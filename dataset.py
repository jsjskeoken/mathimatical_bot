"""
dataset.py — REAL OCR image dataset pipeline.

Two roles:

1. ACTIVE COLLECTOR (runtime): when attached to BotCore.dataset_collector,
   record_selection() saves ONLY the token/glyph crops involved in a
   selection decision (never a whole screenshot) plus an append-only JSONL
   metadata line. Difficult cases (low confidence, ambiguous glyphs,
   rejected candidates) become reviewable training data.

2. DATASET LOADER (training): loads a dataset laid out as

       <root>/images/*.png        (raw or processed crops)
       <root>/labels.jsonl        (one line per labelled example)

   from the secondary repository (jentrenert/optical-reader-math-solver)
   and records the dataset revision for reproducibility.

Privacy rules enforced here:
  - only the OCR crop region is stored, never the full screen;
  - no window titles, no credentials, no unrelated content — the JSONL
    carries OCR metadata only;
  - labelling is explicit: automatically-generated records have
    true_label=null and are distinguishable from human-confirmed labels;
  - the model's own predictions are recorded as ml_prediction, never as
    ground truth (no silent self-training).

JSONL schema (one JSON object per line):
  record_id           stable id
  image_path          path (relative when loading from a dataset root)
  raw_or_processed    "raw" | "processed"
  true_label          null until a human confirms (explicit labelling)
  label_source        "unlabelled" | "human" | "ocr_accepted" (provisional)
  ocr_prediction      EasyOCR token text
  ocr_confidence      float | null
  bbox                [x1, y1, x2, y2] | null
  token_position      index of token in the reading
  preprocessing_version string (pipeline version that produced the crop)
  glyph_class         "digit" | "operator" | "letter"
  operator_category   "digit" | "+-*/×÷:x" family | null
  ambiguity_category  e.g. "8/B" | null
  deterministic_correction bool (deterministic rules changed this token)
  ml_prediction       null | {char, prob}
  ml_changed          bool (ML layer replaced the OCR character)
  final_accepted_label what the solver finally used for this token
  question_id         id of the expression this token appeared in
  selected_expression the candidate the solver selected ("" when rejected)
  solver_accepted     bool (the selection was solved/accepted)
  enabled_operations  sorted list
  fast_mode           bool
  dataset_split       "unassigned" (assigned at training time, grouped by
                      question_id/source — never random per-crop)
  dataset_revision    git revision of the dataset repo (loader-filled)
  captured_at         unix timestamp
"""

import datetime
import hashlib
import json
import os
import subprocess
import time

DIGITS = set("0123456789")
OPERATOR_CHARS = set("+-*/×÷:xX")
AMBIGUOUS_PAIRS = [("8", "B"), ("0", "O"), ("1", "I"), ("1", "l"), ("5", "S"),
                   ("6", "G"), ("9", "g"), ("2", "z"), ("x", "×"), ("*", "x"),
                   ("÷", "+"), ("÷", "/"), (":", "/")]

PREPROCESSING_VERSION = "v1-gray-blur-unsharp-clahe-otsu"


def classify_char(ch):
    if ch in DIGITS:
        return "digit", None
    if ch in OPERATOR_CHARS:
        return "operator", ch
    return "letter", None


def ambiguity_of(ch):
    for a, b in AMBIGUOUS_PAIRS:
        if ch == a:
            return f"{a}/{b}"
        if ch == b:
            return f"{b}/{a}"
    return None


class GlyphDatasetCollector:
    """
    Append-only collector. Attach to BotCore.dataset_collector to activate:

        core.dataset_collector = GlyphDatasetCollector("ml_dataset")
    """

    def __init__(self, out_dir, capture_trigger_confidence=0.75):
        self.out_dir = out_dir
        self.images_dir = os.path.join(out_dir, "images")
        self.manifest_path = os.path.join(out_dir, "labels.jsonl")
        # Tokens whose confidence is below this are captured; higher-conf
        # tokens are only captured when a correction/ML changed them.
        self.capture_trigger_confidence = capture_trigger_confidence
        self._seq = 0
        os.makedirs(self.images_dir, exist_ok=True)

    # ── token-level capture ──────────────────────────────────────────────────

    def record_selection(self, tokens, confidences, bboxes, image, selected,
                         confidence, enabled_operations, fast_mode,
                         raw_or_processed="processed", ml_info=None):
        """
        Called from select_math_ocr_text for every selection decision.
        question_id is derived from the selected expression + config so the
        same question across retries shares an id (needed for grouped
        train/test splits).
        """
        question_id = self._question_id(selected, enabled_operations, fast_mode)
        solver_accepted = bool(selected)
        for ti, (tok, conf, bbox) in enumerate(zip(tokens, confidences, bboxes)):
            low_conf = (conf is None) or (conf < self.capture_trigger_confidence)
            ambiguous = any(ambiguity_of(ch) for ch in tok)
            changed = ml_info and any(r["token_index"] == ti for r in ml_info)
            if not (low_conf or ambiguous or changed):
                continue
            self._record_token(
                image=image, bbox=bbox, token=tok, conf=conf, position=ti,
                question_id=question_id, selected=selected,
                solver_accepted=solver_accepted,
                enabled_operations=enabled_operations, fast_mode=fast_mode,
                raw_or_processed=raw_or_processed, ml_info=ml_info)

    def _record_token(self, image, bbox, token, conf, position, question_id,
                      selected, solver_accepted, enabled_operations,
                      fast_mode, raw_or_processed, ml_info):
        import cv2
        self._seq += 1
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        rec_id = f"{stamp}_{self._seq:06d}"
        image_name = f"{rec_id}_token{position}.png"
        image_path = os.path.join(self.images_dir, image_name)
        try:
            if bbox is not None and image is not None:
                xs = [int(p[0]) for p in bbox]
                ys = [int(p[1]) for p in bbox]
                x1, x2 = max(0, min(xs)), max(0, max(xs))
                y1, y2 = max(0, min(ys)), max(0, max(ys))
                if x2 > x1 and y2 > y1:
                    cv2.imwrite(image_path, image[y1:y2, x1:x2])
                else:
                    return
            else:
                return
        except Exception as e:
            print(f"[DATASET] crop save failed: {e}")
            return

        glyph_class, op_cat = classify_char(token[0] if token else "?")
        ambiguous = ambiguity_of(token[0]) if token else None
        ml_preds = [r for r in (ml_info or []) if r["token_index"] == position]
        record = {
            "record_id": rec_id,
            "image_path": image_path,
            "raw_or_processed": raw_or_processed,
            "true_label": None,
            "label_source": "unlabelled",
            "ocr_prediction": token,
            "ocr_confidence": None if conf is None else round(float(conf), 4),
            "bbox": [x1, y1, x2, y2] if bbox is not None else None,
            "token_position": position,
            "preprocessing_version": PREPROCESSING_VERSION,
            "glyph_class": glyph_class,
            "operator_category": op_cat,
            "ambiguity_category": ambiguous,
            "deterministic_correction": False,
            "ml_prediction": (ml_preds[-1]["ml_char"] if ml_preds else None),
            "ml_changed": bool(ml_preds),
            "final_accepted_label": token,
            "question_id": question_id,
            "selected_expression": selected,
            "solver_accepted": solver_accepted,
            "enabled_operations": sorted(enabled_operations),
            "fast_mode": bool(fast_mode),
            "dataset_split": "unassigned",
            "dataset_revision": None,
            "captured_at": time.time(),
        }
        self._append(record)

    def _append(self, record):
        try:
            with open(self.manifest_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[DATASET] manifest append failed: {e}")

    @staticmethod
    def _question_id(selected, enabled_operations, fast_mode):
        h = hashlib.blake2b(digest_size=8)
        h.update((selected or "<rejected>").encode("utf-8", "replace"))
        h.update(("".join(sorted(enabled_operations))).encode())
        h.update(b"fast" if fast_mode else b"std")
        return h.hexdigest()


def label_record(manifest_path, record_id, true_label, reviewer="manual"):
    """
    Explicit human labelling — the ONLY path that sets a true_label.
    Appends a corrected copy of the record (append-only; original retained).
    """
    records = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("record_id") == record_id:
                r = dict(r)
                r["true_label"] = true_label
                r["label_source"] = "human"
                r["reviewer"] = reviewer
            records.append(r)
    with open(manifest_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ── Dataset-root loading (secondary repository) ──────────────────────────────

EXPECTED_DATA_DIRS = ("dataset", "ocr_captures", "ml_dataset", "data")


def dataset_revision(repo_path):
    """Git revision of the dataset repo — recorded for reproducibility."""
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_path,
            capture_output=True, text=True, check=True, timeout=10)
        return rev.stdout.strip()
    except Exception:
        return None


def load_dataset(repo_or_dir, require_labels=True):
    """
    Load a real-image dataset from the secondary repository (or any
    directory following the same layout). Returns (records, info) where
    info documents the revision + counts — required fields for any
    reported real-data metric.
    """
    repo_or_dir = os.path.abspath(repo_or_dir)
    if not os.path.isdir(repo_or_dir):
        raise FileNotFoundError(f"dataset root not found: {repo_or_dir}")

    manifest = None
    for sub in EXPECTED_DATA_DIRS:
        for cand in (os.path.join(repo_or_dir, sub, "labels.jsonl"),
                     os.path.join(repo_or_dir, "labels.jsonl")):
            if os.path.isfile(cand):
                manifest = cand
                break
        if manifest:
            break

    info = {
        "root": repo_or_dir,
        "revision": dataset_revision(repo_or_dir),
        "manifest": manifest,
        "images": 0,
        "labelled": 0,
        "raw": 0,
        "processed": 0,
    }
    if manifest is None:
        raise FileNotFoundError(
            f"no labels.jsonl found under {repo_or_dir} (searched "
            f"{EXPECTED_DATA_DIRS} + root). The dataset layout must be "
            f"<root>/images/*.png + <root>/labels.jsonl — see dataset.py "
            f"docstring. If the secondary repository has not had its image "
            f"data pushed yet (ocr_captures/ is gitignored there), this is "
            f"the expected outcome: real-data training cannot start until "
            f"the images + labels.jsonl are actually committed.")

    records = []
    with open(manifest, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not os.path.isabs(r.get("image_path", "")):
                r["image_path"] = os.path.join(os.path.dirname(manifest),
                                               r["image_path"])
            r["dataset_revision"] = info["revision"]
            records.append(r)
            if os.path.exists(r.get("image_path", "")):
                info["images"] += 1
            if r.get("true_label") is not None:
                info["labelled"] += 1
            if r.get("raw_or_processed") == "raw":
                info["raw"] += 1
            else:
                info["processed"] += 1

    if require_labels and info["labelled"] == 0:
        raise ValueError(
            "dataset contains images but no human-confirmed labels "
            "(label_source='human'); run dataset.label_record() on reviewed "
            "records first. Automatically-generated predictions must never "
            "be trained against as ground truth.")
    return records, info


def grouped_split(records, train=0.7, val=0.15, seed=0):
    """
    Split at question_id level (never per-crop): near-identical crops from
    the same question/screen stay in the same fold, so test accuracy is
    not inflated by leakage.
    """
    import random
    qids = sorted({r.get("question_id") or r["record_id"] for r in records})
    rng = random.Random(seed)
    rng.shuffle(qids)
    n = len(qids)
    n_train = int(n * train)
    n_val = int(n * val)
    assignment = {}
    for i, q in enumerate(qids):
        assignment[q] = ("train" if i < n_train
                         else "val" if i < n_train + n_val
                         else "test")
    for r in records:
        r["dataset_split"] = assignment[r.get("question_id") or r["record_id"]]
    return records
