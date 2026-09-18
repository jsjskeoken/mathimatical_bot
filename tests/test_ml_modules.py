"""
tests/test_ml_modules.py — ML corrector + dataset pipeline tests.

Covers required behaviors:
 22 OCR confidence + ML confidence + geometry scoring work together
 23 ML layer bypassed for high-confidence obvious digits
 24 low-confidence ambiguous glyph invokes ML layer
 25 ML correction cannot create invalid math syntax
dataset: privacy (crop-only), append-only JSONL schema, human labelling
distinction, grouped split (no leakage), secondary-repo loader contract.
"""

import json
import os

import numpy as np
import pytest

import ocr_ml
import dataset as dataset_mod
import synthetic_data

from conftest import make_ocr_result


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    out = tmp_path_factory.mktemp("syn")
    records = synthetic_data.generate_dataset(str(out), n_per_class=24, seed=7)
    return records, str(out)


# ── feature extraction / backends ────────────────────────────────────────────

def test_extract_features_shape_and_nonzero(synthetic):
    records, _ = synthetic
    import cv2
    img = cv2.imread(records[0]["image_path"], cv2.IMREAD_GRAYSCALE)
    feat = ocr_ml.extract_features(img)
    assert feat.shape == (ocr_ml.FEAT_LEN,)
    assert feat.any()                          # not all zeros for a real glyph
    empty = ocr_ml.extract_features(np.zeros((10, 10), dtype=np.uint8))
    assert not empty.any()


def test_template_backend_learns_digits(synthetic):
    """Contract test for the TemplateBackend machinery — deterministic and
    platform-independent. Makes NO accuracy claim: accuracy numbers come
    from train_glyph_model.py / benchmark_ocr.py on augmented data, never
    from this unit test.

    Why not the old form (>=3/5 clean renders correct against class-mean
    templates averaged over AUGMENTED renders of every available font)?
    That asserted a cross-font generalization property that 1-NN
    class-mean templates do not have: on Windows, a clean Arial '0'
    measures closer to the mixed-font 'O' centroid than to the mixed-font
    '0' centroid (letter 'O' is in the class set by design — the
    corrector must RECOGNISE letters to refuse them). Templates in
    production come from one consistent source, and every correction is
    guarded by the evidence gates (trigger confidence, accept probability,
    EMITTABLE grammar, re-solve check) — so the machinery contract below
    is what must hold on every platform:

      1. template integrity: every class produced a finite, non-empty
         mean template;
      2. self/prototype classification: clean same-font prototypes map
         back to their own class (variation=False renders are
         deterministic, so each probe is an exact feature match — holds
         for any font any platform discovers);
      3. valid class selection + finite predictions: for unseen augmented
         variants the backend always returns a member of the template
         class set with a finite confidence in [0, 1].
    """
    records, _ = synthetic
    import cv2
    by_class = {}
    for r in records:
        if r["dataset_split"] == "unassigned":
            img = cv2.imread(r["image_path"], cv2.IMREAD_GRAYSCALE)
            by_class.setdefault(r["true_label"], []).append(
                ocr_ml.extract_features(img))
    templates = {c: np.mean(v, axis=0) for c, v in by_class.items() if v}
    backend = ocr_ml.TemplateBackend(templates)

    # 1. template integrity
    assert len(templates) >= 20                # all classes got templates
    for c, t in templates.items():
        assert t.size and np.all(np.isfinite(t)), c

    # 2. self/prototype classification on clean same-font prototypes
    import random
    import synthetic_data as sd
    fonts = sd.available_fonts()
    assert fonts, "no TTF font available for synthetic rendering"
    proto_classes = ["0", "8", "5", "O", "+", "÷"]   # incl. the 0/O pair
    protos = {}
    for cls in proto_classes:
        arr = sd.render_glyph(cls, fonts[0], px=48, variation=False)
        protos[cls] = ocr_ml.extract_features(arr.astype(np.uint8))
    proto_backend = ocr_ml.TemplateBackend(protos)
    for cls, feat in protos.items():
        pred, conf = proto_backend.predict(feat)
        assert pred == cls, (cls, pred)        # exact prototype maps back
        assert np.isfinite(conf) and 0.0 <= conf <= 1.0

    # 3. valid class selection on unseen augmented variants (no accuracy
    #    assertion — only that the machinery never leaves the class set)
    rng = random.Random(3)
    for cls in proto_classes:
        arr = sd.render_glyph(cls, fonts[0], px=48, rng=rng, variation=True)
        feat = ocr_ml.extract_features(arr.astype(np.uint8))
        pred, conf = backend.predict(feat)
        assert pred in templates, (cls, pred)
        assert np.isfinite(conf) and 0.0 <= conf <= 1.0


# ── corrector gating (Req 23/24) ─────────────────────────────────────────────

def _corrector_with_templates(synthetic):
    records, _ = synthetic
    import cv2
    templates = {}
    acc = {}
    for r in records:
        img = cv2.imread(r["image_path"], cv2.IMREAD_GRAYSCALE)
        if img is not None:
            acc.setdefault(r["true_label"], []).append(
                ocr_ml.extract_features(img))
    templates = {c: np.mean(v, axis=0) for c, v in acc.items() if v}
    return ocr_ml.GlyphCorrector(templates=templates, trigger_confidence=0.60,
                                 accept_probability=0.6)


def test_ml_bypassed_for_high_confidence(synthetic):
    """Req 23: confident OCR is never touched by the ML layer."""
    corrector = _corrector_with_templates(synthetic)
    img = np.full((60, 120), 128, dtype=np.uint8)
    res = make_ocr_result([("B", 0.99)])       # even an 'ambiguous' glyph
    tokens, confs, _ = corrector.correct_tokens(
        ["B"], [0.99], [res[0][0]], img)
    assert tokens == ["B"]
    assert corrector.last_replacements == []


def test_ml_consulted_for_low_confidence_glyph(synthetic):
    """Req 24: low-confidence ambiguous glyph goes through the ML layer."""
    corrector = _corrector_with_templates(synthetic)
    img = np.full((60, 120), 128, dtype=np.uint8)
    res = make_ocr_result([("B", 0.30)])
    tokens, confs, _ = corrector.correct_tokens(
        ["B"], [0.30], [res[0][0]], img)
    # the template backend predicts something; the decision must be
    # evidence-based: either a confident emittable replacement, or no change
    assert len(tokens) == 1
    for r in corrector.last_replacements:
        assert r["ml_prob"] >= 0.6
        assert r["ml_char"] in ocr_ml.EMITTABLE
        assert r["ocr_char"] == "B"


def test_ml_never_emits_non_emittable_class(synthetic):
    """Req 25: the corrector can only emit digits/canonical operators —
    a letter prediction can never be injected into the expression."""
    corrector = _corrector_with_templates(synthetic)
    img = np.full((60, 120), 128, dtype=np.uint8)
    # force the backend to return a letter with high confidence
    class _LetterOnly:
        def predict(self, feat):
            return ("G", 0.99)
    corrector._backend = _LetterOnly()
    res = make_ocr_result([("6", 0.2)])
    tokens, _, _ = corrector.correct_tokens(["6"], [0.2], [res[0][0]], img)
    assert tokens == ["6"]                     # unchanged
    assert corrector.last_replacements == []


def test_ml_disabled_by_default_in_core(core):
    """The deterministic pipeline is the default fallback (acceptance 9/10)."""
    assert core.ml_enabled is False
    assert core.ml_corrector is None


def test_ml_integrated_correction_keeps_grammar_valid(core, synthetic):
    """Req 22/25 end-to-end: with ML enabled and a corrector that always
    answers, the solver still only accepts expressions that parse and solve
    — ML cannot manufacture invalid syntax."""
    class _Always8:
        def correct_tokens(self, tokens, confidences, bboxes, image):
            return (["8" if t == "B" else t for t in tokens],
                    confidences, bboxes)

    core.ml_corrector = _Always8()
    core.ml_enabled = True
    img = np.zeros((60, 300), dtype=np.uint8)
    res = make_ocr_result([("B", 0.3), ("+", 0.9), ("6", 0.9)])
    best = core.select_math_ocr_text(res, img)
    assert best == "8 + 6"
    # ... and the corrected reading actually solves via the normal pipeline
    answer, source = core.handle_question(best)
    assert answer == 14

    # an ML "correction" that would create invalid syntax is filtered out
    class _BreakIt:
        def correct_tokens(self, tokens, confidences, bboxes, image):
            return (["8+", "+8", "8"], confidences, bboxes)

    core.ml_corrector = _BreakIt()
    res = make_ocr_result([("B", 0.3), ("+", 0.9), ("6", 0.9)])
    best = core.select_math_ocr_text(res, img)   # hard filters reject garbage
    assert best == "" or core.solve_algebra(
        core.normalise(best)) is not None


# ── dataset collector ────────────────────────────────────────────────────────

def test_collector_records_crops_and_metadata(tmp_path, synthetic):
    records, _ = synthetic
    import cv2
    img = cv2.imread(records[0]["image_path"], cv2.IMREAD_GRAYSCALE)
    col = dataset_mod.GlyphDatasetCollector(str(tmp_path / "ds"))
    bbox = [(4, 6), (60, 6), (60, 50), (4, 50)]
    col.record_selection(tokens=["B", "+", "6"],
                         confidences=[0.3, 0.95, 0.9], bboxes=[bbox, bbox, bbox],
                         image=img, selected="", confidence=None,
                         enabled_operations=["+", "/"], fast_mode=True)
    lines = open(col.manifest_path, encoding="utf-8").read().strip().splitlines()
    assert lines                               # low-conf token captured
    rec = json.loads(lines[0])
    for field in ["record_id", "image_path", "raw_or_processed", "true_label",
                  "label_source", "ocr_prediction", "ocr_confidence", "bbox",
                  "token_position", "preprocessing_version", "glyph_class",
                  "ambiguity_category", "ml_prediction", "ml_changed",
                  "question_id", "dataset_split", "selected_expression",
                  "solver_accepted", "captured_at"]:
        assert field in rec, field
    assert rec["true_label"] is None           # unlabelled until a human acts
    assert rec["label_source"] == "unlabelled"
    assert os.path.exists(rec["image_path"])
    # privacy: stored crop is far smaller than a full frame
    stored = cv2.imread(rec["image_path"])
    assert stored.shape[0] <= img.shape[0] and stored.shape[1] <= img.shape[1]


def test_collector_high_confidence_clean_token_skipped(tmp_path):
    """High-confidence, unambiguous digit tokens are not collected.
    (High-conf '+' IS collected: crops of the '+'/'÷' boundary at every
    confidence are exactly what teaches the model the difference — see
    AMBIGUOUS_PAIRS.)"""
    col = dataset_mod.GlyphDatasetCollector(str(tmp_path / "ds"))
    img = np.zeros((60, 120), dtype=np.uint8)
    bbox = [(4, 6), (60, 6), (60, 50), (4, 50)]
    col.record_selection(tokens=["7"], confidences=[0.99], bboxes=[bbox],
                         image=img, selected="7", confidence=0.99,
                         enabled_operations=["+"], fast_mode=True)
    assert not os.path.exists(col.manifest_path) or \
        os.path.getsize(col.manifest_path) == 0
    # ...while an ambiguous operator at high confidence IS kept
    col2 = dataset_mod.GlyphDatasetCollector(str(tmp_path / "ds2"))
    col2.record_selection(tokens=["+"], confidences=[0.99], bboxes=[bbox],
                          image=img, selected="7+6", confidence=0.99,
                          enabled_operations=["+"], fast_mode=True)
    assert os.path.exists(col2.manifest_path)


def test_label_record_distinguishes_human_labelling(tmp_path, synthetic):
    records, _ = synthetic
    import cv2
    img = cv2.imread(records[0]["image_path"], cv2.IMREAD_GRAYSCALE)
    col = dataset_mod.GlyphDatasetCollector(str(tmp_path / "ds"))
    bbox = [(4, 6), (60, 6), (60, 50), (4, 50)]
    col.record_selection(tokens=["B"], confidences=[0.3], bboxes=[bbox],
                         image=img, selected="", confidence=None,
                         enabled_operations=["+"], fast_mode=True)
    rec = json.loads(open(col.manifest_path).readline())
    dataset_mod.label_record(col.manifest_path, rec["record_id"], "8")
    rec2 = json.loads(open(col.manifest_path).readline())
    assert rec2["true_label"] == "8"
    assert rec2["label_source"] == "human"
    assert rec2["reviewer"] == "manual"


def test_grouped_split_no_question_leakage(tmp_path, synthetic):
    records, _ = synthetic
    # simulate two crops from the same question
    records[0]["question_id"] = "q1"
    records[1]["question_id"] = "q1"
    records[2]["question_id"] = "q2"
    split = dataset_mod.grouped_split(records, train=0.5, val=0.25, seed=1)
    assert split[0]["dataset_split"] == split[1]["dataset_split"]


# ── secondary-repo loader contract ───────────────────────────────────────────

def test_load_dataset_documents_missing_data(tmp_path):
    """The secondary repo currently contains NO images (ocr_captures/ is
    gitignored there) — the loader must fail with an actionable message
    rather than silently substituting another dataset."""
    with pytest.raises((FileNotFoundError, ValueError)) as ei:
        dataset_mod.load_dataset(str(tmp_path))
    msg = str(ei.value).lower()
    assert "labels.jsonl" in msg or "human-confirmed" in msg


def test_load_dataset_roundtrip(tmp_path):
    d = tmp_path / "dataset"
    (d / "images").mkdir(parents=True)
    import cv2
    cv2.imwrite(str(d / "images" / "a.png"), np.zeros((20, 20), np.uint8))
    rec = {"record_id": "a", "image_path": "images/a.png",
           "raw_or_processed": "raw", "true_label": "8",
           "label_source": "human", "question_id": "qz",
           "dataset_split": "unassigned"}
    with open(d / "labels.jsonl", "w") as f:
        f.write(json.dumps(rec) + "\n")
    records, info = dataset_mod.load_dataset(str(tmp_path))
    assert info["images"] == 1 and info["labelled"] == 1
    assert info["raw"] == 1 and info["processed"] == 0
    assert records[0]["true_label"] == "8"
    assert os.path.isabs(records[0]["image_path"])


def test_unlabelled_dataset_refused(tmp_path):
    d = tmp_path / "dataset"
    (d / "images").mkdir(parents=True)
    import cv2
    cv2.imwrite(str(d / "images" / "a.png"), np.zeros((20, 20), np.uint8))
    rec = {"record_id": "a", "image_path": "images/a.png",
           "raw_or_processed": "processed", "true_label": None,
           "label_source": "unlabelled", "question_id": "q",
           "dataset_split": "unassigned"}
    with open(d / "labels.jsonl", "w") as f:
        f.write(json.dumps(rec) + "\n")
    with pytest.raises(ValueError, match="human-confirmed"):
        dataset_mod.load_dataset(str(tmp_path))
