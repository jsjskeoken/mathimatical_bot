"""
ocr_ml.py — Optional ML glyph CORRECTOR / RERANKER for the OCR pipeline.

Design rules (from the project requirements):
  - It is NOT a replacement for EasyOCR. It only re-scores individual,
    SUSPECT glyphs (low OCR confidence / ambiguous shape) and reports class
    PROBABILITIES as evidence.
  - A high-confidence EasyOCR reading is never touched.
  - A replacement is applied only when the ML probability clears
    ML_ACCEPT_PROBABILITY, and the corrected token must still pass every
    downstream hard filter (operator set, digit-group count, solvability)
    in select_math_ocr_text() — the ML layer alone can never manufacture an
    invalid expression.
  - Smallest-model-first: template matching baseline, then a tiny
    single-hidden-layer MLP (numpy only — no sklearn/torch dependency at
    runtime). Models are produced by train_glyph_model.py.
  - The deterministic pipeline must remain complete and valid when this
    module is disabled (bot_core.ml_enabled=False, the default).

Only numpy + cv2 + PIL at module level (all already project dependencies).
"""

import json
import os
import time

import numpy as np

try:
    import cv2
except Exception:            # pragma: no cover - cv2 is a hard dep of bot_core
    cv2 = None

# The class set the corrector distinguishes. Digits + the operators this
# solver understands + the letters EasyOCR realistically confuses with them
# (allowlist leakage happens in practice even with an allowlist set).
ML_CLASSES = list("0123456789") + ["+", "-", "*", "/", "×", "÷", ":", "x",
                                   "B", "O", "S", "G", "g", "I", "l"]
CLASS_TO_IDX = {c: i for i, c in enumerate(ML_CLASSES)}

# Pairs we explicitly expect to be confusable (documented; used for dataset
# ambiguity tagging and benchmark reporting).
CONFUSION_GROUPS = [
    ("8", "B"), ("0", "O"), ("1", "I"), ("1", "l"), ("5", "S"),
    ("6", "G"), ("9", "g"), ("2", "z"), ("x", "×"), ("*", "x"),
    ("÷", "+"), ("÷", "/"), (":", "/"), ("-", "artefact"),
]

FEATURE_W, FEATURE_H = 16, 24          # downsampled glyph grid
FEAT_LEN = FEATURE_W * FEATURE_H + 16  # + gradient-energy histograms

# Characters the corrector is allowed to EMIT. Deliberately narrower than the
# class set: it may replace a glyph only with something the solver's
# downstream canonicalisation understands (a digit or a canonical operator).
EMITTABLE = set("0123456789+-*/x×÷:")


def _ensure_cv():
    if cv2 is None:
        raise RuntimeError("ocr_ml requires cv2 (already a bot_core dependency)")


def extract_features(gray_crop) -> np.ndarray:
    """
    Deterministic feature extraction for one glyph crop.

    1. grayscale → Otsu binarise, polarity-normalised so INK = 1
       (whichever polarity the frame happens to use)
    2. tight-crop to ink bounding box, paste centred on a fixed canvas,
       resize to FEATURE_H x FEATURE_W
    3. raw grid values (normalised 0..1)
    4. HOG-lite: 8-bin horizontal + 8-bin vertical gradient-energy
       histograms — gives the classifier stroke-orientation cues
       (1 vs I vs l, + vs × differ mainly in stroke structure)
    """
    _ensure_cv()
    if gray_crop is None or gray_crop.size == 0:
        return np.zeros(FEAT_LEN, dtype=np.float32)
    gray = gray_crop if gray_crop.ndim == 2 else cv2.cvtColor(
        gray_crop, cv2.COLOR_BGR2GRAY)
    if gray.size < 4:
        return np.zeros(FEAT_LEN, dtype=np.float32)

    # Polarity-normalised binarisation: pick the variant with the smaller
    # ink fraction as ink (glyphs are always the minority of pixels).
    best = None
    for tt in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
        if gray.std() < 1e-3:
            best = (gray > 0).astype(np.uint8)
            break
        _, m = cv2.threshold(gray, 0, 255, tt + cv2.THRESH_OTSU)
        ink = (m > 0).mean()
        if best is None or ink < best[0]:
            best = (ink, (m > 0).astype(np.uint8))
    ink_mask = best[1] if isinstance(best, tuple) else best

    ys, xs = np.nonzero(ink_mask)
    if len(xs) == 0:
        return np.zeros(FEAT_LEN, dtype=np.float32)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    glyph = ink_mask[y0:y1, x0:x1].astype(np.float32)

    canvas = np.zeros((FEATURE_H * 4, FEATURE_W * 4), dtype=np.float32)
    scale = min((FEATURE_W * 4) / glyph.shape[1],
                (FEATURE_H * 4) / glyph.shape[0], 1.0)
    if scale < 1.0:
        glyph = cv2.resize(glyph, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
    gh, gw = glyph.shape
    oy, ox = (canvas.shape[0] - gh) // 2, (canvas.shape[1] - gw) // 2
    canvas[oy:oy + gh, ox:ox + gw] = glyph
    grid = cv2.resize(canvas, (FEATURE_W, FEATURE_H),
                      interpolation=cv2.INTER_AREA).reshape(-1)

    gx = np.abs(np.diff(canvas, axis=1)).sum(axis=0)
    gy = np.abs(np.diff(canvas, axis=0)).sum(axis=1)
    hx = np.array_split(gx, 8)
    hy = np.array_split(gy, 8)
    hist = np.array([h.sum() for h in hx] + [h.sum() for h in hy],
                    dtype=np.float32)
    total = hist.sum()
    if total > 0:
        hist /= total

    return np.concatenate([grid, hist]).astype(np.float32)


def _crop_glyph(image, bbox, char_index, char_count):
    """
    Equal-width monospace glyph column crop — same approximation as
    BotCore._extract_glyph_crop (duplicated here to keep ocr_ml importable
    without bot_core; documented, not silent).
    """
    try:
        xs = [int(p[0]) for p in bbox]
        ys = [int(p[1]) for p in bbox]
    except (TypeError, ValueError, IndexError):
        return None
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    char_count = max(1, char_count)
    w = (x2 - x1) / char_count
    pad = w * 0.35
    cx1 = max(x1, int(x1 + char_index * w - pad))
    cx2 = min(x2, int(x1 + (char_index + 1) * w + pad))
    cy1, cy2 = max(0, y1), min(image.shape[0], y2)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return image[cy1:cy2, cx1:cx2]


class TemplateBackend:
    """
    Class-mean bitmap templates (cosine similarity). Dependency-free
    baseline; also the fallback when no trained model is available.
    Templates are supplied by the caller (built from synthetic data or from
    verified dataset crops) as {class: feature_vector}.
    """

    def __init__(self, templates=None):
        self.templates = templates or {}

    def predict(self, feat: np.ndarray):
        if not self.templates:
            return None
        best_c, best_s = None, -1.0
        for c, t in self.templates.items():
            denom = (np.linalg.norm(feat) * np.linalg.norm(t)) or 1.0
            s = float(np.dot(feat, t) / denom)
            if s > best_s:
                best_c, best_s = c, s
        # softmax-ish over similarities for a comparable confidence scale
        return best_c, min(1.0, max(0.0, (best_s + 1.0) / 2.0))


class MLPBackend:
    """
    Tiny MLP: input -> 32 tanh -> softmax(classes). Weights live in a plain
    JSON file produced by train_glyph_model.py. Numpy-only forward pass —
    sub-millisecond on one glyph, no framework dependency.
    """

    def __init__(self, weights: dict):
        self.classes = weights["classes"]
        self.W1 = np.array(weights["W1"], dtype=np.float32)
        self.b1 = np.array(weights["b1"], dtype=np.float32)
        self.W2 = np.array(weights["W2"], dtype=np.float32)
        self.b2 = np.array(weights["b2"], dtype=np.float32)
        if self.W1.shape[0] != FEAT_LEN:
            raise ValueError(
                f"model expects {self.W1.shape[0]} features, got {FEAT_LEN}")

    def predict(self, feat: np.ndarray):
        h = np.tanh(feat @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        idx = int(np.argmax(probs))
        return self.classes[idx], float(probs[idx])

    def predict_full(self, feat: np.ndarray):
        h = np.tanh(feat @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        return {c: float(p) for c, p in zip(self.classes, probs)}


class GlyphCorrector:
    """
    Stage-3 corrector: consulted only for SUSPECT glyphs.

    Usage from bot_core (already wired):
        core.ml_corrector = GlyphCorrector(model_path="ml_model/glyph_model.json")
        core.ml_enabled = True
    """

    def __init__(self, model_path=None, templates=None,
                 trigger_confidence=0.60, accept_probability=0.85,
                 min_crop_size=6):
        self.trigger_confidence = trigger_confidence
        self.accept_probability = accept_probability
        self.min_crop_size = min_crop_size
        self.last_replacements = []    # diagnostics: what ML changed (per call)
        self._backend = None
        self._backend_name = "none"
        if model_path and os.path.exists(model_path):
            try:
                with open(model_path, "r") as f:
                    weights = json.load(f)
                self._backend = MLPBackend(weights)
                self._backend_name = "mlp"
            except Exception as e:
                print(f"[ML] model load failed ({e}) — falling back to templates")
        if self._backend is None and templates:
            self._backend = TemplateBackend(templates)
            self._backend_name = "template"

    @property
    def backend_name(self):
        return self._backend_name

    def predict(self, gray_crop):
        """Return (class, confidence) or None when no backend/crop usable."""
        if self._backend is None:
            return None
        if gray_crop is None or gray_crop.size == 0:
            return None
        if min(gray_crop.shape[:2]) < self.min_crop_size:
            return None
        feat = extract_features(gray_crop)
        return self._backend.predict(feat)

    def correct_tokens(self, tokens, confidences, bboxes, image):
        """
        Evidence-only correction pass over the token list BEFORE candidate
        generation. Returns (tokens, confidences, bboxes) with at most
        per-suspect-glyph single-character replacements.

        A character qualifies only when ALL of:
          - it is not whitespace/bracket punctuation,
          - its token's OCR confidence < trigger_confidence,
          - a glyph crop can be extracted,
          - the backend predicts an EMITTABLE class with probability
            >= accept_probability,
          - the prediction differs from the current character.
        """
        self.last_replacements = []
        if self._backend is None or image is None:
            return tokens, confidences, bboxes
        out = list(tokens)
        for ti, (tok, conf, bbox) in enumerate(zip(tokens, confidences, bboxes)):
            if conf is not None and conf >= self.trigger_confidence:
                continue
            chars = list(tok)
            changed = False
            for ci, ch in enumerate(chars):
                if ch.strip() == "" or ch in "()=?":
                    continue
                crop = _crop_glyph(image, bbox, ci, len(chars))
                pred = self.predict(crop)
                if pred is None:
                    continue
                pcls, pprob = pred
                if pcls == ch or pcls not in EMITTABLE:
                    continue
                if pprob < self.accept_probability:
                    continue
                self.last_replacements.append({
                    "token_index": ti, "char_index": ci,
                    "ocr_char": ch, "ml_char": pcls, "ml_prob": round(pprob, 4),
                    "ocr_conf": None if conf is None else round(float(conf), 4),
                })
                chars[ci] = pcls
                changed = True
            if changed:
                out[ti] = "".join(chars)
                print(f"[ML] corrected token {tok!r} -> {out[ti]!r} "
                      f"(backend={self._backend_name}, "
                      f"replacements={self.last_replacements[-1]})")
        return out, list(confidences), list(bboxes)


def build_templates_from_crops(samples, classes=None):
    """
    Build TemplateBackend templates from dataset samples
    (list of dicts with 'image_path' + 'true_label').
    """
    _ensure_cv()
    classes = classes or ML_CLASSES
    acc = {c: [] for c in classes}
    for s in samples:
        label = s.get("true_label")
        if label not in CLASS_TO_IDX or not os.path.exists(s["image_path"]):
            continue
        img = cv2.imread(s["image_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        acc[label].append(extract_features(img))
    templates = {c: np.mean(v, axis=0) for c, v in acc.items() if v}
    return templates


def measure_latency(corrector, n=200, size=(24, 32), seed=0):
    """Latency micro-benchmark: predict() on random noise crops (µs/predict)."""
    rng = np.random.default_rng(seed)
    crops = [(rng.random((size[1], size[0])) * 255).astype(np.uint8)
             for _ in range(n)]
    t0 = time.perf_counter()
    for c in crops:
        corrector.predict(c)
    dt = time.perf_counter() - t0
    return dt / n * 1e6
