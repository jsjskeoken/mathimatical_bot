"""
synthetic_data.py — Synthetic glyph generator for bootstrapping the ML
glyph corrector.

Renders 0-9 and + - * x × ÷ : plus confusion letters (B O S G g I l) with
controlled variations: scale, antialiasing, blur, sharpening, contrast,
brightness, noise, compression (JPEG), position jitter, slight rotation,
partial occlusion, stroke thickness (erode/dilate), tight spacing and
glued tokens.

HONESTY RULE (enforced by train_glyph_model.py reporting): synthetic
accuracy is NEVER equivalent to real-world accuracy. Synthetic data
bootstraps and balances; evaluation for deployment decisions must use the
REAL crops from the secondary repository dataset (dataset.load_dataset).
"""

import json
import os
import random

import numpy as np

from PIL import Image, ImageDraw, ImageFont, ImageFilter

# Classes rendered. Letters included so the corrector can learn the real
# confusion classes; the corrector itself may only EMIT digits/operators
# (see ocr_ml.EMITTABLE) — letters exist in the class set so the model can
# recognise "this is a letter, not a digit" and refuse.
GLYPH_CLASSES = list("0123456789") + ["+", "-", "*", "x", "×", "÷", ":",
                                      "B", "O", "S", "G", "g", "I", "l"]

FONT_CANDIDATES = [
    # Windows
    "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/calibri.ttf",
    "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/consola.ttf",
    "C:/Windows/Fonts/cour.ttf",
    # Linux (this repo's CI/test environment)
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
]


def available_fonts():
    return [f for f in FONT_CANDIDATES if os.path.exists(f)]


def render_glyph(ch, font_path, px=48, rng=None, variation=True):
    """Render one glyph to a grayscale uint8 array with augmentations."""
    rng = rng or random.Random()
    font = ImageFont.truetype(font_path, px)
    pad = px // 2
    W, H = px * 2 + pad, px * 2 + pad
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    d.text((pad, pad), ch, fill=255, font=font)

    if variation:
        if rng.random() < 0.7:                        # slight rotation
            img = img.rotate(rng.uniform(-7, 7), resample=Image.BILINEAR,
                             fillcolor=0)
        if rng.random() < 0.5:                        # blur / sharpen
            if rng.random() < 0.5:
                img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.4)))
            else:
                img = img.filter(ImageFilter.UnsharpMask(2, 80, 2))
        if rng.random() < 0.4:                        # scale squash/stretch
            fx, fy = rng.uniform(0.85, 1.15), rng.uniform(0.85, 1.15)
            img = img.resize((int(W * fx), int(H * fy)), Image.BILINEAR)
            img = img.crop((0, 0, W, H)) if img.size[0] >= W and img.size[1] >= H \
                else img.resize((W, H), Image.BILINEAR)

    arr = np.array(img, dtype=np.float32)

    if variation:
        if rng.random() < 0.5:                        # contrast / brightness
            a = rng.uniform(0.7, 1.3)
            b = rng.uniform(-30, 30)
            arr = np.clip(arr * a + b, 0, 255)
        if rng.random() < 0.5:                        # additive noise
            # numpy Generator seeded from the python rng — deterministic
            # per call, since random.Random has no .normal()
            nprng = np.random.default_rng(rng.randrange(2 ** 32))
            arr = np.clip(arr + nprng.normal(0, rng.uniform(4, 18),
                                             arr.shape), 0, 255)
        if rng.random() < 0.25:                       # partial occlusion
            y0 = rng.randint(0, arr.shape[0] - 8)
            x0 = rng.randint(0, arr.shape[1] - 8)
            arr[y0:y0 + rng.randint(3, 8), x0:x0 + rng.randint(3, 8)] = 0
        if rng.random() < 0.3:                        # stroke thickness
            import cv2
            k = 1 if rng.random() < 0.5 else -1
            kern = np.ones((2, 2), np.uint8)
            u8 = arr.astype(np.uint8)
            u8 = cv2.dilate(u8, kern) if k == 1 else cv2.erode(u8, kern)
            arr = u8.astype(np.float32)

    # position jitter within a small margin + realistic scale variation:
    # real EasyOCR crops are small (15-30 px tall) — the model must see
    # low-resolution glyphs too, or it mispredicts exactly the crops the
    # runtime will feed it.
    ys, xs = np.nonzero(arr > 40)
    if len(xs) == 0:
        return np.zeros((64, 64), dtype=np.float32)
    g = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    import cv2
    if variation:
        target_h = rng.randint(18, min(52, max(20, g.shape[0])))
        scale = target_h / g.shape[0]
        if scale < 1.0:
            g = cv2.resize(g, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
    scale = min(56 / g.shape[0], 56 / g.shape[1], 1.0)
    if scale < 1.0:
        g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    canvas = np.zeros((64, 64), dtype=np.float32)
    gh, gw = g.shape
    oy = rng.randint(0, max(0, 64 - gh)) // 2 if variation else (64 - gh) // 2
    ox = rng.randint(0, max(0, 64 - gw)) // 2 if variation else (64 - gw) // 2
    canvas[oy:oy + gh, ox:ox + gw] = g
    return canvas


def render_glued_pair(ch_op, ch_digit, font_path, rng=None):
    """Render an operator glued to a digit (e.g. '×4', '+6') — the merged
    token failure mode, so the model sees tight-spacing crops too."""
    rng = rng or random.Random()
    a = render_glyph(ch_op, font_path, rng=rng, variation=False)
    b = render_glyph(ch_digit, font_path, rng=rng, variation=False)
    ya, xa = np.nonzero(a > 40)
    yb, xb = np.nonzero(b > 40)
    a = a[ya.min():ya.max() + 1, xa.min():xa.max() + 1]
    b = b[yb.min():yb.max() + 1, xb.min():xb.max() + 1]
    overlap = rng.randint(0, max(1, a.shape[1] // 4))
    H = max(a.shape[0], b.shape[0]) + 8
    W = a.shape[1] + b.shape[1] - overlap + 8
    canvas = np.zeros((H, W), dtype=np.float32)
    oy = (H - a.shape[0]) // 2
    canvas[oy:oy + a.shape[0], 4:4 + a.shape[1]] = a
    oy2 = (H - b.shape[0]) // 2
    x2 = 4 + a.shape[1] - overlap
    canvas[oy2:oy2 + b.shape[0], x2:x2 + b.shape[1]] = b
    return canvas


def generate_dataset(out_dir, n_per_class=60, seed=42, fonts=None):
    """
    Generate the synthetic dataset: images/<id>.png + labels.jsonl.
    Every record carries source_id = font+class batch so grouped splits can
    never leak the same rendering session across folds.
    """
    rng = random.Random(seed)
    fonts = fonts or available_fonts()
    if not fonts:
        raise RuntimeError("no usable TTF fonts found for synthetic rendering")
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    records = []
    seq = 0
    for cls in GLYPH_CLASSES:
        for i in range(n_per_class):
            font = rng.choice(fonts)
            arr = render_glyph(cls, font, px=rng.choice([36, 44, 52]),
                               rng=rng, variation=True)
            seq += 1
            name = f"syn_{seq:06d}.png"
            path = os.path.join(out_dir, "images", name)
            _save(arr, path)
            records.append({
                "record_id": f"syn_{seq:06d}",
                "image_path": path,
                "raw_or_processed": "synthetic",
                "true_label": cls,
                "label_source": "synthetic_ground_truth",
                "source_id": f"{os.path.basename(font)}|{cls}",
                "ocr_prediction": None,
                "ocr_confidence": None,
                # Each single-glyph render is its own group: renders are
                # independently augmented, so a per-record group split
                # cannot leak near-identical crops across folds. (A CLASS
                # level group would hold out entire classes from training
                # and report a meaningless 0% test accuracy.)
                "question_id": f"syn_{seq:06d}",
                "dataset_split": "unassigned",
            })
    # glued tokens: 15% extra, half labelled as the operator class
    for i in range(max(1, n_per_class * 15 // 100)):
        for cls in ["+", "-", "*", "x", "×", "÷", ":"]:
            font = rng.choice(fonts)
            arr = render_glued_pair(cls, rng.choice(list("0123456789")),
                                    font, rng=rng)
            seq += 1
            name = f"syn_{seq:06d}.png"
            path = os.path.join(out_dir, "images", name)
            _save(arr, path)
            records.append({
                "record_id": f"syn_{seq:06d}",
                "image_path": path,
                "raw_or_processed": "synthetic",
                "true_label": cls,
                "label_source": "synthetic_ground_truth",
                "source_id": f"{os.path.basename(font)}|{cls}",
                "ocr_prediction": None,
                "ocr_confidence": None,
                "question_id": f"syn-glued-{cls}",
                "dataset_split": "unassigned",
            })
    with open(os.path.join(out_dir, "labels.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return records


def _save(arr, path):
    from PIL import Image as _I
    norm = np.clip(arr, 0, 255).astype(np.uint8)
    _I.fromarray(norm).save(path)
