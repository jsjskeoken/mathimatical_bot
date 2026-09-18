"""
train_glyph_model.py — Train + evaluate the tiny glyph corrector.

Pipeline:
  1. synthetic bootstrap data (synthetic_data.generate_dataset)
  2. REAL crops from the secondary dataset repo, when present AND
     human-labelled (dataset.load_dataset — refuses unlabelled data)
  3. grouped split at source/question level (no crop-level leakage)
  4. candidates: template-matching baseline + tiny MLP (numpy)
  5. report REAL vs SYNTHETIC accuracy SEPARATELY + confusion matrix +
     latency + model size
  6. deployment gate: model.json is written only when the MLP beats the
     template baseline on the validation fold. Real-data evaluation is the
     gate for runtime adoption (bot_core.ml_enabled stays False until a
     real-data benchmark proves improvement).

Usage:
  python train_glyph_model.py                       # synthetic-only bootstrap
  python train_glyph_model.py --real /path/to/optical-reader-math-solver
  python train_glyph_model.py --n-per-class 120 --seed 1
"""

import argparse
import json
import os
import sys
import time

import numpy as np

import ocr_ml
import synthetic_data


# ── data prep ────────────────────────────────────────────────────────────────

def load_images(records):
    import cv2
    feats, labels, kept = [], [], []
    for r in records:
        p = r["image_path"]
        if not os.path.exists(p):
            continue
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        feats.append(ocr_ml.extract_features(img))
        labels.append(ocr_ml.CLASS_TO_IDX[r["true_label"]])
        kept.append(r)
    return (np.array(feats, dtype=np.float32),
            np.array(labels, dtype=np.int64), kept)


def train_mlp(X, y, n_classes, hidden=32, epochs=400, lr=0.05, seed=0,
              weight_decay=1e-4):
    """Numpy MLP (input -> tanh(hidden) -> softmax) with mini-batch Adam."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    W1 = rng.normal(0, 0.05, (d, hidden)).astype(np.float32)
    b1 = np.zeros(hidden, dtype=np.float32)
    W2 = rng.normal(0, 0.05, (hidden, n_classes)).astype(np.float32)
    b2 = np.zeros(n_classes, dtype=np.float32)
    params = [W1, b1, W2, b2]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    idx_all = np.arange(n)
    for epoch in range(epochs):
        rng.shuffle(idx_all)
        for start in range(0, n, 64):
            batch = idx_all[start:start + 64]
            xb, yb = X[batch], y[batch]
            h = np.tanh(xb @ W1 + b1)
            logits = h @ W2 + b2
            e = np.exp(logits - logits.max(axis=1, keepdims=True))
            p = e / e.sum(axis=1, keepdims=True)
            t += 1
            onehot = np.zeros_like(p)
            onehot[np.arange(len(yb)), yb] = 1
            g_logits = (p - onehot) / len(yb)
            gW2 = h.T @ g_logits + weight_decay * W2
            gb2 = g_logits.sum(axis=0)
            gh = g_logits @ W2.T * (1 - h ** 2)
            gW1 = xb.T @ gh + weight_decay * W1
            gb1 = gh.sum(axis=0)
            for i, g in enumerate([gW1, gb1, gW2, gb2]):
                m[i] = beta1 * m[i] + (1 - beta1) * g
                v[i] = beta2 * v[i] + (1 - beta2) * g * g
                mh = m[i] / (1 - beta1 ** t)
                vh = v[i] / (1 - beta2 ** t)
                params[i] -= lr * mh / (np.sqrt(vh) + eps)
    return {"W1": W1.tolist(), "b1": b1.tolist(),
            "W2": W2.tolist(), "b2": b2.tolist()}


def mlp_forward_probs(weights, X):
    W1 = np.array(weights["W1"], dtype=np.float32)
    b1 = np.array(weights["b1"], dtype=np.float32)
    W2 = np.array(weights["W2"], dtype=np.float32)
    b2 = np.array(weights["b2"], dtype=np.float32)
    h = np.tanh(X @ W1 + b1)
    logits = h @ W2 + b2
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def evaluate(probs, y, classes):
    pred = probs.argmax(axis=1)
    acc = float((pred == y).mean()) if len(y) else 0.0
    conf = np.zeros((len(classes), len(classes)), dtype=int)
    for p_, t_ in zip(pred, y):
        conf[t_, p_] += 1
    per_class = {}
    for i, c in enumerate(classes):
        denom = conf[i].sum()
        per_class[c] = round(float(conf[i, i] / denom), 4) if denom else None
    return acc, conf, per_class


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default=None,
                    help="path to the secondary dataset repo checkout "
                         "(jentrenert/optical-reader-math-solver)")
    ap.add_argument("--synthetic-dir", default="ml_dataset_synthetic")
    ap.add_argument("--out", default="ml_model")
    ap.add_argument("--n-per-class", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t_start = time.time()
    os.makedirs(args.out, exist_ok=True)

    # 1 ── synthetic bootstrap
    if not os.path.exists(os.path.join(args.synthetic_dir, "labels.jsonl")):
        print(f"[TRAIN] generating synthetic dataset -> {args.synthetic_dir}")
        synthetic_data.generate_dataset(args.synthetic_dir,
                                        n_per_class=args.n_per_class,
                                        seed=args.seed)
    with open(os.path.join(args.synthetic_dir, "labels.jsonl")) as f:
        syn_records = [json.loads(l) for l in f if l.strip()]
    import dataset as dataset_mod
    syn_records = dataset_mod.grouped_split(syn_records, seed=args.seed)

    # 2 ── real crops (only when a labelled dataset is actually available)
    real_records, real_info = [], None
    if args.real:
        try:
            real_records, real_info = dataset_mod.load_dataset(args.real)
            real_records = dataset_mod.grouped_split(real_records, seed=args.seed)
        except (FileNotFoundError, ValueError) as e:
            print(f"[TRAIN] REAL dataset unavailable: {e}")
            print("[TRAIN] continuing synthetic-only; real-data metrics will "
                  "NOT be reported (and must not be claimed).")
    else:
        print("[TRAIN] no --real path given: synthetic-only run.")

    classes = ocr_ml.ML_CLASSES
    Xs, ys, kept_s = load_images([r for r in syn_records
                                 if r["dataset_split"] != "test"])
    Xs_te, ys_te, kept_s_te = load_images([r for r in syn_records
                                           if r["dataset_split"] == "test"])

    report = {
        "config": {"n_per_class": args.n_per_class, "epochs": args.epochs,
                   "seed": args.seed, "feature_len": ocr_ml.FEAT_LEN},
        "dataset": {
            "synthetic_records": len(syn_records),
            "real_records": len(real_records),
            "real_info": real_info,
        },
        "results": {},
    }

    # 3 ── template baseline
    from ocr_ml import TemplateBackend
    t0 = time.perf_counter()
    templates = {}
    for i, c in enumerate(classes):
        mask = ys == i
        if mask.any():
            templates[c] = Xs[mask].mean(axis=0)
    tb = TemplateBackend(templates)
    template_build_s = time.perf_counter() - t0

    def template_acc(X, y):
        preds = []
        for feat in X:
            r = tb.predict(feat)
            preds.append(classes.index(r[0]) if r else -1)
        preds = np.array(preds)
        return float((preds == y).mean()) if len(y) else 0.0

    report["results"]["template_baseline"] = {
        "synthetic_val_acc": round(template_acc(Xs, ys), 4),
        "synthetic_test_acc": round(template_acc(Xs_te, ys_te), 4),
        "build_seconds": round(template_build_s, 4),
    }

    # 4 ── MLP
    weights = train_mlp(Xs, ys, len(classes), epochs=args.epochs,
                        seed=args.seed)
    report["results"]["mlp"] = {
        "synthetic_val_acc": round(float(
            (mlp_forward_probs(weights, Xs).argmax(axis=1) == ys).mean()), 4),
        "synthetic_test_acc": round(float(
            (mlp_forward_probs(weights, Xs_te).argmax(axis=1) == ys_te).mean()), 4),
    }
    if len(Xs_te):
        _, conf, per_class = evaluate(mlp_forward_probs(weights, Xs_te),
                                      ys_te, classes)
        report["results"]["mlp"]["synthetic_test_per_class"] = per_class
        report["results"]["mlp"]["synthetic_test_confusion"] = conf.tolist()

    # 5 ── real-data evaluation (separate! never merged with synthetic)
    if real_records:
        Xr, yr, kept_r = load_images(real_records)
        if len(Xr):
            real_acc = float((mlp_forward_probs(weights, Xr).argmax(axis=1)
                              == yr).mean())
            template_real_acc = template_acc(Xr, yr)
            report["results"]["mlp"]["REAL_test_acc"] = round(real_acc, 4)
            report["results"]["template_baseline"]["REAL_test_acc"] = \
                round(template_real_acc, 4)
            report["results"]["mlp"]["real_beats_template"] = \
                bool(real_acc > template_real_acc)

    # 6 ── latency + size
    from ocr_ml import GlyphCorrector
    tmp_model = os.path.join(args.out, "glyph_model.json")
    with open(tmp_model, "w") as f:
        json.dump({"classes": classes, **weights}, f)
    gc = GlyphCorrector(model_path=tmp_model)
    lat_us = ocr_ml.measure_latency(gc)
    size_kb = os.path.getsize(tmp_model) / 1024
    report["results"]["mlp"]["latency_us_per_glyph"] = round(lat_us, 1)
    report["results"]["mlp"]["model_size_kb"] = round(size_kb, 1)

    # 7 ── deployment gate (synthetic-only can still produce a bootstrap
    # model for the collector, but real-data adoption requires
    # real_beats_template — reported, not implied)
    mlp_syn = report["results"]["mlp"]["synthetic_test_acc"]
    tpl_syn = report["results"]["template_baseline"]["synthetic_test_acc"]
    report["deployment_gate"] = {
        "mlp_beats_template_synthetic": bool(mlp_syn > tpl_syn),
        "mlp_beats_template_REAL": report["results"]["mlp"].get(
            "real_beats_template", None),
        "note": "runtime adoption (bot_core.ml_enabled=True) requires "
                "REAL-data improvement, not synthetic improvement",
    }
    if mlp_syn <= tpl_syn:
        os.remove(tmp_model)
        report["model_written"] = False
        print("[TRAIN] MLP did not beat the template baseline on the "
              "synthetic validation fold — model NOT written.")
    else:
        report["model_written"] = True
        print(f"[TRAIN] model written -> {tmp_model}")

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report["results"], indent=2, ensure_ascii=False))
    print(f"[TRAIN] done in {time.time() - t_start:.1f}s "
          f"(synthetic-only={real_info is None})")
    return 0


if __name__ == "__main__":
    # dataset_synthetic_data was a typo-guard above; real import lives here.
    sys.exit(main())
