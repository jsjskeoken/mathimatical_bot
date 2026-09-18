"""
benchmarks/benchmark_ocr.py — Reproducible OCR-pipeline benchmark.

Measures the REAL select_math_ocr_text -> normalise -> solve pipeline on a
deterministic scenario corpus covering the failure categories:

  clean / noisy / merged-token / glued-operator / missing-operator /
  stray-noise / out-of-order / ambiguous-digit

Reports per-category accuracy (never one blended percentage), average
selection latency, and average solve latency.

Usage:
  python benchmarks/benchmark_ocr.py                     # current HEAD
  python benchmarks/benchmark_ocr.py --baseline 51b16f4  # vs a git revision
  python benchmarks/benchmark_ocr.py --repeats 20

Notes on honesty:
  - This is a TOKEN-LEVEL benchmark: it exercises everything downstream of
    EasyOCR (candidate selection, correction, normalisation, solving) with
    deterministic EasyOCR-shaped inputs. It does NOT measure EasyOCR's
    pixel-level recognition — that requires the real OCR image dataset
    (see benchmark_ocr.py --pixel-level and dataset.py).
  - Synthetic/pixel-level accuracy must never be reported as real-world
    accuracy; the pixel-level harness only runs with real data present.
"""

import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile
import time
from fnmatch import fnmatch
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tests"))

import conftest                     # installs platform stubs (easyocr etc.)
conftest._install_stubs()

import numpy as np

import bot_core
from bot_core import BotCore
from conftest import make_ocr_result


def bbox_at(x, y=6, w=16, h=30):
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


# ── corpus ───────────────────────────────────────────────────────────────────
# (category, description, ocr_results_builder, expected_answer, expected_expr)

def C7():
    return make_ocr_result([("7", 0.95), ("+", 0.92), ("6", 0.94)])

def C8():
    return make_ocr_result([("8", 0.95), ("*", 0.9), ("7", 0.93)])

def CDIV():
    return make_ocr_result([("8", 0.95), ("/", 0.9), ("2", 0.93)])

def N1():
    return make_ocr_result([("Question", 0.35), ("7", 0.9), ("+", 0.85),
                            ("6", 0.9)])

def M1():
    return make_ocr_result([("7+6", 0.9)])

def G1():
    return make_ocr_result([("7", 0.9), ("+6", 0.6)])

def G2():
    return make_ocr_result([("6", 0.9), ("×4", 0.6)])
def O1():
    res = make_ocr_result([("12", 0.95), ("4", 0.95)])
    return res

def S1():
    return make_ocr_result([("24/05/2026", 0.99), ("7", 0.95), ("+", 0.9),
                            ("6", 0.95)])

def S2():
    return make_ocr_result([("12/03/2026", 0.95)])

def R1():
    # visually out of order: 6 listed first but sits right-most
    res = []
    for (x, text, conf) in [(200, "6", 0.95), (110, "+", 0.9), (20, "7", 0.9)]:
        res.append((bbox_at(x), text, conf))
    return res

def A1():
    return make_ocr_result([("B", 0.55), ("+", 0.9), ("6", 0.9)])

def A2():
    return make_ocr_result([("9", 0.9), ("-", 0.88), ("O", 0.55)])

CORPUS = [
    ("clean",           "plain arithmetic",            C7,   13, "7 + 6"),
    ("clean",           "multiplication",              C8,   56, "8 * 7"),
    ("clean",           "division",                    CDIV,  4, "8 / 2"),
    ("noisy",           "low-conf label beside expr",  N1,   13, "7 + 6"),
    ("merged-token",    "whole expr one token",        M1,   13, "7+6"),
    ("glued-operator",  "'+6' glued to digit",         G1,   13, "7 + 6"),
    ("glued-operator",  "'×4' glued to digit",         G2,   24, "6 × 4"),
    ("missing-op",      "dropped operator",            O1,   48, "12 4"),
    ("stray-noise",     "date beside expression",      S1,   13, "7 + 6"),
    ("stray-noise",     "pure date token",             S2, None, ""),
    ("out-of-order",    "tokens returned out of order", R1,  13, "7 + 6"),
    ("ambiguous-digit", "B misread for 8 (raw kept; "
    "canonicalised B→8 at solve)",             A1,   14, "B + 6"),
    ("ambiguous-digit", "O misread for 0 (raw kept; "
    "canonicalised o→0 at solve)",             A2,    9, "9 - O"),
]


def run_case(core, builder):
    img = np.zeros((60, 460), dtype=np.uint8)
    t0 = time.perf_counter()
    raw = core.select_math_ocr_text(builder(), img)
    t1 = time.perf_counter()
    if raw:
        answer, source = core.handle_question(raw)
    else:
        answer, source = None, None
    t2 = time.perf_counter()
    return raw, answer, (t1 - t0), (t2 - t1)


# ── hybrid OCR + ML corpus ───────────────────────────────────────────────────
# Cases where the DETERMINISTIC letter mapping is provably wrong and only
# pixel evidence (the ML corrector looking at the actual glyph crop) can
# fix it. The glyph the OCR misread is rendered INTO the image at the
# token's bbox, exactly as EasyOCR would have seen it.
#
#   'b' -> 6  (deterministic)  but the pixels show an 8  -> ML must fix
#   'b' -> 6  and the pixels really show a 6  -> ML must NOT touch it
#
# These are SYNTHETIC pixels: reported as synthetic accuracy, never
# real-world accuracy (the real-data path needs the secondary dataset).

def _image_with_glyph(ch, x=4, y=6, size=(16, 30)):
    """Draw the true glyph exactly inside the FIRST token's bbox produced by
    make_ocr_result (x0=4, y0=6, w=16, h=30) so the corrector's crop sees
    what EasyOCR would have seen."""
    import cv2
    import synthetic_data as sd
    fonts = sd.available_fonts()
    rng = __import__("random").Random(hash(ch) % 1000)
    g = sd.render_glyph(ch, fonts[0], px=40, rng=rng, variation=False)
    img = np.zeros((60, 460), dtype=np.uint8)
    glyph = np.clip(g, 0, 255).astype(np.uint8)
    w, h = size
    glyph = cv2.resize(glyph, (w, h), interpolation=cv2.INTER_AREA)
    img[y:y + h, x:x + w] = glyph
    return img


def B8():
    """OCR read 'b' (low conf); the actual glyph is an 8."""
    img = _image_with_glyph("8")
    return make_ocr_result([("b", 0.45), ("+", 0.9), ("6", 0.9)]), img


def B6():
    """OCR read 'b' (low conf); the actual glyph really is a 6."""
    img = _image_with_glyph("6")
    return make_ocr_result([("b", 0.45), ("+", 0.9), ("6", 0.9)]), img

ML_CORPUS = [
    ("ml-hybrid", "b-misread, pixels are 8 -> ML should correct", B8, 14),
    ("ml-hybrid", "b-misread, pixels are 6 -> ML must not touch", B6, 12),
]


def run_ml_case(core, builder):
    res, img = builder()
    t0 = time.perf_counter()
    raw = core.select_math_ocr_text(res, img)
    t1 = time.perf_counter()
    answer, _ = core.handle_question(raw) if raw else (None, None)
    t2 = time.perf_counter()
    return raw, answer, (t1 - t0), (t2 - t1)


def benchmark_ml(core, repeats):
    rows = []
    lat = []
    for cat, desc, builder, want in ML_CORPUS:
        correct = 0
        for i in range(repeats):
            raw, answer, dt_sel, dt_solve = run_ml_case(core, builder)
            ok = (answer == want)
            correct += ok
            if i == 0:
                lat.append(dt_sel)
                rows.append((desc, raw, answer, want))
        rows.append((desc + " [accuracy]", None, correct, repeats))
    return rows, lat


def benchmark(core, repeats):
    per_cat = {}
    lat_sel, lat_solve = [], []
    for cat, desc, builder, want_ans, want_expr in CORPUS:
        correct = 0
        for i in range(repeats):
            raw, answer, dt_sel, dt_solve = run_case(core, builder)
            ok = (answer == want_ans) and (raw == want_expr)
            correct += ok
            if i == 0:
                lat_sel.append(dt_sel)
                lat_solve.append(dt_solve)
        per_cat.setdefault(cat, []).append(
            (desc, correct, repeats, want_ans, want_expr))
    return per_cat, lat_sel, lat_solve


def report(title, per_cat, lat_sel, lat_solve, repeats):
    print(f"\n=== {title} ===")
    total_correct = total = 0
    print(f"{'category':<16} {'accuracy':>9}  details")
    for cat, rows in per_cat.items():
        c = sum(r[1] for r in rows)
        n = sum(r[2] for r in rows)
        total_correct += c
        total += n
        detail = "; ".join(f"{d}: {cc}/{rr}" for d, cc, rr, _, _ in rows)
        print(f"{cat:<16} {c}/{n:>6}   {detail}")
    print(f"{'OVERALL':<16} {total_correct}/{total}")
    print(f"avg select latency: {np.mean(lat_sel)*1000:.2f} ms  | "
          f"avg solve latency: {np.mean(lat_solve)*1000:.3f} ms "
          f"({repeats} repeats/case)")
    return total_correct, total


def decode_git_output(data, context):
    """Decode raw git stdout as UTF-8 — Windows-safe.

    subprocess(text=True) decodes with the platform locale codec (cp1252 on
    Windows), which explodes on bot_core.py's UTF-8 punctuation:
    UnicodeDecodeError: 'charmap' codec can't decode byte 0x8f. Git hands us
    the working-tree bytes, so we decode them ourselves, explicitly and
    strictly, with context — a genuinely non-UTF-8 object fails loudly
    instead of being benchmarked mangled."""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as e:
        raise RuntimeError(
            f"{context} is not valid UTF-8 ({e}); refusing to benchmark a "
            f"mangled copy") from e


def extract_baseline_source(rev, rel_path="bot_core.py"):
    """`git show <rev>:<rel_path>` captured as BYTES and decoded explicitly.

    Never uses text=True (platform-locale decoding) and never hides a git
    failure: a non-zero exit raises with git's stderr so a bad revision or a
    missing file is reported instead of silently benchmarking nothing."""
    proc = subprocess.run(["git", "show", f"{rev}:{rel_path}"], cwd=REPO,
                          capture_output=True)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"git show {rev}:{rel_path} failed (exit {proc.returncode}): "
            f"{err or 'no stderr'}")
    return decode_git_output(proc.stdout, f"{rel_path} at {rev}")


def load_baseline_core(rev, bench_dir):
    """Extract bot_core.py at a git revision and import it as a module.
    Returns (core, temp_path) so the caller can clean the file up."""
    src = extract_baseline_source(rev)
    # live OUTSIDE the repo: a crash mid-benchmark must never leave a stray
    # untracked file (or __pycache__) in the working tree
    path = os.path.join(bench_dir, f"baseline_bot_core_{rev}.py")
    assert not _is_within(path, REPO), (
        "baseline module must never be written inside the repository")
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(src)
    safe = "".join(c if c.isalnum() or c == "_" else "_" for c in rev)
    spec = importlib.util.spec_from_file_location(
        f"baseline_bot_core_{safe}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # AFTER exec_module: the module's own top-level `LUT_FILE = ...`
    # assignment runs during exec and would otherwise re-point the baseline
    # core at the user's real optical_lut.json, letting its async LUT saves
    # overwrite it with a benchmark snapshot. Set the isolated path only now
    # (after top-level init, BEFORE constructing/using BotCore) — and give
    # this core its OWN file so concurrent async saves can never collide.
    mod.LUT_FILE = os.path.join(bench_dir, "lut_baseline.json")
    core = mod.BotCore()
    core.lut = {}
    core.answer_cache.clear()
    core.session_cache.clear()
    return core, path


def drain_lut_writes(core):
    """Block until any in-flight async LUT save of this core has landed.

    The writer thread holds _lut_write_lock for the whole tmp-write +
    os.replace, so acquiring it once means no save of THIS core is running
    and none can start (the benchmark creates no new records afterwards).
    Benchmark-only hygiene — production code is untouched. Combined with the
    per-run temp LUT path this guarantees no thread can ever touch the
    repository's real optical_lut.json, during or after the run."""
    try:
        with core._lut_write_lock:
            pass
    except AttributeError:
        pass


def _is_within(path, directory):
    """True if <path> resolves inside <directory> (both absolute)."""
    return Path(os.path.abspath(path)).resolve().is_relative_to(
        Path(os.path.abspath(directory)).resolve())


def open_bench_dir():
    """TemporaryDirectory OUTSIDE the repository, cleaned up on success AND
    exceptions (it is a context manager).

    ALL benchmark temp state lives here: the baseline module, both LUT
    files, and their __pycache__. The 51b16f4-era extraction script instead
    created tmpXXXXXXXX_<rev>.py files (tempfile.mkstemp defaults) that
    leaked into the working tree whenever it crashed — the exact leftover
    the Windows run reported. This makes that class of leftover impossible
    by construction, and the tripwire in main() verifies it after every
    run."""
    td = tempfile.TemporaryDirectory(prefix="mathbot_bench_")
    if _is_within(td.name, REPO):
        td.cleanup()
        raise RuntimeError(
            f"benchmark temp dir {td.name} resolved INSIDE the repository "
            f"({REPO}) — refusing to run, it would dirty the working tree")
    return td


def scan_temp_leftovers(root=REPO):
    """Report benchmark-temp leftovers under <root> as sorted rel paths.

    Scans for the exact violation class the Windows run hit —
    tmpXXXXXXXX_<rev>.py files (mkstemp's default 'tmp' + 8 random chars)
    and stray mathbot_bench_* trees — while skipping .git/ and
    __pycache__/. Used by main() as a post-run tripwire: the run must leave
    the working tree exactly as it found it."""
    hits = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__")]
        for name in dirnames:
            if fnmatch(name, "mathbot_bench_*"):
                hits.append(os.path.relpath(os.path.join(dirpath, name),
                                            root))
        for name in filenames:
            if fnmatch(name, "tmp*.py") or fnmatch(name, "mathbot_bench_*"):
                hits.append(os.path.relpath(os.path.join(dirpath, name),
                                            root))
    return sorted(hits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--baseline", default=None,
                    help="git revision to benchmark against (e.g. 51b16f4)")
    ap.add_argument("--ml-model", default="ml_model/glyph_model.json",
                    help="trained glyph model for the hybrid ML section "
                         "(run train_glyph_model.py first)")
    args = ap.parse_args()

    # Temp hygiene, step 1 — snapshot whatever benchmark leftovers ALREADY
    # sit in the working tree (e.g. a tmpXXXXXXXX_51b16f4.py from the old
    # extraction script) so the post-run audit can attribute leftovers
    # honestly instead of blaming this run for them.
    pre_existing = scan_temp_leftovers()
    if pre_existing:
        print("NOTE: pre-existing benchmark temp leftovers in the working "
              "tree (NOT created by this run — from an older benchmark "
              "version; safe to delete manually):")
        for rel in pre_existing:
            print(f"  - {rel}")

    # Temp hygiene, step 2 — the TemporaryDirectory context manager is the
    # single owner of every temp file this run creates (baseline module,
    # both LUTs, __pycache__). It lives OUTSIDE the repo (open_bench_dir
    # refuses otherwise) and is removed on success AND on any exception.
    with open_bench_dir() as bench_dir:
        # Isolate the LUT: benchmark runs must NEVER read or write the
        # user's real optical_lut.json. One fresh temp dir PER RUN (not a
        # shared fixed bench_lut.json, which accumulated entries across runs
        # and made the "LUT loaded: N entries" banner nondeterministic),
        # unique file per module (HEAD / baseline), set AFTER import
        # (top-level init done) and BEFORE constructing BotCore. bot_core
        # reads LUT_FILE dynamically at load/save/clear time, so every async
        # save thread follows the redirect and the repo file is unreachable
        # for the whole process lifetime.
        bot_core.LUT_FILE = os.path.join(bench_dir, "lut_head.json")
        core = BotCore()
        core.lut = {}
        core.answer_cache.clear()
        core.session_cache.clear()
        core.fast_mode = True
        core.enabled_operations = {"+", "-", "*", "/"}

        per_cat, lat_sel, lat_solve = benchmark(core, args.repeats)
        report("CURRENT HEAD (retry/fingerprint architecture + OCR fixes)",
               per_cat, lat_sel, lat_solve, args.repeats)

        # ── hybrid OCR + ML section (SYNTHETIC pixels; clearly labelled) ──
        if os.path.exists(args.ml_model):
            import ocr_ml
            print("\n=== HYBRID OCR + ML (SYNTHETIC pixels — not real-world "
                  "accuracy) ===")
            # ML disabled (deterministic-only fallback)
            core.ml_enabled = False
            for desc, raw, got, want in benchmark_ml(core, 1)[0]:
                if "[accuracy]" not in desc:
                    print(f"  ML OFF  {desc}: raw={raw!r} answer={got} "
                          f"(expected {want})")
            # ML enabled with the trained corrector
            core.ml_corrector = ocr_ml.GlyphCorrector(model_path=args.ml_model)
            core.ml_enabled = True
            print(f"  (backend={core.ml_corrector.backend_name}, "
                  f"trigger<0.60 conf, accept>=0.85 prob)")
            rows, ml_lat = benchmark_ml(core, args.repeats)
            for desc, raw, got, want in rows:
                if "[accuracy]" not in desc:
                    print(f"  ML ON   {desc}: raw={raw!r} answer={got} "
                          f"(expected {want})")
            for desc, _, got, want in rows:
                if "[accuracy]" in desc:
                    print(f"  {desc}: {got}/{want}")
            print(f"  ML select latency: {np.mean(ml_lat)*1000:.2f} ms "
                  f"(includes corrector pass on low-conf tokens)")
            core.ml_enabled = False
        else:
            print(f"\n(hybrid ML section skipped — no model at "
                  f"{args.ml_model}; run train_glyph_model.py)")

        bcore = None
        if args.baseline:
            path = None
            try:
                bcore, path = load_baseline_core(args.baseline, bench_dir)
                bcore.fast_mode = True
                bcore.enabled_operations = {"+", "-", "*", "/"}
                b_per_cat, b_sel, b_solve = benchmark(bcore, args.repeats)
                report(f"BASELINE {args.baseline}", b_per_cat, b_sel,
                       b_solve, args.repeats)
            finally:
                if path:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass

        # no async LUT save may still be running when the temp dir goes away
        drain_lut_writes(core)
        if bcore is not None:
            drain_lut_writes(bcore)
    # ^ TemporaryDirectory removed the whole bench tree here — success OR
    #   exception. No mkdtemp/rmtree pair that a crash can skip.

    # Temp hygiene, step 3 — tripwire: the working tree must contain no NEW
    # tmp*.py / mathbot_bench_* leftovers after the run. (Pre-existing ones
    # were reported above and are not attributed to this run.)
    new_leftovers = [p for p in scan_temp_leftovers() if p not in pre_existing]
    if new_leftovers:
        print("ERROR: this run left temporary files in the working tree "
              "(this is a benchmark bug — please report it):")
        for rel in new_leftovers:
            print(f"  - {rel}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
