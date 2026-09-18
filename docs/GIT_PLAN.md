# GIT PLAN — how to get this into `jsjskeoken/mathimatical_bot`

The agent did **not** commit or push anything. You are the committer. Everything below
assumes you extracted the download package **over a clean clone at `093fd3c`** (or later),
e.g.:

```
git clone https://github.com/jsjskeoken/mathimatical_bot
cd mathimatical_bot
git checkout 093fd3c          # the revision this work was built on
# extract mathimatical_bot-upload.zip here (merge/overwrite when asked)
```

## Step 0 — review before you commit

```
git status --short                # expect: 5 modified files + new untracked files
git diff --stat                   # size of the change in modified files
```

The five unified review diffs (bot_core, gui, .gitignore, CHANGELOG, README) are
**not shipped inside the ZIP** — the ZIP mirrors the exact commit list, so
extracting it can never leave a stray untracked `patches/` folder. Download them
from the download site's **Patches (review diffs)** section instead and open the
files locally if you want to review with `less` / `type`.

The five diffs cover every *modified* file so you can review the exact change without
trusting the shipped full copies. The full copies in the package root **are** the
working tree that passed the tests — if you prefer `git apply`, the downloaded diffs
reproduce them.

## Step 1 — verify on your machine

```
python -m pytest tests/ -q
```

Expected: `115 passed` (requires `pip install numpy opencv-python sympy pillow
scikit-learn pytest`; easyocr/mss/pynput/pyautogui are stubbed in tests, but the real
app still needs them at runtime).

Benchmarks (optional, reproducible):

```
python benchmarks/benchmark_ocr.py --baseline 51b16f4   # synthetic corpus: 45/65 -> 65/65
python benchmarks/benchmark_retry.py                    # 8 virtual-clock scenarios
```

(The OCR corpus is a synthetic token benchmark of the cleanup stages — it is
not a real-world OCR accuracy figure.)

## Step 2 — commit in three logical units

```
:: (a) core state machine + retry engine
git add question_state.py bot_core.py .gitignore
git commit -m "Add layered question state machine and bounded retry engine

- exact-frame digest (BLAKE2b) now only confirms clicks; question identity
  is a semantic fingerprint over the canonical expression
- time-based RetryPolicy (0.25 s base, x2 backoff, attempt cap) replaces
  the frame-hash gate that permanently suppressed unchanged frames
- frame_answer_cache replaced by a TTL cache carrying the fingerprint
- fixes dead B->8 mapping and SymPy Float integrality bug in solve_math
- auto-sequence scheduling safe in UI-less runs
- unresolved visual-question identity: unreadable frames are grouped into
  pixel-evidence episodes (visual_signature + signature_distance), so OCR
  garbage flicker can neither mint fresh retry budgets nor share one
  constant identity; unchanged frames still revalidate on schedule"

:: (b) GUI decoupling + independent controls
git add gui.py
git commit -m "Decouple preview from pause and split automation controls

- single capture loop: digest + preview run paused or not; solver gated on
  pause state and the retry engine
- separate 'Answer clicks' and 'AUTO sequence' switches; the unconfirmed-
  click safety net disables clicking and cancels scheduled AUTO actions
  without flipping the AUTO setting
- removes raw-text last_question debounce (OCR identity now lives in the
  state machine); loop errors log full tracebacks"

:: (c) ML layer, data pipeline, tests, benchmarks, docs
git add ocr_ml.py dataset.py synthetic_data.py train_glyph_model.py tests/ benchmarks/ docs/ CHANGELOG.md README.md
git commit -m "Add ML layer, data pipeline, tests, benchmarks and current README

- ocr_ml.py: template + tiny MLP backends, evidence-only gates, ships
  disabled (ml_enabled=False) pending real labelled data
- dataset.py: JSONL collector, glyph-crop-only capture, grouped splits,
  loader for the secondary dataset repository
- 115 tests incl. all 17 preserved OCR fixes + the unresolved-identity
  suite; two reproducible benchmarks (65-case synthetic OCR corpus vs any
  git revision; 8 virtual-clock retry scenarios)
- README.md rewritten from the current code (pipeline, state machine,
  retry/backoff, controls, ML status, synthetic-benchmark labelling,
  known limitations)"
```

## Step 3 — push

```
git push origin master            # or 'main' — match your branch
```

## What must NOT be committed

| Path | Why |
|------|-----|
| `ml_model/` | generated model artefacts (gitignored) |
| `ml_dataset_synthetic/` | generated synthetic dataset (gitignored) |
| `ml_dataset/`, `bench_lut.json`, `__pycache__/` | generated (gitignored) |
| `optical_lut.json` | runtime cache state; benchmark/test runs can append entries — restore with `git checkout -- optical_lut.json` before committing |
| `README-UPLOAD.txt` | must not exist in the repo (kept out of the package) |

If `git status` shows `optical_lut.json` modified after a local run, restore it with
`git checkout -- optical_lut.json`.

## Suggested follow-ups (separate commits, later)

1. Capture real glyph crops on your Windows machine (`dataset.py` collector), label them,
   push them to `jentrenert/optical-reader-math-solver`, retrain, and only then consider
   `ml_enabled=True` (deployment gate must beat the template baseline on real data).
2. Tune `SAME_FRAME_RETRY_DELAY` / `MAX_RETRIES` / TTLs against real app timings if S1/S7
   behaviour feels too eager or too slow in live use.
3. Record a real-screen validation session (the risk section of `docs/REDESIGN_REPORT.md`
   lists what to watch for).
