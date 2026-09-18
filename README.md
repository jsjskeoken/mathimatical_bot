# Optical Reader & Math Solver (`mathimatical_bot`)

A Windows screen-automation bot that watches a region of the screen, OCRs an
on-screen math question, solves it, and (optionally) clicks the answer onto
the on-screen keypad. It is built around a layered question state machine so
that animated screens, OCR flicker and repeated frames cannot cause duplicate
submissions, click-spam, or permanently stuck questions.

## Platform requirements

- **Windows only.** Capture, click automation and DPI awareness go through
  the Windows API directly (`ctypes.windll`); it will not run unmodified on
  macOS/Linux. (The test suite runs anywhere; the app itself needs Windows.)
- **Python 3.10–3.13.** `opencv-python` does not ship prebuilt 3.14 wheels at
  the time of writing, so newer interpreters fall back to a from-source
  build and usually fail. `launch_solver.bat` looks for 3.10–3.13 in that
  order (preferring 3.12).

## Install & run

```bash
pip install --user easyocr opencv-python numpy sympy pyautogui mss pillow pynput certifi
python gui.py
```

Or just run **`launch_solver.bat`** — it finds a usable Python 3.10–3.13
install and installs missing dependencies automatically, retrying with
`--user` if the system-wide install hits a permissions error.

- `certifi` is optional: EasyOCR's first run downloads its recognition model
  over HTTPS, and on machines with a stale certificate store the download
  fails without it. If present it is used automatically; if absent nothing
  changes.
- If imports stay broken after a `--user` install (a broken copy sitting in
  the global site-packages can shadow your own), force a clean user copy:
  `pip install --user --force-reinstall easyocr opencv-python numpy sympy pyautogui mss pillow pynput certifi`.
- `F8` (global hotkey, via `pynput`) toggles pause/resume.

## Pipeline

Per cycle, `bot_core.py` runs:

1. **Capture** — `mss` grabs the configured question region as a raw BGRA
   array.
2. **Preprocess** — greyscale straight from the raw capture, Gaussian blur +
   unsharp mask, cached CLAHE pass, Otsu binarization. No upscaling.
3. **OCR** — EasyOCR reads the processed frame (GPU if available, CPU
   fallback).
4. **Candidate selection** — if the box caught stray tokens (dates, labels,
   fragments), the most plausible expression is picked from the OCR tokens:
   sorted by bounding box into reading order, filtered by enabled operators,
   digit-group shape and date-shape, then the shortest span that actually
   solves wins. A `+` that OCR reports is only converted to `/` when the
   pixels under it show the two-dot `÷` glyph shape (connected-blob check on
   the token's own bounding box); ambiguous cases stay as `+`.
5. **Normalization** — operators unified (`×`/`x`/`X` → `*`, `÷`/`:` → `/`),
   common letter/digit misreads corrected (`o`→`0`, `s`→`5`, `B`→`8`, …),
   hallucinated characters stripped, result collapsed into a canonical
   expression string.
6. **Solve** — checked in this order: **LUT** (persistent, on disk) →
   **session cache** (in-memory) → **`eval()`** for plain arithmetic →
   **SymPy** for equations/expressions with a variable. New solves are
   written back to the LUT.
7. **Optional click** — if answer clicking is enabled, the answer is typed
   onto the on-screen keypad via the active coordinate profile, with
   verified cursor placement and small delays between keys.

Three solver modes: **Hybrid** (default; LUT + cache + solver, saves back to
the LUT), **Calc only** (always recomputes, never touches the LUT), and
**LUT only** (answers only from cache/LUT, skips unknown questions). Capture
mode trades polling interval for OCR load: **Fast** = 10 ms, **Standard** =
150 ms.

## Question identity, retries and safety nets

The old design equated "frame hash changed" with "question changed", which
silently suppressed retries on unchanged frames and duplicated work on
animated ones. `question_state.py` now separates the layers explicitly
(pure stdlib, unit-testable on any OS):

- **Visual state** — an exact BLAKE2b pixel digest, used *only* for click
  confirmation and freshness. Never used to suppress retries.
- **Question identity** — a semantic fingerprint over the *canonical
  expression* (plus enabled ops, mode, capture region). Drives caches and
  retry scheduling.
- **Unresolved visual identity** — when OCR produces no canonical expression
  at all, identity comes from the question-region *pixels*, never from the
  raw garbage text: a 96×16 block-mean signature of the preprocessed image
  (`bot_core.visual_signature()`), compared with a polarity-normalized,
  dead-zoned L1 distance with a small alignment search
  (`question_state.signature_distance()`). Small visual noise, animation
  jitter and blink stay inside the same unresolved episode (and the same
  bounded retry budget); a genuinely different question starts a new episode
  with a fresh budget. Calibrated separation: same-question noise ≤ 0.0043,
  one-digit change ≥ 0.0090, full question change ≥ 0.0171, match threshold
  0.0065 (details in `docs/REDESIGN_REPORT.md` §2.5).
- **Retry state** — time-based, bounded, per question: first retry after
  0.25 s, then 0.5 s base with ×2 backoff capped at 4 s, at most 8
  processing/click retries per question. An unchanged frame is *always*
  revalidated on schedule — retries are never suppressed just because the
  pixels stayed the same.
- **Click confirmation** — after a click, the pixel digest must change
  within 0.5 s or the click counts as unconfirmed. **Three consecutive
  unconfirmed clicks trip the safety net**: answer clicking disables itself
  (and in-flight scheduled AUTO actions are cancelled) instead of clicking
  blind. A confirmed answer suppresses re-clicks of the same question
  (1 s cooldown + 2 s off-screen reappearance rule).

The frame-level answer cache carries the question fingerprint and a 30 s TTL,
so stale entries expire and a cache hit can never re-submit a completed
question.

## GUI (`gui.py`)

- **Live preview keeps running while paused.** The single capture loop feeds
  the pixel digest, the overlays and the preview every cycle; pause gates
  only the solver. Resuming processes immediately.
- **`Answer clicks` and `AUTO sequence` are two independent switches.**
  Answer clicks control keypad submission; the AUTO sequence is the optional
  delayed 1/2/3 click sequence. The unconfirmed-click safety net disables
  answer clicking specifically (cancelling scheduled AUTO actions as a
  documented one-way safety rule) without touching your AUTO setting.
- Draggable/resizable region overlays for the question area and keypad; 3
  numbered coordinate profile slots saved to `optical_coords.json`.
- **Verify LUT** button — re-checks every cached answer against the real
  solver, auto-corrects wrong-but-valid entries, removes non-integer ones.
- **Known operations** checkboxes (Advanced) — `+ − × ÷` filters which
  operators a candidate may use during selection; disabling one never
  converts it into another.
- **Save OCR captures** (Advanced, **off by default**) — writes the original
  + processed image for each solved question to `ocr_captures/` for
  diagnosing OCR misreads. Off by default so a long unattended run does not
  accumulate images without bound.
- Cache and LUT hit counters; pause/resume via `F8`.

## Supported math operations

- Integer arithmetic with `+ − * /` (on-screen `×`, `x`, `÷`, `:` are
  normalized to the same operators).
- Equations / expressions containing a variable are solved with SymPy
  (e.g. `4x4=16`-style missing-operand forms, `?` placeholders).
- Only integer answers are submitted; non-integer or negative results are
  computed and shown but not clicked (see limitations).

## Optional ML glyph corrector (`ocr_ml.py`) — disabled by default

A per-glyph corrector that can rescue low-confidence OCR tokens (template
matching or a tiny NumPy MLP backend). It is **off by default**
(`ml_enabled = False`) and is heavily gated when enabled: a correction is
accepted only when the token confidence is low, the backend is confident,
the replacement is in the solver's grammar, and the result still solves.

- `dataset.py` — append-only JSONL collector + glyph-crop-only capture
  (crops, never full screenshots), a human-labelling path, grouped splits,
  and a loader for the secondary dataset repository
  (`jentrenert/optical-reader-math-solver`).
- `synthetic_data.py` — deterministic synthetic glyph generator. **Synthetic
  data is for bootstrapping and testing the plumbing only** — it is not a
  substitute for real OCR data and says nothing about real-world accuracy.
- `train_glyph_model.py` — train/eval CLI with a deployment gate: a model
  that does not beat the template baseline is not written to disk.
- **Real labelled OCR data is still required before enabling ML.** No
  real-image dataset has been trained on yet; generated model/dataset
  artefacts (`ml_model/`, `ml_dataset*/`) are local build outputs and are
  gitignored.

## Tests & benchmarks

```bash
python -m pytest tests/ -q            # 115 tests
python benchmarks/benchmark_ocr.py --baseline 51b16f4
python benchmarks/benchmark_retry.py
```

- `tests/` — 115 tests: the preserved OCR regression corpus (17 fixes),
  retry-engine timing, GUI loop behaviour (pause/preview/switch
  independence/safety net), ML gates, the unresolved-identity suite, and
  the benchmark tooling's Windows-safety contract (UTF-8 git output) and
  temp-hygiene guarantee (nothing temporary ever touches the working
  tree).
  Windows-specific pieces (`easyocr`, `mss`, `pynput`, `pyautogui`) are
  stubbed, so the suite runs on any OS.
- `benchmarks/benchmark_ocr.py` — a **65-case synthetic OCR token corpus**
  (rendered/glued/misread token fixtures, not screenshots). Current tree:
  **65/65** vs **45/65** at baseline `51b16f4`. **This is a synthetic
  benchmark of the text-cleanup stages — it is not a real-world OCR
  accuracy figure.** Raw logs in `docs/benchmark-logs/`.
- `benchmarks/benchmark_retry.py` — 8 virtual-clock scenarios (frozen
  screen, animation, OCR flicker, question change, garbage flicker, …)
  comparing the old loop reconstruction against the current one, e.g.
  animation duplicate submissions 39 → 1, alternating-wrong submissions
  59 → 2, and one constant identity for unreadable garbage → correct
  per-question episodes.

## Known limitations

- **Negative answers can't be submitted** — the on-screen keypad has no
  minus key. The answer is computed and shown, the click is skipped
  (`[CORE] [SKIP] '-N' has unmapped chars`).
- **Clicking requires the target window focused** — otherwise the click is
  skipped rather than sent to the wrong place
  (`[CORE] [SKIP] Target window not focused`).
- **Repeated unconfirmed clicks auto-disable answer clicking** via the
  safety net described above (`[GUI] [WARN] Click unconfirmed`).
- **Coordinate profiles are machine/layout specific.** Defaults were tuned
  against one specific app layout and resolution; they will not line up
  with a different setup. Capture your own question/keypad regions with the
  overlays (and `logger.py` for raw coordinates) and save a profile.
- **Real-world validation is still required.** Everything above was
  developed and verified in a sandboxed environment with stubbed capture and
  input; recognition quality, click reliability and overlay rendering on a
  real Windows machine still need a validation pass, and the retry/TTL
  constants may want tuning against real app timings.
- Screen automation like this may violate the terms of service of whatever
  application it is pointed at; treat it as a reference implementation for
  the techniques.

## Files

| File | Purpose |
|---|---|
| `gui.py` | Tkinter GUI — overlays, controls, main loop |
| `bot_core.py` | OCR, candidate selection, normalization, solving, caching, LUT, click automation, visual signatures |
| `question_state.py` | Layered question/retry/click state machine (pure stdlib) |
| `ocr_ml.py` | Optional ML glyph corrector (disabled by default) |
| `dataset.py` | JSONL dataset collector + real-data loader |
| `synthetic_data.py` | Synthetic glyph generator (bootstrapping/testing only) |
| `train_glyph_model.py` | Train/eval CLI with deployment gate |
| `backup.py` | Standalone legacy single-file implementation, kept for reference |
| `logger.py` | Standalone mouse-position logger, useful for finding coordinates |
| `optical_lut.json` | Persistent expression → answer lookup table (runtime-maintained cache) |
| `optical_coords.json` | Saved coordinate profiles (machine-specific) |
| `launch_solver.bat` | Windows launcher — finds Python 3.10–3.13, installs missing deps |
| `tests/`, `benchmarks/`, `docs/` | Test suite, reproducible benchmarks, design report + raw benchmark logs |
