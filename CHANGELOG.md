# Changelog

## BUG-5 regression threshold made platform-robust (test-only change)

### Changed
- `tests/test_forensic_audit.py::test_bug5_ring_scan_worst_case_bounded`
  no longer asserts the audit machine's absolute speed (`median < 3.0 ms`).
  The first Windows verification run measured **3.52 ms** — a legitimate
  pass behaviourally (2.8× inside the 10 ms fast-poll budget), but a red
  test, because the old threshold accidentally encoded the Linux
  benchmark environment rather than the architectural requirement.
- The test now encodes the documented BUG-5 contract explicitly:
  **hard limit = the 10 ms `FAST_MODE_POLLING` budget itself**
  (`bot_core.py`) — detection margin 2.6× against the original ~26 ms
  stdlib regression, headroom 2.8× against the slowest platform measured
  so far — plus a **5.0 ms soft margin** (half the budget) that still
  passes but emits a `RuntimeWarning`, so a ~2× platform slowdown shows
  up in the pytest summary before it can threaten the contract. The
  threshold rationale, all measured medians (stdlib ~26 ms, Linux numpy
  ~0.93 ms, Windows numpy ~3.52 ms) and the sampling protocol are
  documented in the test's docstring and
  `docs/GLM_FORENSIC_AUDIT.md` §2 BUG-5.
- Sampling hardened: 3 warm-up iterations, then the median of 30 timed
  8-ring all-no-match scans (median is robust to one-off OS scheduler/GC
  spikes); the failure message now reports min/median/max for diagnosis.

### Fixed
- Nothing — `question_state.py` is untouched. The Windows 3.52 ms median
  demonstrated no correctness or performance defect: the numpy path keeps
  ~2.8× headroom on the slowest environment measured, and the regression
  class the test guards against (the ~26 ms stdlib scan) is still caught
  decisively.

## Forensic audit of a1886ec: six confirmed defects fixed, three additional findings addressed

See `docs/GLM_FORENSIC_AUDIT.md` for the full evidence, ledger and
measured before/after numbers. 130 tests (was 115).

### Fixed
- **BUG-1 (HIGH) — animation starved retries completely.** `observe_visual`
  re-pushed the retry deadline into the future on every visual change, so
  any screen animating faster than `same_frame_retry_delay` never reached
  its deadline (measured: 0 attempts / 30 s vs 10 static). A visual change
  may now only pull the deadline earlier (`min()`), never push it later.
- **BUG-2 (HIGH) — the retry gate was bypassed on any visual change.** A
  full EasyOCR pass ran on every poll whenever anything animated (measured:
  101 passes / 5 s). Replaced by a signature-classified probe gate:
  animation-level deltas stand down; genuine content changes get one
  immediate identity probe (capped); only a genuinely new question
  proceeds. New-question discovery stays immediate — a naive hard gate
  would have deadlocked the quiz flow, since the gate decision belongs to
  the previous frame's identity.
- **BUG-3 (MED-HIGH) — "exhausted" questions clicked forever** (measured:
  153 clicks / 600 s). The `exhausted` flag is now enforced: clicking
  stops for that question until its identity changes, OCR re-validation
  continues at the capped cadence, and a returning exhausted question gets
  a fresh episode (mirroring the done rule).
- **BUG-4 (MEDIUM) — no-identity gate failed open** and outcomes were
  constructed then silently discarded. Fail closed now: `no_identity` is
  refused, outcomes without identity are dropped loudly, and first-frame
  discovery is handled by the probe path (verified by test).
- **BUG-5 (MEDIUM) — signature worst case ~26 ms on the Tk thread.**
  Vectorised with numpy (stdlib fallback kept, byte-identical math):
  8-ring all-no-match p50 0.93 ms (was ~26 ms).
- **BUG-6 (LOW) — `success_cooldown` was dead code.** Now enforced where
  `done` no longer protects: a completed question returning as a new
  episode waits out the cooldown before any click.
- **NEW (HIGH) — hostile re-render churn OCR'd more than the old loop**
  (measured 399 passes / 20 s vs the 51b16f4 loop's 200): fully re-rendered
  frames minted a fresh immediately-allowed unresolved episode per poll.
  Bounded by an unresolved first-look delay plus an unreadable-look probe
  floor: now 80 / 20 s (hard 4/s = first-tier cadence), retries still never
  starve.
- **NEW (hygiene) — `make_new_core()` could wipe the real LUT** when the
  benchmark module was imported instead of run (async saver + `core.lut={}`
  before any redirect). The constructor now redirects to a temp path and
  disables persistence itself; the shipped `optical_lut.json` is verified
  byte-identical (md5 `a2322984…`) across the full test + benchmark chain.
- **NEW — git-dependent tests failed red outside a checkout.** The two
  baseline-extraction tests are now explicitly environment-specific
  (skip without git): extracted-copy run = 128 passed, 2 skipped.

### Added
- `tests/test_forensic_audit.py` — 14 deterministic, fake-clock regression
  tests covering every fix above (animation never starves, OCR rate floor,
  exhaustion click-stop survives re-enable, fail-closed discovery,
  vectorised-vs-reference distance equality, live cooldown).
- Retry benchmark scenarios S9–S12 (animation starvation, hostile OCR
  budget, exhausted answer, no-fingerprint) + `keypad_after_exhausted`
  metric; benchmark logs refreshed.

## Benchmark temp hygiene: nothing temporary ever touches the working tree

### Fixed
- The 51b16f4-era extraction script leaked `tmpXXXXXXXX_51b16f4.py` files
  into the repository root whenever a run crashed — the Windows cp1252
  crash left exactly such a file behind (`tempfile.mkstemp` defaults:
  'tmp' + 8 random chars + the rev as suffix). The rewritten extractor
  already wrote its baseline module OUTSIDE the repository, but both
  benchmarks still cleaned their temp dir only on the SUCCESS path
  (`mkdtemp` + trailing `rmtree`), so a mid-run crash skipped the cleanup.
  Both benchmarks now own a `TemporaryDirectory` context manager
  (`open_bench_dir()`) that refuses to resolve inside the repository and
  removes the whole tree on success AND on exceptions; the baseline module
  additionally asserts it was written outside the repo.
- Post-run tripwire: after every benchmark, the working tree is scanned
  for the exact leftover class (`tmp*.py` and `mathbot_bench_*`; `.git/`
  and `__pycache__/` skipped). Any NEW leftover fails the run with exit
  code 1; pre-existing debris from older versions is reported (and safe
  to delete manually) but never attributed to the current run.

### Added
- 4 temp-hygiene regression tests in `tests/test_benchmark_tooling.py`:
  the bench dir resolves outside the repo and self-cleans; the extracted
  baseline module lands only inside the bench dir with the working tree
  untouched; a failed (bad-rev) extraction leaves zero files anywhere;
  the leftover detector recognises the exact reported filename
  (`tmpj39knlqs_51b16f4.py`) and asserts the live repository is clean.
  115 tests total (was 111).

## Benchmark tooling: Windows-safe baseline extraction + airtight LUT isolation

### Fixed
- `benchmarks/benchmark_ocr.py --baseline <rev>` crashed on Windows while
  extracting the baseline revision: `subprocess(text=True)` decodes git
  output with the platform locale codec (cp1252), which explodes on
  bot_core.py's UTF-8 punctuation — '←'/'→' encode to bytes cp1252 leaves
  undefined (`UnicodeDecodeError: 'charmap' codec can't decode byte
  0x8f`), and the mangled pipeline then surfaced as `TypeError: write()
  argument must be str, not None`. Git output is now captured as BYTES and
  decoded explicitly as UTF-8 — strict, with context — so a genuinely
  non-UTF-8 object fails loudly instead of being benchmarked mangled, and
  genuine git failures (bad revision, missing object) raise with git's
  stderr instead of being swallowed. The extracted baseline module is
  written to a per-run temp dir OUTSIDE the repository, so a crash can no
  longer leave stray files (or `__pycache__`) in the working tree.
- Benchmark LUT isolation hardened. Both benchmarks redirected the LUT to
  one SHARED, predictable `bench_lut.json` in the system temp dir: entries
  accumulated across runs (nondeterministic "LUT loaded: N entries"), and
  two cores' async saves could in principle race their tmp files (each
  core owns a sequence counter, and the shared path is read dynamically at
  save time). Each run now gets a fresh `mkdtemp`'d directory; HEAD and
  baseline cores get distinct LUT files; the redirect happens strictly
  AFTER module execution and BEFORE any BotCore is constructed (the
  module's own top-level `LUT_FILE = os.path.join(_BASE, ...)` cannot
  overwrite it — that ordering bug had been caught once before); and a
  drain helper acquires each core's `_lut_write_lock` before the temp dir
  is removed, so no async save thread can write anywhere after the
  benchmark finishes. The repository's real `optical_lut.json` is
  unreachable for the whole process; verified byte-identical before/after
  both benchmarks.
- `benchmarks/benchmark_retry.py` output relabeled so the numbers cannot
  be misread: "clicks" is now "keypad actions" (individual key presses; a
  2-digit submission costs 3), with a new explicit "submission attempts"
  metric (distinct `click_answer` calls that fired). NEW S1/S7's 9 keypad
  actions are exactly the 3 bounded unconfirmed submission attempts at the
  RetryPolicy cap — production retry behaviour unchanged, presentation
  only. The benchmark table in docs/REDESIGN_REPORT.md §6.2 was updated to
  the same naming.

### Added
- `tests/test_benchmark_tooling.py` — 4 regression tests pinning the
  Windows-safe extraction contract: '←'/'→' UTF-8 round-trip (the exact
  chars that crashed cp1252), loud failure on mangled bytes, loud failure
  on genuine git errors. 111 tests total (was 107).

## Unresolved visual question identity (OCR-flicker loophole fix)

### Fixed
- When OCR produced no canonical expression, question identity fell back
  to a hash of the RAW OCR text — different garbage readings from one
  unchanged screen could mint independent fingerprints, each with a fresh
  retry budget (unbounded reprocessing). Identity for unreadable frames
  now comes from the question-region PIXELS: `visual_signature()`
  (96x16 block-mean of the already-preprocessed OCR input, polarity-
  normalized) + `signature_distance()` (dead-zone L1 with a small
  alignment search, calibrated: same-question noise <= 0.0043, one-digit
  change >= 0.0090, threshold 0.0065). An "unresolved episode" anchors on
  the first unreadable frame; small noise/jitter/blink stays inside the
  same episode and the SAME bounded retry budget; genuinely changed
  pixels (or a 30 s timeout) start a fresh episode with a fresh budget.
  The previous constant-identity degenerate (every unreadable frame
  sharing one budget) is also gone. Exact unchanged frames are still
  revalidated exactly on the RetryPolicy schedule — B1 stays dead.
- Windows test compatibility (found by the first real Windows run:
  97 passed / 10 failed): the identity tests hardcoded a Linux-only font
  path — the rendering helper now obtains its font through the existing
  cross-platform discovery (`synthetic_data.available_fonts()`) and
  draws text with weight parity (`stroke_width=1`), because absolute
  signature distances are font-dependent (a one-digit swap on narrow
  Arial-metric digits measures ~half the reference calibration). The
  separation assertions now lock the threshold CONTRACT (same-question
  noise at/below `unresolved_match_threshold`, genuinely changed
  questions above it) instead of reference-font absolute numbers, and
  the one-digit episode boundary is locked exactly with synthetic grid
  signatures. `test_template_backend_learns_digits` no longer asserts a
  cross-font accuracy property 1-NN class-mean templates do not have
  (a clean Arial '0' sits closer to the mixed-font 'O' centroid); it
  verifies the machinery contract instead — template integrity,
  self/prototype classification, valid class selection, finite
  predictions — and claims no accuracy. Production code untouched.

### Added
- `question_state.observe_unresolved()` + `UnresolvedEpisode` + policy
  knobs (`unresolved_match_threshold`, `unresolved_recent_max`,
  `unresolved_episode_timeout`, `unresolved_min_ink`).
- `tests/test_unresolved_identity.py` — 12 regression tests incl. the six
  required scenarios (garbage A/B/C one episode; noise same episode;
  genuine change new episode; unchanged-screen retry schedule; alternating
  garbage cannot reset the budget; real new question fresh budget).
- `benchmarks/benchmark_retry.py` scenario S8 (garbage-flicker) + a
  `question_states` metric; raw benchmark logs under docs/benchmark-logs/.
- README rewritten from the current code; 107 tests total (was 95).

## Layered question state machine, retry engine and preview decoupling

### Changed
- Replaced the frame-hash-as-question-state architecture with a layered
  state machine (`question_state.py`): exact-frame digest (BLAKE2b) is now
  used only for click confirmation, question identity is a semantic
  fingerprint over the canonical expression (+ enabled ops, mode, capture
  region), and retry is time-based and bounded (`RetryPolicy`) instead of
  pixel-gated. An unchanged frame with an unsettled question is retried
  with backoff and can never be permanently suppressed.
- Preview now updates while the solver is paused (single shared capture);
  resume updates the preview immediately and processes at once.
- The single "Automation" switch became two independent controls (Answer
  clicks / AUTO sequence). The unconfirmed-click safety net disables answer
  clicking specifically and, as a documented rule, cancels in-flight
  scheduled AUTO actions without flipping the AUTO setting.
- `frame_answer_cache` replaced by a TTL'd frame cache carrying the
  question fingerprint (stale entries expire; a hit can never re-click a
  completed question).

### Fixed
- `clean_hallucinations` mapped uppercase B to 6 (dead 'B'->'8' entry ran
  after `.lower()`), so a misread 8 became 6; B->8 now runs before
  lowercasing.
- `solve_math`'s SymPy path discarded integer algebra solutions in hybrid
  mode (SymPy `Float.is_integer` is not an integrality test); the value is
  tested instead.
- Auto-sequence scheduling crashed in core-only (UI-less) runs.
- Loop errors now print a full traceback.

### Added
- Optional ML glyph corrector (`ocr_ml.py`, disabled by default) with
  template + tiny numpy MLP backends, evidence-only correction thresholds
  and grammar-gated acceptance.
- Dataset pipeline (`dataset.py`): append-only JSONL + glyph-crop
  collector (privacy: crops only), human-labelling path, grouped splits,
  loader for the secondary dataset repository (real-image training starts
  once labelled crops are pushed there).
- Synthetic glyph generator (`synthetic_data.py`) and training/eval script
  (`train_glyph_model.py`) with a deployment gate.
- Test suite (`tests/`, 95 tests at the time) and two reproducible
  benchmarks (`benchmarks/`).

## Earlier history (pre-redesign sessions)

### Fixed
- `÷` misread as `+`: never a blanket text rule — the pixels under each
  reported `+` are checked against the glyph's own bounding box
  (`is_division_glyph()`, both Otsu polarities, dot-blobs top+bottom,
  optional middle bar); only corrected to `/` on that visual evidence,
  ambiguous cases stay `+`.
- `4x4` / direct `÷` candidates rejected: the candidate filter only
  recognized the four literal operator characters and silently rejected
  candidates using `× x X ÷ :` before normalization could map them; every
  character `normalise()` understands is now mapped before the check.
- Silent click failures: `fast_click()` now verifies with `GetCursorPos`
  that the cursor actually landed, falls back once to `pyautogui.moveTo`,
  and raises rather than clicking blind; `KEY_PRESS_DELAY` /
  `POST_ANSWER_DELAY` raised from 0 to 0.025 s.
- Clicks landing on the wrong window after focus passed through the GUI:
  the target window is tracked continuously on every poll instead of once
  on resume.
- LUT audit: `"63/9"` was cached as `72` (corrected to `7`), `"2/19"`
  (non-integer) removed; every other entry re-checked against the solver.
  The **Verify LUT** button automates this audit going forward.
- `launch_solver.bat` retries with `pip install --user` automatically when
  the system-wide install hits a permissions wall (locked-down
  school/lab accounts), so imports stop failing in VS Code.

### Added
- `select_math_ocr_text()` candidate selection: when the capture box
  catches stray tokens, the most plausible expression is picked from the
  OCR tokens (operator/digit-group/date-shape filters, shortest solvable
  span wins). `enabled_operations` is a **filter only** — it never converts
  one operator into another. OCR tokens are sorted into reading order by
  bounding box first, since EasyOCR's return order isn't guaranteed.
- "Known Operations" checkboxes (Advanced) driving `enabled_operations`.
- "Save OCR captures" toggle (Advanced, off by default) writing original +
  processed images per solved question to `ocr_captures/`.
- "Verify LUT" button (Advanced): re-checks every cached answer, auto-
  corrects wrong-but-valid entries, removes non-integer ones.
- Optional `certifi` support: EasyOCR's first-run model download can fail
  with an SSL error on stale certificate stores; if `certifi` is installed
  its CA bundle is used, otherwise nothing changes.
