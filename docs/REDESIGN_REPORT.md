# Redesign Report — Question State Machine, Retry Engine, Preview Decoupling, Optional ML Layer

- **Repository:** `jsjskeoken/mathimatical_bot`
- **Baseline audited:** `093fd3c` (*Improve OCR noise correction* — one commit ahead of the user-reported `51b16f4`; the delta `51b16f4..093fd3c` only added `_split_glued_token`, fast-mode `=`-without-`?` rejection, an operator-less-candidate digit-group rule and `B→8` to `bot_core.py`)
- **Report date:** 2026-09-17
- **Scope:** architecture fix for the frame-hash suppression bug, bounded retry engine, pause/preview decoupling, control separation, cache separation, optional evidence-only ML glyph corrector, dataset pipeline, tests and reproducible benchmarks.
- **Verification environment:** Linux sandbox, Python 3.12.14, numpy / OpenCV / SymPy / PIL / scikit-learn installed; `easyocr`, `mss`, `pynput`, `pyautogui` are **stubbed** in tests (platform layer only — the real solver code runs).

Every number in sections 5 and 6 below comes from commands that were actually executed; nothing is projected or estimated.

---

## 1. AUDIT — the exact bugs, with responsibility

All line numbers refer to `093fd3c` (the code before this redesign).

| ID | Location (093fd3c) | Bug | Consequence |
|----|--------------------|-----|-------------|
| **B1** | `gui.py:1353–1355` | `if current_hash == self.last_frame_hash: … return` — the frame MD5 short-circuits the whole OCR/solve path | **Architectural:** when the screen does not change, the question is *permanently suppressed*. If a click landed but the app did not repaint, or OCR failed on an unchanged frame, no retry ever happens — the loop sleeps at `current_polling` forever until pixels change. The frame hash answers "did pixels change?", not "is the question answered?" |
| **B2** | `gui.py:1315` | The whole `capture → OCR → preview` block lives inside `if not self.core.paused:` | **Pause kills the preview.** While paused the user sees a frozen screenshot, although `_track_target_window()` (which must run paused, per its own docstring) proves a paused capture path already existed. Preview is a *display* concern; solving is a *control* concern — they were wrongly coupled. |
| **B3** | `gui.py:863–864` | `_set_automation_enabled()` writes `self.core.answer_clicks_enabled = enabled` **and** `self.core.auto_sequence_enabled = enabled` | Three distinct concepts (solver running / may click answers / may run AUTO 1-2-3 sequence) are bound to one switch. The unconfirmed-click safety net, which should only withdraw *clicking*, also kills the AUTO sequence setting. |
| **B4** | `gui.py:1359–1363` | `frame_answer_cache` maps `frame_hash → (answer, source)` with **no timestamp and no TTL** | A stale entry is servable indefinitely. Any cache hit re-clicks a completed answer for a frame that recurs (round reset, animation returning to the same pixels). One dict serves three roles: frame dedup, answer memo *and* retry suppression. |
| **B5** | `gui.py:1414–1415` (plus 50 ms reset at 1388–1389) | `last_question` compares **raw OCR text** to debounce new solving | Conflates OCR identity with retry suppression: if OCR *fails differently* each pass (garbage), every reading is "new" and can re-solve; if OCR *repeats the same wrong reading*, it is suppressed until pixels change — i.e. the same permanent-suppression hole as B1, at text level. |
| **B6** | `bot_core.py:1293–1294` and `1322, 1331, 1368, 1376` | `click_answer()` writes `answer_cache[norm]` regardless of click outcome; auto-sequence scheduling calls `self.ui.root.after(...)` unguarded | Retry state never lived at the click layer, so "was the click confirmed?" is unknowable after the fact; a core-only run (no UI) crashes on `None.root` when the AUTO sequence fires. |
| **B7** | whole tree | No retry/backoff/TTL machinery exists anywhere in either file | Combined with B1/B4/B5 this yields exactly the three failure modes the redesign targets: permanent suppression, unbounded click storms on animation, and unbounded submissions on OCR flicker. |

**Latent bugs found during the audit (fixed in passing):**

- `clean_hallucinations` mapped uppercase `B→6`; the `B→8` entry ran **after** `.lower()`, so it was dead code and a misread `8` became `6`.
- `solve_math`'s SymPy path tested `Float.is_integer` as an integrality test, discarding *integer* algebra solutions in hybrid mode (SymPy `Float(14.0).is_integer` is not a valid integrality check — the value must be tested).
- The main-loop `except` printed a one-line message, hiding tracebacks.

**Preserved fixes — confirmed present and untouched** (each guarded by a regression test): division-vs-plus glyph classification, enabled-operator filtering, candidate geometry scoring, confidence ranking, glued-token splitting (`_split_glued_token`), `B→8` correction (now actually working), missing-operator repair, date/noise rejection, reading-order sort, `CLICK_RESULT_*` states, foreground-window guard, LUT verify + atomic writes, coord-profile isolation via `_clear_transient_state`.

---

## 2. ARCHITECTURE

### 2.1 Layered question state machine (`question_state.py`, pure stdlib)

```
visual layer    EXACT FRAME DIGEST   BLAKE2b(raw BGRA)   → click confirmation only
identity layer  SEMANTIC FINGERPRINT blake2b(canonical(expr) | enabled-ops | mode | region)
                → cache key, retry scheduling, "same question?" decisions
solve layer     PENDING → SOLVED(answer) → SUBMITTED
click layer     CLICKED / NOT_ENABLED / WRONG_WINDOW / UNMAPPED / ERROR / UNCONFIRMED(n)
retry layer     RetryPolicy: attempt count, next-due time, backoff, settle window
```

The crucial inversion: the exact-pixel digest **no longer gates OCR or solving**. It is consulted only to *confirm* that a click changed the screen. Question identity is decided **after normalisation** (whitespace, case, canonical operators), so `7 + 6`, `7+6` and a flickering `7+6 ` share one identity while genuinely different questions never collide.

### 2.2 Controlled retry (`RetryPolicy`)

- First retry due after **250 ms**, backoff ×2 → `0 → +250 → +500 → +1000 → +2000 …` ms, capped by `max_attempts = 8` and a settle window; every attempt is logged with fingerprint, attempt number, elapsed time and reason (throttled so the console cannot be flooded).
- Retry is **time-based, not pixel-gated**: an unchanged frame with an unsettled question is retried on schedule and can never be permanently suppressed (direct fix for B1/B5/B7).
- Hard safety net: after `UNCONFIRMED_THRESHOLD` (3) unconfirmed clicks, `answer_clicks_enabled` is switched **off** and all scheduled AUTO actions are cancelled — clicking stops, the AUTO setting itself survives, and nothing can click again until a human re-enables it.

### 2.3 Cache separation (fixes B4)

| Cache | Purpose | Eviction |
|-------|---------|----------|
| `TTLFrameCache` (`question_state.py`) | frame digest → solved question fingerprint + answer | TTL (`frame_cache_ttl`, default 90 s) **and** explicit invalidation on round reset; entries carry their fingerprint so a hit can never re-click a completed question |
| question state (state machine) | identity, solve state, click results, retry schedule | episode end / question change / round reset |
| `answer_cache` (`bot_core.py`, untouched role) | "answered this round" memo | round reset — unchanged semantics |

### 2.4 Pause / preview decoupling (fixes B2) and control separation (fixes B3)

`_main_loop` was rewritten as a **single capture loop**: every tick captures once, computes the frame digest, updates the preview and tracks the target window — **paused or not**. The solver (OCR → solve → click) runs only when *not paused* **and** when the retry engine says the current question is due. Tk operations stay on the main thread via `root.after`.

The single "Automation" switch became two independent switches: **Answer clicks** and **AUTO sequence**. The documented safety rule: the unconfirmed-click safety net disables answer clicking and cancels in-flight scheduled AUTO actions *without* flipping the AUTO setting.

### 2.5 Unresolved visual identity (OCR-flicker loophole, added after first review)

**The loophole.** `semantic_fingerprint()`'s contract fell back to hashing the **raw OCR text** when no canonical expression existed. In today's tree `select_math_ocr_text` shields the branch (every readable candidate canonicalises, and unreadable OCR collapses to `raw=""`, hashing to one constant identity) — but that shielding is an accident of two evolving heuristics, and the constant identity is itself wrong: **every unreadable frame shares one budget**, so a genuinely different unreadable question can never get a fresh one. Any future loosening of the selection filter would re-open the exact flicker hole: garbage A/B/C from one unchanged screen → three fingerprints → three fresh budgets → unbounded reprocessing.

**The mechanism** (identity for unreadable frames now comes from pixels, never OCR text):

1. `bot_core.visual_signature(bin_img)` — reuses the **already-preprocessed** Otsu-binarized OCR input (no extra capture, no new dependency): block-mean downsample to 96×16 cells, polarity-normalized (ink = minority, theme-independent) → 1536-byte signature. 0.08 ms.
2. `question_state.signature_distance(a, b)` — dead-zone L1 (per-cell deltas ≤ 48/255 are rendering noise) minimized over a small alignment search (±2×±1 cells, `(0,0)` first, early-exit below threshold). Pure stdlib; ~1.7 ms hot path.
3. `QuestionStateMachine.observe_unresolved(sig, ctx, now)` — anchors an **episode**: fingerprint = `blake2b(b"U\x00" + anchor + context)`. Same episode while the min aligned distance to the recent ring (anchor + up to 7 jitter/blink variants) stays ≤ `unresolved_match_threshold` (0.0065). Distance jump vs every ring entry, or a 30 s timeout, closes the episode: a new anchor mints a new fingerprint → `observe_question` grants a **fresh budget**. A readable canonical question closes the episode. `reset_round()` clears it.

**Calibration evidence** (`scripts/calibrate_signature.py`, production region 311×49 px, production preprocessing): same question + Gaussian noise / blur / 1–3 px shift / cursor blink → **0.0000–0.0043**; one digit changed (7+6→7+2) → **0.0090**; full change (7+6→8+2) → **0.0171**; (→12×4) → **0.0354**. A plain binary-grid Hamming metric was measured first and **rejected**: 1 px shift (0.0059–0.0143) overlaps one-digit change (0.0124). Threshold 0.0065 sits mid-gap with margin on both sides. Failure bias is deliberate: a false "same" shares a bounded budget (under-processing — safe); a false "new" resets a budget (the loophole itself).

**B1 stays dead:** the mechanism only decides *which episode owns an attempt*; `RetryPolicy` timing is untouched — unchanged frames still revalidate exactly on schedule (test `test_unchanged_screen_retries_on_schedule` asserts the 0.25/0.50/1.00/2.00/4.00 s chain).

---

## 3. PATCH — what changed, file by file

No file was wholesale-replaced; `bot_core.py` received surgical edits, `gui.py`'s changes concentrate in the loop/control methods. All 17 preserved OCR fixes pass dedicated regression tests (`tests/test_core_regression.py`).

| File | Status | Summary |
|------|--------|---------|
| `bot_core.py` | modified (focused edits) | wires QSM + `RetryPolicy` + TTL frame cache; `canonicalise()` extracted; `click_answer(..., norm_expr=)` returns richer outcome; ML corrector + dataset-collector hooks (inert unless enabled); auto-sequence scheduling guarded for `ui=None`; round-reset fallback; **fixed** dead `B→8` ordering and SymPy `Float` integrality bug; full-traceback loop errors; `visual_signature()` for the unresolved-visual-identity layer (§2.5) |
| `gui.py` | modified (loop rewrite) | `_main_loop` / `_process_frame` / `_run_click_confirmation` rewritten per §2.4; preview decoupled from pause; two-row controls (Answer clicks / AUTO sequence); removed `last_question` debounce machinery; full-traceback logging; unreadable frames routed through `observe_unresolved` (§2.5) |
| `question_state.py` | **new** | state machine, `RetryPolicy`, `TTLFrameCache`, digests, throttled logger (stdlib only); plus unresolved-visual-identity layer: `visual_signature` contract, `signature_distance`, `UnresolvedEpisode` (§2.5) |
| `ocr_ml.py` | **new** | `GlyphCorrector` — template-match backend + tiny NumPy MLP backend, evidence-only gates |
| `dataset.py` | **new** | append-only JSONL collector, glyph-crop collector (crops only — privacy), human-labelling path, grouped splits, loader for the secondary data repo |
| `synthetic_data.py` | **new** | deterministic glyph renderer + augmentation (blur, noise, erosion, low-res) |
| `train_glyph_model.py` | **new** | train/eval CLI with a **deployment gate** (a model that does not beat the template baseline on the same split is not written) |
| `.gitignore` | modified | adds `ml_model/`, `ml_dataset_synthetic/`, `ml_dataset/`, `bench_lut.json` (generated artefacts) |
| `CHANGELOG.md` | modified | session changelog updated |
| `tests/` | **new** | 115 tests across 6 files (see §5) |
| `benchmarks/` | **new** | `benchmark_ocr.py` (65-case token corpus + `--baseline <rev>`), `benchmark_retry.py` (8 virtual-clock scenarios with a faithful `51b16f4` loop reconstruction) |
| `docs/` | **new** | this report + `GIT_PLAN.md` |

Review-friendly unified diffs for the five modified files are served by the download package site (Patches section; the ZIP itself mirrors the exact commit list).

---

## 4. OCR / ML — layered design and honest status

**Design (unchanged):** EasyOCR stays the recogniser. On top of it: deterministic rules (always on) → optional small ML classifier **only** for low-confidence/ambiguous tokens → mathematical grammar validation as the final gate. The ML never replaces EasyOCR and never overrides high-confidence readings.

**Data reality check (audited, not assumed):** the secondary repository `jentrenert/optical-reader-math-solver` currently contains **zero images** — `ocr_captures/` is gitignored and its code files are byte-identical to the main repo's HEAD. It is a parallel dev branch, **not yet a dataset**. `dataset.py` already contains the loader for it, so real-image training starts the moment labelled crops are pushed there; until then there is no real-data accuracy claim to make — and none is made.

**What exists today:** `synthetic_data.py` renders digit/operator glyphs (this project's own font set) with augmentation; `train_glyph_model.py` trained a tiny NumPy MLP on a **synthetic** split: token accuracy **73.6 % (MLP) vs 46.8 % (template)** on that synthetic test split. In the end-to-end hybrid benchmark the gates behaved exactly as designed but conservatively: the b→8 correction case scores **0/5 corrected** (the gate refuses weak corrections) and the must-not-touch case scores **5/5** — i.e. **0 false corrections**, which is the property that matters most.

**`ml_enabled` ships `False`.** The benchmark output is explicitly labelled *"SYNTHETIC pixels — not real-world accuracy"*. Path to enabling: collect real crops (`dataset.py` collector) → label them → push to the secondary repo → retrain → the deployment gate must show the model beating the template baseline *on real data* before `ml_enabled=True`.

---

## 5. TEST RESULTS — exact commands, fresh run

Executed **2026-09-18**, in the project root:

```
$ python -m pytest tests/ -q
........................................................................ [ 62%]
...........................................                              [100%]
115 passed in 7.18s
```

| File | Covers |
|------|--------|
| `tests/test_core_regression.py` | the 17 preserved OCR fixes (division glyph, glued tokens, `B→8`, missing-op repair, date rejection, geometry ranking, …) — guards against regressions |
| `tests/test_retry_engine.py` | `RetryPolicy` timings/backoff/caps, TTL expiry, no-permanent-suppression, episode reset on question re-appear, throttle |
| `tests/test_gui_loop.py` | pause-with-live-preview, resume-immediate-processing, click-independence of the two switches, safety-net rule (clicks off, AUTO setting survives), cache-TTL behaviour |
| `tests/test_ml_modules.py` | corrector gates (0 false corrections), dataset JSONL integrity, grouped splits, synthetic generator determinism, deployment-gate refusal |
| `tests/test_unresolved_identity.py` | the OCR-flicker loophole regression set (§2.5): garbage A/B/C → one episode; noise/blink → same episode; genuinely changed pixels → new episode + fresh budget; unchanged screen still retries at the exact scheduled times; alternating garbage cannot reset the budget (monotonic → exhausted); timeout + canonical-resolution lifecycle; signature invariants and the measured separation classes; GUI end-to-end single-episode assertion |
| `tests/test_benchmark_tooling.py` | the benchmark tools' own Windows-safety contract: git output captured as bytes and decoded strictly as UTF-8 ('←'/'→' round-trip — the exact chars that crashed cp1252), mangled bytes fail loudly, genuine git failures surface with stderr |

Platform honesty: `easyocr`/`mss`/`pynput`/`pyautogui` are stubbed by `tests/conftest.py` (a `FakeGrab` supplies frames and a headless GUI double replaces Tk). Everything **above** the platform layer — normalisation, solving, state machine, retry scheduling, cache logic, GUI decision logic — runs as real production code. What is *not* covered here is real EasyOCR inference on a live Windows screen; the benchmark corpora are synthetic pixels (see §7 Risks).

---

## 6. BENCHMARK RESULTS — before / after

### 6.1 OCR token corpus (65 cases, synthetic pixel renders)

```
$ python benchmarks/benchmark_ocr.py --baseline 51b16f4
```

| Category | 51b16f4 (baseline) | HEAD (this redesign) |
|---|---|---|
| clean (arithmetic / × / ÷) | 15/15 | 15/15 |
| noisy (low-conf label beside expr) | 5/5 | 5/5 |
| merged-token (whole expr one token) | 5/5 | 5/5 |
| **glued-operator** (`+6`, `×4` glued to digit) | **0/10** | **10/10** |
| **missing-op** (dropped operator) | **0/5** | **5/5** |
| stray-noise (dates) | 10/10 | 10/10 |
| out-of-order tokens | 5/5 | 5/5 |
| **ambiguous-digit** (B→8, O→0) | **5/10** | **10/10** |
| **OVERALL** | **45/65 (69 %)** | **65/65 (100 %)** |
| avg select latency | 0.12 ms | 0.44 ms |
| avg solve latency | 0.154 ms | 0.222 ms |

The gains come from the pre-existing-but-broken paths the audit identified (dead `B→8`, glued-token handling) plus the redesign's re-selection order; latency growth is sub-millisecond and irrelevant next to OCR itself.

### 6.2 Retry behaviour (8 virtual-clock scenarios, OLD = faithful `51b16f4` loop reconstruction)

```
$ python benchmarks/benchmark_retry.py
```

| Scenario | Metric | OLD | NEW |
|---|---|---|---|
| S1 frozen-feedback (click lands, screen never repaints) | keypad actions / submission attempts | 3 / 1 | 9 keypad actions = **3 bounded unconfirmed attempts** (RetryPolicy cap) / 9 — then **safety net disables clicking** (`clicks enabled at end: False`) |
| S2 frozen-failure (frozen screenshot 10 s, OCR garbage) | OCR passes | 1 (suppressed forever) | **7, bounded** by retry policy; 0 keypad actions |
| S3 clicks-off | keypad actions | 0 | 0 (stays off) |
| S4 animation (repaint every 200 ms, same question) | keypad actions / submissions | **120 / 39** | **3 / 1** |
| S5 alternating-wrong (OCR flickers two wrong readings) | keypad actions / submissions | **180 / 59** | **6 / 2** |
| S6 question-change (7+6 → 8+2 mid-session) | keypad actions beyond confirmed | 6 | **0** |
| S7 ocr-improves (garbage 1 s → clean render) | OCR passes | 2 | 4 (recovers through the garbage window; safety net armed) |
| S8 garbage-flicker (unreadable question: animated + garbage OCR 2 s → frozen 3 s → genuinely different unreadable question) | question identities / OCR passes | **1 constant identity** (cannot tell the two questions apart) / 35 with the frozen phase fully suppressed (B1) | **2 episodes** (correct anchoring, fresh budget for the new question) / 35, all bounded |

Metric naming (deliberate, anti-misread): a "keypad action" is one key press
sent by `click_answer` — a 2-digit submission costs 3 (two digits + Enter).
"Submission attempts" counts distinct `click_answer` calls that fired. NEW
S1/S7's 9 keypad actions are therefore exactly the 3 bounded unconfirmed
attempts of the RetryPolicy cap, not 9 separate submissions; production
retry behaviour is unchanged, only the labelling.

Headlines: animation duplicate submissions **39 → 1**; alternating-wrong submissions **59 → 2**; unconfirmed click leakage in S4/S5/S6 **→ 0**; the frozen-screen permanent-suppression hole is gone (bounded revalidation), and when feedback is absent the safety net — not an infinite loop — ends clicking. S8 adds the OCR-flicker loophole evidence (§2.5): one unreadable screen = one episode with a monotonic budget; a genuinely different unreadable screen = a fresh episode.

### 6.3 ML gates (synthetic pixels only)

```
b-misread, pixels are 8 → should correct : 0/5   (gate refuses weak correction)
b-misread, pixels are 6 → must not touch : 5/5   (0 false corrections)
ML select latency: 0.71–0.85 ms
```

---

## 7. RISKS

1. **Real-screen validation is still required.** OCR corpora are synthetic pixel renders; the retry scenarios are virtual-clock reconstructions of the old loop. On a real Windows machine, poll timing, multi-monitor DPI and EasyOCR GPU/CPU variance can shift the tuned constants (`SAME_FRAME_RETRY_DELAY`, `MAX_RETRIES`, TTLs).
2. **ML is unproven on real data** — by design it ships disabled; enabling before real labelled crops exist would violate the project's own evidence rules.
3. **The two latent bug fixes change behaviour**: `B→8` now actually fires (previously `B→6`), and hybrid-mode algebra questions that SymPy answered as `Float` are no longer discarded. Both are covered by tests, but they *are* user-visible answer changes.
4. **Click-confirmation depends on the screen actually changing.** If the target app gives no visual feedback at all, the safety net will (correctly) pause answer clicking; the user must re-enable manually.
5. **The 100 % OCR corpus score must not be read as a real-world accuracy claim** — the corpus is small and synthetic; the secondary dataset repo will provide the honest measure.

---

## 8. GIT PLAN

Nothing was committed or pushed by the agent. Suggested review order and commit sequence (see `docs/GIT_PLAN.md` for the copy-paste version):

1. Review `patches/*.patch` (or `git diff` after extracting) — 5 modified files. Patches are review artefacts: never `git add` them.
2. Extract the package over a clean clone of `093fd3c`.
3. `python -m pytest tests/ -q` → expect `115 passed`.
4. Commit in three logical units: (a) core state machine + retry + unresolved identity (`question_state.py`, `bot_core.py`, `.gitignore`), (b) GUI decoupling + controls (`gui.py`), (c) ML/data/benchmarks/docs/tests (`ocr_ml.py`, `dataset.py`, `synthetic_data.py`, `train_glyph_model.py`, `tests/`, `benchmarks/`, `docs/`, `CHANGELOG.md`, `README.md`).
5. **Never commit:** `ml_model/`, `ml_dataset_synthetic/`, `ml_dataset/`, `bench_lut.json`, `__pycache__/` (all gitignored), `patches/` (review-only), and `optical_lut.json` churn — it is a runtime-maintained cache; if a test/benchmark run appended entries, restore with `git checkout -- optical_lut.json`.
