# GLM Forensic Audit — Redesign HEAD `a1886ec`

**Scope.** Independent forensic audit of the current repository at
`a1886ec` ("Redesign OCR question state and bounded retry handling"),
executed against the 30-part specification supplied after the independent
review (`REVIEW_FINDINGS.md`, BUG-1..BUG-6). Every claim below was
reproduced by driving the real modules on a virtual clock; every number is
measured output, not a code-reading impression. Where a suspicion did not
survive testing, that is recorded too (§5).

**Verdict in one line.** All six reported bugs were confirmed, reproduced,
fixed and regression-tested; the audit additionally found and fixed two NEW
defects (one of which made the redesigned loop *worse than the old loop*
on adversarial screens), the full suite grew 115 → 130 tests, both
benchmarks are unchanged-or-better, and the repository is left clean with
`optical_lut.json` byte-identical (md5 `a232298478264dcbef937f109e956160`).

---

## 1. Verified baseline (PART 1)

| Item | Value |
|---|---|
| Commit audited | `a1886ec` (worktree checkout; user's `main`) |
| Python / pytest | 3.12.14 / 9.0.2 |
| Existing suite | **115 passed in 7.11s** (matches the review's stated baseline) |
| `optical_lut.json` md5 before work | `a232298478264dcbef937f109e956160` |
| Benchmark tooling temp hygiene | `benchmark_ocr.py` already used a repo-external `TemporaryDirectory`; a baseline run left **no `tmp*.py`** and no Git changes |
| numpy / cv2 / sympy / PIL / sklearn | available; EasyOCR GPU banner (stubbed reader in tests) |

The benchmark execution check (PART 1.5) initially appeared clean and was
re-verified at the end (§7); during the audit an import-path LUT hazard was
found and hardened (§4.2).

## 2. Confirmed bugs — reproduction, fix, regression test

### BUG-1 — Retry starvation on animated screens (HIGH) — CONFIRMED
**Reproduction** (virtual clock, 30 s, failure every allowed tick):
static → 10 attempts; animated (50/100/200 ms changes) → **0 attempts**.
The audit's reported "animated ≈ 1" was itself optimistic — with this poll
grid the re-pushed deadline is never reached at all.

**Root cause.** `observe_visual()` executed
`rt.next_allowed = now + same_frame_retry_delay` whenever the deadline was
still in the future, so pixels changing faster than 0.25 s re-pushed the
deadline every frame, forever.

**Fix.** A visual change may only ever **pull** the deadline earlier:
`rt.next_allowed = min(rt.next_allowed, now + same_frame_retry_delay)`.
Exhausted questions are excluded (animation must not accelerate their calm
`max_backoff` re-validation); done questions stay gated by `done`. The
backoff-**tier** reset on visual change is kept deliberately: it only
accelerates, and the attempt budget + exhausted click-stop still bound
everything — animation can spend the budget faster, never exceed it.

**Regression tests.** `test_bug1_animation_never_starves_retries` (the
precise invariant: `observe_visual` never *increases* `next_allowed`; both
static and animated screens reach ≥10 attempts / 30 s) and
`test_bug1_tier_reset_on_visual_change_is_bounded` (50 ticks of animation →
exactly 9 attempts → exhausted, budget intact).

### BUG-2 — Retry gate bypassed on any visual change (HIGH) — CONFIRMED
**Reproduction.** Simulated 5 s, 10 ms poll, animation every 50 ms:
**101 full OCR passes** — the retry engine's 4/s design ceiling simply did
not exist for any screen with animation.

**Root cause and the trap the naive fix would fall into.**
`gui._process_frame` ran the full pipeline whenever `visual_changed` was
true, discarding `should_process`. The obvious repair — honour
`decision.allowed` unconditionally — **deadlocks the primary quiz flow**:
`should_process` runs *before* `observe_question`, so its decision belongs
to the PREVIOUS frame's identity; once that identity is `done` (confirmed
click), the gate is closed forever and the next question would never be
OCR'd. The bypass was load-bearing for discovery; it had to be replaced,
not removed.

**Fix (signature probe gate).** When the gate is closed:
- unchanged frame → gate stands (unchanged behaviour);
- changed frame → the WHY is classified with the existing
  aligned/dead-zoned pixel-signature machinery (no OCR):
  - **animation-level delta** (≤ `unresolved_match_threshold`; measured
    classes: same-question noise ≤ 0.0043) → stand down, zero OCR;
  - **genuine content change** → ONE immediate identity probe; a probe that
    discovers a genuinely NEW question proceeds at once (new-question
    latency stays at one poll), a probe that re-reads the SAME question or
    an unreadable frame stands down **without spending retry budget**;
  - adversarial bound: same-identity probes capped at
    `VISUAL_LOOK_MAX_PER_WINDOW = 3` per closed-gate window;
  - **unreadable-look floor** (§4.1): after a look that read nothing
    canonical, further probes wait one `same_frame_retry_delay`.
- no identity at all (first frame / `reset_round`) → discovery probe, full
  processing — fail-closed discovery (BUG-4) without deadlock.

**Regression tests.** `test_bug2_animation_never_drives_ocr` (invariants:
no two OCR passes closer than the schedule floor; total bounded),
`test_bug2_new_question_discovered_immediately_while_gate_closed`,
`test_bug2_same_question_content_change_spends_no_budget`,
`test_bug2_probe_cap_bounds_adversarial_animation` (OCR count stays flat
under 20 further hostile frames).

### BUG-3 — "Exhausted" questions click forever (MEDIUM-HIGH) — CONFIRMED
**Reproduction.** 600 s stuck question: **153 attempts, 1 per 3.9 s,
forever** — `_schedule_retry` set `exhausted` but nothing ever read it,
while the docstring promised "clicking stops for that question (until its
identity changes)".

**Fix.** Enforced exactly where the documented contract lives:
1. `gui._process_frame` refuses `click_answer` while `rt.exhausted`
   (mirroring the existing `done` / `awaiting_confirmation` click blocks);
   the answer is still displayed and frame-cached, and OCR re-validation
   continues at the capped cadence (`should_process` reason
   `"revalidate_exhausted"` names it in logs).
2. A returning exhausted question (absence > `question_reappear_reset`)
   now starts a NEW episode with a fresh budget — mirroring the done rule.
   Without this, "until its identity changes" would effectively mean
   "until the process restarts".
3. Boundary note (measured, deliberate): the 9th processing attempt is the
   one that *discovers* exhaustion (`attempts > max` is only knowable once
   its outcome is recorded), so its click still lands — bounded, one-off;
   every pass after it is click-free.

**Regression tests.** `test_bug3_exhausted_question_stops_clicking`
(~40 simulated seconds; click log frozen from the moment of exhaustion),
`test_bug3_click_stop_survives_reenable_until_identity_changes` (PART 5
scenarios 3/5: re-enabling answer clicking does NOT resurrect a click;
a genuinely new question clicks immediately),
`test_bug3_returning_exhausted_question_gets_a_fresh_episode`.

### BUG-4 — No current fingerprint: fail-open gate + silent outcome drop (MEDIUM) — CONFIRMED
**Reproduction.** `should_process()` before any observation →
`ProcessDecision(True, 'new_question')`; `record_outcome()` constructed a
runtime and discarded it (`stored runtimes = 0`) — a fail-open default in
the one component whose job is to bound things.

**Fix.** Fail closed: `should_process` returns `(False, "no_identity")`;
`record_outcome` drops the outcome **loudly** (throttled debug line) and
returns `None` instead of minting a runtime it then discards. Legitimate
discovery is preserved by the BUG-2 probe path (the `no_identity` reason
always allows a look), verified by
`test_bug4_fail_closed_gate_still_discovers_first_question` — the very
first frame still solves and displays 13.

**Regression tests.** `test_should_process_without_identity_fails_closed`
(includes the post-`reset_round` case),
`test_bug4_fail_closed_gate_still_discovers_first_question`,
`test_bug4_outcome_without_identity_is_dropped_loudly`. The pre-existing
`test_record_outcome_without_observed_question_is_safe` pinned the OLD
fail-open behaviour and was updated to pin the new contract.

### BUG-5 — `signature_distance` worst case ~26 ms on the Tk main thread (MEDIUM) — CONFIRMED
**Reproduction (a1886ec, pure stdlib, 96×16 grid):** match (early exit)
p50 0.23 ms; single no-match p50 3.24 ms; an all-no-match 8-signature ring
scan ≈ 8 × 3.3 ms ≈ **26 ms — 2.6× the 10 ms fast-poll budget**, on the Tk
thread, exactly at episode boundaries.

**Fix.** Vectorised with numpy (already a hard dependency of `bot_core`;
question_state keeps a byte-identical pure-stdlib fallback for
environments without it). Identical integer math, identical early-exit
order, `int64` accumulation (1536 × 207 would overflow `int16`).

**Measured after fix:** match 0.012 ms; single no-match 0.109 ms;
8-ring all-no-match **p50 0.93 ms / max 0.95 ms** — a ~28× reduction, now
comfortably inside the poll budget.

**Regression tests.** `test_bug5_vectorised_distance_matches_reference_exactly`
(5-pair corpus × 3 bail settings vs the verbatim a1886ec reference
implementation) and `test_bug5_ring_scan_worst_case_bounded` (skips
honestly when numpy is absent).

**Platform note (post-audit, Windows verification round).** The first
Windows run failed `test_bug5_ring_scan_worst_case_bounded` with a median
of **3.52 ms** against the original `median < 3.0 ms` assertion. The
implementation is NOT at fault: 3.52 ms is 2.8× inside the 10 ms
fast-poll budget, and the ~3.8× Linux→Windows factor is ordinary
interpreter/numpy dispatch overhead, not a regression (the stdlib bug
class is ~26 ms, 7× slower than the Windows numpy path). The original
3.0 ms threshold encoded the audit machine's absolute speed instead of
the architectural requirement — exactly the kind of accidental
platform-coupling a regression test must not have. The test has been
redefined against the documented contract: **hard limit = the 10 ms
`FAST_MODE_POLLING` budget itself** (detection margin 2.6× vs the 26 ms
bug; headroom 2.8× vs the measured Windows median), plus a **5.0 ms soft
margin** that passes but emits a `RuntimeWarning` so a ~2× platform
slowdown becomes visible in the pytest summary before it can threaten
the contract. Sampling was hardened at the same time (3 warm-up
iterations, median of 30 timed ring scans). `question_state.py` is
untouched by this change.

### BUG-6 — `success_cooldown` dead code (LOW) — CONFIRMED, knob made live
**Reproduction.** After CONFIRMED: `done=True`, `next_allowed = now +
cooldown`; `should_process` checks `done` first → the cooldown value never
influenced any decision on the primary path (t=0.5 and t=999 both
`question_completed`).

**Analysis.** Moving the check above `done` would be cosmetic (blocked
either way, reason string aside). The knob's documented intent
("belt-and-braces against duplicate clicks") maps to the one path where
`done` no longer protects: **a completed question returning as a NEW
episode**.

**Fix.** The reappear-reset branch now sets
`rt.next_allowed = now + success_cooldown`, so a re-answered question's
fresh episode waits out the cooldown before any click; `RetryPolicy`
comments document the enforced meaning. Verified live: cooldown 1.0 vs 3.0
changes the wait (`test_bug6_reappeared_confirmed_question_waits_success_cooldown`).

## 3. Adversarial scenarios (PART 9/22) — results

Driven on the real state machine / real `_process_frame` (fake clocks; no
real-time sleeps in the new tests):

| Scenario | Result |
|---|---|
| Frozen screen after confirmed click | PASS (pre-existing suite + audit suite) |
| Unmapped answer terminal, no retries | PASS |
| Pause mid-backoff, resume | PASS — processes immediately |
| Same question returns after long absence | PASS — fresh episode, cooldown applied |
| Brief flicker back to a completed question | PASS — no re-click |
| Same unreadable question, garbage varying | PASS — one episode / one budget |
| **Animated screen, retry needed** | **FAIL → FIXED (BUG-1/2)** — S9: 13 bounded OCR passes vs 200 |
| **Exhausted question + clicking re-enabled** | **FAIL → FIXED (BUG-3)** — zero clicks, new identity clicks |
| **No identity (start-up / reset)** | **FAIL(open) → FIXED (BUG-4)** — discovery intact |
| **Hostile re-render churn** | **NEW defect → FIXED (§4.1)** — S10: 80 vs 399 |
| Stale queued events | Verified: clicks are armed per-question via `arm_confirmation` and watched by digest in-loop; AUTO scheduling is pruned each cycle (`_prune_scheduled_events`) and cancelled by the safety net / mode changes. No stale action survives an identity change in any driven scenario. |

## 4. NEW bugs discovered by this audit (PART 25)

### 4.1 — Hostile re-render churn: the redesigned loop OCR'd *more* than the old loop (HIGH, fixed)
**Found while adding benchmark scenario S10.** A screen that re-renders
with LARGE per-frame deltas (video background behind the question) defeats
pixel-identity matching: every frame mints a fresh unresolved episode.
Each fresh runtime was immediately `first_sighting`-allowed, and the probe
cap keyed on `(reason, fingerprint)` reset with every fingerprint change.
Measured: **399 OCR passes / 20 s — worse than the 51b16f4 loop's 200.**

**Fix (two-part bound, semantics-preserving).**
1. `observe_question`: a fresh runtime with an EMPTY canonical (unresolved
   identity) waits one `same_frame_retry_delay` before its first re-look
   (readable questions still process immediately). Genuine unreadable
   questions are re-read exactly on the normal first-retry schedule.
2. `gui` probe gate: after a look that read nothing canonical, further
   closed-gate probes wait one `same_frame_retry_delay` — a hard rate
   floor no per-identity bookkeeping can defeat, because hostile frames
   share no identity to cap against. Looks that READ something are never
   floored, so genuine new-question discovery stays immediate.

**Measured after fix:** S10 = **80 passes / 20 s** (4/s = the first-tier
cadence, vs 20/s on 51b16f4 and ~20/s on a1886ec), retries still never
starve (S9 unchanged at 13). `question_states` remains 24 — pixel identity
churn is by design; the bounded resource is OCR throughput. Documented as
a residual risk in §10.

### 4.2 — `make_new_core()` LUT footgun (repository-hygiene, fixed)
**Found by self-inflicted damage during tracing.** `benchmark_retry.main()`
isolates `bot_core.LUT_FILE` inside its `open_bench_dir()` block — but
`make_new_core()` itself also assigns `core.lut = {}` and left the async
saver armed. Importing the module and driving `make_new_core()` directly
(as this audit's traces did) therefore wrote `{}` over the **real**
`optical_lut.json` (md5 changed; file truncated 53 → 3 lines).

**Fix.** `make_new_core()` now redirects `LUT_FILE` to a temp path itself
and disables `_save_lut_async` outright — the constructor is safe in every
context; `main()`'s per-run redirect still takes precedence. The shipped
LUT was restored from git and verified byte-identical; the FINAL gate re-ran
pytest + both benchmarks with the md5 constant before and after every step.

### 4.3 — Git-dependent tests failed (not failed-false) outside a checkout (PART 26, fixed)
An extracted copy without `.git` failed 2 of 130 tests (`git show`-based
baseline-extraction tests). They are now explicitly marked
environment-specific with a `_requires_git` skipif — option B of PART 26 —
so a zip-download user gets `128 passed, 2 skipped`, not red.

## 5. Suspicions tested and DISPROVED (preserved per PART 24)

| Suspicion | Verdict |
|---|---|
| `_prune()` loses the current question when it is the oldest entry | **Not reachable** — pruning holds the dict at capacity continuously; covered by existing tests, untouched by the fixes |
| Alternating OCR garbage mints separate retry budgets | **Correctly prevented** by `observe_unresolved` when driven through the GUI path (existing tests) |
| SymPy in candidate scoring stalls the loop | **No** — ≥2-digit-group filter rejects `?`-bearing candidates pre-solve (existing tests) |
| Tk touched from the LUT writer thread | **No** — the only `Thread` is the LUT writer; touches no Tk |
| *(audit-internal)* A simple time floor on identity probes would fix BUG-2 | **Disproved by the suite** — it deadlocked/ delayed the load-bearing new-question discovery path; replaced by the signature gate + cap + unreadable-look floor |
| *(audit-internal)* The BUG-3 click-stop alone would satisfy "clicking stops" | **Partially disproved** — without the exhausted-reappear reset (§2 BUG-3 fix 2) the stop would last forever, violating the documented "until its identity changes" |

## 6. Performance results (PART 20/23)

`signature_distance` (96×16, measured, lower is better):

| Case | a1886ec (stdlib) | after (numpy) |
|---|---|---|
| match, early exit | 0.23 ms | **0.012 ms** |
| single no-match | 3.24 ms | **0.109 ms** |
| 8-ring all-no-match | ≈ 26 ms | **0.93 ms** |

Retry benchmark (OLD = faithful 51b16f4 reconstruction, NEW = current
code; production `RetryPolicy`; virtual clock):

| Scenario | Metric | OLD | NEW (post-audit) | a1886ec NEW |
|---|---|---|---|---|
| S2 frozen-failure | ocr passes | 1 | 7 (bounded revalidation) | 7 — unchanged |
| S4 animation | keypad actions | 120 | 3 | 3 — unchanged |
| S8 garbage-flicker | ocr passes | 35 | **19** | 35 — improved by the gate |
| **S9 animation-starve** | ocr passes | 200 | **13** (never starves) | *(new scenario)* |
| **S10 animation-ocr-budget** | ocr passes | 200 | **80** (hard 4/s bound) | ~400 (worse than OLD) |
| **S11 exhausted-answer** | submission attempts | 1 | **9 then zero, keypad_after_exhausted = 0** | clicks forever (153/600 s) |
| **S12 no-fingerprint** | question states | 0 | **1** episode, 0 clicks | fail-open |

OCR benchmark (token-level, synthetic corpus): HEAD **65/65** vs baseline
51b16f4 **45/65** — unchanged by the audit, as required (PART 16: no OCR
normalisation regression; the ÷-glyph, B→8, glued-operator and missing-op
preserved-fixes all still pass).

## 7. Test results & clean verification (PART 30)

- Suite: **130 passed** (115 baseline + 15 net new; one a1886ec test updated
  to the corrected BUG-4 contract, one a1886ec discovery test retimed for
  the documented floor window). Two consecutive clean runs.
- Outside a git checkout: **128 passed, 2 skipped** (explicitly marked).
- `python benchmarks/benchmark_ocr.py --baseline 51b16f4` → HEAD 65/65,
  baseline 45/65.
- `python benchmarks/benchmark_retry.py` → S1–S12, table in
  `docs/benchmark-logs/retry_scenarios.txt`.
- `optical_lut.json` md5 `a232298478264dcbef937f109e956160` before pytest,
  between benchmarks, and after both — **no diff**.
- No `tmp*.py`, no model/cache junk, `git status` contains only intentional
  files (§11).

## 8. Documentation changes (PART 21/28.5)

- `question_state.py` — docstrings now state what the code enforces:
  deadline pull-in never pushes (BUG-1), exhausted click-stop + reappear
  semantics (BUG-3), fail-closed no-identity (BUG-4), vectorisation
  numbers (BUG-5), `success_cooldown`'s enforced meaning (BUG-6).
- `gui.py` — the retry-gate comment block documents the full closed-gate
  decision procedure (animation vs content change, probe cap,
  unreadable-look floor) and `VISUAL_LOOK_MAX_PER_WINDOW`.
- `benchmarks/benchmark_retry.py` — S9–S12 registered with their
  invariants; `keypad_after_exhausted` metric added; `make_new_core()`
  safety contract documented.
- `docs/benchmark-logs/*` — refreshed from the post-fix runs (pytest, OCR
  HEAD + vs-baseline, retry S1–S12).
- `CHANGELOG.md` — new "Forensic audit" entry.
- This document.

## 9. Remaining limitations (honesty first)

1. **Real-data accuracy remains unmeasured.** Every OCR corpus is
   synthetic; ML stays `ml_enabled = False`; nothing here may be read as a
   real-world accuracy claim. The unblocking step is unchanged: zip real
   `ocr_captures/` off the college machine (`.gitignore` guarantees git
   will never carry them).
2. **Hostile-churn residual risk.** On a screen that re-renders with large
   deltas forever, OCR runs at the first-tier cadence (4/s production)
   indefinitely — bounded and calm, but not free. Eliminating it entirely
   would require OCR-free change classification beyond pixel signatures.
3. **Confirmation remains digest-based** (screen changed ⇒ confirmed) —
   the same deliberately-kept semantics as a1886ec. It is click-safe
   (worst case: a duplicate-confirmation is impossible because `done`
   latches), but "screen changed" is not "answer accepted"; upgrading it
   to semantic confirmation is a design decision, not a bug fix, and was
   not smuggled in here.
4. Windows-specific behaviour (mss/pynput/pyautogui, PowerShell, CRLF,
   real EasyOCR latency) could not be exercised on this Linux audit
   environment; the platform layer is stubbed exactly as the existing
   suite stubs it.

## 10. Bug ledger

| ID | Severity | Status | Root cause (one line) | Fix | Regression test |
|---|---|---|---|---|---|
| BUG-1 | HIGH | FIXED | visual change re-pushed the retry deadline every frame | `min()` pull-in; exhausted excluded | test_bug1_* |
| BUG-2 | HIGH | FIXED | `visual_changed` bypassed the retry gate entirely | signature probe gate + probe cap + stand-down | test_bug2_* |
| BUG-3 | MED-HIGH | FIXED | `exhausted` flag never read | click-stop in click policy; exhausted reappear reset | test_bug3_* |
| BUG-4 | MEDIUM | FIXED | fail-open gate + constructed-and-discarded outcomes | fail closed `no_identity`; loud drop | test_bug4_* |
| BUG-5 | MEDIUM | FIXED | 15-shift × 1536-cell pure-Python scan per signature | numpy vectorisation + stdlib fallback | test_bug5_* |
| BUG-6 | LOW | FIXED | `done` short-circuited before the cooldown mattered | cooldown enforced on reappear-as-new-episode | test_bug6_* |
| NEW-A | HIGH | FIXED | hostile re-render churn minted immediately-allowed episodes per frame | unresolved first-look delay + unreadable-look probe floor | S10 benchmark |
| NEW-B | HYGIENE | FIXED | `make_new_core()` safe only when `main()` ran | temp redirect + async saver disabled in constructor | final gate (md5 constant) |
| NEW-C | LOW | FIXED | git-dependent tests failed red outside a checkout | `_requires_git` skipif | no-git run |

## 11. Files modified

```
benchmarks/benchmark_retry.py      S9-S12, exhaustion metric, make_new_core hardening
gui.py                             BUG-2 probe gate, BUG-3 click-stop, bookkeeping
question_state.py                  BUG-1/3/4/6 fixes, BUG-5 vectorisation
tests/conftest.py                  FakeGrab.mutated = realistic content change
tests/test_benchmark_tooling.py    _requires_git skip marks
tests/test_gui_loop.py             discovery test retimed for the documented floor
tests/test_retry_engine.py         BUG-4 contract tests (updated + new)
tests/test_forensic_audit.py       NEW — 14 deterministic regressions
docs/benchmark-logs/*.txt          refreshed
docs/GLM_FORENSIC_AUDIT.md         NEW — this document
CHANGELOG.md                       audit entry
```

## 12. Git status

Working tree intentionally dirty, nothing staged, nothing committed, the
user is the committer:

```
 M benchmarks/benchmark_retry.py
 M docs/benchmark-logs/ocr_vs_51b16f4.txt
 M docs/benchmark-logs/pytest.txt
 M docs/benchmark-logs/retry_scenarios.txt
 M gui.py
 M question_state.py
 M tests/conftest.py
 M tests/test_benchmark_tooling.py
 M tests/test_gui_loop.py
 M tests/test_retry_engine.py
?? docs/GLM_FORENSIC_AUDIT.md
?? tests/test_forensic_audit.py
```

`git diff -- optical_lut.json` → empty. `git diff --check` → clean.

## 13. Recommended next step

1. Review `gui.py::_process_frame` (the probe gate) and
   `question_state.py::observe_visual/observe_question/should_process` —
   the four semantic changes of this audit.
2. Run the verification sequence on Windows
   (`pytest tests/ -q`, both benchmarks, `git status`, LUT diff) — the
   fake-clock suite is designed to behave identically there.
3. Commit as one audit commit (suggested:
   "Forensic audit: fix retry starvation, gate bypass, exhaustion
   enforcement, fail-open identity, signature perf; add S9-S12").
4. The ZIP freeze stays until you have re-verified on Windows; rebuild the
   download-package only after that.
