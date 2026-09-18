"""
tests/test_forensic_audit.py — Regression tests for the six defects confirmed
by the independent forensic audit of a1886ec (see docs/GLM_FORENSIC_AUDIT.md).

Every test is deterministic: the state machine tests run on a fake monotonic
clock; the GUI-level tests patch gui.time.monotonic with the same fake clock
and drive _process_frame / _run_click_confirmation directly, so no test ever
sleeps waiting for real time.

  BUG-1  observe_visual pushed next_allowed forward on every visual change
         -> animated screens starved retries completely.
  BUG-2  gui gate bypassed the retry engine on any visual change
         -> EasyOCR ran on every poll on animated screens.
  BUG-3  rt.exhausted was never enforced -> "clicking stops" was fiction.
  BUG-4  no-identity gate failed OPEN and outcomes were silently dropped.
  BUG-5  signature_distance worst case ~26 ms on the Tk main thread.
  BUG-6  success_cooldown was dead code on every path.
"""

import time as _time
import warnings

import numpy as np
import pytest

import gui as gui_mod
from conftest import FakeGrab, make_ocr_result
from dataclasses import replace
from question_state import (
    QuestionStateMachine, RetryPolicy, frame_digest,
    signature_distance, signature_ink, SIG_GRID_W, SIG_GRID_H,
    OUTCOME_OCR_EMPTY, OUTCOME_UNSOLVED, OUTCOME_NOT_ENABLED,
    OUTCOME_CLICKED, OUTCOME_CONFIRMED, OUTCOME_UNCONFIRMED,
    OUTCOME_UNMAPPED,
)
from bot_core import (
    CLICK_RESULT_AUTOMATION_OFF, CLICK_RESULT_CLICKED,
)


# ── shared helpers ───────────────────────────────────────────────────────────

class FakeClock:
    """Deterministic monotonic clock (start away from 0 to avoid 0-edges)."""

    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def make_machine(**policy_kw):
    clock = FakeClock()
    defaults = dict(same_frame_retry_delay=0.25,
                    same_question_retry_delay=0.5,
                    retry_backoff=2.0, max_backoff=4.0,
                    max_retries_per_question=8,
                    success_cooldown=1.0,
                    click_confirm_timeout=0.5,
                    question_reappear_reset=2.0)
    defaults.update(policy_kw)
    return QuestionStateMachine(RetryPolicy(**defaults), clock=clock), clock


def script_reader(results_queue):
    """Fake EasyOCR reader popping scripted per-call results (last repeats).
    Double-wrapped queue items ([[res]]) are unwrapped, same as the
    test_gui_loop helper."""
    calls = {"n": 0}

    def readtext(arr, **kwargs):
        calls["n"] += 1
        r = results_queue.pop(0) if len(results_queue) > 1 else results_queue[0]
        if r and isinstance(r, list) and isinstance(r[0], list):
            r = r[0]
        return r

    return readtext, calls


def blink_frame(base: FakeGrab, i: int) -> FakeGrab:
    """Animation-level change: digest differs, pixel signature does NOT.

    A one-pixel flip lands inside one 96x16 signature cell's deadzone after
    INTER_AREA downscale, so the probe gate must treat it as animation —
    which is exactly what a cursor blink is on a real screen."""
    clone = FakeGrab.__new__(FakeGrab)
    clone._seed = base._seed
    clone._arr = base._arr.copy()
    y = (i * 7) % clone._arr.shape[0]
    x = (i * 13) % clone._arr.shape[1]
    clone._arr[y, x, 0] ^= 0xFF
    return clone


@pytest.fixture
def gui(gui_double, core, click_log, monkeypatch):
    """GUI double + fast fake-clock drive of the real _process_frame.

    time.monotonic is patched module-wide (same object stdlib time uses),
    so gui._process_frame, gui._run_click_confirmation and every qsm call
    that receives `now` all share one deterministic clock. qsm's internal
    default clock is never consulted on these paths (gui always passes now
    explicitly)."""
    gg = gui_double
    gg.core.last_question = ""
    gg.core.paused = False
    gg._clock = FakeClock()
    monkeypatch.setattr(gui_mod.time, "monotonic", gg._clock)
    return gg


# ── BUG-1: animation must never push a retry deadline later ─────────────────

def test_bug1_animation_never_starves_retries():
    """Static screens and animated screens must BOTH reach their retries.

    a1886ec measured: static 10-11 attempts / 30 s, animated 0-1. After the
    min() fix the animated case retries on schedule (tier resets keep the
    first-tier cadence until the budget exhausts, then revalidation at
    max_backoff) — bounded, never zero. The precise invariant: observe_visual
    must never INCREASE next_allowed — a change may only leave it or pull
    it earlier."""
    for label, interval in (("static", None), ("anim-50ms", 0.05)):
        m, ck = make_machine()
        m.observe_visual("d0", ck())
        m.observe_question("fp1", "7+6", "7 + 6", ck())
        m.record_outcome(OUTCOME_UNSOLVED, ck())          # attempt 1
        allowed = 0
        for i in range(1, 601):                           # 30 s, 50 ms steps
            if interval is not None and i % int(interval / 0.05) == 0:
                before = m.runtime().next_allowed         # THE BUG-1 INVARIANT
                m.observe_visual(f"d{i}", ck())           # pixels changed
                assert m.runtime().next_allowed <= before + 1e-9, \
                    f"{label}: visual change pushed the deadline later"
            dec = m.should_process(ck())
            if dec.allowed:
                allowed += 1
                m.record_outcome(OUTCOME_UNSOLVED, ck())
            ck.advance(0.05)
        assert allowed >= 10, f"{label}: retries starved ({allowed})"


def test_bug1_tier_reset_on_visual_change_is_bounded():
    """Animation resetting the backoff tier only ever ACCELERATES a retry;
    the attempt budget still caps everything (8 retries, then exhausted)."""
    m, ck = make_machine()
    m.observe_visual("d0", ck())
    m.observe_question("fp1", "7+6", "7 + 6", ck())
    for i in range(50):                                   # constant animation
        m.observe_visual(f"d{i}", ck())
        if m.should_process(ck()).allowed:
            m.record_outcome(OUTCOME_UNSOLVED, ck())
        ck.advance(0.05)
    rt = m.runtime()
    assert rt.exhausted                                   # budget ran out
    assert rt.attempts == 9                               # 8 retries + 1st


# ── BUG-2: the gate is never bypassed; discovery stays immediate ────────────

def test_bug2_animation_never_drives_ocr(gui, core):
    """Cursor-blink-level pixel churn on a blocked gate -> OCR passes stay
    ON THE RETRY SCHEDULE. a1886ec ran one pass per 10 ms poll (101 / 5 s).
    Invariants: (a) no two OCR passes closer than the schedule floor
    (same_frame_retry_delay - visual pull-ins may accelerate a look to the
    first tier, never past it), and (b) the total stays bounded (attempt
    budget + revalidation cadence) instead of tracking the poll count."""
    timestamps = []
    results = [make_ocr_result([("garbled", 0.5)])]

    def readtext(arr, **kwargs):
        timestamps.append(gui._clock.t)
        return results.pop(0) if len(results) > 1 else results[0]

    core.reader.readtext = readtext
    base = FakeGrab()
    gui._process_frame(base, frame_digest(base.bgra))       # discovery (OCR #1)
    assert len(timestamps) == 1
    for i in range(1, 41):                                # 40 polls, 10 ms apart
        f = blink_frame(base, i)
        gui._process_frame(f, frame_digest(f.bgra))
        gui._clock.advance(0.01)
    floor = core.retry_policy.same_frame_retry_delay
    gaps = [b - a for a, b in zip(timestamps, timestamps[1:])]
    assert min(gaps) >= floor - 1e-6, \
        f"OCR ran faster than the retry schedule: {min(gaps)*1000:.1f} ms"
    assert len(timestamps) < 15, f"animation drove {len(timestamps)} OCR passes"
    assert len(timestamps) > 3, "retries stopped entirely"


def test_bug2_new_question_discovered_immediately_while_gate_closed(gui, core):
    """The load-bearing case: the previous question is mid-backoff, the
    screen shows a GENUINELY different question -> one probe looks NOW,
    the new identity proceeds, and the new runtime starts at zero."""
    readtext, calls = script_reader([
        make_ocr_result([("7", 0.95), ("+", 0.9), ("6", 0.9)]),
        make_ocr_result([("8", 0.95), ("*", 0.9), ("2", 0.9)]),
    ])
    core.reader.readtext = readtext
    core.answer_clicks_enabled = False
    a = FakeGrab()
    gui._process_frame(a, frame_digest(a.bgra))             # 7+6 solved, click off
    assert core.qsm.runtime().canonical == "7+6"
    assert core.qsm.runtime().attempts == 1               # NOT_ENABLED
    gui._clock.advance(0.01)                                # still inside backoff
    b = a.mutated()                                       # genuine content change
    gui._process_frame(b, frame_digest(b.bgra))
    rt = core.qsm.runtime()
    assert rt.canonical == "8*2"                          # discovered immediately
    assert rt.attempts == 1                               # fresh runtime, own budget
    assert calls["n"] == 2


def test_bug2_same_question_content_change_spends_no_budget(gui, core):
    """Pixels genuinely changed but OCR still reads the SAME question while
    the gate is closed -> probe stands down, budget untouched, answer shown."""
    readtext, _ = script_reader([
        make_ocr_result([("7", 0.95), ("+", 0.9), ("6", 0.9)]),
    ])
    core.reader.readtext = readtext
    core.answer_clicks_enabled = False
    a = FakeGrab()
    gui._process_frame(a, frame_digest(a.bgra))
    assert core.qsm.runtime().attempts == 1
    gui._clock.advance(0.01)
    b = a.mutated()                                       # new pixels, same question
    gui._process_frame(b, frame_digest(b.bgra))
    assert core.qsm.runtime().attempts == 1               # no budget spent
    assert gui._display_calls[-1][1] == 13                  # answer still displayed


def test_bug2_probe_cap_bounds_adversarial_animation(gui, core):
    """A hostile screen re-rendering with LARGE deltas every frame cannot
    turn the probe into a per-poll OCR loop: same-identity probes are capped
    (VISUAL_LOOK_MAX_PER_WINDOW) between gate openings — the OCR count stays
    flat no matter how many hostile frames arrive afterwards."""
    readtext, calls = script_reader([
        make_ocr_result([("7", 0.95), ("+", 0.9), ("6", 0.9)]),
    ])
    core.reader.readtext = readtext
    core.answer_clicks_enabled = False
    a = FakeGrab()
    gui._process_frame(a, frame_digest(a.bgra))             # discovery
    gui._clock.advance(0.001)
    for i in range(gui_mod.VISUAL_LOOK_MAX_PER_WINDOW):   # cap not yet hit
        f = FakeGrab(seed=100 + i)                        # distinct big-delta frame
        gui._process_frame(f, frame_digest(f.bgra))
        gui._clock.advance(0.001)                         # stay inside backoff
    assert calls["n"] == 1 + gui_mod.VISUAL_LOOK_MAX_PER_WINDOW
    assert core.qsm.runtime().attempts == 1
    for i in range(20):                                   # 20 MORE hostile frames
        f = FakeGrab(seed=200 + i)
        gui._process_frame(f, frame_digest(f.bgra))
        gui._clock.advance(0.001)                         # still inside the window
    assert calls["n"] == 1 + gui_mod.VISUAL_LOOK_MAX_PER_WINDOW   # flat: capped
    assert core.qsm.runtime().attempts == 1


# ── BUG-3: exhausted means clicks stop (until the identity changes) ─────────

def test_bug3_exhausted_question_stops_clicking(gui, core, click_log):
    """A stuck question whose clicks never confirm: after the retry budget
    is spent, click_answer is never called again, while processing/re-OCR
    continues at the capped cadence. (a1886ec measured 153 clicks / 600 s.)"""
    readtext, calls = script_reader([
        make_ocr_result([("7", 0.95), ("+", 0.9), ("6", 0.9)]),
    ])
    core.reader.readtext = readtext
    core.qsm.policy = replace(core.retry_policy, unconfirmed_threshold=10 ** 9)
    gui.UNCONFIRMED_THRESHOLD = 10 ** 9
    frame = FakeGrab()
    digest = frame_digest(frame.bgra)
    clicks_seen = 0
    exhausted_at = None
    for i in range(200):                                  # ~40 simulated seconds
        gui._run_click_confirmation(digest)                 # deadline -> UNCONFIRMED
        gui._process_frame(frame, digest)
        gui._clock.advance(0.2)
        rt = core.qsm.runtime()
        if rt is not None and rt.exhausted and exhausted_at is None:
            exhausted_at = i
            clicks_seen = len(click_log)
        if exhausted_at is not None:
            assert len(click_log) == clicks_seen, "clicked after exhaustion!"
    assert exhausted_at is not None
    rt = core.qsm.runtime()
    assert rt.exhausted and rt.attempts >= 9
    # Steady state: exactly one click delivery per attempt. 9 attempts x 3
    # keys: the 9th processing pass is the one that DISCOVERS exhaustion
    # (attempts > max is only knowable once its outcome is recorded), so its
    # click still lands — bounded, one-off, and every pass after it is
    # click-free (asserted above).


def test_bug3_click_stop_survives_reenable_until_identity_changes(gui, core,
                                                                  click_log):
    """PART 5 scenario 5: exhausted question + answer clicking re-enabled
    -> still no click. A genuinely new question -> fresh budget, clicks."""
    readtext, _ = script_reader([
        make_ocr_result([("7", 0.95), ("+", 0.9), ("6", 0.9)]),
        make_ocr_result([("8", 0.95), ("*", 0.9), ("2", 0.9)]),
    ])
    core.reader.readtext = readtext
    core.answer_clicks_enabled = False                    # clicks off: no safety net
    frame = FakeGrab()
    digest = frame_digest(frame.bgra)
    for _ in range(12):                                   # spend the whole budget
        gui._process_frame(frame, digest)
        gui._clock.advance(5.0)                             # past every backoff
    assert core.qsm.runtime().exhausted
    core.answer_clicks_enabled = True                     # operator re-enables
    gui._process_frame(frame, digest)
    gui._clock.advance(5.0)
    assert click_log == []                                # STILL no click
    b = frame.mutated()                                   # genuine new question
    gui._process_frame(b, frame_digest(b.bgra))
    assert len(click_log) == 3                            # fresh episode clicks
    assert core.qsm.runtime().canonical == "8*2"


def test_bug3_returning_exhausted_question_gets_a_fresh_episode():
    """PART 5 scenario 4: the same exhausted question returning after a long
    absence is a NEW instance (mirrors the done-question rule) — otherwise
    'until its identity changes' would mean 'until the process restarts'."""
    m, ck = make_machine()
    m.observe_question("fp1", "7+6", "7+6", ck())
    for _ in range(9):
        m.record_outcome(OUTCOME_UNSOLVED, ck())
        ck.advance(5.0)
    assert m.runtime("fp1").exhausted
    m.observe_question("fp2", "8+2", "8+2", ck())         # other content on screen
    ck.advance(10.0)                                      # longer than reappear_reset
    m.observe_question("fp1", "7+6", "7+6", ck())         # 7+6 returns
    rt = m.runtime("fp1")
    assert not rt.exhausted and rt.attempts == 0
    assert not rt.done
    dec = m.should_process(ck())
    assert not dec.allowed                                # success_cooldown window
    ck.advance(m.policy.success_cooldown)
    assert m.should_process(ck()).allowed


# ── BUG-4: no-identity must fail closed yet still discover ──────────────────

def test_bug4_fail_closed_gate_still_discovers_first_question(gui, core):
    """Fail-closed must not deadlock start-up: the very first frame has no
    identity, the gate refuses, and the discovery path still processes it."""
    assert core.qsm.current_fingerprint is None
    readtext, calls = script_reader([
        make_ocr_result([("7", 0.95), ("+", 0.9), ("6", 0.9)]),
    ])
    core.reader.readtext = readtext
    core.answer_clicks_enabled = False
    frame = FakeGrab()
    gui._process_frame(frame, frame_digest(frame.bgra))
    rt = core.qsm.runtime()
    assert rt is not None and rt.solved and rt.answer == 13
    assert gui._display_calls[-1][1] == 13
    assert calls["n"] == 1


def test_bug4_outcome_without_identity_is_dropped_loudly():
    m, ck = make_machine()
    assert m.record_outcome(OUTCOME_OCR_EMPTY, ck()) is None
    assert len(m._runtime) == 0
    assert not m.should_process(ck()).allowed


# ── BUG-5: vectorised signature math must stay semantically identical ───────

# Performance contract (documented — see test_bug5_ring_scan_worst_case_bounded
# and docs/GLM_FORENSIC_AUDIT.md §2 BUG-5). The numbers encode the
# ARCHITECTURAL requirement — the ring scan shares the Tk main thread's
# fast-poll budget (FAST_MODE_POLLING = 10 ms, bot_core.py) — not any single
# machine's absolute speed.
BUG5_FAST_POLL_BUDGET_MS = 10.0   # hard limit: the fast-poll budget itself
BUG5_SOFT_MARGIN_MS = 5.0         # half-budget visibility band (warn, not fail)
BUG5_SAMPLES = 30                 # median-of-N: robust to one-off scheduler/GC spikes

def _reference_distance(a, b, min_ink, bail_below):
    """The pure-stdlib implementation, verbatim from a1886ec."""
    if len(a) != len(b) or len(a) != SIG_GRID_W * SIG_GRID_H:
        return 1.0
    if signature_ink(a) < min_ink and signature_ink(b) < min_ink:
        return 0.0
    deadzone, span = 48, 255 - 48
    best = 1.0
    w, h = SIG_GRID_W, SIG_GRID_H
    shifts = [(0, 0)] + [(dy, dx)
                         for dy in range(-1, 2) for dx in range(-2, 3)
                         if (dy, dx) != (0, 0)]
    for dy, dx in shifts:
        total = 0
        y0a, y1a = max(0, dy), min(h, h + dy)
        x0a, x1a = max(0, dx), min(w, w + dx)
        y0b, y1b = max(0, -dy), min(h, h - dy)
        x0b, x1b = max(0, -dx), min(w, w - dx)
        for y in range(y1a - y0a):
            ra = (y0a + y) * w
            rb = (y0b + y) * w
            for x in range(x1a - x0a):
                va = a[ra + x0a + x]
                vb = b[rb + x0b + x]
                d = va - vb if va > vb else vb - va
                if d > deadzone:
                    total += d - deadzone
        dist = total / (span * (x1a - x0a) * (y1a - y0a))
        if dist < best:
            best = dist
            if best == 0.0 or (bail_below is not None and best <= bail_below):
                return best
    return best


def test_bug5_vectorised_distance_matches_reference_exactly():
    import question_state as qs
    rng = np.random.default_rng(42)
    corpus = []
    base = rng.integers(0, 256, SIG_GRID_W * SIG_GRID_H, dtype=np.uint8)
    corpus.append((base.tobytes(), base.tobytes()))                     # identical
    corpus.append((base.tobytes(),
                   np.clip(base.astype(int) + rng.integers(-3, 4, base.size),
                           0, 255).astype(np.uint8).tobytes()))         # noise
    corpus.append((base.tobytes(),
                   rng.integers(0, 256, SIG_GRID_W * SIG_GRID_H,
                                dtype=np.uint8).tobytes()))             # no match
    blank = np.full(SIG_GRID_W * SIG_GRID_H, 10, dtype=np.uint8)
    corpus.append((blank.tobytes(), blank.tobytes()))                   # blank
    shifted = np.roll(base.reshape(SIG_GRID_H, SIG_GRID_W), 2, axis=1)
    corpus.append((base.tobytes(), shifted.tobytes()))                  # shift
    for a, b in corpus:
        for bail in (None, 0.0065, 0.5):
            assert qs.signature_distance(a, b, 0.015, bail) == \
                _reference_distance(a, b, 0.015, bail)


def test_bug5_ring_scan_worst_case_bounded():
    """BUG-5 performance contract: the 8-ring all-no-match scan must fit
    inside the 10 ms fast-poll budget it shares on the Tk main thread.

    Threshold rationale (documented contract — NOT a tuned-to-one-machine
    number; see docs/GLM_FORENSIC_AUDIT.md §2 BUG-5):

    signature_distance runs on the Tk MAIN thread inside the fast poll
    loop (FAST_MODE_POLLING = 10 ms in bot_core.py), sharing the budget
    with frame capture, digesting and the retry decision. The pre-audit
    stdlib implementation measured ~26 ms for this exact workload — 2.6x
    the budget, i.e. visible UI jank exactly when a question changes.
    That regression class is what this test exists to catch.

    Measured medians for the same 8-ring all-no-match workload:
        a1886ec stdlib path   ~26.0 ms   (the bug — 2.6x the budget)
        numpy path, Linux     ~0.93 ms   (audit machine)
        numpy path, Windows   ~3.52 ms   (user verification machine)

    The assert therefore encodes the ARCHITECTURAL requirement — the
    worst case must fit INSIDE the fast-poll budget — and deliberately
    not any single machine's absolute speed:
      * hard limit  10.0 ms = the budget itself. Detection margin against
        the original 26 ms bug is 2.6x; headroom on the slowest platform
        measured so far (Windows, 3.52 ms) is 2.8x.
      * soft margin  5.0 ms = half the budget. Machines landing between
        the soft margin and the budget still PASS but emit a RuntimeWarning,
        so a ~2x platform slowdown becomes visible in the pytest summary
        long before it can threaten the real contract.
    """
    import question_state as qs
    if qs._np is None:                                    # pragma: no cover
        pytest.skip("numpy unavailable — stdlib fallback in use; the "
                    "budget contract is guaranteed by the vectorised path")
    rng = np.random.default_rng(7)
    sigs = [rng.integers(0, 256, SIG_GRID_W * SIG_GRID_H, dtype=np.uint8).tobytes()
            for _ in range(9)]
    probe = sigs[-1]
    ring = sigs[:8]

    def one_ring_scan():
        for s in ring:
            d = qs.signature_distance(probe, s, 0.015, bail_below=0.0065)
            if d <= 0.0065:
                break

    for _ in range(3):                    # warmup: allocator, caches, page-in
        one_ring_scan()
    samples = []
    for _ in range(BUG5_SAMPLES):         # median-of-N: robust to one-off
        t0 = _time.perf_counter()         # OS scheduler / GC spikes
        one_ring_scan()
        samples.append((_time.perf_counter() - t0) * 1000)
    samples.sort()
    median = samples[len(samples) // 2]
    assert median < BUG5_FAST_POLL_BUDGET_MS, (
        f"BUG-5 contract violated: 8-ring all-no-match median {median:.2f} ms "
        f">= {BUG5_FAST_POLL_BUDGET_MS:.1f} ms fast-poll budget on the Tk main "
        f"thread (samples min/median/max: {samples[0]:.2f}/{median:.2f}/"
        f"{samples[-1]:.2f} ms). This is the ~26 ms stdlib regression class "
        "that BUG-5 fixed — see docs/GLM_FORENSIC_AUDIT.md §2 BUG-5.")
    if median >= BUG5_SOFT_MARGIN_MS:                     # pragma: no cover
        warnings.warn(
            f"BUG-5 ring scan median {median:.2f} ms exceeds the "
            f"{BUG5_SOFT_MARGIN_MS:.1f} ms soft margin (budget "
            f"{BUG5_FAST_POLL_BUDGET_MS:.1f} ms). The contract still holds; "
            "this platform is simply slower than every environment measured "
            "so far (Linux ~0.93 ms, Windows ~3.52 ms).",
            RuntimeWarning, stacklevel=2)


# ── BUG-6: success_cooldown must be a live knob ─────────────────────────────

def test_bug6_reappeared_confirmed_question_waits_success_cooldown():
    """a1886ec: done=True short-circuited should_process, so the cooldown
    value never influenced anything. Enforced meaning: a confirmed question
    returning as a NEW episode waits out success_cooldown before clicking."""
    m, ck = make_machine()
    m.observe_question("fp1", "7+6", "7+6", ck())
    m.record_outcome(OUTCOME_CLICKED, ck(), answer=13, source="lut")
    m.record_outcome(OUTCOME_CONFIRMED, ck())
    m.observe_question("fp2", "8+2", "8+2", ck())         # leaves the screen
    ck.advance(10.0)                                      # > question_reappear_reset
    m.observe_question("fp1", "7+6", "7+6", ck())         # returns as new episode
    rt = m.runtime("fp1")
    assert not rt.done and rt.attempts == 0
    dec = m.should_process(ck())
    assert not dec.allowed
    assert rt.next_allowed == pytest.approx(ck.t + m.policy.success_cooldown)
    ck.advance(m.policy.success_cooldown + 0.01)
    assert m.should_process(ck()).allowed
    # a DIFFERENT cooldown value changes the wait -> the knob is live
    m2, ck2 = make_machine(success_cooldown=3.0)
    m2.observe_question("fp1", "7+6", "7+6", ck2())
    m2.record_outcome(OUTCOME_CLICKED, ck2(), answer=13, source="lut")
    m2.record_outcome(OUTCOME_CONFIRMED, ck2())
    m2.observe_question("fp2", "8+2", "8+2", ck2())
    ck2.advance(10.0)
    m2.observe_question("fp1", "7+6", "7+6", ck2())
    ck2.advance(2.0)
    assert not m2.should_process(ck2()).allowed           # still cooling down
