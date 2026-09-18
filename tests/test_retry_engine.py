"""
tests/test_retry_engine.py — Layered state machine unit tests.

Covers the required behaviors:
  2  unchanged valid question retries after TTL
  3  unchanged invalid OCR retries after TTL
  4  retry counter resets after question change
  17 frame cache TTL expires
  18 semantic question cache behaves correctly
  20 same pixels / same question can still retry
  21 different pixels / different question resets retry state
plus backoff, bounds, confirmation semantics, pause/resume, pruning.
"""

import pytest

from question_state import (
    QuestionStateMachine, RetryPolicy, TTLFrameCache, frame_digest,
    semantic_fingerprint, QuestionRuntime,
    OUTCOME_OCR_EMPTY, OUTCOME_UNSOLVED, OUTCOME_CLICKED, OUTCOME_CONFIRMED,
    OUTCOME_UNCONFIRMED, OUTCOME_NOT_ENABLED, OUTCOME_WRONG_WINDOW,
    OUTCOME_UNMAPPED, OUTCOME_ERROR,
)


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt
        return self.t


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
    policy = RetryPolicy(**defaults)
    m = QuestionStateMachine(policy, clock=clock)
    return m, clock


# ── basic scheduling ─────────────────────────────────────────────────────────

def test_first_sighting_processes_immediately():
    m, ck = make_machine()
    m.observe_visual("d1", ck())
    m.observe_question("fp1", "7+6", "7 + 6", ck())
    assert m.should_process(ck()).allowed
    assert m.should_process(ck()).reason == "first_sighting"


def test_unchanged_valid_question_retries_after_ttl():
    """Req 2: a solved-but-unsubmitted question is retried, not suppressed."""
    m, ck = make_machine()
    m.observe_visual("d1", ck())
    m.observe_question("fp1", "7+6", "7 + 6", ck())
    # answer known, click disabled -> NOT_ENABLED, retry scheduled
    m.record_outcome(OUTCOME_NOT_ENABLED, ck(), answer=13, source="solve")
    assert not m.should_process(ck()).allowed
    ck.advance(0.25)
    assert m.should_process(ck()).allowed
    assert m.should_process(ck()).reason == "retry_due"


def test_unchanged_invalid_ocr_retries_after_ttl():
    """Req 3: a failed reading on unchanged pixels retries after the delay."""
    m, ck = make_machine()
    m.observe_visual("d1", ck())
    m.observe_question("fp1", "", "garbled", ck())
    m.record_outcome(OUTCOME_UNSOLVED, ck())
    assert not m.should_process(ck()).allowed
    ck.advance(0.25)
    assert m.should_process(ck()).allowed


def test_retry_backoff_schedule_and_cap():
    """Req: attempt 1 immediate, then 0.25 -> 0.5 -> 1.0 -> 2.0 -> 4.0 cap."""
    m, ck = make_machine()
    m.observe_visual("d1", ck())
    m.observe_question("fp1", "9+1", "9+1", ck())
    expected = [0.25, 0.5, 1.0, 2.0, 4.0, 4.0, 4.0]
    delays = []
    for i in range(len(expected)):
        m.record_outcome(OUTCOME_UNSOLVED, ck())
        rt = m.runtime()
        delays.append(round(rt.next_allowed - ck.t, 3))
        ck.advance(delays[-1] + 0.01)
    assert delays == expected, delays


def test_retry_counter_resets_after_question_change():
    """Req 4: a genuinely different question starts with a fresh budget."""
    m, ck = make_machine()
    m.observe_visual("d1", ck())
    m.observe_question("fp1", "7+6", "7 + 6", ck())
    for _ in range(4):
        m.record_outcome(OUTCOME_UNSOLVED, ck())
        ck.advance(0.3)
    assert m.runtime().attempts == 4
    m.observe_visual("d2", ck())                      # pixels changed too
    m.observe_question("fp2", "8+2", "8 + 2", ck())
    assert m.runtime().fingerprint == "fp2"
    assert m.runtime().attempts == 0
    assert m.should_process(ck()).allowed


def test_visual_change_resets_backoff_tier_but_not_budget():
    """Animation may buy a sooner retry, never more retries."""
    m, ck = make_machine()
    m.observe_visual("d1", ck())
    m.observe_question("fp1", "7+6", "7+6", ck())
    m.record_outcome(OUTCOME_UNSOLVED, ck())          # tier -> next at +0.25
    assert m.runtime().backoff_tier == 1
    ck.advance(0.05)
    m.observe_visual("d2", ck())                      # animation changed pixels
    m.record_outcome(OUTCOME_UNSOLVED, ck())
    rt = m.runtime()
    assert rt.attempts == 2                            # budget kept
    assert round(rt.next_allowed - ck.t, 3) == 0.25    # tier reset: base delay


def test_click_then_confirm_prevents_duplicate_clicking():
    """Req 5: CLICKED -> awaiting blocks; CONFIRMED -> done blocks."""
    m, ck = make_machine()
    m.observe_visual("d1", ck())
    m.observe_question("fp1", "7+6", "7+6", ck())
    m.record_outcome(OUTCOME_CLICKED, ck(), answer=13)
    assert m.runtime().awaiting_confirmation
    assert not m.should_process(ck()).allowed          # no click on top of it
    assert m.runtime("fp1").awaiting_confirmation
    m.record_outcome(OUTCOME_CONFIRMED, ck())
    rt = m.runtime()
    assert rt.done and rt.click_confirmed
    assert not m.should_process(ck()).allowed
    assert m.should_process(ck() + 10).reason == "question_completed"


def test_unconfirmed_click_enters_bounded_retry():
    """Req 6/7: unconfirmed -> retry path, still inside the budget."""
    m, ck = make_machine()
    m.observe_visual("d1", ck())
    m.observe_question("fp1", "7+6", "7+6", ck())
    m.record_outcome(OUTCOME_CLICKED, ck(), answer=13)
    m.record_outcome(OUTCOME_UNCONFIRMED, ck())
    rt = m.runtime()
    assert not rt.awaiting_confirmation
    assert rt.attempts == 1
    assert not m.should_process(ck()).allowed
    ck.advance(0.25)
    assert m.should_process(ck()).allowed


def test_max_retry_limit_then_capped_revalidation():
    """Req 7: after MAX_RETRIES the question is never click-spammed but is
    also never permanently suppressed (bounded revalidation)."""
    m, ck = make_machine(max_retries_per_question=3)
    m.observe_visual("d1", ck())
    m.observe_question("fp1", "7+6", "7+6", ck())
    exhausted_at = None
    for i in range(6):
        m.record_outcome(OUTCOME_UNSOLVED, ck())
        rt = m.runtime()
        if rt.exhausted:
            exhausted_at = i
            break
        ck.advance(rt.next_allowed - ck.t + 0.01)
    assert exhausted_at is not None
    gap = rt.next_allowed - ck.t
    assert gap == pytest.approx(4.0)                   # max_backoff cap
    ck.advance(4.1)
    assert m.should_process(ck()).allowed              # still re-validates
    m.record_outcome(OUTCOME_UNSOLVED, ck())
    assert round(m.runtime().next_allowed - ck.t, 3) == 4.0


def test_unmapped_answer_is_terminal():
    """Req Part 7: no infinite retry of an impossible click."""
    m, ck = make_machine()
    m.observe_visual("d1", ck())
    m.observe_question("fp1", "7+6", "7+6", ck())
    m.record_outcome(OUTCOME_UNMAPPED, ck(), answer=13)
    assert m.runtime().done
    assert not m.should_process(ck() + 60).allowed


def test_click_error_is_retryable():
    m, ck = make_machine()
    m.observe_visual("d1", ck())
    m.observe_question("fp1", "7+6", "7+6", ck())
    m.record_outcome(OUTCOME_ERROR, ck(), answer=13)
    ck.advance(0.25)
    assert m.should_process(ck()).allowed


def test_wrong_window_does_not_solve_or_confirm():
    """Req: WRONG_WINDOW is 'known, not submitted' — not a failure of the
    question and not a success either."""
    m, ck = make_machine()
    m.observe_visual("d1", ck())
    m.observe_question("fp1", "7+6", "7+6", ck())
    rt = m.record_outcome(OUTCOME_WRONG_WINDOW, ck(), answer=13, source="lut")
    assert rt.solved and rt.answer == 13
    assert not rt.done
    ck.advance(0.25)
    assert m.should_process(ck()).allowed


# ── pause / resume ───────────────────────────────────────────────────────────

def test_pause_drops_pending_confirmation_and_streak():
    m, ck = make_machine()
    m.observe_visual("d1", ck())
    m.observe_question("fp1", "7+6", "7+6", ck())
    m.record_outcome(OUTCOME_CLICKED, ck(), answer=13)
    m.arm_confirmation("d1", ck())
    m.consecutive_unconfirmed = 2
    m.on_pause()
    assert m.pending_confirm_digest is None
    assert m.consecutive_unconfirmed == 0


def test_resume_processes_immediately():
    """Req 5 (preview section): resuming restores processing at once."""
    m, ck = make_machine()
    m.observe_visual("d1", ck())
    m.observe_question("fp1", "7+6", "7+6", ck())
    m.record_outcome(OUTCOME_UNSOLVED, ck())
    ck.advance(10.0)
    m.observe_question("fp1", "7+6", "7+6", ck())      # still same question
    # a long wait already passed; force a mid-backoff state to be sure:
    m.record_outcome(OUTCOME_UNSOLVED, ck())
    m.on_resume()
    assert m.should_process(ck()).allowed


# ── fingerprints / identity ──────────────────────────────────────────────────

def test_same_pixels_same_question_can_still_retry():
    """Req 20: unchanged pixels + unchanged question do not permanently
    suppress retries — the retry engine is the only gate."""
    m, ck = make_machine()
    m.observe_visual("d1", ck())
    m.observe_question("fp1", "7+6", "7 + 6", ck())
    for i in range(5):
        m.record_outcome(OUTCOME_OCR_EMPTY, ck())
        rt = m.runtime()
        ck.advance(max(0.3, rt.next_allowed - ck.t + 0.01))
        m.observe_visual("d1", ck())                   # pixels NEVER change
        assert m.should_process(ck()).allowed
        m.observe_question("fp1", "7+6", "7 + 6", ck())


def test_different_pixels_different_question_resets_state():
    """Req 21: changed pixels + changed question -> fresh runtime."""
    m, ck = make_machine()
    m.observe_visual("d1", ck())
    m.observe_question("fp1", "7+6", "7+6", ck())
    m.record_outcome(OUTCOME_UNSOLVED, ck())
    m.observe_visual("d2", ck())
    m.observe_question("fp2", "3*4", "3×4", ck())
    rt = m.runtime()
    assert rt.attempts == 0 and not rt.solved and not rt.done


def test_fingerprint_converges_canonical_readings():
    """Req Part 2: "7+6", "7 + 6", "7×?6" all canonicalise to the same
    identity (canonicalisation happens before hashing)."""
    ctx = "ops=['+','*']|fast=True|area=(1,2,3,4)"
    assert semantic_fingerprint("7+6", "7 + 6", ctx) == \
           semantic_fingerprint("7+6", "7 + 6 ", ctx)
    assert semantic_fingerprint("7*6", "7×?6", ctx) != \
           semantic_fingerprint("7+6", "7 + 6", ctx)


def test_fingerprint_raw_fallback_for_unreadable():
    """LOW-LEVEL building block only: semantic_fingerprint still hashes
    distinct raw garbage distinctly. The MACHINE must never use this for
    unreadable frames — identity there is the unresolved visual episode
    (see tests/test_unresolved_identity.py): garbage A/B/C from one
    unchanged question share ONE episode and ONE bounded retry budget."""
    ctx = "ctx"
    assert semantic_fingerprint("", "garbled A", ctx) != \
           semantic_fingerprint("", "garbled B", ctx)
    assert semantic_fingerprint("", "garbled A", ctx) == \
           semantic_fingerprint("", "garbled A", ctx)


def test_fingerprint_context_separates_configurations():
    """ops / mode / region changes change identity — profile isolation."""
    base = ("7+6", "7+6")
    assert semantic_fingerprint(*base, "ops=['+']|fast=True|area=a") != \
           semantic_fingerprint(*base, "ops=['+','/']|fast=True|area=a")
    assert semantic_fingerprint(*base, "ops=['+']|fast=True|area=a") != \
           semantic_fingerprint(*base, "ops=['+']|fast=False|area=a")
    assert semantic_fingerprint(*base, "ops=['+']|fast=True|area=a") != \
           semantic_fingerprint(*base, "ops=['+']|fast=True|area=b")


def test_completed_question_flicker_does_not_reclick_but_reappear_does():
    """Same question reappearing quickly = flicker (stays done); after a
    longer absence = a new episode (fresh click allowed)."""
    m, ck = make_machine()
    m.observe_visual("d1", ck())
    m.observe_question("fp1", "7+6", "7+6", ck())
    m.record_outcome(OUTCOME_CLICKED, ck(), answer=13)
    m.record_outcome(OUTCOME_CONFIRMED, ck())
    # quick flicker through another frame and back
    ck.advance(0.1)
    m.observe_question("fp2", "9-4", "9-4", ck())
    m.observe_question("fp1", "7+6", "7+6", ck())
    assert m.runtime().done                            # still completed
    # long absence -> genuinely new episode
    ck.advance(3.0)
    m.observe_question("fp2", "9-4", "9-4", ck())
    ck.advance(3.0)
    m.observe_question("fp1", "7+6", "7+6", ck())
    rt = m.runtime()
    assert not rt.done and rt.attempts == 0


# ── frame cache TTL ──────────────────────────────────────────────────────────

def test_frame_cache_ttl_expires():
    """Req 17: stale frame entries expire instead of living forever."""
    clock = FakeClock()
    cache = TTLFrameCache(ttl=30.0, clock=clock)
    cache.put("aa", 42, "solve", "fp1", "7+6")
    entry = cache.get("aa")
    assert entry.answer == 42 and entry.fingerprint == "fp1"
    clock.advance(29)
    assert cache.get("aa") is not None
    clock.advance(2)
    assert cache.get("aa") is None                     # expired


def test_frame_cache_carries_identity():
    clock = FakeClock()
    cache = TTLFrameCache(ttl=30.0, clock=clock)
    cache.put("aa", 13, "cache", "fp7", "7+6")
    e = cache.get("aa")
    assert (e.answer, e.source, e.fingerprint, e.canonical) == \
           (13, "cache", "fp7", "7+6")
    cache.clear()
    assert cache.get("aa") is None


def test_frame_cache_fifo_cap():
    clock = FakeClock()
    cache = TTLFrameCache(ttl=100.0, max_entries=3, clock=clock)
    for i in range(5):
        cache.put(f"k{i}", i, "solve", f"fp{i}", str(i))
    assert len(cache) == 3
    assert cache.get("k0") is None and cache.get("k1") is None
    assert cache.get("k4") is not None


# ── frame digest ─────────────────────────────────────────────────────────────

def test_frame_digest_changes_with_pixels():
    a = frame_digest(b"\x00" * 64)
    b = frame_digest(b"\x01" + b"\x00" * 63)
    c = frame_digest(b"\x00" * 64)
    assert a == c and a != b
    assert len(a) == 32                                # blake2b-128 hex


def test_runtime_pruning_keeps_current_drops_oldest():
    """Memory bound: old question states expire; the current one survives."""
    m, ck = make_machine(question_state_max_entries=4, question_state_ttl=100.0)
    m.observe_visual("d0", ck())
    m.observe_question("fp0", "0+0", "0+0", ck())
    for i in range(1, 8):
        ck.advance(0.1)
        m.observe_question(f"fp{i}", f"{i}+{i}", f"{i}+{i}", ck())
    snap = m.snapshot()
    assert "fp0" not in snap and "fp1" not in snap    # oldest evicted
    assert "fp7" in snap                              # current kept
    assert len(snap) <= 4


def test_record_outcome_without_observed_question_is_safe():
    """Defensive: an outcome for a question never observed must not crash."""
    m, ck = make_machine()
    rt = m.record_outcome(OUTCOME_OCR_EMPTY, ck())
    assert rt is not None
