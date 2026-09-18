"""
tests/test_gui_loop.py — Headless integration tests for the reworked main
loop: preview-while-paused, retry of unchanged frames, click separation,
safety-net scoping, frame cache, confirmation flow.

Real code under test: gui._main_loop / _process_frame / _run_click_confirmation
/ _update_preview / _set_answer_clicks_enabled / _set_auto_sequence_enabled,
driven against a real BotCore with a scripted EasyOCR stub.
"""

import time
import types

import pytest

import bot_core
import gui as gui_mod
from bot_core import CLICK_RESULT_CLICKED
from conftest import FakeGrab, make_ocr_result
from question_state import OUTCOME_NOT_ENABLED


def script_reader(results_queue):
    """Fake EasyOCR reader popping scripted results per call.

    Each queue item is one frame's result list [(bbox, text, conf), ...].
    Double-wrapped items ([[...]]) are tolerated and unwrapped, so call
    sites can pass either [res] or [[res]]."""
    calls = {"n": 0, "frames": []}

    def readtext(arr, **kwargs):
        calls["n"] += 1
        calls["frames"].append(arr)
        r = results_queue.pop(0) if len(results_queue) > 1 else results_queue[0]
        if r and isinstance(r, list) and isinstance(r[0], list):
            r = r[0]
        return r

    return readtext, calls


def run_loop_once(g, frame, patch_process=None):
    """Drive one _main_loop iteration synchronously (no Tk scheduler)."""
    g._sct = types.SimpleNamespace(grab=lambda monitor: frame)
    if patch_process is not None:
        g._process_frame = patch_process
    g._main_loop()
    g.root.callbacks.clear()


@pytest.fixture
def gui(gui_double, core, click_log):
    g = gui_double
    g.core.last_question = ""
    return g


# ── pause / preview (Part 5) ─────────────────────────────────────────────────

def test_paused_loop_skips_processing_but_captures(gui, core):
    processed = []
    core.paused = True
    frame = FakeGrab()
    run_loop_once(gui, frame, patch_process=lambda *a, **k: processed.append(1))
    assert processed == []                       # solver path stopped
    assert gui.last_frame_hash is not None       # capture still ran


def test_paused_preview_continues_updating(gui, core):
    """Req: PAUSED -> preview STILL LIVE."""
    core.paused = True
    core.preview_enabled = True
    frame = FakeGrab()
    # force the cadence: PREVIEW_UPDATE_INTERVAL iterations normally gate it
    for _ in range(gui_mod.PREVIEW_UPDATE_INTERVAL):
        run_loop_once(gui, frame)
    assert len(gui.preview_canvas.images) >= 1   # preview was rendered


def test_preview_disabled_while_paused_renders_nothing(gui, core):
    core.paused = True
    core.preview_enabled = False
    run_loop_once(gui, FakeGrab())
    assert gui.preview_canvas.images == []


def test_resume_updates_preview_immediately(gui, core):
    """Req: resuming -> preview updates at once, not after N cycles."""
    core.paused = False
    core.preview_enabled = True
    gui.sync_pause_state()                       # sets _preview_force
    assert gui._preview_force is True
    run_loop_once(gui, FakeGrab())
    assert len(gui.preview_canvas.images) == 1
    assert gui._preview_force is False


def test_resume_processes_immediately_after_backoff(gui, core):
    """Req: resuming restores solver operation without waiting out a backoff."""
    core.paused = False
    core.preview_enabled = False
    readtext, calls = script_reader([[]])          # OCR reads nothing
    core.reader.readtext = readtext
    frame = FakeGrab()
    run_loop_once(gui, frame)                    # first sighting -> OCR_EMPTY
    rt = core.qsm.runtime()
    assert rt is not None and rt.attempts == 1
    run_loop_once(gui, frame)                    # inside backoff -> skipped
    assert calls["n"] == 1
    core.paused = True
    gui.sync_pause_state()
    core.paused = False
    gui.sync_pause_state()                       # resume -> on_resume()
    run_loop_once(gui, frame)                    # processes IMMEDIATELY
    assert calls["n"] == 2


def test_run_loop_when_unpaused_processes(gui, core):
    core.paused = False
    core.preview_enabled = False
    readtext, calls = script_reader([[make_ocr_result([("7", 0.95), ("+", 0.9),
                                                     ("6", 0.9)])]])
    core.reader.readtext = readtext
    run_loop_once(gui, FakeGrab())
    assert calls["n"] == 1
    assert gui._display_calls[-1][1] == 13       # solved


# ── retry of unchanged frames (Part 3) ───────────────────────────────────────

def test_unchanged_frame_invalid_ocr_is_retried(gui, core):
    """Req 1/2: failed OCR on a STATIC screen retries after the TTL."""
    core.paused = False
    core.preview_enabled = False
    readtext, calls = script_reader([[make_ocr_result([("garbled", 0.5)])]])
    core.reader.readtext = readtext
    frame = FakeGrab()
    run_loop_once(gui, frame)                    # attempt 1: unsolved
    assert calls["n"] == 1
    run_loop_once(gui, frame)                    # unchanged + backoff: skip
    assert calls["n"] == 1
    time.sleep(core.retry_policy.same_frame_retry_delay + 0.01)
    run_loop_once(gui, frame)                    # RETRIED despite same pixels
    assert calls["n"] == 2


def test_unchanged_frame_valid_question_unsubmitted_is_retried(gui, core,
                                                               click_log):
    """Req 1: valid question, clicks disabled -> bounded click retries."""
    core.paused = False
    core.preview_enabled = False
    core.answer_clicks_enabled = False
    readtext, calls = script_reader([[make_ocr_result([("7", 0.95), ("+", 0.9),
                                                     ("6", 0.9)])]])
    core.reader.readtext = readtext
    frame = FakeGrab()
    run_loop_once(gui, frame)
    rt = core.qsm.runtime()
    assert rt.solved and rt.last_outcome == OUTCOME_NOT_ENABLED
    time.sleep(core.retry_policy.same_frame_retry_delay + 0.01)
    run_loop_once(gui, frame)                    # retry tick: answer reused,
    # click re-attempted (still off) -> attempt budget now 2
    rt = core.qsm.runtime()
    assert rt.attempts == 2 and rt.solved
    assert click_log == []                       # still never clicked


def test_retry_backoff_grows(gui, core):
    core.paused = False
    core.preview_enabled = False
    readtext, calls = script_reader([[(make_ocr_result([("garbled", 0.5)]))]])
    core.reader.readtext = readtext
    frame = FakeGrab()
    gaps = []
    prev_calls = 0
    for _ in range(3):
        run_loop_once(gui, frame)
        rt = core.qsm.runtime()
        gaps.append(round(rt.next_allowed - rt.last_attempt, 3))
        time.sleep(max(0.0, rt.next_allowed - time.monotonic()) + 0.005)
        prev_calls = calls["n"]
    assert gaps[0] < gaps[1] < gaps[2]           # 0.03 < 0.06 < 0.12(cap)
    assert calls["n"] >= 3


def test_question_change_resets_retry_state(gui, core):
    core.paused = False
    core.preview_enabled = False
    readtext, calls = script_reader([
        [make_ocr_result([("garbled", 0.5)])],
        [make_ocr_result([("8", 0.95), ("*", 0.9), ("2", 0.9)])],
    ])
    core.reader.readtext = readtext
    frame = FakeGrab()
    run_loop_once(gui, frame)
    assert core.qsm.runtime().attempts == 1
    frame2 = frame.mutated()                     # pixels change -> new question
    # The forensic-audit gate floors probes that follow an UNREADABLE look
    # to same_frame_retry_delay; at +1 tick past that floor the retry gate
    # itself opens and the new question is processed on the allowed path —
    # discovery is immediate either way, never later than one floor window.
    time.sleep(core.retry_policy.same_frame_retry_delay + 0.01)
    run_loop_once(gui, frame2)
    rt = core.qsm.runtime()
    assert rt.canonical == "8*2" and rt.attempts == 0


def test_ocr_confidence_change_between_retries_recovers(gui, core):
    """A question that failed with low confidence succeeds when a retry
    reads it cleanly (same pixels)."""
    core.paused = False
    core.preview_enabled = False
    bad = make_ocr_result([("7", 0.3), ("?", 0.2), ("6", 0.3)])
    good = make_ocr_result([("7", 0.95), ("+", 0.94), ("6", 0.95)])
    readtext, calls = script_reader([[bad], [good]])
    core.reader.readtext = readtext
    frame = FakeGrab()
    run_loop_once(gui, frame)
    assert gui._display_calls[-1][1] is None
    time.sleep(core.retry_policy.same_frame_retry_delay + 0.01)
    run_loop_once(gui, frame)
    assert gui._display_calls[-1][1] == 13


# ── anti-spam: duplicate click prevention ────────────────────────────────────

def test_click_awaits_confirmation_no_duplicate_click(gui, core, click_log):
    core.paused = False
    core.preview_enabled = False
    readtext, calls = script_reader([[make_ocr_result([("7", 0.95), ("+", 0.9),
                                                     ("6", 0.9)])]])
    core.reader.readtext = readtext
    frame = FakeGrab()
    run_loop_once(gui, frame)                    # click issued
    assert len(click_log) == 3                   # 7, 6, OK
    for _ in range(5):                           # static screen, awaiting...
        run_loop_once(gui, frame)
    assert len(click_log) == 3                   # NEVER re-clicked


def test_confirmed_click_done_no_reclick_on_flicker(gui, core, click_log):
    core.paused = False
    core.preview_enabled = False
    readtext, calls = script_reader([[make_ocr_result([("7", 0.95), ("+", 0.9),
                                                     ("6", 0.9)])]])
    core.reader.readtext = readtext
    frame = FakeGrab()
    run_loop_once(gui, frame)
    assert len(click_log) == 3
    run_loop_once(gui, frame.mutated())          # screen changed -> CONFIRMED
    assert core.qsm.runtime().done
    # brief flicker back to the old pixels (within reappear window)
    run_loop_once(gui, frame)
    run_loop_once(gui, frame.mutated())
    assert len(click_log) == 3                   # still exactly one submission


def test_unconfirmed_click_retries_bounded_then_safety_net(gui, core, click_log):
    """Req 6/16: unconfirmed clicks retry bounded; 3 strikes disable answer
    clicking SPECIFICALLY."""
    core.paused = False
    core.preview_enabled = False
    core.auto_sequence_enabled = True            # must remain untouched
    core.preview_enabled_before = core.preview_enabled
    readtext, _ = script_reader([[make_ocr_result([("7", 0.95), ("+", 0.9),
                                                 ("6", 0.9)])]])
    core.reader.readtext = readtext
    frame = FakeGrab()
    clicks_after_first = None
    for cycle in range(200):
        run_loop_once(gui, frame)
        time.sleep(0.004)
        if clicks_after_first is None and len(click_log) >= 3:
            clicks_after_first = len(click_log)
        if not core.answer_clicks_enabled:
            break
    # Safety net tripped: answer clicks disabled...
    assert core.answer_clicks_enabled is False
    # ...specifically: AUTO sequence setting untouched
    assert core.auto_sequence_enabled is True
    assert core.paused is False
    assert core.preview_enabled == core.preview_enabled_before
    # total keypad clicks stayed bounded (first submission + <=2 retries)
    assert len(click_log) <= 9                   # <=3 submissions × 3 clicks


def test_safety_net_cancels_scheduled_auto_actions(gui, core, click_log):
    """Documented rule: safety net cancels IN-FLIGHT scheduled AUTO actions
    (stale-screen rule) but leaves the AUTO setting itself alone."""
    core.paused = False
    core.preview_enabled = False
    core.auto_sequence_enabled = True
    core.scheduled_events.append(4242)           # simulate a scheduled step
    cancelled_ids = []
    gui.root.after = lambda ms, fn=None: 5150    # not used here
    readtext, _ = script_reader([[make_ocr_result([("7", 0.95), ("+", 0.9),
                                                 ("6", 0.9)])]])
    core.reader.readtext = readtext
    frame = FakeGrab()
    original_cancel = core.cancel_all_scheduled_events

    def cancel_recorder():
        cancelled_ids.append(list(core.scheduled_events))
        original_cancel()

    core.cancel_all_scheduled_events = cancel_recorder
    for cycle in range(200):
        run_loop_once(gui, frame)
        time.sleep(0.004)
        if not core.answer_clicks_enabled:
            break
    assert cancelled_ids and cancelled_ids[0] == [4242]
    assert core.scheduled_events == []
    assert core.auto_sequence_enabled is True


# ── answer clicks vs AUTO sequence (Part 6) ─────────────────────────────────

def test_answer_clicks_off_still_ocrs_and_solves(gui, core, click_log):
    """Req 11/12/13: clicks OFF — OCR + solving continue, keypad untouched."""
    core.paused = False
    core.preview_enabled = False
    core.answer_clicks_enabled = False
    readtext, calls = script_reader([[make_ocr_result([("7", 0.95), ("+", 0.9),
                                                     ("6", 0.9)])]])
    core.reader.readtext = readtext
    run_loop_once(gui, FakeGrab())
    assert calls["n"] == 1                       # OCR ran
    assert gui._display_calls[-1][1] == 13       # solved + displayed
    assert click_log == []                       # no keypad clicks


def test_auto_off_does_not_stop_answer_clicks(gui, core, click_log):
    core.paused = False
    core.preview_enabled = False
    core.auto_sequence_enabled = False
    readtext, _ = script_reader([[make_ocr_result([("7", 0.95), ("+", 0.9),
                                                 ("6", 0.9)])]])
    core.reader.readtext = readtext
    run_loop_once(gui, FakeGrab())
    assert len(click_log) == 3                   # normal submission still works


def test_auto_on_does_not_enable_answer_clicks(gui, core, click_log):
    """Req 15: AUTO ON + answer clicks OFF must NOT re-enable submission."""
    core.paused = False
    core.preview_enabled = False
    core.answer_clicks_enabled = False
    core.auto_sequence_enabled = True
    readtext, _ = script_reader([[make_ocr_result([("7", 0.95), ("+", 0.9),
                                                 ("6", 0.9)])]])
    core.reader.readtext = readtext
    run_loop_once(gui, FakeGrab())
    assert click_log == []
    # and the AUTO macro itself remains independently runnable
    assert core._can_auto("AUTO 1") is True


def test_manual_answer_clicks_off_leaves_scheduled_auto_alone(gui, core):
    """Manual toggle (cancel_auto=False) does NOT cancel scheduled AUTO."""
    core.scheduled_events.append(777)
    gui._set_answer_clicks_enabled(False)
    assert core.answer_clicks_enabled is False
    assert core.scheduled_events == [777]


def test_auto_seq_off_cancels_only_auto_scheduling(gui, core):
    core.answer_clicks_enabled = True
    core.scheduled_events.append(888)
    gui._set_auto_sequence_enabled(False)
    assert core.auto_sequence_enabled is False
    assert core.answer_clicks_enabled is True
    assert core.scheduled_events == []
    assert core.extended_sequence_active is False


# ── frame cache (Part 4) ─────────────────────────────────────────────────────

def test_frame_cache_serves_known_pixels_within_ttl(gui, core):
    """Same pixels returning within TTL -> answer served WITHOUT OCR."""
    core.paused = False
    core.preview_enabled = False
    readtext, calls = script_reader([[make_ocr_result([("7", 0.95), ("+", 0.9),
                                                     ("6", 0.9)])]])
    core.reader.readtext = readtext
    frame_a = FakeGrab(seed=7)
    frame_b = frame_a.mutated()
    run_loop_once(gui, frame_a)                  # solve + cache   (OCR #1)
    run_loop_once(gui, frame_b)                  # change           (OCR #2)
    run_loop_once(gui, frame_a)                  # back to A: cache hit
    assert calls["n"] == 2                       # no third pass — A was served
    assert gui._display_calls[-1][0] == "(cached frame)"


def test_frame_cache_expired_triggers_ocr(gui, core):
    core.frame_cache.ttl = 0.05
    core.paused = False
    core.preview_enabled = False
    readtext, calls = script_reader([[make_ocr_result([("7", 0.95), ("+", 0.9),
                                                     ("6", 0.9)])]])
    core.reader.readtext = readtext
    frame_a = FakeGrab(seed=11)
    frame_b = frame_a.mutated()
    run_loop_once(gui, frame_a)
    run_loop_once(gui, frame_b)                  # (OCR call #2: B is new)
    time.sleep(0.06)                             # TTL expires
    run_loop_once(gui, frame_a)
    assert calls["n"] == 3                       # re-OCR'd, NOT stale-served


def test_frame_cache_hit_never_reclicks_completed_question(gui, core, click_log):
    core.paused = False
    core.preview_enabled = False
    readtext, calls = script_reader([[make_ocr_result([("7", 0.95), ("+", 0.9),
                                                     ("6", 0.9)])]])
    core.reader.readtext = readtext
    frame_a = FakeGrab(seed=3)
    frame_b = frame_a.mutated()
    run_loop_once(gui, frame_a)                  # solve + click
    assert len(click_log) == 3
    run_loop_once(gui, frame_b)                  # confirm
    run_loop_once(gui, frame_a)                  # cache hit
    run_loop_once(gui, frame_b)
    assert len(click_log) == 3                   # no duplicate submission


def test_new_round_resets_done_state(gui, core, click_log):
    """Req: same question after a NEW ROUND is answered again."""
    core.paused = False
    core.preview_enabled = False
    readtext, _ = script_reader([[make_ocr_result([("7", 0.95), ("+", 0.9),
                                                 ("6", 0.9)])]])
    core.reader.readtext = readtext
    frame = FakeGrab()
    run_loop_once(gui, frame)
    run_loop_once(gui, frame.mutated())          # confirm
    assert len(click_log) == 3
    gui._clear_transient_state("test new round") # the New Round button path
    run_loop_once(gui, frame.mutated())
    assert len(click_log) == 6                   # answered again


def test_frame_changed_but_same_question_handled(gui, core, click_log):
    """Req 19: pixels change (animation) but the semantic question is the
    same -> one submission, no duplicate clicking."""
    core.paused = False
    core.preview_enabled = False
    readtext, _ = script_reader([[make_ocr_result([("7", 0.95), ("+", 0.9),
                                                 ("6", 0.9)])]])
    core.reader.readtext = readtext
    run_loop_once(gui, FakeGrab(seed=5))
    assert len(click_log) == 3
    run_loop_once(gui, FakeGrab(seed=6))         # same question, new pixels
    run_loop_once(gui, FakeGrab(seed=7))
    assert len(click_log) == 3


# ── main-loop resilience (Part 17) ──────────────────────────────────────────

def test_loop_exception_reschedules(gui, core):
    """GUI loop throws -> error printed, loop still reschedules itself."""
    core.paused = False
    core.preview_enabled = False
    def boom(monitor):
        raise RuntimeError("capture exploded")
    g = gui
    g._sct = types.SimpleNamespace(grab=boom)
    g._main_loop()
    assert g.root.callbacks and g.root.callbacks[0][0] == core.current_polling
