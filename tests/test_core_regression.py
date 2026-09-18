"""
tests/test_core_regression.py — Regression suite for the EXISTING solver /
OCR-selection fixes (Part 13) running the REAL bot_core logic.

These pin the behaviour that must survive the architecture refactor:
canonicalisation convergence, B->8, division-glyph pixel evidence,
enabled-operation filtering, glued-token splitting, date/noise rejection,
reading-order sorting, candidate scoring (confidence + geometry),
missing-operator repair, solve paths, LUT verify, click-result semantics.
"""

import numpy as np
import pytest

import bot_core
from bot_core import (
    CLICK_RESULT_CLICKED, CLICK_RESULT_AUTOMATION_OFF,
    CLICK_RESULT_WRONG_WINDOW, CLICK_RESULT_UNMAPPED_ANSWER,
    CLICK_RESULT_ERROR, MODE_LUT_ONLY, MODE_CALC,
)

from conftest import make_ocr_result


# ── canonicalisation (the semantic key) ──────────────────────────────────────

def test_canonicalise_converges_representations(core):
    """Req Part 2: spacing/case/trailing-mark variants of a reading all
    converge to ONE canonical key before any hashing."""
    assert core.canonicalise("7+6") == "7+6"
    assert core.canonicalise("7 + 6") == "7+6"
    assert core.canonicalise("7+6=") == "7+6"
    assert core.canonicalise("7×?6") == "7*6"
    assert core.canonicalise("7 X 6") == "7*6"
    assert core.canonicalise("7÷6") == "7/6"
    plus_variants = {core.canonicalise(s) for s in ["7+6", "7 + 6", "7+6="]}
    mul_variants = {core.canonicalise(s) for s in ["7×?6", "7 X 6", "7x6"]}
    assert plus_variants == {"7+6"}
    assert mul_variants == {"7*6"}


def test_b_uppercase_maps_to_eight_not_six(core):
    """FIX (audit): 'B':'8' was dead code (lower() ran first), so a misread
    8 was corrupted into 6 via the lowercase 'b'->'6' rule."""
    assert core.canonicalise("B + 6") == "8+6"
    assert core.canonicalise("B+6") == "8+6"
    # lowercase b keeps its historical ->6 mapping
    assert core.canonicalise("b + 6") == "6+6"


def test_letter_digit_corrections(core):
    assert core.canonicalise("o + 5") == "0+5"
    assert core.canonicalise("S + 5") == "5+5"
    assert core.canonicalise("i x 3") == "1*3"
    assert core.canonicalise("g - 4") == "9-4"


# ── solver paths ─────────────────────────────────────────────────────────────

def test_handle_question_arithmetic_fast_path(core):
    answer, source = core.handle_question("8 × 7 = ?")
    assert answer == 56 and source == "solve"
    # second sighting: LUT (written by the fresh solve) answers instantly
    answer2, source2 = core.handle_question("8*7")
    assert answer2 == 56 and source2 == "lut"
    # a third identical question inside the same round, LUT disabled,
    # still hits the session memoization cache
    del core.lut["8*7"]
    answer3, source3 = core.handle_question("8*7")
    assert answer3 == 56 and source3 == "cache"


def test_handle_question_non_integer_not_entered(core):
    """Keypad has no '.' key — non-integer results must not be truncated."""
    answer, source = core.handle_question("7/2")
    assert answer is None and source is None


def test_handle_question_algebra_when_not_fast_mode(core):
    core.fast_mode = False
    answer, source = core.handle_question("3*?=9")
    assert answer == 3


def test_handle_question_lut_only_mode(core):
    core.solve_mode = MODE_LUT_ONLY
    assert core.handle_question("5+5") == (None, None)   # unknown, skipped
    core.answer_cache["5+5"] = 10
    assert core.handle_question("5 + 5") == (10, "cache")


def test_handle_question_calc_mode_bypasses_lut(core):
    core.solve_mode = MODE_CALC
    core.lut["2+2"] = 999                                # poison the LUT
    assert core.handle_question("2+2") == (4, "solve")


def test_lut_record_and_verify_roundtrip(core):
    core.handle_question("12+13")                        # fresh solve -> LUT
    assert core.lut["12+13"] == 25
    core.lut["63/9"] = 72                                # wrong cached answer
    core.lut["1?"] = 5                                   # unsolvable key
    report = core.verify_lut(auto_fix=True)
    assert ("63/9", 72, 7) in report["corrected"]
    assert ("1?", 5) in report["removed"]
    assert core.lut["63/9"] == 7
    assert "1?" not in core.lut


# ── candidate selection ──────────────────────────────────────────────────────

def test_select_prefers_real_expression_over_noise_tokens(core):
    """A stray label/date beside the question must not win."""
    img = np.zeros((60, 400), dtype=np.uint8)
    res = make_ocr_result([("07/03/2026", 0.99), ("7", 0.95), ("+", 0.94),
                           ("6", 0.96)], char_w=22)
    best = core.select_math_ocr_text(res, img)
    assert best == "7 + 6"


def test_select_rejects_date_shaped_single_token(core):
    img = np.zeros((60, 300), dtype=np.uint8)
    res = make_ocr_result([("12/03/2026", 0.95)])
    assert core.select_math_ocr_text(res, img) == ""


def test_select_respects_enabled_operations_filter(core):
    img = np.zeros((60, 300), dtype=np.uint8)
    core.enabled_operations = {"+"}
    res = make_ocr_result([("7", 0.95), ("/", 0.94), ("6", 0.96)])
    assert core.select_math_ocr_text(res, img) == ""     # '/' disabled
    core.enabled_operations = {"/", "+"}
    assert core.select_math_ocr_text(res, img) == "7 / 6"


def test_select_handles_glued_operator_token(core):
    """093fd3c fix: '+6' glued token must split, not get the candidate
    rejected (previously a 0%-accuracy failure mode)."""
    img = np.zeros((60, 300), dtype=np.uint8)
    res = make_ocr_result([("7", 0.95), ("+6", 0.9)])
    assert core.select_math_ocr_text(res, img) == "7 + 6"


def test_select_rejects_malformed_glued_candidate_in_favour_of_real(core):
    img = np.zeros((60, 400), dtype=np.uint8)
    res = make_ocr_result([("7", 0.95), ("+", 0.93), ("6", 0.96),
                           ("+2", 0.55)])
    best = core.select_math_ocr_text(res, img)
    assert best == "7 + 6"


def test_select_out_of_order_tokens_sorted_into_reading_order(core):
    """Reading-order fix: EasyOCR returns tokens out of order (here '6'
    first in the list but right-most on screen); candidates must be built
    in left-to-right order after the row-aware sort."""
    img = np.zeros((60, 300), dtype=np.uint8)
    positions = [(200, 6), (110, 6), (20, 6)]          # 6 -> '+' -> '7'
    res = []
    for (x, y), (text, conf) in zip(positions,
                                    [("6", 0.95), ("+", 0.94), ("7", 0.95)]):
        bbox = [(x, y), (x + 16, y), (x + 16, y + 30), (x, y + 30)]
        res.append((bbox, text, conf))
    best = core.select_math_ocr_text(res, img)
    assert best == "7 + 6"


def test_select_operator_less_candidate_needs_two_digit_groups(core):
    img = np.zeros((60, 300), dtype=np.uint8)
    res = make_ocr_result([("42", 0.95)])
    assert core.select_math_ocr_text(res, img) == ""     # bare number
    res = make_ocr_result([("12", 0.95), ("4", 0.95)])
    assert core.select_math_ocr_text(res, img) == "12 4"  # repaired later


def test_candidate_scoring_confidence_and_geometry(core):
    """Confidence dominates; geometry nudges; both can't rescue a hard
    filter failure."""
    img = np.zeros((60, 400), dtype=np.uint8)
    good = make_ocr_result([("7", 0.98), ("+", 0.97), ("6", 0.98)])
    bad_low_conf = make_ocr_result([("7", 0.55), ("+", 0.53), ("6", 0.54)])
    assert core.select_math_ocr_text(good, img) == "7 + 6"
    # geometry: a same-line coherent pair scores above a scattered one
    aligned = make_ocr_result([("8", 0.9), ("*", 0.9), ("2", 0.9)])
    assert core.select_math_ocr_text(aligned, img) == "8 * 2"
    # malformed fragment can't win by geometry
    frag = make_ocr_result([("+6", 0.99)])
    assert core.select_math_ocr_text(frag, img) == ""


def test_fast_mode_rejects_equals_without_question(core):
    """093fd3c fix: an 'x=5'-style token must be rejected in fast mode and
    must not beat the real expression."""
    img = np.zeros((60, 500), dtype=np.uint8)
    core.fast_mode = True
    res = make_ocr_result([("x", 0.95), ("=", 0.95), ("5", 0.95),
                           ("7", 0.97), ("+", 0.96), ("6", 0.98)])
    best = core.select_math_ocr_text(res, img)
    assert best != "x = 5"
    assert "=" not in best
    # and with a low-confidence stray 'x=5' beside it, the real one wins
    res = make_ocr_result([("x", 0.4), ("=", 0.4), ("5", 0.4),
                           ("7", 0.97), ("+", 0.96), ("6", 0.98)])
    assert core.select_math_ocr_text(res, img) == "7 + 6"


def test_last_ocr_confidence_is_measured(core):
    img = np.zeros((60, 300), dtype=np.uint8)
    res = make_ocr_result([("7", 0.95), ("+", 0.9), ("6", 0.85)])
    core.select_math_ocr_text(res, img)
    assert core.last_ocr_confidence == pytest.approx((0.95 + 0.9 + 0.85) / 3)


# ── division-glyph pixel evidence ────────────────────────────────────────────

def _glyph_image(kind):
    """White-on-black glyph: 'plus' = cross, 'division' = dot/bar/dot."""
    img = np.zeros((60, 60), dtype=np.uint8)
    if kind == "plus":
        img[28:32, 12:48] = 255   # horizontal bar
        img[12:48, 28:32] = 255   # vertical bar
    else:
        img[8:14, 26:34] = 255    # top dot
        img[27:33, 12:48] = 255   # middle bar (wider than dots)
        img[46:52, 26:34] = 255   # bottom dot
    return img


def test_is_division_glyph_pixel_evidence(core):
    div = _glyph_image("division")
    plus = _glyph_image("plus")
    assert core.is_division_glyph(div) is True
    assert core.is_division_glyph(plus) is False


def test_correct_ocr_operators_corrects_only_on_visual_evidence(core):
    div_img = _glyph_image("division")
    bbox = [(4, 6), (52, 6), (52, 54), (4, 54)]
    out = core.correct_ocr_operators([(bbox, "+", 0.9)], div_img)
    assert out == "/"
    plus_img = _glyph_image("plus")
    out = core.correct_ocr_operators([(bbox, "+", 0.9)], plus_img)
    assert out == "+"


# ── click_answer semantics ───────────────────────────────────────────────────

def test_click_answer_clicks_and_counts(click_log, core):
    core.last_question = "7 + 6"
    result = core.click_answer(76, "solve", norm_expr="7+6")
    assert result == CLICK_RESULT_CLICKED
    assert len(click_log) == 3                  # '7' '6' OK
    assert core.answers_count == 1
    assert core.answer_cache.get("7+6") == 76   # keyed by norm_expr


def test_click_answer_respects_answer_clicks_enabled(click_log, core):
    core.answer_clicks_enabled = False
    core.last_question = "7 + 6"
    result = core.click_answer(76, "solve", norm_expr="7+6")
    assert result == CLICK_RESULT_AUTOMATION_OFF
    assert click_log == []
    assert core.answers_count == 0
    assert core.answer_cache.get("7+6") == 76   # answer still "known"


def test_click_answer_wrong_window(click_log, core, monkeypatch):
    class _FakeUser32:
        @staticmethod
        def GetForegroundWindow():
            return 999                            # some other window

    class _FakeWindll:
        user32 = _FakeUser32

    monkeypatch.setattr(bot_core.ctypes, "windll", _FakeWindll, raising=False)
    core.target_hwnd = 1234
    result = core.click_answer(76, "solve", norm_expr="7+6")
    assert result == CLICK_RESULT_WRONG_WINDOW
    assert click_log == []


def test_click_answer_unmapped_is_not_click_error(click_log, core):
    core.key_coords = {"1": (1, 1), "OK": (2, 2)}      # no '3'
    result = core.click_answer(13, "solve", norm_expr="1+2")
    assert result == CLICK_RESULT_UNMAPPED_ANSWER
    assert click_log == []


def test_click_error_result(click_log, core, monkeypatch):
    def boom(x, y):
        raise RuntimeError("injected")
    monkeypatch.setattr(bot_core, "fast_click", boom)
    result = core.click_answer(13, "solve", norm_expr="1+2")
    assert result == CLICK_RESULT_ERROR


def test_click_answer_resets_extended_sequence(click_log, core, monkeypatch):
    core.extended_sequence_active = True
    cancelled = []
    monkeypatch.setattr(core, "cancel_all_scheduled_events",
                        lambda: cancelled.append(1))
    core.click_answer(13, "solve", norm_expr="1+2")
    assert core.extended_sequence_active is False
    assert cancelled == [1]


def test_auto_sequence_independent_of_answer_clicks(core):
    """_can_auto gates on auto_sequence_enabled — NOT on answer clicks."""
    core.answer_clicks_enabled = False
    core.auto_sequence_enabled = True
    assert core._can_auto("AUTO 1") is True
    core.auto_sequence_enabled = False
    assert core._can_auto("AUTO 1") is False


def test_new_round_clears_transients_but_not_lut(click_log, core):
    core.handle_question("5+5")
    core.lut["3+3"] = 6
    core.qsm.observe_question("fp", "5+5", "5+5", 0.0)
    core.frame_cache.put("digest", 10, "solve", "fp", "5+5")
    core.clear_cache_for_new_session()
    assert core.answer_cache == {} and core.session_cache == {}
    assert len(core.frame_cache) == 0
    assert core.qsm.current_fingerprint is None
    assert core.lut["3+3"] == 6                        # LUT untouched
