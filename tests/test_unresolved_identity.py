"""
tests/test_unresolved_identity.py — Regression tests for the OCR-flicker
loophole fix (unresolved visual question identity).

The loophole: when OCR produced no canonical expression, question identity
fell back to a hash of the RAW OCR TEXT, so garbage A / B / C from one
unchanged question minted three independent fingerprints — each with a
fresh retry budget (unbounded reprocessing).

The fix: identity for unreadable frames comes from the question-region
PIXELS (bot_core.visual_signature on the already-preprocessed OCR input),
matched by aligned, dead-zoned distance (question_state.signature_distance)
against the current unresolved episode. Small noise/jitter/blink stays in
the same episode; genuinely changed pixels start a new one.

Required scenarios (each mapped to a test below):
  1. same visual question + garbage A/B/C      -> same unresolved episode
  2. small visual noise + garbage OCR          -> same episode
  3. genuinely changed question + garbage OCR  -> new episode
  4. unchanged screen                          -> retries at scheduled times
  5. alternating garbage cannot reset budget   -> budget monotonic, bounded
  6. real new question                         -> fresh retry budget
"""

import types

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

import cv2
import bot_core
import synthetic_data as sd
from bot_core import visual_signature
from conftest import FakeGrab, make_ocr_result
from question_state import (
    SIG_GRID_W, SIG_GRID_H, RetryPolicy, signature_distance, signature_ink,
    OUTCOME_UNSOLVED,
)
from test_retry_engine import make_machine

W, H = 311, 49                       # production DEFAULT_QUESTION_AREA size
GARBAGE = ["###", "???", "~~~"]      # canonicalise() -> "" for all of these
THR = RetryPolicy().unresolved_match_threshold   # 0.0065 — production knob

# Reference calibration (docs/REDESIGN_REPORT.md §2.5: same-question noise
# <= 0.0043, one-digit change >= 0.0090, full change >= 0.0171) was measured
# on ONE reference render (bold 30px sans). Absolute distances vary with the
# platform font discovered below (Arial's narrower digits measure roughly
# half the DejaVu-Bold one-digit distance); what must hold EVERYWHERE is the
# production CONTRACT: same-question noise at/below THR, genuinely changed
# questions above it. The assertions below lock the contract, not the
# reference numbers.

_FONT_CACHE = {}


def _test_font(size=30):
    """Cross-platform test font via the EXISTING discovery mechanism
    (synthetic_data.available_fonts(): Windows + Linux candidates) — no
    bundled font file, no extra dependency, no duplicated candidate list.

    Text is drawn with stroke_width=1 (weight parity across platforms):
    thin regular-weight fonts spread so little ink over the 96x16 block
    grid that genuinely different questions can fall under the dead-zone;
    semibold strokes match the production question-region rendering and
    make the same/different separation font-independent."""
    if size not in _FONT_CACHE:
        font = None
        for path in sd.available_fonts():
            try:
                font = ImageFont.truetype(path, size)
                break
            except OSError:
                continue
        if font is None:
            pytest.skip("no usable TTF font found by "
                        "synthetic_data.available_fonts()")
        _FONT_CACHE[size] = font
    return _FONT_CACHE[size]


# ── frame/signature helpers (real preprocess + real signature code) ─────────

def render_gray(text, cursor=False, shift=0, w=W, h=H, noise_sigma=0.0,
                rng=None):
    img = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)
    font = _test_font(30)                # resolved even for cursor-only draws
    if text:
        d.text((20 + shift, 8), text, fill=20, font=font,
               stroke_width=1, stroke_fill=20)
    if cursor:
        x = 20 + shift + (d.textlength(text, font=font) if text else 0) + 4
        d.rectangle([x, 8, x + 2, 8 + 30], fill=20)
    arr = np.asarray(img)
    if noise_sigma:
        r = rng or np.random.default_rng(0)
        arr = np.clip(arr.astype(np.int16)
                      + r.normal(0, noise_sigma, arr.shape), 0, 255)
    return arr.astype(np.uint8)


def gray_to_bgra(gray):
    g = gray.astype(np.uint8)
    alpha = np.full_like(g, 255)
    return np.stack([g, g, g, alpha], axis=-1)


def sig_of(core, gray):
    """Real production chain: preprocess_for_ocr -> visual_signature."""
    return visual_signature(core.preprocess_for_ocr(gray_to_bgra(gray)))


# ── signature invariants ─────────────────────────────────────────────────────

def test_signature_invariants(core):
    blank = sig_of(core, render_gray(""))
    text = sig_of(core, render_gray("7+6"))
    assert len(blank) == len(text) == SIG_GRID_W * SIG_GRID_H
    assert signature_ink(blank) < 0.015                  # blank is blank
    assert signature_ink(text) >= 0.015
    # determinism
    assert sig_of(core, render_gray("7+6")) == text
    # polarity independence (light-on-dark theme = same identity evidence):
    # the Otsu/CLAHE chain is not bit-symmetric, but the signatures must be
    # well inside the same-episode threshold (dead-zone absorbs edge cells)
    inverted = sig_of(core, 255 - render_gray("7+6"))
    assert signature_distance(text, inverted) <= 0.0045
    # malformed signatures never match
    assert signature_distance(b"x", b"y") == 1.0
    # two blanks are the same "identity" (nothing readable in either)
    assert signature_distance(blank, blank) == 0.0


def test_signature_distance_separation_classes(core):
    """Locks the PRODUCTION separation contract the threshold choice is
    built on (RetryPolicy.unresolved_match_threshold = 0.0065):
      - same-question noise / blur / 1-3px jitter / cursor blink stays at
        or below the threshold -> same unresolved episode;
      - genuinely changed questions measure strictly above it -> new
        episode with a fresh budget;
      - and the two classes never overlap (min changed > max same).
    Absolute distances are font-dependent (see the module docstring): the
    REFERENCE calibration (bold 30px render) measured one-digit changes
    at ~0.0090 and full changes at ~0.0171 (docs/REDESIGN_REPORT.md
    §2.5), but narrower platform fonts measure lower, so the rendered
    assertions here use content changes large enough to clear the
    threshold on EVERY discoverable font (full change / different
    question). The one-digit boundary behaviour itself is locked exactly,
    font-independently, by test_one_digit_change_is_a_new_episode."""
    anchor = sig_of(core, render_gray("7+6"))
    rng = np.random.default_rng(7)
    same = {
        "noise":  sig_of(core, render_gray("7+6", noise_sigma=14, rng=rng)),
        "blur":   sig_of(core, cv2.GaussianBlur(render_gray("7+6"), (5, 5), 1.2)),
        "1px":    sig_of(core, render_gray("7+6", shift=1)),
        "3px":    sig_of(core, render_gray("7+6", shift=3)),
        "cursor": sig_of(core, render_gray("7+6", cursor=True)),
    }
    for name, s in same.items():
        d = signature_distance(anchor, s)
        assert d <= THR, f"{name} distance {d:.4f} above match threshold"
    changed = {
        "full":  sig_of(core, render_gray("8+2")),
        "new_q": sig_of(core, render_gray("12x4")),
    }
    for name, s in changed.items():
        d = signature_distance(anchor, s)
        assert d > THR, f"{name} change {d:.4f} did not exceed threshold"
    assert min(signature_distance(anchor, s) for s in changed.values()) \
        > max(signature_distance(anchor, s) for s in same.values()), \
        "same-question and changed-question distance classes overlap"


# ── scenario 1: same visual question + garbage A/B/C -> same episode ────────

def test_same_visual_question_garbage_abc_same_episode(core):
    m, ck = make_machine()
    sig = sig_of(core, render_gray("7+6"))
    m.observe_visual("d1", ck())
    fps = []
    for garbage in GARBAGE:
        fp = m.observe_unresolved(sig, "ctx", ck())
        fps.append(fp)
        m.observe_question(fp, "", garbage, ck())       # unreadable
        m.record_outcome(OUTCOME_UNSOLVED, ck())        # attempts += 1
    assert len(set(fps)) == 1                            # ONE identity
    assert len(m.snapshot()) == 1                        # ONE runtime
    rt = m.runtime(fps[0])
    assert rt.attempts == 3                              # ONE shared budget


# ── scenario 2: small visual noise + garbage OCR -> same episode ─────────────

def test_small_noise_same_episode(core):
    m, ck = make_machine()
    rng = np.random.default_rng(3)
    variants = [
        sig_of(core, render_gray("7+6")),
        sig_of(core, render_gray("7+6", noise_sigma=14, rng=rng)),
        sig_of(core, render_gray("7+6", shift=2)),
        sig_of(core, render_gray("7+6", cursor=True)),
    ]
    m.observe_visual("d1", ck())
    seen_fps = set()
    for i, sig in enumerate(variants * 2):               # two rounds
        fp = m.observe_unresolved(sig, "ctx", ck())
        seen_fps.add(fp)
        m.observe_question(fp, "", GARBAGE[i % 3], ck())
        m.record_outcome(OUTCOME_UNSOLVED, ck())
    assert len(seen_fps) == 1
    assert len(m.snapshot()) == 1
    assert m.runtime(seen_fps.pop()).attempts == 8       # budget accumulated


# ── scenario 3: genuinely changed question + garbage OCR -> new episode ─────

def test_genuine_change_new_episode(core):
    m, ck = make_machine()
    sig_a = sig_of(core, render_gray("7+6"))
    m.observe_visual("d1", ck())
    fp1 = m.observe_unresolved(sig_a, "ctx", ck())
    m.observe_question(fp1, "", "###", ck())
    m.record_outcome(OUTCOME_UNSOLVED, ck())
    m.record_outcome(OUTCOME_UNSOLVED, ck())
    assert m.runtime(fp1).attempts == 2

    sig_b = sig_of(core, render_gray("8+2"))             # genuinely changed
    fp2 = m.observe_unresolved(sig_b, "ctx", ck())
    assert fp2 != fp1                                    # new identity
    m.observe_question(fp2, "", "???", ck())
    rt2 = m.runtime(fp2)
    assert rt2.attempts == 0                             # fresh budget
    # re-showing A anchors a fresh episode on A's pixels: the derived fp is
    # A's fingerprint again, and A's budget RESUMES (monotonic, bounded) —
    # it is never reset by the identity churn.
    fp3 = m.observe_unresolved(sig_a, "ctx", ck())
    assert fp3 == fp1
    m.observe_question(fp3, "", "~~~", ck())
    assert m.runtime(fp3).attempts == 2


def test_one_digit_change_is_a_new_episode(core):
    """The episode boundary is exactly the match threshold: a change that
    measures ABOVE it (on the production reference render a one-digit swap
    measures ~0.0090) starts a new episode with a fresh budget; a change
    below it stays inside the same episode and shares the budget.

    Synthetic grid signatures are used instead of rendered digits so the
    boundary check is exact and platform-independent — rendered one-digit
    distances vary with the discovered font (see module docstring), while
    THIS behaviour (threshold comparison -> episode split) must not."""
    n = SIG_GRID_W * SIG_GRID_H

    def pattern(off=(), on=()):
        s = bytearray(n)
        for i in range(40):                       # 40 bright cells: ink 0.026
            s[int(i * n / 40) + 7] = 255          # >= min_ink -> not "blank"
        for i in off:
            s[i] = 0
        for i in on:
            s[i] = 255
        return bytes(s)

    base_cells = [int(i * n / 40) + 7 for i in range(40)]
    dark = [i for i in range(n) if i not in base_cells]
    sig_a = pattern()
    # Sub-threshold change: 4 cells differ (distance 4/1536 = 0.0026).
    # Above-threshold change: 12 cells differ (12/1536 = 0.0078). The two
    # variants touch DISJOINT cell groups so neither can bridge to the
    # other through the episode's recent-variant ring (a variant within
    # threshold of the anchor is itself added to the ring; matching is the
    # minimum over anchor + ring, by design — blink tolerance).
    sig_small = pattern(off=base_cells[20:22], on=dark[500:502])
    sig_big = pattern(off=base_cells[0:6], on=dark[900:906])
    assert signature_distance(sig_a, sig_small) <= THR
    assert signature_distance(sig_a, sig_big) > THR
    assert signature_distance(sig_small, sig_big) > THR

    m, ck = make_machine()
    m.observe_visual("d1", ck())
    fp1 = m.observe_unresolved(sig_a, "ctx", ck())
    m.observe_question(fp1, "", "###", ck())
    m.record_outcome(OUTCOME_UNSOLVED, ck())
    fp_small = m.observe_unresolved(sig_small, "ctx", ck())
    assert fp_small == fp1                         # same episode, same budget
    fp2 = m.observe_unresolved(sig_big, "ctx", ck())
    assert fp2 != fp1                              # new identity
    m.observe_question(fp2, "", "???", ck())
    assert m.runtime(fp2).attempts == 0            # fresh budget


# ── scenario 4: unchanged screen -> retries still occur on schedule ─────────

def test_unchanged_screen_retries_on_schedule():
    """B1 must stay dead: the identity mechanism never gates retry TIMING.
    An unchanged frame's unresolved question is retried exactly per
    RetryPolicy: +0.25, +0.50, +1.00, +2.00, then capped at +4.00."""
    m, ck = make_machine()
    sig = b"\x01" * (SIG_GRID_W * SIG_GRID_H)            # synthetic but valid
    m.observe_visual("d1", ck())
    fp = m.observe_unresolved(sig, "ctx", ck())
    m.observe_question(fp, "", "###", ck())
    m.record_outcome(OUTCOME_UNSOLVED, ck())             # attempt 1

    expected_gaps = [0.25, 0.50, 1.00, 2.00, 4.00, 4.00, 4.00]
    for gap in expected_gaps:
        rt = m.runtime()
        due = rt.last_attempt + gap
        ck.advance(gap - 0.01)
        assert not m.should_process(ck()).allowed        # not yet
        ck.advance(0.01)
        dec = m.should_process(ck())
        assert dec.allowed and dec.reason == "retry_due"
        # same episode on every retry — the fingerprint never changes
        assert m.observe_unresolved(sig, "ctx", ck()) == fp
        m.observe_question(fp, "", "???", ck())
        m.record_outcome(OUTCOME_UNSOLVED, ck())
    # bounded: after max_retries the question is exhausted, never suppressed
    ck.advance(4.0)
    assert m.observe_unresolved(sig, "ctx", ck()) == fp
    m.observe_question(fp, "", "~~~", ck())
    m.record_outcome(OUTCOME_UNSOLVED, ck())
    assert m.runtime(fp).exhausted


# ── scenario 5: alternating garbage cannot reset the budget indefinitely ────

def test_alternating_garbage_cannot_reset_budget(core):
    """30 sightings alternating 3 pixel variants AND 3 garbage strings:
    exactly one runtime, attempts monotonic to exhaustion, no resets."""
    m, ck = make_machine()
    rng = np.random.default_rng(11)
    variants = [
        sig_of(core, render_gray("7+6")),
        sig_of(core, render_gray("7+6", noise_sigma=10, rng=rng)),
        sig_of(core, render_gray("7+6", shift=1)),
    ]
    m.observe_visual("d1", ck())
    attempts_seen = []
    for i in range(30):
        sig = variants[i % 3]
        fp = m.observe_unresolved(sig, "ctx", ck())
        m.observe_question(fp, "", GARBAGE[i % 3], ck())
        m.record_outcome(OUTCOME_UNSOLVED, ck())
        attempts_seen.append(m.runtime(fp).attempts)
    assert len(m.snapshot()) == 1                        # ONE identity ever
    assert attempts_seen == sorted(attempts_seen)        # never reset
    assert attempts_seen[0] == 1
    assert attempts_seen[-1] >= 8                        # bounded exhaustion
    rt = m.runtime()
    assert rt.exhausted and rt.next_allowed - rt.last_attempt == 4.0


# ── scenario 6: a real new question receives a fresh retry budget ───────────

def test_real_new_question_gets_fresh_budget(core):
    m, ck = make_machine()
    sig_a = sig_of(core, render_gray("7+6"))
    m.observe_visual("d1", ck())
    fp1 = m.observe_unresolved(sig_a, "ctx", ck())
    m.observe_question(fp1, "", "###", ck())
    for _ in range(5):                                   # mid-budget
        m.record_outcome(OUTCOME_UNSOLVED, ck())
        ck.advance(4.0)
    assert m.runtime(fp1).attempts == 5

    sig_b = sig_of(core, render_gray("12x4"))            # different question
    fp2 = m.observe_unresolved(sig_b, "ctx", ck())
    m.observe_question(fp2, "", "???", ck())
    rt2 = m.runtime(fp2)
    assert fp2 != fp1 and rt2.attempts == 0
    m.record_outcome(OUTCOME_UNSOLVED, ck())
    assert rt2.attempts == 1                             # own fresh budget


def test_readable_question_closes_the_episode(core):
    """Once OCR CAN read the question, the unresolved episode is closed:
    later garbage anchors a fresh episode instead of inheriting state."""
    m, ck = make_machine()
    sig_a = sig_of(core, render_gray("7+6"))
    m.observe_visual("d1", ck())
    fp1 = m.observe_unresolved(sig_a, "ctx", ck())
    m.observe_question(fp1, "", "###", ck())
    m.record_outcome(OUTCOME_UNSOLVED, ck())
    assert m._unresolved is not None

    # OCR recovers: canonical identity takes over
    fp_q = "canonical-fp"
    m.observe_question(fp_q, "8*2", "8*2", ck())
    m.record_outcome(OUTCOME_UNSOLVED, ck())
    assert m._unresolved is None                         # episode resolved

    # unreadable again -> fresh episode anchored on current pixels. The
    # derived fingerprint for IDENTICAL pixels + context is the same, so
    # the budget RESUMES monotonically (bounded — never reset): attempts
    # continue from 1, they do not restart at 0.
    fp2 = m.observe_unresolved(sig_a, "ctx", ck())
    assert fp2 == fp1
    m.observe_question(fp2, "", "???", ck())
    rt2 = m.runtime(fp2)
    assert rt2.attempts == 1
    assert m._unresolved is not None
    assert m._unresolved.fingerprint == fp2              # fresh episode bound


def test_episode_times_out(core):
    """An episode unseen for > unresolved_episode_timeout is replaced by a
    NEW episode object (cannot bind stale pixels to new content). Re-showing
    the IDENTICAL pixels derives the same fingerprint, so the budget
    continues monotonically — never reset."""
    m, ck = make_machine(unresolved_episode_timeout=5.0)
    sig = b"\x02" * (SIG_GRID_W * SIG_GRID_H)
    m.observe_visual("d1", ck())
    fp1 = m.observe_unresolved(sig, "ctx", ck())
    m.observe_question(fp1, "", "###", ck())
    m.record_outcome(OUTCOME_UNSOLVED, ck())
    ep_before = m._unresolved
    assert ep_before.first_seen == 1000.0
    ck.advance(6.0)                                      # past the timeout
    fp2 = m.observe_unresolved(sig, "ctx", ck())
    assert fp2 == fp1                                    # same anchor pixels
    assert m._unresolved is not ep_before                # NEW episode object
    assert m._unresolved.first_seen == 1006.0            # fresh lifecycle
    m.observe_question(fp2, "", "???", ck())
    m.record_outcome(OUTCOME_UNSOLVED, ck())
    assert m.runtime(fp2).attempts == 2                  # monotonic, not reset


# ── GUI end-to-end: the loophole is closed in the real loop ─────────────────

@pytest.fixture
def gui(gui_double):
    """Same convention as test_gui_loop.py's local gui fixture."""
    gui_double.core.last_question = ""
    return gui_double


def test_gui_alternating_garbage_single_episode(gui, core):
    """Drive the REAL _process_frame on a static screen while EasyOCR
    alternates three garbage readings: one runtime, attempts never reset."""
    core.paused = False
    core.preview_enabled = False
    calls = {"n": 0}

    def readtext(arr, **kwargs):
        r = make_ocr_result([(GARBAGE[calls["n"] % 3], 0.5)])
        calls["n"] += 1
        return r

    core.reader.readtext = readtext
    frame = FakeGrab()
    attempts = []
    import time
    for _ in range(10):
        # honour the retry schedule (fast test policy: 0.03/0.06/0.12s)
        rt = core.qsm.runtime()
        if rt is not None:
            wait = rt.next_allowed - time.monotonic()
            if wait > 0:
                time.sleep(wait + 0.005)
        gui._sct = types.SimpleNamespace(grab=lambda monitor: frame)
        gui._main_loop()
        gui.root.callbacks.clear()
        rt = core.qsm.runtime()
        assert rt is not None
        attempts.append(rt.attempts)
    assert calls["n"] >= 3                               # OCR actually ran
    assert len(core.qsm.snapshot()) == 1                 # ONE identity
    assert attempts == sorted(attempts)                  # monotonic budget
    assert attempts[-1] >= 3
