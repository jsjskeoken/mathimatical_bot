"""
benchmarks/benchmark_retry.py — Retry / fingerprint architecture benchmark.

Drives a virtual-clock simulation of screen frames + OCR outcomes through:

  NEW: the real gui._process_frame / _run_click_confirmation pipeline
       (real BotCore, real question_state.RetryPolicy)
  OLD: a faithful transcription of the 51b16f4 loop semantics
       (frame-hash early return, un-TTL'd frame cache, 50 ms last_question
       debounce, confirm watcher, coupled automation kill-switch) — clearly
       labelled a reconstruction, since no benchmark harness existed at
       that revision.

Scenarios (virtual 30 s windows, 10 ms cycle):
  S1 static screen, click lands but app never progresses (frozen feedback)
  S2 frozen screenshot for 10 s after a failed reading
  S3 static screen, valid question, answer clicking OFF
  S4 animation jitter, same question, click confirms normally
  S5 OCR alternates between two bad readings (animation between them)
  S6 question changes (7+6 -> 8+2)
  S7 static screen, low-confidence garbage then clean reading (OCR "improves")
  S8 unreadable question: animation cycles 3 near-identical frames while OCR
     returns garbage (raw "") for 2 s, the screen freezes for 3 s, then a
     genuinely DIFFERENT unreadable question appears. OLD gives every
     unreadable frame ONE constant identity (cannot tell questions apart)
     and suppresses the frozen phase entirely (B1); NEW anchors one episode
     for the first question, revalidates the frozen phase on schedule
     (bounded), and starts a FRESH episode for the new question.

Metrics: keypad actions, submission attempts, OCR passes, confirmed
submissions, keypad actions beyond confirmed submissions, and question
identities minted. "Keypad actions" are the individual key presses
.click_answer sends (a 2-digit submission costs 3: two digits + Enter), so
NEW S1/S7's 9 keypad actions are exactly the 3 bounded unconfirmed
submission attempts of the RetryPolicy cap — NOT 9 separate submissions.
"Keypad actions beyond confirmed" includes those DELIBERATE bounded
unconfirmed retries (NEW) — they stop via backoff + the safety net.
No uncontrolled click spam may appear in any scenario for either loop.

Usage:  python benchmarks/benchmark_retry.py [--window 30]
"""

import argparse
import os
import sys
import tempfile
import time as time_mod
from fnmatch import fnmatch
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tests"))

import conftest                     # installs platform stubs
conftest._install_stubs()

import numpy as np
import types

import bot_core
import gui as gui_mod
from bot_core import BotCore, CLICK_RESULT_CLICKED
from conftest import FakeGrab, make_ocr_result
from question_state import (QuestionStateMachine, RetryPolicy, TTLFrameCache,
                            frame_digest)

# All keypad clicks are recorded, never performed.
CLICK_LOG = []
bot_core.fast_click = lambda x, y: CLICK_LOG.append((x, y))

GOOD = lambda: make_ocr_result([("7", 0.95), ("+", 0.9), ("6", 0.9)])
GOOD2 = lambda: make_ocr_result([("8", 0.95), ("+", 0.9), ("2", 0.9)])
BAD = lambda: make_ocr_result([("garbled", 0.5)])
LOWBAD = lambda: make_ocr_result([("7", 0.2), ("?", 0.15), ("6", 0.2)])
BADA = lambda: make_ocr_result([("7", 0.5), ("#", 0.4), ("6", 0.5)])
BADB = lambda: make_ocr_result([("8", 0.5), ("#", 0.4), ("2", 0.5)])

# Virtual clock shared by every driver: the whole simulation runs in real
# milliseconds but must behave as if the full scenario window elapsed.
VCLOCK = {"t": 0.0}
time_mod.monotonic = lambda: VCLOCK["t"]      # gui._process_frame reads this
t_now = VCLOCK


class Scenario:
    def __init__(self, name, desc, window, frame_fn, ocr_fn,
                 clicks_enabled=True, no_safety_net=False):
        self.name = name
        self.desc = desc
        self.window = window
        self.frame_fn = frame_fn        # t -> FakeGrab frame
        self.ocr_fn = ocr_fn            # t -> ocr results
        self.clicks_enabled = clicks_enabled
        # S11: disable the 3-strike safety net so the EXHAUSTION stop (not
        # the safety net) is what ends the clicking.
        self.no_safety_net = no_safety_net


# NOTE: every lambda binds its frame objects via DEFAULT ARGS (f=..., g=...)
# so no scenario's state can be aliased by another's.
S1 = Scenario(
    "S1 frozen-feedback",
    "static screen, valid question, click lands but screen NEVER changes",
    12.0,
    frame_fn=lambda t, f=FakeGrab(seed=1): f,
    ocr_fn=lambda t: GOOD(),
)
S2 = Scenario(
    "S2 frozen-failure",
    "frozen screenshot 10 s, OCR returns garbage the whole time",
    12.0,
    frame_fn=lambda t, f=FakeGrab(seed=2): f,
    ocr_fn=lambda t: BAD(),
)
S3 = Scenario(
    "S3 clicks-off",
    "static screen, valid question, answer clicking OFF",
    8.0,
    frame_fn=lambda t, f=FakeGrab(seed=3): f,
    ocr_fn=lambda t: GOOD(),
    clicks_enabled=False,
)
S4 = Scenario(
    "S4 animation",
    "animation every 200 ms, same question, click CONFIRMS (screen moves on)",
    8.0,
    frame_fn=lambda t, f=FakeGrab(seed=4): (
        f.mutated() if int(t / 0.2) % 2 else f),
    ocr_fn=lambda t: GOOD(),
)
S5 = Scenario(
    "S5 alternating-wrong",
    "OCR alternates two WRONG readings, each solving to a different wrong "
    "answer; frames alternate (flicker)",
    12.0,
    frame_fn=lambda t, f=FakeGrab(seed=5), g=FakeGrab(seed=6): (
        f if int(t / 0.2) % 2 else g),
    ocr_fn=lambda t: BADA() if int(t / 0.2) % 2 else BADB(),
)
S6 = Scenario(
    "S6 question-change",
    "7+6 solved & confirmed (screen updates at t=0.3); at t=4 question "
    "becomes 8+2, confirmed at t=4.3",
    8.0,
    frame_fn=lambda t, f=FakeGrab(seed=7), g=FakeGrab(seed=8): (
        f if t < 0.3 else
        f.mutated() if t < 4 else
        g if t < 4.3 else g.mutated()),
    ocr_fn=lambda t: GOOD() if t < 4 else GOOD2(),
)
S7 = Scenario(
    "S7 ocr-improves",
    "static screen: garbage for 1 s, then the app rerenders once and the "
    "clean reading appears",
    10.0,
    frame_fn=lambda t, f=FakeGrab(seed=9): f if t < 1.0 else f.mutated(),
    ocr_fn=lambda t: BAD() if t < 1.0 else GOOD(),
)


# ── S8 helpers: three near-identical frames + three garbage readings ───────

def _noise_variant(base, y, x):
    """A FakeGrab differing from `base` by ONE pixel (same screen content,
    different digest — exactly what animation jitter produces)."""
    clone = FakeGrab.__new__(FakeGrab)
    clone._arr = base._arr.copy()
    clone._arr[y, x, 0] ^= 0xFF
    return clone


_S8_A = FakeGrab(seed=8)                      # unreadable question 1
_S8_A_VARIANTS = [_noise_variant(_S8_A, 0, 0),
                  _noise_variant(_S8_A, 1, 1),
                  _noise_variant(_S8_A, 2, 2)]
_S8_B = FakeGrab(seed=12)                     # genuinely different question
_S8_B_VARIANTS = [_noise_variant(_S8_B, 0, 0),
                  _noise_variant(_S8_B, 1, 1)]
_S8_GARBAGE = [
    lambda: make_ocr_result([("garbled", 0.5)]),
    lambda: make_ocr_result([("#?!", 0.5)]),
    lambda: make_ocr_result([("~x~", 0.5)]),
]

S8 = Scenario(
    "S8 garbage-flicker",
    "unreadable question: animated screen (3 pixel variants @200 ms) with "
    "garbage OCR for 2 s, frozen 3 s, then a genuinely DIFFERENT unreadable "
    "question — episode anchoring vs one constant identity",
    10.0,
    frame_fn=lambda t: (_S8_A_VARIANTS[int(t / 0.2) % 3] if t < 2.0
                        else _S8_A_VARIANTS[0] if t < 5.0
                        else _S8_B_VARIANTS[int(t / 0.2) % 2]),
    ocr_fn=lambda t: _S8_GARBAGE[int(t / 0.2) % 3](),
)


# ── S9-S12 (forensic audit PART 23): regression scenarios for the six ────────
#    confirmed defects. Every scenario's NEW column must show the invariant
#    named in its description.

def _blink_variants(base, n):
    """n near-identical frames (1-pixel flips): same content, new digests —
    cursor-blink / progress-bar-level animation."""
    return [_noise_variant(base,
                           (i * 7) % base._arr.shape[0],
                           (i * 13) % base._arr.shape[1])
            for i in range(n)]


def _hostile_frames(n):
    """n genuinely DIFFERENT renders (video-background-level churn): every
    digest AND every pixel signature differs from every other."""
    return [FakeGrab(seed=900 + i) for i in range(n)]


_S9_FRAMES = _blink_variants(FakeGrab(seed=21), 4)     # blink cycle @100 ms
S9 = Scenario(
    "S9 animation-starve",
    "unsolved question on a blinking screen (near-identical frames @100 ms, "
    "faster than same_frame_retry_delay): retries must PROCEED on schedule "
    "(bounded by the attempt budget), never starve",
    20.0,
    frame_fn=lambda t: _S9_FRAMES[int(t / 0.1) % 4],
    ocr_fn=lambda t: BAD(),
)

_S10_FRAMES = _hostile_frames(24)                      # big-delta churn @100 ms
S10 = Scenario(
    "S10 animation-ocr-budget",
    "hostile screen: FULLY re-rendered frames every 100 ms (defeats "
    "pixel-identity matching), unsolved question — OCR passes must stay "
    "bounded by the retry schedule + probe cap, never one per poll",
    20.0,
    frame_fn=lambda t: _S10_FRAMES[int(t / 0.1) % 24],
    ocr_fn=lambda t: BAD(),
)

S11 = Scenario(
    "S11 exhausted-answer",
    "valid question, click lands but screen NEVER confirms, safety net "
    "disabled: after the retry budget is spent, clicking stops for this "
    "question (forever, until its identity changes) while OCR re-validation "
    "continues",
    40.0,
    frame_fn=lambda t, f=FakeGrab(seed=22): f,
    ocr_fn=lambda t: GOOD(),
    no_safety_net=True,
)

_S12_BLACK = FakeGrab.__new__(FakeGrab)
_S12_BLACK._arr = np.zeros((50, 300, 4), dtype=np.uint8)
_S12_BLACK._seed = -1
S12 = Scenario(
    "S12 no-fingerprint",
    "blank/unreadable screen from the very first poll: the fail-closed gate "
    "must still discover the unresolved identity (exactly ONE episode, no "
    "spurious states, no clicks) and stay bounded",
    10.0,
    frame_fn=lambda t: _S12_BLACK,
    ocr_fn=lambda t: [],
)

SCENARIOS = [S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12]


# ── NEW loop driver ──────────────────────────────────────────────────────────

def make_new_core():
    # LUT isolation: main() redirected bot_core.LUT_FILE to a fresh per-run
    # temp dir BEFORE any core was constructed. bot_core reads LUT_FILE
    # dynamically at load/save/clear time, so this core (and every async
    # save thread it spawns) can only ever touch the temp file — never the
    # repository's real optical_lut.json.
    #
    # Forensic-audit PART 17 hardening: main()'s redirect only exists when
    # the benchmark is RUN. When this module is IMPORTED and make_new_core()
    # is driven directly (as the forensic audit's own traces did), the
    # redirect never happened — and `core.lut = {}` + the async saver then
    # truncated the real shipped LUT to "{}" at process exit. Two defenses,
    # so the constructor is safe in EVERY context:
    #   1. a guaranteed temp redirect set here (main() overrides it later);
    #   2. async persistence disabled outright — benchmarks measure
    #      in-memory behaviour, and no daemon thread may ever touch disk.
    import tempfile as _tempfile
    bot_core.LUT_FILE = os.path.join(_tempfile.gettempdir(),
                                     "mathbot_bench_lut_isolated.json")
    core = BotCore()
    core.lut = {}
    core._save_lut_async = lambda: None
    policy = RetryPolicy()                     # production defaults
    core.retry_policy = policy
    # Explicit virtual clock: the RetryPolicy defaults bind time.monotonic
    # at import time, so the machine must be handed the patched clock.
    core.qsm = QuestionStateMachine(policy, clock=lambda: VCLOCK["t"])
    core.frame_cache = TTLFrameCache(policy.frame_cache_ttl,
                                     clock=lambda: VCLOCK["t"])
    core.target_hwnd = None
    return core


def make_gui_double(core):
    g = gui_mod.OpticalReaderSolverGUI.__new__(gui_mod.OpticalReaderSolverGUI)
    g.core = core
    core.ui = None
    g.last_frame_hash = None
    g._preview_counter = 0
    g._preview_force = False
    g.CONFIRM_TIMEOUT = core.retry_policy.click_confirm_timeout
    g.UNCONFIRMED_THRESHOLD = core.retry_policy.unconfirmed_threshold
    g.preview_canvas = types.SimpleNamespace(
        images=[], create_image=lambda *a, **k: 1,
        itemconfig=lambda *a, **k: None, delete=lambda *a: None,
        create_text=lambda *a, **k: 0)
    g.preview_img_tk = None
    g.save_ocr_captures_var = types.SimpleNamespace(get=lambda: False)
    g.root = types.SimpleNamespace(after=lambda *a: None, callbacks=[])
    g.answer_clicks_btn = types.SimpleNamespace(config=lambda **k: None)
    g.auto_seq_btn = types.SimpleNamespace(config=lambda **k: None)
    g.set_auto_status = lambda text, color="gray": None
    g._display_calls = []
    g._update_detected_display = lambda *a, **k: g._display_calls.append(a)
    return g


class NewDriver:
    """Real NEW pipeline on a virtual clock."""

    def __init__(self, scenario, click_log):
        self.sc = scenario
        self.click_log = click_log
        self.core = make_new_core()
        self.core.answer_clicks_enabled = scenario.clicks_enabled
        if getattr(scenario, "no_safety_net", False):
            # S11: the 3-strike safety net must not fire before the retry
            # budget does, so the exhaustion stop is what ends the clicking.
            from dataclasses import replace as _dc_replace
            self.core.qsm.policy = _dc_replace(
                self.core.retry_policy, unconfirmed_threshold=10 ** 9)
            self.core.retry_policy = self.core.qsm.policy
        self.ocr_calls = {"n": 0}

        def readtext(arr, **kw):
            self.ocr_calls["n"] += 1
            return scenario.ocr_fn(VCLOCK["t"])

        self.core.reader.readtext = readtext
        # count distinct submission attempts (click_answer calls that fired);
        # each 2-digit submission costs 3 keypad actions in CLICK_LOG
        self.submission_attempts = 0
        real_click = self.core.click_answer

        def _counting_click(answer, source, **kw):
            r = real_click(answer, source, **kw)
            if r == CLICK_RESULT_CLICKED:
                self.submission_attempts += 1
            return r

        self.core.click_answer = _counting_click
        self.g = make_gui_double(self.core)

    def run(self):
        self.ocr_times = []
        self._keypad_at_exhausted = None      # forensic-audit S11 metric
        for t in np.arange(0, self.sc.window, 0.01):
            VCLOCK["t"] = float(t)
            frame = self.sc.frame_fn(t)
            # NEW pipeline: capture -> digest -> confirm watcher -> solver
            digest = frame_digest(frame.bgra)
            self.g.last_frame_hash = digest
            self.g._run_click_confirmation(digest)
            if not self.core.paused:
                prev = self.ocr_calls["n"]
                self.g._process_frame(frame, digest)
                if self.ocr_calls["n"] > prev:
                    self.ocr_times.append(round(t, 3))
                rt = self.core.qsm.runtime()
                if (rt is not None and rt.exhausted
                        and self._keypad_at_exhausted is None):
                    self._keypad_at_exhausted = len(self.click_log)

    def metrics(self):
        c = self.core
        return {
            "keypad_actions": len(self.click_log),
            "submission_attempts": self.submission_attempts,
            "ocr_passes": self.ocr_calls["n"],
            "clicks_enabled_at_end": c.answer_clicks_enabled,
            "submissions_confirmed": sum(
                1 for rt in c.qsm.snapshot().values()
                if rt.get("click_confirmed")),
            # OCR-flicker loophole metric: how many distinct question
            # identities (retry budgets) were minted. One unreadable screen
            # must produce exactly ONE.
            "question_states": len(c.qsm.snapshot()),
            # Forensic-audit S11 metric: keypad actions delivered AFTER the
            # retry budget was exhausted. Contract: always 0.
            "keypad_after_exhausted": (
                len(self.click_log) - self._keypad_at_exhausted
                if self._keypad_at_exhausted is not None else 0),
        }


t_now = [0.0]


# ── OLD loop (faithful transcription of 51b16f4 semantics) ───────────────────

class OldDriver:
    """
    Reconstruction of the 51b16f4 _main_loop decision logic, driving the
    same BotCore: exact frame-hash early return, un-TTL'd frame cache,
    raw-text last_question debounce with 50 ms reset, confirm watcher,
    and the coupled automation kill-switch (both flags flipped together).
    """

    CONFIRM_TIMEOUT = 0.5
    UNCONFIRMED_THRESHOLD = 3

    def __init__(self, scenario, click_log):
        self.sc = scenario
        self.click_log = click_log
        self.core = make_new_core()            # same solver underneath
        self.core.answer_clicks_enabled = scenario.clicks_enabled
        self.last_frame_hash = None
        self.frame_answer_cache = {}
        self._pending_confirm_hash = None
        self._pending_confirm_deadline = None
        self._consecutive_unconfirmed = 0
        self.core.last_question = ""
        self._reset_at = None
        self.ocr_calls = {"n": 0}
        self.readings = set()                  # distinct raw readings seen

        def readtext(arr, **kw):
            # NOTE: the counter is incremented in run() next to the OCR
            # block; this closure must stay side-effect-free or the count
            # would be doubled.
            return scenario.ocr_fn(VCLOCK["t"])

        self.core.reader.readtext = readtext
        self.confirmed = 0
        # count distinct submission attempts (click_answer calls that fired);
        # each 2-digit submission costs 3 keypad actions in CLICK_LOG
        self.submission_attempts = 0
        real_click = self.core.click_answer

        def _counting_click(answer, source, **kw):
            r = real_click(answer, source, **kw)
            if r == CLICK_RESULT_CLICKED:
                self.submission_attempts += 1
            return r

        self.core.click_answer = _counting_click

    def _click(self, answer, source, digest, t):
        r = self.core.click_answer(answer, source)
        if r == CLICK_RESULT_CLICKED:
            self._pending_confirm_hash = digest
            self._pending_confirm_deadline = t + self.CONFIRM_TIMEOUT

    def run(self):
        self.ocr_times = []
        for t in np.arange(0, self.sc.window, 0.01):
            VCLOCK["t"] = float(t)
            core = self.core
            if self._reset_at is not None and t >= self._reset_at:
                core.last_question = ""
                self._reset_at = None
            frame = self.sc.frame_fn(t)
            digest = frame_digest(frame.bgra)

            # click confirmation (identical to 51b16f4)
            if self._pending_confirm_hash is not None:
                if digest != self._pending_confirm_hash:
                    self._pending_confirm_hash = None
                    self._consecutive_unconfirmed = 0
                    self.confirmed += 1
                elif t >= self._pending_confirm_deadline:
                    self._pending_confirm_hash = None
                    self._consecutive_unconfirmed += 1
                    if (self._consecutive_unconfirmed
                            >= self.UNCONFIRMED_THRESHOLD):
                        # OLD coupling: one switch kills answer clicks AND
                        # the auto sequence
                        core.answer_clicks_enabled = False
                        core.auto_sequence_enabled = False
                        self._consecutive_unconfirmed = 0

            # frame-hash early return (the bug under test)
            if digest == self.last_frame_hash:
                continue
            self.last_frame_hash = digest

            # un-TTL'd frame cache
            if digest in self.frame_answer_cache:
                cached_answer, cached_source = self.frame_answer_cache[digest]
                if core.last_question == "":
                    self._click(cached_answer, cached_source, digest, t)
                    self._reset_at = t + 0.05
                continue

            raw_np = np.array(frame)
            arr = core.preprocess_for_ocr(raw_np)
            result = core.reader.readtext(
                arr, allowlist='0123456789+-*/()=?xX×÷: ',
                low_text=0.3, batch_size=1, paragraph=False, min_size=5)
            self.ocr_calls["n"] += 1
            self.ocr_times.append(round(t, 3))

            answer, source = None, None
            if result:
                raw = core.select_math_ocr_text(result, arr)
                self.readings.add(raw)
                if raw != core.last_question:
                    core.last_question = raw
                    answer, source = core.handle_question(raw)
                    self._reset_at = t + 0.05
                    if answer is not None:
                        self._click(answer, source, digest, t)
            if answer is not None:
                if len(self.frame_answer_cache) >= 500:
                    oldest = next(iter(self.frame_answer_cache))
                    del self.frame_answer_cache[oldest]
                self.frame_answer_cache[digest] = (answer, source)

    def metrics(self):
        c = self.core
        return {
            "keypad_actions": len(self.click_log),
            "submission_attempts": self.submission_attempts,
            "ocr_passes": self.ocr_calls["n"],
            "clicks_enabled_at_end": c.answer_clicks_enabled,
            "submissions_confirmed": self.confirmed,
            "ocr_times": self.ocr_times,
            # OLD semantics: every distinct raw reading was its own
            # "question" (raw-text identity) with a fresh implicit budget.
            "question_states": len(self.readings),
            # The OLD loop has no exhaustion concept (it used a global
            # kill-switch instead), so it has no post-exhaustion clicks by
            # definition.
            "keypad_after_exhausted": 0,
        }


# ── measurement ──────────────────────────────────────────────────────────────

def drain_lut_writes(core):
    """Block until any in-flight async LUT save of this core has landed.

    The writer thread holds _lut_write_lock for the whole tmp-write +
    os.replace, so acquiring it once means this core has no save in flight
    and cannot start one. Called between drivers so two cores' async saves
    (each with its own sequence counter, sharing one temp LUT path) can
    never race their tmp files. Benchmark-only hygiene — production code
    is untouched."""
    try:
        with core._lut_write_lock:
            pass
    except AttributeError:
        pass


def _is_within(path, directory):
    """True if <path> resolves inside <directory> (both absolute)."""
    return Path(os.path.abspath(path)).resolve().is_relative_to(
        Path(os.path.abspath(directory)).resolve())


def open_bench_dir():
    """TemporaryDirectory OUTSIDE the repository, cleaned up on success AND
    exceptions (it is a context manager).

    Every temp file this benchmark creates (the per-run LUT) lives here.
    The old mkdtemp/rmtree pair was only reached on the success path — a
    crash mid-run leaked the tree (into the system temp dir, never the
    repo, but still hygiene). Same helper as benchmark_ocr.py; kept
    self-contained so each benchmark script runs standalone."""
    td = tempfile.TemporaryDirectory(prefix="mathbot_bench_")
    if _is_within(td.name, REPO):
        td.cleanup()
        raise RuntimeError(
            f"benchmark temp dir {td.name} resolved INSIDE the repository "
            f"({REPO}) — refusing to run, it would dirty the working tree")
    return td


def scan_temp_leftovers(root=REPO):
    """Report benchmark-temp leftovers under <root> as sorted rel paths.

    The exact violation class the Windows run hit for the OCR benchmark —
    tmpXXXXXXXX_<rev>.py files (mkstemp's default 'tmp' + 8 random chars)
    and stray mathbot_bench_* trees — while skipping .git/ and
    __pycache__/. main() uses it as a post-run tripwire: the run must leave
    the working tree exactly as it found it. Same helper as
    benchmark_ocr.py; kept self-contained."""
    hits = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__")]
        for name in dirnames:
            if fnmatch(name, "mathbot_bench_*"):
                hits.append(os.path.relpath(os.path.join(dirpath, name),
                                            root))
        for name in filenames:
            if fnmatch(name, "tmp*.py") or fnmatch(name, "mathbot_bench_*"):
                hits.append(os.path.relpath(os.path.join(dirpath, name),
                                            root))
    return sorted(hits)


def measure(scenario):
    results = {}
    for label, Driver in (("OLD (51b16f4 reconstruction)", OldDriver),
                          ("NEW (retry/fingerprint)", NewDriver)):
        click_log = []
        # drivers append to their own list via the patched fast_click
        bot_core.fast_click = lambda x, y, _l=click_log: _l.append((x, y))
        VCLOCK["t"] = 0.0
        drv = Driver(scenario, click_log)
        drv.run()
        drain_lut_writes(drv.core)
        m = drv.metrics()
        m["keypad_beyond_confirmed"] = max(
            0, m["keypad_actions"] - (3 * m["submissions_confirmed"])
        ) if m["submissions_confirmed"] else m["keypad_actions"]
        results[label] = m
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=None,
                    help="override scenario window (seconds)")
    args = ap.parse_args()
    if args.window:
        for s in SCENARIOS:
            s.window = args.window

    # Temp hygiene, step 1 — snapshot pre-existing leftovers (if any) so the
    # post-run audit never blames this run for older debris.
    pre_existing = scan_temp_leftovers()
    if pre_existing:
        print("NOTE: pre-existing benchmark temp leftovers in the working "
              "tree (NOT created by this run — from an older benchmark "
              "version; safe to delete manually):")
        for rel in pre_existing:
            print(f"  - {rel}")

    # Temp hygiene, step 2 — the TemporaryDirectory context manager owns
    # every temp file this run creates and is removed on success AND on any
    # exception (the old mkdtemp/rmtree pair was only reached on success).
    # Isolate the LUT BEFORE any core exists: every BotCore below (OLD and
    # NEW drivers) loads from — and asynchronously saves to — this fresh
    # per-run temp file only. bot_core reads LUT_FILE dynamically at
    # load/save/clear time, so the repository's real optical_lut.json is
    # unreachable for the whole process, even after the benchmark exits.
    with open_bench_dir() as bench_dir:
        bot_core.LUT_FILE = os.path.join(bench_dir, "lut.json")

        print("Retry / fingerprint architecture benchmark")
        print("(keypad actions = individual key presses; a 2-digit submission "
              "costs 3 keypad actions")
        print(" — so e.g. NEW S1's 9 keypad actions are its 3 bounded "
              "unconfirmed submission")
        print(" attempts at the RetryPolicy cap, NOT 9 separate submissions. "
              "Both loops use")
        print(" two-digit answers (13/14/12) throughout, so every submission "
              "costs exactly 3.\n")
        header = (f"{'scenario':<20} {'metric':<32} "
                  f"{'OLD (51b16f4 recon)':>20} {'NEW (retry/fingerprint)':>24}")
        print(header)
        print("-" * len(header))
        for sc in SCENARIOS:
            res = measure(sc)
            o, n = (res["OLD (51b16f4 reconstruction)"],
                    res["NEW (retry/fingerprint)"])
            for metric, label in [
                    ("keypad_actions", "keypad actions"),
                    ("submission_attempts", "submission attempts"),
                    ("ocr_passes", "ocr passes"),
                    ("submissions_confirmed", "confirmed submissions"),
                    ("keypad_beyond_confirmed",
                     "keypad actions beyond confirmed"),
                    ("question_states", "question states"),
                    ("keypad_after_exhausted",
                     "keypad actions after exhausted")]:
                print(f"{sc.name:<20} {label:<32} "
                      f"{o[metric]:>20} {n[metric]:>24}")
            print(f"{sc.name:<20} {'clicks enabled at end':<32} "
                  f"{str(o['clicks_enabled_at_end']):>20} "
                  f"{str(n['clicks_enabled_at_end']):>24}")
            print(f"{sc.name:<20} {'description':<32} {sc.desc}")
            print()
    # ^ TemporaryDirectory removed the whole bench tree here — success OR
    #   exception.

    # Temp hygiene, step 3 — tripwire: no NEW tmp*.py / mathbot_bench_*
    # leftovers in the working tree after the run.
    new_leftovers = [p for p in scan_temp_leftovers() if p not in pre_existing]
    if new_leftovers:
        print("ERROR: this run left temporary files in the working tree "
              "(this is a benchmark bug — please report it):")
        for rel in new_leftovers:
            print(f"  - {rel}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
