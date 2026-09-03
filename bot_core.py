"""
bot_core.py — Business logic, solver, OCR, LUT, coordinate management.
No tkinter imports here. All UI callbacks are injected via the `ui` reference
that the GUI sets after construction so the core can update labels etc.
"""

import easyocr
from PIL import ImageGrab
import pyautogui
import numpy as np
import cv2
from sympy import symbols, Eq, solve, sympify, N
import re
import time
import sys
import os
import json
import ctypes
import ctypes.wintypes  # must be imported explicitly — accessing
                         # ctypes.wintypes without this only worked before
                         # because some other import happened to pull it in
                         # as a side effect; that's not guaranteed.
import threading
from pynput import keyboard as pynput_keyboard

# ── Safety ──────────────────────────────────────────────────────────────────
# Disable pyautogui fail-safe — without this, moving the mouse to (0,0)
# raises FailSafeException which kills the process instantly.
pyautogui.FAILSAFE = False
# Remove the 0.1s hidden pause PyAutoGUI injects after every action.
# A 3-digit answer + OK = 4 actions = 0.4s of dead time without this.
pyautogui.PAUSE = 0

# ── File paths ───────────────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.abspath(__file__))
SLOTS_FILE = os.path.join(_BASE, "optical_coords.json")
LUT_FILE   = os.path.join(_BASE, "optical_lut.json")
MAX_SLOTS  = 3

# ── DPI / scaling ────────────────────────────────────────────────────────────
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

# ── Elevation check ──────────────────────────────────────────────────────────
# The F8 hotkey (see _start_hotkey_listener) uses a global OS-level keyboard
# hook, so it doesn't depend on this GUI having focus. But if the target
# application is running elevated (as Administrator) and this script is not,
# Windows can block a lower-privilege process from receiving input while that
# elevated window is in the foreground — the hotkey silently stops firing.
# If F8 does nothing while the target app is focused, this is almost always why.
try:
    if not ctypes.windll.shell32.IsUserAnAdmin():
        print("[CORE] WARNING: not running as Administrator. If F8 stops "
              "responding while the target app is focused, that app is "
              "probably elevated — run this script as Administrator too.")
except Exception:
    pass

REF_W, REF_H = 1366, 768
CURR_W, CURR_H = pyautogui.size()
SCALE_X = CURR_W / REF_W
SCALE_Y = CURR_H / REF_H
print(f"[CORE] Screen: {CURR_W}x{CURR_H}  scale: {SCALE_X:.2f}x {SCALE_Y:.2f}x")


def s_xy(x, y):
    return (int(x * SCALE_X), int(y * SCALE_Y))

def s_bbox(bbox):
    return (int(bbox[0]*SCALE_X), int(bbox[1]*SCALE_Y),
            int(bbox[2]*SCALE_X), int(bbox[3]*SCALE_Y))


def fast_click(x, y):
    """
    Bypass PyAutoGUI entirely for zero-latency hardware-level clicks.
    Uses Windows API directly: SetCursorPos + mouse_event (down then up).
    No Python overhead, no metric recalculation — the OS receives the input
    in the same instruction cycle.
    """
    ctypes.windll.user32.SetCursorPos(int(x), int(y))
    ctypes.windll.user32.mouse_event(0x0002, 0, 0, 0, 0)  # MOUSEEVENTF_LEFTDOWN
    ctypes.windll.user32.mouse_event(0x0004, 0, 0, 0, 0)  # MOUSEEVENTF_LEFTUP


# ── Taskbar detection ────────────────────────────────────────────────────────
def get_taskbar_position():
    try:
        wa = ctypes.wintypes.RECT()
        ctypes.windll.user32.SystemParametersInfoW(48, 0, ctypes.byref(wa), 0)
        sw = ctypes.windll.user32.GetSystemMetrics(0)
        sh = ctypes.windll.user32.GetSystemMetrics(1)
        xo, yo = wa.left, wa.top
        if wa.top > 0:         pos = "top"
        elif wa.left > 0:      pos = "left"
        elif wa.right < sw:    pos = "right"
        elif wa.bottom < sh:   pos = "bottom"
        else:                  pos = "hidden"
        return (xo, yo, pos)
    except Exception as e:
        print(f"[CORE] Taskbar detection error: {e}")
        return (0, 0, "bottom")

TASKBAR_X_OFFSET, TASKBAR_Y_OFFSET, TASKBAR_POSITION = get_taskbar_position()
print(f"[CORE] Taskbar: {TASKBAR_POSITION}  offset: ({TASKBAR_X_OFFSET}, {TASKBAR_Y_OFFSET})")


def apply_scaling_and_offset(bbox):
    s = s_bbox(bbox)
    return (s[0]+TASKBAR_X_OFFSET, s[1]+TASKBAR_Y_OFFSET,
            s[2]+TASKBAR_X_OFFSET, s[3]+TASKBAR_Y_OFFSET)

def apply_scaling_and_offset_xy(x, y):
    s = s_xy(x, y)
    return (s[0]+TASKBAR_X_OFFSET, s[1]+TASKBAR_Y_OFFSET)


# ── Default / original coordinate tables ────────────────────────────────────
# These are the factory defaults (1366×768, bottom taskbar).
# ORIGINAL_* are mutable and updated when coords are edited.
# DEFAULT_* are never mutated and are used to reset.

DEFAULT_QUESTION_AREA      = (411, 182, 722, 231)
DEFAULT_QUESTION_AREA_FAST = (423, 182, 710, 231)
DEFAULT_KEY_COORDS = {
    '0': (486, 613), '1': (464, 537), '2': (533, 534), '3': (606, 530),
    '4': (453, 456), '5': (531, 460), '6': (615, 455),
    '7': (453, 378), '8': (530, 381), '9': (613, 381), 'OK': (689, 576)
}
DEFAULT_AUTO_AREA_1 = (510, 686)
DEFAULT_AUTO_AREA_2 = (157, 745)
DEFAULT_AUTO_AREA_3 = (274, 430)

ORIGINAL_QUESTION_AREA      = DEFAULT_QUESTION_AREA
ORIGINAL_QUESTION_AREA_FAST = DEFAULT_QUESTION_AREA_FAST
ORIGINAL_KEY_COORDS         = dict(DEFAULT_KEY_COORDS)
ORIGINAL_AUTO_AREA_1        = DEFAULT_AUTO_AREA_1
ORIGINAL_AUTO_AREA_2        = DEFAULT_AUTO_AREA_2
ORIGINAL_AUTO_AREA_3        = DEFAULT_AUTO_AREA_3

# Scaled runtime values (recomputed whenever ORIGINAL_* change)
QUESTION_AREA      = apply_scaling_and_offset(ORIGINAL_QUESTION_AREA)
QUESTION_AREA_FAST = apply_scaling_and_offset(ORIGINAL_QUESTION_AREA_FAST)
KEY_COORDS         = {k: apply_scaling_and_offset_xy(*v) for k, v in ORIGINAL_KEY_COORDS.items()}
AUTO_AREA_1        = apply_scaling_and_offset_xy(*ORIGINAL_AUTO_AREA_1)
AUTO_AREA_2        = apply_scaling_and_offset_xy(*ORIGINAL_AUTO_AREA_2)
AUTO_AREA_3        = apply_scaling_and_offset_xy(*ORIGINAL_AUTO_AREA_3)

# ── Polling / timing constants ───────────────────────────────────────────────
FAST_MODE_POLLING     = 10    # ms
STANDARD_MODE_POLLING    = 150   # ms
KEY_PRESS_DELAY       = 0     # s between digit clicks
POST_ANSWER_DELAY     = 0     # s after OK click
TASKBAR_CHECK_INTERVAL = 2000 # ms
PREVIEW_UPDATE_INTERVAL = 5   # loop iterations between preview refreshes

# ── Solver modes ─────────────────────────────────────────────────────────────
MODE_HYBRID   = "hybrid"   # LUT race + solver in parallel (default)
MODE_CALC     = "calc"     # Force-solve every question, ignore LUT
MODE_LUT_ONLY = "lut"      # Only use LUT/cache; skip if unknown

# ── Global hotkey ─────────────────────────────────────────────────────────────
# Press this key combination from any window to toggle pause.
# Default: F8  (change to e.g. pynput_keyboard.Key.f9, or a hotcombo like
#  {pynput_keyboard.Key.ctrl, pynput_keyboard.KeyCode.from_char('p')} )
PAUSE_HOTKEY = pynput_keyboard.Key.f8


# ─────────────────────────────────────────────────────────────────────────────
class BotCore:
    """
    Pure business-logic layer.  No tkinter here.
    The GUI creates a BotCore instance then sets `core.ui = gui_instance`
    so the core can call back into the UI for label updates.
    """

    def __init__(self):
        # UI back-reference — set by the GUI after construction
        self.ui = None

        # Coordinates (local copies that follow the global vars)
        self.question_area      = QUESTION_AREA
        self.question_area_fast = QUESTION_AREA_FAST
        self.key_coords         = dict(KEY_COORDS)
        self.current_taskbar_position = TASKBAR_POSITION

        # OCR reader
        try:
            self.reader = easyocr.Reader(['en'], gpu=True)
            print("[CORE] EasyOCR: GPU")
        except Exception:
            self.reader = easyocr.Reader(['en'], gpu=False)
            print("[CORE] EasyOCR: CPU")

        # State
        self.last_question            = ""
        self._last_question_reset_id  = None
        self.paused                   = False
        self.fast_mode                = True
        self.current_polling          = FAST_MODE_POLLING
        self.solve_mode               = MODE_HYBRID
        self.preview_enabled          = True
        self.preview_loop_counter     = 0

        # Automation
        self.answers_count            = 0
        self.ready_count              = 0
        self.is_answering             = False
        self.extended_sequence_active = False
        self.automation_enabled       = True   # toggled by the Automation button
        self.scheduled_events         = []

        # Session cache — tracks questions answered this round (fast mode only).
        # Cleared on Reset / mode switch / new-session trigger.
        # Completely separate from the LUT.
        self.answer_cache = {}

        # High-speed session memory: expr → answer for the current round.
        # Checked after the LUT so the same question never hits the solver twice.
        self.session_cache = {}

        # Persistent LUT — loaded from disk, never cleared except by user action
        self.lut        = self._load_lut()
        self._lut_dirty = False
        # Serializes disk writes and lets a late-finishing stale write detect
        # that a newer one has since been queued and skip itself, instead of
        # two concurrent writers racing and the older one winning last.
        self._lut_write_lock = threading.Lock()
        self._lut_write_seq  = 0

        # Pre-compiled regex
        self._op_clean  = re.compile(r'[^\d\s\+\-\*/\(\)=?]')
        self._sp_clean  = re.compile(r'\s*([\+\-\*/\(\)=])\s*')
        self._div_pat   = re.compile(r"(\d{3})\s+(\d{1,2})\s*=\s*\?")
        self._mult_pat  = re.compile(r"(\d+)\s+(\d+)")
        self._x         = symbols('x')

        # Cache CLAHE once — recreating it every frame wastes time
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Start global hotkey listener on a daemon thread so it works
        # regardless of which window/application is currently focused.
        self._start_hotkey_listener()

    # ── Global hotkey listener ────────────────────────────────────────────────

    def _start_hotkey_listener(self):
        """
        Spawn a daemon thread running a pynput keyboard listener.
        Pressing PAUSE_HOTKEY (default F8) from any window toggles pause.
        The thread is a daemon so it dies automatically when the main process exits.
        """
        def on_press(key):
            try:
                if key == PAUSE_HOTKEY:
                    self._hotkey_toggle_pause()
            except Exception:
                pass   # never let a listener error crash the bot

        listener = pynput_keyboard.Listener(on_press=on_press)
        listener.daemon = True
        listener.start()
        print(f"[CORE] Hotkey listener active — press {PAUSE_HOTKEY} to toggle pause")

    def _hotkey_toggle_pause(self):
        """
        Called from the pynput listener thread.
        Flips self.paused and — if the GUI is attached — schedules the
        matching UI update on the tkinter main thread via root.after().
        """
        self.paused = not self.paused
        state = "PAUSED" if self.paused else "RUNNING"
        print(f"[HOTKEY] Bot {state}")
        if self.ui:
            # root.after is thread-safe; directly touching widgets is not
            self.ui.root.after(0, self.ui.sync_pause_state)

    # ── Image pre-processing ──────────────────────────────────────────────────

    def preprocess_for_ocr(self, np_img):
        """
        Zero-copy pre-processing pipeline.
        Accepts the raw BGRA numpy array directly from mss (no PIL conversion).

          1. BGRA → greyscale in one pass (avoids the old RGB→BGR→grey double-convert)
          2. Gaussian blur 3×3 — remove compression noise before sharpening
          3. Unsharp mask — sharpen digit edges without ringing artefacts
          4. CLAHE (cached) — adaptive local contrast for uneven lighting
          5. Otsu threshold — auto-selects the optimal cut-point per frame

        No 2× upscale: EasyOCR processing scales quadratically so the upscale
        was the single biggest performance killer.
        """
        gray     = cv2.cvtColor(np_img, cv2.COLOR_BGRA2GRAY)
        blurred  = cv2.GaussianBlur(gray, (3, 3), 0)
        sharp    = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
        enhanced = self._clahe.apply(sharp)
        _, thresh = cv2.threshold(enhanced, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    # ── Normalisation ─────────────────────────────────────────────────────────

    def clean_hallucinations(self, expr):
        """
        Cleans raw OCR text.
        × and ÷ are converted first so they survive to the solver.
        Uses anchored regex to strip '= ?' only at the end (arithmetic → LUT hit)
        and '? =' only at the start, leaving internal '=' intact for algebra.
        """
        expr = expr.lower().strip()
        replacements = {
            'z': '2', 's': '5', 'o': '0', 'i': '1',
            '|': '1', 'l': '1', 'g': '9', 'b': '6',
            ':': '/',
            '×': '*',   # must convert BEFORE garbage strip
            '÷': '/',   # must convert BEFORE garbage strip
        }
        for char, rep in replacements.items():
            expr = expr.replace(char, rep)
        # Strip trailing '= ?' or '= x' → arithmetic, forces LUT cache hit
        expr = re.sub(r'=[ \t]*[?x]\s*$', '', expr)
        # Strip leading '? =' or 'x =' → rare OCR artefact
        expr = re.sub(r'^[?x][ \t]*=', '', expr)
        # Keep only valid math / algebra characters
        return re.sub(r'[^0-9+\-*/().=?x ]', '', expr).strip()

    def normalize_operators(self, expr):
        expr = (expr.replace('×','*').replace('x','*').replace('X','*')
                    .replace('÷','/').replace(':','/'))
        return self._op_clean.sub('', expr)

    def fix_missing_operator(self, expr):
        """
        Heuristic repair for an operator OCR dropped entirely (leaving two
        number groups separated only by whitespace).

        Previously this treated EVERY '+' with no parens as a misread '/' —
        that's wrong far more often than it's right (it corrupted plain
        addition like "7+6" into "7/6"). '+' is a real, unambiguous
        character once OCR reads it; there's no evidence a genuinely-read
        '+' should ever be reinterpreted as something else, so that branch
        is removed entirely.

        The only remaining repair is for a dropped operator between two
        bare number groups (e.g. "123 4 = ?"), which we treat as implicit
        multiplication — the safer default, since division would need to
        guess digit-grouping that OCR gives no evidence for.
        """
        return self._mult_pat.sub(r"\1 * \2", expr, count=1)

    def finalize_expression(self, expr):
        expr = self._sp_clean.sub(r'\1', expr)
        diff = expr.count('(') - expr.count(')')
        if diff > 0:
            expr += ')' * diff
        return expr

    def normalise(self, raw):
        """
        Full pipeline: raw OCR text → clean expression key.
        Order matters:
          1. clean_hallucinations — fix letter/digit misreads (e.g. o→0, s→5)
          2. normalize_operators  — unify ×/÷/x/: to */
          3. fix_missing_operator — heuristic fixes for OCR-dropped operators
          4. finalize_expression  — strip spaces around operators, balance parens
        """
        return self.finalize_expression(
            self.fix_missing_operator(
                self.normalize_operators(
                    self.clean_hallucinations(raw)
                )
            )
        )

    # ── Solver ────────────────────────────────────────────────────────────────

    def solve_algebra(self, expr):
        """
        Solve an expression. Returns int/float or None.

        Fast path: pure arithmetic (no '?' or '=') uses Python eval() ~57× faster
        than SymPy. In fast mode with the = ? strip applied, this is always hit.
        SymPy only runs for algebraic equations with '?'.
        """
        try:
            if '?' not in expr and '=' not in expr:
                result = eval(expr, {"__builtins__": {}})
                if isinstance(result, (int, float)) and float(result).is_integer():
                    return int(result)
                return float(result) if isinstance(result, float) else None
            if '?' not in expr:
                return None
            lhs, rhs = expr.replace('?', 'x').split('=')
            sol = solve(Eq(sympify(lhs), sympify(rhs)), self._x)
            if sol:
                result = N(sol[0])
                return int(result) if result.is_integer else float(result)
        except Exception as e:
            print(f"[CORE] Solver error: {e}", file=sys.stderr)
        return None

    # ── Unified solve pipeline ────────────────────────────────────────────────

    def solve_math(self, expr):
        """
        Unified high-speed solving pipeline.

        IMPORTANT: `expr` must already be the fully-normalised expression
        (the output of self.normalise(), fast-mode-stripped by the caller) —
        the SAME string that gets used as the LUT/session-cache key.

        Previously this method re-derived its own key by running only
        clean_hallucinations() on the raw OCR text, while handle_question()
        stored results under a DIFFERENT key built by the full normalise()
        pipeline (which also runs fix_missing_operator/normalize_operators).
        Those two pipelines could disagree — e.g. solve_math would correctly
        eval "7+6" = 13, but it would get written to the LUT under the key
        "7/6" (because fix_missing_operator rewrote the '+' to '/' for the
        *key* but the '+' → 13 result had already been computed). A later,
        genuine "7/6" question would then hit the LUT and get told the
        answer is 13. Using one shared, already-normalised string for both
        the computation AND the cache key removes that whole class of bug.

        Priority order:
          1. LUT        — permanent JSON cache, instant dict lookup
          2. session_cache — in-memory, populated during this round
          3. eval()     — pure arithmetic, ~57× faster than SymPy
          4. SymPy      — algebraic equations with '?' only
        Returns (answer, source) where source is 'lut' | 'cache' | 'solve' | None.
        """
        if not expr:
            return None, None

        # 1. LUT — permanent JSON cache
        if expr in self.lut:
            return self.lut[expr], 'lut'

        # 2. Session cache — current round memory
        if expr in self.session_cache:
            return self.session_cache[expr], 'cache'

        # 3. eval() fast path — pure arithmetic (no '=' or '?')
        if '=' not in expr and '?' not in expr:
            try:
                result = eval(expr, {"__builtins__": {}})
                # The on-screen keypad only has digit keys, so a non-integer
                # result can't be entered — treat it as unsolved rather than
                # silently rounding/truncating to a wrong integer.
                answer = int(result) if float(result).is_integer() else None
                if answer is not None:
                    self.session_cache[expr] = answer
                    return answer, 'solve'
                return None, None
            except Exception as e:
                print(f"[CORE] eval error on '{expr}': {e}", file=sys.stderr)
                return None, None

        # 4. SymPy — algebraic equations
        try:
            algebra_expr = expr.replace('?', 'x').replace('=', '==')
            lhs, rhs = algebra_expr.split('==')
            sol = solve(Eq(sympify(lhs), sympify(rhs)), self._x)
            if sol:
                result = N(sol[0])
                # Same keypad constraint as above: don't truncate a
                # non-integer solution into a confidently wrong integer.
                if result.is_integer:
                    answer = int(result)
                    self.session_cache[expr] = answer
                    return answer, 'solve'
                return None, None
        except Exception as e:
            print(f"[CORE] SymPy error on '{expr}': {e}", file=sys.stderr)

        return None, None

    # ── Question handling (called from GUI main_loop) ──────────────────────────

    def handle_question(self, raw_ocr):
        """
        Decide how to answer a question.
        Returns (answer_int, source_tag) or (None, None).
        source_tag: 'cache' | 'lut' | 'solve'

        Cache  = per-session answered-questions store (fast mode, clears on reset).
        LUT    = permanent file-backed store (survives restarts).
        Solver = SymPy — runs on a miss; result goes to LUT only (not cache).
        Cache is populated only when a question is *answered this session*.

        Fast mode optimisation: fast questions are always simple arithmetic
        (e.g. '8 × 7 = ?').  Any surviving '?', '=' or surrounding whitespace
        is stripped immediately after normalisation so the key is a bare
        arithmetic expression ('8*7').  This guarantees:
          • LUT/cache hit on the first look-up (no '= ?' variant stored)
          • eval() fast path used instead of SymPy on a cache/LUT miss
          • SymPy is never called at all in fast mode
        """
        norm = self.normalise(raw_ocr)

        # ── Fast mode: discard algebra markers, force arithmetic key ──────────
        if self.fast_mode:
            # Strip any remaining '=', '?' and surrounding whitespace.
            # '8*7' stays '8*7'; '8*7=' or '8*7=?' both become '8*7'.
            norm = re.sub(r'[=?]', '', norm).strip()

        print(f"[CORE] Detected: {raw_ocr!r}  norm: {norm!r}")

        mode = self.solve_mode

        # ── MODE: LUT only — never runs the solver ────────────────────────────
        if mode == MODE_LUT_ONLY:
            # Check session cache first (fastest)
            if self.fast_mode and norm in self.answer_cache:
                return int(self.answer_cache[norm]), 'cache'
            # Then persistent LUT
            if norm in self.lut:
                return int(self.lut[norm]), 'lut'
            print(f"[CORE] [LUT-ONLY] '{norm}' unknown — skipping")
            return None, None

        # ── MODE: Pure calculation — solver every time, LUT never consulted ───
        if mode == MODE_CALC:
            answer = self.solve_algebra(norm)
            if answer is not None and float(answer).is_integer():
                # Do NOT write to LUT in calc mode — user chose to bypass it
                return int(answer), 'solve'
            return None, None

        # ── MODE: Hybrid — LUT → session cache → eval → SymPy ────────────────
        # `norm` was already computed above via self.normalise() (+ the same
        # fast-mode strip applied everywhere else) — solve_math uses that
        # exact string as its key, so whatever gets written to the LUT here
        # is guaranteed to be the same key a later identical question will
        # look itself up under. `source` now reflects where the answer
        # actually came from instead of always being hard-coded to 'solve'.
        answer, source = self.solve_math(norm)
        if answer is not None:
            if source == 'solve':
                self._lut_record(norm, int(answer))
            return int(answer), source

        return None, None

    # ── Click answer ──────────────────────────────────────────────────────────

    def click_answer(self, answer, source='solve'):
        """
        Click the answer on the on-screen keypad.
        source: 'cache' | 'lut' | 'solve'
        The session cache is updated here — this is the correct moment because
        it reflects questions actually answered this session, not what the LUT knows.
        """
        self.is_answering = True
        try:
            # Cancel automation sequence if a fresh solve interrupts it
            if self.extended_sequence_active and source == 'solve':
                print("[CORE] New solve interrupted extended sequence — resetting")
                self.cancel_all_scheduled_events()
                self.extended_sequence_active = False
                if self.ui:
                    self.ui.set_auto_status("⚠ Sequence reset", "orange")
                    self.ui.root.after(2000, lambda: self.ui.set_auto_status("", "gray"))

            answer_str = str(int(answer))

            # ── Global safety switch ───────────────────────────────────────────
            # This is the ONLY place that actually issues clicks for an answer,
            # so it's the one place that must respect automation_enabled.
            # Bookkeeping (session cache, counters) below still runs — the bot
            # still "knows" the answer — it just never touches the mouse.
            if not self.automation_enabled:
                print(f"[CORE] [{source.upper()}] Automation OFF — not clicking {answer_str}")
            else:
                print(f"[CORE] [{source.upper()}] Clicking: {answer_str}")

                if not all(d in self.key_coords for d in answer_str):
                    print(f"[CORE] [SKIP] '{answer_str}' has unmapped chars")
                    return

                for d in answer_str:
                    x, y = self.key_coords[d]
                    fast_click(x, y)
                    if KEY_PRESS_DELAY > 0:
                        time.sleep(KEY_PRESS_DELAY)

                ok_x, ok_y = self.key_coords['OK']
                fast_click(ok_x, ok_y)
                if POST_ANSWER_DELAY > 0:
                    time.sleep(POST_ANSWER_DELAY)

            # ── Update session cache ──────────────────────────────────────────
            # Record every answered question in the session cache (fast mode).
            # This is what makes cache hits happen for repeated questions within
            # a round — completely independent of the LUT.
            norm = self.normalise(self.last_question) if self.last_question else None
            if norm and self.fast_mode and norm not in self.answer_cache:
                self.answer_cache[norm] = int(answer)
                if self.ui:
                    self.ui.update_cache_label(len(self.answer_cache))

            # ── Automation counter ────────────────────────────────────────────
            # Only counts when automation is enabled and in fast mode.
            # Cache hits still count — the question was answered regardless.
            if self.automation_enabled and self.fast_mode and not self.extended_sequence_active:
                self.answers_count += 1
                if self.ui:
                    self.ui.update_counter_label(self.answers_count, self.ready_count)

                if self.answers_count >= 10:
                    self.answers_count = 0
                    self.ready_count  += 1
                    print(f"[CORE] 10 answers! Ready: {self.ready_count}")
                    if self.ui:
                        self.ui.set_auto_status("⏳ Auto sequence starting…", "orange")
                        self.ui.update_counter_label(self.answers_count, self.ready_count)

                    delay = 10000 if self.ready_count >= 3 else 2500
                    if self.ready_count >= 3:
                        self.extended_sequence_active = True
                        print("[CORE] Ready=3 — extended sequence")

                    eid = self.ui.root.after(delay, self.auto_click_area_1_initial)
                    self.scheduled_events.append(eid)

                    if self.ready_count >= 3:
                        self.ready_count = 0
                        if self.ui:
                            self.ui.update_counter_label(self.answers_count, self.ready_count)
                        eid = self.ui.root.after(15000, self.clear_cache_for_new_session)
                        self.scheduled_events.append(eid)
                        eid = self.ui.root.after(10000, self.auto_click_area_2)
                        self.scheduled_events.append(eid)

        except Exception as e:
            print(f"[CORE] Click error: {e}")
        finally:
            self.is_answering = False

    # ── Auto-click sequence ───────────────────────────────────────────────────

    def _can_auto(self, area_name):
        if not self.automation_enabled:
            print(f"[CORE] Skip {area_name}: automation disabled")
            return False
        if self.is_answering or self.paused or not self.fast_mode:
            reason = ("answering" if self.is_answering else
                      "paused"    if self.paused         else "not FAST mode")
            print(f"[CORE] Skip {area_name}: {reason}")
            return False
        return True

    def auto_click_area_1_initial(self):
        if self._can_auto("AUTO 1"):
            fast_click(*AUTO_AREA_1)
            print(f"[CORE] AUTO 1 clicked at {AUTO_AREA_1}")
            if self.ui: self.ui.set_auto_status("✓ AUTO 1 clicked", "green")
        else:
            if self.ui: self.ui.set_auto_status("", "gray")

    def auto_click_area_2(self):
        if self._can_auto("AUTO 2"):
            fast_click(*AUTO_AREA_2)
            print(f"[CORE] AUTO 2 clicked at {AUTO_AREA_2}")
            if self.ui: self.ui.set_auto_status("✓ AUTO 2 clicked", "green")
            eid = self.ui.root.after(2000, self.auto_click_area_3)
            self.scheduled_events.append(eid)

    def auto_click_area_3(self):
        if self._can_auto("AUTO 3"):
            fast_click(*AUTO_AREA_3)
            print(f"[CORE] AUTO 3 clicked at {AUTO_AREA_3}")
            if self.ui: self.ui.set_auto_status("✓ AUTO 3 clicked", "green")
            eid = self.ui.root.after(8000, self.auto_click_area_1_final)
            self.scheduled_events.append(eid)

    def auto_click_area_1_final(self):
        if self._can_auto("AUTO 1 final"):
            fast_click(*AUTO_AREA_1)
            print(f"[CORE] AUTO 1 (final) clicked")
            self.extended_sequence_active = False
            if self.ui:
                self.ui.set_auto_status("✓ Sequence complete", "green")
                eid = self.ui.root.after(2000, lambda: self.ui.set_auto_status("", "gray"))
                self.scheduled_events.append(eid)
        else:
            self.extended_sequence_active = False

    def clear_cache_for_new_session(self):
        """Clear per-round caches at the start of a new round.
        The LUT is untouched — it is never part of the session cache."""
        self.answer_cache.clear()
        self.session_cache.clear()
        print("[CORE] Session caches cleared for new round")
        if self.ui:
            self.ui.update_cache_label(0)

    def cancel_all_scheduled_events(self):
        for eid in self.scheduled_events:
            try:
                self.ui.root.after_cancel(eid)
            except Exception:
                pass
        self.scheduled_events.clear()
        print("[CORE] All scheduled events cancelled")

    def _prune_scheduled_events(self):
        if len(self.scheduled_events) > 40:
            self.scheduled_events = self.scheduled_events[-20:]

    # ── LUT management ────────────────────────────────────────────────────────

    def _load_lut(self):
        try:
            if os.path.exists(LUT_FILE):
                with open(LUT_FILE, 'r') as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    print(f"[CORE] LUT loaded: {len(data)} entries")
                    return data
        except Exception as e:
            print(f"[CORE] LUT load error: {e}")
        return {}

    def _save_lut_async(self):
        """
        Write LUT to disk on a daemon thread — never blocks the main loop.

        Every call gets a sequence number. If two writes are ever in flight
        at once, whichever one is *not* the latest checks self._lut_write_seq
        right before writing (under the lock) and bails out instead of
        writing — so an older, smaller snapshot can never overwrite a newer
        one just because its thread happened to get scheduled second.
        """
        self._lut_write_seq += 1
        seq = self._lut_write_seq
        snapshot = dict(self.lut)
        self._lut_dirty = False
        def _write(snap, seq):
            try:
                with self._lut_write_lock:
                    if seq != self._lut_write_seq:
                        print(f"[CORE] LUT write #{seq} superseded — skipping")
                        return
                    with open(LUT_FILE, 'w') as f:
                        json.dump(snap, f, indent=2)
            except Exception as e:
                print(f"[CORE] LUT save error: {e}")
        threading.Thread(target=_write, args=(snapshot, seq), daemon=True).start()

    def _lut_record(self, norm, answer):
        """Store a fresh solve result in the LUT and persist to disk.
        Does NOT write to answer_cache — the cache is populated in click_answer
        when the question is actually answered this session."""
        if norm not in self.lut:
            self.lut[norm]  = answer
            self._lut_dirty = True
            self._save_lut_async()
            print(f"[CORE] LUT ← '{norm}' → {answer}  ({len(self.lut)} total)")
            if self.ui:
                self.ui.update_lut_label(len(self.lut))

    def clear_lut(self):
        """Called by the GUI Clear LUT button. Only wipes the LUT — session cache is unaffected."""
        self.lut.clear()
        # Invalidate any write already queued on a background thread — without
        # this, a snapshot taken just before Clear was pressed can finish
        # writing just after and silently recreate the file we just deleted.
        self._lut_write_seq += 1
        try:
            if os.path.exists(LUT_FILE):
                os.remove(LUT_FILE)
        except Exception as e:
            print(f"[CORE] LUT delete error: {e}")
        print("[CORE] LUT cleared")
        if self.ui:
            self.ui.update_lut_label(0)

    # ── Coord slot management ─────────────────────────────────────────────────

    def read_slots(self):
        """Return list of MAX_SLOTS dicts (or None for empty)."""
        try:
            if os.path.exists(SLOTS_FILE):
                with open(SLOTS_FILE, 'r') as f:
                    data = json.load(f)
                slots = data.get("slots", [])
                while len(slots) < MAX_SLOTS:
                    slots.append(None)
                return slots[:MAX_SLOTS]
        except Exception as e:
            print(f"[CORE] Slots read error: {e}")
        return [None] * MAX_SLOTS

    def write_slots(self, slots):
        try:
            with open(SLOTS_FILE, 'w') as f:
                json.dump({"slots": slots}, f, indent=2)
            print(f"[CORE] Slots saved to {SLOTS_FILE}")
        except Exception as e:
            print(f"[CORE] Slots write error: {e}")

    def build_coord_snapshot(self):
        """Current ORIGINAL_* values as a JSON-serialisable dict."""
        return {
            "question_area":      list(ORIGINAL_QUESTION_AREA),
            "question_area_fast": list(ORIGINAL_QUESTION_AREA_FAST),
            "key_coords":         {k: list(v) for k, v in ORIGINAL_KEY_COORDS.items()},
            "auto_area_1":        list(ORIGINAL_AUTO_AREA_1),
            "auto_area_2":        list(ORIGINAL_AUTO_AREA_2),
            "auto_area_3":        list(ORIGINAL_AUTO_AREA_3),
        }

    def load_coord_slot(self, idx):
        """
        Apply saved coords from slot idx.
        Returns True on success, False if slot empty.
        Caller must rebuild overlays after this.
        """
        global ORIGINAL_QUESTION_AREA, ORIGINAL_QUESTION_AREA_FAST
        global ORIGINAL_KEY_COORDS
        global ORIGINAL_AUTO_AREA_1, ORIGINAL_AUTO_AREA_2, ORIGINAL_AUTO_AREA_3
        global QUESTION_AREA, QUESTION_AREA_FAST, KEY_COORDS
        global AUTO_AREA_1, AUTO_AREA_2, AUTO_AREA_3

        slots = self.read_slots()
        slot  = slots[idx]
        if slot is None:
            print(f"[CORE] Slot {idx+1} empty")
            return False

        c = slot["coords"]
        ORIGINAL_QUESTION_AREA      = tuple(c["question_area"])
        ORIGINAL_QUESTION_AREA_FAST = tuple(c["question_area_fast"])
        ORIGINAL_KEY_COORDS         = {k: tuple(v) for k, v in c["key_coords"].items()}
        ORIGINAL_AUTO_AREA_1        = tuple(c["auto_area_1"])
        ORIGINAL_AUTO_AREA_2        = tuple(c["auto_area_2"])
        ORIGINAL_AUTO_AREA_3        = tuple(c["auto_area_3"])

        self._recompute_scaled_coords()
        print(f"[CORE] Slot {idx+1} loaded")
        return True

    def reset_coords_to_defaults(self):
        """Restore factory defaults."""
        global ORIGINAL_QUESTION_AREA, ORIGINAL_QUESTION_AREA_FAST
        global ORIGINAL_KEY_COORDS
        global ORIGINAL_AUTO_AREA_1, ORIGINAL_AUTO_AREA_2, ORIGINAL_AUTO_AREA_3
        ORIGINAL_QUESTION_AREA      = DEFAULT_QUESTION_AREA
        ORIGINAL_QUESTION_AREA_FAST = DEFAULT_QUESTION_AREA_FAST
        ORIGINAL_KEY_COORDS         = dict(DEFAULT_KEY_COORDS)
        ORIGINAL_AUTO_AREA_1        = DEFAULT_AUTO_AREA_1
        ORIGINAL_AUTO_AREA_2        = DEFAULT_AUTO_AREA_2
        ORIGINAL_AUTO_AREA_3        = DEFAULT_AUTO_AREA_3
        self._recompute_scaled_coords()
        print("[CORE] Coords reset to factory defaults")

    def _recompute_scaled_coords(self):
        """Rebuild all scaled runtime globals from ORIGINAL_* values."""
        global QUESTION_AREA, QUESTION_AREA_FAST, KEY_COORDS
        global AUTO_AREA_1, AUTO_AREA_2, AUTO_AREA_3
        QUESTION_AREA      = apply_scaling_and_offset(ORIGINAL_QUESTION_AREA)
        QUESTION_AREA_FAST = apply_scaling_and_offset(ORIGINAL_QUESTION_AREA_FAST)
        KEY_COORDS         = {k: apply_scaling_and_offset_xy(*v) for k, v in ORIGINAL_KEY_COORDS.items()}
        AUTO_AREA_1        = apply_scaling_and_offset_xy(*ORIGINAL_AUTO_AREA_1)
        AUTO_AREA_2        = apply_scaling_and_offset_xy(*ORIGINAL_AUTO_AREA_2)
        AUTO_AREA_3        = apply_scaling_and_offset_xy(*ORIGINAL_AUTO_AREA_3)
        self.question_area      = QUESTION_AREA
        self.question_area_fast = QUESTION_AREA_FAST
        self.key_coords         = dict(KEY_COORDS)

    def apply_overlay_positions(self, key_overlays, auto_overlays_map,
                                question_overlay, fast_mode):
        """
        Read drag-tracked _edit_x/_edit_y from every overlay window and push the
        values into the live globals and ORIGINAL_* tables.
        Called by the GUI after the user clicks Done Editing.
        """
        global KEY_COORDS, AUTO_AREA_1, AUTO_AREA_2, AUTO_AREA_3
        global QUESTION_AREA, QUESTION_AREA_FAST
        global ORIGINAL_KEY_COORDS, ORIGINAL_AUTO_AREA_1, ORIGINAL_AUTO_AREA_2, ORIGINAL_AUTO_AREA_3
        global ORIGINAL_QUESTION_AREA, ORIGINAL_QUESTION_AREA_FAST

        def rev(fx, fy):
            return (int((fx - TASKBAR_X_OFFSET) / SCALE_X),
                    int((fy - TASKBAR_Y_OFFSET) / SCALE_Y))

        bs  = int(50 * SCALE_X)   # keypad box half-size
        abs_ = int(60 * SCALE_X)  # auto-area box half-size

        for key, win in key_overlays.items():
            cx, cy = win._edit_x + bs//2, win._edit_y + bs//2
            KEY_COORDS[key] = (cx, cy)
            self.key_coords[key] = (cx, cy)
            ORIGINAL_KEY_COORDS[key] = rev(cx, cy)

        for attr, gvar, ogvar in [
            ('a1', 'AUTO_AREA_1', 'ORIGINAL_AUTO_AREA_1'),
            ('a2', 'AUTO_AREA_2', 'ORIGINAL_AUTO_AREA_2'),
            ('a3', 'AUTO_AREA_3', 'ORIGINAL_AUTO_AREA_3'),
        ]:
            win = auto_overlays_map.get(attr)
            if win is None: continue
            cx, cy = win._edit_x + abs_//2, win._edit_y + abs_//2
            globals()[gvar]  = (cx, cy)
            globals()[ogvar] = rev(cx, cy)

        qw = question_overlay
        qx, qy, qw2, qh2 = qw._edit_x, qw._edit_y, qw._edit_w, qw._edit_h
        inset = int(12 * SCALE_X)

        if fast_mode:
            QUESTION_AREA_FAST = (qx, qy, qx+qw2, qy+qh2)
            QUESTION_AREA      = (qx-inset, qy, qx+qw2+inset, qy+qh2)
        else:
            QUESTION_AREA      = (qx, qy, qx+qw2, qy+qh2)
            QUESTION_AREA_FAST = (qx+inset, qy, qx+qw2-inset, qy+qh2)

        self.question_area      = QUESTION_AREA
        self.question_area_fast = QUESTION_AREA_FAST

        ox1, oy1 = rev(QUESTION_AREA[0], QUESTION_AREA[1])
        ox2, oy2 = rev(QUESTION_AREA[2], QUESTION_AREA[3])
        ORIGINAL_QUESTION_AREA      = (ox1, oy1, ox2, oy2)
        ORIGINAL_QUESTION_AREA_FAST = (ox1+12, oy1, ox2-12, oy2)

        print(f"[CORE] Overlay positions applied — Q={QUESTION_AREA}")

    def check_taskbar_position(self):
        """
        Call from GUI's periodic check.
        Returns True if the taskbar moved and coords were recalculated.
        """
        global TASKBAR_X_OFFSET, TASKBAR_Y_OFFSET
        try:
            xo, yo, pos = get_taskbar_position()
            if pos != self.current_taskbar_position:
                print(f"[CORE] Taskbar moved: {self.current_taskbar_position} → {pos}")
                self.current_taskbar_position = pos
                TASKBAR_X_OFFSET = xo
                TASKBAR_Y_OFFSET = yo
                self._recompute_scaled_coords()
                return True
        except Exception as e:
            print(f"[CORE] Taskbar check error: {e}")
        return False
