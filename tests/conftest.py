"""
tests/conftest.py — Test harness for running the REAL bot_core/gui logic on
any OS (developed and run on Linux; the production target is Windows).

Injects stubs for the platform layer (easyocr, mss, pynput, pyautogui)
BEFORE importing the project modules. The actual solver, normalisation,
candidate selection, LUT, cache and retry logic all run as real code —
only screen/keyboard/click primitives are faked.

tkinter is importable headless (no window is ever created); PIL.ImageTk
imports fine but PhotoImage needs a display, so gui tests patch
gui.ImageTk.PhotoImage with an identity function.
"""

import os
import sys
import types

import numpy as np
import pytest

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)


# ── platform stubs (must exist before bot_core import) ──────────────────────

def _install_stubs():
    # easyocr — Reader is constructed in BotCore.__init__
    if "easyocr" not in sys.modules:
        easyocr = types.ModuleType("easyocr")

        class _StubReader:
            def __init__(self, *args, **kwargs):
                self.last_call = None

            def readtext(self, img, **kwargs):
                self.last_call = kwargs
                return []

        easyocr.Reader = _StubReader
        sys.modules["easyocr"] = easyocr

    # pyautogui — screen size queried at bot_core import time
    if "pyautogui" not in sys.modules:
        pyautogui = types.ModuleType("pyautogui")
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0
        pyautogui.size = lambda: (1366, 768)
        pyautogui.moveTo = lambda *a, **k: None
        pyautogui.position = lambda: (0, 0)
        pyautogui.click = lambda *a, **k: None
        sys.modules["pyautogui"] = pyautogui

    # mss — capture object; GUI tests never call grab() directly
    if "mss" not in sys.modules:
        mss = types.ModuleType("mss")

        class _StubSct:
            def grab(self, monitor):
                raise RuntimeError("mss stub: no real screen in tests")

        mss.mss = _StubSct
        sys.modules["mss"] = mss

    # pynput — hotkey listener thread
    if "pynput" not in sys.modules:
        pynput = types.ModuleType("pynput")
        kb = types.ModuleType("pynput.keyboard")

        class _StubListener:
            def __init__(self, *a, **k):
                self.daemon = None

            def start(self):
                pass

            def stop(self):
                pass

        class _StubKey:
            f8 = "f8"
            f9 = "f9"

        kb.Listener = _StubListener
        kb.Key = _StubKey
        kb.KeyCode = lambda **k: k.get("char", "key")
        pynput.keyboard = kb
        sys.modules["pynput"] = pynput
        sys.modules["pynput.keyboard"] = kb


_install_stubs()


# ── shared fixtures ──────────────────────────────────────────────────────────

class FakeGrab:
    """mss-shaped frame: BGRA buffer + size, numpy-viewable."""

    def __init__(self, w=300, h=50, seed=0, invert=False):
        self._seed = seed
        rng = np.random.default_rng(seed)
        arr = (rng.random((h, w, 4)) * 255).astype(np.uint8)
        if invert:
            arr = 255 - arr
        self._arr = arr

    @property
    def bgra(self):
        return self._arr.tobytes()

    @property
    def size(self):
        return (self._arr.shape[1], self._arr.shape[0])

    def __array__(self, dtype=None, copy=None):
        out = self._arr
        if dtype is not None:
            out = out.astype(dtype)
        if copy:
            out = out.copy()
        return out

    def mutated(self):
        """A copy representing NEW SCREEN CONTENT: digest differs (layer A)
        AND the question-region rendering differs far beyond animation-level
        pixel noise, so the forensic-audit probe gate (gui._process_frame)
        treats it as a genuine content change — which is what every test
        using mutated() means by it. (The old single-pixel flip simulated
        exactly the animation the new gate must IGNORE.)"""
        clone = FakeGrab.__new__(FakeGrab)
        clone._seed = self._seed ^ 0x9E3779B9
        rng = np.random.default_rng(clone._seed)
        clone._arr = (rng.random(self._arr.shape) * 255).astype(np.uint8)
        return clone


class DummyCanvas:
    def __init__(self):
        self.images = []
        self._next_id = 1

    def create_image(self, *a, **k):
        self.images.append(("create", a, k))   # first render
        self._next_id += 1
        return self._next_id - 1

    def itemconfig(self, item, image=None):
        self.images.append(image)

    def delete(self, *a):
        pass

    def create_text(self, *a, **k):
        return 0


class DummyVar:
    def __init__(self, value=False):
        self._v = value

    def get(self):
        return self._v


class DummyRoot:
    """Tk root stand-in: records after() callbacks instead of scheduling."""

    def __init__(self):
        self.callbacks = []

    def after(self, ms, fn=None):
        self.callbacks.append((ms, fn))
        return len(self.callbacks)

    def after_cancel(self, eid):
        pass

    def after_idle(self, fn=None):
        if fn:
            fn()


@pytest.fixture
def core(tmp_path, monkeypatch):
    """A real BotCore (stubbed platform layer), tuned for fast tests.

    The LUT is redirected to a scratch file and emptied so tests never
    depend on (or pollute) the repository's shipped optical_lut.json."""
    import bot_core
    from question_state import QuestionStateMachine, RetryPolicy, TTLFrameCache

    monkeypatch.setattr(bot_core, "LUT_FILE", str(tmp_path / "test_lut.json"))
    c = bot_core.BotCore()
    c.lut = {}
    c._lut_dirty = False
    # Disable async LUT persistence: _save_lut_async spawns a daemon thread
    # that can fire AFTER this fixture restored LUT_FILE to the real path
    # and overwrite the repository's optical_lut.json with a test snapshot.
    # Persistence is not under test; keep every write in-process and inert.
    c._save_lut_async = lambda: None
    policy = RetryPolicy(
        same_frame_retry_delay=0.03,
        same_question_retry_delay=0.06,
        retry_backoff=2.0,
        max_backoff=0.12,
        max_retries_per_question=8,
        success_cooldown=0.05,
        click_confirm_timeout=0.04,
        unconfirmed_threshold=3,
        question_reappear_reset=0.2,
    )
    c.retry_policy = policy
    c.qsm = QuestionStateMachine(policy)
    c.frame_cache = TTLFrameCache(policy.frame_cache_ttl)
    c.target_hwnd = None          # skip the wrong-window GetForegroundWindow path
    yield c


@pytest.fixture
def click_log(monkeypatch):
    """Record every click fast_click() would make; returns the log list."""
    import bot_core
    log = []
    monkeypatch.setattr(bot_core, "fast_click", lambda x, y: log.append((x, y)))
    return log


@pytest.fixture
def gui_double(core, monkeypatch):
    """
    OpticalReaderSolverGUI without __init__/Tk: real loop methods driven
    against a fake capture + dummy widgets.
    """
    import gui as gui_mod

    # PhotoImage needs a display — identity-patch it for preview tests.
    monkeypatch.setattr(gui_mod.ImageTk, "PhotoImage", lambda img: img)

    g = gui_mod.OpticalReaderSolverGUI.__new__(gui_mod.OpticalReaderSolverGUI)
    g.core = core
    core.ui = None
    g.last_frame_hash = None
    g._preview_counter = 0
    g._preview_force = False
    g.CONFIRM_TIMEOUT = core.retry_policy.click_confirm_timeout
    g.UNCONFIRMED_THRESHOLD = core.retry_policy.unconfirmed_threshold
    g.preview_canvas = DummyCanvas()
    g.preview_img_tk = None
    g.save_ocr_captures_var = DummyVar(False)
    g.root = DummyRoot()
    g._sct = None                # _main_loop tests install their own frames

    g._display_calls = []

    def _display(expr, answer, source=None, click_result=None):
        g._display_calls.append((expr, answer, source, click_result))

    g._update_detected_display = _display

    # silence widget-config paths we never exercise
    g.preview_toggle_btn = types.SimpleNamespace(
        config=lambda **k: None, bind=lambda *a, **k: None)
    g.answer_clicks_btn = types.SimpleNamespace(config=lambda **k: None)
    g.auto_seq_btn = types.SimpleNamespace(config=lambda **k: None)
    g.set_auto_status = lambda text, color="gray": None
    g.update_cache_label = lambda count: None
    g._set_status = lambda text, color: None
    g.pause_btn = types.SimpleNamespace(config=lambda **k: None)
    return g


def make_ocr_result(tokens_confs, x0=4, y0=6, height=30, char_w=16):
    """
    Build EasyOCR-shaped results [(bbox, text, conf), ...] laid out left to
    right: bbox = 4-corner list, matching production shape.
    """
    results = []
    x = x0
    for text, conf in tokens_confs:
        w = max(len(text), 1) * char_w
        bbox = [(x, y0), (x + w, y0), (x + w, y0 + height), (x, y0 + height)]
        results.append((bbox, text, conf))
        x += w + 6
    return results
