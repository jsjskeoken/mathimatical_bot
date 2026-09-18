"""
gui.py — Tkinter UI, overlay management, main loop scheduler.
Imports BotCore from bot_core.py and calls back into it for all logic.
"""

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import mss
import numpy as np
import ctypes
import datetime
import os
import re
import time

from bot_core import (
    BotCore,
    QUESTION_AREA, QUESTION_AREA_FAST, KEY_COORDS,
    AUTO_AREA_1, AUTO_AREA_2, AUTO_AREA_3,
    TASKBAR_CHECK_INTERVAL, PREVIEW_UPDATE_INTERVAL,
    MAX_SLOTS, SCALE_X, SCALE_Y,
    MODE_HYBRID, MODE_CALC, MODE_LUT_ONLY,
    FAST_MODE_POLLING, STANDARD_MODE_POLLING,
    CLICK_RESULT_CLICKED, CLICK_RESULT_AUTOMATION_OFF,
    CLICK_RESULT_WRONG_WINDOW, CLICK_RESULT_UNMAPPED_ANSWER,
    CLICK_RESULT_ERROR,
)
from question_state import (
    frame_digest, semantic_fingerprint, debug_throttled, signature_distance,
    OUTCOME_OCR_EMPTY, OUTCOME_UNSOLVED, OUTCOME_NOT_ENABLED,
    OUTCOME_WRONG_WINDOW, OUTCOME_UNMAPPED, OUTCOME_CLICKED, OUTCOME_ERROR,
    OUTCOME_CONFIRMED, OUTCOME_UNCONFIRMED,
)
from bot_core import visual_signature  # unresolved-visual-identity pixels
import bot_core  # for the mutable globals QUESTION_AREA etc.

# Windows constants for click-through overlays
WS_EX_TRANSPARENT = 0x00000020
WS_EX_LAYERED     = 0x00080000
GWL_EXSTYLE       = -20
# Off by default (see the "Save OCR captures" advanced toggle) — only
# created on disk the first time a capture is actually saved.
OCR_CAPTURE_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "ocr_captures")

# ─────────────────────────────────────────────────────────────────────────────
# Design tokens — the single source of truth for every colour/font/spacing
# value used below. Nothing past this block should hardcode a one-off hex
# code or point size; add a token here instead so the whole app stays
# visually consistent and themeable from one place.
# ─────────────────────────────────────────────────────────────────────────────

# Colour — restrained dark graphite + muted indigo. Purple/indigo is
# reserved for active mode / primary action / selected state — it should
# never be the dominant colour of every card, or the whole app reads as
# "saturated purple" instead of "one accent among calm neutrals".
C_BG          = "#101116"   # app background
C_SURFACE     = "#181a21"   # primary card
C_SURFACE_ALT = "#1e2029"   # secondary panel (advanced area, modals)
C_BORDER      = "#2a2d38"   # card/section border
C_DIVIDER     = "#232631"   # hairline between rows within a card

C_ACCENT       = "#8b82d9"  # primary accent — mode selection, primary button
C_ACCENT_HOVER = "#9e96e8"

C_GREEN   = "#4ade80"   # success / running / enabled
C_GREEN_BG = "#193021"
C_RED     = "#f87171"   # danger / paused / disabled
C_RED_BG  = "#321e21"
C_ORANGE  = "#fbbf24"   # warning — used sparingly, never as a default state
C_ORANGE_BG = "#332a13"
C_CYAN    = "#67e8f9"   # informational accent — "live"/reading indicator only

C_FG        = "#eceef5"  # primary text
C_MUTED     = "#969aaa"  # secondary text / labels
C_MUTED_DIM = "#606474"  # tertiary / helper text

# Typography — one Windows-safe family; hierarchy comes from size/weight,
# not from mixing typefaces. A monospace face is used only where alignment
# of digits actually matters (the detected expression / result).
FONT_FAMILY = "Segoe UI"
FONT_MONO   = "Consolas"

F_STATUS   = (FONT_FAMILY, 13, "bold")   # the one big "what's happening" line
F_SECTION  = (FONT_FAMILY, 8,  "bold")   # section headings (Solver Mode, Advanced…)
F_BODY     = (FONT_FAMILY, 9)            # buttons, normal controls
F_BODY_B   = (FONT_FAMILY, 9,  "bold")
F_LABEL    = (FONT_FAMILY, 8)            # secondary labels
F_HELPER   = (FONT_FAMILY, 7)            # meta/helper text, mode subtitles
F_RESULT   = (FONT_MONO,   16, "bold")   # the solved answer — biggest number on screen
F_DETECTED = (FONT_MONO,   10)           # the raw detected expression

# Spacing scale (px) — every pack/grid padding below is one of these.
SP_1, SP_2, SP_3, SP_4, SP_5 = 4, 8, 12, 16, 20

# Button colour system — a handful of named *kinds* rather than one-off
# colours per button, so "this is primary" / "this is dangerous" reads
# consistently everywhere.
BTN_KINDS = {
    "primary":   {"bg": C_ACCENT,     "fg": "#ffffff", "hover": C_ACCENT_HOVER},
    "secondary": {"bg": C_SURFACE_ALT,"fg": C_FG,       "hover": C_BORDER},
    "ghost":     {"bg": C_SURFACE,    "fg": C_MUTED,    "hover": C_SURFACE_ALT},
    "success":   {"bg": C_GREEN_BG,   "fg": C_GREEN,    "hover": "#24432f"},
    "danger":    {"bg": C_RED_BG,     "fg": C_RED,      "hover": "#472a2a"},
    "muted_off": {"bg": C_SURFACE_ALT,"fg": C_MUTED,    "hover": C_BORDER},
}


# Forensic audit BUG-2 — how a CLOSED retry gate reacts to changed pixels.
#
# A visual change is a reason to LOOK sooner, not a licence to bypass the
# retry engine: the old bypass (`if not visual_changed and not
# decision.allowed: return`) ran a full EasyOCR pass on EVERY poll whenever
# anything on screen animated (cursor blink, spinner, progress bar), so the
# entire retry/backoff design never applied to animated screens.
#
# The gate instead distinguishes WHY the frame changed, using the same
# aligned, dead-zoned pixel-signature machinery as unresolved-question
# identity (measured classes: same-question noise <= 0.0043, content change
# >= 0.0171 at the reference grid):
#
#   - animation-level pixel delta  -> NOT a reason to look: the retry
#     schedule owns the look (a changed frame also pulls next_allowed in
#     via observe_visual, so a blocked gate still opens on time).
#   - genuine content change       -> ONE immediate identity probe; a probe
#     that discovers a genuinely NEW question proceeds (new-question
#     latency stays at one poll), while a probe that re-reads the SAME
#     question or an unreadable frame stands down without spending retry
#     budget.
#   - adversarial bound: a hostile screen can defeat the signature gate by
#     re-rendering with large per-frame deltas (video-like backgrounds).
#     VISUAL_LOOK_MAX_PER_WINDOW caps same-identity probes between gate
#     openings / state changes, so even that degenerates to a few OCR
#     passes per backoff window instead of one per poll.
VISUAL_LOOK_MAX_PER_WINDOW = 3


class OpticalReaderSolverGUI:
    """Full GUI shell. Creates a BotCore, wires up callbacks, owns the main loop."""

    # Pixel-signature anchor of the last look + blocked-gate probe bookkeeping
    # (forensic audit BUG-2). Class-level defaults so the object is consistent
    # before the first pass (and in headless test doubles).
    _last_look_sig = None       # signature of the pixels last OCR'd/probed
    _blocked_key = None         # (reason-prefix, fingerprint) of the closed gate
    _blocked_looks = 0          # identity probes spent under this closed gate
    _last_look = (None, False)  # (time of last look, was it UNREADABLE?)

    def __init__(self):
        self.core = BotCore()
        self.core.ui = self            # back-reference so core can call UI methods

        # Overlay state
        self.overlay_windows     = []
        self.auto_overlays       = []
        self.key_overlays        = {}
        self.auto_area_1_overlay = None
        self.auto_area_2_overlay = None
        self.auto_area_3_overlay = None
        self.question_overlay    = None
        self.overlays_visible    = True

        # Edit-mode state
        self.edit_mode          = False
        self._edit_save_pending = False
        self._resize_handles    = []
        self.active_slot        = None

        # Advanced section — collapsed by default so the window opens
        # showing only what you need for a normal run (status, preview,
        # pause/automation, mode). Everything else is one click away.
        self._advanced_visible = False

        # High-speed screen capture — mss is 3-5× faster than PIL screen capture
        self._sct             = mss.mss()
        # Exact-pixel digest of the last frame (BLAKE2b). INFORMATIONAL + click
        # confirmation ONLY — it never gates OCR or retries. An unchanged frame
        # with an unsettled question is retried on the retry engine's schedule
        # (question_state.RetryPolicy), not suppressed by this value.
        self.last_frame_hash  = None

        # Preview scheduling — preview is decoupled from the solver: the loop
        # captures and (when preview is enabled) updates the preview EVERY
        # cycle, paused or not. _preview_force makes the very first frame after
        # Resume update the preview immediately instead of waiting out the
        # every-Nth-iteration cadence.
        self._preview_counter = 0
        self._preview_force   = False

        # ── Click confirmation — now owned by the state machine (qsm), which
        # also understands what a confirmed/unconfirmed click MEANS for retry
        # scheduling (question_state.py). The GUI keeps the two constants as
        # attributes so existing tuning/readouts keep working; their values
        # come from the shared RetryPolicy, not a magic number here.
        self.CONFIRM_TIMEOUT       = self.core.retry_policy.click_confirm_timeout
        self.UNCONFIRMED_THRESHOLD = self.core.retry_policy.unconfirmed_threshold

        self._build_root()
        self._build_gui()
        self._build_overlays()
        self._schedule_taskbar_check()

    # ─────────────────────────────────────────────────────────────────────────
    # Root window
    # ─────────────────────────────────────────────────────────────────────────

    def _build_root(self):
        self.root = tk.Tk()
        self.root.title("Optical Reader & Solver")
        self.root.geometry("372x50+50+50")   # height is set for real once content is built
        self.root.resizable(False, False)
        self.root.attributes("-topmost", True)
        self.root.configure(bg=C_BG)

        # Custom ttk style for the solver-mode segmented control
        style = ttk.Style(self.root)
        style.theme_use("clam")
        for name, bg, fg in [
            ("ModeActive.TButton",   C_ACCENT,     "#ffffff"),
            ("ModeInactive.TButton", C_SURFACE_ALT, C_MUTED),
        ]:
            style.configure(name, background=bg, foreground=fg,
                            font=F_BODY, padding=(SP_2, SP_2),
                            relief="flat", borderwidth=0)
            style.map(name, background=[("active", bg)])

        # Register our own window handle so the core's foreground-window
        # tracker can tell "the GUI has focus" apart from "the target app
        # has focus" — without this, clicking Resume would always look
        # like the GUI itself just became the target, since Windows
        # focuses a window as part of delivering a click to it.
        raw_id = self.root.winfo_id()
        parent = ctypes.windll.user32.GetParent(raw_id)
        self.core.set_gui_window(parent if parent else raw_id)

    def _resize_to_fit(self):
        """
        Re-fit the fixed-size window to whatever's currently packed —
        called after toggling the Advanced section (or anything else that
        changes how much vertical content exists), so the window grows or
        shrinks instead of clipping content or leaving dead space.
        """
        self.root.update_idletasks()
        w = self.root.winfo_reqwidth()
        h = self.root.winfo_reqheight()
        self.root.geometry(f"{w}x{h}")

    # ─────────────────────────────────────────────────────────────────────────
    # Small reusable widget builders — keeps every card/button/section
    # visually consistent without repeating style kwargs everywhere.
    # ─────────────────────────────────────────────────────────────────────────

    def _card(self, parent, bg=C_SURFACE, inner_padx=SP_3, inner_pady=SP_2,
              outer_padx=SP_3, outer_pady=(SP_2, 0)):
        card = tk.Frame(parent, bg=bg, padx=inner_padx, pady=inner_pady)
        card.pack(fill="x", padx=outer_padx, pady=outer_pady)
        return card

    def _section_label(self, parent, text):
        return tk.Label(parent, text=text.upper(), fg=C_MUTED, bg=parent["bg"],
                         font=F_SECTION)

    def _pill(self, parent, text, fg, bg, font=None):
        """
        A small badge: a Label whose background genuinely contrasts with
        its parent, read as a compact "pill" even without a true rounded
        corner (which Tkinter can't do without the Canvas-based approach
        this file deliberately avoids — see the note above _build_gui).
        """
        return tk.Label(parent, text=text, fg=fg, bg=bg,
                        font=font or F_LABEL, padx=SP_2, pady=1)

    def _style_button(self, btn, kind):
        """(Re)apply a named colour kind to a button, including live hover
        feedback. Called at creation and again whenever a stateful button
        (Automation, Edit) changes what state it represents."""
        colors = BTN_KINDS[kind]
        btn.config(fg=colors["fg"], bg=colors["bg"],
                   activeforeground=colors["fg"], activebackground=colors["hover"])
        rest, hover = colors["bg"], colors["hover"]
        btn.bind("<Enter>", lambda e, b=btn, c=hover: b.config(bg=c))
        btn.bind("<Leave>", lambda e, b=btn, c=rest: b.config(bg=c))

    def _button(self, parent, text, command, kind="secondary", **kw):
        btn = tk.Button(parent, text=text, command=command,
                        relief="flat", bd=0, cursor="hand2",
                        font=kw.pop("font", F_BODY),
                        padx=SP_3, pady=SP_2, justify="center", **kw)
        self._style_button(btn, kind)
        return btn

    # ─────────────────────────────────────────────────────────────────────────
    # GUI layout
    # ─────────────────────────────────────────────────────────────────────────

    def _build_gui(self):
        root = self.root

        # ── Status card — the single most important thing on screen: is it
        # running, and what has it just seen? Status is a real pill badge
        # (matching bg+fg, see _set_status) rather than large coloured
        # text, and the mode indicator is a small muted badge rather than
        # a second competing headline. ───────────────────────────────────
        status_card = self._card(root, outer_pady=(SP_3, 0), inner_pady=SP_3)

        top_row = tk.Frame(status_card, bg=C_SURFACE)
        top_row.pack(fill="x")
        self.status_pill = tk.Frame(top_row, bg=C_GREEN_BG)
        self.status_pill.pack(side="left")
        self.status_label = tk.Label(self.status_pill, text="●  Running",
                                     fg=C_GREEN, bg=C_GREEN_BG,
                                     font=(FONT_FAMILY, 10, "bold"),
                                     padx=SP_2, pady=1)
        self.status_label.pack()
        self.mode_label = self._pill(top_row, "FAST", C_MUTED, C_SURFACE_ALT)
        self.mode_label.pack(side="right")

        tk.Frame(status_card, bg=C_DIVIDER, height=1).pack(fill="x", pady=(SP_3, SP_2))

        meta1 = tk.Frame(status_card, bg=C_SURFACE)
        meta1.pack(fill="x")
        self.counter_label = tk.Label(meta1, text="Answers 0/10  ·  Ready 0",
                                      fg=C_MUTED, bg=C_SURFACE, font=F_LABEL)
        self.counter_label.pack(side="left")

        meta2 = tk.Frame(status_card, bg=C_SURFACE)
        meta2.pack(fill="x", pady=(2, 0))
        self.cache_label = tk.Label(meta2, text="Cache 0",
                                    fg=C_MUTED_DIM, bg=C_SURFACE, font=F_HELPER)
        self.cache_label.pack(side="left")
        tk.Label(meta2, text="   ", bg=C_SURFACE).pack(side="left")
        self.lut_label = tk.Label(meta2, text=f"{len(self.core.lut)} saved answers",
                                  fg=C_MUTED_DIM, bg=C_SURFACE, font=F_HELPER)
        self.lut_label.pack(side="left")

        self.auto_status_label = tk.Label(status_card, text="", fg=C_MUTED,
                                          bg=C_SURFACE, font=F_HELPER, wraplength=320,
                                          justify="left", anchor="w")
        self.auto_status_label.pack(fill="x", pady=(SP_1, 0))

        # ── Detected / Result — the visual centrepiece: what did it see,
        # what did it decide the answer is. The system line underneath
        # shows only real, measured data (OCR confidence, cache/LUT
        # source, click outcome) — never a fabricated number — and is
        # deliberately small/muted so it reads as supporting detail, not
        # a second headline competing with the result. ────────────────────
        det_card = self._card(root, inner_pady=SP_3)
        det_cols = tk.Frame(det_card, bg=C_SURFACE)
        det_cols.pack(fill="x")

        det_left = tk.Frame(det_cols, bg=C_SURFACE)
        det_left.pack(side="left", fill="x", expand=True)
        self._section_label(det_left, "Detected").pack(anchor="w")
        self.detected_label = tk.Label(det_left, text="—", fg=C_FG, bg=C_SURFACE,
                                       font=F_DETECTED, anchor="w")
        self.detected_label.pack(anchor="w", fill="x", pady=(2, 0))

        det_right = tk.Frame(det_cols, bg=C_SURFACE)
        det_right.pack(side="right")
        self._section_label(det_right, "Result").pack(anchor="e")
        self.result_label = tk.Label(det_right, text="—", fg=C_MUTED, bg=C_SURFACE,
                                     font=F_RESULT, anchor="e")
        self.result_label.pack(anchor="e")

        tk.Frame(det_card, bg=C_DIVIDER, height=1).pack(fill="x", pady=(SP_2, SP_1))
        self.system_line_label = tk.Label(det_card, text="Waiting for a question…",
                                          fg=C_MUTED_DIM, bg=C_SURFACE, font=F_HELPER,
                                          anchor="w")
        self.system_line_label.pack(anchor="w", fill="x")

        # ── OCR preview — framed like the "live vision" of the app: a
        # single clickable status pill ("●  Live" / "Off") rather than a
        # separate dot + text button competing for the same space. ────────
        prev_card = self._card(root)
        hdr = tk.Frame(prev_card, bg=C_SURFACE)
        hdr.pack(fill="x")
        self._section_label(hdr, "OCR Preview").pack(side="left")
        self.preview_toggle_btn = tk.Label(
            hdr, text="●  Live", fg=C_CYAN, bg=C_SURFACE, font=F_LABEL, cursor="hand2")
        self.preview_toggle_btn.pack(side="right")
        self.preview_toggle_btn.bind("<Button-1>", lambda e: self._toggle_preview())

        preview_frame = tk.Frame(prev_card, bg=C_BORDER, padx=1, pady=1)
        preview_frame.pack(pady=(SP_2, 0))
        self.preview_canvas = tk.Canvas(preview_frame, width=286, height=60,
                                        bg="#0c0c14", highlightthickness=0)
        self.preview_canvas.pack()
        self.preview_img_tk = None

        # ── Primary controls — Pause/Resume and Automation are the two
        # actions that matter most. Each is one row: a plain label on the
        # left, a small state pill on the right — rather than the state
        # being the button's entire two-line caption, which made them the
        # loudest thing in the whole window. ───────────────────────────────
        primary_card = self._card(root)

        pause_row = tk.Frame(primary_card, bg=C_SURFACE_ALT, cursor="hand2")
        pause_row.pack(fill="x")
        tk.Label(pause_row, text="❚❚  Pause", fg=C_FG, bg=C_SURFACE_ALT,
                 font=F_BODY_B, padx=SP_3, pady=SP_2).pack(side="left")
        self.pause_btn = tk.Label(pause_row, text="RUNNING", fg=C_GREEN,
                                  bg=C_SURFACE_ALT, font=F_LABEL, padx=SP_3)
        self.pause_btn.pack(side="right")
        for w in (pause_row, *pause_row.winfo_children()):
            w.bind("<Button-1>", lambda e: self._toggle_pause())

        auto_row = tk.Frame(primary_card, bg=C_SURFACE_ALT, cursor="hand2")
        auto_row.pack(fill="x", pady=(SP_2, 0))
        tk.Label(auto_row, text="Answer clicks", fg=C_FG, bg=C_SURFACE_ALT,
                 font=F_BODY_B, padx=SP_3, pady=SP_2).pack(side="left")
        self.answer_clicks_btn = tk.Label(auto_row, text="ON", fg=C_GREEN,
                                          bg=C_SURFACE_ALT, font=F_LABEL, padx=SP_3)
        self.answer_clicks_btn.pack(side="right")
        for w in (auto_row, *auto_row.winfo_children()):
            w.bind("<Button-1>", lambda e: self._toggle_answer_clicks())

        seq_row = tk.Frame(primary_card, bg=C_SURFACE_ALT, cursor="hand2")
        seq_row.pack(fill="x", pady=(SP_2, 0))
        tk.Label(seq_row, text="AUTO sequence", fg=C_FG, bg=C_SURFACE_ALT,
                 font=F_BODY_B, padx=SP_3, pady=SP_2).pack(side="left")
        self.auto_seq_btn = tk.Label(seq_row, text="ON", fg=C_GREEN,
                                     bg=C_SURFACE_ALT, font=F_LABEL, padx=SP_3)
        self.auto_seq_btn.pack(side="right")
        for w in (seq_row, *seq_row.winfo_children()):
            w.bind("<Button-1>", lambda e: self._toggle_auto_sequence())

        # ── Solver mode — segmented control with a short description of
        # whichever mode is currently active, instead of assuming the
        # names ("Hybrid", "LUT") are self-explanatory. ────────────────────
        mode_card = self._card(root)
        self._section_label(mode_card, "Solver Mode").pack(anchor="w")

        pills = tk.Frame(mode_card, bg=C_SURFACE)
        pills.pack(fill="x", pady=(SP_1, 0))

        self._mode_pills = {}
        self._mode_subtitles = {
            MODE_HYBRID:   "Fastest — calculates, remembers answers",
            MODE_CALC:     "Always calculates from scratch",
            MODE_LUT_ONLY: "Only answers questions seen before",
        }
        defs = [(MODE_HYBRID, "Hybrid"), (MODE_CALC, "Calculate"), (MODE_LUT_ONLY, "Saved")]
        for i, (mode, label) in enumerate(defs):
            btn = ttk.Button(pills, text=label, style="ModeInactive.TButton",
                             command=lambda m=mode: self._set_solver_mode(m))
            btn.grid(row=0, column=i, padx=(0 if i == 0 else SP_1, 0), sticky="ew")
            self._mode_pills[mode] = (btn, "ModeActive.TButton")
            pills.columnconfigure(i, weight=1)

        self.mode_subtitle_label = tk.Label(mode_card, text="", fg=C_MUTED_DIM,
                                            bg=C_SURFACE, font=F_HELPER, anchor="w")
        self.mode_subtitle_label.pack(anchor="w", pady=(SP_1, 0))

        self.lut_warn_label = tk.Label(mode_card,
                                       text="Skips any question it hasn't seen before",
                                       fg=C_ORANGE, bg=C_SURFACE, font=F_HELPER)
        # (shown/hidden by _set_solver_mode)

        self._set_solver_mode(MODE_HYBRID)  # set default highlight

        # ── Advanced — everything that isn't a per-round action lives
        # behind one disclosure toggle, collapsed by default. ──────────────
        adv_wrap = self._card(root, bg=C_BG, inner_padx=0, inner_pady=0,
                              outer_pady=(SP_2, SP_3))

        self.advanced_toggle_btn = tk.Button(
            adv_wrap, text="▸  Advanced", command=self._toggle_advanced,
            fg=C_MUTED, bg=C_BG, activeforeground=C_FG, activebackground=C_BG,
            relief="flat", bd=0, cursor="hand2", font=F_LABEL, anchor="w")
        self.advanced_toggle_btn.pack(fill="x")

        self.advanced_frame = tk.Frame(adv_wrap, bg=C_SURFACE_ALT, padx=SP_3, pady=SP_2)
        # not packed yet — _toggle_advanced() packs/unpacks it

        # LAYOUT ─────────────────────────────────────────────────────────
        self._section_label(self.advanced_frame, "Layout").pack(anchor="w")
        row1 = tk.Frame(self.advanced_frame, bg=C_SURFACE_ALT)
        row1.pack(fill="x", pady=(SP_1, 0))
        self.overlays_toggle_btn = self._button(row1, "Overlays: Shown",
                                                self._toggle_overlays, kind="ghost")
        self.overlays_toggle_btn.grid(row=0, column=0, padx=(0, SP_1), sticky="ew")
        self.edit_btn = self._button(row1, "Edit Layout", self._toggle_edit_mode,
                                     kind="ghost")
        self.edit_btn.grid(row=0, column=1, sticky="ew")
        row1.columnconfigure(0, weight=1)
        row1.columnconfigure(1, weight=1)

        row1b = tk.Frame(self.advanced_frame, bg=C_SURFACE_ALT)
        row1b.pack(fill="x", pady=(SP_1, 0))
        saves_btn = self._button(row1b, "Saved Layouts", self._open_saves_modal,
                                 kind="ghost")
        saves_btn.grid(row=0, column=0, padx=(0, SP_1), sticky="ew")
        reset_def_btn = self._button(row1b, "Reset to Default",
                                     self._reset_to_defaults, kind="danger")
        reset_def_btn.grid(row=0, column=1, sticky="ew")
        row1b.columnconfigure(0, weight=1)
        row1b.columnconfigure(1, weight=1)

        # SESSION ────────────────────────────────────────────────────────
        self._section_label(self.advanced_frame, "Session").pack(anchor="w", pady=(SP_3, 0))
        row2 = tk.Frame(self.advanced_frame, bg=C_SURFACE_ALT)
        row2.pack(fill="x", pady=(SP_1, 0))
        session_reset_btn = self._button(row2, "New Round", self._reset_counter,
                                         kind="ghost")
        session_reset_btn.pack(fill="x")

        # MEMORY — Verify LUT re-checks every cached answer against the
        # real solver and fixes/removes anything wrong (see verify_lut()'s
        # docstring for the two kinds of bad entry this catches). ─────────
        self._section_label(self.advanced_frame, "Memory").pack(anchor="w", pady=(SP_3, 0))
        row3 = tk.Frame(self.advanced_frame, bg=C_SURFACE_ALT)
        row3.pack(fill="x", pady=(SP_1, 0))
        verify_lut_btn = self._button(row3, "Verify LUT",
                                      self._verify_lut, kind="secondary")
        verify_lut_btn.grid(row=0, column=0, padx=(0, SP_1), sticky="ew")
        clear_lut_btn = self._button(row3, "Clear Saved Answers",
                                     self.core.clear_lut, kind="danger")
        clear_lut_btn.grid(row=0, column=1, sticky="ew")
        row3.columnconfigure(0, weight=1)
        row3.columnconfigure(1, weight=1)
        self.lut_verify_label = tk.Label(
            self.advanced_frame, text="", fg=C_MUTED, bg=C_SURFACE_ALT,
            font=F_HELPER, justify="left", anchor="w")
        self.lut_verify_label.pack(fill="x", pady=(SP_1, 0))

        # OCR — Known operations FILTERS which operators a candidate
        # expression may use (select_math_ocr_text rejects any candidate
        # containing a disabled operator); it never converts one operator
        # into another. Disabling '+' does NOT make a '+' become '/' —
        # only is_division_glyph's pixel evidence can do that, in
        # correct_ocr_operators. See select_math_ocr_text's docstring. ──
        self._section_label(self.advanced_frame, "OCR").pack(anchor="w", pady=(SP_3, 0))
        operations_label = tk.Label(self.advanced_frame, text="Known operations",
                                    fg=C_MUTED, bg=C_SURFACE_ALT, font=F_LABEL)
        operations_label.pack(anchor="w", pady=(SP_1, SP_1))
        operations_row = tk.Frame(self.advanced_frame, bg=C_SURFACE_ALT)
        operations_row.pack(fill="x")
        operation_defs = [
            ("+", "Addition"), ("-", "Subtraction"),
            ("*", "Multiply"), ("/", "Division"),
        ]
        self.operation_vars = {}
        for column, (operator, label) in enumerate(operation_defs):
            var = tk.BooleanVar(value=True)
            self.operation_vars[operator] = var
            check = tk.Checkbutton(
                operations_row, text=label, variable=var,
                command=lambda op=operator: self._operation_changed(op),
                fg=C_FG, bg=C_SURFACE_ALT,
                activeforeground=C_FG, activebackground=C_SURFACE_ALT,
                selectcolor=C_SURFACE, highlightthickness=0,
                bd=0, padx=SP_1, pady=SP_1, font=F_LABEL,
            )
            check.grid(row=0, column=column, sticky="w")
            operations_row.columnconfigure(column, weight=1)

        # OCR capture debug logging — off by default. Each solved question
        # writes two PNGs (original + processed) to ocr_captures/, useful
        # for diagnosing misreads but grows without bound if left on for a
        # long run, so this is opt-in rather than always-on.
        self.save_ocr_captures_var = tk.BooleanVar(value=False)
        capture_check = tk.Checkbutton(
            self.advanced_frame, text="Save OCR captures (debug)",
            variable=self.save_ocr_captures_var,
            command=self._toggle_ocr_captures,
            fg=C_FG, bg=C_SURFACE_ALT,
            activeforeground=C_FG, activebackground=C_SURFACE_ALT,
            selectcolor=C_SURFACE, highlightthickness=0,
            bd=0, padx=SP_1, pady=SP_1, font=F_LABEL,
        )
        capture_check.pack(anchor="w", pady=(SP_1, 0))

        self.root.after_idle(self._resize_to_fit)

    # ─────────────────────────────────────────────────────────────────────────
    # UI update callbacks (called from BotCore)
    # ─────────────────────────────────────────────────────────────────────────

    def update_cache_label(self, count):
        self.cache_label.config(text=f"Cache {count}")

    def update_lut_label(self, count):
        self.lut_label.config(text=f"{count} saved answers")

    def _set_status(self, text, color):
        """
        Single place that sets the status pill's text AND matching
        background, so "Paused" always renders as a solid red badge and
        never green-background-red-text — every other place that used to
        call self.status_label.config(...) directly now goes through this
        instead, so the pill's background can never drift out of sync
        with its text colour.
        """
        bg_map = {C_GREEN: C_GREEN_BG, C_RED: C_RED_BG, C_ORANGE: C_ORANGE_BG}
        bg = bg_map.get(color, C_SURFACE_ALT)
        self.status_pill.config(bg=bg)
        self.status_label.config(text=text, fg=color, bg=bg)

    def _verify_lut(self):
        """
        Re-checks every cached LUT answer against the real solver and
        fixes/removes anything wrong — see BotCore.verify_lut()'s
        docstring for the two kinds of bad entry this catches. Runs
        synchronously; the LUT is a small dict (tens to low hundreds of
        entries for this app) and solve_algebra() on each is fast, so
        this doesn't need to be backgrounded the way OCR/clicking do.
        """
        report = self.core.verify_lut(auto_fix=True)
        parts = [f"{report['valid']} valid"]
        if report["corrected"]:
            parts.append(f"{len(report['corrected'])} corrected")
        if report["removed"]:
            parts.append(f"{len(report['removed'])} removed")
        summary = f"{report['total']} entries — " + ", ".join(parts)
        self.lut_verify_label.config(text=summary)
        if report["corrected"] or report["removed"]:
            self.lut_verify_label.config(fg=C_RED)
        else:
            self.lut_verify_label.config(fg=C_GREEN)

    def update_counter_label(self, answers, ready):
        self.counter_label.config(text=f"Answers {answers}/10  ·  Ready {ready}")

    def set_auto_status(self, text, color="gray"):
        # Map the legacy colour-name strings used by bot_core.py onto the
        # design-token palette so callers don't need to know hex codes.
        color_map = {"gray": C_MUTED, "grey": C_MUTED, "orange": C_ORANGE,
                     "green": C_GREEN, "red": C_RED}
        self.auto_status_label.config(text=text, fg=color_map.get(color, color))

    def sync_pause_state(self):
        """
        Mirror core.paused into the GUI widgets — the one authoritative
        place that reacts to a pause/resume transition, regardless of which
        of the two triggers caused it (the Pause/Resume button, handled in
        _toggle_pause below, or the F8 hotkey, handled in bot_core.py —
        both funnel through this).
        """
        if self.core.paused:
            self._set_status("●  Paused", C_RED)
            self.pause_btn.config(text="PAUSED", fg=C_RED)
            # Pausing stops the solver — and with it the digest watcher that
            # would have resolved a pending click confirmation. Drop it via
            # the state machine (which also resets the unconfirmed streak): a
            # deliberate pause is a clean boundary, and a deadline elapsing
            # unobserved must never later read as a false "unconfirmed".
            self.core.qsm.on_pause()
        else:
            self._set_status("●  Running", C_GREEN)
            self.pause_btn.config(text="RUNNING", fg=C_GREEN)
            # Resume must restore normal operation IMMEDIATELY: the state
            # machine treats the current question as freshly sighted (no
            # waiting out a backoff that started before the pause), and the
            # next loop cycle updates the preview at once instead of waiting
            # for the every-Nth-iteration cadence.
            self.core.qsm.on_resume()
            self._preview_force = True

    _CLICK_RESULT_TEXT = {
        CLICK_RESULT_CLICKED:         "CLICKED",
        CLICK_RESULT_AUTOMATION_OFF:  "NOT CLICKED · automation off",
        CLICK_RESULT_WRONG_WINDOW:    "NOT CLICKED · wrong window",
        CLICK_RESULT_UNMAPPED_ANSWER: "NOT CLICKED · no key for this answer",
        CLICK_RESULT_ERROR:           "NOT CLICKED · click error",
    }

    def _update_detected_display(self, expr, answer, source=None, click_result=None):
        """
        GUI-only bookkeeping: surfaces the last thing the solver saw/solved.
        Reads state _main_loop already computes — doesn't change what gets
        detected, solved, or clicked.

        The result colour communicates outcome at a glance (muted = nothing
        detected yet, dim amber = detected but didn't solve, accent =
        solved) and the system line underneath spells out source/confidence
        and whether a click actually happened — using only values the
        caller actually passed in. source/click_result/confidence default
        to None (nothing to report) rather than a guessed value, so this
        never displays something the code doesn't actually know.
        """
        self.detected_label.config(text=expr if expr else "—")

        if not expr:
            self.result_label.config(text="—", fg=C_MUTED)
            self.system_line_label.config(text="Waiting for a question…")
            return

        if answer is None:
            self.result_label.config(text="—", fg=C_ORANGE)
            self.system_line_label.config(text="Detected, not solved yet")
            return

        self.result_label.config(text=str(answer), fg=C_ACCENT)

        parts = []
        if source == "lut":
            parts.append("LUT HIT")
        elif source == "cache":
            parts.append("CACHE HIT")
        elif source == "solve" and self.core.last_ocr_confidence is not None:
            parts.append(f"OCR {round(self.core.last_ocr_confidence * 100)}%")
        elif source:
            parts.append(source.upper())
        if click_result is not None:
            parts.append(self._CLICK_RESULT_TEXT.get(click_result, click_result))
        self.system_line_label.config(text=" · ".join(parts) if parts else "Solved")

    def _toggle_ocr_captures(self):
        enabled = self.save_ocr_captures_var.get()
        print(f"[GUI] OCR capture saving {'ENABLED' if enabled else 'disabled'}"
              + (f" → {OCR_CAPTURE_DIR}" if enabled else ""))

    def _save_ocr_capture(self, sct_img, processed, raw_text, answer, source):
        """
        Save the original screen crop and the exact image passed to
        EasyOCR, for diagnosing OCR mistakes later (e.g. "what did the
        classifier actually see when it called this division-like").

        Off by default — gated on the "Save OCR captures" advanced toggle,
        checked here rather than at each call site, so callers don't need
        to know about the setting. Only called for a question that
        actually reached a solved answer (answer is not None); a run that
        never solves anything creates no directory and no files, so
        leaving this on for a long unattended session still can't run
        away with disk space the way logging every raw OCR attempt would.
        """
        if not self.save_ocr_captures_var.get() or answer is None:
            return
        try:
            os.makedirs(OCR_CAPTURE_DIR, exist_ok=True)
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            safe_text = raw_text
            for operator, name in {
                "+": "_add_", "-": "_sub_", "*": "_mul_",
                "/": "_div_", "=": "_eq_", ":": "_div_",
            }.items():
                safe_text = safe_text.replace(operator, name)
            safe_text = re.sub(r"[^A-Za-z0-9_]+", "_", safe_text).strip("_")
            safe_text = (safe_text[:60] or "unknown")
            prefix = f"{stamp}_{safe_text}_answer-{answer}_{source}"

            original = Image.frombytes(
                "RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            original.save(os.path.join(OCR_CAPTURE_DIR, prefix + "_original.png"))
            Image.fromarray(processed).save(
                os.path.join(OCR_CAPTURE_DIR, prefix + "_processed.png"))
            print(f"[GUI] OCR capture saved: {prefix}")
        except Exception as exc:
            print(f"[GUI] OCR capture save failed: {exc}")

    def _operation_changed(self, operator):
        """
        Updates core.enabled_operations from the checkbox state. This is a
        FILTER only — select_math_ocr_text() rejects a candidate expression
        that uses a disabled operator; it never converts one operator into
        another. Disabling '+' does not make a stray '+' become '/' — only
        is_division_glyph's pixel evidence can do that, in
        correct_ocr_operators(), completely independently of this setting.
        """
        enabled = {op for op, var in self.operation_vars.items() if var.get()}
        self.core.enabled_operations = enabled
        self._clear_transient_state("operation filters changed")
        names = {"+": "addition", "-": "subtraction", "*": "multiplication", "/": "division"}
        state = "enabled" if operator in enabled else "disabled"
        print(f"[GUI] Operation {names[operator]} {state}; active={sorted(enabled)}")

    # ─────────────────────────────────────────────────────────────────────────
    # Solver mode
    # ─────────────────────────────────────────────────────────────────────────

    def _clear_transient_state(self, reason=""):
        """
        Full reset of every cache/pending-state that's only valid for the
        CURRENT capture configuration and round — the one authoritative
        invalidation point, called from every place that changes what's
        being captured or solved (mode switch, coordinate profile load,
        manual reset, reset-to-defaults) and from core's own new-round
        trigger.

        This now also resets the layered question state machine (per-question
        solve/click/retry runtime — retry budgets from the old configuration
        must never gate the new one) and the TTL frame cache.
        """
        self.core.answer_cache.clear()
        self.core.session_cache.clear()
        self.core.frame_cache.clear()
        self.core.qsm.reset_round()
        self.last_frame_hash    = None
        self.core.last_question = ""
        self.update_cache_label(0)
        if reason:
            print(f"[GUI] Cleared session/frame cache ({reason})")

    def _set_solver_mode(self, mode):
        self.core.solve_mode = mode
        for m, (btn, active_style) in self._mode_pills.items():
            btn.configure(style=active_style if m == mode else "ModeInactive.TButton")
        self.mode_subtitle_label.config(text=self._mode_subtitles.get(mode, ""))
        if mode == MODE_LUT_ONLY:
            self.lut_warn_label.pack(anchor="w", pady=(2, 0))
        else:
            self.lut_warn_label.pack_forget()
        print(f"[GUI] Solver mode → {mode}")

    # ─────────────────────────────────────────────────────────────────────────
    # Button handlers
    # ─────────────────────────────────────────────────────────────────────────

    def _toggle_pause(self):
        self.core.paused = not self.core.paused
        self.sync_pause_state()
        # If user resumes while save-pending, cancel that state
        if not self.core.paused and self._edit_save_pending:
            self._edit_save_pending = False
        print(f"[GUI] Bot {'PAUSED' if self.core.paused else 'RUNNING'}")

    def _toggle_advanced(self):
        self._advanced_visible = not self._advanced_visible
        if self._advanced_visible:
            self.advanced_toggle_btn.config(text="▾  Advanced", fg=C_FG)
            self.advanced_frame.pack(fill="x", pady=(SP_1, 0))
        else:
            self.advanced_toggle_btn.config(text="▸  Advanced", fg=C_MUTED)
            self.advanced_frame.pack_forget()
        self._resize_to_fit()

    def _toggle_game_mode(self):
        self.core.cancel_all_scheduled_events()
        self.core.fast_mode = not self.core.fast_mode
        self.core.current_polling = (FAST_MODE_POLLING if self.core.fast_mode
                                     else STANDARD_MODE_POLLING)
        label = "FAST" if self.core.fast_mode else "STANDARD"
        self.mode_label.config(text=label)
        # Clear ALL transient state on mode switch (LUT is untouched — it's
        # persistent and mode-independent by design)
        self._clear_transient_state(f"mode → {label}")
        self.set_auto_status("")
        self.core.extended_sequence_active = False
        # Update OCR box size
        if self.overlays_visible and self.question_overlay:
            self.question_overlay.destroy()
            self.overlay_windows.remove(self.question_overlay)
            x1, y1, x2, y2 = (self.core.question_area_fast if self.core.fast_mode
                               else self.core.question_area)
            self.question_overlay = self._create_overlay(
                x1, y1, x2-x1, y2-y1, "red", "", is_auto=False)
        # Show/hide auto overlays
        for w in self.auto_overlays:
            if self.core.fast_mode: w.deiconify()
            else:                    w.withdraw()
        print(f"[GUI] Mode → {label}")

    def _toggle_overlays(self):
        self.overlays_visible = not self.overlays_visible
        for w in self.overlay_windows:
            if self.overlays_visible:
                if w not in self.auto_overlays or self.core.fast_mode:
                    w.deiconify()
            else:
                w.withdraw()
        self.overlays_toggle_btn.config(
            text=f"Overlays: {'Shown' if self.overlays_visible else 'Hidden'}")

    def _reset_counter(self):
        self.core.cancel_all_scheduled_events()
        self.core.answers_count = 0
        self.core.ready_count   = 0
        self.core.extended_sequence_active = False
        self._clear_transient_state("manual reset")
        self.update_counter_label(0, 0)
        self.set_auto_status("")

    def _set_answer_clicks_enabled(self, enabled, reason="", cancel_auto=False):
        """
        Set ANSWER-CLICK submission only — nothing else.

        This deliberately does NOT touch auto_sequence_enabled, preview,
        pause, or any solver state: whether a solved answer may be typed on
        the keypad is an independent control from the AUTO 1/2/3 macro
        sequence, and the old single "Automation" flag that coupled them
        meant there was no way to stop just the answer submission.

        cancel_auto is a DELIBERATE safety rule, not flag coupling:
          - The confirmation-failure safety net passes cancel_auto=True. If
            clicks are demonstrably not landing, any AUTO 1/2/3 step already
            scheduled by Tk is acting on a screen state that is now
            suspect — those in-flight actions are cancelled. The AUTO
            *setting* itself is left alone (the user's choice stays as it
            was; only the stale scheduled actions are dropped).
          - A manual toggle passes cancel_auto=False: turning answer clicks
            off by hand must not reach into the AUTO sequence at all.
        """
        self.core.answer_clicks_enabled = enabled
        if enabled:
            self.answer_clicks_btn.config(text="ON", fg=C_GREEN)
            print("[GUI] Answer clicks ENABLED")
        else:
            # A pending confirmation was waiting to see whether the click
            # that started it landed — with answer clicking now off there is
            # nothing further to click, so its timeout no longer means
            # anything. Drop it and reset the streak: (re)enabling answer
            # clicks is a deliberate boundary the user took specifically to
            # reset the subsystem.
            self.core.qsm.drop_confirmation()
            self.core.qsm.consecutive_unconfirmed = 0
            if cancel_auto:
                self.core.cancel_all_scheduled_events()
                self.core.extended_sequence_active = False
            self.answer_clicks_btn.config(text="OFF", fg=C_RED)
            if reason:
                self.set_auto_status(reason, "red")
            else:
                self.set_auto_status("")
            print("[GUI] Answer clicks DISABLED"
                  + (" — scheduled AUTO actions cancelled (safety rule)" if cancel_auto else "")
                  + (f" ({reason})" if reason else ""))

    def _set_auto_sequence_enabled(self, enabled):
        """
        Set the AUTO 1/2/3 sequence switch only. Turning it off cancels any
        already-scheduled sequence steps (immediate effect instead of each
        step no-op'ing through _can_auto). It NEVER touches
        answer_clicks_enabled: switching the AUTO sequence on does not
        enable answer submission, and switching it off does not stop
        ordinary answer clicks.
        """
        self.core.auto_sequence_enabled = enabled
        if enabled:
            self.auto_seq_btn.config(text="ON", fg=C_GREEN)
            print("[GUI] AUTO sequence ENABLED")
        else:
            self.core.cancel_all_scheduled_events()
            self.core.extended_sequence_active = False
            self.auto_seq_btn.config(text="OFF", fg=C_RED)
            print("[GUI] AUTO sequence DISABLED — scheduled steps cancelled")

    def _toggle_answer_clicks(self):
        self._set_answer_clicks_enabled(not self.core.answer_clicks_enabled)

    def _toggle_auto_sequence(self):
        self._set_auto_sequence_enabled(not self.core.auto_sequence_enabled)

    def _toggle_preview(self):
        self.core.preview_enabled = not self.core.preview_enabled
        if self.core.preview_enabled:
            self.preview_toggle_btn.config(text="●  Live", fg=C_CYAN)
            self._preview_force = True   # show something on the next cycle
        else:
            self.preview_toggle_btn.config(text="Off", fg=C_MUTED_DIM)
            self.preview_canvas.delete("all")
            self.preview_canvas.create_text(143, 30, text="Preview disabled",
                                            fill=C_MUTED, font=F_LABEL)

    # ─────────────────────────────────────────────────────────────────────────
    # Saves modal (replaces the old inline slot buttons)
    # ─────────────────────────────────────────────────────────────────────────

    def _open_saves_modal(self, save_mode=False):
        """
        Open a clean Toplevel window showing the 3 coord slots.
        save_mode=True: buttons say "Save here"; False: buttons say "Load".
        """
        modal = tk.Toplevel(self.root)
        modal.title("Saved Layouts")
        modal.geometry("300x260")
        modal.resizable(False, False)
        modal.configure(bg=C_BG)
        modal.attributes("-topmost", True)
        modal.grab_set()   # modal behaviour

        tk.Label(modal, text="Saved Layouts",
                 fg=C_FG, bg=C_BG, font=F_STATUS).pack(pady=(SP_4, SP_1))

        hint = ("Choose a slot to save this layout into."
                if save_mode else
                "Choose a saved layout to load.")
        self.modal_hint = tk.Label(modal, text=hint,
                                   fg=C_MUTED, bg=C_BG, font=F_LABEL)
        self.modal_hint.pack(pady=(0, SP_3))

        slots_frame = tk.Frame(modal, bg=C_BG)
        slots_frame.pack(fill="x", padx=SP_4)

        slots = self.core.read_slots()
        for i in range(MAX_SLOTS):
            slot      = slots[i]
            is_active = (i == self.active_slot)

            row = tk.Frame(slots_frame, bg=C_SURFACE, padx=SP_2, pady=SP_2)
            row.pack(fill="x", pady=SP_1 // 2)

            left = tk.Frame(row, bg=C_SURFACE)
            left.pack(side="left", fill="x", expand=True)
            name = f"Layout {i + 1}"
            if is_active:
                name += "  ·  active"
            tk.Label(left, text=name, fg=C_FG if not is_active else C_ACCENT,
                     bg=C_SURFACE, font=F_BODY_B if is_active else F_BODY,
                     anchor="w").pack(anchor="w")
            sub = f"Saved {slot['saved']}" if slot else "Empty"
            tk.Label(left, text=sub, fg=C_MUTED_DIM, bg=C_SURFACE,
                     font=F_HELPER, anchor="w").pack(anchor="w")

            if save_mode:
                btn_kind, btn_text = "primary", "Save here"
                btn_cmd = lambda idx=i, m=modal: self._save_to_slot(idx, m)
            elif slot:
                btn_kind, btn_text = "secondary", "Load"
                btn_cmd = lambda idx=i, m=modal: self._load_slot(idx, m)
            else:
                btn_kind, btn_text = "ghost", "Empty"
                btn_cmd = None

            slot_btn = self._button(row, btn_text, btn_cmd or (lambda: None),
                                    kind=btn_kind, font=F_LABEL)
            if not btn_cmd:
                slot_btn.config(state="disabled", cursor="arrow")
            slot_btn.pack(side="right")

        close_btn = self._button(modal, "Close", modal.destroy, kind="ghost")
        close_btn.pack(pady=(SP_3, SP_3))

    def _save_to_slot(self, idx, modal=None):
        slots = self.core.read_slots()
        slots[idx] = {
            "label":  f"Slot {idx+1}",
            "saved":  datetime.datetime.now().strftime("%d/%m %H:%M"),
            "coords": self.core.build_coord_snapshot(),
        }
        self.core.write_slots(slots)
        self.active_slot        = idx
        self._edit_save_pending = False
        print(f"[GUI] Slot {idx+1} saved")
        if modal:
            modal.destroy()

    def _load_slot(self, idx, modal=None):
        if not self.core.paused:
            self._toggle_pause()
        ok = self.core.load_coord_slot(idx)
        if ok:
            self.active_slot = idx
            # New coordinates mean a totally different screen region — any
            # cached frame hash / session answer from the old region is
            # meaningless (and dangerous: the same pixel hash is very
            # unlikely but the same STALE answer being auto-clicked into a
            # different question is exactly the kind of bug that's hard to
            # notice until it's already clicked something wrong).
            self._clear_transient_state(f"coordinate slot {idx+1} loaded")
            self._rebuild_overlays()
            self.sync_pause_state()
            print(f"[GUI] Slot {idx+1} loaded — click Resume when ready")
        if modal:
            modal.destroy()

    # ─────────────────────────────────────────────────────────────────────────
    # Reset to defaults
    # ─────────────────────────────────────────────────────────────────────────

    def _reset_to_defaults(self):
        if self.edit_mode:
            self._cancel_edit_mode()
        if not self.core.paused:
            self._toggle_pause()
        self.core.reset_coords_to_defaults()
        self.active_slot = None
        self._clear_transient_state("coordinates reset to defaults")
        self._rebuild_overlays()
        self.sync_pause_state()
        print("[GUI] Reset to defaults — click Resume when ready")

    # ─────────────────────────────────────────────────────────────────────────
    # Overlay creation & management
    # ─────────────────────────────────────────────────────────────────────────

    def _build_overlays(self):
        self.overlay_windows     = []
        self.auto_overlays       = []
        self.key_overlays        = {}
        self.auto_area_1_overlay = None
        self.auto_area_2_overlay = None
        self.auto_area_3_overlay = None

        qa = self.core.question_area_fast if self.core.fast_mode else self.core.question_area
        x1, y1, x2, y2 = qa
        self.question_overlay = self._create_overlay(
            x1, y1, x2-x1, y2-y1, "red", "", is_auto=False)

        bs = int(50 * SCALE_X)
        for key, (x, y) in self.core.key_coords.items():
            w = self._create_overlay(x-bs//2, y-bs//2, bs, bs,
                                     "cyan", key, is_auto=False)
            self.key_overlays[key] = w

        abs_ = int(60 * SCALE_X)
        areas = [
            (bot_core.AUTO_AREA_1, "yellow",  "AUTO 1", 'auto_area_1_overlay'),
            (bot_core.AUTO_AREA_2, "magenta", "AUTO 2", 'auto_area_2_overlay'),
            (bot_core.AUTO_AREA_3, "lime",    "AUTO 3", 'auto_area_3_overlay'),
        ]
        for (ax, ay), color, label, attr in areas:
            w = self._create_overlay(ax-abs_//2, ay-abs_//2,
                                     abs_, abs_, color, label, is_auto=True)
            setattr(self, attr, w)

        # Hide auto overlays in standard mode
        if not self.core.fast_mode:
            for w in self.auto_overlays:
                w.withdraw()

    def _rebuild_overlays(self):
        """Destroy all overlays and recreate from current core coords."""
        for w in self.overlay_windows:
            try: w.destroy()
            except Exception: pass
        self._build_overlays()
        if not self.overlays_visible:
            for w in self.overlay_windows:
                w.withdraw()
        elif not self.core.fast_mode:
            for w in self.auto_overlays:
                w.withdraw()

    def _create_overlay(self, x, y, w, h, color, label, is_auto=False):
        win = tk.Toplevel(self.root)
        win.overrideredirect(True)
        win.attributes("-topmost", True)
        win.geometry(f"{w}x{h}+{x}+{y}")
        win._edit_x = x
        win._edit_y = y
        win._edit_w = w
        win._edit_h = h

        bg = "black"
        win.config(bg=bg)
        win.attributes("-transparentcolor", bg)

        cv = tk.Canvas(win, width=w, height=h, bg=bg, highlightthickness=0)
        cv.pack()
        cv.create_rectangle(2, 2, w-2, h-2, outline=color, width=3)
        if label:
            cv.create_text(w//2, 10, text=label, fill=color,
                           font=("Arial", 10, "bold"), anchor="n")

        hwnd = ctypes.windll.user32.GetParent(win.winfo_id())
        st   = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE,
                                            st | WS_EX_LAYERED | WS_EX_TRANSPARENT)

        self.overlay_windows.append(win)
        if is_auto:
            self.auto_overlays.append(win)
        return win

    # ─────────────────────────────────────────────────────────────────────────
    # Edit mode
    # ─────────────────────────────────────────────────────────────────────────

    def _toggle_edit_mode(self):
        if not self.edit_mode:
            self._enter_edit_mode()
        else:
            self._exit_edit_mode()

    def _enter_edit_mode(self):
        if not self.core.paused:
            self._toggle_pause()
        self.edit_mode = True
        self._style_button(self.edit_btn, "primary")
        self.edit_btn.config(text="✓  Done editing")
        self._set_status("●  Editing", C_ORANGE)
        self.set_auto_status("Drag boxes to move them, corners to resize", "orange")

        for w in self.overlay_windows:
            w.deiconify()
            self._set_click_through(w, False)
            self._make_draggable(w)

        self._attach_resize_handles()

        # Auto-open saves modal in save mode
        self._open_saves_modal(save_mode=True)
        print("[GUI] Edit mode ON")

    def _exit_edit_mode(self):
        self._remove_resize_handles()

        # Push new positions into core
        auto_map = {
            'a1': self.auto_area_1_overlay,
            'a2': self.auto_area_2_overlay,
            'a3': self.auto_area_3_overlay,
        }
        self.core.apply_overlay_positions(
            self.key_overlays, auto_map,
            self.question_overlay, self.core.fast_mode
        )

        self.edit_mode          = False
        self._edit_save_pending = True
        self._style_button(self.edit_btn, "ghost")
        self.edit_btn.config(text="Edit Layout")

        for w in self.overlay_windows:
            cv = w.winfo_children()[0]
            cv.unbind("<ButtonPress-1>")
            cv.unbind("<B1-Motion>")
            self._set_click_through(w, True)

        if not self.overlays_visible:
            for w in self.overlay_windows: w.withdraw()
        elif not self.core.fast_mode:
            for w in self.auto_overlays: w.withdraw()

        self.sync_pause_state()
        self.set_auto_status("Layout updated — save it, or Resume to try it out", "gray")
        print("[GUI] Edit mode OFF — coords applied. Save to slot or Resume to skip.")

    def _cancel_edit_mode(self):
        """Cancel edit mode without applying positions (used by Reset to Defaults)."""
        self._remove_resize_handles()
        for w in self.overlay_windows:
            cv = w.winfo_children()[0]
            cv.unbind("<ButtonPress-1>")
            cv.unbind("<B1-Motion>")
            self._set_click_through(w, True)
        self.edit_mode = False
        self._style_button(self.edit_btn, "ghost")
        self.edit_btn.config(text="Edit Layout")

    # ── Drag ──────────────────────────────────────────────────────────────────

    def _make_draggable(self, win):
        cv = win.winfo_children()[0]
        cv._dsx, cv._dsy = 0, 0

        def on_press(e, w=win, c=cv):
            c._dsx = e.x_root - w._edit_x
            c._dsy = e.y_root - w._edit_y

        def on_drag(e, w=win, c=cv):
            nx, ny = e.x_root - c._dsx, e.y_root - c._dsy
            w._edit_x, w._edit_y = nx, ny
            w.geometry(f"+{nx}+{ny}")

        cv.bind("<ButtonPress-1>", on_press)
        cv.bind("<B1-Motion>",     on_drag)

    # ── Resize handles ────────────────────────────────────────────────────────

    def _attach_resize_handles(self):
        HSIZE = 12
        qw    = self.question_overlay
        for corner in ("tl", "tr", "bl", "br"):
            h = tk.Toplevel(self.root)
            h.overrideredirect(True)
            h.attributes("-topmost", True)
            h._corner      = corner
            h._handle_size = HSIZE
            hx, hy = self._handle_pos(qw, corner, HSIZE)
            h.geometry(f"{HSIZE}x{HSIZE}+{hx}+{hy}")
            cv = tk.Canvas(h, width=HSIZE, height=HSIZE, bg="red",
                           highlightthickness=0, cursor="sizing")
            cv.pack()
            cv.create_rectangle(1, 1, HSIZE-1, HSIZE-1,
                                 fill="red", outline="white", width=1)
            cv._dsx, cv._dsy = 0, 0

            def on_press(e, c=cv): c._dsx, c._dsy = e.x_root, e.y_root
            def on_drag(e, c=cv, hw=h, qwin=qw):
                dx, dy = e.x_root - c._dsx, e.y_root - c._dsy
                c._dsx, c._dsy = e.x_root, e.y_root
                self._resize_ocr_box(qwin, hw._corner, dx, dy)
                for rh in self._resize_handles:
                    rx, ry = self._handle_pos(qwin, rh._corner, rh._handle_size)
                    rh.geometry(f"+{rx}+{ry}")

            cv.bind("<ButtonPress-1>", on_press)
            cv.bind("<B1-Motion>",     on_drag)
            self._resize_handles.append(h)

    def _handle_pos(self, qw, corner, size):
        x, y, w, h = qw._edit_x, qw._edit_y, qw._edit_w, qw._edit_h
        half = size // 2
        return {"tl": (x-half, y-half), "tr": (x+w-half, y-half),
                "bl": (x-half, y+h-half), "br": (x+w-half, y+h-half)}[corner]

    def _resize_ocr_box(self, qw, corner, dx, dy):
        MIN_W, MIN_H = 40, 15
        x, y, w, h = qw._edit_x, qw._edit_y, qw._edit_w, qw._edit_h
        if corner == "br":
            w = max(MIN_W, w+dx); h = max(MIN_H, h+dy)
        elif corner == "bl":
            nw = max(MIN_W, w-dx)
            if nw > MIN_W: x += dx
            w = nw; h = max(MIN_H, h+dy)
        elif corner == "tr":
            nh = max(MIN_H, h-dy)
            if nh > MIN_H: y += dy
            h = nh; w = max(MIN_W, w+dx)
        elif corner == "tl":
            nw = max(MIN_W, w-dx); nh = max(MIN_H, h-dy)
            if nw > MIN_W: x += dx
            if nh > MIN_H: y += dy
            w = nw; h = nh
        qw._edit_x, qw._edit_y, qw._edit_w, qw._edit_h = x, y, w, h
        qw.geometry(f"{w}x{h}+{x}+{y}")
        cv = qw.winfo_children()[0]
        cv.config(width=w, height=h)
        cv.delete("all")
        cv.create_rectangle(2, 2, w-2, h-2, outline="red", width=3)

    def _remove_resize_handles(self):
        for h in self._resize_handles:
            try: h.destroy()
            except Exception: pass
        self._resize_handles.clear()

    # ── Click-through helpers ─────────────────────────────────────────────────

    def _set_click_through(self, win, enabled):
        hwnd = ctypes.windll.user32.GetParent(win.winfo_id())
        st   = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        if enabled:
            new = st | WS_EX_LAYERED | WS_EX_TRANSPARENT
        else:
            new = (st | WS_EX_LAYERED) & ~WS_EX_TRANSPARENT
        ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new)

    # ─────────────────────────────────────────────────────────────────────────
    # Taskbar monitoring
    # ─────────────────────────────────────────────────────────────────────────

    def _schedule_taskbar_check(self):
        self.root.after(TASKBAR_CHECK_INTERVAL, self._check_taskbar)

    def _check_taskbar(self):
        if self.core.check_taskbar_position():
            print("[GUI] Taskbar moved — rebuilding overlays")
            self._rebuild_overlays()
        self._schedule_taskbar_check()

    # ─────────────────────────────────────────────────────────────────────────
    # Main detection loop
    # ─────────────────────────────────────────────────────────────────────────

    def _main_loop(self):
        """
        The heartbeat — one capture, three independent consumers.

        Layered scheduling (see question_state.py):

          EVERY cycle, paused or not:
            1. capture frame (single shared mss grab — no duplicate capture)
            2. exact-frame digest (BLAKE2b) + click-confirmation watcher
            3. preview update when preview enabled — PAUSE-INDEPENDENT
            4. target-window tracking

          Only when NOT paused (the solver gate):
            5. question processing — OCR / solve / click, gated by the
               retry engine (time-based, bounded), NOT by frame-hash
               equality. An unchanged frame with an unsettled question is
               retried on schedule instead of being suppressed forever.
        """
        self.core._prune_scheduled_events()
        # Runs every poll, paused or not — see _track_target_window()'s
        # docstring for why this can't be a one-shot "capture on resume"
        # instead.
        self.core._track_target_window()
        try:
            area = (self.core.question_area_fast if self.core.fast_mode
                    else self.core.question_area)

            # ── 1. Capture ────────────────────────────────────────────────
            monitor = {
                "left":   area[0], "top":    area[1],
                "width":  area[2] - area[0],
                "height": area[3] - area[1],
            }
            sct_img = self._sct.grab(monitor)

            # ── 2. Exact frame digest (layer A). Used ONLY for click
            # confirmation and as a freshness signal — never as a
            # suppression gate.
            digest = frame_digest(sct_img.bgra)
            self.last_frame_hash = digest

            # ── 3. Click-confirmation watcher (watch-only; runs paused or
            # not, before any processing, since an unchanged frame after a
            # click is exactly the failure case being checked).
            self._run_click_confirmation(digest)

            # ── 4. Preview: decoupled from the solver. Runs while PAUSED so
            # the operator keeps a live view; the capture above is shared by
            # both paths (no duplicate screen-capture system).
            if self.core.preview_enabled:
                self._update_preview(sct_img, force=self._preview_force)
                self._preview_force = False

            # ── 5. Solver path — the ONLY thing pause stops.
            if not self.core.paused:
                self._process_frame(sct_img, digest)

        except Exception:
            # Full traceback, not just the message: a bare print() made
            # loop bugs (which repeat every cycle) nearly impossible to
            # diagnose from the console.
            import traceback
            print("[GUI] Loop error:")
            traceback.print_exc()

        # Reschedule with crash recovery
        try:
            self.root.after(self.core.current_polling, self._main_loop)
        except Exception as e:
            print(f"[GUI] FATAL: loop reschedule failed: {e} — retry in 500ms")
            try:
                self.root.after(500, self._main_loop)
            except Exception:
                pass

    # ─────────────────────────────────────────────────────────────────────────
    # Frame processing (solver path; called only when NOT paused)
    # ─────────────────────────────────────────────────────────────────────────

    def _fingerprint_context(self):
        """
        Everything besides the expression itself that changes what a
        question MEANS to the solver: enabled operations, fast/standard
        mode, and the capture region. Feeds the semantic fingerprint, so a
        coordinate-profile or mode switch can never collide with the
        previous configuration's question state.
        """
        area = (self.core.question_area_fast if self.core.fast_mode
                else self.core.question_area)
        return (f"ops={sorted(self.core.enabled_operations)}|"
                f"fast={self.core.fast_mode}|area={tuple(area)}")

    def _process_frame(self, sct_img, digest):
        """
        OCR → canonical question → semantic fingerprint → retry gate →
        solve → click policy.

        Replaces the old frame-hash early-return: the frame digest NEVER
        decides whether processing happens — the retry engine does, on a
        time schedule, so an unchanged frame with an unsettled question is
        retried (with backoff, bounded) instead of suppressed.
        """
        core = self.core
        qsm  = core.qsm
        now  = time.monotonic()

        visual_changed = qsm.observe_visual(digest, now).changed

        # ── Retry gate (forensic audit BUG-2). The decision below belongs
        # to the PREVIOUS frame's identity, so when the gate is closed:
        #   - unchanged frame -> the closed gate stands; return.
        #   - changed frame (or no identity at all — fail-closed discovery,
        #     e.g. first frame / after reset_round) -> look at the WHY via
        #     the cheap pixel signature: animation-level deltas stand down;
        #     genuine content changes get ONE immediate identity probe
        #     (capped per closed-gate window), and only a genuinely NEW
        #     question may proceed to solve/click. Same identity or an
        #     unresolved reading only updates tracking — no budget spent —
        #     and the scheduled retry still fires on time.
        pre_fp = qsm.current_fingerprint
        decision = qsm.should_process(now)
        arr = None              # preprocessed frame, shared with the OCR path
        frame_sig = None        # pixel signature of this frame (when computed)
        forced_look = False
        if not decision.allowed:
            reason_key = decision.reason.split("_")[0]      # backoff/awaiting/question/no
            if (reason_key, pre_fp) != self._blocked_key:
                self._blocked_key = (reason_key, pre_fp)
                self._blocked_looks = 0
            may_probe = (visual_changed
                         or decision.reason == "no_identity")
            if not may_probe:
                debug_throttled(f"gate:{decision.reason}",
                                f"frame unchanged — processing skipped ({decision.reason})")
                return
            if visual_changed and pre_fp is not None:
                # WHY did the pixels change? Cheap numpy signature — no OCR.
                # Animation-level change: the gate stands (observe_visual
                # already pulled the retry deadline in for a due look).
                raw_np = np.array(sct_img)
                arr = core.preprocess_for_ocr(raw_np)
                frame_sig = visual_signature(arr)
                if self._last_look_sig is not None and signature_distance(
                        frame_sig, self._last_look_sig,
                        qsm.policy.unresolved_min_ink,
                        bail_below=qsm.policy.unresolved_match_threshold,
                ) <= qsm.policy.unresolved_match_threshold:
                    debug_throttled("gate:anim",
                                    "visual change is animation-level — retry gate stands")
                    return
                # Unreadable-look floor (S10 finding): if the previous look
                # read NOTHING canonical (hostile re-render churn mints a
                # fresh unresolved episode per frame), probes are floored to
                # the first retry tier — a hard OCR rate bound that no
                # per-identity bookkeeping can defeat, because hostile
                # frames share no identity to cap against. A look that READ
                # something is never floored: genuine new-question discovery
                # stays immediate.
                last_t, last_unreadable = self._last_look
                if (last_unreadable and last_t is not None
                        and now - last_t < qsm.policy.same_frame_retry_delay):
                    return
                # Adversarial bound (video-like re-renders defeat the
                # signature gate): cap same-identity probes per window.
                if self._blocked_looks >= VISUAL_LOOK_MAX_PER_WINDOW:
                    debug_throttled("gate:probe_cap",
                                    f"identity-probe cap ({VISUAL_LOOK_MAX_PER_WINDOW}) "
                                    f"reached for this gate window — standing down")
                    return
            self._blocked_looks += 1
            forced_look = True
        else:
            self._blocked_key = None
            self._blocked_looks = 0

        # ── Frame cache: exact pixels recently solved → answer without OCR.
        # The entry carries the SEMANTIC fingerprint it was solved under, so
        # a hit still flows through full identity + click-policy checks — a
        # completed question can never be re-clicked just because its pixels
        # reappeared (this replaces the old fragile `last_question == ""`
        # guard). Only consulted on a visual change or a forced identity
        # probe: on unchanged pixels the retry engine owns the decision
        # above.
        cached = core.frame_cache.get(digest) if (visual_changed
                                                  or forced_look) else None
        fingerprint = None
        rt = qsm.runtime()
        if cached is not None:
            fingerprint = cached.fingerprint
            canonical   = cached.canonical
            raw         = cached.canonical
            conf        = None
            display     = "(cached frame)"
            qsm.observe_question(fingerprint, canonical, raw, now)
            answer, source = cached.answer, cached.source
            print(f"[GUI] [FRAME CACHE] {answer} (ttl-valid)")
        elif (not visual_changed and rt is not None and rt.solved
              and rt.answer is not None):
            # Retry tick on unchanged pixels with the answer already known
            # this round: skip the EasyOCR pass entirely — the pixels are
            # bit-identical, so OCR would return the same reading. The
            # retry is a bounded CLICK re-attempt, not a re-solve.
            fingerprint = rt.fingerprint
            canonical   = rt.canonical
            raw         = rt.canonical
            conf        = rt.last_ocr_confidence
            display     = rt.canonical
            qsm.observe_question(fingerprint, canonical, raw, now)
            answer, source = rt.answer, rt.source
        else:
            # ── OCR (new pixels, or retry due on unchanged pixels) ────────
            if arr is None:            # probe gate may have preprocessed already
                raw_np = np.array(sct_img)
                arr = core.preprocess_for_ocr(raw_np)
            frame_sig = visual_signature(arr)
            self._last_look_sig = frame_sig        # anchor for the probe gate

            result = core.reader.readtext(
                arr,
                allowlist='0123456789+-*/()=?xX×÷: ',
                low_text=0.3, batch_size=1, paragraph=False, min_size=5
            )

            raw = core.select_math_ocr_text(result, arr) if result else ""
            core.last_question = raw
            canonical = core.canonicalise(raw) if raw else ""
            conf = core.last_ocr_confidence
            display = raw
            if canonical:
                fingerprint = semantic_fingerprint(
                    canonical, raw, self._fingerprint_context())
            else:
                # UNRESOLVED VISUAL IDENTITY (OCR-flicker loophole fix).
                # OCR could not produce a canonical expression, so the raw
                # text is garbage and must NEVER decide identity: garbage
                # A/B/C from the same screen would otherwise mint three
                # independent fingerprints, each with a fresh retry budget.
                # Identity here is the question-region PIXELS — the same
                # preprocessed image EasyOCR just consumed — matched by
                # aligned, dead-zoned distance against the current
                # unresolved episode (small noise/jitter/blink = same
                # episode, same budget; genuinely changed pixels = new
                # episode, fresh budget). See docs/REDESIGN_REPORT.md §2.5.
                fingerprint = qsm.observe_unresolved(
                    visual_signature(arr), self._fingerprint_context(), now)
            qsm.observe_question(fingerprint, canonical, raw, now)
            # (S10 floor bookkeeping) an unreadable reading floors the next
            # closed-gate probe; a readable one never does.
            self._last_look = (now, not bool(canonical))

            # ── Solve. If the machine already knows this exact question's
            # answer this round (retry path), don't re-solve — reuse it so
            # the retry is a bounded click re-attempt, not a solve loop.
            rt = qsm.runtime()
            if rt is not None and rt.solved and rt.canonical == canonical \
                    and rt.answer is not None:
                answer, source = rt.answer, rt.source
            elif raw:
                answer, source = core.handle_question(raw)
            else:
                answer, source = None, None

        # ── Outcome recording + click policy (shared by OCR path and
        # frame-cache path). A forced probe must justify itself first:
        # only a genuinely different question proceeds while the old
        # identity's gate is closed; an unchanged identity or an unreadable
        # frame only updates tracking — budget stays untouched until the
        # gate opens (BUG-2 contract).
        if forced_look:
            if frame_sig is not None:
                self._last_look_sig = frame_sig
            if pre_fp is not None and (qsm.current_fingerprint == pre_fp
                                       or answer is None):
                # Probe did not discover a new question: show what was seen
                # (if anything) and stand down — the retry gate stays closed
                # and no retry budget is spent (BUG-2 contract).
                debug_throttled("gate:probe",
                                "identity probe: same question or unreadable — "
                                "retry gate stays closed, no budget spent")
                self._update_detected_display(display, answer, source=source)
                return

        if answer is None:
            outcome = OUTCOME_OCR_EMPTY if not raw else OUTCOME_UNSOLVED
            qsm.record_outcome(outcome, now, ocr_confidence=conf)
            self._update_detected_display(display, None)
            return

        rt = qsm.runtime()
        if rt is not None and rt.done:
            # Question already completed this round (click confirmed) — the
            # answer may be re-displayed but must NEVER be re-clicked while
            # the identity is unchanged. Cache the frame so animation-heavy
            # screens do not re-OCR a completed question on every change.
            debug_throttled("click:done",
                            f"answer {answer} known but question already "
                            f"completed this round — not re-clicking")
            self._update_detected_display(display, answer, source=source)
            if cached is None and fingerprint:
                core.frame_cache.put(digest, answer, source,
                                     fingerprint, canonical)
            return
        if rt is not None and rt.awaiting_confirmation:
            # A previous click for THIS question is still being watched —
            # never click again on top of it (anti-spam core).
            debug_throttled("click:awaiting",
                            "click awaiting confirmation — no further click")
            self._update_detected_display(display, answer, source=source)
            return
        if rt is not None and rt.exhausted:
            # Forensic audit BUG-3 — enforce the documented contract of
            # max_retries_per_question: after exhaustion CLICKING stops for
            # this question (until its identity changes, which resets the
            # flag), while OCR re-validation continues at the capped cadence
            # (should_process reason "revalidate_exhausted"). The answer is
            # still displayed and cached so animation never re-OCRs it.
            debug_throttled("click:exhausted",
                            f"answer {answer} known but retry budget exhausted — "
                            f"not clicking (re-validation continues)")
            self._update_detected_display(display, answer, source=source)
            if cached is None and fingerprint:
                core.frame_cache.put(digest, answer, source,
                                     fingerprint, canonical)
            return

        click_result = core.click_answer(answer, source, norm_expr=canonical)
        qsm.record_outcome(self._outcome_for_click(click_result), now,
                           answer=answer, source=source, ocr_confidence=conf)
        self._update_detected_display(display, answer, source=source,
                                      click_result=click_result)
        # Only a VERIFIED click arms confirmation — "known but not
        # submitted" results must not start a confirmation watch.
        if click_result == CLICK_RESULT_CLICKED:
            qsm.arm_confirmation(digest, now)
        # Frame cache stores successes only (unchanged rule), now with
        # identity + TTL.
        if cached is None:
            core.frame_cache.put(digest, answer, source, fingerprint, canonical)

    @staticmethod
    def _outcome_for_click(click_result):
        """Map a CLICK_RESULT_* onto the retry engine's outcome vocabulary."""
        return {
            CLICK_RESULT_CLICKED:         OUTCOME_CLICKED,
            CLICK_RESULT_AUTOMATION_OFF:  OUTCOME_NOT_ENABLED,
            CLICK_RESULT_WRONG_WINDOW:    OUTCOME_WRONG_WINDOW,
            CLICK_RESULT_UNMAPPED_ANSWER: OUTCOME_UNMAPPED,
            CLICK_RESULT_ERROR:           OUTCOME_ERROR,
        }.get(click_result, OUTCOME_ERROR)

    def _run_click_confirmation(self, digest):
        """
        Watch a pending click's outcome. The frame digest changing after a
        click confirms it (screen-change proof, not acceptance proof — the
        same semantics as before, deliberately kept); the deadline elapsing
        with unchanged pixels counts one unconfirmed strike, and the
        threshold trips the safety net — which now disables ANSWER CLICKING
        specifically (plus, as a documented rule, cancels already-scheduled
        AUTO actions whose screen state is now suspect) instead of flipping
        unrelated solver or preview state.
        """
        qsm = self.core.qsm
        if qsm.pending_confirm_digest is None:
            return
        now = time.monotonic()
        if digest != qsm.pending_confirm_digest:
            qsm.record_outcome(OUTCOME_CONFIRMED, now)
            qsm.drop_confirmation()
            qsm.consecutive_unconfirmed = 0
            print("[GUI] Click confirmed — screen changed after click")
        elif now >= (qsm.pending_confirm_deadline or 0):
            qsm.record_outcome(OUTCOME_UNCONFIRMED, now)
            qsm.drop_confirmation()
            qsm.consecutive_unconfirmed += 1
            print(f"[GUI] [WARN] Click unconfirmed — screen unchanged after "
                  f"click ({qsm.consecutive_unconfirmed}/{self.UNCONFIRMED_THRESHOLD})")
            if qsm.consecutive_unconfirmed >= self.UNCONFIRMED_THRESHOLD:
                self._set_answer_clicks_enabled(
                    False,
                    f"⚠ {self.UNCONFIRMED_THRESHOLD} unconfirmed clicks — answer clicking paused",
                    cancel_auto=True)
                qsm.consecutive_unconfirmed = 0

    def _update_preview(self, sct_img, force=False):
        """
        Preview rendering — runs on the Tk main thread (called from the
        loop), every PREVIEW_UPDATE_INTERVAL cycles or immediately when
        forced (resume / re-enable). Independent of pause state.
        """
        self._preview_counter += 1
        if not force and self._preview_counter < PREVIEW_UPDATE_INTERVAL:
            return
        self._preview_counter = 0
        try:
            img = Image.frombytes("RGB", sct_img.size,
                                  sct_img.bgra, "raw", "BGRX")
            prev = img.resize((286, 60))
            self.preview_img_tk = ImageTk.PhotoImage(prev)
            if getattr(self.preview_canvas, '_img_id', None):
                self.preview_canvas.itemconfig(
                    self.preview_canvas._img_id,
                    image=self.preview_img_tk)
            else:
                self.preview_canvas._img_id = self.preview_canvas.create_image(
                    0, 0, anchor=tk.NW, image=self.preview_img_tk)
        except Exception as e:
            debug_throttled("preview:error", f"preview update failed: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Entry point
    # ─────────────────────────────────────────────────────────────────────────

    def run(self):
        print("[GUI] Starting — FAST mode, 10 ms polling")
        self.root.after(200, self._main_loop)
        self.root.mainloop()


if __name__ == "__main__":
    OpticalReaderSolverGUI().run()
