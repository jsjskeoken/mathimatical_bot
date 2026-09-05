"""
gui.py — Tkinter UI, overlay management, main loop scheduler.
Imports BotCore from bot_core.py and calls back into it for all logic.
"""

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import mss
import hashlib
import numpy as np
import ctypes
import datetime
import time

from bot_core import (
    BotCore,
    QUESTION_AREA, QUESTION_AREA_FAST, KEY_COORDS,
    AUTO_AREA_1, AUTO_AREA_2, AUTO_AREA_3,
    TASKBAR_CHECK_INTERVAL, PREVIEW_UPDATE_INTERVAL,
    MAX_SLOTS, SCALE_X, SCALE_Y,
    MODE_HYBRID, MODE_CALC, MODE_LUT_ONLY,
    FAST_MODE_POLLING, STANDARD_MODE_POLLING,
)
import bot_core  # for the mutable globals QUESTION_AREA etc.

# Windows constants for click-through overlays
WS_EX_TRANSPARENT = 0x00000020
WS_EX_LAYERED     = 0x00080000
GWL_EXSTYLE       = -20

# ─────────────────────────────────────────────────────────────────────────────
# Design tokens — the single source of truth for every colour/font/spacing
# value used below. Nothing past this block should hardcode a one-off hex
# code or point size; add a token here instead so the whole app stays
# visually consistent and themeable from one place.
# ─────────────────────────────────────────────────────────────────────────────

# Colour — dark utility palette. The purple accent is the one saturated
# colour in the UI; everything else is either neutral or a single-purpose
# status colour (success/warning/danger), so nothing competes with it.
C_BG          = "#14141c"   # app background
C_SURFACE     = "#1c1c28"   # primary card
C_SURFACE_ALT = "#20202e"   # secondary panel (advanced area, modals)
C_BORDER      = "#32324a"   # card/section border
C_DIVIDER     = "#26263a"   # hairline between rows within a card

C_ACCENT       = "#7c6af7"  # primary accent — mode selection, primary button
C_ACCENT_HOVER = "#9384fa"

C_GREEN   = "#4ade80"   # success / running / enabled
C_GREEN_BG = "#1c3326"
C_RED     = "#f87171"   # danger / paused / disabled
C_RED_BG  = "#3a2222"
C_ORANGE  = "#fbbf24"   # warning — used sparingly, never as a default state
C_CYAN    = "#67e8f9"   # informational accent — "live" indicator only

C_FG        = "#e6e6f0"  # primary text
C_MUTED     = "#8a8aa3"  # secondary text / labels
C_MUTED_DIM = "#5c5c73"  # tertiary / helper text

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
SP_1, SP_2, SP_3, SP_4 = 4, 8, 12, 16

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


class OpticalReaderSolverGUI:
    """Full GUI shell. Creates a BotCore, wires up callbacks, owns the main loop."""

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
        # MD5 of the last captured frame — if unchanged, skip OCR entirely
        self.last_frame_hash  = None
        # Maps frame hash → (answer, source).  After the first OCR pass for any
        # question, subsequent appearances skip EasyOCR entirely and answer in
        # microseconds.  Capped at 500 entries (oldest evicted first).
        self.frame_answer_cache = {}

        # ── Click confirmation ──────────────────────────────────────────────
        # We already hash every frame anyway (for the frame-answer-cache) —
        # reuse that to check whether the screen actually changed after a
        # click, instead of assuming click_answer() worked just because it
        # didn't raise.
        self._pending_confirm_hash     = None  # frame hash right before the click
        self._pending_confirm_deadline = None  # time.time() by which we expect a change
        self._consecutive_unconfirmed  = 0
        self.CONFIRM_TIMEOUT       = 0.5  # seconds to wait for the screen to change
        self.UNCONFIRMED_THRESHOLD = 3    # auto-disable automation after this many in a row

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
        self.root.geometry("340x50+50+50")   # height is set for real once content is built
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
        # running, and what has it just seen? ─────────────────────────────
        status_card = self._card(root, outer_pady=(SP_3, 0))

        top_row = tk.Frame(status_card, bg=C_SURFACE)
        top_row.pack(fill="x")
        self.status_label = tk.Label(top_row, text="●  Running",
                                     fg=C_GREEN, bg=C_SURFACE, font=F_STATUS)
        self.status_label.pack(side="left")
        self.mode_label = tk.Label(top_row, text="FAST",
                                   fg=C_MUTED, bg=C_SURFACE, font=F_LABEL)
        self.mode_label.pack(side="right", anchor="s", pady=(0, 2))

        tk.Frame(status_card, bg=C_DIVIDER, height=1).pack(fill="x", pady=(SP_2, SP_2))

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
                                          bg=C_SURFACE, font=F_HELPER, wraplength=290,
                                          justify="left", anchor="w")
        self.auto_status_label.pack(fill="x", pady=(SP_1, 0))

        # ── Detected / Result — surfaces what the solver is doing right now
        # without requiring the person to read the tiny preview image. ─────
        det_card = self._card(root)
        det_cols = tk.Frame(det_card, bg=C_SURFACE)
        det_cols.pack(fill="x")

        det_left = tk.Frame(det_cols, bg=C_SURFACE)
        det_left.pack(side="left", fill="x", expand=True)
        self._section_label(det_left, "Detected").pack(anchor="w")
        self.detected_label = tk.Label(det_left, text="—", fg=C_FG, bg=C_SURFACE,
                                       font=F_DETECTED, anchor="w")
        self.detected_label.pack(anchor="w", fill="x")

        det_right = tk.Frame(det_cols, bg=C_SURFACE)
        det_right.pack(side="right")
        self._section_label(det_right, "Result").pack(anchor="e")
        self.result_label = tk.Label(det_right, text="—", fg=C_ACCENT, bg=C_SURFACE,
                                     font=F_RESULT, anchor="e")
        self.result_label.pack(anchor="e")

        # ── OCR preview — framed like the "live vision" of the app ─────────
        prev_card = self._card(root)
        hdr = tk.Frame(prev_card, bg=C_SURFACE)
        hdr.pack(fill="x")
        self._section_label(hdr, "OCR Preview").pack(side="left")
        self._live_dot = tk.Label(hdr, text="●", fg=C_CYAN, bg=C_SURFACE, font=F_HELPER)
        self._live_dot.pack(side="left", padx=(SP_1, 0))
        self.preview_toggle_btn = tk.Button(
            hdr, text="On", fg=C_GREEN, bg=C_SURFACE, bd=0, font=F_LABEL,
            activeforeground=C_GREEN, activebackground=C_SURFACE,
            command=self._toggle_preview, cursor="hand2")
        self.preview_toggle_btn.pack(side="right")

        preview_frame = tk.Frame(prev_card, bg=C_BORDER, padx=1, pady=1)
        preview_frame.pack(pady=(SP_2, 0))
        self.preview_canvas = tk.Canvas(preview_frame, width=286, height=60,
                                        bg="#0c0c14", highlightthickness=0)
        self.preview_canvas.pack()
        self.preview_img_tk = None

        # ── Primary controls — Pause/Resume and Automation are the two
        # actions that matter most, so they're the two largest buttons in
        # the app and nothing else competes with them for attention. ──────
        primary_card = self._card(root)
        self.pause_btn = self._button(primary_card, "❚❚  Pause", self._toggle_pause,
                                      kind="secondary", font=F_BODY_B)
        self.pause_btn.pack(fill="x")

        self.auto_btn = self._button(
            primary_card, "Automation\nEnabled", self._toggle_automation,
            kind="success", font=F_BODY_B)
        self.auto_btn.pack(fill="x", pady=(SP_2, 0))

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

        row1 = tk.Frame(self.advanced_frame, bg=C_SURFACE_ALT)
        row1.pack(fill="x")
        self.overlays_toggle_btn = self._button(row1, "Overlays: Shown",
                                                self._toggle_overlays, kind="ghost")
        self.overlays_toggle_btn.grid(row=0, column=0, padx=(0, SP_1), sticky="ew")
        self.edit_btn = self._button(row1, "Edit Layout", self._toggle_edit_mode,
                                     kind="ghost")
        self.edit_btn.grid(row=0, column=1, sticky="ew")
        row1.columnconfigure(0, weight=1)
        row1.columnconfigure(1, weight=1)

        row2 = tk.Frame(self.advanced_frame, bg=C_SURFACE_ALT)
        row2.pack(fill="x", pady=(SP_1, 0))
        saves_btn = self._button(row2, "Saved Layouts", self._open_saves_modal,
                                 kind="ghost")
        saves_btn.grid(row=0, column=0, padx=(0, SP_1), sticky="ew")
        session_reset_btn = self._button(row2, "New Round", self._reset_counter,
                                         kind="ghost")
        session_reset_btn.grid(row=0, column=1, sticky="ew")
        row2.columnconfigure(0, weight=1)
        row2.columnconfigure(1, weight=1)

        row3 = tk.Frame(self.advanced_frame, bg=C_SURFACE_ALT)
        row3.pack(fill="x", pady=(SP_1, 0))
        reset_def_btn = self._button(row3, "Reset Layout to Default",
                                     self._reset_to_defaults, kind="danger")
        reset_def_btn.grid(row=0, column=0, padx=(0, SP_1), sticky="ew")
        clear_lut_btn = self._button(row3, "Clear Saved Answers",
                                     self.core.clear_lut, kind="danger")
        clear_lut_btn.grid(row=0, column=1, sticky="ew")
        row3.columnconfigure(0, weight=1)
        row3.columnconfigure(1, weight=1)

        self.root.after_idle(self._resize_to_fit)

    # ─────────────────────────────────────────────────────────────────────────
    # UI update callbacks (called from BotCore)
    # ─────────────────────────────────────────────────────────────────────────

    def update_cache_label(self, count):
        self.cache_label.config(text=f"Cache {count}")

    def update_lut_label(self, count):
        self.lut_label.config(text=f"{count} saved answers")

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
        Mirror core.paused into the GUI widgets.
        Called via root.after(0, ...) from the hotkey listener thread
        so it always runs on the tkinter main thread.
        """
        if self.core.paused:
            self.status_label.config(text="●  Paused", fg=C_RED)
            self.pause_btn.config(text="▶  Resume")
        else:
            self.status_label.config(text="●  Running", fg=C_GREEN)
            self.pause_btn.config(text="❚❚  Pause")

    def _update_detected_display(self, expr, answer):
        """
        GUI-only bookkeeping: surfaces the last thing the solver saw/solved.
        Reads state _main_loop already computes — doesn't change what gets
        detected, solved, or clicked.
        """
        self.detected_label.config(text=expr if expr else "—")
        self.result_label.config(text=str(answer) if answer is not None else "—")

    # ─────────────────────────────────────────────────────────────────────────
    # Solver mode
    # ─────────────────────────────────────────────────────────────────────────

    def _clear_transient_state(self, reason=""):
        """
        Full reset of every cache that's only valid for the CURRENT capture
        configuration (which screen region + which mode). Previously a
        Fast/Standard switch only cleared answer_cache, and loading a
        coordinate profile cleared nothing at all — so session_cache and
        frame_answer_cache (keyed on pixel hashes from the OLD region/mode)
        could keep serving answers that have nothing to do with what's now
        on screen. Any change to what we're capturing or how we solve it
        should invalidate all of these together.
        """
        self.core.answer_cache.clear()
        self.core.session_cache.clear()
        self.frame_answer_cache.clear()
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
        if self.core.paused:
            self.status_label.config(text="●  Paused", fg=C_RED)
            self.pause_btn.config(text="▶  Resume")
        else:
            self.status_label.config(text="●  Running", fg=C_GREEN)
            self.pause_btn.config(text="❚❚  Pause")
            self.core._capture_target_window()
            # If user resumes while save-pending, cancel that state
            if self._edit_save_pending:
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

    def _set_automation_enabled(self, enabled, reason=""):
        """
        Single place that flips automation on/off and updates the button —
        used by the manual toggle AND by the auto-pause-on-unconfirmed-clicks
        safety net, so both stay visually consistent.
        """
        self.core.automation_enabled = enabled
        if enabled:
            self.auto_btn.config(text="Automation\nEnabled")
            self._style_button(self.auto_btn, "success")
            print("[GUI] Automation ENABLED")
        else:
            # Cancel any in-progress sequences immediately
            self.core.cancel_all_scheduled_events()
            self.core.extended_sequence_active = False
            self.auto_btn.config(text="Automation\nOff")
            self._style_button(self.auto_btn, "muted_off")
            if reason:
                self.set_auto_status(reason, "red")
            else:
                self.set_auto_status("")
            print(f"[GUI] Automation DISABLED — all sequences cancelled"
                  + (f" ({reason})" if reason else ""))

    def _toggle_automation(self):
        self._set_automation_enabled(not self.core.automation_enabled)

    def _toggle_preview(self):
        self.core.preview_enabled = not self.core.preview_enabled
        if self.core.preview_enabled:
            self.preview_toggle_btn.config(text="On", fg=C_GREEN)
            self._live_dot.config(fg=C_CYAN)
        else:
            self.preview_toggle_btn.config(text="Off", fg=C_MUTED)
            self._live_dot.config(fg=C_MUTED_DIM)
            self.preview_canvas.delete("all")
            self.preview_canvas.create_text(143, 30, text="Preview off",
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
            self.status_label.config(text="●  Paused", fg=C_RED)
            self.pause_btn.config(text="▶  Resume")
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
        self.status_label.config(text="●  Paused", fg=C_RED)
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
        self.status_label.config(text="●  Editing", fg=C_ORANGE)
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

        self.status_label.config(text="●  Paused", fg=C_RED)
        self.pause_btn.config(text="▶  Resume")
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
        The heartbeat: grab screen, run OCR, dispatch to core.handle_question.

        Pipeline (fastest path first):
          1. mss capture into raw BGRA buffer
          2. Hash .bgra — same frame → return immediately (zero work)
          3. Frame answer cache — known frame → answer without OCR (microseconds)
          4. Zero-copy numpy → OpenCV preprocess → EasyOCR (first sight only)
          5. PIL for UI preview only, completely off the solver hot path
        """
        self.core._prune_scheduled_events()
        try:
            if not self.core.paused:
                area = (self.core.question_area_fast if self.core.fast_mode
                        else self.core.question_area)

                # ── 1. Capture ────────────────────────────────────────────────
                monitor = {
                    "left":   area[0], "top":    area[1],
                    "width":  area[2] - area[0],
                    "height": area[3] - area[1],
                }
                sct_img = self._sct.grab(monitor)

                # ── 2. Hash on native BGRA buffer (zero-copy) ─────────────────
                current_hash = hashlib.md5(sct_img.bgra).hexdigest()

                # ── Click confirmation ─────────────────────────────────────────
                # Runs BEFORE the "unchanged frame" early-return below, since
                # an unchanged frame after a click is exactly the failure case
                # we're checking for (the click didn't land, or landed on the
                # wrong window, and nothing on screen moved).
                if self._pending_confirm_hash is not None:
                    if current_hash != self._pending_confirm_hash:
                        # Screen moved on since the click — good enough
                        # confirmation without needing to know *why* it moved.
                        self._pending_confirm_hash = None
                        self._consecutive_unconfirmed = 0
                    elif time.time() >= self._pending_confirm_deadline:
                        self._pending_confirm_hash = None
                        self._consecutive_unconfirmed += 1
                        print(f"[GUI] [WARN] Click unconfirmed — screen unchanged "
                              f"after click ({self._consecutive_unconfirmed}/"
                              f"{self.UNCONFIRMED_THRESHOLD})")
                        if self._consecutive_unconfirmed >= self.UNCONFIRMED_THRESHOLD:
                            self._set_automation_enabled(
                                False,
                                f"⚠ {self.UNCONFIRMED_THRESHOLD} unconfirmed clicks — automation paused")
                            self._consecutive_unconfirmed = 0

                if current_hash == self.last_frame_hash:
                    self.root.after(self.core.current_polling, self._main_loop)
                    return
                self.last_frame_hash = current_hash

                # ── 3. Frame answer cache — skip OCR on known frames ──────────
                if current_hash in self.frame_answer_cache:
                    cached_answer, cached_source = self.frame_answer_cache[current_hash]
                    if cached_answer is not None and self.core.last_question == "":
                        print(f"[GUI] [FRAME CACHE] {cached_answer}")
                        self.core.click_answer(cached_answer, cached_source)
                        if self.core.automation_enabled:
                            self._pending_confirm_hash     = current_hash
                            self._pending_confirm_deadline = time.time() + self.CONFIRM_TIMEOUT
                        if self.core._last_question_reset_id is not None:
                            try:
                                self.root.after_cancel(
                                    self.core._last_question_reset_id)
                            except Exception:
                                pass
                        self.core._last_question_reset_id = self.root.after(
                            50, self._reset_last_question)
                    self.root.after(self.core.current_polling, self._main_loop)
                    return

                # ── 4. Zero-copy numpy → OpenCV → EasyOCR (new frame) ─────────
                raw_np = np.array(sct_img)
                arr    = self.core.preprocess_for_ocr(raw_np)

                result = self.core.reader.readtext(
                    arr,
                    allowlist='0123456789+-*/()=?xX×÷: ',
                    low_text=0.3, batch_size=1, paragraph=False, min_size=5
                )

                answer, source = None, None
                if result:
                    raw = " ".join(t for _, t, _ in result)
                    if raw != self.core.last_question:
                        self.core.last_question = raw
                        answer, source = self.core.handle_question(raw)
                        self._update_detected_display(raw, answer)
                        if answer is not None:
                            self.core.click_answer(answer, source)
                            if self.core.automation_enabled:
                                self._pending_confirm_hash     = current_hash
                                self._pending_confirm_deadline = time.time() + self.CONFIRM_TIMEOUT
                            if self.core._last_question_reset_id is not None:
                                try:
                                    self.root.after_cancel(
                                        self.core._last_question_reset_id)
                                except Exception:
                                    pass
                            self.core._last_question_reset_id = self.root.after(
                                50, self._reset_last_question)

                # Cache successful results only. A failed OCR/solve attempt
                # used to be cached as (None, None) too — meant to save a
                # wasted OCR pass on a repeated animation frame, but it also
                # meant one bad frame (blur, glare, a half-drawn digit) could
                # get its failure "stuck": if those exact pixels reappeared
                # later, OCR would be skipped and the earlier failure reused
                # instead of trying again. A skipped OCR pass is cheap; a
                # permanently unsolvable question is not.
                if answer is not None:
                    # Evict the oldest entry instead of refusing new ones once
                    # full — dict preserves insertion order, so this is a
                    # simple FIFO/LRU-ish cap. Previously the cache just
                    # stopped accepting new frames forever once it hit 500.
                    if len(self.frame_answer_cache) >= 500:
                        oldest = next(iter(self.frame_answer_cache))
                        del self.frame_answer_cache[oldest]
                    self.frame_answer_cache[current_hash] = (answer, source)

                # ── 5. UI preview — PIL only if enabled ───────────────────────
                if self.core.preview_enabled:
                    self.core.preview_loop_counter += 1
                    if self.core.preview_loop_counter >= PREVIEW_UPDATE_INTERVAL:
                        self.core.preview_loop_counter = 0
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
            print(f"[GUI] Loop error: {e}")

        # Reschedule with crash recovery
        try:
            self.root.after(self.core.current_polling, self._main_loop)
        except Exception as e:
            print(f"[GUI] FATAL: loop reschedule failed: {e} — retry in 500ms")
            try:
                self.root.after(500, self._main_loop)
            except Exception:
                pass

    def _reset_last_question(self):
        self.core.last_question           = ""
        self.core._last_question_reset_id = None

    # ─────────────────────────────────────────────────────────────────────────
    # Entry point
    # ─────────────────────────────────────────────────────────────────────────

    def run(self):
        print("[GUI] Starting — FAST mode, 10 ms polling")
        self.root.after(200, self._main_loop)
        self.root.mainloop()


if __name__ == "__main__":
    OpticalReaderSolverGUI().run()
