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

# ── Colour palette ────────────────────────────────────────────────────────────
C_BG       = "#1e1e2e"   # dark background
C_SURFACE  = "#2a2a3e"   # card/panel
C_BORDER   = "#44446a"   # subtle border
C_ACCENT   = "#7c6af7"   # purple accent
C_GREEN    = "#50fa7b"
C_RED      = "#ff5555"
C_ORANGE   = "#ffb86c"
C_CYAN     = "#8be9fd"
C_FG       = "#cdd6f4"   # foreground text
C_MUTED    = "#6c7086"   # muted labels


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

        # High-speed screen capture — mss is 3-5× faster than PIL screen capture
        self._sct             = mss.mss()
        # MD5 of the last captured frame — if unchanged, skip OCR entirely
        self.last_frame_hash  = None
        # Maps frame hash → (answer, source).  After the first OCR pass for any
        # question, subsequent appearances skip EasyOCR entirely and answer in
        # microseconds.  Transition frames are cached as (None, None) so OCR is
        # never wasted on them either.  Capped at 500 entries.
        self.frame_answer_cache = {}

        # ── Click confirmation ──────────────────────────────────────────────
        # We already hash every frame anyway (for the frame-answer-cache) —
        # reuse that to check whether the screen actually changed after a
        # click, instead of assuming click_answer() worked just because it
        # didn't raise.
        self._pending_confirm_hash     = None  # frame hash right before the click
        self._pending_confirm_deadline = None  # time.time() by which we expect a change
        self._consecutive_unconfirmed  = 0
        self.CONFIRM_TIMEOUT      = 0.5  # seconds to wait for the screen to change
        self.UNCONFIRMED_THRESHOLD = 3   # auto-disable automation after this many in a row

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
        self.root.geometry("310x540+50+50")
        self.root.resizable(False, False)
        self.root.attributes("-topmost", True)
        self.root.configure(bg=C_BG)

        # Custom ttk style for the solver-mode radio buttons
        style = ttk.Style(self.root)
        style.theme_use("clam")
        for name, bg in [("Hybrid.TButton", C_ACCENT),
                          ("Calc.TButton",   "#44b5a0"),
                          ("Lut.TButton",    "#e07b54"),
                          ("Inactive.TButton", C_SURFACE)]:
            style.configure(name, background=bg, foreground="white",
                            font=("Segoe UI", 8, "bold"), padding=3,
                            relief="flat", borderwidth=0)
            style.map(name, background=[("active", bg)])

    # ─────────────────────────────────────────────────────────────────────────
    # GUI layout
    # ─────────────────────────────────────────────────────────────────────────

    def _build_gui(self):
        root = self.root

        # ── Status card ──────────────────────────────────────────────────────
        card = tk.Frame(root, bg=C_SURFACE, padx=10, pady=6)
        card.pack(fill="x", padx=10, pady=(10, 4))

        # Row 0: status badge + mode label
        self.status_label = tk.Label(card, text="● RUNNING",
                                     fg=C_GREEN, bg=C_SURFACE,
                                     font=("Segoe UI", 11, "bold"))
        self.status_label.grid(row=0, column=0, sticky="w")

        self.mode_label = tk.Label(card, text="FAST · 10 ms",
                                   fg=C_ACCENT, bg=C_SURFACE,
                                   font=("Segoe UI", 9))
        self.mode_label.grid(row=0, column=1, sticky="e", padx=(10, 0))
        card.columnconfigure(1, weight=1)

        # Row 1: counters
        self.counter_label = tk.Label(card, text="Answers: 0/10  Ready: 0",
                                      fg=C_ORANGE, bg=C_SURFACE,
                                      font=("Segoe UI", 8))
        self.counter_label.grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))

        # Row 2: cache + LUT on one line
        stats_row = tk.Frame(card, bg=C_SURFACE)
        stats_row.grid(row=2, column=0, columnspan=2, sticky="w", pady=(2, 0))
        self.cache_label = tk.Label(stats_row,
                                    text=f"Cache: {len(self.core.answer_cache)}",
                                    fg="#bd93f9", bg=C_SURFACE, font=("Segoe UI", 8))
        self.cache_label.pack(side="left")
        tk.Label(stats_row, text="  ", bg=C_SURFACE).pack(side="left")
        self.lut_label = tk.Label(stats_row,
                                  text=f"LUT: {len(self.core.lut)} entries",
                                  fg=C_GREEN, bg=C_SURFACE, font=("Segoe UI", 8, "bold"))
        self.lut_label.pack(side="left")

        # Row 3: automation status
        self.auto_status_label = tk.Label(card, text="", fg=C_MUTED,
                                          bg=C_SURFACE, font=("Segoe UI", 7))
        self.auto_status_label.grid(row=3, column=0, columnspan=2, sticky="w")

        # ── OCR preview ──────────────────────────────────────────────────────
        prev_card = tk.Frame(root, bg=C_SURFACE, padx=6, pady=4)
        prev_card.pack(fill="x", padx=10, pady=4)

        hdr = tk.Frame(prev_card, bg=C_SURFACE)
        hdr.pack(fill="x")
        tk.Label(hdr, text="OCR Preview", fg=C_MUTED, bg=C_SURFACE,
                 font=("Segoe UI", 8)).pack(side="left")
        self.preview_toggle_btn = tk.Button(hdr, text="ON", fg=C_GREEN, bg=C_SURFACE,
                                            bd=0, font=("Segoe UI", 8, "bold"),
                                            command=self._toggle_preview,
                                            cursor="hand2")
        self.preview_toggle_btn.pack(side="right")

        self.preview_canvas = tk.Canvas(prev_card, width=286, height=60,
                                        bg="#111122", highlightthickness=1,
                                        highlightbackground=C_BORDER)
        self.preview_canvas.pack(pady=(4, 0))
        self.preview_img_tk = None

        # ── Solver-mode pills ─────────────────────────────────────────────────
        mode_card = tk.Frame(root, bg=C_SURFACE, padx=8, pady=6)
        mode_card.pack(fill="x", padx=10, pady=4)
        tk.Label(mode_card, text="Solver Mode", fg=C_MUTED, bg=C_SURFACE,
                 font=("Segoe UI", 7, "bold")).pack(anchor="w")

        pills = tk.Frame(mode_card, bg=C_SURFACE)
        pills.pack(fill="x", pady=(4, 0))

        self._mode_pills = {}
        defs = [
            (MODE_HYBRID,   "⚡ Hybrid",    "Hybrid.TButton"),
            (MODE_CALC,     "🔢 Calc Only", "Calc.TButton"),
            (MODE_LUT_ONLY, "📖 LUT Only",  "Lut.TButton"),
        ]
        for i, (mode, label, style_name) in enumerate(defs):
            btn = ttk.Button(pills, text=label, style=style_name,
                             command=lambda m=mode: self._set_solver_mode(m))
            btn.grid(row=0, column=i, padx=3, sticky="ew")
            self._mode_pills[mode] = (btn, style_name)
            pills.columnconfigure(i, weight=1)

        self.lut_warn_label = tk.Label(mode_card,
                                       text="⚠ Will skip unknown questions",
                                       fg=C_ORANGE, bg=C_SURFACE,
                                       font=("Segoe UI", 7))
        # (shown/hidden by _set_solver_mode)

        self._set_solver_mode(MODE_HYBRID)  # set default highlight

        # ── Control buttons ───────────────────────────────────────────────────
        btn_card = tk.Frame(root, bg=C_SURFACE, padx=8, pady=6)
        btn_card.pack(fill="x", padx=10, pady=4)

        def _btn(parent, text, cmd, fg=C_FG, bg=C_BORDER, **kw):
            b = tk.Button(parent, text=text, command=cmd, fg=fg, bg=bg,
                          activeforeground=C_FG, activebackground=C_ACCENT,
                          relief="flat", bd=0, cursor="hand2",
                          font=("Segoe UI", 8), padx=6, pady=4, **kw)
            return b

        # Row 0: Pause / Mode / Overlays / Reset
        self.pause_btn  = _btn(btn_card, "⏸ Pause",    self._toggle_pause)
        self.mode_btn   = _btn(btn_card, "🔄 Mode",     self._toggle_game_mode)
        overlays_btn    = _btn(btn_card, "👁 Overlays", self._toggle_overlays)
        reset_btn       = _btn(btn_card, "↺ Reset",     self._reset_counter)

        for col, b in enumerate([self.pause_btn, self.mode_btn,
                                  overlays_btn, reset_btn]):
            b.grid(row=0, column=col, padx=2, pady=2, sticky="ew")
            btn_card.columnconfigure(col, weight=1)

        # Row 1: Edit / Reset Default / Saves / Clear LUT
        self.edit_btn = _btn(btn_card, "✏ Edit",   self._toggle_edit_mode,
                             bg="#3a3a50")
        reset_def_btn = _btn(btn_card, "⟳ Default", self._reset_to_defaults,
                             bg="#4a2a2a", fg=C_RED)
        saves_btn     = _btn(btn_card, "💾 Saves",   self._open_saves_modal,
                             bg="#2a3a2a", fg=C_GREEN)
        clear_lut_btn = _btn(btn_card, "🗑 LUT",    self.core.clear_lut,
                             bg="#4a2a2a", fg=C_ORANGE)

        for col, b in enumerate([self.edit_btn, reset_def_btn,
                                  saves_btn, clear_lut_btn]):
            b.grid(row=1, column=col, padx=2, pady=2, sticky="ew")

        # Row 2: Automation toggle (full width so it's obvious)
        self.auto_btn = _btn(btn_card, "🤖 Automation: ON",
                             self._toggle_automation,
                             fg=C_GREEN, bg="#1e3a1e")
        self.auto_btn.grid(row=2, column=0, columnspan=4, padx=2, pady=(4, 2), sticky="ew")

    # ─────────────────────────────────────────────────────────────────────────
    # UI update callbacks (called from BotCore)
    # ─────────────────────────────────────────────────────────────────────────

    def update_cache_label(self, count):
        self.cache_label.config(text=f"Cache: {count}")

    def update_lut_label(self, count):
        self.lut_label.config(text=f"LUT: {count} entries")

    def update_counter_label(self, answers, ready):
        self.counter_label.config(text=f"Answers: {answers}/10  Ready: {ready}")

    def set_auto_status(self, text, color="gray"):
        self.auto_status_label.config(text=text, fg=color)

    def sync_pause_state(self):
        """
        Mirror core.paused into the GUI widgets.
        Called via root.after(0, ...) from the hotkey listener thread
        so it always runs on the tkinter main thread.
        """
        if self.core.paused:
            self.status_label.config(text="● PAUSED", fg=C_RED)
            self.pause_btn.config(text="▶ Resume")
        else:
            self.status_label.config(text="● RUNNING", fg=C_GREEN)
            self.pause_btn.config(text="⏸ Pause")

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
            btn.configure(style=active_style if m == mode else "Inactive.TButton")
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
            self.status_label.config(text="● PAUSED", fg=C_RED)
            self.pause_btn.config(text="▶ Resume")
        else:
            self.status_label.config(text="● RUNNING", fg=C_GREEN)
            self.pause_btn.config(text="⏸ Pause")
            self.core._capture_target_window()
            # If user resumes while save-pending, cancel that state
            if self._edit_save_pending:
                self._edit_save_pending = False
        print(f"[GUI] Bot {'PAUSED' if self.core.paused else 'RUNNING'}")

    def _toggle_game_mode(self):
        self.core.cancel_all_scheduled_events()
        self.core.fast_mode = not self.core.fast_mode
        self.core.current_polling = (FAST_MODE_POLLING if self.core.fast_mode
                                     else STANDARD_MODE_POLLING)
        label = "FAST · 10 ms" if self.core.fast_mode else "STANDARD · 150 ms"
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
            self.auto_btn.config(text="🤖 Automation: ON",
                                 fg=C_GREEN, bg="#1e3a1e")
            print("[GUI] Automation ENABLED")
        else:
            # Cancel any in-progress sequences immediately
            self.core.cancel_all_scheduled_events()
            self.core.extended_sequence_active = False
            self.auto_btn.config(text="🤖 Automation: OFF",
                                 fg=C_MUTED, bg="#2a2a2a")
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
            self.preview_toggle_btn.config(text="ON", fg=C_GREEN)
        else:
            self.preview_toggle_btn.config(text="OFF", fg=C_MUTED)
            self.preview_canvas.delete("all")
            self.preview_canvas.create_text(143, 30, text="Preview OFF",
                                            fill=C_MUTED, font=("Segoe UI", 9))

    # ─────────────────────────────────────────────────────────────────────────
    # Saves modal (replaces the old inline slot buttons)
    # ─────────────────────────────────────────────────────────────────────────

    def _open_saves_modal(self, save_mode=False):
        """
        Open a clean Toplevel window showing the 3 coord slots.
        save_mode=True: buttons say "Save here"; False: buttons say "Load".
        """
        modal = tk.Toplevel(self.root)
        modal.title("Coordinate Saves")
        modal.geometry("300x220")
        modal.resizable(False, False)
        modal.configure(bg=C_BG)
        modal.attributes("-topmost", True)
        modal.grab_set()   # modal behaviour

        tk.Label(modal, text="Coord Slots",
                 fg=C_FG, bg=C_BG, font=("Segoe UI", 11, "bold")).pack(pady=(12, 4))

        hint = ("Click a slot to SAVE your new layout." if save_mode
                else "Click a slot to LOAD saved coordinates.")
        self.modal_hint = tk.Label(modal, text=hint,
                                   fg=C_MUTED, bg=C_BG, font=("Segoe UI", 8))
        self.modal_hint.pack(pady=(0, 8))

        slots_frame = tk.Frame(modal, bg=C_BG)
        slots_frame.pack(fill="x", padx=12)

        slots = self.core.read_slots()
        for i in range(MAX_SLOTS):
            slot     = slots[i]
            is_active = (i == self.active_slot)
            row_bg   = "#aaddaa" if is_active else C_SURFACE

            row = tk.Frame(slots_frame, bg=row_bg, pady=4, padx=8)
            row.pack(fill="x", pady=2)

            label_text = (f"Slot {i+1}  —  {slot['saved']}" if slot
                          else f"Slot {i+1}  —  (empty)")
            tk.Label(row, text=label_text, fg=C_FG, bg=row_bg,
                     font=("Segoe UI", 9)).pack(side="left")

            if save_mode:
                btn_text = "💾 Save here"
                btn_cmd  = lambda idx=i, m=modal: self._save_to_slot(idx, m)
                btn_col  = C_ORANGE
            elif slot:
                btn_text = "↩ Load"
                btn_cmd  = lambda idx=i, m=modal: self._load_slot(idx, m)
                btn_col  = C_CYAN
            else:
                btn_text = "(empty)"
                btn_cmd  = None
                btn_col  = C_MUTED

            tk.Button(row, text=btn_text, fg=btn_col, bg=row_bg,
                      bd=0, relief="flat", cursor="hand2",
                      font=("Segoe UI", 8, "bold"),
                      state="normal" if btn_cmd else "disabled",
                      command=btn_cmd).pack(side="right")

        tk.Button(modal, text="Close", command=modal.destroy,
                  fg=C_MUTED, bg=C_SURFACE, relief="flat", bd=0,
                  font=("Segoe UI", 8), cursor="hand2").pack(pady=(10, 6))

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
            self.status_label.config(text="● PAUSED", fg=C_RED)
            self.pause_btn.config(text="▶ Resume")
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
        self.status_label.config(text="● PAUSED", fg=C_RED)
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
        self.edit_btn.config(text="✔ Done", bg=C_ORANGE, fg="black")
        self.status_label.config(text="● EDITING", fg=C_ORANGE)

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
        self.edit_btn.config(text="✏ Edit", bg="#3a3a50", fg=C_FG)

        for w in self.overlay_windows:
            cv = w.winfo_children()[0]
            cv.unbind("<ButtonPress-1>")
            cv.unbind("<B1-Motion>")
            self._set_click_through(w, True)

        if not self.overlays_visible:
            for w in self.overlay_windows: w.withdraw()
        elif not self.core.fast_mode:
            for w in self.auto_overlays: w.withdraw()

        self.status_label.config(text="● PAUSED", fg=C_RED)
        self.pause_btn.config(text="▶ Resume")
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
        self.edit_btn.config(text="✏ Edit", bg="#3a3a50", fg=C_FG)

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
