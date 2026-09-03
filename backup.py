import easyocr
from PIL import ImageGrab, Image, ImageTk
import pyautogui
import numpy as np
import tkinter as tk
from sympy import symbols, Eq, solve, sympify, N
import re
import time
import sys
import os
import json
import ctypes
import threading

# Disable pyautogui fail-safe — without this, moving the mouse to the top-left
# corner raises FailSafeException which kills the entire process instantly.
pyautogui.FAILSAFE = False

# Path to the coordinate slots save file (same folder as this script)
SLOTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "optical_coords.json")
MAX_SLOTS = 3

# Persistent look-up table — survives restarts, separate from session cache
LUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "optical_lut.json")

# ----------------- DPI & SCALING SETUP -----------------
# Enable DPI awareness for proper coordinate scaling on high-DPI displays
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

# Reference resolution (coordinates were designed for this resolution)
REF_W = 1366
REF_H = 768

# Get current screen resolution
CURR_W, CURR_H = pyautogui.size()
SCALE_X = CURR_W / REF_W
SCALE_Y = CURR_H / REF_H

print(f"Screen: {CURR_W}x{CURR_H} (Scaling: {SCALE_X:.2f}x, {SCALE_Y:.2f}x)")

def s_xy(x, y):
    """Scale a single coordinate pair based on screen resolution"""
    return (int(x * SCALE_X), int(y * SCALE_Y))

def s_bbox(bbox):
    """Scale a bounding box (x1, y1, x2, y2) based on screen resolution"""
    return (int(bbox[0] * SCALE_X), int(bbox[1] * SCALE_Y), 
            int(bbox[2] * SCALE_X), int(bbox[3] * SCALE_Y))

# ----------------- WINDOWS CONSTANTS -----------------
# Windows API constants for transparent, click-through overlays
WS_EX_TRANSPARENT = 0x00000020
WS_EX_LAYERED = 0x00080000
GWL_EXSTYLE = -20

# ----------------- TASKBAR DETECTION -----------------
def get_taskbar_position():
    """
    Detect taskbar position and return offset coordinates.
    Returns: (x_offset, y_offset) - amount to shift coordinates based on taskbar location
    """
    try:
        # Get the work area (screen minus taskbar)
        work_area = ctypes.wintypes.RECT()
        ctypes.windll.user32.SystemParametersInfoW(48, 0, ctypes.byref(work_area), 0)
        
        # Get full screen dimensions
        screen_w = ctypes.windll.user32.GetSystemMetrics(0)
        screen_h = ctypes.windll.user32.GetSystemMetrics(1)
        
        # Calculate taskbar position based on work area vs full screen
        x_offset = work_area.left
        y_offset = work_area.top
        
        # Determine taskbar location
        if work_area.top > 0:
            position = "top"
        elif work_area.left > 0:
            position = "left"
        elif work_area.right < screen_w:
            position = "right"
        elif work_area.bottom < screen_h:
            position = "bottom"
        else:
            position = "hidden"
            
        return (x_offset, y_offset, position)
    except Exception as e:
        print(f"Taskbar detection error: {e}")
        return (0, 0, "bottom")  # Default to bottom taskbar

# Get initial taskbar position
TASKBAR_X_OFFSET, TASKBAR_Y_OFFSET, TASKBAR_POSITION = get_taskbar_position()
print(f"Taskbar detected: {TASKBAR_POSITION} (offset: {TASKBAR_X_OFFSET}, {TASKBAR_Y_OFFSET})")

# ----------------- CONFIG (AUTO-SCALED) -----------------
# Original coordinates (designed for 1366x768 with bottom taskbar)
ORIGINAL_QUESTION_AREA = (411, 182, 722, 231)
ORIGINAL_QUESTION_AREA_FAST = (423, 182, 710, 231)  # 12px smaller on each side for fast mode
ORIGINAL_KEY_COORDS = {
    '0': (486, 613), '1': (464, 537), '2': (533, 534), '3': (606, 530),
    '4': (453, 456), '5': (531, 460), '6': (615, 455),
    '7': (453, 378), '8': (530, 381), '9': (613, 381), 'OK': (689, 576)
}

# Automation click areas
ORIGINAL_AUTO_AREA_1 = (510, 686)
ORIGINAL_AUTO_AREA_2 = (157, 745)
ORIGINAL_AUTO_AREA_3 = (274, 430)

# ── Factory defaults (never mutated — used by Reset to Default) ──────────────
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

# Apply scaling and taskbar offset to all coordinates
def apply_scaling_and_offset(bbox):
    """Apply both screen scaling and taskbar offset to a bounding box"""
    scaled = s_bbox(bbox)
    return (scaled[0] + TASKBAR_X_OFFSET, scaled[1] + TASKBAR_Y_OFFSET,
            scaled[2] + TASKBAR_X_OFFSET, scaled[3] + TASKBAR_Y_OFFSET)

def apply_scaling_and_offset_xy(x, y):
    """Apply both screen scaling and taskbar offset to a coordinate pair"""
    scaled = s_xy(x, y)
    return (scaled[0] + TASKBAR_X_OFFSET, scaled[1] + TASKBAR_Y_OFFSET)

# Calculate final coordinates with scaling and taskbar offset
QUESTION_AREA = apply_scaling_and_offset(ORIGINAL_QUESTION_AREA)
QUESTION_AREA_FAST = apply_scaling_and_offset(ORIGINAL_QUESTION_AREA_FAST)
KEY_COORDS = {k: apply_scaling_and_offset_xy(v[0], v[1]) for k, v in ORIGINAL_KEY_COORDS.items()}
AUTO_AREA_1 = apply_scaling_and_offset_xy(ORIGINAL_AUTO_AREA_1[0], ORIGINAL_AUTO_AREA_1[1])
AUTO_AREA_2 = apply_scaling_and_offset_xy(ORIGINAL_AUTO_AREA_2[0], ORIGINAL_AUTO_AREA_2[1])
AUTO_AREA_3 = apply_scaling_and_offset_xy(ORIGINAL_AUTO_AREA_3[0], ORIGINAL_AUTO_AREA_3[1])

# Mode-specific polling intervals
FAST_MODE_POLLING = 10
STANDARD_MODE_POLLING = 50

KEY_PRESS_DELAY = 0
POST_ANSWER_DELAY = 0

# Taskbar check interval (check every 2 seconds for taskbar movement)
TASKBAR_CHECK_INTERVAL = 2000

# Preview update frequency (update preview every N loops to reduce overhead)
PREVIEW_UPDATE_INTERVAL = 5  # Update preview every 5 loops

# ----------------- MATH SOLVER CLASS -----------------
class OpticalReaderSolver:

    def __init__(self, q_area, key_coords):
        self.question_area = q_area
        self.question_area_fast = QUESTION_AREA_FAST
        self.key_coords = key_coords
        
        # Store current taskbar position for dynamic updates
        self.current_taskbar_position = TASKBAR_POSITION
        
        # Try GPU first, fallback to CPU
        try:
            self.reader = easyocr.Reader(['en'], gpu=True)
            print("✓ EasyOCR using GPU")
        except:
            self.reader = easyocr.Reader(['en'], gpu=False)
            print("✓ EasyOCR using CPU (GPU not available)")
        
        self.last_question = ""
        self._last_question_reset_id = None  # scheduled event to clear last_question
        self.paused = False
        self.overlays_visible = True
        self.preview_enabled = True  # Preview toggle state
        self.preview_loop_counter = 0  # Counter for preview update frequency
        self.fast_mode = True
        self.current_polling = FAST_MODE_POLLING
        
        # Automation counters
        self.answers_count = 0
        self.ready_count = 0
        
        # Persistent look-up table (all modes, survives restarts)
        self.lut = self._load_lut()
        self._lut_dirty = False      # True when an unsaved entry is pending write

        # Pre-warm session cache with every LUT entry so fast mode gets instant
        # hits from the very first question without waiting for a fresh solve
        self.answer_cache = dict(self.lut)
        
        # Flag to prevent auto-clicks during answering
        self.is_answering = False
        
        # Flag to track if extended sequence (ready=3) is active
        self.extended_sequence_active = False
        
        # Store scheduled automation events to cancel them
        self.scheduled_events = []
        
        # Pre-compile regex patterns for speed
        self.operator_clean = re.compile(r'[^\d\s\+\-\*/\(\)=?]')
        self.space_clean = re.compile(r'\s*([\+\-\*/\(\)=])\s*')
        self.div_pattern = re.compile(r"(\d{3})\s+(\d{1,2})\s*=\s*\?")
        self.mult_pattern = re.compile(r"(\d+)\s+(\d+)")
        
        # Cache for sympy symbol
        self.x_symbol = symbols('x')
        
        # Store auto overlay windows separately
        self.auto_overlays = []
        
        # Edit mode state
        self.edit_mode = False
        self._edit_save_pending = False   # True after Done Editing until a slot is chosen
        self._resize_handles = []         # Corner handle windows for OCR box resize
        
        # Slot system
        self.active_slot = None           # Which slot (0-based) is currently loaded
        self.slot_btns = []               # List of 3 slot Button widgets
        
        # Per-overlay references for coordinate saving
        self.key_overlays = {}           # key → Toplevel win
        self.auto_area_1_overlay = None
        self.auto_area_2_overlay = None
        self.auto_area_3_overlay = None

        self.setup_gui()
        self.setup_overlay_boxes()
        
        # Start taskbar monitoring
        self.start_taskbar_monitoring()

    def setup_gui(self):
        """Setup the main control GUI with status display and question preview"""
        self.root = tk.Tk()
        self.root.title("Optical Reader & Solver")
        self.root.geometry("320x600+50+50")  # Larger to accommodate preview + slots
        self.root.resizable(False, False)
        self.root.attributes("-topmost", True)

        # === STATUS SECTION ===
        status_label = tk.Label(self.root, text="Status:", font=("Arial", 10, "bold"))
        status_label.pack(pady=5)
        
        self.status_text = tk.Label(self.root, text="RUNNING", fg="green", font=("Arial", 12, "bold"))
        self.status_text.pack(pady=2)
        
        self.mode_text = tk.Label(self.root, text="Mode: FAST (10ms)", fg="blue", font=("Arial", 9, "bold"))
        self.mode_text.pack(pady=2)
        
        # Automation counter display
        self.counter_text = tk.Label(self.root, text="Answers: 0/10 | Ready: 0", fg="orange", font=("Arial", 9, "bold"))
        self.counter_text.pack(pady=2)
        
        self.cache_text = tk.Label(self.root, text=f"Cache: {len(self.answer_cache)} questions", fg="purple", font=("Arial", 8))
        self.cache_text.pack(pady=2)

        # LUT status — persistent across restarts
        lut_frame = tk.Frame(self.root)
        lut_frame.pack(pady=1)
        self.lut_text = tk.Label(lut_frame, text=f"LUT: {len(self.lut)} entries",
                                 fg="#007700", font=("Arial", 8, "bold"))
        self.lut_text.grid(row=0, column=0, padx=4)
        clear_lut_btn = tk.Button(lut_frame, text="Clear LUT", font=("Arial", 7),
                                  bg="#ffe0e0", command=self._clear_lut)
        clear_lut_btn.grid(row=0, column=1, padx=4)
        
        # Automation status indicator
        self.auto_status_text = tk.Label(self.root, text="", fg="gray", font=("Arial", 7))
        self.auto_status_text.pack(pady=1)
        
        # === QUESTION PREVIEW SECTION ===
        preview_label = tk.Label(self.root, text="Question Preview:", font=("Arial", 9, "bold"))
        preview_label.pack(pady=(10, 2))
        
        # Canvas for displaying captured question image
        self.preview_canvas = tk.Canvas(self.root, width=300, height=90, bg="gray")
        self.preview_canvas.pack(pady=5)
        self.preview_img_tk = None
        
        # === CONTROL BUTTONS ===
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=5)
        
        # Control buttons (3 rows, 2 columns)
        pause_btn = tk.Button(btn_frame, text="Pause", command=self.toggle_pause, width=8)
        pause_btn.grid(row=0, column=0, padx=2)
        self.pause_btn = pause_btn
        
        overlay_btn = tk.Button(btn_frame, text="Overlays", command=self.toggle_overlays, width=8)
        overlay_btn.grid(row=0, column=1, padx=2)
        
        mode_btn = tk.Button(btn_frame, text="Mode", command=self.toggle_mode, width=8)
        mode_btn.grid(row=1, column=0, padx=2, pady=2)
        
        reset_btn = tk.Button(btn_frame, text="Reset", command=self.reset_counter, width=8)
        reset_btn.grid(row=1, column=1, padx=2, pady=2)
        
        # NEW: Preview toggle button
        preview_btn = tk.Button(btn_frame, text="Preview", command=self.toggle_preview, width=8, bg="lightgreen")
        preview_btn.grid(row=2, column=0, columnspan=2, padx=2, pady=2)
        self.preview_btn = preview_btn
        
        # Edit mode button + Reset Default on the same row
        edit_btn = tk.Button(btn_frame, text="Edit", command=self.toggle_edit_mode,
                             width=9, bg="lightyellow", font=("Arial", 9, "bold"))
        edit_btn.grid(row=3, column=0, padx=2, pady=4)
        self.edit_btn = edit_btn

        reset_default_btn = tk.Button(btn_frame, text="Reset Default",
                                      command=self.reset_to_defaults,
                                      width=9, bg="#ffcccc", font=("Arial", 9, "bold"))
        reset_default_btn.grid(row=3, column=1, padx=2, pady=4)

        # === COORD SLOTS SECTION ===
        sep = tk.Frame(self.root, height=1, bg="gray")
        sep.pack(fill="x", padx=10, pady=(6, 0))

        slots_label = tk.Label(self.root, text="Coord Slots", font=("Arial", 9, "bold"), fg="#444")
        slots_label.pack(pady=(4, 2))

        self.slot_hint = tk.Label(self.root, text="Click a slot to load saved coords",
                                  font=("Arial", 7), fg="gray")
        self.slot_hint.pack(pady=(0, 4))

        slots_frame = tk.Frame(self.root)
        slots_frame.pack()
        self.slot_btns = []
        for i in range(MAX_SLOTS):
            btn = tk.Button(
                slots_frame, text=f"Slot {i+1}\n(empty)",
                width=8, height=2, font=("Arial", 7),
                bg="#ddd", fg="gray",
                command=lambda n=i: self._slot_button_clicked(n)
            )
            btn.grid(row=0, column=i, padx=4)
            self.slot_btns.append(btn)

        # Populate slot buttons from any existing save file
        self._refresh_slot_buttons()

    def toggle_pause(self):
        """Toggle pause state - stops/resumes question detection and solving"""
        self.paused = not self.paused
        status = "PAUSED" if self.paused else "RUNNING"
        color = "red" if self.paused else "green"
        self.status_text.config(text=status, fg=color)
        self.pause_btn.config(text="Resume" if self.paused else "Pause")
        # If user resumes while save was pending, cancel save mode cleanly
        if not self.paused and self._edit_save_pending:
            self._edit_save_pending = False
            self.slot_hint.config(text="Click a slot to load saved coords", fg="gray")
            self._refresh_slot_buttons()
        print(f"Bot {status}")
    
    def toggle_mode(self):
        """Switch between FAST mode (fast) and STANDARD mode (slow)"""
        # Cancel any pending automation events
        self.cancel_all_scheduled_events()
        
        self.fast_mode = not self.fast_mode
        self.current_polling = FAST_MODE_POLLING if self.fast_mode else STANDARD_MODE_POLLING
        mode_name = "FAST (10ms)" if self.fast_mode else "STANDARD (50ms)"
        self.mode_text.config(text=f"Mode: {mode_name}")
        
        # Clear cache when switching modes
        self.answer_cache = {}
        self.cache_text.config(text="Cache: 0 questions")
        
        # Clear automation status and reset sequence flag
        self.auto_status_text.config(text="")
        self.extended_sequence_active = False
        
        # Update question area overlay size based on mode
        if self.overlays_visible:
            # Destroy old question overlay
            self.question_overlay.destroy()
            self.overlay_windows.remove(self.question_overlay)
            
            # Create new question overlay with correct size
            x1, y1, x2, y2 = self.question_area_fast if self.fast_mode else self.question_area
            self.question_overlay = self.create_overlay_box(
                x1, y1, x2 - x1, y2 - y1,
                "red", "", is_auto=False
            )
            
            # Show/hide auto overlays based on mode
            for win in self.auto_overlays:
                if self.fast_mode:
                    win.deiconify()
                else:
                    win.withdraw()
        
        print(f"Switched to {mode_name} mode - Cache cleared, automation cancelled")
    
    def reset_counter(self):
        """Reset all counters, cache, and automation sequences"""
        # Cancel any pending automation events
        self.cancel_all_scheduled_events()
        
        self.answers_count = 0
        self.ready_count = 0
        self.answer_cache = {}
        self.extended_sequence_active = False  # Reset extended sequence flag
        self.update_counter_display()
        self.cache_text.config(text="Cache: 0 questions")
        self.auto_status_text.config(text="")
        print("[RESET] Counter, cache, automation, and sequence flags reset to 0")

    def setup_overlay_boxes(self):
        """Create transparent overlay boxes showing detection areas on screen"""
        self.overlay_windows = []

        # Question area box - will be updated based on mode
        x1, y1, x2, y2 = self.question_area_fast if self.fast_mode else self.question_area
        self.question_overlay = self.create_overlay_box(
            x1, y1, x2 - x1, y2 - y1,
            "red", "", is_auto=False
        )

        # Keypad button boxes
        box_size = int(50 * SCALE_X)
        for key, (x, y) in self.key_coords.items():
            win = self.create_overlay_box(
                x - (box_size // 2), y - (box_size // 2), box_size, box_size,
                "cyan", key, is_auto=False
            )
            self.key_overlays[key] = win
        
        # Automation area highlights (only visible in fast mode)
        auto_box_size = int(60 * SCALE_X)
        self.auto_area_1_overlay = self.create_overlay_box(
            AUTO_AREA_1[0] - (auto_box_size // 2), AUTO_AREA_1[1] - (auto_box_size // 2),
            auto_box_size, auto_box_size,
            "yellow", "AUTO 1", is_auto=True
        )
        self.auto_area_2_overlay = self.create_overlay_box(
            AUTO_AREA_2[0] - (auto_box_size // 2), AUTO_AREA_2[1] - (auto_box_size // 2),
            auto_box_size, auto_box_size,
            "magenta", "AUTO 2", is_auto=True
        )
        self.auto_area_3_overlay = self.create_overlay_box(
            AUTO_AREA_3[0] - (auto_box_size // 2), AUTO_AREA_3[1] - (auto_box_size // 2),
            auto_box_size, auto_box_size,
            "lime", "AUTO 3", is_auto=True
        )

    def create_overlay_box(self, x, y, w, h, color, label, is_auto=False):
        """
        Create a transparent, click-through overlay box at specified position
        Args:
            x, y: Position of overlay
            w, h: Width and height
            color: Outline color
            label: Text to display (empty string for no label)
            is_auto: Whether this is an automation area (affects visibility in different modes)
        """
        win = tk.Toplevel(self.root)
        win.overrideredirect(True)
        win.attributes("-topmost", True)
        win.geometry(f"{w}x{h}+{x}+{y}")
        # Track dragged position ourselves (winfo_x/y unreliable on Windows DPI)
        win._edit_x = x
        win._edit_y = y
        win._edit_w = w
        win._edit_h = h

        transparent_bg = "black"
        win.config(bg=transparent_bg)
        win.attributes("-transparentcolor", transparent_bg)

        canvas = tk.Canvas(win, width=w, height=h, bg=transparent_bg, highlightthickness=0)
        canvas.pack()
        
        canvas.create_rectangle(
            2, 2, w - 2, h - 2,
            outline=color,
            width=3
        )
        
        if label:
            canvas.create_text(
                w // 2, 10, 
                text=label,
                fill=color,
                font=("Arial", 10, "bold"),
                anchor="n"
            )

        # Make window click-through using Windows API
        hwnd = ctypes.windll.user32.GetParent(win.winfo_id())
        styles = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        new_style = styles | WS_EX_LAYERED | WS_EX_TRANSPARENT
        ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new_style)

        self.overlay_windows.append(win)
        
        if is_auto:
            self.auto_overlays.append(win)
        
        return win

    def toggle_overlays(self):
        """Show/hide all overlay boxes"""
        self.overlays_visible = not self.overlays_visible
        
        for win in self.overlay_windows:
            if self.overlays_visible:
                # Show overlay if it's not an auto overlay, or if it's auto and in fast mode
                if win not in self.auto_overlays or self.fast_mode:
                    win.deiconify()
            else:
                # Hide all overlays when overlays are turned off
                win.withdraw()
        
        print(f"Overlays {'SHOWN' if self.overlays_visible else 'HIDDEN'}")
    
    def toggle_preview(self):
        """
        Toggle question preview on/off for performance
        When disabled, saves 2-5ms per loop by skipping image resize and display
        """
        self.preview_enabled = not self.preview_enabled
        
        if self.preview_enabled:
            self.preview_btn.config(bg="lightgreen")
            print(f"Preview ENABLED (updates every {PREVIEW_UPDATE_INTERVAL} loops)")
        else:
            self.preview_btn.config(bg="lightgray")
            # Clear preview and show disabled message
            self.preview_canvas.delete("all")
            self.preview_canvas.create_text(
                150, 45,
                text="Preview Disabled\n(for max speed)",
                fill="white",
                font=("Arial", 10, "bold")
            )
            print("Preview DISABLED (max performance mode)")

    # ─────────────── EDIT MODE ───────────────

    def toggle_edit_mode(self):
        """Enter or exit overlay edit mode"""
        if not self.edit_mode:
            self._enter_edit_mode()
        else:
            self._exit_edit_mode()

    def _enter_edit_mode(self):
        """Pause the bot and make all overlays draggable for repositioning"""
        if not self.paused:
            self.toggle_pause()

        self.edit_mode = True
        self._edit_save_pending = False
        self.edit_btn.config(text="Done Editing", bg="orange")
        self.status_text.config(text="EDIT MODE", fg="darkorange")

        # Show all overlays while editing
        for win in self.overlay_windows:
            win.deiconify()

        # Remove click-through and bind drag to every overlay
        for win in self.overlay_windows:
            self._remove_click_through(win)
            self._make_draggable(win)

        # Add resize corner handles to the OCR box only
        self._attach_resize_handles()

        # Slot buttons switch to SAVE mode
        self.slot_hint.config(text="Drag/resize OCR box, then click a slot to SAVE", fg="darkorange")
        self._refresh_slot_buttons(save_mode=True)

        print("[EDIT] Edit mode ON  - drag overlays, resize red OCR box corners, then Done Editing")

    def _exit_edit_mode(self):
        """Apply dragged/resized positions live, prompt user to pick a save slot"""
        # Destroy resize handles before reading final OCR box size
        self._remove_resize_handles()

        # Apply coordinates to live runtime variables
        self._apply_overlay_positions()

        self.edit_mode = False
        self._edit_save_pending = True
        self.edit_btn.config(text="Edit", bg="lightyellow")

        # Remove drag bindings and restore click-through
        for win in self.overlay_windows:
            canvas = win.winfo_children()[0]
            canvas.unbind("<ButtonPress-1>")
            canvas.unbind("<B1-Motion>")
            self._restore_click_through(win)

        # Restore correct visibility state
        if not self.overlays_visible:
            for win in self.overlay_windows:
                win.withdraw()
        elif not self.fast_mode:
            for win in self.auto_overlays:
                win.withdraw()

        # Keep paused
        self.status_text.config(text="PAUSED", fg="red")
        self.pause_btn.config(text="Resume")

        self.slot_hint.config(text="Click a slot to SAVE  |  or Resume to skip", fg="darkorange")
        self._refresh_slot_buttons(save_mode=True)

        print("[EDIT] Coords applied live. Click a slot to save, or Resume to continue without saving.")

    # ── Resize handles (OCR box only) ─────────────────────────────────────────

    def _attach_resize_handles(self):
        """Create 4 solid corner handles on the OCR (question) overlay for resizing."""
        HANDLE = 12
        qw = self.question_overlay

        for corner in ("tl", "tr", "bl", "br"):
            h = tk.Toplevel(self.root)
            h.overrideredirect(True)
            h.attributes("-topmost", True)
            h._corner = corner
            h._handle_size = HANDLE

            hx, hy = self._handle_pos(qw, corner, HANDLE)
            h.geometry(f"{HANDLE}x{HANDLE}+{hx}+{hy}")

            cv = tk.Canvas(h, width=HANDLE, height=HANDLE, bg="red",
                           highlightthickness=0, cursor="sizing")
            cv.pack()
            cv.create_rectangle(1, 1, HANDLE-1, HANDLE-1,
                                 fill="red", outline="white", width=1)
            cv._drag_sx = 0
            cv._drag_sy = 0

            def on_press(event, c=cv):
                c._drag_sx = event.x_root
                c._drag_sy = event.y_root

            def on_drag(event, c=cv, hwin=h, qwin=qw):
                dx = event.x_root - c._drag_sx
                dy = event.y_root - c._drag_sy
                c._drag_sx = event.x_root
                c._drag_sy = event.y_root
                self._resize_ocr_box(qwin, hwin._corner, dx, dy)
                for rh in self._resize_handles:
                    rx, ry = self._handle_pos(qwin, rh._corner, rh._handle_size)
                    rh.geometry(f"+{rx}+{ry}")

            cv.bind("<ButtonPress-1>", on_press)
            cv.bind("<B1-Motion>", on_drag)
            self._resize_handles.append(h)

    def _handle_pos(self, qwin, corner, size):
        """Return (x, y) screen position for a handle corner on qwin."""
        x, y, w, h = qwin._edit_x, qwin._edit_y, qwin._edit_w, qwin._edit_h
        half = size // 2
        return {
            "tl": (x - half,     y - half),
            "tr": (x + w - half, y - half),
            "bl": (x - half,     y + h - half),
            "br": (x + w - half, y + h - half),
        }[corner]

    def _resize_ocr_box(self, qwin, corner, dx, dy):
        """Apply dx/dy delta from dragging a corner handle to the OCR overlay."""
        MIN_W, MIN_H = 40, 15
        x, y, w, h = qwin._edit_x, qwin._edit_y, qwin._edit_w, qwin._edit_h

        if corner == "br":
            w = max(MIN_W, w + dx)
            h = max(MIN_H, h + dy)
        elif corner == "bl":
            new_x = x + dx;  new_w = max(MIN_W, w - dx)
            if new_w > MIN_W: x = new_x
            w = new_w;  h = max(MIN_H, h + dy)
        elif corner == "tr":
            new_y = y + dy;  new_h = max(MIN_H, h - dy)
            if new_h > MIN_H: y = new_y
            h = new_h;  w = max(MIN_W, w + dx)
        elif corner == "tl":
            new_x = x + dx;  new_w = max(MIN_W, w - dx)
            new_y = y + dy;  new_h = max(MIN_H, h - dy)
            if new_w > MIN_W: x = new_x
            if new_h > MIN_H: y = new_y
            w = new_w;  h = new_h

        qwin._edit_x, qwin._edit_y, qwin._edit_w, qwin._edit_h = x, y, w, h
        qwin.geometry(f"{w}x{h}+{x}+{y}")

        # Redraw rectangle to match new canvas size
        canvas = qwin.winfo_children()[0]
        canvas.config(width=w, height=h)
        canvas.delete("all")
        canvas.create_rectangle(2, 2, w-2, h-2, outline="red", width=3)

    def _remove_resize_handles(self):
        """Destroy all corner handle windows."""
        for h in self._resize_handles:
            try:
                h.destroy()
            except Exception:
                pass
        self._resize_handles.clear()

    # ── Reset to defaults ─────────────────────────────────────────────────────

    def reset_to_defaults(self):
        """Restore all coords to the hardcoded factory defaults and rebuild overlays."""
        global KEY_COORDS, AUTO_AREA_1, AUTO_AREA_2, AUTO_AREA_3
        global QUESTION_AREA, QUESTION_AREA_FAST
        global ORIGINAL_KEY_COORDS, ORIGINAL_AUTO_AREA_1, ORIGINAL_AUTO_AREA_2, ORIGINAL_AUTO_AREA_3
        global ORIGINAL_QUESTION_AREA, ORIGINAL_QUESTION_AREA_FAST

        ORIGINAL_QUESTION_AREA       = DEFAULT_QUESTION_AREA
        ORIGINAL_QUESTION_AREA_FAST  = DEFAULT_QUESTION_AREA_FAST
        ORIGINAL_KEY_COORDS          = dict(DEFAULT_KEY_COORDS)
        ORIGINAL_AUTO_AREA_1         = DEFAULT_AUTO_AREA_1
        ORIGINAL_AUTO_AREA_2         = DEFAULT_AUTO_AREA_2
        ORIGINAL_AUTO_AREA_3         = DEFAULT_AUTO_AREA_3

        QUESTION_AREA      = apply_scaling_and_offset(ORIGINAL_QUESTION_AREA)
        QUESTION_AREA_FAST = apply_scaling_and_offset(ORIGINAL_QUESTION_AREA_FAST)
        KEY_COORDS         = {k: apply_scaling_and_offset_xy(v[0], v[1]) for k, v in ORIGINAL_KEY_COORDS.items()}
        AUTO_AREA_1        = apply_scaling_and_offset_xy(*ORIGINAL_AUTO_AREA_1)
        AUTO_AREA_2        = apply_scaling_and_offset_xy(*ORIGINAL_AUTO_AREA_2)
        AUTO_AREA_3        = apply_scaling_and_offset_xy(*ORIGINAL_AUTO_AREA_3)

        self.question_area      = QUESTION_AREA
        self.question_area_fast = QUESTION_AREA_FAST
        self.key_coords         = KEY_COORDS

        # Cancel edit mode cleanly if active
        if self.edit_mode:
            self._remove_resize_handles()
            for win in self.overlay_windows:
                canvas = win.winfo_children()[0]
                canvas.unbind("<ButtonPress-1>")
                canvas.unbind("<B1-Motion>")
            self.edit_mode = False
            self.edit_btn.config(text="Edit", bg="lightyellow")

        if not self.paused:
            self.toggle_pause()

        # Rebuild overlays at default positions
        for win in self.overlay_windows:
            win.destroy()
        self.overlay_windows     = []
        self.auto_overlays       = []
        self.key_overlays        = {}
        self.auto_area_1_overlay = None
        self.auto_area_2_overlay = None
        self.auto_area_3_overlay = None
        self.active_slot         = None
        self.setup_overlay_boxes()

        if not self.overlays_visible:
            for win in self.overlay_windows:
                win.withdraw()
        elif not self.fast_mode:
            for win in self.auto_overlays:
                win.withdraw()

        self._edit_save_pending = False
        self.slot_hint.config(text="Click a slot to load saved coords", fg="gray")
        self._refresh_slot_buttons()
        self.status_text.config(text="PAUSED", fg="red")
        print("[RESET] All coords restored to factory defaults. Click Resume when ready.")

    def _make_draggable(self, win):
        """Attach mouse drag handlers to an overlay window's canvas"""
        canvas = win.winfo_children()[0]
        canvas._drag_start_x = 0
        canvas._drag_start_y = 0

        def on_press(event, w=win, c=canvas):
            # Offset from window top-left to where mouse pressed
            c._drag_start_x = event.x_root - w._edit_x
            c._drag_start_y = event.y_root - w._edit_y

        def on_drag(event, w=win, c=canvas):
            new_x = event.x_root - c._drag_start_x
            new_y = event.y_root - c._drag_start_y
            w._edit_x = new_x
            w._edit_y = new_y
            w.geometry(f"+{new_x}+{new_y}")

        canvas.bind("<ButtonPress-1>", on_press)
        canvas.bind("<B1-Motion>", on_drag)

    def _remove_click_through(self, win):
        """Remove WS_EX_TRANSPARENT so the overlay can receive mouse events"""
        hwnd = ctypes.windll.user32.GetParent(win.winfo_id())
        styles = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        new_style = (styles | WS_EX_LAYERED) & ~WS_EX_TRANSPARENT
        ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new_style)

    def _restore_click_through(self, win):
        """Restore WS_EX_TRANSPARENT so clicks pass through the overlay"""
        hwnd = ctypes.windll.user32.GetParent(win.winfo_id())
        styles = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        new_style = styles | WS_EX_LAYERED | WS_EX_TRANSPARENT
        ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new_style)

    def _apply_overlay_positions(self):
        """
        Read _edit_x/_edit_y from every overlay window (set by the drag handler —
        not winfo_x/y which is unreliable under Windows DPI) and update all runtime
        coordinate globals immediately so the bot uses the new positions right away.
        """
        global KEY_COORDS, AUTO_AREA_1, AUTO_AREA_2, AUTO_AREA_3
        global QUESTION_AREA, QUESTION_AREA_FAST
        global ORIGINAL_KEY_COORDS, ORIGINAL_AUTO_AREA_1, ORIGINAL_AUTO_AREA_2, ORIGINAL_AUTO_AREA_3
        global ORIGINAL_QUESTION_AREA, ORIGINAL_QUESTION_AREA_FAST

        def reverse(fx, fy):
            return (
                int((fx - TASKBAR_X_OFFSET) / SCALE_X),
                int((fy - TASKBAR_Y_OFFSET) / SCALE_Y),
            )

        box_size = int(50 * SCALE_X)
        auto_box_size = int(60 * SCALE_X)

        # Keypad overlays — centre of box
        for key, win in self.key_overlays.items():
            cx = win._edit_x + box_size // 2
            cy = win._edit_y + box_size // 2
            KEY_COORDS[key] = (cx, cy)
            self.key_coords[key] = (cx, cy)
            ORIGINAL_KEY_COORDS[key] = reverse(cx, cy)

        # AUTO area overlays — centre of box
        for attr, gvar, orig_gvar in [
            ('auto_area_1_overlay', 'AUTO_AREA_1', 'ORIGINAL_AUTO_AREA_1'),
            ('auto_area_2_overlay', 'AUTO_AREA_2', 'ORIGINAL_AUTO_AREA_2'),
            ('auto_area_3_overlay', 'AUTO_AREA_3', 'ORIGINAL_AUTO_AREA_3'),
        ]:
            win = getattr(self, attr)
            if win is None:
                continue
            cx = win._edit_x + auto_box_size // 2
            cy = win._edit_y + auto_box_size // 2
            globals()[gvar] = (cx, cy)
            globals()[orig_gvar] = reverse(cx, cy)

        # Question / OCR area overlay — top-left + size
        # IMPORTANT: setup_overlay_boxes places the red box at QUESTION_AREA_FAST in fast mode
        # and at QUESTION_AREA in standard mode. So we must derive the "other" value from the box.
        qw = self.question_overlay
        qx, qy = qw._edit_x, qw._edit_y
        qw2, qh2 = qw._edit_w, qw._edit_h

        inset_x = int(12 * SCALE_X)   # 12 reference-px → screen-px

        if self.fast_mode:
            # The dragged box IS the fast (inset) area — full area is wider by inset_x each side
            QUESTION_AREA_FAST = (qx, qy, qx + qw2, qy + qh2)
            QUESTION_AREA      = (qx - inset_x, qy, qx + qw2 + inset_x, qy + qh2)
        else:
            # The dragged box IS the full area — fast area is narrower by inset_x each side
            QUESTION_AREA      = (qx, qy, qx + qw2, qy + qh2)
            QUESTION_AREA_FAST = (qx + inset_x, qy, qx + qw2 - inset_x, qy + qh2)

        self.question_area      = QUESTION_AREA
        self.question_area_fast = QUESTION_AREA_FAST

        # ── ORIGINAL_ values (resolution-independent, used by taskbar recalc) ──
        # Always derive from the full (non-fast) area
        ox1, oy1 = reverse(QUESTION_AREA[0], QUESTION_AREA[1])
        ox2, oy2 = reverse(QUESTION_AREA[2], QUESTION_AREA[3])
        ORIGINAL_QUESTION_AREA      = (ox1, oy1, ox2, oy2)
        ORIGINAL_QUESTION_AREA_FAST = (ox1 + 12, oy1, ox2 - 12, oy2)

        mode_tag = "FAST" if self.fast_mode else "STANDARD"
        print(f"[EDIT] Live coords applied ({mode_tag} mode):")
        print(f"  Q_AREA     ={QUESTION_AREA}")
        print(f"  Q_AREA_FAST={QUESTION_AREA_FAST}  (inset {inset_x}px each side)")
        print(f"  AUTO1={AUTO_AREA_1}  AUTO2={AUTO_AREA_2}  AUTO3={AUTO_AREA_3}")

    def _build_coord_snapshot(self):
        """Return a dict of the current ORIGINAL_* coords (resolution-independent)."""
        return {
            "question_area":      list(ORIGINAL_QUESTION_AREA),
            "question_area_fast": list(ORIGINAL_QUESTION_AREA_FAST),
            "key_coords":         {k: list(v) for k, v in ORIGINAL_KEY_COORDS.items()},
            "auto_area_1":        list(ORIGINAL_AUTO_AREA_1),
            "auto_area_2":        list(ORIGINAL_AUTO_AREA_2),
            "auto_area_3":        list(ORIGINAL_AUTO_AREA_3),
        }

    # ── Slot file helpers ─────────────────────────────────────────────────────

    def _read_slots_file(self):
        """Load slots from JSON; returns list of MAX_SLOTS dicts (None = empty)."""
        try:
            if os.path.exists(SLOTS_FILE):
                with open(SLOTS_FILE, "r") as f:
                    data = json.load(f)
                slots = data.get("slots", [])
                # Pad / trim to exactly MAX_SLOTS
                while len(slots) < MAX_SLOTS:
                    slots.append(None)
                return slots[:MAX_SLOTS]
        except Exception as e:
            print(f"[SLOTS] Error reading {SLOTS_FILE}: {e}")
        return [None] * MAX_SLOTS

    def _write_slots_file(self, slots):
        """Persist the slots list to JSON."""
        try:
            with open(SLOTS_FILE, "w") as f:
                json.dump({"slots": slots}, f, indent=2)
            print(f"[SLOTS] Saved to {SLOTS_FILE}")
        except Exception as e:
            print(f"[SLOTS] Error writing {SLOTS_FILE}: {e}")

    def _slot_button_clicked(self, idx):
        """Handle a slot button press — save if pending, load otherwise."""
        if self._edit_save_pending:
            self._save_to_slot(idx)
        else:
            self._load_slot(idx)

    def _save_to_slot(self, idx):
        """Save current coords to slot idx (0-based) and write to disk."""
        import datetime
        slots = self._read_slots_file()
        slots[idx] = {
            "label":   f"Slot {idx + 1}",
            "saved":   datetime.datetime.now().strftime("%d/%m %H:%M"),
            "coords":  self._build_coord_snapshot(),
        }
        self._write_slots_file(slots)
        self.active_slot = idx
        self._edit_save_pending = False

        self.slot_hint.config(text="Click a slot to load saved coords", fg="gray")
        self._refresh_slot_buttons()
        print(f"[SLOTS] Slot {idx + 1} saved.")

    def _load_slot(self, idx):
        """Load coords from slot idx and rebuild overlays."""
        global KEY_COORDS, AUTO_AREA_1, AUTO_AREA_2, AUTO_AREA_3
        global QUESTION_AREA, QUESTION_AREA_FAST
        global ORIGINAL_KEY_COORDS, ORIGINAL_AUTO_AREA_1, ORIGINAL_AUTO_AREA_2, ORIGINAL_AUTO_AREA_3
        global ORIGINAL_QUESTION_AREA, ORIGINAL_QUESTION_AREA_FAST

        slots = self._read_slots_file()
        slot = slots[idx]
        if slot is None:
            print(f"[SLOTS] Slot {idx + 1} is empty — nothing to load.")
            return

        c = slot["coords"]

        # Restore ORIGINAL_* (resolution-independent) values
        ORIGINAL_QUESTION_AREA       = tuple(c["question_area"])
        ORIGINAL_QUESTION_AREA_FAST  = tuple(c["question_area_fast"])
        ORIGINAL_KEY_COORDS          = {k: tuple(v) for k, v in c["key_coords"].items()}
        ORIGINAL_AUTO_AREA_1         = tuple(c["auto_area_1"])
        ORIGINAL_AUTO_AREA_2         = tuple(c["auto_area_2"])
        ORIGINAL_AUTO_AREA_3         = tuple(c["auto_area_3"])

        # Re-derive scaled runtime values
        QUESTION_AREA      = apply_scaling_and_offset(ORIGINAL_QUESTION_AREA)
        QUESTION_AREA_FAST = apply_scaling_and_offset(ORIGINAL_QUESTION_AREA_FAST)
        KEY_COORDS         = {k: apply_scaling_and_offset_xy(v[0], v[1]) for k, v in ORIGINAL_KEY_COORDS.items()}
        AUTO_AREA_1        = apply_scaling_and_offset_xy(*ORIGINAL_AUTO_AREA_1)
        AUTO_AREA_2        = apply_scaling_and_offset_xy(*ORIGINAL_AUTO_AREA_2)
        AUTO_AREA_3        = apply_scaling_and_offset_xy(*ORIGINAL_AUTO_AREA_3)

        # Update instance references
        self.question_area      = QUESTION_AREA
        self.question_area_fast = QUESTION_AREA_FAST
        self.key_coords         = KEY_COORDS

        # Rebuild overlays at new positions
        was_paused = self.paused
        if not was_paused:
            self.toggle_pause()

        for win in self.overlay_windows:
            win.destroy()
        self.overlay_windows = []
        self.auto_overlays   = []
        self.key_overlays    = {}
        self.auto_area_1_overlay = None
        self.auto_area_2_overlay = None
        self.auto_area_3_overlay = None
        self.setup_overlay_boxes()

        # Restore visibility state
        if not self.overlays_visible:
            for win in self.overlay_windows:
                win.withdraw()
        elif not self.fast_mode:
            for win in self.auto_overlays:
                win.withdraw()

        self.active_slot = idx
        self._refresh_slot_buttons()
        print(f"[SLOTS] Slot {idx + 1} loaded. Bot paused — click Resume when ready.")
        self.status_text.config(text="PAUSED", fg="red")
        self.pause_btn.config(text="Resume")

    def _refresh_slot_buttons(self, save_mode=False):
        """Update all 3 slot button labels and colours to reflect current state."""
        slots = self._read_slots_file()
        for i, btn in enumerate(self.slot_btns):
            slot = slots[i]
            is_active = (i == self.active_slot)

            if save_mode:
                btn.config(
                    text=f"Save → Slot {i+1}",
                    bg="#ff9944", fg="black",
                    relief="raised"
                )
            elif slot is None:
                btn.config(
                    text=f"Slot {i+1}\n(empty)",
                    bg="#ddd", fg="gray",
                    relief="flat"
                )
            else:
                btn.config(
                    text=f"Slot {i+1}\n{slot['saved']}",
                    bg="#aaddaa" if is_active else "#cceecc",
                    fg="black",
                    relief="sunken" if is_active else "raised"
                )

    # ─────────────── END EDIT MODE ───────────────

    # ─────────────── PERSISTENT LOOK-UP TABLE ───────────────

    def _load_lut(self):
        """Load the LUT from disk at startup. Returns a plain dict {question: answer}."""
        try:
            if os.path.exists(LUT_FILE):
                with open(LUT_FILE, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    print(f"[LUT] Loaded {len(data)} entries from {LUT_FILE}")
                    return data
        except Exception as e:
            print(f"[LUT] Error loading {LUT_FILE}: {e}")
        return {}

    def _save_lut_async(self):
        """Write the LUT to disk on a background thread so the main loop isn't stalled."""
        def _write(snapshot):
            try:
                with open(LUT_FILE, "w") as f:
                    json.dump(snapshot, f, indent=2)
            except Exception as e:
                print(f"[LUT] Error saving {LUT_FILE}: {e}")

        snapshot = dict(self.lut)   # shallow copy — values are ints/floats, safe
        self._lut_dirty = False
        threading.Thread(target=_write, args=(snapshot,), daemon=True).start()

    def _lut_record(self, norm_expr, answer):
        """
        Add a normalised-expression→answer pair to the LUT.
        Keyed on norm_expr (not raw OCR) so minor OCR variations of the same
        question always hit the same entry.
        Only called after a fresh solve so we never overwrite a known-good answer.
        """
        if norm_expr not in self.lut:
            self.lut[norm_expr] = answer
            self._lut_dirty = True
            self.lut_text.config(text=f"LUT: {len(self.lut)} entries")
            # Also warm session cache immediately
            self.answer_cache[norm_expr] = answer
            self._save_lut_async()
            print(f"[LUT] Saved  '{norm_expr}' → {answer}  ({len(self.lut)} total)")

    def _clear_lut(self):
        """Wipe the in-memory LUT, the pre-warmed session cache, and delete the file."""
        self.lut.clear()
        self.answer_cache.clear()
        self.lut_text.config(text="LUT: 0 entries")
        self.cache_text.config(text="Cache: 0 questions")
        try:
            if os.path.exists(LUT_FILE):
                os.remove(LUT_FILE)
        except Exception as e:
            print(f"[LUT] Error deleting {LUT_FILE}: {e}")
        print("[LUT] Cleared all entries and deleted file.")

    # ─────────────── END PERSISTENT LOOK-UP TABLE ───────────────

    def start_taskbar_monitoring(self):
        """Start periodic taskbar position checking"""
        self.check_taskbar_position()
    
    def check_taskbar_position(self):
        """
        Periodically check if taskbar has moved and recalculate coordinates if needed.
        Runs every TASKBAR_CHECK_INTERVAL milliseconds.
        """
        try:
            # Get current taskbar position
            x_offset, y_offset, position = get_taskbar_position()
            
            # Check if taskbar position has changed
            if position != self.current_taskbar_position:
                print(f"\n[TASKBAR] Position changed: {self.current_taskbar_position} → {position}")
                print(f"[TASKBAR] Recalculating coordinates...")
                
                # Update stored position
                self.current_taskbar_position = position
                
                # Recalculate all coordinates with new offset
                global TASKBAR_X_OFFSET, TASKBAR_Y_OFFSET, QUESTION_AREA, QUESTION_AREA_FAST
                global KEY_COORDS, AUTO_AREA_1, AUTO_AREA_2, AUTO_AREA_3
                
                TASKBAR_X_OFFSET = x_offset
                TASKBAR_Y_OFFSET = y_offset
                
                # Recalculate coordinates
                QUESTION_AREA = apply_scaling_and_offset(ORIGINAL_QUESTION_AREA)
                QUESTION_AREA_FAST = apply_scaling_and_offset(ORIGINAL_QUESTION_AREA_FAST)
                KEY_COORDS = {k: apply_scaling_and_offset_xy(v[0], v[1]) for k, v in ORIGINAL_KEY_COORDS.items()}
                AUTO_AREA_1 = apply_scaling_and_offset_xy(ORIGINAL_AUTO_AREA_1[0], ORIGINAL_AUTO_AREA_1[1])
                AUTO_AREA_2 = apply_scaling_and_offset_xy(ORIGINAL_AUTO_AREA_2[0], ORIGINAL_AUTO_AREA_2[1])
                AUTO_AREA_3 = apply_scaling_and_offset_xy(ORIGINAL_AUTO_AREA_3[0], ORIGINAL_AUTO_AREA_3[1])
                
                # Update instance variables
                self.question_area = QUESTION_AREA
                self.question_area_fast = QUESTION_AREA_FAST
                self.key_coords = KEY_COORDS
                
                # Always recreate overlays with new positions, regardless of visibility
                print("[TASKBAR] Updating overlay positions...")
                for win in self.overlay_windows:
                    win.destroy()
                self.overlay_windows = []
                self.auto_overlays = []
                self.key_overlays = {}
                self.auto_area_1_overlay = None
                self.auto_area_2_overlay = None
                self.auto_area_3_overlay = None
                self.setup_overlay_boxes()

                # Restore correct visibility state after recreation
                if not self.overlays_visible:
                    # Overlays are toggled off — hide everything
                    for win in self.overlay_windows:
                        win.withdraw()
                elif not self.fast_mode:
                    # Overlays on but in standard mode — hide auto overlays
                    for win in self.auto_overlays:
                        win.withdraw()

                print("[TASKBAR] ✓ Coordinates updated successfully")
                
        except Exception as e:
            print(f"[TASKBAR] Error checking position: {e}")
        
        # Schedule next check
        self.root.after(TASKBAR_CHECK_INTERVAL, self.check_taskbar_position)
    
    def cancel_all_scheduled_events(self):
        """Cancel all pending automation timer events"""
        for event_id in self.scheduled_events:
            try:
                self.root.after_cancel(event_id)
            except Exception:
                pass
        self.scheduled_events.clear()
        print("[AUTO] All scheduled automation events cancelled")

    def _reset_last_question(self):
        """
        Clear the dedup guard 100ms after answering so the same question can be
        answered again immediately if the game re-shows it (e.g. next round starts
        with the same question).
        """
        self.last_question = ""
        self._last_question_reset_id = None

    def _prune_scheduled_events(self):
        """
        Remove already-fired event IDs from the list to prevent it growing unbounded.
        tkinter doesn't tell us which events have fired, so we attempt to cancel each
        one — a no-op for events that already ran — and rebuild the list keeping only
        those that cancel_all_scheduled_events would actually need to cancel.
        We approximate this by just capping the list to the last 20 entries; any entry
        older than the most recent batch will have already fired.
        """
        if len(self.scheduled_events) > 40:
            # Keep only the most recent 20 IDs — older ones have definitely fired
            self.scheduled_events = self.scheduled_events[-20:]

    def normalize_operators(self, expr):
        """Convert various operator symbols (×, ÷) to standard Python operators (*, /)"""
        expr = expr.replace('×', '*').replace('x', '*').replace('X', '*').replace('÷', '/').replace(':', '/')
        return self.operator_clean.sub('', expr)

    def fix_missing_operator(self, expr):
        """Apply heuristics to fix common OCR mistakes like missing operators"""
        if '(' not in expr and ')' not in expr and '+' in expr:
            return expr.replace('+', '/', 1)
        match = self.div_pattern.search(expr)
        if match:
            return f"{match.group(1)} / {match.group(2)} = ?"
        return self.mult_pattern.sub(r"\1 * \2", expr, count=1)

    def finalize_expression(self, expr):
        """Clean up expression formatting and balance parentheses"""
        expr = self.space_clean.sub(r'\1', expr)
        paren_diff = expr.count('(') - expr.count(')')
        if paren_diff > 0:
            expr += ')' * paren_diff
        return expr

    def solve_algebra(self, expr):
        """
        Solve mathematical expression using sympy
        Handles both simple arithmetic and algebraic equations with '?'
        """
        try:
            if '?' not in expr:
                # Simple arithmetic evaluation
                result = N(sympify(expr))
                return int(result) if result.is_integer else float(result)
            # Algebraic equation solving
            lhs, rhs = expr.replace('?', 'x').split('=')
            sol = solve(Eq(sympify(lhs), sympify(rhs)), self.x_symbol)
            if sol:
                result = N(sol[0])
                return int(result) if result.is_integer else float(result)
        except Exception as e:
            print(f"Solver Error: {e}", file=sys.stderr)
        return None

    def click_answer(self, answer, is_cached=False):
        """
        Click the answer on the on-screen keypad
        Handles automation counting and triggers auto-click sequences
        """
        self.is_answering = True
        try:
            # If extended sequence is active and a new question is answered, cancel it
            if self.extended_sequence_active and not is_cached:
                print("[AUTO] New question answered during extended sequence - RESETTING sequence")
                print("[AUTO] Note: Counter frozen during extended sequence - will resume after reset")
                self.cancel_all_scheduled_events()
                self.extended_sequence_active = False
                self.auto_status_text.config(text="⚠ Sequence reset", fg="orange")
                # Clear status after 2 seconds
                self.root.after(2000, lambda: self.auto_status_text.config(text=""))
            
            answer_int = int(answer)
            answer_str = str(answer_int)
            cache_tag = "[CACHED] " if is_cached else ""
            print(f"{cache_tag}Clicking answer: {answer_str}")
            
            # Guard: only click if every digit has a mapped key
            # (negative numbers contain '-' which has no key)
            if not all(d in self.key_coords for d in answer_str):
                print(f"[SKIP] Answer '{answer_str}' contains characters not in keypad — skipping click")
                return
            
            # Click each digit
            for d in answer_str:
                pyautogui.click(self.key_coords[d])
                if KEY_PRESS_DELAY > 0:
                    time.sleep(KEY_PRESS_DELAY)
            
            # Click OK button
            pyautogui.click(self.key_coords['OK'])
            if POST_ANSWER_DELAY > 0:
                time.sleep(POST_ANSWER_DELAY)
            
            # Only increment counter if NOT cached, in FAST mode, AND not during extended sequence
            if not is_cached and self.fast_mode and not self.extended_sequence_active:
                self.answers_count += 1
                self.update_counter_display()
                
                # Check if we hit 10 answers - trigger automation
                if self.answers_count >= 10:
                    self.answers_count = 0
                    self.ready_count += 1
                    print(f"[AUTO] 10 answers reached! Ready count: {self.ready_count}")
                    self.auto_status_text.config(text="⏳ Auto sequence starting...", fg="orange")
                    
                    # Schedule AUTO_AREA_1 click with different timing based on ready count
                    if self.ready_count >= 3:
                        auto1_delay = 10000  # 10 seconds when ready reaches 3
                        print(f"[AUTO] Ready=3 reached! AUTO 1 will click in 10 seconds")
                        self.extended_sequence_active = True
                    else:
                        auto1_delay = 2500  # 2.5 seconds before ready reaches 3
                        print(f"[AUTO] AUTO 1 will click in 2.5 seconds (Ready: {self.ready_count}/3)")
                    
                    event_id = self.root.after(auto1_delay, self.auto_click_area_1_initial)
                    self.scheduled_events.append(event_id)
                    
                    # Check if ready count is 3 - trigger extended sequence
                    if self.ready_count >= 3:
                        print("[AUTO] Ready count is 3! Starting timed sequence...")
                        self.ready_count = 0
                        self.update_counter_display()
                        
                        event_id = self.root.after(15000, self.clear_cache_for_new_session)
                        self.scheduled_events.append(event_id)
                        
                        event_id = self.root.after(10000, self.auto_click_area_2)
                        self.scheduled_events.append(event_id)

        except Exception as e:
            print(f"[CLICK ERROR] {e}")
        finally:
            # Always reset is_answering — even if an exception was raised mid-click
            self.is_answering = False
    
    def auto_click_area_1_initial(self):
        """Click AUTO_AREA_1 after delay (2.5s if ready<3, 10s if ready=3)"""
        if not self.is_answering and not self.paused and self.fast_mode:
            print(f"[AUTO] Clicking AUTO AREA 1 at {AUTO_AREA_1}")
            pyautogui.click(AUTO_AREA_1)
            self.auto_status_text.config(text="✓ AUTO 1 clicked", fg="green")
        else:
            if not self.fast_mode:
                reason = "not in FAST mode"
            else:
                reason = "bot is answering" if self.is_answering else "bot is paused"
            print(f"[AUTO] Skipping AUTO 1 click - {reason}")
            self.auto_status_text.config(text="", fg="gray")
    
    def auto_click_area_2(self):
        """Click AUTO_AREA_2 after 10 second delay"""
        if not self.is_answering and not self.paused and self.fast_mode:
            print(f"[AUTO] Clicking AUTO AREA 2 at {AUTO_AREA_2}")
            pyautogui.click(AUTO_AREA_2)
            self.auto_status_text.config(text="✓ AUTO 2 clicked", fg="green")
            
            # Schedule AUTO_AREA_3 click after 2 seconds
            event_id = self.root.after(2000, self.auto_click_area_3)
            self.scheduled_events.append(event_id)
        else:
            if not self.fast_mode:
                reason = "not in FAST mode"
            else:
                reason = "bot is answering" if self.is_answering else "bot is paused"
            print(f"[AUTO] Skipping AUTO 2 click - {reason}")
    
    def auto_click_area_3(self):
        """Click AUTO_AREA_3 after 2 second delay"""
        if not self.is_answering and not self.paused and self.fast_mode:
            print(f"[AUTO] Clicking AUTO AREA 3 at {AUTO_AREA_3}")
            pyautogui.click(AUTO_AREA_3)
            self.auto_status_text.config(text="✓ AUTO 3 clicked", fg="green")
            
            # Schedule AUTO_AREA_1 final click after 8 seconds
            event_id = self.root.after(8000, self.auto_click_area_1_final)
            self.scheduled_events.append(event_id)
        else:
            if not self.fast_mode:
                reason = "not in FAST mode"
            else:
                reason = "bot is answering" if self.is_answering else "bot is paused"
            print(f"[AUTO] Skipping AUTO 3 click - {reason}")
    
    def auto_click_area_1_final(self):
        """Click AUTO_AREA_1 after 8 second delay (final step in sequence)"""
        if not self.is_answering and not self.paused and self.fast_mode:
            print(f"[AUTO] Clicking AUTO AREA 1 (final) at {AUTO_AREA_1}")
            pyautogui.click(AUTO_AREA_1)
            self.auto_status_text.config(text="✓ Sequence complete", fg="green")
            # Mark extended sequence as complete
            self.extended_sequence_active = False
            # Clear status after 2 seconds
            event_id = self.root.after(2000, lambda: self.auto_status_text.config(text=""))
            self.scheduled_events.append(event_id)
        else:
            if not self.fast_mode:
                reason = "not in FAST mode"
            else:
                reason = "bot is answering" if self.is_answering else "bot is paused"
            print(f"[AUTO] Skipping AUTO 1 (final) click - {reason}")
            self.extended_sequence_active = False  # Reset flag even if skipped
    
    def clear_cache_for_new_session(self):
        """Clear answer cache 8 seconds after ready=3 (for new session)"""
        self.answer_cache = {}
        self.cache_text.config(text="Cache: 0 questions")
        print("[AUTO] Cache cleared for new session (8s delay)")
    
    def update_counter_display(self):
        """Update the answer counter display in GUI"""
        self.counter_text.config(text=f"Answers: {self.answers_count}/10 | Ready: {self.ready_count}")

    def main_loop(self):
        """
        Main detection and solving loop
        Runs continuously at FAST_MODE_POLLING or STANDARD_MODE_POLLING interval
        """
        # Prune stale event IDs so the list doesn't grow unbounded
        self._prune_scheduled_events()
        try:
            if not self.paused:
                # Use the correct question area based on mode
                current_area = self.question_area_fast if self.fast_mode else self.question_area
                img = ImageGrab.grab(bbox=current_area)
                
                # Update preview only if enabled and at specified interval
                if self.preview_enabled:
                    self.preview_loop_counter += 1
                    if self.preview_loop_counter >= PREVIEW_UPDATE_INTERVAL:
                        self.preview_loop_counter = 0
                        
                        # Update preview in GUI (only every N loops for performance)
                        img_preview = img.resize((300, 90))
                        self.preview_img_tk = ImageTk.PhotoImage(img_preview)
                        if getattr(self.preview_canvas, 'img_id', None):
                            self.preview_canvas.itemconfig(self.preview_canvas.img_id, image=self.preview_img_tk)
                        else:
                            self.preview_canvas.img_id = self.preview_canvas.create_image(
                                0, 0, anchor=tk.NW, image=self.preview_img_tk
                            )
                
                # Fast image preprocessing for better OCR
                img_array = np.array(img.convert("L"))
                img_array[img_array < 140] = 0
                img_array[img_array >= 140] = 255

                # Optimized EasyOCR parameters
                result = self.reader.readtext(
                    img_array,
                    allowlist='0123456789+-*/()=?xX×÷: ',
                    low_text=0.3,
                    batch_size=1,
                    paragraph=False,
                    min_size=5
                )

                if result:
                    question = " ".join(t for _, t, _ in result)
                    
                    if question != self.last_question:
                        self.last_question = question

                        # Normalise the raw OCR string into a clean expression.
                        # ALL lookups and storage use this key so OCR whitespace /
                        # symbol variations of the same question always hit the same entry.
                        norm = self.finalize_expression(
                            self.fix_missing_operator(
                                self.normalize_operators(question)
                            )
                        )
                        print(f"Detected: {question}  →  norm: {norm}")
                        
                        # 1. Session cache — keyed by norm, fast mode only, very fast
                        if self.fast_mode and norm in self.answer_cache:
                            answer = self.answer_cache[norm]
                            print(f"[CACHE HIT] {norm} → {answer}")
                            self.click_answer(int(answer), is_cached=True)
                        
                        # 2. Persistent LUT — keyed by norm, all modes, survives restarts
                        elif norm in self.lut:
                            answer = self.lut[norm]
                            print(f"[LUT HIT] {norm} → {answer}")
                            # Warm session cache so next hit is even faster
                            if self.fast_mode:
                                self.answer_cache[norm] = answer
                                self.cache_text.config(text=f"Cache: {len(self.answer_cache)} questions")
                            self.click_answer(int(answer), is_cached=True)
                        
                        # 3. Solve fresh — store result in both LUT and session cache
                        else:
                            answer = self.solve_algebra(norm)
                            if answer is not None and float(answer).is_integer():
                                if self.fast_mode:
                                    self.answer_cache[norm] = answer
                                    self.cache_text.config(text=f"Cache: {len(self.answer_cache)} questions")
                                self._lut_record(norm, int(answer))
                                self.click_answer(int(answer), is_cached=False)

                        # Schedule last_question reset 100ms after answering.
                        # If the game re-shows the same question (e.g. next round),
                        # the dedup guard won't block it after this window expires.
                        if self._last_question_reset_id is not None:
                            try:
                                self.root.after_cancel(self._last_question_reset_id)
                            except Exception:
                                pass
                        self._last_question_reset_id = self.root.after(
                            100, self._reset_last_question
                        )
        except Exception as e:
            print(f"Loop Error: {e}")
        
        # Reschedule — wrapped so a tkinter hiccup can't silently kill the loop
        try:
            self.root.after(self.current_polling, self.main_loop)
        except Exception as e:
            print(f"[FATAL] Failed to reschedule main_loop: {e}  — attempting recovery in 500ms")
            try:
                self.root.after(500, self.main_loop)
            except Exception:
                pass

    def run(self):
        """Start the math solver bot"""
        print("OpticalReaderSolver running.")
        print("Use buttons in GUI for controls")
        print("Starting in FAST mode (10ms polling)")
        print("Note: Automation only active in FAST mode")
        print(f"Taskbar monitoring: Checking every {TASKBAR_CHECK_INTERVAL/1000}s for position changes")
        print(f"Preview: Enabled (updates every {PREVIEW_UPDATE_INTERVAL} loops) - toggle for max speed")
        self.root.after(200, self.main_loop)
        self.root.mainloop()

if __name__ == "__main__":
    OpticalReaderSolver(QUESTION_AREA, KEY_COORDS).run()
