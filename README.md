# Optical Reader & Math Solver

A screen-based OCR pipeline that reads an on-screen math expression, solves it, and (optionally) clicks the answer — built around a normalize → LUT/cache → solve flow so repeated or known questions never touch OCR or the solver twice.

Shared as a reference implementation for the OCR cleanup, caching, and coordinate-mapping techniques — not maintained as an active bot.

![Demo](demo.gif)

## How it works

1. **Capture** — `mss` grabs the configured screen region as a raw BGRA array.
2. **Preprocess** — greyscale conversion straight from the raw capture (no PIL round-trip), Gaussian blur + unsharp mask to sharpen digit edges, a cached CLAHE pass for uneven lighting, then Otsu thresholding to binarize. No upscaling — it was the single biggest performance cost in an earlier version.
3. **OCR** — EasyOCR reads the processed frame (GPU if available, CPU fallback).
4. **Clean & normalize** — strips OCR hallucinations, corrects common misreads (e.g. a `/` that OCR read as `+`), and collapses the result into a canonical expression string.
5. **Solve** — checked in this order: **LUT** (persistent, on disk) → **session cache** (in-memory, this round only) → **`eval()`** for plain arithmetic → **SymPy** for anything with a variable. A fresh solve gets written back to the LUT.
6. **Click (optional)** — if automation is on, the answer is typed onto the on-screen keypad using coordinates from the active profile.

## Solver modes

| Mode | Behavior |
|---|---|
| **Hybrid** (default) | LUT → session cache → `eval()` → SymPy, in that order. New solves are saved to the LUT. |
| **Calc Only** | Ignores the LUT and cache both ways — always recomputes with `eval()`/SymPy, and does *not* write the result back to the LUT. |
| **LUT Only** | Checks the session cache, then the LUT. Never calls the solver — an unrecognized expression is skipped. |

Fast and Standard capture modes trade polling interval (10ms vs 150ms) for OCR load.

## GUI

Tkinter GUI (`gui.py`) with:
- Live OCR preview
- Draggable/resizable region overlays for the question area and keypad
- Solver-mode switch, pause/resume (bindable globally — default hotkey `F8`, via a `pynput` listener so it works while the window isn't focused)
- Save/load up to 3 numbered coordinate profiles ("slots") to `optical_coords.json`
- Cache and LUT hit counters
- Automation on/off toggle, independent of solving

## Files

| File | Purpose |
|---|---|
| `gui.py` | Tkinter GUI — overlays, controls, main loop |
| `bot_core.py` | OCR, cleanup/normalization, solving, caching, LUT, click automation |
| `backup.py` | Standalone legacy single-file implementation, kept for reference |
| `optical_lut.json` | Persistent expression → answer lookup table |
| `optical_coords.json` | Saved coordinate profiles (question area, keypad, auto-click zones) |
| `logger.py` | Standalone mouse-position logger, useful for finding new coordinates |
| `launch_solver.bat` | Windows launcher — finds a working Python install and installs missing deps automatically |

## Requirements

- **Windows only.** Click automation and DPI-awareness go through the Windows API directly (`ctypes.windll`) — it won't run unmodified on macOS/Linux.
- **Python 3.10–3.13.** As of this writing, `opencv-python` still doesn't ship prebuilt Python 3.14 wheels, so `pip install` on 3.14 tends to fall back to a from-source build and fail without a C compiler. Check the [opencv-python PyPI page](https://pypi.org/project/opencv-python/) for current wheel support before assuming this has changed.
- Don't have Python installed? Grab it here:
  - [python.org/downloads](https://www.python.org/downloads/) — auto-detects your OS
  - [python.org/downloads/windows](https://www.python.org/downloads/windows/) — Windows-specific builds/installers
  - Tick **"Add python.exe to PATH"** during install, or the launcher script won't find it.

```bash
pip install easyocr opencv-python numpy sympy pyautogui mss pillow pynput
```

## Run

```bash
python gui.py
```

or on Windows, just run `launch_solver.bat`, which auto-detects a working Python 3.10–3.13 interpreter and installs missing dependencies for you.

`F8` toggles pause/resume globally.

## Notes

This was built and tuned against one specific on-screen layout, so the coordinate profiles won't line up with a different app or resolution out of the box — capture new ones with `logger.py` and save a new profile. Screen automation like this may violate the terms of service of whatever application it's pointed at, so treat this as a reference for the technique rather than a drop-in tool.
