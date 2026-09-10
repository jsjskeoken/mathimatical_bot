# Optical Reader & Math Solver

A screen-based OCR pipeline that reads an on-screen math expression, solves it, and (optionally) clicks the answer — built around a normalize → LUT/cache → solve flow so repeated or known questions never touch OCR or the solver twice.

Shared as a reference implementation for the OCR cleanup, caching, and coordinate-mapping techniques — not maintained as an active bot.

![Demo](demo.gif)

## How it works

1. **Capture** — `mss` grabs the configured screen region as a raw BGRA array.
2. **Preprocess** — greyscale conversion straight from the raw capture (no PIL round-trip), Gaussian blur + unsharp mask to sharpen digit edges, a cached CLAHE pass for uneven lighting, then Otsu thresholding to binarize. No upscaling — it was the single biggest performance cost in an earlier version.
3. **OCR** — EasyOCR reads the processed frame (GPU if available, CPU fallback).
4. **Operator disambiguation** — EasyOCR sometimes misreads the on-screen `÷` glyph as `+`. Rather than a blanket text-level rule (`+` almost always really is `+`), the pixels under each reported `+` are checked against its own bounding box: a genuine plus is one connected blob (a cross), while `÷` is two dot-shaped blobs stacked with a gap, optionally with a wider bar between them. Only corrected to `/` on that visual evidence — anything ambiguous is left as `+` rather than guessed.
5. **Candidate selection** — if the capture box catches more than the question itself (a stray date, label, or fragment), the most plausible expression is picked from the OCR tokens rather than naively joining everything: candidates are filtered by enabled operators (see "Known operations" below), digit-group count, and date-shape, then only the ones that actually solve are considered, preferring the shortest span that does.
6. **Clean & normalize** — strips OCR hallucinations, corrects common letter/digit misreads (`o`→`0`, `s`→`5`, etc.), and collapses the result into a canonical expression string.
7. **Solve** — checked in this order: **LUT** (persistent, on disk) → **session cache** (in-memory, this round only) → **`eval()`** for plain arithmetic → **SymPy** for anything with a variable. A fresh solve gets written back to the LUT.
8. **Click (optional)** — if answer clicking is on, the answer is typed onto the on-screen keypad using coordinates from the active profile, with a small delay between keys and a verified cursor placement (see below).

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
- Solver-mode switch, pause/resume (bindable globally — default hotkey `F8`, via a `pynput` listener so it works while the window isn't focused; pauses OCR/answer-clicking AND the AUTO 1/2/3 sequence together)
- Save/load up to 3 numbered coordinate profiles ("slots") to `optical_coords.json`
- Cache and LUT hit counters
- Automation on/off toggle — stops answer-submission clicking and the AUTO 1/2/3 sequence together; a repeated-unconfirmed-click safety net can also trip this automatically (see Known limitations)
- **Known operations** (Advanced) — checkboxes for +, −, ×, ÷ that *filter* which operators a candidate expression may use during candidate selection; disabling one never converts it into another operator
- **Save OCR captures** (Advanced, off by default) — writes the original + processed image for each solved question to `ocr_captures/`, for diagnosing OCR misreads. Off by default since a long unattended run would otherwise accumulate images without bound.

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
pip install --user easyocr opencv-python numpy sympy pyautogui mss pillow pynput certifi
```

`--user` installs into your own profile instead of Python's shared site-packages — needed on locked-down accounts (school/lab computers) that can't write there without admin rights. `launch_solver.bat` already does this automatically (see below); use `--user` yourself only if you're installing by hand, e.g. from VS Code's integrated terminal.

`certifi` is optional — EasyOCR's first run downloads its recognition model over HTTPS, and on some machines (locked-down accounts, some corporate networks) Python's bundled certificate store is missing or stale enough to make that fail with an SSL error. If `certifi` is installed, its up-to-date CA bundle is used automatically; if it isn't, the app runs exactly as before. Nothing depends on it being present.

## Run

```bash
python gui.py
```

or on Windows, just run `launch_solver.bat`, which auto-detects a working Python 3.10–3.13 interpreter and installs missing dependencies for you — retrying with `--user` automatically if the system-wide install fails from a permissions error, so it works on restricted accounts without any manual steps.

`F8` toggles pause/resume globally.

## Known limitations

- **Negative answers can't be submitted.** The on-screen keypad has no minus-sign key, so a question that resolves to a negative number is solved correctly (and shown in the GUI) but the click is skipped — check the console for `[CORE] [SKIP] '-N' has unmapped chars`.
- **Click automation needs the target window focused.** If a different window has focus at the moment an answer is ready, the click is skipped rather than sent to the wrong place — check the console for `[CORE] [SKIP] Target window not focused`.
- **Repeated unconfirmed clicks auto-disable answer clicking.** If the screen doesn't visibly change after several consecutive clicks (the click isn't landing, or the target UI isn't responding), the Automation toggle switches itself off rather than continuing to click blind — check the console for `[GUI] [WARN] Click unconfirmed`.

## Notes

This was built and tuned against one specific on-screen layout, so the coordinate profiles won't line up with a different app or resolution out of the box — capture new ones with `logger.py` and save a new profile. Screen automation like this may violate the terms of service of whatever application it's pointed at, so treat this as a reference for the technique rather than a drop-in tool.
