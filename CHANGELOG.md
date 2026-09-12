# Optical Reader & Math Solver — Session Changelog

Everything done to `bot_core.py`, `gui.py`, `launch_solver.bat`, `README.md`,
`.gitignore`, and `optical_lut.json` in this session, in one place.

## Files in this download

| File | What changed |
|---|---|
| `bot_core.py` | Most of the real work — see below |
| `gui.py` | New Advanced-section controls, safety-net fix, main-loop wiring |
| `launch_solver.bat` | `--user` pip fallback, `certifi` added |
| `README.md` | Documents all of the below, plus a Troubleshooting section |
| `.gitignore` | Added `ocr_captures/` |
| `optical_lut.json` | Two bad cached answers fixed (see below) |

Not touched: `optical_coords.json` (machine-specific, never safe to share
across machines), `backup.py`, `logger.py`.

---

## Bugs fixed

### 1. `÷` misread as `+`
EasyOCR sometimes reads the on-screen `÷` glyph as `+`. Rather than a
blanket `+`→`/` text rule (which was tried once before, found to be wrong
far more often than right, and removed), the pixels under every reported
`+` are checked against the glyph's own bounding box: a genuine plus is
one connected blob (a cross); `÷` is two dot-shaped blobs stacked with a
gap, optionally with a wider bar between them. Only corrected to `/` on
that visual evidence.

`is_division_glyph()` checks both Otsu threshold polarities and requires
a compact blob in the top third AND bottom third, with an optional
middle bar that must be visibly wider than the dots to count as
confirmation. Ambiguous cases are left as `+` rather than guessed.

### 2. `4x4` / `8x8` (and direct `÷`) not solving
A regression introduced earlier in this same session: when
`select_math_ocr_text()` was added to filter out noise tokens (stray
dates/labels), its operator-detection check only recognized the four
literal characters `+ - * /`. Your `normalise()` pipeline has always
handled `×`, `x`, `X`, `÷`, and `:` too — but the new candidate filter
didn't know that, so it silently rejected any candidate using one of
those as "no operator found" before it ever reached the solver. Fixed by
mapping every character `normalise()` understands to its canonical
operator before checking it.

### 3. Click regression (from an earlier debugging session)
Historically, `target_hwnd` was captured once on Resume, which could get
stuck pointing at the GUI's own window if focus passed through it. Fixed
before this session by tracking the foreground window continuously on
every poll instead.

### 4. Silent click failures — `fast_click()` had no verification
`SetCursorPos`'s return value was never checked, and there was zero delay
between clicks. A silently-failed cursor move (e.g. blocked input to an
elevated target window) or a click firing faster than the target UI could
register looked identical to a successful click from the console's
perspective. Fixed: verify with `GetCursorPos` that the cursor actually
landed, fall back once to `pyautogui.moveTo`, raise rather than click
blind if it still didn't land. `KEY_PRESS_DELAY`/`POST_ANSWER_DELAY`
raised from 0 to 0.025s (tunable constants).

### 5. LUT contained two bad cached answers
Audited all 52 entries against the real solver. `"63/9"` was cached as
`72` (should be `7`) — corrected. `"2/19"` was cached as `21`, but `2/19`
isn't even an integer (~0.105) — removed entirely, since there's no way
to know what the "real" intended expression was for a key that doesn't
solve to anything sensible. Everything else in the LUT checked out
correct.

### 6. "10000 errors" in VS Code
Caused by `pip install` needing admin rights to write to Python's shared
site-packages — on a locked-down school/lab account without admin, the
install fails or partially fails, leaving imports unresolved, which
Pylance then flags on every single downstream usage across the ~2,500-line
codebase. Fixed: `launch_solver.bat` now retries with `pip install --user`
automatically if the system-wide install fails from a permissions error.

### 7. README `demo.gif`
Went back and forth on this one — worth being upfront about. Removed it
first based on a zip download not containing the file, then confirmed
directly against the **live** GitHub repo that `demo.gif` does exist
there and renders fine (the zip just doesn't include it, likely a
"Download ZIP" quirk with binary assets). Restored the line. Current
README has it back in.

---

## New features

### `answer_clicks_enabled` / `auto_sequence_enabled` (was one `automation_enabled`)
Two separate concepts that used to share one flag: whether a solved
answer may be submitted on the keypad, vs. whether the optional AUTO
1/2/3 delayed sequence may run. Splitting them mattered because the
confirmation-failure safety net (3 consecutive unconfirmed clicks →
auto-disable) needs to specifically stop answer-submission clicking, not
just the bonus sequence — a naive rename would have left the safety net
protecting the wrong thing. Verified directly (not just asserted) that
Pause already independently gates both systems and wasn't affected by
this split. The single "Automation" button UI is unchanged — still
controls both together, same as before from your perspective.

### `select_math_ocr_text()` — candidate selection
When the capture box catches more than just the question (a stray date,
label, or fragment), picks the most plausible expression out of the OCR
tokens instead of naively joining everything. Filters by enabled
operators, digit-group count, and date-shape, then only considers
candidates that actually solve, preferring the shortest solvable span.
`enabled_operations` is a **filter only** — it rejects a candidate using
a disabled operator, it never converts one operator into another (an
earlier version of this idea had a regression doing exactly that, which
was deliberately not carried over).

Tokens are now also sorted top-to-bottom/left-to-right by bounding box
before candidates are built, since EasyOCR's return order isn't
guaranteed to match reading order.

### "Known Operations" checkboxes (Advanced)
+, −, ×, ÷ toggles that drive `enabled_operations` above.

### "Save OCR captures" toggle (Advanced, off by default)
Writes the original + processed image for each solved question to
`ocr_captures/`, for diagnosing OCR mistakes. Off by default so a long
unattended run doesn't accumulate images without bound.

### "Verify LUT" button (Advanced)
Re-checks every cached answer against the real solver, auto-corrects
wrong-but-valid entries, removes non-integer ones, shows a live
`N entries — X valid, Y corrected, Z removed` readout. This is what
would have caught bug #5 automatically going forward.

### `certifi` support
EasyOCR's first-run model download can fail with an SSL error on
machines with a stale certificate store. If `certifi` is installed, its
CA bundle is used automatically; if not, nothing changes. Fully optional,
graceful fallback either way.

---

## Testing performed

Everything above was tested by actually importing and running the real
`bot_core.py`/`gui.py` in a Linux sandbox — mocking only what's
Windows-specific or unavailable (`easyocr`, `pyautogui`, `pynput`,
`ctypes.windll`, `mss`), running the real Tkinter GUI under a virtual
display — rather than testing simplified reimplementations. Confirmed
working end-to-end: the division-glyph classifier, candidate selection
against the real `normalise()`/`solve_algebra()` (including the exact
`4x4` and direct-`÷` cases you reported), `verify_lut()`'s correction and
persistence, `fast_click()`'s verification and fallback path, all three
`click_answer()` gates, the `_can_auto()` gates, the confirmation safety
net actually disabling the correct flag (and `click_answer()` genuinely
refusing afterward, not just a flag changing), the Known Operations
checkboxes actually affecting candidate selection, and the OCR capture
toggle defaulting off and not touching disk while off.

**Not testable here, needs your real machine:** actual EasyOCR
recognition against your real screen/font, real mouse movement, and the
overlay rendering itself (Windows-only transparency).

---

## Known limitations (documented in the README)

- **Negative answers can't be submitted.** No minus-sign key exists on
  the keypad. The answer is computed and shown correctly, just never
  clicked. Console shows `[CORE] [SKIP] '-N' has unmapped chars`.
- **Click automation needs the target window focused.** Console shows
  `[CORE] [SKIP] Target window not focused` when skipped for this reason.
- **Repeated unconfirmed clicks auto-disable answer clicking** via the
  safety net described above.

## Deliberately not done this session

- Alternate OCR preprocessing (2× upscale, keep grayscale) and EasyOCR
  threshold tuning — both need real-screen benchmarking that can't be
  done blind; current (faster, working) pipeline kept as-is.
- Confidence-weighted candidate scoring, adaptive second-pass OCR,
  temporal frame-to-frame voting, a diagnostic panel, and an Advanced
  section visual reorganization — all reasonable ideas from a later
  review, not implemented since each needs either real-machine
  threshold-tuning or your input on what you actually want the UI to
  look like.
