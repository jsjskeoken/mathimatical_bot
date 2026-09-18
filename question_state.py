"""
question_state.py — Layered question/retry/click state machine.

Replaces the old "frame hash == question state" concept with explicit layers:

  A. VISUAL STATE      exact-pixel digest. Answers "did literally these pixels
                       change?". Used ONLY for click confirmation and as a
                       freshness signal. NEVER used to suppress retries.
  B. QUESTION IDENTITY semantic fingerprint over the CANONICAL expression
                       (after correction/normalisation), plus the enabled-
                       operation set, fast/standard mode, and capture-region
                       identity. Answers "is OCR referring to the same
                       question?". Drives caches and retry scheduling.
  C. SOLVE STATE       per-question "do we already know the answer this round".
  D. CLICK STATE       per-question "was a click issued / confirmed / blocked".
  E. RETRY STATE       per-question time-based, bounded retry scheduling with
                       backoff. This is what guarantees an unchanged question
                       is retried after a delay instead of being suppressed
                       forever by an unchanged frame hash.

  F. PAUSE STATE    -> core.paused (single gate in the GUI loop; preview does
                       not go through this gate)
  G. PREVIEW STATE  -> core.preview_enabled (single gate on the preview path)
  H. AUTO SEQ STATE -> core.auto_sequence_enabled (single gate in _can_auto)

This module is pure stdlib: no numpy/cv2/tkinter/Windows dependencies, so the
whole machine is unit-testable on any OS and importable from bot_core safely.

Timing: all deadlines/elapsed values use time.monotonic() — immune to wall
clock changes (NTP jumps, manual clock edits) which would otherwise corrupt
retry scheduling and confirmation timeouts.
"""

import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

# ── Attempt / outcome vocabulary ─────────────────────────────────────────────
# One vocabulary shared by the retry engine, the GUI loop and the logs, so
# every retry decision can name exactly why it happened.

OUTCOME_OCR_EMPTY     = "ocr_empty"        # OCR produced nothing usable
OUTCOME_UNSOLVED      = "unsolved"         # text found, no integer answer
OUTCOME_NOT_ENABLED   = "not_enabled"      # answer known; clicking disabled
OUTCOME_WRONG_WINDOW  = "wrong_window"     # answer known; target not focused
OUTCOME_UNMAPPED      = "unmapped_answer"  # answer known; no keypad key — terminal
OUTCOME_CLICKED       = "clicked"          # click issued; awaiting confirmation
OUTCOME_ERROR         = "click_error"      # click attempt raised
OUTCOME_CONFIRMED     = "confirmed"        # screen changed after click — success
OUTCOME_UNCONFIRMED   = "unconfirmed"      # deadline elapsed; screen unchanged


# ── Retry policy — every knob in one place, nothing magic inline ────────────

@dataclass(frozen=True)
class RetryPolicy:
    # Delay before the FIRST retry of an unchanged frame's question.
    same_frame_retry_delay: float = 0.25
    # Base delay for retries after the first (before backoff multiplier).
    same_question_retry_delay: float = 0.50
    # Multiplier applied per consecutive retry tier.
    retry_backoff: float = 2.0
    # Cap for a single retry delay.
    max_backoff: float = 4.0
    # Maximum click/processing retries per semantic question. After this,
    # clicking stops for that question (until its identity changes) but
    # re-validation OCR continues at max_backoff — a question is never
    # permanently suppressed, but it also can never click-spam.
    max_retries_per_question: int = 8
    # After a CONFIRMED click, suppress any further click of the SAME
    # question for at least this long (belt-and-braces against duplicate
    # clicks while the screen is still transitioning).
    success_cooldown: float = 1.0
    # How long a question's fingerprint must have been OFF-screen before a
    # reappearance counts as a NEW episode (fresh click allowed) rather
    # than a flicker of the already-completed frame (no re-click).
    question_reappear_reset: float = 2.0
    # How long to wait after CLICKED for the frame digest to change.
    click_confirm_timeout: float = 0.5
    # Consecutive unconfirmed clicks before the safety net disables answer
    # clicking.
    unconfirmed_threshold: int = 3
    # Frame-answer cache entry lifetime (seconds). Stale entries expire.
    frame_cache_ttl: float = 30.0
    # Per-question runtime state TTL + cap (housekeeping; questions that
    # stopped appearing lose their state instead of leaking).
    question_state_ttl: float = 120.0
    question_state_max_entries: int = 64
    # ── Unresolved visual identity (OCR-flicker loophole) ─────────────
    # When OCR yields no canonical expression, question identity comes
    # from the QUESTION-REGION PIXELS (see bot_core.visual_signature),
    # never from the raw garbage text. Two sightings belong to the same
    # unresolved episode while their aligned, dead-zoned signature
    # distance stays at or below this threshold (measured separation:
    # same-question noise ≤ 0.0043, one-digit change ≥ 0.0090, full
    # question change ≥ 0.0171 — see docs/REDESIGN_REPORT.md §2.5).
    unresolved_match_threshold: float = 0.0065
    # Signatures kept per episode (anchor + tolerated blink/jitter
    # variants). Matching checks the minimum distance over ALL of them.
    unresolved_recent_max: int = 8
    # An episode unseen for longer than this can never match again — the
    # next unreadable sighting starts a fresh episode with a fresh budget.
    unresolved_episode_timeout: float = 30.0
    # Two signatures whose ink is below this fraction of cells are treated
    # as identical (a nearly blank region cannot define an identity, and
    # must not churn episodes on a blinking cursor alone).
    unresolved_min_ink: float = 0.015


# ── Small state objects ──────────────────────────────────────────────────────

@dataclass
class VisualObservation:
    changed: bool
    digest: str


@dataclass
class QuestionObservation:
    fingerprint: str
    canonical: str
    new_question: bool


@dataclass
class ProcessDecision:
    allowed: bool
    reason: str


@dataclass
class QuestionRuntime:
    """Layered C/D/E state for ONE semantic question."""
    fingerprint: str
    canonical: str
    first_seen: float = 0.0
    last_attempt: float = 0.0
    attempts: int = 0                      # click/processing retries used
    backoff_tier: int = 0                  # consecutive retries at this tier
    next_allowed: float = 0.0              # monotonic time of next permitted attempt
    # C. solve state
    solved: bool = False                   # answer known this round
    answer: Optional[int] = None
    source: Optional[str] = None
    # D. click state
    clicked: bool = False
    awaiting_confirmation: bool = False
    click_confirmed: bool = False
    done: bool = False                     # completed; never re-click this round
    exhausted: bool = False                # max retries used; clicks stop, OCR revalidates
    # misc
    last_outcome: str = ""
    inactive_since: Optional[float] = None   # fingerprint left the screen
    last_ocr_confidence: Optional[float] = None
    last_ml_confidence: Optional[float] = None


def frame_digest(bgra_bytes) -> str:
    """
    EXACT FRAME DIGEST (layer A). BLAKE2b-128 over the raw BGRA buffer.

    Deliberate note: switching MD5 -> BLAKE2b is NOT the architectural fix
    (any exact-pixel digest has the same suppression problem if used as a
    gate); BLAKE2b is chosen simply because it is fast, modern and needs no
    third-party dependency. This digest is used only where exact pixels
    matter: click confirmation and literal screen-change detection.
    """
    h = hashlib.blake2b(bgra_bytes, digest_size=16)
    return h.hexdigest()


def semantic_fingerprint(canonical_expr: str, raw_ocr: str, context_key: str) -> str:
    """
    SEMANTIC QUESTION KEY (layer B).

    canonical_expr non-empty  -> identity is the question itself: "7+6",
    "7 + 6", "7×?6"-style readings all converge BEFORE this point because
    they were canonicalised through the normalise() pipeline (correction ->
    token ordering -> operator correction -> hallucination cleanup ->
    normalise). Raw OCR text is never hashed as identity when a canonical
    form exists.

    canonical_expr empty      -> UNRESOLVED. The raw-text hash that used to
    live here is retained ONLY as a low-level building block: the machine
    routes unreadable frames through observe_unresolved() instead, whose
    identity is the question-region PIXELS (an OCR-flicker loophole made
    different garbage strings mint independent retry budgets — garbage A,
    B, C from one unchanged question must share ONE episode/ONE budget).
    Nothing in the GUI is expected to call this branch again.

    context_key carries everything that changes what the question MEANS to
    the solver: enabled operations, fast/standard mode, capture region — so
    a coordinate-profile switch can never collide with the old profile's
    question state.
    """
    h = hashlib.blake2b(digest_size=16)
    if canonical_expr:
        h.update(b"Q\x00")
        h.update(canonical_expr.encode("utf-8", "replace"))
    else:
        h.update(b"R\x00")
        h.update(raw_ocr.encode("utf-8", "replace"))
    h.update(b"\x00")
    h.update(context_key.encode("utf-8", "replace"))
    return h.hexdigest()


# ── Unresolved visual identity (layer B' — pixel evidence) ───────────────────
# Signatures are produced by bot_core.visual_signature() from the SAME
# preprocessed question-region image EasyOCR already consumed — no extra
# capture, no new dependency. They arrive here as plain bytes (one uint8
# cell value each, ink-normalized to minority polarity), so this module
# stays pure-stdlib and unit-testable anywhere.

SIG_GRID_W = 96
SIG_GRID_H = 16
_SIG_DEADZONE = 48          # per-cell delta below this = rendering noise
_SIG_SHIFT_X = 2            # alignment search, cells
_SIG_SHIFT_Y = 1
_SIG_SPAN = 255 - _SIG_DEADZONE
# (0,0) first — identical/near-identical frames exit after one scan
_SHIFT_ORDER = [(0, 0)] + [(dy, dx)
                           for dy in range(-_SIG_SHIFT_Y, _SIG_SHIFT_Y + 1)
                           for dx in range(-_SIG_SHIFT_X, _SIG_SHIFT_X + 1)
                           if (dy, dx) != (0, 0)]


def signature_ink(sig: bytes) -> float:
    """Fraction of cells counted as ink (polarity already normalized)."""
    if not sig:
        return 0.0
    return sum(1 for v in sig if v >= 128) / len(sig)


def signature_distance(a: bytes, b: bytes, min_ink: float = 0.015,
                       bail_below: Optional[float] = None) -> float:
    """
    Aligned, dead-zoned L1 distance between two signatures, in 0.0..1.0.

    The distance is minimized over a small integer-cell alignment search
    (±SIG_SHIFT_X x, ±SIG_SHIFT_Y y, tried (0,0) first): whole-image render
    jitter must not count as a content change. The dead-zone absorbs
    sub-cell edge differences (partial glyph coverage from
    antialiasing/subpixel shifts). Measured classes at SIG_GRID 96x16,
    deadzone 48 (REFERENCE render: bold 30px sans — absolute distances
    vary with the actual platform font; the threshold CONTRACT is what
    must hold everywhere, see tests/test_unresolved_identity.py):
        same question + noise/blur/1-3px shift/cursor blink: 0.0000-0.0043
        one digit changed:                                   0.0090
        full question change:                                0.0171-0.0354

    min_ink: two signatures whose ink fraction is below this are treated
    as identical (blank regions define no identity — see policy knob
    unresolved_min_ink; the QSM passes its policy value here).

    bail_below: early-exit for match decisions — return as soon as a shift
    scores at or below this value (the caller only needs to know "within
    threshold", not the exact minimum). Keeps the hot per-frame path to a
    single grid scan (~1.7 ms in pure stdlib Python).
    """
    if len(a) != len(b) or len(a) != SIG_GRID_W * SIG_GRID_H:
        return 1.0
    if signature_ink(a) < min_ink and signature_ink(b) < min_ink:
        # Two essentially blank regions define no identity at all — treat
        # as identical so a blank/cursor-blink area cannot churn episodes.
        return 0.0

    best = 1.0
    w, h = SIG_GRID_W, SIG_GRID_H
    for dy, dx in _SHIFT_ORDER:
        total = 0
        y0a, y1a = max(0, dy), min(h, h + dy)
        x0a, x1a = max(0, dx), min(w, w + dx)
        y0b, y1b = max(0, -dy), min(h, h - dy)
        x0b, x1b = max(0, -dx), min(w, w - dx)
        for y in range(y1a - y0a):
            ra = (y0a + y) * w
            rb = (y0b + y) * w
            for x in range(x1a - x0a):
                va = a[ra + x0a + x]
                vb = b[rb + x0b + x]
                d = va - vb if va > vb else vb - va
                if d > _SIG_DEADZONE:
                    total += d - _SIG_DEADZONE
        dist = total / (_SIG_SPAN * (x1a - x0a) * (y1a - y0a))
        if dist < best:
            best = dist
            if best == 0.0 or (bail_below is not None
                               and best <= bail_below):
                return best
    return best


def unresolved_fingerprint(anchor: bytes, context_key: str) -> str:
    """Stable episode identity: derived from the ANCHOR pixels + context —
    never from OCR text. A new episode anchors on different pixels and
    therefore mints a different fingerprint (fresh budget); every sighting
    inside the episode reuses the same fingerprint (shared budget)."""
    h = hashlib.blake2b(digest_size=16)
    h.update(b"U\x00")
    h.update(anchor)
    h.update(b"\x00")
    h.update(context_key.encode("utf-8", "replace"))
    return h.hexdigest()


@dataclass
class UnresolvedEpisode:
    """One 'the screen shows a question OCR cannot read' episode.

    anchor    first signature of the episode (fixed — no drift accrual)
    recent    anchor + up to unresolved_recent_max-1 tolerated variants
              (cursor blink, 2-frame animations); all are matched against
    fingerprint  the stable identity this episode reports to the machine
    """
    anchor: bytes
    recent: List[bytes]
    fingerprint: str
    first_seen: float
    last_seen: float


# ── TTL frame cache (layer A' — "these exact pixels recently meant this") ───

@dataclass
class FrameCacheEntry:
    answer: int
    source: str
    fingerprint: str            # semantic identity this frame was solved under
    canonical: str
    created: float


class TTLFrameCache:
    """
    FRAME CACHE: "these exact pixels were recently associated with this
    answer." Entries carry timestamps and expire; nothing here can suppress
    retries (the retry engine does not consult it) and nothing here is
    served after its TTL. Only successful answers are ever stored.
    """

    def __init__(self, ttl: float, max_entries: int = 500,
                 clock: Callable[[], float] = time.monotonic):
        self.ttl = ttl
        self.max_entries = max_entries
        self._clock = clock
        self._entries: Dict[str, FrameCacheEntry] = {}

    def get(self, digest: str) -> Optional[FrameCacheEntry]:
        entry = self._entries.get(digest)
        if entry is None:
            return None
        if self._clock() - entry.created > self.ttl:
            del self._entries[digest]
            return None
        return entry

    def put(self, digest: str, answer: int, source: str,
            fingerprint: str, canonical: str) -> None:
        if len(self._entries) >= self.max_entries:
            oldest = next(iter(self._entries))
            del self._entries[oldest]
        self._entries[digest] = FrameCacheEntry(
            answer=answer, source=source, fingerprint=fingerprint,
            canonical=canonical, created=self._clock())

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


# ── Throttled debug logging ──────────────────────────────────────────────────
# Per-key minimum interval so a 10 ms poll loop can never flood the console.
# Enable with BOT_DEBUG=1 (or any non-empty value) in the environment.

_DEBUG_ENABLED = bool(os.environ.get("BOT_DEBUG"))
_debug_last: Dict[str, float] = {}


def debug_throttled(key: str, msg: str, interval: float = 0.25,
                    clock: Callable[[], float] = time.monotonic) -> None:
    if not _DEBUG_ENABLED:
        return
    now = clock()
    last = _debug_last.get(key)
    if last is not None and (now - last) < interval:
        return
    _debug_last[key] = now
    print(f"[STATE] {msg}")


# ── The state machine ────────────────────────────────────────────────────────

class QuestionStateMachine:
    """
    Owns layers A–E for all live questions.

    The GUI loop drives it in this order every cycle:

        obs  = qsm.observe_visual(digest, now)          # A
        qobs = qsm.observe_question(fp, canon, raw, now) # B   (after OCR,
                                                         #     or from a
                                                         #     frame-cache hit)
        dec  = qsm.should_process(now)                   # E gate
        ... OCR / solve / click ...
        qsm.record_outcome(...)                          # C/D/E update
    """

    def __init__(self, policy: Optional[RetryPolicy] = None,
                 clock: Callable[[], float] = time.monotonic):
        self.policy = policy or RetryPolicy()
        self._clock = clock
        self._last_digest: Optional[str] = None
        self._runtime: Dict[str, QuestionRuntime] = {}
        self._current_fp: Optional[str] = None
        self._unresolved: Optional[UnresolvedEpisode] = None
        self.pending_confirm_digest: Optional[str] = None
        self.pending_confirm_deadline: Optional[float] = None
        self.consecutive_unconfirmed: int = 0

    # ── A. visual layer ──────────────────────────────────────────────────────

    def observe_visual(self, digest: str, now: Optional[float] = None) -> VisualObservation:
        now = self._clock() if now is None else now
        changed = (self._last_digest is not None and digest != self._last_digest)
        self._last_digest = digest
        if changed:
            # A visual change is evidence the screen moved on: reset the
            # backoff TIER (next retry sooner) but NOT the per-question
            # attempt budget — animation jitter must not buy unlimited
            # retries.
            rt = self._runtime.get(self._current_fp)
            if rt is not None and not rt.done:
                rt.backoff_tier = 0
                if rt.next_allowed > now:
                    rt.next_allowed = now + self.policy.same_frame_retry_delay
        return VisualObservation(changed=changed, digest=digest)

    @property
    def last_digest(self) -> Optional[str]:
        return self._last_digest

    # ── B. question identity layer ───────────────────────────────────────────

    def observe_question(self, fingerprint: str, canonical: str, raw: str,
                         now: Optional[float] = None) -> QuestionObservation:
        """
        Observe a question sighting. Identity changes drive:
          - retry-budget reset (a genuinely different question starts fresh)
          - episode lifecycle: a COMPLETED question seen again after a brief
            absence (flicker / animation) stays completed — no re-click;
            the same question reappearing after it has been off-screen for
            longer than policy.question_reappear_reset starts a NEW episode
            (fresh click allowed — it is a genuinely new instance).
        """
        now = self._clock() if now is None else now
        if canonical:
            # A readable question resolves (and closes) any unresolved
            # visual episode: the next unreadable frame must anchor fresh.
            self._unresolved = None
        new_question = (fingerprint != self._current_fp)
        if new_question:
            # The previously-current question just left the screen.
            old_rt = self._runtime.get(self._current_fp)
            if old_rt is not None and old_rt.inactive_since is None:
                old_rt.inactive_since = now
            self._current_fp = fingerprint
            rt = self._runtime.get(fingerprint)
            if rt is None:
                rt = QuestionRuntime(fingerprint=fingerprint, canonical=canonical,
                                     first_seen=now)
                self._runtime[fingerprint] = rt
            else:
                inactive_for = (now - rt.inactive_since
                                if rt.inactive_since is not None else 0.0)
                rt.inactive_since = None
                if rt.done and inactive_for > self.policy.question_reappear_reset:
                    # Reappearance after other content — a new episode of
                    # this question: fresh click/solve/retry lifecycle.
                    rt.done = False
                    rt.clicked = False
                    rt.awaiting_confirmation = False
                    rt.click_confirmed = False
                    rt.exhausted = False
                    rt.solved = False
                    rt.answer = None
                    rt.source = None
                    rt.attempts = 0
                    rt.backoff_tier = 0
                    rt.first_seen = now
            self._prune(now)
        return QuestionObservation(fingerprint=fingerprint, canonical=canonical,
                                   new_question=new_question)

    @property
    def current_fingerprint(self) -> Optional[str]:
        return self._current_fp

    # ── B'. unresolved visual identity (OCR-flicker loophole) ────────────────

    def observe_unresolved(self, signature: bytes, context_key: str,
                           now: Optional[float] = None) -> str:
        """
        Identity for a frame whose OCR produced NO canonical expression.

        Returns the episode fingerprint to feed observe_question(). The
        fingerprint is derived from the episode's ANCHOR PIXELS + context,
        never from the garbage text, so alternating garbage readings from
        the same screen share ONE runtime and ONE bounded retry budget.

        Same episode  : min aligned signature distance over the recent ring
                        ≤ policy.unresolved_match_threshold (small noise,
                        blink, render jitter) → the current fingerprint is
                        returned unchanged.
        New episode   : distance above the threshold vs EVERY recent
                        signature (genuinely changed pixels) or the episode
                        timed out → the old episode is closed and a new
                        anchor mints a new fingerprint (fresh budget via
                        observe_question's new-question path).
        """
        now = self._clock() if now is None else now
        p = self.policy
        ep = self._unresolved
        if ep is not None:
            expired = (now - ep.last_seen) > p.unresolved_episode_timeout
            if not expired:
                # first hit within the threshold decides — no need for the
                # exact minimum (early-exit keeps the hot path cheap)
                match = None
                no_match_dists = []
                for s in ep.recent:
                    d = signature_distance(signature, s,
                                           p.unresolved_min_ink,
                                           bail_below=p.unresolved_match_threshold)
                    if d <= p.unresolved_match_threshold:
                        match = d
                        break
                    no_match_dists.append(d)
                if match is not None:
                    ep.last_seen = now
                    if signature not in ep.recent:
                        ep.recent.append(signature)
                        if len(ep.recent) > p.unresolved_recent_max:
                            # keep the anchor (index 0) and the newest
                            # variants; drop the oldest middle sample
                            del ep.recent[1]
                    return ep.fingerprint
                best = min(no_match_dists)
                debug_throttled(
                    "unresolved:new",
                    "unresolved identity changed — new visual episode "
                    f"(min distance {best:.4f} > {p.unresolved_match_threshold})")
            else:
                debug_throttled(
                    "unresolved:new",
                    "unresolved episode timed out — new visual episode")
        # First unreadable sighting, the previous episode expired, or the
        # pixels genuinely changed: anchor a new episode.
        fp = unresolved_fingerprint(signature, context_key)
        self._unresolved = UnresolvedEpisode(
            anchor=signature, recent=[signature], fingerprint=fp,
            first_seen=now, last_seen=now)
        return fp

    def runtime(self, fingerprint: Optional[str] = None) -> Optional[QuestionRuntime]:
        return self._runtime.get(self._current_fp if fingerprint is None else fingerprint)

    # ── E. retry gate ────────────────────────────────────────────────────────

    def should_process(self, now: Optional[float] = None) -> ProcessDecision:
        now = self._clock() if now is None else now
        rt = self.runtime()
        if rt is None:
            return ProcessDecision(True, "new_question")
        if rt.done:
            return ProcessDecision(False, "question_completed")
        if rt.awaiting_confirmation:
            return ProcessDecision(False, "awaiting_confirmation")
        if now >= rt.next_allowed:
            return ProcessDecision(True, "retry_due" if rt.attempts else "first_sighting")
        return ProcessDecision(False, f"backoff_{rt.next_allowed - now:.2f}s_remaining")

    # ── C/D/E. outcome recording ─────────────────────────────────────────────

    def record_outcome(self, outcome: str, now: Optional[float] = None,
                       answer: Optional[int] = None, source: Optional[str] = None,
                       ocr_confidence: Optional[float] = None,
                       ml_confidence: Optional[float] = None) -> QuestionRuntime:
        now = self._clock() if now is None else now
        rt = self.runtime()
        if rt is None:
            # Outcomes may arrive for a question never observed (defensive) —
            # create it so nothing is silently dropped.
            rt = QuestionRuntime(fingerprint=self._current_fp or "?", canonical="",
                                 first_seen=now)
            if self._current_fp:
                self._runtime[self._current_fp] = rt
        rt.last_outcome = outcome
        rt.last_attempt = now
        if ocr_confidence is not None:
            rt.last_ocr_confidence = ocr_confidence
        if ml_confidence is not None:
            rt.last_ml_confidence = ml_confidence

        if answer is not None:
            rt.solved = True
            rt.answer = answer
            rt.source = source

        if outcome == OUTCOME_CONFIRMED:
            rt.click_confirmed = True
            rt.awaiting_confirmation = False
            rt.done = True
            rt.next_allowed = now + self.policy.success_cooldown
        elif outcome == OUTCOME_UNCONFIRMED:
            rt.awaiting_confirmation = False
            self._schedule_retry(rt, now, failed=True)
        elif outcome in (OUTCOME_OCR_EMPTY, OUTCOME_UNSOLVED, OUTCOME_ERROR):
            self._schedule_retry(rt, now, failed=True)
        elif outcome in (OUTCOME_NOT_ENABLED, OUTCOME_WRONG_WINDOW):
            # Answer known; the click was not possible. Bounded retry so a
            # later re-enable / re-focus still leads to a click — but never
            # a rapid loop.
            self._schedule_retry(rt, now, failed=True)
        elif outcome == OUTCOME_UNMAPPED:
            # No keypad key can ever represent this answer — retrying is
            # pointless (Part 7: do not retry infinitely). Terminal.
            rt.done = True
        elif outcome == OUTCOME_CLICKED:
            rt.clicked = True
            rt.awaiting_confirmation = True
            rt.backoff_tier = 0
            rt.next_allowed = now + self.policy.click_confirm_timeout
        return rt

    def _schedule_retry(self, rt: QuestionRuntime, now: float, failed: bool) -> None:
        p = self.policy
        rt.attempts += 1
        if rt.attempts > p.max_retries_per_question:
            # Bounded-exhausted: clicks stop for this question, but OCR
            # re-validation continues at the capped interval so the question
            # can never be permanently suppressed (and can never spam).
            rt.exhausted = True
            rt.next_allowed = now + p.max_backoff
            return
        if rt.backoff_tier == 0:
            delay = p.same_frame_retry_delay
        else:
            delay = min(p.same_question_retry_delay * (p.retry_backoff ** (rt.backoff_tier - 1)),
                        p.max_backoff)
        rt.backoff_tier += 1
        rt.next_allowed = now + delay

    def record_attempt_now(self, now: Optional[float] = None) -> None:
        """Mark that processing ran right now (first sighting path)."""
        now = self._clock() if now is None else now
        rt = self.runtime()
        if rt is not None and rt.last_attempt == 0.0:
            rt.last_attempt = now

    # ── D. confirmation layer (driven by the GUI's digest watcher) ───────────

    def arm_confirmation(self, digest: str, now: Optional[float] = None) -> None:
        now = self._clock() if now is None else now
        self.pending_confirm_digest = digest
        self.pending_confirm_deadline = now + self.policy.click_confirm_timeout

    def drop_confirmation(self) -> None:
        self.pending_confirm_digest = None
        self.pending_confirm_deadline = None

    # ── Lifecycle helpers ────────────────────────────────────────────────────

    def on_pause(self) -> None:
        """
        Pausing stops watching for click confirmation (nothing is processed
        while paused, so the deadline would silently elapse and later read
        as a false 'unconfirmed'). The streak is deliberately reset too —
        a pause is a clean boundary.
        """
        self.drop_confirmation()
        self.consecutive_unconfirmed = 0

    def on_resume(self, now: Optional[float] = None) -> None:
        """
        Resuming must restore normal processing IMMEDIATELY: the current
        question is treated as freshly sighted (any remaining backoff from
        before the pause is not worth waiting out).
        """
        now = self._clock() if now is None else now
        rt = self.runtime()
        if rt is not None and not rt.done:
            rt.next_allowed = now
            rt.backoff_tier = 0

    def reset_round(self) -> None:
        """
        New round / configuration change: per-question runtime, retry
        budgets and the frame cache all describe the OLD round — drop them.
        (The persistent LUT is not part of this state and is untouched.)
        """
        self._runtime.clear()
        self._current_fp = None
        self._unresolved = None
        self.drop_confirmation()
        self.consecutive_unconfirmed = 0

    def _prune(self, now: float) -> None:
        p = self.policy
        if len(self._runtime) <= p.question_state_max_entries:
            return
        # Drop expired first, then oldest.
        expired = [fp for fp, rt in self._runtime.items()
                   if now - rt.first_seen > p.question_state_ttl
                   and fp != self._current_fp]
        for fp in expired:
            del self._runtime[fp]
        while len(self._runtime) > p.question_state_max_entries:
            oldest = min(self._runtime.items(),
                         key=lambda kv: kv[1].first_seen,
                         default=None)
            if oldest is None or oldest[0] == self._current_fp:
                break
            del self._runtime[oldest[0]]

    # ── Introspection (logging / tests) ──────────────────────────────────────

    def snapshot(self) -> Dict[str, dict]:
        return {fp: {
            "canonical": rt.canonical, "attempts": rt.attempts,
            "solved": rt.solved, "answer": rt.answer, "source": rt.source,
            "clicked": rt.clicked, "awaiting_confirmation": rt.awaiting_confirmation,
            "click_confirmed": rt.click_confirmed, "done": rt.done,
            "exhausted": rt.exhausted, "last_outcome": rt.last_outcome,
            "backoff_tier": rt.backoff_tier,
        } for fp, rt in self._runtime.items()}
