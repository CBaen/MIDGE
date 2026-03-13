"""CircuitBreaker — per-source API failure protection for the sensing pipeline.

Automatically pauses data sources that start erroring, protecting API keys
from bans caused by hammering failing endpoints.

Three states per source:
  CLOSED    — healthy; calls are allowed through
  OPEN      — failing; calls are blocked until cooldown expires
  HALF_OPEN — testing recovery; one probe call allowed

Transition rules:
  CLOSED -> OPEN      : 3 consecutive failures
  OPEN -> HALF_OPEN   : after cooldown expires
  HALF_OPEN -> CLOSED : probe call succeeded (source recovers)
  HALF_OPEN -> OPEN   : probe call failed (cooldown doubles)

Cooldown schedule: 60s → 120s → 240s → 480s → 960s → cap 1800s (30 min)

Thread-safe: all state mutations are protected by a threading.Lock since
the sensing pipeline dispatches fetches via ThreadPoolExecutor.

Usage:
    cb = CircuitBreaker()
    if cb.can_call("sec_form4"):
        try:
            signals = fetch_sec_form4(...)
            cb.record_success("sec_form4")
        except Exception as e:
            cb.record_failure("sec_form4", str(e))
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("midge.market.circuit_breaker")

# State constants
_CLOSED = "CLOSED"
_OPEN = "OPEN"
_HALF_OPEN = "HALF_OPEN"

_FAILURE_THRESHOLD = 3          # consecutive failures before OPEN
_BASE_COOLDOWN = 60.0           # seconds for first OPEN cooldown
_MAX_COOLDOWN = 1800.0          # 30 minutes cap
_STATE_FILE = Path("data/market/circuit_breaker_state.json")


def _next_cooldown(current: float) -> float:
    """Double the current cooldown, capped at max."""
    return min(current * 2.0, _MAX_COOLDOWN)


class CircuitBreaker:
    """Per-source circuit breaker for the MIDGE sensing pipeline.

    Instantiated once and stored on ctx.circuit_breaker so every sensing
    component shares the same state.
    """

    def __init__(self, state_file: Optional[Path] = None) -> None:
        self._state_file = Path(state_file) if state_file else _STATE_FILE
        self._lock = threading.Lock()
        # Per-source record:
        # { "state": CLOSED|OPEN|HALF_OPEN,
        #   "consecutive_failures": int,
        #   "total_failures": int,
        #   "cooldown": float (seconds),
        #   "tripped_at": float (epoch seconds, or None),
        #   "last_error": str }
        self._sources: Dict[str, dict] = {}
        self._load_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def can_call(self, source_name: str) -> bool:
        """Return True if the source is allowed to make a call right now."""
        with self._lock:
            rec = self._sources.get(source_name)
            if rec is None:
                return True  # Unknown source — allow (first call ever)

            state = rec["state"]

            if state == _CLOSED:
                return True

            if state == _OPEN:
                elapsed = time.time() - (rec["tripped_at"] or 0.0)
                remaining = rec["cooldown"] - elapsed
                if remaining > 0:
                    logger.info(
                        "CircuitBreaker: blocking %s (OPEN, %.0fs remaining)",
                        source_name, remaining,
                    )
                    return False
                # Cooldown expired — promote to HALF_OPEN for a probe
                rec["state"] = _HALF_OPEN
                logger.info(
                    "CircuitBreaker: %s HALF_OPEN — probe call allowed",
                    source_name,
                )
                self._persist_state_locked()
                return True

            if state == _HALF_OPEN:
                # Allow exactly one probe call; subsequent calls blocked until resolved
                return True

            return True  # Unknown state — safe default

    def record_success(self, source_name: str) -> None:
        """Signal that a fetch completed successfully. Resets failure count."""
        with self._lock:
            rec = self._sources.get(source_name)
            if rec is None:
                # First seen — no need to persist
                return

            prev_state = rec["state"]
            if prev_state in (_CLOSED,) and rec["consecutive_failures"] == 0:
                return  # Already healthy, nothing to do

            recovered = prev_state in (_OPEN, _HALF_OPEN)
            trip_duration = (
                time.time() - (rec["tripped_at"] or time.time())
                if recovered else 0.0
            )
            rec["state"] = _CLOSED
            rec["consecutive_failures"] = 0
            rec["cooldown"] = _BASE_COOLDOWN  # Reset for next trip
            rec["tripped_at"] = None

            if recovered:
                logger.info(
                    "CircuitBreaker: %s recovered after %.0fs cooldown",
                    source_name, trip_duration,
                )

            self._persist_state_locked()

    def record_failure(self, source_name: str, error: Optional[str] = None) -> None:
        """Signal that a fetch failed. May trip the breaker to OPEN."""
        with self._lock:
            rec = self._sources.setdefault(source_name, {
                "state": _CLOSED,
                "consecutive_failures": 0,
                "total_failures": 0,
                "cooldown": _BASE_COOLDOWN,
                "tripped_at": None,
                "last_error": None,
            })

            rec["consecutive_failures"] += 1
            rec["total_failures"] += 1
            rec["last_error"] = error

            state = rec["state"]

            if state == _HALF_OPEN:
                # Probe failed — double cooldown and go back to OPEN
                rec["cooldown"] = _next_cooldown(rec["cooldown"])
                rec["state"] = _OPEN
                rec["tripped_at"] = time.time()
                logger.warning(
                    "CircuitBreaker: %s OPEN (probe failed). Cooldown: %.0fs",
                    source_name, rec["cooldown"],
                )
                self._persist_state_locked()
                return

            if rec["consecutive_failures"] >= _FAILURE_THRESHOLD and state == _CLOSED:
                rec["state"] = _OPEN
                rec["tripped_at"] = time.time()
                logger.warning(
                    "CircuitBreaker: %s OPEN after %d consecutive failures. Cooldown: %.0fs",
                    source_name, rec["consecutive_failures"], rec["cooldown"],
                )
                self._persist_state_locked()

    def get_status(self) -> dict:
        """Return all source states. Safe to call from any thread."""
        with self._lock:
            result = {}
            now = time.time()
            for name, rec in self._sources.items():
                state = rec["state"]
                remaining = 0.0
                if state == _OPEN and rec["tripped_at"] is not None:
                    remaining = max(0.0, rec["cooldown"] - (now - rec["tripped_at"]))
                result[name] = {
                    "state": state,
                    "consecutive_failures": rec["consecutive_failures"],
                    "total_failures": rec["total_failures"],
                    "cooldown_remaining_s": round(remaining, 1),
                    "last_error": rec.get("last_error"),
                }
            return result

    def get_blocked_sources(self) -> List[str]:
        """Return source names currently in OPEN state (calls blocked)."""
        with self._lock:
            now = time.time()
            blocked = []
            for name, rec in self._sources.items():
                if rec["state"] == _OPEN:
                    elapsed = now - (rec["tripped_at"] or 0.0)
                    if elapsed < rec["cooldown"]:
                        blocked.append(name)
            return blocked

    # ------------------------------------------------------------------
    # Persistence (atomic write — same pattern as deception_detector)
    # ------------------------------------------------------------------

    def _persist_state_locked(self) -> None:
        """Write state to disk. Called while _lock is held."""
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_file.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(self._sources, indent=2, default=str),
                encoding="utf-8",
            )
            tmp.replace(self._state_file)
        except Exception:
            logger.debug("CircuitBreaker: failed to persist state", exc_info=True)

    def _load_state(self) -> None:
        """Load persisted state from disk on init. Non-blocking."""
        if not self._state_file.exists():
            return
        try:
            raw = json.loads(self._state_file.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                self._sources = raw
                logger.info(
                    "CircuitBreaker: loaded state for %d sources from disk",
                    len(self._sources),
                )
        except Exception:
            logger.debug("CircuitBreaker: failed to load state — starting fresh", exc_info=True)
