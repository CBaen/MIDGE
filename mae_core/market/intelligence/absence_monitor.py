"""Absence Monitor — detects when normally-active signal sources go silent.

Pattern recognition includes noticing what SHOULD be there but ISN'T.
This is fundamentally different from all other market intelligence systems,
which operate on signals that ARE present. Informed actors going silent
is one of the strongest asymmetric signals in markets.

Examples:
  - Congressional member who trades weekly goes silent before a vote
  - SEC Form 4 filings drop off for a normally-active company
  - COT data stops updating (CFTC reporting lag = data quality issue)

Design:
  - Cadence learned from arrival history (median inter-arrival time)
  - Bootstrap from signal archive on startup
  - AbsenceSignal fires when silence_ratio > threshold (2.5x expected cadence)
  - Source-specific overrides for known cadence patterns
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Minimum arrivals before cadence estimate is trusted
MIN_OBSERVATIONS = 5

# Default silence threshold: fire at 2.5x expected cadence
DEFAULT_ABSENCE_THRESHOLD = 2.5

# Source-specific expected cadence overrides (hours).
# Used as seed — actual cadence is learned from data when available.
_KNOWN_CADENCES: Dict[str, float] = {
    "congressional": 168.0,      # Weekly during session
    "senate": 168.0,             # Weekly during session
    "cot_positioning": 168.0,    # CFTC reports weekly (Fridays)
    "fred_macro": 336.0,         # FRED updates biweekly for many series
    "finra_short": 336.0,        # FINRA short interest is semi-monthly
    "sec_form4": 48.0,           # Form 4 filings arrive ~daily for active tickers
    "sec_form8k": 72.0,          # Form 8-K events vary, ~3 day average
    "finnhub_earnings": 720.0,   # Earnings are quarterly (~30 days)
}

# Maximum history entries per source (memory bound)
MAX_HISTORY = 100


@dataclass
class AbsenceSignal:
    """A detected absence — something that should be here but isn't."""
    source: str
    expected_hours: float
    actual_hours: float
    silence_ratio: float  # actual / expected — higher = more anomalous
    direction: str = "bearish"  # Silence from informed actors defaults bearish
    last_seen: Optional[datetime] = None


class AbsenceMonitor:
    """Tracks per-source signal cadence and detects unexpected silence.

    The cognitive strategy: pattern recognition includes noticing what
    SHOULD be there but ISN'T. This is fundamentally different from
    signal-present analysis — it detects the dog that didn't bark.
    """

    def __init__(
        self,
        event_bus=None,
        archive_reader=None,
        absence_threshold: float = DEFAULT_ABSENCE_THRESHOLD,
        min_observations: int = MIN_OBSERVATIONS,
    ):
        self._bus = event_bus
        self._archive_reader = archive_reader
        self._absence_threshold = absence_threshold
        self._min_observations = min_observations

        # Per-source state
        self._last_seen: Dict[str, datetime] = {}
        self._cadence: Dict[str, float] = {}  # source -> expected hours
        self._arrival_history: Dict[str, deque] = {}  # source -> recent timestamps

        # Seed known cadences
        for source, hours in _KNOWN_CADENCES.items():
            self._cadence[source] = hours

    def record_arrival(self, source: str, timestamp: Optional[datetime] = None) -> None:
        """Record that a signal arrived from this source.

        Called from the sensing hook when signals are ingested.
        Updates cadence estimates from inter-arrival times.
        """
        ts = timestamp or datetime.now()
        self._last_seen[source] = ts

        # Maintain arrival history
        if source not in self._arrival_history:
            self._arrival_history[source] = deque(maxlen=MAX_HISTORY)
        self._arrival_history[source].append(ts)

        # Update cadence estimate from history
        history = self._arrival_history[source]
        if len(history) >= 2:
            self._update_cadence(source, history)

    def _update_cadence(self, source: str, history: deque) -> None:
        """Recompute expected cadence from inter-arrival times.

        Uses median inter-arrival time (robust to outliers).
        """
        sorted_times = sorted(history)
        intervals = []
        for i in range(1, len(sorted_times)):
            delta = (sorted_times[i] - sorted_times[i - 1]).total_seconds() / 3600.0
            if delta > 0:
                intervals.append(delta)

        if not intervals:
            return

        # Median inter-arrival time
        intervals.sort()
        mid = len(intervals) // 2
        if len(intervals) % 2 == 0 and len(intervals) >= 2:
            median = (intervals[mid - 1] + intervals[mid]) / 2.0
        else:
            median = intervals[mid]

        if median > 0:
            self._cadence[source] = median

    def check_absences(self, now: Optional[datetime] = None) -> List[AbsenceSignal]:
        """Check all tracked sources for unexpected silence.

        Returns AbsenceSignal for each source that has been silent
        longer than absence_threshold * expected_cadence.
        """
        check_time = now or datetime.now()
        absences = []

        for source, last_ts in self._last_seen.items():
            expected_hours = self._cadence.get(source)
            if expected_hours is None:
                continue

            # Need enough observations to trust cadence estimate
            history = self._arrival_history.get(source, deque())
            if len(history) < self._min_observations:
                # But allow known-cadence sources with seed values
                if source not in _KNOWN_CADENCES:
                    continue

            actual_hours = (check_time - last_ts).total_seconds() / 3600.0
            if actual_hours <= 0:
                continue

            silence_ratio = actual_hours / expected_hours
            if silence_ratio >= self._absence_threshold:
                absences.append(AbsenceSignal(
                    source=source,
                    expected_hours=expected_hours,
                    actual_hours=actual_hours,
                    silence_ratio=silence_ratio,
                    last_seen=last_ts,
                ))

        # Publish to EventBus if available
        if absences and self._bus is not None:
            try:
                from mae_core.market.channels import CH_ABSENCE_DETECTED
                for absence in absences:
                    self._bus.publish(CH_ABSENCE_DETECTED, {
                        "source": absence.source,
                        "expected_hours": absence.expected_hours,
                        "actual_hours": absence.actual_hours,
                        "silence_ratio": absence.silence_ratio,
                    })
            except Exception:
                logger.debug("CH_ABSENCE_DETECTED publish failed", exc_info=True)

        return absences

    def bootstrap_from_archives(self, archive_reader=None) -> int:
        """Learn historical cadences from signal archive files.

        Loads the full archive and computes per-source cadence from
        actual historical inter-arrival times. This gives MIDGE accurate
        cadence baselines from day one.

        Returns the number of sources with learned cadences.
        """
        reader = archive_reader or self._archive_reader
        if reader is None:
            return 0

        try:
            from datetime import date, timedelta as td

            # Load a generous window — 90 days of history
            end = date.today()
            start = end - td(days=90)
            reader.load_range(start, end)

            sources_learned = 0
            # Iterate over all sources the archive knows about
            source_keys = list(reader._by_source.keys()) if hasattr(reader, "_by_source") else []

            for source in source_keys:
                records = reader.query_source(source, start=start, end=end)
                if len(records) < self._min_observations:
                    continue

                # Extract timestamps and sort
                timestamps = sorted(
                    r.timestamp for r in records
                    if hasattr(r, "timestamp") and r.timestamp != datetime.min
                )
                if len(timestamps) < 2:
                    continue

                # Record each arrival
                for ts in timestamps:
                    self.record_arrival(source, ts)

                sources_learned += 1

            logger.info(
                "AbsenceMonitor: bootstrapped cadences for %d sources from archives",
                sources_learned,
            )
            return sources_learned

        except Exception:
            logger.debug("AbsenceMonitor bootstrap failed", exc_info=True)
            return 0

    def get_statistics(self) -> dict:
        """Summary statistics for monitoring."""
        return {
            "tracked_sources": len(self._last_seen),
            "learned_cadences": len(self._cadence),
            "sources_with_history": len(self._arrival_history),
        }
