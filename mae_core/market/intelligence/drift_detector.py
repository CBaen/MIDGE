"""Drift Detector — ADWIN concept drift detection for signal distribution shifts.

Uses River's ADWIN (Adaptive Windowing) algorithm when available, with a
pure-Python ADWIN fallback for environments where River cannot be installed
(e.g. Python 3.14 before River publishes wheels).

ADWIN maintains a sliding window that automatically shrinks when the data
distribution changes. When it detects drift, it reports the mean before and
after the change point, enabling RegimeClassifier to trigger re-evaluation.

Tracked streams (configurable):
  - "price_returns":  daily return series
  - "volume":         normalised trading volume
  - "vix":            VIX index level
  - "sentiment":      composite sentiment score

Constructor args:
    delta: ADWIN sensitivity (lower = more sensitive). Default 0.002.

Usage:
    dd = DriftDetector()
    drift, old_mean, new_mean = dd.update("price_returns", 0.012)
    status = dd.get_drift_status()
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Attempt to use River ADWIN. Falls back to pure-Python implementation.
try:
    from river.drift import ADWIN as _RiverADWIN  # type: ignore

    HAS_RIVER = True
    logger.debug("drift_detector: using River ADWIN")
except ImportError:
    HAS_RIVER = False
    logger.debug("drift_detector: River not available, using pure-Python ADWIN")


# ---------------------------------------------------------------------------
# Pure-Python ADWIN implementation (fallback)
# Based on: Bifet & Gavalda (2007), "Learning from Time-Changing Data with
# Adaptive Windowing". Simplified for production stability.
# ---------------------------------------------------------------------------

class _PurePythonADWIN:
    """Simplified ADWIN for streaming concept drift detection.

    Maintains a list of buckets (exponential histogram). On each update,
    tests all sub-windows for a statistically significant mean shift using
    Hoeffding's bound.
    """

    def __init__(self, delta: float = 0.002) -> None:
        self.delta = delta
        self._window: List[float] = []
        self._n = 0
        self._sum = 0.0
        self._variance = 0.0
        self.drift_detected = False
        self._min_window = 5   # need at least this many obs before testing

    def update(self, value: float) -> bool:
        """Feed one value. Returns True if drift was detected."""
        self.drift_detected = False
        self._window.append(value)
        self._n += 1
        self._sum += value

        if self._n < self._min_window:
            return False

        # Test all splits of the window for a significant mean difference
        for split in range(self._min_window, self._n - self._min_window + 1):
            w0 = self._window[:split]
            w1 = self._window[split:]
            if not w0 or not w1:
                continue
            m0 = sum(w0) / len(w0)
            m1 = sum(w1) / len(w1)
            diff = abs(m0 - m1)

            # Hoeffding bound: eps = sqrt((1/(2m0) + 1/(2m1)) * ln(4*n/delta))
            n0, n1 = len(w0), len(w1)
            harmonic = (1.0 / (2 * n0)) + (1.0 / (2 * n1))
            log_term = math.log(4.0 * self._n / self.delta) if self.delta > 0 else 0
            eps = math.sqrt(harmonic * log_term)

            if diff >= eps:
                # Drift: drop the older sub-window
                self._window = w1
                self._n = len(self._window)
                self._sum = sum(self._window)
                self.drift_detected = True
                logger.debug(
                    "ADWIN drift: diff=%.4f eps=%.4f split=%d/%d",
                    diff, eps, split, self._n + split,
                )
                break

        return self.drift_detected

    @property
    def mean(self) -> float:
        if not self._window:
            return 0.0
        return self._sum / self._n


# ---------------------------------------------------------------------------
# Public DriftDetector
# ---------------------------------------------------------------------------

@dataclass
class DriftEvent:
    """A detected drift event in a named stream."""

    stream_name: str
    old_mean: float
    new_mean: float
    drift_magnitude: float     # abs(old - new)
    timestamp: datetime = field(default_factory=datetime.now)
    n_observations: int = 0


@dataclass
class StreamStatus:
    """Current state of one tracked stream."""

    name: str
    current_mean: float = 0.0
    drift_count: int = 0
    observation_count: int = 0
    last_drift: Optional[datetime] = None


class DriftDetector:
    """Multi-stream concept drift detector using ADWIN.

    Wraps one ADWIN instance per tracked stream. When drift is detected,
    captures old/new means and returns them so callers can react.

    Constructor args:
        delta: ADWIN sensitivity parameter (0.002 = low sensitivity, fewer alerts;
               0.05 = high sensitivity, more alerts). Default 0.002.
    """

    def __init__(self, delta: float = 0.002) -> None:
        self.delta = delta
        self._detectors: Dict[str, object] = {}
        self._prev_means: Dict[str, float] = {}
        self._statuses: Dict[str, StreamStatus] = {}
        self._total_drifts = 0
        self._recent_drifts: List[DriftEvent] = []   # last 50 drift events
        self._max_drift_history = 50

    def _get_or_create(self, stream_name: str) -> object:
        """Lazily create a detector for the named stream."""
        if stream_name not in self._detectors:
            if HAS_RIVER:
                self._detectors[stream_name] = _RiverADWIN(delta=self.delta)
            else:
                self._detectors[stream_name] = _PurePythonADWIN(delta=self.delta)
            self._statuses[stream_name] = StreamStatus(name=stream_name)
            self._prev_means[stream_name] = 0.0
        return self._detectors[stream_name]

    def update(self, stream_name: str, value: float) -> Tuple[bool, float, float]:
        """Feed one value to the named stream.

        Args:
            stream_name: identifier for this signal stream (e.g. "vix", "volume")
            value: the latest observation

        Returns:
            (drift_detected, old_mean, new_mean)
            drift_detected is True only when a statistically significant
            distribution shift has been detected.
        """
        if not isinstance(value, (int, float)) or math.isnan(value) or math.isinf(value):
            return (False, 0.0, 0.0)

        detector = self._get_or_create(stream_name)
        status = self._statuses[stream_name]
        status.observation_count += 1

        old_mean = self._prev_means.get(stream_name, 0.0)

        drift_detected = False
        new_mean = old_mean

        try:
            if HAS_RIVER:
                detector.update(value)                      # type: ignore
                drift_detected = detector.drift_detected    # type: ignore
                new_mean = getattr(detector, "estimation", old_mean)  # type: ignore
            else:
                drift_detected = detector.update(value)     # type: ignore
                new_mean = detector.mean                     # type: ignore
        except Exception:
            logger.debug("ADWIN update failed for stream %s", stream_name, exc_info=True)
            return (False, old_mean, old_mean)

        if drift_detected:
            self._total_drifts += 1
            status.drift_count += 1
            status.last_drift = datetime.now()
            status.current_mean = new_mean

            event = DriftEvent(
                stream_name=stream_name,
                old_mean=old_mean,
                new_mean=new_mean,
                drift_magnitude=abs(new_mean - old_mean),
                n_observations=status.observation_count,
            )
            self._recent_drifts.append(event)
            if len(self._recent_drifts) > self._max_drift_history:
                self._recent_drifts.pop(0)

            logger.info(
                "DriftDetector: drift in '%s' (old_mean=%.4f new_mean=%.4f obs=%d)",
                stream_name, old_mean, new_mean, status.observation_count,
            )

        self._prev_means[stream_name] = new_mean
        status.current_mean = new_mean

        return (drift_detected, old_mean, new_mean)

    def get_drift_status(self) -> Dict[str, dict]:
        """Return current status for all tracked streams.

        Returns a dict keyed by stream_name with:
            current_mean, drift_count, observation_count, last_drift (ISO str or None)
        """
        return {
            name: {
                "current_mean": round(s.current_mean, 6),
                "drift_count": s.drift_count,
                "observation_count": s.observation_count,
                "last_drift": s.last_drift.isoformat() if s.last_drift else None,
            }
            for name, s in self._statuses.items()
        }

    def get_recent_drifts(self, limit: int = 10) -> List[DriftEvent]:
        """Return the most recent drift events, newest first."""
        return list(reversed(self._recent_drifts))[:limit]

    def get_statistics(self) -> dict:
        """For HolonProxy.sense() delegation."""
        return {
            "tracked_streams": len(self._detectors),
            "total_drifts": self._total_drifts,
            "delta": self.delta,
            "has_river": HAS_RIVER,
            "stream_names": list(self._statuses.keys()),
        }
