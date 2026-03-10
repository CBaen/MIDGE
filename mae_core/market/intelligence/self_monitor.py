"""SelfMonitor — Behavioral risk monitoring for alert emission patterns.

Watches the stream of convergence alerts that MIDGE emits and detects four
classes of behavioral anomaly that indicate a feedback loop or runaway state:

1. runaway_rate    — More than ``max_alerts_per_window`` alerts in the rolling
                     deque window.  Classic runaway: something is triggering the
                     alerter in a tight loop.

2. direction_bias  — More than ``bias_threshold`` (default 80%) of the last
                     ``direction_window`` alerts point the same direction.
                     May be legitimate (trending market) — does NOT auto-suppress.

3. confidence_clustering — Standard deviation of the last 20 confidence values
                     is below 0.02.  All alerts landing at nearly identical
                     confidence levels strongly suggests a feedback loop has
                     frozen the learning engine.

4. ticker_flooding — A single ticker accounts for more than 50% of the last 30
                     alerts.  May be legitimate (breaking news) — does NOT
                     auto-suppress.

Auto-suppression (circuit breaker):
    runaway_rate       → suppresses
    confidence_clustering → suppresses
    direction_bias     → WARNING only, does NOT suppress
    ticker_flooding    → WARNING only, does NOT suppress

Usage:
    monitor = SelfMonitor(event_bus=ctx.bus)

    # After each convergence alert is generated:
    monitor.record_alert(direction="bullish", confidence=0.72, ticker="AAPL", step=42)

    # Before emitting an alert downstream:
    if not monitor.is_alerting_suppressed():
        emit(alert)

    # Manual recovery after investigation:
    monitor.reset_suppression()
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Channel constant — will be added to channels.py by Wiring Builder in Round 2.
# Using string literal here to avoid import dependency during Round 1 builds.
CH_BEHAVIORAL_ANOMALY = "market.risk.behavioral_anomaly"

# Anomaly string constants — matches build brief spec exactly.
ANOMALY_RUNAWAY_RATE = "runaway_rate"
ANOMALY_DIRECTION_BIAS = "direction_bias"
ANOMALY_CONFIDENCE_CLUSTERING = "confidence_clustering"
ANOMALY_TICKER_FLOODING = "ticker_flooding"

# Anomalies that trigger auto-suppression (feedback loop indicators).
_SUPPRESSION_TRIGGERS = frozenset({ANOMALY_RUNAWAY_RATE, ANOMALY_CONFIDENCE_CLUSTERING})

# Thresholds for sub-window checks (not configurable — these are biological constants).
_CLUSTERING_WINDOW = 20    # alerts examined for confidence std dev
_CLUSTERING_STD_THRESHOLD = 0.02  # std dev below this = clustering
_FLOODING_WINDOW = 30      # alerts examined for ticker concentration
_FLOODING_THRESHOLD = 0.50  # single ticker fraction above this = flooding


@dataclass
class _AlertRecord:
    """Lightweight record stored per alert in the rolling deque."""

    direction: str
    confidence: float
    ticker: str
    step: int
    timestamp: float = field(default_factory=time.time)


class SelfMonitor:
    """Behavioral anomaly detector for MIDGE alert emission.

    All public methods are thread-safe (protected by ``_lock``).
    All dependencies (event_bus) are optional — the monitor degrades
    gracefully to logging-only when they are absent.

    Args:
        event_bus: Optional EventBus for publishing CH_BEHAVIORAL_ANOMALY.
        rate_window: Size of the rolling deque (default 100). An alert is
            considered "runaway" when this deque is full AND more than
            ``max_alerts_per_window`` entries reside in it.
        direction_window: Number of recent alerts examined for direction bias
            (default 50). Must be <= rate_window.
        max_alerts_per_window: Maximum alert count before runaway_rate fires
            (default 10).  Read as ">10 alerts in the last rate_window steps
            constitutes runaway."
        bias_threshold: Fraction of direction_window that must share the same
            direction before direction_bias fires (default 0.80 = 80%).
    """

    def __init__(
        self,
        event_bus: Any = None,
        rate_window: int = 100,
        direction_window: int = 50,
        max_alerts_per_window: int = 10,
        bias_threshold: float = 0.80,
    ) -> None:
        self._bus = event_bus
        self._rate_window = rate_window
        self._direction_window = min(direction_window, rate_window)
        self._max_alerts_per_window = max_alerts_per_window
        self._bias_threshold = bias_threshold

        # Core state
        self._lock = threading.RLock()
        self._recent_alerts: deque[_AlertRecord] = deque(maxlen=rate_window)
        self._alert_count: int = 0
        self._anomaly_flags: list[str] = []
        self._alerting_suppressed: bool = False

        # Track which anomalies were already published to avoid event spam.
        # Reset when anomaly clears.
        self._published_anomalies: set[str] = set()

        logger.info(
            "SelfMonitor initialized (rate_window=%d, direction_window=%d, "
            "max_alerts=%d, bias_threshold=%.2f)",
            rate_window,
            self._direction_window,
            max_alerts_per_window,
            bias_threshold,
        )

    # =========================================================================
    # Primary interface
    # =========================================================================

    def record_alert(
        self,
        direction: str,
        confidence: float,
        ticker: str,
        step: int = 0,
    ) -> None:
        """Record a new alert and run anomaly checks.

        Args:
            direction: Alert direction (e.g. "bullish", "bearish", "neutral").
            confidence: Alert confidence score in [0, 1].
            ticker: Ticker symbol the alert concerns.
            step: Current simulation/daemon step counter.
        """
        record = _AlertRecord(
            direction=str(direction),
            confidence=float(confidence),
            ticker=str(ticker),
            step=int(step),
        )
        with self._lock:
            self._recent_alerts.append(record)
            self._alert_count += 1
            self._run_anomaly_checks()

    def check_anomalies(self) -> list[str]:
        """Return the current list of active anomaly flags.

        Does NOT re-run checks — returns the last computed state.  The checks
        are run inside ``record_alert`` so callers can poll this cheaply.

        Returns:
            List of active anomaly strings, empty if all clear.
        """
        with self._lock:
            return list(self._anomaly_flags)

    def is_alerting_suppressed(self) -> bool:
        """Return True if any suppression-triggering anomaly is active.

        Suppression triggers: runaway_rate, confidence_clustering.
        Direction bias and ticker flooding are WARNING-only — they do NOT
        suppress alert emission.
        """
        with self._lock:
            return self._alerting_suppressed

    def reset_suppression(self) -> None:
        """Manually clear the circuit-breaker suppression flag.

        Called during recovery after manual investigation confirms the
        underlying issue has been resolved.  Does NOT clear anomaly flags —
        those clear automatically on the next check_anomalies run when the
        deque has rotated past the problematic entries.
        """
        with self._lock:
            self._alerting_suppressed = False
            # Also clear the published set so re-suppression events are
            # re-published if the anomaly re-occurs.
            self._published_anomalies.discard(ANOMALY_RUNAWAY_RATE)
            self._published_anomalies.discard(ANOMALY_CONFIDENCE_CLUSTERING)
        logger.info("SelfMonitor: suppression manually reset")

    def get_statistics(self) -> dict[str, Any]:
        """Return current statistics for HolonProxy/SomaticMap integration.

        Returns:
            Dict with keys: alert_count, active_anomalies, alerting_suppressed,
            recent_alert_rate, direction_distribution.
        """
        with self._lock:
            deque_len = len(self._recent_alerts)
            # recent_alert_rate: fraction of the rate window that is filled
            recent_alert_rate = deque_len / max(self._rate_window, 1)

            # direction_distribution from the full deque
            direction_counts: Counter[str] = Counter(
                r.direction for r in self._recent_alerts
            )
            total = max(deque_len, 1)
            direction_distribution = {
                d: count / total for d, count in direction_counts.items()
            }

            return {
                "alert_count": self._alert_count,
                "active_anomalies": list(self._anomaly_flags),
                "alerting_suppressed": self._alerting_suppressed,
                "recent_alert_rate": recent_alert_rate,
                "direction_distribution": direction_distribution,
            }

    # =========================================================================
    # Internal anomaly detection
    # =========================================================================

    def _run_anomaly_checks(self) -> None:
        """Run all four anomaly checks and update state.

        MUST be called while holding ``_lock``.
        """
        alerts = list(self._recent_alerts)  # snapshot under lock
        new_flags: list[str] = []

        # --- 1. Runaway rate ---
        if len(alerts) > self._max_alerts_per_window:
            new_flags.append(ANOMALY_RUNAWAY_RATE)

        # --- 2. Direction bias ---
        if len(alerts) >= self._direction_window:
            recent_dir = [r.direction for r in alerts[-self._direction_window:]]
            if recent_dir:
                dominant = Counter(recent_dir).most_common(1)[0]
                dominant_fraction = dominant[1] / len(recent_dir)
                if dominant_fraction > self._bias_threshold:
                    new_flags.append(ANOMALY_DIRECTION_BIAS)

        # --- 3. Confidence clustering ---
        if len(alerts) >= _CLUSTERING_WINDOW:
            confidences = [r.confidence for r in alerts[-_CLUSTERING_WINDOW:]]
            std = _std_dev(confidences)
            if std < _CLUSTERING_STD_THRESHOLD:
                new_flags.append(ANOMALY_CONFIDENCE_CLUSTERING)

        # --- 4. Ticker flooding ---
        if len(alerts) >= _FLOODING_WINDOW:
            tickers = [r.ticker for r in alerts[-_FLOODING_WINDOW:]]
            top_ticker, top_count = Counter(tickers).most_common(1)[0]
            if top_count / len(tickers) > _FLOODING_THRESHOLD:
                new_flags.append(ANOMALY_TICKER_FLOODING)

        # Update suppression flag
        has_suppression_trigger = bool(
            _SUPPRESSION_TRIGGERS.intersection(new_flags)
        )
        if has_suppression_trigger and not self._alerting_suppressed:
            self._alerting_suppressed = True
            logger.warning(
                "SelfMonitor: alerting SUPPRESSED — circuit breaker triggered by: %s",
                ", ".join(_SUPPRESSION_TRIGGERS.intersection(new_flags)),
            )

        # Update flag list
        self._anomaly_flags = new_flags

        # Publish new anomalies (avoid spamming the bus for every alert)
        self._publish_new_anomalies(new_flags)

    def _publish_new_anomalies(self, current_flags: list[str]) -> None:
        """Publish CH_BEHAVIORAL_ANOMALY for newly appearing anomalies.

        MUST be called while holding ``_lock``.
        Only publishes when an anomaly is seen for the first time after
        it was absent — prevents bus flooding.
        """
        current_set = set(current_flags)
        cleared = self._published_anomalies - current_set
        appeared = current_set - self._published_anomalies

        # Update tracking set
        self._published_anomalies = current_set

        for anomaly_type in appeared:
            details = self._build_anomaly_details(anomaly_type)
            suppresses = anomaly_type in _SUPPRESSION_TRIGGERS
            payload = {
                "anomaly_type": anomaly_type,
                "details": details,
                "suppresses_alerting": suppresses,
                "alert_count": self._alert_count,
                "timestamp": time.time(),
            }
            logger.warning(
                "SelfMonitor: anomaly detected — %s (suppresses=%s) — %s",
                anomaly_type,
                suppresses,
                details,
            )
            if self._bus is not None:
                try:
                    self._bus.publish(CH_BEHAVIORAL_ANOMALY, payload)
                except Exception:
                    logger.debug(
                        "SelfMonitor: failed to publish anomaly event", exc_info=True
                    )

        for anomaly_type in cleared:
            logger.info("SelfMonitor: anomaly cleared — %s", anomaly_type)

    def _build_anomaly_details(self, anomaly_type: str) -> str:
        """Build a human-readable detail string for a given anomaly.

        MUST be called while holding ``_lock``.
        """
        alerts = list(self._recent_alerts)
        if anomaly_type == ANOMALY_RUNAWAY_RATE:
            return (
                f"{len(alerts)} alerts in deque "
                f"(window={self._rate_window}, max={self._max_alerts_per_window})"
            )
        if anomaly_type == ANOMALY_DIRECTION_BIAS:
            if len(alerts) >= self._direction_window:
                recent_dir = [r.direction for r in alerts[-self._direction_window:]]
                dominant = Counter(recent_dir).most_common(1)[0]
                pct = dominant[1] / max(len(recent_dir), 1) * 100
                return (
                    f"{dominant[0]} direction in {pct:.1f}% of last "
                    f"{self._direction_window} alerts "
                    f"(threshold={self._bias_threshold * 100:.0f}%)"
                )
            return f"direction_bias check (insufficient window)"
        if anomaly_type == ANOMALY_CONFIDENCE_CLUSTERING:
            if len(alerts) >= _CLUSTERING_WINDOW:
                confidences = [r.confidence for r in alerts[-_CLUSTERING_WINDOW:]]
                std = _std_dev(confidences)
                mean = sum(confidences) / len(confidences)
                return (
                    f"confidence std={std:.4f} < {_CLUSTERING_STD_THRESHOLD} "
                    f"(mean={mean:.3f}, last {_CLUSTERING_WINDOW} alerts)"
                )
            return f"confidence_clustering check (insufficient window)"
        if anomaly_type == ANOMALY_TICKER_FLOODING:
            if len(alerts) >= _FLOODING_WINDOW:
                tickers = [r.ticker for r in alerts[-_FLOODING_WINDOW:]]
                top_ticker, top_count = Counter(tickers).most_common(1)[0]
                pct = top_count / len(tickers) * 100
                return (
                    f"{top_ticker} in {pct:.1f}% of last {_FLOODING_WINDOW} alerts "
                    f"(threshold={_FLOODING_THRESHOLD * 100:.0f}%)"
                )
            return "ticker_flooding check (insufficient window)"
        return ""


# =========================================================================
# Module-level utility
# =========================================================================

def _std_dev(values: list[float]) -> float:
    """Population standard deviation of a list of floats.

    Returns 0.0 for lists with fewer than 2 elements (no variance defined).
    Uses math.fsum for numeric stability with floats.
    """
    n = len(values)
    if n < 2:
        return 0.0
    mean = math.fsum(values) / n
    variance = math.fsum((x - mean) ** 2 for x in values) / n
    return math.sqrt(variance)
