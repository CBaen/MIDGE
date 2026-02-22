"""Vestibular system - balance, orientation, and stability detection.

Biological analogy: The inner ear's vestibular apparatus detects head
position, acceleration, and balance. It signals when the body is
stable, wobbling, or in free-fall (vertigo). The brain uses this
information to maintain posture and coordinate movement.

Mae's vestibular system:
- "Balance" = stability of key system metrics over a rolling window
- High variance across metrics = unstable / dizzy
- Low variance = stable and oriented
- Vertigo alert when stability drops below threshold
- Rapid-change detection identifies metrics diverging abruptly

Law compliance:
- Law 6 (Autopoietic Closure): Stability monitoring feeds back into
  system regulation — the vestibular signal itself triggers corrective
  action that restores stability.
- Law 8 (Consciousness): (8) prediction/error-correction — detecting
  deviation from expected stability baseline.

EventBus channels:
- coordination.balance_update (periodic stability report)
- coordination.vertigo (stability critically low)
"""

from __future__ import annotations

import logging
import statistics
from typing import Any, Optional

from mae_core.backbone.event_bus import EventBus

logger = logging.getLogger(__name__)

# EventBus channels
CH_BALANCE_UPDATE = "coordination.balance_update"
CH_VERTIGO = "coordination.vertigo"

# Rolling window size for metric histories
_WINDOW_SIZE = 10


class VestibularSystem:
    """Mae's balance and stability monitoring system.

    Tracks rolling windows of metric values reported by other systems.
    Computes per-metric stability (inverse of normalized variance) and
    an overall stability score. Publishes vertigo, wobble, or stable
    status each step.
    """

    CH_BALANCE_UPDATE = CH_BALANCE_UPDATE
    CH_VERTIGO = CH_VERTIGO

    def __init__(self, event_bus: Optional[EventBus] = None) -> None:
        self.event_bus = event_bus

        # Rolling metric histories: metric_name -> list of recent values
        self._metric_history: dict[str, list[float]] = {}
        self._tracked_metrics: set[str] = set()

        # Stability
        self._stability_score: float = 1.0  # 1.0 = perfectly stable
        self._vertigo_threshold: float = 0.3  # Below this = vertigo

        # Step tracking
        self._step_count: int = 0

        logger.info(
            "VestibularSystem initialized: vertigo_threshold=%.2f",
            self._vertigo_threshold,
        )

    # =========================================================================
    # Metric Reporting (called by other systems)
    # =========================================================================

    def report_metric(self, name: str, value: float) -> None:
        """Report a metric value for stability tracking.

        Args:
            name: Metric identifier (e.g. "cpu_load", "memory_pressure").
            value: Current value of the metric.
        """
        self._tracked_metrics.add(name)

        if name not in self._metric_history:
            self._metric_history[name] = []

        history = self._metric_history[name]
        history.append(value)

        # Enforce rolling window
        if len(history) > _WINDOW_SIZE:
            self._metric_history[name] = history[-_WINDOW_SIZE:]

    # =========================================================================
    # Stability Computation
    # =========================================================================

    def _compute_stability(self, history: list[float]) -> float:
        """Compute stability for a single metric's history.

        Stability = 1.0 - normalized_standard_deviation.
        When all values are identical, stability is 1.0.
        When variance is very high relative to the mean, stability
        approaches 0.0.

        Args:
            history: List of recent metric values.

        Returns:
            Stability score between 0.0 and 1.0.
        """
        if len(history) < 2:
            return 1.0  # Not enough data to judge instability

        mean = statistics.mean(history)
        stdev = statistics.stdev(history)

        # Normalize: coefficient of variation, capped at 1.0
        if abs(mean) < 1e-10:
            # Mean near zero — use raw stdev as instability signal
            normalized = min(1.0, stdev)
        else:
            normalized = min(1.0, stdev / abs(mean))

        return max(0.0, 1.0 - normalized)

    def _detect_rapid_change(self, history: list[float]) -> bool:
        """Detect whether recent values are diverging rapidly.

        True if the last 3 values differ by more than 50% from
        the series mean. This catches sudden spikes/drops that
        the standard deviation might smooth over.

        Args:
            history: List of recent metric values.

        Returns:
            True if rapid change is detected.
        """
        if len(history) < 3:
            return False

        mean = statistics.mean(history)
        if abs(mean) < 1e-10:
            # Near-zero mean: check if last 3 are all significantly nonzero
            return any(abs(v) > 0.5 for v in history[-3:])

        threshold = abs(mean) * 0.5
        last_three = history[-3:]
        return all(abs(v - mean) > threshold for v in last_three)

    # =========================================================================
    # Step (Law 8: prediction/error-correction)
    # =========================================================================

    def step(self, current_step: int = 0) -> None:
        """Run one vestibular assessment cycle.

        1. Compute per-metric stability scores.
        2. Compute overall stability as the mean of all metric stabilities.
        3. Publish vertigo, wobbling, or stable status.
        """
        self._step_count = current_step if current_step else self._step_count + 1

        if not self._metric_history:
            # No metrics reported yet — assume stable
            self._stability_score = 1.0
            if self.event_bus is not None:
                self.event_bus.publish(
                    CH_BALANCE_UPDATE,
                    {
                        "stability": self._stability_score,
                        "status": "stable",
                        "metrics_tracked": 0,
                        "step": self._step_count,
                    },
                )
            return

        # 1. Per-metric stability
        metric_stabilities: dict[str, float] = {}
        rapid_changes: list[str] = []

        for name, history in self._metric_history.items():
            stability = self._compute_stability(history)
            metric_stabilities[name] = stability

            if self._detect_rapid_change(history):
                rapid_changes.append(name)

        # 2. Overall stability = mean of all metric stabilities
        if metric_stabilities:
            self._stability_score = statistics.mean(metric_stabilities.values())
        else:
            self._stability_score = 1.0

        # Penalize rapid changes: each rapid-change metric reduces score
        if rapid_changes:
            penalty = 0.1 * len(rapid_changes)
            self._stability_score = max(0.0, self._stability_score - penalty)

        # 3. Publish status
        if self.event_bus is not None:
            status_data = {
                "stability": self._stability_score,
                "metric_stabilities": metric_stabilities,
                "rapid_changes": rapid_changes,
                "metrics_tracked": len(self._metric_history),
                "step": self._step_count,
            }

            if self._stability_score < self._vertigo_threshold:
                status_data["status"] = "vertigo"
                self.event_bus.publish(CH_VERTIGO, status_data)
                logger.warning(
                    "VERTIGO: stability=%.2f (threshold %.2f) at step %d, "
                    "rapid_changes=%s",
                    self._stability_score,
                    self._vertigo_threshold,
                    self._step_count,
                    rapid_changes,
                )
            elif self._stability_score > 0.7:
                status_data["status"] = "stable"
                self.event_bus.publish(CH_BALANCE_UPDATE, status_data)
            else:
                status_data["status"] = "wobbling"
                self.event_bus.publish(CH_BALANCE_UPDATE, status_data)

    # =========================================================================
    # Query
    # =========================================================================

    def get_stability_score(self) -> float:
        """Return overall stability score (0.0 - 1.0)."""
        return self._stability_score

    def is_stable(self) -> bool:
        """True when stability is above the vertigo threshold."""
        return self._stability_score >= self._vertigo_threshold

    # =========================================================================
    # Observability
    # =========================================================================

    def get_statistics(self) -> dict[str, Any]:
        """Return vestibular system statistics."""
        metric_stabilities = {}
        for name, history in self._metric_history.items():
            metric_stabilities[name] = self._compute_stability(history)

        return {
            "stability_score": self._stability_score,
            "is_stable": self.is_stable(),
            "vertigo_threshold": self._vertigo_threshold,
            "metrics_tracked": len(self._tracked_metrics),
            "metric_names": sorted(self._tracked_metrics),
            "metric_stabilities": metric_stabilities,
            "step_count": self._step_count,
        }

    # =========================================================================
    # Tier 2 Persistence
    # =========================================================================

    def serialize(self) -> dict[str, Any]:
        """Serialize state for Tier 2 persistence."""
        return {
            "metric_history": {k: list(v) for k, v in self._metric_history.items()},
            "tracked_metrics": sorted(self._tracked_metrics),
            "stability_score": self._stability_score,
            "vertigo_threshold": self._vertigo_threshold,
            "step_count": self._step_count,
        }

    def restore(self, data: dict[str, Any]) -> None:
        """Restore state from serialized data."""
        if "metric_history" in data:
            self._metric_history = {
                k: list(v) for k, v in data["metric_history"].items()
            }
        if "tracked_metrics" in data:
            self._tracked_metrics = set(data["tracked_metrics"])
        self._stability_score = data.get("stability_score", self._stability_score)
        self._vertigo_threshold = data.get("vertigo_threshold", self._vertigo_threshold)
        self._step_count = data.get("step_count", self._step_count)
