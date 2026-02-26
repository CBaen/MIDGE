"""Metacognition Monitor - Thinking about thinking.

Biological basis: Flavell (1979) introduced metacognition as monitoring
and controlling one's own cognitive processes. Prefrontal cortex
executive functions enable error detection, confidence calibration,
and adaptive learning rate adjustment.

Mae uses metacognition to:
1. Track decision quality over time (am I getting better or worse?)
2. Detect performance degradation before it cascades
3. Calibrate confidence (are my confidence estimates accurate?)
4. Suggest learning rate adjustments (speed up or slow down learning)

Connection points:
- EventBus publishes metacognition_update every step
- EventBus publishes metacognition_alert when degradation detected
- Learning systems consume alerts to adjust their rates
- TheoryOfMind can use calibration data for self-model accuracy

Law compliance:
- Law 3 (Holon Protocol): Self-reference — this IS self-reference, the
  system monitors its own cognitive performance
- Law 8 (Consciousness): Property 8 — prediction/error-correction
  (the core mechanism: predict outcomes, measure errors, correct)
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class DecisionRecord:
    """Record of a single decision and its outcome."""

    step: int
    predicted_outcome: float
    actual_outcome: float
    decision_type: str = "general"
    error: float = 0.0


# Configuration constants
_MAX_HISTORY = 200
_PERFORMANCE_WINDOW = 20
_EMA_ALPHA = 0.05
_DEGRADATION_DEFAULT_THRESHOLD = 0.3
_LEARNING_RATE_BOOST = 1.5
_LEARNING_RATE_REDUCE = 0.8


class MetacognitionMonitor:
    """Monitors and evaluates the quality of Mae's own decisions.

    Maintains a sliding window of decision records, computes recent
    performance, detects degradation relative to a running baseline,
    and suggests learning rate adjustments.

    Args:
        event_bus: Optional EventBus for publishing updates and alerts.
    """

    CH_META_UPDATE = "cognition.metacognition_update"
    CH_META_ALERT = "cognition.metacognition_alert"

    def __init__(self, event_bus: Any | None = None, **kwargs: Any) -> None:
        self._bus = event_bus
        self._decision_history: deque[DecisionRecord] = deque(maxlen=_MAX_HISTORY)
        self._baseline_performance: float = 0.5
        self._confidence_calibration: float = 1.0
        self._degradation_threshold: float = _DEGRADATION_DEFAULT_THRESHOLD
        self._current_step: int = 0

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_decision(
        self,
        step: int,
        predicted: float,
        actual: float,
        decision_type: str = "general",
    ) -> DecisionRecord:
        """Record a decision with its predicted and actual outcomes.

        Computes error as abs(predicted - actual), appends to history,
        and updates the running baseline via exponential moving average.

        Args:
            step: The simulation step when the decision was made.
            predicted: The predicted outcome value (0-1 scale).
            actual: The actual outcome value (0-1 scale).
            decision_type: A category label for the decision.

        Returns:
            The created DecisionRecord.
        """
        error = abs(predicted - actual)
        record = DecisionRecord(
            step=step,
            predicted_outcome=predicted,
            actual_outcome=actual,
            decision_type=decision_type,
            error=error,
        )
        self._decision_history.append(record)

        # Update baseline with EMA
        accuracy = 1.0 - error
        self._baseline_performance = (
            (1 - _EMA_ALPHA) * self._baseline_performance
            + _EMA_ALPHA * accuracy
        )

        return record

    # ------------------------------------------------------------------
    # Performance Computation
    # ------------------------------------------------------------------

    def _compute_recent_performance(self) -> float:
        """Compute mean accuracy over the last _PERFORMANCE_WINDOW decisions.

        Accuracy for each decision is 1.0 - error. Returns 0.5 if
        no decisions have been recorded.
        """
        if not self._decision_history:
            return 0.5

        recent = list(self._decision_history)[-_PERFORMANCE_WINDOW:]
        mean_error = sum(r.error for r in recent) / len(recent)
        return 1.0 - mean_error

    def _detect_degradation(self) -> bool:
        """Check whether recent performance has degraded significantly.

        Degradation is defined as recent performance falling below
        the baseline by more than the degradation threshold.
        """
        recent = self._compute_recent_performance()
        return recent < (self._baseline_performance - self._degradation_threshold)

    def _compute_calibration(self) -> float:
        """Compute how well-calibrated predictions are.

        Measures the correlation between prediction confidence
        (1 - abs(predicted - 0.5) * 2, i.e., how far from uncertainty)
        and actual accuracy (1 - error).

        Returns a value in [0, 1] where 1.0 means perfectly calibrated.
        """
        if len(self._decision_history) < 2:
            return 1.0

        recent = list(self._decision_history)[-_PERFORMANCE_WINDOW:]
        if len(recent) < 2:
            return 1.0

        # Measure: are confident predictions more accurate?
        calibration_errors: list[float] = []
        for record in recent:
            # How "confident" the prediction was (distance from 0.5)
            confidence_proxy = abs(record.predicted_outcome - 0.5) * 2.0
            accuracy = 1.0 - record.error
            calibration_error = abs(confidence_proxy - accuracy)
            calibration_errors.append(calibration_error)

        mean_cal_error = sum(calibration_errors) / len(calibration_errors)
        self._confidence_calibration = max(0.0, 1.0 - mean_cal_error)
        return self._confidence_calibration

    # ------------------------------------------------------------------
    # Step (Autopoietic Maintenance)
    # ------------------------------------------------------------------

    def step(self, current_step: int = 0) -> None:
        """Advance one time step. Check for degradation and publish updates.

        Law 6 (Autopoietic Closure): The step method self-maintains by
        continuously monitoring its own performance and publishing alerts
        when degradation is detected.
        """
        self._current_step = current_step if current_step > 0 else self._current_step + 1

        recent_performance = self._compute_recent_performance()
        calibration = self._compute_calibration()
        degraded = self._detect_degradation()

        # Publish alert if degraded
        if degraded and self._bus is not None:
            deficit = self._baseline_performance - recent_performance
            try:
                self._bus.publish(self.CH_META_ALERT, {
                    "recent_performance": round(recent_performance, 4),
                    "baseline": round(self._baseline_performance, 4),
                    "deficit": round(deficit, 4),
                    "step": self._current_step,
                })
            except Exception:
                logger.debug("EventBus publish failed for metacognition_alert")

        # Always publish update
        if self._bus is not None:
            try:
                self._bus.publish(self.CH_META_UPDATE, {
                    "recent_performance": round(recent_performance, 4),
                    "baseline": round(self._baseline_performance, 4),
                    "calibration": round(calibration, 4),
                    "decision_count": len(self._decision_history),
                    "degraded": degraded,
                    "step": self._current_step,
                })
            except Exception:
                logger.debug("EventBus publish failed for metacognition_update")

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_performance_score(self) -> float:
        """Return the recent performance score."""
        return self._compute_recent_performance()

    def is_performing_well(self) -> bool:
        """Return True if recent performance is at or above baseline."""
        return not self._detect_degradation()

    def should_adjust_learning_rate(self) -> Optional[float]:
        """Suggest a learning rate multiplier based on performance.

        If degraded: suggest increasing learning rate by 1.5x to
        adapt faster. If performing well: suggest decreasing by 0.8x
        to consolidate. If neither: return None (no change).
        """
        if self._detect_degradation():
            return _LEARNING_RATE_BOOST
        recent = self._compute_recent_performance()
        if recent > self._baseline_performance + 0.1:
            return _LEARNING_RATE_REDUCE
        return None

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def get_statistics(self) -> dict[str, Any]:
        """Return current system statistics."""
        return {
            "decision_count": len(self._decision_history),
            "recent_performance": round(self._compute_recent_performance(), 4),
            "baseline_performance": round(self._baseline_performance, 4),
            "confidence_calibration": round(self._confidence_calibration, 4),
            "degraded": self._detect_degradation(),
            "current_step": self._current_step,
        }

    # ------------------------------------------------------------------
    # Tier 2 Persistence
    # ------------------------------------------------------------------

    def serialize(self, persist_dir: Any = None) -> dict[str, Any]:
        """Serialize state for Tier 2 persistence.

        Args:
            persist_dir: Accepted for compatibility with the shared-system
                persistence loop in model.py. Ignored — all state fits
                inline in the returned metadata dict.
        """
        history = [
            {
                "step": r.step,
                "predicted_outcome": r.predicted_outcome,
                "actual_outcome": r.actual_outcome,
                "decision_type": r.decision_type,
                "error": r.error,
            }
            for r in self._decision_history
        ]

        return {
            "decision_history": history,
            "baseline_performance": self._baseline_performance,
            "confidence_calibration": self._confidence_calibration,
            "degradation_threshold": self._degradation_threshold,
            "current_step": self._current_step,
        }

    def restore(self, persist_dir_or_data: Any = None, metadata: dict[str, Any] | None = None) -> None:
        """Restore state from serialized data.

        Supports two call styles:
          - restore(data_dict)           — direct dict (unit tests)
          - restore(persist_dir, meta)   — shared-system pattern (model.py)
        """
        if metadata is not None:
            data = metadata
        elif isinstance(persist_dir_or_data, dict):
            data = persist_dir_or_data
        else:
            return  # nothing to restore
        self._baseline_performance = data.get("baseline_performance", 0.5)
        self._confidence_calibration = data.get("confidence_calibration", 1.0)
        self._degradation_threshold = data.get(
            "degradation_threshold", _DEGRADATION_DEFAULT_THRESHOLD
        )
        self._current_step = data.get("current_step", 0)

        self._decision_history.clear()
        for record_data in data.get("decision_history", []):
            record = DecisionRecord(
                step=record_data["step"],
                predicted_outcome=record_data["predicted_outcome"],
                actual_outcome=record_data["actual_outcome"],
                decision_type=record_data.get("decision_type", "general"),
                error=record_data.get("error", 0.0),
            )
            self._decision_history.append(record)

    def __repr__(self) -> str:
        return (
            f"MetacognitionMonitor(decisions={len(self._decision_history)}, "
            f"baseline={self._baseline_performance:.3f})"
        )
