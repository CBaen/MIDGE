"""Validated Imagination - Tracks WorldModel prediction accuracy.

Records agent predictions, validates against actual outcomes via
consensus, and maintains per-agent per-domain accuracy metrics.
Feeds into expertise tracking: accurate imaginers get more influence.

Biological analogy: Learning from prediction error (dopamine system).
Based on: Doll et al. (2012) "Model-based vs model-free learning".
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ImaginationPrediction:
    """A recorded prediction from an agent's imagination."""

    prediction_id: str
    agent_id: str
    domain: str
    state: Any
    action: Any
    predicted_next_state: Any
    predicted_reward: float
    predicted_confidence: float
    timestamp: float = field(default_factory=time.time)

    # Filled during validation
    actual_state: Any | None = None
    actual_reward: float | None = None
    consensus_confirms: bool = False
    consensus_confidence: float = 0.0
    validated: bool = False
    validation_timestamp: float | None = None


@dataclass
class ImaginationAccuracy:
    """Per-agent per-domain accuracy tracking."""

    agent_id: str
    domain: str
    total_predictions: int = 0
    accurate_predictions: int = 0
    inaccurate_predictions: int = 0
    total_reward_error: float = 0.0
    mean_reward_error: float = 0.0
    overconfident_count: int = 0
    underconfident_count: int = 0
    well_calibrated_count: int = 0

    @property
    def prediction_accuracy(self) -> float:
        if self.total_predictions == 0:
            return 0.5
        return self.accurate_predictions / self.total_predictions

    @property
    def confidence_calibration(self) -> float:
        total_checks = (
            self.overconfident_count + self.underconfident_count + self.well_calibrated_count
        )
        if total_checks == 0:
            return 0.5
        return self.well_calibrated_count / total_checks


@dataclass
class TrajectoryStep:
    """A single step in a validated planning trajectory."""

    state: Any
    action: Any
    predicted_next_state: Any
    predicted_reward: float
    step_index: int
    validation_confidence: float = 0.0
    consensus_supports: bool = False
    validation_message: str = ""


class ValidatedImagination:
    """Validates WorldModel predictions using consensus reality checks.

    Cycle:
    1. Agent imagines: record_imagination()
    2. Agent acts in real world
    3. Consensus observes outcome
    4. System validates: validate_with_consensus()
    5. Accuracy updated → feeds into expertise
    """

    def __init__(
        self,
        reward_error_threshold: float = 0.2,
        confidence_threshold: float = 0.7,
        expertise_callback: Any | None = None,
        event_bus: Any = None,
    ) -> None:
        self._reward_error_threshold = reward_error_threshold
        self._conf_threshold = confidence_threshold
        self._expertise_cb = expertise_callback
        self._bus = event_bus

        self._pending: dict[str, ImaginationPrediction] = {}
        self._accuracy: dict[tuple[str, str], ImaginationAccuracy] = {}

        self._total_imaginations = 0
        self._total_validations = 0
        self._total_accurate = 0
        self._total_inaccurate = 0

        # Subscribe to prediction events for auto-tracking
        if self._bus is not None:
            try:
                self._bus.register_callback(
                    "cognition.prediction_made",
                    self._on_prediction_made,
                )
            except Exception:
                logger.debug("Failed to subscribe to cognition.prediction_made")

    def record_imagination(
        self,
        agent_id: str,
        domain: str,
        state: Any,
        action: Any,
        predicted_next_state: Any,
        predicted_reward: float,
        confidence: float,
    ) -> str:
        """Record an agent's prediction for later validation."""
        self._total_imaginations += 1
        pred_id = f"pred-{agent_id}-{domain}-{self._total_imaginations}"

        prediction = ImaginationPrediction(
            prediction_id=pred_id,
            agent_id=agent_id,
            domain=domain,
            state=state,
            action=action,
            predicted_next_state=predicted_next_state,
            predicted_reward=predicted_reward,
            predicted_confidence=confidence,
        )
        self._pending[pred_id] = prediction
        return pred_id

    def validate_with_consensus(
        self,
        prediction_id: str,
        actual_state: Any,
        actual_reward: float,
        consensus_confidence: float = 0.5,
    ) -> bool:
        """Validate a prediction against actual outcome."""
        prediction = self._pending.pop(prediction_id, None)
        if prediction is None:
            return False

        # Compare predicted vs actual
        reward_error = abs(prediction.predicted_reward - actual_reward)
        is_accurate = reward_error <= self._reward_error_threshold

        prediction.actual_state = actual_state
        prediction.actual_reward = actual_reward
        prediction.consensus_confirms = is_accurate
        prediction.consensus_confidence = consensus_confidence
        prediction.validated = True
        prediction.validation_timestamp = time.time()

        # Update accuracy tracking
        self._update_accuracy(prediction, reward_error)

        # Notify expertise system
        if self._expertise_cb is not None:
            try:
                self._expertise_cb(
                    agent_id=prediction.agent_id,
                    domain=prediction.domain,
                    was_correct=is_accurate,
                    confidence=prediction.predicted_confidence,
                )
            except Exception:
                logger.exception("Expertise callback failed")

        self._total_validations += 1
        if is_accurate:
            self._total_accurate += 1
        else:
            self._total_inaccurate += 1

        # Publish validation result on EventBus
        if self._bus is not None:
            try:
                total = max(self._total_validations, 1)
                self._bus.publish("cognition.imagination_validated", {
                    "prediction_id": prediction.prediction_id,
                    "agent_id": prediction.agent_id,
                    "domain": prediction.domain,
                    "was_accurate": is_accurate,
                    "reward_error": reward_error,
                    "overall_accuracy": self._total_accurate / total,
                    "total_validations": self._total_validations,
                    # Actual prediction data for WorldModel training —
                    # without these, downstream consumers must use synthetic
                    # zeros, making training meaningless.
                    "state": prediction.state,
                    "action": prediction.action,
                    "predicted_next_state": prediction.predicted_next_state,
                    "actual_state": prediction.actual_state,
                    "actual_reward": actual_reward,
                    "predicted_reward": prediction.predicted_reward,
                })
            except Exception:
                logger.debug("EventBus publish failed for imagination_validated")

        return is_accurate

    def get_imagination_accuracy(
        self, agent_id: str, domain: str
    ) -> ImaginationAccuracy | None:
        return self._accuracy.get((agent_id, domain))

    def get_top_imaginers(
        self, domain: str, count: int = 10, min_predictions: int = 5
    ) -> list[tuple[str, float]]:
        """Rank agents by imagination accuracy in a domain."""
        candidates = []
        for (aid, dom), acc in self._accuracy.items():
            if dom == domain and acc.total_predictions >= min_predictions:
                score = acc.prediction_accuracy * 0.7 + acc.confidence_calibration * 0.3
                candidates.append((aid, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:count]

    def get_statistics(self) -> dict[str, Any]:
        total = max(self._total_validations, 1)
        return {
            "total_imaginations": self._total_imaginations,
            "total_validations": self._total_validations,
            "pending_validations": len(self._pending),
            "overall_accuracy": self._total_accurate / total,
            "total_accurate": self._total_accurate,
            "total_inaccurate": self._total_inaccurate,
            "tracked_agents": len(set(k[0] for k in self._accuracy)),
            "tracked_domains": len(set(k[1] for k in self._accuracy)),
        }

    # --- Internal ---

    def _on_prediction_made(self, channel: str, message: Any) -> None:
        """Auto-track predictions from EventBus cognition.prediction_made events."""
        try:
            import json
            data = json.loads(message) if isinstance(message, str) else message
            # Auto-record prediction if it has the required fields
            uncertainty = data.get("uncertainty", 0.0)
            reward = data.get("reward", 0.5)
            self.record_imagination(
                agent_id=data.get("agent_id", "world_model"),
                domain=data.get("domain", "general"),
                state=None,
                action=None,
                predicted_next_state=None,
                predicted_reward=reward,
                confidence=max(0.0, 1.0 - uncertainty),
            )
        except Exception:
            logger.debug("Failed to auto-track prediction from EventBus")

    def _update_accuracy(
        self, prediction: ImaginationPrediction, reward_error: float
    ) -> None:
        """Update per-agent per-domain accuracy metrics."""
        key = (prediction.agent_id, prediction.domain)
        if key not in self._accuracy:
            self._accuracy[key] = ImaginationAccuracy(
                agent_id=prediction.agent_id,
                domain=prediction.domain,
            )

        acc = self._accuracy[key]
        acc.total_predictions += 1

        if prediction.consensus_confirms:
            acc.accurate_predictions += 1
        else:
            acc.inaccurate_predictions += 1

        acc.total_reward_error += reward_error
        acc.mean_reward_error = acc.total_reward_error / acc.total_predictions

        # Confidence calibration check
        high_confidence = prediction.predicted_confidence >= self._conf_threshold
        if high_confidence:
            if prediction.consensus_confirms:
                acc.well_calibrated_count += 1
            else:
                acc.overconfident_count += 1
        else:
            if not prediction.consensus_confirms:
                acc.well_calibrated_count += 1
            else:
                acc.underconfident_count += 1

    def __repr__(self) -> str:
        return (
            f"ValidatedImagination(imaginations={self._total_imaginations}, "
            f"validations={self._total_validations})"
        )


class ValidatedImaginationPlanner:
    """Plans trajectories with step-by-step consensus validation.

    Each planning step:
    1. WorldModel predicts next state
    2. Consensus validates prediction
    3. If supported: continue planning
    4. If rejected: stop (prediction unrealistic)
    """

    def __init__(
        self,
        world_model: Any,
        validator: ValidatedImagination | None = None,
        min_validation_confidence: float = 0.5,
    ) -> None:
        self._world_model = world_model
        self._validator = validator
        self._min_confidence = min_validation_confidence

        self._total_plans = 0
        self._total_steps = 0
        self._total_validated = 0
        self._total_rejected = 0

    def plan_with_validation(
        self,
        agent_id: str,
        initial_state: Any,
        policy: Any,
        horizon: int = 10,
        domain: str = "general",
    ) -> list[TrajectoryStep]:
        """Plan trajectory, validating each step."""
        self._total_plans += 1
        trajectory: list[TrajectoryStep] = []
        current_state = initial_state

        for i in range(horizon):
            # Get action from policy
            action = self._select_action(policy, current_state)

            # Predict next state
            predicted_next, predicted_reward = self._predict_transition(
                current_state, action
            )

            step = TrajectoryStep(
                state=current_state,
                action=action,
                predicted_next_state=predicted_next,
                predicted_reward=predicted_reward,
                step_index=i,
            )

            # Validate step
            validation = self._validate_step(agent_id, step, domain)
            step.validation_confidence = validation["confidence"]
            step.consensus_supports = validation["supports"]
            step.validation_message = validation["message"]

            self._total_steps += 1

            if step.consensus_supports:
                trajectory.append(step)
                self._total_validated += 1
                current_state = predicted_next
            else:
                self._total_rejected += 1
                break  # Stop planning at first rejected step

        return trajectory

    def get_statistics(self) -> dict[str, Any]:
        total = max(self._total_steps, 1)
        plans = max(self._total_plans, 1)
        return {
            "total_plans": self._total_plans,
            "total_steps_imagined": self._total_steps,
            "total_steps_validated": self._total_validated,
            "total_steps_rejected": self._total_rejected,
            "validation_rate": self._total_validated / total,
            "rejection_rate": self._total_rejected / total,
            "avg_validated_per_plan": self._total_validated / plans,
        }

    # --- Internal ---

    def _select_action(self, policy: Any, state: Any) -> Any:
        if hasattr(policy, "select_action"):
            return policy.select_action(state)
        if callable(policy):
            return policy(state)
        return {"type": "forward"}

    def _predict_transition(self, state: Any, action: Any) -> tuple[Any, float]:
        if hasattr(self._world_model, "predict"):
            next_state = self._world_model.predict(state, action)
            reward = (
                self._world_model.predict_reward(state, action)
                if hasattr(self._world_model, "predict_reward")
                else 0.5
            )
            return next_state, reward
        return state, 0.5

    def _validate_step(
        self, agent_id: str, step: TrajectoryStep, domain: str
    ) -> dict[str, Any]:
        """Three-strategy validation cascade."""
        # Strategy 1: Use validator's historical accuracy
        if self._validator is not None:
            accuracy = self._validator.get_imagination_accuracy(agent_id, domain)
            if accuracy is not None and accuracy.total_predictions >= 5:
                confidence = accuracy.prediction_accuracy
                return {
                    "confidence": confidence,
                    "supports": confidence >= self._min_confidence,
                    "message": f"Historical accuracy: {confidence:.2f}",
                }

        # Strategy 2: Default confidence (no history available)
        default_confidence = 0.7
        return {
            "confidence": default_confidence,
            "supports": default_confidence >= self._min_confidence,
            "message": "Default confidence (no history)",
        }

    def __repr__(self) -> str:
        return (
            f"ValidatedImaginationPlanner(plans={self._total_plans}, "
            f"validated={self._total_validated})"
        )
