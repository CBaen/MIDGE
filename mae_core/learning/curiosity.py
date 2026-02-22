"""Curiosity Drive - Intrinsic motivation via novelty and prediction error.

Computes intrinsic rewards from state novelty, information gain, and
prediction error. Drives exploration in sparse-reward environments.

Biological analogy: Dopaminergic curiosity response.
Based on: Pathak et al. (2017) "ICM", Burda et al. (2019) "RND".
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class CuriosityType(Enum):
    EPISTEMIC = "epistemic"  # Uncertainty-driven
    PERCEPTUAL = "perceptual"  # Novelty-driven
    SPECIFIC = "specific"  # Goal-directed
    DIVERSIVE = "diversive"  # Exploration-driven


@dataclass
class CuriositySignal:
    """A curiosity reward signal combining multiple components."""

    total_reward: float
    novelty: float = 0.0
    information_gain: float = 0.0
    prediction_error: float = 0.0
    curiosity_type: CuriosityType = CuriosityType.PERCEPTUAL


class CuriosityDrive:
    """Intrinsic motivation system that rewards exploration.

    Three components:
    1. Novelty: How different is this state from previously seen states?
    2. Information gain: How much does this transition reduce uncertainty?
    3. Prediction error: How surprising is this outcome?

    EventBus integration:
    - Subscribes to memory.novel_experience to boost curiosity when
      MemoryCoordinator detects genuinely novel states.
    """

    def __init__(
        self,
        novelty_weight: float = 0.4,
        info_gain_weight: float = 0.3,
        prediction_error_weight: float = 0.3,
        exploration_bonus: float = 0.1,
        novelty_decay: float = 0.95,
        visit_count_threshold: int = 10,
        event_bus: Any = None,
    ) -> None:
        self._novelty_w = novelty_weight
        self._info_gain_w = info_gain_weight
        self._pred_error_w = prediction_error_weight
        self._exploration_bonus = exploration_bonus
        self._novelty_decay = novelty_decay
        self._visit_threshold = visit_count_threshold

        self._state_visits: dict[str, int] = defaultdict(int)
        self._state_predictions: dict[str, np.ndarray] = {}
        self._novelty_scores: dict[str, float] = defaultdict(lambda: 1.0)
        self._lock = threading.RLock()

        # Statistics
        self._total_signals = 0
        self._exploration_count = 0
        self._exploitation_count = 0
        self._novel_experience_count = 0

        # EventBus: subscribe to novel experience events from memory
        self._event_bus = event_bus
        if event_bus is not None:
            event_bus.register_callback(
                "memory.novel_experience", self._on_novel_experience
            )
            event_bus.register_callback(
                "memory.experience_stored", self._on_experience_stored
            )
            event_bus.register_callback(
                "memory.consolidation_complete", self._on_consolidation_complete
            )

    def compute_curiosity_reward(
        self,
        state: np.ndarray,
        action: Any,
        next_state: np.ndarray,
        predicted_next_state: np.ndarray | None = None,
    ) -> CuriositySignal:
        """Compute intrinsic curiosity reward for a transition."""
        novelty = self.compute_novelty(state)
        info_gain = self.compute_information_gain(state, next_state)
        pred_error = (
            self.compute_prediction_error(predicted_next_state, next_state)
            if predicted_next_state is not None
            else 0.0
        )

        total = (
            self._novelty_w * novelty
            + self._info_gain_w * info_gain
            + self._pred_error_w * pred_error
        )

        self.update_state_model(state, action, next_state)
        self._total_signals += 1

        return CuriositySignal(
            total_reward=total,
            novelty=novelty,
            information_gain=info_gain,
            prediction_error=pred_error,
        )

    def compute_novelty(self, state: np.ndarray) -> float:
        """State novelty based on visit frequency. Returns [0, 1]."""
        key = self._state_key(state)
        with self._lock:
            visits = self._state_visits[key]
            self._state_visits[key] += 1

        if visits == 0:
            return 1.0
        # Diminishing novelty with visits
        return 1.0 / (1.0 + visits)

    def compute_information_gain(
        self, state: np.ndarray, next_state: np.ndarray
    ) -> float:
        """Information gain from state transition. Returns [0, 1]."""
        # Approximation: distance between states normalized
        diff = np.linalg.norm(next_state - state)
        return float(min(1.0, diff / (np.linalg.norm(state) + 1e-8)))

    def compute_prediction_error(
        self, predicted: np.ndarray, actual: np.ndarray
    ) -> float:
        """How surprising was the outcome? Returns [0, 1]."""
        error = np.linalg.norm(predicted - actual)
        # Normalize by state magnitude
        return float(min(1.0, error / (np.linalg.norm(actual) + 1e-8)))

    def should_explore(self, state: np.ndarray) -> bool:
        """Explore vs exploit decision based on novelty."""
        novelty = self.compute_novelty(state)
        if novelty > 0.5:
            self._exploration_count += 1
            return True
        self._exploitation_count += 1
        return False

    def update_state_model(
        self, state: np.ndarray, action: Any, next_state: np.ndarray
    ) -> None:
        """Update internal model of state transitions."""
        key = self._state_key(state)
        with self._lock:
            # Simple running average of next states
            if key in self._state_predictions:
                self._state_predictions[key] = (
                    0.9 * self._state_predictions[key] + 0.1 * next_state
                )
            else:
                self._state_predictions[key] = next_state.copy()

            # Decay novelty scores
            for k in self._novelty_scores:
                self._novelty_scores[k] *= self._novelty_decay

    def get_exploration_targets(self, num_targets: int = 5) -> list[str]:
        """Get states with highest novelty for targeted exploration."""
        with self._lock:
            scored = [
                (k, self._novelty_scores[k])
                for k in self._novelty_scores
                if self._state_visits.get(k, 0) < self._visit_threshold
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            return [k for k, _ in scored[:num_targets]]

    def decay_novelty_scores(self) -> None:
        """Apply decay to all novelty scores."""
        with self._lock:
            for k in self._novelty_scores:
                self._novelty_scores[k] *= self._novelty_decay

    def combine_rewards(
        self, extrinsic: float, intrinsic: float, extrinsic_weight: float = 0.7
    ) -> float:
        """Blend extrinsic and intrinsic rewards."""
        return extrinsic_weight * extrinsic + (1 - extrinsic_weight) * intrinsic

    def _on_novel_experience(self, channel: str, data: Any) -> None:
        """Handle novel experience events from MemoryCoordinator.

        When memory detects a genuinely novel state (high semantic distance),
        boost the global exploration bonus temporarily. This creates the
        dopaminergic feedback loop: novel experience -> curiosity spike ->
        more exploration -> more novel experiences.
        """
        import json
        if isinstance(data, str):
            data = json.loads(data)
        novelty_score = data.get("novelty_score", 0.5)
        with self._lock:
            self._novel_experience_count += 1
            # Boost exploration bonus proportional to novelty
            # (decays naturally via novelty_decay each step)
            self._exploration_bonus = min(
                0.5, self._exploration_bonus + novelty_score * 0.05
            )
        logger.debug(
            "Curiosity boosted by novel experience (score=%.2f, agent=%s)",
            novelty_score, data.get("agent_id", "?"),
        )

    def _on_experience_stored(self, channel: str, data: Any) -> None:
        """Track novelty of stored experiences.

        As more experiences accumulate, gradually decay exploration bonus
        to shift toward exploitation (diminishing novelty over time).
        """
        import json
        if isinstance(data, str):
            data = json.loads(data)
        total = data.get("total_stored", 0)
        with self._lock:
            # Periodically decay exploration bonus as experience grows
            if total > 0 and total % 100 == 0:
                self._exploration_bonus = max(
                    0.01, self._exploration_bonus * self._novelty_decay
                )
        logger.debug(
            "Experience stored event received (total=%d, agent=%s)",
            total, data.get("agent_id", "?"),
        )

    def _on_consolidation_complete(self, channel: str, data: Any) -> None:
        """Reset exploration after memory consolidation.

        After consolidation, the agent's model has been updated.
        Boost exploration briefly to discover whether the updated
        model reveals new opportunities.
        """
        import json
        if isinstance(data, str):
            data = json.loads(data)
        with self._lock:
            # Post-consolidation exploration boost
            self._exploration_bonus = min(
                0.5, self._exploration_bonus + 0.1
            )
        logger.debug(
            "Curiosity boosted after consolidation (agent=%s, loss_reduction=%.3f)",
            data.get("agent_id", "?"),
            data.get("loss_reduction", 0.0),
        )

    def get_curiosity_metrics(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_signals": self._total_signals,
                "unique_states_visited": len(self._state_visits),
                "predictions_stored": len(self._state_predictions),
                "exploration_count": self._exploration_count,
                "exploitation_count": self._exploitation_count,
                "novel_experience_count": self._novel_experience_count,
                "explore_ratio": (
                    self._exploration_count / (self._exploration_count + self._exploitation_count)
                    if (self._exploration_count + self._exploitation_count) > 0
                    else 0.5
                ),
            }

    @staticmethod
    def _state_key(state: np.ndarray) -> str:
        return np.round(state, decimals=2).tobytes()[:32].hex()

    def __repr__(self) -> str:
        return (
            f"CuriosityDrive(states={len(self._state_visits)}, "
            f"signals={self._total_signals})"
        )
