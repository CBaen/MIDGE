"""Convergence Mixin - Safety + Satisfaction tracking.

Prevents infinite learning loops by tracking policy improvement rates
and determining when an agent has reached a stable, optimal policy.

Ported from v5-pivot base_agent.py lines 226-238, 874-1058.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ConvergenceMixin:
    """Adds convergence tracking and satisfaction checking to agents."""

    def _init_convergence(self, agent_config: dict[str, Any]) -> None:
        """Initialize convergence attributes."""
        self.learning_iterations: int = 0
        self.max_learning_iterations: int = agent_config.get("max_learning_iterations", 100)
        self.convergence_threshold: float = agent_config.get("convergence_threshold", 0.01)
        self.policy_improvement_window: deque[float] = deque(maxlen=50)
        self.last_n_improvements: int = 10
        self.has_reached_convergence: bool = False

        self.satisfaction_score: float = 0.0
        self.satisfaction_threshold: float = agent_config.get("satisfaction_threshold", 0.85)
        self.team_satisfaction: Optional[float] = None
        self.is_satisfied_state: bool = False

    def has_converged(self) -> bool:
        """Check if agent has converged (policy improvement below threshold)."""
        if len(self.policy_improvement_window) < self.last_n_improvements:
            return False

        recent = list(self.policy_improvement_window)[-self.last_n_improvements :]
        avg_improvement = float(np.mean(recent))

        has_converged = avg_improvement < self.convergence_threshold

        if has_converged and not self.has_reached_convergence:
            self.has_reached_convergence = True
            logger.info(
                "%s CONVERGED (avg improvement: %.4f < %.4f)",
                getattr(self, "unique_id", "?"),
                avg_improvement,
                self.convergence_threshold,
            )

        return has_converged

    def record_policy_improvement(
        self, old_performance: float, new_performance: float
    ) -> None:
        """Record improvement for convergence tracking."""
        improvement = new_performance - old_performance
        self.policy_improvement_window.append(improvement)

    def _compute_improvement_rate(self) -> float:
        """Compute rate of performance improvement over recent history."""
        performance_history = getattr(self, "performance_history", [])
        if len(performance_history) < 10:
            return 0.0

        window_size = min(20, len(performance_history))
        recent = list(performance_history)[-window_size:]
        half = len(recent) // 2
        first_half_avg = float(np.mean(recent[:half]))
        second_half_avg = float(np.mean(recent[half:]))

        if first_half_avg > 0:
            return (second_half_avg - first_half_avg) / first_half_avg
        return 0.0

    def compute_satisfaction(self) -> float:
        """Compute current satisfaction score based on performance."""
        performance_history = getattr(self, "performance_history", [])
        if not performance_history:
            return 0.0

        recent = list(performance_history)[-20:]
        avg_performance = float(np.mean(recent))
        improvement_rate = self._compute_improvement_rate()

        self.satisfaction_score = min(1.0, avg_performance * 0.7 + max(0, improvement_rate) * 0.3)
        self.is_satisfied_state = self.satisfaction_score >= self.satisfaction_threshold
        return self.satisfaction_score

    def is_satisfied(self) -> bool:
        """Check if agent is satisfied with current performance."""
        self.compute_satisfaction()
        return self.is_satisfied_state

    def should_continue_learning(self) -> bool:
        """Determine if agent should continue active learning."""
        step_count = getattr(self, "step_count", 0)

        if self.learning_iterations >= self.max_learning_iterations:
            return False

        if self.has_converged():
            return step_count % 10 == 0

        if self.is_satisfied():
            return step_count % 5 == 0

        return True

    def _serialize_convergence(self) -> dict:
        return {
            "learning_iterations": self.learning_iterations,
            "policy_improvement_window": list(self.policy_improvement_window),
            "has_reached_convergence": self.has_reached_convergence,
            "satisfaction_score": self.satisfaction_score,
            "is_satisfied_state": self.is_satisfied_state,
        }

    def _restore_convergence(self, data: dict) -> None:
        self.learning_iterations = data.get("learning_iterations", 0)
        if "policy_improvement_window" in data:
            for v in data["policy_improvement_window"]:
                self.policy_improvement_window.append(v)
        self.has_reached_convergence = data.get("has_reached_convergence", False)
        self.satisfaction_score = data.get("satisfaction_score", 0.0)
        self.is_satisfied_state = data.get("is_satisfied_state", False)
