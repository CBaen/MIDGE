"""Imitation Learning - Learning from demonstration.

Agents observe expert behavior and learn to imitate it.
Supports behavioral cloning, selective imitation, and
teacher evaluation.

Biological analogy: Social learning, mirror neurons.
Based on: Ross et al. (2011) "DAgger", Ho & Ermon (2016) "GAIL".
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ObservedBehavior:
    """A single observed behavior from an expert."""

    actor_id: str
    action: Any
    context: dict[str, Any]
    outcome: float  # reward or success signal
    timestamp: float = field(default_factory=time.time)


@dataclass
class DemonstrationTrajectory:
    """A complete demonstration sequence."""

    teacher_id: str
    behaviors: list[ObservedBehavior]
    total_reward: float = 0.0
    success: bool = False
    task_id: str = ""


@dataclass
class ImitationPolicy:
    """A learned state-action mapping from demonstrations."""

    policy_id: str
    source_teacher: str
    context_action_map: dict[str, Any] = field(default_factory=dict)
    performance: float = 0.0
    training_samples: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class TeacherProfile:
    """Reliability profile of a demonstration source."""

    teacher_id: str
    demonstrations_observed: int = 0
    average_outcome: float = 0.0
    reliability_score: float = 0.5  # [0, 1]
    last_observed: float = 0.0


class ImitationLearning:
    """Learning from demonstration via behavioral cloning and selective imitation."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        similarity_threshold: float = 0.7,
        max_observations: int = 10000,
    ) -> None:
        self._lr = learning_rate
        self._sim_threshold = similarity_threshold

        self._observations: deque[ObservedBehavior] = deque(maxlen=max_observations)
        self._learned_policies: dict[str, ImitationPolicy] = {}
        self._teacher_profiles: dict[str, TeacherProfile] = {}
        self._behavioral_library: dict[str, Any] = {}  # context_hash -> action
        self._lock = threading.RLock()

        # Statistics
        self._total_observations = 0
        self._successful_imitations = 0
        self._failed_imitations = 0

    def observe_behavior(
        self,
        actor_id: str,
        action: Any,
        context: dict[str, Any],
        outcome: float,
    ) -> ObservedBehavior:
        """Record an observed behavior from an expert."""
        obs = ObservedBehavior(
            actor_id=actor_id,
            action=action,
            context=context,
            outcome=outcome,
        )

        with self._lock:
            self._observations.append(obs)
            self._total_observations += 1

            # Update teacher profile
            if actor_id not in self._teacher_profiles:
                self._teacher_profiles[actor_id] = TeacherProfile(teacher_id=actor_id)
            profile = self._teacher_profiles[actor_id]
            profile.demonstrations_observed += 1
            # EMA of outcomes
            profile.average_outcome = (
                0.9 * profile.average_outcome + 0.1 * outcome
            )
            profile.last_observed = time.time()
            profile.reliability_score = min(1.0, profile.average_outcome)

            # Store in behavioral library if outcome is good
            if outcome > 0:
                ctx_key = self._context_key(context)
                self._behavioral_library[ctx_key] = action

        return obs

    def learn_from_demonstration(
        self, trajectory: DemonstrationTrajectory
    ) -> ImitationPolicy:
        """Learn a policy from a complete demonstration."""
        context_action_map: dict[str, Any] = {}

        for behavior in trajectory.behaviors:
            ctx_key = self._context_key(behavior.context)
            context_action_map[ctx_key] = behavior.action

        policy = ImitationPolicy(
            policy_id=f"imit-{len(self._learned_policies)}",
            source_teacher=trajectory.teacher_id,
            context_action_map=context_action_map,
            performance=trajectory.total_reward,
            training_samples=len(trajectory.behaviors),
        )

        with self._lock:
            self._learned_policies[policy.policy_id] = policy

        return policy

    def imitate(self, context: dict[str, Any]) -> Any | None:
        """Execute learned behavior for a given context."""
        ctx_key = self._context_key(context)

        with self._lock:
            # Check behavioral library first
            if ctx_key in self._behavioral_library:
                self._successful_imitations += 1
                return self._behavioral_library[ctx_key]

            # Check learned policies
            for policy in self._learned_policies.values():
                if ctx_key in policy.context_action_map:
                    self._successful_imitations += 1
                    return policy.context_action_map[ctx_key]

            # Find similar context
            similar = self.find_similar_context(context)
            if similar:
                self._successful_imitations += 1
                return self._behavioral_library.get(similar)

            self._failed_imitations += 1
            return None

    def find_similar_context(self, context: dict[str, Any]) -> str | None:
        """Find the most similar stored context."""
        target_key = self._context_key(context)
        target_set = set(str(v) for v in context.values())

        best_key = None
        best_sim = 0.0

        with self._lock:
            for stored_key in self._behavioral_library:
                # Simple Jaccard similarity on context keys
                stored_set = set(stored_key.split("|"))
                if not target_set or not stored_set:
                    continue
                intersection = len(target_set & stored_set)
                union = len(target_set | stored_set)
                sim = intersection / union if union > 0 else 0.0
                if sim > best_sim and sim >= self._sim_threshold:
                    best_sim = sim
                    best_key = stored_key

        return best_key

    def selective_imitation(self, behavior: ObservedBehavior) -> bool:
        """Decide whether to imitate a behavior (quality filter)."""
        profile = self._teacher_profiles.get(behavior.actor_id)
        if profile is None:
            return behavior.outcome > 0

        # Imitate if teacher is reliable AND outcome was good
        return profile.reliability_score > 0.5 and behavior.outcome > 0

    def evaluate_teacher(self, teacher_id: str) -> float:
        """Get reliability score for a teacher. Returns [0, 1]."""
        profile = self._teacher_profiles.get(teacher_id)
        return profile.reliability_score if profile else 0.0

    def combine_demonstrations(
        self, trajectories: list[DemonstrationTrajectory]
    ) -> ImitationPolicy:
        """Learn from multiple demonstrations (multi-teacher)."""
        combined_map: dict[str, Any] = {}
        total_reward = 0.0
        total_samples = 0

        for traj in trajectories:
            weight = self.evaluate_teacher(traj.teacher_id)
            for behavior in traj.behaviors:
                ctx_key = self._context_key(behavior.context)
                # Higher-reliability teacher overwrites
                if ctx_key not in combined_map or weight > 0.5:
                    combined_map[ctx_key] = behavior.action
            total_reward += traj.total_reward
            total_samples += len(traj.behaviors)

        policy = ImitationPolicy(
            policy_id=f"imit-combined-{len(self._learned_policies)}",
            source_teacher="combined",
            context_action_map=combined_map,
            performance=total_reward / max(len(trajectories), 1),
            training_samples=total_samples,
        )

        with self._lock:
            self._learned_policies[policy.policy_id] = policy
        return policy

    def get_imitation_success_rate(self) -> float:
        total = self._successful_imitations + self._failed_imitations
        return self._successful_imitations / total if total > 0 else 0.0

    def get_statistics(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_observations": self._total_observations,
                "learned_policies": len(self._learned_policies),
                "behavioral_library_size": len(self._behavioral_library),
                "teacher_profiles": len(self._teacher_profiles),
                "successful_imitations": self._successful_imitations,
                "failed_imitations": self._failed_imitations,
                "success_rate": self.get_imitation_success_rate(),
            }

    @staticmethod
    def _context_key(context: dict[str, Any]) -> str:
        """Create a hashable key from context."""
        return "|".join(f"{k}={v}" for k, v in sorted(context.items()))

    def __repr__(self) -> str:
        return (
            f"ImitationLearning(observations={self._total_observations}, "
            f"policies={len(self._learned_policies)})"
        )
