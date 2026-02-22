"""Transfer Learning Engine - Knowledge reuse across tasks.

Finds similar past tasks, transfers policies/experiences/value functions.
Measures transfer speedup (jumpstart + asymptotic improvement).

Biological analogy: Generalization from past experience.
Based on: Taylor & Stone (2009) "Transfer Learning for RL".
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

from .knowledge_base import KnowledgeBase, TaskDescriptor

logger = logging.getLogger(__name__)


class TransferStrategy(Enum):
    POLICY_TRANSFER = "policy_transfer"
    EXPERIENCE_REPLAY = "experience_replay"
    VALUE_INITIALIZATION = "value_initialization"
    FEATURE_EXTRACTION = "feature_extraction"
    CURRICULUM = "curriculum"
    COMBINED = "combined"


@dataclass
class TransferResult:
    """Outcome of a transfer operation."""

    target_task_id: str
    source_task_ids: list[str]
    strategy: str
    success: bool
    jumpstart: float = 0.0  # Initial performance boost
    speedup: float = 1.0  # Learning speed multiplier
    policy_transferred: bool = False
    experiences_transferred: int = 0
    value_initialized: bool = False
    timestamp: float = field(default_factory=time.time)


class TransferLearningEngine:
    """Finds similar tasks and transfers knowledge to accelerate learning.

    Works with KnowledgeBase for episode/policy storage and retrieval.
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        default_strategy: TransferStrategy = TransferStrategy.COMBINED,
        min_source_performance: float = 0.5,
    ) -> None:
        self._kb = knowledge_base
        self._default_strategy = default_strategy
        self._min_perf = min_source_performance
        self._transfer_history: list[TransferResult] = []
        self._lock = threading.RLock()

    def initiate_transfer(
        self,
        target_task: TaskDescriptor,
        agent_id: str,
        strategy: TransferStrategy | None = None,
        min_similarity: float = 0.3,
        k_source_tasks: int = 3,
        same_type_only: bool = False,
    ) -> TransferResult:
        """Begin knowledge transfer to a new task."""
        strategy = strategy or self._default_strategy
        self._kb.register_task(target_task)

        # Find similar source tasks
        similar = self._kb.retrieve_similar_tasks(
            target_task, min_similarity, k_source_tasks
        )

        if not similar:
            result = TransferResult(
                target_task_id=target_task.task_id,
                source_task_ids=[],
                strategy=strategy.value,
                success=False,
            )
            self._transfer_history.append(result)
            return result

        source_ids = [t.task_id for t, _ in similar]
        policy_transferred = False
        experiences_transferred = 0
        value_initialized = False

        for source_task, sim_score in similar:
            sid = source_task.task_id
            perf = self._kb.get_task_performance(sid)
            if perf < self._min_perf:
                continue

            if strategy in (TransferStrategy.POLICY_TRANSFER, TransferStrategy.COMBINED):
                best = self._kb.retrieve_best_policy(sid)
                if best:
                    # Store as initial policy for target
                    self._kb.store_policy(
                        target_task.task_id, agent_id,
                        best["policy_state"], best["performance"] * sim_score,
                    )
                    policy_transferred = True

            if strategy in (TransferStrategy.EXPERIENCE_REPLAY, TransferStrategy.COMBINED):
                episodes = self._kb.retrieve_successful_episodes(sid, limit=10)
                experiences_transferred += len(episodes)
                for ep in episodes:
                    self._kb.store_episode(
                        target_task.task_id, agent_id,
                        ep.transitions, ep.total_reward * sim_score,
                        ep.success, {"source_task": sid, "transfer": True},
                    )

            if strategy in (TransferStrategy.VALUE_INITIALIZATION, TransferStrategy.COMBINED):
                vf = self._kb.retrieve_value_function(sid)
                if vf:
                    self._kb.store_value_function(
                        target_task.task_id, agent_id, vf["value_state"]
                    )
                    value_initialized = True

        result = TransferResult(
            target_task_id=target_task.task_id,
            source_task_ids=source_ids,
            strategy=strategy.value,
            success=policy_transferred or experiences_transferred > 0 or value_initialized,
            policy_transferred=policy_transferred,
            experiences_transferred=experiences_transferred,
            value_initialized=value_initialized,
        )
        self._transfer_history.append(result)
        return result

    def record_performance(
        self,
        agent_id: str,
        task_id: str,
        performance: float,
    ) -> TransferResult | None:
        """Record agent performance and attempt transfer if history is sufficient.

        Convenience method that wraps TaskDescriptor creation. Called from
        agent lifecycle when reward is high — accumulates task history, then
        initiates transfer when enough related tasks exist.
        """
        td = TaskDescriptor(
            task_id=task_id,
            task_type="agent_step",
            description=f"Agent {agent_id} step performance",
        )
        self._kb.register_task(td)
        self._kb.store_episode(
            task_id, agent_id, transitions=[], total_reward=performance,
            success=performance > 0.5,
        )
        # Attempt transfer once enough history accumulates
        history_count = len(self._kb.retrieve_successful_episodes(task_id, limit=100))
        if history_count >= 3:
            return self.initiate_transfer(td, agent_id)
        return None

    def evaluate_transfer(
        self,
        baseline_performance: float,
        transfer_performance: float,
        baseline_samples: int,
        transfer_samples: int,
    ) -> dict[str, float]:
        """Measure transfer effectiveness."""
        jumpstart = transfer_performance - baseline_performance
        speedup = (
            baseline_samples / max(transfer_samples, 1)
            if baseline_samples > 0
            else 1.0
        )
        return {
            "jumpstart": jumpstart,
            "speedup": speedup,
            "baseline_performance": baseline_performance,
            "transfer_performance": transfer_performance,
        }

    def get_average_speedup(self) -> float:
        successful = [r for r in self._transfer_history if r.success]
        if not successful:
            return 1.0
        return float(np.mean([r.speedup for r in successful]))

    def get_statistics(self) -> dict[str, Any]:
        return {
            "total_transfers": len(self._transfer_history),
            "successful_transfers": sum(1 for r in self._transfer_history if r.success),
            "average_speedup": self.get_average_speedup(),
        }

    def __repr__(self) -> str:
        return f"TransferLearning(transfers={len(self._transfer_history)})"
