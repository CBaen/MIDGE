"""Transfer Learning Mixin - Knowledge transfer across tasks.

Enables agents to transfer learned knowledge between tasks,
with MAML meta-learning for few-shot adaptation.

Ported from v5-pivot base_agent.py Big Rock 8 transfer methods.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class TransferLearningMixin:
    """Adds transfer learning and MAML meta-learning to agents."""

    def _init_transfer_learning(
        self,
        knowledge_base: Any = None,
        transfer_engine: Any = None,
        maml_learner: Any = None,
        agent_config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize transfer learning attributes."""
        config = agent_config or {}
        self.knowledge_base = knowledge_base
        self.transfer_engine = transfer_engine
        self.maml_learner = maml_learner
        self.current_task: Optional[Any] = None
        self.episode_transitions: deque[Any] = deque(maxlen=1000)
        self.transfer_enabled: bool = config.get("transfer_enabled", False)
        self.maml_enabled: bool = config.get("maml_enabled", False)

    def begin_new_task(
        self,
        task_descriptor: Any,
        use_transfer: bool = True,
        use_maml: bool = False,
        min_similarity: float = 0.5,
        k_source_tasks: int = 3,
    ) -> dict[str, Any]:
        """Begin learning a new task with optional transfer/MAML."""
        self.current_task = task_descriptor
        result: dict[str, Any] = {
            "task_id": getattr(task_descriptor, "task_id", "unknown"),
            "transfer_used": False,
            "maml_used": False,
            "speedup_estimate": 1.0,
        }

        if use_transfer and self.transfer_engine and self.transfer_enabled:
            try:
                transfer_result = self.transfer_engine.initiate_transfer(
                    target_task=task_descriptor,
                    agent_id=getattr(self, "unique_id", "unknown"),
                    min_similarity=min_similarity,
                    k_source_tasks=k_source_tasks,
                )
                result["transfer_used"] = True
                result["transfer_result"] = transfer_result
            except Exception:
                logger.exception("Transfer learning failed")

        if use_maml and self.maml_learner and self.maml_enabled:
            try:
                if self.knowledge_base:
                    support = self.knowledge_base.retrieve_successful_episodes(
                        task_id=getattr(task_descriptor, "task_id", ""), k=5
                    )
                    if support:
                        adaptation = self.maml_learner.adapt_to_task(
                            target_task=task_descriptor,
                            agent_id=getattr(self, "unique_id", "unknown"),
                            support_episodes=support,
                        )
                        result["maml_used"] = True
                        result["maml_result"] = adaptation
            except Exception:
                logger.exception("MAML adaptation failed")

        return result

    def store_transition_for_transfer(
        self,
        state: np.ndarray,
        action: Any,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Store experience transition for future transfer."""
        if not self.knowledge_base or not self.current_task:
            return
        transition = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
            "metadata": metadata or {},
        }
        self.episode_transitions.append(transition)

    def store_episode_for_transfer(
        self, total_reward: float, success: bool, clear_buffer: bool = True
    ) -> Optional[str]:
        """Store completed episode for future transfer."""
        if not self.knowledge_base or not self.current_task or not self.episode_transitions:
            return None
        task_id = getattr(self.current_task, "task_id", "unknown")
        agent_id = str(getattr(self, "unique_id", "unknown"))
        episode_id = self.knowledge_base.store_episode(
            task_id=task_id,
            agent_id=agent_id,
            transitions=list(self.episode_transitions),
            total_reward=total_reward,
            success=success,
        )
        if clear_buffer:
            self.episode_transitions.clear()
        return episode_id

    def store_policy_for_transfer(self, policy: Any) -> None:
        """Store current policy for future transfer."""
        if self.knowledge_base and self.current_task:
            task_id = getattr(self.current_task, "task_id", "unknown")
            agent_id = str(getattr(self, "unique_id", "unknown"))
            policy_state = policy if isinstance(policy, dict) else {"weights": policy}
            self.knowledge_base.store_policy(
                task_id=task_id,
                agent_id=agent_id,
                policy_state=policy_state,
            )

    def store_value_function_for_transfer(self, value_function: Any) -> None:
        """Store current value function for future transfer."""
        if self.knowledge_base and self.current_task:
            task_id = getattr(self.current_task, "task_id", "unknown")
            agent_id = str(getattr(self, "unique_id", "unknown"))
            value_state = value_function if isinstance(value_function, dict) else {"weights": value_function}
            self.knowledge_base.store_value_function(
                task_id=task_id,
                agent_id=agent_id,
                value_state=value_state,
            )

    def evaluate_transfer_performance(
        self,
        baseline_performance: float,
        current_performance: float,
        baseline_samples: int,
        current_samples: int,
    ) -> dict[str, Any]:
        """Evaluate effectiveness of transfer learning."""
        if not self.transfer_engine or not self.current_task:
            return {}
        return self.transfer_engine.evaluate_transfer(
            baseline_performance=baseline_performance,
            transfer_performance=current_performance,
            baseline_samples=baseline_samples,
            transfer_samples=current_samples,
        )

    def get_transfer_statistics(self) -> Optional[dict[str, Any]]:
        """Get transfer learning statistics."""
        if not self.transfer_engine:
            return None
        return {
            "transfer_enabled": self.transfer_enabled,
            "maml_enabled": self.maml_enabled,
            "current_task": getattr(self.current_task, "task_id", None) if self.current_task else None,
            "episode_buffer_size": len(self.episode_transitions),
        }

    def get_maml_statistics(self) -> Optional[dict[str, Any]]:
        """Get MAML meta-learning statistics."""
        if not self.maml_learner:
            return None
        return {
            "maml_enabled": self.maml_enabled,
            "meta_initialized": getattr(self.maml_learner, "meta_initialized", False),
        }

    def _serialize_transfer_learning(self) -> dict:
        return {
            "transfer_enabled": getattr(self, "transfer_enabled", False),
            "maml_enabled": getattr(self, "maml_enabled", False),
        }

    def _restore_transfer_learning(self, data: dict) -> None:
        if "transfer_enabled" in data:
            self.transfer_enabled = data["transfer_enabled"]
        if "maml_enabled" in data:
            self.maml_enabled = data["maml_enabled"]
