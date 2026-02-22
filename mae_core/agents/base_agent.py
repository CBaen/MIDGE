"""Base Agent - Mesa 3.4 Agent with lifecycle and state management.

Thin base providing Mesa integration, step lifecycle, and common state.
All domain capabilities come from mixins composed in MycelialAgent.

Ported from v5-pivot base_agent.py core scaffolding (~200 lines).
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any, Optional

import numpy as np
from mesa import Agent

logger = logging.getLogger(__name__)


class BaseAgent(Agent):
    """Minimal Mesa agent with lifecycle hooks and common state.

    This agent provides:
    - Mesa 3.4 integration (auto-registered via super().__init__)
    - Step lifecycle: observe -> decide -> act -> learn -> communicate
    - Common state tracking (step count, reward, performance)
    - Override points for subclasses
    """

    def __init__(
        self,
        model: Any,
        agent_type: str = "base",
        agent_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(model)

        self.agent_type: str = agent_type
        self.agent_config: dict[str, Any] = agent_config or {}

        # Learning rate (used by MemoryConsolidator during sleep replay)
        self._learning_rate: float = 0.01

        # Core state
        self.step_count: int = 0
        self.cumulative_reward: float = 0.0
        self.current_state: Optional[dict[str, Any]] = None
        self.last_action: Optional[Any] = None
        self.last_reward: float = 0.0
        self.risk_score: float = 0.0

        # Performance tracking
        self.performance_history: deque[float] = deque(maxlen=200)
        self.reward_history: deque[float] = deque(maxlen=200)
        self.creation_time: float = time.time()

    def step(self) -> None:
        """Execute one agent lifecycle step.

        The canonical lifecycle:
        1. Observe environment
        2. Decide on action (policy or world model)
        3. Act (execute action)
        4. Learn (update policy, replay memory)
        5. Communicate (signals, stigmergy, GNN)
        """
        self.step_count += 1

        # Triage queued signals before processing (thalamus)
        resolver = getattr(self, "_signal_resolver", None)
        if resolver is not None:
            resolver.process()

        try:
            self._observe()
            action = self._decide()
            reward = self._act(action)
            self._learn(action, reward)
            self._communicate()
        except Exception:
            logger.exception(
                "%s: Error in step %d", self.unique_id, self.step_count
            )

    def _observe(self) -> None:
        """Observe the environment. Override in subclasses."""
        pass

    def _decide(self) -> Any:
        """Decide on an action. Override in subclasses."""
        return self._select_action(self.current_state)

    def _act(self, action: Any) -> float:
        """Execute action, return reward. Override in subclasses."""
        self.last_action = action
        return 0.0

    def _learn(self, action: Any, reward: float) -> None:
        """Update learning from the step. Override in subclasses."""
        self.last_reward = reward
        self.cumulative_reward += reward
        self.reward_history.append(reward)

    def _communicate(self) -> None:
        """Post-step communication. Override in subclasses."""
        pass

    def _select_action(self, state: Optional[dict[str, Any]]) -> Any:
        """Select action from policy. Override in subclasses."""
        return 0

    def get_state(self) -> dict[str, Any]:
        """Get agent state for persistence or monitoring."""
        return {
            "unique_id": self.unique_id,
            "agent_type": self.agent_type,
            "step_count": self.step_count,
            "cumulative_reward": self.cumulative_reward,
            "last_reward": self.last_reward,
            "risk_score": self.risk_score,
            "alive_seconds": time.time() - self.creation_time,
        }

    def serialize_state(self) -> dict[str, Any]:
        """Serialize accumulated agent state for persistence across restarts."""
        return {
            "unique_id": self.unique_id,
            "agent_type": self.agent_type,
            "step_count": self.step_count,
            "cumulative_reward": self.cumulative_reward,
            "last_reward": self.last_reward,
            "risk_score": self.risk_score,
            "performance_history": list(self.performance_history),
            "reward_history": list(self.reward_history),
        }

    def restore_state(self, data: dict[str, Any]) -> None:
        """Restore agent state from persisted data."""
        self.step_count = data.get("step_count", 0)
        self.cumulative_reward = data.get("cumulative_reward", 0.0)
        self.last_reward = data.get("last_reward", 0.0)
        self.risk_score = data.get("risk_score", 0.0)
        if "performance_history" in data:
            self.performance_history = deque(data["performance_history"], maxlen=200)
        if "reward_history" in data:
            self.reward_history = deque(data["reward_history"], maxlen=200)

    def get_performance_summary(self) -> dict[str, Any]:
        """Get performance metrics summary."""
        rewards = list(self.reward_history)
        if not rewards:
            return {
                "mean_reward": 0.0,
                "total_reward": 0.0,
                "steps": self.step_count,
            }

        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "total_reward": self.cumulative_reward,
            "recent_mean": float(np.mean(rewards[-20:])) if len(rewards) >= 20 else float(np.mean(rewards)),
            "steps": self.step_count,
        }

    def get_learning_rate(self) -> float:
        """Return the agent's current learning rate.

        Used by MemoryConsolidator to lower the rate during
        consolidation (sleep replay) and restore it afterwards.
        """
        return self._learning_rate

    def set_learning_rate(self, rate: float) -> None:
        """Set the agent's learning rate.

        Used by MemoryConsolidator to scale the rate down during
        consolidation and restore the original value afterwards.
        """
        self._learning_rate = rate

    def get_local_reward(self, global_reward: float) -> float:
        """Receive global reward for VDN distribution.

        Called by MycelialModel._distribute_rewards() to pass the
        system-wide reward to each agent. Stores the value for
        downstream learning use and returns it unchanged.
        """
        self.last_reward = global_reward
        return global_reward
