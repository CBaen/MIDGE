"""Advanced Features Mixin - World model, morphogenesis, planning.

Phase 4 capabilities: validated imagination (world model planning),
dynamic organ formation (morphogenesis), and generative memory
coordination.

Ported from v5-pivot base_agent.py Phase 4 methods.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class AdvancedFeaturesMixin:
    """Adds world model planning, morphogenesis, and Phase 4 coordination."""

    def _init_advanced_features(
        self,
        world_model: Any = None,
        morphogenesis_enabled: bool = False,
        decision_router: Any = None,
        causal_engine: Any = None,
        agent_config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize Phase 4 advanced feature attributes."""
        config = agent_config or {}
        self.world_model = world_model
        self.decision_router = decision_router
        self.causal_engine = causal_engine
        self.morphogenesis_enabled: bool = morphogenesis_enabled
        self.world_model_enabled: bool = config.get("world_model_enabled", False)
        self.planning_horizon: int = config.get("planning_horizon", 5)
        self.use_world_model_planning: bool = False

    def use_world_model(self) -> Any:
        """Plan actions using world model imagination.

        Imagines multiple action sequences and selects the best one
        based on predicted rewards. Enables model-based planning
        without environment interaction.
        """
        if not self.world_model or not self.world_model_enabled:
            select_action = getattr(self, "_select_action", None)
            current_state = getattr(self, "current_state", None)
            if select_action and current_state is not None:
                return select_action(current_state)
            return None

        try:
            import numpy as np

            current_state = getattr(self, "current_state", {})
            if isinstance(current_state, dict):
                state_vec = current_state.get("state_vector", current_state.get("observation", None))
            else:
                state_vec = None

            if state_vec is None:
                return self._select_action(current_state) if hasattr(self, "_select_action") else None

            state_np = np.asarray(state_vec, dtype=np.float32)

            best_action = None
            best_value = float("-inf")

            num_actions = getattr(self, "agent_config", {}).get("num_actions", 4)
            for action_idx in range(num_actions):
                pred = self.world_model.step(state_np, action_idx, deterministic=True)
                reward_value = pred.reward

                if reward_value > best_value:
                    best_value = reward_value
                    best_action = action_idx

            return best_action

        except Exception:
            logger.exception("Error in world model planning")
            select_action = getattr(self, "_select_action", None)
            current_state = getattr(self, "current_state", None)
            if select_action and current_state is not None:
                return select_action(current_state)
            return None

    def enable_morphogenesis(self) -> None:
        """Enable agent participation in dynamic organ formation.

        When enabled, the agent can be recruited into specialized teams
        (organs) that emerge in response to novel problem signatures.
        """
        self.morphogenesis_enabled = True

        emit_signal = getattr(self, "emit_signal", None)
        if emit_signal is not None:
            compute_satisfaction = getattr(self, "compute_satisfaction", lambda: 0.0)
            emit_signal(
                "COLLABORATION",
                {
                    "event": "morphogenesis_enabled",
                    "agent_id": getattr(self, "unique_id", "unknown"),
                    "capabilities": list(getattr(self, "capabilities", [])),
                    "performance": compute_satisfaction(),
                },
            )

    def get_advanced_statistics(self) -> dict[str, Any]:
        """Get comprehensive Phase 4 statistics."""
        stats: dict[str, Any] = {
            "world_model_enabled": self.world_model_enabled,
            "morphogenesis_enabled": self.morphogenesis_enabled,
            "use_world_model_planning": self.use_world_model_planning,
            "planning_horizon": self.planning_horizon,
        }

        if self.world_model:
            stats["world_model"] = {
                "state_dim": getattr(self.world_model, "state_dim", None),
                "action_dim": getattr(self.world_model, "action_dim", None),
                "use_ensemble": getattr(self.world_model, "use_ensemble", False),
            }

        generative_memory = getattr(self, "generative_memory", None)
        if generative_memory:
            get_stats = getattr(generative_memory, "get_statistics", None)
            if get_stats:
                stats["generative_memory"] = get_stats()

        return stats

    def _serialize_advanced_features(self) -> dict:
        return {
            "morphogenesis_enabled": getattr(self, "morphogenesis_enabled", False),
            "world_model_enabled": getattr(self, "world_model_enabled", False),
            "use_world_model_planning": getattr(self, "use_world_model_planning", False),
        }

    def _restore_advanced_features(self, data: dict) -> None:
        if "morphogenesis_enabled" in data:
            self.morphogenesis_enabled = data["morphogenesis_enabled"]
        if "world_model_enabled" in data:
            self.world_model_enabled = data["world_model_enabled"]
        if "use_world_model_planning" in data:
            self.use_world_model_planning = data["use_world_model_planning"]
