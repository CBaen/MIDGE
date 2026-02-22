"""Value Decomposition Networks - Multi-agent credit assignment.

Decomposes joint team value into individual agent contributions.
Supports additive VDN, difference rewards, and counterfactual baselines.

Based on: Sunehag et al. (2018) "VDN", Rashid et al. (2018) "QMIX".
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

from ..backbone.event_bus import EventBus

logger = logging.getLogger(__name__)

CH_VALUE_SHARE = "vdn.value_share"


class DecompositionMethod(Enum):
    ADDITIVE = "additive"  # Q_tot = sum(Q_i)
    MONOTONIC = "monotonic"  # Q_tot = f(Q_1...Q_n) monotonic (QMIX)
    WEIGHTED = "weighted"  # Q_tot = sum(w_i * Q_i)


class CreditAssignmentStrategy(Enum):
    DIFFERENCE_REWARDS = "difference_rewards"
    SHAPLEY_VALUE = "shapley_value"
    COUNTERFACTUAL = "counterfactual"
    ATTENTION_BASED = "attention_based"


@dataclass
class CreditResult:
    """Result of credit assignment for one agent."""

    agent_id: str
    local_value: float
    credit: float
    marginal_contribution: float = 0.0


class ValueDecompositionEngine:
    """Multi-agent value decomposition with credit assignment.

    Each agent computes local Q-values. The engine decomposes joint
    value and assigns credit for cooperative learning.
    """

    def __init__(
        self,
        agent_id: str,
        event_bus: EventBus,
        state_dim: int = 10,
        action_dim: int = 5,
        decomposition: DecompositionMethod = DecompositionMethod.ADDITIVE,
        credit_strategy: CreditAssignmentStrategy = CreditAssignmentStrategy.DIFFERENCE_REWARDS,
        learning_rate: float = 0.01,
        discount: float = 0.99,
    ) -> None:
        self._agent_id = agent_id
        self._bus = event_bus
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._decomposition = decomposition
        self._credit_strategy = credit_strategy
        self._lr = learning_rate
        self._gamma = discount

        # Simple tabular Q-values
        self._q_table: dict[str, np.ndarray] = {}
        self._peer_values: dict[str, float] = {}
        self._credit_history: list[float] = []
        self._lock = threading.RLock()

        # Subscribe to value broadcasts
        self._bus.register_callback(CH_VALUE_SHARE, self._on_value_share)

    def compute_local_value(
        self, state: np.ndarray, action: int
    ) -> float:
        """Compute this agent's Q-value for a state-action pair."""
        state_key = self._state_key(state)
        with self._lock:
            if state_key not in self._q_table:
                self._q_table[state_key] = np.zeros(self._action_dim)
            return float(self._q_table[state_key][action])

    def compute_joint_value(
        self, individual_values: dict[str, float]
    ) -> float:
        """Compute joint value from individual agent values."""
        values = list(individual_values.values())
        if self._decomposition == DecompositionMethod.ADDITIVE:
            return sum(values)
        if self._decomposition == DecompositionMethod.WEIGHTED:
            n = len(values)
            weights = [1.0 / n] * n
            return sum(w * v for w, v in zip(weights, values))
        # MONOTONIC fallback to additive
        return sum(values)

    def assign_credit(
        self,
        global_reward: float,
        state: np.ndarray,
        action: int,
        individual_values: dict[str, float] | None = None,
    ) -> float:
        """Assign credit to this agent from global reward."""
        if self._credit_strategy == CreditAssignmentStrategy.DIFFERENCE_REWARDS:
            credit = self._difference_reward(global_reward, individual_values)
        elif self._credit_strategy == CreditAssignmentStrategy.COUNTERFACTUAL:
            credit = self._counterfactual_credit(global_reward, state, action, individual_values)
        else:
            # Equal split
            n_agents = max(1, len(individual_values or {}))
            credit = global_reward / n_agents

        self._credit_history.append(credit)
        return credit

    def update_value_function(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> float:
        """Update Q-value from experience. Returns TD error."""
        state_key = self._state_key(state)
        next_key = self._state_key(next_state)

        with self._lock:
            if state_key not in self._q_table:
                self._q_table[state_key] = np.zeros(self._action_dim)
            if next_key not in self._q_table:
                self._q_table[next_key] = np.zeros(self._action_dim)

            current_q = self._q_table[state_key][action]
            next_max = 0.0 if done else float(np.max(self._q_table[next_key]))
            target = reward + self._gamma * next_max
            td_error = target - current_q

            self._q_table[state_key][action] += self._lr * td_error
            return td_error

    def share_value_information(
        self, state: np.ndarray, action: int, value: float
    ) -> None:
        """Broadcast local value to peers."""
        self._bus.publish(CH_VALUE_SHARE, {
            "agent_id": self._agent_id,
            "state_hash": self._state_key(state),
            "action": action,
            "value": value,
            "timestamp": time.time(),
        })

    def get_peer_values(self) -> dict[str, float]:
        """Get most recent values from peers."""
        with self._lock:
            return dict(self._peer_values)

    def get_credit_statistics(self) -> dict[str, Any]:
        return {
            "agent_id": self._agent_id,
            "q_table_size": len(self._q_table),
            "credit_history_size": len(self._credit_history),
            "mean_credit": float(np.mean(self._credit_history)) if self._credit_history else 0.0,
            "decomposition": self._decomposition.value,
            "credit_strategy": self._credit_strategy.value,
        }

    def _difference_reward(
        self, global_reward: float, individual_values: dict[str, float] | None
    ) -> float:
        """Credit = global_reward - reward_without_this_agent."""
        if not individual_values or self._agent_id not in individual_values:
            return global_reward
        total = sum(individual_values.values())
        without_me = total - individual_values[self._agent_id]
        return global_reward - without_me

    def _counterfactual_credit(
        self,
        global_reward: float,
        state: np.ndarray,
        action: int,
        individual_values: dict[str, float] | None,
    ) -> float:
        """Credit = actual_value - average_over_all_actions."""
        state_key = self._state_key(state)
        with self._lock:
            if state_key not in self._q_table:
                return global_reward / max(1, len(individual_values or {}))
            mean_value = float(np.mean(self._q_table[state_key]))
            actual = self._q_table[state_key][action]
            return float(actual - mean_value)

    def _on_value_share(self, channel: str, data: Any) -> None:
        import json
        if isinstance(data, str):
            data = json.loads(data)
        peer_id = data.get("agent_id", "")
        if peer_id and peer_id != self._agent_id:
            with self._lock:
                self._peer_values[peer_id] = data.get("value", 0.0)

    @staticmethod
    def _state_key(state: np.ndarray) -> str:
        """Hash state for tabular lookup."""
        return str(np.round(state, decimals=2).tobytes()[:32].hex())

    # -- persistence --

    def serialize(self, persist_dir: "Path") -> dict:
        """Save VDN state to disk. Returns JSON-safe metadata."""
        import pickle
        from pathlib import Path

        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "q_table": dict(self._q_table),
            "peer_values": dict(self._peer_values),
            "credit_history": list(self._credit_history),
        }
        with open(persist_dir / "vdn.pkl", "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

        return {
            "q_table_size": len(self._q_table),
            "credit_history_size": len(self._credit_history),
            "agent_id": self._agent_id,
        }

    def restore(self, persist_dir: "Path", metadata: dict) -> None:
        """Restore VDN state from disk."""
        import pickle
        from pathlib import Path

        persist_dir = Path(persist_dir)
        path = persist_dir / "vdn.pkl"
        if not path.exists():
            logger.warning("No VDN file at %s, starting fresh", path)
            return

        try:
            with open(path, "rb") as f:
                state = pickle.load(f)

            with self._lock:
                self._q_table = state["q_table"]
                self._peer_values = state["peer_values"]
                self._credit_history = state["credit_history"]
            logger.info(
                "Restored VDN (q_states=%d, credits=%d)",
                len(self._q_table), len(self._credit_history),
            )
        except Exception:
            logger.warning("Failed to restore VDN", exc_info=True)

    def __repr__(self) -> str:
        return f"VDN(agent={self._agent_id}, q_states={len(self._q_table)})"
