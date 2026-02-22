"""World Model - Imagination engine for predicting state transitions.

Predicts next states, rewards, and observations from (state, action) pairs.
Supports ensemble uncertainty, stochastic transitions, and distributional rewards.
Foundation for validated imagination, dream planning, and causal reasoning.

Biological analogy: Mental simulation / imagination.
Based on: Ha & Schmidhuber (2018) "World Models", Hafner et al. (2023) "DreamerV3".
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WorldModelConfig:
    """Configuration for the world model."""

    state_dim: int = 10
    action_dim: int = 5
    hidden_dim: int = 128
    use_ensemble: bool = False
    num_ensemble: int = 5
    stochastic: bool = True
    learning_rate: float = 3e-4
    device: str = "cpu"


@dataclass
class Prediction:
    """Result of a world model prediction."""

    next_state: np.ndarray
    reward: float
    uncertainty: float = 0.0
    observation: np.ndarray | None = None
    ensemble_disagreement: float = 0.0


class WorldModel:
    """Predicts state transitions, rewards, and observations.

    Core imagination engine used by:
    - CollectiveDreamPlanner (multi-agent dream planning)
    - ValidatedImagination (accuracy tracking)
    - DreamEnvironment (imagined policy training)
    - DecisionRouter (prefrontal deliberation)

    Supports ensemble mode for uncertainty quantification.
    """

    def __init__(
        self,
        config: WorldModelConfig | None = None,
        transition_fn: Callable[..., np.ndarray] | None = None,
        reward_fn: Callable[..., float] | None = None,
        event_bus: Any | None = None,
    ) -> None:
        self._config = config or WorldModelConfig()
        self._transition_fn = transition_fn or self._default_transition
        self._reward_fn = reward_fn or self._default_reward
        self._bus = event_bus

        # Ensemble parameters (simple learned mappings)
        self._ensemble_weights: list[np.ndarray] = []
        if self._config.use_ensemble:
            for _ in range(self._config.num_ensemble):
                w = np.random.randn(
                    self._config.state_dim + self._config.action_dim,
                    self._config.state_dim,
                ).astype(np.float32) * 0.1
                self._ensemble_weights.append(w)

        # Training state
        self._training_steps = 0
        self._training_losses: list[float] = []
        self._lock = threading.RLock()

    def step(
        self,
        state: np.ndarray,
        action: np.ndarray | int,
        deterministic: bool = False,
    ) -> Prediction:
        """Single-step prediction: (state, action) -> (next_state, reward)."""
        action_vec = self._encode_action(action)

        if self._config.use_ensemble and self._ensemble_weights:
            return self._ensemble_step(state, action_vec, deterministic)

        next_state = self._transition_fn(state, action_vec, deterministic)
        reward = self._reward_fn(state, action_vec)

        uncertainty = 0.0
        if self._config.stochastic and not deterministic:
            noise = np.random.randn(*state.shape) * 0.01
            next_state = next_state + noise
            uncertainty = float(np.linalg.norm(noise))

        return Prediction(
            next_state=next_state,
            reward=reward,
            uncertainty=uncertainty,
        )

    def rollout(
        self,
        initial_state: np.ndarray,
        policy: Any,
        horizon: int,
        deterministic: bool = False,
    ) -> dict[str, Any]:
        """Multi-step trajectory simulation using a policy."""
        states = [initial_state]
        actions = []
        rewards = []
        uncertainties = []

        current_state = initial_state
        for _ in range(horizon):
            action = self._get_action(policy, current_state)
            pred = self.step(current_state, action, deterministic)

            actions.append(action)
            rewards.append(pred.reward)
            uncertainties.append(pred.uncertainty)
            states.append(pred.next_state)
            current_state = pred.next_state

        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "uncertainties": uncertainties,
            "total_reward": sum(rewards),
            "avg_uncertainty": float(np.mean(uncertainties)) if uncertainties else 0.0,
        }

    def get_uncertainty(
        self, state: np.ndarray, action: np.ndarray | int
    ) -> dict[str, float]:
        """Estimate prediction uncertainty via ensemble disagreement."""
        if not self._config.use_ensemble or not self._ensemble_weights:
            return {"transition_disagreement": 0.0, "total_disagreement": 0.0}

        action_vec = self._encode_action(action)
        inp = np.concatenate([state.flatten(), action_vec.flatten()])

        predictions = []
        for weights in self._ensemble_weights:
            pred = inp @ weights
            predictions.append(pred)

        preds_array = np.array(predictions)
        disagreement = float(np.mean(np.std(preds_array, axis=0)))

        return {
            "transition_disagreement": disagreement,
            "total_disagreement": disagreement,
        }

    def train_step(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        rewards: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> float:
        """Single training step on a batch. Returns loss."""
        if weights is None:
            weights = np.ones(len(states))

        batch_size = len(states)
        total_loss = 0.0

        if self._config.use_ensemble and self._ensemble_weights:
            for i, ew in enumerate(self._ensemble_weights):
                for j in range(batch_size):
                    action_vec = self._encode_action(actions[j])
                    inp = np.concatenate([states[j].flatten(), action_vec.flatten()])
                    predicted = inp @ ew
                    error = predicted - next_states[j].flatten()[: predicted.shape[0]]
                    grad = np.outer(inp, error * weights[j])
                    self._ensemble_weights[i] -= self._config.learning_rate * grad
                    total_loss += float(np.mean(error**2) * weights[j])

            total_loss /= max(batch_size * len(self._ensemble_weights), 1)
        else:
            for j in range(batch_size):
                action_vec = self._encode_action(actions[j])
                pred = self._transition_fn(states[j], action_vec, True)
                error = pred - next_states[j]
                total_loss += float(np.mean(error**2) * weights[j])
            total_loss /= max(batch_size, 1)

        with self._lock:
            self._training_steps += 1
            self._training_losses.append(total_loss)

        if self._bus is not None:
            try:
                self._bus.publish("cognition.model_trained", {
                    "training_step": self._training_steps,
                    "loss": total_loss,
                })
            except Exception:
                logger.debug("EventBus publish failed for model_trained")

        return total_loss

    def predict(self, state: np.ndarray, action: Any) -> np.ndarray:
        """Convenience: predict next state only."""
        pred = self.step(state, action, deterministic=True)
        if self._bus is not None:
            try:
                self._bus.publish("cognition.prediction_made", {
                    "uncertainty": pred.uncertainty,
                    "ensemble_disagreement": pred.ensemble_disagreement,
                    "reward": pred.reward,
                })
            except Exception:
                logger.debug("EventBus publish failed for prediction_made")
        return pred.next_state

    def predict_reward(self, state: np.ndarray, action: Any) -> float:
        """Convenience: predict reward only."""
        pred = self.step(state, action, deterministic=True)
        return pred.reward

    def get_training_stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "training_steps": self._training_steps,
                "recent_loss": (
                    float(np.mean(self._training_losses[-100:]))
                    if self._training_losses
                    else 0.0
                ),
                "ensemble_size": len(self._ensemble_weights),
                "stochastic": self._config.stochastic,
            }

    # --- Internal ---

    def _ensemble_step(
        self, state: np.ndarray, action_vec: np.ndarray, deterministic: bool
    ) -> Prediction:
        """Predict using ensemble of models."""
        inp = np.concatenate([state.flatten(), action_vec.flatten()])
        predictions = []
        for weights in self._ensemble_weights:
            pred = inp @ weights
            predictions.append(pred)

        preds_array = np.array(predictions)
        mean_pred = np.mean(preds_array, axis=0)
        std_pred = np.std(preds_array, axis=0)
        disagreement = float(np.mean(std_pred))

        if not deterministic and self._config.stochastic:
            noise = np.random.randn(*mean_pred.shape) * np.minimum(std_pred, 0.1)
            mean_pred = mean_pred + noise

        reward = self._reward_fn(state, action_vec)

        return Prediction(
            next_state=mean_pred,
            reward=reward,
            uncertainty=disagreement,
            ensemble_disagreement=disagreement,
        )

    def _encode_action(self, action: np.ndarray | int) -> np.ndarray:
        """Convert action to vector representation."""
        if isinstance(action, (int, np.integer)):
            vec = np.zeros(self._config.action_dim, dtype=np.float32)
            if 0 <= action < self._config.action_dim:
                vec[action] = 1.0
            return vec
        if isinstance(action, dict):
            return np.zeros(self._config.action_dim, dtype=np.float32)
        return np.asarray(action, dtype=np.float32).flatten()

    @staticmethod
    def _get_action(policy: Any, state: np.ndarray) -> Any:
        """Extract action from various policy types."""
        if hasattr(policy, "select_action"):
            return policy.select_action(state)
        if callable(policy):
            return policy(state)
        return np.random.randint(0, 3)

    @staticmethod
    def _default_transition(
        state: np.ndarray, action: np.ndarray, deterministic: bool
    ) -> np.ndarray:
        """Default: small perturbation based on action."""
        delta = action[: len(state)] * 0.1 if len(action) >= len(state) else action[:1] * 0.1
        result = state + np.resize(delta, state.shape)
        if not deterministic:
            result += np.random.randn(*state.shape) * 0.01
        return result

    @staticmethod
    def _default_reward(state: np.ndarray, action: np.ndarray) -> float:
        """Default: small positive reward."""
        return float(0.5 + np.random.randn() * 0.1)

    # -- persistence --

    def serialize(self, persist_dir: "Path") -> dict:
        """Save world model state to disk. Returns JSON-safe metadata."""
        import pickle
        from pathlib import Path

        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "ensemble_weights": self._ensemble_weights,
            "training_steps": self._training_steps,
            "training_losses": self._training_losses,
        }
        with open(persist_dir / "world_model.pkl", "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

        return {
            "training_steps": self._training_steps,
            "ensemble_size": len(self._ensemble_weights),
            "recent_loss": (
                float(np.mean(self._training_losses[-100:]))
                if self._training_losses else 0.0
            ),
        }

    def restore(self, persist_dir: "Path", metadata: dict) -> None:
        """Restore world model state from disk."""
        import pickle
        from pathlib import Path

        persist_dir = Path(persist_dir)
        path = persist_dir / "world_model.pkl"
        if not path.exists():
            logger.warning("No world model file at %s, starting fresh", path)
            return

        try:
            with open(path, "rb") as f:
                state = pickle.load(f)

            self._ensemble_weights = state["ensemble_weights"]
            self._training_steps = state["training_steps"]
            self._training_losses = state["training_losses"]
            logger.info(
                "Restored world model (steps=%d, ensemble=%d)",
                self._training_steps, len(self._ensemble_weights),
            )
        except Exception:
            logger.warning("Failed to restore world model", exc_info=True)

    def __repr__(self) -> str:
        return (
            f"WorldModel(state_dim={self._config.state_dim}, "
            f"ensemble={self._config.use_ensemble}, "
            f"steps={self._training_steps})"
        )
