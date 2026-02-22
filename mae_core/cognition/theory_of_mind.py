"""Theory of Mind - Modeling other agents' mental states.

Biological basis: Premack & Woodruff (1978) showed that primates attribute
mental states (beliefs, desires, intentions) to others. Goldman (2006)
proposes simulation theory: we model others by running a simulated copy
of our own decision machinery with their parameters.

Mae uses Theory of Mind to:
1. Track what each agent is likely thinking/feeling/intending
2. Predict other agents' next actions
3. Estimate cooperation probability
4. Enable strategic social behaviour

Connection points:
- EventBus publishes model updates on cognition.tom_update
- Agents feed observations via update_model()
- DecisionRouter can use predictions for social planning
- NociceptionSystem can incorporate social pain (rejection/betrayal)

Law compliance:
- Law 3 (Holon Protocol): Self-reference — modeling others requires a self-model
- Law 8 (Consciousness): Property 1 — integration (combining disparate
  observations into a coherent model of another mind)
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentModel:
    """Internal model of another agent's mental state."""

    agent_id: int
    estimated_emotion: str = "calm"
    estimated_goal: str = "explore"
    recent_actions: list[str] = field(default_factory=list)
    trust_level: float = 0.5
    last_updated: int = 0
    confidence: float = 0.3


# Emotion -> cooperation modifier mapping
_EMOTION_COOPERATION_MODIFIERS: dict[str, float] = {
    "calm": 0.0,
    "joy": 0.15,
    "happy": 0.15,
    "excited": 0.1,
    "curious": 0.05,
    "fear": -0.2,
    "anger": -0.15,
    "sad": -0.1,
    "frustrated": -0.1,
}

# Signal type -> inferred goal mapping
_SIGNAL_GOAL_MAP: dict[str, str] = {
    "communicate": "collaborate",
    "collaborate": "collaborate",
    "trade": "exchange",
    "help": "collaborate",
    "attack": "compete",
    "flee": "survive",
    "explore": "explore",
    "gather": "resource_acquisition",
    "defend": "protect",
}

_MAX_RECENT_ACTIONS = 10
_MAX_CONFIDENCE = 0.9
_CONFIDENCE_INCREMENT = 0.05
_CONFIDENCE_DECAY = 0.95
_MIN_CONFIDENCE_THRESHOLD = 0.1


class TheoryOfMind:
    """Models other agents' mental states from observed behaviour.

    Maintains a dictionary of AgentModels, one per known agent.
    Each model tracks the agent's estimated emotion, goal, recent
    actions, trust level, and the model's own confidence.

    Args:
        event_bus: Optional EventBus for publishing tom_update events.
    """

    CH_TOM_UPDATE = "cognition.tom_update"

    def __init__(self, event_bus: Any | None = None, **kwargs: Any) -> None:
        self._bus = event_bus
        self._agent_models: dict[int, AgentModel] = {}
        self._observation_count: int = 0
        self._current_step: int = 0

    # ------------------------------------------------------------------
    # Model Updates
    # ------------------------------------------------------------------

    def update_model(
        self,
        agent_id: int,
        action: str | None = None,
        signal_type: str | None = None,
        emotion: str | None = None,
    ) -> AgentModel:
        """Update the internal model of an agent based on observed behaviour.

        Each observation increases model confidence (capped at 0.9).
        Multiple observation types can be provided in a single call.

        Args:
            agent_id: The unique identifier of the observed agent.
            action: An action the agent performed (e.g. "explore", "attack").
            signal_type: A signal the agent emitted (used to infer goal).
            emotion: A detected or reported emotional state.

        Returns:
            The updated AgentModel.
        """
        model = self._agent_models.get(agent_id)
        if model is None:
            model = AgentModel(agent_id=agent_id)
            self._agent_models[agent_id] = model

        updated = False

        if action is not None:
            model.recent_actions.append(action)
            # Trim to max length (FIFO)
            if len(model.recent_actions) > _MAX_RECENT_ACTIONS:
                model.recent_actions = model.recent_actions[-_MAX_RECENT_ACTIONS:]
            updated = True

        if signal_type is not None:
            inferred_goal = _SIGNAL_GOAL_MAP.get(signal_type)
            if inferred_goal is not None:
                model.estimated_goal = inferred_goal
            updated = True

        if emotion is not None:
            model.estimated_emotion = emotion
            updated = True

        if updated:
            model.confidence = min(
                _MAX_CONFIDENCE,
                model.confidence + _CONFIDENCE_INCREMENT,
            )
            model.last_updated = self._current_step
            self._observation_count += 1

        return model

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------

    def predict_action(self, agent_id: int) -> str:
        """Predict the next action of an agent based on recent behaviour.

        Uses simple majority voting over recent_actions. If no model
        exists or no actions have been recorded, returns "unknown".
        """
        model = self._agent_models.get(agent_id)
        if model is None or not model.recent_actions:
            return "unknown"

        counts = Counter(model.recent_actions)
        most_common_action, _ = counts.most_common(1)[0]
        return most_common_action

    def predict_cooperation(self, agent_id: int) -> float:
        """Estimate the probability that an agent will cooperate.

        Based on trust_level and estimated_emotion. Emotions like
        fear reduce cooperation probability; joy increases it.

        Returns:
            A float in [0, 1] representing cooperation probability.
        """
        model = self._agent_models.get(agent_id)
        if model is None:
            return 0.5  # Neutral assumption for unknown agents

        base = model.trust_level
        modifier = _EMOTION_COOPERATION_MODIFIERS.get(
            model.estimated_emotion, 0.0
        )
        return max(0.0, min(1.0, base + modifier))

    def get_agent_model(self, agent_id: int) -> Optional[AgentModel]:
        """Retrieve the internal model for a specific agent, or None."""
        return self._agent_models.get(agent_id)

    # ------------------------------------------------------------------
    # Step (Autopoietic Maintenance)
    # ------------------------------------------------------------------

    def step(self, current_step: int = 0) -> None:
        """Advance one time step. Decays stale models and publishes updates.

        Models not updated recently have their confidence decayed by
        _CONFIDENCE_DECAY per step. Models whose confidence drops below
        _MIN_CONFIDENCE_THRESHOLD are removed entirely.

        Law 6 (Autopoietic Closure): The step method maintains the
        system's internal state, pruning stale models and publishing
        health signals.
        """
        self._current_step = current_step if current_step > 0 else self._current_step + 1

        # Decay confidence of stale models
        to_remove: list[int] = []
        for agent_id, model in self._agent_models.items():
            if model.last_updated < self._current_step:
                model.confidence *= _CONFIDENCE_DECAY
                if model.confidence < _MIN_CONFIDENCE_THRESHOLD:
                    to_remove.append(agent_id)

        for agent_id in to_remove:
            del self._agent_models[agent_id]

        # Publish update
        if self._bus is not None:
            avg_confidence = 0.0
            if self._agent_models:
                avg_confidence = sum(
                    m.confidence for m in self._agent_models.values()
                ) / len(self._agent_models)

            try:
                self._bus.publish(self.CH_TOM_UPDATE, {
                    "num_models": len(self._agent_models),
                    "avg_confidence": round(avg_confidence, 4),
                    "step": self._current_step,
                })
            except Exception:
                logger.debug("EventBus publish failed for tom_update")

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def get_statistics(self) -> dict[str, Any]:
        """Return current system statistics."""
        avg_conf = 0.0
        avg_trust = 0.0
        if self._agent_models:
            avg_conf = sum(m.confidence for m in self._agent_models.values()) / len(self._agent_models)
            avg_trust = sum(m.trust_level for m in self._agent_models.values()) / len(self._agent_models)

        return {
            "num_models": len(self._agent_models),
            "observation_count": self._observation_count,
            "avg_confidence": round(avg_conf, 4),
            "avg_trust": round(avg_trust, 4),
            "current_step": self._current_step,
        }

    # ------------------------------------------------------------------
    # Tier 2 Persistence
    # ------------------------------------------------------------------

    def serialize(self) -> dict[str, Any]:
        """Serialize state for Tier 2 persistence."""
        models_data = {}
        for agent_id, model in self._agent_models.items():
            models_data[str(agent_id)] = {
                "agent_id": model.agent_id,
                "estimated_emotion": model.estimated_emotion,
                "estimated_goal": model.estimated_goal,
                "recent_actions": list(model.recent_actions),
                "trust_level": model.trust_level,
                "last_updated": model.last_updated,
                "confidence": model.confidence,
            }

        return {
            "agent_models": models_data,
            "observation_count": self._observation_count,
            "current_step": self._current_step,
        }

    def restore(self, data: dict[str, Any]) -> None:
        """Restore state from serialized data."""
        self._observation_count = data.get("observation_count", 0)
        self._current_step = data.get("current_step", 0)
        self._agent_models.clear()

        for _key, model_data in data.get("agent_models", {}).items():
            model = AgentModel(
                agent_id=model_data["agent_id"],
                estimated_emotion=model_data.get("estimated_emotion", "calm"),
                estimated_goal=model_data.get("estimated_goal", "explore"),
                recent_actions=model_data.get("recent_actions", []),
                trust_level=model_data.get("trust_level", 0.5),
                last_updated=model_data.get("last_updated", 0),
                confidence=model_data.get("confidence", 0.3),
            )
            self._agent_models[model.agent_id] = model

    def __repr__(self) -> str:
        return (
            f"TheoryOfMind(models={len(self._agent_models)}, "
            f"observations={self._observation_count})"
        )
