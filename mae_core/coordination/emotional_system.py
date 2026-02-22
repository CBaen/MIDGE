"""Emotional system - somatic marker hypothesis for Mae.

Biological basis: Antonio Damasio's somatic marker hypothesis (1994).
Emotions are not separate from rational thought - they ARE bodily signals
(hormone combinations + recent experiences) that bias decision-making.
Without emotions, organisms make catastrophically poor decisions because
they cannot rapidly evaluate options against past experience.

Mae's 6 emotional states emerge from hormone combinations:

| Emotion   | Valence | Arousal | Trigger                           |
|-----------|---------|---------|-----------------------------------|
| FEAR      | -0.8    | 0.9     | High cortisol + adrenaline        |
| CURIOSITY | 0.6     | 0.7     | High dopamine, low cortisol       |
| JOY       | 0.9     | 0.6     | High dopamine + serotonin         |
| ANGER     | -0.7    | 0.8     | High adrenaline + cortisol, low serotonin |
| CALM      | 0.3     | 0.2     | High serotonin + oxytocin         |
| SURPRISE  | 0.1     | 0.9     | Adrenaline spike + dopamine       |

Integration:
- Reads hormone levels from EndocrineSystem via EventBus
- Publishes emotion updates for DecisionRouter and other systems
- Emotion inertia prevents rapid oscillation (biological realism)

Law compliance:
- Law 8 property (2): differentiation - rich internal emotional states
- Law 8 property (4): recurrence - emotion-hormone feedback loop
- Law 6: autopoietic - emotions modulate hormones which modulate emotions
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from mae_core.backbone.event_bus import EventBus

logger = logging.getLogger(__name__)

# EventBus channel
CH_EMOTION_UPDATE = "coordination.emotion_update"


@dataclass(frozen=True)
class EmotionalState:
    """A single emotional state with valence and arousal dimensions.

    Valence: -1.0 (negative) to 1.0 (positive)
    Arousal: 0.0 (calm) to 1.0 (activated)
    """

    name: str
    valence: float
    arousal: float
    dominant_hormone: str
    description: str


# The 6 canonical emotional states
FEAR = EmotionalState(
    name="FEAR",
    valence=-0.8,
    arousal=0.9,
    dominant_hormone="cortisol",
    description="Threat detected - mobilize defenses, avoid risk",
)
CURIOSITY = EmotionalState(
    name="CURIOSITY",
    valence=0.6,
    arousal=0.7,
    dominant_hormone="dopamine",
    description="Novelty detected - explore, investigate, learn",
)
JOY = EmotionalState(
    name="JOY",
    valence=0.9,
    arousal=0.6,
    dominant_hormone="dopamine",
    description="Reward achieved - reinforce behavior, share with peers",
)
ANGER = EmotionalState(
    name="ANGER",
    valence=-0.7,
    arousal=0.8,
    dominant_hormone="adrenaline",
    description="Frustration or threat - confront, assert boundaries",
)
CALM = EmotionalState(
    name="CALM",
    valence=0.3,
    arousal=0.2,
    dominant_hormone="serotonin",
    description="Stable state - maintain, cooperate, consolidate",
)
SURPRISE = EmotionalState(
    name="SURPRISE",
    valence=0.1,
    arousal=0.9,
    dominant_hormone="adrenaline",
    description="Unexpected event - orient attention, reassess",
)

ALL_EMOTIONS = [FEAR, CURIOSITY, JOY, ANGER, CALM, SURPRISE]

# Emotion profiles: maps emotion -> list of (hormone, operator, threshold)
# Each condition is a tuple: (hormone_name, ">" or "<", threshold_value)
EMOTION_PROFILES: dict[str, list[tuple[str, str, float]]] = {
    "FEAR": [
        ("cortisol", ">", 0.6),
        ("adrenaline", ">", 0.4),
    ],
    "CURIOSITY": [
        ("dopamine", ">", 0.5),
        ("cortisol", "<", 0.3),
    ],
    "JOY": [
        ("dopamine", ">", 0.4),
        ("serotonin", ">", 0.4),
        ("cortisol", "<", 0.3),
    ],
    "ANGER": [
        ("adrenaline", ">", 0.6),
        ("cortisol", ">", 0.4),
        ("serotonin", "<", 0.3),
    ],
    "CALM": [
        ("serotonin", ">", 0.5),
        ("oxytocin", ">", 0.3),
        ("adrenaline", "<", 0.3),
    ],
    "SURPRISE": [
        ("adrenaline", ">", 0.5),
        ("dopamine", ">", 0.3),
    ],
}

_EMOTION_BY_NAME: dict[str, EmotionalState] = {e.name: e for e in ALL_EMOTIONS}


class EmotionalSystem:
    """Mae's emotional system - somatic markers for decision biasing.

    Reads hormone levels from the endocrine system, evaluates them
    against emotion profiles, and publishes the current dominant
    emotion. Emotion inertia prevents unrealistic rapid flipping.
    """

    CH_EMOTION_UPDATE = CH_EMOTION_UPDATE

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        endocrine_system: Any = None,
        emotion_inertia: float = 0.7,
    ) -> None:
        self.event_bus = event_bus
        self._endocrine_system = endocrine_system
        self._emotion_inertia = max(0.0, min(1.0, emotion_inertia))

        # Current state
        self._current_emotion: EmotionalState = CALM
        self._current_confidence: float = 0.5
        self._emotion_history: list[tuple[int, str]] = []
        self._max_history = 100

        # Hormone snapshot (updated via EventBus or direct polling)
        # Defaults tuned so CALM is the baseline emotion (serotonin and
        # oxytocin above CALM thresholds, adrenaline below).
        self._hormone_levels: dict[str, float] = {
            "adrenaline": 0.1,
            "cortisol": 0.2,
            "dopamine": 0.2,
            "serotonin": 0.55,
            "oxytocin": 0.35,
        }

        # Scores from last evaluation (for debugging / statistics)
        self._last_scores: dict[str, float] = {}
        self._blended_scores: dict[str, float] = {}

        # Fear reinforcement from threat events
        self._fear_reinforcement: float = 0.0
        self._fear_decay: float = 0.1

        # Surprise boost from pattern consolidation (novelty)
        self._surprise_boost: float = 0.0
        self._surprise_decay: float = 0.15

        self._step_count = 0

        # Subscribe to EventBus channels
        if self.event_bus is not None:
            self.event_bus.register_callback(
                "endocrine.state_update", self._on_hormone_update
            )
            self.event_bus.register_callback(
                "defense.threat_detected", self._on_threat_detected
            )
            self.event_bus.register_callback(
                "pattern.consolidation", self._on_pattern_consolidation
            )

        logger.info("EmotionalSystem initialized (inertia=%.2f)", self._emotion_inertia)

    # =========================================================================
    # EventBus Handlers
    # =========================================================================

    def _on_hormone_update(self, channel: str, message: Any) -> None:
        """Update hormone snapshot from endocrine state broadcast."""
        data = self._parse_message(message)
        if data is None:
            return
        # The endocrine system publishes {hormone_name: level} dict
        for hormone in self._hormone_levels:
            if hormone in data:
                self._hormone_levels[hormone] = float(data[hormone])

    def _on_threat_detected(self, channel: str, message: Any) -> None:
        """Reinforce fear when threats are detected."""
        self._fear_reinforcement = min(1.0, self._fear_reinforcement + 0.3)

    def _on_pattern_consolidation(self, channel: str, message: Any) -> None:
        """Boost surprise when novel patterns are consolidated."""
        self._surprise_boost = min(1.0, self._surprise_boost + 0.25)

    # =========================================================================
    # Core Logic
    # =========================================================================

    def _evaluate_profile(self, emotion_name: str) -> float:
        """Score how well current hormones match an emotion profile.

        Returns a score between 0.0 and 1.0. Each condition contributes
        equally. A condition is either fully met (1.0) or partially met
        (proportional to how close the value is to the threshold).
        """
        conditions = EMOTION_PROFILES.get(emotion_name, [])
        if not conditions:
            return 0.0

        total_score = 0.0
        for hormone, op, threshold in conditions:
            level = self._hormone_levels.get(hormone, 0.0)
            if op == ">":
                if level > threshold:
                    total_score += 1.0
                else:
                    # Partial credit: how close are we?
                    total_score += max(0.0, level / threshold) * 0.5
            elif op == "<":
                if level < threshold:
                    total_score += 1.0
                else:
                    # Partial credit: inverse proximity
                    total_score += max(0.0, (1.0 - level) / (1.0 - threshold)) * 0.5

        # Normalize to 0-1
        return total_score / len(conditions)

    def _apply_reinforcements(self, scores: dict[str, float]) -> dict[str, float]:
        """Apply external reinforcement signals to emotion scores."""
        adjusted = dict(scores)
        # Fear reinforcement from threat events
        if self._fear_reinforcement > 0.0:
            adjusted["FEAR"] = min(1.0, adjusted.get("FEAR", 0.0) + self._fear_reinforcement * 0.4)
        # Surprise boost from novelty
        if self._surprise_boost > 0.0:
            adjusted["SURPRISE"] = min(1.0, adjusted.get("SURPRISE", 0.0) + self._surprise_boost * 0.4)
        return adjusted

    def step(self, current_step: int = 0) -> None:
        """Evaluate emotional state from current hormones.

        1. Score all emotion profiles against current hormone levels
        2. Apply external reinforcements (threat, novelty)
        3. Blend with previous scores via inertia
        4. Pick the dominant emotion
        5. Publish update
        """
        self._step_count = current_step if current_step > 0 else self._step_count + 1

        # If we have a direct reference to the endocrine system, poll it
        if self._endocrine_system is not None:
            if hasattr(self._endocrine_system, "get_all_levels"):
                levels = self._endocrine_system.get_all_levels()
                for hormone in self._hormone_levels:
                    if hormone in levels:
                        self._hormone_levels[hormone] = float(levels[hormone])

        # 1. Score all profiles
        raw_scores: dict[str, float] = {}
        for emotion_name in EMOTION_PROFILES:
            raw_scores[emotion_name] = self._evaluate_profile(emotion_name)
        self._last_scores = dict(raw_scores)

        # 2. Apply reinforcements
        adjusted_scores = self._apply_reinforcements(raw_scores)

        # 3. Blend with previous scores (inertia)
        for emotion_name in adjusted_scores:
            prev = self._blended_scores.get(emotion_name, 0.0)
            new = adjusted_scores[emotion_name]
            self._blended_scores[emotion_name] = (
                self._emotion_inertia * prev + (1.0 - self._emotion_inertia) * new
            )

        # 4. Pick dominant emotion
        best_name = max(self._blended_scores, key=self._blended_scores.get)
        best_score = self._blended_scores[best_name]

        # If no emotion scores above a minimum threshold, default to CALM
        if best_score < 0.15:
            best_name = "CALM"
            best_score = 0.5

        self._current_emotion = _EMOTION_BY_NAME[best_name]
        self._current_confidence = min(1.0, best_score)

        # 5. Record history
        self._emotion_history.append((self._step_count, best_name))
        if len(self._emotion_history) > self._max_history:
            self._emotion_history = self._emotion_history[-self._max_history:]

        # 6. Decay reinforcements
        self._fear_reinforcement = max(0.0, self._fear_reinforcement - self._fear_decay)
        self._surprise_boost = max(0.0, self._surprise_boost - self._surprise_decay)

        # 7. Publish
        if self.event_bus is not None:
            self.event_bus.publish(
                CH_EMOTION_UPDATE,
                {
                    "emotion_name": self._current_emotion.name,
                    "valence": self._current_emotion.valence,
                    "arousal": self._current_emotion.arousal,
                    "confidence": self._current_confidence,
                    "step": self._step_count,
                },
            )

    # =========================================================================
    # Public Query API
    # =========================================================================

    def get_current_emotion(self) -> EmotionalState:
        """Return the current dominant emotional state."""
        return self._current_emotion

    def get_emotional_valence(self) -> float:
        """Return current emotional valence for decision biasing.

        -1.0 = strongly negative, 0.0 = neutral, 1.0 = strongly positive.
        Weighted by confidence so uncertain emotions have less bias.
        """
        return self._current_emotion.valence * self._current_confidence

    def get_emotional_arousal(self) -> float:
        """Return current arousal level (0.0 = calm, 1.0 = activated)."""
        return self._current_emotion.arousal * self._current_confidence

    # =========================================================================
    # Observability
    # =========================================================================

    def get_statistics(self) -> dict[str, Any]:
        """Return emotional system statistics for monitoring."""
        recent_emotions: dict[str, int] = {}
        for _, emotion_name in self._emotion_history[-20:]:
            recent_emotions[emotion_name] = recent_emotions.get(emotion_name, 0) + 1

        return {
            "current_emotion": self._current_emotion.name,
            "valence": self._current_emotion.valence,
            "arousal": self._current_emotion.arousal,
            "confidence": self._current_confidence,
            "effective_valence": self.get_emotional_valence(),
            "hormone_levels": dict(self._hormone_levels),
            "raw_scores": dict(self._last_scores),
            "blended_scores": dict(self._blended_scores),
            "fear_reinforcement": self._fear_reinforcement,
            "surprise_boost": self._surprise_boost,
            "history_length": len(self._emotion_history),
            "recent_distribution": recent_emotions,
            "step_count": self._step_count,
        }

    # =========================================================================
    # Persistence (Tier 2)
    # =========================================================================

    def serialize(self) -> dict:
        """Serialize emotional state for persistence."""
        return {
            "current_emotion": self._current_emotion.name,
            "current_confidence": self._current_confidence,
            "emotion_history": list(self._emotion_history),
            "hormone_levels": dict(self._hormone_levels),
            "blended_scores": dict(self._blended_scores),
            "fear_reinforcement": self._fear_reinforcement,
            "surprise_boost": self._surprise_boost,
            "emotion_inertia": self._emotion_inertia,
            "step_count": self._step_count,
        }

    def restore(self, data: dict) -> None:
        """Restore emotional state from serialized data."""
        if not isinstance(data, dict):
            return

        emotion_name = data.get("current_emotion", "CALM")
        self._current_emotion = _EMOTION_BY_NAME.get(emotion_name, CALM)
        self._current_confidence = float(data.get("current_confidence", 0.5))
        self._emotion_history = [
            (int(s), str(n)) for s, n in data.get("emotion_history", [])
        ]
        self._hormone_levels.update(data.get("hormone_levels", {}))
        self._blended_scores = dict(data.get("blended_scores", {}))
        self._fear_reinforcement = float(data.get("fear_reinforcement", 0.0))
        self._surprise_boost = float(data.get("surprise_boost", 0.0))
        self._emotion_inertia = float(data.get("emotion_inertia", self._emotion_inertia))
        self._step_count = int(data.get("step_count", 0))

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    @staticmethod
    def _parse_message(message: Any) -> Optional[dict]:
        """Parse an EventBus message (may be JSON string or dict)."""
        if isinstance(message, dict):
            return message
        if isinstance(message, str):
            try:
                parsed = json.loads(message)
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass
        return None
