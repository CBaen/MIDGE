"""OrganismState output API - query, feedback, serialize, restore.

Contains get_body_state, get_reflex_override, get_decision_context,
report_action_outcome, get_statistics, serialize, restore, _parse_message.
Extracted from organism_state.py to stay under 500-line limit.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Channel published by OrganismState itself
CH_ORGANISM_ACTION_OUTCOME = "organism.action_outcome"

_OUTCOME_WINDOW = 20        # Rolling window size for action outcomes
_VITALITY_ALPHA = 0.1       # Exponential moving average smoothing factor


class OrganismStateOutputsMixin:
    """Public query API, feedback, observability, and persistence for OrganismState."""

    # =====================================================================
    # Public Query API
    # =====================================================================

    def get_body_state(self) -> dict[str, Any]:
        """Return the unified body state aggregating all 18 systems."""
        return {
            # Metabolic
            "energy_level": self._energy_level,
            "energy_critical": self._energy_critical,
            "oxygen_level": self._oxygen_level,
            "toxin_load": self._toxin_load,
            "circulation_adequate": self._circulation_adequate,
            "digestion_active": self._digestion_active,
            # Emotional / Social
            "emotional_valence": self._emotional_valence,
            "emotional_arousal": self._emotional_arousal,
            "dominant_emotion": self._dominant_emotion,
            "metacognition_score": self._metacognition_score,
            "social_confidence": self._social_confidence,
            # Pain / Stability
            "pain_load": self._pain_load,
            "stability": self._stability,
            "temperature_zone": self._temperature_zone,
            # Maintenance
            "organism_age": self._organism_age,
            "waste_capacity_used": self._waste_capacity_used,
            "membrane_permeability": self._membrane_permeability,
            "microbiome_diversity": self._microbiome_diversity,
            # Regulation
            "homeostasis_deviation": self._homeostasis_deviation,
            "population_healthy": self._population_healthy,
        }

    def get_reflex_override(self) -> Optional[str]:
        """Check for emergency conditions requiring immediate reflexive action.

        Returns an action string if an override is active, or None if the
        normal decision cascade should proceed. Priority order (highest
        first): pain > stability > oxygen > energy > toxin.

        MIDGE: disabled — fictional physiology harms trading daemon.
        All five reflex conditions (pain, stability, oxygen, toxin, homeostasis)
        map to biological metaphors with no real-world meaning for a software
        trading daemon. When active, they silently override market intelligence
        and force agents to "rest" during the moments MIDGE most needs to act
        (high volatility, many convergence alerts, threat detection events).
        The dead code below is preserved for mae-core compatibility and future
        repurposing if real market equivalents are defined for these signals.
        """
        # MIDGE: disabled — fictional physiology harms trading daemon
        return None

        # --- Dead code below: mae-core compatibility, do not remove ---
        # Priority 1: Acute pain withdrawal
        if self._pain_load > 0.8:  # noqa: unreachable
            return "rest"

        # Priority 2: Vertigo stabilization
        if self._stability < 0.3:
            return "rest"

        # Priority 3: Hypoxia conservation
        if self._oxygen_level < 0.3:
            return "rest"

        # Priority 4: Starvation foraging — previously neutralized.
        if self._energy_critical:
            return None

        # Priority 5: Kidney stress
        if self._toxin_load > 4.0:
            return "rest"

        # Priority 6: Homeostasis deviation
        if self._homeostasis_deviation >= self._HOMEOSTASIS_URGENCY_THRESHOLD:
            return "rest"

        return None

    def get_decision_context(self) -> dict[str, Any]:
        """Return enriched context for the advisory decision routing."""
        # Threat level: composite of pain, instability, and toxins
        pain_factor = min(1.0, self._pain_load)
        instability_factor = max(0.0, 1.0 - self._stability)
        toxin_factor = min(1.0, self._toxin_load / 5.0)
        body_threat_level = (pain_factor + instability_factor + toxin_factor) / 3.0

        # Opportunity level: composite of energy, stability, and low pain
        energy_factor = self._energy_level
        stability_factor = self._stability
        low_pain_factor = max(0.0, 1.0 - self._pain_load)
        body_opportunity_level = (
            energy_factor + stability_factor + low_pain_factor
        ) / 3.0

        # Emotional bias
        if self._emotional_valence > 0.3:
            emotional_bias = "approach"
        elif self._emotional_valence < -0.3:
            emotional_bias = "avoid"
        else:
            emotional_bias = "neutral"

        return {
            "body_threat_level": round(body_threat_level, 4),
            "body_opportunity_level": round(body_opportunity_level, 4),
            "emotional_bias": emotional_bias,
            "metacognitive_confidence": round(self._metacognition_score, 4),
            "organism_vitality": round(self._vitality, 4),
        }

    # =====================================================================
    # Action Outcome Feedback
    # =====================================================================

    def report_action_outcome(
        self, action: str, reward: float, step: int,
    ) -> None:
        """Record the outcome of an agent action for vitality tracking."""
        self._recent_outcomes.append(reward)

        # Update vitality via EMA
        self._vitality = (
            (1.0 - _VITALITY_ALPHA) * self._vitality
            + _VITALITY_ALPHA * reward
        )

        # Publish outcome for other systems to consume
        if self.event_bus is not None:
            try:
                self.event_bus.publish(
                    CH_ORGANISM_ACTION_OUTCOME,
                    {
                        "action": action,
                        "reward": round(reward, 4),
                        "vitality": round(self._vitality, 4),
                        "step": step,
                    },
                )
            except Exception:
                logger.debug("EventBus publish failed for action_outcome")

    # =====================================================================
    # Observability
    # =====================================================================

    def get_statistics(self) -> dict[str, Any]:
        """Return organism state statistics for monitoring."""
        reflex = self.get_reflex_override()
        context = self.get_decision_context()

        return {
            "body_state": self.get_body_state(),
            "reflex_override": reflex,
            "decision_context": context,
            "vitality": round(self._vitality, 4),
            "recent_outcomes_count": len(self._recent_outcomes),
            "step_count": self._step_count,
        }

    # =====================================================================
    # Persistence (Tier 2)
    # =====================================================================

    def serialize(self) -> dict[str, Any]:
        """Serialize organism state for persistence."""
        return {
            # Metabolic
            "energy_level": self._energy_level,
            "energy_critical": self._energy_critical,
            "oxygen_level": self._oxygen_level,
            "toxin_load": self._toxin_load,
            "circulation_adequate": self._circulation_adequate,
            "digestion_active": self._digestion_active,
            # Emotional / Social
            "emotional_valence": self._emotional_valence,
            "emotional_arousal": self._emotional_arousal,
            "dominant_emotion": self._dominant_emotion,
            "metacognition_score": self._metacognition_score,
            "social_confidence": self._social_confidence,
            # Pain / Stability
            "pain_load": self._pain_load,
            "stability": self._stability,
            "temperature_zone": self._temperature_zone,
            # Maintenance
            "organism_age": self._organism_age,
            "waste_capacity_used": self._waste_capacity_used,
            "membrane_permeability": self._membrane_permeability,
            "microbiome_diversity": self._microbiome_diversity,
            # Regulation
            "homeostasis_deviation": self._homeostasis_deviation,
            "population_healthy": self._population_healthy,
            # Vitality
            "vitality": self._vitality,
            "recent_outcomes": list(self._recent_outcomes),
            # Step
            "step_count": self._step_count,
        }

    def restore(self, data: dict[str, Any]) -> None:
        """Restore organism state from serialized data."""
        if not isinstance(data, dict):
            return

        # Metabolic
        self._energy_level = float(data.get("energy_level", self._energy_level))
        self._energy_critical = bool(data.get("energy_critical", self._energy_critical))
        self._oxygen_level = float(data.get("oxygen_level", self._oxygen_level))
        self._toxin_load = float(data.get("toxin_load", self._toxin_load))
        self._circulation_adequate = bool(
            data.get("circulation_adequate", self._circulation_adequate)
        )
        self._digestion_active = bool(
            data.get("digestion_active", self._digestion_active)
        )

        # Emotional / Social
        self._emotional_valence = float(
            data.get("emotional_valence", self._emotional_valence)
        )
        self._emotional_arousal = float(
            data.get("emotional_arousal", self._emotional_arousal)
        )
        self._dominant_emotion = str(
            data.get("dominant_emotion", self._dominant_emotion)
        )
        self._metacognition_score = float(
            data.get("metacognition_score", self._metacognition_score)
        )
        self._social_confidence = float(
            data.get("social_confidence", self._social_confidence)
        )

        # Pain / Stability
        self._pain_load = float(data.get("pain_load", self._pain_load))
        self._stability = float(data.get("stability", self._stability))
        self._temperature_zone = str(
            data.get("temperature_zone", self._temperature_zone)
        )

        # Maintenance
        self._organism_age = float(data.get("organism_age", self._organism_age))
        self._waste_capacity_used = float(
            data.get("waste_capacity_used", self._waste_capacity_used)
        )
        self._membrane_permeability = float(
            data.get("membrane_permeability", self._membrane_permeability)
        )
        self._microbiome_diversity = float(
            data.get("microbiome_diversity", self._microbiome_diversity)
        )

        # Regulation
        self._homeostasis_deviation = float(
            data.get("homeostasis_deviation", self._homeostasis_deviation)
        )
        self._population_healthy = bool(
            data.get("population_healthy", self._population_healthy)
        )

        # Vitality
        self._vitality = float(data.get("vitality", self._vitality))
        raw_outcomes = data.get("recent_outcomes", [])
        self._recent_outcomes = deque(
            (float(v) for v in raw_outcomes), maxlen=_OUTCOME_WINDOW,
        )

        # Step
        self._step_count = int(data.get("step_count", 0))

    # =====================================================================
    # Internal Helpers
    # =====================================================================

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
