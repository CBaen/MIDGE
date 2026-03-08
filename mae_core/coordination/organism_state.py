"""Organism state aggregator - Mae's hypothalamus.

Biological basis: The hypothalamus integrates signals from every major body
system (temperature, hunger, thirst, fatigue, pain, hormones) into a unified
internal state representation. This allows the organism to answer "how am I
doing right now?" with a single coherent snapshot, rather than consulting
dozens of independent sensors.

Mae's OrganismState serves the same purpose:
- Subscribes to EventBus channels from all 18 biological systems
- Aggregates them into a single body_state dict
- Provides reflex overrides for emergency conditions (pain, starvation, etc.)
- Enriches the decision context for the advisory routing cascade
- Tracks action outcomes to build an overall vitality metric

Connection points:
- All 18 biological systems publish to EventBus; OrganismState listens
- Agents call get_body_state() for holistic awareness
- DecisionRouter calls get_reflex_override() before the advisory cascade
- DecisionRouter calls get_decision_context() to enrich routing context
- Agents call report_action_outcome() to close the feedback loop

Law compliance:
- Law 3 (Holon Protocol): know_self - the organism knows its own state
- Law 6 (Autopoietic Closure): signals produce the state that drives the
  actions that produce the signals
- Law 8 (Consciousness): Property 1 - integration (combining all signals
  into a unified representation); Property 3 - self-reference (the organism
  monitoring itself)
"""

from __future__ import annotations

import json
import logging
from collections import deque
from typing import Any, Optional

from mae_core.backbone.event_bus import EventBus

logger = logging.getLogger(__name__)

# Channel published by OrganismState itself
CH_ORGANISM_ACTION_OUTCOME = "organism.action_outcome"

# =========================================================================
# Channel constants from all 18 biological systems (verified from source)
# =========================================================================

# Coordination systems
_CH_EMOTION_UPDATE = "coordination.emotion_update"
_CH_HOMEOSTASIS_CORRECTION = "coordination.homeostasis_correction"
_CH_COOLING_NEEDED = "coordination.cooling_needed"
_CH_WARMING_NEEDED = "coordination.warming_needed"
_CH_TEMPERATURE_NORMAL = "coordination.temperature_normal"
_CH_DIGESTION_COMPLETE = "coordination.digestion_complete"
_CH_RESPIRATION_UPDATE = "coordination.respiration_update"
_CH_HYPOXIA = "coordination.hypoxia"
_CH_HYPERCAPNIA = "coordination.hypercapnia"
_CH_BALANCE_UPDATE = "coordination.balance_update"
_CH_VERTIGO = "coordination.vertigo"

# Communication systems
_CH_ACUTE_PAIN = "communication.acute_pain"
_CH_PAIN_OVERLOAD = "communication.pain_overload"
_CH_PAIN_UPDATE = "communication.pain_update"

# Emergent systems
_CH_LYMPH_STATUS = "emergent.lymph_status"
_CH_LYMPH_OVERFLOW = "emergent.lymph_overflow"
_CH_REJUVENATION = "emergent.rejuvenation_needed"
_CH_SENESCENT = "emergent.system_senescent"
_CH_AGE_UPDATE = "emergent.age_update"
_CH_PROPRIOCEPTION = "emergent.proprioception_update"
_CH_TOPOLOGY_CHANGE = "emergent.topology_changed"
_CH_MICROBIOME_STATUS = "emergent.microbiome_status"
_CH_DYSBIOSIS = "emergent.microbiome_imbalanced"

# Defense systems
_CH_MEMBRANE_STATUS = "defense.membrane_status"
_CH_QUARANTINE = "defense.quarantine_event"
_CH_RENAL_STATUS = "defense.renal_status"
_CH_KIDNEY_FAILURE = "defense.kidney_failure"
_CH_TOXIC_FILTERED = "defense.toxic_filtered"

# Memory systems
_CH_ENERGY_STATUS = "memory.energy_status"
_CH_STARVATION = "memory.starvation_alert"
_CH_RESERVES_FULL = "memory.reserves_full"

# Substrate systems
_CH_CIRCULATION = "substrate.circulation_update"
_CH_SUPPLY_LOW = "substrate.supply_low"

# Morphogenesis systems
_CH_SPAWN = "morphogenesis.spawn_request"
_CH_RETIRE = "morphogenesis.retire_request"
_CH_POPULATION = "morphogenesis.population_status"

# Cognition systems
_CH_TOM_UPDATE = "cognition.tom_update"
_CH_META_UPDATE = "cognition.metacognition_update"
_CH_META_ALERT = "cognition.metacognition_alert"

# =========================================================================
# Configuration constants
# =========================================================================

_OUTCOME_WINDOW = 20        # Rolling window size for action outcomes
_VITALITY_ALPHA = 0.1       # Exponential moving average smoothing factor


class OrganismState:
    """Mae's hypothalamus - unified body state aggregator.

    Subscribes to EventBus channels from all 18 biological systems and
    aggregates their signals into a single coherent body state. Provides
    reflex overrides for emergencies and enriched decision context.

    Args:
        event_bus: The EventBus instance for pub/sub integration.
    """

    CH_ORGANISM_ACTION_OUTCOME = CH_ORGANISM_ACTION_OUTCOME

    # Tunable threshold for homeostasis-driven rest reflex (Priority 6).
    # When _homeostasis_deviation exceeds this value, the organism rests to
    # allow internal systems to restabilize before continuing normal activity.
    _HOMEOSTASIS_URGENCY_THRESHOLD: float = 0.7

    def __init__(self, event_bus: Optional[EventBus] = None) -> None:
        self.event_bus = event_bus
        self._step_count: int = 0

        # ===================================================================
        # Body state — all metrics with healthy baselines
        # ===================================================================

        # Metabolic
        self._energy_level: float = 1.0
        self._energy_critical: bool = False
        self._oxygen_level: float = 1.0
        self._toxin_load: float = 0.0
        self._circulation_adequate: bool = True
        self._digestion_active: bool = False

        # Emotional / Social
        self._emotional_valence: float = 0.3       # CALM baseline
        self._emotional_arousal: float = 0.2        # CALM baseline
        self._dominant_emotion: str = "CALM"
        self._metacognition_score: float = 0.5
        self._social_confidence: float = 0.3

        # Pain / Stability
        self._pain_load: float = 0.0
        self._stability: float = 1.0
        self._temperature_zone: str = "optimal"

        # Maintenance
        self._organism_age: float = 0.0
        self._waste_capacity_used: float = 0.0
        self._membrane_permeability: float = 0.5
        self._microbiome_diversity: float = 1.0

        # Regulation
        self._homeostasis_deviation: float = 0.0
        self._population_healthy: bool = True

        # ===================================================================
        # Action outcome tracking
        # ===================================================================
        self._recent_outcomes: deque[float] = deque(maxlen=_OUTCOME_WINDOW)
        self._vitality: float = 0.5     # EMA of rewards

        # ===================================================================
        # Subscribe to all 18 biological system channels
        # ===================================================================
        if self.event_bus is not None:
            self._subscribe_all()

        logger.info("OrganismState initialized (hypothalamus aggregator)")

    # =====================================================================
    # Subscription Wiring
    # =====================================================================

    def _subscribe_all(self) -> None:
        """Register callbacks for all biological system channels."""
        bus = self.event_bus

        # -- Emotional System --
        bus.register_callback(_CH_EMOTION_UPDATE, self._on_emotion_update)

        # -- Homeostasis Regulator --
        bus.register_callback(
            _CH_HOMEOSTASIS_CORRECTION, self._on_homeostasis_correction,
        )

        # -- Thermoregulation System --
        bus.register_callback(_CH_COOLING_NEEDED, self._on_thermoregulation)
        bus.register_callback(_CH_WARMING_NEEDED, self._on_thermoregulation)
        bus.register_callback(_CH_TEMPERATURE_NORMAL, self._on_thermoregulation)

        # -- Digestive System --
        bus.register_callback(_CH_DIGESTION_COMPLETE, self._on_digestion_complete)

        # -- Respiratory System --
        bus.register_callback(_CH_RESPIRATION_UPDATE, self._on_respiration)
        bus.register_callback(_CH_HYPOXIA, self._on_respiration)
        bus.register_callback(_CH_HYPERCAPNIA, self._on_respiration)

        # -- Vestibular System --
        bus.register_callback(_CH_BALANCE_UPDATE, self._on_balance_update)
        bus.register_callback(_CH_VERTIGO, self._on_balance_update)

        # -- Nociception System --
        bus.register_callback(_CH_ACUTE_PAIN, self._on_pain_update)
        bus.register_callback(_CH_PAIN_OVERLOAD, self._on_pain_update)
        bus.register_callback(_CH_PAIN_UPDATE, self._on_pain_update)

        # -- Lymphatic System --
        bus.register_callback(_CH_LYMPH_STATUS, self._on_lymph_status)
        bus.register_callback(_CH_LYMPH_OVERFLOW, self._on_lymph_overflow)

        # -- Senescence Manager --
        bus.register_callback(_CH_AGE_UPDATE, self._on_age_update)
        bus.register_callback(_CH_REJUVENATION, self._on_age_update)
        bus.register_callback(_CH_SENESCENT, self._on_age_update)

        # -- Proprioception System --
        bus.register_callback(_CH_PROPRIOCEPTION, self._on_proprioception)
        bus.register_callback(_CH_TOPOLOGY_CHANGE, self._on_proprioception)

        # -- Microbiome --
        bus.register_callback(_CH_MICROBIOME_STATUS, self._on_microbiome_status)
        bus.register_callback(_CH_DYSBIOSIS, self._on_microbiome_dysbiosis)

        # -- Boundary Membrane --
        bus.register_callback(_CH_MEMBRANE_STATUS, self._on_membrane_status)
        bus.register_callback(_CH_QUARANTINE, self._on_membrane_status)

        # -- Renal Filter --
        bus.register_callback(_CH_RENAL_STATUS, self._on_renal_status)
        bus.register_callback(_CH_KIDNEY_FAILURE, self._on_kidney_failure)
        bus.register_callback(_CH_TOXIC_FILTERED, self._on_toxic_filtered)

        # -- Energy Reserve --
        bus.register_callback(_CH_ENERGY_STATUS, self._on_energy_status)
        bus.register_callback(_CH_STARVATION, self._on_starvation)
        bus.register_callback(_CH_RESERVES_FULL, self._on_energy_status)

        # -- Circulatory System --
        bus.register_callback(_CH_CIRCULATION, self._on_circulation)
        bus.register_callback(_CH_SUPPLY_LOW, self._on_supply_low)

        # -- Reproductive System --
        bus.register_callback(_CH_POPULATION, self._on_population_status)
        bus.register_callback(_CH_SPAWN, self._on_population_status)
        bus.register_callback(_CH_RETIRE, self._on_population_status)

        # -- Theory of Mind --
        bus.register_callback(_CH_TOM_UPDATE, self._on_tom_update)

        # -- Metacognition Monitor --
        bus.register_callback(_CH_META_UPDATE, self._on_meta_update)
        bus.register_callback(_CH_META_ALERT, self._on_meta_alert)

    # =====================================================================
    # EventBus Callback Handlers
    # =====================================================================

    def _on_emotion_update(self, channel: str, message: Any) -> None:
        """Handle emotion updates from EmotionalSystem."""
        data = self._parse_message(message)
        if data is None:
            return
        self._emotional_valence = float(data.get("valence", self._emotional_valence))
        self._emotional_arousal = float(data.get("arousal", self._emotional_arousal))
        self._dominant_emotion = str(data.get("emotion_name", self._dominant_emotion))

    def _on_homeostasis_correction(self, channel: str, message: Any) -> None:
        """Handle correction signals from HomeostasisRegulator.

        Each message is a single parameter correction. We track the maximum
        urgency across all corrections as the deviation score.
        """
        data = self._parse_message(message)
        if data is None:
            return
        urgency = float(data.get("urgency", 0.0))
        # Keep the worst deviation seen this step cycle
        self._homeostasis_deviation = max(self._homeostasis_deviation, urgency)

    def _on_thermoregulation(self, channel: str, message: Any) -> None:
        """Handle warming/cooling/normal signals from ThermoregulationSystem."""
        if channel == _CH_TEMPERATURE_NORMAL:
            self._temperature_zone = "optimal"
        elif channel == _CH_COOLING_NEEDED:
            self._temperature_zone = "hot"
        elif channel == _CH_WARMING_NEEDED:
            self._temperature_zone = "cold"

    def _on_digestion_complete(self, channel: str, message: Any) -> None:
        """Handle digestion complete signals from DigestiveSystem."""
        data = self._parse_message(message)
        if data is None:
            return
        # Active if there are items still in the queue
        queue_size = int(data.get("queue_size", 0))
        self._digestion_active = queue_size > 0

    def _on_respiration(self, channel: str, message: Any) -> None:
        """Handle respiration updates, hypoxia, and hypercapnia."""
        data = self._parse_message(message)
        if data is None:
            return
        self._oxygen_level = float(data.get("oxygen", self._oxygen_level))

    def _on_balance_update(self, channel: str, message: Any) -> None:
        """Handle balance/vertigo updates from VestibularSystem."""
        data = self._parse_message(message)
        if data is None:
            return
        self._stability = float(data.get("stability", self._stability))

    def _on_pain_update(self, channel: str, message: Any) -> None:
        """Handle pain signals from NociceptionSystem."""
        data = self._parse_message(message)
        if data is None:
            return
        self._pain_load = float(data.get("total_pain_load", self._pain_load))

    def _on_lymph_status(self, channel: str, message: Any) -> None:
        """Handle status updates from LymphaticSystem."""
        data = self._parse_message(message)
        if data is None:
            return
        self._waste_capacity_used = float(
            data.get("capacity_used", self._waste_capacity_used)
        )

    def _on_lymph_overflow(self, channel: str, message: Any) -> None:
        """Handle overflow alerts from LymphaticSystem."""
        data = self._parse_message(message)
        if data is None:
            return
        self._waste_capacity_used = float(
            data.get("capacity_used", self._waste_capacity_used)
        )

    def _on_age_update(self, channel: str, message: Any) -> None:
        """Handle age updates from SenescenceManager."""
        data = self._parse_message(message)
        if data is None:
            return
        self._organism_age = float(data.get("organism_age", self._organism_age))

    def _on_proprioception(self, channel: str, message: Any) -> None:
        """Handle proprioception updates and topology changes."""
        # Proprioception data is informational; topology changes
        # feed into stability awareness via the vestibular system.
        pass

    def _on_microbiome_status(self, channel: str, message: Any) -> None:
        """Handle status updates from Microbiome."""
        data = self._parse_message(message)
        if data is None:
            return
        self._microbiome_diversity = float(
            data.get("diversity", self._microbiome_diversity)
        )

    def _on_microbiome_dysbiosis(self, channel: str, message: Any) -> None:
        """Handle dysbiosis alerts from Microbiome."""
        data = self._parse_message(message)
        if data is None:
            return
        self._microbiome_diversity = float(
            data.get("diversity", self._microbiome_diversity)
        )

    def _on_membrane_status(self, channel: str, message: Any) -> None:
        """Handle status and quarantine events from BoundaryMembrane."""
        data = self._parse_message(message)
        if data is None:
            return
        if "permeability" in data:
            self._membrane_permeability = float(data["permeability"])

    def _on_renal_status(self, channel: str, message: Any) -> None:
        """Handle status updates from RenalFilter."""
        data = self._parse_message(message)
        if data is None:
            return
        self._toxin_load = float(data.get("toxin_load", self._toxin_load))

    def _on_kidney_failure(self, channel: str, message: Any) -> None:
        """Handle kidney failure alerts from RenalFilter."""
        data = self._parse_message(message)
        if data is None:
            return
        self._toxin_load = float(data.get("toxin_load", self._toxin_load))

    def _on_toxic_filtered(self, channel: str, message: Any) -> None:
        """Handle individual toxin filtered events from RenalFilter."""
        # Per-item events; toxin_load is tracked via renal_status
        pass

    def _on_energy_status(self, channel: str, message: Any) -> None:
        """Handle energy status from EnergyReserve."""
        data = self._parse_message(message)
        if data is None:
            return
        capacity_pct = data.get("capacity_pct")
        if capacity_pct is not None:
            self._energy_level = max(0.0, min(1.0, float(capacity_pct) / 100.0))
        self._energy_critical = bool(data.get("is_critical", self._energy_critical))

    def _on_starvation(self, channel: str, message: Any) -> None:
        """Handle starvation alerts from EnergyReserve."""
        self._energy_critical = True

    def _on_circulation(self, channel: str, message: Any) -> None:
        """Handle circulation updates from CirculatorySystem."""
        data = self._parse_message(message)
        if data is None:
            return
        unfulfilled = float(data.get("unfulfilled", 0.0))
        # Adequate if unfulfilled demand is low relative to distribution
        distributed = float(data.get("distributed", 1.0))
        if distributed > 0:
            self._circulation_adequate = (unfulfilled / distributed) < 0.3
        else:
            self._circulation_adequate = unfulfilled == 0.0

    def _on_supply_low(self, channel: str, message: Any) -> None:
        """Handle supply-low alerts from CirculatorySystem."""
        self._circulation_adequate = False

    def _on_population_status(self, channel: str, message: Any) -> None:
        """Handle population status from ReproductiveSystem."""
        data = self._parse_message(message)
        if data is None:
            return
        recommendation = data.get("recommendation")
        if recommendation is not None:
            self._population_healthy = str(recommendation) == "stable"
        else:
            # Spawn/retire events indicate the system is adjusting
            self._population_healthy = True

    def _on_tom_update(self, channel: str, message: Any) -> None:
        """Handle Theory of Mind updates."""
        data = self._parse_message(message)
        if data is None:
            return
        self._social_confidence = float(
            data.get("avg_confidence", self._social_confidence)
        )

    def _on_meta_update(self, channel: str, message: Any) -> None:
        """Handle metacognition updates."""
        data = self._parse_message(message)
        if data is None:
            return
        self._metacognition_score = float(
            data.get("recent_performance", self._metacognition_score)
        )

    def _on_meta_alert(self, channel: str, message: Any) -> None:
        """Handle metacognition degradation alerts."""
        data = self._parse_message(message)
        if data is None:
            return
        self._metacognition_score = float(
            data.get("recent_performance", self._metacognition_score)
        )

    # =====================================================================
    # Core Step
    # =====================================================================

    def step(self, current_step: int = 0) -> None:
        """Advance one organism state cycle.

        Resets per-step accumulators (homeostasis deviation is re-populated
        by incoming corrections each step). The actual state updates happen
        reactively via EventBus callbacks.
        """
        self._step_count = current_step if current_step > 0 else self._step_count + 1

        # Reset homeostasis deviation so it is freshly computed from
        # this step's correction messages
        self._homeostasis_deviation = 0.0

    # =====================================================================
    # Public Query API
    # =====================================================================

    def get_body_state(self) -> dict[str, Any]:
        """Return the unified body state aggregating all 18 systems.

        This is the single authoritative snapshot of Mae's internal
        condition. All values are kept at healthy baselines until
        overwritten by incoming EventBus signals.
        """
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

        Reflexes bypass the advisory decision routing entirely, just as a
        pain withdrawal reflex bypasses conscious processing.
        """
        # Priority 1: Acute pain withdrawal
        if self._pain_load > 0.8:
            return "rest"

        # Priority 2: Vertigo stabilization
        if self._stability < 0.3:
            return "rest"

        # Priority 3: Hypoxia conservation
        if self._oxygen_level < 0.3:
            return "rest"

        # Priority 4: Starvation foraging
        if self._energy_critical:
            return "explore"

        # Priority 5: Kidney stress
        if self._toxin_load > 4.0:
            return "rest"

        # Priority 6: Homeostasis deviation — internal systems significantly
        # out of balance. The organism needs to pause and stabilize before
        # taking further action. This is less urgent than acute emergencies
        # (pain, hypoxia, starvation) but more important than normal routing.
        if self._homeostasis_deviation >= self._HOMEOSTASIS_URGENCY_THRESHOLD:
            return "rest"

        # No emergency — proceed with normal decision cascade
        return None

    def get_decision_context(self) -> dict[str, Any]:
        """Return enriched context for the advisory decision routing.

        Provides composite metrics that the decision router can use
        to bias choices without needing to query individual systems.
        """
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

        # Organism vitality: overall health score (0-1)
        organism_vitality = self._vitality

        return {
            "body_threat_level": round(body_threat_level, 4),
            "body_opportunity_level": round(body_opportunity_level, 4),
            "emotional_bias": emotional_bias,
            "metacognitive_confidence": round(self._metacognition_score, 4),
            "organism_vitality": round(organism_vitality, 4),
        }

    # =====================================================================
    # Action Outcome Feedback
    # =====================================================================

    def report_action_outcome(
        self, action: str, reward: float, step: int,
    ) -> None:
        """Record the outcome of an agent action for vitality tracking.

        Maintains a rolling window of recent rewards and updates the
        vitality metric using an exponential moving average.

        Args:
            action: The action that was taken (e.g. "explore", "rest").
            reward: The reward received (-1 to 1 scale).
            step: The simulation step when the action occurred.
        """
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
        """Parse an EventBus message (may be JSON string or dict).

        Handles both pre-parsed dicts and JSON-serialized strings
        gracefully, returning None for malformed messages.
        """
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
