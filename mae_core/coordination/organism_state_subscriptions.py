"""OrganismState subscription wiring and EventBus callback handlers.

Contains _subscribe_all and all _on_* callbacks for the 18 biological systems.
Extracted from organism_state.py to stay under 500-line limit.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

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


class OrganismStateSubscriptionsMixin:
    """EventBus subscription wiring and all _on_* callback handlers."""

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
        """Handle correction signals from HomeostasisRegulator."""
        data = self._parse_message(message)
        if data is None:
            return
        urgency = float(data.get("urgency", 0.0))
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
