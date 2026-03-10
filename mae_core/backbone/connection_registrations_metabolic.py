"""Connection Registrations — Group 1: Metabolic/Biological -> OrganismState.

OrganismState is Mae's hypothalamus — integrates signals from every biological
subsystem. Witnesses are PEER systems from the same organ or closely related
domain (not backbone governance).

Extracted from connection_registrations.py for single-responsibility.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from mae_core.backbone.connection_registry import (
    ConnectionCriticality,
    ConnectionRegistry,
    ConnectionType,
)

logger = logging.getLogger(__name__)


def register_metabolic_connections(
    registry: ConnectionRegistry,
    systems: dict[str, Any],
    _reg: Callable,
) -> None:
    """Register Group 1: metabolic/biological -> OrganismState callbacks.

    Args:
        registry: ConnectionRegistry instance.
        systems: System dict (unused here, kept for uniform signature).
        _reg: Inner registration helper from register_all_connections.
    """
    cb = ConnectionType.CALLBACK_REGISTRATION

    # -- Coordination organ (peers witness each other) --
    _reg("emotional_system", "organism_state", cb,
         channel="coordination.emotion_update",
         witnesses=["homeostasis", "endocrine"],
         description="Emotional valence feeds whole-organism state")
    _reg("homeostasis", "organism_state", cb,
         channel="coordination.homeostasis_correction",
         witnesses=["thermoregulation", "endocrine"],
         description="Homeostatic corrections modulate organism state")
    _reg("thermoregulation", "organism_state", cb,
         channel="coordination.cooling_needed",
         witnesses=["respiratory_system", "homeostasis"],
         description="Overheating triggers organism-level response")
    _reg("thermoregulation", "organism_state", cb,
         channel="coordination.warming_needed",
         witnesses=["respiratory_system", "homeostasis"],
         description="Hypothermia triggers organism-level response")
    _reg("thermoregulation", "organism_state", cb,
         channel="coordination.temperature_normal",
         witnesses=["homeostasis", "respiratory_system"],
         description="Temperature normalization updates organism state")
    _reg("digestive_system", "organism_state", cb,
         channel="coordination.digestion_complete",
         witnesses=["energy_reserve", "endocrine"],
         description="Digestion completion updates energy awareness")
    _reg("respiratory_system", "organism_state", cb,
         channel="coordination.respiration_update",
         witnesses=["circulatory_system", "thermoregulation"],
         description="Breathing rate feeds organism awareness")
    _reg("respiratory_system", "organism_state", cb,
         channel="coordination.hypoxia",
         witnesses=["circulatory_system", "auto_healer"],
         criticality=ConnectionCriticality.CRITICAL,
         description="Oxygen deprivation -- emergency organism signal")
    _reg("respiratory_system", "organism_state", cb,
         channel="coordination.hypercapnia",
         witnesses=["circulatory_system", "thermoregulation"],
         criticality=ConnectionCriticality.IMPORTANT,
         description="CO2 buildup -- urgent organism signal")
    _reg("vestibular_system", "organism_state", cb,
         channel="coordination.balance_update",
         witnesses=["proprioception", "somatic_map"],
         description="Balance sense feeds spatial awareness")
    _reg("vestibular_system", "organism_state", cb,
         channel="coordination.vertigo",
         witnesses=["proprioception", "auto_healer"],
         criticality=ConnectionCriticality.IMPORTANT,
         description="Vertigo -- disorientation alarm to organism")
    _reg("arousal_regulator", "organism_state", cb,
         channel="coordination.regulation_signal",
         witnesses=["endocrine", "homeostasis"],
         description="Arousal regulation feeds organism state")

    # -- Communication organ (pain channels -- sensory peers witness) --
    _reg("nociception", "organism_state", cb,
         channel="communication.acute_pain",
         witnesses=["auto_healer", "threat_detector"],
         criticality=ConnectionCriticality.CRITICAL,
         description="Acute pain -- immediate organism emergency")
    _reg("nociception", "organism_state", cb,
         channel="communication.pain_overload",
         witnesses=["auto_healer", "endocrine"],
         criticality=ConnectionCriticality.CRITICAL,
         description="Pain overload -- system shutdown threshold")
    _reg("nociception", "organism_state", cb,
         channel="communication.pain_update",
         witnesses=["proprioception", "emotional_system"],
         description="Pain status feeds body/emotional awareness")

    # -- Emergent organ (maintenance peers witness each other) --
    _reg("lymphatic_system", "organism_state", cb,
         channel="emergent.lymph_status",
         witnesses=["microbiome", "auto_healer"],
         description="Lymph health feeds organism immune picture")
    _reg("lymphatic_system", "organism_state", cb,
         channel="emergent.lymph_overflow",
         witnesses=["renal_filter", "auto_healer"],
         criticality=ConnectionCriticality.IMPORTANT,
         description="Lymph overflow -- waste system backing up")
    _reg("senescence", "organism_state", cb,
         channel="emergent.rejuvenation_needed",
         witnesses=["auto_healer", "lymphatic_system"],
         description="Aging signal requests rejuvenation")
    _reg("senescence", "organism_state", cb,
         channel="emergent.system_senescent",
         witnesses=["auto_healer", "lymphatic_system"],
         description="System marked senescent -- organism awareness")
    _reg("senescence", "organism_state", cb,
         channel="emergent.age_update",
         witnesses=["lymphatic_system", "auto_healer"],
         description="Age tracking -- cleanup + repair peers verify aging")
    _reg("proprioception", "organism_state", cb,
         channel="emergent.proprioception_update",
         witnesses=["vestibular_system", "somatic_map"],
         description="Body position sense feeds organism awareness")
    _reg("microbiome", "organism_state", cb,
         channel="emergent.microbiome_status",
         witnesses=["lymphatic_system", "digestive_system"],
         description="Microbiome health -- gut peers witness")
    _reg("microbiome", "organism_state", cb,
         channel="emergent.microbiome_imbalanced",
         witnesses=["lymphatic_system", "auto_healer"],
         criticality=ConnectionCriticality.IMPORTANT,
         description="Dysbiosis -- gut flora destabilized")

    # -- Defense organ (defense peers witness each other) --
    _reg("boundary_membrane", "organism_state", cb,
         channel="defense.membrane_status",
         witnesses=["renal_filter", "threat_detector"],
         description="Boundary integrity -- defense peers witness")
    _reg("boundary_membrane", "organism_state", cb,
         channel="defense.quarantine_event",
         witnesses=["threat_detector", "pearl_defense"],
         criticality=ConnectionCriticality.IMPORTANT,
         description="Quarantine event -- defense triad witnesses")
    _reg("renal_filter", "organism_state", cb,
         channel="defense.renal_status",
         witnesses=["boundary_membrane", "lymphatic_system"],
         description="Kidney function -- filtering peers witness")
    _reg("renal_filter", "organism_state", cb,
         channel="defense.kidney_failure",
         witnesses=["auto_healer", "boundary_membrane"],
         criticality=ConnectionCriticality.CRITICAL,
         description="Kidney failure -- critical organ emergency")
    _reg("renal_filter", "organism_state", cb,
         channel="defense.toxic_filtered",
         witnesses=["boundary_membrane", "lymphatic_system"],
         description="Toxin filtered -- waste processing peers witness")

    # -- Memory organ (energy -- metabolic peers witness) --
    _reg("energy_reserve", "organism_state", cb,
         channel="memory.energy_status",
         witnesses=["digestive_system", "circulatory_system"],
         description="Energy reserves -- metabolic peers witness")
    _reg("energy_reserve", "organism_state", cb,
         channel="memory.starvation_alert",
         witnesses=["auto_healer", "circulatory_system"],
         criticality=ConnectionCriticality.CRITICAL,
         description="Starvation -- critical energy emergency")
    _reg("energy_reserve", "organism_state", cb,
         channel="memory.reserves_full",
         witnesses=["digestive_system", "endocrine"],
         description="Full reserves -- metabolic peers witness")

    # -- Substrate organ (circulatory -- transport peers witness) --
    _reg("circulatory_system", "organism_state", cb,
         channel="substrate.circulation_update",
         witnesses=["respiratory_system", "energy_reserve"],
         description="Circulation -- oxygen/energy delivery peers witness")
    _reg("circulatory_system", "organism_state", cb,
         channel="substrate.supply_low",
         witnesses=["energy_reserve", "auto_healer"],
         criticality=ConnectionCriticality.IMPORTANT,
         description="Supply shortage -- resource peers witness")

    # -- Morphogenesis organ (lifecycle peers witness) --
    _reg("reproductive_system", "organism_state", cb,
         channel="morphogenesis.retire_request",
         witnesses=["morph_coordinator", "stem_cell_registry"],
         description="Agent retirement -- lifecycle peers witness")
    _reg("reproductive_system", "organism_state", cb,
         channel="morphogenesis.population_status",
         witnesses=["morph_coordinator", "stem_cell_registry"],
         description="Population metrics -- lifecycle peers witness")
