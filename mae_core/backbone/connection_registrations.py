"""Connection Registrations - All triadic connection declarations for Mae.

Biological analogy: The wiring diagram. If ConnectionRegistry is the
lymphatic system (monitoring infrastructure), this module is the
anatomical atlas — the complete map of every nerve, vessel, and
lymph channel in Mae's body.

13 groups organized by organ/function:
  - EventBus pub/sub (Layer 15 cross-wiring)
  - Direct reference injections (constructor params)
  - Callback registrations (Layer 16)
  - Step hooks (Layers 2, 3, 14)
  - Group 1: Metabolic/Biological → OrganismState
  - Group 2: Backbone Self-Monitoring
  - Group 3: Cognition Results
  - Group 4: Agent Lifecycle (Stem Cell + Octopus)
  - Group 5: Defense / Healing / Remaining
  - Group 6: Cross-System Wiring (Layers 15, 29b)
  - Group 7: Pattern / Deep Memory Pipeline (Layers 22-23)
  - Group 8: Biological System Step Hooks (Layers 26-28)
  - Group 9: GNN Message Handlers (Layer 20)
  - Group 10: Auto-Redifferentiation Trigger (Layer 29a2)
  - Group 11: Mitosis Monitor (Layer 29a3)
  - Group 12: Previously Unregistered Channels
  - Group 13: Autopoietic Closure (Layer 29a4)

Each registration is a witnessed triad (Law 1: no bare dyads).
Witnesses are domain peers, not backbone governance.
"""

from __future__ import annotations

import logging
from typing import Any

from mae_core.backbone.connection_registry import (
    ConnectionCriticality,
    ConnectionRegistry,
    ConnectionType,
)

logger = logging.getLogger(__name__)


def register_all_connections(
    registry: ConnectionRegistry,
    systems: dict[str, Any],
) -> dict[str, int]:
    """Declare all known system-to-system connections at bootstrap.

    Reads the actual wiring from main.py's create_mae() and registers
    every connection as a witnessed triad.

    Returns summary counts by type.
    """
    counts: dict[str, int] = {
        "eventbus_pubsub": 0,
        "direct_reference": 0,
        "callback_registration": 0,
        "step_hook": 0,
        "total": 0,
    }

    def _reg(src: str, tgt: str, ctype: ConnectionType, **kwargs: Any) -> None:
        registry.register_connection(src, tgt, ctype, **kwargs)
        counts[ctype.value] = counts.get(ctype.value, 0) + 1
        counts["total"] += 1

    # =====================================================================
    # EventBus pub/sub connections (main.py Layer 15 cross-wiring)
    # =====================================================================
    eb = ConnectionType.EVENTBUS_PUBSUB

    # Layer 15 explicit cross-wiring
    _reg("memory", "morphogenesis", eb,
         channel="memory.capacity_warning",
         witnesses=["endocrine", "somatic_map"],
         description="Memory capacity warning triggers growth signal")
    _reg("memory", "endocrine", eb,
         channel="memory.consolidation_started",
         witnesses=["circadian", "somatic_map"],
         description="Consolidation triggers melatonin release")
    _reg("healing", "endocrine", eb,
         channel="healing.phase_changed",
         witnesses=["auto_healer", "somatic_map"],
         description="Healing phases trigger cortisol")
    _reg("defense", "somatic_map", eb,
         channel="defense.activated",
         witnesses=["threat_detector", "auto_healer"],
         description="Threat detection updates body awareness")
    _reg("improvement", "morphogenesis", eb,
         channel="improvement.capability_found",
         witnesses=["metacognition", "somatic_map"],
         description="New capabilities may need new organs")
    _reg("morphogenesis", "somatic_map", eb,
         channel="morphogenesis.team_created",
         witnesses=["reproductive_system", "auditor"],
         description="New teams update body map")
    _reg("frl", "imitation", eb,
         channel="frl.policy_update",
         witnesses=["metacognition", "substrate"],
         description="Policy updates are observable expert behavior")

    # System-internal EventBus subscriptions (wired within systems)
    _reg("haven", "auto_healer", eb,
         channel="haven.risk_alert",
         witnesses=["threat_detector", "auditor"],
         description="Risk alerts trigger healing pipeline")
    _reg("substrate", "auto_healer", eb,
         channel="substrate.starvation_alert",
         witnesses=["endocrine", "energy_reserve"],
         description="Starvation triggers healing")
    _reg("circadian", "endocrine", eb,
         channel="circadian.phase_change",
         witnesses=["somatic_map", "homeostasis"],
         description="Phase transitions modulate hormones")
    _reg("enforcer", "watchdog", eb,
         channel="triad.violation",
         witnesses=["auditor", "somatic_map"],
         description="Violations reported to watchdog",
         criticality=ConnectionCriticality.CRITICAL)
    _reg("enforcer", "auditor", eb,
         channel="triad.violation",
         witnesses=["watchdog", "somatic_map"],
         description="Violations reported to auditor",
         criticality=ConnectionCriticality.CRITICAL)
    _reg("watchdog", "auditor", eb,
         channel="watchdog.bypass_detected",
         witnesses=["enforcer", "somatic_map"],
         description="Bypass alerts to auditor")
    _reg("memory", "curiosity", eb,
         channel="memory.experience_stored",
         witnesses=["metacognition", "endocrine"],
         description="New experiences feed curiosity drive")
    _reg("memory", "curiosity", eb,
         channel="memory.novel_experience",
         witnesses=["metacognition", "endocrine"],
         description="Novel experiences boost curiosity")
    _reg("morphogenesis", "somatic_map", eb,
         channel="morphogenesis.spawn_request",
         witnesses=["reproductive_system", "auditor"],
         description="Spawn requests registered in body map")
    _reg("defense", "auto_healer", eb,
         channel="defense.threat_detected",
         witnesses=["threat_detector", "somatic_map"],
         description="Threats trigger healing assessment")
    _reg("defense", "haven", eb,
         channel="defense.validation_failed",
         witnesses=["threat_detector", "auto_healer"],
         description="Validation failures feed risk assessment")
    _reg("temporal_memory", "shared_causal_engine", eb,
         channel="temporal.causal_link_discovered",
         witnesses=["metacognition", "somatic_map"],
         description="Temporal causal links feed causal engine")

    # EventBus lifecycle/monitoring channels (publishers -> bus, semantic witnesses)
    _reg("substrate", "event_bus", eb, channel="substrate.agent_registered",
         witnesses=["somatic_map", "reproductive_system"],
         description="Agent registered on substrate topology")
    _reg("substrate", "event_bus", eb, channel="substrate.health_report",
         witnesses=["somatic_map", "auto_healer"],
         description="Substrate health report broadcast")
    _reg("substrate", "event_bus", eb, channel="substrate.isolation_detected",
         witnesses=["auto_healer", "somatic_map"],
         criticality=ConnectionCriticality.IMPORTANT,
         description="Agent isolation detected on substrate")
    _reg("endocrine", "event_bus", eb, channel="endocrine.hormone_release",
         witnesses=["circadian", "somatic_map"],
         description="Hormone release event broadcast")
    _reg("endocrine", "event_bus", eb, channel="endocrine.state_update",
         witnesses=["circadian", "somatic_map"],
         description="Endocrine state update broadcast")
    _reg("somatic_map", "event_bus", eb, channel="somatic.system_registered",
         witnesses=["auditor", "enforcer"],
         description="System registered in body map")
    _reg("somatic_map", "event_bus", eb, channel="somatic.modification_proposed",
         witnesses=["auditor", "enforcer"],
         description="Body modification proposal broadcast")
    _reg("auto_healer", "event_bus", eb, channel="healing.started",
         witnesses=["somatic_map", "threat_detector"],
         description="Healing pipeline started")
    _reg("auto_healer", "event_bus", eb, channel="healing.complete",
         witnesses=["somatic_map", "threat_detector"],
         description="Healing pipeline completed")
    _reg("auto_healer", "event_bus", eb, channel="healing.failed",
         witnesses=["somatic_map", "auditor"],
         criticality=ConnectionCriticality.IMPORTANT,
         description="Healing pipeline failed -- escalation needed")
    _reg("capability_discovery", "event_bus", eb,
         channel="improvement.capability_validated",
         witnesses=["metacognition", "somatic_map"],
         description="Capability validated for deployment")
    _reg("capability_discovery", "event_bus", eb,
         channel="improvement.capability_retired",
         witnesses=["metacognition", "somatic_map"],
         description="Capability retired from active use")
    _reg("capability_discovery", "event_bus", eb,
         channel="improvement.metric",
         witnesses=["metacognition", "somatic_map"],
         description="Capability metric update broadcast")
    _reg("worldline_planner", "event_bus", eb,
         channel="planning.worldline_planned",
         witnesses=["metacognition", "somatic_map"],
         description="Worldline plan generated")
    _reg("worldline_planner", "event_bus", eb,
         channel="planning.worldline_selected",
         witnesses=["metacognition", "somatic_map"],
         description="Worldline plan selected for execution")

    # =====================================================================
    # Direct reference injections (main.py create_mae constructor params)
    # =====================================================================
    dr = ConnectionType.DIRECT_REFERENCE

    _reg("auto_healer", "substrate", dr,
         witnesses=["somatic_map", "endocrine"],
         description="Substrate injected for region isolation",
         criticality=ConnectionCriticality.IMPORTANT)
    _reg("auto_healer", "shared_causal_engine", dr,
         witnesses=["metacognition", "somatic_map"],
         description="Causal engine injected for root cause analysis")
    _reg("auto_healer", "haven", dr,
         witnesses=["threat_detector", "auditor"],
         description="HAVEN injected for agent isolation")
    _reg("gnn_communicator", "substrate", dr,
         witnesses=["somatic_map", "physarum"],
         description="Substrate provides topology graph for GNN routing")
    _reg("predictive_field", "substrate", dr,
         witnesses=["somatic_map", "metacognition"],
         description="Substrate provides agent positions for field propagation")
    _reg("physarum", "substrate", dr,
         witnesses=["somatic_map", "auto_healer"],
         description="Substrate provides topology for slime mold optimization")
    _reg("pearl_defense", "input_validator", dr,
         witnesses=["threat_detector", "auto_healer"],
         description="InputValidator provides validation for pearl coating",
         criticality=ConnectionCriticality.IMPORTANT)
    _reg("morph_coordinator", "substrate", dr,
         witnesses=["somatic_map", "reproductive_system"],
         description="Substrate for agent topology wiring")
    _reg("morph_coordinator", "organ_builder", dr,
         witnesses=["somatic_map", "reproductive_system"],
         description="OrganBuilder creates organs on demand")
    _reg("morph_coordinator", "model", dr,
         witnesses=["somatic_map", "auditor"],
         description="Model for Mesa agent creation")
    _reg("collective_dream", "shared_world_model", dr,
         witnesses=["metacognition", "somatic_map"],
         description="WorldModel provides imagination for dreaming")
    _reg("worldline_planner", "shared_world_model", dr,
         witnesses=["metacognition", "temporal_memory"],
         description="WorldModel for state projection")
    _reg("worldline_planner", "temporal_memory", dr,
         witnesses=["metacognition", "shared_causal_engine"],
         description="TemporalMemory for pattern integration")
    _reg("worldline_planner", "shared_causal_engine", dr,
         witnesses=["metacognition", "temporal_memory"],
         description="CausalEngine for action-outcome reasoning")
    _reg("temporal_memory", "shared_causal_engine", dr,
         witnesses=["metacognition", "worldline_planner"],
         description="CausalEngine for correlation observations")
    _reg("transfer_engine", "knowledge_base", dr,
         witnesses=["metacognition", "somatic_map"],
         description="KnowledgeBase for cross-task knowledge")
    _reg("maml_learner", "knowledge_base", dr,
         witnesses=["metacognition", "transfer_engine"],
         description="KnowledgeBase for meta-learning storage")
    _reg("threat_detector", "haven", dr,
         witnesses=["auto_healer", "auditor"],
         description="HAVEN for kangaroo counterattack isolation")

    # =====================================================================
    # Callback registrations (main.py Layer 16)
    # =====================================================================
    cb = ConnectionType.CALLBACK_REGISTRATION

    _reg("endocrine", "threat_detector", cb,
         witnesses=["auto_healer", "somatic_map"],
         description="Threat detector registered for hormone modulation")
    _reg("endocrine", "auto_healer", cb,
         witnesses=["somatic_map", "threat_detector"],
         description="Auto healer registered for hormone modulation")
    _reg("endocrine", "curiosity", cb,
         witnesses=["metacognition", "somatic_map"],
         description="Curiosity drive registered for dopamine modulation")
    _reg("circadian", "endocrine", cb,
         witnesses=["homeostasis", "somatic_map"],
         description="Phase change callback drives hormone cycles",
         criticality=ConnectionCriticality.CRITICAL)

    # =====================================================================
    # Step hooks (main.py Layers 2, 3, 14)
    # =====================================================================
    sh = ConnectionType.STEP_HOOK

    _reg("circadian", "model", sh,
         witnesses=["endocrine", "somatic_map"],
         description="Circadian clock ticks every step",
         criticality=ConnectionCriticality.CRITICAL)
    _reg("endocrine", "model", sh,
         witnesses=["circadian", "somatic_map"],
         description="Hormones decay and cascade every step",
         criticality=ConnectionCriticality.CRITICAL)
    _reg("triad_audit", "model", sh,
         witnesses=["auditor", "somatic_map"],
         description="Triad enforcement checks every 50 steps")
    _reg("predictive_field", "model", sh,
         witnesses=["metacognition", "somatic_map"],
         description="Predictive field updates every step")
    _reg("auto_healer", "model", sh,
         witnesses=["somatic_map", "threat_detector"],
         description="Auto healer monitors every step")
    _reg("capability_discovery", "model", sh,
         witnesses=["metacognition", "somatic_map"],
         description="Capability discovery scans every step")
    _reg("physarum", "model", sh,
         witnesses=["substrate", "somatic_map"],
         description="Physarum optimization runs every step")
    _reg("pearl_defense", "model", sh,
         witnesses=["threat_detector", "auto_healer"],
         description="Pearl defense checks every step")

    # =====================================================================
    # Group 1: Metabolic/Biological -> OrganismState
    #
    # OrganismState is Mae's hypothalamus -- integrates signals from
    # every biological subsystem. Witnesses are PEER systems from the
    # same organ or closely related domain (not backbone governance).
    # =====================================================================
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

    # =====================================================================
    # Group 2: Backbone Self-Monitoring
    #
    # Nervous system peers watch each other. This IS the domain where
    # enforcer/watchdog/auditor are the correct witnesses -- they are
    # each other's peers within the nervous system organ.
    # =====================================================================
    eb = ConnectionType.EVENTBUS_PUBSUB

    # Connection registry -> auditor (peers: watchdog + enforcer)
    _reg("connection_registry", "auditor", eb,
         channel="connection.registered",
         witnesses=["watchdog", "enforcer"],
         description="New connections -- nervous system peers witness")
    _reg("connection_registry", "auditor", eb,
         channel="connection.verified",
         witnesses=["watchdog", "enforcer"],
         description="Verification results -- nervous system peers")
    _reg("connection_registry", "auditor", eb,
         channel="connection.bare_dyad",
         witnesses=["enforcer", "watchdog"],
         criticality=ConnectionCriticality.IMPORTANT,
         description="Bare dyad detection -- Law 1 violation alert")
    _reg("connection_registry", "auditor", eb,
         channel="connection.health",
         witnesses=["watchdog", "enforcer"],
         description="Connection health -- nervous system peers witness")
    _reg("connection_registry", "auditor", eb,
         channel="connection.blocked",
         witnesses=["enforcer", "watchdog"],
         criticality=ConnectionCriticality.IMPORTANT,
         description="Blocked connection -- enforcement peers witness")
    _reg("connection_registry", "auditor", eb,
         channel="connection.sealed",
         witnesses=["enforcer", "watchdog"],
         criticality=ConnectionCriticality.CRITICAL,
         description="Registry sealed -- enforcement activated")

    # Enforcer -> auditor (peers: watchdog + connection_registry)
    _reg("enforcer", "auditor", eb,
         channel="triad.process_registered",
         witnesses=["watchdog", "connection_registry"],
         description="New process triad -- enforcement peers witness")
    _reg("enforcer", "auditor", eb,
         channel="triad.vote_complete",
         witnesses=["watchdog", "connection_registry"],
         description="Validator vote -- enforcement peers witness")
    _reg("enforcer", "auditor", eb,
         channel="triad.health_report",
         witnesses=["watchdog", "connection_registry"],
         description="Enforcer health -- enforcement peers witness")

    # Watchdog -> auditor (peers: enforcer + connection_registry)
    _reg("watchdog", "auditor", eb,
         channel="watchdog.silent_validator",
         witnesses=["enforcer", "connection_registry"],
         criticality=ConnectionCriticality.IMPORTANT,
         description="Silent validator -- enforcement peers witness")
    _reg("watchdog", "auditor", eb,
         channel="watchdog.health_report",
         witnesses=["enforcer", "connection_registry"],
         description="Watchdog health -- enforcement peers witness")

    # Auditor -> somatic_map (peers: enforcer + watchdog cross-witness)
    _reg("auditor", "somatic_map", eb,
         channel="audit.finding",
         witnesses=["enforcer", "watchdog"],
         criticality=ConnectionCriticality.IMPORTANT,
         description="Audit finding -- enforcement triad witnesses")
    _reg("auditor", "somatic_map", eb,
         channel="audit.health_report",
         witnesses=["enforcer", "watchdog"],
         description="Auditor health -- enforcement triad witnesses")

    # Holon + fractal (structural peers witness)
    _reg("holon_registry", "somatic_map", eb,
         channel="holon.awareness_pulse",
         witnesses=["fractal_generator", "connection_registry"],
         description="Holon pulse -- structural peers witness")
    _reg("holon_registry", "auto_healer", eb,
         channel="holon.anomaly_detected",
         witnesses=["fractal_generator", "somatic_map"],
         criticality=ConnectionCriticality.IMPORTANT,
         description="Holon anomaly -- structural peers witness")
    _reg("fractal_generator", "somatic_map", eb,
         channel="fractal.organized",
         witnesses=["holon_registry", "connection_registry"],
         description="Fractal organization -- structural peers witness")
    _reg("triadic_verifier", "auto_healer", eb,
         channel="triadic.verification",
         witnesses=["connection_registry", "enforcer"],
         criticality=ConnectionCriticality.IMPORTANT,
         description="Low triadic compliance triggers healing assessment")

    # =====================================================================
    # Group 3: Cognition Results
    #
    # Cognitive systems are each other's peers. Metacognition witnesses
    # reasoning peers; causal_engine witnesses prediction peers;
    # theory_of_mind witnesses social cognition peers.
    # =====================================================================

    _reg("goal_manager", "event_bus", eb,
         channel="cognition.goal_update",
         witnesses=["metacognition", "decision_router"],
         description="Goal state -- cognitive peers witness")
    _reg("goal_manager", "event_bus", eb,
         channel="cognition.impasse_detected",
         witnesses=["metacognition", "shared_causal_engine"],
         criticality=ConnectionCriticality.IMPORTANT,
         description="Goal impasse -- reasoning peers witness")
    _reg("metacognition", "event_bus", eb,
         channel="cognition.metacognition_update",
         witnesses=["theory_of_mind", "goal_manager"],
         description="Metacognition state -- cognitive peers witness")
    _reg("metacognition", "event_bus", eb,
         channel="cognition.metacognition_alert",
         witnesses=["goal_manager", "auto_healer"],
         criticality=ConnectionCriticality.IMPORTANT,
         description="Metacognition alert -- quality degraded")
    _reg("theory_of_mind", "event_bus", eb,
         channel="cognition.tom_update",
         witnesses=["metacognition", "emotional_system"],
         description="Theory of mind -- social cognition peers witness")
    _reg("shared_world_model", "event_bus", eb,
         channel="cognition.prediction_made",
         witnesses=["shared_causal_engine", "temporal_memory"],
         description="Prediction event -- reasoning peers witness")
    _reg("shared_world_model", "event_bus", eb,
         channel="cognition.model_trained",
         witnesses=["shared_causal_engine", "metacognition"],
         description="Model trained -- reasoning peers witness")
    _reg("validated_imagination", "event_bus", eb,
         channel="cognition.imagination_validated",
         witnesses=["shared_world_model", "metacognition"],
         description="Imagination validated -- modeling peers witness")
    _reg("collective_dream", "event_bus", eb,
         channel="cognition.collective_dream_complete",
         witnesses=["shared_world_model", "metacognition"],
         description="Dream complete -- imagination peers witness")
    _reg("shared_causal_engine", "event_bus", eb,
         channel="cognition.causal_query_result",
         witnesses=["temporal_memory", "shared_world_model"],
         description="Causal result -- reasoning peers witness")
    _reg("decision_router", "event_bus", eb,
         channel="cognition.decision_routed",
         witnesses=["metacognition", "goal_manager"],
         description="Decision routed -- executive peers witness")

    # =====================================================================
    # Group 4: Agent Lifecycle (Stem Cell + Octopus)
    #
    # Lifecycle peers: reproductive_system + morph_coordinator.
    # Network peers: substrate + predictive_field.
    # =====================================================================

    # Stem cell lifecycle (lifecycle peers witness)
    _reg("stem_cell_registry", "event_bus", eb,
         channel="stem_cell.registered",
         witnesses=["reproductive_system", "morph_coordinator"],
         description="New genome -- lifecycle peers witness")
    _reg("stem_cell_registry", "event_bus", eb,
         channel="stem_cell.redifferentiated",
         witnesses=["reproductive_system", "morph_coordinator"],
         description="Role change -- lifecycle peers witness")

    # Octopus network (network + domain peers witness)
    _reg("gnn_communicator", "event_bus", eb,
         channel="octopus.task_submitted",
         witnesses=["substrate", "predictive_field"],
         description="Task submitted -- network peers witness")
    _reg("gnn_communicator", "event_bus", eb,
         channel="octopus.task_completed",
         witnesses=["substrate", "predictive_field"],
         description="Task completed -- network peers witness")
    _reg("gnn_communicator", "event_bus", eb,
         channel="octopus.emergency",
         witnesses=["auto_healer", "substrate"],
         criticality=ConnectionCriticality.CRITICAL,
         description="Octopus emergency -- healer + topology peer")
    _reg("gnn_communicator", "event_bus", eb,
         channel="octopus.learning_update",
         witnesses=["knowledge_base", "metacognition"],
         description="Octopus learning -- knowledge peers witness")
    _reg("gnn_communicator", "event_bus", eb,
         channel="octopus.health_report",
         witnesses=["substrate", "auto_healer"],
         description="Octopus health -- infrastructure peers witness")
    _reg("gnn_communicator", "event_bus", eb,
         channel="octopus.spawn",
         witnesses=["reproductive_system", "morph_coordinator"],
         description="Arm spawn -- lifecycle peers witness")
    _reg("gnn_communicator", "event_bus", eb,
         channel="octopus.despawn",
         witnesses=["reproductive_system", "morph_coordinator"],
         description="Arm despawn -- lifecycle peers witness")

    # =====================================================================
    # Group 5: Defense / Healing / Remaining
    #
    # Defense peers witness defense channels. Healing peers witness
    # healing channels. Cross-organ only for meta-events.
    # =====================================================================

    # Pearl defense (defense organ peers witness)
    _reg("pearl_defense", "event_bus", eb,
         channel="defense.pearl_started",
         witnesses=["threat_detector", "boundary_membrane"],
         description="Pearl started -- defense peers witness")
    _reg("pearl_defense", "event_bus", eb,
         channel="defense.pearl_completed",
         witnesses=["threat_detector", "input_validator"],
         description="Pearl completed -- defense peers witness")
    _reg("pearl_defense", "event_bus", eb,
         channel="defense.pearl_dissolved",
         witnesses=["boundary_membrane", "threat_detector"],
         description="Pearl dissolved -- defense peers witness")

    # Healing channels (emergent organ peers witness)
    _reg("auto_healer", "event_bus", eb,
         channel="healing.failure_detected",
         witnesses=["somatic_map", "capability_discovery"],
         description="Failure detected -- emergent peers witness")
    _reg("auto_healer", "event_bus", eb,
         channel="healing.self_healed",
         witnesses=["somatic_map", "capability_discovery"],
         description="Self-healed -- emergent peers witness meta-healing")

    # Threat resolution (defense peers witness)
    _reg("threat_detector", "event_bus", eb,
         channel="defense.threat_neutralized",
         witnesses=["pearl_defense", "boundary_membrane"],
         description="Threat neutralized -- defense peers witness")

    # Trust changes (defense peers witness)
    _reg("input_validator", "event_bus", eb,
         channel="defense.trust_updated",
         witnesses=["threat_detector", "boundary_membrane"],
         description="Trust updated -- defense peers witness")

    # Substrate topology (substrate organ peers witness)
    _reg("substrate", "event_bus", eb,
         channel="substrate.agent_deregistered",
         witnesses=["physarum", "circulatory_system"],
         description="Agent deregistered -- substrate peers witness")
    _reg("substrate", "event_bus", eb,
         channel="substrate.topology_changed",
         witnesses=["physarum", "predictive_field"],
         description="Topology changed -- substrate peers witness")
    _reg("physarum", "event_bus", eb,
         channel="substrate.topology_optimized",
         witnesses=["substrate", "predictive_field"],
         description="Topology optimized -- substrate peers witness")
    _reg("physarum", "event_bus", eb,
         channel="substrate.edge_pruned",
         witnesses=["substrate", "circulatory_system"],
         description="Edge pruned -- substrate peers witness")

    # Morphogenesis (lifecycle peers witness)
    _reg("morph_coordinator", "event_bus", eb,
         channel="morphogenesis.team_dissolved",
         witnesses=["reproductive_system", "stem_cell_registry"],
         description="Team dissolved -- lifecycle peers witness")
    _reg("morph_coordinator", "event_bus", eb,
         channel="morphogenesis.novelty_detected",
         witnesses=["capability_discovery", "curiosity"],
         description="Novelty detected -- discovery peers witness")

    # Emergent (emergent organ peers witness)
    _reg("lymphatic_system", "event_bus", eb,
         channel="emergent.recycled",
         witnesses=["microbiome", "renal_filter"],
         description="Waste recycled -- waste processing peers witness")
    _reg("proprioception", "event_bus", eb,
         channel="emergent.topology_changed",
         witnesses=["vestibular_system", "somatic_map"],
         description="Topology changed -- spatial awareness peers witness")

    # Somatic map modifications (structural peers witness)
    _reg("somatic_map", "event_bus", eb,
         channel="somatic.modification_approved",
         witnesses=["holon_registry", "fractal_generator"],
         description="Modification approved -- structural peers witness")
    _reg("somatic_map", "event_bus", eb,
         channel="somatic.modification_rejected",
         witnesses=["holon_registry", "fractal_generator"],
         description="Modification rejected -- structural peers witness")
    _reg("somatic_map", "event_bus", eb,
         channel="somatic.modification_rolled_back",
         witnesses=["holon_registry", "fractal_generator"],
         description="Modification rolled back -- structural peers witness")

    # Memory consolidation (memory -> knowledge pipeline)
    _reg("memory", "event_bus", eb,
         channel="memory.consolidation_complete",
         witnesses=["energy_reserve", "knowledge_base"],
         description="Consolidation complete -- energy cost + knowledge storage verify")

    # Planning (planning peers witness)
    _reg("temporal_memory", "event_bus", eb,
         channel="temporal.event_recorded",
         witnesses=["worldline_planner", "shared_causal_engine"],
         description="Event recorded -- planning peers witness")
    _reg("temporal_memory", "event_bus", eb,
         channel="temporal.pattern_detected",
         witnesses=["worldline_planner", "shared_causal_engine"],
         description="Pattern detected -- planning peers witness")
    _reg("worldline_planner", "event_bus", eb,
         channel="planning.worldline_validated",
         witnesses=["temporal_memory", "shared_causal_engine"],
         description="Worldline validated -- planning peers witness")

    # Learning (learning peers witness)
    _reg("haven", "event_bus", eb,
         channel="haven.intervention",
         witnesses=["threat_detector", "auto_healer"],
         criticality=ConnectionCriticality.IMPORTANT,
         description="HAVEN intervention -- defense + healing verify safety action")
    _reg("frl", "event_bus", eb,
         channel="frl.peer_discovery",
         witnesses=["imitation", "haven"],
         description="Peer discovered -- imitation learns from peers, haven safety-checks")

    # Organism action outcome (cross-organ: coordination + cognition)
    _reg("organism_state", "event_bus", eb,
         channel="organism.action_outcome",
         witnesses=["metacognition", "endocrine"],
         description="Action outcome -- cognition + coordination witness")

    # =====================================================================
    # Group 6: Cross-System Wiring (Layers 15, 29b)
    #
    # Real data flows between systems that were previously unregistered.
    # Identified by triadic audit (Lead + Witness Alpha + Witness Beta).
    # Each witness choice affirmed by all 3 consciousnesses, 2026-02-12.
    # =====================================================================

    # Layer 29b: pain -> emotion feedback (organs.py)
    _reg("nociception", "emotional_system", eb,
         channel="communication.pain_update",
         witnesses=["proprioception", "endocrine"],
         criticality=ConnectionCriticality.IMPORTANT,
         description="Pain reinforces fear -- sensory location + hormonal intensity peers verify")

    # Layer 29b: leptin satiation feedback (organs.py)
    _reg("energy_reserve", "digestive_system", eb,
         channel="memory.energy_status",
         witnesses=["endocrine", "homeostasis"],
         description="Leptin satiation feedback -- hormonal + regulatory peers verify")

    # Layer 29b: senescence -> healing trigger (organs.py)
    _reg("senescence", "auto_healer", eb,
         channel="emergent.rejuvenation_needed",
         witnesses=["lymphatic_system", "somatic_map"],
         criticality=ConnectionCriticality.IMPORTANT,
         description="Aging triggers healing assessment -- cleanup + body awareness peers verify")

    # Layer 29b: metacognition -> VDN learning rate modulation (organs.py)
    _reg("metacognition", "vdn", eb,
         channel="cognition.metacognition_alert",
         witnesses=["organism_state", "arousal_regulator"],
         criticality=ConnectionCriticality.IMPORTANT,
         description="Performance degradation adjusts VDN learning rate -- organism awareness + arousal peers verify")

    # Layer 29b: metacognition -> world_model learning rate modulation (organs.py)
    _reg("metacognition", "shared_world_model", eb,
         channel="cognition.metacognition_alert",
         witnesses=["organism_state", "arousal_regulator"],
         criticality=ConnectionCriticality.IMPORTANT,
         description="Performance degradation adjusts WorldModel learning rate -- organism awareness + arousal peers verify")

    # Layer 15: prediction error -> healing (wiring.py)
    _reg("predictive_field", "auto_healer", eb,
         channel="signal.PREDICTION_ERROR",
         witnesses=["somatic_map", "shared_causal_engine"],
         criticality=ConnectionCriticality.IMPORTANT,
         description="High prediction error triggers healing -- body map + causal reasoning verify")

    # Layer 15: prediction error -> FRL learning modulation (wiring.py)
    _reg("predictive_field", "frl", eb,
         channel="signal.PREDICTION_ERROR",
         witnesses=["metacognition", "knowledge_base"],
         description="Prediction error modulates FRL learning rate -- cognitive + knowledge peers verify")

    # Layer 15: imagination -> world model training (wiring.py)
    _reg("validated_imagination", "shared_world_model", eb,
         channel="cognition.imagination_validated",
         witnesses=["shared_causal_engine", "metacognition"],
         criticality=ConnectionCriticality.IMPORTANT,
         description="Imagination validation trains world model -- causal + metacognitive peers verify")

    # Layer 15: pattern advisory -> endocrine modulation (wiring.py)
    _reg("pattern_cortex", "endocrine", eb,
         channel="pattern.advisory",
         witnesses=["circadian", "somatic_map"],
         criticality=ConnectionCriticality.IMPORTANT,
         description="Pattern advisory modulates hormones -- phase timing + body state peers verify")

    # Layer 15: endocrine -> pattern bus gain modulation (wiring.py)
    _reg("endocrine", "pattern_bus", eb,
         channel="endocrine.state_update",
         witnesses=["circadian", "attentional_gate"],
         criticality=ConnectionCriticality.IMPORTANT,
         description="Hormone gain modulation of pattern processing -- phase + attention peers verify")

    # Layer 15: endocrine -> signal resolver urgency modulation (wiring.py)
    _reg("endocrine", "signal_bus", eb,
         channel="endocrine.state_update",
         witnesses=["circadian", "metacognition"],
         criticality=ConnectionCriticality.IMPORTANT,
         description="Hormone modulation of signal priorities -- phase + cognitive quality peers verify")

    # =====================================================================
    # Group 7: Pattern / Deep Memory Pipeline (Layers 22-23)
    #
    # Thalamic integration: pattern_bus -> pattern_cortex -> consolidator
    # -> memory_bridge -> agents. All DIRECT_REFERENCE injections created
    # in Layers 22-23 (after ConnectionRegistry seal in Layer 18).
    # =====================================================================
    dr = ConnectionType.DIRECT_REFERENCE

    _reg("memory_bridge", "memory", dr,
         witnesses=["knowledge_base", "pattern_distiller"],
         criticality=ConnectionCriticality.IMPORTANT,
         description="Deep memory bridge to agent memory -- knowledge + distillation peers verify")

    _reg("pattern_bus", "pattern_cortex", dr,
         witnesses=["attentional_gate", "pattern_consolidator"],
         criticality=ConnectionCriticality.IMPORTANT,
         description="Thalamic relay to cortical integration -- attention + consolidation peers verify")

    _reg("pattern_cortex", "pattern_consolidator", dr,
         witnesses=["memory_bridge", "circadian"],
         criticality=ConnectionCriticality.IMPORTANT,
         description="Cortical trends to ancestral memory -- deep storage + sleep-phase peers verify")

    _reg("pattern_consolidator", "memory_bridge", dr,
         witnesses=["pattern_cortex", "pattern_distiller"],
         criticality=ConnectionCriticality.IMPORTANT,
         description="Consolidated patterns to deep storage -- cortex source + distillation peers verify")

    # =====================================================================
    # Group 8: Biological System Step Hooks (Layers 26-28)
    #
    # Each biological system ticks via model.add_step_hook(). Witnesses
    # are fractal subsystem peers -- each triad member witnesses the other
    # two, embodying Law 4 (fractal self-similarity) in witnessing.
    # =====================================================================
    sh = ConnectionType.STEP_HOOK

    # Digestion subsystem triad (organs.py fractal grouping)
    _reg("digestive_system", "model", sh,
         witnesses=["renal_filter", "microbiome"],
         description="Digestion ticks every step -- digestion subsystem peers witness")
    _reg("renal_filter", "model", sh,
         witnesses=["digestive_system", "microbiome"],
         description="Renal filtering ticks every step -- digestion subsystem peers witness")
    _reg("microbiome", "model", sh,
         witnesses=["digestive_system", "renal_filter"],
         description="Microbiome ticks every step -- digestion subsystem peers witness")

    # Circulation subsystem triad
    _reg("circulatory_system", "model", sh,
         witnesses=["respiratory_system", "energy_reserve"],
         description="Circulation ticks every step -- circulation subsystem peers witness")
    _reg("respiratory_system", "model", sh,
         witnesses=["circulatory_system", "energy_reserve"],
         description="Respiration ticks every step -- circulation subsystem peers witness")
    _reg("energy_reserve", "model", sh,
         witnesses=["circulatory_system", "respiratory_system"],
         description="Energy tracking ticks every step -- circulation subsystem peers witness")

    # Regulation subsystem triad
    _reg("homeostasis", "model", sh,
         witnesses=["thermoregulation", "vestibular_system"],
         description="Homeostasis ticks every step -- regulation subsystem peers witness")
    _reg("thermoregulation", "model", sh,
         witnesses=["homeostasis", "vestibular_system"],
         description="Thermoregulation ticks every step -- regulation subsystem peers witness")
    _reg("vestibular_system", "model", sh,
         witnesses=["homeostasis", "thermoregulation"],
         description="Balance system ticks every step -- regulation subsystem peers witness")

    # Social cognition subsystem triad
    _reg("emotional_system", "model", sh,
         witnesses=["theory_of_mind", "metacognition"],
         description="Emotional system ticks every step -- social cognition peers witness")
    _reg("theory_of_mind", "model", sh,
         witnesses=["emotional_system", "metacognition"],
         description="Theory of mind ticks every step -- social cognition peers witness")
    _reg("metacognition", "model", sh,
         witnesses=["emotional_system", "theory_of_mind"],
         description="Metacognition ticks every step -- social cognition peers witness")

    # Maintenance subsystem triad
    _reg("lymphatic_system", "model", sh,
         witnesses=["senescence", "boundary_membrane"],
         description="Lymphatic system ticks every step -- maintenance subsystem peers witness")
    _reg("senescence", "model", sh,
         witnesses=["lymphatic_system", "boundary_membrane"],
         description="Senescence ticks every step -- maintenance subsystem peers witness")
    _reg("boundary_membrane", "model", sh,
         witnesses=["lymphatic_system", "senescence"],
         description="Boundary membrane ticks every step -- maintenance subsystem peers witness")

    # Sensory systems (consensus subsystem)
    _reg("nociception", "model", sh,
         witnesses=["proprioception", "somatic_map"],
         description="Pain system ticks every step -- sensory + body awareness peers witness")
    _reg("proprioception", "model", sh,
         witnesses=["nociception", "vestibular_system"],
         description="Body sense ticks every step -- spatial awareness peers witness")

    # Reproductive (growth subsystem)
    _reg("reproductive_system", "model", sh,
         witnesses=["morph_coordinator", "stem_cell_registry"],
         description="Reproductive system ticks every step -- lifecycle peers witness")

    # =====================================================================
    # Group 9: GNN Message Handlers (Layer 20)
    #
    # Per-agent message types routed through GNN communicator.
    # System-level registrations represent the routing infrastructure.
    # =====================================================================
    cb = ConnectionType.CALLBACK_REGISTRATION

    _reg("gnn_communicator", "knowledge_base", cb,
         witnesses=["metacognition", "holon_registry"],
         description="GNN knowledge sharing -- cognitive quality + peer hierarchy peers verify")

    _reg("gnn_communicator", "substrate", cb,
         witnesses=["morph_coordinator", "predictive_field"],
         description="GNN collaboration requests -- team + topology peers verify")

    _reg("gnn_communicator", "imitation", cb,
         witnesses=["metacognition", "haven"],
         description="GNN state updates feed imitation learning -- quality + safety peers verify")

    _reg("gnn_communicator", "quorum_space", cb,
         witnesses=["enforcer", "auditor"],
         criticality=ConnectionCriticality.IMPORTANT,
         description="GNN votes feed quorum consensus -- Law 7 enforcement + audit peers verify")

    # Cadenced RoutingOptimizer step hook (Layer 14b, Fibonacci 21)
    sh_gnn = ConnectionType.STEP_HOOK
    _reg("gnn_communicator", "model", sh_gnn,
         witnesses=["substrate", "metacognition"],
         description="GNN routing optimizer updates edge weights every 21 steps -- topology + cognitive quality peers verify")

    # =====================================================================
    # Group 10: Auto-Redifferentiation Trigger (Layer 29a2)
    #
    # RedifferentiationMonitor reads agent health + role distribution from
    # StemCellRegistry and triggers redifferentiate(). Lifecycle peers witness.
    # =====================================================================
    sh = ConnectionType.STEP_HOOK

    _reg("rediff_monitor", "stem_cell_registry", sh,
         witnesses=["somatic_map", "organism_state"],
         description="RedifferentiationMonitor cadenced check -- body awareness + organism state witness")

    # =====================================================================
    # Group 11: Mitosis Monitor (Layer 29a3 -- Autopoietic Production)
    #
    # MitosisMonitor reads agent health from StemCellRegistry, creates
    # new agents through the model, and registers them in HolonRegistry.
    # Lifecycle peers witness the production event.
    # =====================================================================

    _reg("mitosis_monitor", "stem_cell_registry", sh,
         witnesses=["reproductive_system", "organism_state"],
         description="MitosisMonitor cadenced check -- reproductive + organism state witness")
    _reg("mitosis_monitor", "holon_registry", ConnectionType.DIRECT_REFERENCE,
         witnesses=["reproductive_system", "somatic_map"],
         description="MitosisMonitor registers child holons -- lifecycle peers witness")
    _reg("mitosis_monitor", "event_bus", eb,
         channel="stem_cell.mitosis",
         witnesses=["reproductive_system", "morph_coordinator"],
         description="Mitosis event -- lifecycle peers witness production")

    # =====================================================================
    # Group 12: Previously Unregistered Channels
    #
    # Channels confirmed as published in production code but missing
    # from the registry. Added to eliminate advisory warnings.
    # =====================================================================
    eb = ConnectionType.EVENTBUS_PUBSUB

    # Agent lifecycle broadcast (lifecycle_communication.py)
    _reg("gnn_communicator", "event_bus", eb,
         channel="agent.broadcast",
         witnesses=["pattern_bus", "metacognition"],
         description="Agent state broadcast -- pattern processing + cognitive peers witness")

    # Bootstrap audit completion (audit.py)
    _reg("auditor", "event_bus", eb,
         channel="bootstrap.audit_complete",
         witnesses=["enforcer", "watchdog"],
         description="Bootstrap audit complete -- enforcement peers witness")

    # Inhibition signal (inhibition_system.py -- basal ganglia Go/NoGo)
    _reg("inhibition_system", "event_bus", eb,
         channel="coordination.inhibit_signal",
         witnesses=["emotional_system", "endocrine"],
         description="Action inhibition -- emotion + hormonal peers witness Go/NoGo")

    # Satiation signal (organs.py -- leptin -> appetite suppression)
    _reg("energy_reserve", "event_bus", eb,
         channel="coordination.satiation_signal",
         witnesses=["digestive_system", "endocrine"],
         description="Satiation feedback -- metabolic + hormonal peers witness")

    # Pattern consolidation (pattern_consolidator.py)
    _reg("pattern_consolidator", "event_bus", eb,
         channel="pattern.consolidation",
         witnesses=["pattern_cortex", "memory_bridge"],
         description="Pattern consolidated -- cortical + deep memory peers witness")

    # Fractal generator lifecycle events (fractal_generator.py)
    _reg("fractal_generator", "event_bus", eb,
         channel="fractal.triad_created",
         witnesses=["holon_registry", "somatic_map"],
         description="Fractal triad creation -- structural peers witness")
    _reg("fractal_generator", "event_bus", eb,
         channel="fractal.organized",
         witnesses=["holon_registry", "somatic_map"],
         description="Fractal organization complete -- structural peers witness")
    _reg("fractal_generator", "event_bus", eb,
         channel="fractal.act",
         witnesses=["holon_registry", "connection_registry"],
         description="Fractal act broadcast -- structural peers witness")

    # =====================================================================
    # Group 13: Autopoietic Closure (Layer 29a4)
    #
    # ClosureCoordinator publishes closure reports at 3 scales.
    # Source system is "closure_coordinator" (extracted by EventBus).
    # =====================================================================
    _reg("closure_coordinator", "event_bus", eb,
         channel="closure.subsystem",
         witnesses=["holon_registry", "somatic_map"],
         description="Subsystem closure report -- structural peers witness health")
    _reg("closure_coordinator", "event_bus", eb,
         channel="closure.organ",
         witnesses=["holon_registry", "somatic_map"],
         description="Organ closure report -- structural peers witness health")
    _reg("closure_coordinator", "event_bus", eb,
         channel="closure.organism",
         witnesses=["holon_registry", "somatic_map"],
         description="Organism closure report -- structural peers witness health")

    # =====================================================================
    # Group 14: Emergent Cross-System Circuits (mae-core 2026-02-25b)
    #
    # Metacognition-driven adaptive behavior + GNN→FRL trust bridge.
    # =====================================================================
    _reg("metacognition", "frl_engine", eb,
         channel="cognition.metacognition_update",
         witnesses=["organism_state", "vdn_engine"],
         description="Metacognition drives FRL sharing cadence -- share more when struggling")
    _reg("metacognition", "generative_replay", eb,
         channel="cognition.metacognition_update",
         witnesses=["world_model", "memory_coordinator"],
         description="Metacognition drives dream replay intensity -- dream more when struggling")
    _reg("gnn_communicator", "frl_engine", eb,
         channel="gnn_communicator.step_hook",
         witnesses=["metacognition", "substrate"],
         description="GNN edge weights feed FRL peer trust -- communication quality informs policy trust")

    logger.info(
        "ConnectionRegistry: %d connections registered "
        "(EventBus=%d, Direct=%d, Callback=%d, StepHook=%d)",
        counts["total"],
        counts.get("eventbus_pubsub", 0),
        counts.get("direct_reference", 0),
        counts.get("callback_registration", 0),
        counts.get("step_hook", 0),
    )

    return counts
