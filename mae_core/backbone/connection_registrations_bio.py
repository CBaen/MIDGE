"""Connection Registrations — Infrastructure + Groups 2-3.

Covers:
  - EventBus pub/sub connections (Layer 15 cross-wiring)
  - Direct reference injections (constructor params)
  - Callback registrations (Layer 16)
  - Step hooks (Layers 2, 3, 14)
  - Group 2: Backbone Self-Monitoring
  - Group 3: Cognition Results

Group 1 (Metabolic/Biological -> OrganismState) lives in
connection_registrations_metabolic.py. Extracted for single-responsibility.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from mae_core.backbone.connection_registry import (
    ConnectionCriticality,
    ConnectionRegistry,
    ConnectionType,
)
from mae_core.backbone.connection_registrations_metabolic import (
    register_metabolic_connections,
)

logger = logging.getLogger(__name__)


def register_bio_connections(
    registry: ConnectionRegistry,
    systems: dict[str, Any],
    _reg: Callable,
) -> None:
    """Register EventBus/DR/CB/SH infrastructure + Groups 1-3.

    Args:
        registry: ConnectionRegistry instance.
        systems: System dict (unused here, kept for uniform signature).
        _reg: Inner registration helper from register_all_connections.
    """
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

    # Group 1: Metabolic/Biological -> OrganismState (see connection_registrations_metabolic.py)
    register_metabolic_connections(registry, systems, _reg)

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
