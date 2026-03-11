"""Connection Registrations — Group 3: Cognition Results.

Cognitive systems are each other's peers. Metacognition witnesses
reasoning peers; causal_engine witnesses prediction peers;
theory_of_mind witnesses social cognition peers.

Extracted from connection_registrations_bio.py for single-responsibility.
"""

from __future__ import annotations

from typing import Any, Callable

from mae_core.backbone.connection_registry import (
    ConnectionCriticality,
    ConnectionRegistry,
    ConnectionType,
)


def register_cognition_connections(
    registry: ConnectionRegistry,
    systems: dict[str, Any],
    _reg: Callable,
) -> None:
    """Register Group 3: Cognition Results connections.

    Args:
        registry: ConnectionRegistry instance.
        systems: System dict (unused here, kept for uniform signature).
        _reg: Inner registration helper from register_all_connections.
    """
    eb = ConnectionType.EVENTBUS_PUBSUB

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
