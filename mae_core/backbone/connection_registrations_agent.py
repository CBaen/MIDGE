"""Connection Registrations — Agent Lifecycle groups (Groups 4-5).

Covers:
  - Group 4: Agent Lifecycle (Stem Cell + Octopus)
  - Group 5: Defense / Healing / Remaining

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


def register_agent_connections(
    registry: ConnectionRegistry,
    systems: dict[str, Any],
    _reg: Callable,
) -> None:
    """Register Groups 4-5: agent lifecycle + defense/healing.

    Args:
        registry: ConnectionRegistry instance.
        systems: System dict (unused here, kept for uniform signature).
        _reg: Inner registration helper from register_all_connections.
    """
    eb = ConnectionType.EVENTBUS_PUBSUB

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
    _reg("stem_cell_registry", "event_bus", eb,
         channel="stem_cell.auto_redifferentiated",
         witnesses=["reproductive_system", "morph_coordinator"],
         description="Automatic role change -- lifecycle peers witness")

    # Genome self-modification events (genome_reader.py, genome_sandbox.py)
    _reg("genome_reader", "event_bus", eb,
         channel="genome.snapshot_taken",
         witnesses=["stem_cell_registry", "enforcer"],
         description="Genome snapshot -- identity + enforcement peers witness")
    _reg("genome_sandbox", "event_bus", eb,
         channel="genome.sandbox_result",
         witnesses=["stem_cell_registry", "enforcer"],
         description="Genome sandbox test result -- identity + enforcement peers witness")

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
