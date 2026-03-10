"""Connection Registrations — Advanced/Autopoietic groups (Groups 10-14).

Covers:
  - Group 10: Auto-Redifferentiation Trigger (Layer 29a2)
  - Group 11: Mitosis Monitor (Layer 29a3)
  - Group 12: Previously Unregistered Channels
  - Group 13: Autopoietic Closure (Layer 29a4)
  - Group 14: Emergent Cross-System Circuits (mae-core 2026-02-25b)

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


def register_advanced_connections(
    registry: ConnectionRegistry,
    systems: dict[str, Any],
    _reg: Callable,
) -> None:
    """Register Groups 10-14: autopoietic, lifecycle monitors, unregistered channels.

    Args:
        registry: ConnectionRegistry instance.
        systems: System dict (unused here, kept for uniform signature).
        _reg: Inner registration helper from register_all_connections.
    """
    eb = ConnectionType.EVENTBUS_PUBSUB
    sh = ConnectionType.STEP_HOOK

    # =====================================================================
    # Group 10: Auto-Redifferentiation Trigger (Layer 29a2)
    #
    # RedifferentiationMonitor reads agent health + role distribution from
    # StemCellRegistry and triggers redifferentiate(). Lifecycle peers witness.
    # =====================================================================

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

    # Signal bus collaboration request (lifecycle_decision.py _act_communicate)
    _reg("signal_bus", "event_bus", eb,
         channel="signal.COLLABORATION_REQUEST",
         witnesses=["gnn_communicator", "pattern_bus"],
         description="Collaboration signal -- communication + pattern peers witness")

    # Agent shared insight (lifecycle_decision.py _act_communicate)
    _reg("gnn_communicator", "event_bus", eb,
         channel="agent.shared",
         witnesses=["pattern_bus", "metacognition"],
         description="Agent insight sharing -- pattern + cognitive peers witness")

    # Topology analysis report (topology_analyzer.py)
    _reg("topology_analyzer", "event_bus", eb,
         channel="topology.analysis",
         witnesses=["somatic_map", "connection_registry"],
         description="Topology analysis report -- body map + connection registry peers witness")

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
    # Metacognition-driven adaptive behavior + GNN->FRL trust bridge.
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
