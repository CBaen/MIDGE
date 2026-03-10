"""Bootstrap Layers 17-21: SomaticMap, ConnectionRegistry, Bidirectional Awareness,
Fractal Generator, and Stem Cell Registry.

Layer 17: Register all systems with SomaticMap and HolonRegistry.
Layer 18: ConnectionRegistry creation, all connections registered, WitnessNotifier.
Layer 19: Bidirectional Awareness — holon proxies injected, AwarenessPulse created.
Layer 20: Fractal Generator — explicit recursive K3 structure, agent triads, GNN handlers.
Layer 21: Stem Cell Registry — agent genome + epigenome tracking.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

logger = logging.getLogger("midge.bootstrap")


def _register_somatic_systems(somatic_map, systems: dict) -> None:
    """Register all systems with SomaticMap for body awareness.

    Uses register_system() if available, falls back to heartbeat().
    """
    for name, system in systems.items():
        try:
            if hasattr(somatic_map, "register_system"):
                somatic_map.register_system(
                    system_id=name,
                    description=type(system).__name__,
                    depends_on=[],
                )
            elif hasattr(somatic_map, "heartbeat"):
                somatic_map.heartbeat(name)
        except Exception:
            logger.debug("Could not register %s with SomaticMap", name)


def _wire_layers_17_21(ctx: SimpleNamespace) -> None:
    """Wire SomaticMap, ConnectionRegistry, Awareness, Fractal, and Stem Cell."""
    from mae_core.backbone.holon_protocol import AwarenessPulse
    from mae_core.backbone.connection_registry import ConnectionRegistry, ConnectionType
    from mae_core.backbone.connection_registrations import register_all_connections

    # =================================================================
    # Layer 17: Register ALL systems with SomaticMap (body awareness)
    # =================================================================
    _register_somatic_systems(ctx.somatic_map, {
        "event_bus": ctx.bus, "circadian": ctx.circadian, "endocrine": ctx.endocrine,
        "enforcer": ctx.enforcer, "watchdog": ctx.watchdog, "auditor": ctx.auditor,
        "substrate": ctx.substrate, "physarum": ctx.physarum,
        "signal_bus": ctx.signal_bus, "gnn_communicator": ctx.gnn_comm,
        "stigmergy": ctx.stigmergy, "quorum_space": ctx.quorum_space,
        "predictive_field": ctx.predictive_field, "knowledge_base": ctx.knowledge_base,
        "transfer_engine": ctx.transfer_engine, "maml_learner": ctx.maml_learner,
        "curiosity": ctx.curiosity, "haven": ctx.haven, "imitation": ctx.imitation,
        "threat_detector": ctx.threat_detector,
        "input_validator": ctx.input_validator, "pearl_defense": ctx.pearl_defense,
        "shared_world_model": ctx.shared_world_model,
        "collective_dream": ctx.collective_dream, "validated_imagination": ctx.validated_imagination,
        "shared_causal_engine": ctx.shared_causal_engine, "auto_healer": ctx.auto_healer,
        "capability_discovery": ctx.capability_discovery, "morph_coordinator": ctx.morph_coordinator,
        "organ_builder": ctx.organ_builder, "temporal_memory": ctx.temporal_memory,
        "worldline_planner": ctx.worldline_planner,
        "holon_registry": ctx.holon_registry,
    })
    # Register abstract/group system names used as connection participants.
    # These are logical groupings (not individual subsystems) that appear as
    # source/target/witness in register_all_connections(). Without SomaticMap
    # entries, verify_all() marks their connections as unhealthy.
    for abstract_name in (
        "agent", "decision_router", "defense", "frl",
        "genome_reader", "genome_sandbox",
        "healing", "improvement", "memory", "morphogenesis", "triad_audit",
    ):
        ctx.somatic_map.register_system(
            system_id=abstract_name,
            description="AbstractGroup",
            depends_on=[],
        )
    # SomaticMap registers itself (used as witness in connection registrations)
    ctx.somatic_map.register_system(
        system_id="somatic_map",
        description=type(ctx.somatic_map).__name__,
        depends_on=[],
    )
    logger.info("Layer 17 - SomaticMap: %d systems registered for body awareness", len(ctx.somatic_map.get_all_systems()))

    # Register all shared systems as holons (children of Mae)
    for name in (
        "event_bus", "circadian", "endocrine", "enforcer", "watchdog", "auditor",
        "substrate", "physarum", "signal_bus", "gnn_communicator", "stigmergy",
        "quorum_space", "predictive_field", "knowledge_base", "transfer_engine",
        "maml_learner", "curiosity", "haven", "imitation", "threat_detector",
        "input_validator", "pearl_defense", "shared_world_model", "collective_dream",
        "validated_imagination", "shared_causal_engine", "auto_healer",
        "capability_discovery", "somatic_map", "morph_coordinator", "organ_builder",
        "temporal_memory", "worldline_planner",
    ):
        ctx.holon_registry.register(name, holon_type="system", parent_id="mae")
    ctx.holon_registry.register("holon_registry", holon_type="system", parent_id="mae")
    logger.info("Layer 17 - HolonRegistry: %d holons registered", len(ctx.holon_registry.get_all_ids()))

    # =================================================================
    # Layer 18: Connection Registry (triadic witnessing for all connections)
    # =================================================================
    ctx.connection_registry = ConnectionRegistry(
        event_bus=ctx.bus,
        somatic_map=ctx.somatic_map,
    )
    ctx.conn_counts = register_all_connections(ctx.connection_registry, {
        "model": ctx.model, "event_bus": ctx.bus,
        "circadian": ctx.circadian, "endocrine": ctx.endocrine,
        "enforcer": ctx.enforcer, "watchdog": ctx.watchdog, "auditor": ctx.auditor,
        "substrate": ctx.substrate, "physarum": ctx.physarum,
        "signal_bus": ctx.signal_bus, "gnn_communicator": ctx.gnn_comm,
        "stigmergy": ctx.stigmergy, "quorum_space": ctx.quorum_space,
        "predictive_field": ctx.predictive_field, "knowledge_base": ctx.knowledge_base,
        "transfer_engine": ctx.transfer_engine, "maml_learner": ctx.maml_learner,
        "curiosity": ctx.curiosity, "haven": ctx.haven, "imitation": ctx.imitation,
        "threat_detector": ctx.threat_detector,
        "input_validator": ctx.input_validator, "pearl_defense": ctx.pearl_defense,
        "shared_world_model": ctx.shared_world_model,
        "collective_dream": ctx.collective_dream, "validated_imagination": ctx.validated_imagination,
        "shared_causal_engine": ctx.shared_causal_engine, "auto_healer": ctx.auto_healer,
        "capability_discovery": ctx.capability_discovery, "somatic_map": ctx.somatic_map,
        "morph_coordinator": ctx.morph_coordinator, "organ_builder": ctx.organ_builder,
        "temporal_memory": ctx.temporal_memory, "worldline_planner": ctx.worldline_planner,
        "holon_registry": ctx.holon_registry,
    })
    ctx.watchdog.set_connection_registry(ctx.connection_registry)
    ctx.model.add_step_hook(ctx.connection_registry.step)
    ctx.holon_registry.register("connection_registry", holon_type="system", parent_id="mae")
    _register_somatic_systems(ctx.somatic_map, {"connection_registry": ctx.connection_registry})
    # Meta-healing triad: AutoHealer monitors itself, witnessed by SomaticMap (Law 6)
    ctx.auto_healer._connection_registry = ctx.connection_registry
    ctx.auto_healer.register_self_healing_triad()
    ctx.connection_registry.seal()  # End bootstrap grace period -> enforcement active
    ctx.bus.set_connection_registry(ctx.connection_registry)  # Advisory witnessing on publish()

    # Layer 18b: WitnessNotifier — operational witnessing (verification pathway)
    from mae_core.backbone.witness_notifier import WitnessNotifier
    ctx.witness_notifier = WitnessNotifier(
        event_bus=ctx.bus,
        connection_registry=ctx.connection_registry,
        holon_registry=ctx.holon_registry,
    )
    ctx.witness_notifier.activate()
    ctx.connection_registry._witness_notifier = ctx.witness_notifier
    ctx.model.add_step_hook(ctx.witness_notifier.step)
    ctx.holon_registry.register("witness_notifier", holon_type="system", parent_id="mae")
    _register_somatic_systems(ctx.somatic_map, {"witness_notifier": ctx.witness_notifier})
    # Register WitnessNotifier's own triadic connections
    ctx.connection_registry.register_connection(
        source="witness_notifier", target="event_bus",
        connection_type=ConnectionType.EVENTBUS_PUBSUB,
        channel="witness_notifier.observation",
        witnesses=["auditor", "connection_registry"],
        description="Shadow notifications for witnessed message flow",
    )
    ctx.connection_registry.register_connection(
        source="witness_notifier", target="event_bus",
        connection_type=ConnectionType.EVENTBUS_PUBSUB,
        channel="witness_notifier.health",
        witnesses=["auditor", "connection_registry"],
        description="Cadenced witnessing health summaries",
    )
    ctx.connection_registry.register_connection(
        source="witness_notifier", target="event_bus",
        connection_type=ConnectionType.EVENTBUS_PUBSUB,
        channel="witness_notifier.digest",
        witnesses=["auditor", "connection_registry"],
        description="Per-witness cadenced digest notifications",
    )
    ctx.connection_registry.register_connection(
        source="witness_notifier", target="event_bus",
        connection_type=ConnectionType.EVENTBUS_PUBSUB,
        channel="witness_notifier.verdict",
        witnesses=["auditor", "connection_registry"],
        description="Balance pathway: witness verdicts on observed connections",
    )
    logger.info(
        "Layer 18b - WitnessNotifier: %d channels subscribed, operational witnessing active",
        len(ctx.witness_notifier._subscribed_channels),
    )

    logger.info(
        "Layer 18 - ConnectionRegistry: %d connections registered (%d witnessed)",
        ctx.conn_counts["total"] + 4,  # +4 for WitnessNotifier's own connections
        ctx.conn_counts["total"] + 3 - len(ctx.connection_registry.get_bare_dyads()),
    )

    # =================================================================
    # Layer 19: Bidirectional Awareness (holon proxies + awareness pulse)
    # =================================================================
    ctx.holon_registry.set_somatic_map(ctx.somatic_map)
    ctx.holon_registry.set_connection_registry(ctx.connection_registry)

    # Inject _holon proxy into every shared system
    _shared_systems = {
        "event_bus": ctx.bus, "circadian": ctx.circadian, "endocrine": ctx.endocrine,
        "enforcer": ctx.enforcer, "watchdog": ctx.watchdog, "auditor": ctx.auditor,
        "substrate": ctx.substrate, "physarum": ctx.physarum,
        "signal_bus": ctx.signal_bus, "gnn_communicator": ctx.gnn_comm,
        "stigmergy": ctx.stigmergy, "quorum_space": ctx.quorum_space,
        "predictive_field": ctx.predictive_field, "knowledge_base": ctx.knowledge_base,
        "transfer_engine": ctx.transfer_engine, "maml_learner": ctx.maml_learner,
        "curiosity": ctx.curiosity, "haven": ctx.haven, "imitation": ctx.imitation,
        "threat_detector": ctx.threat_detector,
        "input_validator": ctx.input_validator, "pearl_defense": ctx.pearl_defense,
        "shared_world_model": ctx.shared_world_model,
        "collective_dream": ctx.collective_dream, "validated_imagination": ctx.validated_imagination,
        "shared_causal_engine": ctx.shared_causal_engine, "auto_healer": ctx.auto_healer,
        "capability_discovery": ctx.capability_discovery, "somatic_map": ctx.somatic_map,
        "morph_coordinator": ctx.morph_coordinator, "organ_builder": ctx.organ_builder,
        "temporal_memory": ctx.temporal_memory, "worldline_planner": ctx.worldline_planner,
        "holon_registry": ctx.holon_registry, "connection_registry": ctx.connection_registry,
        "witness_notifier": ctx.witness_notifier,
    }
    proxy_count = 0
    for name, system in _shared_systems.items():
        proxy = ctx.holon_registry.get_proxy(name)
        system._holon = proxy
        proxy.set_system_ref(system)  # Enable sense/decide/learn/heal/act delegation
        proxy_count += 1

    ctx.awareness_pulse = AwarenessPulse(
        registry=ctx.holon_registry,
        event_bus=ctx.bus,
        interval=25,
    )
    ctx.model.add_step_hook(ctx.awareness_pulse.step)
    ctx.holon_registry.register("awareness_pulse", holon_type="system", parent_id="mae")
    _register_somatic_systems(ctx.somatic_map, {"awareness_pulse": ctx.awareness_pulse})
    ctx.awareness_pulse._holon = ctx.holon_registry.get_proxy("awareness_pulse")
    proxy_count += 1

    logger.info(
        "Layer 19 - Bidirectional Awareness: %d holon proxies injected, AwarenessPulse active (interval=%d)",
        proxy_count, 25,
    )

    # =================================================================
    # Layer 20: Fractal Generator (explicit recursive structure)
    # =================================================================
    from mae_core.backbone.fractal_generator import FractalGenerator
    from mae_core.patterns.pattern_sharer import PatternSharer

    ctx.fractal_generator = FractalGenerator(
        holon_registry=ctx.holon_registry,
        connection_registry=ctx.connection_registry,
        event_bus=ctx.bus,
    )
    ctx.fractal_report = ctx.fractal_generator.organize()

    ctx.holon_registry.register("fractal_generator", holon_type="system", parent_id="mae")
    _register_somatic_systems(ctx.somatic_map, {"fractal_generator": ctx.fractal_generator})
    ctx.fractal_generator._holon = ctx.holon_registry.get_proxy("fractal_generator")

    # Group agents into triads (tissue-level grouping)
    _agent_ids = [str(a.unique_id) for a in ctx.agents]
    _triads_created = 0
    for i in range(0, len(_agent_ids), 3):
        chunk = _agent_ids[i:i + 3]
        triad_name = f"agent_triad_{i // 3}"
        ctx.fractal_generator.generate_triad(
            name=triad_name,
            holon_type="tissue",
            children_ids=chunk,
            parent_id="colony",
        )
        _triads_created += 1

    # Inject PatternSharer into each agent (requires triads to exist)
    def _make_pattern_handler(sharer_ref):
        """Factory to capture sharer reference -- routes GNN pattern shares to inbox."""
        def handler(message):
            content = getattr(message, "content", {})
            if isinstance(content, dict) and content.get("type") == "pattern_share":
                sharer_ref.receive_peer_signal(
                    content.get("sender", "unknown"), content,
                )
        return handler

    # FIX-7: Register handlers for COLLABORATION_REQUEST, STATE_UPDATE, VOTE
    # (GNN message types existed but had zero handlers)

    def _make_collaboration_handler(agent_ref):
        """Handle collaboration requests from peers."""
        def handler(message):
            try:
                content = getattr(message, "content", {})
                sender = getattr(message, "sender_id", "unknown")
                required_caps = content.get("required_capabilities", []) if isinstance(content, dict) else []
                agent_caps = set(getattr(agent_ref, "capabilities", []))
                can_help = bool(agent_caps) and any(cap in agent_caps for cap in required_caps)
                if can_help and hasattr(agent_ref, "send_gnn_message"):
                    agent_ref.send_gnn_message(
                        content={"response": "available", "capabilities": list(agent_caps)},
                        message_type="COLLABORATION_RESPONSE",
                        target_ids=[sender],
                        priority=0.7,
                    )
            except Exception:
                logger.debug("collaboration handler failed", exc_info=True)
        return handler

    def _make_state_update_handler(agent_ref):
        """Handle state updates from peers — feed to imitation learning."""
        def handler(message):
            try:
                content = getattr(message, "content", {})
                sender = getattr(message, "sender_id", "unknown")
                if isinstance(content, dict):
                    il = getattr(agent_ref, "_imitation_learner", None)
                    if il is not None and hasattr(il, "observe_behavior"):
                        il.observe_behavior(
                            actor_id=sender,
                            action=content.get("action"),
                            context=content.get("state", {}),
                            outcome=content.get("reward", 0.0),
                        )
            except Exception:
                logger.debug("state_update handler failed", exc_info=True)
        return handler

    def _make_vote_handler(agent_ref):
        """Handle vote messages — feed to quorum sensor."""
        def handler(message):
            try:
                content = getattr(message, "content", {})
                sender = getattr(message, "sender_id", "unknown")
                if isinstance(content, dict):
                    qs = getattr(agent_ref, "quorum_sensor", None)
                    if qs is not None and hasattr(qs, "record_vote"):
                        qs.record_vote(
                            content.get("vote_type", "unknown"),
                            content.get("value", 0.0),
                            sender,
                        )
            except Exception:
                logger.debug("vote handler failed", exc_info=True)
        return handler

    for agent in ctx.agents:
        agent_id_str = str(agent.unique_id)
        sharer = PatternSharer(
            agent_id=agent_id_str,
            holon_registry=ctx.holon_registry,
            gnn_communicator=ctx.gnn_comm,
            event_bus=ctx.bus,
        )
        agent._pattern_sharer = sharer
        ctx.per_agent_systems[agent.unique_id]["pattern_sharer"] = sharer

        # Register GNN handler so incoming KNOWLEDGE_SHARE messages
        # reach PatternSharer.receive_peer_signal()
        agent.register_gnn_message_handler(
            "KNOWLEDGE_SHARE", _make_pattern_handler(sharer),
        )
        agent.register_gnn_message_handler(
            "COLLABORATION_REQUEST", _make_collaboration_handler(agent),
        )
        agent.register_gnn_message_handler(
            "STATE_UPDATE", _make_state_update_handler(agent),
        )
        agent.register_gnn_message_handler(
            "VOTE", _make_vote_handler(agent),
        )

    logger.info(
        "Layer 20 - Fractal Generator: %d organs, %d subsystems, %d K3 connections, depth=%d, %d agent triads",
        ctx.fractal_report.organs_created,
        ctx.fractal_report.subsystems_created,
        ctx.fractal_report.connections_created,
        ctx.fractal_report.max_depth,
        _triads_created,
    )

    # =================================================================
    # Layer 21: Stem Cell Registry (agent genome + epigenome tracking)
    # =================================================================
    from mae_core.agents.stem_cell import StemCellRegistry

    ctx.stem_cell_registry = StemCellRegistry(event_bus=ctx.bus)
    for agent in ctx.agents:
        ctx.stem_cell_registry.register(agent)

    ctx.holon_registry.register("stem_cell_registry", holon_type="system", parent_id="mae")
    _register_somatic_systems(ctx.somatic_map, {"stem_cell_registry": ctx.stem_cell_registry})
    ctx.stem_cell_registry._holon = ctx.holon_registry.get_proxy("stem_cell_registry")

    logger.info(
        "Layer 21 - Stem Cell Registry: %d agents registered as STEM, genome=%d genes",
        len(ctx.agents), ctx.stem_cell_registry.get_statistics()["genome_size"],
    )
