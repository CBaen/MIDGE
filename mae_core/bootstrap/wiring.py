"""Bootstrap Layers 14-21: Step hooks, EventBus wiring, registrations.

Wires all cross-system connections, endocrine consumers, somatic map,
connection registry, bidirectional awareness, fractal generator,
and stem cell registry.
"""

from __future__ import annotations

import json
import logging
import time
from types import SimpleNamespace

import numpy as np

logger = logging.getLogger("mae.bootstrap")


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


def bootstrap_wiring(ctx: SimpleNamespace) -> None:
    """Wire all cross-system connections and registrations (Layers 14-21)."""
    from mae_core.coordination.endocrine_system import HormoneType
    from mae_core.emergent.auto_healer import FailureReport, FailureType
    from mae_core.backbone.holon_protocol import AwarenessPulse
    from mae_core.backbone.connection_registry import ConnectionRegistry, ConnectionType
    from mae_core.backbone.connection_registrations import register_all_connections

    # =================================================================
    # Layer 14: Step hooks for systems that need periodic ticking
    # =================================================================
    ctx.model.add_step_hook(ctx.predictive_field.step if hasattr(ctx.predictive_field, "step") else lambda: None)
    ctx.model.add_step_hook(ctx.auto_healer.step if hasattr(ctx.auto_healer, "step") else lambda: None)
    ctx.model.add_step_hook(ctx.capability_discovery.step if hasattr(ctx.capability_discovery, "step") else lambda: None)
    ctx.model.add_step_hook(ctx.physarum.step)
    ctx.model.add_step_hook(ctx.pearl_defense.step)
    logger.info("Layer 14 - Step hooks registered for PredictiveField, AutoHealer, CapabilityDiscovery, Physarum, PearlDefense")

    # =================================================================
    # Layer 15: EventBus cross-wiring (orphan channels get subscribers)
    # =================================================================

    # Memory -> Morphogenesis: capacity warnings trigger growth
    def _on_capacity_warning(channel: str, serialized: str) -> None:
        msg = json.loads(serialized) if isinstance(serialized, str) else serialized
        logger.warning(
            "Memory capacity warning (agent %s, %.0f%% full) - triggering growth signal",
            msg.get("agent_id", "?"),
            msg.get("utilization", 0) * 100,
        )
        ctx.bus.publish("morphogenesis.spawn_request", {
            "trigger": "memory_capacity",
            "agent_id": msg.get("agent_id"),
            "urgency": msg.get("utilization", 0.9),
        })

    ctx.bus.register_callback("memory.capacity_warning", _on_capacity_warning)

    # Memory consolidation -> Endocrine: melatonin boost during consolidation
    def _on_consolidation_started(channel: str, serialized: str) -> None:
        msg = json.loads(serialized) if isinstance(serialized, str) else serialized
        ctx.endocrine.release_hormone(HormoneType.MELATONIN, 0.3, "memory_consolidation")

    ctx.bus.register_callback("memory.consolidation_started", _on_consolidation_started)

    # Healing -> Endocrine: cortisol during healing
    def _on_healing_phase_changed(channel: str, serialized: str) -> None:
        msg = json.loads(serialized) if isinstance(serialized, str) else serialized
        if msg.get("phase") in ("ISOLATE", "ASSESS"):
            ctx.endocrine.release_hormone(HormoneType.CORTISOL, 0.2, "healing_response")

    ctx.bus.register_callback("healing.phase_changed", _on_healing_phase_changed)

    # Defense -> SomaticMap: threat detection updates body awareness
    def _on_defense_activated(channel: str, serialized: str) -> None:
        msg = json.loads(serialized) if isinstance(serialized, str) else serialized
        if hasattr(ctx.somatic_map, "heartbeat"):
            ctx.somatic_map.heartbeat("threat_detector")

    ctx.bus.register_callback("defense.activated", _on_defense_activated)

    # Improvement -> Morphogenesis: new capabilities may need new organs
    def _on_capability_found(channel: str, serialized: str) -> None:
        msg = json.loads(serialized) if isinstance(serialized, str) else serialized
        if hasattr(ctx.morph_coordinator, "handle_novel_problem"):
            logger.info("Capability found: %s - checking morphogenesis need", msg.get("name", "?"))

    ctx.bus.register_callback("improvement.capability_found", _on_capability_found)

    # Morphogenesis -> SomaticMap: new teams update body map
    def _on_team_created(channel: str, serialized: str) -> None:
        msg = json.loads(serialized) if isinstance(serialized, str) else serialized
        if hasattr(ctx.somatic_map, "heartbeat"):
            ctx.somatic_map.heartbeat("morphogenesis")

    ctx.bus.register_callback("morphogenesis.team_created", _on_team_created)

    # FRL -> Imitation: policy updates are observable expert behavior
    def _on_policy_update(channel: str, serialized: str) -> None:
        msg = json.loads(serialized) if isinstance(serialized, str) else serialized
        agent_id = msg.get("agent_id", "unknown")
        policy = msg.get("policy", {})
        reward = msg.get("reward", 0.0)
        ctx.imitation.observe_behavior(
            actor_id=agent_id,
            action=policy,
            context={"trigger": "frl_policy_update", "step": int(ctx.model.time)},
            outcome=reward,
        )

    ctx.bus.register_callback("frl.policy_update", _on_policy_update)

    # FIX 2.2: ValidatedImagination -> WorldModel training feedback
    def _on_imagination_validated(channel: str, serialized: str) -> None:
        msg = json.loads(serialized) if isinstance(serialized, str) else serialized
        try:
            was_accurate = msg.get("was_accurate", False)
            actual_reward = msg.get("actual_reward")

            raw_state = msg.get("state")
            raw_action = msg.get("action")
            raw_actual_state = msg.get("actual_state")

            sdim = ctx.shared_world_model._config.state_dim
            adim = ctx.shared_world_model._config.action_dim

            if raw_state is not None:
                state = np.asarray(raw_state, dtype=np.float32).flatten()[:sdim]
                if len(state) < sdim:
                    state = np.pad(state, (0, sdim - len(state)))
            else:
                state = np.zeros(sdim, dtype=np.float32)

            if raw_action is not None:
                action = np.asarray(raw_action, dtype=np.float32).flatten()[:adim]
                if len(action) < adim:
                    action = np.pad(action, (0, adim - len(action)))
            else:
                action = np.zeros(adim, dtype=np.float32)

            if raw_actual_state is not None:
                next_state = np.asarray(raw_actual_state, dtype=np.float32).flatten()[:sdim]
                if len(next_state) < sdim:
                    next_state = np.pad(next_state, (0, sdim - len(next_state)))
            else:
                reward_error = msg.get("reward_error", 0.0)
                correction = -reward_error if not was_accurate else reward_error * 0.1
                next_state = state + correction

            reward = actual_reward if actual_reward is not None else (1.0 if was_accurate else 0.0)

            ctx.shared_world_model.train_step(
                states=np.array([state]),
                actions=np.array([action]),
                next_states=np.array([next_state]),
                rewards=np.array([reward]),
            )
        except Exception:
            logger.debug("Failed to train WorldModel from imagination validation", exc_info=True)

    ctx.bus.register_callback("cognition.imagination_validated", _on_imagination_validated)

    # FIX 2.4: AwarenessPulse anomalies -> AutoHealer assessment
    def _on_holon_anomaly(channel: str, serialized: str) -> None:
        msg = json.loads(serialized) if isinstance(serialized, str) else serialized
        try:
            anomalies = msg.get("anomalies", [])
            for anomaly in anomalies:
                anomaly_type = anomaly.get("type", "unknown")
                affected = []
                if anomaly_type == "orphaned_systems":
                    affected = anomaly.get("holon_ids", [])
                elif anomaly_type == "health_gradient":
                    affected = [issue.get("holon_id", "unknown")
                                for issue in anomaly.get("issues", [])]
                if affected:
                    failure = FailureReport(
                        failure_id=f"holon-anomaly-{anomaly_type}-{int(time.time())}",
                        failure_type=FailureType.PERFORMANCE_DEGRADATION,
                        affected_agents=affected,
                        severity=0.5,
                        metadata={"source": "awareness_pulse", "anomaly_type": anomaly_type},
                    )
                    ctx.auto_healer.report_failure(failure)
        except Exception:
            logger.debug("Failed to route holon anomaly to AutoHealer", exc_info=True)

    ctx.bus.register_callback("holon.anomaly_detected", _on_holon_anomaly)

    # FIX 2.8: TriadicVerifier low compliance -> AutoHealer assessment
    def _on_triadic_low_compliance(channel: str, serialized: str) -> None:
        msg = json.loads(serialized) if isinstance(serialized, str) else serialized
        try:
            overall_pct = msg.get("overall_pct", 100.0)
            if overall_pct < 80.0:
                severity = max(0.3, min(1.0, (100.0 - overall_pct) / 100.0))
                failure = FailureReport(
                    failure_id=f"triadic-low-compliance-{int(time.time())}",
                    failure_type=FailureType.PERFORMANCE_DEGRADATION,
                    affected_agents=[],
                    severity=severity,
                    metadata={
                        "source": "triadic_verifier",
                        "overall_pct": overall_pct,
                        "laman_pct": msg.get("laman_pct", 0),
                        "peirce_pct": msg.get("peirce_pct", 0),
                        "hegel_pct": msg.get("hegel_pct", 0),
                        "simmel_pct": msg.get("simmel_pct", 0),
                    },
                )
                ctx.auto_healer.report_failure(failure)
        except Exception:
            logger.debug("Failed to route triadic verification to AutoHealer", exc_info=True)

    ctx.bus.register_callback("triadic.verification", _on_triadic_low_compliance)

    # FIX 2.5: PatternAdvisory -> EndocrineSystem hormonal response
    def _on_pattern_advisory(channel: str, serialized: str) -> None:
        msg = json.loads(serialized) if isinstance(serialized, str) else serialized
        try:
            threat_level = msg.get("threat_level", 0.0)
            opportunity_level = msg.get("opportunity_level", 0.0)
            if threat_level > 0.3:
                ctx.endocrine.release_hormone(
                    HormoneType.ADRENALINE,
                    min(0.5, threat_level * 0.5),
                    "pattern_advisory_threat",
                )
            if opportunity_level > 0.3:
                ctx.endocrine.release_hormone(
                    HormoneType.DOPAMINE,
                    min(0.4, opportunity_level * 0.4),
                    "pattern_advisory_opportunity",
                )
        except Exception:
            logger.debug("Failed to route pattern advisory to EndocrineSystem", exc_info=True)

    ctx.bus.register_callback("pattern.advisory", _on_pattern_advisory)

    # FIX-1: Subscribe to PREDICTION_ERROR (core FEP signal was orphaned)
    def _on_prediction_error(channel: str, serialized: str) -> None:
        msg = json.loads(serialized) if isinstance(serialized, str) else serialized
        payload = msg.get("payload", msg)
        error = payload.get("prediction_error", 0.0)
        agent_id = payload.get("agent_id", "")

        # High prediction error triggers healing assessment
        if error > 0.7 and hasattr(ctx.auto_healer, "report_anomaly"):
            try:
                ctx.auto_healer.report_anomaly(
                    source=f"agent:{agent_id}",
                    severity=min(1.0, error),
                    description=f"High prediction error {error:.3f} at step {payload.get('step', '?')}",
                )
            except Exception:
                logger.debug("prediction_error -> auto_healer failed", exc_info=True)

        # Modulate learning: boost FRL learning rate for high-error agents
        agent_uid = None
        for a in ctx.agents:
            if str(a.unique_id) == agent_id:
                agent_uid = a.unique_id
                break
        if agent_uid is not None:
            frl = ctx.frl_engines.get(agent_uid)
            if frl is not None and error > 0.3:
                try:
                    # Boost learning rate proportional to prediction error
                    if hasattr(frl, "learning_rate"):
                        frl.learning_rate = min(0.1, frl.learning_rate * (1 + error))
                except Exception:
                    pass

    ctx.bus.register_callback("signal.PREDICTION_ERROR", _on_prediction_error)

    # IntegrationMeter blanket effectiveness -> BoundaryMembrane permeability
    # Weak blankets tighten the membrane; strong blankets relax it.
    boundary = getattr(ctx, "boundary_membrane", None)
    if boundary is not None:
        ctx.bus.register_callback(
            "integration.phi_measurement", boundary._on_blanket_report
        )

    logger.info("Layer 15 - EventBus cross-wiring complete")

    # =================================================================
    # Layer 15b: HAVEN risk validators → TriadEnforcer (Law 7 compliance)
    # =================================================================
    # Register HAVEN's 5 validators on key processes for immune-system defense
    def _register_haven_validators() -> None:
        from mae_core.backbone.triad_enforcer import ValidatorType

        # Decision making: validate agent decisions against risk assessment
        ctx.enforcer.add_validator(
            "decision_making",
            "haven_decision",
            ValidatorType.BEHAVIORAL,
            lambda ctx_dict: ctx.haven.validate_decision(ctx_dict),
            "HAVEN: Agent decision risk assessment",
        )

        # Learning policy: validate policy learning during contagion detection
        ctx.enforcer.add_validator(
            "learning_policy",
            "haven_contagion",
            ValidatorType.BEHAVIORAL,
            lambda ctx_dict: ctx.haven.validate_modification(ctx_dict),
            "HAVEN: Policy contagion detection",
        )

        # Healing: validate recovery strategy matches risk level
        ctx.enforcer.add_validator(
            "healing",
            "haven_healing",
            ValidatorType.BEHAVIORAL,
            lambda ctx_dict: ctx.haven.validate_healing(ctx_dict),
            "HAVEN: Healing strategy risk alignment",
        )

        # Self modification: validate agent self-modification against contagion
        ctx.enforcer.add_validator(
            "self_modification",
            "haven_modification",
            ValidatorType.BEHAVIORAL,
            lambda ctx_dict: ctx.haven.validate_modification(ctx_dict),
            "HAVEN: Self-modification contagion check",
        )

        # Resource allocation: validate resource requests from high-risk agents
        ctx.enforcer.add_validator(
            "resource_allocation",
            "haven_resources",
            ValidatorType.BEHAVIORAL,
            lambda ctx_dict: ctx.haven.validate_decision(ctx_dict),
            "HAVEN: Resource allocation risk check",
        )

        # Threat detection: validate threat responses against HAVEN's immune assessment
        ctx.enforcer.add_validator(
            "threat_detection",
            "haven_threat_validate",
            ValidatorType.BEHAVIORAL,
            lambda ctx_dict: ctx.haven.validate_threat(ctx_dict),
            "HAVEN: Threat response validation",
        )

    _register_haven_validators()
    logger.info(
        "Layer 15b - HAVEN: 6 validators registered on TriadEnforcer "
        "(immune-system defense)"
    )

    # =================================================================
    # Layer 16: Endocrine hormone consumers (hormones modulate behavior)
    # =================================================================
    ctx.endocrine.register_threat_detector(ctx.threat_detector)
    ctx.endocrine.register_auto_healer(ctx.auto_healer)
    ctx.endocrine.register_curiosity_drive(ctx.curiosity)

    # Per-agent hormone consumers (each agent's decision router + memory)
    for agent_uid, agent_sys in ctx.per_agent_systems.items():
        dr = agent_sys.get("decision_router")
        mc = agent_sys.get("memory_coordinator")
        qs = agent_sys.get("quorum_sensor")
        if dr is not None:
            ctx.endocrine.register_decision_router(dr)
        if mc is not None and hasattr(mc, "_consolidator"):
            ctx.endocrine.register_memory_consolidator(mc._consolidator)
        if qs is not None:
            ctx.endocrine.register_quorum_sensor(qs)

    # FIX 2.6: EndocrineSystem -> SignalPriorityResolver gain modulation
    # NOTE: pattern_bus may not exist yet (created in Layer 23). We store a
    # reference holder that gets populated later. The callback captures ctx
    # which will have ctx.pattern_bus set by bootstrap_patterns.
    def _on_hormone_state_update(channel: str, serialized: str) -> None:
        msg = json.loads(serialized) if isinstance(serialized, str) else serialized
        try:
            adrenaline = msg.get("adrenaline", 0.1)
            dopamine = msg.get("dopamine", 0.3)
            melatonin = msg.get("melatonin", 0.2)
            # Endocrine gain modulation of PatternBus signal processing
            pattern_bus = getattr(ctx, "pattern_bus", None)
            if pattern_bus is not None:
                pattern_bus.set_hormone_levels(msg)
            for agent in ctx.agents:
                resolver = getattr(agent, "_signal_resolver", None)
                if resolver is None:
                    continue
                # Adrenaline: preserve DANGER default (1.0) as floor, only boost
                danger_default = resolver._urgency_map.get("DANGER", 1.0)
                resolver._urgency_map["DANGER"] = max(danger_default, min(1.0, 0.8 + adrenaline * 0.2))
                # Dopamine boosts OPPORTUNITY urgency
                resolver._urgency_map["OPPORTUNITY"] = min(1.0, 0.4 + dopamine * 0.3)
                # Melatonin reduces budget (fewer signals processed when resting)
                if melatonin > 0.5:
                    resolver._config.budget_per_step = max(3, int(10 * (1.0 - melatonin * 0.5)))
                else:
                    resolver._config.budget_per_step = 10
        except Exception:
            logger.debug("Failed to modulate SignalPriorityResolver from hormones", exc_info=True)

    ctx.bus.register_callback("endocrine.state_update", _on_hormone_state_update)

    logger.info("Layer 16 - Endocrine: hormone consumers registered (shared + per-agent)")

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
        "decision_router", "defense", "frl", "healing",
        "improvement", "memory", "morphogenesis", "triad_audit",
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
    logger.info("Layer 17 - SomaticMap: %d systems registered for body awareness", 41)

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
