"""Bootstrap Layers 14-16: Step hooks, EventBus cross-wiring, HAVEN, endocrine consumers.

Layer 14: Step hooks for PredictiveField, AutoHealer, CapabilityDiscovery,
          Physarum, PearlDefense, GNN RoutingOptimizer.
Layer 15: EventBus cross-wiring (memory, healing, defense, FRL, imagination,
          holon anomalies, triadic compliance, pattern advisory, prediction error).
Layer 15b: HAVEN risk validators registered on TriadEnforcer.
Layer 16: Endocrine hormone consumers (shared + per-agent).
"""

from __future__ import annotations

import json
import logging
import time
from types import SimpleNamespace

import numpy as np

logger = logging.getLogger("midge.bootstrap")


def _wire_layers_14_16(ctx: SimpleNamespace) -> None:
    """Wire step hooks, EventBus cross-connections, HAVEN, and endocrine consumers."""
    from mae_core.coordination.endocrine_system import HormoneType
    from mae_core.emergent.auto_healer import FailureReport, FailureType
    from mae_core.backbone.holon_protocol import AwarenessPulse

    # =================================================================
    # Layer 14: Step hooks for systems that need periodic ticking
    # =================================================================
    ctx.model.add_step_hook(ctx.predictive_field.step if hasattr(ctx.predictive_field, "step") else lambda: None)
    ctx.model.add_step_hook(ctx.auto_healer.step if hasattr(ctx.auto_healer, "step") else lambda: None)
    ctx.model.add_step_hook(ctx.capability_discovery.step if hasattr(ctx.capability_discovery, "step") else lambda: None)
    ctx.model.add_step_hook(ctx.physarum.step)
    ctx.model.add_step_hook(ctx.pearl_defense.step)

    # GNN RoutingOptimizer: cadenced edge weight optimization (Fibonacci 21)
    _gnn_opt_counter = [0]

    def _gnn_optimize_hook() -> None:
        _gnn_opt_counter[0] += 1
        if _gnn_opt_counter[0] % 21 == 0:
            optimizer = getattr(ctx.gnn_comm, "_optimizer", None)
            graph = getattr(ctx.gnn_comm, "_graph", None)
            if optimizer is not None and graph is not None:
                updated = optimizer.optimize_graph(graph, min_samples=5)
                if updated > 0:
                    logger.debug(
                        "GNN RoutingOptimizer: %d edges updated at step %d",
                        updated, _gnn_opt_counter[0],
                    )
                    # Circuit C: Push GNN edge weights → FRL peer trust
                    frl_engines = getattr(ctx, "frl_engines", None)
                    if frl_engines and hasattr(graph, "edges"):
                        trust_updates = 0
                        for (src, tgt), edge in graph.edges.items():
                            src_key = int(src) if isinstance(src, str) and src.isdigit() else src
                            src_frl = frl_engines.get(src_key)
                            if src_frl is not None:
                                try:
                                    src_frl.update_peer_trust(
                                        str(tgt), edge.weight, 0.0,
                                    )
                                    trust_updates += 1
                                except Exception:
                                    pass
                        if trust_updates > 0:
                            logger.debug(
                                "GNN->FRL trust: %d peer trust updates at step %d",
                                trust_updates, _gnn_opt_counter[0],
                            )

    ctx.model.add_step_hook(_gnn_optimize_hook)
    logger.info("Layer 14 - Step hooks registered for PredictiveField, AutoHealer, CapabilityDiscovery, Physarum, PearlDefense, GNN RoutingOptimizer")

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
