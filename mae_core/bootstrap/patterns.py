"""Bootstrap Layers 22-25: Deep memory, pattern ecosystem, action environment.

Creates: deep_store, memory_bridge, pattern_distiller, narrator,
pattern_bus, pattern_cortex, pattern_consolidator, attentional_gate,
task_pool, organism_action.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

logger = logging.getLogger("mae.bootstrap")


def _register_somatic_systems(somatic_map, systems: dict) -> None:
    """Register all systems with SomaticMap for body awareness."""
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


def bootstrap_patterns(ctx: SimpleNamespace) -> None:
    """Create deep memory, pattern ecosystem, and action environment (Layers 22-25)."""
    from mae_core.memory.deep_memory import DeepMemoryStore, QdrantConfig
    from mae_core.memory.experience_narrator import ExperienceNarrator
    from mae_core.memory.memory_bridge import MemoryBridge
    from mae_core.memory.pattern_distiller import PatternDistiller
    from mae_core.environment.task_pool import TaskPool
    from mae_core.backbone.fractal_act import build_fractal_action

    # =================================================================
    # Layer 22: Deep Memory (Qdrant-backed permanent memory + bridge)
    # =================================================================
    # Graceful degradation: if Qdrant is down, Mae runs without deep memory.
    ctx.deep_store = None
    ctx.memory_bridge = None
    ctx.pattern_distiller = None
    ctx.narrator = None

    try:
        qdrant_config = QdrantConfig()  # localhost:6335 (mae-qdrant container)
        ctx.deep_store = DeepMemoryStore(config=qdrant_config)

        if ctx.deep_store.is_available():
            # Create collections if needed
            collection_results = ctx.deep_store.ensure_collections()
            collections_ok = sum(1 for v in collection_results.values() if v)

            # Narrator bridges numerical state -> searchable text
            ctx.narrator = ExperienceNarrator()

            # Bridge connects agent memory coordinators to deep store
            ctx.memory_bridge = MemoryBridge(
                deep_store=ctx.deep_store,
                narrator=ctx.narrator,
                event_bus=ctx.bus,
                haven=ctx.haven,
            )

            # PatternDistiller extracts ancestral patterns (Rule of Three)
            ctx.pattern_distiller = PatternDistiller(min_occurrences=3)

            # Inject bridge into each agent's memory coordinator
            bridge_count = 0
            for agent_uid, agent_sys in ctx.per_agent_systems.items():
                mc = agent_sys.get("memory_coordinator")
                if mc is not None:
                    mc._bridge = ctx.memory_bridge
                    bridge_count += 1

            # Store references for inspection (not in _tier2_refs — deep memory
            # persists in Qdrant, not pickle)
            ctx.model._deep_memory_refs = {
                "deep_store": ctx.deep_store,
                "memory_bridge": ctx.memory_bridge,
                "pattern_distiller": ctx.pattern_distiller,
                "narrator": ctx.narrator,
            }

            # Inject MemoryBridge into agents for lifecycle access
            # (agents created at Layer 12, bridge created here at Layer 22)
            for agent in ctx.agents:
                agent._memory_bridge = ctx.memory_bridge

            logger.info(
                "Layer 22 - Deep Memory: Qdrant connected, %d/%d collections ready, "
                "%d agents bridged",
                collections_ok, len(collection_results), bridge_count,
            )
        else:
            logger.info("Layer 22 - Deep Memory: Qdrant unavailable, running without deep memory")
    except Exception:
        logger.warning("Layer 22 - Deep Memory: failed to initialize (graceful degradation)", exc_info=True)
        ctx.deep_store = None
        ctx.memory_bridge = None

    # Register deep memory systems with holon + somatic (even if None, for awareness)
    if ctx.deep_store is not None:
        ctx.holon_registry.register("deep_memory", holon_type="system", parent_id="mae")
        ctx.holon_registry.register("memory_bridge", holon_type="system", parent_id="mae")
        ctx.holon_registry.register("pattern_distiller", holon_type="system", parent_id="mae")
        _register_somatic_systems(ctx.somatic_map, {
            "deep_memory": ctx.deep_store,
            "memory_bridge": ctx.memory_bridge,
            "pattern_distiller": ctx.pattern_distiller,
        })
        if hasattr(ctx.deep_store, "_holon"):
            ctx.deep_store._holon = ctx.holon_registry.get_proxy("deep_memory")
        if hasattr(ctx.memory_bridge, "_holon"):
            ctx.memory_bridge._holon = ctx.holon_registry.get_proxy("memory_bridge")

        # Register triadic connections for MemoryBridge recall pathways (Law 1)
        from mae_core.backbone.connection_registry import ConnectionType as _MCT
        _mb_reg = ctx.connection_registry.register_connection
        _mb_reg("memory_bridge", "deep_memory", _MCT.DIRECT_REFERENCE,
                witnesses=["haven", "event_bus"],
                description="MemoryBridge recall from Qdrant — HAVEN filters + EventBus witnessing")
        _mb_reg("memory_bridge", "haven", _MCT.DIRECT_REFERENCE,
                witnesses=["deep_memory", "event_bus"],
                description="HAVEN isolation check before cross-agent recall")
        _mb_reg("memory_bridge", "event_bus", _MCT.EVENTBUS_PUBSUB,
                channel="memory.recall_completed",
                witnesses=["deep_memory", "haven"],
                description="Recall event notification for triadic witnessing")
        _mb_reg("memory_bridge", "event_bus", _MCT.EVENTBUS_PUBSUB,
                channel="memory.recall_verified",
                witnesses=["deep_memory", "haven"],
                description="Recall verification event — witness hash presence check")

    # =================================================================
    # Layer 23: Pattern Recognition Ecosystem (thalamic integration)
    # =================================================================
    ctx.pattern_bus = None
    ctx.pattern_cortex = None
    ctx.pattern_consolidator = None
    ctx.attentional_gate = None
    try:
        from mae_core.patterns.attentional_gate import AttentionalGate
        from mae_core.patterns.pattern_bus import PatternBus
        from mae_core.patterns.pattern_cortex import PatternCortex
        from mae_core.patterns.translators.cognition import (
            CausalEngineTranslator,
            DecisionRouterTranslator,
            WorldModelTranslator,
        )
        from mae_core.patterns.translators.defense import (
            AutoHealerTranslator,
            HAVENTranslator,
            ThreatTranslator,
        )
        from mae_core.patterns.translators.emergent import CapabilityTranslator
        from mae_core.patterns.translators.learning import CuriosityTranslator
        from mae_core.patterns.translators.memory import PatternDistillerTranslator
        from mae_core.patterns.translators.opportunity import OpportunityTranslator
        from mae_core.patterns.translators.triadic import TriadicPatternTranslator

        # TRN analog: top-down attentional gating (Law 8, Property 8)
        ctx.attentional_gate = AttentionalGate()
        ctx.pattern_bus = PatternBus(event_bus=ctx.bus, attentional_gate=ctx.attentional_gate)

        # Register translators (one per source system)
        _translators = [
            WorldModelTranslator(),
            CausalEngineTranslator(),
            DecisionRouterTranslator(),
            CuriosityTranslator(),
            AutoHealerTranslator(),
            HAVENTranslator(),
            ThreatTranslator(),
            CapabilityTranslator(),
            PatternDistillerTranslator(),
            OpportunityTranslator(),
            TriadicPatternTranslator(),
        ]
        for t in _translators:
            ctx.pattern_bus.register_translator(t)

        # PatternCortex: temporal integration over 13-step window.
        # Inject MemoryBridge for ancestral recall (graceful if None).
        ctx.pattern_cortex = PatternCortex(memory_bridge=ctx.memory_bridge)

        # Step hook: process signals, then integrate in cortex
        _pattern_step_counter = [0]
        _latest_advisory: dict = {"advisory": None}
        # Store advisory ref on ctx so Layer 30 can inject into agents
        ctx._latest_advisory = _latest_advisory

        def _pattern_step_hook():
            _pattern_step_counter[0] += 1

            # Feed aggregate prediction error from agents into the gate
            if ctx.attentional_gate is not None:
                pred_errors = []
                for agent in ctx.agents:
                    pe = getattr(agent, "_prediction_error", 0.0)
                    if pe > 0:
                        pred_errors.append(pe)
                if pred_errors:
                    avg_error = sum(pred_errors) / len(pred_errors)
                    ctx.attentional_gate.set_prediction_error(avg_error)

            digest = ctx.pattern_bus.process_step(_pattern_step_counter[0])
            advisory = ctx.pattern_cortex.process_digest(digest)
            _latest_advisory["advisory"] = advisory

            # Top-down feedback: advisory updates attentional gate for next step
            if ctx.attentional_gate is not None and advisory is not None:
                ctx.attentional_gate.update_attention(advisory)

            # Publish advisory to EventBus for any interested subscriber
            if advisory is not None:
                try:
                    ctx.bus.publish("pattern.advisory", {
                        "step": advisory.step,
                        "threat_level": advisory.threat_level,
                        "opportunity_level": advisory.opportunity_level,
                        "novelty_level": advisory.novelty_level,
                        "recommended_tier": advisory.recommended_tier,
                        "active_trends": advisory.active_trends,
                        "confidence": advisory.confidence,
                    })
                except Exception:
                    logger.debug("Failed to publish pattern advisory", exc_info=True)

        ctx.model.add_step_hook(_pattern_step_hook)

        # Inject advisory mailbox into agents for _observe() access
        for agent in ctx.agents:
            agent._pattern_advisory_ref = _latest_advisory

        # Register with holon + somatic
        ctx.holon_registry.register("pattern_bus", holon_type="system", parent_id="mae")
        ctx.holon_registry.register("pattern_cortex", holon_type="system", parent_id="mae")
        ctx.holon_registry.register("attentional_gate", holon_type="system", parent_id="mae")
        _register_somatic_systems(ctx.somatic_map, {
            "pattern_bus": ctx.pattern_bus,
            "pattern_cortex": ctx.pattern_cortex,
            "attentional_gate": ctx.attentional_gate,
        })

        # PatternConsolidator: every 89 steps, distill trends into ancestral memory
        from mae_core.patterns.pattern_consolidator import (
            CONSOLIDATION_INTERVAL,
            PatternConsolidator,
        )

        ctx.pattern_consolidator = PatternConsolidator(
            pattern_cortex=ctx.pattern_cortex,
            memory_bridge=ctx.memory_bridge,
            pattern_distiller=ctx.pattern_distiller,
            event_bus=ctx.bus,
            circadian=ctx.circadian,
        )

        _consolidator_step_counter = [0]

        def _consolidator_step_hook():
            _consolidator_step_counter[0] += 1
            if _consolidator_step_counter[0] % CONSOLIDATION_INTERVAL == 0:
                ctx.pattern_consolidator.consolidate(_consolidator_step_counter[0], force=True)

        ctx.model.add_step_hook(_consolidator_step_hook)

        ctx.holon_registry.register("pattern_consolidator", holon_type="system", parent_id="mae")
        _register_somatic_systems(ctx.somatic_map, {
            "pattern_consolidator": ctx.pattern_consolidator,
        })

        logger.info(
            "Layer 23 - Pattern Ecosystem: %d translators, cortex active (bridge=%s), "
            "TRN gate active, consolidator active",
            ctx.pattern_bus.translator_count,
            "yes" if ctx.memory_bridge is not None else "no",
        )
    except Exception:
        logger.warning(
            "Layer 23 - Pattern Ecosystem: failed to initialize (graceful degradation)",
            exc_info=True,
        )
        ctx.pattern_bus = None
        ctx.pattern_cortex = None
        ctx.pattern_consolidator = None
        ctx.attentional_gate = None

    # =================================================================
    # Layer 24: Action Environment (TaskPool — Mae's body)
    # =================================================================
    ctx.task_pool = TaskPool(generation_rate=3)

    # Inject into every agent (biological: connecting motor cortex to muscles)
    for agent in ctx.agents:
        agent._task_pool = ctx.task_pool

    # Step hook: advance the pool (generate new tasks, expire old ones)
    def _task_pool_step_hook() -> None:
        ctx.task_pool.step(current_step=int(ctx.model.time))

    ctx.model.add_step_hook(_task_pool_step_hook)

    # Register with holon + somatic for observability
    ctx.holon_registry.register("task_pool", holon_type="system", parent_id="mae")
    _register_somatic_systems(ctx.somatic_map, {"task_pool": ctx.task_pool})
    ctx.task_pool._holon = ctx.holon_registry.get_proxy("task_pool")

    logger.info(
        "Layer 24 - Action Environment: TaskPool created (rate=%d), injected into %d agents",
        ctx.task_pool.generation_rate, len(ctx.agents),
    )

    # =================================================================
    # Layer 25: Fractal ACT (action at every scale — Law 3 compliance)
    # =================================================================
    # Inject system references into holon proxies so act() can
    # delegate to the system's step/process method.
    _all_system_refs = {
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
        "awareness_pulse": ctx.awareness_pulse, "fractal_generator": ctx.fractal_generator,
        "stem_cell_registry": ctx.stem_cell_registry, "task_pool": ctx.task_pool,
        # NOTE: integration_meter proxy ref is set in Layer 25b (after creation)
        # NOTE: 18 new biological systems are added to proxy refs in Layer 28
        # after they're created. See _new_systems_for_proxy in organs.py.
    }
    _proxy_refs_injected = 0
    for sys_name, sys_obj in _all_system_refs.items():
        proxy = ctx.holon_registry.get_proxy(sys_name)
        if proxy is not None:
            proxy.set_system_ref(sys_obj)
            _proxy_refs_injected += 1

    # Build fractal action hierarchy (SubsystemAction -> OrganAction -> OrganismAction)
    ctx.organism_action = build_fractal_action(
        registry=ctx.holon_registry,
        event_bus=ctx.bus,
    )

    # Wire *Action objects as system_ref on their virtual holon proxies.
    # This connects the two abstraction layers so HolonProxy.sense/decide/learn/heal
    # delegate to the *Action classes at subsystem/organ/organism level.
    for organ_name, organ_action in ctx.organism_action._organ_actions.items():
        organ_proxy = ctx.holon_registry.get_proxy(organ_name)
        if organ_proxy is not None:
            organ_proxy.set_system_ref(organ_action)
        for sub_name, sub_action in organ_action._subsystem_actions.items():
            sub_proxy = ctx.holon_registry.get_proxy(sub_name)
            if sub_proxy is not None:
                sub_proxy.set_system_ref(sub_action)
    mae_proxy = ctx.holon_registry.get_proxy("mae")
    if mae_proxy is not None:
        mae_proxy.set_system_ref(ctx.organism_action)

    # Step hook — run organism-level ACT every 10 steps (advisory, not every step)
    _fractal_act_counter = [0]
    _fractal_act_interval = 10

    def _fractal_act_hook() -> None:
        _fractal_act_counter[0] += 1
        if _fractal_act_counter[0] % _fractal_act_interval == 0:
            ctx.organism_action.act()

    ctx.model.add_step_hook(_fractal_act_hook)

    logger.info(
        "Layer 25 - Fractal ACT: %d proxy refs injected, %d organs, %d subsystems, interval=%d",
        _proxy_refs_injected,
        len(ctx.organism_action._organ_actions),
        sum(len(o._subsystem_actions) for o in ctx.organism_action._organ_actions.values()),
        _fractal_act_interval,
    )

    # =================================================================
    # Layer 25b: Integration Meter (IIT Phi + Markov blanket — Law 8)
    # =================================================================
    # Measures integrated information (Phi) at every scale and computes
    # Markov blankets for each subsystem. Advisory only — never blocks.
    # Addresses Part 5 Properties 1 (Integration) and 6 (Boundary).
    from mae_core.backbone.integration_meter import IntegrationMeter

    ctx.integration_meter = IntegrationMeter(
        holon_registry=ctx.holon_registry,
        connection_registry=ctx.connection_registry,
        event_bus=ctx.bus,
        cadence=89,       # Fibonacci — expensive computation
        buffer_size=50,
        bins=8,
    )
    ctx.model.add_step_hook(ctx.integration_meter.step)

    # Register as holon + somatic
    ctx.holon_registry.register(
        "integration_meter", holon_type="system", parent_id="mae",
    )
    _register_somatic_systems(ctx.somatic_map, {"integration_meter": ctx.integration_meter})
    _im_proxy = ctx.holon_registry.get_proxy("integration_meter")
    if _im_proxy is not None:
        _im_proxy.set_system_ref(ctx.integration_meter)

    # Register triadic connections (after seal — advisory mode)
    from mae_core.backbone.connection_registry import ConnectionType as _CT
    _im_reg = ctx.connection_registry.register_connection
    _im_reg("integration_meter", "holon_registry", _CT.DIRECT_REFERENCE,
            witnesses=["fractal_generator", "somatic_map"])
    _im_reg("integration_meter", "connection_registry", _CT.DIRECT_REFERENCE,
            witnesses=["holon_registry", "somatic_map"])
    _im_reg("integration_meter", "event_bus", _CT.EVENTBUS_PUBSUB,
            witnesses=["holon_registry", "fractal_generator"])
    _im_reg("integration_meter", "model", _CT.STEP_HOOK,
            witnesses=["awareness_pulse", "fractal_generator"])

    logger.info("Layer 25b - Integration Meter: IIT Phi + Markov blanket (cadence=89)")

    # =================================================================
    # Layer 25c: Topology Analyzer (transfractal metrics — Part 6)
    # =================================================================
    # Measures clustering coefficient, average path length, small-world
    # sigma, and graph density. Quantifies the transfractal compromise:
    # high clustering (triadic local structure) + short paths (EventBus).
    # Advisory only — never blocks.
    from mae_core.backbone.topology_analyzer import TopologyAnalyzer

    ctx.topology_analyzer = TopologyAnalyzer(
        connection_registry=ctx.connection_registry,
        event_bus=ctx.bus,
        cadence=55,       # Fibonacci — BFS over connection graph
    )
    ctx.model.add_step_hook(ctx.topology_analyzer.step)

    # Register as holon + somatic
    ctx.holon_registry.register(
        "topology_analyzer", holon_type="system", parent_id="mae",
    )
    _register_somatic_systems(ctx.somatic_map, {"topology_analyzer": ctx.topology_analyzer})
    _ta_proxy = ctx.holon_registry.get_proxy("topology_analyzer")
    if _ta_proxy is not None:
        _ta_proxy.set_system_ref(ctx.topology_analyzer)

    # Register triadic connections (after seal — advisory mode)
    _ta_reg = ctx.connection_registry.register_connection
    _ta_reg("topology_analyzer", "connection_registry", _CT.DIRECT_REFERENCE,
            witnesses=["holon_registry", "somatic_map"])
    _ta_reg("topology_analyzer", "event_bus", _CT.EVENTBUS_PUBSUB,
            witnesses=["holon_registry", "fractal_generator"])
    _ta_reg("topology_analyzer", "model", _CT.STEP_HOOK,
            witnesses=["awareness_pulse", "fractal_generator"])

    logger.info("Layer 25c - Topology Analyzer: clustering/path_length/sigma/density (cadence=55)")

    # =================================================================
    # Layer 25d: Triadic Verifier (Part 1 proof verification)
    # =================================================================
    # Verifies four mathematical proofs: Laman rigidity, Peirce
    # irreducibility, Hegel synthesis, Simmel witness cycles.
    # Completes Part 1's six independent proofs (Byzantine + IIT already
    # implemented). Advisory only — never blocks.
    from mae_core.backbone.triadic_verifiers import TriadicVerifier

    ctx.triadic_verifier = TriadicVerifier(
        connection_registry=ctx.connection_registry,
        triad_enforcer=ctx.enforcer,
        event_bus=ctx.bus,
        cadence=89,       # Fibonacci — structural invariants change slowly
    )
    ctx.model.add_step_hook(ctx.triadic_verifier.step)

    # Register as holon + somatic
    ctx.holon_registry.register(
        "triadic_verifier", holon_type="system", parent_id="mae",
    )
    _register_somatic_systems(ctx.somatic_map, {"triadic_verifier": ctx.triadic_verifier})
    _tv_proxy = ctx.holon_registry.get_proxy("triadic_verifier")
    if _tv_proxy is not None:
        _tv_proxy.set_system_ref(ctx.triadic_verifier)

    # Register triadic connections (after seal — advisory mode)
    _tv_reg = ctx.connection_registry.register_connection
    _tv_reg("triadic_verifier", "connection_registry", _CT.DIRECT_REFERENCE,
            witnesses=["holon_registry", "somatic_map"])
    _tv_reg("triadic_verifier", "enforcer", _CT.DIRECT_REFERENCE,
            witnesses=["watchdog", "auditor"])
    _tv_reg("triadic_verifier", "event_bus", _CT.EVENTBUS_PUBSUB,
            witnesses=["holon_registry", "fractal_generator"])
    _tv_reg("triadic_verifier", "model", _CT.STEP_HOOK,
            witnesses=["awareness_pulse", "fractal_generator"])

    logger.info("Layer 25d - Triadic Verifier: laman/peirce/hegel/simmel (cadence=89)")
