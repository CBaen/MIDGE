"""Bootstrap Layers 26-30: Biological systems, organism state, lifecycle.

Creates: digestive_system, respiratory_system, vestibular_system,
homeostasis, thermoregulation, energy_reserve, circulatory_system,
renal_filter, microbiome, emotional_system, theory_of_mind,
metacognition, nociception, proprioception, lymphatic_system,
senescence, boundary_membrane, reproductive_system, organism_state,
inhibition_system, goal_manager, arousal_regulator.
"""

from __future__ import annotations

import json
import logging
import time
from types import SimpleNamespace

logger = logging.getLogger("midge.bootstrap")


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


def bootstrap_organs(ctx: SimpleNamespace) -> None:
    """Create biological systems, organism state, and lifecycle steps (Layers 26-30)."""
    from mae_core.coordination.emotional_system import EmotionalSystem
    from mae_core.coordination.homeostasis import HomeostasisRegulator
    from mae_core.coordination.thermoregulation import ThermoregulationSystem
    from mae_core.coordination.digestive_system import DigestiveSystem
    from mae_core.coordination.respiratory_system import RespiratorySystem
    from mae_core.coordination.vestibular_system import VestibularSystem
    from mae_core.coordination.organism_state import OrganismState
    from mae_core.coordination.inhibition_system import InhibitionSystem
    from mae_core.coordination.arousal_regulator import ArousalRegulator
    from mae_core.cognition.goal_manager import GoalManager
    from mae_core.cognition.theory_of_mind import TheoryOfMind
    from mae_core.cognition.metacognition import MetacognitionMonitor
    from mae_core.cognition.collective_dream import DreamAgent
    from mae_core.communication.nociception import NociceptionSystem
    from mae_core.defense.boundary_membrane import BoundaryMembrane
    from mae_core.defense.renal_filter import RenalFilter
    from mae_core.memory.energy_reserve import EnergyReserve
    from mae_core.substrate.circulatory_system import CirculatorySystem
    from mae_core.emergent.lymphatic_system import LymphaticSystem
    from mae_core.emergent.senescence import SenescenceManager
    from mae_core.emergent.proprioception import ProprioceptionSystem
    from mae_core.emergent.microbiome import Microbiome
    from mae_core.morphogenesis.reproductive_system import ReproductiveSystem
    from mae_core.emergent.auto_healer import FailureReport, FailureType
    from mae_core.backbone.fractal_generator import FractalLevel

    # =================================================================
    # Layer 26: Metabolic Systems (digestion, circulation, regulation)
    # =================================================================
    ctx.digestive_system = DigestiveSystem(event_bus=ctx.bus)
    ctx.respiratory_system = RespiratorySystem(event_bus=ctx.bus)
    ctx.vestibular_system = VestibularSystem(event_bus=ctx.bus)
    ctx.homeostasis = HomeostasisRegulator(event_bus=ctx.bus)
    ctx.thermoregulation = ThermoregulationSystem(event_bus=ctx.bus)
    ctx.energy_reserve = EnergyReserve(event_bus=ctx.bus)
    ctx.circulatory_system = CirculatorySystem(event_bus=ctx.bus, substrate=ctx.substrate)
    ctx.renal_filter = RenalFilter(event_bus=ctx.bus)
    ctx.microbiome = Microbiome(event_bus=ctx.bus)

    # Step hooks for metabolic systems
    ctx.model.add_step_hook(lambda s=ctx.digestive_system: s.step(current_step=int(ctx.model.time)))
    ctx.model.add_step_hook(lambda s=ctx.respiratory_system: s.step(current_step=int(ctx.model.time)))
    ctx.model.add_step_hook(lambda s=ctx.vestibular_system: s.step(current_step=int(ctx.model.time)))
    ctx.model.add_step_hook(lambda s=ctx.homeostasis: s.step(current_step=int(ctx.model.time)))
    ctx.model.add_step_hook(lambda s=ctx.thermoregulation: s.step(current_step=int(ctx.model.time)))
    ctx.model.add_step_hook(lambda s=ctx.energy_reserve: s.step(current_step=int(ctx.model.time)))
    ctx.model.add_step_hook(lambda s=ctx.circulatory_system: s.step(current_step=int(ctx.model.time)))
    ctx.model.add_step_hook(lambda s=ctx.renal_filter: s.step(current_step=int(ctx.model.time)))

    # Wire microbiome: feed BEFORE step() so _process_counts are populated
    # when _evolve_populations checks idle status (step resets counts at end)
    _micro = ctx.microbiome
    _micro_types = ["pattern", "anomaly", "weak_signal", "noisy", "data"]

    def _feed_microbiome(channel, data, input_type="pattern"):
        try:
            if isinstance(data, dict):
                _micro.process_input(input_type, data)
            else:
                _micro.process_input(input_type, {"raw": data})
        except Exception:
            pass

    # Event-driven feeding from high-frequency channels
    ctx.bus.register_callback("signal.PREDICTION_ERROR", lambda ch, d: _feed_microbiome(ch, d, "anomaly"))
    ctx.bus.register_callback("external.response_received", lambda ch, d: _feed_microbiome(ch, d, "data"))

    # Step-driven: feed all specializations BEFORE microbiome.step() each step.
    # Must run before step() because step() resets _process_counts at the end.
    def _microbiome_step_feed(step):
        try:
            for input_type in _micro_types:
                _micro.process_input(input_type, {"step": step, "source": "organism_rhythm"})
        except Exception:
            pass
    ctx.model.add_step_hook(lambda s=None: _microbiome_step_feed(int(ctx.model.time)))
    # Microbiome step AFTER feed hook — checks _process_counts, then resets
    ctx.model.add_step_hook(lambda s=ctx.microbiome: s.step(current_step=int(ctx.model.time)))

    logger.info(
        "Layer 26 - Metabolic Systems: 9 systems created (digestion, respiration, vestibular, "
        "homeostasis, thermoregulation, energy, circulation, renal, microbiome)"
    )

    # =================================================================
    # Layer 27: Social Cognition + Sensory Extension
    # =================================================================
    ctx.emotional_system = EmotionalSystem(event_bus=ctx.bus)
    ctx.theory_of_mind = TheoryOfMind(event_bus=ctx.bus)
    ctx.metacognition = MetacognitionMonitor(event_bus=ctx.bus)
    ctx.nociception = NociceptionSystem(event_bus=ctx.bus, somatic_map=ctx.somatic_map)
    ctx.proprioception = ProprioceptionSystem(
        event_bus=ctx.bus, somatic_map=ctx.somatic_map, fractal_generator=ctx.fractal_generator,
    )

    # Step hooks for social cognition + sensory
    ctx.model.add_step_hook(lambda s=ctx.emotional_system: s.step(current_step=int(ctx.model.time)))
    ctx.model.add_step_hook(lambda s=ctx.theory_of_mind: s.step(current_step=int(ctx.model.time)))
    ctx.model.add_step_hook(lambda s=ctx.metacognition: s.step(current_step=int(ctx.model.time)))
    ctx.model.add_step_hook(lambda s=ctx.nociception: s.step(current_step=int(ctx.model.time)))
    ctx.model.add_step_hook(lambda s=ctx.proprioception: s.step(current_step=int(ctx.model.time)))

    # Wire metacognition into Tier 2 persistence (created here at Layer 27,
    # after _tier2_refs was built at Layer 12 in agents.py)
    if hasattr(ctx.model, "_tier2_refs") and "shared_systems" in ctx.model._tier2_refs:
        ctx.model._tier2_refs["shared_systems"]["metacognition"] = ctx.metacognition

    # Restore metacognition from prior run (must happen here, after creation)
    metacog_meta = ctx.model.load_subsystem_metadata("subsystem:shared:metacognition")
    if metacog_meta:
        ctx.metacognition.restore(metacog_meta)
        logger.info("Layer 27 - Metacognition: restored %d decisions from prior run",
                     len(ctx.metacognition._decision_history))

    # Wire theory_of_mind into Tier 2 persistence
    if hasattr(ctx.model, "_tier2_refs") and "shared_systems" in ctx.model._tier2_refs:
        ctx.model._tier2_refs["shared_systems"]["theory_of_mind"] = ctx.theory_of_mind

    # Restore theory_of_mind from prior run
    tom_meta = ctx.model.load_subsystem_metadata("subsystem:shared:theory_of_mind")
    if tom_meta:
        ctx.theory_of_mind.restore(tom_meta)
        logger.info("Layer 27 - TheoryOfMind: restored %d agent models from prior run",
                     len(ctx.theory_of_mind._agent_models))

    logger.info(
        "Layer 27 - Social Cognition + Sensory: 5 systems created "
        "(emotions, theory-of-mind, metacognition, nociception, proprioception)"
    )

    # =================================================================
    # Layer 28: Maintenance + Growth + Boundary
    # =================================================================
    ctx.lymphatic_system = LymphaticSystem(event_bus=ctx.bus)
    ctx.senescence = SenescenceManager(event_bus=ctx.bus)
    ctx.boundary_membrane = BoundaryMembrane(event_bus=ctx.bus)
    ctx.reproductive_system = ReproductiveSystem(
        event_bus=ctx.bus,
        morph_coordinator=ctx.morph_coordinator,
    )

    # Step hooks for maintenance + growth
    ctx.model.add_step_hook(lambda s=ctx.lymphatic_system: s.step(current_step=int(ctx.model.time)))
    ctx.model.add_step_hook(lambda s=ctx.senescence: s.step(current_step=int(ctx.model.time)))
    ctx.model.add_step_hook(lambda s=ctx.boundary_membrane: s.step(current_step=int(ctx.model.time)))

    # Reproductive system: feed agent loads before each step so
    # _should_spawn() sees real population, not zeros.
    def _reproductive_step_hook() -> None:
        agent_loads = {}
        for agent in ctx.agents:
            uid = getattr(agent, "unique_id", id(agent))
            # Use task completion rate as proxy for load (0.0-1.0)
            tasks = getattr(agent, "_active_tasks", [])
            load = min(1.0, len(tasks) * 0.2) if tasks else 0.1
            agent_loads[uid] = load
        ctx.reproductive_system.update_metrics(agent_loads)
        ctx.reproductive_system.step(current_step=int(ctx.model.time))

    ctx.model.add_step_hook(_reproductive_step_hook)

    # Register all 18 new systems with SomaticMap
    _new_systems = {
        "digestive_system": ctx.digestive_system,
        "respiratory_system": ctx.respiratory_system,
        "vestibular_system": ctx.vestibular_system,
        "homeostasis": ctx.homeostasis,
        "thermoregulation": ctx.thermoregulation,
        "energy_reserve": ctx.energy_reserve,
        "circulatory_system": ctx.circulatory_system,
        "renal_filter": ctx.renal_filter,
        "microbiome": ctx.microbiome,
        "emotional_system": ctx.emotional_system,
        "theory_of_mind": ctx.theory_of_mind,
        "metacognition": ctx.metacognition,
        "nociception": ctx.nociception,
        "proprioception": ctx.proprioception,
        "lymphatic_system": ctx.lymphatic_system,
        "senescence": ctx.senescence,
        "boundary_membrane": ctx.boundary_membrane,
        "reproductive_system": ctx.reproductive_system,
    }
    _register_somatic_systems(ctx.somatic_map, _new_systems)

    # Register all 18 new system holons (temporarily under "mae")
    for name in _new_systems:
        ctx.holon_registry.register(name, holon_type="system", parent_id="mae")

    # Build missing fractal structure
    ctx.fractal_generator.generate_triad(
        "social-cognition", FractalLevel.SUBSYSTEM.value,
        ["emotional_system", "theory_of_mind", "metacognition"], "cognitive-system",
    )
    ctx.fractal_generator.generate_triad(
        "maintenance", FractalLevel.SUBSYSTEM.value,
        ["lymphatic_system", "senescence", "boundary_membrane"], "somatic-system",
    )

    # New metabolic-system organ (all 3 subsystems are new)
    ctx.fractal_generator.generate_triad(
        "digestion", FractalLevel.SUBSYSTEM.value,
        ["digestive_system", "renal_filter", "microbiome"], "metabolic-system",
    )
    ctx.fractal_generator.generate_triad(
        "circulation", FractalLevel.SUBSYSTEM.value,
        ["circulatory_system", "respiratory_system", "energy_reserve"], "metabolic-system",
    )
    ctx.fractal_generator.generate_triad(
        "regulation", FractalLevel.SUBSYSTEM.value,
        ["homeostasis", "thermoregulation", "vestibular_system"], "metabolic-system",
    )
    ctx.fractal_generator.generate_triad(
        "metabolic-system", FractalLevel.ORGAN.value,
        ["digestion", "circulation", "regulation"], "mae",
    )

    # Reparent systems into correct subsystems
    ctx.holon_registry.set_parent("nociception", "consensus")
    ctx.holon_registry.set_parent("proprioception", "consensus")
    ctx.holon_registry.set_parent("reproductive_system", "growth")
    ctx.holon_registry.set_parent("colony", "growth")

    # Inject holon proxies into all 18 new systems
    for name, system in _new_systems.items():
        system._holon = ctx.holon_registry.get_proxy(name)

    # Inject system refs into proxies (for Fractal ACT — continues Layer 25 work)
    for name, system in _new_systems.items():
        proxy = ctx.holon_registry.get_proxy(name)
        if proxy is not None:
            proxy.set_system_ref(system)

    logger.info(
        "Layer 28 - Maintenance + Growth: 4 systems created "
        "(lymphatic, senescence, boundary, reproductive). "
        "All 18 new systems registered with SomaticMap + HolonRegistry"
    )

    # =================================================================
    # Layer 29: Organism State + Deep Integration
    # =================================================================

    # -- 29a: Create OrganismState --
    ctx.organism_state = OrganismState(event_bus=ctx.bus)
    ctx.model.add_step_hook(lambda: ctx.organism_state.step(current_step=int(ctx.model.time)))

    ctx.holon_registry.register("organism_state", holon_type="system", parent_id="mae")
    _register_somatic_systems(ctx.somatic_map, {"organism_state": ctx.organism_state})
    ctx.organism_state._holon = ctx.holon_registry.get_proxy("organism_state")
    proxy = ctx.holon_registry.get_proxy("organism_state")
    if proxy is not None:
        proxy.set_system_ref(ctx.organism_state)

    # -- 29a2: RedifferentiationMonitor (auto stem cell triggers) --
    from mae_core.agents.redifferentiation_triggers import RedifferentiationMonitor

    ctx.rediff_monitor = RedifferentiationMonitor(
        stem_cell_registry=ctx.stem_cell_registry, event_bus=ctx.bus,
    )
    ctx.model.add_step_hook(lambda: ctx.rediff_monitor.step(current_step=int(ctx.model.time)))
    _register_somatic_systems(ctx.somatic_map, {"rediff_monitor": ctx.rediff_monitor})
    logger.info("Layer 29a2 - RedifferentiationMonitor: auto triggers active (cadence=21)")

    # -- 29a3: MitosisMonitor (autopoietic production loop) --
    from mae_core.agents.mitosis import MitosisMonitor

    ctx.mitosis_monitor = MitosisMonitor(
        model=ctx.model,
        stem_cell_registry=ctx.stem_cell_registry,
        holon_registry=ctx.holon_registry,
        event_bus=ctx.bus,
        max_agents=ctx.num_agents * 2,
        shared_systems={
            "signal_bus": ctx.signal_bus,
            "stigmergy": ctx.stigmergy,
            "gnn_communicator": ctx.gnn_comm,
            "knowledge_base": ctx.knowledge_base,
            "transfer_engine": ctx.transfer_engine,
            "maml_learner": ctx.maml_learner,
            "somatic_map": ctx.somatic_map,
            "substrate": ctx.substrate,
        },
    )
    ctx.model.add_step_hook(lambda: ctx.mitosis_monitor.step(current_step=int(ctx.model.time)))
    _register_somatic_systems(ctx.somatic_map, {"mitosis_monitor": ctx.mitosis_monitor})
    logger.info("Layer 29a3 - MitosisMonitor: autopoietic production loop active (cadence=13)")

    # -- 29a4: ClosureCoordinator (autopoietic closure at subsystem/organ/organism) --
    from mae_core.backbone.autopoietic_closure import ClosureCoordinator

    ctx.closure_coordinator = ClosureCoordinator(
        holon_registry=ctx.holon_registry,
        event_bus=ctx.bus,
    )
    ctx.model.add_step_hook(lambda: ctx.closure_coordinator.step(current_step=int(ctx.model.time)))
    _register_somatic_systems(ctx.somatic_map, {"closure_coordinator": ctx.closure_coordinator})
    logger.info(
        "Layer 29a4 - ClosureCoordinator: autopoietic closure active at 3 scales "
        "(subsystem=5, organ=8, organism=13 steps)"
    )

    # -- 29b: Cross-system EventBus wiring --

    # 1. Nociception -> EmotionalSystem: Pain reinforces fear response.
    def _pain_to_emotion(channel, message) -> None:
        try:
            data = json.loads(message) if isinstance(message, str) else message
            if not isinstance(data, dict):
                return
            pain = data.get("total_pain_load", 0)
            if pain > 0.5:
                ctx.emotional_system._fear_reinforcement = min(
                    1.0, ctx.emotional_system._fear_reinforcement + pain * 0.4,
                )
        except Exception:
            pass

    ctx.bus.register_callback("communication.pain_update", _pain_to_emotion)

    # 2. EnergyReserve -> DigestiveSystem: Leptin suppresses appetite.
    def _leptin_to_satiation(channel, message) -> None:
        try:
            data = json.loads(message) if isinstance(message, str) else message
            if not isinstance(data, dict):
                return
            leptin = data.get("leptin_level", 0.0)
            if leptin > 0.7:
                ctx.bus.publish("coordination.satiation_signal", {
                    "source": "energy_reserve",
                    "leptin_level": leptin,
                    "suppress_appetite": True,
                })
        except Exception:
            pass

    ctx.bus.register_callback("memory.energy_status", _leptin_to_satiation)

    # 3. SenescenceManager -> AutoHealer: Rejuvenation requests trigger healing.
    def _senescence_to_healing(channel, message) -> None:
        try:
            data = json.loads(message) if isinstance(message, str) else message
            if not isinstance(data, dict):
                return
            system_id = data.get("system_id", "unknown")
            failure = FailureReport(
                failure_id=f"senescence-rejuv-{system_id}-{int(time.time())}",
                failure_type=FailureType.PERFORMANCE_DEGRADATION,
                affected_agents=[system_id],
                severity=0.4,
                metadata={
                    "source": "senescence_manager",
                    "trigger": "rejuvenation_needed",
                },
            )
            ctx.auto_healer.report_failure(failure)
        except Exception:
            pass

    ctx.bus.register_callback("emergent.rejuvenation_needed", _senescence_to_healing)

    # 4. MetacognitionMonitor -> VDN + WorldModel: adaptive learning rates.
    # Biological basis: prefrontal executive modulation of learning speed.
    def _metacognition_to_learning_rate(channel, message) -> None:
        try:
            monitor = getattr(ctx, "metacognition", None)
            if monitor is None:
                return
            multiplier = monitor.should_adjust_learning_rate()
            if multiplier is None:
                return
            for agent in ctx.agents:
                vdn = getattr(agent, "_vdn_engine", None)
                if vdn is not None:
                    new_lr = float(vdn._lr) * multiplier
                    vdn._lr = max(0.001, min(0.1, new_lr))
                wm = getattr(agent, "world_model", None)
                if wm is not None:
                    new_lr = float(wm._config.learning_rate) * multiplier
                    wm._config.learning_rate = max(1e-5, min(1e-2, new_lr))
            logger.debug(
                "Metacognition LR bridge: multiplier=%.2f applied to %d agents",
                multiplier, len(ctx.agents),
            )
        except Exception:
            pass

    ctx.bus.register_callback("cognition.metacognition_alert", _metacognition_to_learning_rate)

    logger.info(
        "Layer 29a/b - OrganismState created + 4 cross-system wires "
        "(pain->emotion, leptin->satiation, senescence->healing, metacognition->learning_rate)"
    )

    # -- 29c: Inject references into agents --
    for agent in ctx.agents:
        agent._organism_state = ctx.organism_state
        agent._worldline_planner = ctx.worldline_planner
        agent._collective_dream = ctx.collective_dream
        agent._predictive_field = ctx.predictive_field
        agent._morphogenesis = ctx.morph_coordinator
        # Theory of Mind created at Layer 27 — inject now that it exists
        agent._theory_of_mind = ctx.theory_of_mind
        # Metacognition created at Layer 27 — inject for prediction pipeline
        agent._metacognition = ctx.metacognition

    logger.info(
        "Layer 29c - Agent injection: organism_state + 5 dormant system refs "
        "injected into %d agents",
        len(ctx.agents),
    )

    # -- 29d: Register agents with CollectiveDreamPlanner --
    if ctx.collective_dream is not None:
        _dream_agents_registered = 0
        for agent in ctx.agents:
            if hasattr(ctx.collective_dream, "register_agent"):
                dream_agent = DreamAgent(
                    agent_id=str(agent.unique_id),
                    expertise=0.5,
                    domain="general",
                )
                ctx.collective_dream.register_agent(dream_agent)
                _dream_agents_registered += 1
        logger.info(
            "Layer 29d - CollectiveDream: %d dream agents registered",
            _dream_agents_registered,
        )

    # -- 29e: TriageClassifier (biological urgency triage for signal processing) --
    from mae_core.communication.triage_classifier import TriageClassifier

    ctx.triage_classifier = TriageClassifier(
        nociception=ctx.nociception,
        threat_detector=ctx.threat_detector,
        endocrine=ctx.endocrine,
    )

    # Inject into each agent's SignalPriorityResolver
    _triage_injected = 0
    for agent in ctx.agents:
        resolver = getattr(agent, "_signal_resolver", None)
        if resolver is not None:
            resolver._triage_classifier = ctx.triage_classifier
            _triage_injected += 1

    _register_somatic_systems(ctx.somatic_map, {"triage_classifier": ctx.triage_classifier})
    ctx.holon_registry.register("triage_classifier", holon_type="system", parent_id="mae")

    # Register triadic connections for TriageClassifier (Law 1)
    from mae_core.backbone.connection_registry import ConnectionType as _TCT
    _tc_reg = ctx.connection_registry.register_connection
    _tc_reg("triage_classifier", "nociception", _TCT.DIRECT_REFERENCE,
            witnesses=["threat_detector", "endocrine"],
            description="Triage reads pain state for urgency classification")
    _tc_reg("triage_classifier", "threat_detector", _TCT.DIRECT_REFERENCE,
            witnesses=["nociception", "endocrine"],
            description="Triage reads threat state for urgency classification")
    _tc_reg("triage_classifier", "endocrine", _TCT.DIRECT_REFERENCE,
            witnesses=["nociception", "threat_detector"],
            description="Triage reads stress/arousal for urgency classification")

    logger.info(
        "Layer 29e - TriageClassifier: biological urgency triage "
        "injected into %d agent signal resolvers",
        _triage_injected,
    )

    logger.info(
        "Layer 29 - Organism State + Deep Integration complete"
    )

    # =================================================================
    # Layer 30: Lifecycle Step Systems (INHIBIT, GOAL, REGULATE)
    # =================================================================
    ctx.inhibition_system = InhibitionSystem(event_bus=ctx.bus)
    ctx.goal_manager = GoalManager(event_bus=ctx.bus)
    ctx.arousal_regulator = ArousalRegulator(event_bus=ctx.bus)

    # Register with holon registry
    ctx.holon_registry.register("inhibition_system", holon_type="system", parent_id="mae")
    ctx.holon_registry.register("goal_manager", holon_type="system", parent_id="mae")
    ctx.holon_registry.register("arousal_regulator", holon_type="system", parent_id="mae")
    _register_somatic_systems(ctx.somatic_map, {
        "inhibition_system": ctx.inhibition_system,
        "goal_manager": ctx.goal_manager,
        "arousal_regulator": ctx.arousal_regulator,
    })

    # Inject into agents
    for agent in ctx.agents:
        agent._inhibition_system = ctx.inhibition_system
        agent._goal_manager = ctx.goal_manager
        agent._arousal_regulator = ctx.arousal_regulator
        # attentional_gate was created in Layer 23 but never injected into agents
        agent._attentional_gate = ctx.attentional_gate
        # PatternBus reference for BROADCAST step
        agent._pattern_bus = ctx.pattern_bus if ctx.pattern_bus is not None else None

    logger.info(
        "Layer 30 - Lifecycle Steps: InhibitionSystem + GoalManager + "
        "ArousalRegulator created, attentional_gate + pattern_bus injected "
        "into %d agents",
        len(ctx.agents),
    )

    # =================================================================
    # Layer 30b: Phi-driven behavioral modulation
    # =================================================================
    # IntegrationMeter publishes organism_mean_phi on integration.phi_measurement.
    # Wire consumers so phi DRIVES behavior, not just gets logged.
    # 1. EndocrineSystem: low phi -> mild cortisol (coordination stress)
    # 2. ArousalRegulator: low phi -> raise target (drive interaction)
    # 3. GlobalWorkspace: low phi -> lower ignition threshold (easier broadcast)

    ctx.bus.register_callback(
        "integration.phi_measurement", ctx.endocrine._on_phi_measurement,
    )

    ctx.bus.register_callback(
        "integration.phi_measurement", ctx.arousal_regulator._on_phi_measurement,
    )

    # GlobalWorkspace lives inside PatternCortex — access via ctx.pattern_cortex
    _pattern_cortex = getattr(ctx, "pattern_cortex", None)
    _gw = getattr(_pattern_cortex, "_workspace", None) if _pattern_cortex else None
    if _gw is not None and hasattr(_gw, "_on_phi_measurement"):
        ctx.bus.register_callback(
            "integration.phi_measurement", _gw._on_phi_measurement,
        )
        _phi_gw_wired = True
    else:
        _phi_gw_wired = False

    # Register triadic connections for phi consumers
    from mae_core.backbone.connection_registry import ConnectionType as _CT
    _phi_reg = ctx.connection_registry.register_connection
    _phi_reg("integration_meter", "endocrine", _CT.EVENTBUS_PUBSUB,
             channel="integration.phi_measurement",
             witnesses=["arousal_regulator", "somatic_map"],
             description="Phi fragmentation/integration modulates cortisol")
    _phi_reg("integration_meter", "arousal_regulator", _CT.EVENTBUS_PUBSUB,
             channel="integration.phi_measurement",
             witnesses=["endocrine", "somatic_map"],
             description="Phi drives arousal target for agent interaction")
    if _phi_gw_wired:
        _phi_reg("integration_meter", "pattern_cortex", _CT.EVENTBUS_PUBSUB,
                 channel="integration.phi_measurement",
                 witnesses=["endocrine", "arousal_regulator"],
                 description="Phi adjusts GWT ignition threshold for integration")

    logger.info(
        "Layer 30b - Phi-driven modulation: endocrine + arousal_regulator + "
        "global_workspace(%s) now respond to integration.phi_measurement",
        "wired" if _phi_gw_wired else "skipped",
    )
