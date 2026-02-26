"""Bootstrap Layers 1-11: Foundation systems (no agent dependency).

Creates: model, bus, holon_registry, circadian, endocrine, enforcer,
watchdog, auditor, substrate, physarum, signal_bus, gnn_comm, stigmergy,
quorum_space, predictive_field, knowledge_base, transfer_engine,
maml_learner, curiosity, haven, imitation, threat_detector,
input_validator, pearl_defense, shared_world_model, collective_dream,
validated_imagination, shared_causal_engine, auto_healer,
capability_discovery, somatic_map, organ_builder, morph_coordinator,
temporal_memory, worldline_planner.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

logger = logging.getLogger("mae.bootstrap")


def bootstrap_foundation(ctx: SimpleNamespace) -> None:
    """Create all shared systems that don't depend on agents (Layers 1-11)."""
    from mae_core.model import MycelialModel
    from mae_core.agents.mycelial_agent import MycelialAgent
    from mae_core.coordination.circadian_rhythm import CircadianPhase, CircadianRhythm
    from mae_core.coordination.endocrine_system import EndocrineSystem
    from mae_core.backbone.triad_auditor import TriadAuditor
    from mae_core.backbone.triad_enforcer import TriadEnforcer
    from mae_core.backbone.triad_registry import register_all_triads, wire_triad_systems
    from mae_core.backbone.triad_watchdog import TriadWatchdog
    from mae_core.substrate.mycelial_substrate import MycelialSubstrate
    from mae_core.substrate.physarum_optimizer import PhysarumOptimizer
    from mae_core.communication.signal_bus import SignalBus
    from mae_core.communication.gnn_communicator import GNNCommunicator
    from mae_core.communication.stigmergy import StigmergicEnvironment
    from mae_core.communication.quorum_space import QuorumSpace
    from mae_core.communication.predictive_field import PredictiveField
    from mae_core.learning.knowledge_base import KnowledgeBase
    from mae_core.learning.transfer_learning import TransferLearningEngine
    from mae_core.learning.maml import MAMLLearner
    from mae_core.learning.curiosity import CuriosityDrive
    from mae_core.learning.haven import HavenRiskCoordinator
    from mae_core.learning.imitation import ImitationLearning
    from mae_core.defense.threat_detector import ThreatDetector
    from mae_core.defense.input_validator import InputValidator
    from mae_core.defense.pearl_defense import PearlDefense
    from mae_core.cognition.world_model import WorldModel
    from mae_core.cognition.collective_dream import CollectiveDreamPlanner
    from mae_core.cognition.validated_imagination import ValidatedImagination
    from mae_core.cognition.causal_reasoning import CausalReasoningEngine
    from mae_core.emergent.auto_healer import AutoHealer
    from mae_core.emergent.capability_discovery import CapabilityDiscovery
    from mae_core.emergent.somatic_map import SomaticMap
    from mae_core.backbone.holon_protocol import HolonRegistry
    from mae_core.morphogenesis.coordinator import MorphogenesisCoordinator
    from mae_core.morphogenesis.organ_builder import OrganBuilder
    from mae_core.planning.temporal_memory import TemporalMemory
    from mae_core.planning.worldline_planner import WorldlinePlanner

    # =================================================================
    # Layer 1: Foundation (EventBus + StateStore + VectorStore)
    # =================================================================
    ctx.model = MycelialModel(persist_dir=Path(ctx.persist_dir))
    ctx.bus = ctx.model.event_bus
    ctx.holon_registry = HolonRegistry()
    ctx.holon_registry.register("mae", holon_type="organism")
    logger.info("Layer 1 - Foundation: MycelialModel + HolonRegistry created (backbone ready)")

    # =================================================================
    # Layer 2: Coordination (Circadian clock + Endocrine hormones)
    # =================================================================
    ctx.circadian = CircadianRhythm(event_bus=ctx.bus, cycle_length=ctx.cycle_length)
    ctx.endocrine = EndocrineSystem(event_bus=ctx.bus)

    def _on_phase_change(old_phase: CircadianPhase, new_phase: CircadianPhase) -> None:
        ctx.endocrine.set_circadian_phase(new_phase.value)

    ctx.circadian.on_phase_change(_on_phase_change)
    ctx.model.add_step_hook(ctx.circadian.step)
    ctx.model.add_step_hook(ctx.endocrine.step)
    logger.info("Layer 2 - Coordination: Circadian + Endocrine wired")

    # =================================================================
    # Layer 3: Enforcement Triad (Rule of 3/5 compliance)
    # =================================================================
    ctx.enforcer = TriadEnforcer(event_bus=ctx.bus)
    ctx.watchdog = TriadWatchdog(event_bus=ctx.bus, triad_enforcer=ctx.enforcer)
    ctx.auditor = TriadAuditor(event_bus=ctx.bus, triad_enforcer=ctx.enforcer)

    ctx.triad_report = register_all_triads(
        enforcer=ctx.enforcer, watchdog=ctx.watchdog, auditor=ctx.auditor,
    )
    logger.info(
        "Layer 3 - Triad enforcement: %d processes, %d validators, %.0f%% compliant",
        ctx.triad_report.get("total_registered", 0),
        ctx.triad_report.get("total_validators_added", 0),
        ctx.triad_report.get("compliance_rate", 0) * 100,
    )

    _triad_step_counter = [0]

    def _triad_audit_hook() -> None:
        _triad_step_counter[0] += 1
        if _triad_step_counter[0] % 50 == 0:
            bypasses = ctx.watchdog.check_for_bypasses()
            if bypasses:
                logger.warning("Triad watchdog: %d bypass alerts", len(bypasses))
            findings = ctx.auditor.run_audit()
            if findings:
                logger.warning("Triad auditor: %d findings", len(findings))

    ctx.model.add_step_hook(_triad_audit_hook)

    # =================================================================
    # Layer 4: Substrate (network fabric)
    # =================================================================
    ctx.substrate = MycelialSubstrate(
        event_bus=ctx.bus,
        topology_type="scale_free",
        initial_nodes=ctx.num_agents,
    )
    ctx.physarum = PhysarumOptimizer(substrate=ctx.substrate, event_bus=ctx.bus)
    logger.info("Layer 4 - Substrate: Mycelial network + Physarum optimizer ready")

    # =================================================================
    # Layer 5: Communication infrastructure (shared)
    # =================================================================
    ctx.signal_bus = SignalBus(event_bus=ctx.bus)
    ctx.gnn_comm = GNNCommunicator(substrate=ctx.substrate, event_bus=ctx.bus)
    ctx.stigmergy = StigmergicEnvironment()
    ctx.quorum_space = QuorumSpace()
    ctx.predictive_field = PredictiveField(substrate=ctx.substrate)
    logger.info("Layer 5 - Communication: SignalBus, GNN, Stigmergy, Quorum, PredictiveField")

    # =================================================================
    # Layer 6: Learning infrastructure (shared)
    # =================================================================
    ctx.knowledge_base = KnowledgeBase(event_bus=ctx.bus)
    ctx.transfer_engine = TransferLearningEngine(knowledge_base=ctx.knowledge_base)
    ctx.maml_learner = MAMLLearner(knowledge_base=ctx.knowledge_base)
    ctx.curiosity = CuriosityDrive(event_bus=ctx.bus)
    ctx.haven = HavenRiskCoordinator(event_bus=ctx.bus)
    ctx.imitation = ImitationLearning()
    logger.info("Layer 6 - Learning: KnowledgeBase, Transfer, MAML, Curiosity, HAVEN, Imitation")

    # =================================================================
    # Layer 7: Defense (shared)
    # =================================================================
    ctx.threat_detector = ThreatDetector(event_bus=ctx.bus, haven=ctx.haven)
    ctx.input_validator = InputValidator(event_bus=ctx.bus)
    ctx.pearl_defense = PearlDefense(input_validator=ctx.input_validator, event_bus=ctx.bus)
    logger.info("Layer 7 - Defense: ThreatDetector + InputValidator + PearlDefense")

    # =================================================================
    # Layer 8: Cognition (shared systems)
    # =================================================================
    ctx.shared_world_model = WorldModel(event_bus=ctx.bus)
    ctx.collective_dream = CollectiveDreamPlanner(world_model=ctx.shared_world_model)
    ctx.validated_imagination = ValidatedImagination(event_bus=ctx.bus)
    ctx.shared_causal_engine = CausalReasoningEngine()
    logger.info("Layer 8 - Cognition: WorldModel, CollectiveDream, ValidatedImagination, CausalEngine")

    # =================================================================
    # Layer 9: Emergent capabilities (shared)
    # =================================================================
    ctx.auto_healer = AutoHealer(
        event_bus=ctx.bus,
        substrate=ctx.substrate,
        causal_engine=ctx.shared_causal_engine,
        haven=ctx.haven,
    )
    ctx.capability_discovery = CapabilityDiscovery(event_bus=ctx.bus)
    ctx.somatic_map = SomaticMap(event_bus=ctx.bus)
    # FIX: AutoHealer created before SomaticMap — inject for proactive scanning
    ctx.auto_healer._somatic_map = ctx.somatic_map
    logger.info("Layer 9 - Emergent: AutoHealer, CapabilityDiscovery, SomaticMap")

    # =================================================================
    # Layer 10: Morphogenesis (shared)
    # =================================================================
    def _morphogenesis_factory(
        m=None, agent_type=None, organ_id=None, model=None, **kw,
    ):
        """Lazy factory: reads shared systems from ctx at invocation time.

        Layer 10 runs before shared systems exist. By the time
        MorphogenesisCoordinator calls this (Layer 29+ or runtime),
        all shared systems are populated. Mirrors mitosis.py pattern.
        """
        return MycelialAgent(
            model or m,
            agent_type=agent_type,
            agent_config={"organ_id": organ_id, **kw},
            signal_bus=getattr(ctx, "signal_bus", None),
            stigmergy_env=getattr(ctx, "stigmergy", None),
            gnn_communicator=getattr(ctx, "gnn_comm", None),
            knowledge_base=getattr(ctx, "knowledge_base", None),
            transfer_engine=getattr(ctx, "transfer_engine", None),
            maml_learner=getattr(ctx, "maml_learner", None),
            holon_registry=getattr(ctx, "holon_registry", None),
            somatic_map=getattr(ctx, "somatic_map", None),
        )

    ctx.organ_builder = OrganBuilder(agent_factory=_morphogenesis_factory)
    ctx.morph_coordinator = MorphogenesisCoordinator(
        event_bus=ctx.bus,
        organ_builder=ctx.organ_builder,
        model=ctx.model,
        substrate=ctx.substrate,
    )
    ctx.model.add_step_hook(ctx.morph_coordinator.step)
    logger.info("Layer 10 - Morphogenesis: Coordinator + OrganBuilder")

    # =================================================================
    # Layer 11: Planning (shared)
    # =================================================================
    ctx.temporal_memory = TemporalMemory(
        event_bus=ctx.bus, causal_engine=ctx.shared_causal_engine,
    )
    ctx.worldline_planner = WorldlinePlanner(
        world_model=ctx.shared_world_model,
        temporal_memory=ctx.temporal_memory,
        causal_engine=ctx.shared_causal_engine,
        event_bus=ctx.bus,
    )
    logger.info("Layer 11 - Planning: TemporalMemory + WorldlinePlanner")

    # =================================================================
    # Layer 11b: Wire real systems into Triad validators
    # =================================================================
    # register_all_triads() ran at Layer 3 with permissive defaults
    # because systems didn't exist yet. Now wire the real methods.
    triad_systems = {
        "haven": ctx.haven,
        "auto_healer": ctx.auto_healer,
        "somatic_map": ctx.somatic_map,
        "causal_engine": ctx.shared_causal_engine,
        "world_model": ctx.shared_world_model,
        "validated_imagination": ctx.validated_imagination,
        "collective_dream": ctx.collective_dream,
        "threat_detector": ctx.threat_detector,
        "input_validator": ctx.input_validator,
        "substrate": ctx.substrate,
        "endocrine": ctx.endocrine,
        "circadian": ctx.circadian,
        "curiosity": ctx.curiosity,
        "temporal_memory": ctx.temporal_memory,
    }
    wired_count = wire_triad_systems(ctx.enforcer, triad_systems)
    logger.info("Layer 11b - Triad wiring: %d validators wired to real systems", wired_count)
