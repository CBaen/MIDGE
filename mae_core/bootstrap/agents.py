"""Bootstrap Layers 12-13: Agent colony and learning engines.

Creates: agents, per_agent_systems, frl_engines, vdn_engines.
Restores Tier 2 persistence and injects engine references into agents.
"""

from __future__ import annotations

import logging
import math
from types import SimpleNamespace

logger = logging.getLogger("midge.bootstrap")


def bootstrap_agents(ctx: SimpleNamespace) -> None:
    """Create agents, per-agent systems, and learning engines (Layers 12-13)."""
    from mae_core.agents.mycelial_agent import MycelialAgent
    from mae_core.memory.coordinator import MemoryCoordinator
    from mae_core.cognition.world_model import WorldModel
    from mae_core.cognition.decision_router import DecisionRouter
    from mae_core.cognition.causal_reasoning import CausalReasoningEngine
    from mae_core.communication.quorum_sensor import QuorumSensor
    from mae_core.patterns.pattern_sense import PatternSense
    from mae_core.learning.frl import FederatedLearningEngine
    from mae_core.learning.vdn import ValueDecompositionEngine

    # =================================================================
    # Layer 12: Agents — create colony with ALL subsystems injected
    # =================================================================
    ctx.holon_registry.register("colony", holon_type="colony", parent_id="mae")
    ctx.agents = []
    ctx.per_agent_systems: dict[int, dict] = {}

    for i in range(ctx.num_agents):
        agent_id_str = f"agent-{i}"

        # Per-agent memory coordinator (own episodic + consolidation)
        agent_memory = MemoryCoordinator(
            event_bus=ctx.bus,
            agent_id=agent_id_str,
            enable_generative=True,
        )

        # Per-agent cognition
        agent_world_model = WorldModel(event_bus=ctx.bus)
        agent_decision_router = DecisionRouter(
            event_bus=ctx.bus,
            world_model=agent_world_model,
            endocrine=ctx.endocrine,
        )
        agent_causal_engine = CausalReasoningEngine()

        # Per-agent pattern sense (cell membrane)
        agent_pattern_sense = PatternSense(agent_id=agent_id_str)

        # Per-agent quorum sensor
        agent_quorum = QuorumSensor(
            agent_id=agent_id_str,
            quorum_space=ctx.quorum_space,
        )

        # Create the fully-wired agent
        agent_cfg = {
            "max_learning_iterations": 200,
            "convergence_threshold": 0.001,
            # FIX BUG-19: Enable recall pathways (defaults are False, gating off memory)
            "semantic_search_enabled": True,
            "replay_enabled": True,
            "consolidation_enabled": True,
            "generative_memory_enabled": True,
            "transfer_enabled": True,
            "quorum_sensing_enabled": True,
        }
        # All agents get an API voice — cycle round-robin through all providers
        # Gateway handles missing keys gracefully (no key = provider not registered = no-op)
        _providers = ("groq", "mistral", "deepseek")
        agent_cfg["api_call_enabled"] = True
        agent_cfg["preferred_provider"] = _providers[i % len(_providers)]

        agent = MycelialAgent(
            ctx.model,
            agent_type="mycelial",
            agent_config=agent_cfg,
            signal_bus=ctx.signal_bus,
            stigmergy_env=ctx.stigmergy,
            gnn_communicator=ctx.gnn_comm,
            knowledge_base=ctx.knowledge_base,
            transfer_engine=ctx.transfer_engine,
            maml_learner=ctx.maml_learner,
            episodic_memory=agent_memory.episodic if hasattr(agent_memory, "episodic") else None,
            memory_consolidator=agent_memory.consolidator if hasattr(agent_memory, "consolidator") else None,
            semantic_retriever=agent_memory.semantic if hasattr(agent_memory, "semantic") else None,
            generative_memory=agent_memory.generative if hasattr(agent_memory, "generative") else None,
            quorum_sensor=agent_quorum,
            world_model=agent_world_model,
            decision_router=agent_decision_router,
            causal_engine=agent_causal_engine,
            pattern_sense=agent_pattern_sense,
            morphogenesis_enabled=True,
            holon_registry=ctx.holon_registry,
            somatic_map=ctx.somatic_map,
        )

        ctx.agents.append(agent)
        ctx.substrate.register_agent(str(agent.unique_id))
        if hasattr(ctx, "gnn_comm") and hasattr(ctx.gnn_comm, "register_agent"):
            ctx.gnn_comm.register_agent(str(agent.unique_id), agent_type="mycelial")
        ctx.holon_registry.register(str(agent.unique_id), holon_type="agent", parent_id="colony")
        # FIX BUG-11: Set parent_id on agent so serialization captures it
        # (_init_holon runs before registry entry exists, leaving parent_id=None)
        agent._holon_parent_id = "colony"
        # FIX: Inject shared CuriosityDrive so _learn() can compute intrinsic reward
        agent.curiosity_drive = ctx.curiosity
        # FIX-5: Inject shared subsystems for lifecycle integration
        # ValidatedImagination lives in foundation.py (Layer 8) — available here.
        # Theory of Mind (Layer 27) and MemoryBridge (Layer 22) are injected
        # later in organs.py and patterns.py respectively.
        if hasattr(ctx, "validated_imagination"):
            agent._validated_imagination = ctx.validated_imagination

        # FIX-6: Assign spatial positions (agents at (0,0) kills stigmergy gradients)
        _grid_size = 100
        _cols = max(1, int(math.ceil(math.sqrt(ctx.num_agents))))
        _row = i // _cols
        _col = i % _cols
        _x = (_col + 0.5) * (_grid_size / _cols) + (i * 7.3 % 10 - 5)  # deterministic jitter
        _y = (_row + 0.5) * (_grid_size / _cols) + (i * 3.7 % 10 - 5)
        agent.pos = (_x, _y)
        if hasattr(agent, "set_stigmergy_position"):
            agent.set_stigmergy_position((_x, _y))

        # Restore prior state if available (agent remembers across restarts)
        prior_state = ctx.model.load_agent_state(agent.unique_id)
        if prior_state:
            agent.restore_state(prior_state)

        # Track per-agent systems for inspection
        ctx.per_agent_systems[agent.unique_id] = {
            "memory_coordinator": agent_memory,
            "world_model": agent_world_model,
            "decision_router": agent_decision_router,
            "causal_engine": agent_causal_engine,
            "quorum_sensor": agent_quorum,
            "pattern_sense": agent_pattern_sense,
        }

    restored_count = sum(1 for a in ctx.agents if a.step_count > 0)
    logger.info(
        "Layer 12 - Colony: %d agents created (%d restored from prior state)",
        ctx.num_agents, restored_count,
    )

    # =================================================================
    # Layer 13: Register learning engines with model
    # =================================================================
    ctx.model.set_haven_coordinator(ctx.haven)

    # Register all agents with HAVEN risk coordinator for monitoring
    for agent in ctx.agents:
        ctx.haven.register_agent(str(agent.unique_id))

    # FRL + VDN (per-agent engines, coordinated at model level)
    ctx.frl_engines = {}
    ctx.vdn_engines = {}
    for agent in ctx.agents:
        agent_id_str = f"agent-{agent.unique_id}"
        frl = FederatedLearningEngine(
            agent_id=agent_id_str,
            event_bus=ctx.bus,
            memory_coordinator=ctx.per_agent_systems[agent.unique_id].get("memory_coordinator"),
        )
        vdn = ValueDecompositionEngine(agent_id=agent_id_str, event_bus=ctx.bus)
        ctx.frl_engines[agent.unique_id] = frl
        ctx.vdn_engines[agent.unique_id] = vdn

    # Register first VDN as model-level coordinator (triggers global reward distribution)
    if ctx.vdn_engines:
        first_vdn = next(iter(ctx.vdn_engines.values()))
        ctx.model.set_vdn_engine(first_vdn)

    logger.info("Layer 13 - Learning engines: HAVEN + %d FRL + %d VDN registered",
                len(ctx.frl_engines), len(ctx.vdn_engines))

    # -- Tier 2 Persistence: restore learned state from prior run --
    tier2_restored = 0
    subsystem_dir = ctx.model._persist_dir / "subsystems"
    for agent_uid, agent_sys in ctx.per_agent_systems.items():
        agent_dir = subsystem_dir / "agents" / str(agent_uid)
        if not agent_dir.exists():
            continue

        # Memory coordinator
        mc = agent_sys.get("memory_coordinator")
        mc_meta = ctx.model.load_subsystem_metadata(f"subsystem:agent:{agent_uid}:memory")
        if mc is not None and mc_meta:
            mc.restore(agent_dir, mc_meta)
            tier2_restored += 1

        # World model
        wm = agent_sys.get("world_model")
        wm_meta = ctx.model.load_subsystem_metadata(f"subsystem:agent:{agent_uid}:world_model")
        if wm is not None and wm_meta:
            wm.restore(agent_dir, wm_meta)
            tier2_restored += 1

    # Per-agent VDN + FRL
    for agent in ctx.agents:
        agent_uid = agent.unique_id
        agent_dir = subsystem_dir / "agents" / str(agent_uid)
        if not agent_dir.exists():
            continue

        vdn = ctx.vdn_engines.get(agent_uid)
        vdn_meta = ctx.model.load_subsystem_metadata(f"subsystem:agent:{agent_uid}:vdn")
        if vdn is not None and vdn_meta:
            vdn.restore(agent_dir, vdn_meta)
            tier2_restored += 1

        frl = ctx.frl_engines.get(agent_uid)
        frl_meta = ctx.model.load_subsystem_metadata(f"subsystem:agent:{agent_uid}:frl")
        if frl is not None and frl_meta:
            frl.restore(agent_dir, frl_meta)
            tier2_restored += 1

    # Shared systems
    shared_dir = subsystem_dir / "shared"
    if shared_dir.exists():
        kb_meta = ctx.model.load_subsystem_metadata("subsystem:shared:knowledge_base")
        if kb_meta:
            ctx.knowledge_base.restore(shared_dir, kb_meta)
            tier2_restored += 1

        maml_meta = ctx.model.load_subsystem_metadata("subsystem:shared:maml")
        if maml_meta:
            ctx.maml_learner.restore(shared_dir, maml_meta)
            tier2_restored += 1

    # Store references for shutdown persistence
    ctx.model._tier2_refs = {
        "memory_coordinators": {
            uid: sys["memory_coordinator"]
            for uid, sys in ctx.per_agent_systems.items()
        },
        "world_models": {
            uid: sys["world_model"]
            for uid, sys in ctx.per_agent_systems.items()
        },
        "vdn_engines": ctx.vdn_engines,
        "frl_engines": ctx.frl_engines,
        "shared_systems": {
            "knowledge_base": ctx.knowledge_base,
            "maml": ctx.maml_learner,
        },
    }

    if tier2_restored > 0:
        logger.info("Tier 2 Persistence: restored %d subsystem states from prior run", tier2_restored)
    else:
        logger.info("Tier 2 Persistence: no prior state found (fresh start)")

    # FIX-4 prep: Inject learning engine references into agents
    # (engines exist from Layers 6+13 but were never accessible in _learn())
    for agent in ctx.agents:
        agent._frl_engine = ctx.frl_engines.get(agent.unique_id)
        agent._vdn_engine = ctx.vdn_engines.get(agent.unique_id)
        agent._maml_learner = ctx.maml_learner
        agent._transfer_engine = ctx.transfer_engine
        agent._imitation_learner = ctx.imitation
