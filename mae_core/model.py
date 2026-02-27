"""MycelialModel - The soil in which Mae's agents grow.

Central orchestrator for the multi-agent system. Provides
infrastructure (backbone, scheduling, coordination) without
centrally controlling individual agents.

Ported from v5-pivot src/core/model.py with:
- Redis replaced by EventBus + StateStore
- ChromaDB replaced by VectorStore (FAISS)
- Mesa 3.4 API (auto agent tracking, universal time)
"""

from __future__ import annotations

import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import mesa

from mae_core.backbone.event_bus import EventBus
from mae_core.backbone.state_store import StateStore
from mae_core.backbone.vector_store import VectorStore

logger = logging.getLogger(__name__)


class MycelialModel(mesa.Model):
    """The soil in which Mae's mycelial agent network grows.

    Provides shared resources (backbone, storage, communication) but
    does not centrally control agents. Agents are autonomous and
    communicate peer-to-peer via the EventBus.

    Mesa 3.4 features used:
    - model.agents: Auto-tracked AgentSet with shuffle_do(), select(), group_by()
    - model.time: Universal time tracking (increments each step)
    - Agent auto-registration on creation (no manual scheduler.add())
    """

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        state_store: Optional[StateStore] = None,
        vector_store: Optional[VectorStore] = None,
        persist_dir: Optional[Path] = None,
        model_params: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        # Persistence directory
        self._persist_dir = Path(persist_dir) if persist_dir else Path("data/midge")

        # Initialize backbone (no external servers needed)
        self.event_bus = event_bus or EventBus()
        self.state_store = state_store or StateStore(
            persist_path=self._persist_dir / "state_store.json"
        )
        self.vector_store = vector_store or VectorStore(
            embedding_dim=128,
            persist_path=self._persist_dir / "vector_store.pkl",
        )

        # Model parameters
        self.model_params = model_params or {}

        # System state
        self.start_time: Optional[float] = None
        self.is_running = False

        # Agent registries (beyond Mesa's auto-tracking)
        self.hibernated_agents: dict[int, Any] = {}

        # Learning engine references (set by components later)
        self.frl_engine: Any = None
        self.vdn_engine: Any = None
        self.haven_coordinator: Any = None

        # Step hooks (coordination systems, triad audits, etc.)
        self._step_hooks: list[Any] = []

        # Parallel agent stepping — agents run concurrently in threads.
        # GIL serializes pure Python, but C extensions (numpy, FAISS) and
        # I/O (async API calls) release the GIL for true parallelism.
        self._parallel = True
        self._agent_executor: ThreadPoolExecutor | None = None

        # Performance metrics
        self.total_agents_created = 0
        self.total_agents_removed = 0
        self.global_reward_history: list[float] = []

        # Load persisted state if available
        self.state_store.load()

        logger.info("MycelialModel initialized (backbone: EventBus + StateStore + VectorStore)")

    # =========================================================================
    # Main Simulation Loop
    # =========================================================================

    def step(self) -> None:
        """Execute one step of the model.

        1. Mark running
        2. Activate all agents (parallel or sequential)
        3. Compute global rewards (if VDN enabled)
        4. Update HAVEN risk assessment (if enabled)
        5. Persist system state
        """
        if not self.is_running:
            self.is_running = True
            self.start_time = time.time()

        # Run step hooks (coordination systems, triad audits, etc.)
        for hook in self._step_hooks:
            try:
                hook()
            except Exception:
                logger.exception("Error in step hook %s", hook)

        # Activate all agents — parallel when enabled, sequential fallback
        if self._parallel and len(self.agents) > 1:
            self._parallel_step()
        else:
            self.agents.shuffle_do("step")

        # Compute and distribute global rewards (VDN)
        if self.vdn_engine is not None:
            global_reward = self._compute_global_reward()
            self.global_reward_history.append(global_reward)
            self._distribute_rewards(global_reward)

        # Update HAVEN risk assessment
        if self.haven_coordinator is not None:
            self._update_risk_assessment()

        # Persist model state periodically
        if int(self.time) % 10 == 0:
            self._save_model_state()

        if int(self.time) % 100 == 0:
            logger.info(
                "Step %d (agents: %d)", int(self.time), len(self.agents)
            )

    def _parallel_step(self) -> None:
        """Step all agents concurrently using a thread pool.

        Agents run in shuffled order within a ThreadPoolExecutor.
        EventBus is thread-safe (locked publish/write_to_stream).
        Each agent only mutates its own state during step().
        """
        agent_list = list(self.agents)
        random.shuffle(agent_list)

        if self._agent_executor is None:
            workers = min(len(agent_list), 8)
            self._agent_executor = ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="mae-agent",
            )

        def _safe_step(agent: Any) -> None:
            try:
                agent.step()
            except Exception:
                logger.exception("Error in parallel step for agent %s", agent.unique_id)

        futures = [self._agent_executor.submit(_safe_step, a) for a in agent_list]
        for f in futures:
            f.result()  # propagate exceptions, wait for all

    def _compute_global_reward(self) -> float:
        """Compute system-wide global reward (sum of agent rewards)."""
        total = 0.0
        for agent in self.agents:
            if hasattr(agent, "last_reward"):
                total += agent.last_reward
        return total

    def _distribute_rewards(self, global_reward: float) -> None:
        """Distribute global reward to agents using VDN."""
        for agent in self.agents:
            if hasattr(agent, "get_local_reward"):
                try:
                    agent.get_local_reward(global_reward)
                except Exception:
                    logger.exception("Error distributing reward to agent %s", agent.unique_id)

    def _update_risk_assessment(self) -> None:
        """Update HAVEN risk assessment for all agents."""
        if self.haven_coordinator is None:
            return
        try:
            self.haven_coordinator.assess_system_risk()
            self.haven_coordinator.detect_policy_contagion()
        except Exception:
            logger.exception("Error in risk assessment")

    def _save_model_state(self) -> None:
        """Persist model state to StateStore."""
        state_data = {
            "step": int(self.time),
            "active_agents": len(self.agents),
            "hibernated_agents": len(self.hibernated_agents),
            "total_agents_created": self.total_agents_created,
            "params": self.model_params,
            "is_running": self.is_running,
        }
        self.state_store.set_key_value("model:state:current", state_data)

    # =========================================================================
    # Agent Management
    # =========================================================================

    def track_agent_created(self, agent: Any) -> None:
        """Track agent creation beyond Mesa's auto-registration.

        Note: Mesa 3.4 auto-adds agents to model.agents on creation.
        Call this for additional metrics tracking if needed.
        """
        self.total_agents_created += 1
        logger.debug(
            "Tracked agent %s (total active: %d)",
            agent.unique_id,
            len(self.agents),
        )

    def hibernate_agent(self, agent: Any) -> None:
        """Move an agent to hibernation (inactive but preserved)."""
        self.hibernated_agents[agent.unique_id] = agent
        agent.remove()  # Mesa 3.4: removes from model.agents
        logger.debug("Hibernated agent %s", agent.unique_id)

    def wake_agent(self, unique_id: int) -> Optional[Any]:
        """Wake a hibernated agent."""
        agent = self.hibernated_agents.pop(unique_id, None)
        if agent is not None:
            logger.debug("Woke agent %s", unique_id)
        return agent

    def get_agent_state(self, agent_id: int) -> Optional[dict[str, Any]]:
        """Retrieve agent state from StateStore."""
        return self.state_store.get_key_value(f"agent:state:{agent_id}")

    # =========================================================================
    # Engine Integration
    # =========================================================================

    def add_step_hook(self, hook: Any) -> None:
        """Register a callable to run at the start of each step.

        Used for coordination systems (circadian, endocrine) and
        periodic checks (triad audit) that need to tick each step.
        """
        self._step_hooks.append(hook)
        logger.debug("Step hook registered: %s", hook)

    def set_frl_engine(self, frl_engine: Any) -> None:
        """Register the Federated Reinforcement Learning engine."""
        self.frl_engine = frl_engine
        logger.info("FRL engine registered with model")

    def set_vdn_engine(self, vdn_engine: Any) -> None:
        """Register the Value-Decomposition Networks engine."""
        self.vdn_engine = vdn_engine
        logger.info("VDN engine registered with model")

    def set_haven_coordinator(self, haven_coordinator: Any) -> None:
        """Register the HAVEN risk coordinator."""
        self.haven_coordinator = haven_coordinator
        logger.info("HAVEN coordinator registered with model")

    # =========================================================================
    # Simulation Control
    # =========================================================================

    def run(self, num_steps: int) -> None:
        """Run the model for a specified number of steps."""
        logger.info("Starting simulation run for %d steps", num_steps)
        try:
            for _ in range(num_steps):
                self.step()
            logger.info("Simulation run completed")
        except KeyboardInterrupt:
            logger.warning("Simulation interrupted at step %d", int(self.time))
            self.shutdown()
        except Exception:
            logger.exception("Simulation error at step %d", int(self.time))
            self.shutdown()
            raise

    def save_all_agent_states(self) -> int:
        """Persist every agent's accumulated state to StateStore.

        Returns the number of agents saved.
        """
        saved = 0
        for agent in self.agents:
            if hasattr(agent, "serialize_state"):
                try:
                    state = agent.serialize_state()
                    self.state_store.set_key_value(
                        f"agent:state:{agent.unique_id}", state,
                    )
                    saved += 1
                except Exception:
                    logger.exception("Failed to save state for agent %s", agent.unique_id)
        if saved:
            logger.info("Saved state for %d agents", saved)
        return saved

    def load_agent_state(self, agent_id: int) -> Optional[dict[str, Any]]:
        """Load persisted state for a specific agent."""
        return self.state_store.get_key_value(f"agent:state:{agent_id}")

    # =========================================================================
    # Tier 2 Persistence (Subsystem Learned State)
    # =========================================================================

    def save_subsystem_states(
        self,
        memory_coordinators: dict[str, Any] | None = None,
        world_models: dict[str, Any] | None = None,
        vdn_engines: dict[str, Any] | None = None,
        frl_engines: dict[str, Any] | None = None,
        shared_systems: dict[str, Any] | None = None,
    ) -> int:
        """Persist Tier 2 learned state for all subsystems.

        Iterates per-agent and shared subsystems, calls serialize(),
        and stores metadata in StateStore under subsystem:* keys.

        Returns the number of subsystems saved.
        """
        saved = 0
        subsystem_dir = self._persist_dir / "subsystems"

        # Per-agent memory coordinators
        if memory_coordinators:
            for agent_id, mc in memory_coordinators.items():
                if hasattr(mc, "serialize"):
                    try:
                        agent_dir = subsystem_dir / "agents" / str(agent_id)
                        meta = mc.serialize(agent_dir)
                        self.state_store.set_key_value(
                            f"subsystem:agent:{agent_id}:memory", meta
                        )
                        saved += 1
                    except Exception:
                        logger.exception("Failed to save memory for agent %s", agent_id)

        # Per-agent world models
        if world_models:
            for agent_id, wm in world_models.items():
                if hasattr(wm, "serialize"):
                    try:
                        agent_dir = subsystem_dir / "agents" / str(agent_id)
                        meta = wm.serialize(agent_dir)
                        self.state_store.set_key_value(
                            f"subsystem:agent:{agent_id}:world_model", meta
                        )
                        saved += 1
                    except Exception:
                        logger.exception("Failed to save world model for agent %s", agent_id)

        # Per-agent VDN
        if vdn_engines:
            for agent_id, vdn in vdn_engines.items():
                if hasattr(vdn, "serialize"):
                    try:
                        agent_dir = subsystem_dir / "agents" / str(agent_id)
                        meta = vdn.serialize(agent_dir)
                        self.state_store.set_key_value(
                            f"subsystem:agent:{agent_id}:vdn", meta
                        )
                        saved += 1
                    except Exception:
                        logger.exception("Failed to save VDN for agent %s", agent_id)

        # Per-agent FRL
        if frl_engines:
            for agent_id, frl in frl_engines.items():
                if hasattr(frl, "serialize"):
                    try:
                        agent_dir = subsystem_dir / "agents" / str(agent_id)
                        meta = frl.serialize(agent_dir)
                        self.state_store.set_key_value(
                            f"subsystem:agent:{agent_id}:frl", meta
                        )
                        saved += 1
                    except Exception:
                        logger.exception("Failed to save FRL for agent %s", agent_id)

        # Shared systems (knowledge_base, maml)
        if shared_systems:
            shared_dir = subsystem_dir / "shared"
            for name, system in shared_systems.items():
                if hasattr(system, "serialize"):
                    try:
                        meta = system.serialize(shared_dir)
                        self.state_store.set_key_value(
                            f"subsystem:shared:{name}", meta
                        )
                        saved += 1
                    except Exception:
                        logger.exception("Failed to save shared system %s", name)

        if saved:
            logger.info("Saved Tier 2 state for %d subsystems", saved)
        return saved

    def load_subsystem_metadata(self, key: str) -> dict | None:
        """Load Tier 2 metadata for a subsystem from StateStore."""
        return self.state_store.get_key_value(key)

    def shutdown(self) -> None:
        """Gracefully shutdown and persist all state."""
        logger.info("Shutting down MycelialModel...")
        self.is_running = False

        # Shut down thread pool
        if self._agent_executor is not None:
            self._agent_executor.shutdown(wait=False)
            self._agent_executor = None

        # Tier 2: save subsystem states if references available
        if hasattr(self, "_tier2_refs"):
            self.save_subsystem_states(**self._tier2_refs)

        self.save_all_agent_states()
        self._save_model_state()
        self.state_store.close()
        self.vector_store.close()
        self.event_bus.close()
        logger.info("MycelialModel shutdown complete")

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_system_stats(self) -> dict[str, Any]:
        """Get comprehensive system statistics."""
        runtime = time.time() - self.start_time if self.start_time else 0
        step_count = int(self.time)

        return {
            "current_step": step_count,
            "runtime_seconds": runtime,
            "steps_per_second": step_count / runtime if runtime > 0 else 0,
            "active_agents": len(self.agents),
            "hibernated_agents": len(self.hibernated_agents),
            "total_agents_created": self.total_agents_created,
            "total_agents_removed": self.total_agents_removed,
            "global_reward_avg": (
                sum(self.global_reward_history[-100:])
                / max(1, len(self.global_reward_history[-100:]))
            ),
            "engines_enabled": {
                "frl": self.frl_engine is not None,
                "vdn": self.vdn_engine is not None,
                "haven": self.haven_coordinator is not None,
            },
            "backbone": {
                "event_bus": self.event_bus.get_stats(),
                "state_store_keys": len(self.state_store.keys()),
                "vector_store": self.vector_store.get_collection_stats(),
            },
        }
