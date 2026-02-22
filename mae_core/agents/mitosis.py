"""Agent Mitosis - Autopoietic production loop for Mae's colony.

When an agent is healthy and the colony has capacity, it can "divide" --
spawning a child agent with a mutated epigenome. The parent's experience
seeds the child's initial state. This is genuine autopoietic production:
the organism's own processes produce new functional components.

Like biological cell division: the cell's own health signals trigger
mitosis, the daughter cell inherits DNA (genome) with slightly different
gene expression (epigenome), and receives cytoplasmic inheritance
(memory seeding) from the parent.

Cadenced every 13 steps (Fibonacci). Advisory only -- logs, never blocks.
Law 5 (Stem Cell) + Law 6 (Autopoietic Closure).
"""

from __future__ import annotations

import logging
import random
from typing import Any

from mae_core.agents.stem_cell import (
    ROLE_PROFILES,
    AgentEpigenome,
    DEFAULT_GENOME,
    redifferentiate,
)

logger = logging.getLogger(__name__)

# EventBus channel for mitosis events
CH_AGENT_MITOSIS = "stem_cell.mitosis"

# Thresholds
HEALTH_THRESHOLD = 0.7       # Parent must be healthier than this to divide
MIN_HISTORY = 13             # Minimum reward_history entries before evaluation
CADENCE = 13                 # Check every 13 steps (Fibonacci)
MUTATION_RATE = 0.15         # Probability of each gene mutating during division
MEMORY_SEED_SIZE = 5         # Number of recent rewards to seed into child


class MitosisMonitor:
    """Monitors colony health and capacity for agent production.

    Like growth factor signaling in biology -- when conditions are right
    (healthy parent, colony below capacity, roles needed), a healthy agent
    divides. The child carries the same genome with a mutated epigenome.

    This closes the autopoietic loop: agent processes (health evaluation,
    role distribution sensing) produce new agents that participate in those
    same processes.
    """

    def __init__(
        self,
        model: Any,
        stem_cell_registry: Any,
        holon_registry: Any,
        event_bus: Any = None,
        max_agents: int = 10,
        shared_systems: dict[str, Any] | None = None,
    ) -> None:
        self._model = model
        self._registry = stem_cell_registry
        self._holon_registry = holon_registry
        self._event_bus = event_bus
        self._max_agents = max_agents
        self._shared = shared_systems or {}
        self._birth_count: int = 0
        self._last_check_step: int = 0

    def step(self, current_step: int = 0) -> None:
        """Run mitosis check. Called from model step hook."""
        if current_step % CADENCE != 0 or current_step == 0:
            return
        if current_step == self._last_check_step:
            return
        self._last_check_step = current_step
        self._check_mitosis(current_step)

    def _agent_health(self, agent: Any) -> float | None:
        """Compute health from reward_history. Returns None if insufficient data."""
        history = getattr(agent, "reward_history", None)
        if history is None or len(history) < MIN_HISTORY:
            return None
        recent = list(history)[-MIN_HISTORY:]
        total = sum(recent)
        return max(0.0, min(1.0, (total / len(recent) + 1.0) / 2.0))

    def _colony_at_capacity(self) -> bool:
        """Check if colony is at or above max agents."""
        return len(self._registry._agents) >= self._max_agents

    def _pick_parent(self) -> tuple[str, Any] | None:
        """Find the healthiest non-locked agent eligible for mitosis."""
        best_id, best_agent, best_health = None, None, 0.0
        for agent_id, agent in self._registry._agents.items():
            epi = self._registry.get_epigenome(agent_id)
            if epi is None or epi.locked:
                continue
            health = self._agent_health(agent)
            if health is None or health < HEALTH_THRESHOLD:
                continue
            if health > best_health:
                best_id, best_agent, best_health = agent_id, agent, health
        if best_id is not None:
            return (best_id, best_agent)
        return None

    def _mutate_epigenome(self, parent_role: str) -> str:
        """Create a child role by possibly mutating the parent's role.

        With MUTATION_RATE probability, the child gets a different role
        from the available profiles. Otherwise inherits parent's role.
        This creates genuine novelty -- not just copying.
        """
        if random.random() < MUTATION_RATE:
            available = [r for r in ROLE_PROFILES if r != parent_role]
            if available:
                return random.choice(available)
        return parent_role

    def _seed_memory(self, parent: Any, child: Any) -> None:
        """Transfer a small excerpt of parent experience to child.

        Like cytoplasmic inheritance -- the daughter cell gets some of the
        parent's molecular machinery, not the full genome expression state.
        """
        parent_history = getattr(parent, "reward_history", None)
        if parent_history and len(parent_history) >= MEMORY_SEED_SIZE:
            seed = list(parent_history)[-MEMORY_SEED_SIZE:]
            child_history = getattr(child, "reward_history", None)
            if child_history is not None:
                for r in seed:
                    child_history.append(r)

    def _create_child(self, parent: Any, parent_id: str, step: int) -> Any:
        """Instantiate a new MycelialAgent as the child of a mitosis event."""
        from mae_core.agents.mycelial_agent import MycelialAgent

        parent_epi = self._registry.get_epigenome(parent_id)
        parent_role = parent_epi.role if parent_epi else "STEM"
        child_role = self._mutate_epigenome(parent_role)

        # Build child config from role profile
        child_config = dict(DEFAULT_GENOME.get_defaults())
        child_config.update(ROLE_PROFILES.get(child_role, {}))

        # Create child with shared systems (Mesa auto-registers in model.agents)
        child = MycelialAgent(
            self._model,
            agent_type="mycelial",
            agent_config=child_config,
            signal_bus=self._shared.get("signal_bus"),
            stigmergy_env=self._shared.get("stigmergy"),
            gnn_communicator=self._shared.get("gnn_communicator"),
            knowledge_base=self._shared.get("knowledge_base"),
            transfer_engine=self._shared.get("transfer_engine"),
            maml_learner=self._shared.get("maml_learner"),
            holon_registry=self._holon_registry,
            somatic_map=self._shared.get("somatic_map"),
        )

        # Register in StemCellRegistry and apply role
        self._registry.register(child)
        child_id = str(child.unique_id)
        redifferentiate(child, child_role, registry=self._registry, step=step)

        # Register in HolonRegistry (child of colony, like all agents)
        self._holon_registry.register(
            child_id, holon_type="agent", parent_id="colony",
        )
        child._holon_parent_id = "colony"

        # Register in substrate if available
        substrate = self._shared.get("substrate")
        if substrate is not None and hasattr(substrate, "register_agent"):
            substrate.register_agent(child_id)

        # Seed child with parent's recent experience
        self._seed_memory(parent, child)

        # Track creation
        if hasattr(self._model, "track_agent_created"):
            self._model.track_agent_created(child)

        return child

    def _check_mitosis(self, current_step: int) -> None:
        """Evaluate conditions and trigger mitosis if appropriate."""
        if self._colony_at_capacity():
            return

        candidate = self._pick_parent()
        if candidate is None:
            return

        parent_id, parent = candidate
        parent_health = self._agent_health(parent)

        child = self._create_child(parent, parent_id, current_step)
        child_id = str(child.unique_id)
        child_epi = self._registry.get_epigenome(child_id)
        child_role = child_epi.role if child_epi else "STEM"

        self._birth_count += 1

        logger.info(
            "Mitosis: parent %s (health=%.2f) -> child %s (role=%s) at step %d",
            parent_id, parent_health or 0.0, child_id, child_role, current_step,
        )

        self._emit(parent_id, child_id, child_role, parent_health or 0.0, current_step)

    def _emit(
        self, parent_id: str, child_id: str, child_role: str,
        parent_health: float, step: int,
    ) -> None:
        """Publish mitosis event on EventBus."""
        if self._event_bus is None:
            return
        self._event_bus.publish(CH_AGENT_MITOSIS, {
            "parent_id": parent_id,
            "child_id": child_id,
            "child_role": child_role,
            "parent_health": round(parent_health, 3),
            "step": step,
            "total_births": self._birth_count,
        })

    def get_statistics(self) -> dict[str, Any]:
        """Monitoring statistics."""
        return {
            "total_births": self._birth_count,
            "max_agents": self._max_agents,
            "current_agents": len(self._registry._agents),
            "at_capacity": self._colony_at_capacity(),
            "last_check_step": self._last_check_step,
        }
