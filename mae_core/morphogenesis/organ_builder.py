"""Organ builder - blueprint design and agent team spawning.

Biological analogy: Morphogenesis in multicellular organisms creates
specialized tissues (organs) from undifferentiated cells. Problem
characteristics determine what kind of organ forms - like how
morphogen gradients guide cell differentiation in embryos.

An "organ" in Mae is a team of specialized agents that forms to
solve a specific class of problems, works together, and may dissolve
when the problem is resolved.

Blueprint design: Problem analysis drives composition (how many of
which types), topology (how they connect), and coordination protocol
(how they decide together).

Agent spawning: Uses Mesa 3.4 dynamic agent creation. Agents are
created via the model and auto-registered, then connected through
substrate topology.

Sub-modules:
  organ_builder_design.py — design_organ() and on_system_senescent() helpers
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from mae_core.morphogenesis.organ_builder_design import (
    design_organ as _design_organ_fn,
    on_system_senescent as _on_system_senescent_fn,
    CH_REBUILD_REQUESTED,
)

logger = logging.getLogger(__name__)


class OrganStatus(Enum):
    """Organ lifecycle status."""

    GROWING = "growing"  # Being constructed
    ACTIVE = "active"  # Operating
    DISSOLVING = "dissolving"  # Being deconstructed
    DISSOLVED = "dissolved"  # No longer active


class CoordinationProtocol(Enum):
    """How agents within an organ coordinate decisions."""

    CONSENSUS = "consensus"  # All agents vote
    HIERARCHICAL = "hierarchical"  # Coordinator decides
    AUCTION = "auction"  # Agents bid for tasks


class OrganTopology(Enum):
    """How agents within an organ are connected."""

    MESH = "mesh"  # All-to-all (small teams)
    STAR = "star"  # Hub-spoke (coordinator-centered)
    RING = "ring"  # Circular chain (ordered pipeline)
    HIERARCHICAL = "hierarchical"  # Two-level coordination


@dataclass
class ProblemSignature:
    """Characterizes a class of problems for morphogenesis.

    Morphogenesis examines these dimensions to decide what kind
    of organ (agent team) to build.
    """

    signature_id: str = ""
    coordination_level: float = 0.5  # 0=independent, 1=tightly coupled
    exploration_level: float = 0.5  # 0=exploit known, 1=explore new
    complexity: float = 0.5  # 0=simple, 1=highly complex
    risk_level: float = 0.3  # 0=safe, 1=dangerous
    temporal_pattern: str = "persistent"  # "episodic" or "persistent"
    sparse_rewards: bool = False
    domain: str = "general"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.signature_id:
            self.signature_id = f"sig_{uuid.uuid4().hex[:8]}"

    def similarity(self, other: ProblemSignature) -> float:
        """How similar is this problem to another? (0-1)."""
        dims = [
            abs(self.coordination_level - other.coordination_level),
            abs(self.exploration_level - other.exploration_level),
            abs(self.complexity - other.complexity),
            abs(self.risk_level - other.risk_level),
        ]
        return 1.0 - (sum(dims) / len(dims))


@dataclass
class OrganBlueprint:
    """Specification for an organ (agent team) to be built.

    Describes composition (agent types and counts), topology
    (how they connect), coordination (how they decide), and
    lifecycle parameters.
    """

    organ_id: str = ""
    name: str = "unnamed_organ"
    purpose: str = ""  # What problem class this organ solves
    composition: dict[str, int] = field(default_factory=dict)  # {agent_type: count}
    topology: OrganTopology = OrganTopology.MESH
    protocol: CoordinationProtocol = CoordinationProtocol.CONSENSUS
    transient: bool = False  # Dissolves after task completion
    max_lifetime: Optional[float] = None  # Seconds (None = permanent)
    max_size: int = 50
    performance_threshold: float = 0.3  # Below this, organ is underperforming
    problem_signature: Optional[ProblemSignature] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.organ_id:
            self.organ_id = f"organ_{uuid.uuid4().hex[:8]}"

    @property
    def total_agents(self) -> int:
        return sum(self.composition.values())

    def validate(self) -> bool:
        """Check blueprint validity."""
        if self.total_agents <= 0:
            return False
        if self.total_agents > self.max_size:
            return False
        return True


class Organ:
    """A living team of agents working together.

    Organs are created by OrganBuilder from blueprints, contain
    agent references, track performance, and can be dissolved.
    """

    def __init__(self, blueprint: OrganBlueprint) -> None:
        self.blueprint = blueprint
        self.organ_id = blueprint.organ_id
        self.status = OrganStatus.GROWING
        self.agents: list[dict[str, Any]] = []
        self.agent_ids: list[int] = []  # Mesa agent unique_ids
        self.coordinator_id: Optional[int] = None

        # Performance tracking
        self.performance_history: list[float] = []
        self.task_count = 0
        self.success_count = 0

        # Lifecycle
        self.creation_time = time.time()
        self.last_active_time = self.creation_time

    def add_agent(self, agent_id: int, agent_type: str, **kwargs: Any) -> None:
        """Register an agent with this organ."""
        self.agents.append(
            {"agent_id": agent_id, "agent_type": agent_type, **kwargs}
        )
        self.agent_ids.append(agent_id)

        # First coordinator-type agent becomes the coordinator
        if self.coordinator_id is None and agent_type == "coordinator":
            self.coordinator_id = agent_id

    def record_task(self, success: bool) -> None:
        """Record a task outcome."""
        self.task_count += 1
        if success:
            self.success_count += 1
        self.last_active_time = time.time()

    def evaluate_performance(self) -> float:
        """Evaluate organ performance (success rate)."""
        if self.task_count == 0:
            return 0.5  # Neutral when no data
        perf = self.success_count / self.task_count
        self.performance_history.append(perf)
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
        return perf

    def should_dissolve(self) -> bool:
        """Check if this organ should be dissolved."""
        # Transient organs dissolve after no activity
        if self.blueprint.transient:
            idle_time = time.time() - self.last_active_time
            if idle_time > 300:  # 5 minutes of no activity
                return True

        # Lifetime exceeded
        if self.blueprint.max_lifetime:
            age = time.time() - self.creation_time
            if age > self.blueprint.max_lifetime:
                return True

        # Persistent underperformance
        if len(self.performance_history) >= 10:
            recent = self.performance_history[-10:]
            avg = sum(recent) / len(recent)
            if avg < self.blueprint.performance_threshold:
                return True

        return False

    def get_statistics(self) -> dict[str, Any]:
        """Get organ statistics."""
        return {
            "organ_id": self.organ_id,
            "name": self.blueprint.name,
            "status": self.status.value,
            "agent_count": len(self.agents),
            "agent_ids": list(self.agent_ids),
            "coordinator_id": self.coordinator_id,
            "task_count": self.task_count,
            "success_count": self.success_count,
            "success_rate": (
                self.success_count / self.task_count if self.task_count > 0 else 0.0
            ),
            "age_seconds": time.time() - self.creation_time,
            "topology": self.blueprint.topology.value,
            "protocol": self.blueprint.protocol.value,
            "transient": self.blueprint.transient,
        }


class OrganBuilder:
    """Designs organ blueprints and spawns agent teams.

    Analyzes problem signatures to determine what kind of team to
    build, then creates agents via the Mesa model and connects them
    through the substrate.

    The builder maintains a registry of active organs and handles
    their full lifecycle.
    """

    def __init__(
        self,
        agent_factory: Optional[Callable[..., Any]] = None,
        event_bus: Optional[Any] = None,
    ) -> None:
        """Initialize the organ builder.

        Args:
            agent_factory: Callable that creates agents. Should accept
                (model, agent_type, organ_id, **kwargs) and return an
                agent with a unique_id attribute. If None, organs are
                created as metadata-only (no actual Mesa agents).
            event_bus: EventBus instance for pub/sub integration. If
                provided, OrganBuilder subscribes to CH_SENESCENT so that
                it can prune organs and request rebuilds when systems
                reach end-of-life. If None, operates without EventBus
                (backward-compatible).
        """
        self._agent_factory = agent_factory
        self._active_organs: dict[str, Organ] = {}
        self._creation_history: list[dict[str, Any]] = []

        self._event_bus = event_bus
        if self._event_bus is not None:
            self._event_bus.register_callback(
                "emergent.system_senescent", self._on_system_senescent
            )

    # =========================================================================
    # Blueprint Design
    # =========================================================================

    def design_organ(
        self,
        signature: ProblemSignature,
        name: Optional[str] = None,
    ) -> OrganBlueprint:
        """Design an organ blueprint based on problem characteristics.

        Composition rules (inspired by v5-pivot OrganBuilder):
        - High coordination -> add coordinators
        - High exploration -> add explorers
        - Sparse rewards or high complexity -> add specialists
        - High risk -> add risk managers
        - Always: generalists fill remaining slots
        """
        composition: dict[str, int] = {}

        # Coordinators: needed when tight coordination required
        if signature.coordination_level > 0.6:
            coord_count = max(1, int(signature.coordination_level * 5))
            composition["coordinator"] = coord_count

        # Explorers: needed when exploration is high
        if signature.exploration_level > 0.5:
            explore_count = max(1, int(signature.exploration_level * 8))
            composition["explorer"] = explore_count

        # Specialists: for complex or sparse-reward problems
        if signature.sparse_rewards or signature.complexity > 0.6:
            spec_count = max(2, int(signature.complexity * 6))
            composition["specialist"] = spec_count

        # Risk managers: for dangerous environments
        if signature.risk_level > 0.6:
            risk_count = max(1, int(signature.risk_level * 3))
            composition["risk_manager"] = risk_count

        # Generalists: fill to minimum team size
        current_total = sum(composition.values())
        min_team = max(3, current_total // 4)  # At least 3, or 1/4 of specialized
        if current_total < min_team:
            composition["generalist"] = min_team - current_total
        elif "generalist" not in composition:
            composition["generalist"] = max(1, current_total // 4)

        # Select topology based on coordination level
        if signature.coordination_level > 0.8:
            topology = OrganTopology.HIERARCHICAL
        elif signature.coordination_level > 0.5:
            topology = OrganTopology.STAR
        else:
            topology = OrganTopology.MESH

        # Select protocol
        if signature.coordination_level > 0.7:
            protocol = CoordinationProtocol.HIERARCHICAL
        elif signature.exploration_level > 0.6:
            protocol = CoordinationProtocol.AUCTION
        else:
            protocol = CoordinationProtocol.CONSENSUS

        # Lifecycle
        transient = signature.temporal_pattern == "episodic"
        max_lifetime = 3600.0 if transient else None

        organ_name = name or f"organ_{signature.domain}_{uuid.uuid4().hex[:4]}"

        blueprint = OrganBlueprint(
            name=organ_name,
            purpose=f"Solve {signature.domain} problems (complexity={signature.complexity:.1f})",
            composition=composition,
            topology=topology,
            protocol=protocol,
            transient=transient,
            max_lifetime=max_lifetime,
            problem_signature=signature,
        )

        logger.info(
            "Designed organ '%s': %d agents, %s topology, %s protocol",
            blueprint.name,
            blueprint.total_agents,
            blueprint.topology.value,
            blueprint.protocol.value,
        )
        return blueprint

    # =========================================================================
    # Organ Growth (Agent Spawning)
    # =========================================================================

    def grow_organ(
        self,
        blueprint: OrganBlueprint,
        model: Optional[Any] = None,
        substrate: Optional[Any] = None,
    ) -> Organ:
        """Grow an organ by spawning agents according to the blueprint.

        If agent_factory is available and model is provided, creates real
        Mesa agents. Otherwise, creates metadata-only organs.

        Args:
            blueprint: The organ specification.
            model: Mesa model for agent creation.
            substrate: MycelialSubstrate for topology registration.
        """
        if not blueprint.validate():
            raise ValueError(f"Invalid blueprint: {blueprint.organ_id}")

        organ = Organ(blueprint)

        # Spawn agents for each type in composition
        for agent_type, count in blueprint.composition.items():
            for i in range(count):
                if self._agent_factory and model:
                    # Create real Mesa agent
                    agent = self._agent_factory(
                        model=model,
                        agent_type=agent_type,
                        organ_id=organ.organ_id,
                    )
                    organ.add_agent(agent.unique_id, agent_type)

                    # Register with substrate if available
                    if substrate:
                        substrate.register_agent(agent.unique_id)
                else:
                    # Metadata-only (no actual agent)
                    fake_id = hash(f"{organ.organ_id}_{agent_type}_{i}") % 100000
                    organ.add_agent(fake_id, agent_type)

        # Connect agents within organ through substrate
        if substrate and organ.agent_ids:
            self._connect_organ_agents(organ, substrate)

        organ.status = OrganStatus.ACTIVE
        self._active_organs[organ.organ_id] = organ

        self._creation_history.append(
            {
                "organ_id": organ.organ_id,
                "name": blueprint.name,
                "agent_count": len(organ.agents),
                "time": time.time(),
            }
        )

        logger.info(
            "Organ '%s' grown: %d agents, status=%s",
            organ.organ_id,
            len(organ.agents),
            organ.status.value,
        )
        return organ

    def dissolve_organ(
        self,
        organ_id: str,
        model: Optional[Any] = None,
        substrate: Optional[Any] = None,
    ) -> bool:
        """Dissolve an organ, removing its agents."""
        organ = self._active_organs.get(organ_id)
        if organ is None:
            return False

        organ.status = OrganStatus.DISSOLVING

        # Remove agents from model and substrate
        for agent_id in organ.agent_ids:
            if substrate:
                substrate.deregister_agent(agent_id)
            if model:
                # Find and remove Mesa agent
                for agent in list(model.agents):
                    if agent.unique_id == agent_id:
                        agent.remove()
                        break

        organ.status = OrganStatus.DISSOLVED
        del self._active_organs[organ_id]

        logger.info("Organ '%s' dissolved", organ_id)
        return True

    def prune_organs(
        self,
        model: Optional[Any] = None,
        substrate: Optional[Any] = None,
    ) -> list[str]:
        """Dissolve organs that should be removed."""
        to_dissolve = [
            oid
            for oid, organ in self._active_organs.items()
            if organ.should_dissolve()
        ]
        for oid in to_dissolve:
            self.dissolve_organ(oid, model=model, substrate=substrate)
        return to_dissolve

    # =========================================================================
    # Queries
    # =========================================================================

    def get_organ(self, organ_id: str) -> Optional[Organ]:
        return self._active_organs.get(organ_id)

    def get_all_organs(self) -> list[Organ]:
        return list(self._active_organs.values())

    @property
    def active_organ_count(self) -> int:
        return len(self._active_organs)

    def get_statistics(self) -> dict[str, Any]:
        return {
            "active_organs": len(self._active_organs),
            "total_created": len(self._creation_history),
            "total_agents_in_organs": sum(
                len(o.agents) for o in self._active_organs.values()
            ),
            "organs": [o.get_statistics() for o in self._active_organs.values()],
        }

    # =========================================================================
    # Senescence Integration
    # =========================================================================

    def _on_system_senescent(self, channel: str, message: Any) -> None:
        """Handle a senescent system event from SenescenceManager.

        Triggered when a system's wear reaches 1.0 (end-of-life). Prunes
        any organs that should dissolve, then publishes a rebuild request
        so downstream systems know replacement is needed.

        Args:
            channel: The EventBus channel (always "emergent.system_senescent").
            message: JSON-serialized payload containing at minimum
                ``system_name``, ``wear_level``, ``total_steps_active``,
                and ``step``.
        """
        # Parse the message payload (EventBus delivers JSON strings)
        if isinstance(message, str):
            try:
                payload = json.loads(message)
            except (json.JSONDecodeError, ValueError):
                logger.warning("OrganBuilder: could not parse senescent message: %s", message)
                return
        elif isinstance(message, dict):
            payload = message
        else:
            logger.warning("OrganBuilder: unexpected message type %s", type(message))
            return

        system_name = payload.get("system_name", "unknown")

        # Prune organs that have reached end-of-life
        self.prune_organs()

        # Publish rebuild notification so downstream systems can act
        if self._event_bus is not None:
            self._event_bus.publish(
                CH_REBUILD_REQUESTED,
                {"system_name": system_name, "reason": "senescence"},
            )

        logger.info(
            "OrganBuilder: senescent event for %s — pruned and rebuild requested",
            system_name,
        )

    # =========================================================================
    # Internal
    # =========================================================================

    def _connect_organ_agents(self, organ: Organ, substrate: Any) -> None:
        """Connect agents within an organ through substrate topology."""
        ids = organ.agent_ids
        if len(ids) < 2:
            return

        topo = organ.blueprint.topology

        if topo == OrganTopology.MESH:
            # All-to-all
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    node_a = substrate.topology.get_agent_node(ids[i])
                    node_b = substrate.topology.get_agent_node(ids[j])
                    if node_a and node_b:
                        substrate.topology.add_edge(node_a, node_b)

        elif topo == OrganTopology.STAR:
            # Hub-spoke: coordinator connects to all
            hub = organ.coordinator_id or ids[0]
            hub_node = substrate.topology.get_agent_node(hub)
            if hub_node:
                for aid in ids:
                    if aid != hub:
                        spoke_node = substrate.topology.get_agent_node(aid)
                        if spoke_node:
                            substrate.topology.add_edge(hub_node, spoke_node)

        elif topo == OrganTopology.RING:
            # Circular chain
            for i in range(len(ids)):
                node_a = substrate.topology.get_agent_node(ids[i])
                node_b = substrate.topology.get_agent_node(ids[(i + 1) % len(ids)])
                if node_a and node_b:
                    substrate.topology.add_edge(node_a, node_b)

        elif topo == OrganTopology.HIERARCHICAL:
            # Coordinators mesh with each other, workers connect to first coordinator
            coord_ids = [
                a["agent_id"]
                for a in organ.agents
                if a["agent_type"] == "coordinator"
            ]
            worker_ids = [aid for aid in ids if aid not in coord_ids]

            # Mesh coordinators
            for i in range(len(coord_ids)):
                for j in range(i + 1, len(coord_ids)):
                    na = substrate.topology.get_agent_node(coord_ids[i])
                    nb = substrate.topology.get_agent_node(coord_ids[j])
                    if na and nb:
                        substrate.topology.add_edge(na, nb)

            # Workers to first coordinator
            if coord_ids:
                coord_node = substrate.topology.get_agent_node(coord_ids[0])
                if coord_node:
                    for wid in worker_ids:
                        wnode = substrate.topology.get_agent_node(wid)
                        if wnode:
                            substrate.topology.add_edge(coord_node, wnode)
