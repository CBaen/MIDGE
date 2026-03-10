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

import logging
import time
from typing import Any, Callable, Optional

# Models and enums (no circular dependency — models don't import from here)
from mae_core.morphogenesis.organ_builder_models import (
    CoordinationProtocol,
    Organ,
    OrganBlueprint,
    OrganStatus,
    OrganTopology,
    ProblemSignature,
)

# Design helpers (imports from organ_builder_models, not from here)
from mae_core.morphogenesis.organ_builder_design import (
    design_organ as _design_organ_fn,
    on_system_senescent as _on_system_senescent_fn,
    CH_REBUILD_REQUESTED,
)

logger = logging.getLogger(__name__)


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

        Delegates to organ_builder_design.design_organ().
        """
        return _design_organ_fn(signature, name)

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
        """Handle a senescent system event — delegates to organ_builder_design."""
        _on_system_senescent_fn(self, channel, message)

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
