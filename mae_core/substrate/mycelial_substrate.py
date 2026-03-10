"""MycelialSubstrate - Mae's body.

The substrate IS the physical world Mae's agents inhabit. Like soil
for a mycelial network: it provides the medium through which everything
connects. Topology defines who can reach whom. Nutrient flow feeds agents.
Signal propagation carries messages along the network fabric.

Every system in Mae ultimately connects through substrate:
- OctopusColony registers octopuses here (peer topology)
- SignalBus propagates signals along substrate edges
- Stigmergy deposits pheromones ON the substrate
- GNNCommunicator routes along substrate graph
- Morphogenesis spawns agents INTO the substrate
- Circadian rhythm pulses THROUGH the substrate
- Endocrine hormones flow THROUGH the substrate
- SpatialConsensus reads positions FROM the substrate
- PredictiveFields propagate THROUGH the substrate
- AutoHealing isolates substrate regions

The substrate is not separate infrastructure. It IS Mae.

EventBus channels:
- substrate.agent_registered: New agent joined network
- substrate.agent_deregistered: Agent left network
- substrate.topology_changed: Edge added/removed
- substrate.health_report: Periodic health metrics
- substrate.starvation_alert: Node below resource threshold
- substrate.isolation_detected: Disconnected node found

Sub-modules:
  mycelial_topology.py — topology factory, node allocation, grow/prune, isolate/restore
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Optional

from mae_core.backbone.event_bus import EventBus
from mae_core.substrate.nutrient_flow import FlowConfig, NutrientFlowEngine
from mae_core.substrate.topology import SubstrateTopology, TopologyType
from mae_core.substrate.mycelial_topology import (
    build_topology,
    find_or_create_node,
    auto_connect,
    grow_node as _grow_node,
    prune_node as _prune_node,
    isolate_region as _isolate_region,
    restore_region as _restore_region,
)

logger = logging.getLogger(__name__)

# EventBus channels
CH_AGENT_REGISTERED = "substrate.agent_registered"
CH_AGENT_DEREGISTERED = "substrate.agent_deregistered"
CH_TOPOLOGY_CHANGED = "substrate.topology_changed"
CH_HEALTH_REPORT = "substrate.health_report"
CH_STARVATION_ALERT = "substrate.starvation_alert"
CH_ISOLATION_DETECTED = "substrate.isolation_detected"


class MycelialSubstrate:
    """Mae's body - the mycelial substrate everything connects through.

    Integrates topology (structure), nutrient flow (resources), and
    signal propagation (communication paths) into a unified biological
    fabric. Agents register with the substrate to participate in Mae's
    network.
    """

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        topology_type: str = "scale_free",
        initial_nodes: int = 10,
        topology_params: Optional[dict[str, Any]] = None,
        flow_config: Optional[FlowConfig] = None,
        health_report_interval: int = 10,
    ) -> None:
        self.event_bus = event_bus or EventBus()
        self._health_report_interval = health_report_interval
        self._step_count = 0

        # Build initial topology
        topo_params = topology_params or {}
        self.topology = build_topology(topology_type, initial_nodes, topo_params)

        # Initialize nutrient flow
        self.nutrient_flow = NutrientFlowEngine(self.topology, flow_config)

        # Signal propagation tracking
        self._signal_history: deque[dict[str, Any]] = deque(maxlen=1000)

        # Phase modulation (set by CircadianRhythm)
        self._current_phase: str = "ACTIVE"

        logger.info(
            "MycelialSubstrate initialized: %s topology, %d nodes, %d edges",
            topology_type,
            self.topology.node_count,
            self.topology.edge_count,
        )

    # =========================================================================
    # Agent Registration (integrates with Mesa model)
    # =========================================================================

    def register_agent(
        self,
        agent_id: int,
        position: Optional[tuple[float, float]] = None,
        connect_to: Optional[list[int]] = None,
    ) -> str:
        """Register an agent with the substrate.

        Creates a node (or reuses an empty one) and assigns the agent.
        Optionally connects to specified peers.

        Args:
            agent_id: Mesa agent unique_id.
            position: Optional 2D position.
            connect_to: Optional list of agent_ids to connect to.

        Returns:
            The node_id assigned to this agent.
        """
        # Check if already registered
        existing = self.topology.get_agent_node(agent_id)
        if existing:
            logger.debug("Agent %d already registered at node %s", agent_id, existing)
            return existing

        # Try to reuse an empty node near requested position
        node_id = find_or_create_node(self.topology, agent_id, position)
        self.topology.assign_agent(agent_id, node_id)

        # Connect to specified peers
        if connect_to:
            for peer_id in connect_to:
                peer_node = self.topology.get_agent_node(peer_id)
                if peer_node:
                    self.topology.add_edge(node_id, peer_node)

        # If no explicit connections, connect to nearest occupied nodes
        if not connect_to:
            auto_connect(self.topology, node_id, max_connections=3)

        self.event_bus.publish(
            CH_AGENT_REGISTERED,
            {"agent_id": agent_id, "node_id": node_id},
        )
        logger.debug("Agent %d registered at node %s", agent_id, node_id)
        return node_id

    def deregister_agent(self, agent_id: int) -> Optional[str]:
        """Remove an agent from the substrate.

        The node remains (empty position) but the agent is unassigned.
        Edges are preserved so the position can be reused.
        """
        node_id = self.topology.unassign_agent(agent_id)
        if node_id:
            self.event_bus.publish(
                CH_AGENT_DEREGISTERED,
                {"agent_id": agent_id, "node_id": node_id},
            )
            logger.debug("Agent %d deregistered from node %s", agent_id, node_id)
        return node_id

    # =========================================================================
    # Signal Propagation
    # =========================================================================

    def propagate_signal(
        self,
        source_agent_id: int,
        signal: dict[str, Any],
        max_hops: Optional[int] = None,
        signal_type: str = "general",
    ) -> list[int]:
        """Propagate a signal from a source agent through the substrate.

        Signals travel along edges, attenuating with distance and
        conductance. Returns list of agent_ids that received the signal.

        Args:
            source_agent_id: The sending agent.
            signal: The signal payload.
            max_hops: Maximum propagation distance (None = unlimited).
            signal_type: Type for filtering/prioritization.

        Returns:
            List of agent_ids that received the signal.
        """
        source_node = self.topology.get_agent_node(source_agent_id)
        if source_node is None:
            return []

        reached_agents: list[int] = []
        visited: set[str] = {source_node}
        queue: deque[tuple[str, int, float]] = deque()  # (node_id, hops, strength)

        # Seed with neighbors
        for neighbor_id in self.topology.get_neighbors(source_node):
            source = self.topology.get_node(source_node)
            conductance = source.conductance.get(neighbor_id, 1.0) if source else 1.0
            queue.append((neighbor_id, 1, conductance))

        while queue:
            current_id, hops, strength = queue.popleft()

            if current_id in visited:
                continue
            if max_hops is not None and hops > max_hops:
                continue
            if strength < 0.01:  # Signal too weak
                continue

            visited.add(current_id)
            node = self.topology.get_node(current_id)
            if node and node.agent_id is not None:
                reached_agents.append(node.agent_id)

            # Propagate to neighbors with attenuation
            for neighbor_id in self.topology.get_neighbors(current_id):
                if neighbor_id not in visited:
                    edge_conductance = (
                        node.conductance.get(neighbor_id, 1.0) if node else 1.0
                    )
                    new_strength = strength * edge_conductance * 0.8  # 20% attenuation per hop
                    queue.append((neighbor_id, hops + 1, new_strength))

        # Record propagation event
        self._signal_history.append(
            {
                "source": source_agent_id,
                "type": signal_type,
                "reached": len(reached_agents),
                "hops_max": max_hops,
                "step": self._step_count,
            }
        )

        return reached_agents

    def get_signal_path(
        self, from_agent_id: int, to_agent_id: int
    ) -> Optional[list[str]]:
        """Get the shortest path between two agents through the substrate."""
        from_node = self.topology.get_agent_node(from_agent_id)
        to_node = self.topology.get_agent_node(to_agent_id)
        if from_node is None or to_node is None:
            return None
        return self.topology.shortest_path(from_node, to_node)

    # =========================================================================
    # Topology Queries (consumed by OctopusColony, GNN, etc.)
    # =========================================================================

    def get_peers(self, agent_id: int, max_peers: int = 3) -> list[int]:
        """Get peer agents connected to this agent through substrate.

        Used by OctopusColony for peer connections, FRL for policy sharing,
        and GNN for message routing. Limited by Rule of 3 convention.
        """
        neighbors = self.topology.get_agent_neighbors(agent_id)
        if len(neighbors) <= max_peers:
            return neighbors

        # Return the most strongly-connected peers
        node_id = self.topology.get_agent_node(agent_id)
        if node_id is None:
            return neighbors[:max_peers]

        node = self.topology.get_node(node_id)
        if node is None:
            return neighbors[:max_peers]

        # Sort by conductance (strongest connections first)
        scored = []
        for peer_id in neighbors:
            peer_node = self.topology.get_agent_node(peer_id)
            if peer_node:
                conductance = node.conductance.get(peer_node, 0.5)
                scored.append((peer_id, conductance))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scored[:max_peers]]

    def get_agent_position(self, agent_id: int) -> Optional[tuple[float, float]]:
        """Get an agent's position in the substrate."""
        node_id = self.topology.get_agent_node(agent_id)
        if node_id is None:
            return None
        node = self.topology.get_node(node_id)
        return node.position if node else None

    def get_all_agent_positions(self) -> dict[int, tuple[float, float]]:
        """Get all agent positions (for spatial consensus, visualization)."""
        positions: dict[int, tuple[float, float]] = {}
        for nid, node in self.topology.get_nodes().items():
            if node.agent_id is not None:
                positions[node.agent_id] = node.position
        return positions

    def get_topology_graph(self) -> dict[str, Any]:
        """Get the full topology as a serializable graph.

        Used by GNNCommunicator for graph neural network routing.
        """
        nodes_data = {}
        for nid, node in self.topology.get_nodes().items():
            nodes_data[nid] = {
                "agent_id": node.agent_id,
                "position": node.position,
                "resource_level": node.resource_level,
                "degree": self.topology.get_degree(nid),
            }

        edges = []
        seen: set[tuple[str, str]] = set()
        for nid, neighbors in self.topology._adjacency.items():
            for neighbor in neighbors:
                edge_key = tuple(sorted([nid, neighbor]))
                if edge_key not in seen:
                    seen.add(edge_key)
                    node = self.topology.get_node(nid)
                    conductance = node.conductance.get(neighbor, 1.0) if node else 1.0
                    edges.append(
                        {"from": nid, "to": neighbor, "conductance": conductance}
                    )

        return {"nodes": nodes_data, "edges": edges}

    # =========================================================================
    # Step (called by MycelialModel each tick)
    # =========================================================================

    def step(self) -> None:
        """Execute one substrate step.

        1. Run nutrient flow (resource diffusion)
        2. Check for starving nodes
        3. Check for isolated nodes
        4. Publish health report periodically
        """
        self._step_count += 1

        # 1. Nutrient flow
        self.nutrient_flow.flow_step()

        # 2. Starvation alerts
        starving = self.nutrient_flow.get_starving_nodes()
        if starving:
            self.event_bus.publish(
                CH_STARVATION_ALERT,
                {"nodes": starving, "step": self._step_count},
            )

        # 3. Isolation detection
        isolated = self.topology.get_isolated_nodes()
        occupied_isolated = [
            nid
            for nid in isolated
            if self.topology.get_node(nid)
            and self.topology.get_node(nid).agent_id is not None
        ]
        if occupied_isolated:
            self.event_bus.publish(
                CH_ISOLATION_DETECTED,
                {"nodes": occupied_isolated, "step": self._step_count},
            )

        # 4. Periodic health report
        if self._step_count % self._health_report_interval == 0:
            self._publish_health_report()

    def _publish_health_report(self) -> None:
        """Publish comprehensive substrate health metrics."""
        report = self.get_health_report()
        self.event_bus.publish(CH_HEALTH_REPORT, report)

    def get_health_report(self) -> dict[str, Any]:
        """Get comprehensive substrate health report."""
        return {
            "step": self._step_count,
            "phase": self._current_phase,
            "topology": {
                "type": self.topology.topology_type.value,
                "nodes": self.topology.node_count,
                "edges": self.topology.edge_count,
                "agents": self.topology.agent_count,
                "connectivity": self.topology.connectivity_ratio(),
                "avg_degree": self.topology.average_degree(),
                "clustering": self.topology.clustering_coefficient(),
                "isolated_nodes": len(self.topology.get_isolated_nodes()),
                "empty_nodes": len(self.topology.get_empty_nodes()),
            },
            "resources": self.nutrient_flow.get_resource_stats(),
            "signals": {
                "recent_count": len(self._signal_history),
                "total_propagated": sum(
                    s["reached"] for s in self._signal_history
                ),
            },
        }

    # =========================================================================
    # Circadian Integration
    # =========================================================================

    def set_phase(self, phase: str) -> None:
        """Set circadian phase (modulates substrate behavior).

        ACTIVE: Normal operation, full flow.
        CONSOLIDATION: Reduced flow, focus on strengthening connections.
        REST: Minimal flow, conserve resources.
        """
        self._current_phase = phase
        if phase == "REST":
            self.nutrient_flow.modulate_decay_rate(0.5)  # Half decay during rest
            self.nutrient_flow.modulate_flow_rate(0.5)  # Half flow during rest
        elif phase == "CONSOLIDATION":
            self.nutrient_flow.modulate_decay_rate(0.7)
            self.nutrient_flow.modulate_flow_rate(0.8)
        else:  # ACTIVE
            # Reset to defaults (would need to store originals for proper reset)
            pass

    # =========================================================================
    # Region Isolation (for AutoHealing) — delegated to mycelial_topology
    # =========================================================================

    def isolate_region(self, node_ids: list[str]) -> dict[str, list[str]]:
        """Isolate a set of nodes from the rest of the network."""
        return _isolate_region(self.topology, self.event_bus, node_ids)

    def restore_region(self, removed_edges: dict[str, list[str]]) -> None:
        """Restore previously isolated edges."""
        _restore_region(self.topology, self.event_bus, removed_edges)

    # =========================================================================
    # Dynamic Topology Growth — delegated to mycelial_topology
    # =========================================================================

    def grow_node(
        self,
        position: Optional[tuple[float, float]] = None,
        connect_to_nearest: int = 3,
    ) -> str:
        """Add a new empty node to the substrate."""
        return _grow_node(self.topology, self.event_bus, position, connect_to_nearest)

    def prune_node(self, node_id: str) -> bool:
        """Remove an empty node from the substrate."""
        return _prune_node(self.topology, self.event_bus, node_id)

    # =========================================================================
    # Internal — delegated to mycelial_topology
    # =========================================================================

    def _build_topology(
        self,
        topology_type: str,
        initial_nodes: int,
        params: dict[str, Any],
    ) -> SubstrateTopology:
        """Build initial topology (delegates to mycelial_topology.build_topology)."""
        return build_topology(topology_type, initial_nodes, params)

    def _find_or_create_node(
        self, agent_id: int, position: Optional[tuple[float, float]]
    ) -> str:
        """Find or create node (delegates to mycelial_topology.find_or_create_node)."""
        return find_or_create_node(self.topology, agent_id, position)

    def _auto_connect(self, node_id: str, max_connections: int = 3) -> None:
        """Auto-connect node (delegates to mycelial_topology.auto_connect)."""
        auto_connect(self.topology, node_id, max_connections)
