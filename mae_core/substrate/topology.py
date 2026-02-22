"""Substrate topology - the physical structure of Mae's mycelial network.

Biological analogy: Mycelial hyphal networks form scale-free graphs with
small-world properties. Hub nodes (high connectivity) provide resilience.
Short average path lengths enable fast signal propagation. High clustering
creates local neighborhoods for cooperation.

Real mycelium: Physarum polycephalum solves shortest-path problems by
reinforcing successful routes and retracting failed ones. Our topology
supports similar dynamic rewiring.

Provides topology generators (ring, scale-free, small-world, mesh) and
a mutable graph manager for dynamic agent registration/deregistration.
"""

from __future__ import annotations

import logging
import math
import random
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class TopologyType(Enum):
    """Available topology generators."""

    RING = "ring"
    SCALE_FREE = "scale_free"
    SMALL_WORLD = "small_world"
    MESH = "mesh"
    CUSTOM = "custom"


@dataclass
class SubstrateNode:
    """A node in the substrate topology.

    Each node represents a position in the network that an agent can occupy.
    Nodes can exist without agents (empty positions for future growth).
    """

    node_id: str
    agent_id: Optional[int] = None  # Mesa agent unique_id
    position: tuple[float, float] = (0.0, 0.0)
    resource_level: float = 0.5  # Current energy (0.0 - 1.0)
    conductance: dict[str, float] = field(default_factory=dict)  # edge health per neighbor
    metadata: dict[str, Any] = field(default_factory=dict)


class SubstrateTopology:
    """Manages the network graph structure for Mae's substrate.

    Supports dynamic node/edge management, topology queries, and health
    metrics. Uses adjacency list representation for sparse biological networks.
    """

    def __init__(self, topology_type: TopologyType = TopologyType.CUSTOM) -> None:
        self.topology_type = topology_type
        self._nodes: dict[str, SubstrateNode] = {}
        self._adjacency: dict[str, set[str]] = {}  # node_id -> set of neighbor ids
        self._agent_to_node: dict[int, str] = {}  # agent_id -> node_id

    # =========================================================================
    # Factory Methods - Topology Generators
    # =========================================================================

    @classmethod
    def ring(cls, n_nodes: int) -> SubstrateTopology:
        """Create a ring topology - each node connects to 2 neighbors.

        Like a circular mycelial network. Simple, connected, but long
        average path length (n/4). Good for ordered communication.
        """
        topo = cls(TopologyType.RING)
        for i in range(n_nodes):
            angle = 2 * math.pi * i / n_nodes
            pos = (math.cos(angle), math.sin(angle))
            topo.add_node(f"node_{i}", position=pos)

        node_ids = list(topo._nodes.keys())
        for i in range(n_nodes):
            topo.add_edge(node_ids[i], node_ids[(i + 1) % n_nodes])
        return topo

    @classmethod
    def scale_free(cls, n_nodes: int, m: int = 3) -> SubstrateTopology:
        """Create a scale-free topology (Barabasi-Albert model).

        Biological reality: Real mycelial networks follow power-law degree
        distributions. A few hub nodes have many connections while most
        nodes have few. This creates resilience - random failures rarely
        hit hubs, but targeted attacks on hubs fragment the network.

        Args:
            n_nodes: Total number of nodes.
            m: Edges per new node (higher = more connected).
        """
        topo = cls(TopologyType.SCALE_FREE)
        if n_nodes < m + 1:
            m = max(1, n_nodes - 1)

        # Seed with a small complete graph
        seed_size = m + 1
        for i in range(seed_size):
            angle = 2 * math.pi * i / seed_size
            topo.add_node(f"node_{i}", position=(math.cos(angle), math.sin(angle)))

        seed_ids = list(topo._nodes.keys())
        for i in range(seed_size):
            for j in range(i + 1, seed_size):
                topo.add_edge(seed_ids[i], seed_ids[j])

        # Preferential attachment: new nodes connect to existing nodes
        # with probability proportional to their degree
        for i in range(seed_size, n_nodes):
            new_id = f"node_{i}"
            angle = 2 * math.pi * i / n_nodes
            topo.add_node(new_id, position=(math.cos(angle) * 1.5, math.sin(angle) * 1.5))

            # Calculate attachment probabilities (degree-proportional)
            existing = [nid for nid in topo._nodes if nid != new_id]
            degrees = [len(topo._adjacency.get(nid, set())) for nid in existing]
            total_degree = sum(degrees) or 1

            # Select m targets by preferential attachment
            targets: set[str] = set()
            attempts = 0
            while len(targets) < min(m, len(existing)) and attempts < m * 10:
                r = random.random() * total_degree
                cumulative = 0.0
                for nid, deg in zip(existing, degrees):
                    cumulative += deg
                    if cumulative >= r:
                        targets.add(nid)
                        break
                attempts += 1

            # Fallback: if preferential attachment didn't find enough, pick randomly
            if len(targets) < min(m, len(existing)):
                remaining = [n for n in existing if n not in targets]
                needed = min(m, len(existing)) - len(targets)
                targets.update(random.sample(remaining, min(needed, len(remaining))))

            for target in targets:
                topo.add_edge(new_id, target)

        return topo

    @classmethod
    def small_world(
        cls, n_nodes: int, k: int = 4, p: float = 0.3
    ) -> SubstrateTopology:
        """Create a small-world topology (Watts-Strogatz model).

        Biological reality: Neural networks and mycelial networks both
        exhibit small-world properties - high clustering (local neighborhoods)
        AND short average path lengths (global reach). The p parameter
        controls the balance: p=0 is a regular lattice, p=1 is random.

        Args:
            n_nodes: Total number of nodes.
            k: Each node connects to k nearest neighbors (must be even).
            p: Rewiring probability (0.0 = lattice, 1.0 = random).
        """
        topo = cls(TopologyType.SMALL_WORLD)
        k = k if k % 2 == 0 else k + 1  # ensure even

        # Create ring lattice with k neighbors
        for i in range(n_nodes):
            angle = 2 * math.pi * i / n_nodes
            topo.add_node(f"node_{i}", position=(math.cos(angle), math.sin(angle)))

        node_ids = list(topo._nodes.keys())
        for i in range(n_nodes):
            for j in range(1, k // 2 + 1):
                topo.add_edge(node_ids[i], node_ids[(i + j) % n_nodes])

        # Rewire edges with probability p
        for i in range(n_nodes):
            for j in range(1, k // 2 + 1):
                if random.random() < p:
                    target = node_ids[(i + j) % n_nodes]
                    # Remove original edge
                    topo.remove_edge(node_ids[i], target)
                    # Rewire to random non-neighbor
                    candidates = [
                        nid
                        for nid in node_ids
                        if nid != node_ids[i]
                        and nid not in topo._adjacency.get(node_ids[i], set())
                    ]
                    if candidates:
                        new_target = random.choice(candidates)
                        topo.add_edge(node_ids[i], new_target)
                    else:
                        # Restore if no candidates
                        topo.add_edge(node_ids[i], target)

        return topo

    @classmethod
    def mesh(cls, n_nodes: int) -> SubstrateTopology:
        """Create a full mesh topology - every node connects to every other.

        Maximum connectivity but O(n^2) edges. Only suitable for small
        networks. Biologically unrealistic at scale but useful for
        tightly-coupled subsystems.
        """
        topo = cls(TopologyType.MESH)
        for i in range(n_nodes):
            angle = 2 * math.pi * i / n_nodes
            topo.add_node(f"node_{i}", position=(math.cos(angle), math.sin(angle)))

        node_ids = list(topo._nodes.keys())
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                topo.add_edge(node_ids[i], node_ids[j])
        return topo

    # =========================================================================
    # Dynamic Node/Edge Management
    # =========================================================================

    def add_node(
        self,
        node_id: str,
        agent_id: Optional[int] = None,
        position: Optional[tuple[float, float]] = None,
    ) -> SubstrateNode:
        """Add a node to the topology."""
        if node_id in self._nodes:
            logger.warning("Node %s already exists", node_id)
            return self._nodes[node_id]

        pos = position or (random.uniform(-2, 2), random.uniform(-2, 2))
        node = SubstrateNode(node_id=node_id, agent_id=agent_id, position=pos)
        self._nodes[node_id] = node
        self._adjacency[node_id] = set()

        if agent_id is not None:
            self._agent_to_node[agent_id] = node_id

        return node

    def remove_node(self, node_id: str) -> Optional[SubstrateNode]:
        """Remove a node and all its edges."""
        node = self._nodes.pop(node_id, None)
        if node is None:
            return None

        # Remove from adjacency lists
        neighbors = self._adjacency.pop(node_id, set())
        for neighbor_id in neighbors:
            self._adjacency.get(neighbor_id, set()).discard(node_id)

        # Clean conductance references
        for neighbor_id in neighbors:
            neighbor = self._nodes.get(neighbor_id)
            if neighbor:
                neighbor.conductance.pop(node_id, None)

        # Clean agent mapping
        if node.agent_id is not None:
            self._agent_to_node.pop(node.agent_id, None)

        return node

    def add_edge(self, node_a: str, node_b: str, conductance: float = 1.0) -> bool:
        """Add an edge between two nodes."""
        if node_a not in self._nodes or node_b not in self._nodes:
            return False
        if node_a == node_b:
            return False

        self._adjacency[node_a].add(node_b)
        self._adjacency[node_b].add(node_a)

        # Set conductance on both sides
        self._nodes[node_a].conductance[node_b] = conductance
        self._nodes[node_b].conductance[node_a] = conductance
        return True

    def remove_edge(self, node_a: str, node_b: str) -> bool:
        """Remove an edge between two nodes."""
        if node_a not in self._adjacency or node_b not in self._adjacency:
            return False

        self._adjacency[node_a].discard(node_b)
        self._adjacency[node_b].discard(node_a)

        # Clean conductance
        if node_a in self._nodes:
            self._nodes[node_a].conductance.pop(node_b, None)
        if node_b in self._nodes:
            self._nodes[node_b].conductance.pop(node_a, None)
        return True

    # =========================================================================
    # Agent-Node Mapping
    # =========================================================================

    def assign_agent(self, agent_id: int, node_id: str) -> bool:
        """Assign an agent to a node position."""
        node = self._nodes.get(node_id)
        if node is None:
            return False

        # Remove old assignment if exists
        old_node_id = self._agent_to_node.get(agent_id)
        if old_node_id and old_node_id in self._nodes:
            self._nodes[old_node_id].agent_id = None

        node.agent_id = agent_id
        self._agent_to_node[agent_id] = node_id
        return True

    def unassign_agent(self, agent_id: int) -> Optional[str]:
        """Remove an agent from its node position."""
        node_id = self._agent_to_node.pop(agent_id, None)
        if node_id and node_id in self._nodes:
            self._nodes[node_id].agent_id = None
        return node_id

    def get_agent_node(self, agent_id: int) -> Optional[str]:
        """Get the node ID for an agent."""
        return self._agent_to_node.get(agent_id)

    def get_node_agent(self, node_id: str) -> Optional[int]:
        """Get the agent ID at a node."""
        node = self._nodes.get(node_id)
        return node.agent_id if node else None

    # =========================================================================
    # Topology Queries
    # =========================================================================

    def get_neighbors(self, node_id: str) -> list[str]:
        """Get all neighbors of a node."""
        return list(self._adjacency.get(node_id, set()))

    def get_agent_neighbors(self, agent_id: int) -> list[int]:
        """Get agent IDs of all agents neighboring this agent."""
        node_id = self._agent_to_node.get(agent_id)
        if node_id is None:
            return []

        result = []
        for neighbor_id in self._adjacency.get(node_id, set()):
            neighbor = self._nodes.get(neighbor_id)
            if neighbor and neighbor.agent_id is not None:
                result.append(neighbor.agent_id)
        return result

    def get_degree(self, node_id: str) -> int:
        """Get the degree (number of connections) of a node."""
        return len(self._adjacency.get(node_id, set()))

    def shortest_path(self, from_id: str, to_id: str) -> Optional[list[str]]:
        """BFS shortest path between two nodes.

        Returns the path as a list of node IDs, or None if no path exists.
        """
        if from_id not in self._nodes or to_id not in self._nodes:
            return None
        if from_id == to_id:
            return [from_id]

        visited: set[str] = {from_id}
        queue: deque[list[str]] = deque([[from_id]])

        while queue:
            path = queue.popleft()
            current = path[-1]

            for neighbor in self._adjacency.get(current, set()):
                if neighbor == to_id:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])
        return None

    def get_hub_nodes(self, top_k: int = 5) -> list[tuple[str, int]]:
        """Get the most connected nodes (hubs).

        In a scale-free network, hubs are critical infrastructure.
        Protecting hubs = protecting the whole network.
        """
        degrees = [(nid, len(neighbors)) for nid, neighbors in self._adjacency.items()]
        degrees.sort(key=lambda x: x[1], reverse=True)
        return degrees[:top_k]

    def get_isolated_nodes(self) -> list[str]:
        """Get nodes with zero connections."""
        return [nid for nid, neighbors in self._adjacency.items() if len(neighbors) == 0]

    def get_empty_nodes(self) -> list[str]:
        """Get nodes without assigned agents (available positions)."""
        return [nid for nid, node in self._nodes.items() if node.agent_id is None]

    def get_occupied_nodes(self) -> list[str]:
        """Get nodes with assigned agents."""
        return [nid for nid, node in self._nodes.items() if node.agent_id is not None]

    # =========================================================================
    # Health Metrics
    # =========================================================================

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return sum(len(n) for n in self._adjacency.values()) // 2

    @property
    def agent_count(self) -> int:
        return len(self._agent_to_node)

    def connectivity_ratio(self) -> float:
        """Fraction of nodes reachable from any starting node.

        1.0 = fully connected. < 1.0 = fragmented network.
        Fragmentation is a health warning.
        """
        if not self._nodes:
            return 0.0

        start = next(iter(self._nodes))
        visited: set[str] = set()
        queue: deque[str] = deque([start])

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            for neighbor in self._adjacency.get(current, set()):
                if neighbor not in visited:
                    queue.append(neighbor)

        return len(visited) / len(self._nodes)

    def average_degree(self) -> float:
        """Average number of connections per node."""
        if not self._adjacency:
            return 0.0
        return sum(len(n) for n in self._adjacency.values()) / len(self._adjacency)

    def clustering_coefficient(self) -> float:
        """Global clustering coefficient.

        Measures how much nodes tend to cluster together.
        High clustering = strong local neighborhoods (biological).
        """
        if not self._nodes:
            return 0.0

        total_cc = 0.0
        count = 0

        for node_id, neighbors in self._adjacency.items():
            k = len(neighbors)
            if k < 2:
                continue

            # Count edges between neighbors
            neighbor_list = list(neighbors)
            edges_between = 0
            for i in range(len(neighbor_list)):
                for j in range(i + 1, len(neighbor_list)):
                    if neighbor_list[j] in self._adjacency.get(neighbor_list[i], set()):
                        edges_between += 1

            possible = k * (k - 1) / 2
            total_cc += edges_between / possible if possible > 0 else 0
            count += 1

        return total_cc / count if count > 0 else 0.0

    def get_nodes(self) -> dict[str, SubstrateNode]:
        """Get all nodes (read-only view)."""
        return dict(self._nodes)

    def get_node(self, node_id: str) -> Optional[SubstrateNode]:
        """Get a specific node."""
        return self._nodes.get(node_id)

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._nodes
