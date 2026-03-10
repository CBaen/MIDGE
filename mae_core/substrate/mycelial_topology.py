"""
mycelial_topology.py - Topology operations for MycelialSubstrate.

Extracted from mycelial_substrate.py. Contains:
  - _build_topology: factory method to create topology from type string
  - _find_or_create_node: node allocation for agent registration
  - _auto_connect: proximity-based automatic edge wiring
  - grow_node / prune_node: dynamic topology mutation
  - isolate_region / restore_region: AutoHealing containment
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, Any, Optional

from mae_core.backbone.event_bus import EventBus
from mae_core.substrate.topology import SubstrateTopology, TopologyType

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

CH_TOPOLOGY_CHANGED = "substrate.topology_changed"


# ---------------------------------------------------------------------------
# Topology factory
# ---------------------------------------------------------------------------

def build_topology(
    topology_type: str,
    initial_nodes: int,
    params: dict[str, Any],
) -> SubstrateTopology:
    """Build initial topology from type string and parameters."""
    if topology_type == "ring":
        return SubstrateTopology.ring(initial_nodes)
    elif topology_type == "scale_free":
        m = params.get("m", 3)
        return SubstrateTopology.scale_free(initial_nodes, m=m)
    elif topology_type == "small_world":
        k = params.get("k", 4)
        p = params.get("p", 0.3)
        return SubstrateTopology.small_world(initial_nodes, k=k, p=p)
    elif topology_type == "mesh":
        return SubstrateTopology.mesh(initial_nodes)
    else:
        # Custom: empty topology, nodes added dynamically
        return SubstrateTopology(TopologyType.CUSTOM)


# ---------------------------------------------------------------------------
# Node allocation
# ---------------------------------------------------------------------------

def find_or_create_node(
    topology: SubstrateTopology,
    agent_id: int,
    position: Optional[tuple[float, float]],
) -> str:
    """Find an empty node near the position, or create a new one."""
    empty_nodes = topology.get_empty_nodes()

    if empty_nodes and position:
        # Find closest empty node
        best_id = None
        best_dist = float("inf")
        for nid in empty_nodes:
            node = topology.get_node(nid)
            if node:
                dx = node.position[0] - position[0]
                dy = node.position[1] - position[1]
                dist = (dx * dx + dy * dy) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_id = nid
        if best_id and best_dist < 1.0:  # Close enough to reuse
            return best_id

    elif empty_nodes and not position:
        # Reuse any empty node
        return empty_nodes[0]

    # Create new node
    node_id = f"agent_{agent_id}"
    topology.add_node(node_id, agent_id=agent_id, position=position)
    return node_id


def auto_connect(
    topology: SubstrateTopology,
    node_id: str,
    max_connections: int = 3,
) -> None:
    """Auto-connect a node to nearest occupied nodes.

    Uses position-based proximity. Respects Rule of 3 convention
    (default max_connections=3).
    """
    node = topology.get_node(node_id)
    if node is None:
        return

    occupied = topology.get_occupied_nodes()
    if not occupied:
        # Connect to any nearby nodes
        candidates = [
            nid
            for nid in topology.get_nodes()
            if nid != node_id and nid not in topology.get_neighbors(node_id)
        ]
    else:
        candidates = [nid for nid in occupied if nid != node_id]

    if not candidates:
        return

    # Sort by distance
    scored = []
    for cid in candidates:
        candidate = topology.get_node(cid)
        if candidate:
            dx = node.position[0] - candidate.position[0]
            dy = node.position[1] - candidate.position[1]
            dist = (dx * dx + dy * dy) ** 0.5
            scored.append((cid, dist))

    scored.sort(key=lambda x: x[1])

    # Connect to nearest, up to max_connections
    existing = len(topology.get_neighbors(node_id))
    for cid, _ in scored:
        if existing >= max_connections:
            break
        if topology.add_edge(node_id, cid):
            existing += 1


# ---------------------------------------------------------------------------
# Dynamic topology mutation
# ---------------------------------------------------------------------------

def grow_node(
    topology: SubstrateTopology,
    event_bus: EventBus,
    position: Optional[tuple[float, float]] = None,
    connect_to_nearest: int = 3,
) -> str:
    """Add a new empty node to the substrate.

    Used by Morphogenesis to prepare positions for new agents.
    """
    node_id = f"node_{topology.node_count}"
    # Ensure unique ID
    while node_id in topology:
        node_id = f"node_{random.randint(1000, 99999)}"

    topology.add_node(node_id, position=position)

    # Connect to nearest nodes
    if connect_to_nearest > 0:
        auto_connect(topology, node_id, max_connections=connect_to_nearest)

    event_bus.publish(
        CH_TOPOLOGY_CHANGED,
        {"action": "grow_node", "node_id": node_id},
    )
    return node_id


def prune_node(
    topology: SubstrateTopology,
    event_bus: EventBus,
    node_id: str,
) -> bool:
    """Remove an empty node from the substrate.

    Only removes if no agent is assigned. Used for cleanup.
    """
    node = topology.get_node(node_id)
    if node is None or node.agent_id is not None:
        return False

    topology.remove_node(node_id)
    event_bus.publish(
        CH_TOPOLOGY_CHANGED,
        {"action": "prune_node", "node_id": node_id},
    )
    return True


# ---------------------------------------------------------------------------
# Region isolation (AutoHealing)
# ---------------------------------------------------------------------------

def isolate_region(
    topology: SubstrateTopology,
    event_bus: EventBus,
    node_ids: list[str],
) -> dict[str, list[str]]:
    """Isolate a set of nodes from the rest of the network.

    Used by AutoHealing to contain failures. Removes edges between
    the isolated region and the rest of the network. Returns the
    removed edges so they can be restored after recovery.

    Returns:
        Dict of node_id -> list of removed neighbor_ids.
    """
    removed_edges: dict[str, list[str]] = {}
    region_set = set(node_ids)

    for node_id in node_ids:
        removed = []
        for neighbor_id in list(topology.get_neighbors(node_id)):
            if neighbor_id not in region_set:
                topology.remove_edge(node_id, neighbor_id)
                removed.append(neighbor_id)
        if removed:
            removed_edges[node_id] = removed

    if removed_edges:
        event_bus.publish(
            CH_TOPOLOGY_CHANGED,
            {
                "action": "isolate_region",
                "nodes": node_ids,
                "edges_removed": sum(len(v) for v in removed_edges.values()),
            },
        )
    return removed_edges


def restore_region(
    topology: SubstrateTopology,
    event_bus: EventBus,
    removed_edges: dict[str, list[str]],
) -> None:
    """Restore previously isolated edges."""
    for node_id, neighbors in removed_edges.items():
        for neighbor_id in neighbors:
            topology.add_edge(node_id, neighbor_id)

    event_bus.publish(
        CH_TOPOLOGY_CHANGED,
        {
            "action": "restore_region",
            "edges_restored": sum(len(v) for v in removed_edges.values()),
        },
    )
