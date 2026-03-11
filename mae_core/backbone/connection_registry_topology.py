"""Connection Registry Topology — Euler invariant analysis.

Extracted from connection_registry.py to keep it under the 500-line cap.

Pure computation: takes the connections dict as input, returns topological
invariants. No threading, no EventBus, no side effects.

Biological analogy: Topological analysis of Mae's nervous system graph.
Euler's formula measures the structural complexity of the connection web —
how many "shortcut" edges exist above a minimal spanning tree. These
shortcuts ARE the transfractal compromise in numbers.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def get_euler_statistics(connections: dict[str, Any]) -> dict[str, Any]:
    """Compute topological invariants of the connection graph.

    Euler's formula for connected planar graphs: V - E + F = 2.
    Mae's graph is neither planar nor simply connected, so we compute
    the generalized Euler characteristic and the "excess edges" above
    a spanning forest. The excess count is the number of non-tree
    shortcuts (EventBus cross-wiring) above the minimal spanning tree.

    Parameters
    ----------
    connections : dict[str, ConnectionTriad]
        The registry's connection dict (connection_id -> ConnectionTriad).

    Returns dict with vertices, edges, components, excess_edges,
    and euler_characteristic (V - E + C for a graph with C components).
    """
    if not connections:
        return {
            "vertices": 0,
            "edges": 0,
            "components": 0,
            "excess_edges": 0,
            "euler_characteristic": 0,
        }

    # V: unique systems participating as source or target
    vertices: set[str] = set()
    # E: unique undirected edges (unordered pairs)
    edges: set[frozenset[str]] = set()

    for triad in connections.values():
        vertices.add(triad.source)
        vertices.add(triad.target)
        edges.add(frozenset((triad.source, triad.target)))

    v = len(vertices)
    e = len(edges)

    # C: connected components via union-find
    parent: dict[str, str] = {node: node for node in vertices}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for edge in edges:
        nodes = list(edge)
        if len(nodes) == 2:
            ra, rb = find(nodes[0]), find(nodes[1])
            if ra != rb:
                parent[ra] = rb

    c = len({find(node) for node in vertices})

    # Spanning forest has V - C edges; excess = E - (V - C)
    excess = e - (v - c)

    return {
        "vertices": v,
        "edges": e,
        "components": c,
        "excess_edges": excess,
        "euler_characteristic": v - e + c,
    }


def check_euler_invariant(connections: dict[str, Any]) -> dict[str, Any]:
    """Advisory check: report topological invariants of the connection graph.

    For a spanning forest: E = V - C exactly (0 excess edges).
    For Mae: E > V - C because EventBus shortcuts add non-tree edges.
    The excess_edges count measures how many shortcuts exist above the
    minimal spanning tree — this IS the transfractal compromise in numbers.

    Returns the Euler statistics plus an advisory interpretation.
    """
    stats = get_euler_statistics(connections)
    if stats["vertices"] == 0:
        stats["interpretation"] = "empty_graph"
        return stats

    if stats["excess_edges"] == 0:
        stats["interpretation"] = "tree_or_forest"
    else:
        stats["interpretation"] = "shortcuts_present"

    logger.info(
        "Euler invariant: V=%d, E=%d, C=%d, excess=%d, chi=%d (%s)",
        stats["vertices"], stats["edges"], stats["components"],
        stats["excess_edges"], stats["euler_characteristic"],
        stats["interpretation"],
    )
    return stats
