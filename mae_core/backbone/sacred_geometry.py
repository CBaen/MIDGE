"""Sacred Geometry - K4 Complete Graph structures.

The tetrahedron (K4) is the minimal 3D stable structure and natural working
group topology. Four nodes, fully connected (6 edges), form 4 triadic faces.

This module extends K3 (proven at every scale) with K4 structures that can
coordinate cross-module groups: 4 subsystems forming a tetrahedron work better
than 3 for certain coordination patterns (voting stability, spatial reasoning).

Design principle: "Blueprint not building" — K4 represents a higher-order
coordination pattern built FROM K3 units (its 4 faces are K3 triangles).

All edges require triadic witnesses (Law 1). Integration with ConnectionRegistry
ensures zero bare dyads and proper triad enforcement.

Data structure: K4CompleteGraph stores vertices, edges with witnessed connections,
and the 4 triadic faces. Provides: tetrahedron geometry, dual structure,
face enumeration, witness integrity checks.

Biological analogy: A tetrahedron represents a working team of 4 agents where
every pair has mutual awareness, witnessed by the other 2. Fault tolerance:
survives loss of 1 member. Consensus: any 3 agree. Rigidity: can't collapse.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from mae_core.backbone.connection_registry import (
    ConnectionRegistry,
    ConnectionTriad,
    ConnectionType,
    ConnectionCriticality,
)

logger = logging.getLogger(__name__)

# EventBus channels
CH_K4_CREATED = "sacred_geometry.k4_created"
CH_K4_FACE_VERIFIED = "sacred_geometry.k4_face_verified"


class GeometryLevel(Enum):
    """Sacred geometry hierarchy."""
    K3 = "k3"  # Triangle (dyad + witness)
    K4 = "k4"  # Tetrahedron (4 triangles, fully connected)


@dataclass
class Edge:
    """A witnessed connection between two nodes in the geometry."""
    node_a: str
    node_b: str
    witnesses: list[str] = field(default_factory=list)
    connection_id: str = ""
    connection_type: ConnectionType = ConnectionType.DIRECT_REFERENCE
    created_at: float = field(default_factory=time.time)

    def __hash__(self) -> int:
        """Hash edge by sorted node pair (undirected)."""
        pair = tuple(sorted([self.node_a, self.node_b]))
        return hash(pair)

    def __eq__(self, other: Any) -> bool:
        """Compare edges by node pair (undirected)."""
        if not isinstance(other, Edge):
            return False
        pair_self = tuple(sorted([self.node_a, self.node_b]))
        pair_other = tuple(sorted([other.node_a, other.node_b]))
        return pair_self == pair_other


@dataclass
class K3Triangle:
    """A K3 (triangle) within a K4 structure.

    K3 is the atomic unit: 3 nodes, fully connected (3 edges), each edge witnessed.
    A tetrahedron (K4) contains 4 overlapping K3 triangles (its faces).

    Law 1: No bare dyads. Each edge in this triangle is a ConnectionTriad.
    """
    name: str
    nodes: list[str]  # Exactly 3 nodes
    edges: list[Edge] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def verify_triadic_integrity(self) -> bool:
        """Check: exactly 3 nodes, 3 witnessed edges, no bare dyads."""
        if len(self.nodes) != 3:
            return False
        if len(self.edges) != 3:
            return False
        # Every edge must have at least 1 witness (Law 1)
        for edge in self.edges:
            if not edge.witnesses:
                return False
        return True


@dataclass
class K4CompleteGraph:
    """K4: Complete graph on 4 nodes forming a tetrahedron.

    4 nodes (vertices), every pair connected (6 edges), all edges witnessed,
    forming 4 triadic faces (K3 triangles).

    Self-dual: the communication graph (who talks to whom) mirrors the
    functional graph (working group coordination structure).

    Tetrahedra are the atoms of 3D structure (like K3 in 2D). Perfect for:
    - Consensus voting (any 3 of 4 agree, witnesses break ties)
    - Spatial reasoning (can map to 3D coordinates)
    - Cross-module working groups (4 subsystems coordinating)
    - Fault tolerance (survives 1 node failure, 2+ still connected)
    """
    name: str
    nodes: list[str]  # Exactly 4 nodes
    edges: list[Edge] = field(default_factory=list)
    faces: list[K3Triangle] = field(default_factory=list)
    connection_registry: Optional[ConnectionRegistry] = None
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        """Validate that we have exactly 4 nodes."""
        if len(self.nodes) != 4:
            raise ValueError(
                f"K4 requires exactly 4 nodes, got {len(self.nodes)}"
            )

    def get_all_edges(self) -> list[tuple[str, str]]:
        """Return all 6 edges of the complete graph as (node_a, node_b) pairs.

        For 4 nodes: C(4,2) = 6 edges.
        """
        edges = []
        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                edges.append((self.nodes[i], self.nodes[j]))
        return edges

    def get_faces(self) -> list[list[str]]:
        """Return the 4 triadic faces (K3 subgraphs).

        Each face is 3 of the 4 nodes (omitting one node, we get a triangle).
        For nodes [A, B, C, D], faces are:
          - [A, B, C] (omit D)
          - [A, B, D] (omit C)
          - [A, C, D] (omit B)
          - [B, C, D] (omit A)
        """
        if len(self.nodes) != 4:
            return []
        faces = []
        for omit_idx in range(4):
            face = [n for i, n in enumerate(self.nodes) if i != omit_idx]
            faces.append(face)
        return faces

    def get_dual_graph(self) -> dict[str, list[str]]:
        """Return dual graph: maps each node to its opposite face.

        In a tetrahedron, every vertex has a unique opposite face
        (the 3 vertices that form the face opposite to it).

        Example: node A is opposite to face [B, C, D].
        This is self-dual (Platonic property of tetrahedra).
        """
        dual = {}
        faces = self.get_faces()
        for node in self.nodes:
            # Find the face that doesn't include this node
            opposite_face = [f for f in faces if node not in f]
            if opposite_face:
                dual[node] = opposite_face[0]
        return dual

    def verify_completeness(self) -> bool:
        """Check: 4 nodes, 6 edges, all edges present."""
        if len(self.nodes) != 4:
            return False
        expected_edge_count = 6  # C(4,2)
        if len(self.edges) != expected_edge_count:
            return False
        # Check all 6 edges are present
        expected = set(self.get_all_edges())
        actual = set(
            (e.node_a, e.node_b) if e.node_a < e.node_b else (e.node_b, e.node_a)
            for e in self.edges
        )
        return expected == actual

    def verify_triadic_witnessing(self) -> bool:
        """Check: all 6 edges have witnesses (Law 1 — no bare dyads).

        Per Law 1, every edge A-B must have at least one witness C.
        """
        for edge in self.edges:
            if not edge.witnesses:
                return False
        return True

    def verify_face_integrity(self) -> bool:
        """Check: 4 K3 faces are valid, each with triadic witnessing."""
        if len(self.faces) != 4:
            return False
        for face in self.faces:
            if not face.verify_triadic_integrity():
                return False
        return True

    def verify_structural_integrity(self) -> bool:
        """Full integrity check: completeness + witnessing + face integrity."""
        return (
            self.verify_completeness()
            and self.verify_triadic_witnessing()
            and self.verify_face_integrity()
        )

    def get_witness_count(self) -> int:
        """Count total witness assignments across all edges."""
        return sum(len(edge.witnesses) for edge in self.edges)

    def get_edge_by_nodes(self, node_a: str, node_b: str) -> Optional[Edge]:
        """Find an edge by its two nodes (undirected)."""
        for edge in self.edges:
            if (
                (edge.node_a == node_a and edge.node_b == node_b)
                or (edge.node_a == node_b and edge.node_b == node_a)
            ):
                return edge
        return None


class K4Generator:
    """Factory for creating K4 structures with proper ConnectionRegistry integration.

    Ensures all edges are registered with triadic witnesses (Law 1).
    """

    def __init__(
        self,
        connection_registry: ConnectionRegistry,
        event_bus: Any = None,
    ) -> None:
        self._connection_registry = connection_registry
        self._bus = event_bus
        logger.info("K4Generator initialized (EventBus=%s)", "present" if event_bus else "none")

    def generate_k4(
        self,
        name: str,
        nodes: list[str],
        parent_id: Optional[str] = None,
        witness_pool: Optional[list[str]] = None,
        connection_type: ConnectionType = ConnectionType.DIRECT_REFERENCE,
    ) -> K4CompleteGraph:
        """Create a K4 complete graph with all edges witnessed.

        Args:
            name: Name of the K4 structure (e.g., "module-tetrahedron-1")
            nodes: Exactly 4 nodes to form the tetrahedron
            parent_id: Optional parent holarchy ID (for context)
            witness_pool: Optional list of witness candidates (defaults to nervous system)
            connection_type: How edges are connected (default: DIRECT_REFERENCE)

        Returns:
            K4CompleteGraph with all 6 edges registered and witnessed

        Raises:
            ValueError: If not exactly 4 nodes
        """
        if len(nodes) != 4:
            raise ValueError(
                f"K4 requires exactly 4 nodes, got {len(nodes)}"
            )

        # Witness pool: default to nervous system + available nodes
        if witness_pool is None:
            witness_pool = list(nodes) + ["enforcer", "watchdog", "auditor"]

        k4 = K4CompleteGraph(name=name, nodes=list(nodes))

        # Create all 6 edges with triadic witnesses
        all_edges = k4.get_all_edges()
        for node_a, node_b in all_edges:
            # Assign witnesses: prefer the 2 other nodes in this edge's K3 faces
            # Each edge appears in 2 faces, giving us 4 candidate witnesses
            candidate_witnesses = [
                n for n in nodes
                if n != node_a and n != node_b
            ]
            # Pick 2 witnesses from the 2 candidates (the other 2 nodes)
            chosen_witnesses = candidate_witnesses[:2]

            # Register the edge with ConnectionRegistry
            # This creates a ConnectionTriad and ensures Law 1 compliance
            triad = self._connection_registry.register_connection(
                source=node_a,
                target=node_b,
                connection_type=connection_type,
                witnesses=chosen_witnesses,
                description=f"K4 tetrahedron {name}: {node_a} <-> {node_b}",
                criticality=ConnectionCriticality.STANDARD,
            )

            edge = Edge(
                node_a=node_a,
                node_b=node_b,
                witnesses=chosen_witnesses,
                connection_type=connection_type,
                connection_id=triad.connection_id,
            )
            k4.edges.append(edge)

        # Generate 4 K3 faces
        faces_nodes = k4.get_faces()
        for i, face_nodes in enumerate(faces_nodes):
            face_name = f"{name}:face-{i}"
            face_edges = []
            for j in range(3):
                for k in range(j + 1, 3):
                    edge = k4.get_edge_by_nodes(
                        face_nodes[j],
                        face_nodes[k],
                    )
                    if edge:
                        face_edges.append(edge)
            face = K3Triangle(
                name=face_name,
                nodes=face_nodes,
                edges=face_edges,
            )
            k4.faces.append(face)

        k4.connection_registry = self._connection_registry

        # Publish creation event
        if self._bus:
            self._bus.publish(CH_K4_CREATED, {
                "k4_name": name,
                "nodes": nodes,
                "parent_id": parent_id,
                "edges_created": len(all_edges),
                "witnesses_total": k4.get_witness_count(),
            })

        logger.info(
            "K4 '%s' generated: 4 nodes, 6 edges, %d witnesses, 4 K3 faces",
            name, k4.get_witness_count(),
        )
        return k4

    def verify_k4_integrity(self, k4: K4CompleteGraph) -> dict[str, Any]:
        """Verify a K4 structure against all mathematical laws.

        Returns detailed report of integrity checks.
        """
        report = {
            "k4_name": k4.name,
            "complete": k4.verify_completeness(),
            "witnessed": k4.verify_triadic_witnessing(),
            "faces_valid": k4.verify_face_integrity(),
            "overall_valid": k4.verify_structural_integrity(),
            "node_count": len(k4.nodes),
            "edge_count": len(k4.edges),
            "witness_count": k4.get_witness_count(),
            "face_count": len(k4.faces),
            "details": [],
        }

        # Detail: each edge and its witnesses
        for edge in k4.edges:
            edge_detail = {
                "edge": f"{edge.node_a}<->{edge.node_b}",
                "witnesses": edge.witnesses,
                "connection_id": edge.connection_id,
            }
            report["details"].append(edge_detail)

        # Publish verification event if valid
        if report["overall_valid"] and self._bus:
            self._bus.publish(CH_K4_FACE_VERIFIED, {
                "k4_name": k4.name,
                "verification": "passed",
                "edge_count": len(k4.edges),
            })

        return report


__all__ = [
    "GeometryLevel",
    "Edge",
    "K3Triangle",
    "K4CompleteGraph",
    "K4Generator",
    "CH_K4_CREATED",
    "CH_K4_FACE_VERIFIED",
]
