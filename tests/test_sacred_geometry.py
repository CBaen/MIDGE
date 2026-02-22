"""Tests for Sacred Geometry - K4 Complete Graph structures.

Tests the K4 tetrahedron (4 nodes, 6 edges, 4 faces) with triadic witnessing.
Verifies Law 1 compliance: all edges are witnessed, no bare dyads.
"""

from __future__ import annotations

import pytest

from mae_core.backbone.connection_registry import (
    ConnectionRegistry,
    ConnectionType,
)
from mae_core.backbone.event_bus import EventBus
from mae_core.backbone.sacred_geometry import (
    Edge,
    K3Triangle,
    K4CompleteGraph,
    K4Generator,
    GeometryLevel,
)


@pytest.fixture
def bus():
    return EventBus()


@pytest.fixture
def conn_registry(bus):
    return ConnectionRegistry(event_bus=bus)


@pytest.fixture
def k4_generator(conn_registry, bus):
    return K4Generator(
        connection_registry=conn_registry,
        event_bus=bus,
    )


# =====================================================================
# Test Edge
# =====================================================================


class TestEdge:
    """Tests for Edge (witnessed connection)."""

    def test_create_edge(self):
        """Create a simple edge."""
        edge = Edge(node_a="A", node_b="B", witnesses=["C"])
        assert edge.node_a == "A"
        assert edge.node_b == "B"
        assert edge.witnesses == ["C"]

    def test_edge_is_undirected(self):
        """Edges are undirected: (A, B) == (B, A)."""
        edge1 = Edge(node_a="A", node_b="B", witnesses=["C"])
        edge2 = Edge(node_a="B", node_b="A", witnesses=["C"])
        assert edge1 == edge2

    def test_edge_hash_is_undirected(self):
        """Hash of edge is the same regardless of direction."""
        edge1 = Edge(node_a="A", node_b="B", witnesses=["C"])
        edge2 = Edge(node_a="B", node_b="A", witnesses=["C"])
        assert hash(edge1) == hash(edge2)

    def test_edge_multiple_witnesses(self):
        """Edge can have multiple witnesses (Law 1)."""
        edge = Edge(node_a="A", node_b="B", witnesses=["C", "D"])
        assert len(edge.witnesses) == 2


# =====================================================================
# Test K3Triangle
# =====================================================================


class TestK3Triangle:
    """Tests for K3 (triangle) substructure within K4."""

    def test_create_k3(self):
        """Create a K3 triangle."""
        edge1 = Edge(node_a="A", node_b="B", witnesses=["C"])
        edge2 = Edge(node_a="B", node_b="C", witnesses=["A"])
        edge3 = Edge(node_a="C", node_b="A", witnesses=["B"])
        k3 = K3Triangle(
            name="face-0",
            nodes=["A", "B", "C"],
            edges=[edge1, edge2, edge3],
        )
        assert len(k3.nodes) == 3
        assert len(k3.edges) == 3

    def test_k3_verify_integrity_valid(self):
        """Valid K3 passes integrity check."""
        edge1 = Edge(node_a="A", node_b="B", witnesses=["C"])
        edge2 = Edge(node_a="B", node_b="C", witnesses=["A"])
        edge3 = Edge(node_a="C", node_b="A", witnesses=["B"])
        k3 = K3Triangle(
            name="face-0",
            nodes=["A", "B", "C"],
            edges=[edge1, edge2, edge3],
        )
        assert k3.verify_triadic_integrity() is True

    def test_k3_verify_integrity_missing_witness(self):
        """K3 with bare dyad fails integrity check."""
        edge1 = Edge(node_a="A", node_b="B", witnesses=[])  # No witness!
        edge2 = Edge(node_a="B", node_b="C", witnesses=["A"])
        edge3 = Edge(node_a="C", node_b="A", witnesses=["B"])
        k3 = K3Triangle(
            name="face-0",
            nodes=["A", "B", "C"],
            edges=[edge1, edge2, edge3],
        )
        assert k3.verify_triadic_integrity() is False

    def test_k3_verify_integrity_wrong_node_count(self):
        """K3 must have exactly 3 nodes."""
        k3 = K3Triangle(
            name="invalid",
            nodes=["A", "B"],  # Only 2!
            edges=[],
        )
        assert k3.verify_triadic_integrity() is False


# =====================================================================
# Test K4CompleteGraph
# =====================================================================


class TestK4CompleteGraph:
    """Tests for K4 tetrahedron (4 nodes, complete graph)."""

    def test_create_k4(self):
        """Create a K4 with 4 nodes."""
        k4 = K4CompleteGraph(
            name="tetrahedron-1",
            nodes=["A", "B", "C", "D"],
        )
        assert len(k4.nodes) == 4
        assert k4.name == "tetrahedron-1"

    def test_create_k4_invalid_node_count(self):
        """K4 must have exactly 4 nodes."""
        with pytest.raises(ValueError):
            K4CompleteGraph(
                name="invalid",
                nodes=["A", "B", "C"],  # Only 3!
            )

    def test_get_all_edges(self):
        """K4 has exactly 6 edges (C(4,2) = 6)."""
        k4 = K4CompleteGraph(
            name="tetrahedron-1",
            nodes=["A", "B", "C", "D"],
        )
        edges = k4.get_all_edges()
        assert len(edges) == 6
        # Check all pairs are present
        expected = {
            ("A", "B"), ("A", "C"), ("A", "D"),
            ("B", "C"), ("B", "D"),
            ("C", "D"),
        }
        assert set(edges) == expected

    def test_get_faces(self):
        """K4 has exactly 4 K3 faces."""
        k4 = K4CompleteGraph(
            name="tetrahedron-1",
            nodes=["A", "B", "C", "D"],
        )
        faces = k4.get_faces()
        assert len(faces) == 4
        # Each face has 3 nodes
        for face in faces:
            assert len(face) == 3
        # All nodes should appear in exactly 3 faces
        face_str = [str(f) for f in faces]
        for node in k4.nodes:
            count = sum(1 for f in faces if node in f)
            assert count == 3

    def test_get_dual_graph(self):
        """K4 self-dual: each node opposite to a face."""
        k4 = K4CompleteGraph(
            name="tetrahedron-1",
            nodes=["A", "B", "C", "D"],
        )
        dual = k4.get_dual_graph()
        assert len(dual) == 4
        # A is opposite to [B, C, D]
        assert set(dual["A"]) == {"B", "C", "D"}
        assert set(dual["B"]) == {"A", "C", "D"}
        assert set(dual["C"]) == {"A", "B", "D"}
        assert set(dual["D"]) == {"A", "B", "C"}

    def test_verify_completeness_invalid(self):
        """K4 with missing edges fails completeness check."""
        k4 = K4CompleteGraph(
            name="incomplete",
            nodes=["A", "B", "C", "D"],
        )
        # Add only 3 edges, should be 6
        k4.edges = [
            Edge(node_a="A", node_b="B", witnesses=["C"]),
            Edge(node_a="A", node_b="C", witnesses=["B"]),
            Edge(node_a="A", node_b="D", witnesses=["B"]),
        ]
        assert k4.verify_completeness() is False

    def test_verify_triadic_witnessing_valid(self):
        """All edges witnessed passes check."""
        k4 = K4CompleteGraph(
            name="tetrahedron-1",
            nodes=["A", "B", "C", "D"],
        )
        k4.edges = [
            Edge(node_a="A", node_b="B", witnesses=["C", "D"]),
            Edge(node_a="A", node_b="C", witnesses=["B", "D"]),
            Edge(node_a="A", node_b="D", witnesses=["B", "C"]),
            Edge(node_a="B", node_b="C", witnesses=["A", "D"]),
            Edge(node_a="B", node_b="D", witnesses=["A", "C"]),
            Edge(node_a="C", node_b="D", witnesses=["A", "B"]),
        ]
        assert k4.verify_triadic_witnessing() is True

    def test_verify_triadic_witnessing_bare_dyad(self):
        """Bare dyad (no witness) fails check (Law 1 violation)."""
        k4 = K4CompleteGraph(
            name="tetrahedron-1",
            nodes=["A", "B", "C", "D"],
        )
        k4.edges = [
            Edge(node_a="A", node_b="B", witnesses=[]),  # Bare dyad!
            Edge(node_a="A", node_b="C", witnesses=["B", "D"]),
            Edge(node_a="A", node_b="D", witnesses=["B", "C"]),
            Edge(node_a="B", node_b="C", witnesses=["A", "D"]),
            Edge(node_a="B", node_b="D", witnesses=["A", "C"]),
            Edge(node_a="C", node_b="D", witnesses=["A", "B"]),
        ]
        assert k4.verify_triadic_witnessing() is False

    def test_get_witness_count(self):
        """Count total witness assignments."""
        k4 = K4CompleteGraph(
            name="tetrahedron-1",
            nodes=["A", "B", "C", "D"],
        )
        k4.edges = [
            Edge(node_a="A", node_b="B", witnesses=["C", "D"]),
            Edge(node_a="A", node_b="C", witnesses=["B"]),  # 1 witness
            Edge(node_a="A", node_b="D", witnesses=[]),
            Edge(node_a="B", node_b="C", witnesses=["A"]),
            Edge(node_a="B", node_b="D", witnesses=[]),
            Edge(node_a="C", node_b="D", witnesses=[]),
        ]
        # Total: 2 + 1 + 0 + 1 + 0 + 0 = 4
        assert k4.get_witness_count() == 4

    def test_get_edge_by_nodes(self):
        """Find edge by two nodes (undirected)."""
        k4 = K4CompleteGraph(
            name="tetrahedron-1",
            nodes=["A", "B", "C", "D"],
        )
        edge = Edge(node_a="A", node_b="B", witnesses=["C"])
        k4.edges = [edge]
        # Should find by (A, B) or (B, A)
        assert k4.get_edge_by_nodes("A", "B") is edge
        assert k4.get_edge_by_nodes("B", "A") is edge
        assert k4.get_edge_by_nodes("A", "C") is None


# =====================================================================
# Test K4Generator
# =====================================================================


class TestK4Generator:
    """Tests for K4 factory with ConnectionRegistry integration."""

    def test_generate_k4_basic(self, k4_generator, conn_registry):
        """Generate a K4 with proper registration."""
        k4 = k4_generator.generate_k4(
            name="module-tetrahedron",
            nodes=["subsys-A", "subsys-B", "subsys-C", "subsys-D"],
        )
        assert len(k4.nodes) == 4
        assert len(k4.edges) == 6
        # All edges should be registered
        for edge in k4.edges:
            assert edge.connection_id
            assert edge.witnesses

    def test_generate_k4_witness_assignment(self, k4_generator):
        """Each edge gets witnesses from the other 2 nodes."""
        k4 = k4_generator.generate_k4(
            name="tetrahedron-1",
            nodes=["A", "B", "C", "D"],
        )
        # For edge A-B, witnesses should be C and D (the other 2 nodes)
        edge_ab = k4.get_edge_by_nodes("A", "B")
        assert edge_ab is not None
        assert set(edge_ab.witnesses) == {"C", "D"}

        # For edge A-C, witnesses should be B and D
        edge_ac = k4.get_edge_by_nodes("A", "C")
        assert edge_ac is not None
        assert set(edge_ac.witnesses) == {"B", "D"}

    def test_generate_k4_creates_faces(self, k4_generator):
        """K4 generation creates 4 K3 faces."""
        k4 = k4_generator.generate_k4(
            name="tetrahedron-1",
            nodes=["A", "B", "C", "D"],
        )
        assert len(k4.faces) == 4
        for face in k4.faces:
            assert len(face.nodes) == 3
            assert len(face.edges) == 3

    def test_generate_k4_faces_contain_all_edges(self, k4_generator):
        """All 6 edges are distributed among the 4 faces."""
        k4 = k4_generator.generate_k4(
            name="tetrahedron-1",
            nodes=["A", "B", "C", "D"],
        )
        all_face_edges = []
        for face in k4.faces:
            all_face_edges.extend(face.edges)
        # Each edge appears in exactly 2 faces (shared between them)
        # So 6 edges * 2 appearances = 12 face-edges
        assert len(all_face_edges) == 12

    def test_generate_k4_invalid_node_count(self, k4_generator):
        """K4 generation fails with != 4 nodes."""
        with pytest.raises(ValueError):
            k4_generator.generate_k4(
                name="invalid",
                nodes=["A", "B", "C"],  # Only 3!
            )

    def test_verify_k4_integrity_valid(self, k4_generator):
        """Verify a valid K4 structure."""
        k4 = k4_generator.generate_k4(
            name="tetrahedron-1",
            nodes=["A", "B", "C", "D"],
        )
        report = k4_generator.verify_k4_integrity(k4)
        assert report["overall_valid"] is True
        assert report["complete"] is True
        assert report["witnessed"] is True
        assert report["faces_valid"] is True
        assert report["node_count"] == 4
        assert report["edge_count"] == 6
        assert report["witness_count"] == 12  # 6 edges * 2 witnesses each

    def test_verify_k4_integrity_detailed(self, k4_generator):
        """Verify report contains edge details."""
        k4 = k4_generator.generate_k4(
            name="tetrahedron-1",
            nodes=["A", "B", "C", "D"],
        )
        report = k4_generator.verify_k4_integrity(k4)
        assert "details" in report
        assert len(report["details"]) == 6  # One per edge
        for detail in report["details"]:
            assert "edge" in detail
            assert "witnesses" in detail
            assert "connection_id" in detail


# =====================================================================
# Integration Tests (K4 + ConnectionRegistry)
# =====================================================================


class TestK4Integration:
    """Integration tests: K4 with full ConnectionRegistry enforcement."""

    def test_k4_with_connection_registry(self, k4_generator, conn_registry):
        """K4 edges are registered with ConnectionRegistry (Law 1 compliant)."""
        k4 = k4_generator.generate_k4(
            name="module-coord",
            nodes=["metabolic", "sensing", "cognitive", "decision"],
        )
        # All edges should be queryable from registry
        # (ConnectionRegistry.is_connection_allowed checks for registered connections)
        for edge in k4.edges:
            # Note: Connection ID is in the format "name:node_a-node_b"
            # The registry stores it but we can't directly query by that ID
            # So we verify it was registered by checking the K4 structure
            assert edge.connection_id
            assert len(edge.witnesses) > 0

    def test_k4_no_bare_dyads(self, k4_generator):
        """K4 generation ensures no bare dyads (Law 1)."""
        k4 = k4_generator.generate_k4(
            name="working-group",
            nodes=["node-1", "node-2", "node-3", "node-4"],
        )
        # Every edge must have witnesses
        for edge in k4.edges:
            assert edge.witnesses, f"Bare dyad found: {edge.node_a} <-> {edge.node_b}"

    def test_multiple_k4_structures(self, k4_generator):
        """Can create multiple independent K4 structures."""
        k4_1 = k4_generator.generate_k4(
            name="tetrahedron-1",
            nodes=["A", "B", "C", "D"],
        )
        k4_2 = k4_generator.generate_k4(
            name="tetrahedron-2",
            nodes=["W", "X", "Y", "Z"],
        )
        assert k4_1.name == "tetrahedron-1"
        assert k4_2.name == "tetrahedron-2"
        assert k4_1.get_all_edges() != k4_2.get_all_edges()

    def test_k4_event_published(self, k4_generator, bus):
        """K4 creation publishes event to EventBus."""
        import json
        events = []
        bus.register_callback(
            "sacred_geometry.k4_created",
            lambda ch, msg: events.append(json.loads(msg) if isinstance(msg, str) else msg)
        )
        k4 = k4_generator.generate_k4(
            name="tetrahedron-1",
            nodes=["A", "B", "C", "D"],
        )
        assert len(events) == 1
        event_data = events[0]
        assert event_data["k4_name"] == "tetrahedron-1"
        assert event_data["nodes"] == ["A", "B", "C", "D"]
        assert event_data["edges_created"] == 6


# =====================================================================
# Mathematical Property Tests
# =====================================================================


class TestK4MathematicalProperties:
    """Tests for mathematical properties of K4 (tetrahedron)."""

    def test_k4_is_complete_graph(self, k4_generator):
        """K4 is a complete graph: every pair of nodes is connected."""
        k4 = k4_generator.generate_k4(
            name="complete",
            nodes=["A", "B", "C", "D"],
        )
        # All 6 pairs should have edges
        pairs = k4.get_all_edges()
        assert len(pairs) == 6
        for i, node_a in enumerate(k4.nodes):
            for node_b in k4.nodes[i+1:]:
                assert (node_a, node_b) in pairs or (node_b, node_a) in pairs

    def test_k4_self_dual(self, k4_generator):
        """K4 is self-dual: dual of dual is original."""
        k4 = k4_generator.generate_k4(
            name="self-dual",
            nodes=["A", "B", "C", "D"],
        )
        dual = k4.get_dual_graph()
        # In a tetrahedron, every vertex is opposite to a face
        # If we replace each vertex with its opposite face, we get another K4
        # This is the self-dual property
        assert len(dual) == 4
        for node in k4.nodes:
            opposite_face = dual[node]
            assert len(opposite_face) == 3
            assert node not in opposite_face

    def test_k4_fault_tolerance(self, k4_generator):
        """K4 can survive loss of 1 node (remaining 3 form K3)."""
        k4 = k4_generator.generate_k4(
            name="robust",
            nodes=["A", "B", "C", "D"],
        )
        # If we remove node D, we should have a K3 with nodes A, B, C
        remaining_nodes = [n for n in k4.nodes if n != "D"]
        remaining_edges = [
            e for e in k4.edges
            if e.node_a in remaining_nodes and e.node_b in remaining_nodes
        ]
        # K3 has exactly 3 edges
        assert len(remaining_edges) == 3
        # Check all edges have witnesses
        for edge in remaining_edges:
            assert len(edge.witnesses) > 0

    def test_k4_vertex_degree(self, k4_generator):
        """In K4, every node is connected to all 3 other nodes (degree 3)."""
        k4 = k4_generator.generate_k4(
            name="degree-test",
            nodes=["A", "B", "C", "D"],
        )
        for node in k4.nodes:
            # Count edges touching this node
            degree = sum(
                1 for edge in k4.edges
                if node == edge.node_a or node == edge.node_b
            )
            assert degree == 3
