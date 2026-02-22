"""Tests for FractalGenerator — Step 4 of fractal architecture.

Tests the recursive triadic structure: generate_triad, generate_level,
organize, verify_triadic_integrity.
"""

from __future__ import annotations

import json

import pytest

from mae_core.backbone.connection_registry import ConnectionRegistry, ConnectionType
from mae_core.backbone.event_bus import EventBus
from mae_core.backbone.fractal_generator import (
    FRACTAL_GROUPING,
    FractalGenerator,
    FractalLevel,
)
from mae_core.backbone.holon_protocol import HolonRegistry


@pytest.fixture
def registry():
    return HolonRegistry()


@pytest.fixture
def bus():
    return EventBus()


@pytest.fixture
def conn_registry(bus):
    return ConnectionRegistry(event_bus=bus)


@pytest.fixture
def generator(registry, conn_registry, bus):
    return FractalGenerator(
        holon_registry=registry,
        connection_registry=conn_registry,
        event_bus=bus,
    )


@pytest.fixture
def populated_registry(registry):
    """Registry with mae + 6 child systems for testing."""
    registry.register("mae", holon_type="organism")
    for name in ("alpha", "beta", "gamma", "delta", "epsilon", "zeta"):
        registry.register(name, holon_type="system", parent_id="mae")
    return registry


@pytest.fixture
def populated_generator(populated_registry, conn_registry, bus):
    return FractalGenerator(
        holon_registry=populated_registry,
        connection_registry=conn_registry,
        event_bus=bus,
    )


# =====================================================================
# TestFractalGeneratorTriad
# =====================================================================


class TestFractalGeneratorTriad:
    """Tests for the atomic generate_triad operation."""

    def test_creates_parent_holon(self, populated_generator, populated_registry):
        result = populated_generator.generate_triad(
            name="triad-1",
            holon_type="subsystem",
            children_ids=["alpha", "beta", "gamma"],
            parent_id="mae",
        )
        entry = populated_registry.get_entry("triad-1")
        assert entry is not None
        assert entry.holon_type == "subsystem"
        assert entry.parent_id == "mae"
        assert result.parent_id == "triad-1"

    def test_reparents_children(self, populated_generator, populated_registry):
        populated_generator.generate_triad(
            name="triad-1",
            holon_type="subsystem",
            children_ids=["alpha", "beta", "gamma"],
            parent_id="mae",
        )
        for name in ("alpha", "beta", "gamma"):
            entry = populated_registry.get_entry(name)
            assert entry.parent_id == "triad-1"

    def test_k3_connections(self, populated_generator, conn_registry):
        result = populated_generator.generate_triad(
            name="triad-1",
            holon_type="subsystem",
            children_ids=["alpha", "beta", "gamma"],
            parent_id="mae",
        )
        # 3 children → 3 K3 connections (alpha-beta, alpha-gamma, beta-gamma)
        assert result.connections_created == 3

    def test_witness_assignment(self, populated_generator, conn_registry):
        populated_generator.generate_triad(
            name="triad-1",
            holon_type="subsystem",
            children_ids=["alpha", "beta", "gamma"],
            parent_id="mae",
        )
        # alpha-beta should be witnessed by gamma
        conn = conn_registry._connections.get("alpha->beta:None")
        if conn is None:
            conn = conn_registry._connections.get("alpha->beta")
        assert conn is not None
        assert conn.witness == "gamma"

    def test_children_become_peers(self, populated_generator, populated_registry):
        populated_generator.generate_triad(
            name="triad-1",
            holon_type="subsystem",
            children_ids=["alpha", "beta", "gamma"],
            parent_id="mae",
        )
        peers = populated_registry.get_peers("alpha")
        assert set(peers) == {"beta", "gamma"}

    def test_non_triadic_warns(self, populated_generator, caplog):
        populated_generator.generate_triad(
            name="pair-1",
            holon_type="subsystem",
            children_ids=["alpha", "beta"],
            parent_id="mae",
        )
        assert "2 children (ideal: 3)" in caplog.text

    def test_non_triadic_still_creates(self, populated_generator, populated_registry):
        result = populated_generator.generate_triad(
            name="pair-1",
            holon_type="subsystem",
            children_ids=["alpha", "beta"],
            parent_id="mae",
        )
        assert populated_registry.get_entry("pair-1") is not None
        assert len(result.children_ids) == 2

    def test_idempotent(self, populated_generator, populated_registry):
        populated_generator.generate_triad(
            name="triad-1",
            holon_type="subsystem",
            children_ids=["alpha", "beta", "gamma"],
            parent_id="mae",
        )
        result2 = populated_generator.generate_triad(
            name="triad-1",
            holon_type="subsystem",
            children_ids=["alpha", "beta", "gamma"],
            parent_id="mae",
        )
        # Second call should return existing, 0 new connections
        assert result2.connections_created == 0
        assert set(result2.children_ids) == {"alpha", "beta", "gamma"}

    def test_missing_child_skipped(self, populated_generator, caplog):
        result = populated_generator.generate_triad(
            name="triad-1",
            holon_type="subsystem",
            children_ids=["alpha", "nonexistent", "gamma"],
            parent_id="mae",
        )
        assert "nonexistent not found" in caplog.text
        assert len(result.children_ids) == 2

    def test_all_missing_aborts(self, populated_generator):
        result = populated_generator.generate_triad(
            name="triad-empty",
            holon_type="subsystem",
            children_ids=["x", "y", "z"],
            parent_id="mae",
        )
        assert result.connections_created == 0
        assert len(result.children_ids) == 0

    def test_publishes_event(self, populated_generator, bus):
        events = []
        bus.register_callback("fractal.triad_created", lambda ch, msg: events.append(json.loads(msg)))
        populated_generator.generate_triad(
            name="triad-1",
            holon_type="subsystem",
            children_ids=["alpha", "beta", "gamma"],
            parent_id="mae",
        )
        assert len(events) == 1
        assert events[0]["parent_id"] == "triad-1"
        assert events[0]["connections"] == 3


# =====================================================================
# TestFractalGeneratorLevel
# =====================================================================


class TestFractalGeneratorLevel:
    """Tests for batch generate_level operation."""

    def test_creates_multiple_triads(self, populated_generator, populated_registry):
        results = populated_generator.generate_level(
            triads_spec={
                "sub-1": ["alpha", "beta", "gamma"],
                "sub-2": ["delta", "epsilon", "zeta"],
            },
            holon_type="subsystem",
            parent_id="mae",
        )
        assert len(results) == 2
        assert populated_registry.get_entry("sub-1") is not None
        assert populated_registry.get_entry("sub-2") is not None

    def test_all_children_reparented(self, populated_generator, populated_registry):
        populated_generator.generate_level(
            triads_spec={
                "sub-1": ["alpha", "beta", "gamma"],
                "sub-2": ["delta", "epsilon", "zeta"],
            },
            holon_type="subsystem",
            parent_id="mae",
        )
        for name in ("alpha", "beta", "gamma"):
            assert populated_registry.get_entry(name).parent_id == "sub-1"
        for name in ("delta", "epsilon", "zeta"):
            assert populated_registry.get_entry(name).parent_id == "sub-2"


# =====================================================================
# TestFractalGeneratorOrganize
# =====================================================================


class TestFractalGeneratorOrganize:
    """Tests for full organize() with FRACTAL_GROUPING."""

    @pytest.fixture
    def mae_registry(self, registry, conn_registry, bus):
        """Registry mimicking Mae's full bootstrap hierarchy."""
        registry.register("mae", holon_type="organism")
        registry.register("colony", holon_type="colony", parent_id="mae")

        all_systems = set()
        for organ_spec in FRACTAL_GROUPING.values():
            for system_ids in organ_spec.values():
                for sid in system_ids:
                    if sid != "colony":
                        all_systems.add(sid)

        for name in sorted(all_systems):
            registry.register(name, holon_type="system", parent_id="mae")

        return FractalGenerator(
            holon_registry=registry,
            connection_registry=conn_registry,
            event_bus=bus,
        )

    def test_creates_full_hierarchy(self, mae_registry, registry):
        report = mae_registry.organize()
        assert report.organs_created == 5
        assert report.organ_clusters_created == 2
        assert report.subsystems_created == 18

    def test_depth_increased(self, mae_registry, registry):
        mae_registry.organize()
        stats = registry.get_statistics()
        # mae → organ → subsystem → system = depth 3 (no agents in this fixture)
        # Real bootstrap with agents under colony reaches depth 4
        assert stats["max_depth"] >= 3

    def test_organs_are_descendants_of_mae(self, mae_registry, registry):
        mae_registry.organize()
        # After ORGAN_GROUPING, clustered organs are under organ clusters.
        # nervous-system stays as direct child of mae (bridge organ).
        # All organs should be in mae's subtree.
        mae_subtree = registry.get_subtree("mae")
        for organ_name in FRACTAL_GROUPING:
            assert organ_name in mae_subtree

    def test_organ_clusters_are_children_of_mae(self, mae_registry, registry):
        from mae_core.backbone.fractal_generator import ORGAN_GROUPING
        mae_registry.organize()
        mae_children = registry.get_children("mae")
        # Organ clusters should be direct children of mae
        for cluster_name in ORGAN_GROUPING:
            assert cluster_name in mae_children
        # nervous-system is the bridge organ — stays as direct mae child
        assert "nervous-system" in mae_children

    def test_subsystems_are_descendants_of_organs(self, mae_registry, registry):
        mae_registry.organize()
        for organ_name, subsystems_spec in FRACTAL_GROUPING.items():
            organ_subtree = registry.get_subtree(organ_name)
            for sub_name in subsystems_spec:
                assert sub_name in organ_subtree, (
                    f"{sub_name} not in {organ_name} subtree"
                )

    def test_all_systems_placed(self, mae_registry, registry):
        mae_registry.organize()
        # Every system should be reachable from mae
        subtree = registry.get_subtree("mae")
        for organ_spec in FRACTAL_GROUPING.values():
            for system_ids in organ_spec.values():
                for sid in system_ids:
                    assert sid in subtree, f"{sid} not in mae subtree"

    def test_idempotent(self, mae_registry, registry):
        report1 = mae_registry.organize()
        report2 = mae_registry.organize()
        # Second call should create 0 new connections
        assert report2.connections_created == 0

    def test_report_statistics(self, mae_registry):
        report = mae_registry.organize()
        assert report.connections_created > 0
        assert report.max_depth >= 3
        assert len(report.non_triadic_groups) > 0  # consensus has 1 member

    def test_publishes_organized_event(self, mae_registry, bus):
        events = []
        bus.register_callback("fractal.organized", lambda ch, msg: events.append(json.loads(msg)))
        mae_registry.organize()
        assert len(events) == 1
        assert events[0]["organs"] == 5
        assert events[0]["organ_clusters"] == 2


# =====================================================================
# TestFractalGeneratorVerification
# =====================================================================


class TestFractalGeneratorVerification:
    """Tests for verify_triadic_integrity."""

    def test_clean_triads(self, populated_generator):
        populated_generator.generate_triad(
            name="triad-1",
            holon_type="subsystem",
            children_ids=["alpha", "beta", "gamma"],
            parent_id="mae",
        )
        report = populated_generator.verify_triadic_integrity()
        assert report["clean_triads"] == 1
        assert len(report["violations"]) == 0

    def test_non_triadic_reported(self, populated_generator):
        populated_generator.generate_triad(
            name="pair-1",
            holon_type="subsystem",
            children_ids=["alpha", "beta"],
            parent_id="mae",
        )
        report = populated_generator.verify_triadic_integrity()
        assert len(report["violations"]) == 1
        assert report["violations"][0]["children_count"] == 2


# =====================================================================
# TestFractalGeneratorStatistics
# =====================================================================


class TestFractalGeneratorStatistics:
    """Tests for get_statistics."""

    def test_tracks_created_holons(self, populated_generator):
        populated_generator.generate_triad(
            name="triad-1",
            holon_type="subsystem",
            children_ids=["alpha", "beta", "gamma"],
            parent_id="mae",
        )
        stats = populated_generator.get_statistics()
        assert stats["created_holons"] == 1
        assert "triad-1" in stats["created_holon_ids"]

    def test_no_connection_registry(self, populated_registry, bus):
        gen = FractalGenerator(
            holon_registry=populated_registry,
            connection_registry=None,
            event_bus=bus,
        )
        result = gen.generate_triad(
            name="triad-1",
            holon_type="subsystem",
            children_ids=["alpha", "beta", "gamma"],
            parent_id="mae",
        )
        assert result.connections_created == 0
        assert len(result.children_ids) == 3
