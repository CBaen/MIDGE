"""Tests for ConnectionRegistry - triadic witnessing for all connections."""

import json

import pytest

from mae_core.backbone.connection_registry import (
    CH_CONNECTION_REGISTERED,
    CH_CONNECTION_VERIFIED,
    ConnectionCriticality,
    ConnectionRegistry,
    ConnectionTriad,
    ConnectionType,
)
from mae_core.backbone.connection_registrations import register_all_connections
from mae_core.backbone.event_bus import EventBus


class MockSomaticMap:
    """Minimal SomaticMap mock for witness assignment tests."""

    def __init__(self):
        self._systems = {}

    def register_system(self, system_id, system_type="mock", dependencies=None):
        node = type("Node", (), {
            "system_id": system_id,
            "upstream": set(),
            "downstream": set(),
        })()
        self._systems[system_id] = node
        return node

    def get_system_info(self, system_id):
        return self._systems.get(system_id)

    def add_dependency(self, child, parent):
        if child in self._systems and parent in self._systems:
            self._systems[child].upstream.add(parent)
            self._systems[parent].downstream.add(child)


# =========================================================================
# ConnectionTriad dataclass
# =========================================================================

class TestConnectionTriad:
    def test_creates_with_defaults(self):
        t = ConnectionTriad(
            connection_id="a->b",
            source="a",
            target="b",
        )
        assert t.source == "a"
        assert t.target == "b"
        assert t.witness is None
        assert t.healthy is True
        assert t.connection_type == ConnectionType.DIRECT_REFERENCE

    def test_creates_with_all_fields(self):
        t = ConnectionTriad(
            connection_id="a->b:channel",
            source="a",
            target="b",
            witnesses=["c", "d"],
            connection_type=ConnectionType.EVENTBUS_PUBSUB,
            channel="test.channel",
            criticality=ConnectionCriticality.CRITICAL,
            description="test connection",
        )
        assert t.witness == "c"
        assert len(t.witnesses) == 2
        assert t.channel == "test.channel"
        assert t.criticality == ConnectionCriticality.CRITICAL

    def test_connection_type_values(self):
        assert ConnectionType.EVENTBUS_PUBSUB.value == "eventbus_pubsub"
        assert ConnectionType.DIRECT_REFERENCE.value == "direct_reference"
        assert ConnectionType.CALLBACK_REGISTRATION.value == "callback_registration"
        assert ConnectionType.STEP_HOOK.value == "step_hook"


# =========================================================================
# Witness Assignment
# =========================================================================

class TestWitnessAssignment:
    def test_fallback_to_nervous_system_without_somatic_map(self):
        registry = ConnectionRegistry()
        triad = registry.register_connection(
            "a", "b", ConnectionType.DIRECT_REFERENCE,
        )
        assert len(triad.witnesses) >= 2
        for w in triad.witnesses:
            assert w in ("enforcer", "watchdog", "auditor", "somatic_map")

    def test_assigns_from_nervous_system(self):
        somatic = MockSomaticMap()
        somatic.register_system("a")
        somatic.register_system("b")
        somatic.register_system("enforcer")
        somatic.register_system("watchdog")
        somatic.register_system("auditor")
        somatic.register_system("somatic_map")

        registry = ConnectionRegistry(somatic_map=somatic)
        triad = registry.register_connection(
            "a", "b", ConnectionType.DIRECT_REFERENCE,
        )
        assert triad.witness in ("enforcer", "watchdog", "auditor", "somatic_map")

    def test_prefers_shared_neighbor(self):
        somatic = MockSomaticMap()
        somatic.register_system("a")
        somatic.register_system("b")
        somatic.register_system("shared_witness")
        somatic.register_system("enforcer")

        # shared_witness connects to both a and b
        somatic.add_dependency("a", "shared_witness")
        somatic.add_dependency("b", "shared_witness")

        registry = ConnectionRegistry(somatic_map=somatic)
        triad = registry.register_connection(
            "a", "b", ConnectionType.DIRECT_REFERENCE,
        )
        assert triad.witness == "shared_witness"

    def test_does_not_witness_self(self):
        somatic = MockSomaticMap()
        somatic.register_system("enforcer")
        somatic.register_system("b")
        somatic.register_system("watchdog")
        somatic.register_system("auditor")
        somatic.register_system("somatic_map")

        registry = ConnectionRegistry(somatic_map=somatic)
        triad = registry.register_connection(
            "enforcer", "b", ConnectionType.DIRECT_REFERENCE,
        )
        # Witness should not be 'enforcer' (the source)
        assert triad.witness != "enforcer"
        assert triad.witness in ("watchdog", "auditor", "somatic_map")


# =========================================================================
# Registration
# =========================================================================

class TestConnectionRegistration:
    def test_register_basic(self):
        registry = ConnectionRegistry()
        triad = registry.register_connection(
            "auto_healer", "substrate", ConnectionType.DIRECT_REFERENCE,
        )
        assert triad.connection_id == "auto_healer->substrate"
        assert triad.source == "auto_healer"
        assert triad.target == "substrate"

    def test_register_with_channel(self):
        registry = ConnectionRegistry()
        triad = registry.register_connection(
            "memory", "endocrine", ConnectionType.EVENTBUS_PUBSUB,
            channel="memory.consolidation_started",
        )
        assert triad.connection_id == "memory->endocrine:memory.consolidation_started"

    def test_duplicate_rejected(self):
        registry = ConnectionRegistry()
        t1 = registry.register_connection("a", "b", ConnectionType.DIRECT_REFERENCE)
        t2 = registry.register_connection("a", "b", ConnectionType.DIRECT_REFERENCE)
        assert t1 is t2
        assert registry.get_statistics()["total_connections"] == 1

    def test_get_connection(self):
        registry = ConnectionRegistry()
        registry.register_connection("a", "b", ConnectionType.DIRECT_REFERENCE)
        found = registry.get_connection("a->b")
        assert found is not None
        assert found.source == "a"

    def test_get_connections_for_system(self):
        registry = ConnectionRegistry()
        registry.register_connection("a", "b", ConnectionType.DIRECT_REFERENCE)
        registry.register_connection("a", "c", ConnectionType.DIRECT_REFERENCE)
        registry.register_connection("d", "b", ConnectionType.DIRECT_REFERENCE)

        a_conns = registry.get_connections_for_system("a")
        assert len(a_conns) == 2

        b_conns = registry.get_connections_for_system("b")
        assert len(b_conns) == 2

    def test_deregister(self):
        registry = ConnectionRegistry()
        registry.register_connection("a", "b", ConnectionType.DIRECT_REFERENCE)
        assert registry.deregister_connection("a->b") is True
        assert registry.get_connection("a->b") is None

    def test_deregister_nonexistent(self):
        registry = ConnectionRegistry()
        assert registry.deregister_connection("nonexistent") is False


# =========================================================================
# Verification
# =========================================================================

class TestVerification:
    def test_verify_all_healthy(self):
        somatic = MockSomaticMap()
        somatic.register_system("a")
        somatic.register_system("b")
        # Register all nervous system witnesses (auto-fill assigns 2)
        for ns in ("enforcer", "watchdog", "auditor", "somatic_map"):
            somatic.register_system(ns)

        registry = ConnectionRegistry(somatic_map=somatic)
        registry.register_connection(
            "a", "b", ConnectionType.DIRECT_REFERENCE, witness="enforcer",
        )

        results = registry.verify_all()
        assert results["total"] == 1
        assert results["healthy"] == 1
        assert results["unhealthy"] == 0

    def test_verify_detects_missing_system(self):
        somatic = MockSomaticMap()
        somatic.register_system("a")
        # 'b' is NOT registered with somatic map — should be unhealthy
        for ns in ("enforcer", "watchdog", "auditor", "somatic_map"):
            somatic.register_system(ns)

        registry = ConnectionRegistry(somatic_map=somatic)
        registry.register_connection(
            "a", "b", ConnectionType.DIRECT_REFERENCE, witness="enforcer",
        )

        results = registry.verify_all()
        assert results["unhealthy"] == 1

    def test_step_triggers_verification(self):
        registry = ConnectionRegistry(verify_interval=5)
        registry.register_connection("a", "b", ConnectionType.DIRECT_REFERENCE)

        # Should not verify until interval
        for i in range(4):
            registry.step()
        assert registry.get_statistics()["total_verifications"] == 0

        # 5th step triggers verification
        registry.step()
        assert registry.get_statistics()["total_verifications"] == 1

    def test_verify_without_somatic_map(self):
        registry = ConnectionRegistry()
        registry.register_connection("a", "b", ConnectionType.DIRECT_REFERENCE)
        results = registry.verify_all()
        # Without SomaticMap, all are considered healthy
        assert results["healthy"] == 1


# =========================================================================
# Bare Dyad Detection
# =========================================================================

class TestBareDyads:
    def test_no_bare_dyads_with_witnesses(self):
        registry = ConnectionRegistry()
        registry.register_connection(
            "a", "b", ConnectionType.DIRECT_REFERENCE, witness="c",
        )
        assert len(registry.get_bare_dyads()) == 0

    def test_detects_bare_dyad(self):
        registry = ConnectionRegistry()
        # Force a bare dyad by creating one with no witnesses
        triad = ConnectionTriad(
            connection_id="bare->dyad",
            source="bare",
            target="dyad",
        )
        registry._connections["bare->dyad"] = triad
        bare = registry.get_bare_dyads()
        assert len(bare) == 1
        assert bare[0].connection_id == "bare->dyad"


# =========================================================================
# Coverage Report
# =========================================================================

class TestCoverageReport:
    def test_full_coverage(self):
        registry = ConnectionRegistry()
        registry.register_connection(
            "a", "b", ConnectionType.DIRECT_REFERENCE, witness="c",
        )
        report = registry.get_coverage_report()
        assert report["total_systems"] == 2
        assert report["systems_with_witnesses"] == 2
        assert report["coverage_pct"] == 100.0

    def test_partial_coverage(self):
        registry = ConnectionRegistry()
        registry.register_connection(
            "a", "b", ConnectionType.DIRECT_REFERENCE, witness="c",
        )
        # Add a bare dyad
        triad = ConnectionTriad(
            connection_id="d->e",
            source="d",
            target="e",
        )
        registry._connections["d->e"] = triad
        registry._system_connections.setdefault("d", []).append("d->e")
        registry._system_connections.setdefault("e", []).append("d->e")

        report = registry.get_coverage_report()
        assert report["total_systems"] == 4
        assert report["bare_dyads"] == 1


# =========================================================================
# Statistics
# =========================================================================

class TestStatistics:
    def test_statistics_structure(self):
        registry = ConnectionRegistry()
        registry.register_connection("a", "b", ConnectionType.DIRECT_REFERENCE)
        registry.register_connection(
            "c", "d", ConnectionType.EVENTBUS_PUBSUB, channel="test",
        )

        stats = registry.get_statistics()
        assert stats["total_connections"] == 2
        assert "type_counts" in stats
        assert "direct_reference" in stats["type_counts"]
        assert "eventbus_pubsub" in stats["type_counts"]
        assert "witness_load" in stats
        assert stats["systems_connected"] >= 4

    def test_unhealthy_count(self):
        registry = ConnectionRegistry()
        triad = registry.register_connection("a", "b", ConnectionType.DIRECT_REFERENCE)
        triad.healthy = False
        stats = registry.get_statistics()
        assert stats["unhealthy"] == 1


# =========================================================================
# EventBus Integration
# =========================================================================

class TestEventBusIntegration:
    def test_publishes_on_registration(self):
        bus = EventBus()
        events = []
        bus.register_callback(
            CH_CONNECTION_REGISTERED,
            lambda ch, msg: events.append(json.loads(msg)),
        )

        registry = ConnectionRegistry(event_bus=bus)
        registry.register_connection("a", "b", ConnectionType.DIRECT_REFERENCE)

        assert len(events) == 1
        assert events[0]["source"] == "a"
        assert events[0]["target"] == "b"

    def test_publishes_on_verification(self):
        bus = EventBus()
        events = []
        bus.register_callback(
            CH_CONNECTION_VERIFIED,
            lambda ch, msg: events.append(json.loads(msg)),
        )

        registry = ConnectionRegistry(event_bus=bus)
        registry.register_connection("a", "b", ConnectionType.DIRECT_REFERENCE)
        registry.verify_all()

        assert len(events) == 1
        assert events[0]["total"] == 1


# =========================================================================
# register_all_connections function
# =========================================================================

class TestRegisterAllConnections:
    def test_registers_connections(self):
        bus = EventBus()
        somatic = MockSomaticMap()

        # Register enough systems for the function to find witnesses
        for name in (
            "enforcer", "watchdog", "auditor", "somatic_map",
            "memory", "morphogenesis", "endocrine", "healing", "defense",
            "improvement", "frl", "imitation", "haven", "auto_healer",
            "substrate", "circadian", "gnn_communicator", "predictive_field",
            "physarum", "pearl_defense", "input_validator", "morph_coordinator",
            "organ_builder", "model", "collective_dream", "shared_world_model",
            "worldline_planner", "temporal_memory", "shared_causal_engine",
            "transfer_engine", "maml_learner", "threat_detector", "curiosity",
            "event_bus", "capability_discovery", "triad_audit",
            # Systems referenced by Group 1-5 registrations
            "organism_state", "emotional_system", "homeostasis",
            "thermoregulation", "digestive_system", "respiratory_system",
            "vestibular_system", "nociception", "lymphatic_system",
            "senescence", "proprioception", "microbiome",
            "boundary_membrane", "renal_filter", "energy_reserve",
            "circulatory_system", "reproductive_system",
            "connection_registry", "holon_registry", "fractal_generator",
            "goal_manager", "metacognition", "theory_of_mind",
            "validated_imagination", "decision_router", "stem_cell_registry",
            # Systems referenced by Group 6-9 registrations (triadic audit 2026-02-12)
            "pattern_cortex", "pattern_bus", "pattern_consolidator",
            "pattern_distiller", "memory_bridge", "attentional_gate",
            "signal_bus",
        ):
            somatic.register_system(name)

        registry = ConnectionRegistry(event_bus=bus, somatic_map=somatic)
        counts = register_all_connections(registry, {})

        assert counts["total"] > 40
        assert counts.get("eventbus_pubsub", 0) > 10
        assert counts.get("direct_reference", 0) > 10
        assert counts.get("callback_registration", 0) > 0
        assert counts.get("step_hook", 0) > 5

    def test_all_connections_have_witnesses(self):
        bus = EventBus()
        somatic = MockSomaticMap()
        for name in (
            "enforcer", "watchdog", "auditor", "somatic_map",
            "memory", "morphogenesis", "endocrine", "healing", "defense",
            "improvement", "frl", "imitation", "haven", "auto_healer",
            "substrate", "circadian", "gnn_communicator", "predictive_field",
            "physarum", "pearl_defense", "input_validator", "morph_coordinator",
            "organ_builder", "model", "collective_dream", "shared_world_model",
            "worldline_planner", "temporal_memory", "shared_causal_engine",
            "transfer_engine", "maml_learner", "threat_detector", "curiosity",
            "event_bus", "capability_discovery", "triad_audit",
            # Systems referenced by Group 1-5 registrations
            "organism_state", "emotional_system", "homeostasis",
            "thermoregulation", "digestive_system", "respiratory_system",
            "vestibular_system", "nociception", "lymphatic_system",
            "senescence", "proprioception", "microbiome",
            "boundary_membrane", "renal_filter", "energy_reserve",
            "circulatory_system", "reproductive_system",
            "connection_registry", "holon_registry", "fractal_generator",
            "goal_manager", "metacognition", "theory_of_mind",
            "validated_imagination", "decision_router", "stem_cell_registry",
            # Systems referenced by Group 6-9 registrations (triadic audit 2026-02-12)
            "pattern_cortex", "pattern_bus", "pattern_consolidator",
            "pattern_distiller", "memory_bridge", "attentional_gate",
            "signal_bus",
        ):
            somatic.register_system(name)

        registry = ConnectionRegistry(event_bus=bus, somatic_map=somatic)
        register_all_connections(registry, {})

        bare = registry.get_bare_dyads()
        assert len(bare) == 0, (
            f"Bare dyads found: {[b.connection_id for b in bare]}"
        )


# =========================================================================
# Euler Topological Invariant (Sacred Geometry — Part 4)
# =========================================================================

class TestEulerInvariant:
    def test_empty_registry(self):
        registry = ConnectionRegistry()
        stats = registry.get_euler_statistics()
        assert stats["vertices"] == 0
        assert stats["edges"] == 0
        assert stats["components"] == 0
        assert stats["excess_edges"] == 0
        assert stats["euler_characteristic"] == 0

    def test_triangle_three_nodes_three_edges(self):
        """A triangle (K3): V=3, E=3, C=1, excess=1, chi=1."""
        registry = ConnectionRegistry()
        registry.register_connection("a", "b", ConnectionType.DIRECT_REFERENCE)
        registry.register_connection("b", "c", ConnectionType.DIRECT_REFERENCE)
        registry.register_connection("a", "c", ConnectionType.DIRECT_REFERENCE)
        stats = registry.get_euler_statistics()
        assert stats["vertices"] == 3
        assert stats["edges"] == 3
        assert stats["components"] == 1
        # Spanning tree for 3 nodes = 2 edges; excess = 3 - 2 = 1
        assert stats["excess_edges"] == 1
        assert stats["euler_characteristic"] == 1  # 3 - 3 + 1

    def test_tree_zero_excess(self):
        """A tree (chain a-b-c-d): V=4, E=3, C=1, excess=0, chi=2."""
        registry = ConnectionRegistry()
        registry.register_connection("a", "b", ConnectionType.DIRECT_REFERENCE)
        registry.register_connection("b", "c", ConnectionType.DIRECT_REFERENCE)
        registry.register_connection("c", "d", ConnectionType.DIRECT_REFERENCE)
        stats = registry.get_euler_statistics()
        assert stats["vertices"] == 4
        assert stats["edges"] == 3
        assert stats["components"] == 1
        assert stats["excess_edges"] == 0
        assert stats["euler_characteristic"] == 2  # 4 - 3 + 1

    def test_shortcuts_increase_excess(self):
        """Adding a shortcut edge increases excess_edges by 1."""
        registry = ConnectionRegistry()
        # Build a tree first
        registry.register_connection("a", "b", ConnectionType.DIRECT_REFERENCE)
        registry.register_connection("b", "c", ConnectionType.DIRECT_REFERENCE)
        registry.register_connection("c", "d", ConnectionType.DIRECT_REFERENCE)

        stats_tree = registry.get_euler_statistics()
        assert stats_tree["excess_edges"] == 0

        # Add a shortcut (a-d bypasses the chain)
        registry.register_connection("a", "d", ConnectionType.EVENTBUS_PUBSUB,
                                     channel="shortcut")
        stats_shortcut = registry.get_euler_statistics()
        assert stats_shortcut["excess_edges"] == 1
        assert stats_shortcut["edges"] == 4

    def test_disconnected_components(self):
        """Two disconnected pairs: V=4, E=2, C=2, excess=0."""
        registry = ConnectionRegistry()
        registry.register_connection("a", "b", ConnectionType.DIRECT_REFERENCE)
        registry.register_connection("c", "d", ConnectionType.DIRECT_REFERENCE)
        stats = registry.get_euler_statistics()
        assert stats["vertices"] == 4
        assert stats["edges"] == 2
        assert stats["components"] == 2
        assert stats["excess_edges"] == 0
        assert stats["euler_characteristic"] == 4  # 4 - 2 + 2

    def test_check_euler_invariant_empty(self):
        registry = ConnectionRegistry()
        result = registry.check_euler_invariant()
        assert result["interpretation"] == "empty_graph"

    def test_check_euler_invariant_tree(self):
        registry = ConnectionRegistry()
        registry.register_connection("a", "b", ConnectionType.DIRECT_REFERENCE)
        result = registry.check_euler_invariant()
        assert result["interpretation"] == "tree_or_forest"

    def test_check_euler_invariant_with_shortcuts(self):
        registry = ConnectionRegistry()
        registry.register_connection("a", "b", ConnectionType.DIRECT_REFERENCE)
        registry.register_connection("b", "c", ConnectionType.DIRECT_REFERENCE)
        registry.register_connection("a", "c", ConnectionType.DIRECT_REFERENCE)
        result = registry.check_euler_invariant()
        assert result["interpretation"] == "shortcuts_present"

    def test_verify_all_includes_euler(self):
        registry = ConnectionRegistry()
        registry.register_connection("a", "b", ConnectionType.DIRECT_REFERENCE)
        results = registry.verify_all()
        assert "euler" in results
        assert results["euler"]["vertices"] == 2
        assert results["euler"]["edges"] == 1
