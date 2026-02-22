"""Tests for Mae's maintenance systems: Lymphatic, Senescence, Proprioception, Microbiome.

These four systems handle waste collection, aging, body awareness, and
symbiotic processing — the maintenance layer that keeps Mae healthy.
"""

from __future__ import annotations

import json

import pytest

from mae_core.backbone.event_bus import EventBus
from mae_core.emergent.lymphatic_system import (
    CH_LYMPH_OVERFLOW,
    CH_LYMPH_STATUS,
    CH_RECYCLED,
    LymphaticSystem,
    WasteItem,
)
from mae_core.emergent.senescence import (
    CH_AGE_UPDATE,
    CH_REJUVENATION,
    CH_SENESCENT,
    SenescenceManager,
    SystemAge,
)
from mae_core.emergent.proprioception import (
    CH_PROPRIOCEPTION,
    CH_TOPOLOGY_CHANGE,
    ProprioceptionSystem,
    SystemPosition,
)
from mae_core.emergent.microbiome import (
    CH_DYSBIOSIS,
    CH_MICROBIOME_STATUS,
    Microbiome,
    MicrobialStrain,
)


# =====================================================================
# Helpers
# =====================================================================

class EventCapture:
    """Captures events published on an EventBus channel."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    def __call__(self, channel: str, message) -> None:
        if isinstance(message, str):
            try:
                self.events.append(json.loads(message))
            except (json.JSONDecodeError, TypeError):
                self.events.append({"raw": message})
        elif isinstance(message, dict):
            self.events.append(message)


@pytest.fixture
def bus():
    return EventBus()


# =====================================================================
# LymphaticSystem Tests
# =====================================================================


class TestLymphaticSystem:
    """Tests for waste collection, processing, recycling, and overflow."""

    def test_collect_waste(self):
        """collect_waste adds items to the waste bin."""
        lymph = LymphaticSystem()
        lymph.collect_waste("memory", "expired_memory", {"key": "old"}, current_step=10)
        assert lymph._total_collected == 1
        assert len(lymph._waste_bin) == 1
        assert lymph._waste_bin[0].source == "memory"
        assert lymph._waste_bin[0].waste_type == "expired_memory"

    def test_process_waste_recycling(self):
        """Recyclable waste (stale_marker, expired_memory) gets recycled."""
        bus = EventBus()
        capture = EventCapture()
        bus.register_callback(CH_RECYCLED, capture)

        lymph = LymphaticSystem(event_bus=bus)
        lymph.collect_waste("stigmergy", "stale_marker", {"id": "m1"})
        lymph.collect_waste("memory", "expired_memory", {"key": "old"})

        recycled, disposed = lymph._process_waste()
        assert recycled == 2
        assert disposed == 0
        assert len(capture.events) == 2

    def test_process_waste_disposal(self):
        """Non-recyclable waste (dead_connection, orphan_subscription) gets disposed."""
        lymph = LymphaticSystem()
        lymph.collect_waste("registry", "dead_connection", {"id": "c1"})
        lymph.collect_waste("bus", "orphan_subscription", {"ch": "x"})

        recycled, disposed = lymph._process_waste()
        assert recycled == 0
        assert disposed == 2
        assert lymph._disposed_count == 2

    def test_sweep_stale_markers(self):
        """_sweep_stale_markers collects old markers from stigmergy."""

        class FakeStigmergy:
            def __init__(self):
                self._markers = {
                    "m1": type("M", (), {"metadata": {"step": 5}, "marker_type": "SUCCESS"})(),
                    "m2": type("M", (), {"metadata": {"step": 100}, "marker_type": "DANGER"})(),
                }

        stig = FakeStigmergy()
        lymph = LymphaticSystem(stigmergy=stig)
        lymph._sweep_stale_markers(current_step=60)

        # m1 is stale (age 55 > 50), m2 is fresh (age -40 won't trigger)
        assert lymph._total_collected == 1
        assert "m1" not in stig._markers
        assert "m2" in stig._markers

    def test_sweep_dead_connections(self):
        """_sweep_dead_connections collects unhealthy connections."""

        class FakeConn:
            def __init__(self):
                self.connection_id = "a->b"
                self.source = "a"
                self.target = "b"

        class FakeRegistry:
            def get_unhealthy_connections(self):
                return [FakeConn()]

        lymph = LymphaticSystem(connection_registry=FakeRegistry())
        lymph._sweep_dead_connections()

        assert lymph._total_collected == 1
        assert lymph._waste_bin[0].waste_type == "dead_connection"

    def test_overflow_alert(self, bus):
        """Overflow event published when waste exceeds capacity."""
        capture = EventCapture()
        bus.register_callback(CH_LYMPH_OVERFLOW, capture)

        lymph = LymphaticSystem(event_bus=bus)
        lymph._max_waste_capacity = 5  # Low threshold for testing

        # Fill beyond capacity
        for i in range(10):
            lymph.collect_waste("test", "orphan_subscription", {"i": i})

        lymph.step(current_step=1)
        assert len(capture.events) >= 1
        assert capture.events[0]["capacity_used"] > 1.0

    def test_capacity_tracking(self):
        """get_capacity_used returns correct fraction."""
        lymph = LymphaticSystem()
        lymph._max_waste_capacity = 10
        for i in range(5):
            lymph.collect_waste("test", "stale_marker")
        assert lymph.get_capacity_used() == 0.5

    def test_serialization(self):
        """serialize/restore preserves state."""
        lymph = LymphaticSystem()
        lymph.collect_waste("test", "stale_marker", current_step=5)
        lymph._recycled_count = 10
        lymph._disposed_count = 3

        data = lymph.serialize()
        lymph2 = LymphaticSystem()
        lymph2.restore(data)

        assert lymph2._recycled_count == 10
        assert lymph2._disposed_count == 3
        assert len(lymph2._waste_bin) == 1

    def test_step_publishes_status(self, bus):
        """step() always publishes lymph status."""
        capture = EventCapture()
        bus.register_callback(CH_LYMPH_STATUS, capture)

        lymph = LymphaticSystem(event_bus=bus)
        lymph.step(current_step=1)

        assert len(capture.events) == 1
        assert "collected" in capture.events[0]
        assert "capacity_used" in capture.events[0]

    def test_graceful_without_event_bus(self):
        """LymphaticSystem works without event_bus."""
        lymph = LymphaticSystem()
        lymph.collect_waste("test", "stale_marker")
        lymph.step(current_step=13)
        stats = lymph.get_statistics()
        assert stats["total_collected"] == 1


# =====================================================================
# SenescenceManager Tests
# =====================================================================


class TestSenescenceManager:
    """Tests for aging, wear accumulation, rejuvenation, and organism age."""

    def test_register_system(self):
        """register_system starts tracking a new system."""
        sm = SenescenceManager()
        sm.register_system("test_sys", creation_step=10)
        assert "test_sys" in sm._system_ages
        assert sm._system_ages["test_sys"].creation_step == 10
        assert sm._system_ages["test_sys"].wear_level == 0.0

    def test_wear_accumulation(self):
        """step() accumulates wear on all tracked systems."""
        sm = SenescenceManager()
        sm.register_system("sys_a")

        for i in range(100):
            sm.step(i)

        age = sm._system_ages["sys_a"]
        assert age.wear_level > 0.0

    def test_idle_penalty(self):
        """Idle systems (activity_rate < 0.1) age faster."""
        sm = SenescenceManager()
        sm.register_system("idle_sys")
        sm.register_system("active_sys")

        for i in range(100):
            sm.report_activity("active_sys", active=True)
            # idle_sys gets no activity reports
            sm.step(i)

        idle_wear = sm._system_ages["idle_sys"].wear_level
        active_wear = sm._system_ages["active_sys"].wear_level
        assert idle_wear > active_wear, "Idle system should wear faster"

    def test_activity_bonus(self):
        """Active systems (activity_rate >= 0.1) age slower."""
        sm = SenescenceManager()
        sm.register_system("active_sys")

        for i in range(50):
            sm.report_activity("active_sys", active=True)
            sm.step(i)

        # Activity bonus = 0.5x base rate
        expected_max = 50 * 0.001 * 0.5 + 0.001  # Small tolerance
        assert sm._system_ages["active_sys"].wear_level < expected_max

    def test_rejuvenation(self):
        """rejuvenate() resets wear to 0.3, not 0.0."""
        sm = SenescenceManager()
        sm.register_system("worn_sys")
        sm._system_ages["worn_sys"].wear_level = 0.9

        result = sm.rejuvenate("worn_sys")
        assert result is True
        assert sm._system_ages["worn_sys"].wear_level == 0.3

    def test_rejuvenation_unknown_system(self):
        """rejuvenate() returns False for unknown systems."""
        sm = SenescenceManager()
        assert sm.rejuvenate("nonexistent") is False

    def test_organism_age(self):
        """get_organism_age returns mean wear across all systems."""
        sm = SenescenceManager()
        sm.register_system("a")
        sm.register_system("b")
        sm._system_ages["a"].wear_level = 0.4
        sm._system_ages["b"].wear_level = 0.6
        assert sm.get_organism_age() == pytest.approx(0.5)

    def test_oldest_systems(self):
        """get_oldest_systems returns top N by wear level."""
        sm = SenescenceManager()
        for name, wear in [("a", 0.1), ("b", 0.9), ("c", 0.5)]:
            sm.register_system(name)
            sm._system_ages[name].wear_level = wear

        oldest = sm.get_oldest_systems(n=2)
        assert len(oldest) == 2
        assert oldest[0].system_name == "b"
        assert oldest[1].system_name == "c"

    def test_rejuvenation_event(self, bus):
        """Publish rejuvenation_needed when wear exceeds threshold."""
        capture = EventCapture()
        bus.register_callback(CH_REJUVENATION, capture)

        sm = SenescenceManager(event_bus=bus)
        sm.register_system("worn_sys")
        # Set wear just below threshold; idle penalty = 0.001 * 2.0 = 0.002 per step
        sm._system_ages["worn_sys"].wear_level = 0.799

        # One step pushes past 0.8 threshold (0.799 + 0.002 = 0.801)
        sm.step(0)

        assert len(capture.events) >= 1
        assert capture.events[0]["system_name"] == "worn_sys"

    def test_senescent_event(self, bus):
        """Publish system_senescent when wear reaches max."""
        capture = EventCapture()
        bus.register_callback(CH_SENESCENT, capture)

        sm = SenescenceManager(event_bus=bus)
        sm.register_system("dying_sys")
        sm._system_ages["dying_sys"].wear_level = 0.999

        sm.step(0)

        assert len(capture.events) >= 1
        assert capture.events[0]["system_name"] == "dying_sys"

    def test_serialization(self):
        """serialize/restore preserves system ages."""
        sm = SenescenceManager()
        sm.register_system("test_sys", creation_step=5)
        sm._system_ages["test_sys"].wear_level = 0.42
        sm._system_ages["test_sys"].activity_rate = 0.7

        data = sm.serialize()
        sm2 = SenescenceManager()
        sm2.restore(data)

        assert "test_sys" in sm2._system_ages
        assert sm2._system_ages["test_sys"].wear_level == pytest.approx(0.42)
        assert sm2._system_ages["test_sys"].activity_rate == pytest.approx(0.7)

    def test_graceful_without_event_bus(self):
        """SenescenceManager works without event_bus."""
        sm = SenescenceManager()
        sm.register_system("solo")
        sm.step(0)
        stats = sm.get_statistics()
        assert stats["tracked_systems"] == 1


# =====================================================================
# ProprioceptionSystem Tests
# =====================================================================


class TestProprioceptionSystem:
    """Tests for body map, topology detection, and relative position."""

    def test_build_body_map(self):
        """build_body_map populates positions from FRACTAL_GROUPING."""
        prop = ProprioceptionSystem()
        prop.build_body_map()

        # Should have systems from all 4 organs
        assert len(prop._system_positions) > 0
        # Check a known system exists
        assert "event_bus" in prop._system_positions
        pos = prop._system_positions["event_bus"]
        assert pos.organ == "nervous-system"
        assert pos.subsystem == "core-backbone"
        assert pos.depth == 3

    def test_position_update(self):
        """update_position modifies activity and health."""
        prop = ProprioceptionSystem()
        prop.update_position("test_sys", activity=0.7, health=0.9)

        pos = prop._system_positions["test_sys"]
        assert pos.activity == pytest.approx(0.7)
        assert pos.health == pytest.approx(0.9)

    def test_topology_change_detection(self):
        """detect_topology_change returns True when systems are added."""
        prop = ProprioceptionSystem()
        prop._previous_topology_hash = prop._compute_topology_hash()

        # No change
        assert prop.detect_topology_change() is False

        # Add a system
        prop.update_position("new_system", activity=0.5)
        assert prop.detect_topology_change() is True
        assert prop._topology_version == 1

    def test_relative_position_same_organ(self):
        """get_relative_position detects systems in the same organ."""
        prop = ProprioceptionSystem()
        prop.build_body_map()

        # event_bus and holon_registry are both in nervous-system/core-backbone
        rel = prop.get_relative_position("event_bus", "holon_registry")
        assert rel["same_organ"] is True
        assert rel["same_subsystem"] is True
        assert rel["known"] is True

    def test_relative_position_different_organ(self):
        """get_relative_position detects systems in different organs."""
        prop = ProprioceptionSystem()
        prop.build_body_map()

        # event_bus (nervous) vs substrate (sensory)
        rel = prop.get_relative_position("event_bus", "substrate")
        assert rel["same_organ"] is False
        assert rel["known"] is True

    def test_relative_position_unknown_system(self):
        """get_relative_position handles unknown systems."""
        prop = ProprioceptionSystem()
        rel = prop.get_relative_position("nonexistent_a", "nonexistent_b")
        assert rel["known"] is False

    def test_body_summary(self):
        """get_body_summary returns organ-level aggregates."""
        prop = ProprioceptionSystem()
        prop.build_body_map()

        summary = prop.get_body_summary()
        assert "nervous-system" in summary
        assert summary["nervous-system"]["systems"] > 0
        assert "mean_health" in summary["nervous-system"]

    def test_serialization(self):
        """serialize/restore preserves positions and topology version."""
        prop = ProprioceptionSystem()
        prop.build_body_map()
        prop._topology_version = 5

        data = prop.serialize()
        prop2 = ProprioceptionSystem()
        prop2.restore(data)

        assert prop2._topology_version == 5
        assert "event_bus" in prop2._system_positions
        assert prop2._system_positions["event_bus"].organ == "nervous-system"

    def test_step_publishes_status(self, bus):
        """step() publishes proprioception update."""
        capture = EventCapture()
        bus.register_callback(CH_PROPRIOCEPTION, capture)

        prop = ProprioceptionSystem(event_bus=bus)
        prop.step(current_step=1)

        assert len(capture.events) == 1
        assert "total_systems" in capture.events[0]

    def test_graceful_without_event_bus(self):
        """ProprioceptionSystem works without event_bus."""
        prop = ProprioceptionSystem()
        prop.build_body_map()
        prop.step(current_step=1)
        stats = prop.get_statistics()
        assert stats["total_systems"] > 0


# =====================================================================
# Microbiome Tests
# =====================================================================


class TestMicrobiome:
    """Tests for input processing, population dynamics, and diversity."""

    def test_default_strains(self):
        """Microbiome starts with 5 default strains, one per specialization."""
        mb = Microbiome()
        assert len(mb._strains) == 5
        specs = {s.specialization for s in mb._strains}
        assert len(specs) == 5  # All 5 specializations represented

    def test_process_pattern_input(self):
        """Pattern input routed to pattern_decomposer."""
        mb = Microbiome()
        result = mb.process_input("pattern", {"a": 1, "b": 2, "c": 3})
        assert result["processed"] is True
        assert result["specialization"] == "pattern_decomposer"
        assert "decomposed_parts" in result["result"]

    def test_process_anomaly_input(self):
        """Anomaly input routed to anomaly_detector."""
        mb = Microbiome()
        result = mb.process_input("anomaly", {"x": 200})
        assert result["processed"] is True
        assert result["specialization"] == "anomaly_detector"
        assert result["result"]["is_anomalous"] is True  # value > 100

    def test_process_signal_amplification(self):
        """Weak signal input routed to signal_amplifier."""
        mb = Microbiome()
        result = mb.process_input("weak_signal", {"strength": 0.1})
        assert result["processed"] is True
        assert result["specialization"] == "signal_amplifier"
        assert result["result"]["amplified"]["strength"] > 0.1

    def test_process_noise_filter(self):
        """Noisy input routed to noise_filter, empty values removed."""
        mb = Microbiome()
        result = mb.process_input("noisy", {"good": 1, "empty": "", "none": None})
        assert result["processed"] is True
        assert result["specialization"] == "noise_filter"
        assert result["result"]["removed_count"] == 2

    def test_process_nutrient_synthesizer(self):
        """Data input routed to nutrient_synthesizer."""
        mb = Microbiome()
        result = mb.process_input("data", {"x": 10, "y": 20, "label": "test"})
        assert result["processed"] is True
        assert result["specialization"] == "nutrient_synthesizer"
        assert result["result"]["numeric_mean"] == pytest.approx(15.0)

    def test_population_evolution(self):
        """Active strains grow, idle strains shrink."""
        mb = Microbiome()

        # Process many pattern inputs — pattern_decomposer should dominate
        for _ in range(20):
            mb.process_input("pattern", {"a": 1})
        mb._evolve_populations()

        decomposer = [s for s in mb._strains if s.specialization == "pattern_decomposer"][0]
        idle_strain = [s for s in mb._strains if s.specialization == "noise_filter"][0]

        assert decomposer.fitness > idle_strain.fitness

    def test_diversity_computation(self):
        """Shannon diversity computed correctly."""
        mb = Microbiome()
        # Default strains have equal fitness — max diversity
        diversity = mb._compute_diversity()
        assert diversity > 1.0  # log(5) ~ 1.609 for 5 equal strains

    def test_diversity_low_with_one_dominant(self):
        """Diversity drops when one strain dominates."""
        mb = Microbiome()
        # Kill all but one strain
        for s in mb._strains[1:]:
            s.population = 0.001
        mb._strains[0].population = 19.0

        diversity = mb._compute_diversity()
        assert diversity < 0.5

    def test_dysbiosis_detection(self, bus):
        """Dysbiosis event published when diversity < 0.5."""
        capture = EventCapture()
        bus.register_callback(CH_DYSBIOSIS, capture)

        mb = Microbiome(event_bus=bus)
        # Force low diversity by setting fitness (step() recalculates population from fitness)
        mb._strains[0].fitness = 0.99
        for s in mb._strains[1:]:
            s.fitness = 0.001

        mb.step(current_step=1)
        assert len(capture.events) >= 1
        assert capture.events[0]["diversity"] < 0.5

    def test_introduce_strain(self):
        """introduce_strain adds a new strain to the ecosystem."""
        mb = Microbiome()
        result = mb.introduce_strain("probiotic_1", "noise_filter")
        assert result is True
        assert len(mb._strains) == 6

    def test_introduce_strain_invalid_specialization(self):
        """introduce_strain rejects unknown specializations."""
        mb = Microbiome()
        result = mb.introduce_strain("bad_strain", "teleportation")
        assert result is False
        assert len(mb._strains) == 5

    def test_introduce_strain_duplicate_name(self):
        """introduce_strain rejects duplicate names."""
        mb = Microbiome()
        existing_name = mb._strains[0].name
        result = mb.introduce_strain(existing_name, "noise_filter")
        assert result is False

    def test_dominant_strain(self):
        """get_dominant_strain returns strain with highest population."""
        mb = Microbiome()
        mb._strains[2].population = 10.0  # Make one dominant
        dominant = mb.get_dominant_strain()
        assert dominant is not None
        assert dominant.name == mb._strains[2].name

    def test_serialization(self):
        """serialize/restore preserves strains and metrics."""
        mb = Microbiome()
        mb._strains[0].fitness = 0.9
        mb._total_processed = 42

        data = mb.serialize()
        mb2 = Microbiome()
        mb2.restore(data)

        assert mb2._total_processed == 42
        assert mb2._strains[0].fitness == pytest.approx(0.9)
        assert len(mb2._strains) == 5

    def test_graceful_without_event_bus(self):
        """Microbiome works without event_bus."""
        mb = Microbiome()
        result = mb.process_input("pattern", {"a": 1})
        assert result["processed"] is True
        mb.step(current_step=1)
        stats = mb.get_statistics()
        assert stats["total_strains"] == 5
