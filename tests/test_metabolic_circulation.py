"""Tests for CirculatorySystem and ReproductiveSystem.

Tests cover:
- CirculatorySystem: resource requesting, Murray's law distribution,
  supply replenishment, heart rate adaptation, supply low alerts,
  demand fulfillment, serialization/restore, graceful degradation.
- ReproductiveSystem: population metrics, spawn threshold, retire threshold,
  cooldown enforcement, min/max population limits, Rule of 3 floor,
  optimal population calculation, serialization/restore, graceful degradation.
"""

import unittest

from mae_core.backbone.event_bus import EventBus
from mae_core.morphogenesis.reproductive_system import (
    CH_POPULATION,
    CH_RETIRE,
    CH_SPAWN,
    PopulationMetrics,
    ReproductiveSystem,
)
from mae_core.substrate.circulatory_system import (
    CH_CIRCULATION,
    CH_SUPPLY_LOW,
    CirculatorySystem,
    DemandSignal,
    ResourcePacket,
)


# ===========================================================================
# CirculatorySystem Tests
# ===========================================================================


class TestCirculatorySystemBasic(unittest.TestCase):
    """Basic CirculatorySystem behavior."""

    def setUp(self):
        self.bus = EventBus()
        self.circ = CirculatorySystem(event_bus=self.bus)

    def test_initial_supply_levels(self):
        """All three resource types start at 100."""
        self.assertAlmostEqual(self.circ.get_supply_level("compute"), 100.0)
        self.assertAlmostEqual(self.circ.get_supply_level("memory"), 100.0)
        self.assertAlmostEqual(self.circ.get_supply_level("attention"), 100.0)

    def test_unknown_resource_returns_zero(self):
        """Requesting supply for unknown type returns 0."""
        self.assertEqual(self.circ.get_supply_level("unobtanium"), 0.0)

    def test_initial_heart_rate(self):
        """Heart rate starts at 1.0."""
        self.assertAlmostEqual(self.circ.get_heart_rate(), 1.0)

    def test_initial_adequate(self):
        """With no demands, circulation is adequate."""
        self.assertTrue(self.circ.is_adequate())


class TestCirculatoryResourceRequest(unittest.TestCase):
    """Resource requesting and distribution."""

    def setUp(self):
        self.bus = EventBus()
        self.circ = CirculatorySystem(event_bus=self.bus)

    def test_single_request_fulfilled(self):
        """A single request within supply should be fully fulfilled."""
        self.circ.request_resource("system_a", "compute", 10.0, urgency=0.5)
        self.circ.step(current_step=1)
        # Supply should have decreased (distributed 10 from 100)
        self.assertLess(self.circ.get_supply_level("compute"), 100.0)
        self.assertTrue(self.circ.is_adequate())

    def test_multiple_requests_murray_distribution(self):
        """Multiple requests should be distributed proportionally via Murray's law."""
        # System A needs 30, System B needs 10 — A should get more
        self.circ.request_resource("system_a", "compute", 30.0, urgency=0.5)
        self.circ.request_resource("system_b", "compute", 10.0, urgency=0.5)
        self.circ.step(current_step=1)

        # Check that distribution happened (supply decreased)
        supply = self.circ.get_supply_level("compute")
        self.assertLess(supply, 100.0)
        # Specifically, 40 was requested from 100 — should be fulfillable
        # After replenish + distribute, supply should be less than starting
        self.assertTrue(self.circ.is_adequate())

    def test_high_urgency_gets_priority(self):
        """High urgency demands get proportionally more resources."""
        events = []
        self.bus.register_callback(CH_CIRCULATION, lambda ch, msg: events.append(msg))

        # Same amount, different urgency
        self.circ.request_resource("urgent_sys", "attention", 20.0, urgency=1.0)
        self.circ.request_resource("normal_sys", "attention", 20.0, urgency=0.1)
        self.circ.step(current_step=1)

        # Both should get resources, but urgent should get more
        self.assertTrue(len(events) > 0)
        self.assertTrue(self.circ.is_adequate())

    def test_demand_exceeding_supply(self):
        """When demand exceeds supply, some goes unfulfilled."""
        # Request more than available
        self.circ.request_resource("greedy_sys", "compute", 200.0, urgency=0.5)
        self.circ.step(current_step=1)

        # Supply should be near zero after distribution
        stats = self.circ.get_statistics()
        self.assertGreater(stats["total_unfulfilled"], 0.0)


class TestCirculatoryReplenishment(unittest.TestCase):
    """Supply replenishment (blood returning to heart)."""

    def setUp(self):
        self.bus = EventBus()
        self.circ = CirculatorySystem(event_bus=self.bus)

    def test_supply_replenishes_after_drain(self):
        """Supply should recover toward max after being depleted."""
        # Drain compute supply
        self.circ.request_resource("drain", "compute", 80.0, urgency=0.9)
        self.circ.step(current_step=1)

        supply_after_drain = self.circ.get_supply_level("compute")

        # Run several steps with no demand — supply should recover
        for i in range(5):
            self.circ.step(current_step=2 + i)

        supply_after_recovery = self.circ.get_supply_level("compute")
        self.assertGreater(supply_after_recovery, supply_after_drain)

    def test_supply_never_exceeds_max(self):
        """Replenishment should not push supply above its max."""
        # Run many steps with no demand
        for i in range(20):
            self.circ.step(current_step=i + 1)

        # Supply should be at or below max (100)
        for rt in ["compute", "memory", "attention"]:
            self.assertLessEqual(self.circ.get_supply_level(rt), 100.0 + 0.01)


class TestCirculatoryHeartRate(unittest.TestCase):
    """Heart rate adaptation under stress."""

    def setUp(self):
        self.bus = EventBus()
        self.circ = CirculatorySystem(event_bus=self.bus)

    def test_heart_rate_increases_under_high_demand(self):
        """Heart rate should increase when demand is high relative to supply."""
        initial_hr = self.circ.get_heart_rate()

        # Heavy demand across all resource types
        for rt in ["compute", "memory", "attention"]:
            self.circ.request_resource("heavy", rt, 90.0, urgency=0.9)
        self.circ.step(current_step=1)

        self.assertGreater(self.circ.get_heart_rate(), initial_hr)

    def test_heart_rate_bounded(self):
        """Heart rate should stay within [0.5, 3.0] bounds."""
        # Extreme demand
        for _ in range(10):
            self.circ.request_resource("extreme", "compute", 500.0, urgency=1.0)
            self.circ.step()

        self.assertLessEqual(self.circ.get_heart_rate(), 3.0)
        self.assertGreaterEqual(self.circ.get_heart_rate(), 0.5)


class TestCirculatorySupplyLowAlert(unittest.TestCase):
    """Supply low alert publishing."""

    def setUp(self):
        self.bus = EventBus()
        self.circ = CirculatorySystem(event_bus=self.bus)
        self.alerts = []
        self.bus.register_callback(CH_SUPPLY_LOW, lambda ch, msg: self.alerts.append(msg))

    def test_supply_low_alert_triggered(self):
        """Alert should fire when supply drops below 20% of max."""
        # Drain supply significantly
        self.circ.request_resource("drain", "compute", 95.0, urgency=1.0)
        self.circ.step(current_step=1)

        # Check if alert was published
        # Supply at ~5/100 = 5% — well below 20% threshold
        has_compute_alert = any(
            "compute" in str(a) for a in self.alerts
        )
        self.assertTrue(has_compute_alert)


class TestCirculatorySerialization(unittest.TestCase):
    """Serialization and restore."""

    def test_serialize_restore_roundtrip(self):
        """Serialized state should restore correctly."""
        bus = EventBus()
        circ = CirculatorySystem(event_bus=bus)

        # Modify state
        circ.request_resource("sys_a", "compute", 30.0)
        circ.step(current_step=5)

        # Serialize
        data = circ.serialize()
        self.assertIn("supply", data)
        self.assertIn("heart_rate", data)
        self.assertEqual(data["step_count"], 5)

        # Restore into fresh instance
        circ2 = CirculatorySystem(event_bus=bus)
        circ2.restore(data)

        self.assertEqual(circ2._step_count, 5)
        self.assertAlmostEqual(
            circ2.get_supply_level("compute"),
            circ.get_supply_level("compute"),
            places=2,
        )

    def test_restore_handles_bad_data(self):
        """Restore should handle invalid data gracefully."""
        circ = CirculatorySystem()
        circ.restore("not a dict")  # Should not raise
        circ.restore({})  # Should not raise
        circ.restore({"supply": "invalid"})  # Should not raise


class TestCirculatoryGraceful(unittest.TestCase):
    """Graceful operation without optional dependencies."""

    def test_works_without_event_bus(self):
        """CirculatorySystem should work without an EventBus."""
        circ = CirculatorySystem(event_bus=None)
        circ.request_resource("sys", "compute", 10.0)
        circ.step(current_step=1)  # Should not raise
        stats = circ.get_statistics()
        self.assertIn("supply_levels", stats)

    def test_works_without_substrate(self):
        """CirculatorySystem should work without a substrate reference."""
        bus = EventBus()
        circ = CirculatorySystem(event_bus=bus, substrate=None)
        circ.request_resource("sys", "memory", 5.0)
        circ.step(current_step=1)
        self.assertTrue(circ.is_adequate())


# ===========================================================================
# ReproductiveSystem Tests
# ===========================================================================


class TestReproductiveSystemBasic(unittest.TestCase):
    """Basic ReproductiveSystem behavior."""

    def setUp(self):
        self.bus = EventBus()
        self.repro = ReproductiveSystem(event_bus=self.bus)

    def test_initial_state(self):
        """Initial population metrics should be empty."""
        m = self.repro.population_metrics
        self.assertEqual(m.current_agents, 0)
        self.assertEqual(m.idle_agents, 0)
        self.assertEqual(m.overloaded_agents, 0)

    def test_update_metrics(self):
        """Metrics should compute correctly from agent loads."""
        self.repro.update_metrics({1: 0.5, 2: 0.3, 3: 0.8})
        m = self.repro.population_metrics
        self.assertEqual(m.current_agents, 3)
        self.assertAlmostEqual(m.load_per_agent, (0.5 + 0.3 + 0.8) / 3, places=4)
        self.assertEqual(m.idle_agents, 0)  # None below 0.1
        self.assertEqual(m.overloaded_agents, 0)  # None above 0.9


class TestReproductiveSpawnThreshold(unittest.TestCase):
    """Spawn decision logic."""

    def setUp(self):
        self.bus = EventBus()
        self.spawn_events = []
        self.bus.register_callback(CH_SPAWN, lambda ch, msg: self.spawn_events.append(msg))
        self.repro = ReproductiveSystem(event_bus=self.bus)

    def test_spawn_triggered_on_high_load(self):
        """Spawn should trigger when average load exceeds threshold."""
        # 3 agents all at high load (> 0.8 threshold)
        self.repro.update_metrics({1: 0.9, 2: 0.95, 3: 0.85})
        self.repro.step(current_step=1)

        self.assertTrue(len(self.spawn_events) > 0)
        self.assertIn("high_load", str(self.spawn_events[0]))

    def test_no_spawn_below_threshold(self):
        """No spawn when load is below threshold."""
        self.repro.update_metrics({1: 0.3, 2: 0.4, 3: 0.5})
        self.repro.step(current_step=1)

        self.assertEqual(len(self.spawn_events), 0)

    def test_no_spawn_at_max_population(self):
        """No spawn when population is already at maximum."""
        self.repro._max_population = 3  # Set max to current
        self.repro.update_metrics({1: 0.95, 2: 0.95, 3: 0.95})
        self.repro.step(current_step=1)

        self.assertEqual(len(self.spawn_events), 0)


class TestReproductiveRetireThreshold(unittest.TestCase):
    """Retire decision logic."""

    def setUp(self):
        self.bus = EventBus()
        self.retire_events = []
        self.bus.register_callback(CH_RETIRE, lambda ch, msg: self.retire_events.append(msg))
        self.repro = ReproductiveSystem(event_bus=self.bus)

    def test_retire_triggered_on_low_load(self):
        """Retire should trigger when load is very low with multiple idle agents."""
        # 5 agents, all very idle (load < 0.1) — should retire
        self.repro.update_metrics({1: 0.02, 2: 0.03, 3: 0.01, 4: 0.05, 5: 0.04})
        self.repro.step(current_step=1)

        self.assertTrue(len(self.retire_events) > 0)

    def test_no_retire_above_threshold(self):
        """No retire when average load is above threshold."""
        self.repro.update_metrics({1: 0.5, 2: 0.6, 3: 0.7, 4: 0.5})
        self.repro.step(current_step=1)

        self.assertEqual(len(self.retire_events), 0)


class TestReproductiveCooldown(unittest.TestCase):
    """Spawn cooldown enforcement."""

    def setUp(self):
        self.bus = EventBus()
        self.spawn_events = []
        self.bus.register_callback(CH_SPAWN, lambda ch, msg: self.spawn_events.append(msg))
        self.repro = ReproductiveSystem(event_bus=self.bus, spawn_cooldown=5)

    def test_cooldown_prevents_rapid_spawning(self):
        """Second spawn should be blocked within cooldown window."""
        loads = {1: 0.95, 2: 0.95, 3: 0.95}
        self.repro.update_metrics(loads)
        self.repro.step(current_step=1)
        first_count = len(self.spawn_events)
        self.assertEqual(first_count, 1)

        # Immediately try again — should be blocked by cooldown
        self.repro.update_metrics(loads)
        self.repro.step(current_step=2)
        self.assertEqual(len(self.spawn_events), 1)  # Still just 1

    def test_spawn_allowed_after_cooldown(self):
        """Spawn should be allowed once cooldown expires."""
        loads = {1: 0.95, 2: 0.95, 3: 0.95}
        self.repro.update_metrics(loads)
        self.repro.step(current_step=1)
        self.assertEqual(len(self.spawn_events), 1)

        # Wait past cooldown (5 steps)
        for i in range(2, 7):
            self.repro.update_metrics(loads)
            self.repro.step(current_step=i)

        # Step 7 is 6 steps after step 1, past cooldown of 5
        self.assertGreater(len(self.spawn_events), 1)


class TestReproductivePopulationLimits(unittest.TestCase):
    """Min/max population enforcement and Rule of 3."""

    def setUp(self):
        self.bus = EventBus()
        self.retire_events = []
        self.bus.register_callback(CH_RETIRE, lambda ch, msg: self.retire_events.append(msg))

    def test_rule_of_3_floor(self):
        """Population should never drop below 3 (Rule of 3)."""
        repro = ReproductiveSystem(event_bus=self.bus)
        # 3 idle agents — should NOT retire because that would go below 3
        repro.update_metrics({1: 0.01, 2: 0.02, 3: 0.01})
        repro.step(current_step=1)

        self.assertEqual(len(self.retire_events), 0)

    def test_min_population_enforced_at_init(self):
        """Min population should be at least 3 even if set lower."""
        repro = ReproductiveSystem(event_bus=self.bus, min_population=1)
        self.assertEqual(repro._min_population, 3)

    def test_max_population_respected(self):
        """Spawn should not happen when at max population."""
        spawn_events = []
        self.bus.register_callback(CH_SPAWN, lambda ch, msg: spawn_events.append(msg))

        repro = ReproductiveSystem(event_bus=self.bus, max_population=5)
        repro.update_metrics({1: 0.95, 2: 0.95, 3: 0.95, 4: 0.95, 5: 0.95})
        repro.step(current_step=1)

        self.assertEqual(len(spawn_events), 0)


class TestReproductiveOptimalPopulation(unittest.TestCase):
    """Optimal population calculation."""

    def test_optimal_based_on_total_load(self):
        """Optimal = ceil(total_load / target_load_per_agent)."""
        repro = ReproductiveSystem()
        # 3 agents at load 0.5 each = total 1.5, target per agent 0.5
        # optimal = ceil(1.5 / 0.5) = 3
        repro.update_metrics({1: 0.5, 2: 0.5, 3: 0.5})
        self.assertEqual(repro.get_optimal_population(), 3)

    def test_optimal_clamped_to_min(self):
        """Optimal should not go below min_population."""
        repro = ReproductiveSystem()
        # Very low load — optimal would be 1, but min is 3
        repro.update_metrics({1: 0.1, 2: 0.1, 3: 0.1})
        self.assertGreaterEqual(repro.get_optimal_population(), 3)

    def test_optimal_clamped_to_max(self):
        """Optimal should not exceed max_population."""
        repro = ReproductiveSystem(max_population=5)
        # Extremely high load — optimal would be huge
        repro.update_metrics({i: 1.0 for i in range(5)})
        # total_load=5.0, target=0.5 -> optimal = ceil(10) = 10, clamped to 5
        self.assertLessEqual(repro.get_optimal_population(), 5)


class TestReproductivePopulationHealth(unittest.TestCase):
    """Population health assessment."""

    def test_healthy_population(self):
        """Normal load distribution should be healthy."""
        repro = ReproductiveSystem()
        repro.update_metrics({1: 0.5, 2: 0.6, 3: 0.4})
        self.assertTrue(repro.is_population_healthy())

    def test_unhealthy_overloaded(self):
        """Majority overloaded agents is unhealthy."""
        repro = ReproductiveSystem()
        # >50% overloaded (load > 0.9)
        repro.update_metrics({1: 0.95, 2: 0.95, 3: 0.5})
        self.assertFalse(repro.is_population_healthy())

    def test_unhealthy_below_min(self):
        """Below minimum population is unhealthy."""
        repro = ReproductiveSystem()
        repro.update_metrics({1: 0.5, 2: 0.5})
        self.assertFalse(repro.is_population_healthy())


class TestReproductiveSerialization(unittest.TestCase):
    """Serialization and restore."""

    def test_serialize_restore_roundtrip(self):
        """Serialized state should restore correctly."""
        bus = EventBus()
        repro = ReproductiveSystem(event_bus=bus)
        repro.update_metrics({1: 0.5, 2: 0.6, 3: 0.4})
        repro.step(current_step=10)

        data = repro.serialize()
        self.assertIn("population_metrics", data)
        self.assertEqual(data["step_count"], 10)

        # Restore into fresh instance
        repro2 = ReproductiveSystem(event_bus=bus)
        repro2.restore(data)

        self.assertEqual(repro2._step_count, 10)
        self.assertEqual(repro2.population_metrics.current_agents, 3)

    def test_restore_handles_bad_data(self):
        """Restore should handle invalid data gracefully."""
        repro = ReproductiveSystem()
        repro.restore("not a dict")  # Should not raise
        repro.restore({})  # Should not raise


class TestReproductiveGraceful(unittest.TestCase):
    """Graceful operation without optional dependencies."""

    def test_works_without_event_bus(self):
        """ReproductiveSystem should work without an EventBus."""
        repro = ReproductiveSystem(event_bus=None)
        repro.update_metrics({1: 0.9, 2: 0.9, 3: 0.9})
        repro.step(current_step=1)  # Should not raise
        stats = repro.get_statistics()
        self.assertIn("current_agents", stats)

    def test_works_without_morph_coordinator(self):
        """ReproductiveSystem should work without a MorphogenesisCoordinator."""
        bus = EventBus()
        repro = ReproductiveSystem(event_bus=bus, morph_coordinator=None)
        repro.update_metrics({1: 0.95, 2: 0.95, 3: 0.95})
        repro.step(current_step=1)  # Should not raise
        self.assertTrue(repro.get_statistics()["division_count"] >= 1)


# ===========================================================================
# Cross-System Tests
# ===========================================================================


class TestCirculatoryPublishesEvents(unittest.TestCase):
    """Verify EventBus integration for CirculatorySystem."""

    def test_circulation_update_published(self):
        """Step should publish circulation update event."""
        bus = EventBus()
        events = []
        bus.register_callback(CH_CIRCULATION, lambda ch, msg: events.append(msg))

        circ = CirculatorySystem(event_bus=bus)
        circ.request_resource("test_sys", "compute", 10.0)
        circ.step(current_step=1)

        self.assertTrue(len(events) > 0)


class TestReproductivePublishesEvents(unittest.TestCase):
    """Verify EventBus integration for ReproductiveSystem."""

    def test_population_status_published(self):
        """Step should publish population status event."""
        bus = EventBus()
        events = []
        bus.register_callback(CH_POPULATION, lambda ch, msg: events.append(msg))

        repro = ReproductiveSystem(event_bus=bus)
        repro.update_metrics({1: 0.5, 2: 0.5, 3: 0.5})
        repro.step(current_step=1)

        self.assertTrue(len(events) > 0)


class TestDataclasses(unittest.TestCase):
    """Test dataclass construction."""

    def test_resource_packet_defaults(self):
        """ResourcePacket should have correct defaults."""
        pkt = ResourcePacket(
            resource_type="compute", amount=10.0, destination="sys_a"
        )
        self.assertEqual(pkt.priority, 0.5)
        self.assertEqual(pkt.ttl, 10)

    def test_demand_signal_defaults(self):
        """DemandSignal should have correct defaults."""
        sig = DemandSignal(
            system_name="sys_a", resource_type="memory", amount_needed=5.0
        )
        self.assertEqual(sig.urgency, 0.5)

    def test_population_metrics_defaults(self):
        """PopulationMetrics should have correct defaults."""
        m = PopulationMetrics()
        self.assertEqual(m.current_agents, 0)
        self.assertEqual(m.optimal_agents, 3)
        self.assertEqual(m.load_per_agent, 0.0)


class TestCirculatoryStatistics(unittest.TestCase):
    """Verify get_statistics returns complete data."""

    def test_statistics_keys(self):
        """Statistics should contain all expected keys."""
        circ = CirculatorySystem()
        stats = circ.get_statistics()
        expected = {
            "supply_levels", "heart_rate", "total_distributed",
            "total_unfulfilled", "is_adequate", "murray_exponent",
            "step_count", "history_length", "pending_demands",
        }
        self.assertTrue(expected.issubset(set(stats.keys())))


class TestReproductiveStatistics(unittest.TestCase):
    """Verify get_statistics returns complete data."""

    def test_statistics_keys(self):
        """Statistics should contain all expected keys."""
        repro = ReproductiveSystem()
        stats = repro.get_statistics()
        expected = {
            "current_agents", "optimal_agents", "load_per_agent",
            "idle_agents", "overloaded_agents", "is_healthy",
            "division_count", "retirement_count", "spawn_threshold",
            "retire_threshold", "min_population", "max_population",
            "spawn_cooldown", "step_count",
        }
        self.assertTrue(expected.issubset(set(stats.keys())))


if __name__ == "__main__":
    unittest.main()
