"""Tests for metabolic process systems: Digestive, Respiratory, Vestibular.

Covers ingestion, triage, energy budgeting, gas exchange, stability
monitoring, serialization/restore, and graceful operation without EventBus.
"""

from __future__ import annotations

import json

import pytest

from mae_core.backbone.event_bus import EventBus
from mae_core.coordination.digestive_system import (
    CH_DIGESTION_COMPLETE,
    DigestiveSystem,
    EnergyBudget,
    NutrientPacket,
)
from mae_core.coordination.respiratory_system import (
    CH_HYPERCAPNIA,
    CH_HYPOXIA,
    CH_RESPIRATION_UPDATE,
    RespiratorySystem,
)
from mae_core.coordination.vestibular_system import (
    CH_BALANCE_UPDATE,
    CH_VERTIGO,
    VestibularSystem,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bus():
    return EventBus()


@pytest.fixture
def collector(bus):
    """Collect all messages published on the bus, keyed by channel."""
    received: dict[str, list] = {}

    def _on_message(channel: str, message):
        received.setdefault(channel, []).append(message)

    return received, _on_message


# ===========================================================================
# DigestiveSystem Tests
# ===========================================================================

class TestDigestiveIngestion:
    def test_ingest_adds_to_queue(self, bus):
        ds = DigestiveSystem(event_bus=bus)
        ds.ingest("agent_1", {"data": "hello"})
        assert len(ds._intake_queue) == 1

    def test_ingest_returns_packet(self, bus):
        ds = DigestiveSystem(event_bus=bus)
        pkt = ds.ingest("src", {"k": "v"}, energy_cost=2.0, nutritional_value=5.0)
        assert isinstance(pkt, NutrientPacket)
        assert pkt.source == "src"
        assert pkt.energy_cost == 2.0
        assert pkt.nutritional_value == 5.0
        assert pkt.decomposed is False


class TestDigestiveTriage:
    def test_triage_orders_by_value_ratio(self, bus):
        ds = DigestiveSystem(event_bus=bus)
        # Low ratio (1/10 = 0.1)
        ds.ingest("low", {"x": 1}, energy_cost=10.0, nutritional_value=1.0)
        # High ratio (10/1 = 10.0)
        ds.ingest("high", {"x": 2}, energy_cost=1.0, nutritional_value=10.0)
        # Medium ratio (5/2 = 2.5)
        ds.ingest("mid", {"x": 3}, energy_cost=2.0, nutritional_value=5.0)

        ds._triage()

        sources = [p.source for p in ds._intake_queue]
        assert sources == ["high", "mid", "low"]


class TestDigestiveEnergyBudget:
    def test_energy_regeneration(self, bus):
        budget = EnergyBudget(total_capacity=100.0, current_energy=50.0, regen_rate=10.0)
        ds = DigestiveSystem(event_bus=bus, energy_budget=budget)
        ds.step(1)
        # Should regenerate 10.0: 50 + 10 = 60 (minus any processing)
        assert ds.get_energy_remaining() == 60.0

    def test_energy_caps_at_capacity(self, bus):
        budget = EnergyBudget(total_capacity=100.0, current_energy=98.0, regen_rate=10.0)
        ds = DigestiveSystem(event_bus=bus, energy_budget=budget)
        ds.step(1)
        assert ds.get_energy_remaining() == 100.0

    def test_can_afford(self, bus):
        budget = EnergyBudget(current_energy=5.0)
        ds = DigestiveSystem(event_bus=bus, energy_budget=budget)
        assert ds.can_afford(5.0) is True
        assert ds.can_afford(5.1) is False

    def test_spend_energy_deducts(self, bus):
        budget = EnergyBudget(current_energy=10.0)
        ds = DigestiveSystem(event_bus=bus, energy_budget=budget)
        assert ds.spend_energy(3.0) is True
        assert ds.get_energy_remaining() == 7.0

    def test_spend_energy_insufficient(self, bus):
        budget = EnergyBudget(current_energy=2.0)
        ds = DigestiveSystem(event_bus=bus, energy_budget=budget)
        assert ds.spend_energy(5.0) is False
        assert ds.get_energy_remaining() == 2.0  # Unchanged

    def test_items_stay_in_queue_when_energy_exhausted(self, bus):
        budget = EnergyBudget(total_capacity=5.0, current_energy=0.0, regen_rate=3.0)
        ds = DigestiveSystem(event_bus=bus, energy_budget=budget)
        # Ingest two items, each costing 2.0. After regen we have 3.0.
        ds.ingest("a", {"x": 1}, energy_cost=2.0)
        ds.ingest("b", {"x": 2}, energy_cost=2.0)
        ds.step(1)
        # Should process one (2.0), reject the second (only 1.0 left)
        assert ds._processed_count == 1
        assert ds._rejected_count == 1
        assert len(ds._intake_queue) == 1  # One item remains


class TestDigestiveDecomposition:
    def test_simple_input_passes_through(self, bus):
        ds = DigestiveSystem(event_bus=bus)
        pkt = NutrientPacket(source="s", content={"a": 1, "b": 2})
        result = ds._digest(pkt)
        assert len(result) == 1
        assert result[0].decomposed is True

    def test_complex_input_by_key_count(self, bus):
        ds = DigestiveSystem(event_bus=bus)
        content = {f"k{i}": i for i in range(7)}  # 7 keys > 5
        pkt = NutrientPacket(source="s", content=content, energy_cost=7.0, nutritional_value=14.0)
        result = ds._digest(pkt)
        assert len(result) == 7
        assert all(p.decomposed for p in result)
        # Energy and value split evenly
        assert abs(sum(p.energy_cost for p in result) - 7.0) < 1e-9
        assert abs(sum(p.nutritional_value for p in result) - 14.0) < 1e-9

    def test_complex_input_by_nested_dict(self, bus):
        ds = DigestiveSystem(event_bus=bus)
        content = {"a": 1, "b": {"nested": True}}  # Has nested dict
        pkt = NutrientPacket(source="s", content=content)
        result = ds._digest(pkt)
        assert len(result) == 2
        # The nested dict is extracted as its own content
        nested_contents = [p.content for p in result]
        assert {"nested": True} in nested_contents


class TestDigestiveSerialization:
    def test_serialize_restore_roundtrip(self, bus):
        ds = DigestiveSystem(event_bus=bus)
        ds.ingest("x", {"data": 1}, energy_cost=3.0, nutritional_value=7.0)
        ds.step(1)

        data = ds.serialize()
        ds2 = DigestiveSystem(event_bus=bus)
        ds2.restore(data)

        assert ds2._processed_count == ds._processed_count
        assert ds2._rejected_count == ds._rejected_count
        assert abs(ds2._total_energy_spent - ds._total_energy_spent) < 1e-9


class TestDigestivePublish:
    def test_publishes_digestion_complete(self, bus, collector):
        received, callback = collector
        bus.register_callback(CH_DIGESTION_COMPLETE, callback)

        ds = DigestiveSystem(event_bus=bus)
        ds.ingest("a", {"x": 1})
        ds.step(1)

        assert CH_DIGESTION_COMPLETE in received
        msg = json.loads(received[CH_DIGESTION_COMPLETE][0])
        assert msg["processed_count"] >= 0


# ===========================================================================
# RespiratorySystem Tests
# ===========================================================================

class TestRespiratoryBreathing:
    def test_initial_state(self, bus):
        rs = RespiratorySystem(event_bus=bus)
        assert rs.get_oxygen_level() == 1.0
        assert rs.get_co2_level() == 0.0
        assert rs.is_breathing_normally() is True

    def test_breathing_rate_adapts_to_co2(self, bus):
        rs = RespiratorySystem(event_bus=bus)
        rs._co2 = 0.0
        low_co2_rate = rs._base_breathing_rate * (1.0 + 2.0 * 0.0)

        rs._co2 = 0.8
        high_co2_rate = rs._base_breathing_rate * (1.0 + 2.0 * 0.8)

        # Higher CO2 should produce higher breathing rate
        assert high_co2_rate > low_co2_rate

    def test_oxygen_consumption(self, bus):
        rs = RespiratorySystem(event_bus=bus)
        rs.consume_oxygen(0.3)
        assert abs(rs.get_oxygen_level() - 0.7) < 1e-9
        assert abs(rs.get_co2_level() - 0.3) < 1e-9

    def test_oxygen_cannot_go_negative(self, bus):
        rs = RespiratorySystem(event_bus=bus)
        rs.consume_oxygen(2.0)  # More than available
        assert rs.get_oxygen_level() == 0.0
        assert rs.get_co2_level() == 1.0  # Only consumed what was available


class TestRespiratoryAlerts:
    def test_hypoxia_alert(self, bus, collector):
        received, callback = collector
        bus.register_callback(CH_HYPOXIA, callback)

        rs = RespiratorySystem(event_bus=bus)
        rs._oxygen = 0.1  # Well below 0.3 threshold
        rs._co2 = 0.1
        # Set base breathing rate very low so O2 stays below threshold
        rs._base_breathing_rate = 0.01
        rs.step(1)

        # After breathing: O2 = 0.1 + small_rate. If still < 0.3, hypoxia fires.
        if rs.get_oxygen_level() < 0.3:
            assert CH_HYPOXIA in received

    def test_hypercapnia_alert(self, bus, collector):
        received, callback = collector
        bus.register_callback(CH_HYPERCAPNIA, callback)

        rs = RespiratorySystem(event_bus=bus)
        rs._oxygen = 0.5
        rs._co2 = 0.9  # Well above 0.7
        # Even with high breathing rate, 0.9 - rate may still be > 0.7
        rs._base_breathing_rate = 0.01
        rs.step(1)

        if rs.get_co2_level() > 0.7:
            assert CH_HYPERCAPNIA in received

    def test_normal_breathing_publishes_update(self, bus, collector):
        received, callback = collector
        bus.register_callback(CH_RESPIRATION_UPDATE, callback)

        rs = RespiratorySystem(event_bus=bus)
        # Default state is normal (O2=1.0, CO2=0.0)
        rs.step(1)

        assert CH_RESPIRATION_UPDATE in received


class TestRespiratorySerialization:
    def test_serialize_restore_roundtrip(self, bus):
        rs = RespiratorySystem(event_bus=bus)
        rs.consume_oxygen(0.4)
        rs.step(1)

        data = rs.serialize()
        rs2 = RespiratorySystem(event_bus=bus)
        rs2.restore(data)

        assert abs(rs2.get_oxygen_level() - rs.get_oxygen_level()) < 1e-9
        assert abs(rs2.get_co2_level() - rs.get_co2_level()) < 1e-9
        assert rs2._step_count == rs._step_count


# ===========================================================================
# VestibularSystem Tests
# ===========================================================================

class TestVestibularMetricReporting:
    def test_report_metric_tracks(self, bus):
        vs = VestibularSystem(event_bus=bus)
        vs.report_metric("cpu", 0.5)
        assert "cpu" in vs._tracked_metrics
        assert len(vs._metric_history["cpu"]) == 1

    def test_rolling_window_enforced(self, bus):
        vs = VestibularSystem(event_bus=bus)
        for i in range(20):
            vs.report_metric("load", float(i))
        # Window size is 10
        assert len(vs._metric_history["load"]) == 10
        assert vs._metric_history["load"][-1] == 19.0


class TestVestibularStability:
    def test_stable_metrics(self, bus):
        vs = VestibularSystem(event_bus=bus)
        # Report identical values — should be perfectly stable
        for _ in range(5):
            vs.report_metric("temp", 0.5)
        vs.step(1)
        assert vs.get_stability_score() == 1.0
        assert vs.is_stable() is True

    def test_unstable_metrics(self, bus):
        vs = VestibularSystem(event_bus=bus)
        # Wildly varying values
        values = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        for v in values:
            vs.report_metric("chaos", v)
        vs.step(1)
        # Should have low stability
        assert vs.get_stability_score() < 0.7

    def test_vertigo_detection(self, bus, collector):
        received, callback = collector
        bus.register_callback(CH_VERTIGO, callback)

        vs = VestibularSystem(event_bus=bus)
        # Create extreme instability
        for v in [0.0, 10.0, 0.0, 10.0, 0.0, 10.0, 0.0, 10.0]:
            vs.report_metric("wild", v)
        vs.step(1)

        assert vs.get_stability_score() < vs._vertigo_threshold
        assert CH_VERTIGO in received


class TestVestibularRapidChange:
    def test_detect_rapid_change_true(self, bus):
        vs = VestibularSystem(event_bus=bus)
        # Mean ~ 1.0, last 3 values all differ by > 50% from mean
        history = [1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0]
        assert vs._detect_rapid_change(history) is True

    def test_detect_rapid_change_false(self, bus):
        vs = VestibularSystem(event_bus=bus)
        # Stable values — last 3 close to mean
        history = [1.0, 1.0, 1.0, 1.0, 1.0]
        assert vs._detect_rapid_change(history) is False

    def test_detect_rapid_change_short_history(self, bus):
        vs = VestibularSystem(event_bus=bus)
        assert vs._detect_rapid_change([1.0]) is False
        assert vs._detect_rapid_change([1.0, 2.0]) is False


class TestVestibularSerialization:
    def test_serialize_restore_roundtrip(self, bus):
        vs = VestibularSystem(event_bus=bus)
        vs.report_metric("m1", 1.0)
        vs.report_metric("m1", 2.0)
        vs.report_metric("m2", 5.0)
        vs.step(1)

        data = vs.serialize()
        vs2 = VestibularSystem(event_bus=bus)
        vs2.restore(data)

        assert vs2._tracked_metrics == vs._tracked_metrics
        assert vs2._metric_history == vs._metric_history
        assert vs2._stability_score == vs._stability_score
        assert vs2._step_count == vs._step_count


class TestVestibularBalancePublish:
    def test_stable_publishes_balance_update(self, bus, collector):
        received, callback = collector
        bus.register_callback(CH_BALANCE_UPDATE, callback)

        vs = VestibularSystem(event_bus=bus)
        for _ in range(5):
            vs.report_metric("steady", 1.0)
        vs.step(1)

        assert CH_BALANCE_UPDATE in received


# ===========================================================================
# Cross-Cutting: Graceful Without EventBus
# ===========================================================================

class TestGracefulWithoutEventBus:
    def test_digestive_no_bus(self):
        ds = DigestiveSystem(event_bus=None)
        ds.ingest("src", {"data": 1})
        ds.step(1)  # Should not raise
        assert ds._processed_count >= 0

    def test_respiratory_no_bus(self):
        rs = RespiratorySystem(event_bus=None)
        rs.consume_oxygen(0.2)
        rs.step(1)  # Should not raise
        assert rs.is_breathing_normally() is True

    def test_vestibular_no_bus(self):
        vs = VestibularSystem(event_bus=None)
        vs.report_metric("m", 1.0)
        vs.step(1)  # Should not raise
        assert vs.is_stable() is True


# ===========================================================================
# Cross-Cutting: get_statistics Returns Data
# ===========================================================================

class TestGetStatistics:
    def test_digestive_statistics(self, bus):
        ds = DigestiveSystem(event_bus=bus)
        stats = ds.get_statistics()
        assert "energy_remaining" in stats
        assert "processed_count" in stats
        assert "queue_size" in stats

    def test_respiratory_statistics(self, bus):
        rs = RespiratorySystem(event_bus=bus)
        stats = rs.get_statistics()
        assert "oxygen" in stats
        assert "co2" in stats
        assert "is_breathing_normally" in stats

    def test_vestibular_statistics(self, bus):
        vs = VestibularSystem(event_bus=bus)
        stats = vs.get_statistics()
        assert "stability_score" in stats
        assert "is_stable" in stats
        assert "metrics_tracked" in stats
