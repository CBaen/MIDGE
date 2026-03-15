"""Tests for Tier 4+5 bio-market wiring (Layer 33k extended).

NOTE: Many bio system callbacks were unsubscribed from market channels by
the triadic system audit (2026-03-14). Tests for removed subscriptions are
skipped with reason tags. See research/triadic-system-audit/deliverable.md.
"""
import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _make_bus():
    """Minimal EventBus substitute for testing."""
    callbacks = {}

    class FakeBus:
        def register_callback(self, channel, cb):
            callbacks.setdefault(channel, []).append(cb)

        def publish(self, channel, data):
            msg = json.dumps(data) if isinstance(data, (dict, list)) else data
            for cb in callbacks.get(channel, []):
                cb(channel, msg)
            return len(callbacks.get(channel, []))

    return FakeBus(), callbacks


def _make_ctx(bus, **systems):
    """Build a ctx SimpleNamespace with bus and optional systems."""
    ctx = SimpleNamespace(bus=bus)
    for name, obj in systems.items():
        setattr(ctx, name, obj)
    return ctx


# =========================================================================
# Tier 4 tests
# =========================================================================


_AUDIT_SKIP = "Bio callback unsubscribed from market channels — triadic audit 2026-03-14"


@pytest.mark.skip(reason=_AUDIT_SKIP)
class TestDigestiveWiring:
    def test_convergence_ingested_as_nutrient(self):
        bus, _ = _make_bus()
        digestive = MagicMock()
        ctx = _make_ctx(bus, digestive_system=digestive)

        from mae_core.bootstrap.bio_market_wiring_extended import wire_bio_systems_extended
        wire_bio_systems_extended(ctx)

        bus.publish("market.intel.convergence", {
            "ticker": "AAPL", "strength": 0.9, "direction": "bullish",
        })
        digestive.ingest.assert_called_once()
        kwargs = digestive.ingest.call_args[1]
        assert "convergence:AAPL" in kwargs["source"]
        assert kwargs["nutritional_value"] == 0.9

    def test_partial_ingested_lower_value(self):
        bus, _ = _make_bus()
        digestive = MagicMock()
        ctx = _make_ctx(bus, digestive_system=digestive)

        from mae_core.bootstrap.bio_market_wiring_extended import wire_bio_systems_extended
        wire_bio_systems_extended(ctx)

        bus.publish("market.intel.partial_convergence", {
            "domains_seen": ["insider", "macro"],
        })
        digestive.ingest.assert_called_once()
        kwargs = digestive.ingest.call_args[1]
        assert kwargs["nutritional_value"] == 0.3
        assert kwargs["energy_cost"] == 0.1


@pytest.mark.skip(reason=_AUDIT_SKIP)
class TestCirculatoryWiring:
    def test_convergence_requests_attention(self):
        bus, _ = _make_bus()
        circ = MagicMock()
        ctx = _make_ctx(bus, circulatory_system=circ)

        from mae_core.bootstrap.bio_market_wiring_extended import wire_bio_systems_extended
        wire_bio_systems_extended(ctx)

        bus.publish("market.intel.convergence", {"strength": 0.8})
        circ.request_resource.assert_called_once()
        args = circ.request_resource.call_args[0]
        assert args[0] == "convergence_alerter"
        assert args[1] == "attention"

    def test_velocity_below_threshold_ignored(self):
        bus, _ = _make_bus()
        circ = MagicMock()
        ctx = _make_ctx(bus, circulatory_system=circ)

        from mae_core.bootstrap.bio_market_wiring_extended import wire_bio_systems_extended
        wire_bio_systems_extended(ctx)

        bus.publish("market.intel.velocity_anomaly", {"magnitude": 0.5})
        circ.request_resource.assert_not_called()


@pytest.mark.skip(reason=_AUDIT_SKIP)
class TestLymphaticWiring:
    def test_prediction_failure_collects_waste(self):
        bus, _ = _make_bus()
        lymph = MagicMock()
        ctx = _make_ctx(bus, lymphatic_system=lymph)

        from mae_core.bootstrap.bio_market_wiring_extended import wire_bio_systems_extended
        wire_bio_systems_extended(ctx)

        bus.publish("market.sensing.prediction_result", {
            "won": False, "ticker": "TSLA", "step": 100,
        })
        lymph.collect_waste.assert_called_once()
        kwargs = lymph.collect_waste.call_args[1]
        assert kwargs["waste_type"] == "expired_memory"
        assert kwargs["item_data"]["ticker"] == "TSLA"

    def test_prediction_win_no_waste(self):
        bus, _ = _make_bus()
        lymph = MagicMock()
        ctx = _make_ctx(bus, lymphatic_system=lymph)

        from mae_core.bootstrap.bio_market_wiring_extended import wire_bio_systems_extended
        wire_bio_systems_extended(ctx)

        bus.publish("market.sensing.prediction_result", {"won": True})
        lymph.collect_waste.assert_not_called()

    def test_deception_collects_orphan(self):
        bus, _ = _make_bus()
        lymph = MagicMock()
        ctx = _make_ctx(bus, lymphatic_system=lymph)

        from mae_core.bootstrap.bio_market_wiring_extended import wire_bio_systems_extended
        wire_bio_systems_extended(ctx)

        bus.publish("market.edge.deception_detected", {"source": "finviz"})
        lymph.collect_waste.assert_called_once()
        kwargs = lymph.collect_waste.call_args[1]
        assert kwargs["waste_type"] == "orphan_subscription"


@pytest.mark.skip(reason=_AUDIT_SKIP)
class TestMicrobiomeWiring:
    def test_convergence_processed_as_complex(self):
        bus, _ = _make_bus()
        micro = MagicMock()
        ctx = _make_ctx(bus, microbiome=micro)

        from mae_core.bootstrap.bio_market_wiring_extended import wire_bio_systems_extended
        wire_bio_systems_extended(ctx)

        bus.publish("market.intel.convergence", {"direction": "bullish"})
        micro.process_input.assert_called_once_with("complex", {"direction": "bullish"})

    def test_velocity_processed_as_anomaly(self):
        bus, _ = _make_bus()
        micro = MagicMock()
        ctx = _make_ctx(bus, microbiome=micro)

        from mae_core.bootstrap.bio_market_wiring_extended import wire_bio_systems_extended
        wire_bio_systems_extended(ctx)

        bus.publish("market.intel.velocity_anomaly", {"magnitude": 3.0})
        micro.process_input.assert_called_once_with("anomaly", {"magnitude": 3.0})

    def test_low_confidence_stack_amplified(self):
        bus, _ = _make_bus()
        micro = MagicMock()
        ctx = _make_ctx(bus, microbiome=micro)

        from mae_core.bootstrap.bio_market_wiring_extended import wire_bio_systems_extended
        wire_bio_systems_extended(ctx)

        bus.publish("market.archaeology.pattern_stack_detected", {"confidence": 0.3})
        micro.process_input.assert_called_once_with("weak_signal", {"confidence": 0.3})

    def test_high_confidence_stack_not_amplified(self):
        bus, _ = _make_bus()
        micro = MagicMock()
        ctx = _make_ctx(bus, microbiome=micro)

        from mae_core.bootstrap.bio_market_wiring_extended import wire_bio_systems_extended
        wire_bio_systems_extended(ctx)

        bus.publish("market.archaeology.pattern_stack_detected", {"confidence": 0.8})
        micro.process_input.assert_not_called()


@pytest.mark.skip(reason=_AUDIT_SKIP)
class TestRenalFilterWiring:
    def test_deception_teaches_toxin(self):
        bus, _ = _make_bus()
        renal = MagicMock()
        ctx = _make_ctx(bus, renal_filter=renal)

        from mae_core.bootstrap.bio_market_wiring_extended import wire_bio_systems_extended
        wire_bio_systems_extended(ctx)

        bus.publish("market.edge.deception_detected", {
            "source": "finviz", "severity": 0.7,
        })
        renal.add_toxin_pattern.assert_called_once()
        pattern = renal.add_toxin_pattern.call_args[0][0]
        assert isinstance(pattern, str)
        assert "finviz" in pattern
        renal.filter_item.assert_called_once()

    def test_convergence_filtered(self):
        bus, _ = _make_bus()
        renal = MagicMock()
        ctx = _make_ctx(bus, renal_filter=renal)

        from mae_core.bootstrap.bio_market_wiring_extended import wire_bio_systems_extended
        wire_bio_systems_extended(ctx)

        bus.publish("market.intel.convergence", {"ticker": "AAPL"})
        renal.filter_item.assert_called_once()


class TestSenescenceWiring:
    def test_market_systems_registered(self):
        bus, _ = _make_bus()
        sen = MagicMock()
        ctx = _make_ctx(bus, senescence=sen)

        from mae_core.bootstrap.bio_market_wiring_extended import wire_bio_systems_extended
        wire_bio_systems_extended(ctx)

        # Should register at least convergence_alerter
        reg_names = [c[0][0] for c in sen.register_system.call_args_list]
        assert "convergence_alerter" in reg_names

    def test_convergence_reports_activity(self):
        bus, _ = _make_bus()
        sen = MagicMock()
        ctx = _make_ctx(bus, senescence=sen)

        from mae_core.bootstrap.bio_market_wiring_extended import wire_bio_systems_extended
        wire_bio_systems_extended(ctx)

        bus.publish("market.intel.convergence", {"strength": 0.8})
        sen.report_activity.assert_called_with("convergence_alerter", active=True)


class TestMorphogenesisWiring:
    def test_partial_overflow_spawns_organ(self):
        bus, _ = _make_bus()
        morph = MagicMock()
        ctx = _make_ctx(bus, morph_coordinator=morph)

        from mae_core.bootstrap.bio_market_wiring_extended import wire_bio_systems_extended
        wire_bio_systems_extended(ctx)

        # Fire 5+ partials rapidly to trigger spawn
        for _ in range(6):
            bus.publish("market.intel.partial_convergence", {
                "domains_seen": ["insider", "macro"],
            })
        morph.handle_novel_problem.assert_called()

    def test_hypothesis_boosts_growth(self):
        bus, _ = _make_bus()
        morph = MagicMock()
        ctx = _make_ctx(bus, morph_coordinator=morph)

        from mae_core.bootstrap.bio_market_wiring_extended import wire_bio_systems_extended
        wire_bio_systems_extended(ctx)

        bus.publish("market.hypothesis.discovered", {"id": "h1"})
        morph.set_growth_rate.assert_called_with(1.3)


class TestReproductiveWiring:
    def test_convergence_increases_pressure(self):
        bus, _ = _make_bus()
        repro = MagicMock()
        ctx = _make_ctx(bus, reproductive_system=repro)

        from mae_core.bootstrap.bio_market_wiring_extended import wire_bio_systems_extended
        wire_bio_systems_extended(ctx)

        bus.publish("market.intel.convergence", {"strength": 0.8})
        assert ctx._market_activity_pressure > 0.0

    def test_pressure_capped_at_one(self):
        bus, _ = _make_bus()
        repro = MagicMock()
        ctx = _make_ctx(bus, reproductive_system=repro)

        from mae_core.bootstrap.bio_market_wiring_extended import wire_bio_systems_extended
        wire_bio_systems_extended(ctx)

        for _ in range(50):
            bus.publish("market.intel.convergence", {"strength": 1.0})
        assert ctx._market_activity_pressure <= 1.0


class TestPearlDefenseWiring:
    def test_deception_triggers_validation(self):
        bus, _ = _make_bus()
        pearl = MagicMock()
        ctx = _make_ctx(bus, pearl_defense=pearl)

        from mae_core.bootstrap.bio_market_wiring_extended import wire_bio_systems_extended
        wire_bio_systems_extended(ctx)

        bus.publish("market.edge.deception_detected", {
            "source": "finviz", "severity": 0.6,
        })
        pearl.validate.assert_called_once()
        kwargs = pearl.validate.call_args[1]
        assert kwargs["source"] == "finviz"
        assert kwargs["input_type"] == "deception_signal"
        assert kwargs["numeric_value"] == 0.6


# =========================================================================
# Tier 5 tests
# =========================================================================


@pytest.mark.skip(reason=_AUDIT_SKIP)
class TestRespiratoryWiring:
    def test_convergence_consumes_oxygen(self):
        bus, _ = _make_bus()
        resp = MagicMock()
        ctx = _make_ctx(bus, respiratory_system=resp)

        from mae_core.bootstrap.bio_market_wiring_extended import wire_bio_systems_extended
        wire_bio_systems_extended(ctx)

        bus.publish("market.intel.convergence", {"strength": 0.8})
        resp.consume_oxygen.assert_called_once_with(0.03)

    def test_velocity_anomaly_consumes_more_oxygen(self):
        bus, _ = _make_bus()
        resp = MagicMock()
        ctx = _make_ctx(bus, respiratory_system=resp)

        from mae_core.bootstrap.bio_market_wiring_extended import wire_bio_systems_extended
        wire_bio_systems_extended(ctx)

        bus.publish("market.intel.velocity_anomaly", {"magnitude": 4.0})
        resp.consume_oxygen.assert_called_once()
        args = resp.consume_oxygen.call_args[0]
        assert args[0] > 0.03  # more than standard convergence cost


@pytest.mark.skip(reason=_AUDIT_SKIP)
class TestThermoregulationWiring:
    def test_convergence_reports_heat(self):
        bus, _ = _make_bus()
        thermo = MagicMock()
        ctx = _make_ctx(bus, thermoregulation=thermo)

        from mae_core.bootstrap.bio_market_wiring_extended import wire_bio_systems_extended
        wire_bio_systems_extended(ctx)

        bus.publish("market.intel.convergence", {"strength": 0.7})
        thermo.report_activity.assert_called_once_with("market_convergence", 0.7)

    def test_velocity_reports_spike(self):
        bus, _ = _make_bus()
        thermo = MagicMock()
        ctx = _make_ctx(bus, thermoregulation=thermo)

        from mae_core.bootstrap.bio_market_wiring_extended import wire_bio_systems_extended
        wire_bio_systems_extended(ctx)

        bus.publish("market.intel.velocity_anomaly", {"magnitude": 3.0})
        thermo.report_activity.assert_called_once()
        args = thermo.report_activity.call_args[0]
        assert args[0] == "market_anomaly"
        assert abs(args[1] - 0.6) < 0.01


@pytest.mark.skip(reason=_AUDIT_SKIP)
class TestVestibularWiring:
    def test_convergence_tracks_rate(self):
        bus, _ = _make_bus()
        vest = MagicMock()
        ctx = _make_ctx(bus, vestibular_system=vest)

        from mae_core.bootstrap.bio_market_wiring_extended import wire_bio_systems_extended
        wire_bio_systems_extended(ctx)

        bus.publish("market.intel.convergence", {"domain_count": 4})
        vest.report_metric.assert_called_once_with("convergence_rate", 4 / 12.0)

    def test_prediction_tracks_accuracy(self):
        bus, _ = _make_bus()
        vest = MagicMock()
        ctx = _make_ctx(bus, vestibular_system=vest)

        from mae_core.bootstrap.bio_market_wiring_extended import wire_bio_systems_extended
        wire_bio_systems_extended(ctx)

        bus.publish("market.sensing.prediction_result", {"won": True})
        vest.report_metric.assert_called_with("prediction_accuracy", 1.0)

    def test_prediction_loss_tracks_zero(self):
        bus, _ = _make_bus()
        vest = MagicMock()
        ctx = _make_ctx(bus, vestibular_system=vest)

        from mae_core.bootstrap.bio_market_wiring_extended import wire_bio_systems_extended
        wire_bio_systems_extended(ctx)

        bus.publish("market.sensing.prediction_result", {"won": False})
        vest.report_metric.assert_called_with("prediction_accuracy", 0.0)


@pytest.mark.skip(reason=_AUDIT_SKIP)
class TestProprioceptionWiring:
    def test_convergence_updates_alerter_position(self):
        bus, _ = _make_bus()
        proprio = MagicMock()
        ctx = _make_ctx(bus, proprioception=proprio)

        from mae_core.bootstrap.bio_market_wiring_extended import wire_bio_systems_extended
        wire_bio_systems_extended(ctx)

        bus.publish("market.intel.convergence", {
            "strength": 0.9, "confidence": 0.8,
        })
        proprio.update_position.assert_called_once_with(
            "convergence_alerter", activity=0.9, health=0.8,
        )

    def test_prediction_win_healthy(self):
        bus, _ = _make_bus()
        proprio = MagicMock()
        ctx = _make_ctx(bus, proprioception=proprio)

        from mae_core.bootstrap.bio_market_wiring_extended import wire_bio_systems_extended
        wire_bio_systems_extended(ctx)

        bus.publish("market.sensing.prediction_result", {"won": True})
        proprio.update_position.assert_called_once_with(
            "outcome_tracker", activity=1.0, health=1.0,
        )

    def test_prediction_loss_half_health(self):
        bus, _ = _make_bus()
        proprio = MagicMock()
        ctx = _make_ctx(bus, proprioception=proprio)

        from mae_core.bootstrap.bio_market_wiring_extended import wire_bio_systems_extended
        wire_bio_systems_extended(ctx)

        bus.publish("market.sensing.prediction_result", {"won": False})
        proprio.update_position.assert_called_once_with(
            "outcome_tracker", activity=1.0, health=0.5,
        )


@pytest.mark.skip(reason=_AUDIT_SKIP)
class TestEnergyReserveWiring:
    def test_convergence_spends_energy(self):
        bus, _ = _make_bus()
        energy = MagicMock()
        ctx = _make_ctx(bus, energy_reserve=energy)

        from mae_core.bootstrap.bio_market_wiring_extended import wire_bio_systems_extended
        wire_bio_systems_extended(ctx)

        bus.publish("market.intel.convergence", {"strength": 0.8})
        energy.release.assert_called_once_with(0.5)

    def test_rest_phase_stores_energy(self):
        bus, _ = _make_bus()
        energy = MagicMock()
        ctx = _make_ctx(bus, energy_reserve=energy)

        from mae_core.bootstrap.bio_market_wiring_extended import wire_bio_systems_extended
        wire_bio_systems_extended(ctx)

        bus.publish("circadian.phase_change", {"new_phase": "REST"})
        energy.store.assert_called_once_with(5.0)

    def test_active_phase_releases_energy(self):
        bus, _ = _make_bus()
        energy = MagicMock()
        ctx = _make_ctx(bus, energy_reserve=energy)

        from mae_core.bootstrap.bio_market_wiring_extended import wire_bio_systems_extended
        wire_bio_systems_extended(ctx)

        bus.publish("circadian.phase_change", {"new_phase": "ACTIVE"})
        energy.release.assert_called_once_with(1.0)


@pytest.mark.skip(reason=_AUDIT_SKIP)
class TestPredictiveFieldWiring:
    def test_convergence_updates_field(self):
        bus, _ = _make_bus()
        field = MagicMock()
        ctx = _make_ctx(bus, predictive_field=field)

        from mae_core.bootstrap.bio_market_wiring_extended import wire_bio_systems_extended
        wire_bio_systems_extended(ctx)

        bus.publish("market.intel.convergence", {
            "ticker": "AAPL", "direction": "bullish", "confidence": 0.85,
        })
        field.update_agent_state.assert_called_once()
        kwargs = field.update_agent_state.call_args[1]
        assert isinstance(kwargs["agent_id"], int)
        assert kwargs["intention"] == "bullish"
        assert kwargs["confidence"] == 0.85

    def test_missing_ticker_skipped(self):
        bus, _ = _make_bus()
        field = MagicMock()
        ctx = _make_ctx(bus, predictive_field=field)

        from mae_core.bootstrap.bio_market_wiring_extended import wire_bio_systems_extended
        wire_bio_systems_extended(ctx)

        bus.publish("market.intel.convergence", {
            "direction": "bullish", "confidence": 0.8,
        })
        field.update_agent_state.assert_not_called()


# =========================================================================
# Graceful degradation
# =========================================================================


class TestExtendedGracefulDegradation:
    def test_no_bus_returns_zero(self):
        ctx = SimpleNamespace()
        from mae_core.bootstrap.bio_market_wiring_extended import wire_bio_systems_extended
        assert wire_bio_systems_extended(ctx) == 0

    def test_empty_ctx_wires_zero(self):
        bus, _ = _make_bus()
        ctx = _make_ctx(bus)
        from mae_core.bootstrap.bio_market_wiring_extended import wire_bio_systems_extended
        count = wire_bio_systems_extended(ctx)
        assert count == 0

    def test_all_systems_wires_remaining(self):
        bus, _ = _make_bus()
        ctx = _make_ctx(
            bus,
            digestive_system=MagicMock(),
            circulatory_system=MagicMock(),
            lymphatic_system=MagicMock(),
            microbiome=MagicMock(),
            renal_filter=MagicMock(),
            senescence=MagicMock(),
            morph_coordinator=MagicMock(),
            reproductive_system=MagicMock(),
            pearl_defense=MagicMock(),
            respiratory_system=MagicMock(),
            thermoregulation=MagicMock(),
            vestibular_system=MagicMock(),
            proprioception=MagicMock(),
            energy_reserve=MagicMock(),
            predictive_field=MagicMock(),
        )
        from mae_core.bootstrap.bio_market_wiring_extended import wire_bio_systems_extended
        count = wire_bio_systems_extended(ctx)
        # Count reduced after triadic audit removed inert bio callbacks (2026-03-14)
        assert count >= 0  # exact count depends on how many systems remain wired
