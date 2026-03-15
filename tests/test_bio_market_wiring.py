"""Tests for bio-market wiring (Layer 33k activation)."""
import json
import threading
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
# Tier 2 tests
# =========================================================================


class TestEmotionalSystemWiring:
    def test_bullish_convergence_boosts_surprise(self):
        bus, _ = _make_bus()
        emo = SimpleNamespace(_surprise_boost=0.0, _fear_reinforcement=0.0)
        ctx = _make_ctx(bus, emotional_system=emo)

        from mae_core.bootstrap.bio_market_wiring import wire_bio_systems_to_market
        wire_bio_systems_to_market(ctx)

        bus.publish("market.intel.convergence", {
            "direction": "bullish", "strength": 0.8,
        })
        assert emo._surprise_boost > 0.0

    def test_bearish_convergence_boosts_fear(self):
        bus, _ = _make_bus()
        emo = SimpleNamespace(_surprise_boost=0.0, _fear_reinforcement=0.0)
        ctx = _make_ctx(bus, emotional_system=emo)

        from mae_core.bootstrap.bio_market_wiring import wire_bio_systems_to_market
        wire_bio_systems_to_market(ctx)

        bus.publish("market.intel.convergence", {
            "direction": "bearish", "strength": 0.9,
        })
        assert emo._fear_reinforcement > 0.0

    def test_deception_spikes_fear(self):
        bus, _ = _make_bus()
        emo = SimpleNamespace(_surprise_boost=0.0, _fear_reinforcement=0.0)
        ctx = _make_ctx(bus, emotional_system=emo)

        from mae_core.bootstrap.bio_market_wiring import wire_bio_systems_to_market
        wire_bio_systems_to_market(ctx)

        bus.publish("market.edge.deception_detected", {
            "severity": 0.7, "source": "test",
        })
        assert emo._fear_reinforcement > 0.0

    def test_weak_convergence_ignored(self):
        bus, _ = _make_bus()
        emo = SimpleNamespace(_surprise_boost=0.0, _fear_reinforcement=0.0)
        ctx = _make_ctx(bus, emotional_system=emo)

        from mae_core.bootstrap.bio_market_wiring import wire_bio_systems_to_market
        wire_bio_systems_to_market(ctx)

        bus.publish("market.intel.convergence", {
            "direction": "bullish", "strength": 0.3,
        })
        assert emo._surprise_boost == 0.0


class TestHomeostasisWiring:
    @pytest.mark.skip(reason="HomeostasisRegulator unsubscribed from CH_CONVERGENCE — "
                             "triadic audit (2026-03-14) confirmed it produces no market output")
    def test_bearish_raises_threat_level(self):
        bus, _ = _make_bus()
        homeo = MagicMock()
        ctx = _make_ctx(bus, homeostasis_regulator=homeo)

        from mae_core.bootstrap.bio_market_wiring import wire_bio_systems_to_market
        wire_bio_systems_to_market(ctx)

        bus.publish("market.intel.convergence", {
            "direction": "bearish", "strength": 0.8,
        })
        homeo.update_current_value.assert_called()
        calls = [c for c in homeo.update_current_value.call_args_list
                 if c[0][0] == "threat_level"]
        assert len(calls) > 0
        assert calls[0][0][1] > 0.3  # significant threat


class TestArousalWiring:
    def test_prediction_win_records_high_reward(self):
        bus, _ = _make_bus()
        arousal = MagicMock()
        ctx = _make_ctx(bus, arousal_regulator=arousal)

        from mae_core.bootstrap.bio_market_wiring import wire_bio_systems_to_market
        wire_bio_systems_to_market(ctx)

        bus.publish("market.sensing.prediction_result", {"won": True})
        arousal.record_reward.assert_called_with(1.0)

    def test_prediction_loss_records_zero_reward(self):
        bus, _ = _make_bus()
        arousal = MagicMock()
        ctx = _make_ctx(bus, arousal_regulator=arousal)

        from mae_core.bootstrap.bio_market_wiring import wire_bio_systems_to_market
        wire_bio_systems_to_market(ctx)

        bus.publish("market.sensing.prediction_result", {"won": False})
        arousal.record_reward.assert_called_with(0.0)


class TestCuriosityWiring:
    def test_partial_convergence_boosts_exploration(self):
        bus, _ = _make_bus()
        curiosity = SimpleNamespace(
            _exploration_bonus=0.1,
            _lock=threading.RLock(),
            set_exploration_bonus=lambda self, lvl: None,
        )
        # Replace with real method behavior
        def set_bonus(lvl):
            curiosity._exploration_bonus = max(0.01, min(0.5, lvl))
        curiosity.set_exploration_bonus = set_bonus

        ctx = _make_ctx(bus, curiosity_drive=curiosity)

        from mae_core.bootstrap.bio_market_wiring import wire_bio_systems_to_market
        wire_bio_systems_to_market(ctx)

        bus.publish("market.intel.partial_convergence", {
            "domains_seen": ["insider", "macro"],
        })
        assert curiosity._exploration_bonus > 0.1


class TestNociceptionWiring:
    def test_deception_causes_acute_pain(self):
        bus, _ = _make_bus()
        noci = MagicMock()
        ctx = _make_ctx(bus, nociception_system=noci)

        from mae_core.bootstrap.bio_market_wiring import wire_bio_systems_to_market
        wire_bio_systems_to_market(ctx)

        bus.publish("market.edge.deception_detected", {
            "severity": 0.8, "source": "finviz",
        })
        noci.report_damage.assert_called_once()
        args = noci.report_damage.call_args[0]
        assert "deception" in args[0]
        assert args[2] == "acute"

    def test_prediction_failure_causes_referred_pain(self):
        bus, _ = _make_bus()
        noci = MagicMock()
        ctx = _make_ctx(bus, nociception_system=noci)

        from mae_core.bootstrap.bio_market_wiring import wire_bio_systems_to_market
        wire_bio_systems_to_market(ctx)

        bus.publish("market.sensing.prediction_result", {
            "won": False, "confidence": 0.8,
        })
        noci.report_damage.assert_called_once()
        args = noci.report_damage.call_args[0]
        assert args[0] == "prediction_failure"
        assert args[2] == "referred"

    def test_prediction_win_no_pain(self):
        bus, _ = _make_bus()
        noci = MagicMock()
        ctx = _make_ctx(bus, nociception_system=noci)

        from mae_core.bootstrap.bio_market_wiring import wire_bio_systems_to_market
        wire_bio_systems_to_market(ctx)

        bus.publish("market.sensing.prediction_result", {"won": True})
        noci.report_damage.assert_not_called()


# =========================================================================
# Tier 3 tests
# =========================================================================


class TestMetacognitionWiring:
    def test_prediction_result_records_decision(self):
        bus, _ = _make_bus()
        metacog = MagicMock()
        ctx = _make_ctx(bus, metacognition_monitor=metacog)

        from mae_core.bootstrap.bio_market_wiring import wire_bio_systems_to_market
        wire_bio_systems_to_market(ctx)

        bus.publish("market.sensing.prediction_result", {
            "won": True, "confidence": 0.75, "step": 100,
        })
        metacog.record_decision.assert_called_once_with(
            step=100, predicted=0.75, actual=1.0,
            decision_type="convergence_alert",
        )


class TestThreatDetectorWiring:
    def test_deception_quill_registered(self):
        bus, _ = _make_bus()
        td = MagicMock(spec=["register_quill", "register_sacrificeable"])
        ctx = _make_ctx(bus, threat_detector=td)

        from mae_core.bootstrap.bio_market_wiring import wire_bio_systems_to_market
        wire_bio_systems_to_market(ctx)

        td.register_quill.assert_called_once()


class TestQuorumWiring:
    def test_convergence_deposits_signal(self):
        bus, _ = _make_bus()
        quorum = MagicMock()
        ctx = _make_ctx(bus, quorum_space=quorum)

        from mae_core.bootstrap.bio_market_wiring import wire_bio_systems_to_market
        wire_bio_systems_to_market(ctx)

        bus.publish("market.intel.convergence", {
            "direction": "bullish", "strength": 0.8,
            "ticker": "AAPL", "domain_count": 4,
        })
        quorum.deposit_signal.assert_called_once()
        args = quorum.deposit_signal.call_args[0]
        assert args[0] == "AAPL.bullish"
        assert args[1] == "convergence_alerter"


class TestCircadianWiring:
    @pytest.mark.skip(reason="CircadianRhythm activity pinned to 1.0 — "
                             "triadic audit (2026-03-14) confirmed phase-based throttling harms market sensing")
    def test_phase_change_sets_ctx_activity(self):
        bus, _ = _make_bus()
        circadian = SimpleNamespace(get_activity_multiplier=lambda: 0.5)
        ctx = _make_ctx(bus, circadian_rhythm=circadian)

        from mae_core.bootstrap.bio_market_wiring import wire_bio_systems_to_market
        wire_bio_systems_to_market(ctx)

        assert ctx._circadian_activity == 1.0  # default
        bus.publish("circadian.phase_change", {"new_phase": "CONSOLIDATION"})
        assert ctx._circadian_activity == 0.5


class TestHavenWiring:
    def test_deception_flags_source(self):
        bus, _ = _make_bus()
        haven = SimpleNamespace()
        ctx = _make_ctx(bus, haven=haven)

        from mae_core.bootstrap.bio_market_wiring import wire_bio_systems_to_market
        wire_bio_systems_to_market(ctx)

        bus.publish("market.edge.deception_detected", {
            "source": "finviz", "severity": 0.5,
        })
        assert ctx._haven_market_flags["finviz"] == 0.5

    def test_success_reduces_flag(self):
        bus, _ = _make_bus()
        haven = SimpleNamespace()
        ctx = _make_ctx(bus, haven=haven)

        from mae_core.bootstrap.bio_market_wiring import wire_bio_systems_to_market
        wire_bio_systems_to_market(ctx)

        bus.publish("market.edge.deception_detected", {
            "source": "finviz", "severity": 0.5,
        })
        bus.publish("market.sensing.prediction_result", {
            "won": True, "sources": ["finviz"],
        })
        assert ctx._haven_market_flags["finviz"] < 0.5


class TestInhibitionWiring:
    def test_deception_raises_caution(self):
        bus, _ = _make_bus()
        inhibition = SimpleNamespace()
        ctx = _make_ctx(bus, inhibition_system=inhibition)

        from mae_core.bootstrap.bio_market_wiring import wire_bio_systems_to_market
        wire_bio_systems_to_market(ctx)

        bus.publish("market.edge.deception_detected", {"severity": 0.8})
        assert ctx._market_caution > 0.0

    def test_high_confidence_lowers_caution(self):
        bus, _ = _make_bus()
        inhibition = SimpleNamespace()
        ctx = _make_ctx(bus, inhibition_system=inhibition)

        from mae_core.bootstrap.bio_market_wiring import wire_bio_systems_to_market
        wire_bio_systems_to_market(ctx)

        ctx._market_caution = 0.5
        bus.publish("market.intel.convergence", {"confidence": 0.85})
        assert ctx._market_caution < 0.5


class TestStigmergyWiring:
    def test_convergence_deposits_marker(self):
        bus, _ = _make_bus()
        stigmergy = MagicMock()
        ctx = _make_ctx(bus, stigmergy=stigmergy)

        from mae_core.bootstrap.bio_market_wiring import wire_bio_systems_to_market
        wire_bio_systems_to_market(ctx)

        bus.publish("market.intel.convergence", {
            "direction": "bullish", "strength": 0.9,
            "ticker": "TSLA", "domain_count": 3,
        })
        stigmergy.deposit_marker.assert_called_once()
        kwargs = stigmergy.deposit_marker.call_args[1]
        assert kwargs["marker_type"] == "convergence.bullish"
        assert kwargs["metadata"]["ticker"] == "TSLA"


# =========================================================================
# Graceful degradation
# =========================================================================


class TestGracefulDegradation:
    def test_missing_systems_wires_zero(self):
        bus, _ = _make_bus()
        ctx = _make_ctx(bus)  # no systems at all

        from mae_core.bootstrap.bio_market_wiring import wire_bio_systems_to_market
        wire_bio_systems_to_market(ctx)
        # Should complete without error

    def test_no_bus_returns_immediately(self):
        ctx = SimpleNamespace()  # no bus

        from mae_core.bootstrap.bio_market_wiring import wire_bio_systems_to_market
        wire_bio_systems_to_market(ctx)
        # Should complete without error

    def test_partial_systems_wires_available(self):
        bus, cbs = _make_bus()
        emo = SimpleNamespace(_surprise_boost=0.0, _fear_reinforcement=0.0)
        ctx = _make_ctx(bus, emotional_system=emo)
        # Only emotional_system set — all others missing

        from mae_core.bootstrap.bio_market_wiring import wire_bio_systems_to_market
        wire_bio_systems_to_market(ctx)

        # Should have registered callbacks for emotional_system channels
        assert "market.intel.convergence" in cbs
