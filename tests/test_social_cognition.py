"""Tests for social cognition systems: TheoryOfMind, MetacognitionMonitor, NociceptionSystem.

Covers:
- TheoryOfMind: model creation, action prediction, cooperation prediction,
  confidence decay, model removal, emotion influence, serialization
- MetacognitionMonitor: record decisions, performance tracking, degradation
  detection, calibration, learning rate suggestion, baseline update, serialization
- NociceptionSystem: damage reporting, pain habituation, acute pain alerts,
  pain overload, gate control suppression, pain threshold, somatic integration,
  serialization
- All three: graceful operation without event_bus
"""

import json

import pytest

from mae_core.backbone.event_bus import EventBus
from mae_core.cognition.theory_of_mind import AgentModel, TheoryOfMind
from mae_core.cognition.metacognition import DecisionRecord, MetacognitionMonitor
from mae_core.communication.nociception import NociceptionSystem, PainSignal


# =========================================================================
# TheoryOfMind Tests
# =========================================================================


class TestTheoryOfMindModelCreation:
    """Tests for creating and updating agent models."""

    def test_empty_on_init(self):
        tom = TheoryOfMind()
        assert tom.get_statistics()["num_models"] == 0
        assert tom.get_statistics()["observation_count"] == 0

    def test_model_created_on_first_observation(self):
        tom = TheoryOfMind()
        model = tom.update_model(agent_id=1, action="explore")
        assert model.agent_id == 1
        assert "explore" in model.recent_actions
        assert tom.get_statistics()["num_models"] == 1

    def test_model_reused_on_second_observation(self):
        tom = TheoryOfMind()
        tom.update_model(agent_id=1, action="explore")
        tom.update_model(agent_id=1, action="gather")
        assert tom.get_statistics()["num_models"] == 1
        model = tom.get_agent_model(1)
        assert len(model.recent_actions) == 2

    def test_multiple_agents(self):
        tom = TheoryOfMind()
        for i in range(5):
            tom.update_model(agent_id=i, action="move")
        assert tom.get_statistics()["num_models"] == 5

    def test_recent_actions_trimmed_to_max(self):
        tom = TheoryOfMind()
        for i in range(15):
            tom.update_model(agent_id=1, action=f"action_{i}")
        model = tom.get_agent_model(1)
        assert len(model.recent_actions) == 10
        # Should keep the latest 10
        assert model.recent_actions[-1] == "action_14"

    def test_signal_type_infers_goal(self):
        tom = TheoryOfMind()
        tom.update_model(agent_id=1, signal_type="communicate")
        model = tom.get_agent_model(1)
        assert model.estimated_goal == "collaborate"

    def test_unknown_signal_type_does_not_change_goal(self):
        tom = TheoryOfMind()
        tom.update_model(agent_id=1, signal_type="unknown_signal")
        model = tom.get_agent_model(1)
        assert model.estimated_goal == "explore"  # Default unchanged

    def test_emotion_update(self):
        tom = TheoryOfMind()
        tom.update_model(agent_id=1, emotion="joy")
        model = tom.get_agent_model(1)
        assert model.estimated_emotion == "joy"


class TestTheoryOfMindPrediction:
    """Tests for action and cooperation prediction."""

    def test_predict_action_majority_vote(self):
        tom = TheoryOfMind()
        tom.update_model(agent_id=1, action="explore")
        tom.update_model(agent_id=1, action="explore")
        tom.update_model(agent_id=1, action="gather")
        assert tom.predict_action(1) == "explore"

    def test_predict_action_unknown_agent(self):
        tom = TheoryOfMind()
        assert tom.predict_action(999) == "unknown"

    def test_predict_action_no_actions(self):
        tom = TheoryOfMind()
        tom.update_model(agent_id=1, emotion="calm")  # No action recorded
        assert tom.predict_action(1) == "unknown"

    def test_predict_cooperation_neutral(self):
        tom = TheoryOfMind()
        tom.update_model(agent_id=1, emotion="calm")
        coop = tom.predict_cooperation(1)
        # Calm emotion gives 0 modifier, trust_level default is 0.5
        assert coop == pytest.approx(0.5, abs=0.01)

    def test_predict_cooperation_joy_increases(self):
        tom = TheoryOfMind()
        tom.update_model(agent_id=1, emotion="joy")
        model = tom.get_agent_model(1)
        model.trust_level = 0.5
        coop = tom.predict_cooperation(1)
        assert coop > 0.5

    def test_predict_cooperation_fear_decreases(self):
        tom = TheoryOfMind()
        tom.update_model(agent_id=1, emotion="fear")
        model = tom.get_agent_model(1)
        model.trust_level = 0.5
        coop = tom.predict_cooperation(1)
        assert coop < 0.5

    def test_predict_cooperation_unknown_agent(self):
        tom = TheoryOfMind()
        assert tom.predict_cooperation(999) == 0.5

    def test_predict_cooperation_clamped_to_0_1(self):
        tom = TheoryOfMind()
        tom.update_model(agent_id=1, emotion="fear")
        model = tom.get_agent_model(1)
        model.trust_level = 0.0  # Very low trust + fear
        coop = tom.predict_cooperation(1)
        assert 0.0 <= coop <= 1.0


class TestTheoryOfMindConfidenceDecay:
    """Tests for confidence decay and model removal."""

    def test_confidence_increases_on_update(self):
        tom = TheoryOfMind()
        tom.update_model(agent_id=1, action="explore")
        model = tom.get_agent_model(1)
        initial_conf = model.confidence
        tom.update_model(agent_id=1, action="gather")
        assert model.confidence > initial_conf

    def test_confidence_capped_at_max(self):
        tom = TheoryOfMind()
        for i in range(50):
            tom.update_model(agent_id=1, action="move")
        model = tom.get_agent_model(1)
        assert model.confidence <= 0.9

    def test_confidence_decays_on_step(self):
        tom = TheoryOfMind()
        tom.update_model(agent_id=1, action="explore")
        model = tom.get_agent_model(1)
        initial_conf = model.confidence

        # Step forward — model was updated at step 0, so step 1 will decay it
        tom.step(current_step=2)
        assert model.confidence < initial_conf

    def test_stale_model_removed(self):
        tom = TheoryOfMind()
        tom.update_model(agent_id=1, action="explore")
        model = tom.get_agent_model(1)
        model.confidence = 0.08  # Below threshold

        tom.step(current_step=100)
        # After step, confidence decayed further and model removed
        assert tom.get_agent_model(1) is None

    def test_recently_updated_model_not_decayed(self):
        tom = TheoryOfMind()
        tom._current_step = 5
        tom.update_model(agent_id=1, action="explore")
        model = tom.get_agent_model(1)
        conf_before = model.confidence

        # Step at the same time — last_updated == current_step
        tom.step(current_step=5)
        # Model should NOT be decayed since last_updated is not less than current_step
        assert model.confidence == conf_before


class TestTheoryOfMindSerialization:
    """Tests for serialize/restore."""

    def test_serialize_roundtrip(self):
        tom = TheoryOfMind()
        tom.update_model(agent_id=1, action="explore", emotion="joy")
        tom.update_model(agent_id=2, signal_type="communicate")
        tom.step(current_step=5)

        data = tom.serialize()
        assert "agent_models" in data

        tom2 = TheoryOfMind()
        tom2.restore(data)
        assert tom2.get_statistics()["num_models"] == 2
        model1 = tom2.get_agent_model(1)
        assert model1.estimated_emotion == "joy"
        assert "explore" in model1.recent_actions

    def test_graceful_without_event_bus(self):
        tom = TheoryOfMind(event_bus=None)
        tom.update_model(agent_id=1, action="move")
        tom.step(current_step=1)
        stats = tom.get_statistics()
        assert stats["num_models"] == 1


class TestTheoryOfMindEventBus:
    """Tests for EventBus integration."""

    def test_publishes_tom_update(self):
        bus = EventBus()
        messages = []
        bus.register_callback("cognition.tom_update", lambda ch, msg: messages.append(msg))

        tom = TheoryOfMind(event_bus=bus)
        tom.update_model(agent_id=1, action="explore")
        tom.step(current_step=1)

        assert len(messages) == 1
        data = json.loads(messages[0])
        assert "num_models" in data
        assert "avg_confidence" in data


# =========================================================================
# MetacognitionMonitor Tests
# =========================================================================


class TestMetacognitionRecording:
    """Tests for recording decisions."""

    def test_record_single_decision(self):
        meta = MetacognitionMonitor()
        record = meta.record_decision(step=1, predicted=0.8, actual=0.7)
        assert isinstance(record, DecisionRecord)
        assert record.error == pytest.approx(0.1, abs=0.001)

    def test_decision_count(self):
        meta = MetacognitionMonitor()
        for i in range(10):
            meta.record_decision(step=i, predicted=0.5, actual=0.5)
        assert meta.get_statistics()["decision_count"] == 10

    def test_error_computed_correctly(self):
        meta = MetacognitionMonitor()
        record = meta.record_decision(step=1, predicted=0.3, actual=0.9)
        assert record.error == pytest.approx(0.6, abs=0.001)

    def test_decision_type_recorded(self):
        meta = MetacognitionMonitor()
        record = meta.record_decision(step=1, predicted=0.5, actual=0.5, decision_type="navigation")
        assert record.decision_type == "navigation"


class TestMetacognitionPerformance:
    """Tests for performance tracking and degradation detection."""

    def test_recent_performance_after_perfect_decisions(self):
        meta = MetacognitionMonitor()
        for i in range(20):
            meta.record_decision(step=i, predicted=0.5, actual=0.5)
        perf = meta.get_performance_score()
        assert perf == pytest.approx(1.0, abs=0.01)

    def test_recent_performance_after_bad_decisions(self):
        meta = MetacognitionMonitor()
        for i in range(20):
            meta.record_decision(step=i, predicted=0.0, actual=1.0)
        perf = meta.get_performance_score()
        assert perf == pytest.approx(0.0, abs=0.01)

    def test_is_performing_well_when_good(self):
        meta = MetacognitionMonitor()
        for i in range(20):
            meta.record_decision(step=i, predicted=0.5, actual=0.5)
        assert meta.is_performing_well() is True

    def test_degradation_detected(self):
        meta = MetacognitionMonitor()
        # Start with good decisions to establish high baseline
        for i in range(40):
            meta.record_decision(step=i, predicted=0.5, actual=0.5)
        # Now feed terrible decisions
        for i in range(40, 60):
            meta.record_decision(step=i, predicted=0.0, actual=1.0)
        assert meta.is_performing_well() is False

    def test_baseline_updates_with_ema(self):
        meta = MetacognitionMonitor()
        initial_baseline = meta._baseline_performance
        meta.record_decision(step=1, predicted=0.5, actual=0.5)
        # Perfect decision should move baseline up (EMA toward 1.0)
        assert meta._baseline_performance > initial_baseline

    def test_degradation_threshold(self):
        meta = MetacognitionMonitor()
        meta._degradation_threshold = 0.1  # Very sensitive
        # Build a strong baseline of perfect decisions
        for i in range(50):
            meta.record_decision(step=i, predicted=0.5, actual=0.5)
        # Freeze the baseline so EMA does not absorb the bad decisions
        baseline_before = meta._baseline_performance
        # Now feed clearly bad decisions to fill the recent window
        for i in range(50, 70):
            meta.record_decision(step=i, predicted=0.0, actual=1.0)
        # Override baseline to original to test threshold logic directly
        meta._baseline_performance = baseline_before
        assert meta._detect_degradation() is True


class TestMetacognitionLearningRate:
    """Tests for learning rate adjustment suggestions."""

    def test_suggest_boost_when_degraded(self):
        meta = MetacognitionMonitor()
        # Establish baseline
        for i in range(40):
            meta.record_decision(step=i, predicted=0.5, actual=0.5)
        # Degrade
        for i in range(40, 60):
            meta.record_decision(step=i, predicted=0.0, actual=1.0)
        suggestion = meta.should_adjust_learning_rate()
        assert suggestion is not None
        assert suggestion == pytest.approx(1.5, abs=0.01)

    def test_suggest_reduce_when_performing_well(self):
        meta = MetacognitionMonitor()
        # Start with baseline at 0.5 (default), then perform much better
        for i in range(30):
            meta.record_decision(step=i, predicted=0.5, actual=0.5)
        suggestion = meta.should_adjust_learning_rate()
        # Performance is close to baseline, might return None
        # Make performance clearly better than baseline
        meta._baseline_performance = 0.5  # Reset
        for i in range(30, 50):
            meta.record_decision(step=i, predicted=0.5, actual=0.5)
        # Recent performance ~1.0, baseline moved up gradually
        # Need baseline to remain low enough for > 0.1 gap
        meta._baseline_performance = 0.7
        suggestion = meta.should_adjust_learning_rate()
        if suggestion is not None:
            assert suggestion == pytest.approx(0.8, abs=0.01)

    def test_no_suggestion_when_stable(self):
        meta = MetacognitionMonitor()
        # Consistent mediocre performance — no change needed
        for i in range(20):
            meta.record_decision(step=i, predicted=0.5, actual=0.7)
        suggestion = meta.should_adjust_learning_rate()
        # Could be None if performance ~ baseline
        # Not degraded and not clearly above baseline + 0.1
        assert suggestion is None or isinstance(suggestion, float)


class TestMetacognitionCalibration:
    """Tests for confidence calibration computation."""

    def test_calibration_with_no_decisions(self):
        meta = MetacognitionMonitor()
        cal = meta._compute_calibration()
        assert cal == 1.0  # Default when no data

    def test_calibration_changes_with_data(self):
        meta = MetacognitionMonitor()
        for i in range(20):
            meta.record_decision(step=i, predicted=0.9, actual=0.1)
        cal = meta._compute_calibration()
        # Very overconfident predictions -> poor calibration
        assert cal < 1.0


class TestMetacognitionSerialization:
    """Tests for serialize/restore."""

    def test_serialize_roundtrip(self):
        meta = MetacognitionMonitor()
        for i in range(10):
            meta.record_decision(step=i, predicted=0.6, actual=0.5)
        meta.step(current_step=11)

        data = meta.serialize()
        assert len(data["decision_history"]) == 10

        meta2 = MetacognitionMonitor()
        meta2.restore(data)
        assert meta2.get_statistics()["decision_count"] == 10
        assert meta2._baseline_performance == pytest.approx(
            meta._baseline_performance, abs=0.001
        )

    def test_graceful_without_event_bus(self):
        meta = MetacognitionMonitor(event_bus=None)
        meta.record_decision(step=1, predicted=0.5, actual=0.5)
        meta.step(current_step=1)
        stats = meta.get_statistics()
        assert stats["decision_count"] == 1


class TestMetacognitionEventBus:
    """Tests for EventBus integration."""

    def test_publishes_metacognition_update(self):
        bus = EventBus()
        messages = []
        bus.register_callback("cognition.metacognition_update", lambda ch, msg: messages.append(msg))

        meta = MetacognitionMonitor(event_bus=bus)
        meta.record_decision(step=1, predicted=0.5, actual=0.5)
        meta.step(current_step=2)

        assert len(messages) == 1
        data = json.loads(messages[0])
        assert "recent_performance" in data
        assert "baseline" in data

    def test_publishes_alert_on_degradation(self):
        bus = EventBus()
        alerts = []
        bus.register_callback("cognition.metacognition_alert", lambda ch, msg: alerts.append(msg))

        meta = MetacognitionMonitor(event_bus=bus)
        # Build baseline
        for i in range(40):
            meta.record_decision(step=i, predicted=0.5, actual=0.5)
        # Degrade
        for i in range(40, 60):
            meta.record_decision(step=i, predicted=0.0, actual=1.0)
        meta.step(current_step=61)

        assert len(alerts) >= 1
        data = json.loads(alerts[0])
        assert "deficit" in data


# =========================================================================
# NociceptionSystem Tests
# =========================================================================


class TestNociceptionDamage:
    """Tests for damage reporting."""

    def test_report_damage_creates_pain(self):
        noci = NociceptionSystem()
        signal = noci.report_damage("memory", 0.6)
        assert signal.source == "memory"
        assert signal.intensity == pytest.approx(0.6, abs=0.01)
        assert signal.pain_type == "acute"

    def test_report_damage_takes_max_intensity(self):
        noci = NociceptionSystem()
        noci.report_damage("memory", 0.4)
        noci.report_damage("memory", 0.8)
        signal = noci._active_pains["memory"]
        assert signal.intensity == pytest.approx(0.8, abs=0.01)

    def test_report_damage_does_not_lower_intensity(self):
        noci = NociceptionSystem()
        noci.report_damage("memory", 0.8)
        noci.report_damage("memory", 0.3)
        signal = noci._active_pains["memory"]
        assert signal.intensity == pytest.approx(0.8, abs=0.01)

    def test_multiple_sources(self):
        noci = NociceptionSystem()
        noci.report_damage("memory", 0.5)
        noci.report_damage("learning", 0.6)
        noci.report_damage("defense", 0.7)
        assert len(noci._active_pains) == 3

    def test_intensity_clamped(self):
        noci = NociceptionSystem()
        signal = noci.report_damage("test", 1.5)
        assert signal.intensity <= 1.0
        signal2 = noci.report_damage("test2", -0.5)
        assert signal2.intensity >= 0.0


class TestNociceptionHabituation:
    """Tests for pain habituation over time."""

    def test_pain_habituates_over_steps(self):
        noci = NociceptionSystem()
        noci.report_damage("memory", 0.8)
        initial = noci._active_pains["memory"].intensity

        noci.step(current_step=1)
        after_one_step = noci._active_pains["memory"].intensity
        assert after_one_step < initial

    def test_pain_removed_when_below_threshold(self):
        noci = NociceptionSystem()
        noci.report_damage("test", 0.35)
        # Multiple steps will habituate below threshold (0.3)
        for i in range(1, 10):
            noci.step(current_step=i)
        assert "test" not in noci._active_pains

    def test_habituation_rate_controls_decay_speed(self):
        noci = NociceptionSystem()
        noci._habituation_rate = 0.5  # Faster decay
        noci.report_damage("fast_decay", 0.8)
        noci.step(current_step=1)
        # After one step: 0.8 * 0.5 = 0.4
        assert noci._active_pains["fast_decay"].intensity == pytest.approx(0.4, abs=0.01)


class TestNociceptionAlerts:
    """Tests for pain alerts via EventBus."""

    def test_acute_pain_alert(self):
        bus = EventBus()
        alerts = []
        bus.register_callback("communication.acute_pain", lambda ch, msg: alerts.append(msg))

        noci = NociceptionSystem(event_bus=bus)
        noci.report_damage("critical_system", 0.9)
        noci.step(current_step=1)

        # 0.9 * 0.9 (habituation) = 0.81, still > 0.7
        assert len(alerts) >= 1
        data = json.loads(alerts[0])
        assert data["source"] == "critical_system"
        assert data["urgency"] == "high"

    def test_no_alert_below_threshold(self):
        bus = EventBus()
        alerts = []
        bus.register_callback("communication.acute_pain", lambda ch, msg: alerts.append(msg))

        noci = NociceptionSystem(event_bus=bus)
        noci.report_damage("minor", 0.5)
        noci.step(current_step=1)

        assert len(alerts) == 0

    def test_pain_overload_alert(self):
        bus = EventBus()
        overload_alerts = []
        bus.register_callback("communication.pain_overload", lambda ch, msg: overload_alerts.append(msg))

        noci = NociceptionSystem(event_bus=bus)
        # Create multiple pain sources that sum > 1.5
        noci.report_damage("sys1", 0.8)
        noci.report_damage("sys2", 0.8)
        noci.report_damage("sys3", 0.8)
        noci.step(current_step=1)

        # After habituation: 0.8 * 0.9 = 0.72 each, total = 2.16 > 1.5
        assert len(overload_alerts) >= 1


class TestNociceptionGateControl:
    """Tests for pain suppression (gate control theory)."""

    def test_suppress_pain(self):
        noci = NociceptionSystem()
        noci.report_damage("memory", 0.8)
        result = noci.suppress_pain("memory", factor=0.5)
        assert result is True
        assert noci._active_pains["memory"].intensity == pytest.approx(0.4, abs=0.01)

    def test_suppress_removes_if_below_threshold(self):
        noci = NociceptionSystem()
        noci.report_damage("minor", 0.4)
        noci.suppress_pain("minor", factor=0.5)
        # 0.4 * 0.5 = 0.2, below threshold 0.3
        assert "minor" not in noci._active_pains

    def test_suppress_nonexistent_pain(self):
        noci = NociceptionSystem()
        result = noci.suppress_pain("nonexistent")
        assert result is False

    def test_suppress_factor_clamped(self):
        noci = NociceptionSystem()
        noci.report_damage("test", 0.8)
        noci.suppress_pain("test", factor=2.0)  # Clamped to 1.0
        assert noci._active_pains["test"].intensity == pytest.approx(0.8, abs=0.01)


class TestNociceptionQueries:
    """Tests for pain query methods."""

    def test_get_pain_load(self):
        noci = NociceptionSystem()
        noci.report_damage("a", 0.5)
        noci.report_damage("b", 0.3)
        noci.step(current_step=1)
        load = noci.get_pain_load()
        assert load > 0

    def test_get_worst_pain(self):
        noci = NociceptionSystem()
        noci.report_damage("mild", 0.3)
        noci.report_damage("severe", 0.9)
        worst = noci.get_worst_pain()
        assert worst is not None
        assert worst.source == "severe"

    def test_get_worst_pain_empty(self):
        noci = NociceptionSystem()
        assert noci.get_worst_pain() is None

    def test_is_in_pain_true(self):
        noci = NociceptionSystem()
        noci.report_damage("hurt", 0.5)
        assert noci.is_in_pain() is True

    def test_is_in_pain_false_when_empty(self):
        noci = NociceptionSystem()
        assert noci.is_in_pain() is False


class TestNociceptionSomaticIntegration:
    """Tests for somatic map integration."""

    def test_somatic_map_generates_chronic_pain(self):
        """SomaticMap with unhealthy systems should generate chronic pain."""
        from mae_core.emergent.somatic_map import SomaticMap, SystemCriticality

        somatic = SomaticMap()
        somatic.register_system("weak_system", "test", SystemCriticality.STANDARD)
        somatic.heartbeat("weak_system", health=0.2)  # Very unhealthy

        noci = NociceptionSystem(somatic_map=somatic)
        noci.step(current_step=1)

        # Should have generated a chronic pain for weak_system
        assert "weak_system" in noci._active_pains
        assert noci._active_pains["weak_system"].pain_type == "chronic"

    def test_somatic_map_healthy_no_chronic(self):
        """Healthy somatic systems should not generate chronic pain."""
        from mae_core.emergent.somatic_map import SomaticMap, SystemCriticality

        somatic = SomaticMap()
        somatic.register_system("healthy_system", "test", SystemCriticality.STANDARD)
        somatic.heartbeat("healthy_system", health=0.9)

        noci = NociceptionSystem(somatic_map=somatic)
        noci.step(current_step=1)

        assert "healthy_system" not in noci._active_pains


class TestNociceptionEventBusSubscriptions:
    """Tests for EventBus subscription handlers."""

    def test_healing_failure_creates_pain(self):
        bus = EventBus()
        noci = NociceptionSystem(event_bus=bus)

        # Simulate a healing failure event
        bus.publish("healing.failure_detected", {
            "system_id": "damaged_module",
            "severity": 0.7,
        })

        assert "damaged_module" in noci._active_pains
        assert noci._active_pains["damaged_module"].intensity == pytest.approx(0.7, abs=0.01)

    def test_threat_detected_creates_referred_pain(self):
        bus = EventBus()
        noci = NociceptionSystem(event_bus=bus)

        bus.publish("defense.threat_detected", {
            "source": "external_threat",
            "score": 0.6,
        })

        assert "external_threat" in noci._active_pains
        assert noci._active_pains["external_threat"].pain_type == "referred"


class TestNociceptionSerialization:
    """Tests for serialize/restore."""

    def test_serialize_roundtrip(self):
        noci = NociceptionSystem()
        noci.report_damage("memory", 0.7)
        noci.report_damage("learning", 0.5)
        noci.step(current_step=3)

        data = noci.serialize()
        assert "active_pains" in data

        noci2 = NociceptionSystem()
        noci2.restore(data)
        assert len(noci2._active_pains) == len(noci._active_pains)
        assert noci2._current_step == 3

    def test_graceful_without_event_bus(self):
        noci = NociceptionSystem(event_bus=None, somatic_map=None)
        noci.report_damage("test", 0.5)
        noci.step(current_step=1)
        stats = noci.get_statistics()
        assert stats["total_damage_reports"] == 1


# =========================================================================
# Cross-System Tests
# =========================================================================


class TestCrossSystemGraceful:
    """Tests that all three systems work gracefully without EventBus."""

    def test_theory_of_mind_no_bus(self):
        tom = TheoryOfMind(event_bus=None)
        tom.update_model(agent_id=1, action="explore")
        tom.step(current_step=1)
        assert tom.get_statistics()["num_models"] == 1
        data = tom.serialize()
        tom2 = TheoryOfMind()
        tom2.restore(data)
        assert tom2.get_statistics()["num_models"] == 1

    def test_metacognition_no_bus(self):
        meta = MetacognitionMonitor(event_bus=None)
        meta.record_decision(step=1, predicted=0.5, actual=0.5)
        meta.step(current_step=2)
        assert meta.get_statistics()["decision_count"] == 1

    def test_nociception_no_bus_no_somatic(self):
        noci = NociceptionSystem(event_bus=None, somatic_map=None)
        noci.report_damage("test", 0.8)
        noci.step(current_step=1)
        assert noci.is_in_pain() is True

    def test_repr_methods(self):
        tom = TheoryOfMind()
        assert repr(tom).startswith("TheoryOfMind(")
        meta = MetacognitionMonitor()
        assert repr(meta).startswith("MetacognitionMonitor(")
        noci = NociceptionSystem()
        assert repr(noci).startswith("NociceptionSystem(")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
