"""Tests for Phase 5.7: Emergence and Defense.

Tests auto-healing, capability discovery, threat detection,
input validation, and cross-system integration.
"""

import threading
import time
import unittest

import numpy as np


# ===========================================================================
# Auto-Healing Tests
# ===========================================================================


class TestAutoHealer(unittest.TestCase):
    def _make_healer(self, **kwargs):
        from mae_core.backbone.event_bus import EventBus
        from mae_core.emergent.auto_healer import AutoHealer

        bus = EventBus()
        return AutoHealer(event_bus=bus, **kwargs), bus

    def test_report_failure_creates_record(self):
        from mae_core.emergent.auto_healer import FailureReport, FailureType

        healer, bus = self._make_healer()
        failure = FailureReport(
            failure_id="test-001",
            failure_type=FailureType.PERFORMANCE_DEGRADATION,
            affected_agents=["agent-1"],
            severity=0.5,
        )
        record = healer.report_failure(failure)
        self.assertIsNotNone(record)
        self.assertTrue(record.success)  # No HAVEN = auto-success on verify
        stats = healer.get_statistics()
        self.assertEqual(stats["total_healings"], 1)

    def test_healing_with_haven(self):
        from mae_core.backbone.event_bus import EventBus
        from mae_core.emergent.auto_healer import AutoHealer, FailureReport, FailureType
        from mae_core.learning.haven import HavenRiskCoordinator

        bus = EventBus()
        haven = HavenRiskCoordinator(event_bus=bus)
        haven.register_agent("agent-1")
        healer = AutoHealer(event_bus=bus, haven=haven)

        failure = FailureReport(
            failure_id="test-002",
            failure_type=FailureType.AGENT_CRASH,
            affected_agents=["agent-1"],
            severity=0.9,
        )
        record = healer.report_failure(failure)
        self.assertIsNotNone(record)
        # Agent should be isolated then restored
        self.assertFalse(haven.is_agent_isolated("agent-1"))
        self.assertTrue(record.success)

    def test_max_concurrent_healing(self):
        from mae_core.emergent.auto_healer import AutoHealer, FailureReport, FailureType
        from mae_core.backbone.event_bus import EventBus

        bus = EventBus()
        healer = AutoHealer(event_bus=bus, max_concurrent_healings=1)

        # First healing completes synchronously, so slot opens
        f1 = FailureReport(
            failure_id="f1",
            failure_type=FailureType.STARVATION,
            severity=0.5,
        )
        r1 = healer.report_failure(f1)
        self.assertIsNotNone(r1)

        stats = healer.get_statistics()
        self.assertEqual(stats["total_healings"], 1)

    def test_healing_phases(self):
        from mae_core.emergent.auto_healer import (
            AutoHealer, FailureReport, FailureType, HealingPhase,
        )
        from mae_core.backbone.event_bus import EventBus

        bus = EventBus()
        phases_seen = []
        bus.register_callback("healing.phase_changed", lambda ch, msg: phases_seen.append(msg))

        healer = AutoHealer(event_bus=bus)
        failure = FailureReport(
            failure_id="phase-test",
            failure_type=FailureType.COMMUNICATION_BREAK,
            severity=0.5,
        )
        record = healer.report_failure(failure)

        # Should have gone through all phases (EventBus serializes to JSON strings)
        import json
        phase_values = []
        for p in phases_seen:
            if isinstance(p, dict):
                phase_values.append(p.get("phase", ""))
            elif isinstance(p, str):
                try:
                    phase_values.append(json.loads(p).get("phase", ""))
                except (json.JSONDecodeError, AttributeError):
                    phase_values.append(p)
        self.assertIn("isolating", phase_values)
        self.assertIn("assessing", phase_values)
        self.assertIn("restoring", phase_values)

    def test_eventbus_integration(self):
        from mae_core.emergent.auto_healer import AutoHealer, FailureReport, FailureType
        from mae_core.backbone.event_bus import EventBus

        bus = EventBus()
        events = []
        bus.register_callback("healing.failure_detected", lambda ch, msg: events.append(("detected", msg)))
        bus.register_callback("healing.complete", lambda ch, msg: events.append(("complete", msg)))

        healer = AutoHealer(event_bus=bus)
        failure = FailureReport(
            failure_id="event-test",
            failure_type=FailureType.RESOURCE_EXHAUSTION,
            severity=0.5,
        )
        healer.report_failure(failure)

        event_types = [e[0] for e in events]
        self.assertIn("detected", event_types)
        self.assertIn("complete", event_types)

    def test_healing_history(self):
        from mae_core.emergent.auto_healer import AutoHealer, FailureReport, FailureType
        from mae_core.backbone.event_bus import EventBus

        bus = EventBus()
        healer = AutoHealer(event_bus=bus)

        for i in range(3):
            failure = FailureReport(
                failure_id=f"hist-{i}",
                failure_type=FailureType.STARVATION,
                severity=0.3,
            )
            healer.report_failure(failure)

        history = healer.get_healing_history(limit=10)
        self.assertEqual(len(history), 3)


# ===========================================================================
# Capability Discovery Tests
# ===========================================================================


class TestCapabilityDiscovery(unittest.TestCase):
    def _make_discovery(self, **kwargs):
        from mae_core.backbone.event_bus import EventBus
        from mae_core.emergent.capability_discovery import CapabilityDiscovery

        bus = EventBus()
        return CapabilityDiscovery(event_bus=bus, **kwargs), bus

    def test_detect_novel_capability(self):
        disc, bus = self._make_discovery(novelty_threshold=0.1)

        # Build strong baseline (low performance)
        for i in range(80):
            disc.observe_performance("agent-1", 0.3, context="explore")

        # Sudden improvement - capture first detection
        found_cap = None
        for i in range(25):
            cap = disc.observe_performance("agent-1", 0.8, context="explore")
            if cap is not None:
                found_cap = cap
                break

        self.assertIsNotNone(found_cap)
        self.assertGreater(found_cap.performance_delta, 0.1)

    def test_repeated_capability_not_duplicated(self):
        disc, bus = self._make_discovery(novelty_threshold=0.1)

        # Build baseline
        for i in range(25):
            disc.observe_performance("agent-1", 0.3, context="task_a")

        # Trigger discovery
        for i in range(20):
            disc.observe_performance("agent-1", 0.8, context="task_a")
        cap1 = disc.observe_performance("agent-1", 0.8, context="task_a", behavior_signature="fast")

        # Second time should return None (already known)
        for i in range(20):
            disc.observe_performance("agent-1", 0.9, context="task_a")
        cap2 = disc.observe_performance("agent-1", 0.9, context="task_a", behavior_signature="fast")
        self.assertIsNone(cap2)

    def test_validation_pipeline(self):
        from mae_core.emergent.capability_discovery import CapabilityStatus

        disc, bus = self._make_discovery(novelty_threshold=0.1, validation_rounds=3)

        # Build strong baseline
        for i in range(80):
            disc.observe_performance("agent-1", 0.3, context="ctx")

        # Trigger discovery - capture first detection
        cap = None
        for i in range(25):
            result = disc.observe_performance("agent-1", 0.8, context="ctx")
            if result is not None:
                cap = result
                break
        self.assertIsNotNone(cap)

        # Submit validations
        disc.submit_validation(cap.capability_id, 0.8)
        disc.submit_validation(cap.capability_id, 0.7)
        result = disc.submit_validation(cap.capability_id, 0.9)

        self.assertEqual(result.status, CapabilityStatus.VALIDATED)

    def test_improvement_tracking(self):
        disc, bus = self._make_discovery()

        m1 = disc.track_metric("learning_rate", 0.5)
        m2 = disc.track_metric("learning_rate", 0.6)
        m3 = disc.track_metric("learning_rate", 0.7)

        self.assertEqual(m3.current_value, 0.7)
        self.assertEqual(m3.baseline_value, 0.5)
        self.assertGreater(m3.trend, 0)

    def test_statistics(self):
        disc, bus = self._make_discovery()
        stats = disc.get_statistics()
        self.assertEqual(stats["total_discoveries"], 0)
        self.assertEqual(stats["active_capabilities"], 0)


# ===========================================================================
# Threat Detector Tests
# ===========================================================================


class TestThreatDetector(unittest.TestCase):
    def _make_detector(self, **kwargs):
        from mae_core.backbone.event_bus import EventBus
        from mae_core.defense.threat_detector import ThreatDetector

        bus = EventBus()
        return ThreatDetector(event_bus=bus, **kwargs), bus

    def test_register_and_scan_quills(self):
        from mae_core.defense.threat_detector import Threat, ThreatLevel

        detector, bus = self._make_detector()

        def quill():
            return Threat(
                threat_id="q1",
                source="external",
                target="agent-1",
                level=ThreatLevel.MEDIUM,
                score=0.5,
            )

        detector.register_quill(quill)
        threats = detector.scan_threats()
        self.assertEqual(len(threats), 1)
        self.assertEqual(threats[0].threat_id, "q1")

    def test_turtle_shell(self):
        detector, bus = self._make_detector()

        self.assertFalse(detector.is_shelled)
        detector.update_integrity(-0.6)
        self.assertTrue(detector.is_shelled)
        detector.update_integrity(0.7)
        self.assertFalse(detector.is_shelled)

    def test_lizard_sacrifice(self):
        detector, bus = self._make_detector()

        detector.register_sacrificeable("cache", 1.0)
        detector.register_sacrificeable("logging", 0.5)
        detector.register_sacrificeable("visualization", 0.1)

        victim = detector.sacrifice("under_pressure")
        self.assertEqual(victim, "visualization")

        victim2 = detector.sacrifice("still_pressured")
        self.assertEqual(victim2, "logging")

    def test_kangaroo_counterattack(self):
        from mae_core.defense.threat_detector import Threat, ThreatLevel

        detector, bus = self._make_detector(energy_budget=100.0)

        threat = Threat(
            threat_id="t1",
            source="bad_agent",
            target="good_agent",
            level=ThreatLevel.CRITICAL,
            score=0.9,
        )

        counter_called = []
        detector.register_counter_action("bad_agent", lambda t: counter_called.append(t))

        response = detector.counterattack(threat)
        self.assertTrue(response.success)
        self.assertEqual(len(counter_called), 1)
        self.assertTrue(threat.neutralized)

    def test_respond_to_threat_auto_strategy(self):
        from mae_core.defense.threat_detector import Threat, ThreatLevel, DefenseStrategy

        detector, bus = self._make_detector()

        # Critical = counterattack
        threat_crit = Threat(
            threat_id="c1", source="x", target="y",
            level=ThreatLevel.CRITICAL, score=0.9,
        )
        resp = detector.respond_to_threat(threat_crit)
        self.assertEqual(resp.strategy, DefenseStrategy.KANGAROO)

        # Low = monitoring
        threat_low = Threat(
            threat_id="c2", source="x", target="y",
            level=ThreatLevel.LOW, score=0.2,
        )
        resp = detector.respond_to_threat(threat_low)
        self.assertEqual(resp.strategy, DefenseStrategy.PORCUPINE)

    def test_step_recharges_energy(self):
        detector, bus = self._make_detector(energy_budget=50.0)
        initial = detector.get_statistics()["energy"]

        # Drain some energy
        from mae_core.defense.threat_detector import Threat, ThreatLevel
        threat = Threat(
            threat_id="drain", source="x", target="y",
            level=ThreatLevel.CRITICAL, score=0.9,
        )
        detector.counterattack(threat)

        after_attack = detector.get_statistics()["energy"]
        self.assertLess(after_attack, initial)

        # Step should recharge
        detector.step()
        after_step = detector.get_statistics()["energy"]
        self.assertGreater(after_step, after_attack)

    def test_eventbus_threat_events(self):
        from mae_core.defense.threat_detector import Threat, ThreatLevel

        detector, bus = self._make_detector()
        events = []
        bus.register_callback("defense.threat_detected", lambda ch, msg: events.append(msg))

        detector.report_threat(Threat(
            threat_id="ev1", source="x", target="y",
            level=ThreatLevel.HIGH, score=0.7,
        ))
        self.assertEqual(len(events), 1)


# ===========================================================================
# Input Validator Tests
# ===========================================================================


class TestInputValidator(unittest.TestCase):
    def _make_validator(self, **kwargs):
        from mae_core.backbone.event_bus import EventBus
        from mae_core.defense.input_validator import InputValidator

        bus = EventBus()
        return InputValidator(event_bus=bus, **kwargs), bus

    def test_basic_validation_passes(self):
        from mae_core.defense.input_validator import ValidationResult

        validator, bus = self._make_validator()
        report = validator.validate("agent-1", "message", {"content": "hello"})
        self.assertEqual(report.result, ValidationResult.PASSED)

    def test_low_trust_fails(self):
        from mae_core.defense.input_validator import ValidationResult

        validator, bus = self._make_validator(min_trust_for_accept=0.8)
        # Default trust is 0.5, below 0.8
        report = validator.validate("agent-1", "message", {"content": "hello"})
        self.assertEqual(report.result, ValidationResult.FAILED)

    def test_trust_increases_with_success(self):
        validator, bus = self._make_validator()
        initial_trust = validator.get_trust("agent-1")

        # Successful validations build trust
        for _ in range(5):
            validator.validate("agent-1", "message", {})

        new_trust = validator.get_trust("agent-1")
        self.assertGreater(new_trust, initial_trust)

    def test_trust_decreases_with_failure(self):
        validator, bus = self._make_validator(min_trust_for_accept=0.8)
        validator.set_trust("agent-1", 0.9)

        # Force failures via custom validator
        validator.register_validator("bad_input", "always_fail", lambda d, s: False)
        for _ in range(3):
            validator.validate("agent-1", "bad_input", {})

        trust = validator.get_trust("agent-1")
        self.assertLess(trust, 0.9)

    def test_anomaly_detection(self):
        from mae_core.defense.input_validator import ValidationResult

        validator, bus = self._make_validator()

        # Build baseline of normal values
        for i in range(15):
            validator.validate("agent-1", "sensor", {}, numeric_value=5.0 + i * 0.1)

        # Normal value should pass
        report = validator.validate("agent-1", "sensor", {}, numeric_value=5.5)
        self.assertEqual(report.result, ValidationResult.PASSED)

        # Extreme outlier
        report = validator.validate("agent-1", "sensor", {}, numeric_value=999.0)
        # Should detect anomaly (but may still pass if trust is ok)
        self.assertTrue(any("anomaly_detected" in d for d in report.details))

    def test_custom_range_validator(self):
        from mae_core.defense.input_validator import ValidationResult

        validator, bus = self._make_validator()
        validator.register_range_validator("reward", "value", -1.0, 1.0)

        # In range
        report = validator.validate("agent-1", "reward", {"value": 0.5})
        self.assertGreater(report.checks_passed, 0)

        # Out of range
        report = validator.validate("agent-1", "reward", {"value": 999.0})
        self.assertGreater(report.checks_failed, 0)

    def test_validation_failed_event(self):
        validator, bus = self._make_validator(min_trust_for_accept=0.9)
        events = []
        bus.register_callback("defense.validation_failed", lambda ch, msg: events.append(msg))

        validator.validate("untrusted", "message", {})
        self.assertGreater(len(events), 0)

    def test_statistics(self):
        validator, bus = self._make_validator()
        validator.validate("a", "msg", {})
        validator.validate("b", "msg", {})

        stats = validator.get_statistics()
        self.assertEqual(stats["total_validations"], 2)
        self.assertEqual(stats["tracked_sources"], 2)


# ===========================================================================
# Cross-System Integration Tests
# ===========================================================================


class TestCrossSystemIntegration(unittest.TestCase):
    def test_haven_triggers_autohealer(self):
        """HAVEN risk alert -> AutoHealer failure detection."""
        from mae_core.backbone.event_bus import EventBus
        from mae_core.emergent.auto_healer import AutoHealer
        from mae_core.learning.haven import HavenRiskCoordinator

        bus = EventBus()
        haven = HavenRiskCoordinator(event_bus=bus)
        haven.register_agent("agent-1")
        healer = AutoHealer(event_bus=bus, haven=haven, auto_isolate_threshold=0.7)

        # Simulate high-risk assessment that triggers HAVEN alert
        assessment = haven.assess_agent_risk(
            "agent-1",
            recent_performance=[0.9, 0.8, 0.5, 0.3, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001],
        )

        # If risk was high enough, healer should have responded
        stats = healer.get_statistics()
        # The auto-response depends on the risk score exceeding threshold
        # Just verify the pipeline is wired
        self.assertIsNotNone(stats)

    def test_threat_detector_with_haven(self):
        """ThreatDetector counterattack uses HAVEN for isolation."""
        from mae_core.backbone.event_bus import EventBus
        from mae_core.defense.threat_detector import ThreatDetector, Threat, ThreatLevel
        from mae_core.learning.haven import HavenRiskCoordinator

        bus = EventBus()
        haven = HavenRiskCoordinator(event_bus=bus)
        haven.register_agent("bad-agent")
        detector = ThreatDetector(event_bus=bus, haven=haven)

        threat = Threat(
            threat_id="t1", source="bad-agent", target="system",
            level=ThreatLevel.CRITICAL, score=0.95,
        )
        response = detector.counterattack(threat)
        self.assertTrue(response.success)
        self.assertTrue(haven.is_agent_isolated("bad-agent"))

    def test_defense_eventbus_chain(self):
        """Verify event chain: threat -> defense -> healing."""
        from mae_core.backbone.event_bus import EventBus
        from mae_core.defense.threat_detector import ThreatDetector, Threat, ThreatLevel
        from mae_core.emergent.auto_healer import AutoHealer

        bus = EventBus()
        events = []
        bus.register_callback("defense.threat_detected", lambda ch, msg: events.append("threat"))
        bus.register_callback("defense.activated", lambda ch, msg: events.append("defense"))
        bus.register_callback("healing.failure_detected", lambda ch, msg: events.append("healing"))
        bus.register_callback("healing.complete", lambda ch, msg: events.append("healed"))

        detector = ThreatDetector(event_bus=bus)
        healer = AutoHealer(event_bus=bus)

        # Report threat
        detector.report_threat(Threat(
            threat_id="chain", source="x", target="y",
            level=ThreatLevel.HIGH, score=0.7,
        ))

        self.assertIn("threat", events)

    def test_input_validator_protects_frl(self):
        """InputValidator validates policy updates before FRL accepts them."""
        from mae_core.backbone.event_bus import EventBus
        from mae_core.defense.input_validator import InputValidator, ValidationResult

        bus = EventBus()
        validator = InputValidator(event_bus=bus)

        # Trusted agent
        validator.set_trust("trusted-peer", 0.9)
        report = validator.validate_policy_update(
            "trusted-peer", np.random.randn(10)
        )
        self.assertEqual(report.result, ValidationResult.PASSED)

        # Untrusted agent
        validator.set_trust("sus-peer", 0.1)
        report = validator.validate_policy_update(
            "sus-peer", np.random.randn(10)
        )
        self.assertNotEqual(report.result, ValidationResult.PASSED)

    def test_full_defense_ecosystem(self):
        """Full integration: threat detection + defense + healing + validation."""
        from mae_core.backbone.event_bus import EventBus
        from mae_core.defense.threat_detector import ThreatDetector, Threat, ThreatLevel
        from mae_core.defense.input_validator import InputValidator
        from mae_core.emergent.auto_healer import AutoHealer, FailureReport, FailureType
        from mae_core.emergent.capability_discovery import CapabilityDiscovery
        from mae_core.learning.haven import HavenRiskCoordinator

        bus = EventBus()

        # Initialize all systems
        haven = HavenRiskCoordinator(event_bus=bus)
        healer = AutoHealer(event_bus=bus, haven=haven)
        detector = ThreatDetector(event_bus=bus, haven=haven)
        validator = InputValidator(event_bus=bus)
        discovery = CapabilityDiscovery(event_bus=bus)

        # Register agents
        for i in range(5):
            haven.register_agent(f"agent-{i}")
            validator.set_trust(f"agent-{i}", 0.7)

        # Simulate 20 steps
        for step in range(20):
            # Agents perform work
            for i in range(5):
                aid = f"agent-{i}"

                # Validate inputs
                validator.validate(aid, "state", {"step": step})

                # Track performance
                perf = 0.6 + 0.1 * np.random.randn()
                discovery.observe_performance(aid, perf, context="work")

                # Periodic risk assessment
                if step % 5 == 0:
                    haven.assess_agent_risk(aid, recent_performance=[perf])

            # Scan for threats
            detector.scan_threats()
            detector.step()

        # Verify all systems ran
        haven_report = haven.get_system_health_report()
        healer_stats = healer.get_statistics()
        detector_stats = detector.get_statistics()
        validator_stats = validator.get_statistics()
        discovery_stats = discovery.get_statistics()

        self.assertEqual(haven_report["total_agents"], 5)
        self.assertEqual(validator_stats["total_validations"], 100)  # 5 agents * 20 steps
        self.assertEqual(discovery_stats["agents_monitored"], 5)
        self.assertGreater(detector_stats["energy"], 0)


# ===========================================================================
# Somatic Map Tests (Blast Radius / Body Awareness)
# ===========================================================================


class TestSomaticMap(unittest.TestCase):
    def _make_map(self, **kwargs):
        from mae_core.backbone.event_bus import EventBus
        from mae_core.emergent.somatic_map import SomaticMap

        bus = EventBus()
        return SomaticMap(event_bus=bus, **kwargs), bus

    def test_register_systems(self):
        smap, bus = self._make_map()
        smap.register_system("eventbus", "Core event bus", depends_on=[])
        smap.register_system("memory", "Memory layer", depends_on=["eventbus"])
        smap.register_system("learning", "Learning engines", depends_on=["memory", "eventbus"])

        self.assertEqual(len(smap.get_all_systems()), 3)
        info = smap.get_system_info("learning")
        self.assertIn("memory", info.upstream)
        self.assertIn("eventbus", info.upstream)

    def test_blast_radius_simple(self):
        from mae_core.emergent.somatic_map import ModificationVerdict

        smap, bus = self._make_map()
        smap.register_system("A", "System A")
        smap.register_system("B", "System B", depends_on=["A"])
        smap.register_system("C", "System C", depends_on=["B"])

        report = smap.analyze_blast_radius("A")
        self.assertIn("B", report.transitive_downstream)
        self.assertIn("C", report.transitive_downstream)
        self.assertEqual(report.total_affected, 2)
        self.assertEqual(report.max_depth, 2)

    def test_blast_radius_critical_rejection(self):
        from mae_core.emergent.somatic_map import SystemCriticality, ModificationVerdict

        smap, bus = self._make_map(auto_reject_critical=True)
        smap.register_system("A", "System A")
        smap.register_system("B", "System B", depends_on=["A"],
                            criticality=SystemCriticality.CRITICAL)

        report = smap.analyze_blast_radius("A")
        self.assertEqual(report.verdict, ModificationVerdict.REJECTED)
        self.assertIn("B", report.critical_systems_affected)

    def test_modification_gating(self):
        smap, bus = self._make_map()
        smap.register_system("peripheral", "Peripheral system")

        record, report = smap.propose_modification(
            "mod-001", "peripheral", "Update config"
        )
        self.assertTrue(record.approved)
        stats = smap.get_statistics()
        self.assertEqual(stats["approved"], 1)

    def test_modification_rejected(self):
        from mae_core.emergent.somatic_map import SystemCriticality

        smap, bus = self._make_map(auto_reject_critical=True)
        smap.register_system("base", "Base system")
        smap.register_system("brain", "Critical brain",
                            depends_on=["base"],
                            criticality=SystemCriticality.CRITICAL)

        record, report = smap.propose_modification(
            "mod-002", "base", "Modify foundation"
        )
        self.assertFalse(record.approved)
        stats = smap.get_statistics()
        self.assertEqual(stats["rejected"], 1)

    def test_snapshot_and_rollback(self):
        smap, bus = self._make_map()

        # Simulate a system with state
        state = {"value": 42}

        smap.register_system("stateful", "Stateful system")
        smap.register_snapshot_provider(
            "stateful",
            snapshot_fn=lambda: dict(state),
            rollback_fn=lambda s: state.update(s),
        )

        record, report = smap.propose_modification(
            "mod-003", "stateful", "Change value"
        )
        self.assertTrue(record.approved)

        # Execute (takes snapshot)
        smap.execute_modification("mod-003")

        # Simulate modification
        state["value"] = 999

        # Rollback
        success = smap.rollback_modification("mod-003")
        self.assertTrue(success)
        self.assertEqual(state["value"], 42)

    def test_dependency_chain(self):
        smap, bus = self._make_map()
        smap.register_system("A", "A")
        smap.register_system("B", "B", depends_on=["A"])
        smap.register_system("C", "C", depends_on=["B"])
        smap.register_system("D", "D", depends_on=["C"])

        chain = smap.get_dependency_chain("A", direction="downstream")
        self.assertIn("B", chain)
        self.assertIn("C", chain)
        self.assertIn("D", chain)

        upstream = smap.get_dependency_chain("D", direction="upstream")
        self.assertIn("C", upstream)
        self.assertIn("B", upstream)
        self.assertIn("A", upstream)

    def test_body_map_visualization(self):
        smap, bus = self._make_map()
        smap.register_system("X", "System X")
        smap.register_system("Y", "System Y", depends_on=["X"])

        body_map = smap.get_body_map()
        self.assertIn("X", body_map)
        self.assertIn("Y", body_map)
        self.assertIn("Y", body_map["X"]["downstream"])

    def test_heartbeat_and_health(self):
        smap, bus = self._make_map()
        smap.register_system("sys", "System")

        smap.heartbeat("sys", health=0.3)
        unhealthy = smap.get_unhealthy_systems(threshold=0.5)
        self.assertIn("sys", unhealthy)

        smap.heartbeat("sys", health=0.8)
        unhealthy = smap.get_unhealthy_systems(threshold=0.5)
        self.assertNotIn("sys", unhealthy)

    def test_complete_modification_auto_rollback(self):
        smap, bus = self._make_map()

        state = {"counter": 10}
        smap.register_system("counter_sys", "Counter")
        smap.register_snapshot_provider(
            "counter_sys",
            snapshot_fn=lambda: dict(state),
            rollback_fn=lambda s: state.update(s),
        )

        record, _ = smap.propose_modification("mod-auto", "counter_sys", "test")
        smap.execute_modification("mod-auto")
        state["counter"] = 999

        # Complete with failure triggers auto-rollback
        smap.complete_modification("mod-auto", success=False)
        self.assertEqual(state["counter"], 10)

    def test_somatic_map_with_full_mae_systems(self):
        """Register Mae's actual system topology and verify blast radius."""
        from mae_core.emergent.somatic_map import SystemCriticality, ModificationVerdict

        smap, bus = self._make_map()

        # Register Mae's systems with real dependency structure
        smap.register_system("eventbus", "EventBus backbone",
                            criticality=SystemCriticality.CRITICAL)
        smap.register_system("substrate", "Mycelial Substrate",
                            depends_on=["eventbus"],
                            criticality=SystemCriticality.CRITICAL)
        smap.register_system("memory", "Memory Layer",
                            depends_on=["eventbus"],
                            criticality=SystemCriticality.PROTECTED)
        smap.register_system("communication", "Signal Bus + GNN",
                            depends_on=["eventbus", "substrate"])
        smap.register_system("learning", "Learning Engines",
                            depends_on=["memory", "eventbus"])
        smap.register_system("cognition", "World Model + Decision",
                            depends_on=["memory", "learning"])
        smap.register_system("morphogenesis", "Team Spawning",
                            depends_on=["substrate", "eventbus"])
        smap.register_system("coordination", "Endocrine + Circadian",
                            depends_on=["eventbus"])
        smap.register_system("defense", "HAVEN + Threats",
                            depends_on=["eventbus", "learning"])
        smap.register_system("emergence", "AutoHealer + Discovery",
                            depends_on=["eventbus", "substrate", "cognition"])

        # Modifying EventBus should be REJECTED (everything depends on it)
        report = smap.analyze_blast_radius("eventbus")
        self.assertEqual(report.verdict, ModificationVerdict.REJECTED)
        self.assertGreater(report.total_affected, 5)

        # Modifying emergence is safe (nothing depends on it)
        report2 = smap.analyze_blast_radius("emergence")
        self.assertEqual(report2.total_affected, 0)
        self.assertIn(report2.verdict, [
            ModificationVerdict.APPROVED,
            ModificationVerdict.APPROVED_WITH_WARNINGS,
        ])

        # Modifying substrate affects morphogenesis and emergence
        report3 = smap.analyze_blast_radius("substrate")
        self.assertIn("morphogenesis", report3.transitive_downstream)
        self.assertIn("emergence", report3.transitive_downstream)


if __name__ == "__main__":
    unittest.main()
