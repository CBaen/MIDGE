"""Tests for TriadWatchdog and TriadAuditor.

Watchdog: Detects when processes bypass their triads.
Auditor: Detects unhealthy voting patterns (echo chambers, captured validators).
"""

import json
import time

import pytest

from mae_core.backbone.event_bus import EventBus
from mae_core.backbone.triad_auditor import (
    CH_AUDIT_FINDING,
    AuditFinding,
    FindingSeverity,
    FindingType,
    TriadAuditor,
)
from mae_core.backbone.triad_enforcer import (
    ProcessCriticality,
    TriadEnforcer,
    ValidatorType,
)
from mae_core.backbone.triad_watchdog import (
    CH_WATCHDOG_BYPASS,
    CH_WATCHDOG_SILENT,
    TriadWatchdog,
)


@pytest.fixture
def bus():
    return EventBus()


@pytest.fixture
def enforcer(bus):
    return TriadEnforcer(event_bus=bus)


@pytest.fixture
def watchdog(bus, enforcer):
    return TriadWatchdog(event_bus=bus, triad_enforcer=enforcer)


@pytest.fixture
def auditor(bus, enforcer):
    return TriadAuditor(event_bus=bus, triad_enforcer=enforcer)


# ===========================================================================
# TestTriadWatchdog
# ===========================================================================


class TestTriadWatchdog:
    """Tests for bypass detection."""

    def test_monitor_channel(self, watchdog, bus):
        """Can monitor a channel for process actions."""
        watchdog.monitor_channel("healing.started", "healing")

        # Simulate action on channel
        bus.publish("healing.started", {"failure_id": "test"})

        activity = watchdog.get_activity("healing")
        assert activity is not None
        assert activity["observed"] == 1
        assert activity["validated"] == 0

    def test_validated_action_recorded(self, watchdog):
        """Validated actions are tracked."""
        watchdog.monitor_channel("test.action", "test")
        watchdog.record_validated_action("test")

        activity = watchdog.get_activity("test")
        assert activity["validated"] == 1

    def test_bypass_detected(self, watchdog, bus):
        """Bypass detected when observed >> validated."""
        watchdog.monitor_channel("healing.action", "healing")

        # 5 observed actions, 0 validated
        for _ in range(5):
            bus.publish("healing.action", {"step": 1})

        alerts = watchdog.check_for_bypasses()
        assert len(alerts) >= 1
        assert alerts[0].process_id == "healing"

    def test_no_bypass_when_validated(self, watchdog, bus):
        """No bypass when actions are properly validated."""
        watchdog.monitor_channel("test.action", "test")

        # 3 observed, 3 validated
        for _ in range(3):
            bus.publish("test.action", {})
            watchdog.record_validated_action("test")

        alerts = watchdog.check_for_bypasses()
        assert len(alerts) == 0

    def test_bypass_eventbus(self, watchdog, bus):
        """Bypass alerts published on EventBus."""
        received = []
        bus.register_callback(
            CH_WATCHDOG_BYPASS,
            lambda ch, msg: received.append(json.loads(msg)),
        )

        watchdog.monitor_channel("healing.action", "healing")
        for _ in range(5):
            bus.publish("healing.action", {})

        watchdog.check_for_bypasses()
        assert len(received) >= 1
        assert received[0]["process_id"] == "healing"

    def test_statistics(self, watchdog, bus):
        """Statistics track monitoring state."""
        watchdog.monitor_channel("ch1", "proc1")
        watchdog.monitor_channel("ch2", "proc2")

        bus.publish("ch1", {})
        bus.publish("ch1", {})
        watchdog.record_validated_action("proc1")

        stats = watchdog.get_statistics()
        assert stats["monitored_processes"] == 2
        assert stats["monitored_channels"] == 2
        assert stats["total_observed_actions"] == 2
        assert stats["total_validated_actions"] == 1

    def test_silent_validator_detection(self, watchdog, bus, enforcer):
        """Detect validators that have never been invoked."""
        enforcer.register_process("test", "Test")
        enforcer.add_validator(
            "test", "v1", ValidatorType.STRUCTURAL,
            lambda ctx: True,
        )
        enforcer.add_validator(
            "test", "v2", ValidatorType.BEHAVIORAL,
            lambda ctx: True,
        )
        enforcer.add_validator(
            "test", "v3", ValidatorType.OPERATIONAL,
            lambda ctx: True,
        )

        # Monitor and observe activity
        watchdog.monitor_channel("test.action", "test")
        bus.publish("test.action", {})

        # None of the validators have been invoked
        silent = watchdog.detect_silent_validators()
        assert len(silent) == 3  # All 3 are silent


# ===========================================================================
# TestTriadAuditor
# ===========================================================================


class TestTriadAuditor:
    """Tests for voting pattern analysis."""

    def test_record_votes(self, auditor):
        """Can record votes for pattern analysis."""
        for _ in range(5):
            auditor.record_vote("healing", "v1", True)
            auditor.record_vote("healing", "v2", True)
            auditor.record_vote("healing", "v3", False)

        stats = auditor.get_statistics()
        assert stats["total_votes_recorded"] == 15
        assert stats["processes_tracked"] == 1

    def test_echo_chamber_detection(self, auditor):
        """Detect suspiciously unanimous voting."""
        # 15 rounds of unanimous approval
        for _ in range(15):
            auditor.record_vote("suspicious", "v1", True)
            auditor.record_vote("suspicious", "v2", True)
            auditor.record_vote("suspicious", "v3", True)

        findings = auditor.run_audit()
        echo_findings = [
            f for f in findings
            if f.finding_type == FindingType.ECHO_CHAMBER
        ]
        assert len(echo_findings) >= 1
        assert echo_findings[0].process_id == "suspicious"

    def test_no_echo_chamber_with_disagreement(self, auditor):
        """No echo chamber when validators disagree sometimes."""
        for i in range(15):
            auditor.record_vote("healthy", "v1", True)
            auditor.record_vote("healthy", "v2", True)
            # v3 disagrees 30% of the time
            auditor.record_vote("healthy", "v3", i % 3 != 0)

        findings = auditor.run_audit()
        echo_findings = [
            f for f in findings
            if f.finding_type == FindingType.ECHO_CHAMBER
            and f.process_id == "healthy"
        ]
        assert len(echo_findings) == 0

    def test_captured_validator_always_yes(self, auditor):
        """Detect validator that always approves."""
        for _ in range(15):
            auditor.record_vote("test", "rubber_stamp", True)
            auditor.record_vote("test", "honest", True if _ % 3 != 0 else False)

        findings = auditor.run_audit()
        captured = [
            f for f in findings
            if f.finding_type == FindingType.CAPTURED_VALIDATOR
            and "rubber_stamp" in f.validator_ids
        ]
        assert len(captured) >= 1

    def test_captured_validator_always_no(self, auditor):
        """Detect validator that always rejects."""
        for _ in range(15):
            auditor.record_vote("test", "blocker", False)
            auditor.record_vote("test", "honest", True if _ % 3 != 0 else False)

        findings = auditor.run_audit()
        captured = [
            f for f in findings
            if f.finding_type == FindingType.CAPTURED_VALIDATOR
            and "blocker" in f.validator_ids
        ]
        assert len(captured) >= 1

    def test_compliance_check_via_auditor(self, auditor, enforcer):
        """Auditor checks Rule of 3 compliance via enforcer."""
        enforcer.register_process(
            "understaffed", "Needs more validators",
            ProcessCriticality.STANDARD,
        )
        enforcer.add_validator(
            "understaffed", "v1", ValidatorType.STRUCTURAL,
            lambda ctx: True,
        )

        findings = auditor.run_audit()
        compliance_findings = [
            f for f in findings
            if f.finding_type == FindingType.LOW_COMPLIANCE
        ]
        assert len(compliance_findings) >= 1

    def test_health_report(self, auditor, enforcer):
        """Health report summarizes audit state."""
        enforcer.register_process("empty", "No validators", ProcessCriticality.CRITICAL)

        report = auditor.get_health_report()
        assert not report["healthy"]
        assert report["critical_count"] >= 1

    def test_audit_eventbus(self, auditor, bus, enforcer):
        """Audit findings published on EventBus."""
        received = []
        bus.register_callback(
            CH_AUDIT_FINDING,
            lambda ch, msg: received.append(json.loads(msg)),
        )

        enforcer.register_process("empty", "No validators")
        auditor.run_audit()

        assert len(received) >= 1


# ===========================================================================
# TestEnforcementTriad
# ===========================================================================


class TestEnforcementTriad:
    """Test all three enforcement systems working together."""

    def test_full_enforcement_triad(self, bus):
        """Enforcer + Watchdog + Auditor form a complete oversight triad."""
        enforcer = TriadEnforcer(event_bus=bus)
        watchdog = TriadWatchdog(event_bus=bus, triad_enforcer=enforcer)
        auditor = TriadAuditor(event_bus=bus, triad_enforcer=enforcer)

        # Register a process with full triad
        enforcer.register_process("healing", "Healing", ProcessCriticality.PROTECTED)
        for vid, vtype in [
            ("auto_healer", ValidatorType.OPERATIONAL),
            ("somatic_map", ValidatorType.STRUCTURAL),
            ("haven", ValidatorType.BEHAVIORAL),
            ("nutrient_flow", ValidatorType.RESOURCE),
            ("causal_engine", ValidatorType.CAUSAL),
        ]:
            enforcer.add_validator(
                "healing", vid, vtype, lambda ctx: True,
            )

        # Watchdog monitors the healing channel
        watchdog.monitor_channel("healing.started", "healing")

        # Simulate: 3 actions observed, 3 validated via enforcer
        for i in range(3):
            bus.publish("healing.started", {"step": i})
            result = enforcer.validate("healing", {"step": i})
            assert result.approved
            watchdog.record_validated_action("healing")

            # Record votes for auditor
            for v in ["auto_healer", "somatic_map", "haven", "nutrient_flow", "causal_engine"]:
                auditor.record_vote("healing", v, True)

        # No bypasses
        bypass_alerts = watchdog.check_for_bypasses()
        assert len(bypass_alerts) == 0

        # Compliance check passes
        compliance = enforcer.check_compliance()
        assert compliance["compliant"] == 1

        # Auditor may flag echo chamber (all unanimous) but no critical issues
        audit = auditor.get_health_report()
        assert audit["critical_count"] == 0

    def test_enforcement_catches_bypass(self, bus):
        """Full triad catches a process that bypasses validation."""
        enforcer = TriadEnforcer(event_bus=bus)
        watchdog = TriadWatchdog(event_bus=bus, triad_enforcer=enforcer, bypass_threshold=3)

        enforcer.register_process("rogue", "Rogue process")
        watchdog.monitor_channel("rogue.action", "rogue")

        # Rogue process acts 5 times without validation
        for _ in range(5):
            bus.publish("rogue.action", {"unauthorized": True})

        # Watchdog catches it
        alerts = watchdog.check_for_bypasses()
        assert len(alerts) >= 1
        assert alerts[0].process_id == "rogue"

        # Enforcer flags non-compliance (no validators)
        compliance = enforcer.check_compliance()
        assert compliance["non_compliant"] >= 1

    def test_three_enforcement_systems_exist(self, bus):
        """Verify we have exactly 3 enforcement systems (the enforcement triad itself)."""
        enforcer = TriadEnforcer(event_bus=bus)
        watchdog = TriadWatchdog(event_bus=bus, triad_enforcer=enforcer)
        auditor = TriadAuditor(event_bus=bus, triad_enforcer=enforcer)

        # Each provides different oversight
        # Enforcer: FORMAL - registration and voting
        assert hasattr(enforcer, 'validate')
        assert hasattr(enforcer, 'check_compliance')

        # Watchdog: OPERATIONAL - bypass detection
        assert hasattr(watchdog, 'check_for_bypasses')
        assert hasattr(watchdog, 'detect_silent_validators')

        # Auditor: BEHAVIORAL - pattern analysis
        assert hasattr(auditor, 'run_audit')
        assert hasattr(auditor, '_check_echo_chambers')
        assert hasattr(auditor, '_check_captured_validators')

        # Three complementary lenses, not copies
        # Enforcer doesn't detect bypasses (Watchdog does)
        # Watchdog doesn't analyze patterns (Auditor does)
        # Auditor doesn't do registration (Enforcer does)
