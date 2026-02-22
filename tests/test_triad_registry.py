"""Tests for TriadRegistry - startup wiring of all Mae processes.

Verifies that register_all_triads() correctly wires:
- 16 processes with appropriate criticality levels
- Correct validator counts (5 for critical/protected, 3 for standard)
- All odd-numbered validator counts (no deadlock)
- Complementary types (2+ per process)
- Watchdog channel monitoring
- Full compliance when all processes registered
"""

import pytest

from mae_core.backbone.event_bus import EventBus
from mae_core.backbone.triad_auditor import TriadAuditor
from mae_core.backbone.triad_enforcer import (
    ProcessCriticality,
    TriadEnforcer,
    ValidatorType,
)
from mae_core.backbone.triad_registry import register_all_triads
from mae_core.backbone.triad_watchdog import TriadWatchdog


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


@pytest.fixture
def report(enforcer, watchdog, auditor):
    """Run registry and return report."""
    return register_all_triads(enforcer, watchdog, auditor)


# ===========================================================================
# Registration Completeness
# ===========================================================================


class TestRegistrationCompleteness:
    """Verify all 16 processes are registered."""

    def test_total_processes(self, report):
        """All 16 Mae processes are registered."""
        assert report["total_registered"] == 16

    def test_total_validators_added(self, report):
        """Correct total of validators across all processes."""
        # 4 critical × 5 = 20
        # 3 protected × 5 = 15
        # 9 standard × 3 = 27
        # Total = 62
        assert report["total_validators_added"] == 62

    def test_compliance_report_included(self, report):
        """Report includes compliance data from enforcer."""
        assert "compliance_rate" in report
        assert "total_processes" in report

    def test_all_processes_in_enforcer(self, enforcer, report):
        """Every registered process is accessible in enforcer."""
        stats = enforcer.get_statistics()
        assert stats["total_processes"] == 16


# ===========================================================================
# Criticality Levels
# ===========================================================================


class TestCriticalityLevels:
    """Verify correct criticality assignments."""

    CRITICAL_PROCESSES = [
        "self_modification",
        "decision_making",
        "healing",
        "threat_detection",
    ]

    PROTECTED_PROCESSES = [
        "agent_lifecycle",
        "resource_allocation",
        "learning_policy",
    ]

    STANDARD_PROCESSES = [
        "temporal_reasoning",
        "causal_reasoning",
        "world_model_predictions",
        "morphogenesis",
        "endocrine_regulation",
        "circadian_rhythm",
        "communication",
        "memory_storage",
        "consensus",
    ]

    def test_critical_processes_count(self, enforcer, report):
        """4 processes at CRITICAL level."""
        critical = 0
        for pid in self.CRITICAL_PROCESSES:
            status = enforcer.get_process_status(pid)
            assert status is not None, f"{pid} not registered"
            assert status["criticality"] == "critical", f"{pid} not critical"
            critical += 1
        assert critical == 4

    def test_protected_processes_count(self, enforcer, report):
        """3 processes at PROTECTED level."""
        for pid in self.PROTECTED_PROCESSES:
            status = enforcer.get_process_status(pid)
            assert status is not None, f"{pid} not registered"
            assert status["criticality"] == "protected", f"{pid} not protected"

    def test_standard_processes_count(self, enforcer, report):
        """9 processes at STANDARD level."""
        for pid in self.STANDARD_PROCESSES:
            status = enforcer.get_process_status(pid)
            assert status is not None, f"{pid} not registered"
            assert status["criticality"] == "standard", f"{pid} not standard"


# ===========================================================================
# Validator Counts
# ===========================================================================


class TestValidatorCounts:
    """Verify correct validator counts per criticality."""

    def test_critical_have_5_validators(self, enforcer, report):
        """CRITICAL processes have exactly 5 validators."""
        for pid in TestCriticalityLevels.CRITICAL_PROCESSES:
            status = enforcer.get_process_status(pid)
            assert status["validator_count"] == 5, (
                f"{pid} has {status['validator_count']} validators, need 5"
            )

    def test_protected_have_5_validators(self, enforcer, report):
        """PROTECTED processes have exactly 5 validators."""
        for pid in TestCriticalityLevels.PROTECTED_PROCESSES:
            status = enforcer.get_process_status(pid)
            assert status["validator_count"] == 5, (
                f"{pid} has {status['validator_count']} validators, need 5"
            )

    def test_standard_have_3_validators(self, enforcer, report):
        """STANDARD processes have exactly 3 validators."""
        for pid in TestCriticalityLevels.STANDARD_PROCESSES:
            status = enforcer.get_process_status(pid)
            assert status["validator_count"] == 3, (
                f"{pid} has {status['validator_count']} validators, need 3"
            )

    def test_all_odd_counts(self, enforcer, report):
        """Every process has an odd number of validators (no deadlock)."""
        all_processes = (
            TestCriticalityLevels.CRITICAL_PROCESSES
            + TestCriticalityLevels.PROTECTED_PROCESSES
            + TestCriticalityLevels.STANDARD_PROCESSES
        )
        for pid in all_processes:
            status = enforcer.get_process_status(pid)
            count = status["validator_count"]
            assert count % 2 == 1, f"{pid} has even count {count}"


# ===========================================================================
# Complementary Types
# ===========================================================================


class TestComplementaryTypes:
    """Verify validators use different approaches."""

    def test_all_have_complementary_types(self, enforcer, report):
        """Every process has at least 2 different validator types."""
        all_processes = (
            TestCriticalityLevels.CRITICAL_PROCESSES
            + TestCriticalityLevels.PROTECTED_PROCESSES
            + TestCriticalityLevels.STANDARD_PROCESSES
        )
        for pid in all_processes:
            status = enforcer.get_process_status(pid)
            types = status["unique_types"]
            assert len(types) >= 2, (
                f"{pid} has only {len(types)} type(s): {types}"
            )

    def test_critical_have_3_plus_types(self, enforcer, report):
        """CRITICAL processes have at least 3 different validator types."""
        for pid in TestCriticalityLevels.CRITICAL_PROCESSES:
            status = enforcer.get_process_status(pid)
            types = status["unique_types"]
            assert len(types) >= 3, (
                f"{pid} has only {len(types)} types: {types}"
            )


# ===========================================================================
# Full Compliance
# ===========================================================================


class TestFullCompliance:
    """Verify 100% compliance when all processes registered."""

    def test_100_percent_compliance(self, report):
        """All 16 processes are compliant."""
        assert report["compliance_rate"] == 1.0

    def test_no_violations(self, report):
        """No Rule of 3 violations."""
        assert len(report.get("violations", [])) == 0

    def test_enforcer_agrees(self, enforcer, report):
        """Enforcer's own compliance check agrees."""
        check = enforcer.check_compliance()
        assert check["compliant"] == 16
        assert check["non_compliant"] == 0


# ===========================================================================
# Watchdog Integration
# ===========================================================================


class TestWatchdogIntegration:
    """Verify watchdog channel monitoring is configured."""

    def test_watchdog_has_monitored_channels(self, watchdog, report):
        """Watchdog monitors multiple channels after registry."""
        stats = watchdog.get_statistics()
        assert stats["monitored_channels"] >= 5

    def test_healing_channels_monitored(self, watchdog, bus, report):
        """Healing channels are monitored by watchdog."""
        # Simulate a healing event
        bus.publish("healing.started", {"failure_id": "test"})

        activity = watchdog.get_activity("healing")
        assert activity is not None
        assert activity["observed"] >= 1

    def test_self_modification_monitored(self, watchdog, bus, report):
        """Self-modification channel is monitored."""
        bus.publish("improvement.capability_found", {"cap": "test"})
        activity = watchdog.get_activity("self_modification")
        assert activity is not None

    def test_threat_detection_monitored(self, watchdog, bus, report):
        """Threat detection channel is monitored."""
        bus.publish("defense.threat_detected", {"threat": "test"})
        activity = watchdog.get_activity("threat_detection")
        assert activity is not None


# ===========================================================================
# Permissive Defaults (no live systems)
# ===========================================================================


class TestPermissiveDefaults:
    """Verify validators work with permissive defaults (no live systems)."""

    def test_validation_passes_without_systems(self, enforcer, report):
        """All permissive default validators approve."""
        result = enforcer.validate("healing", {"action": "test"})
        assert result.approved
        assert result.unanimous

    def test_all_processes_validate(self, enforcer, report):
        """Every process can run validation with permissive defaults."""
        all_processes = (
            TestCriticalityLevels.CRITICAL_PROCESSES
            + TestCriticalityLevels.PROTECTED_PROCESSES
            + TestCriticalityLevels.STANDARD_PROCESSES
        )
        for pid in all_processes:
            result = enforcer.validate(pid, {"test": True})
            assert result.approved, f"{pid} validation failed"
            assert result.total_validators >= 3, (
                f"{pid} only has {result.total_validators} validators"
            )


# ===========================================================================
# With Live Systems (mock)
# ===========================================================================


class TestWithLiveSystems:
    """Verify validators call live systems when available."""

    def test_live_system_called(self, bus):
        """A live system's validate method gets called."""
        call_log = []

        class MockHaven:
            def validate_modification(self, ctx):
                call_log.append(("haven", ctx))
                return True

        enforcer = TriadEnforcer(event_bus=bus)
        watchdog = TriadWatchdog(event_bus=bus, triad_enforcer=enforcer)
        auditor = TriadAuditor(event_bus=bus, triad_enforcer=enforcer)

        register_all_triads(
            enforcer, watchdog, auditor,
            systems={"haven": MockHaven()},
        )

        # Validate self_modification - haven should be called
        result = enforcer.validate("self_modification", {"test": True})
        assert result.approved

        # Haven's validate_modification was called
        haven_calls = [c for c in call_log if c[0] == "haven"]
        assert len(haven_calls) == 1

    def test_rejecting_system_affects_vote(self, bus):
        """A rejecting validator affects the vote result."""

        class RejectingSystem:
            def validate_modification(self, ctx):
                return False

        enforcer = TriadEnforcer(event_bus=bus)
        watchdog = TriadWatchdog(event_bus=bus, triad_enforcer=enforcer)
        auditor = TriadAuditor(event_bus=bus, triad_enforcer=enforcer)

        register_all_triads(
            enforcer, watchdog, auditor,
            systems={"haven": RejectingSystem()},
        )

        # self_modification has 5 validators; haven rejects, 4 others approve
        result = enforcer.validate("self_modification", {"test": True})
        assert result.approved  # 4 vs 1 - majority wins
        assert not result.unanimous
        assert result.votes_against == 1
        assert "haven" in result.dissenting_validators
