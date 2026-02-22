"""Tests for TriadEnforcer - Rule of 3 structural guarantee.

Tests that every major process can register validator triads,
enforce minimum counts, run majority votes, and detect violations.
"""

import json
import time

import pytest

from mae_core.backbone.event_bus import EventBus
from mae_core.backbone.triad_enforcer import (
    CH_TRIAD_REGISTERED,
    CH_TRIAD_VIOLATION,
    CH_TRIAD_VOTE_COMPLETE,
    MINIMUM_VALIDATORS,
    ProcessCriticality,
    ProcessTriad,
    TriadEnforcer,
    Validator,
    ValidatorType,
    VoteResult,
)


@pytest.fixture
def bus():
    return EventBus()


@pytest.fixture
def enforcer(bus):
    return TriadEnforcer(event_bus=bus)


def always_approve(ctx):
    return True


def always_reject(ctx):
    return False


def approve_if_safe(ctx):
    return ctx.get("risk_score", 1.0) < 0.8


# ===========================================================================
# TestProcessRegistration
# ===========================================================================


class TestProcessRegistration:
    """Tests for registering processes and validators."""

    def test_register_process(self, enforcer):
        """Can register a major process."""
        triad = enforcer.register_process(
            "self_modification", "Self-improvement oversight",
            ProcessCriticality.CRITICAL,
        )
        assert triad.process_id == "self_modification"
        assert triad.criticality == ProcessCriticality.CRITICAL

    def test_add_validators(self, enforcer):
        """Can add validators to a process."""
        enforcer.register_process("healing", "Healing oversight")

        enforcer.add_validator(
            "healing", "somatic_map", ValidatorType.STRUCTURAL,
            always_approve, "Structural blast radius check",
        )
        enforcer.add_validator(
            "healing", "haven", ValidatorType.BEHAVIORAL,
            always_approve, "Behavioral trust check",
        )
        enforcer.add_validator(
            "healing", "auto_healer", ValidatorType.OPERATIONAL,
            always_approve, "Operational recovery check",
        )

        status = enforcer.get_process_status("healing")
        assert status["validator_count"] == 3
        assert status["is_compliant"]

    def test_duplicate_validator_rejected(self, enforcer):
        """Cannot add the same validator twice."""
        enforcer.register_process("test", "Test process")
        enforcer.add_validator("test", "v1", ValidatorType.STRUCTURAL, always_approve)
        result = enforcer.add_validator("test", "v1", ValidatorType.BEHAVIORAL, always_approve)
        assert result is False

    def test_unregistered_process_rejected(self, enforcer):
        """Cannot add validator to unregistered process."""
        result = enforcer.add_validator(
            "nonexistent", "v1", ValidatorType.STRUCTURAL, always_approve
        )
        assert result is False

    def test_eventbus_on_registration(self, enforcer, bus):
        """EventBus receives registration events."""
        received = []
        bus.register_callback(
            CH_TRIAD_REGISTERED,
            lambda ch, msg: received.append(json.loads(msg)),
        )
        enforcer.register_process("test", "Test", ProcessCriticality.STANDARD)
        assert len(received) == 1
        assert received[0]["process_id"] == "test"


# ===========================================================================
# TestComplianceChecking
# ===========================================================================


class TestComplianceChecking:
    """Tests for Rule of 3 compliance validation."""

    def test_compliant_triad(self, enforcer):
        """3 validators with 2+ types is compliant."""
        enforcer.register_process("good", "Good process")
        enforcer.add_validator("good", "v1", ValidatorType.STRUCTURAL, always_approve)
        enforcer.add_validator("good", "v2", ValidatorType.BEHAVIORAL, always_approve)
        enforcer.add_validator("good", "v3", ValidatorType.OPERATIONAL, always_approve)

        status = enforcer.get_process_status("good")
        assert status["is_compliant"]

    def test_non_compliant_too_few(self, enforcer):
        """Fewer than 3 validators is non-compliant."""
        enforcer.register_process("bad", "Understaffed process")
        enforcer.add_validator("bad", "v1", ValidatorType.STRUCTURAL, always_approve)

        status = enforcer.get_process_status("bad")
        assert not status["is_compliant"]

    def test_non_compliant_even_count(self, enforcer):
        """Even number of validators is non-compliant (deadlock risk)."""
        enforcer.register_process("even", "Even process")
        for i in range(4):
            vtype = ValidatorType.STRUCTURAL if i % 2 == 0 else ValidatorType.BEHAVIORAL
            enforcer.add_validator(f"even", f"v{i}", vtype, always_approve)

        status = enforcer.get_process_status("even")
        assert not status["is_compliant"]
        assert status["validator_count"] == 4

    def test_non_compliant_same_type(self, enforcer):
        """All validators same type is non-compliant (not complementary)."""
        enforcer.register_process("copies", "Copy process")
        enforcer.add_validator("copies", "v1", ValidatorType.STRUCTURAL, always_approve)
        enforcer.add_validator("copies", "v2", ValidatorType.STRUCTURAL, always_approve)
        enforcer.add_validator("copies", "v3", ValidatorType.STRUCTURAL, always_approve)

        status = enforcer.get_process_status("copies")
        assert not status["is_compliant"]

    def test_critical_needs_five(self, enforcer):
        """CRITICAL processes need 5 validators."""
        enforcer.register_process("crit", "Critical", ProcessCriticality.CRITICAL)
        for i in range(3):
            vtype = [ValidatorType.STRUCTURAL, ValidatorType.BEHAVIORAL, ValidatorType.OPERATIONAL][i]
            enforcer.add_validator("crit", f"v{i}", vtype, always_approve)

        status = enforcer.get_process_status("crit")
        assert not status["is_compliant"]
        assert status["minimum_required"] == 5

    def test_compliance_report(self, enforcer):
        """Full compliance report across all processes."""
        # Good process
        enforcer.register_process("good", "Good")
        for i, vt in enumerate([ValidatorType.STRUCTURAL, ValidatorType.BEHAVIORAL, ValidatorType.OPERATIONAL]):
            enforcer.add_validator("good", f"g{i}", vt, always_approve)

        # Bad process
        enforcer.register_process("bad", "Bad")
        enforcer.add_validator("bad", "b1", ValidatorType.STRUCTURAL, always_approve)

        report = enforcer.check_compliance()
        assert report["total_processes"] == 2
        assert report["compliant"] == 1
        assert report["non_compliant"] == 1
        assert report["compliance_rate"] == 0.5


# ===========================================================================
# TestMajorityVoting
# ===========================================================================


class TestMajorityVoting:
    """Tests for majority vote validation."""

    def _setup_triad(self, enforcer, process_id, validators):
        """Helper to set up a process with given validators."""
        enforcer.register_process(process_id, f"Test {process_id}")
        for vid, vtype, fn in validators:
            enforcer.add_validator(process_id, vid, vtype, fn)

    def test_unanimous_approval(self, enforcer):
        """All validators approve -> approved."""
        self._setup_triad(enforcer, "test", [
            ("v1", ValidatorType.STRUCTURAL, always_approve),
            ("v2", ValidatorType.BEHAVIORAL, always_approve),
            ("v3", ValidatorType.OPERATIONAL, always_approve),
        ])

        result = enforcer.validate("test", {"action": "modify"})
        assert result.approved
        assert result.unanimous
        assert result.votes_for == 3
        assert result.votes_against == 0

    def test_majority_approval(self, enforcer):
        """2 of 3 approve -> approved."""
        self._setup_triad(enforcer, "test", [
            ("v1", ValidatorType.STRUCTURAL, always_approve),
            ("v2", ValidatorType.BEHAVIORAL, always_approve),
            ("v3", ValidatorType.OPERATIONAL, always_reject),
        ])

        result = enforcer.validate("test", {"action": "modify"})
        assert result.approved
        assert not result.unanimous
        assert result.votes_for == 2
        assert result.votes_against == 1
        assert "v3" in result.dissenting_validators

    def test_majority_rejection(self, enforcer):
        """2 of 3 reject -> rejected."""
        self._setup_triad(enforcer, "test", [
            ("v1", ValidatorType.STRUCTURAL, always_approve),
            ("v2", ValidatorType.BEHAVIORAL, always_reject),
            ("v3", ValidatorType.OPERATIONAL, always_reject),
        ])

        result = enforcer.validate("test", {"action": "risky"})
        assert not result.approved
        assert result.votes_for == 1
        assert result.votes_against == 2

    def test_context_based_voting(self, enforcer):
        """Validators can use context to decide."""
        self._setup_triad(enforcer, "test", [
            ("v1", ValidatorType.STRUCTURAL, always_approve),
            ("v2", ValidatorType.BEHAVIORAL, approve_if_safe),
            ("v3", ValidatorType.OPERATIONAL, always_approve),
        ])

        # Safe context
        safe = enforcer.validate("test", {"risk_score": 0.3})
        assert safe.approved
        assert safe.votes_for == 3

        # Risky context
        risky = enforcer.validate("test", {"risk_score": 0.9})
        assert risky.approved  # 2 of 3 still approve
        assert risky.votes_for == 2

    def test_validator_exception_counts_as_rejection(self, enforcer):
        """Validator that throws exception is counted as rejection."""
        def exploding_validator(ctx):
            raise RuntimeError("validator crashed")

        self._setup_triad(enforcer, "test", [
            ("v1", ValidatorType.STRUCTURAL, always_approve),
            ("v2", ValidatorType.BEHAVIORAL, exploding_validator),
            ("v3", ValidatorType.OPERATIONAL, always_approve),
        ])

        result = enforcer.validate("test", {})
        assert result.approved  # 2 approve, 1 crashed = rejection
        assert result.votes_for == 2
        assert result.votes_against == 1

    def test_no_validators_rejects(self, enforcer):
        """Process with no validators always rejects."""
        enforcer.register_process("empty", "Empty")
        result = enforcer.validate("empty", {})
        assert not result.approved
        assert result.total_validators == 0

    def test_unregistered_process_rejects(self, enforcer):
        """Unregistered process always rejects."""
        result = enforcer.validate("ghost", {})
        assert not result.approved

    def test_vote_eventbus(self, enforcer, bus):
        """Vote results published on EventBus."""
        received = []
        bus.register_callback(
            CH_TRIAD_VOTE_COMPLETE,
            lambda ch, msg: received.append(json.loads(msg)),
        )

        enforcer.register_process("test", "Test")
        enforcer.add_validator("test", "v1", ValidatorType.STRUCTURAL, always_approve)
        enforcer.add_validator("test", "v2", ValidatorType.BEHAVIORAL, always_approve)
        enforcer.add_validator("test", "v3", ValidatorType.OPERATIONAL, always_reject)

        enforcer.validate("test", {"action": "test"})

        assert len(received) == 1
        assert received[0]["approved"] is True
        assert received[0]["votes_for"] == 2


# ===========================================================================
# TestViolationDetection
# ===========================================================================


class TestViolationDetection:
    """Tests for detecting and reporting Rule of 3 violations."""

    def test_violation_on_empty_validate(self, enforcer, bus):
        """Violations published when validating process with no validators."""
        violations = []
        bus.register_callback(
            CH_TRIAD_VIOLATION,
            lambda ch, msg: violations.append(json.loads(msg)),
        )

        enforcer.register_process("empty", "Empty")
        enforcer.validate("empty", {})

        assert len(violations) == 1
        assert violations[0]["type"] == "no_validators"

    def test_compliance_check_records_violations(self, enforcer):
        """Compliance check records violations for non-compliant processes."""
        enforcer.register_process("solo", "Solo process")
        enforcer.add_validator("solo", "v1", ValidatorType.STRUCTURAL, always_approve)

        enforcer.check_compliance()

        violations = enforcer.get_violations()
        assert len(violations) >= 1
        assert any(v["process_id"] == "solo" for v in violations)

    def test_health_report(self, enforcer):
        """Health report summarizes triad system state."""
        # One good, one bad
        enforcer.register_process("good", "Good")
        for i, vt in enumerate([ValidatorType.STRUCTURAL, ValidatorType.BEHAVIORAL, ValidatorType.OPERATIONAL]):
            enforcer.add_validator("good", f"g{i}", vt, always_approve)

        enforcer.register_process("bad", "Bad")

        report = enforcer.get_health_report()
        assert not report["healthy"]  # < 80% compliance
        assert len(report["recommendations"]) >= 1


# ===========================================================================
# TestStatistics
# ===========================================================================


class TestStatistics:
    """Tests for triad system statistics."""

    def test_statistics_tracking(self, enforcer):
        """Statistics accurately track state."""
        enforcer.register_process("p1", "Process 1")
        enforcer.register_process("p2", "Process 2")

        for i, vt in enumerate([ValidatorType.STRUCTURAL, ValidatorType.BEHAVIORAL, ValidatorType.OPERATIONAL]):
            enforcer.add_validator("p1", f"v{i}", vt, always_approve)

        stats = enforcer.get_statistics()
        assert stats["total_processes"] == 2
        assert stats["compliant_processes"] == 1
        assert stats["total_validators"] == 3

    def test_vote_statistics(self, enforcer):
        """Vote statistics track approvals and rejections."""
        enforcer.register_process("test", "Test")
        enforcer.add_validator("test", "v1", ValidatorType.STRUCTURAL, always_approve)
        enforcer.add_validator("test", "v2", ValidatorType.BEHAVIORAL, always_approve)
        enforcer.add_validator("test", "v3", ValidatorType.OPERATIONAL, always_approve)

        enforcer.validate("test", {})
        enforcer.validate("test", {})

        status = enforcer.get_process_status("test")
        assert status["total_votes"] == 2
        assert status["unanimous_approvals"] == 2


# ===========================================================================
# TestMaeSystemTriads
# ===========================================================================


class TestMaeSystemTriads:
    """Integration test: Register Mae's actual system triads."""

    def test_mae_full_triad_registration(self, enforcer):
        """Register all major Mae processes with their triads."""

        # --- Threat Detection (PASSES Rule of 3) ---
        enforcer.register_process(
            "threat_detection", "Threat detection and response",
            ProcessCriticality.CRITICAL,
        )
        enforcer.add_validator(
            "threat_detection", "threat_detector",
            ValidatorType.BEHAVIORAL,
            lambda ctx: ctx.get("threat_level", 0) < 5,
            "ThreatDetector - 4 bio strategies",
        )
        enforcer.add_validator(
            "threat_detection", "input_validator",
            ValidatorType.FORMAL,
            lambda ctx: ctx.get("trust_score", 0) > 0.3,
            "InputValidator - zero-trust scoring",
        )
        enforcer.add_validator(
            "threat_detection", "haven",
            ValidatorType.BEHAVIORAL,
            lambda ctx: ctx.get("risk_score", 0) < 0.8,
            "HAVEN - immune system",
        )
        enforcer.add_validator(
            "threat_detection", "somatic_map",
            ValidatorType.STRUCTURAL,
            lambda ctx: ctx.get("blast_radius", 0) < 5,
            "SomaticMap - structural analysis",
        )
        enforcer.add_validator(
            "threat_detection", "causal_engine",
            ValidatorType.CAUSAL,
            lambda ctx: True,
            "CausalEngine - root cause analysis",
        )

        # --- Self-Modification (needs triad!) ---
        enforcer.register_process(
            "self_modification", "Self-improvement and capability changes",
            ProcessCriticality.CRITICAL,
        )
        enforcer.add_validator(
            "self_modification", "somatic_map",
            ValidatorType.STRUCTURAL,
            lambda ctx: ctx.get("blast_risk", 1.0) < 0.7,
            "SomaticMap - blast radius before modification",
        )
        enforcer.add_validator(
            "self_modification", "haven",
            ValidatorType.BEHAVIORAL,
            lambda ctx: ctx.get("policy_safe", False),
            "HAVEN - policy contagion check",
        )
        enforcer.add_validator(
            "self_modification", "capability_discovery",
            ValidatorType.OPERATIONAL,
            lambda ctx: ctx.get("validated", False),
            "CapabilityDiscovery - validation pipeline",
        )
        enforcer.add_validator(
            "self_modification", "causal_engine",
            ValidatorType.CAUSAL,
            lambda ctx: True,
            "CausalEngine - predict modification consequences",
        )
        enforcer.add_validator(
            "self_modification", "triad_enforcer",
            ValidatorType.FORMAL,
            lambda ctx: True,
            "TriadEnforcer - Rule of 3 meta-check",
        )

        # --- Healing (needs triad!) ---
        enforcer.register_process(
            "healing", "Failure detection and recovery",
            ProcessCriticality.PROTECTED,
        )
        enforcer.add_validator(
            "healing", "auto_healer",
            ValidatorType.OPERATIONAL,
            lambda ctx: True,
            "AutoHealer - 3-phase biological recovery",
        )
        enforcer.add_validator(
            "healing", "somatic_map",
            ValidatorType.STRUCTURAL,
            lambda ctx: ctx.get("blast_risk", 1.0) < 0.8,
            "SomaticMap - structural impact of healing actions",
        )
        enforcer.add_validator(
            "healing", "haven",
            ValidatorType.BEHAVIORAL,
            lambda ctx: True,
            "HAVEN - was the failure adversarial?",
        )
        enforcer.add_validator(
            "healing", "nutrient_flow",
            ValidatorType.RESOURCE,
            lambda ctx: True,
            "NutrientFlow - resource availability for recovery",
        )
        enforcer.add_validator(
            "healing", "causal_engine",
            ValidatorType.CAUSAL,
            lambda ctx: True,
            "CausalEngine - root cause before healing",
        )

        # --- Decision Making ---
        enforcer.register_process(
            "decision_making", "Agent action selection",
            ProcessCriticality.STANDARD,
        )
        enforcer.add_validator(
            "decision_making", "decision_router",
            ValidatorType.OPERATIONAL,
            lambda ctx: True,
            "DecisionRouter - 3-tier reflex/habit/prefrontal",
        )
        enforcer.add_validator(
            "decision_making", "collective_dream",
            ValidatorType.CONSENSUS,
            lambda ctx: ctx.get("consensus", 0) > 0.4,
            "CollectiveDream - swarm validation",
        )
        enforcer.add_validator(
            "decision_making", "world_model",
            ValidatorType.PREDICTIVE,
            lambda ctx: True,
            "WorldModel - predict consequences",
        )

        # Verify compliance
        report = enforcer.check_compliance()
        assert report["compliant"] >= 3  # At least threat, healing, decision pass

        # Run validation on self-modification
        result = enforcer.validate("self_modification", {
            "blast_risk": 0.3,
            "policy_safe": True,
            "validated": True,
        })
        assert result.approved
        assert result.votes_for == 5

        # Test rejection cascade
        risky = enforcer.validate("self_modification", {
            "blast_risk": 0.9,  # Too high
            "policy_safe": False,  # Not safe
            "validated": False,  # Not validated
        })
        assert not risky.approved

    def test_five_validator_critical_system(self, enforcer):
        """Critical system with 5 validators (biological standard)."""
        enforcer.register_process(
            "system_health", "Overall Mae health monitoring",
            ProcessCriticality.CRITICAL,
        )

        validators = [
            ("somatic_map", ValidatorType.STRUCTURAL, "Architecture inspection"),
            ("haven", ValidatorType.BEHAVIORAL, "Behavioral anomaly detection"),
            ("auto_healer", ValidatorType.OPERATIONAL, "Runtime recovery"),
            ("causal_engine", ValidatorType.CAUSAL, "Root cause analysis"),
            ("triad_enforcer", ValidatorType.FORMAL, "Rule of 3 meta-check"),
        ]

        for vid, vtype, desc in validators:
            enforcer.add_validator(
                "system_health", vid, vtype, always_approve, desc,
            )

        status = enforcer.get_process_status("system_health")
        assert status["is_compliant"]
        assert status["validator_count"] == 5
        assert len(status["unique_types"]) == 5  # All different types
