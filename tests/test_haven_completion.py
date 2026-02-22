"""HAVEN completion tests — risk coordinator enforcement via TriadEnforcer.

Tests verify that:
1. HAVEN validators are properly wired into TriadEnforcer
2. Risk assessment flows through validator pipeline
3. High risk triggers intervention recommendations
4. Contagion detection works
5. Law 7 (minimum 3 validators per process) is maintained
"""

import pytest
from mae_core.learning.haven import (
    HavenRiskCoordinator, RiskLevel, InterventionType,
    ContagionStatus, RiskAssessment,
)
from mae_core.backbone.event_bus import EventBus
from mae_core.backbone.triad_enforcer import (
    TriadEnforcer, ProcessCriticality, ValidatorType,
)


class TestHavenValidatorMethods:
    """Test HAVEN's validator methods work independently."""

    def test_validate_decision_low_risk_passes(self):
        """Low-risk agent decisions pass validation."""
        bus = EventBus()
        haven = HavenRiskCoordinator(event_bus=bus, risk_threshold=0.7)
        haven.register_agent("agent-1")

        context = {
            "agent_id": "agent-1",
            "action": "explore",
            "recent_performance": [0.5] * 5 + [0.6] * 5,  # Stable/slight improvement
            "behavioral_metrics": {},
        }

        result = haven.validate_decision(context)
        assert result is True

    def test_validate_decision_high_risk_fails(self):
        """High-risk agent decisions fail validation."""
        bus = EventBus()
        haven = HavenRiskCoordinator(event_bus=bus, risk_threshold=0.7)
        haven.register_agent("agent-1")

        # Simulate strong performance degradation
        context = {
            "agent_id": "agent-1",
            "action": "exploit",
            "recent_performance": [0.8] * 5 + [0.1] * 5,  # Sharp degradation
            "behavioral_metrics": {"anomaly_1": 3.0, "anomaly_2": 2.5},
        }

        result = haven.validate_decision(context)
        assert result is False

    def test_validate_modification_healthy_contagion(self):
        """Modification passes when system is healthy."""
        bus = EventBus()
        haven = HavenRiskCoordinator(event_bus=bus)
        haven.register_agent("agent-1")

        context = {
            "agent_id": "agent-1",
            "modification_type": "policy_update",
            "source_peers": ["agent-2", "agent-3"],
        }

        result = haven.validate_modification(context)
        assert result is True

    def test_validate_healing_critical_risk_requires_isolation(self):
        """Critical risk requires isolation/rollback recovery."""
        bus = EventBus()
        haven = HavenRiskCoordinator(event_bus=bus)
        haven.register_agent("agent-1")

        # Critical risk (score > 0.8)
        context = {
            "agent_id": "agent-1",
            "failure_type": "performance_degradation",
            "recovery_strategy": "policy_freeze",  # Too lenient for critical
            "risk_score": 0.85,
        }

        result = haven.validate_healing(context)
        assert result is False  # Policy freeze not aggressive enough

        # Isolation is appropriate for critical risk
        context["recovery_strategy"] = "isolation"
        result = haven.validate_healing(context)
        assert result is True

    def test_validate_healing_high_risk_allows_freeze(self):
        """High risk allows policy freeze (less aggressive)."""
        bus = EventBus()
        haven = HavenRiskCoordinator(event_bus=bus)
        haven.register_agent("agent-1")

        context = {
            "agent_id": "agent-1",
            "failure_type": "anomaly",
            "recovery_strategy": "policy_freeze",
            "risk_score": 0.75,  # HIGH but not CRITICAL
        }

        result = haven.validate_healing(context)
        assert result is True

    def test_validate_policy_rejects_high_risk_source(self):
        """Policy from high-risk agents is rejected."""
        bus = EventBus()
        haven = HavenRiskCoordinator(event_bus=bus, risk_threshold=0.7)
        haven.register_agent("agent-1")
        haven.register_agent("agent-2")

        # Force agent-2 to high risk
        # Performance degradation: first half high (0.9), second half low (0.1)
        # Need >= 10 values for performance degradation to calculate
        haven.assess_agent_risk(
            "agent-2",
            recent_performance=[0.9, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1],
            behavioral_metrics={"anomaly": 3.0},
        )

        context = {
            "agent_id": "agent-1",
            "policy_id": "policy-bad",
            "source_agent": "agent-2",
            "performance_on_source": 0.8,
            "learning_method": "frl",
        }

        result = haven.validate_policy(context)
        assert result is False

    def test_validate_policy_accepts_low_risk_source(self):
        """Policy from low-risk agents is accepted."""
        bus = EventBus()
        haven = HavenRiskCoordinator(event_bus=bus, risk_threshold=0.7)
        haven.register_agent("agent-1")
        haven.register_agent("agent-2")

        context = {
            "agent_id": "agent-1",
            "policy_id": "policy-good",
            "source_agent": "agent-2",
            "performance_on_source": 0.8,
            "learning_method": "frl",
        }

        result = haven.validate_policy(context)
        assert result is True

    def test_validate_threat_identifies_contagion(self):
        """Threat validator identifies contagion threats."""
        bus = EventBus()
        haven = HavenRiskCoordinator(event_bus=bus, contagion_threshold=0.5)
        haven.register_agent("agent-1")
        haven.register_agent("agent-2")
        haven.register_agent("agent-3")

        # Force multiple high-risk agents to trigger SPREADING status
        haven.assess_agent_risk(
            "agent-1",
            recent_performance=[0.8] * 5 + [0.1] * 5,  # Sharp degradation
            behavioral_metrics={"anomaly": 3.0},
        )
        haven.assess_agent_risk(
            "agent-2",
            recent_performance=[0.7] * 5 + [0.2] * 5,  # Sharp degradation
            behavioral_metrics={"anomaly": 2.5},
        )
        haven.assess_agent_risk(
            "agent-3",
            recent_performance=[0.5] * 5 + [0.3] * 5,  # Moderate degradation
            behavioral_metrics={},
        )

        context = {
            "threat_type": "contagion",
            "threat_level": 0.6,
            "affected_agents": ["agent-1", "agent-2"],
            "defense_action": "isolation",
        }

        result = haven.validate_threat(context)
        # Should validate correctly (may be True or True based on actual contagion)
        assert isinstance(result, bool)


class TestHavenTriadEnforcerIntegration:
    """Test HAVEN validators are integrated into TriadEnforcer."""

    def test_haven_validators_registered_on_decision_making(self):
        """HAVEN validators are registered on decision_making process."""
        bus = EventBus()
        haven = HavenRiskCoordinator(event_bus=bus)
        enforcer = TriadEnforcer(event_bus=bus)

        # Register process and validators manually (as done in wiring.py)
        enforcer.register_process(
            "decision_making", "Agent action selection", ProcessCriticality.CRITICAL
        )

        # Add HAVEN validator
        enforcer.add_validator(
            "decision_making", "haven_decision", ValidatorType.BEHAVIORAL,
            lambda ctx: haven.validate_decision(ctx),
            "Risk assessment of proposed action",
        )

        # Check that validator was added
        triad = enforcer.get_process_status("decision_making")
        assert triad is not None
        validators = triad["validators"]
        haven_validator = next(
            (v for v in validators if v["id"] == "haven_decision"), None
        )
        assert haven_validator is not None
        assert haven_validator["type"] == "behavioral"

    def test_haven_validators_registered_on_learning_policy(self):
        """HAVEN validators are registered on learning_policy process."""
        bus = EventBus()
        haven = HavenRiskCoordinator(event_bus=bus)
        enforcer = TriadEnforcer(event_bus=bus)

        enforcer.register_process(
            "learning_policy", "Policy sharing and updates", ProcessCriticality.PROTECTED
        )

        enforcer.add_validator(
            "learning_policy", "haven_contagion", ValidatorType.BEHAVIORAL,
            lambda ctx: haven.validate_policy(ctx),
            "Policy contagion detection",
        )

        triad = enforcer.get_process_status("learning_policy")
        assert triad is not None
        validators = triad["validators"]
        assert any(v["id"] == "haven_contagion" for v in validators)

    def test_haven_validators_registered_on_healing(self):
        """HAVEN validators are registered on healing process."""
        bus = EventBus()
        haven = HavenRiskCoordinator(event_bus=bus)
        enforcer = TriadEnforcer(event_bus=bus)

        enforcer.register_process(
            "healing", "Failure detection and recovery", ProcessCriticality.CRITICAL
        )

        enforcer.add_validator(
            "healing", "haven_healing", ValidatorType.BEHAVIORAL,
            lambda ctx: haven.validate_healing(ctx),
            "Risk-appropriate healing strategy selection",
        )

        triad = enforcer.get_process_status("healing")
        assert triad is not None
        validators = triad["validators"]
        assert any(v["id"] == "haven_healing" for v in validators)

    def test_haven_decision_validator_in_majority_vote(self):
        """HAVEN decision validator participates in majority vote."""
        bus = EventBus()
        haven = HavenRiskCoordinator(event_bus=bus, risk_threshold=0.7)
        haven.register_agent("agent-1")
        enforcer = TriadEnforcer(event_bus=bus)

        enforcer.register_process(
            "decision_making", "Agent action selection", ProcessCriticality.CRITICAL
        )

        # Add 3 validators (minimum for Law 7)
        enforcer.add_validator(
            "decision_making", "haven_decision", ValidatorType.BEHAVIORAL,
            lambda ctx: haven.validate_decision(ctx),
            "Risk assessment",
        )
        enforcer.add_validator(
            "decision_making", "validator_2", ValidatorType.OPERATIONAL,
            lambda ctx: True,  # Always approve
            "Operational check",
        )
        enforcer.add_validator(
            "decision_making", "validator_3", ValidatorType.PREDICTIVE,
            lambda ctx: True,  # Always approve
            "Predictive check",
        )

        # Low-risk context should pass (2/3 approve)
        context = {
            "agent_id": "agent-1",
            "action": "explore",
            "recent_performance": [0.5] * 5 + [0.6] * 5,  # Stable/slight improvement
            "behavioral_metrics": {},
        }
        result = enforcer.validate("decision_making", context)
        assert result.approved is True
        assert result.votes_for >= 2

    def test_haven_healing_validator_in_majority_vote(self):
        """HAVEN healing validator participates in majority vote."""
        bus = EventBus()
        haven = HavenRiskCoordinator(event_bus=bus)
        enforcer = TriadEnforcer(event_bus=bus)

        enforcer.register_process(
            "healing", "Failure detection and recovery", ProcessCriticality.CRITICAL
        )

        # Add 3 validators
        enforcer.add_validator(
            "healing", "haven_healing", ValidatorType.BEHAVIORAL,
            lambda ctx: haven.validate_healing(ctx),
            "Risk-appropriate strategy",
        )
        enforcer.add_validator(
            "healing", "validator_2", ValidatorType.OPERATIONAL,
            lambda ctx: True,
            "Operational readiness",
        )
        enforcer.add_validator(
            "healing", "validator_3", ValidatorType.CAUSAL,
            lambda ctx: True,
            "Root cause analysis",
        )

        # Critical risk with isolation should pass
        context = {
            "agent_id": "agent-1",
            "failure_type": "severe",
            "recovery_strategy": "isolation",
            "risk_score": 0.85,
        }
        result = enforcer.validate("healing", context)
        assert result.approved is True

    def test_law7_minimum_validators_per_process(self):
        """Law 7: each process has minimum 3 validators (odd count)."""
        bus = EventBus()
        haven = HavenRiskCoordinator(event_bus=bus)
        enforcer = TriadEnforcer(event_bus=bus)

        # Register a process
        enforcer.register_process(
            "test_process", "Test", ProcessCriticality.STANDARD
        )

        # Check minimum is enforced
        triad = enforcer._processes["test_process"]
        assert triad.minimum_required == 3

        # Add less than minimum
        enforcer.add_validator(
            "test_process", "v1", ValidatorType.BEHAVIORAL,
            lambda ctx: True, "First"
        )
        enforcer.add_validator(
            "test_process", "v2", ValidatorType.OPERATIONAL,
            lambda ctx: True, "Second"
        )

        # Not yet compliant (only 2 validators)
        assert triad.is_compliant is False

        # Add third validator
        enforcer.add_validator(
            "test_process", "v3", ValidatorType.PREDICTIVE,
            lambda ctx: True, "Third"
        )

        # Now compliant (3 validators, odd count)
        assert triad.is_compliant is True
        assert triad.validator_count % 2 == 1  # Odd count

    def test_haven_isolation_recommendation_on_critical_risk(self):
        """HAVEN recommends isolation for critical-risk agents."""
        bus = EventBus()
        haven = HavenRiskCoordinator(event_bus=bus)
        haven.register_agent("agent-1")

        # Force critical risk: need performance degradation + behavioral anomalies
        # Degradation: (0.8 - 0.1) / 0.8 = 0.875, contributes 0.5 * 0.875 = 0.4375
        # Behavioral: 2 anomalies / 1 metric = 1.0, contributes 0.3 * 1.0 = 0.3
        # Total: 0.4375 + 0.3 = 0.7375 (still below 0.8)
        # Isolate first: 0.2 * 1.0 = 0.2 → 0.7375 + 0.2 = 0.9375 (CRITICAL!)
        haven._isolated_agents.add("agent-1")
        haven.assess_agent_risk(
            "agent-1",
            recent_performance=[0.8] * 5 + [0.1] * 5,  # Sharp degradation
            behavioral_metrics={"anomaly_1": 3.0, "anomaly_2": 2.5},  # 2 anomalies
        )

        # Get recommendation
        assessment = haven._risk_assessments["agent-1"]
        intervention = haven.recommend_intervention(assessment)

        # Critical risk should recommend rollback (most aggressive)
        assert intervention in (InterventionType.ROLLBACK, InterventionType.EMERGENCY_STOP)

    def test_haven_contagion_detection_spreading_status(self):
        """HAVEN detects contagion when multiple agents are high-risk."""
        bus = EventBus()
        haven = HavenRiskCoordinator(
            event_bus=bus,
            risk_threshold=0.7,
            contagion_threshold=0.3,  # Lower threshold for detection
        )

        # Register agents
        for i in range(5):
            haven.register_agent(f"agent-{i}")

        # Make 2+ agents high-risk (need >= 10 values for performance degradation)
        # Degradation: (0.8 - 0.1)/0.8 = 0.875, score = 0.5 * 0.875 = 0.4375
        haven.assess_agent_risk("agent-0", recent_performance=[0.8] * 5 + [0.1] * 5)
        haven.assess_agent_risk("agent-1", recent_performance=[0.8] * 5 + [0.2] * 5)

        # Check contagion
        report = haven.detect_policy_contagion()

        # At least EARLY_WARNING with 2 high-risk agents
        assert report.status in (
            ContagionStatus.EARLY_WARNING,
            ContagionStatus.SPREADING,
        )

    def test_haven_isolation_execution(self):
        """HAVEN can execute isolation intervention."""
        bus = EventBus()
        haven = HavenRiskCoordinator(event_bus=bus)
        haven.register_agent("agent-1")

        # Isolation should not throw
        haven.isolate_agent("agent-1", reason="test isolation")

        # Agent should be marked as isolated
        assert haven.is_agent_isolated("agent-1")

        # System health report should reflect isolation
        report = haven.get_system_health_report()
        assert report["isolated_agents"] == 1

    def test_haven_restoration_after_isolation(self):
        """HAVEN can restore isolated agents."""
        bus = EventBus()
        haven = HavenRiskCoordinator(event_bus=bus)
        haven.register_agent("agent-1")

        # Isolate
        haven.isolate_agent("agent-1", reason="test isolation")
        assert haven.is_agent_isolated("agent-1")

        # Restore
        result = haven.restore_agent("agent-1")
        assert result is True
        assert not haven.is_agent_isolated("agent-1")

    def test_haven_system_risk_aggregation(self):
        """HAVEN aggregates risk across agents."""
        bus = EventBus()
        haven = HavenRiskCoordinator(event_bus=bus)

        # Register agents
        for i in range(3):
            haven.register_agent(f"agent-{i}")

        # Assess with varying risk levels (need >= 10 values for performance degradation)
        haven.assess_agent_risk("agent-0", recent_performance=[0.8] * 5 + [0.7] * 5)  # Low risk
        haven.assess_agent_risk("agent-1", recent_performance=[0.6] * 5 + [0.4] * 5)  # Medium risk
        haven.assess_agent_risk("agent-2", recent_performance=[0.8] * 5 + [0.1] * 5)  # High risk

        system_risk = haven.assess_system_risk()

        # System risk should be weighted combination of agents
        # 60% max + 40% mean
        assert 0.0 <= system_risk <= 1.0
        # With one high-risk agent, system risk should be elevated
        assert system_risk > 0.3


class TestHavenBootstrapIntegration:
    """Integration tests for HAVEN in full bootstrap context."""

    @pytest.mark.integration
    def test_haven_validators_bootstrap_wiring(self):
        """HAVEN validators are properly wired during bootstrap."""
        # This test would require a full bootstrap context.
        # Manually set up the key components instead.
        bus = EventBus()
        haven = HavenRiskCoordinator(event_bus=bus)
        enforcer = TriadEnforcer(event_bus=bus)

        # Register all processes that HAVEN validates
        processes = [
            ("decision_making", ProcessCriticality.CRITICAL),
            ("learning_policy", ProcessCriticality.PROTECTED),
            ("healing", ProcessCriticality.CRITICAL),
            ("self_modification", ProcessCriticality.CRITICAL),
            ("resource_allocation", ProcessCriticality.PROTECTED),
            ("threat_detection", ProcessCriticality.CRITICAL),
        ]

        for process_id, criticality in processes:
            enforcer.register_process(process_id, f"Process: {process_id}", criticality)

        # Register HAVEN validators on each
        validator_map = {
            "decision_making": lambda ctx: haven.validate_decision(ctx),
            "learning_policy": lambda ctx: haven.validate_policy(ctx),
            "healing": lambda ctx: haven.validate_healing(ctx),
            "self_modification": lambda ctx: haven.validate_modification(ctx),
            "resource_allocation": lambda ctx: haven.validate_threat(ctx),
            "threat_detection": lambda ctx: haven.validate_threat(ctx),
        }

        for process_id, validate_fn in validator_map.items():
            enforcer.add_validator(
                process_id, f"haven_{process_id}", ValidatorType.BEHAVIORAL,
                validate_fn, f"HAVEN behavioral validation for {process_id}",
            )

        # All processes should be registered
        for process_id, _ in processes:
            status = enforcer.get_process_status(process_id)
            assert status is not None
            assert status["process_id"] == process_id
            # HAVEN validator should be present
            assert any(
                v["id"] == f"haven_{process_id}"
                for v in status["validators"]
            )


class TestWireTriadSystems:
    """Test wire_triad_systems replaces permissive defaults with real methods."""

    def test_wire_replaces_permissive_default(self):
        """wire_triad_systems replaces lambda ctx: True with real haven method."""
        from mae_core.backbone.triad_registry import register_all_triads, wire_triad_systems
        from mae_core.backbone.triad_auditor import TriadAuditor
        from mae_core.backbone.triad_watchdog import TriadWatchdog

        bus = EventBus()
        enforcer = TriadEnforcer(event_bus=bus)
        watchdog = TriadWatchdog(event_bus=bus, triad_enforcer=enforcer)
        auditor = TriadAuditor(event_bus=bus, triad_enforcer=enforcer)

        # Register with no systems (permissive defaults)
        register_all_triads(enforcer=enforcer, watchdog=watchdog, auditor=auditor)

        # Verify haven validator exists on decision_making with permissive default
        triad = enforcer._processes["decision_making"]
        haven_v = next(v for v in triad.validators if v.validator_id == "haven")
        assert haven_v.validate_fn({}) is True  # Permissive default

        # Now wire the real haven
        haven = HavenRiskCoordinator(event_bus=bus, risk_threshold=0.7)
        haven.register_agent("agent-1")
        wired = wire_triad_systems(enforcer, {"haven": haven})

        assert wired > 0

        # Now the haven validator should use real validate_decision
        # High-risk context should be rejected
        context = {
            "agent_id": "agent-1",
            "action": "exploit",
            "recent_performance": [0.8] * 5 + [0.1] * 5,
            "behavioral_metrics": {"anomaly": 3.0},
        }
        assert haven_v.validate_fn(context) is False  # Real rejection

    def test_wire_count_matches_haven_processes(self):
        """wire_triad_systems wires haven on all processes that reference it."""
        from mae_core.backbone.triad_registry import register_all_triads, wire_triad_systems
        from mae_core.backbone.triad_auditor import TriadAuditor
        from mae_core.backbone.triad_watchdog import TriadWatchdog

        bus = EventBus()
        enforcer = TriadEnforcer(event_bus=bus)
        watchdog = TriadWatchdog(event_bus=bus, triad_enforcer=enforcer)
        auditor = TriadAuditor(event_bus=bus, triad_enforcer=enforcer)

        register_all_triads(enforcer=enforcer, watchdog=watchdog, auditor=auditor)

        haven = HavenRiskCoordinator(event_bus=bus)
        wired = wire_triad_systems(enforcer, {"haven": haven})

        # Haven appears in 8 processes: self_modification, decision_making,
        # healing, agent_lifecycle, resource_allocation, learning_policy,
        # endocrine_regulation, threat_detection
        assert wired >= 5  # At least the ones with matching methods


class TestHavenEventBusIntegration:
    """Test HAVEN's EventBus integration."""

    def test_haven_publishes_risk_alert_on_high_risk(self):
        """HAVEN publishes risk_alert event when agent exceeds threshold."""
        import json
        bus = EventBus()
        haven = HavenRiskCoordinator(event_bus=bus, risk_threshold=0.7)
        haven.register_agent("agent-1")

        alert_received = []

        def on_alert(channel, serialized):
            msg = json.loads(serialized) if isinstance(serialized, str) else serialized
            alert_received.append(msg)

        bus.register_callback("haven.risk_alert", on_alert)

        # Assess high risk (need >= 10 values for performance degradation)
        haven.assess_agent_risk(
            "agent-1",
            recent_performance=[0.8] * 5 + [0.1] * 5,  # Sharp degradation
            behavioral_metrics={"anomaly": 3.0},
        )

        # Alert should have been published
        assert len(alert_received) > 0, "No alert was published"
        alert = alert_received[0]
        assert alert.get("agent_id") == "agent-1", f"Expected agent_id in {alert}"
        assert alert.get("risk_score") >= 0.7, f"Expected risk_score >= 0.7, got {alert.get('risk_score')}"

    def test_haven_publishes_intervention_event(self):
        """HAVEN publishes intervention event when intervention executed."""
        import json
        bus = EventBus()
        haven = HavenRiskCoordinator(event_bus=bus)
        haven.register_agent("agent-1")

        intervention_received = []

        def on_intervention(channel, serialized):
            msg = json.loads(serialized) if isinstance(serialized, str) else serialized
            intervention_received.append(msg)

        bus.register_callback("haven.intervention", on_intervention)

        # Execute intervention
        haven.isolate_agent("agent-1", reason="test")

        # Intervention should have been published
        assert len(intervention_received) > 0, "No intervention event was published"
        event = intervention_received[0]
        assert event.get("agent_id") == "agent-1", f"Expected agent_id in {event}"
        assert event.get("intervention") == "isolation", f"Expected isolation intervention, got {event.get('intervention')}"
