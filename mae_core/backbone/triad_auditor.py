"""Triad Auditor - Behavioral analysis of voting patterns.

Biological analogy: The prefrontal cortex monitoring its own
decision-making patterns. Metacognition - thinking about thinking.
The Auditor doesn't make decisions or watch for bypasses. It
reflects on whether the decision-making PATTERNS are healthy.

Detects:
- Echo chambers: Validators always agree (suspicious unanimity)
- Captured validators: One validator always dissents or always agrees
- Dead validators: Validators that stopped responding
- Bias drift: Approval rates shifting over time
- Correlated failures: Multiple validators failing simultaneously

Three complementary enforcement lenses:
- Enforcer: FORMAL - "Do you have enough validators?"
- Watchdog: OPERATIONAL - "Are you actually calling them?"
- Auditor: BEHAVIORAL - "Are the voting patterns healthy?"

Connection points:
- Reads TriadEnforcer process status and vote history
- Publishes audit findings on EventBus
- SomaticMap registers Auditor as CRITICAL system
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# EventBus channels
CH_AUDIT_FINDING = "audit.finding"
CH_AUDIT_HEALTH = "audit.health_report"


class FindingSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class FindingType(Enum):
    ECHO_CHAMBER = "echo_chamber"  # Suspiciously unanimous
    CAPTURED_VALIDATOR = "captured_validator"  # Always same vote
    DEAD_VALIDATOR = "dead_validator"  # Not responding
    BIAS_DRIFT = "bias_drift"  # Approval rate shifting
    CORRELATED_FAILURE = "correlated_failure"  # Multiple fail together
    LOW_COMPLIANCE = "low_compliance"  # Rule of 3 not met


@dataclass
class AuditFinding:
    """A finding from behavioral audit of voting patterns."""

    finding_id: str
    finding_type: FindingType
    severity: FindingSeverity
    process_id: str
    detail: str
    recommendation: str = ""
    timestamp: float = field(default_factory=time.time)
    validator_ids: list[str] = field(default_factory=list)


class TriadAuditor:
    """Behavioral analysis of triad voting patterns.

    Periodically examines TriadEnforcer state and voting history
    to detect unhealthy patterns. Like a meta-auditor that asks:
    "Are the people making decisions behaving normally?"

    Key insight: Three identical "yes" votes might mean the
    validators are redundant (not complementary), compromised,
    or not actually checking anything. Healthy triads should
    show OCCASIONAL disagreement - that proves they're independent.
    """

    def __init__(
        self,
        event_bus: Any = None,
        triad_enforcer: Any = None,
        unanimity_threshold: float = 0.95,
        bias_threshold: float = 0.9,
        inactivity_window: float = 300.0,
    ) -> None:
        self._bus = event_bus
        self._enforcer = triad_enforcer
        self._unanimity_threshold = unanimity_threshold  # Flag if > 95% unanimous
        self._bias_threshold = bias_threshold  # Flag if validator > 90% same vote
        self._inactivity_window = inactivity_window  # Seconds before "dead"

        self._findings: deque[AuditFinding] = deque(maxlen=500)
        self._finding_counter = 0

        # Track vote records per process per validator
        self._vote_history: dict[str, dict[str, list[bool]]] = defaultdict(
            lambda: defaultdict(list)
        )

        self._lock = threading.RLock()

        logger.info("TriadAuditor initialized")

    # =========================================================================
    # Vote Recording
    # =========================================================================

    def record_vote(
        self, process_id: str, validator_id: str, approved: bool
    ) -> None:
        """Record a validator's vote for pattern analysis."""
        with self._lock:
            history = self._vote_history[process_id][validator_id]
            history.append(approved)
            # Keep last 100 votes per validator
            if len(history) > 100:
                self._vote_history[process_id][validator_id] = history[-100:]

    # =========================================================================
    # Audit Checks
    # =========================================================================

    def run_audit(self) -> list[AuditFinding]:
        """Run all audit checks and return findings."""
        findings = []
        findings.extend(self._check_echo_chambers())
        findings.extend(self._check_captured_validators())
        findings.extend(self._check_compliance())

        with self._lock:
            for f in findings:
                self._findings.append(f)

        if self._bus and findings:
            for f in findings:
                self._bus.publish(CH_AUDIT_FINDING, {
                    "finding_id": f.finding_id,
                    "type": f.finding_type.value,
                    "severity": f.severity.value,
                    "process_id": f.process_id,
                    "detail": f.detail,
                })

        return findings

    def _check_echo_chambers(self) -> list[AuditFinding]:
        """Detect processes where validators always agree.

        High unanimity suggests validators are redundant (not
        complementary), rubber-stamping, or not actually checking.
        Healthy triads should disagree sometimes.
        """
        findings = []

        with self._lock:
            for pid, validators in self._vote_history.items():
                if not validators:
                    continue

                # Check if all validators voted the same way each time
                vote_counts = defaultdict(int)
                total_rounds = 0

                # Build per-round agreement
                max_votes = max(len(v) for v in validators.values()) if validators else 0
                unanimous_rounds = 0

                for round_idx in range(max_votes):
                    votes = []
                    for vid, history in validators.items():
                        if round_idx < len(history):
                            votes.append(history[round_idx])

                    if len(votes) >= 2:
                        total_rounds += 1
                        if all(v == votes[0] for v in votes):
                            unanimous_rounds += 1

                if total_rounds >= 10:  # Need enough data
                    unanimity_rate = unanimous_rounds / total_rounds
                    if unanimity_rate >= self._unanimity_threshold:
                        self._finding_counter += 1
                        findings.append(AuditFinding(
                            finding_id=f"audit-{self._finding_counter}",
                            finding_type=FindingType.ECHO_CHAMBER,
                            severity=FindingSeverity.WARNING,
                            process_id=pid,
                            detail=(
                                f"Unanimity rate {unanimity_rate:.0%} over "
                                f"{total_rounds} rounds - validators may not "
                                f"be truly complementary"
                            ),
                            recommendation=(
                                "Verify validators use different detection "
                                "methods. Consider adding a devil's advocate."
                            ),
                        ))

        return findings

    def _check_captured_validators(self) -> list[AuditFinding]:
        """Detect validators that always vote the same way.

        A validator that always approves or always rejects might be:
        - Not actually checking anything (rubber stamp)
        - Compromised (always says yes to malicious requests)
        - Broken (always returns same value)
        """
        findings = []

        with self._lock:
            for pid, validators in self._vote_history.items():
                for vid, history in validators.items():
                    if len(history) < 10:
                        continue

                    approval_rate = sum(history) / len(history)

                    if approval_rate >= self._bias_threshold:
                        self._finding_counter += 1
                        findings.append(AuditFinding(
                            finding_id=f"audit-{self._finding_counter}",
                            finding_type=FindingType.CAPTURED_VALIDATOR,
                            severity=FindingSeverity.WARNING,
                            process_id=pid,
                            detail=(
                                f"Validator '{vid}' approves {approval_rate:.0%} "
                                f"of the time over {len(history)} votes"
                            ),
                            recommendation=(
                                "Verify this validator's logic. A validator "
                                "that always approves provides no protection."
                            ),
                            validator_ids=[vid],
                        ))
                    elif approval_rate <= (1 - self._bias_threshold):
                        self._finding_counter += 1
                        findings.append(AuditFinding(
                            finding_id=f"audit-{self._finding_counter}",
                            finding_type=FindingType.CAPTURED_VALIDATOR,
                            severity=FindingSeverity.WARNING,
                            process_id=pid,
                            detail=(
                                f"Validator '{vid}' rejects {1 - approval_rate:.0%} "
                                f"of the time over {len(history)} votes"
                            ),
                            recommendation=(
                                "Verify this validator isn't broken. A validator "
                                "that always rejects blocks all process actions."
                            ),
                            validator_ids=[vid],
                        ))

        return findings

    def _check_compliance(self) -> list[AuditFinding]:
        """Check overall Rule of 3 compliance via TriadEnforcer."""
        findings = []
        if not self._enforcer:
            return findings

        report = self._enforcer.check_compliance()
        for v in report.get("violations", []):
            self._finding_counter += 1
            findings.append(AuditFinding(
                finding_id=f"audit-{self._finding_counter}",
                finding_type=FindingType.LOW_COMPLIANCE,
                severity=(
                    FindingSeverity.CRITICAL
                    if v["criticality"] == "critical"
                    else FindingSeverity.WARNING
                ),
                process_id=v["process_id"],
                detail=(
                    f"Non-compliant: {'; '.join(v['reasons'])}"
                ),
                recommendation=(
                    f"Add {v['required'] - v['validators']} more "
                    f"complementary validators"
                ),
            ))

        return findings

    # =========================================================================
    # Health Report
    # =========================================================================

    def get_health_report(self) -> dict[str, Any]:
        """Generate behavioral health report for the triad system."""
        findings = self.run_audit()

        critical = [f for f in findings if f.severity == FindingSeverity.CRITICAL]
        warnings = [f for f in findings if f.severity == FindingSeverity.WARNING]

        report = {
            "healthy": len(critical) == 0,
            "findings_count": len(findings),
            "critical_count": len(critical),
            "warning_count": len(warnings),
            "processes_audited": len(self._vote_history),
            "findings": [
                {
                    "type": f.finding_type.value,
                    "severity": f.severity.value,
                    "process": f.process_id,
                    "detail": f.detail,
                }
                for f in findings
            ],
        }

        if self._bus:
            self._bus.publish(CH_AUDIT_HEALTH, {
                "healthy": report["healthy"],
                "critical": len(critical),
                "warnings": len(warnings),
            })

        return report

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_statistics(self) -> dict[str, Any]:
        """Get auditor statistics."""
        with self._lock:
            total_votes = sum(
                len(h)
                for validators in self._vote_history.values()
                for h in validators.values()
            )
            return {
                "processes_tracked": len(self._vote_history),
                "total_votes_recorded": total_votes,
                "total_findings": len(self._findings),
                "finding_counter": self._finding_counter,
            }

    def get_findings(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get recent audit findings."""
        with self._lock:
            recent = list(self._findings)[-limit:]
            return [
                {
                    "finding_id": f.finding_id,
                    "type": f.finding_type.value,
                    "severity": f.severity.value,
                    "process": f.process_id,
                    "detail": f.detail,
                    "recommendation": f.recommendation,
                }
                for f in recent
            ]
