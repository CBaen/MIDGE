"""Triad Enforcer - Rule of 3 structural guarantee for all Mae processes.

Biological analogy: Regulatory gene networks that enforce minimum
redundancy. In biology, critical pathways ALWAYS have multiple
independent regulators. The p53 gene (the "guardian of the genome")
is monitored by MDM2, MDM4, ARF, and ATM - four different systems
using different detection methods. If any ONE fails, the others
catch it.

The Rule of 3:
- Every major process needs AT LEAST 3 complementary monitors
- "Complementary" = different approaches, not copies
- Always odd numbers for consensus (3, 5, 7) to prevent deadlock
- Critical systems need 5+ (matching biological standard)
- n >= 3f+1 for Byzantine fault tolerance (f = tolerated failures)

This module:
1. Registers processes and their validator triads
2. Enforces minimum validator counts at registration time
3. Provides majority-vote validation for any process action
4. Detects and reports triad violations (processes with < 3 validators)
5. Publishes health status of the triad system itself

Connection points:
- EventBus publishes triad health and violation events
- SomaticMap registers TriadEnforcer as a CRITICAL system
- All major Mae subsystems register their validator triads here
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# EventBus channels
CH_TRIAD_VIOLATION = "triad.violation"
CH_TRIAD_REGISTERED = "triad.process_registered"
CH_TRIAD_VOTE_COMPLETE = "triad.vote_complete"
CH_TRIAD_HEALTH = "triad.health_report"


class ProcessCriticality(Enum):
    """How critical a process is - determines minimum validator count."""

    PERIPHERAL = "peripheral"  # 3 minimum (non-critical)
    STANDARD = "standard"  # 3 minimum (normal operations)
    PROTECTED = "protected"  # 5 minimum (important systems)
    CRITICAL = "critical"  # 5 minimum (blood-brain barrier level)


# Minimum validators by criticality
MINIMUM_VALIDATORS = {
    ProcessCriticality.PERIPHERAL: 3,
    ProcessCriticality.STANDARD: 3,
    ProcessCriticality.PROTECTED: 5,
    ProcessCriticality.CRITICAL: 5,
}


class ValidatorType(Enum):
    """The approach/lens a validator uses - ensures complementary coverage."""

    STRUCTURAL = "structural"  # Architecture/dependency analysis (SomaticMap)
    BEHAVIORAL = "behavioral"  # Trust/anomaly detection (HAVEN)
    OPERATIONAL = "operational"  # Runtime health/recovery (AutoHealer)
    PREDICTIVE = "predictive"  # Prediction accuracy (WorldModel/ValidatedImagination)
    CONSENSUS = "consensus"  # Collective agreement (Quorum/Dream)
    CAUSAL = "causal"  # Cause-effect reasoning (CausalEngine)
    TEMPORAL = "temporal"  # Time-based patterns (TemporalMemory)
    RESOURCE = "resource"  # Resource availability (NutrientFlow/Endocrine)
    FORMAL = "formal"  # Contract/invariant checking


@dataclass
class Validator:
    """A single validator in a process triad.

    Each validator uses a different approach to validate
    the same process, ensuring independent failure modes.
    """

    validator_id: str
    validator_type: ValidatorType
    validate_fn: Callable[[dict[str, Any]], bool]
    description: str = ""
    last_invoked: float = 0.0
    total_invocations: int = 0
    approvals: int = 0
    rejections: int = 0


@dataclass
class ProcessTriad:
    """A registered process with its validator triad.

    The triad must have:
    - At least 3 validators (or 5 for critical processes)
    - Always an odd number (to prevent deadlock)
    - At least 2 different ValidatorTypes (complementary, not copies)
    """

    process_id: str
    description: str
    criticality: ProcessCriticality
    validators: list[Validator] = field(default_factory=list)
    registered_at: float = field(default_factory=time.time)

    # Statistics
    total_votes: int = 0
    unanimous_approvals: int = 0
    majority_approvals: int = 0
    rejections: int = 0

    @property
    def validator_count(self) -> int:
        return len(self.validators)

    @property
    def minimum_required(self) -> int:
        return MINIMUM_VALIDATORS[self.criticality]

    @property
    def is_compliant(self) -> bool:
        """Check if this process meets Rule of 3 requirements."""
        if self.validator_count < self.minimum_required:
            return False
        if self.validator_count % 2 == 0:
            return False  # Must be odd
        types = {v.validator_type for v in self.validators}
        if len(types) < 2:
            return False  # Must be complementary
        return True

    @property
    def unique_types(self) -> set[ValidatorType]:
        return {v.validator_type for v in self.validators}


@dataclass
class VoteResult:
    """Result of a majority vote across validators."""

    process_id: str
    approved: bool
    votes_for: int
    votes_against: int
    total_validators: int
    unanimous: bool
    dissenting_validators: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0


class TriadEnforcer:
    """Enforces Rule of 3 across all Mae processes.

    Every major process registers here with its validator triad.
    The enforcer:
    1. Validates that enough complementary validators exist
    2. Provides majority-vote validation for process actions
    3. Reports violations when processes lack proper triads
    4. Monitors triad health over time

    Like a regulatory body ensuring every critical department
    has independent oversight from multiple angles.
    """

    def __init__(
        self,
        event_bus: Any = None,
        strict_mode: bool = False,
    ) -> None:
        self._bus = event_bus
        self._strict = strict_mode  # If True, reject registration of non-compliant processes

        self._processes: dict[str, ProcessTriad] = {}
        self._violations: list[dict[str, Any]] = []
        self._lock = threading.RLock()

        logger.info("TriadEnforcer initialized (strict=%s)", strict_mode)

    # =========================================================================
    # Process Registration
    # =========================================================================

    def register_process(
        self,
        process_id: str,
        description: str,
        criticality: ProcessCriticality = ProcessCriticality.STANDARD,
    ) -> ProcessTriad:
        """Register a major process that needs triad oversight."""
        with self._lock:
            triad = ProcessTriad(
                process_id=process_id,
                description=description,
                criticality=criticality,
            )
            self._processes[process_id] = triad

            if self._bus:
                self._bus.publish(CH_TRIAD_REGISTERED, {
                    "process_id": process_id,
                    "criticality": criticality.value,
                    "minimum_validators": triad.minimum_required,
                })

            return triad

    def add_validator(
        self,
        process_id: str,
        validator_id: str,
        validator_type: ValidatorType,
        validate_fn: Callable[[dict[str, Any]], bool],
        description: str = "",
    ) -> bool:
        """Add a validator to a process triad.

        Returns False if adding would create an even count
        (caller should add another to maintain odd numbers).
        """
        with self._lock:
            triad = self._processes.get(process_id)
            if not triad:
                logger.warning("Process %s not registered", process_id)
                return False

            # Check for duplicate
            existing_ids = {v.validator_id for v in triad.validators}
            if validator_id in existing_ids:
                logger.warning("Validator %s already registered", validator_id)
                return False

            validator = Validator(
                validator_id=validator_id,
                validator_type=validator_type,
                validate_fn=validate_fn,
                description=description,
            )
            triad.validators.append(validator)

            # Warn if even count (deadlock risk)
            if triad.validator_count % 2 == 0 and triad.validator_count > 0:
                logger.warning(
                    "Process '%s' has %d validators (even count - add one more for odd)",
                    process_id,
                    triad.validator_count,
                )

            return True

    # =========================================================================
    # Majority Vote Validation
    # =========================================================================

    def validate(
        self, process_id: str, context: dict[str, Any]
    ) -> VoteResult:
        """Run majority vote across all validators for a process action.

        Each validator independently evaluates the context.
        Majority rules. Ties (even count) reject by default (safe).
        """
        start = time.time()

        with self._lock:
            triad = self._processes.get(process_id)
            if not triad:
                return VoteResult(
                    process_id=process_id,
                    approved=False,
                    votes_for=0,
                    votes_against=0,
                    total_validators=0,
                    unanimous=False,
                    context=context,
                )

            if not triad.validators:
                self._record_violation(
                    process_id, "no_validators",
                    f"Process '{process_id}' has no validators",
                )
                return VoteResult(
                    process_id=process_id,
                    approved=False,
                    votes_for=0,
                    votes_against=0,
                    total_validators=0,
                    unanimous=False,
                    context=context,
                )

        # Collect votes (outside lock to avoid holding it during callbacks)
        votes_for = 0
        votes_against = 0
        dissenters = []

        for validator in triad.validators:
            try:
                approved = validator.validate_fn(context)
                validator.last_invoked = time.time()
                validator.total_invocations += 1
                if approved:
                    votes_for += 1
                    validator.approvals += 1
                else:
                    votes_against += 1
                    validator.rejections += 1
                    dissenters.append(validator.validator_id)
            except Exception as e:
                logger.error(
                    "Validator %s failed: %s", validator.validator_id, e
                )
                votes_against += 1
                dissenters.append(validator.validator_id)

        # Majority vote (strict majority required)
        total = votes_for + votes_against
        approved = votes_for > votes_against
        unanimous = votes_against == 0

        # Update stats
        with self._lock:
            triad.total_votes += 1
            if unanimous:
                triad.unanimous_approvals += 1
            elif approved:
                triad.majority_approvals += 1
            else:
                triad.rejections += 1

        duration = time.time() - start
        result = VoteResult(
            process_id=process_id,
            approved=approved,
            votes_for=votes_for,
            votes_against=votes_against,
            total_validators=total,
            unanimous=unanimous,
            dissenting_validators=dissenters,
            context=context,
            duration=duration,
        )

        if self._bus:
            self._bus.publish(CH_TRIAD_VOTE_COMPLETE, {
                "process_id": process_id,
                "approved": approved,
                "votes_for": votes_for,
                "votes_against": votes_against,
                "unanimous": unanimous,
            })

        return result

    # =========================================================================
    # Compliance Checking
    # =========================================================================

    def check_compliance(self) -> dict[str, Any]:
        """Check all registered processes for Rule of 3 compliance.

        Returns a report of compliant and non-compliant processes.
        """
        with self._lock:
            compliant = []
            non_compliant = []

            for pid, triad in self._processes.items():
                if triad.is_compliant:
                    compliant.append(pid)
                else:
                    reasons = []
                    if triad.validator_count < triad.minimum_required:
                        reasons.append(
                            f"need {triad.minimum_required}, have {triad.validator_count}"
                        )
                    if triad.validator_count > 0 and triad.validator_count % 2 == 0:
                        reasons.append("even count (deadlock risk)")
                    if len(triad.unique_types) < 2:
                        reasons.append("not complementary (need 2+ types)")
                    non_compliant.append({
                        "process_id": pid,
                        "criticality": triad.criticality.value,
                        "validators": triad.validator_count,
                        "required": triad.minimum_required,
                        "reasons": reasons,
                    })

                    self._record_violation(
                        pid, "non_compliant", "; ".join(reasons)
                    )

            total = len(self._processes)
            return {
                "total_processes": total,
                "compliant": len(compliant),
                "non_compliant": len(non_compliant),
                "compliance_rate": (
                    len(compliant) / max(total, 1)
                ),
                "compliant_processes": compliant,
                "violations": non_compliant,
            }

    def get_process_status(self, process_id: str) -> dict[str, Any] | None:
        """Get detailed status for a specific process triad."""
        with self._lock:
            triad = self._processes.get(process_id)
            if not triad:
                return None

            return {
                "process_id": triad.process_id,
                "description": triad.description,
                "criticality": triad.criticality.value,
                "is_compliant": triad.is_compliant,
                "validator_count": triad.validator_count,
                "minimum_required": triad.minimum_required,
                "unique_types": [t.value for t in triad.unique_types],
                "validators": [
                    {
                        "id": v.validator_id,
                        "type": v.validator_type.value,
                        "invocations": v.total_invocations,
                        "approval_rate": (
                            v.approvals / max(v.total_invocations, 1)
                        ),
                    }
                    for v in triad.validators
                ],
                "total_votes": triad.total_votes,
                "unanimous_approvals": triad.unanimous_approvals,
                "majority_approvals": triad.majority_approvals,
                "rejections": triad.rejections,
            }

    # =========================================================================
    # Violation Tracking
    # =========================================================================

    def _record_violation(
        self, process_id: str, violation_type: str, detail: str
    ) -> None:
        """Record a Rule of 3 violation."""
        violation = {
            "process_id": process_id,
            "type": violation_type,
            "detail": detail,
            "timestamp": time.time(),
        }
        self._violations.append(violation)

        if self._bus:
            self._bus.publish(CH_TRIAD_VIOLATION, violation)

        logger.warning(
            "Rule of 3 violation: %s - %s: %s",
            process_id, violation_type, detail,
        )

    def get_violations(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get recent Rule of 3 violations."""
        with self._lock:
            return list(self._violations[-limit:])

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_statistics(self) -> dict[str, Any]:
        """Get overall triad system statistics."""
        with self._lock:
            total = len(self._processes)
            compliant = sum(
                1 for t in self._processes.values() if t.is_compliant
            )
            total_validators = sum(
                t.validator_count for t in self._processes.values()
            )
            total_votes = sum(
                t.total_votes for t in self._processes.values()
            )

            by_criticality = defaultdict(lambda: {"total": 0, "compliant": 0})
            for triad in self._processes.values():
                crit = triad.criticality.value
                by_criticality[crit]["total"] += 1
                if triad.is_compliant:
                    by_criticality[crit]["compliant"] += 1

            return {
                "total_processes": total,
                "compliant_processes": compliant,
                "compliance_rate": compliant / max(total, 1),
                "total_validators": total_validators,
                "total_votes": total_votes,
                "total_violations": len(self._violations),
                "by_criticality": dict(by_criticality),
            }

    def get_health_report(self) -> dict[str, Any]:
        """Generate a health report for the triad system itself."""
        stats = self.get_statistics()
        compliance = self.check_compliance()

        report = {
            "healthy": stats["compliance_rate"] >= 0.8,
            "compliance_rate": stats["compliance_rate"],
            "critical_violations": [
                v
                for v in compliance["violations"]
                if v["criticality"] == "critical"
            ],
            "recommendations": [],
        }

        for v in compliance["violations"]:
            report["recommendations"].append(
                f"Process '{v['process_id']}' needs "
                f"{v['required'] - v['validators']} more validators "
                f"({'; '.join(v['reasons'])})"
            )

        if self._bus:
            self._bus.publish(CH_TRIAD_HEALTH, {
                "healthy": report["healthy"],
                "compliance_rate": stats["compliance_rate"],
                "violations": len(compliance["violations"]),
            })

        return report
