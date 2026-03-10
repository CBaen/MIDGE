"""Triad Enforcer data models — ProcessCriticality, ValidatorType, Validator, ProcessTriad, VoteResult.

Extracted from triad_enforcer.py for single-responsibility.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


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
