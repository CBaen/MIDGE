"""Somatic Map data models -- enums and dataclasses only.

Extracted from somatic_map.py to keep the core class under the 500-line cap.
Import from mae_core.emergent.somatic_map for all public names.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class SystemCriticality(Enum):
    """How critical a system is - determines protection level."""

    PERIPHERAL = "peripheral"  # Can be modified freely (like skin cells)
    STANDARD = "standard"  # Normal protection (like muscle)
    PROTECTED = "protected"  # Extra validation required (like organs)
    CRITICAL = "critical"  # Blood-brain barrier level protection (like brainstem)


class ModificationVerdict(Enum):
    """Result of blast radius analysis."""

    APPROVED = "approved"  # Safe to proceed
    APPROVED_WITH_WARNINGS = "approved_with_warnings"  # Proceed with caution
    REJECTED = "rejected"  # Too dangerous
    NEEDS_REVIEW = "needs_review"  # Human/higher authority needed


@dataclass
class SystemNode:
    """A registered system in the somatic map."""

    system_id: str
    description: str
    criticality: SystemCriticality = SystemCriticality.STANDARD
    upstream: set[str] = field(default_factory=set)  # Systems this depends on
    downstream: set[str] = field(default_factory=set)  # Systems that depend on this
    health: float = 1.0  # [0, 1]
    registered_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BlastRadiusReport:
    """Analysis of what a modification would affect."""

    target_system: str
    direct_downstream: list[str]  # Immediately affected
    transitive_downstream: list[str]  # All affected (recursive)
    critical_systems_affected: list[str]  # CRITICAL systems in blast radius
    protected_systems_affected: list[str]  # PROTECTED systems in blast radius
    total_affected: int
    max_depth: int  # How deep the cascade goes
    risk_score: float  # [0, 1] overall risk
    verdict: ModificationVerdict
    warnings: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ModificationRecord:
    """Record of a proposed or executed modification."""

    modification_id: str
    target_system: str
    description: str
    blast_radius: Optional[BlastRadiusReport] = None
    approved: bool = False
    executed: bool = False
    rolled_back: bool = False
    snapshot: Optional[dict[str, Any]] = None  # Pre-modification state
    proposed_at: float = field(default_factory=time.time)
    executed_at: Optional[float] = None
    completed_at: Optional[float] = None
