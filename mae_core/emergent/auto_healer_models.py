"""Auto-Healer data models -- enums and dataclasses only.

Extracted from auto_healer.py to keep the core class under the 500-line cap.
Import from mae_core.emergent.auto_healer for all public names.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class HealingPhase(Enum):
    DETECTING = "detecting"
    ISOLATING = "isolating"
    ASSESSING = "assessing"
    RESTORING = "restoring"
    VERIFYING = "verifying"
    COMPLETE = "complete"
    FAILED = "failed"


class FailureType(Enum):
    PERFORMANCE_DEGRADATION = "performance_degradation"
    AGENT_CRASH = "agent_crash"
    COMMUNICATION_BREAK = "communication_break"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    POLICY_CONTAGION = "policy_contagion"
    STARVATION = "starvation"
    CASCADE_FAILURE = "cascade_failure"


@dataclass
class FailureReport:
    """Detected failure requiring healing."""

    failure_id: str
    failure_type: FailureType
    affected_agents: list[str] = field(default_factory=list)
    affected_region: Optional[str] = None
    severity: float = 0.5  # [0, 1]
    detected_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HealingAction:
    """A specific recovery action taken."""

    action: str
    target: str
    success: bool = False
    details: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class HealingRecord:
    """Complete record of a healing operation."""

    failure: FailureReport
    phase: HealingPhase = HealingPhase.DETECTING
    root_cause: Optional[str] = None
    causal_path: list[str] = field(default_factory=list)
    actions_taken: list[HealingAction] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    success: bool = False
