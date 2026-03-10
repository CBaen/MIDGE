"""Threat Detector data models -- enums and dataclasses only.

Extracted from threat_detector.py to keep the core class under the 500-line cap.
Import from mae_core.defense.threat_detector for all public names.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ThreatLevel(Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @classmethod
    def from_score(cls, score: float) -> "ThreatLevel":
        if score < 0.1:
            return cls.NONE
        if score < 0.3:
            return cls.LOW
        if score < 0.6:
            return cls.MEDIUM
        if score < 0.8:
            return cls.HIGH
        return cls.CRITICAL


class DefenseStrategy(Enum):
    PORCUPINE = "porcupine"  # Proactive detection
    TURTLE = "turtle"  # Passive resilience
    LIZARD = "lizard"  # Adaptive sacrifice
    KANGAROO = "kangaroo"  # Aggressive counter


@dataclass
class Threat:
    """A detected threat."""

    threat_id: str
    source: str  # What/who is threatening
    target: str  # What's being threatened
    level: ThreatLevel
    score: float  # [0, 1]
    description: str = ""
    detected_at: float = field(default_factory=time.time)
    neutralized: bool = False
    strategy_used: Optional[DefenseStrategy] = None


@dataclass
class DefenseResponse:
    """Response to a threat."""

    threat: Threat
    strategy: DefenseStrategy
    success: bool
    actions: list[str] = field(default_factory=list)
    cost: float = 0.0  # Resource cost of defense
