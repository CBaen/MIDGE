"""Quorum Signal Types - Chemical signal definitions for quorum sensing.

Defines the signal types that agents can deposit and sense in the
quorum space, analogous to bacterial autoinducers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import math
import time


class DecayType(Enum):
    """How signal concentration decays over time."""

    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    STEP = "step"
    NONE = "none"


@dataclass
class QuorumSignalType:
    """Definition of a quorum signal species."""

    name: str
    description: str = ""
    decay_type: DecayType = DecayType.EXPONENTIAL
    decay_rate: float = 0.1
    base_strength: float = 1.0
    max_concentration: float = 1.0
    metadata_schema: dict[str, str] | None = None

    def compute_decay(self, initial: float, elapsed_seconds: float) -> float:
        """Compute decayed concentration."""
        if self.decay_type == DecayType.NONE:
            return initial
        if self.decay_type == DecayType.EXPONENTIAL:
            return initial * math.exp(-self.decay_rate * elapsed_seconds)
        if self.decay_type == DecayType.LINEAR:
            return max(0.0, initial - self.decay_rate * elapsed_seconds)
        if self.decay_type == DecayType.STEP:
            half_life = math.log(2) / self.decay_rate if self.decay_rate > 0 else float("inf")
            return 0.0 if elapsed_seconds > half_life else initial
        return initial

    def get_half_life(self) -> float | None:
        if self.decay_type == DecayType.EXPONENTIAL and self.decay_rate > 0:
            return math.log(2) / self.decay_rate
        return None


@dataclass
class SenseEvent:
    """Record of a deposit or sense operation."""

    timestamp: float = field(default_factory=time.time)
    event_type: str = "sense"  # "deposit" or "sense"
    signal_type: str = ""
    agent_id: str = ""
    concentration: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "signal_type": self.signal_type,
            "agent_id": self.agent_id,
            "concentration": self.concentration,
            "metadata": self.metadata,
        }


class SignalTypes:
    """Standard quorum signal definitions (constants)."""

    OPPORTUNITY = QuorumSignalType(
        "OPPORTUNITY", "Resource or reward detected",
        DecayType.EXPONENTIAL, 0.05, 1.0, 1.0,
        {"location": "str", "reward_estimate": "float"},
    )
    DANGER = QuorumSignalType(
        "DANGER", "Threat or high risk detected",
        DecayType.EXPONENTIAL, 0.02, 1.0, 1.0,
        {"risk_level": "float", "source": "str"},
    )
    CONSENSUS = QuorumSignalType(
        "CONSENSUS", "Agreement on a decision",
        DecayType.EXPONENTIAL, 0.08, 1.0, 1.0,
        {"proposal_id": "str", "vote": "str"},
    )
    READINESS = QuorumSignalType(
        "READINESS", "Agent ready for coordination",
        DecayType.EXPONENTIAL, 0.15, 1.0, 1.0,
    )
    EXPLORE = QuorumSignalType(
        "EXPLORE", "Exploration needed in area",
        DecayType.EXPONENTIAL, 0.1, 1.0, 1.0,
    )
    EXPLOIT = QuorumSignalType(
        "EXPLOIT", "Exploitation recommended",
        DecayType.EXPONENTIAL, 0.1, 1.0, 1.0,
    )
    HELP = QuorumSignalType(
        "HELP", "Agent requesting assistance",
        DecayType.EXPONENTIAL, 0.03, 1.0, 1.0,
    )
    SUCCESS = QuorumSignalType(
        "SUCCESS", "Task completed successfully",
        DecayType.EXPONENTIAL, 0.12, 1.0, 1.0,
    )


# Global registry for signal type lookup
SIGNAL_TYPE_REGISTRY: dict[str, QuorumSignalType] = {
    attr: getattr(SignalTypes, attr)
    for attr in dir(SignalTypes)
    if isinstance(getattr(SignalTypes, attr), QuorumSignalType)
}


def get_signal_type(name: str) -> QuorumSignalType | None:
    return SIGNAL_TYPE_REGISTRY.get(name)


def register_custom_signal_type(signal_type: QuorumSignalType) -> None:
    SIGNAL_TYPE_REGISTRY[signal_type.name] = signal_type


def list_signal_types() -> list[str]:
    return list(SIGNAL_TYPE_REGISTRY.keys())
