"""Temporal Decay Manager - Time-based value decay for communication systems.

Provides exponential decay calculations used by quorum sensing,
message aggregation, and stigmergy systems.

Biological analogy: Chemical concentration dissipation.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any


class DecayProfile(Enum):
    """Preset decay profiles with (half_life_seconds, removal_threshold)."""

    FAST = (30.0, 0.01)
    NORMAL = (60.0, 0.01)
    SLOW = (120.0, 0.01)
    PERSISTENT = (300.0, 0.01)


@dataclass
class DecayConfig:
    """Configuration for a decay curve."""

    half_life: float  # seconds until value halves
    removal_threshold: float = 0.01  # remove when below this

    @property
    def decay_constant(self) -> float:
        """Lambda for exponential decay: ln(2) / half_life."""
        return math.log(2) / self.half_life if self.half_life > 0 else float("inf")


class TemporalDecayManager:
    """Manages time-based decay for messages and signals.

    Not a singleton - instantiate per subsystem or share via injection.
    """

    def __init__(
        self,
        default_half_life: float = 60.0,
        default_removal_threshold: float = 0.01,
    ) -> None:
        self._default = DecayConfig(default_half_life, default_removal_threshold)
        self._configs: dict[str, DecayConfig] = {}
        self.total_decays_calculated = 0
        self.messages_removed = 0

    def set_decay_profile(self, signal_type: str, profile: DecayProfile) -> None:
        half_life, threshold = profile.value
        self._configs[signal_type] = DecayConfig(half_life, threshold)

    def set_custom_decay(
        self, signal_type: str, half_life: float, removal_threshold: float = 0.01
    ) -> None:
        self._configs[signal_type] = DecayConfig(half_life, removal_threshold)

    def get_config(self, signal_type: str) -> DecayConfig:
        return self._configs.get(signal_type, self._default)

    def calculate_decay_factor(self, age_seconds: float, config: DecayConfig | None = None) -> float:
        """Exponential decay: exp(-lambda * t). Returns value in [0, 1]."""
        cfg = config or self._default
        self.total_decays_calculated += 1
        if age_seconds <= 0:
            return 1.0
        return math.exp(-cfg.decay_constant * age_seconds)

    def should_remove(self, age_seconds: float, signal_type: str = "") -> bool:
        """Check if a value has decayed below removal threshold."""
        cfg = self.get_config(signal_type)
        factor = self.calculate_decay_factor(age_seconds, cfg)
        if factor < cfg.removal_threshold:
            self.messages_removed += 1
            return True
        return False

    def get_decayed_value(
        self, original_value: float, age_seconds: float, signal_type: str = ""
    ) -> float:
        """Apply decay to a value."""
        cfg = self.get_config(signal_type)
        return original_value * self.calculate_decay_factor(age_seconds, cfg)

    def get_statistics(self) -> dict[str, Any]:
        return {
            "total_decays_calculated": self.total_decays_calculated,
            "messages_removed": self.messages_removed,
            "configured_types": len(self._configs),
            "default_half_life": self._default.half_life,
        }


def calculate_time_decay(age_seconds: float, half_life: float = 60.0) -> float:
    """Standalone decay function for simple use cases."""
    if age_seconds <= 0 or half_life <= 0:
        return 1.0
    return math.exp(-math.log(2) / half_life * age_seconds)
