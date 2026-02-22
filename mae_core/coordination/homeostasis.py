"""Homeostasis regulator - setpoint-based internal stability for Mae.

Biological basis: Walter Cannon's concept of homeostasis (1929). All living
systems maintain internal stability by detecting deviations from setpoints
and generating corrective signals. Body temperature, blood sugar, pH,
electrolytes - all are held within narrow ranges by negative feedback loops.

Mae's homeostasis monitors 7 vital parameters:

| Parameter        | Target | Range       | What It Tracks                |
|------------------|--------|-------------|-------------------------------|
| energy_level     | 0.7    | [0.3, 1.0]  | Available processing capacity |
| cortisol         | 0.2    | [0.0, 0.8]  | Stress hormone level          |
| dopamine         | 0.3    | [0.0, 0.8]  | Reward/exploration hormone    |
| serotonin        | 0.4    | [0.1, 0.9]  | Stability hormone level       |
| processing_load  | 0.5    | [0.1, 0.9]  | CPU/memory utilization        |
| memory_pressure  | 0.4    | [0.0, 0.8]  | Memory subsystem strain       |
| threat_level     | 0.1    | [0.0, 0.5]  | Current threat assessment     |

For each parameter, the regulator computes:
- Error: target - current (positive = too low, negative = too high)
- Correction: proportional signal clamped to [-1, 1]
- Urgency: how far outside the acceptable range

Law compliance:
- Law 6: autopoietic closure - the system regulates itself to maintain itself
- Law 8 property (8): prediction/error-correction - canonical PID-style control

EventBus channel: coordination.homeostasis_correction
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional

from mae_core.backbone.event_bus import EventBus

logger = logging.getLogger(__name__)

# EventBus channel
CH_HOMEOSTASIS_CORRECTION = "coordination.homeostasis_correction"


@dataclass
class Setpoint:
    """A single regulated parameter with target and acceptable range."""

    parameter_name: str
    target_value: float
    min_acceptable: float
    max_acceptable: float
    current_value: float
    gain: float  # Proportional gain for correction signal

    @property
    def range_width(self) -> float:
        """Width of the acceptable range."""
        return self.max_acceptable - self.min_acceptable

    @property
    def error(self) -> float:
        """Current error: target - current."""
        return self.target_value - self.current_value

    @property
    def in_range(self) -> bool:
        """Whether current value is within acceptable bounds."""
        return self.min_acceptable <= self.current_value <= self.max_acceptable


# Default setpoint definitions
DEFAULT_SETPOINTS: list[dict[str, Any]] = [
    {
        "parameter_name": "energy_level",
        "target_value": 0.7,
        "min_acceptable": 0.3,
        "max_acceptable": 1.0,
        "current_value": 0.7,
        "gain": 0.5,
    },
    {
        "parameter_name": "cortisol",
        "target_value": 0.2,
        "min_acceptable": 0.0,
        "max_acceptable": 0.8,
        "current_value": 0.2,
        "gain": 0.3,
    },
    {
        "parameter_name": "dopamine",
        "target_value": 0.3,
        "min_acceptable": 0.0,
        "max_acceptable": 0.8,
        "current_value": 0.3,
        "gain": 0.3,
    },
    {
        "parameter_name": "serotonin",
        "target_value": 0.4,
        "min_acceptable": 0.1,
        "max_acceptable": 0.9,
        "current_value": 0.4,
        "gain": 0.3,
    },
    {
        "parameter_name": "processing_load",
        "target_value": 0.5,
        "min_acceptable": 0.1,
        "max_acceptable": 0.9,
        "current_value": 0.5,
        "gain": 0.4,
    },
    {
        "parameter_name": "memory_pressure",
        "target_value": 0.4,
        "min_acceptable": 0.0,
        "max_acceptable": 0.8,
        "current_value": 0.4,
        "gain": 0.3,
    },
    {
        "parameter_name": "threat_level",
        "target_value": 0.1,
        "min_acceptable": 0.0,
        "max_acceptable": 0.5,
        "current_value": 0.1,
        "gain": 0.6,
    },
]

# Hormone names that map directly to setpoints
_HORMONE_SETPOINT_MAP = {"cortisol", "dopamine", "serotonin"}


class HomeostasisRegulator:
    """Mae's homeostasis regulator - maintains internal stability.

    Monitors vital parameters against setpoints and generates
    proportional correction signals. Other systems consume these
    corrections to adjust their behavior toward equilibrium.
    """

    CH_HOMEOSTASIS_CORRECTION = CH_HOMEOSTASIS_CORRECTION

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        endocrine_system: Any = None,
        setpoint_configs: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        self.event_bus = event_bus
        self._endocrine_system = endocrine_system
        self._step_count = 0

        # Initialize setpoints
        configs = setpoint_configs or DEFAULT_SETPOINTS
        self._setpoints: dict[str, Setpoint] = {}
        for cfg in configs:
            sp = Setpoint(**cfg)
            self._setpoints[sp.parameter_name] = sp

        # Correction history for trend analysis
        self._correction_history: list[dict[str, Any]] = []
        self._max_history = 200

        # Subscribe to endocrine state updates
        if self.event_bus is not None:
            self.event_bus.register_callback(
                "endocrine.state_update", self._on_hormone_update
            )

        logger.info(
            "HomeostasisRegulator initialized with %d setpoints",
            len(self._setpoints),
        )

    # =========================================================================
    # EventBus Handlers
    # =========================================================================

    def _on_hormone_update(self, channel: str, message: Any) -> None:
        """Update setpoints that track hormone levels."""
        data = self._parse_message(message)
        if data is None:
            return
        for hormone_name in _HORMONE_SETPOINT_MAP:
            if hormone_name in data and hormone_name in self._setpoints:
                self._setpoints[hormone_name].current_value = float(data[hormone_name])

    # =========================================================================
    # Core Logic
    # =========================================================================

    def _compute_error(self, setpoint: Setpoint) -> float:
        """Compute error for a setpoint: target - current.

        Positive error means the value is too low (needs increase).
        Negative error means the value is too high (needs decrease).
        """
        return setpoint.target_value - setpoint.current_value

    def _compute_correction(self, error: float, gain: float) -> float:
        """Compute proportional correction signal, clamped to [-1, 1].

        correction = error * gain, clamped.
        """
        raw = error * gain
        return max(-1.0, min(1.0, raw))

    def _compute_urgency(self, setpoint: Setpoint) -> float:
        """Compute urgency of correction.

        Urgency = |error| / range_width.
        0.0 = at target, 1.0+ = at or beyond range boundary.
        """
        if setpoint.range_width <= 0:
            return 0.0
        return abs(setpoint.error) / setpoint.range_width

    def step(self, current_step: int = 0) -> None:
        """Evaluate all setpoints and generate correction signals.

        For each parameter:
        1. Compute error (target - current)
        2. Generate proportional correction signal
        3. Compute urgency
        4. Publish correction on EventBus
        """
        self._step_count = current_step if current_step > 0 else self._step_count + 1

        # Poll endocrine system directly if available
        if self._endocrine_system is not None:
            if hasattr(self._endocrine_system, "get_all_levels"):
                levels = self._endocrine_system.get_all_levels()
                for hormone_name in _HORMONE_SETPOINT_MAP:
                    if hormone_name in levels and hormone_name in self._setpoints:
                        self._setpoints[hormone_name].current_value = float(
                            levels[hormone_name]
                        )

        corrections: list[dict[str, Any]] = []

        for name, sp in self._setpoints.items():
            error = self._compute_error(sp)
            correction = self._compute_correction(error, sp.gain)
            urgency = self._compute_urgency(sp)

            correction_data = {
                "parameter": name,
                "current": sp.current_value,
                "target": sp.target_value,
                "error": round(error, 4),
                "correction": round(correction, 4),
                "urgency": round(urgency, 4),
                "in_range": sp.in_range,
            }
            corrections.append(correction_data)

            # Publish individual corrections for parameters with significant error
            if self.event_bus is not None and abs(error) > 0.01:
                self.event_bus.publish(CH_HOMEOSTASIS_CORRECTION, correction_data)

        # Record history
        self._correction_history.append({
            "step": self._step_count,
            "corrections": corrections,
            "deviation_score": self.get_deviation_score(),
            "stable": self.is_stable(),
        })
        if len(self._correction_history) > self._max_history:
            self._correction_history = self._correction_history[-self._max_history:]

    # =========================================================================
    # Public Query API
    # =========================================================================

    def get_deviation_score(self) -> float:
        """Root mean square of all setpoint errors.

        0.0 = perfect homeostasis (all at target).
        Higher values indicate greater overall deviation.
        """
        if not self._setpoints:
            return 0.0
        sum_sq = sum(sp.error ** 2 for sp in self._setpoints.values())
        return math.sqrt(sum_sq / len(self._setpoints))

    def is_stable(self) -> bool:
        """Whether all setpoints are within their acceptable ranges."""
        return all(sp.in_range for sp in self._setpoints.values())

    def get_setpoint(self, parameter_name: str) -> Optional[Setpoint]:
        """Get a specific setpoint by name."""
        return self._setpoints.get(parameter_name)

    def update_current_value(self, parameter_name: str, value: float) -> None:
        """Externally update a parameter's current value.

        Used by systems that report their own metrics (processing_load,
        memory_pressure, energy_level, threat_level).
        """
        sp = self._setpoints.get(parameter_name)
        if sp is not None:
            sp.current_value = max(0.0, min(1.0, value))

    # =========================================================================
    # Observability
    # =========================================================================

    def get_statistics(self) -> dict[str, Any]:
        """Return homeostasis statistics for monitoring."""
        setpoint_stats = {}
        for name, sp in self._setpoints.items():
            setpoint_stats[name] = {
                "target": sp.target_value,
                "current": sp.current_value,
                "error": round(sp.error, 4),
                "in_range": sp.in_range,
                "urgency": round(self._compute_urgency(sp), 4),
            }

        return {
            "setpoints": setpoint_stats,
            "deviation_score": round(self.get_deviation_score(), 4),
            "is_stable": self.is_stable(),
            "setpoint_count": len(self._setpoints),
            "history_length": len(self._correction_history),
            "step_count": self._step_count,
        }

    # =========================================================================
    # Persistence (Tier 2)
    # =========================================================================

    def serialize(self) -> dict:
        """Serialize homeostasis state for persistence."""
        setpoint_data = {}
        for name, sp in self._setpoints.items():
            setpoint_data[name] = {
                "parameter_name": sp.parameter_name,
                "target_value": sp.target_value,
                "min_acceptable": sp.min_acceptable,
                "max_acceptable": sp.max_acceptable,
                "current_value": sp.current_value,
                "gain": sp.gain,
            }
        return {
            "setpoints": setpoint_data,
            "step_count": self._step_count,
        }

    def restore(self, data: dict) -> None:
        """Restore homeostasis state from serialized data."""
        if not isinstance(data, dict):
            return

        setpoint_data = data.get("setpoints", {})
        for name, sp_data in setpoint_data.items():
            if name in self._setpoints:
                sp = self._setpoints[name]
                sp.current_value = float(sp_data.get("current_value", sp.current_value))
                sp.target_value = float(sp_data.get("target_value", sp.target_value))
                sp.gain = float(sp_data.get("gain", sp.gain))
            else:
                # Restore a setpoint that doesn't exist yet
                self._setpoints[name] = Setpoint(**sp_data)

        self._step_count = int(data.get("step_count", 0))

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    @staticmethod
    def _parse_message(message: Any) -> Optional[dict]:
        """Parse an EventBus message (may be JSON string or dict)."""
        if isinstance(message, dict):
            return message
        if isinstance(message, str):
            try:
                parsed = json.loads(message)
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass
        return None
