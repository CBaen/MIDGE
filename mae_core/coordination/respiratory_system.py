"""Respiratory system - O2/CO2 gas exchange analog.

Biological analogy: The lungs exchange oxygen (fresh processing capacity)
for carbon dioxide (accumulated processing waste). Breathing rate increases
when CO2 builds up (respiratory drive). Hypoxia (low O2) and hypercapnia
(high CO2) trigger emergency alerts to other systems.

Mae's respiratory system:
- O2 represents fresh processing capacity (1.0 = fully oxygenated)
- CO2 represents accumulated processing waste (0.0 = no waste)
- Breathing exchanges: brings O2 in, pushes CO2 out
- Respiratory drive: higher CO2 -> faster breathing
- Hypoxia alert when O2 < 0.3
- Hypercapnia alert when CO2 > 0.7

Law compliance:
- Law 6 (Autopoietic Closure): Breathing is circular self-maintenance —
  the system produces fresh capacity that enables the processing that
  produces the waste that drives the breathing.
- Law 8 (Consciousness): (4) recurrence/feedback (O2-CO2 feedback loop)

EventBus channels:
- coordination.respiration_update (normal state)
- coordination.hypoxia (O2 critically low)
- coordination.hypercapnia (CO2 critically high)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from mae_core.backbone.event_bus import EventBus

logger = logging.getLogger(__name__)

# EventBus channels
CH_RESPIRATION_UPDATE = "coordination.respiration_update"
CH_HYPOXIA = "coordination.hypoxia"
CH_HYPERCAPNIA = "coordination.hypercapnia"


class RespiratorySystem:
    """Mae's gas-exchange system for processing capacity management.

    Maintains O2 (fresh capacity) and CO2 (waste) levels. Breathing
    rate adapts to CO2 concentration via respiratory drive. Publishes
    alerts when levels reach dangerous thresholds.
    """

    CH_RESPIRATION_UPDATE = CH_RESPIRATION_UPDATE
    CH_HYPOXIA = CH_HYPOXIA
    CH_HYPERCAPNIA = CH_HYPERCAPNIA

    def __init__(self, event_bus: Optional[EventBus] = None) -> None:
        self.event_bus = event_bus

        # Gas levels (0.0 - 1.0)
        self._oxygen: float = 1.0  # Fresh processing capacity
        self._co2: float = 0.0  # Accumulated processing waste

        # Breathing parameters
        self._base_breathing_rate: float = 0.1  # Baseline exchange rate per step
        self._breathing_rate: float = 0.1  # Current effective rate (adapts to CO2)
        self._consumption_rate: float = 0.0  # O2 usage from system activity

        # Step tracking
        self._step_count: int = 0

        # Subscribe to thermoregulation channels for load awareness
        if self.event_bus is not None:
            self.event_bus.register_callback(
                "coordination.temperature_normal", self._on_load_update
            )
            self.event_bus.register_callback(
                "coordination.cooling_needed", self._on_load_update
            )
            self.event_bus.register_callback(
                "coordination.warming_needed", self._on_load_update
            )

        logger.info(
            "RespiratorySystem initialized: O2=%.2f, CO2=%.2f, rate=%.2f",
            self._oxygen,
            self._co2,
            self._breathing_rate,
        )

    # =========================================================================
    # Load Awareness (EventBus subscriber)
    # =========================================================================

    def _on_load_update(self, channel: str, message: Any) -> None:
        """React to thermoregulation signals for load awareness.

        High load (cooling needed) increases consumption rate.
        Low load (warming needed) decreases consumption rate.
        Normal maintains a moderate baseline.
        """
        if "cooling_needed" in channel:
            self._consumption_rate = min(0.15, self._consumption_rate + 0.03)
        elif "warming_needed" in channel:
            self._consumption_rate = max(0.0, self._consumption_rate - 0.02)
        else:
            # Normal — settle toward baseline
            self._consumption_rate = max(0.0, self._consumption_rate - 0.01)

    # =========================================================================
    # Gas Exchange
    # =========================================================================

    def breathe(self) -> None:
        """Perform one breathing cycle: O2 in, CO2 out.

        The effective breathing rate is proportional to CO2 level
        (respiratory drive). Higher CO2 triggers faster breathing
        to clear the waste.
        """
        # Respiratory drive: higher CO2 = faster breathing
        # Rate scales from base_rate (at CO2=0) to 3x base_rate (at CO2=1)
        drive_multiplier = 1.0 + 2.0 * self._co2
        effective_rate = self._base_breathing_rate * drive_multiplier
        self._breathing_rate = effective_rate

        # Exchange
        self._oxygen = min(1.0, self._oxygen + effective_rate)
        self._co2 = max(0.0, self._co2 - effective_rate)

    def consume_oxygen(self, amount: float) -> None:
        """Consume O2 and produce CO2 from system activity.

        Args:
            amount: How much O2 to consume (also produces equal CO2).
        """
        actual = min(amount, self._oxygen)
        self._oxygen = max(0.0, self._oxygen - actual)
        self._co2 = min(1.0, self._co2 + actual)

    # =========================================================================
    # Step (Law 6: autopoietic closure, Law 8: recurrence/feedback)
    # =========================================================================

    def step(self, current_step: int = 0) -> None:
        """Run one respiratory cycle.

        1. Apply ongoing consumption from system activity.
        2. Compute breathing rate (proportional to CO2).
        3. Breathe (exchange gases).
        4. Publish appropriate alert or normal status.
        """
        self._step_count = current_step if current_step else self._step_count + 1

        # 1. Apply ongoing consumption
        if self._consumption_rate > 0:
            self.consume_oxygen(self._consumption_rate)

        # 2-3. Breathe (includes respiratory drive computation)
        self.breathe()

        # 4. Publish status
        if self.event_bus is not None:
            status_data = {
                "oxygen": self._oxygen,
                "co2": self._co2,
                "breathing_rate": self._breathing_rate,
                "step": self._step_count,
            }

            if self._oxygen < 0.3:
                self.event_bus.publish(CH_HYPOXIA, status_data)
                logger.warning(
                    "HYPOXIA: O2=%.2f (threshold 0.3) at step %d",
                    self._oxygen,
                    self._step_count,
                )
            elif self._co2 > 0.7:
                self.event_bus.publish(CH_HYPERCAPNIA, status_data)
                logger.warning(
                    "HYPERCAPNIA: CO2=%.2f (threshold 0.7) at step %d",
                    self._co2,
                    self._step_count,
                )
            else:
                self.event_bus.publish(CH_RESPIRATION_UPDATE, status_data)

    # =========================================================================
    # Query
    # =========================================================================

    def get_oxygen_level(self) -> float:
        """Return current oxygen level (0.0 - 1.0)."""
        return self._oxygen

    def get_co2_level(self) -> float:
        """Return current CO2 level (0.0 - 1.0)."""
        return self._co2

    def is_breathing_normally(self) -> bool:
        """True when O2 > 0.3 and CO2 < 0.7 (no alerts active)."""
        return self._oxygen > 0.3 and self._co2 < 0.7

    # =========================================================================
    # Observability
    # =========================================================================

    def get_statistics(self) -> dict[str, Any]:
        """Return respiratory system statistics."""
        return {
            "oxygen": self._oxygen,
            "co2": self._co2,
            "breathing_rate": self._breathing_rate,
            "base_breathing_rate": self._base_breathing_rate,
            "consumption_rate": self._consumption_rate,
            "is_breathing_normally": self.is_breathing_normally(),
            "step_count": self._step_count,
        }

    # =========================================================================
    # Tier 2 Persistence
    # =========================================================================

    def serialize(self) -> dict[str, Any]:
        """Serialize state for Tier 2 persistence."""
        return {
            "oxygen": self._oxygen,
            "co2": self._co2,
            "base_breathing_rate": self._base_breathing_rate,
            "breathing_rate": self._breathing_rate,
            "consumption_rate": self._consumption_rate,
            "step_count": self._step_count,
        }

    def restore(self, data: dict[str, Any]) -> None:
        """Restore state from serialized data."""
        self._oxygen = data.get("oxygen", self._oxygen)
        self._co2 = data.get("co2", self._co2)
        self._base_breathing_rate = data.get("base_breathing_rate", self._base_breathing_rate)
        self._breathing_rate = data.get("breathing_rate", self._breathing_rate)
        self._consumption_rate = data.get("consumption_rate", self._consumption_rate)
        self._step_count = data.get("step_count", self._step_count)
