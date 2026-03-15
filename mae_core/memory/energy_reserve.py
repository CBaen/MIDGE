"""EnergyReserve - Adipose tissue-inspired energy storage with leptin signaling.

Biological analogy: Adipose tissue (fat) stores excess energy as
triglycerides and releases it during fasting. Leptin, a hormone produced
by fat cells, signals satiety to the hypothalamus -- high fat stores
produce high leptin, suppressing appetite. Low fat stores produce low
leptin, triggering hunger.

This creates a feedback loop: the storage system itself signals when to
stop storing and when to start consuming, without requiring a central
controller. The fat "knows" how full it is and broadcasts that state.

Mae's EnergyReserve does the same:
- Stores surplus computational energy (excess capacity beyond immediate need)
- Releases energy when systems need it (during high load or recovery)
- Leptin signal indicates fullness (high = stop consuming, low = need more)
- Storage has efficiency loss (like metabolic heat -- not all surplus is stored)

Connection points:
- Endocrine system: leptin level feeds into hormonal signaling
- AutoHealer: draws energy for healing operations
- ThreatDetector: draws energy for defense responses
- EventBus: publishes starvation alerts and reserves-full signals

Law compliance:
- Law 6 (Autopoietic Closure): stores the energy that sustains its own
  operation and the operation of systems that depend on it.
- Law 8 (Consciousness): contributes to prediction/error-correction (property 8) --
  leptin anticipates future energy needs based on current reserves.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# EventBus channels
CH_ENERGY_STATUS = "memory.energy_status"
CH_STARVATION = "memory.starvation_alert"
CH_RESERVES_FULL = "memory.reserves_full"


class EnergyReserve:
    """Adipose tissue-inspired energy storage with leptin signaling.

    Stores surplus energy with efficiency loss (20% lost as heat).
    Releases stored energy on demand up to a per-step rate limit.
    Leptin level is proportional to reserves/capacity, signaling
    appetite suppression when reserves are high.
    """

    CH_ENERGY_STATUS = CH_ENERGY_STATUS
    CH_STARVATION = CH_STARVATION
    CH_RESERVES_FULL = CH_RESERVES_FULL

    def __init__(self, event_bus: Any = None, **kwargs: Any) -> None:
        self._bus = event_bus

        # Energy storage
        # MIDGE: reserves pinned to 100.0 (50% capacity) — fictional physiology
        # harms trading daemon. Default 50.0 caused immediate starvation during
        # busy sessions, publishing CH_STARVATION every step and setting
        # OrganismState._energy_critical = True, which fed InhibitionSystem's
        # NoGo branch and suppressed agent actions. Starting at 100.0 ensures
        # the system begins in a healthy state even if drain is re-enabled.
        self._reserves: float = 100.0  # was: 50.0
        self._max_capacity: float = 200.0
        self._min_critical: float = 10.0  # below this = starvation

        # Dynamics
        self._storage_efficiency: float = 0.8  # 80% stored, 20% lost as heat
        self._release_rate: float = 5.0  # max energy released per step

        # Leptin signaling
        self._leptin_level: float = 0.5  # appetite signal

        # Step tracking
        self._step_count: int = 0

        # Statistics
        self._total_stored: float = 0.0
        self._total_released: float = 0.0
        self._total_heat_lost: float = 0.0
        self._starvation_events: int = 0
        self._full_events: int = 0

        # Initialize leptin
        self._update_leptin()

        logger.info(
            "EnergyReserve initialized (reserves=%.1f, capacity=%.1f, leptin=%.2f)",
            self._reserves,
            self._max_capacity,
            self._leptin_level,
        )

    # =========================================================================
    # Energy Storage
    # =========================================================================

    def store(self, amount: float) -> float:
        """Store surplus energy. Returns amount actually stored.

        Storage has efficiency loss: only storage_efficiency fraction
        is stored, the rest is lost as heat.
        """
        if amount <= 0:
            return 0.0

        effective = amount * self._storage_efficiency
        heat_loss = amount - effective

        available_space = self._max_capacity - self._reserves
        actually_stored = min(effective, available_space)

        # If we can't store everything, proportionally reduce heat loss
        if effective > 0 and actually_stored < effective:
            ratio = actually_stored / effective
            heat_loss = amount * (1 - self._storage_efficiency) * ratio

        self._reserves += actually_stored
        self._total_stored += actually_stored
        self._total_heat_lost += heat_loss
        self._update_leptin()

        return actually_stored

    def release(self, amount: float) -> float:
        """Release energy from reserves. Returns amount actually released.

        Cannot release more than available reserves or the per-step
        release rate.
        """
        if amount <= 0:
            return 0.0

        capped = min(amount, self._release_rate)
        actually_released = min(capped, self._reserves)

        self._reserves -= actually_released
        self._total_released += actually_released
        self._update_leptin()

        return actually_released

    # =========================================================================
    # Leptin Signaling
    # =========================================================================

    def _update_leptin(self) -> None:
        """Update leptin level proportional to reserves/capacity."""
        if self._max_capacity > 0:
            self._leptin_level = self._reserves / self._max_capacity
        else:
            self._leptin_level = 0.0

    def get_leptin_level(self) -> float:
        """Return leptin level (appetite suppression signal).

        High = reserves are full, reduce intake.
        Low = reserves are low, increase intake.
        """
        return self._leptin_level

    # =========================================================================
    # Health Monitoring
    # =========================================================================

    def get_reserves(self) -> float:
        """Return current energy reserves."""
        return self._reserves

    def is_critical(self) -> bool:
        """Return True if reserves are critically low (starvation).

        MIDGE: always returns False — fictional physiology harms trading daemon.
        The "critical" state propagates to OrganismState._energy_critical, which
        feeds InhibitionSystem's NoGo branch and was causing agent suppression.
        Pinned to False so the starvation chain never fires regardless of actual
        reserve level. The underlying reserve tracking still works normally.
        """
        # MIDGE: disabled — fictional physiology harms trading daemon
        return False  # was: self._reserves < self._min_critical

    def is_full(self) -> bool:
        """Return True if reserves are near capacity (>90%)."""
        return self._reserves > self._max_capacity * 0.9

    # =========================================================================
    # Step (Periodic Maintenance)
    # =========================================================================

    def step(self, current_step: int = 0) -> None:
        """Per-step maintenance: update leptin, publish status alerts."""
        self._step_count = current_step if current_step > 0 else self._step_count + 1

        self._update_leptin()

        # Starvation alert
        if self.is_critical():
            self._starvation_events += 1
            if self._bus is not None:
                self._bus.publish(CH_STARVATION, {
                    "reserves": self._reserves,
                    "min_critical": self._min_critical,
                    "leptin_level": self._leptin_level,
                    "step": self._step_count,
                    "message": "STARVATION: energy reserves critically low",
                })
                # Throttled: log only once every 500 steps to avoid flooding.
                # At 0.0 reserves this fires every step — that's thousands of
                # identical warnings per session with zero diagnostic value.
                if self._step_count % 500 == 0:
                    logger.warning(
                        "STARVATION ALERT: reserves=%.1f (critical=%.1f)",
                        self._reserves,
                        self._min_critical,
                    )

        # Reserves full alert
        if self.is_full():
            self._full_events += 1
            if self._bus is not None:
                self._bus.publish(CH_RESERVES_FULL, {
                    "reserves": self._reserves,
                    "max_capacity": self._max_capacity,
                    "leptin_level": self._leptin_level,
                    "step": self._step_count,
                    "message": "Reserves near capacity, reduce intake",
                })

        # Regular status
        if self._bus is not None:
            capacity_pct = (
                self._reserves / self._max_capacity * 100.0
                if self._max_capacity > 0
                else 0.0
            )
            self._bus.publish(CH_ENERGY_STATUS, {
                "reserves": self._reserves,
                "capacity_pct": capacity_pct,
                "leptin_level": self._leptin_level,
                "is_critical": self.is_critical(),
                "step": self._step_count,
            })

    # =========================================================================
    # Observability
    # =========================================================================

    def get_statistics(self) -> dict[str, Any]:
        """Return energy reserve statistics for observability."""
        capacity_pct = (
            self._reserves / self._max_capacity * 100.0
            if self._max_capacity > 0
            else 0.0
        )
        return {
            "reserves": self._reserves,
            "max_capacity": self._max_capacity,
            "min_critical": self._min_critical,
            "capacity_pct": capacity_pct,
            "leptin_level": self._leptin_level,
            "storage_efficiency": self._storage_efficiency,
            "release_rate": self._release_rate,
            "total_stored": self._total_stored,
            "total_released": self._total_released,
            "total_heat_lost": self._total_heat_lost,
            "starvation_events": self._starvation_events,
            "full_events": self._full_events,
            "is_critical": self.is_critical(),
            "is_full": self.is_full(),
            "step": self._step_count,
        }

    # =========================================================================
    # Persistence (Tier 2)
    # =========================================================================

    def serialize(self) -> dict[str, Any]:
        """Serialize energy reserve state for persistence."""
        return {
            "reserves": self._reserves,
            "max_capacity": self._max_capacity,
            "min_critical": self._min_critical,
            "storage_efficiency": self._storage_efficiency,
            "release_rate": self._release_rate,
            "leptin_level": self._leptin_level,
            "step_count": self._step_count,
            "total_stored": self._total_stored,
            "total_released": self._total_released,
            "total_heat_lost": self._total_heat_lost,
            "starvation_events": self._starvation_events,
            "full_events": self._full_events,
        }

    def restore(self, data: dict[str, Any]) -> None:
        """Restore energy reserve state from serialized data."""
        self._reserves = data.get("reserves", 50.0)
        self._max_capacity = data.get("max_capacity", 200.0)
        self._min_critical = data.get("min_critical", 10.0)
        self._storage_efficiency = data.get("storage_efficiency", 0.8)
        self._release_rate = data.get("release_rate", 5.0)
        self._leptin_level = data.get("leptin_level", 0.5)
        self._step_count = data.get("step_count", 0)
        self._total_stored = data.get("total_stored", 0.0)
        self._total_released = data.get("total_released", 0.0)
        self._total_heat_lost = data.get("total_heat_lost", 0.0)
        self._starvation_events = data.get("starvation_events", 0)
        self._full_events = data.get("full_events", 0)

        self._update_leptin()

        logger.info(
            "EnergyReserve restored (reserves=%.1f, leptin=%.2f)",
            self._reserves,
            self._leptin_level,
        )
