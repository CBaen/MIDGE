"""Senescence Manager - Programmed aging and rejuvenation.

Biological basis: Hayflick limit (1961). Cells have limited replication
capacity — telomeres shorten with each division. But aging is not just
wear: it is programmed. Idle muscles atrophy faster than exercised ones.
Sleep rejuvenates but does not reverse aging. The body tracks wear at
every scale.

Mae's SenescenceManager tracks wear on every system:
- Active systems age slowly (exercise preserves youth)
- Idle systems age rapidly (use it or lose it)
- Worn systems request rejuvenation (sleep/repair cycle)
- Fully senescent systems signal end-of-life (replacement needed)

Connection points:
- Every system reports activity via report_activity()
- LymphaticSystem may listen for senescent systems (waste to collect)
- AutoHealer may listen for rejuvenation requests
- Circadian may trigger rejuvenation during rest phases
- EventBus publishes age updates, rejuvenation needs, senescence events

Law compliance:
- Law 6 (Autopoietic Closure): aging drives rejuvenation which extends
  life which produces more aging. Circular causation.
- Law 8, Property 5 (Multi-scale Hierarchy): wear accumulates at
  every level — system, subsystem, organ, organism.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# EventBus channels
CH_REJUVENATION = "emergent.rejuvenation_needed"
CH_SENESCENT = "emergent.system_senescent"
CH_AGE_UPDATE = "emergent.age_update"


@dataclass
class SystemAge:
    """Age and wear tracking for a single system."""

    system_name: str
    creation_step: int = 0
    total_steps_active: int = 0
    wear_level: float = 0.0
    last_rejuvenation: int = 0
    activity_rate: float = 0.0


class SenescenceManager:
    """Tracks wear and aging across all of Mae's systems.

    Every system accumulates wear over time. The rate depends on
    activity: idle systems age faster (atrophy), active systems age
    slower (exercise). When wear exceeds a threshold, the system
    requests rejuvenation. If wear reaches maximum, the system is
    senescent and needs replacement.

    Like biological aging: sleep repairs but does not reverse.
    Rejuvenation resets wear to 0.3, not 0.0.
    """

    CH_REJUVENATION = CH_REJUVENATION
    CH_SENESCENT = CH_SENESCENT
    CH_AGE_UPDATE = CH_AGE_UPDATE

    def __init__(self, event_bus: Any = None, **kwargs: Any) -> None:
        self._bus = event_bus

        # Per-system age tracking
        self._system_ages: dict[str, SystemAge] = {}

        # Wear rate parameters
        self._wear_rate: float = 0.001        # Base wear per step
        self._idle_penalty: float = 2.0        # Idle systems age 2x faster
        self._activity_bonus: float = 0.5      # Active systems age 0.5x
        self._rejuvenation_threshold: float = 0.8  # Triggers rejuvenation request
        self._max_wear: float = 1.0            # System failure threshold

        # Track which systems already triggered events (avoid spam)
        self._rejuvenation_requested: set[str] = set()
        self._senescent_notified: set[str] = set()

        logger.info("SenescenceManager initialized (wear_rate=%.4f)", self._wear_rate)

    # =========================================================================
    # Registration
    # =========================================================================

    def register_system(self, name: str, creation_step: int = 0) -> None:
        """Start tracking a system's age and wear."""
        if name not in self._system_ages:
            self._system_ages[name] = SystemAge(
                system_name=name,
                creation_step=creation_step,
            )
            logger.debug("Senescence: tracking %s (created at step %d)", name, creation_step)

    def report_activity(self, name: str, active: bool = True) -> None:
        """Update a system's activity rate using exponential moving average.

        Called by systems or by a monitoring process to report whether
        a system was active this step.

        Args:
            name: System name (must be registered).
            active: Whether the system was active this step.
        """
        age = self._system_ages.get(name)
        if age is None:
            return

        alpha = 0.3  # EMA smoothing factor
        new_value = 1.0 if active else 0.0
        age.activity_rate = alpha * new_value + (1.0 - alpha) * age.activity_rate

        if active:
            age.total_steps_active += 1

    # =========================================================================
    # Step
    # =========================================================================

    def step(self, current_step: int) -> None:
        """Advance aging for all tracked systems.

        For each system:
        1. Compute effective wear rate (idle penalty or activity bonus)
        2. Accumulate wear
        3. Check thresholds and publish events
        """
        for name, age in self._system_ages.items():
            # Compute effective wear rate
            if age.activity_rate < 0.1:
                effective_rate = self._wear_rate * self._idle_penalty
            else:
                effective_rate = self._wear_rate * self._activity_bonus

            # Accumulate wear (capped at max)
            age.wear_level = min(self._max_wear, age.wear_level + effective_rate)

            # Rejuvenation threshold
            if (age.wear_level >= self._rejuvenation_threshold
                    and name not in self._rejuvenation_requested):
                self._rejuvenation_requested.add(name)
                if self._bus:
                    self._bus.publish(CH_REJUVENATION, {
                        "system_name": name,
                        "wear_level": age.wear_level,
                        "activity_rate": age.activity_rate,
                        "step": current_step,
                    })
                logger.info("Senescence: %s needs rejuvenation (wear=%.3f)",
                            name, age.wear_level)

            # Senescence (end of life)
            if (age.wear_level >= self._max_wear
                    and name not in self._senescent_notified):
                self._senescent_notified.add(name)
                if self._bus:
                    self._bus.publish(CH_SENESCENT, {
                        "system_name": name,
                        "wear_level": age.wear_level,
                        "total_steps_active": age.total_steps_active,
                        "step": current_step,
                    })
                logger.warning("Senescence: %s is SENESCENT (wear=%.3f)",
                               name, age.wear_level)

        # Publish organism-level age update
        if self._bus and self._system_ages:
            self._bus.publish(CH_AGE_UPDATE, {
                "organism_age": self.get_organism_age(),
                "tracked_systems": len(self._system_ages),
                "senescent_count": len(self._senescent_notified),
                "rejuvenation_pending": len(self._rejuvenation_requested),
                "step": current_step,
            })

    # =========================================================================
    # Rejuvenation
    # =========================================================================

    def rejuvenate(self, system_name: str) -> bool:
        """Reset wear to 0.3 (not fully — sleep repairs but does not reverse aging).

        Returns True if the system was found and rejuvenated.
        """
        age = self._system_ages.get(system_name)
        if age is None:
            return False

        age.wear_level = 0.3  # Partial reset (biological realism)
        age.last_rejuvenation = age.total_steps_active
        self._rejuvenation_requested.discard(system_name)
        self._senescent_notified.discard(system_name)

        logger.info("Senescence: %s rejuvenated (wear reset to 0.3)", system_name)
        return True

    # =========================================================================
    # Queries
    # =========================================================================

    def get_organism_age(self) -> float:
        """Mean wear level across all tracked systems."""
        if not self._system_ages:
            return 0.0
        total = sum(a.wear_level for a in self._system_ages.values())
        return total / len(self._system_ages)

    def get_oldest_systems(self, n: int = 3) -> list[SystemAge]:
        """Return the N most worn systems, sorted by wear descending."""
        sorted_ages = sorted(
            self._system_ages.values(),
            key=lambda a: a.wear_level,
            reverse=True,
        )
        return sorted_ages[:n]

    def get_system_age(self, name: str) -> SystemAge | None:
        """Get age info for a specific system."""
        return self._system_ages.get(name)

    # =========================================================================
    # Observability
    # =========================================================================

    def get_statistics(self) -> dict[str, Any]:
        return {
            "tracked_systems": len(self._system_ages),
            "organism_age": self.get_organism_age(),
            "senescent_count": len(self._senescent_notified),
            "rejuvenation_pending": len(self._rejuvenation_requested),
            "wear_rate": self._wear_rate,
            "idle_penalty": self._idle_penalty,
            "activity_bonus": self._activity_bonus,
            "system_wear": {
                name: age.wear_level
                for name, age in self._system_ages.items()
            },
        }

    # =========================================================================
    # Tier 2 Persistence
    # =========================================================================

    def serialize(self) -> dict:
        return {
            "wear_rate": self._wear_rate,
            "idle_penalty": self._idle_penalty,
            "activity_bonus": self._activity_bonus,
            "system_ages": {
                name: {
                    "creation_step": age.creation_step,
                    "total_steps_active": age.total_steps_active,
                    "wear_level": age.wear_level,
                    "last_rejuvenation": age.last_rejuvenation,
                    "activity_rate": age.activity_rate,
                }
                for name, age in self._system_ages.items()
            },
        }

    def restore(self, data: dict) -> None:
        self._wear_rate = data.get("wear_rate", 0.001)
        self._idle_penalty = data.get("idle_penalty", 2.0)
        self._activity_bonus = data.get("activity_bonus", 0.5)
        for name, age_data in data.get("system_ages", {}).items():
            self._system_ages[name] = SystemAge(
                system_name=name,
                creation_step=age_data.get("creation_step", 0),
                total_steps_active=age_data.get("total_steps_active", 0),
                wear_level=age_data.get("wear_level", 0.0),
                last_rejuvenation=age_data.get("last_rejuvenation", 0),
                activity_rate=age_data.get("activity_rate", 0.0),
            )
