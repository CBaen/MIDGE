"""Endocrine system - hormonal modulation of Mae's behavior.

Biological analogy: The endocrine system releases hormones into the
bloodstream, affecting distant organs over time. Unlike electrical
signals (fast, targeted), hormones are slow, broad, and persistent.
They create global mood states that modulate many systems at once.

Mae's 6 hormones:

| Hormone    | Trigger                | Effect                           |
|------------|------------------------|----------------------------------|
| Dopamine   | Reward, novelty        | Increases exploration, creativity |
| Serotonin  | Success, stability     | Increases cooperation, patience  |
| Cortisol   | Stress, failure        | Increases urgency, lowers quality |
| Oxytocin   | Cooperation success    | Increases trust, peer sharing    |
| Adrenaline | Emergency              | Maximizes speed, minimizes deliberation |
| Melatonin  | Circadian REST phase   | Promotes consolidation, reduces activity |

Hormones have:
- Levels (0.0 - 1.0): Current concentration
- Decay rates: How fast they return to baseline
- Cascade effects: One hormone triggers another
- Optimal ranges: Where the system functions best
- Critical thresholds: Where behavior changes dramatically

Integration: Published on EventBus for all systems to subscribe.
Individual systems read hormone levels to modulate their parameters.

This file is a composition hub. Sub-modules:
  - endocrine_consumers.py  (EndocrineConsumersMixin)
  - endocrine_step.py       (EndocrineStepMixin)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from mae_core.backbone.event_bus import EventBus

logger = logging.getLogger(__name__)

# EventBus channels
CH_HORMONE_RELEASE = "endocrine.hormone_release"
CH_HORMONE_STATE = "endocrine.state_update"


class HormoneType(Enum):
    """Mae's 6 hormones."""

    DOPAMINE = "dopamine"  # Reward, exploration, creativity
    SEROTONIN = "serotonin"  # Stability, cooperation, patience
    CORTISOL = "cortisol"  # Stress, urgency, resource mobilization
    OXYTOCIN = "oxytocin"  # Trust, bonding, peer sharing
    ADRENALINE = "adrenaline"  # Emergency, speed, reflex bias
    MELATONIN = "melatonin"  # Sleep, consolidation, rest


@dataclass
class HormoneConfig:
    """Configuration for a single hormone."""

    baseline: float = 0.3
    decay_rate: float = 0.05
    min_level: float = 0.0
    max_level: float = 1.0
    optimal_low: float = 0.2
    optimal_high: float = 0.7
    critical_threshold: float = 0.9


# Default configurations for each hormone (biologically tuned)
DEFAULT_HORMONE_CONFIGS: dict[HormoneType, HormoneConfig] = {
    HormoneType.DOPAMINE: HormoneConfig(
        baseline=0.3, decay_rate=0.08, optimal_low=0.2, optimal_high=0.7,
        critical_threshold=0.9,
    ),
    HormoneType.SEROTONIN: HormoneConfig(
        baseline=0.4, decay_rate=0.03, optimal_low=0.3, optimal_high=0.7,
        critical_threshold=0.85,
    ),
    HormoneType.CORTISOL: HormoneConfig(
        baseline=0.2, decay_rate=0.06, optimal_low=0.1, optimal_high=0.5,
        critical_threshold=0.8,
    ),
    HormoneType.OXYTOCIN: HormoneConfig(
        baseline=0.3, decay_rate=0.04, optimal_low=0.2, optimal_high=0.7,
        critical_threshold=0.9,
    ),
    HormoneType.ADRENALINE: HormoneConfig(
        baseline=0.1, decay_rate=0.15, optimal_low=0.0, optimal_high=0.4,
        critical_threshold=0.7,
    ),
    HormoneType.MELATONIN: HormoneConfig(
        baseline=0.2, decay_rate=0.02, optimal_low=0.1, optimal_high=0.6,
        critical_threshold=0.8,
    ),
}

# Cascade effects: releasing one hormone affects others
# Format: (source, target, multiplier)
HORMONE_CASCADES: list[tuple[HormoneType, HormoneType, float]] = [
    (HormoneType.ADRENALINE, HormoneType.CORTISOL, 0.3),
    (HormoneType.CORTISOL, HormoneType.SEROTONIN, -0.2),
    (HormoneType.DOPAMINE, HormoneType.OXYTOCIN, 0.1),
    (HormoneType.SEROTONIN, HormoneType.CORTISOL, -0.15),
    (HormoneType.MELATONIN, HormoneType.ADRENALINE, -0.2),
    (HormoneType.ADRENALINE, HormoneType.MELATONIN, -0.3),
]


class _EndocrineSystemCore:
    """Core hormone management: release, suppress, read, modulate."""

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        configs: Optional[dict[HormoneType, HormoneConfig]] = None,
        state_publish_interval: int = 5,
    ) -> None:
        self.event_bus = event_bus or EventBus()
        self._state_publish_interval = state_publish_interval
        self._step_count = 0
        self._lock = threading.Lock()

        # Initialize hormone levels and configs
        self._configs = configs or dict(DEFAULT_HORMONE_CONFIGS)
        self._levels: dict[HormoneType, float] = {
            ht: cfg.baseline for ht, cfg in self._configs.items()
        }

        # Subscriber callbacks
        self._subscribers: dict[HormoneType, list[Callable[[HormoneType, float], None]]] = {
            ht: [] for ht in HormoneType
        }

        # Cross-system hormone consumers
        self._hormone_consumers: dict[HormoneType, list[tuple[Any, str]]] = {
            ht: [] for ht in HormoneType
        }

        self._release_history: list[dict[str, Any]] = []

        # Subscribe to healing phase changes for cortisol modulation
        self.event_bus.register_callback(
            "healing.phase_changed", self._on_healing_phase_changed
        )

        logger.info("EndocrineSystem initialized with %d hormones", len(self._levels))

    # =========================================================================
    # Hormone Release
    # =========================================================================

    def release_hormone(
        self,
        hormone: HormoneType,
        amount: float,
        trigger: str = "",
    ) -> float:
        """Release a hormone (increase its level)."""
        with self._lock:
            config = self._configs[hormone]
            old_level = self._levels[hormone]
            new_level = min(config.max_level, old_level + amount)
            self._levels[hormone] = new_level

            self._release_history.append(
                {
                    "hormone": hormone.value,
                    "amount": amount,
                    "trigger": trigger,
                    "new_level": new_level,
                    "step": self._step_count,
                }
            )
            if len(self._release_history) > 500:
                self._release_history = self._release_history[-500:]

            # Apply cascade effects
            self._apply_cascades(hormone, amount)

            # Publish release event
            self.event_bus.publish(
                CH_HORMONE_RELEASE,
                {
                    "hormone": hormone.value,
                    "amount": amount,
                    "level": new_level,
                    "trigger": trigger,
                },
            )

            # Notify subscribers
            for callback in self._subscribers.get(hormone, []):
                try:
                    callback(hormone, new_level)
                except Exception:
                    logger.exception("Error in hormone subscriber")

            # Dispatch to registered hormone consumers
            for consumer, method_name in self._hormone_consumers.get(hormone, []):
                try:
                    if consumer is None:
                        continue
                    method = getattr(consumer, method_name, None)
                    if method is not None:
                        method(new_level)
                except Exception:
                    logger.exception(
                        "Error dispatching %s to %s.%s",
                        hormone.value, type(consumer).__name__, method_name,
                    )

            if new_level >= config.critical_threshold:
                logger.warning(
                    "Hormone %s at critical level: %.2f (trigger: %s)",
                    hormone.value, new_level, trigger,
                )

            return new_level

    def suppress_hormone(
        self,
        hormone: HormoneType,
        amount: float,
        trigger: str = "",
    ) -> float:
        """Suppress a hormone (decrease its level)."""
        with self._lock:
            config = self._configs[hormone]
            old_level = self._levels[hormone]
            new_level = max(config.min_level, old_level - amount)
            self._levels[hormone] = new_level
            return new_level

    # =========================================================================
    # Hormone Reading
    # =========================================================================

    def get_level(self, hormone: HormoneType) -> float:
        """Get current level of a hormone."""
        return self._levels.get(hormone, 0.0)

    def get_all_levels(self) -> dict[str, float]:
        """Get all hormone levels as a dict."""
        return {ht.value: level for ht, level in self._levels.items()}

    def get_global_state(self) -> dict[str, Any]:
        """Get comprehensive endocrine state."""
        state: dict[str, Any] = {}
        for ht, level in self._levels.items():
            config = self._configs[ht]
            if level < config.optimal_low:
                zone = "deficient"
            elif level > config.critical_threshold:
                zone = "critical"
            elif level > config.optimal_high:
                zone = "elevated"
            else:
                zone = "optimal"

            state[ht.value] = {
                "level": level,
                "zone": zone,
                "baseline": config.baseline,
            }
        return state

    def is_stressed(self) -> bool:
        """Is Mae under stress? (cortisol or adrenaline elevated)."""
        return (
            self._levels[HormoneType.CORTISOL]
            > self._configs[HormoneType.CORTISOL].optimal_high
            or self._levels[HormoneType.ADRENALINE]
            > self._configs[HormoneType.ADRENALINE].optimal_high
        )

    def is_resting(self) -> bool:
        """Is Mae in rest mode? (melatonin elevated)."""
        return (
            self._levels[HormoneType.MELATONIN]
            > self._configs[HormoneType.MELATONIN].optimal_high
        )

    # =========================================================================
    # Modulation Helpers
    # =========================================================================

    def get_exploration_bias(self) -> float:
        """How much should agents explore vs exploit?"""
        dopamine = self._levels[HormoneType.DOPAMINE]
        serotonin = self._levels[HormoneType.SEROTONIN]
        return min(1.0, max(0.0, (dopamine - serotonin * 0.5 + 0.3)))

    def get_trust_level(self) -> float:
        """How much should agents trust peers?"""
        oxytocin = self._levels[HormoneType.OXYTOCIN]
        serotonin = self._levels[HormoneType.SEROTONIN]
        cortisol = self._levels[HormoneType.CORTISOL]
        return min(1.0, max(0.0, (oxytocin + serotonin * 0.3 - cortisol * 0.4)))

    def get_urgency_level(self) -> float:
        """How urgent is the current situation?"""
        adrenaline = self._levels[HormoneType.ADRENALINE]
        cortisol = self._levels[HormoneType.CORTISOL]
        return min(1.0, max(0.0, adrenaline * 0.6 + cortisol * 0.4))

    def get_reflex_bias(self) -> float:
        """How much should DecisionRouter bias toward reflexes?"""
        adrenaline = self._levels[HormoneType.ADRENALINE]
        melatonin = self._levels[HormoneType.MELATONIN]
        return min(1.0, max(0.0, adrenaline * 0.7 + melatonin * 0.3))

    def get_statistics(self) -> dict[str, Any]:
        """Get endocrine system statistics."""
        return {
            "levels": self.get_all_levels(),
            "state": self.get_global_state(),
            "is_stressed": self.is_stressed(),
            "is_resting": self.is_resting(),
            "exploration_bias": self.get_exploration_bias(),
            "trust_level": self.get_trust_level(),
            "urgency_level": self.get_urgency_level(),
            "reflex_bias": self.get_reflex_bias(),
            "release_count": len(self._release_history),
            "step_count": self._step_count,
        }


# Compose the final EndocrineSystem class by mixing in the extracted sub-modules.
# Imports here (after class definitions) avoid circular import issues.
from mae_core.coordination.endocrine_consumers import EndocrineConsumersMixin  # noqa: E402
from mae_core.coordination.endocrine_step import EndocrineStepMixin  # noqa: E402


class EndocrineSystem(_EndocrineSystemCore, EndocrineConsumersMixin, EndocrineStepMixin):
    """Mae's hormonal regulation system.

    Manages hormone levels, decay, cascade effects, and publishes
    state changes for other systems to consume.

    Composed from:
    - _EndocrineSystemCore: hormone release/suppress/read/modulate
    - EndocrineConsumersMixin: subscribe + 14 register_* methods
    - EndocrineStepMixin: step, cascades, circadian, EventBus callbacks
    """


__all__ = [
    "EndocrineSystem",
    "HormoneType",
    "HormoneConfig",
    "DEFAULT_HORMONE_CONFIGS",
    "HORMONE_CASCADES",
    "CH_HORMONE_RELEASE",
    "CH_HORMONE_STATE",
    "EndocrineConsumersMixin",
    "EndocrineStepMixin",
]
