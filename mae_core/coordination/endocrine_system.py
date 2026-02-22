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
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from mae_core.backbone.event_bus import EventBus

logger = logging.getLogger(__name__)

# EventBus channel
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

    baseline: float = 0.3  # Resting level
    decay_rate: float = 0.05  # Per-step decay toward baseline
    min_level: float = 0.0
    max_level: float = 1.0
    optimal_low: float = 0.2  # Below this = deficiency
    optimal_high: float = 0.7  # Above this = excess
    critical_threshold: float = 0.9  # Above this = emergency behavior


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
        critical_threshold=0.7,  # Lower threshold - adrenaline is powerful
    ),
    HormoneType.MELATONIN: HormoneConfig(
        baseline=0.2, decay_rate=0.02, optimal_low=0.1, optimal_high=0.6,
        critical_threshold=0.8,
    ),
}

# Cascade effects: releasing one hormone affects others
# Format: (source, target, multiplier)
# Positive multiplier = boost target. Negative = suppress target.
HORMONE_CASCADES: list[tuple[HormoneType, HormoneType, float]] = [
    # Adrenaline triggers cortisol (fight-or-flight cascade)
    (HormoneType.ADRENALINE, HormoneType.CORTISOL, 0.3),
    # High cortisol suppresses serotonin (stress reduces calm)
    (HormoneType.CORTISOL, HormoneType.SEROTONIN, -0.2),
    # Dopamine boosts oxytocin slightly (reward enhances bonding)
    (HormoneType.DOPAMINE, HormoneType.OXYTOCIN, 0.1),
    # Serotonin suppresses cortisol (calm reduces stress)
    (HormoneType.SEROTONIN, HormoneType.CORTISOL, -0.15),
    # Melatonin suppresses adrenaline (sleep reduces alertness)
    (HormoneType.MELATONIN, HormoneType.ADRENALINE, -0.2),
    # Adrenaline suppresses melatonin (emergency overrides sleep)
    (HormoneType.ADRENALINE, HormoneType.MELATONIN, -0.3),
]


class EndocrineSystem:
    """Mae's hormonal regulation system.

    Manages hormone levels, decay, cascade effects, and publishes
    state changes for other systems to consume.
    """

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        configs: Optional[dict[HormoneType, HormoneConfig]] = None,
        state_publish_interval: int = 5,  # Publish state every N steps
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

        # Subscriber callbacks for hormone-specific events
        self._subscribers: dict[HormoneType, list[Callable[[HormoneType, float], None]]] = {
            ht: [] for ht in HormoneType
        }

        # Cross-system hormone consumers: hormone_type -> [(system_ref, method_name)]
        # These are external systems that react to hormone changes.
        self._hormone_consumers: dict[HormoneType, list[tuple[Any, str]]] = {
            ht: [] for ht in HormoneType
        }

        # Release history for tracking
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
        """Release a hormone (increase its level).

        Args:
            hormone: Which hormone to release.
            amount: How much to increase (0.0-1.0 scale).
            trigger: Description of what caused the release.

        Returns:
            New hormone level after release.
        """
        with self._lock:
            config = self._configs[hormone]
            old_level = self._levels[hormone]
            new_level = min(config.max_level, old_level + amount)
            self._levels[hormone] = new_level

            # Record release
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
                    hormone.value,
                    new_level,
                    trigger,
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
    # Modulation Helpers (convenience for consuming systems)
    # =========================================================================

    def get_exploration_bias(self) -> float:
        """How much should agents explore vs exploit?

        High dopamine + low serotonin = high exploration.
        Low dopamine + high serotonin = high exploitation.
        Returns 0.0 (full exploit) to 1.0 (full explore).
        """
        dopamine = self._levels[HormoneType.DOPAMINE]
        serotonin = self._levels[HormoneType.SEROTONIN]
        return min(1.0, max(0.0, (dopamine - serotonin * 0.5 + 0.3)))

    def get_trust_level(self) -> float:
        """How much should agents trust peers?

        High oxytocin + high serotonin = high trust.
        High cortisol = low trust.
        """
        oxytocin = self._levels[HormoneType.OXYTOCIN]
        serotonin = self._levels[HormoneType.SEROTONIN]
        cortisol = self._levels[HormoneType.CORTISOL]
        return min(1.0, max(0.0, (oxytocin + serotonin * 0.3 - cortisol * 0.4)))

    def get_urgency_level(self) -> float:
        """How urgent is the current situation?

        High adrenaline + cortisol = high urgency.
        """
        adrenaline = self._levels[HormoneType.ADRENALINE]
        cortisol = self._levels[HormoneType.CORTISOL]
        return min(1.0, max(0.0, adrenaline * 0.6 + cortisol * 0.4))

    def get_reflex_bias(self) -> float:
        """How much should DecisionRouter bias toward reflexes?

        High adrenaline = strong reflex bias.
        High melatonin = also reflex bias (too tired for deliberation).
        """
        adrenaline = self._levels[HormoneType.ADRENALINE]
        melatonin = self._levels[HormoneType.MELATONIN]
        return min(1.0, max(0.0, adrenaline * 0.7 + melatonin * 0.3))

    # =========================================================================
    # Subscription
    # =========================================================================

    def subscribe(
        self, hormone: HormoneType, callback: Callable[[HormoneType, float], None]
    ) -> None:
        """Subscribe to changes in a specific hormone."""
        self._subscribers[hormone].append(callback)

    def register_consumer(
        self,
        hormone_type: HormoneType,
        consumer_name: str,
        callback: Callable[[HormoneType, float], None],
    ) -> None:
        """Register a named consumer for a specific hormone.

        When that hormone is released, the callback is invoked with the
        hormone type and new level.  This is a thin wrapper around
        subscribe() that adds logging for traceability.
        """
        logger.info(
            "Registered consumer '%s' for %s", consumer_name, hormone_type.value
        )
        self.subscribe(hormone_type, callback)

    def register_hormone_consumer(
        self, hormone_type: HormoneType, consumer: Any, method_name: str
    ) -> None:
        """Register an external system as a hormone consumer.

        When the specified hormone is released, the consumer's method
        will be called with the new hormone level as argument.

        Args:
            hormone_type: Which hormone to react to.
            consumer: Reference to the consuming system.
            method_name: Name of the method to call on the consumer.
        """
        if consumer is None:
            return
        self._hormone_consumers[hormone_type].append((consumer, method_name))

    # =========================================================================
    # Convenience Consumer Registration
    # =========================================================================

    def register_threat_detector(self, td: Any) -> None:
        """Wire cortisol -> ThreatDetector sensitivity."""
        def _on_cortisol(_ht: HormoneType, level: float) -> None:
            if hasattr(td, "set_sensitivity"):
                td.set_sensitivity(level)
            elif hasattr(td, "set_hormone_level"):
                td.set_hormone_level("cortisol", level)

        self.register_consumer(HormoneType.CORTISOL, "threat_detector", _on_cortisol)

    def register_auto_healer(self, ah: Any) -> None:
        """Wire cortisol -> AutoHealer priority."""
        def _on_cortisol(_ht: HormoneType, level: float) -> None:
            if hasattr(ah, "set_cortisol_priority"):
                ah.set_cortisol_priority(level)
            elif hasattr(ah, "set_priority"):
                ah.set_priority(level)
            elif hasattr(ah, "set_hormone_level"):
                ah.set_hormone_level("cortisol", level)

        self.register_consumer(HormoneType.CORTISOL, "auto_healer", _on_cortisol)

    def register_curiosity_drive(self, cd: Any) -> None:
        """Wire dopamine -> CuriosityDrive exploration bonus."""
        def _on_dopamine(_ht: HormoneType, level: float) -> None:
            if hasattr(cd, "set_exploration_bonus"):
                cd.set_exploration_bonus(level)
            elif hasattr(cd, "set_hormone_level"):
                cd.set_hormone_level("dopamine", level)

        self.register_consumer(HormoneType.DOPAMINE, "curiosity_drive", _on_dopamine)

    def register_quorum_sensor(self, qs: Any) -> None:
        """Wire serotonin -> QuorumSensor threshold modifier."""
        def _on_serotonin(_ht: HormoneType, level: float) -> None:
            if hasattr(qs, "set_threshold_modifier"):
                qs.set_threshold_modifier(level)
            elif hasattr(qs, "set_hormone_level"):
                qs.set_hormone_level("serotonin", level)

        self.register_consumer(HormoneType.SEROTONIN, "quorum_sensor", _on_serotonin)

    def register_memory_consolidator(self, mc: Any) -> None:
        """Wire melatonin -> MemoryConsolidator consolidation trigger."""
        def _on_melatonin(_ht: HormoneType, level: float) -> None:
            if hasattr(mc, "trigger_consolidation"):
                mc.trigger_consolidation()
            elif hasattr(mc, "set_hormone_level"):
                mc.set_hormone_level("melatonin", level)

        self.register_consumer(HormoneType.MELATONIN, "memory_consolidator", _on_melatonin)

    def register_decision_router(self, dr: Any) -> None:
        """Wire adrenaline -> DecisionRouter reflex bias."""
        def _on_adrenaline(_ht: HormoneType, level: float) -> None:
            if hasattr(dr, "set_reflex_bias"):
                dr.set_reflex_bias(level)
            elif hasattr(dr, "set_hormone_level"):
                dr.set_hormone_level("adrenaline", level)

        self.register_consumer(HormoneType.ADRENALINE, "decision_router", _on_adrenaline)

    def register_gamification_mixin(self, gm: Any) -> None:
        """Wire dopamine -> GamificationMixin motivation."""
        def _on_dopamine(_ht: HormoneType, level: float) -> None:
            if hasattr(gm, "set_motivation"):
                gm.set_motivation(level)
            elif hasattr(gm, "set_hormone_level"):
                gm.set_hormone_level("dopamine", level)

        self.register_consumer(HormoneType.DOPAMINE, "gamification_mixin", _on_dopamine)

    def register_frl(self, frl: Any) -> None:
        """Wire serotonin -> FRL trust in peer policies, oxytocin -> cooperation weight."""
        def _on_serotonin(_ht: HormoneType, level: float) -> None:
            if hasattr(frl, "set_peer_trust"):
                frl.set_peer_trust(level)
            elif hasattr(frl, "set_hormone_level"):
                frl.set_hormone_level("serotonin", level)

        def _on_oxytocin(_ht: HormoneType, level: float) -> None:
            if hasattr(frl, "set_cooperation_weight"):
                frl.set_cooperation_weight(level)
            elif hasattr(frl, "set_hormone_level"):
                frl.set_hormone_level("oxytocin", level)

        self.register_consumer(HormoneType.SEROTONIN, "frl_serotonin", _on_serotonin)
        self.register_consumer(HormoneType.OXYTOCIN, "frl_oxytocin", _on_oxytocin)

    def register_vdn(self, vdn: Any) -> None:
        """Wire oxytocin -> VDN cooperation weight."""
        def _on_oxytocin(_ht: HormoneType, level: float) -> None:
            if hasattr(vdn, "set_cooperation_weight"):
                vdn.set_cooperation_weight(level)
            elif hasattr(vdn, "set_hormone_level"):
                vdn.set_hormone_level("oxytocin", level)

        self.register_consumer(HormoneType.OXYTOCIN, "vdn", _on_oxytocin)

    # =========================================================================
    # EventBus Handlers
    # =========================================================================

    def _on_phi_measurement(self, channel: str, message: Any) -> None:
        """Respond to IIT Phi measurements from IntegrationMeter.

        Low organism-level phi (< 0.3) signals fragmentation — subsystems
        are operating independently. Release mild cortisol to mobilize
        coordination resources (like the body's stress response to
        disconnection). High phi (> 0.7) signals healthy integration —
        suppress cortisol slightly to reward coherent operation.

        Advisory: effect size +-0.05, never blocks, clamped by hormone bounds.
        """
        import json as _json
        try:
            msg = _json.loads(message) if isinstance(message, str) else message
        except (TypeError, ValueError):
            return
        if not isinstance(msg, dict):
            return

        phi = msg.get("organism_mean_phi")
        if phi is None or not isinstance(phi, (int, float)):
            return

        if phi < 0.3:
            # Fragmentation detected — mild coordination stress
            self.release_hormone(HormoneType.CORTISOL, 0.05, "phi_fragmentation")
        elif phi > 0.7:
            # Strong integration — reduce stress, reward coherence
            self.suppress_hormone(HormoneType.CORTISOL, 0.05, "phi_integration")
            self.release_hormone(HormoneType.SEROTONIN, 0.03, "phi_integration")

    def _on_healing_phase_changed(self, channel: str, message: Any) -> None:
        """Respond to healing phase changes by modulating cortisol.

        When healing is active (isolating/assessing/restoring), release
        cortisol to signal stress to the rest of the system.  When
        healing completes, suppress cortisol.
        """
        if isinstance(message, str):
            try:
                import json
                message = json.loads(message)
            except (json.JSONDecodeError, TypeError):
                return
        if not isinstance(message, dict):
            return

        phase = message.get("phase", "")
        if phase in ("isolating", "assessing", "restoring"):
            self.release_hormone(HormoneType.CORTISOL, 0.15, f"healing_{phase}")
        elif phase == "complete":
            self.suppress_hormone(HormoneType.CORTISOL, 0.1, "healing_complete")

    # =========================================================================
    # Step (decay + publish)
    # =========================================================================

    def step(self) -> None:
        """Decay hormones toward baseline and publish state.

        Called each model step. Hormones naturally decay toward their
        baseline levels, simulating metabolic clearance.
        """
        self._step_count += 1

        with self._lock:
            for ht, config in self._configs.items():
                level = self._levels[ht]
                # Decay toward baseline
                if level > config.baseline:
                    self._levels[ht] = max(
                        config.baseline,
                        level - config.decay_rate,
                    )
                elif level < config.baseline:
                    self._levels[ht] = min(
                        config.baseline,
                        level + config.decay_rate * 0.5,  # Recovery slower than decay
                    )

        # Periodic state publish
        if self._step_count % self._state_publish_interval == 0:
            self.event_bus.publish(CH_HORMONE_STATE, self.get_all_levels())

    # =========================================================================
    # Circadian Integration
    # =========================================================================

    def set_circadian_phase(self, phase: str) -> None:
        """Respond to circadian phase changes.

        ACTIVE: Suppress melatonin, boost dopamine.
        CONSOLIDATION: Moderate melatonin, suppress adrenaline.
        REST: High melatonin, suppress cortisol and adrenaline.
        """
        if phase == "REST":
            self.release_hormone(HormoneType.MELATONIN, 0.3, "circadian_rest")
            self.suppress_hormone(HormoneType.ADRENALINE, 0.2, "circadian_rest")
            self.suppress_hormone(HormoneType.CORTISOL, 0.1, "circadian_rest")
        elif phase == "CONSOLIDATION":
            self.release_hormone(HormoneType.MELATONIN, 0.15, "circadian_consolidation")
            self.suppress_hormone(HormoneType.ADRENALINE, 0.1, "circadian_consolidation")
        elif phase == "ACTIVE":
            self.suppress_hormone(HormoneType.MELATONIN, 0.2, "circadian_active")
            self.release_hormone(HormoneType.DOPAMINE, 0.1, "circadian_active")

    # =========================================================================
    # Internal
    # =========================================================================

    def _apply_cascades(self, source: HormoneType, amount: float) -> None:
        """Apply cascade effects from a hormone release."""
        for src, target, multiplier in HORMONE_CASCADES:
            if src != source:
                continue

            cascade_amount = amount * multiplier
            config = self._configs[target]
            old_level = self._levels[target]

            if cascade_amount > 0:
                self._levels[target] = min(config.max_level, old_level + cascade_amount)
            else:
                self._levels[target] = max(config.min_level, old_level + cascade_amount)

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
