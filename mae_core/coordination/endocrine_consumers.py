"""Endocrine consumer registration mixin.

Contains subscribe() and all 14 register_* convenience methods that wire
external systems to specific hormones. Extracted from endocrine_system.py
to stay under 500-line limit.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from mae_core.coordination.endocrine_system import HormoneType

logger = logging.getLogger(__name__)


class EndocrineConsumersMixin:
    """Hormone consumer registration: subscribe + 14 register_* methods."""

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
        """Register a named consumer for a specific hormone."""
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

    def register_resource_governor(self, rg: Any) -> None:
        """Wire cortisol -> ResourceGovernor budget tightening/relaxation.

        High cortisol (> 0.6) signals organism stress — tighten EXPLORE
        budgets to conserve API quota. Low cortisol (< 0.3) signals calm
        — relax EXPLORE budgets to allow more exploration.
        """
        def _on_cortisol(_ht: HormoneType, level: float) -> None:
            if level > 0.6:
                if hasattr(rg, "tighten_budgets"):
                    rg.tighten_budgets(level)
                elif hasattr(rg, "set_hormone_level"):
                    rg.set_hormone_level("cortisol", level)
            elif level < 0.3:
                if hasattr(rg, "relax_budgets"):
                    rg.relax_budgets(1.0 + (0.3 - level))
                elif hasattr(rg, "set_hormone_level"):
                    rg.set_hormone_level("cortisol", level)
            # Neutral zone [0.3, 0.6]: no action

        self.register_consumer(HormoneType.CORTISOL, "resource_governor", _on_cortisol)
