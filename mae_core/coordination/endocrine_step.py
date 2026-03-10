"""Endocrine step, cascade, and circadian integration mixin.

Contains step(), _apply_cascades(), set_circadian_phase(), and EventBus
callbacks _on_phi_measurement() and _on_healing_phase_changed(). Extracted
from endocrine_system.py to stay under 500-line limit.
"""

from __future__ import annotations

import logging

from mae_core.coordination.endocrine_system import CH_HORMONE_STATE, HormoneType

logger = logging.getLogger(__name__)


class EndocrineStepMixin:
    """Step, cascade, circadian, and EventBus callback logic for EndocrineSystem."""

    # =========================================================================
    # EventBus Handlers
    # =========================================================================

    def _on_phi_measurement(self, channel: str, message) -> None:
        """Respond to IIT Phi measurements from IntegrationMeter.

        Low organism-level phi (< 0.3) signals fragmentation — release mild
        cortisol to mobilize coordination resources. High phi (> 0.7) signals
        healthy integration — suppress cortisol slightly to reward coherent
        operation.

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
            self.release_hormone(HormoneType.CORTISOL, 0.05, "phi_fragmentation")
        elif phi > 0.7:
            self.suppress_hormone(HormoneType.CORTISOL, 0.05, "phi_integration")
            self.release_hormone(HormoneType.SEROTONIN, 0.03, "phi_integration")

    def _on_healing_phase_changed(self, channel: str, message) -> None:
        """Respond to healing phase changes by modulating cortisol.

        When healing is active (isolating/assessing/restoring), release
        cortisol to signal stress to the rest of the system. When
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
    # Internal cascade
    # =========================================================================

    def _apply_cascades(self, source: HormoneType, amount: float) -> None:
        """Apply cascade effects from a hormone release."""
        from mae_core.coordination.endocrine_system import HORMONE_CASCADES
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
