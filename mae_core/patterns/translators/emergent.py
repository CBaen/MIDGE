"""Emergent translators -- CapabilityDiscovery.

Translates emergent capability events into PatternSignals.
"""

from __future__ import annotations

from typing import Any

from mae_core.patterns.pattern_signal import (
    PatternDomain,
    PatternForm,
    PatternSignal,
)


class CapabilityTranslator:
    """Translates CapabilityDiscovery events into PatternSignals.

    Listens to: improvement.capability_found, improvement.capability_validated
    """

    @property
    def source_name(self) -> str:
        return "capability_discovery"

    @property
    def channels(self) -> list[str]:
        return ["improvement.capability_found", "improvement.capability_validated"]

    def translate(self, channel: str, message: Any) -> PatternSignal | None:
        if not isinstance(message, dict):
            return None

        if channel == "improvement.capability_found":
            return self._translate_found(message)
        elif channel == "improvement.capability_validated":
            return self._translate_validated(message)
        return None

    def _translate_found(self, data: dict[str, Any]) -> PatternSignal | None:
        agent_id = data.get("agent_id", "unknown")
        context = data.get("context", "unknown")
        delta = data.get("performance_delta", 0.0)

        description = (
            f"New capability found by agent {agent_id}: "
            f"context={context}, performance_delta={delta:.3f}"
        )

        return PatternSignal(
            source_system=self.source_name,
            domain=PatternDomain.CAPABILITY,
            form=PatternForm.REACTIVE,
            confidence=0.5,  # Just discovered, not yet validated
            salience=min(1.0, abs(delta)),
            description=description,
            evidence=data,
        )

    def _translate_validated(self, data: dict[str, Any]) -> PatternSignal | None:
        capability_id = data.get("capability_id", "unknown")
        validation_score = data.get("validation_score", 0.0)

        description = (
            f"Capability validated: {capability_id} "
            f"(score={validation_score:.2f})"
        )

        return PatternSignal(
            source_system=self.source_name,
            domain=PatternDomain.CAPABILITY,
            form=PatternForm.REACTIVE,
            confidence=min(1.0, validation_score),
            salience=0.6,  # Validated capabilities are moderately important
            description=description,
            evidence=data,
        )
