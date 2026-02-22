"""Memory translators -- PatternDistiller (consolidation events).

Translates memory consolidation events into PatternSignals.
The PatternDistiller's output patterns are translated as ANCESTRAL form
since they've already survived the Rule of Three.
"""

from __future__ import annotations

from typing import Any

from mae_core.patterns.pattern_signal import (
    PatternDomain,
    PatternForm,
    PatternSignal,
)


class PatternDistillerTranslator:
    """Translates memory consolidation events into PatternSignals.

    Listens to: memory.consolidation_complete
    Payload: {agent_id, episodes_consolidated, patterns_found}

    Also accepts direct injection of distilled patterns via
    inject_patterns() for use during consolidation.
    """

    @property
    def source_name(self) -> str:
        return "pattern_distiller"

    @property
    def channels(self) -> list[str]:
        return ["memory.consolidation_complete"]

    def translate(self, channel: str, message: Any) -> PatternSignal | None:
        if not isinstance(message, dict):
            return None

        agent_id = message.get("agent_id", "unknown")
        episodes = message.get("episodes_consolidated", 0)

        if episodes == 0:
            return None

        description = (
            f"Memory consolidation complete for agent {agent_id}: "
            f"{episodes} episodes consolidated"
        )

        return PatternSignal(
            source_system=self.source_name,
            domain=PatternDomain.BEHAVIORAL,
            form=PatternForm.REACTIVE,
            confidence=0.6,
            salience=0.3,  # Consolidation is background, not urgent
            description=description,
            evidence=message,
        )

    def translate_distilled_pattern(
        self, pattern: dict[str, Any],
    ) -> PatternSignal | None:
        """Translate a distilled pattern dict into an ANCESTRAL PatternSignal.

        Called directly by the consolidation pipeline when PatternDistiller
        produces new patterns. These are already Rule-of-Three validated.
        """
        pattern_type = pattern.get("pattern_type", "unknown")
        domain_map = {
            "behavioral": PatternDomain.BEHAVIORAL,
            "state": PatternDomain.STATE,
            "temporal": PatternDomain.BEHAVIORAL,
            "causal": PatternDomain.CAUSATION,
        }

        domain = domain_map.get(pattern_type, PatternDomain.BEHAVIORAL)
        confidence = pattern.get("confidence", 0.5)
        description = pattern.get("description", f"Distilled {pattern_type} pattern")
        occurrence_count = pattern.get("occurrence_count", 3)

        return PatternSignal(
            source_system=self.source_name,
            domain=domain,
            form=PatternForm.ANCESTRAL,
            confidence=confidence,
            salience=min(1.0, confidence * 0.8),
            description=description,
            evidence=pattern,
            occurrence_count=occurrence_count,
            ttl_steps=21,  # Ancestral patterns live longer (Fibonacci)
        )
