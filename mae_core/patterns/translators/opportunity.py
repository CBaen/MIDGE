"""Opportunity translators -- positive events that benefit the organism.

Mae can see threats, failures, and novelty. This translator gives her
eyes for good things: validated capabilities and novel experiences that
paid off. Without this, the OPPORTUNITY domain in PatternSignal was
defined but never fed.

Biological analogy: The dopaminergic reward system -- neurons that fire
when something good happens, reinforcing the behaviors that led to it.
"""

from __future__ import annotations

from typing import Any

from mae_core.patterns.pattern_signal import (
    PatternDomain,
    PatternForm,
    PatternSignal,
)


class OpportunityTranslator:
    """Translates positive events into OPPORTUNITY PatternSignals.

    Listens to:
    - improvement.capability_validated: a confirmed new ability
    - memory.novel_experience: only when reward > 0 (novel AND beneficial)
    """

    @property
    def source_name(self) -> str:
        return "opportunity"

    @property
    def channels(self) -> list[str]:
        return ["improvement.capability_validated", "memory.novel_experience"]

    def translate(self, channel: str, message: Any) -> PatternSignal | None:
        if not isinstance(message, dict):
            return None

        if channel == "improvement.capability_validated":
            return self._translate_validated(message)
        elif channel == "memory.novel_experience":
            return self._translate_novel_reward(message)
        return None

    def _translate_validated(self, data: dict[str, Any]) -> PatternSignal | None:
        capability_id = data.get("capability_id", "unknown")
        validation_score = data.get("validation_score", 0.0)

        if validation_score < 0.3:
            return None  # Weak validation is not a real opportunity

        return PatternSignal(
            source_system=self.source_name,
            domain=PatternDomain.OPPORTUNITY,
            form=PatternForm.REACTIVE,
            confidence=min(1.0, validation_score),
            salience=min(0.7, 0.3 + validation_score * 0.4),
            description=(
                f"Opportunity: validated capability {capability_id} "
                f"(score={validation_score:.2f})"
            ),
            evidence=data,
        )

    def _translate_novel_reward(self, data: dict[str, Any]) -> PatternSignal | None:
        reward = data.get("reward", 0.0)
        if reward <= 0:
            return None  # Only positive novel experiences are opportunities

        novelty = data.get("novelty_score", 0.5)
        agent_id = data.get("agent_id", "unknown")

        return PatternSignal(
            source_system=self.source_name,
            domain=PatternDomain.OPPORTUNITY,
            form=PatternForm.REACTIVE,
            confidence=min(1.0, novelty),
            salience=min(0.7, 0.3 + reward),
            description=(
                f"Opportunity: novel experience by agent {agent_id} "
                f"with positive reward={reward:.3f}"
            ),
            evidence=data,
        )
