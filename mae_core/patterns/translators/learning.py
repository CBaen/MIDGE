"""Learning translators -- CuriosityDrive.

Translates learning system events into PatternSignals.
"""

from __future__ import annotations

from typing import Any

from mae_core.patterns.pattern_signal import (
    PatternDomain,
    PatternForm,
    PatternSignal,
)


class CuriosityTranslator:
    """Translates CuriosityDrive / MemoryCoordinator novelty events.

    Listens to: memory.novel_experience
    Payload: {agent_id, novelty_score, reward}
    """

    @property
    def source_name(self) -> str:
        return "curiosity"

    @property
    def channels(self) -> list[str]:
        return ["memory.novel_experience"]

    def translate(self, channel: str, message: Any) -> PatternSignal | None:
        if not isinstance(message, dict):
            return None

        novelty = message.get("novelty_score", 0.5)
        agent_id = message.get("agent_id", "unknown")
        reward = message.get("reward", 0.0)

        description = (
            f"Novel experience by agent {agent_id}: "
            f"novelty={novelty:.2f}, reward={reward:.3f}"
        )

        return PatternSignal(
            source_system=self.source_name,
            domain=PatternDomain.NOVELTY,
            form=PatternForm.REACTIVE,
            confidence=min(1.0, novelty),
            salience=min(1.0, novelty),
            description=description,
            evidence=message,
        )
