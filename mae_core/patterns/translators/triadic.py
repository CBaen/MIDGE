"""Triadic Pattern Translator - Upward propagation of tissue-level consensus.

When agents in a triad reach consensus (2/3 agreement), they publish
to EventBus channel `pattern.triadic_correlation`. This translator
listens for those events and converts them into PatternSignals for
the organism-level PatternBus.

Biological analogy: When enough cells in a tissue agree on a stimulus,
the tissue sends a signal to the organ level. This translator is the
bridge between tissue and organ.
"""

from __future__ import annotations

from typing import Any

from mae_core.patterns.pattern_signal import (
    PatternDomain,
    PatternForm,
    PatternSignal,
)


class TriadicPatternTranslator:
    """Converts triadic consensus events into PatternBus signals."""

    @property
    def source_name(self) -> str:
        return "triadic_consensus"

    @property
    def channels(self) -> list[str]:
        return ["pattern.triadic_correlation"]

    def translate(self, channel: str, message: Any) -> PatternSignal | None:
        if not isinstance(message, dict):
            return None

        domain_str = message.get("domain")
        if domain_str is None:
            return None

        try:
            domain = PatternDomain(domain_str)
        except ValueError:
            return None

        voters = message.get("voters", [])
        triad_size = message.get("triad_size", 3)
        ratio = message.get("agreement_ratio", 0.67)

        return PatternSignal(
            source_system=self.source_name,
            domain=domain,
            form=PatternForm.CORRELATED,
            confidence=min(1.0, 0.5 + ratio * 0.3),
            salience=min(0.6, 0.3 + 0.1 * len(voters)),
            description=(
                f"Triadic consensus: {domain_str} from "
                f"{len(voters)}/{triad_size} agents"
            ),
            evidence=message,
        )
