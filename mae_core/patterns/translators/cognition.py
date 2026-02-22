"""Cognition translators -- WorldModel, CausalEngine, DecisionRouter.

Translates cognitive system events into PatternSignals.
"""

from __future__ import annotations

from typing import Any

from mae_core.patterns.pattern_signal import (
    PatternDomain,
    PatternForm,
    PatternSignal,
)


class WorldModelTranslator:
    """Translates WorldModel prediction events into PatternSignals.

    Listens to: cognition.prediction_made
    Payload: {uncertainty, ensemble_disagreement, reward}
    """

    @property
    def source_name(self) -> str:
        return "world_model"

    @property
    def channels(self) -> list[str]:
        return ["cognition.prediction_made"]

    def translate(self, channel: str, message: Any) -> PatternSignal | None:
        if not isinstance(message, dict):
            return None

        uncertainty = message.get("uncertainty", 0.0)
        disagreement = message.get("ensemble_disagreement", 0.0)

        # High uncertainty = prediction error = important pattern
        salience = min(1.0, max(uncertainty, disagreement))
        if salience < 0.1:
            return None  # Not interesting enough

        confidence = 1.0 - uncertainty  # We're confident about the error
        description = (
            f"World model prediction: uncertainty={uncertainty:.3f}, "
            f"ensemble_disagreement={disagreement:.3f}"
        )

        return PatternSignal(
            source_system=self.source_name,
            domain=PatternDomain.PREDICTION,
            form=PatternForm.REACTIVE,
            confidence=max(0.0, confidence),
            salience=salience,
            description=description,
            evidence=message,
        )


class CausalEngineTranslator:
    """Translates CausalEngine events into PatternSignals.

    Listens to: cognition.causal_query_result, temporal.causal_link_discovered
    """

    @property
    def source_name(self) -> str:
        return "causal_engine"

    @property
    def channels(self) -> list[str]:
        return ["cognition.causal_query_result", "temporal.causal_link_discovered"]

    def translate(self, channel: str, message: Any) -> PatternSignal | None:
        if not isinstance(message, dict):
            return None

        if channel == "cognition.causal_query_result":
            return self._translate_query(message)
        elif channel == "temporal.causal_link_discovered":
            return self._translate_link(message)
        return None

    def _translate_query(self, data: dict[str, Any]) -> PatternSignal | None:
        is_causal = data.get("is_causal", False)
        if not is_causal:
            return None  # Only report confirmed causal links

        strength = data.get("causal_strength", 0.0)
        confidence = data.get("confidence", 0.5)
        cause = data.get("cause", "?")
        effect = data.get("effect", "?")

        return PatternSignal(
            source_system=self.source_name,
            domain=PatternDomain.CAUSATION,
            form=PatternForm.REACTIVE,
            confidence=confidence,
            salience=strength,
            description=f"Causal link confirmed: {cause} -> {effect} (strength={strength:.2f})",
            evidence=data,
        )

    def _translate_link(self, data: dict[str, Any]) -> PatternSignal | None:
        cause = data.get("cause", data.get("var_a", "?"))
        effect = data.get("effect", data.get("var_b", "?"))
        strength = data.get("strength", 0.5)

        return PatternSignal(
            source_system=self.source_name,
            domain=PatternDomain.CAUSATION,
            form=PatternForm.REACTIVE,
            confidence=0.5,  # Correlations start at medium confidence
            salience=min(1.0, strength),
            description=f"Temporal causal link discovered: {cause} -> {effect}",
            evidence=data,
        )


class DecisionRouterTranslator:
    """Translates DecisionRouter events into PatternSignals.

    Listens to: cognition.decision_routed
    Payload: {decision_id, tier, stimulus, confidence, response_time_ms}
    """

    @property
    def source_name(self) -> str:
        return "decision_router"

    @property
    def channels(self) -> list[str]:
        return ["cognition.decision_routed"]

    def translate(self, channel: str, message: Any) -> PatternSignal | None:
        if not isinstance(message, dict):
            return None

        tier = message.get("tier", "unknown")
        confidence = message.get("confidence", 0.5)
        stimulus = message.get("stimulus", "unknown")

        # Only report reflex decisions (interesting behavioral pattern)
        # and low-confidence decisions (potential problem)
        if tier == "reflex" or confidence < 0.3:
            salience = 0.7 if tier == "reflex" else 0.5
            description = (
                f"Decision routed to {tier} tier: '{stimulus}' "
                f"(confidence={confidence:.2f})"
            )
            return PatternSignal(
                source_system=self.source_name,
                domain=PatternDomain.BEHAVIORAL,
                form=PatternForm.REACTIVE,
                confidence=confidence,
                salience=salience,
                description=description,
                evidence=message,
            )

        return None
