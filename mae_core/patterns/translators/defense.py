"""Defense translators -- AutoHealer, HAVEN, ThreatDetector.

Translates defense/safety system events into PatternSignals.
"""

from __future__ import annotations

from typing import Any

from mae_core.patterns.pattern_signal import (
    PatternDomain,
    PatternForm,
    PatternSignal,
)


class AutoHealerTranslator:
    """Translates AutoHealer failure events into PatternSignals.

    Listens to: healing.failure_detected
    Payload: {failure_id, failure_type, severity, affected_agents}
    """

    @property
    def source_name(self) -> str:
        return "auto_healer"

    @property
    def channels(self) -> list[str]:
        return ["healing.failure_detected"]

    def translate(self, channel: str, message: Any) -> PatternSignal | None:
        if not isinstance(message, dict):
            return None

        severity = message.get("severity", 0.5)
        failure_type = message.get("failure_type", "unknown")
        affected = message.get("affected_agents", [])

        description = (
            f"Failure detected: {failure_type} "
            f"(severity={severity:.2f}, affecting {len(affected)} agents)"
        )

        return PatternSignal(
            source_system=self.source_name,
            domain=PatternDomain.FAILURE,
            form=PatternForm.REACTIVE,
            confidence=0.9,  # AutoHealer is high-confidence
            salience=min(1.0, severity),
            description=description,
            evidence=message,
        )


class HAVENTranslator:
    """Translates HAVEN risk alert events into PatternSignals.

    Listens to: haven.risk_alert
    Payload: {agent_id, risk_score, risk_level}
    """

    @property
    def source_name(self) -> str:
        return "haven"

    @property
    def channels(self) -> list[str]:
        return ["haven.risk_alert"]

    def translate(self, channel: str, message: Any) -> PatternSignal | None:
        if not isinstance(message, dict):
            return None

        risk_score = message.get("risk_score", 0.5)
        risk_level = message.get("risk_level", "moderate")
        agent_id = message.get("agent_id", "unknown")

        description = (
            f"Risk alert for agent {agent_id}: "
            f"score={risk_score:.2f}, level={risk_level}"
        )

        return PatternSignal(
            source_system=self.source_name,
            domain=PatternDomain.THREAT,
            form=PatternForm.REACTIVE,
            confidence=0.8,
            salience=min(1.0, risk_score),
            description=description,
            evidence=message,
        )


class ThreatTranslator:
    """Translates ThreatDetector defense events into PatternSignals.

    Listens to: defense.activated
    Payload: {strategy, integrity/component/reason, action}
    """

    @property
    def source_name(self) -> str:
        return "threat_detector"

    @property
    def channels(self) -> list[str]:
        return ["defense.activated"]

    def translate(self, channel: str, message: Any) -> PatternSignal | None:
        if not isinstance(message, dict):
            return None

        strategy = message.get("strategy", "unknown")
        action = message.get("action", "unknown")

        description = (
            f"Defense activated: strategy={strategy}, action={action}"
        )

        return PatternSignal(
            source_system=self.source_name,
            domain=PatternDomain.THREAT,
            form=PatternForm.REACTIVE,
            confidence=0.9,
            salience=0.8,  # Defense activation is always important
            description=description,
            evidence=message,
        )
