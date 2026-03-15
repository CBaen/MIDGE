"""Bio-market wiring Tier 2: one hop away from market (already hormone-connected).

Functions:
  _wire_emotional_system  — convergence/deception -> EmotionalSystem
  _wire_homeostasis       — convergence/velocity -> HomeostasisRegulator
  _wire_arousal           — prediction results/convergence -> ArousalRegulator
  _wire_curiosity         — partial convergence/hypothesis -> CuriosityDrive
  _wire_nociception       — deception/failure/velocity -> NociceptionSystem

Tier 3 functions (clear market jobs, need EventBus wiring):
  _wire_metacognition     — prediction results -> MetacognitionMonitor
  _wire_threat_detector   — deception -> ThreatDetector quills + sacrificeables
"""
from __future__ import annotations

import logging
import threading
import time
from types import SimpleNamespace
from typing import Any

logger = logging.getLogger("midge.bootstrap")


def _parse(data: Any) -> dict:
    """Safe parse of EventBus message to dict."""
    if isinstance(data, dict):
        return data
    if isinstance(data, str):
        import json
        try:
            return json.loads(data)
        except (json.JSONDecodeError, TypeError):
            pass
    return {}


# =========================================================================
# Tier 2: One hop away (already hormone-connected via EndocrineSystem)
# =========================================================================


def _wire_emotional_system(ctx: SimpleNamespace, bus: Any) -> int:
    """EmotionalSystem: market sentiment oracle.

    Bullish convergence -> surprise/curiosity boost.
    Bearish convergence -> fear reinforcement.
    Deception detected -> direct fear spike.
    """
    emo = getattr(ctx, "emotional_system", None)
    if emo is None:
        return 0

    from mae_core.market.channels import (
        CH_CONVERGENCE,
        CH_DECEPTION_DETECTED,
        CH_DUAL_CONFIRMATION,
    )

    def _on_convergence(channel, data):
        msg = _parse(data)
        direction = msg.get("direction", "neutral")
        strength = msg.get("strength", 0.0)
        if direction == "bullish" and strength > 0.5:
            emo._surprise_boost = min(0.5, emo._surprise_boost + strength * 0.15)
        elif direction == "bearish" and strength > 0.5:
            emo._fear_reinforcement = min(
                0.5, emo._fear_reinforcement + strength * 0.2
            )

    def _on_deception(channel, data):
        msg = _parse(data)
        severity = msg.get("severity", 0.5)
        emo._fear_reinforcement = min(0.5, emo._fear_reinforcement + severity * 0.3)

    def _on_dual_confirm(channel, data):
        emo._surprise_boost = min(0.5, emo._surprise_boost + 0.2)

    bus.register_callback(CH_CONVERGENCE, _on_convergence)
    bus.register_callback(CH_DECEPTION_DETECTED, _on_deception)
    bus.register_callback(CH_DUAL_CONFIRMATION, _on_dual_confirm)
    return 1


def _wire_homeostasis(ctx: SimpleNamespace, bus: Any) -> int:
    """HomeostasisRegulator: volatility balance detector.

    Velocity anomaly -> elevated processing_load setpoint.

    NOTE: CH_CONVERGENCE subscription REMOVED (2026-03-14, triadic audit P1).
    HomeostasisRegulator's convergence callback updated internal threat_level /
    energy_level variables that nothing market-relevant reads — confirmed inert
    by triadic audit. Velocity anomaly callback retained (legitimate load signal).
    """
    homeo = getattr(ctx, "homeostasis_regulator", None)
    if homeo is None:
        return 0

    from mae_core.market.channels import CH_VELOCITY_ANOMALY

    def _on_velocity(channel, data):
        msg = _parse(data)
        magnitude = msg.get("magnitude", 0.0)
        if magnitude > 2.0:
            homeo.update_current_value(
                "processing_load", min(0.9, magnitude * 0.15)
            )

    bus.register_callback(CH_VELOCITY_ANOMALY, _on_velocity)
    return 1


def _wire_arousal(ctx: SimpleNamespace, bus: Any) -> int:
    """ArousalRegulator: Yerkes-Dodson for trading.

    Prediction outcomes -> reward signal (win=high, loss=low).
    Convergence -> moderate reward (opportunity detected).
    """
    arousal = getattr(ctx, "arousal_regulator", None)
    if arousal is None:
        return 0

    from mae_core.market.channels import CH_CONVERGENCE, CH_PREDICTION_RESULT

    def _on_prediction_result(channel, data):
        msg = _parse(data)
        won = msg.get("won")
        if won is True:
            arousal.record_reward(1.0)
        elif won is False:
            arousal.record_reward(0.0)

    def _on_convergence(channel, data):
        msg = _parse(data)
        confidence = msg.get("confidence", 0.0)
        if confidence > 0.6:
            arousal.record_reward(confidence * 0.5)

    bus.register_callback(CH_PREDICTION_RESULT, _on_prediction_result)
    bus.register_callback(CH_CONVERGENCE, _on_convergence)
    return 1


def _wire_curiosity(ctx: SimpleNamespace, bus: Any) -> int:
    """CuriosityDrive: investigate partial convergences.

    Partial convergence -> exploration bonus (something brewing).
    Novel hypothesis discovered -> exploration bonus.
    Low-confidence pattern stack -> boost (novel territory).
    """
    curiosity = getattr(ctx, "curiosity_drive", None)
    if curiosity is None:
        return 0

    from mae_core.market.channels import (
        CH_HYPOTHESIS_DISCOVERED,
        CH_PARTIAL_CONVERGENCE,
        CH_PATTERN_STACK_DETECTED,
    )

    def _on_partial(channel, data):
        msg = _parse(data)
        n_domains = len(msg.get("domains_seen", []))
        bonus = min(0.15, n_domains * 0.05)
        curiosity.set_exploration_bonus(curiosity._exploration_bonus + bonus)

    def _on_hypothesis(channel, data):
        curiosity.set_exploration_bonus(curiosity._exploration_bonus + 0.1)

    def _on_pattern_stack(channel, data):
        msg = _parse(data)
        confidence = msg.get("confidence", 0.0)
        if confidence < 0.5:
            curiosity.set_exploration_bonus(curiosity._exploration_bonus + 0.12)

    bus.register_callback(CH_PARTIAL_CONVERGENCE, _on_partial)
    bus.register_callback(CH_HYPOTHESIS_DISCOVERED, _on_hypothesis)
    bus.register_callback(CH_PATTERN_STACK_DETECTED, _on_pattern_stack)
    return 1


def _wire_nociception(ctx: SimpleNamespace, bus: Any) -> int:
    """NociceptionSystem: pain from market failures.

    Deception detected -> acute pain (organism under attack).
    Prediction failure -> referred pain (wrong about the world).
    Velocity anomaly -> chronic pain (market instability).
    """
    noci = getattr(ctx, "nociception_system", None)
    if noci is None:
        return 0

    from mae_core.market.channels import (
        CH_DECEPTION_DETECTED,
        CH_PREDICTION_RESULT,
        CH_VELOCITY_ANOMALY,
    )

    def _on_deception(channel, data):
        msg = _parse(data)
        severity = msg.get("severity", 0.5)
        source = msg.get("source", "deception")
        noci.report_damage(f"market_deception:{source}", severity, "acute")

    def _on_prediction_fail(channel, data):
        msg = _parse(data)
        if msg.get("won") is False:
            confidence = msg.get("confidence", 0.5)
            intensity = min(0.6, confidence * 0.5)
            noci.report_damage("prediction_failure", intensity, "referred")

    def _on_velocity_anomaly(channel, data):
        msg = _parse(data)
        magnitude = msg.get("magnitude", 0.0)
        if magnitude > 3.0:
            noci.report_damage(
                "market_instability", min(0.3, magnitude * 0.05), "chronic"
            )

    bus.register_callback(CH_DECEPTION_DETECTED, _on_deception)
    bus.register_callback(CH_PREDICTION_RESULT, _on_prediction_fail)
    bus.register_callback(CH_VELOCITY_ANOMALY, _on_velocity_anomaly)
    return 1


# =========================================================================
# Tier 3: Clear market jobs, need EventBus wiring
# =========================================================================


def _wire_metacognition(ctx: SimpleNamespace, bus: Any) -> int:
    """MetacognitionMonitor: prediction confidence calibration.

    Track convergence confidence vs actual outcomes. Feeds the
    already-wired learning rate adjustment bridge.
    """
    metacog = getattr(ctx, "metacognition_monitor", None)
    if metacog is None:
        return 0

    from mae_core.market.channels import CH_PREDICTION_RESULT

    def _on_prediction_result(channel, data):
        msg = _parse(data)
        confidence = msg.get("confidence", 0.5)
        won = msg.get("won")
        if won is not None:
            actual = 1.0 if won else 0.0
            try:
                metacog.record_decision(
                    step=msg.get("step", 0),
                    predicted=confidence,
                    actual=actual,
                    decision_type="convergence_alert",
                )
            except Exception:
                logger.debug("MetacognitionMonitor record failed", exc_info=True)

    bus.register_callback(CH_PREDICTION_RESULT, _on_prediction_result)
    return 1


def _wire_threat_detector(ctx: SimpleNamespace, bus: Any) -> int:
    """ThreatDetector: market threat quills.

    Register quills that fire on deception events.
    Register sacrificeable market components for lizard autotomy.
    """
    td = getattr(ctx, "threat_detector", None)
    if td is None:
        return 0

    from mae_core.defense.threat_detector import Threat, ThreatLevel
    from mae_core.market.channels import CH_DECEPTION_DETECTED

    _deception_queue: list[dict] = []
    _deception_lock = threading.Lock()

    def _on_deception(channel, data):
        msg = _parse(data)
        with _deception_lock:
            if len(_deception_queue) < 50:
                _deception_queue.append(msg)

    bus.register_callback(CH_DECEPTION_DETECTED, _on_deception)

    def _deception_quill():
        with _deception_lock:
            if not _deception_queue:
                return None
            evt = _deception_queue.pop(0)
        severity = evt.get("severity", 0.3)
        source = evt.get("source", "unknown")
        return Threat(
            threat_id=f"deception_{source}_{int(time.time())}",
            source=f"market_deception:{source}",
            target="signal_pipeline",
            level=ThreatLevel.from_score(severity),
            score=severity,
            description=f"Deception detected in {source}",
        )

    td.register_quill(_deception_quill)

    for component, priority in [
        ("finnhub_websocket", 0.2),
        ("apewisdom_client", 0.3),
        ("fractal_resonance_detector", 0.5),
        ("pattern_completion_engine", 0.6),
    ]:
        if getattr(ctx, component, None) is not None:
            td.register_sacrificeable(component, priority)

    return 1
