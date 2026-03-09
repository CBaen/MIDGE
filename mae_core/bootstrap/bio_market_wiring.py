"""Bootstrap Layer 33k: Biological system market activation.

Wire biological systems to market EventBus channels, giving each a real
market job. Each wiring function subscribes a bio system to market channels
and translates market events into system-appropriate inputs.

Tier 2 (one hop away — already hormone-connected via EndocrineSystem):
  EmotionalSystem, HomeostasisRegulator, ArousalRegulator,
  CuriosityDrive, NociceptionSystem

Tier 3 (clear market jobs, need EventBus subscriptions):
  CircadianRhythm, MetacognitionMonitor, ThreatDetector, HAVEN,
  InhibitionSystem, MemoryConsolidator, CollectiveDreamPlanner,
  QuorumSpace, Stigmergy
"""
from __future__ import annotations

import json
import logging
import time
from types import SimpleNamespace
from typing import Any

logger = logging.getLogger("midge.bootstrap")


def _parse(data: Any) -> dict:
    """Safe parse of EventBus message to dict."""
    if isinstance(data, dict):
        return data
    if isinstance(data, str):
        try:
            return json.loads(data)
        except (json.JSONDecodeError, TypeError):
            pass
    return {}


def wire_bio_systems_to_market(ctx: SimpleNamespace) -> None:
    """Activate biological systems with market intelligence jobs.

    Called after both core bio systems (Layers 2-30) and market systems
    (Layer 33) are fully initialized. All wiring is additive — existing
    bio system behavior is unchanged, market signals provide new inputs.
    """
    bus = getattr(ctx, "bus", None)
    if bus is None:
        return

    count = 0
    # Tier 2: one hop away
    count += _wire_emotional_system(ctx, bus)
    count += _wire_homeostasis(ctx, bus)
    count += _wire_arousal(ctx, bus)
    count += _wire_curiosity(ctx, bus)
    count += _wire_nociception(ctx, bus)
    # Tier 3: clear market jobs
    count += _wire_metacognition(ctx, bus)
    count += _wire_threat_detector(ctx, bus)
    count += _wire_quorum(ctx, bus)
    count += _wire_circadian(ctx, bus)
    count += _wire_haven(ctx, bus)
    count += _wire_inhibition(ctx, bus)
    count += _wire_memory_consolidator(ctx, bus)
    count += _wire_collective_dream(ctx, bus)
    count += _wire_stigmergy(ctx, bus)

    # Tier 4+5: extended wiring (separate file to prevent monolith)
    try:
        from mae_core.bootstrap.bio_market_wiring_extended import (
            wire_bio_systems_extended,
        )
        count += wire_bio_systems_extended(ctx)
    except Exception:
        logger.debug("Tier 4+5 bio wiring failed", exc_info=True)

    # Endocrine → ResourceGovernor cortisol coupling.
    # High cortisol (stress) tightens EXPLORE budgets; low cortisol relaxes them.
    # Must run after both endocrine (Layer 26) and resource_governor (Layer 33a)
    # are fully initialized.
    if hasattr(ctx, "resource_governor") and ctx.resource_governor is not None:
        endocrine = getattr(ctx, "endocrine", None)
        if endocrine is not None and hasattr(endocrine, "register_resource_governor"):
            try:
                endocrine.register_resource_governor(ctx.resource_governor)
                count += 1
                logger.debug(
                    "Layer 33k - Endocrine → ResourceGovernor cortisol coupling wired"
                )
            except Exception:
                logger.debug(
                    "Layer 33k - Endocrine → ResourceGovernor coupling failed",
                    exc_info=True,
                )

    logger.info(
        "Layer 33k - Bio-market activation: %d systems wired to market channels",
        count,
    )


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

    Bearish convergence -> elevated threat_level setpoint.
    Velocity anomaly -> elevated processing_load setpoint.
    Bullish convergence -> lower threat, higher energy.
    """
    homeo = getattr(ctx, "homeostasis_regulator", None)
    if homeo is None:
        return 0

    from mae_core.market.channels import CH_CONVERGENCE, CH_VELOCITY_ANOMALY

    def _on_convergence(channel, data):
        msg = _parse(data)
        direction = msg.get("direction", "neutral")
        strength = msg.get("strength", 0.0)
        if direction == "bearish" and strength > 0.5:
            homeo.update_current_value("threat_level", min(0.8, strength * 0.7))
        elif direction == "bullish" and strength > 0.5:
            homeo.update_current_value(
                "threat_level", max(0.05, 0.1 - strength * 0.05)
            )
            homeo.update_current_value(
                "energy_level", min(1.0, 0.7 + strength * 0.2)
            )

    def _on_velocity(channel, data):
        msg = _parse(data)
        magnitude = msg.get("magnitude", 0.0)
        if magnitude > 2.0:
            homeo.update_current_value(
                "processing_load", min(0.9, magnitude * 0.15)
            )

    bus.register_callback(CH_CONVERGENCE, _on_convergence)
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

    import threading

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


def _wire_quorum(ctx: SimpleNamespace, bus: Any) -> int:
    """QuorumSpace: organism-level vote on convergence signals.

    Convergence alerts deposit signal per ticker.
    Pattern stacks deposit confirmation signal.
    Dual confirmation deposits strong agreement signal.
    """
    quorum = getattr(ctx, "quorum_space", None)
    if quorum is None:
        return 0

    from mae_core.market.channels import (
        CH_CONVERGENCE,
        CH_DUAL_CONFIRMATION,
        CH_PATTERN_STACK_DETECTED,
    )

    def _on_convergence(channel, data):
        msg = _parse(data)
        direction = msg.get("direction", "neutral")
        strength = msg.get("strength", 0.0)
        ticker = msg.get("ticker", "")
        if ticker and direction != "neutral":
            signal_type = f"{ticker}.{direction}"
            quorum.deposit_signal(
                signal_type, "convergence_alerter", strength,
                {"domains": msg.get("domain_count", 0)},
            )

    def _on_pattern_stack(channel, data):
        msg = _parse(data)
        ticker = msg.get("ticker", "")
        direction = msg.get("direction", "")
        confidence = msg.get("confidence", 0.0)
        if ticker and direction:
            quorum.deposit_signal(
                f"{ticker}.{direction}", "pattern_archaeology", confidence,
            )

    def _on_dual_confirm(channel, data):
        msg = _parse(data)
        ticker = msg.get("ticker", "")
        direction = msg.get("direction", "")
        if ticker and direction:
            quorum.deposit_signal(
                f"{ticker}.{direction}", "dual_confirmation", 0.9,
            )

    bus.register_callback(CH_CONVERGENCE, _on_convergence)
    bus.register_callback(CH_PATTERN_STACK_DETECTED, _on_pattern_stack)
    bus.register_callback(CH_DUAL_CONFIRMATION, _on_dual_confirm)
    return 1


def _wire_circadian(ctx: SimpleNamespace, bus: Any) -> int:
    """CircadianRhythm: market-aware activity modulation.

    When circadian phase changes, store activity multiplier on ctx
    for the sensing hook to read and scale worker count.
    ACTIVE -> full sensing. CONSOLIDATION -> reduced. REST -> minimal.
    """
    circadian = getattr(ctx, "circadian_rhythm", None)
    if circadian is None:
        return 0

    def _on_phase_change(channel, data):
        msg = _parse(data)
        new_phase = msg.get("new_phase", "")
        multiplier = circadian.get_activity_multiplier()
        ctx._circadian_activity = multiplier
        ctx._circadian_phase = new_phase
        logger.debug(
            "Circadian phase -> %s, activity multiplier %.1f",
            new_phase, multiplier,
        )

    from mae_core.coordination.circadian_rhythm import CH_PHASE_CHANGE
    bus.register_callback(CH_PHASE_CHANGE, _on_phase_change)
    ctx._circadian_activity = 1.0
    ctx._circadian_phase = "ACTIVE"
    return 1


def _wire_haven(ctx: SimpleNamespace, bus: Any) -> int:
    """HAVEN: immune check on market signal sources.

    Deception events flag the source for heightened scrutiny.
    Flags accumulate — cleared only by successful outcomes.
    """
    haven = getattr(ctx, "haven", None)
    if haven is None:
        return 0

    from mae_core.market.channels import CH_DECEPTION_DETECTED, CH_PREDICTION_RESULT

    if not hasattr(ctx, "_haven_market_flags"):
        ctx._haven_market_flags = {}

    def _on_deception(channel, data):
        msg = _parse(data)
        source = msg.get("source", "unknown")
        severity = msg.get("severity", 0.3)
        flags = ctx._haven_market_flags
        flags[source] = flags.get(source, 0.0) + severity

    def _on_prediction_success(channel, data):
        msg = _parse(data)
        if msg.get("won") is True:
            # Successful prediction reduces suspicion on contributing sources
            sources = msg.get("sources", [])
            for src in sources:
                if src in ctx._haven_market_flags:
                    ctx._haven_market_flags[src] = max(
                        0.0, ctx._haven_market_flags[src] - 0.2
                    )

    bus.register_callback(CH_DECEPTION_DETECTED, _on_deception)
    bus.register_callback(CH_PREDICTION_RESULT, _on_prediction_success)
    return 1


def _wire_inhibition(ctx: SimpleNamespace, bus: Any) -> int:
    """InhibitionSystem: market-aware Go/NoGo.

    Deception events raise caution (NoGo bias).
    High-confidence convergence lowers caution (Go bias).
    Stored on ctx for agent lifecycle to read.
    """
    inhibition = getattr(ctx, "inhibition_system", None)
    if inhibition is None:
        return 0

    from mae_core.market.channels import CH_CONVERGENCE, CH_DECEPTION_DETECTED

    if not hasattr(ctx, "_market_caution"):
        ctx._market_caution = 0.0

    def _on_deception(channel, data):
        msg = _parse(data)
        severity = msg.get("severity", 0.3)
        ctx._market_caution = min(1.0, ctx._market_caution + severity * 0.4)

    def _on_convergence(channel, data):
        msg = _parse(data)
        confidence = msg.get("confidence", 0.0)
        if confidence > 0.7:
            ctx._market_caution = max(0.0, ctx._market_caution - 0.2)

    bus.register_callback(CH_DECEPTION_DETECTED, _on_deception)
    bus.register_callback(CH_CONVERGENCE, _on_convergence)
    return 1


def _wire_memory_consolidator(ctx: SimpleNamespace, bus: Any) -> int:
    """MemoryConsolidator: circadian-gated market pattern consolidation.

    CONSOLIDATION phase -> run hypothesis engine step (RSI Layer 2).
    REST phase -> run excavation daemon step (background archaeology).
    """
    hypothesis_engine = getattr(ctx, "hypothesis_engine", None)
    excavation_daemon = getattr(ctx, "excavation_daemon", None)
    if hypothesis_engine is None and excavation_daemon is None:
        return 0

    def _on_phase_change(channel, data):
        msg = _parse(data)
        new_phase = msg.get("new_phase", "")
        if new_phase == "CONSOLIDATION" and hypothesis_engine is not None:
            try:
                hypothesis_engine.step()
            except Exception:
                logger.debug("Hypothesis consolidation failed", exc_info=True)
        elif new_phase == "REST" and excavation_daemon is not None:
            try:
                excavation_daemon.step()
            except Exception:
                logger.debug("Excavation rest-cycle failed", exc_info=True)

    from mae_core.coordination.circadian_rhythm import CH_PHASE_CHANGE
    bus.register_callback(CH_PHASE_CHANGE, _on_phase_change)
    return 1


def _wire_collective_dream(ctx: SimpleNamespace, bus: Any) -> int:
    """CollectiveDreamPlanner: market-expertise-weighted dreaming.

    Fix missing event_bus. When convergence fires, nudge expertise
    weights so the collective dream reflects market-active agents.
    """
    dream = getattr(ctx, "collective_dream", None)
    if dream is None:
        return 0

    if getattr(dream, "_bus", None) is None:
        dream._bus = bus

    from mae_core.market.channels import CH_CONVERGENCE

    def _on_convergence(channel, data):
        agents = getattr(dream, "_agents", {})
        for _agent_id, dreamer in list(agents.items()):
            try:
                dreamer.expertise = min(1.0, dreamer.expertise + 0.02)
            except Exception:
                pass

    bus.register_callback(CH_CONVERGENCE, _on_convergence)
    return 1


def _wire_stigmergy(ctx: SimpleNamespace, bus: Any) -> int:
    """Stigmergy: market ticker trail markers.

    Convergence alerts deposit pheromone per ticker.
    Prediction outcomes deposit success/danger markers.
    Agents following trails converge on high-activity tickers.
    """
    stigmergy = getattr(ctx, "stigmergy", None)
    if stigmergy is None:
        return 0

    from mae_core.market.channels import CH_CONVERGENCE, CH_PREDICTION_RESULT

    def _ticker_position(ticker: str) -> tuple[float, float]:
        """Deterministic 2D position from ticker string."""
        h = hash(ticker) & 0xFFFFFFFF
        return ((h >> 16) / 65535.0, (h & 0xFFFF) / 65535.0)

    def _on_convergence(channel, data):
        msg = _parse(data)
        ticker = msg.get("ticker", "")
        direction = msg.get("direction", "neutral")
        strength = msg.get("strength", 0.0)
        if ticker and direction != "neutral":
            try:
                stigmergy.deposit_marker(
                    marker_type=f"convergence.{direction}",
                    position=_ticker_position(ticker),
                    intensity=strength,
                    depositor_id="convergence_alerter",
                    metadata={"ticker": ticker, "domains": msg.get("domain_count", 0)},
                )
            except Exception:
                logger.debug("Stigmergy convergence deposit failed", exc_info=True)

    def _on_prediction_result(channel, data):
        msg = _parse(data)
        ticker = msg.get("ticker", "")
        won = msg.get("won")
        if not ticker:
            return
        try:
            pos = _ticker_position(ticker)
            if won is True:
                stigmergy.deposit_marker(
                    "SUCCESS", pos, 0.8, "outcome_tracker", {"ticker": ticker},
                )
            elif won is False:
                stigmergy.deposit_marker(
                    "DANGER", pos, 0.6, "outcome_tracker", {"ticker": ticker},
                )
        except Exception:
            logger.debug("Stigmergy prediction deposit failed", exc_info=True)

    bus.register_callback(CH_CONVERGENCE, _on_convergence)
    bus.register_callback(CH_PREDICTION_RESULT, _on_prediction_result)
    return 1
