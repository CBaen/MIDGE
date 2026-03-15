"""Bio-market wiring Tier 3 (continued): QuorumSpace, CircadianRhythm, HAVEN,
InhibitionSystem, MemoryConsolidator, CollectiveDreamPlanner, Stigmergy.

Each function wires a biological system to market EventBus channels, giving
it a real market intelligence job that maps its biological metaphor to actual
market activity.
"""
from __future__ import annotations

import logging
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

    Phase-change callbacks trigger hypothesis consolidation and excavation
    daemon steps (legitimate market jobs). The sensing worker count scaling
    is DISABLED (MIDGE: fictional physiology harms trading daemon) — during
    REST phase the real circadian multiplier drops to 0.1, cutting sensing
    workers from 12 to 3. Markets do not pause for simulated sleep; a 24/7
    trading daemon should not lose 75% of sensing capacity on a phase cycle.
    ctx._circadian_activity is pinned to 1.0 so the scheduler always sees
    full capacity regardless of phase.
    """
    circadian = getattr(ctx, "circadian_rhythm", None)
    if circadian is None:
        return 0

    def _on_phase_change(channel, data):
        msg = _parse(data)
        new_phase = msg.get("new_phase", "")
        # MIDGE: disabled — fictional physiology harms trading daemon.
        # Do NOT read circadian.get_activity_multiplier() here — that value
        # drops to 0.1 in REST phase, throttling sensing workers to 25% capacity.
        # Pin to 1.0 permanently so workers are always at full strength.
        ctx._circadian_activity = 1.0  # was: circadian.get_activity_multiplier()
        ctx._circadian_phase = new_phase
        logger.debug(
            "Circadian phase -> %s, activity multiplier pinned to 1.0 (sensing throttle disabled)",
            new_phase,
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
        flags[source] = min(1.0, flags.get(source, 0.0) + severity)

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

    Fix missing event_bus so CollectiveDreamPlanner can publish internally.

    NOTE: CH_CONVERGENCE subscription REMOVED (2026-03-14, triadic audit P1).
    The convergence callback nudged dreamer expertise weights, but the outputs
    of the collective dream planner are not consumed by any market decision
    pathway. Confirmed inert by triadic audit. EventBus reference wired so
    internal planner operations remain functional.
    """
    dream = getattr(ctx, "collective_dream", None)
    if dream is None:
        return 0

    if getattr(dream, "_bus", None) is None:
        dream._bus = bus

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
