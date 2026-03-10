"""Bio-market wiring extended, part A: Tier 4 systems (market jobs exist but less direct).

Functions:
  _wire_digestive    — convergence/partial -> DigestiveSystem (data as nutrients)
  _wire_circulatory  — convergence/velocity -> CirculatorySystem (attention resources)
  _wire_lymphatic    — failed predictions/deception -> LymphaticSystem (waste cleanup)
  _wire_microbiome   — convergence/velocity/patterns -> Microbiome (pre-processing)
  _wire_renal_filter — deception/convergence -> RenalFilter (toxin filtering)
  _wire_senescence   — convergence/prediction -> SenescenceManager (wear tracking)
  _wire_morphogenesis — partial convergence/hypothesis -> MorphogenesisCoordinator
  _wire_reproductive — convergence/partial -> ReproductiveSystem (population scaling)
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
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


def _wire_digestive(ctx: SimpleNamespace, bus: Any) -> int:
    """DigestiveSystem: data = nutrients.

    Convergence alerts are high-nutrition data (worth processing).
    Partial convergences are lower value (still worth ingesting).
    The digestive system's energy budget gates how much the organism
    can process — overfed = reject low-value signals.
    """
    digestive = getattr(ctx, "digestive_system", None)
    if digestive is None:
        return 0

    from mae_core.market.channels import CH_CONVERGENCE, CH_PARTIAL_CONVERGENCE

    def _on_convergence(channel, data):
        msg = _parse(data)
        strength = msg.get("strength", 0.5)
        ticker = msg.get("ticker", "unknown")
        try:
            digestive.ingest(
                source=f"convergence:{ticker}",
                content=msg,
                energy_cost=0.3,
                nutritional_value=strength,
            )
        except Exception:
            pass

    def _on_partial(channel, data):
        msg = _parse(data)
        try:
            digestive.ingest(
                source="partial_convergence",
                content=msg,
                energy_cost=0.1,
                nutritional_value=0.3,
            )
        except Exception:
            pass

    bus.register_callback(CH_CONVERGENCE, _on_convergence)
    bus.register_callback(CH_PARTIAL_CONVERGENCE, _on_partial)
    return 1


def _wire_circulatory(ctx: SimpleNamespace, bus: Any) -> int:
    """CirculatorySystem: distribute attention resources by market priority.

    Convergence alerts request attention resources (high urgency).
    Velocity anomalies request compute resources (investigation needed).
    The circulatory system's heart rate rises under load — observable
    by thermoregulation and other body-awareness systems.
    """
    circulatory = getattr(ctx, "circulatory_system", None)
    if circulatory is None:
        return 0

    from mae_core.market.channels import CH_CONVERGENCE, CH_VELOCITY_ANOMALY

    def _on_convergence(channel, data):
        msg = _parse(data)
        strength = msg.get("strength", 0.5)
        try:
            circulatory.request_resource(
                "convergence_alerter", "attention", strength, urgency=0.8,
            )
        except Exception:
            pass

    def _on_velocity(channel, data):
        msg = _parse(data)
        magnitude = msg.get("magnitude", 0.0)
        if magnitude > 1.5:
            try:
                circulatory.request_resource(
                    "velocity_detector", "compute",
                    min(1.0, magnitude * 0.2), urgency=0.7,
                )
            except Exception:
                pass

    bus.register_callback(CH_CONVERGENCE, _on_convergence)
    bus.register_callback(CH_VELOCITY_ANOMALY, _on_velocity)
    return 1


def _wire_lymphatic(ctx: SimpleNamespace, bus: Any) -> int:
    """LymphaticSystem: clean up failed prediction waste.

    Failed predictions = expired memory waste. The organism must
    metabolize wrong beliefs to stay healthy. Deception events
    generate orphan subscriptions that need cleanup.
    """
    lymphatic = getattr(ctx, "lymphatic_system", None)
    if lymphatic is None:
        return 0

    from mae_core.market.channels import CH_DECEPTION_DETECTED, CH_PREDICTION_RESULT

    def _on_prediction_fail(channel, data):
        msg = _parse(data)
        if msg.get("won") is False:
            ticker = msg.get("ticker", "unknown")
            try:
                lymphatic.collect_waste(
                    source="convergence_alerter",
                    waste_type="expired_memory",
                    item_data={"ticker": ticker, "type": "failed_prediction"},
                    current_step=msg.get("step", 0),
                )
            except Exception:
                pass

    def _on_deception(channel, data):
        msg = _parse(data)
        source = msg.get("source", "unknown")
        try:
            lymphatic.collect_waste(
                source=source,
                waste_type="orphan_subscription",
                item_data={"type": "deception_cleanup", "source": source},
                current_step=0,
            )
        except Exception:
            pass

    bus.register_callback(CH_PREDICTION_RESULT, _on_prediction_fail)
    bus.register_callback(CH_DECEPTION_DETECTED, _on_deception)
    return 1


def _wire_microbiome(ctx: SimpleNamespace, bus: Any) -> int:
    """Microbiome: microbial strains pre-process market data.

    Convergence alerts are complex data routed to the decomposer strain.
    Velocity anomalies are unusual readings routed to the detector strain.
    Pattern stacks are weak signals routed to the amplifier strain.
    """
    microbiome = getattr(ctx, "microbiome", None)
    if microbiome is None:
        return 0

    from mae_core.market.channels import (
        CH_CONVERGENCE,
        CH_PATTERN_STACK_DETECTED,
        CH_VELOCITY_ANOMALY,
    )

    def _on_convergence(channel, data):
        msg = _parse(data)
        try:
            microbiome.process_input("complex", msg)
        except Exception:
            pass

    def _on_velocity(channel, data):
        msg = _parse(data)
        try:
            microbiome.process_input("anomaly", msg)
        except Exception:
            pass

    def _on_pattern_stack(channel, data):
        msg = _parse(data)
        confidence = msg.get("confidence", 0.0)
        if confidence < 0.5:
            try:
                microbiome.process_input("weak_signal", msg)
            except Exception:
                pass

    bus.register_callback(CH_CONVERGENCE, _on_convergence)
    bus.register_callback(CH_VELOCITY_ANOMALY, _on_velocity)
    bus.register_callback(CH_PATTERN_STACK_DETECTED, _on_pattern_stack)
    return 1


def _wire_renal_filter(ctx: SimpleNamespace, bus: Any) -> int:
    """RenalFilter: filter corrupted/toxic market data.

    Deception events teach new toxin patterns so future data from
    that source gets scrutinized. Convergence signals are filtered
    for integrity before reaching downstream systems.
    """
    renal = getattr(ctx, "renal_filter", None)
    if renal is None:
        return 0

    from mae_core.market.channels import CH_CONVERGENCE, CH_DECEPTION_DETECTED

    def _on_deception(channel, data):
        msg = _parse(data)
        source = msg.get("source", "unknown")
        severity = msg.get("severity", 0.5)
        try:
            renal.add_toxin_pattern(f"deception:{source}:{severity:.1f}")
            renal.filter_item(
                item_id=f"deception_{source}_{time.monotonic_ns()}",
                source=source,
                data=msg,
            )
        except Exception:
            pass

    def _on_convergence(channel, data):
        msg = _parse(data)
        ticker = msg.get("ticker", "unknown")
        try:
            result = renal.filter_item(
                item_id=f"conv_{ticker}_{time.monotonic_ns()}",
                source="convergence_alerter",
                data=msg,
            )
            if hasattr(result, "verdict") and result.verdict == "toxic":
                logger.warning("RenalFilter: toxic convergence signal for %s", ticker)
        except Exception:
            pass

    bus.register_callback(CH_DECEPTION_DETECTED, _on_deception)
    bus.register_callback(CH_CONVERGENCE, _on_convergence)
    return 1


def _wire_senescence(ctx: SimpleNamespace, bus: Any) -> int:
    """SenescenceManager: track aging of market subsystems.

    Register key market systems on wiring, then report activity
    whenever they fire. Systems that stop firing accumulate wear.
    High wear triggers rejuvenation or retirement.
    """
    senescence = getattr(ctx, "senescence", None)
    if senescence is None:
        return 0

    from mae_core.market.channels import CH_CONVERGENCE, CH_PREDICTION_RESULT

    # Register market systems for wear tracking
    market_systems = [
        "convergence_alerter", "thompson_sampler", "velocity_detector",
        "pattern_watcher", "outcome_tracker",
    ]
    for name in market_systems:
        senescence.register_system(name, creation_step=0)

    def _on_convergence(channel, data):
        try:
            senescence.report_activity("convergence_alerter", active=True)
        except Exception:
            pass

    def _on_prediction(channel, data):
        try:
            senescence.report_activity("outcome_tracker", active=True)
        except Exception:
            pass

    bus.register_callback(CH_CONVERGENCE, _on_convergence)
    bus.register_callback(CH_PREDICTION_RESULT, _on_prediction)
    return 1


def _wire_morphogenesis(ctx: SimpleNamespace, bus: Any) -> int:
    """MorphogenesisCoordinator: spawn investigation organs.

    When partial convergences pile up beyond a threshold, spawn a
    new investigation organ to handle the overflow. Novel hypotheses
    increase growth rate temporarily.
    """
    morph = getattr(ctx, "morph_coordinator", None)
    if morph is None:
        return 0

    from mae_core.market.channels import (
        CH_HYPOTHESIS_DISCOVERED,
        CH_PARTIAL_CONVERGENCE,
    )
    from mae_core.morphogenesis.organ_builder import ProblemSignature

    # Track recent partials with thread-safe deque
    _partial_window: deque[float] = deque(maxlen=100)
    _partial_lock = threading.Lock()

    def _on_partial(channel, data):
        msg = _parse(data)
        now = time.time()
        with _partial_lock:
            _partial_window.append(now)
            # Trim to last 10 minutes
            cutoff = now - 600.0
            while _partial_window and _partial_window[0] < cutoff:
                _partial_window.popleft()
            recent_count = len(_partial_window)

        if recent_count >= 5:
            domains = msg.get("domains_seen", [])
            sig_id = f"partial_overflow_{'_'.join(sorted(domains))}"
            sig = ProblemSignature(
                signature_id=sig_id,
                coordination_level=0.7,
                exploration_level=0.8,
                complexity=len(domains) / 12.0,
                risk_level=0.2,
                temporal_pattern="episodic",
                domain="market_investigation",
                metadata={"domains": sorted(domains)},
            )
            try:
                morph.handle_novel_problem(sig, f"partial_investigation_{len(domains)}")
            except Exception:
                logger.debug("Morphogenesis spawn failed", exc_info=True)

    def _on_hypothesis(channel, data):
        try:
            morph.set_growth_rate(1.3)
        except Exception:
            pass

    bus.register_callback(CH_PARTIAL_CONVERGENCE, _on_partial)
    bus.register_callback(CH_HYPOTHESIS_DISCOVERED, _on_hypothesis)
    return 1


def _wire_reproductive(ctx: SimpleNamespace, bus: Any) -> int:
    """ReproductiveSystem: market load drives agent population scaling.

    Convergence activity increases market pressure (decays each call).
    Pressure is consumed by `consume_market_pressure()` which the
    market_hooks step hook calls to feed update_metrics().
    """
    repro = getattr(ctx, "reproductive_system", None)
    if repro is None:
        return 0

    from mae_core.market.channels import CH_CONVERGENCE, CH_PARTIAL_CONVERGENCE

    if not hasattr(ctx, "_market_activity_pressure"):
        ctx._market_activity_pressure = 0.0

    _pressure_lock = threading.Lock()

    def _on_convergence(channel, data):
        msg = _parse(data)
        strength = msg.get("strength", 0.5)
        with _pressure_lock:
            ctx._market_activity_pressure = min(
                1.0, ctx._market_activity_pressure + strength * 0.1,
            )

    def _on_partial(channel, data):
        with _pressure_lock:
            ctx._market_activity_pressure = min(
                1.0, ctx._market_activity_pressure + 0.05,
            )

    def consume_market_pressure() -> float:
        """Read and decay pressure. Called from step hook."""
        with _pressure_lock:
            val = ctx._market_activity_pressure
            ctx._market_activity_pressure = max(0.0, val * 0.9 - 0.01)
            return val

    ctx._consume_market_pressure = consume_market_pressure

    bus.register_callback(CH_CONVERGENCE, _on_convergence)
    bus.register_callback(CH_PARTIAL_CONVERGENCE, _on_partial)
    return 1
