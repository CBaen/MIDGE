"""Bootstrap Layer 33k extended: Tier 4+5 biological system market activation.

Wire remaining biological systems to market EventBus channels. Each system
gets a real market job that maps its biological metaphor to actual market
intelligence work.

Tier 4 (market jobs exist but less direct):
  DigestiveSystem, CirculatorySystem, LymphaticSystem, Microbiome,
  RenalFilter, SenescenceManager, MorphogenesisCoordinator,
  ReproductiveSystem, PearlDefense

Tier 5 (purpose emerges through market connection):
  RespiratorySystem, ThermoregulationSystem, VestibularSystem,
  ProprioceptionSystem, EnergyReserve, PredictiveField

Note: GenerativeReplayMemory is not bootstrapped (needs state_dim/action_dim
design decisions). Deferred to a separate build.
"""
from __future__ import annotations

import json
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
        try:
            return json.loads(data)
        except (json.JSONDecodeError, TypeError):
            pass
    return {}


def wire_bio_systems_extended(ctx: SimpleNamespace) -> int:
    """Wire Tier 4+5 biological systems to market channels.

    Returns the number of systems successfully wired.
    """
    bus = getattr(ctx, "bus", None)
    if bus is None:
        return 0

    count = 0
    # Tier 4: market jobs exist but less direct
    count += _wire_digestive(ctx, bus)
    count += _wire_circulatory(ctx, bus)
    count += _wire_lymphatic(ctx, bus)
    count += _wire_microbiome(ctx, bus)
    count += _wire_renal_filter(ctx, bus)
    count += _wire_senescence(ctx, bus)
    count += _wire_morphogenesis(ctx, bus)
    count += _wire_reproductive(ctx, bus)
    count += _wire_pearl_defense(ctx, bus)
    # Tier 5: purpose emerges through connection
    count += _wire_respiratory(ctx, bus)
    count += _wire_thermoregulation(ctx, bus)
    count += _wire_vestibular(ctx, bus)
    count += _wire_proprioception(ctx, bus)
    count += _wire_energy_reserve(ctx, bus)
    count += _wire_predictive_field(ctx, bus)

    logger.info(
        "Layer 33k extended - Tier 4+5 bio activation: %d systems wired", count,
    )
    return count


# =========================================================================
# Tier 4: Market jobs exist but less direct
# =========================================================================


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
            renal.add_toxin_pattern(
                {"source": source, "type": "deception", "severity": severity},
            )
            renal.filter_item(
                item_id=f"deception_{source}_{int(time.time())}",
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
                item_id=f"conv_{ticker}_{int(time.time())}",
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
        try:
            senescence.register_system(name, creation_step=0)
        except Exception:
            pass  # may already be registered

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

    # Track recent partials with thread-safe lock
    _partial_window: list[float] = []
    _partial_lock = threading.Lock()

    def _on_partial(channel, data):
        msg = _parse(data)
        now = time.time()
        with _partial_lock:
            _partial_window.append(now)
            # Trim to last 10 minutes
            cutoff = now - 600.0
            while _partial_window and _partial_window[0] < cutoff:
                _partial_window.pop(0)
            recent_count = len(_partial_window)

        if recent_count >= 5:
            domains = msg.get("domains_seen", [])
            sig = f"partial_overflow_{'_'.join(sorted(domains))}"
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

    Convergence activity increases market pressure. The step hook
    reads ctx._market_activity_pressure to feed update_metrics().
    High pressure = organism needs more agents. Low = shed.
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

    bus.register_callback(CH_CONVERGENCE, _on_convergence)
    bus.register_callback(CH_PARTIAL_CONVERGENCE, _on_partial)
    return 1


def _wire_pearl_defense(ctx: SimpleNamespace, bus: Any) -> int:
    """PearlDefense: quarantine suspicious market signals.

    Deception events trigger validation — suspicious sources enter
    a multi-layer nacre process (quarantine → review → accept/reject).
    This is slower but more thorough than binary accept/reject.
    """
    pearl = getattr(ctx, "pearl_defense", None)
    if pearl is None:
        return 0

    from mae_core.market.channels import CH_DECEPTION_DETECTED

    def _on_deception(channel, data):
        msg = _parse(data)
        source = msg.get("source", "unknown")
        severity = msg.get("severity", 0.5)
        try:
            pearl.validate(
                source=source,
                input_type="deception_signal",
                data=msg,
                numeric_value=severity,
            )
        except Exception:
            pass

    bus.register_callback(CH_DECEPTION_DETECTED, _on_deception)
    return 1


# =========================================================================
# Tier 5: Purpose emerges through market connection
# =========================================================================


def _wire_respiratory(ctx: SimpleNamespace, bus: Any) -> int:
    """RespiratorySystem: processing throughput as oxygen.

    Each convergence alert costs metabolic oxygen to process.
    High sensing load depletes O2. Low O2 = gasping = the organism
    should throttle market sensing to recover capacity.
    """
    respiratory = getattr(ctx, "respiratory_system", None)
    if respiratory is None:
        return 0

    from mae_core.market.channels import CH_CONVERGENCE, CH_VELOCITY_ANOMALY

    def _on_convergence(channel, data):
        try:
            respiratory.consume_oxygen(0.03)
        except Exception:
            pass

    def _on_velocity(channel, data):
        msg = _parse(data)
        magnitude = msg.get("magnitude", 0.0)
        if magnitude > 2.0:
            try:
                respiratory.consume_oxygen(min(0.1, magnitude * 0.02))
            except Exception:
                pass

    bus.register_callback(CH_CONVERGENCE, _on_convergence)
    bus.register_callback(CH_VELOCITY_ANOMALY, _on_velocity)
    return 1


def _wire_thermoregulation(ctx: SimpleNamespace, bus: Any) -> int:
    """ThermoregulationSystem: computational load as temperature.

    Convergence = heat from processing. Velocity anomaly = spike.
    Overheating (too many signals at once) triggers sweating
    (shedding low-priority tasks). Cold = idle market = shivering
    (increase scanning to find signals).
    """
    thermo = getattr(ctx, "thermoregulation", None)
    if thermo is None:
        return 0

    from mae_core.market.channels import CH_CONVERGENCE, CH_VELOCITY_ANOMALY

    def _on_convergence(channel, data):
        msg = _parse(data)
        strength = msg.get("strength", 0.5)
        try:
            thermo.report_activity("market_convergence", min(1.0, strength))
        except Exception:
            pass

    def _on_velocity(channel, data):
        msg = _parse(data)
        magnitude = msg.get("magnitude", 0.0)
        try:
            thermo.report_activity("market_anomaly", min(1.0, magnitude * 0.2))
        except Exception:
            pass

    bus.register_callback(CH_CONVERGENCE, _on_convergence)
    bus.register_callback(CH_VELOCITY_ANOMALY, _on_velocity)
    return 1


def _wire_vestibular(ctx: SimpleNamespace, bus: Any) -> int:
    """VestibularSystem: detect instability in market sensing.

    Track rolling convergence rate and prediction accuracy. If either
    changes rapidly (spike or crash), the vestibular system fires
    vertigo — alerting other systems that market sensing is unstable.
    """
    vestibular = getattr(ctx, "vestibular_system", None)
    if vestibular is None:
        return 0

    from mae_core.market.channels import CH_CONVERGENCE, CH_PREDICTION_RESULT

    def _on_convergence(channel, data):
        msg = _parse(data)
        domain_count = msg.get("domain_count", 3)
        try:
            vestibular.report_metric("convergence_rate", domain_count / 12.0)
        except Exception:
            pass

    def _on_prediction(channel, data):
        msg = _parse(data)
        won = msg.get("won")
        if won is not None:
            try:
                vestibular.report_metric(
                    "prediction_accuracy", 1.0 if won else 0.0,
                )
            except Exception:
                pass

    bus.register_callback(CH_CONVERGENCE, _on_convergence)
    bus.register_callback(CH_PREDICTION_RESULT, _on_prediction)
    return 1


def _wire_proprioception(ctx: SimpleNamespace, bus: Any) -> int:
    """ProprioceptionSystem: market system positions in body map.

    Update the body map with market organ health and activity levels.
    Convergence alerts show the alerter is active and healthy.
    Prediction results show the outcome tracker's health (wins=healthy).
    """
    proprio = getattr(ctx, "proprioception", None)
    if proprio is None:
        return 0

    from mae_core.market.channels import CH_CONVERGENCE, CH_PREDICTION_RESULT

    def _on_convergence(channel, data):
        msg = _parse(data)
        strength = msg.get("strength", 0.5)
        confidence = msg.get("confidence", 0.5)
        try:
            proprio.update_position(
                "convergence_alerter", activity=strength, health=confidence,
            )
        except Exception:
            pass

    def _on_prediction(channel, data):
        msg = _parse(data)
        won = msg.get("won")
        if won is not None:
            try:
                proprio.update_position(
                    "outcome_tracker",
                    activity=1.0,
                    health=1.0 if won else 0.5,
                )
            except Exception:
                pass

    bus.register_callback(CH_CONVERGENCE, _on_convergence)
    bus.register_callback(CH_PREDICTION_RESULT, _on_prediction)
    return 1


def _wire_energy_reserve(ctx: SimpleNamespace, bus: Any) -> int:
    """EnergyReserve: API budget as metabolic energy.

    Convergence signals spend energy (processing cost). During
    circadian REST, bank surplus energy. During ACTIVE, release
    stored energy. Low leptin = hungry = increase scanning.
    """
    energy = getattr(ctx, "energy_reserve", None)
    if energy is None:
        return 0

    from mae_core.market.channels import CH_CONVERGENCE

    def _on_convergence(channel, data):
        try:
            energy.release(0.5)
        except Exception:
            pass

    def _on_phase_change(channel, data):
        msg = _parse(data)
        phase = msg.get("new_phase", "")
        try:
            if phase == "REST":
                energy.store(5.0)
            elif phase == "ACTIVE":
                energy.release(1.0)
        except Exception:
            pass

    bus.register_callback(CH_CONVERGENCE, _on_convergence)
    bus.register_callback("circadian.phase_change", _on_phase_change)
    return 1


def _wire_predictive_field(ctx: SimpleNamespace, bus: Any) -> int:
    """PredictiveField: aggregate market predictions into spatial field.

    Convergence alerts update the field with market activity. Agents
    can read the field gradient to find coordination opportunities
    and avoid duplicating work on the same tickers.
    """
    field = getattr(ctx, "predictive_field", None)
    if field is None:
        return 0

    from mae_core.market.channels import CH_CONVERGENCE

    def _ticker_position(ticker: str) -> tuple[float, float]:
        """Deterministic 2D position from ticker string."""
        h = hash(ticker) & 0xFFFFFFFF
        return ((h >> 16) / 65535.0, (h & 0xFFFF) / 65535.0)

    def _on_convergence(channel, data):
        msg = _parse(data)
        ticker = msg.get("ticker", "")
        direction = msg.get("direction", "neutral")
        confidence = msg.get("confidence", 0.5)
        if not ticker:
            return
        try:
            pos = _ticker_position(ticker)
            field.update_agent_state(
                agent_id="convergence_alerter",
                position=pos,
                velocity=(0.0, 0.0),
                intention=direction,
                confidence=confidence,
            )
        except Exception:
            pass

    bus.register_callback(CH_CONVERGENCE, _on_convergence)
    return 1
