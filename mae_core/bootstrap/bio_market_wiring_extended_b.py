"""Bio-market wiring extended, part B: Tier 4 (PearlDefense) + Tier 5 systems.

Tier 4:
  _wire_pearl_defense  — deception -> PearlDefense quarantine

Tier 5 (purpose emerges through market connection):
  _wire_respiratory    — convergence/velocity -> RespiratorySystem (O2 budget)
  _wire_thermoregulation — convergence/velocity -> ThermoregulationSystem (load as heat)
  _wire_vestibular     — convergence/prediction -> VestibularSystem (instability detection)
  _wire_proprioception — convergence/prediction -> ProprioceptionSystem (body map)
  _wire_energy_reserve — convergence/circadian -> EnergyReserve (API budget)
  _wire_predictive_field — convergence -> PredictiveField (spatial field update)
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

    During circadian REST, bank surplus energy. During ACTIVE, release
    stored energy. Low leptin = hungry = increase scanning.

    NOTE: The convergence drain (energy.release(0.5) per alert) was
    removed because convergence alerts fire faster than REST phases
    can refill, permanently locking reserves at 0.0. At 0.0 the
    organism enters starvation reflex every step, hijacking agent
    decisions before market intelligence runs. The REST-phase store
    remains so reserves can actually accumulate.
    """
    energy = getattr(ctx, "energy_reserve", None)
    if energy is None:
        return 0

    # Convergence drain REMOVED — see note above.
    # do NOT register a CH_CONVERGENCE callback here that calls energy.release().

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

    from mae_core.coordination.circadian_rhythm import CH_PHASE_CHANGE
    bus.register_callback(CH_PHASE_CHANGE, _on_phase_change)
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

    # Stable integer ID for the convergence alerter "virtual agent"
    _ALERTER_AGENT_ID = hash("convergence_alerter") & 0x7FFFFFFF

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
                agent_id=_ALERTER_AGENT_ID,
                position=pos,
                velocity=(0.0, 0.0),
                intention=direction,
                confidence=confidence,
            )
        except Exception:
            pass

    bus.register_callback(CH_CONVERGENCE, _on_convergence)
    return 1
