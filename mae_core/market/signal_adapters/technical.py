"""Technical analysis signal adapters — TA indicators and session sweeps.

Converts technical analysis signals into normalized MarketSignal objects.
"""

from __future__ import annotations

from datetime import datetime

from mae_core.market.signal import MarketSignal, _ensure_datetime


def from_ta_signal(signal) -> MarketSignal:
    """Convert a TASignalBase (RSI, MACD, Bollinger, Structure, Candle) to MarketSignal.

    All TA indicators share the same adapter — they differ only in source key
    and metadata. The indicator type is encoded in signal.indicator.
    """
    event_dt = _ensure_datetime(signal.detected_at)

    # Source key = "ta_{indicator}" for Thompson key lookup
    source = f"ta_{signal.indicator}"

    # Outcome window: structure/candle are faster-acting than oscillators
    outcome_days = 7 if signal.indicator in ("structure", "candle") else 14

    return MarketSignal(
        signal_id=signal.signal_id,
        source=source,
        symbol=signal.symbol,
        asset_class="stock",
        domain="technical",
        direction=signal.direction,
        strength=signal.strength,
        confidence=signal.confidence,
        decay_rate=signal.decay_rate,
        timestamp=event_dt,
        received_at=datetime.now(),
        outcome_symbol=signal.symbol,
        outcome_window_days=outcome_days,
        raw_id=signal.signal_id,
        raw_type=type(signal).__name__,
        metadata=signal.metadata,
    )


def from_session_sweep(sweep) -> MarketSignal:
    """Convert a SessionSweepSignal to a MarketSignal.

    ICT session liquidity sweep -> tactical signal. Decay is hourly-scale
    (~18h half-life). Outcome checked by next session (1 day).
    """
    event_dt = _ensure_datetime(sweep.detected_at)

    signal_id = (
        f"session_sweep:{sweep.symbol}:{sweep.session_swept}:"
        f"{sweep.sweep_type}:{sweep.detected_at}"
    )

    # IFVG signals get separate Thompson tracking
    is_ifvg = getattr(sweep, "is_ifvg", False)
    source = "session_sweep_ifvg" if is_ifvg else "session_sweep"

    return MarketSignal(
        signal_id=signal_id,
        source=source,
        symbol=sweep.symbol,
        asset_class="futures",
        domain="technical",
        direction=sweep.direction,
        strength=sweep.strength,
        confidence=sweep.confidence,
        decay_rate=sweep.decay_rate,
        timestamp=event_dt,
        received_at=datetime.now(),
        outcome_symbol=sweep.symbol,
        outcome_window_days=1,
        raw_id=sweep.sweep_id or "",
        raw_type="SessionSweepSignal",
        metadata={
            "sweep_type": sweep.sweep_type,
            "session_swept": sweep.session_swept,
            "sweep_level": sweep.sweep_level,
            "fvg_top": sweep.fvg_top,
            "fvg_bottom": sweep.fvg_bottom,
            "entry_zone_top": sweep.entry_zone_top,
            "entry_zone_bottom": sweep.entry_zone_bottom,
            "stop_level": sweep.stop_level,
            "target_level": sweep.target_level,
            "rr_ratio": sweep.rr_ratio,
            "kill_zone": sweep.kill_zone,
            "is_ifvg": is_ifvg,
            "displacement_score": getattr(sweep, "displacement_score", 0.0),
            "fvg_atr_ratio": getattr(sweep, "fvg_atr_ratio", 0.0),
            "quality_score": getattr(sweep, "quality_score", 0.0),
        },
    )
