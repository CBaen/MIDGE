"""Technical analysis signal adapters — TA indicators, session sweeps, order flow.

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


def from_order_flow(signal) -> MarketSignal:
    """Convert an OrderFlowSignal to a MarketSignal.

    Intraday volume-imbalance signal. Decay is fast (~2.3 days half-life).
    Outcome checked within 2 days (intraday signal).
    """
    event_dt = _ensure_datetime(signal.timestamp)

    signal_id = (
        f"order_flow:{signal.ticker}:{signal.direction}:"
        f"{signal.timestamp}"
    )

    return MarketSignal(
        signal_id=signal_id,
        source="order_flow",
        symbol=signal.ticker,
        asset_class="stock",
        domain="technical",
        direction=signal.direction,
        strength=signal.strength,
        confidence=0.5,  # Neutral prior — let Thompson learn
        decay_rate=0.30,
        timestamp=event_dt,
        received_at=datetime.now(),
        outcome_symbol=signal.ticker,
        outcome_window_days=2,
        raw_id=signal_id,
        raw_type="OrderFlowSignal",
        metadata={
            "volume_zscore": signal.volume_zscore,
            "close_position_in_range": signal.close_position_in_range,
            "consecutive_directional": signal.consecutive_directional,
            "relative_volume": signal.relative_volume,
        },
    )


def from_fractal_resonance(result) -> MarketSignal:
    """Convert a FractalResonanceResult to a MarketSignal.

    Multi-timeframe structure alignment signal. Slow decay (fractal signals
    persist over weeks). Outcome window 14 days (weekly-scale patterns).
    """
    signal_id = (
        f"fractal_resonance:{result.ticker}:{result.dominant_direction}:"
        f"{result.aligned_timeframes}tf"
    )

    return MarketSignal(
        signal_id=signal_id,
        source="fractal_resonance",
        symbol=result.ticker,
        asset_class="stock",
        domain="technical",
        direction=result.dominant_direction,
        strength=result.resonance_score,
        confidence=0.55,  # Slightly optimistic prior
        decay_rate=0.05,  # Slow decay — multi-timeframe signals persist
        timestamp=datetime.now(),
        received_at=datetime.now(),
        outcome_symbol=result.ticker,
        outcome_window_days=14,
        raw_id=signal_id,
        raw_type="FractalResonanceResult",
        metadata={
            "aligned_timeframes": result.aligned_timeframes,
            "resonance_score": result.resonance_score,
            "patterns": {
                tf: {
                    "structure": p.structure,
                    "pattern_type": p.pattern_type,
                    "strength": p.strength,
                }
                for tf, p in result.patterns_by_timeframe.items()
            },
        },
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
