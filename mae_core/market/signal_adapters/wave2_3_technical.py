"""wave2_3_technical.py - Technical/market Wave 2+3 signal adapters.

Extracted from wave2_3.py. Converts FinViz unusual volume, FinViz short
squeeze, Massive/Polygon snapshots, economic suppression events, economic
surprise signals, and Finnhub real-time WebSocket signals to MarketSignal objects.
"""

from __future__ import annotations

from datetime import datetime

from mae_core.market.signal import MarketSignal, _ensure_datetime


def from_finviz_unusual_volume(uv) -> MarketSignal:
    """Convert an UnusualVolume (FinViz screener) to a MarketSignal.

    Volume anomalies signal informed participation. The actual field on the
    dataclass is `volume_ratio` (volume / avg_volume). Direction follows the
    intraday price change (±2% gate). Strength scales: 4× average = full.
    """
    ticker = getattr(uv, "ticker", "") or ""
    change_pct = getattr(uv, "change_pct", 0.0) or 0.0
    # `volume_ratio` is the actual field name on UnusualVolume dataclass
    relative_volume = getattr(uv, "volume_ratio", 1.0) or 1.0

    if change_pct > 2.0:
        direction = "bullish"
    elif change_pct < -2.0:
        direction = "bearish"
    else:
        direction = "neutral"

    # relative_volume of 4.0 (4× average) = full strength; scale from 1.0 baseline
    strength = min(1.0, (relative_volume - 1.0) / 3.0)

    signal_id = f"finviz_uvol:{ticker}:{datetime.now().date()}"

    return MarketSignal(
        signal_id=signal_id,
        source="finviz_unusual_volume",
        symbol=ticker,
        asset_class="stock",
        domain="technical",
        direction=direction,
        strength=strength,
        confidence=0.50,
        decay_rate=0.30,
        timestamp=datetime.now(),
        received_at=datetime.now(),
        outcome_symbol=ticker,
        outcome_window_days=5,
        raw_id="",
        raw_type="UnusualVolume",
        metadata={
            "company": getattr(uv, "company", ""),
            "sector": getattr(uv, "sector", ""),
            "price": getattr(uv, "price", 0.0),
            "change_pct": change_pct,
            "volume": getattr(uv, "volume", 0),
            "avg_volume": getattr(uv, "avg_volume", 0),
            "volume_ratio": relative_volume,
        },
    )


def from_finviz_short_squeeze(sq) -> MarketSignal:
    """Convert a ShortSqueezeCandidate (FinViz screener) to a MarketSignal.

    High short float creates mechanical upward pressure on any positive catalyst.
    Direction is always bullish — squeeze pressure is inherently upward.
    Strength scales to 40% short float (extreme squeeze setup).
    """
    ticker = getattr(sq, "ticker", "") or ""
    short_float_pct = getattr(sq, "short_float_pct", 0.0) or 0.0

    strength = min(1.0, short_float_pct / 40.0)

    signal_id = f"finviz_squeeze:{ticker}:{datetime.now().date()}"

    return MarketSignal(
        signal_id=signal_id,
        source="finviz_short_squeeze",
        symbol=ticker,
        asset_class="stock",
        domain="institutional",
        direction="bullish",
        strength=strength,
        confidence=0.50,
        decay_rate=0.10,
        timestamp=datetime.now(),
        received_at=datetime.now(),
        outcome_symbol=ticker,
        outcome_window_days=14,
        raw_id="",
        raw_type="ShortSqueezeCandidate",
        metadata={
            "company": getattr(sq, "company", ""),
            "sector": getattr(sq, "sector", ""),
            "price": getattr(sq, "price", 0.0),
            "short_float_pct": short_float_pct,
            "short_ratio": getattr(sq, "short_ratio", 0.0),
            "float_shares": getattr(sq, "float_shares", 0),
            "short_interest": getattr(sq, "short_interest", 0),
        },
    )


def from_finnhub_realtime(signal_dict) -> MarketSignal:
    """Convert a Finnhub WebSocket signal dict to a MarketSignal.

    Signal dicts are emitted by FinnhubWebSocket._emit_signal() with keys:
    signal_type, symbol, price, value, threshold, direction, timestamp.

    Volume spike strength scales relative to threshold (3× threshold = full).
    Rapid move strength scales to 3% move (= full). Real-time signals decay
    within hours (decay_rate=0.50).
    """
    signal_type = signal_dict.get("signal_type", "")
    symbol = signal_dict.get("symbol", "") or ""
    direction = signal_dict.get("direction", "neutral")
    value = signal_dict.get("value", 0.0) or 0.0
    threshold = signal_dict.get("threshold", 1.0) or 1.0
    price = signal_dict.get("price", 0.0)
    timestamp = signal_dict.get("timestamp", datetime.now())

    if signal_type == "volume_spike":
        strength = min(1.0, value / (threshold * 3.0))
    elif signal_type == "rapid_move":
        strength = min(1.0, abs(value) / 0.03)
    else:
        strength = 0.5

    event_dt = _ensure_datetime(timestamp) if not isinstance(timestamp, datetime) else timestamp

    signal_id = f"finnhub_rt:{symbol}:{signal_type}:{event_dt.strftime('%Y%m%dT%H%M%S')}"

    return MarketSignal(
        signal_id=signal_id,
        source="finnhub_realtime",
        symbol=symbol,
        asset_class="stock",
        domain="technical",
        direction=direction,
        strength=strength,
        confidence=0.50,
        decay_rate=0.50,
        timestamp=event_dt,
        received_at=datetime.now(),
        outcome_symbol=symbol,
        outcome_window_days=1,
        raw_id="",
        raw_type="FinnhubWebSocketSignal",
        metadata={
            "signal_type": signal_type,
            "price": price,
            "value": value,
            "threshold": threshold,
        },
    )


def from_suppression_event(event) -> MarketSignal:
    """Convert an EconomicEvent (economic_calendar) to a suppression MarketSignal.

    Suppression signals are neutral — they don't predict direction but flag
    that scheduled volatility makes other signals unreliable.  The convergence
    alerter reads `metadata["suppress"]` to apply a confidence penalty.
    Confidence is high (0.70) because these events are scheduled in advance.
    """
    name = getattr(event, "name", "") or ""
    impact = getattr(event, "impact", "low") or "low"
    event_date = getattr(event, "event_date", None)
    description = getattr(event, "description", "") or ""

    if impact == "high":
        strength = 0.8
    elif impact == "medium":
        strength = 0.5
    else:
        strength = 0.3

    event_dt = _ensure_datetime(str(event_date)) if event_date is not None else datetime.now()

    signal_id = f"econ_suppress:{name}:{event_date}"

    return MarketSignal(
        signal_id=signal_id,
        source="economic_calendar",
        symbol="",
        asset_class="macro",
        domain="events",
        direction="neutral",
        strength=strength,
        confidence=0.70,
        decay_rate=0.50,
        timestamp=event_dt,
        received_at=datetime.now(),
        outcome_symbol="SPY",
        outcome_window_days=3,
        raw_id="",
        raw_type="EconomicEvent",
        metadata={
            "name": name,
            "impact": impact,
            "description": description,
            "suppress": True,
        },
    )


def from_binance_funding(rate) -> MarketSignal:
    """Convert a BinanceFundingClient FundingRate to a MarketSignal.

    Funding rate is the cost of holding perpetual futures long vs short.
    Positive = longs pay shorts (overleveraged long, contrarian bearish).
    Negative = shorts pay longs (overleveraged short, contrarian bullish).
    Uses FundingRate's built-in direction/strength properties.
    """
    symbol = getattr(rate, "symbol", "") or ""
    funding_rate = getattr(rate, "rate", 0.0) or 0.0
    ts = getattr(rate, "timestamp", None)
    mark_price = getattr(rate, "mark_price", None)

    direction = getattr(rate, "direction", "neutral")
    strength = getattr(rate, "strength", 0.1)

    event_dt = ts if isinstance(ts, datetime) else datetime.now()

    # Strip USDT suffix for clean ticker
    clean_symbol = symbol.replace("USDT", "").replace("BUSD", "").replace("PERP", "")
    signal_id = f"binance_funding:{symbol}:{event_dt.strftime('%Y%m%dT%H%M')}"

    return MarketSignal(
        signal_id=signal_id,
        source="binance_funding",
        symbol=clean_symbol,
        asset_class="crypto",
        domain="positioning",
        direction=direction,
        strength=strength,
        confidence=0.55,
        decay_rate=0.70,
        timestamp=event_dt,
        received_at=datetime.now(),
        outcome_symbol=clean_symbol + "-USD",
        outcome_window_days=3,
        raw_id=symbol,
        raw_type="FundingRate",
        metadata={
            "symbol": symbol,
            "funding_rate": funding_rate,
            "funding_rate_pct": funding_rate * 100,
            "annualized_rate": funding_rate * 3 * 365 * 100,
            "mark_price": mark_price,
        },
    )


def from_massive_snapshot(snapshot) -> MarketSignal:
    """Convert a TickerSnapshot (Massive/Polygon) to a MarketSignal.

    Generates signals from daily return magnitude, volume anomalies, and
    price gaps.  Free tier provides institutional-grade EOD data that
    complements yfinance (different data source = domain diversity).

    Signal priority:
    - Volume anomaly (volume_ratio > 2.0): strongest indicator
    - Large daily return (>2%): directional conviction
    - Gap (>1% open vs prev close): overnight news impact
    Picks the strongest signal type for this snapshot.
    """
    ticker = getattr(snapshot, "ticker", "") or ""
    daily_return = getattr(snapshot, "daily_return_pct", 0.0) or 0.0
    volume_ratio = getattr(snapshot, "volume_ratio", 0.0) or 0.0
    gap_pct = getattr(snapshot, "gap_pct", 0.0) or 0.0
    close = getattr(snapshot, "close", 0.0) or 0.0
    prev_close = getattr(snapshot, "prev_close", 0.0) or 0.0
    volume = getattr(snapshot, "volume", 0.0) or 0.0
    bar_date = getattr(snapshot, "bar_date", "") or ""

    # Determine signal type by strongest feature
    vol_strength = min(1.0, max(0.0, (volume_ratio - 1.0) / 3.0)) if volume_ratio > 1.5 else 0.0
    move_strength = min(1.0, abs(daily_return) / 5.0) if abs(daily_return) > 1.0 else 0.0
    gap_strength = min(1.0, abs(gap_pct) / 3.0) if abs(gap_pct) > 0.5 else 0.0

    strength = max(vol_strength, move_strength, gap_strength)
    if strength <= 0.0:
        # Sub-threshold — still create signal but with minimal strength
        strength = 0.1

    # Pick dominant signal type for the ID
    if vol_strength >= move_strength and vol_strength >= gap_strength:
        signal_type = "volume_anomaly"
    elif gap_strength >= move_strength:
        signal_type = "gap"
    else:
        signal_type = "price_move"

    # Direction from daily return
    if daily_return > 1.0:
        direction = "bullish"
    elif daily_return < -1.0:
        direction = "bearish"
    else:
        direction = "neutral"

    signal_id = f"massive:{signal_type}:{ticker}:{bar_date}"

    return MarketSignal(
        signal_id=signal_id,
        source="massive_snapshot",
        symbol=ticker,
        asset_class="stock",
        domain="technical",
        direction=direction,
        strength=strength,
        confidence=0.55,
        decay_rate=0.30,
        timestamp=datetime.now(),
        received_at=datetime.now(),
        outcome_symbol=ticker,
        outcome_window_days=5,
        raw_id="",
        raw_type="TickerSnapshot",
        metadata={
            "close": close,
            "prev_close": prev_close,
            "daily_return_pct": daily_return,
            "volume": volume,
            "volume_ratio": volume_ratio,
            "gap_pct": gap_pct,
            "signal_type": signal_type,
            "vwap": getattr(snapshot, "vwap", 0.0),
            "high": getattr(snapshot, "high", 0.0),
            "low": getattr(snapshot, "low", 0.0),
            "data_source": "massive_polygon",
        },
    )


def from_economic_surprise(event) -> MarketSignal:
    """Convert a Finnhub EconomicEvent with actual/estimate data to a surprise MarketSignal.

    Economic surprise — actual minus consensus estimate — is one of the most
    tradeable macro signals.  A positive surprise on growth indicators (NFP,
    GDP, PMI, retail sales) is bullish; a positive surprise on inflation
    indicators (CPI, PPI, PCE) is bearish because it raises rate-hike odds.

    Direction logic:
      - Growth indicators: actual > estimate → bullish, actual < estimate → bearish
      - Inflation indicators: actual > estimate → bearish (higher rates), vice versa

    Threshold: fires only when |surprise| > 0.5% to filter measurement noise.
    Strength: scales linearly, full strength at 10% surprise.
    Domain: "macro" — distinct from "events" used by from_economic_event so both
    can contribute to convergence independently.
    """
    actual = getattr(event, "actual", None)
    estimate = getattr(event, "estimate", None)
    previous = getattr(event, "previous", None)
    event_name = getattr(event, "event", "") or ""
    country = getattr(event, "country", "") or ""
    impact = getattr(event, "impact", "low") or "low"
    unit = getattr(event, "unit", "") or ""
    event_date = getattr(event, "date", None)

    # Require both actual and estimate to compute surprise
    if actual is None or estimate is None or estimate == 0:
        return None  # type: ignore[return-value]

    surprise = (actual - estimate) / abs(estimate)

    # Filter noise — only fire when surprise magnitude exceeds 0.5%
    if abs(surprise) < 0.005:
        return None  # type: ignore[return-value]

    # Classify as inflation or growth to determine direction polarity
    _INFLATION_KEYWORDS = {"CPI", "PPI", "PCE", "Inflation", "Core", "HICP"}
    is_inflation = any(kw.lower() in event_name.lower() for kw in _INFLATION_KEYWORDS)

    if is_inflation:
        # Inflation beats estimate → bearish (rate-hike pressure)
        direction = "bearish" if surprise > 0 else "bullish"
    else:
        # Growth beats estimate → bullish
        direction = "bullish" if surprise > 0 else "bearish"

    # Strength: scales to 10% surprise = full strength
    strength = min(1.0, abs(surprise) / 0.10)

    # Confidence scales with impact level
    if impact == "high":
        confidence = 0.65
    elif impact == "medium":
        confidence = 0.55
    else:
        confidence = 0.45

    event_dt = _ensure_datetime(str(event_date)) if event_date is not None else datetime.now()

    signal_id = f"econ_surprise:{country}:{event_name}:{event_date}"

    return MarketSignal(
        signal_id=signal_id,
        source="economic_surprise",
        symbol="",
        asset_class="macro",
        domain="macro",
        direction=direction,
        strength=strength,
        confidence=confidence,
        decay_rate=0.50,  # Surprises price in fast — 1-2 day half-life
        timestamp=event_dt,
        received_at=datetime.now(),
        outcome_symbol="SPY",
        outcome_window_days=3,
        raw_id="",
        raw_type="EconomicEvent",
        metadata={
            "event": event_name,
            "country": country,
            "impact": impact,
            "actual": actual,
            "estimate": estimate,
            "previous": previous,
            "unit": unit,
            "surprise_pct": round(surprise * 100, 3),
            "is_inflation_indicator": is_inflation,
        },
    )


def from_kalshi_mover(mover) -> MarketSignal:
    """Convert a KalshiMover (prediction market probability shift) to MarketSignal.

    Kalshi prices ARE probabilities (1-99 cents = 1-99%). A 10% price move
    means crowd consensus shifted by 10 percentage points. This is a powerful
    independent signal — thousands of traders pricing real money on outcomes.

    Domain: "prediction_market" — intentionally a NEW domain (not "macro") to
    maximize convergence diversity.
    """
    market = getattr(mover, "market", None)
    if market is None:
        market = mover

    ticker = getattr(market, "ticker", "") or ""
    title = getattr(market, "title", "") or ""
    event_ticker = getattr(market, "event_ticker", "") or ""
    yes_price = getattr(market, "yes_price", 0.5) or 0.5
    volume = getattr(market, "volume_24h", 0) or 0
    category = getattr(market, "category", "other") or "other"

    price_change = getattr(mover, "price_change", 0.0) or 0.0
    direction = getattr(mover, "direction", "neutral") or "neutral"
    strength = getattr(mover, "strength", 0.3) or 0.3

    confidence = min(0.75, 0.45 + (volume / 50000) * 0.30)

    signal_id = f"kalshi:{ticker}:{datetime.now().strftime('%Y%m%dT%H%M')}"

    return MarketSignal(
        signal_id=signal_id,
        source="kalshi_market",
        symbol=ticker,
        asset_class="prediction_market",
        domain="prediction_market",
        direction=direction,
        strength=strength,
        confidence=round(confidence, 3),
        decay_rate=0.15,
        timestamp=datetime.now(),
        received_at=datetime.now(),
        outcome_symbol=ticker,
        outcome_window_days=7,
        raw_id=event_ticker,
        raw_type="KalshiMover",
        metadata={
            "title": title,
            "event_ticker": event_ticker,
            "yes_price": yes_price,
            "price_change_pct": round(price_change * 100, 2),
            "volume_24h": volume,
            "category": category,
            "symbol": ticker,
        },
    )
