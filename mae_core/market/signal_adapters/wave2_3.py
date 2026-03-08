"""Wave 2+3 signal adapters — crypto, OpenInsider, 13F, FinViz, Finnhub real-time, suppression.

Converts Wave 2 (real-time + crypto) and Wave 3 (data enrichment) sources
into normalized MarketSignal objects.
"""

from __future__ import annotations

from datetime import datetime

from mae_core.market.signal import MarketSignal, _ensure_datetime


def from_crypto_signal(crypto_price) -> MarketSignal:
    """Convert a CryptoPrice (CoinGecko) or CryptoAsset (CoinCap) to a MarketSignal.

    Both sources expose a `change_24h_pct` field and `price_usd`. The adapter
    accepts either via duck-typing — CoinGecko objects have `coin_id`, CoinCap
    objects have `asset_id`. Direction gates at ±3% (crypto noise floor).
    Strength scales linearly to 10% (full-strength move).
    """
    change_24h = getattr(crypto_price, "change_24h_pct", 0.0) or 0.0
    symbol = getattr(crypto_price, "symbol", "") or ""
    price_usd = getattr(crypto_price, "price_usd", 0.0) or 0.0
    volume = (
        getattr(crypto_price, "volume_24h", None)
        or getattr(crypto_price, "volume_24h_usd", None)
        or 0.0
    )
    market_cap = (
        getattr(crypto_price, "market_cap", None)
        or getattr(crypto_price, "market_cap_usd", None)
        or 0.0
    )

    # Detect source by distinguishing attribute
    if hasattr(crypto_price, "coin_id"):
        source = "crypto_coingecko"
        raw_id = getattr(crypto_price, "coin_id", "")
        raw_type = "CryptoPrice"
        last_updated = getattr(crypto_price, "last_updated", "")
        event_dt = _ensure_datetime(last_updated) if last_updated else datetime.now()
        extra_meta = {
            "coin_id": raw_id,
            "change_7d_pct": getattr(crypto_price, "change_7d_pct", 0.0),
        }
    else:
        source = "crypto_coincap"
        raw_id = getattr(crypto_price, "asset_id", "")
        raw_type = "CryptoAsset"
        event_dt = datetime.now()
        extra_meta = {
            "asset_id": raw_id,
            "rank": getattr(crypto_price, "rank", 0),
            "vwap_24h": getattr(crypto_price, "vwap_24h", None),
        }

    if change_24h > 3.0:
        direction = "bullish"
    elif change_24h < -3.0:
        direction = "bearish"
    else:
        direction = "neutral"

    strength = min(1.0, abs(change_24h) / 10.0)

    outcome_symbol = symbol.upper() + "-USD"
    signal_id = f"crypto:{source}:{symbol}:{event_dt.date()}"

    return MarketSignal(
        signal_id=signal_id,
        source=source,
        symbol=symbol.upper(),
        asset_class="crypto",
        domain="crypto",
        direction=direction,
        strength=strength,
        confidence=0.50,
        decay_rate=0.50,
        timestamp=event_dt,
        received_at=datetime.now(),
        outcome_symbol=outcome_symbol,
        outcome_window_days=3,
        raw_id=raw_id,
        raw_type=raw_type,
        metadata={
            "price_usd": price_usd,
            "volume_24h": volume,
            "change_24h_pct": change_24h,
            "market_cap": market_cap,
            **extra_meta,
        },
    )


def from_openinsider(purchase) -> MarketSignal:
    """Convert an InsiderPurchase (OpenInsider) to a MarketSignal.

    OpenInsider pre-filters RSU grants and 10b5-1 plan exercises, so the
    signal quality is higher than raw SEC EDGAR Form 4 data.  All records
    are purchase-only — direction is always bullish.  Strength scales to
    $500K (a large but realistic single-insider buy).
    """
    value = getattr(purchase, "value", 0.0) or 0.0
    ticker = getattr(purchase, "ticker", "") or ""
    filing_date = getattr(purchase, "filing_date", "") or ""
    trade_date = getattr(purchase, "trade_date", "") or ""

    strength = min(1.0, value / 500_000.0)

    # Use trade_date for the underlying event; fall back to filing_date
    event_dt = _ensure_datetime(trade_date or filing_date)

    signal_id = f"openinsider:{ticker}:{trade_date or filing_date}:{getattr(purchase, 'insider_name', '')}"

    return MarketSignal(
        signal_id=signal_id,
        source="openinsider_purchase",
        symbol=ticker,
        asset_class="stock",
        domain="insider",
        direction="bullish",
        strength=strength,
        confidence=0.55,
        decay_rate=0.05,
        timestamp=event_dt,
        received_at=datetime.now(),
        outcome_symbol=ticker,
        outcome_window_days=30,
        raw_id="",
        raw_type="InsiderPurchase",
        metadata={
            "insider_name": getattr(purchase, "insider_name", ""),
            "title": getattr(purchase, "title", ""),
            "trade_type": getattr(purchase, "trade_type", ""),
            "price": getattr(purchase, "price", 0.0),
            "quantity": getattr(purchase, "quantity", 0),
            "value": value,
            "delta_owned_pct": getattr(purchase, "delta_owned_pct", 0.0),
            "company_name": getattr(purchase, "company_name", ""),
        },
    )


def from_13f_holding(holding) -> MarketSignal:
    """Convert an InstitutionalHolding (SEC 13F) to a MarketSignal.

    13F filings have a 45-day lag — confidence is neutral (0.50) to reflect
    that the position may have changed since the reporting period.  Strength
    scales to a $10M position (large institutional conviction).
    value_usd is reported in thousands by SEC convention; multiply by 1000
    for the actual dollar figure.
    """
    ticker = getattr(holding, "ticker", "") or ""
    value_usd = getattr(holding, "value_usd", 0.0) or 0.0
    filing_date = getattr(holding, "filing_date", "") or ""
    period = getattr(holding, "period_of_report", "") or ""

    # value_usd is in thousands — convert to actual dollars for strength calc
    actual_value = value_usd * 1000.0
    strength = min(1.0, actual_value / 10_000_000.0)

    event_dt = _ensure_datetime(period or filing_date)

    signal_id = f"13f:{getattr(holding, 'filer_cik', '')}:{ticker}:{period or filing_date}"

    return MarketSignal(
        signal_id=signal_id,
        source="institutional_13f",
        symbol=ticker,
        asset_class="stock",
        domain="institutional",
        direction="bullish",
        strength=strength,
        confidence=0.50,
        decay_rate=0.03,
        timestamp=event_dt,
        received_at=datetime.now(),
        outcome_symbol=ticker,
        outcome_window_days=90,
        raw_id=getattr(holding, "filer_cik", ""),
        raw_type="InstitutionalHolding",
        metadata={
            "filer_name": getattr(holding, "filer_name", ""),
            "filer_cik": getattr(holding, "filer_cik", ""),
            "company_name": getattr(holding, "company_name", ""),
            "shares": getattr(holding, "shares", 0),
            "value_usd_thousands": value_usd,
            "period_of_report": period,
            "filing_date": filing_date,
        },
    )


def from_activist_filing(filing) -> MarketSignal:
    """Convert an ActivistFiling (SC 13D) to a MarketSignal.

    Activist investors acquiring >5% stakes signal intent to unlock value —
    historically bullish for the target company.  Strength is fixed at 0.85
    (activist campaigns are inherently high-conviction events).
    Decay rate is very slow (0.02) because campaigns play out over months.
    """
    subject_ticker = getattr(filing, "subject_ticker", "") or ""
    filing_date = getattr(filing, "filing_date", "") or ""

    event_dt = _ensure_datetime(filing_date)

    signal_id = f"activist13d:{getattr(filing, 'filer_cik', '')}:{subject_ticker}:{filing_date}"

    return MarketSignal(
        signal_id=signal_id,
        source="activist_13d",
        symbol=subject_ticker,
        asset_class="stock",
        domain="institutional",
        direction="bullish",
        strength=0.85,
        confidence=0.60,
        decay_rate=0.02,
        timestamp=event_dt,
        received_at=datetime.now(),
        outcome_symbol=subject_ticker,
        outcome_window_days=90,
        raw_id=getattr(filing, "filer_cik", ""),
        raw_type="ActivistFiling",
        metadata={
            "filer_name": getattr(filing, "filer_name", ""),
            "filer_cik": getattr(filing, "filer_cik", ""),
            "subject_company": getattr(filing, "subject_company", ""),
            "form_type": getattr(filing, "form_type", "SC 13D"),
            "percent_owned": getattr(filing, "percent_owned", 0.0),
            "purpose": getattr(filing, "purpose", ""),
        },
    )


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


def from_finviz_insider(trade) -> MarketSignal:
    """Convert a FinVizInsiderTrade (Buy) to a MarketSignal.

    FinViz scrapes the same SEC Form 4 filings as EDGAR and OpenInsider but
    presents them through a different lens (screener context, owner titles).
    Using all three sources gives cross-validation: if all three flag the same
    ticker on the same day, the signal confidence is substantially higher.

    Only Buy transactions should be passed here (filtered in fetch_finviz).
    Strength scales to $1M trade value.
    """
    ticker = getattr(trade, "ticker", "") or ""
    value = getattr(trade, "value", 0.0) or 0.0
    date_str = getattr(trade, "date", "") or ""

    strength = min(1.0, value / 1_000_000.0)

    event_dt = _ensure_datetime(date_str) if date_str else datetime.now()

    signal_id = (
        f"finviz_insider:{ticker}:{date_str}:{getattr(trade, 'owner_name', '')}"
    )

    return MarketSignal(
        signal_id=signal_id,
        source="finviz_insider",
        symbol=ticker,
        asset_class="stock",
        domain="insider",
        direction="bullish",
        strength=strength,
        confidence=0.50,
        decay_rate=0.05,
        timestamp=event_dt,
        received_at=datetime.now(),
        outcome_symbol=ticker,
        outcome_window_days=30,
        raw_id="",
        raw_type="FinVizInsiderTrade",
        metadata={
            "owner_name": getattr(trade, "owner_name", ""),
            "relationship": getattr(trade, "relationship", ""),
            "transaction_type": getattr(trade, "transaction_type", ""),
            "date": date_str,
            "shares_traded": getattr(trade, "shares_traded", 0),
            "value": value,
            "shares_owned": getattr(trade, "shares_owned", 0),
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
