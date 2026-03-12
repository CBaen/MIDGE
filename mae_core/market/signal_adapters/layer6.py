"""Layer 6 signal adapters — COT, StockTwits, VIX, Google Trends, Finnhub extras.

Converts Layer 6 free data sources into normalized MarketSignal objects.
These were the fifth expansion of MIDGE's sensing capabilities (2026-02-27).
"""

from __future__ import annotations

from datetime import datetime

from mae_core.market.signal import MarketSignal, _ensure_datetime


def from_cot_positioning(cot) -> MarketSignal:
    """Convert a COTSignal to a MarketSignal.

    Commercial net positioning is the primary signal. When commercials are
    heavily long it's bullish; when heavily short, bearish.

    Two enhanced metrics improve signal quality when history is available:

    change_commercial_net — week-over-week momentum.  A large positive WoW
    change while already net-long amplifies the bullish read; a reversal
    (WoW negative while net-long) weakens it.  Threshold: ±5% of open
    interest is considered a meaningful shift.

    cot_index — percentile rank vs 52-week range.  Extremes (>0.85 or <0.15)
    are the historically high-signal zones where commercial positioning is at
    a structural extreme and mean reversion or trend continuation is most
    likely.  A COT Index > 0.85 on a bullish net adds a confidence boost; an
    extreme in the opposite direction of the net acts as a caution flag.
    """
    # --- Direction (unchanged logic) ---
    if cot.commercial_net > 0 and cot.pct_commercial_long > 0.55:
        direction = "bullish"
    elif cot.commercial_net < 0 and cot.pct_commercial_long < 0.45:
        direction = "bearish"
    else:
        direction = "neutral"

    # --- Base strength from absolute net/OI ratio ---
    net_ratio = abs(cot.commercial_net) / max(1, cot.open_interest)
    base_strength = min(1.0, net_ratio * 5)  # 20% net/OI = full strength

    # --- WoW momentum modifier ---
    # A large same-direction weekly change boosts strength; a reversal damps it.
    wow_modifier = 0.0
    change_net = getattr(cot, "change_commercial_net", None)
    oi = max(1, cot.open_interest)
    if change_net is not None:
        change_ratio = change_net / oi  # signed, relative to OI
        if direction == "bullish":
            wow_modifier = min(0.15, change_ratio * 3)   # +15% max boost
        elif direction == "bearish":
            wow_modifier = min(0.15, -change_ratio * 3)  # bearish: negative change boosts

    # --- COT Index modifier ---
    # Extremes (>0.85 bullish, <0.15 bearish) add confidence at structural turning points.
    index_modifier = 0.0
    cot_idx = getattr(cot, "cot_index", None)
    if cot_idx is not None:
        if direction == "bullish" and cot_idx > 0.85:
            index_modifier = 0.10
        elif direction == "bearish" and cot_idx < 0.15:
            index_modifier = 0.10
        elif direction == "bullish" and cot_idx < 0.15:
            index_modifier = -0.10  # net long but at 52-week low — contrarian caution
        elif direction == "bearish" and cot_idx > 0.85:
            index_modifier = -0.10  # net short but at 52-week high — contrarian caution

    strength = min(1.0, max(0.0, base_strength + wow_modifier + index_modifier))

    event_dt = _ensure_datetime(cot.report_date)

    signal_id = f"cot:{cot.ticker}:{cot.report_date}"

    return MarketSignal(
        signal_id=signal_id,
        source="cot_positioning",
        symbol=cot.ticker,
        asset_class="futures",
        domain="positioning",
        direction=direction,
        strength=strength,
        confidence=cot.confidence,
        decay_rate=cot.decay_rate,
        timestamp=event_dt,
        received_at=datetime.now(),
        outcome_symbol=cot.ticker,
        outcome_window_days=21,  # Weekly data — check 3 weeks out
        raw_id="",
        raw_type="COTSignal",
        metadata={
            "contract_name": cot.contract_name,
            "commercial_net": cot.commercial_net,
            "noncommercial_net": cot.noncommercial_net,
            "small_trader_net": cot.small_trader_net,
            "open_interest": cot.open_interest,
            "pct_commercial_long": cot.pct_commercial_long,
            "change_commercial_net": change_net,
            "cot_index": cot_idx,
        },
    )


def from_stocktwits_sentiment(st) -> MarketSignal:
    """Convert a StockTwitsSentiment to a MarketSignal.

    Bull/bear ratio is the signal. Extreme readings (>70% or <30%)
    are most actionable, either as confirmation or contrarian signals.
    """
    if st.bull_ratio >= 0.70:
        direction = "bullish"
    elif st.bull_ratio <= 0.30:
        direction = "bearish"
    else:
        direction = "neutral"

    # Strength from extremeness of sentiment
    deviation = abs(st.bull_ratio - 0.50)
    strength = min(1.0, deviation * 3)  # 50% = 0.0, 83% = 1.0

    event_dt = _ensure_datetime(st.detected_at)

    signal_id = f"stocktwits:{st.ticker}:{st.detected_at}"

    return MarketSignal(
        signal_id=signal_id,
        source="stocktwits_sentiment",
        symbol=st.ticker,
        asset_class="stock",
        domain="sentiment",
        direction=direction,
        strength=strength,
        confidence=st.confidence,
        decay_rate=st.decay_rate,
        timestamp=event_dt,
        received_at=event_dt,
        outcome_symbol=st.ticker,
        raw_id="",
        raw_type="StockTwitsSentiment",
        metadata={
            "bull_count": st.bull_count,
            "bear_count": st.bear_count,
            "bull_ratio": st.bull_ratio,
            "total_messages": st.total_messages,
            "trending": st.trending,
        },
    )


def from_vix_structure(vix) -> MarketSignal:
    """Convert a VIXSignal to a MarketSignal.

    VIX term structure is a regime signal:
    - Backwardation = fear, bearish for equities
    - Contango = complacency, bullish for equities
    - High absolute VIX (>30) = fear regardless of structure
    """
    if vix.structure_type == "backwardation" or vix.vix_spot > 30:
        direction = "bearish"
    elif vix.structure_type == "contango" and vix.vix_spot < 20:
        direction = "bullish"
    else:
        direction = "neutral"

    # Strength from VIX level (higher = stronger signal)
    strength = min(1.0, vix.vix_spot / 40.0)

    event_dt = _ensure_datetime(vix.date)

    signal_id = f"vix:{vix.structure_type}:{vix.date}"

    return MarketSignal(
        signal_id=signal_id,
        source="vix_term_structure",
        symbol="",  # VIX is market-wide, no specific ticker
        asset_class="macro",
        domain="volatility",
        direction=direction,
        strength=strength,
        confidence=vix.confidence,
        decay_rate=vix.decay_rate,
        timestamp=event_dt,
        received_at=datetime.now(),
        outcome_symbol="SPY",  # Track against SPY
        outcome_window_days=7,
        raw_id="",
        raw_type="VIXSignal",
        metadata={
            "vix_spot": vix.vix_spot,
            "vix_1m": vix.vix_1m,
            "vix_3m": vix.vix_3m,
            "term_spread": vix.term_spread,
            "structure_type": vix.structure_type,
        },
    )


def from_trends_signal(trend) -> MarketSignal:
    """Convert a TrendsSignal to a MarketSignal.

    Search interest spikes indicate retail attention.
    High interest + rising = momentum building.
    Macro fear terms (recession, crash) are bearish.
    """
    fear_keywords = {"recession", "market crash", "bear market", "unemployment"}
    is_fear = trend.keyword.lower() in fear_keywords

    if is_fear:
        direction = "bearish" if trend.interest_score > 50 else "neutral"
    elif trend.interest_delta_7d > 20:
        direction = "bullish"  # Rising attention on a ticker
    elif trend.interest_delta_7d < -20:
        direction = "bearish"
    else:
        direction = "neutral"

    strength = min(1.0, trend.interest_score / 100.0)

    event_dt = _ensure_datetime(trend.detected_at)

    signal_id = f"trends:{trend.keyword}:{trend.detected_at}"

    # Symbol is the keyword if it looks like a ticker (all caps, short)
    symbol = trend.keyword if trend.keyword.isupper() and len(trend.keyword) <= 5 else ""

    return MarketSignal(
        signal_id=signal_id,
        source="google_trends",
        symbol=symbol,
        asset_class="stock" if symbol else "macro",
        domain="sentiment",
        direction=direction,
        strength=strength,
        confidence=trend.confidence,
        decay_rate=trend.decay_rate,
        timestamp=event_dt,
        received_at=event_dt,
        outcome_symbol=symbol or "SPY",
        outcome_window_days=7,
        raw_id="",
        raw_type="TrendsSignal",
        metadata={
            "keyword": trend.keyword,
            "interest_score": trend.interest_score,
            "interest_delta_7d": trend.interest_delta_7d,
            "is_breakout": trend.is_breakout,
            "related_queries": trend.related_queries[:5],
        },
    )


def from_economic_event(event) -> MarketSignal:
    """Convert a Finnhub EconomicEvent to a MarketSignal.

    High-impact events (FOMC, CPI, NFP) are catalysts.
    Pre-event: neutral catalyst approaching.
    Post-event with surprise: directional from actual vs estimate.
    """
    if event.actual is not None and event.estimate is not None:
        surprise = event.surprise_pct()
        if surprise is not None and abs(surprise) > 0.05:
            direction = "bullish" if surprise > 0 else "bearish"
            strength = min(1.0, abs(surprise) * 5)
        else:
            direction = "neutral"
            strength = 0.3
    else:
        direction = "neutral"
        strength = 0.4 if event.impact == "high" else 0.2

    event_dt = _ensure_datetime(event.date)

    signal_id = f"econ:{event.country}:{event.event}:{event.date}"

    return MarketSignal(
        signal_id=signal_id,
        source="finnhub_economic",
        symbol="",
        asset_class="macro",
        domain="events",
        direction=direction,
        strength=strength,
        confidence=event.confidence,
        decay_rate=event.decay_rate,
        timestamp=event_dt,
        received_at=datetime.now(),
        outcome_symbol="SPY",
        outcome_window_days=3,
        raw_id="",
        raw_type="EconomicEvent",
        metadata={
            "event": event.event,
            "country": event.country,
            "impact": event.impact,
            "actual": event.actual,
            "estimate": event.estimate,
            "previous": event.previous,
            "unit": event.unit,
        },
    )


def from_analyst_recommendation(rec) -> MarketSignal:
    """Convert a Finnhub AnalystRec to a MarketSignal.

    Strong consensus (>80% buy or >40% sell) is a signal.
    Analyst sentiment is slow-moving and reflects institutional thinking.
    """
    buy_ratio = rec.buy_ratio
    if buy_ratio >= 0.80:
        direction = "bullish"
    elif buy_ratio <= 0.30:
        direction = "bearish"
    else:
        direction = "neutral"

    # Strength from consensus strength (how extreme the skew is)
    deviation = abs(buy_ratio - 0.50)
    strength = min(1.0, deviation * 3)

    event_dt = _ensure_datetime(rec.period)

    signal_id = f"analyst:{rec.symbol}:{rec.period}"

    return MarketSignal(
        signal_id=signal_id,
        source="finnhub_analyst",
        symbol=rec.symbol,
        asset_class="stock",
        domain="fundamentals",
        direction=direction,
        strength=strength,
        confidence=rec.confidence,
        decay_rate=rec.decay_rate,
        timestamp=event_dt,
        received_at=datetime.now(),
        outcome_symbol=rec.symbol,
        outcome_window_days=30,  # Analyst recs have longer time horizon
        raw_id="",
        raw_type="AnalystRec",
        metadata={
            "strong_buy": rec.strong_buy,
            "buy": rec.buy,
            "hold": rec.hold,
            "sell": rec.sell,
            "strong_sell": rec.strong_sell,
            "buy_ratio": buy_ratio,
            "total": rec.total,
        },
    )


def from_yahoo_rss_signal(sig) -> MarketSignal:
    """Convert a YahooHeadlineSignal to a MarketSignal.

    Headline velocity (news acceleration) is the primary signal.
    A sudden burst of headlines means something is happening — market
    participants are reacting to news. Directional sentiment from
    keyword polarity provides the bullish/bearish tilt.

    Velocity >= 3x prior window = strong signal.
    Sentiment keywords sharpen direction.
    Domain: "events" (breaking news is a catalyst domain).
    """
    ticker = getattr(sig, "ticker", "") or ""
    headline_count = getattr(sig, "headline_count", 0)
    velocity_change = getattr(sig, "velocity_change", 1.0)
    keyword_polarity = getattr(sig, "keyword_polarity", 0)
    detected_at = getattr(sig, "detected_at", "")

    # Direction from keyword polarity
    if keyword_polarity > 0:
        direction = "bullish"
    elif keyword_polarity < 0:
        direction = "bearish"
    else:
        direction = "neutral"

    # Strength: velocity change (capped at 1.0) combined with headline volume
    # 2x = 0.40, 3x = 0.60, 5x+ = 0.80+
    velocity_strength = min(0.90, (velocity_change - 1.0) / 6.0 + 0.30)
    # Boost slightly if many absolute headlines (sustained coverage)
    volume_boost = min(0.10, headline_count * 0.005)
    strength = min(1.0, velocity_strength + volume_boost)

    event_dt = _ensure_datetime(detected_at) if detected_at else datetime.now()
    signal_id = f"yahoo_rss:{ticker}:{event_dt.strftime('%Y%m%dT%H%M')}"

    return MarketSignal(
        signal_id=signal_id,
        source="yahoo_rss",
        symbol=ticker,
        asset_class="stock",
        domain="events",
        direction=direction,
        strength=strength,
        confidence=getattr(sig, "confidence", 0.45),
        decay_rate=getattr(sig, "decay_rate", 0.70),
        timestamp=event_dt,
        received_at=datetime.now(),
        outcome_symbol=ticker,
        outcome_window_days=3,   # News impact resolves quickly
        raw_id="",
        raw_type="YahooHeadlineSignal",
        metadata={
            "headline_count": headline_count,
            "velocity_change": velocity_change,
            "keyword_polarity": keyword_polarity,
            "sentiment_keywords": getattr(sig, "sentiment_keywords", [])[:10],
            "latest_headline": (getattr(sig, "latest_headline", "") or "")[:200],
        },
    )


def from_social_text_signal(sig) -> MarketSignal:
    """Convert a SocialTextSignal to a MarketSignal.

    StockTwits message text carries cultural momentum that the simple
    bull/bear ratio misses.  A surge in options_flow chatter predates
    volume; earnings_play dominance signals a catalyst is being positioned
    for.  Text theme signals have faster decay (0.60) than sentiment ratios.
    """
    ticker = getattr(sig, "ticker", "") or ""
    direction = getattr(sig, "direction", "neutral")
    strength = getattr(sig, "strength", 0.0) or 0.0
    dominant_theme = getattr(sig, "dominant_theme", "unknown")
    message_count = getattr(sig, "message_count", 0)
    theme_score = getattr(sig, "theme_score", 0.0)
    detected_at = getattr(sig, "detected_at", "")

    event_dt = _ensure_datetime(detected_at) if detected_at else datetime.now()

    signal_id = f"social_text:{ticker}:{dominant_theme}:{event_dt.strftime('%Y%m%dT%H%M')}"

    return MarketSignal(
        signal_id=signal_id,
        source="social_text",
        symbol=ticker,
        asset_class="stock",
        domain="sentiment",
        direction=direction,
        strength=strength,
        confidence=getattr(sig, "confidence", 0.40),
        decay_rate=getattr(sig, "decay_rate", 0.60),
        timestamp=event_dt,
        received_at=datetime.now(),
        outcome_symbol=ticker,
        outcome_window_days=3,
        raw_id="",
        raw_type="SocialTextSignal",
        metadata={
            "dominant_theme": dominant_theme,
            "theme_score": theme_score,
            "intensity": getattr(sig, "intensity", 0.0),
            "message_count": message_count,
            "themes": getattr(sig, "themes", {}),
        },
    )
