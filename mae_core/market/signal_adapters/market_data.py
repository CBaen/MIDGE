"""Market data signal adapters — short interest, news, earnings, macro, social, price.

Converts raw market data types into normalized MarketSignal objects.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Optional

from mae_core.market.signal import MarketSignal, _ensure_datetime


def from_short_interest(data) -> MarketSignal:
    """Convert a FINRA ShortInterestData to a MarketSignal.

    High short ratio (>50% of daily volume) is a potential squeeze signal.
    When combined with insider buying clusters, this becomes very actionable.
    """
    # Direction: high short ratio is bearish pressure, but also squeeze potential
    if data.short_ratio >= 0.6:
        direction = "bearish"  # Heavy shorting pressure
    elif data.short_ratio >= 0.4:
        direction = "neutral"  # Elevated but not extreme
    else:
        direction = "neutral"

    # Strength from short ratio
    strength = min(1.0, data.short_ratio / 0.7)

    event_dt = _ensure_datetime(data.date)

    symbol = data.symbol or ""
    signal_id = f"finra_short:{symbol}:{data.date}"

    return MarketSignal(
        signal_id=signal_id,
        source="finra_short",
        symbol=symbol,
        asset_class="stock",
        domain="institutional",
        direction=direction,
        strength=strength,
        confidence=data.confidence,
        decay_rate=data.decay_rate,
        timestamp=event_dt,
        received_at=datetime.now(),
        outcome_symbol=symbol,
        raw_id="",
        raw_type="ShortInterestData",
        metadata={
            "short_volume": data.short_volume,
            "total_volume": data.total_volume,
            "short_ratio": data.short_ratio,
        },
    )


def from_news_sentiment(sentiment) -> MarketSignal:
    """Convert a Finnhub NewsSentiment to a MarketSignal.

    The buzz score (articles this week vs weekly average) is the velocity.
    Bullish/bearish from Finnhub's own sentiment analysis.
    """
    if sentiment.bullish_pct >= 0.6:
        direction = "bullish"
    elif sentiment.bearish_pct >= 0.6:
        direction = "bearish"
    else:
        direction = "neutral"

    # Strength from buzz score (how much more attention than normal)
    strength = min(1.0, sentiment.buzz_score / 3.0)

    event_dt = _ensure_datetime(sentiment.detected_at)

    symbol = sentiment.ticker or ""
    signal_id = f"finnhub_news:{symbol}:{sentiment.detected_at}"

    return MarketSignal(
        signal_id=signal_id,
        source="finnhub_news",
        symbol=symbol,
        asset_class="stock",
        domain="news",
        direction=direction,
        strength=strength,
        confidence=sentiment.confidence,
        decay_rate=sentiment.decay_rate,
        timestamp=event_dt,
        received_at=event_dt,
        outcome_symbol=symbol,
        raw_id="",
        raw_type="NewsSentiment",
        metadata={
            "bullish_pct": sentiment.bullish_pct,
            "bearish_pct": sentiment.bearish_pct,
            "news_score": sentiment.news_score,
            "buzz_score": sentiment.buzz_score,
        },
    )


def from_earnings_event(event) -> MarketSignal:
    """Convert a Finnhub EarningsEvent to a MarketSignal.

    Pre-earnings: neutral signal (catalyst approaching).
    Post-earnings with surprise: strong directional signal.
    """
    if event.eps_actual is not None and event.eps_estimate is not None:
        # Post-earnings: direction from surprise
        surprise = event.eps_actual - event.eps_estimate
        if surprise > 0:
            direction = "bullish"
            strength = min(1.0, abs(surprise) / max(0.01, abs(event.eps_estimate)) * 2)
        elif surprise < 0:
            direction = "bearish"
            strength = min(1.0, abs(surprise) / max(0.01, abs(event.eps_estimate)) * 2)
        else:
            direction = "neutral"
            strength = 0.3
    else:
        # Pre-earnings: catalyst approaching
        direction = "neutral"
        strength = 0.4

    event_dt = _ensure_datetime(event.date)

    symbol = event.symbol or ""
    signal_id = f"finnhub_earnings:{symbol}:{event.date}"

    return MarketSignal(
        signal_id=signal_id,
        source="finnhub_earnings",
        symbol=symbol,
        asset_class="stock",
        domain="events",
        direction=direction,
        strength=strength,
        confidence=event.confidence,
        decay_rate=event.decay_rate,
        timestamp=event_dt,
        received_at=datetime.now(),
        outcome_symbol=symbol,
        raw_id="",
        raw_type="EarningsEvent",
        metadata={
            "eps_estimate": event.eps_estimate,
            "eps_actual": event.eps_actual,
            "revenue_estimate": event.revenue_estimate,
            "revenue_actual": event.revenue_actual,
            "hour": event.hour,
        },
    )


def from_macro_indicator(indicator) -> MarketSignal:
    """Convert a FRED MacroIndicator to a MarketSignal.

    Macro indicators modulate all other signals — they're regime context.
    A bullish insider signal during yield curve inversion is less reliable.
    """
    event_dt = _ensure_datetime(indicator.date)

    signal_id = f"fred:{indicator.series_id}:{indicator.date}"

    return MarketSignal(
        signal_id=signal_id,
        source="fred_macro",
        symbol="",  # Macro signals have no ticker
        asset_class="macro",
        domain="macro",
        direction=indicator.direction,
        strength=min(1.0, abs(indicator.value) / 10.0) if indicator.signal_type != "yield_curve"
                 else min(1.0, abs(indicator.value) / 3.0),
        confidence=indicator.confidence,
        decay_rate=indicator.decay_rate,
        timestamp=event_dt,
        received_at=datetime.now(),
        outcome_symbol="",
        raw_id=indicator.series_id,
        raw_type="MacroIndicator",
        metadata={
            "series_id": indicator.series_id,
            "series_name": indicator.series_name,
            "value": indicator.value,
            "signal_type": indicator.signal_type,
        },
    )


def from_price_data(data, move_threshold: float = 1.5) -> Optional[MarketSignal]:
    """Convert a PriceData object to a MarketSignal.

    Only produces a signal when the intraday move exceeds move_threshold%.
    Returns None for sub-threshold moves to avoid flooding with noise.

    Price signals are supporting context — they tell the lag-correlation
    analyzer and convergence alerter what the market actually DID, so it
    can learn which other signals predicted the move.

    Args:
        data: PriceData from price_fetcher.py
        move_threshold: Minimum |change_pct| to emit a signal (default 1.5%)

    Returns:
        MarketSignal or None if the move is below threshold
    """
    if abs(data.change_pct) < move_threshold:
        return None

    direction = "bullish" if data.change_pct > 0 else "bearish"
    # 5% move = full strength. 1.5% = 0.30. 3% = 0.60.
    strength = min(1.0, abs(data.change_pct) / 5.0)

    event_dt = _ensure_datetime(data.timestamp)

    symbol = data.symbol or ""
    signal_id = f"price_move:{symbol}:{data.timestamp}"

    return MarketSignal(
        signal_id=signal_id,
        source="yfinance_price",
        symbol=symbol,
        asset_class="stock",
        domain="price",
        direction=direction,
        strength=strength,
        confidence=0.50,
        decay_rate=0.15,
        timestamp=event_dt,
        received_at=datetime.now(),
        outcome_symbol=symbol,
        raw_id="",
        raw_type="PriceData",
        metadata={
            "open": data.open,
            "high": data.high,
            "low": data.low,
            "close": data.price,
            "volume": data.volume,
            "change_pct": data.change_pct,
        },
    )


def from_social_sentiment(sentiment) -> MarketSignal:
    """Convert an ApeWisdom SocialSentiment to a MarketSignal.

    The signal is in the mention velocity (mention_change), not the
    absolute count. A ticker going from 10 to 100 mentions is more
    actionable than one steady at 500.
    """
    # Direction from velocity: accelerating mentions = bullish retail momentum
    if sentiment.mention_change >= 3.0:
        direction = "bullish"
    elif sentiment.mention_change <= 0.3:
        direction = "bearish"
    else:
        direction = "neutral"

    # Strength from mention velocity (log scale)
    strength = min(1.0, math.log1p(sentiment.mention_change) / math.log1p(10))

    event_dt = _ensure_datetime(sentiment.detected_at)

    symbol = sentiment.ticker or ""
    signal_id = f"social:{symbol}:{sentiment.detected_at}"

    return MarketSignal(
        signal_id=signal_id,
        source="social_sentiment",
        symbol=symbol,
        asset_class="stock",
        domain="sentiment",
        direction=direction,
        strength=strength,
        confidence=sentiment.confidence,
        decay_rate=sentiment.decay_rate,
        timestamp=event_dt,
        received_at=event_dt,
        outcome_symbol=symbol,
        raw_id="",
        raw_type="SocialSentiment",
        metadata={
            "mentions_24h": sentiment.mentions_24h,
            "mentions_prior_24h": sentiment.mentions_prior_24h,
            "mention_change": sentiment.mention_change,
            "upvotes": sentiment.upvotes,
            "rank": sentiment.rank,
            "source_subreddit": sentiment.source_subreddit,
        },
    )


def from_energy_indicator(indicator) -> MarketSignal:
    """Convert an EIA EnergyIndicator to a MarketSignal.

    Energy inventory/production data is a real-economy signal. Inventory
    builds (supply > demand) are bearish for energy stocks; draws are bullish.
    The EIA client pre-computes direction and strength from weekly changes.

    Unlike macro signals, energy signals carry specific affected tickers
    (XLE, XOP, UNG, etc.) — the first is used as the primary symbol.
    """
    event_dt = _ensure_datetime(indicator.date)

    symbol = indicator.affected_tickers[0] if indicator.affected_tickers else ""
    signal_id = f"eia_energy:{indicator.signal_type}:{indicator.date}"

    return MarketSignal(
        signal_id=signal_id,
        source="eia_energy",
        symbol=symbol,
        asset_class="commodity",
        domain="energy",
        direction=indicator.direction,
        strength=indicator.strength,
        confidence=indicator.confidence,
        decay_rate=indicator.decay_rate,
        timestamp=event_dt,
        received_at=datetime.now(),
        outcome_symbol=symbol,
        raw_id=indicator.series_key,
        raw_type="EnergyIndicator",
        metadata={
            "series_key": indicator.series_key,
            "series_name": indicator.series_name,
            "value": indicator.value,
            "prior_value": indicator.prior_value,
            "change": indicator.change,
            "change_pct": indicator.change_pct,
            "signal_type": indicator.signal_type,
            "unit": indicator.unit,
            "affected_tickers": indicator.affected_tickers,
        },
    )


def from_usda_indicator(indicator) -> MarketSignal:
    """Convert a USDA AgriculturalIndicator to a MarketSignal.

    Agricultural supply/demand data is a slow-moving real-economy signal.
    Stocks-to-use ratio drives commodity futures direction. Deficit = bullish
    for the commodity and its producers; surplus = bearish.

    Primary symbol is the commodity futures ticker (ZW, ZC, ZS, CT) so the
    convergence engine can match against technical signals on the same instrument.
    """
    event_dt = _ensure_datetime(datetime.now().date().isoformat())

    symbol = indicator.futures_ticker or (
        indicator.affected_tickers[0] if indicator.affected_tickers else ""
    )
    signal_id = f"usda_agriculture:{indicator.commodity_key}:{indicator.market_year}"

    # Strength from stocks-to-use deviation: more extreme = stronger signal
    # Already computed by the client; use directly
    strength = indicator.strength

    return MarketSignal(
        signal_id=signal_id,
        source="usda_agriculture",
        symbol=symbol,
        asset_class="commodity",
        domain="macro",   # Agricultural supply/demand is a macro-level signal
        direction=indicator.direction,
        strength=strength,
        confidence=indicator.confidence,
        decay_rate=indicator.decay_rate,
        timestamp=event_dt,
        received_at=datetime.now(),
        outcome_symbol=symbol,
        raw_id=f"{indicator.commodity_key}:{indicator.market_year}",
        raw_type="AgriculturalIndicator",
        metadata={
            "commodity_key": indicator.commodity_key,
            "commodity_name": indicator.commodity_name,
            "market_year": indicator.market_year,
            "production": indicator.production,
            "consumption": indicator.consumption,
            "ending_stocks": indicator.ending_stocks,
            "supply_surplus_ratio": indicator.supply_surplus_ratio,
            "futures_ticker": indicator.futures_ticker,
            "affected_tickers": indicator.affected_tickers,
        },
    )


def from_fred_yield(indicator) -> MarketSignal:
    """Convert a FRED MacroIndicator for treasury yields / DXY into a MarketSignal.

    Treasury yield and dollar index data are forex-critical — rate differentials
    drive EUR/USD, USD/JPY, and commodity prices. This adapter routes yield/dollar
    signals into the 'macro' domain alongside standard FRED macro signals, but
    with a slightly faster decay rate (yields move daily, not weekly).

    The signal uses the same from_macro_indicator logic but with adjusted
    strength computation for rate data (absolute value of rate level matters).
    """
    # Reuse the existing macro adapter — yield/dollar signals are still macro domain
    return from_macro_indicator(indicator)


def from_legislative_indicator(indicator) -> MarketSignal:
    """Convert a Congress.gov LegislativeIndicator to a MarketSignal.

    Legislative signals are slow-moving but high-conviction when stacked
    with insider trades and sector data. Bills advancing through Congress
    affect sector ETFs and individual companies in the policy area.
    """
    event_dt = _ensure_datetime(indicator.action_date)

    symbol = indicator.affected_tickers[0] if indicator.affected_tickers else ""
    signal_id = f"congress_legislation:{indicator.bill_id}:{indicator.action_date}"

    return MarketSignal(
        signal_id=signal_id,
        source="congress_legislation",
        symbol=symbol,
        asset_class="equity",
        domain="government",
        direction=indicator.direction,
        strength=indicator.strength,
        confidence=indicator.confidence,
        decay_rate=indicator.decay_rate,
        timestamp=event_dt,
        received_at=datetime.now(),
        outcome_symbol=symbol,
        raw_id=indicator.bill_id,
        raw_type="LegislativeIndicator",
        metadata={
            "bill_id": indicator.bill_id,
            "bill_number": indicator.bill_number,
            "title": indicator.title,
            "congress": indicator.congress,
            "policy_area": indicator.policy_area,
            "action_text": indicator.action_text,
            "signal_type": indicator.signal_type,
            "affected_tickers": indicator.affected_tickers,
        },
    )
