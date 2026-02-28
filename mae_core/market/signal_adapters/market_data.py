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
