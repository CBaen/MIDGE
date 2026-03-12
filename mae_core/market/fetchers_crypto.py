"""Crypto and miscellaneous fetch functions — CoinGecko, CoinCap, FINRA, Economic Calendar.

Covers 24/7 crypto price feeds, short volume data, and economic event suppression signals.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger("midge.market.sensing")


def fetch_crypto_prices(coingecko_client: Any, converter: Callable) -> list:
    """Fetch crypto prices from CoinGecko."""
    if coingecko_client is None:
        return []
    signals = []
    try:
        prices = coingecko_client.get_prices()
        for price in prices:
            try:
                signals.append(converter(price))
            except Exception:
                pass
    except Exception as e:
        logger.debug("CoinGecko fetch failed: %s", e)
    return signals


def fetch_crypto_exchange(coincap_client: Any, converter: Callable) -> list:
    """Fetch crypto from CoinCap."""
    if coincap_client is None:
        return []
    signals = []
    try:
        assets = coincap_client.get_assets(limit=10)
        for asset in assets:
            try:
                signals.append(converter(asset))
            except Exception:
                pass
    except Exception as e:
        logger.debug("CoinCap fetch failed: %s", e)
    return signals


def fetch_finra_short(finra_client: Any, watchlist: dict, converter: Callable) -> list:
    """Fetch FINRA daily short volume — high short ratio tickers."""
    if finra_client is None:
        return []

    signals = []
    try:
        # Get tickers with >50% short volume ratio
        high_short = finra_client.get_high_short_ratio(min_ratio=0.5)
        # Filter to watchlist + top 10 highest ratios
        watchlist_tickers = set(watchlist.get("tickers", []))
        for i, record in enumerate(high_short):
            try:
                if record.symbol in watchlist_tickers or i < 10:
                    signals.append(converter(record))
            except Exception:
                pass
    except Exception as e:
        logger.debug("FINRA short volume fetch failed: %s", e)
    return signals


def fetch_binance_funding(binance_funding_client: Any, converter: Callable) -> list:
    """Fetch perpetual futures funding rates from Binance."""
    if binance_funding_client is None:
        return []
    signals = []
    try:
        rates = binance_funding_client.get_funding_rates()
        for rate in rates:
            try:
                signals.append(converter(rate))
            except Exception:
                pass
    except Exception as e:
        logger.debug("Binance funding fetch failed: %s", e)
    return signals


def fetch_kalshi_movers(kalshi_client: Any, converter: Callable) -> list:
    """Fetch significant probability shifts from Kalshi prediction markets."""
    if kalshi_client is None:
        return []
    signals = []
    try:
        movers = kalshi_client.get_market_movers()
        for mover in movers:
            try:
                signals.append(converter(mover))
            except Exception:
                pass
    except Exception as e:
        logger.debug("Kalshi movers fetch failed: %s", e)
    return signals


def fetch_economic_calendar(calendar_client: Any, converter: Callable) -> list:
    """Fetch upcoming high-impact economic events as suppression signals."""
    if calendar_client is None:
        return []
    signals = []
    try:
        events = calendar_client.get_upcoming_events(days=7)
        # Only high-impact events
        high_impact = [e for e in events if e.impact == "high"]
        for event in high_impact:
            try:
                signals.append(converter(event))
            except Exception:
                pass
    except Exception as e:
        logger.debug("Economic calendar fetch failed: %s", e)
    return signals
