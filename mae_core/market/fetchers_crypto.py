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


def fetch_cboe_options(cboe_options_client: Any) -> list:
    """Fetch CBOE VIX family + put/call ratio — options domain signals."""
    if cboe_options_client is None:
        return []
    from datetime import datetime, timezone
    signals = []
    try:
        # VIX family signals (VVIX, VIX9D, OVX, GVZ)
        vix_signals = cboe_options_client.get_vix_family()
        for sig in (vix_signals or []):
            signals.append({
                "source": f"cboe_{sig.index_name.lower()}" if hasattr(sig, "index_name") else sig.signal_source,
                "symbol": getattr(sig, "index_name", "VIX_FAMILY"),
                "asset_class": "index",
                "domain": "options",
                "direction": sig.direction,
                "strength": sig.strength,
                "confidence": sig.confidence,
                "decay_rate": sig.decay_rate,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "outcome_window_days": 7,
                "metadata": {
                    "value": getattr(sig, "value", None),
                    "note": getattr(sig, "note", ""),
                    "signal_source": sig.signal_source,
                },
            })
    except Exception as e:
        logger.debug("CBOE VIX family fetch failed: %s", e)

    try:
        # Put/Call ratio
        pc = cboe_options_client.get_put_call_ratio()
        if pc is not None:
            signals.append({
                "source": "cboe_put_call",
                "symbol": "SPX",
                "asset_class": "index",
                "domain": "options",
                "direction": pc.direction,
                "strength": pc.strength,
                "confidence": pc.confidence,
                "decay_rate": pc.decay_rate,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "outcome_window_days": 7,
                "metadata": {
                    "equity_pc": getattr(pc, "equity_pc", None),
                    "index_pc": getattr(pc, "index_pc", None),
                    "total_pc": getattr(pc, "total_pc", None),
                    "signal_source": pc.signal_source,
                },
            })
    except Exception as e:
        logger.debug("CBOE put/call fetch failed: %s", e)
    return signals


def fetch_crypto_fear_greed(fear_greed_client: Any) -> list:
    """Fetch Crypto Fear & Greed Index — contrarian sentiment signal."""
    if fear_greed_client is None:
        return []
    from datetime import datetime, timezone
    signals = []
    try:
        fg = fear_greed_client.get_fear_greed()
        if fg is not None and fg.direction != "neutral":
            signals.append({
                "source": "crypto_fear_greed",
                "symbol": "BTC",
                "asset_class": "crypto",
                "domain": "sentiment",
                "direction": fg.direction,
                "strength": fg.strength,
                "confidence": fg.confidence,
                "decay_rate": fg.decay_rate,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "outcome_window_days": 7,
                "metadata": {
                    "value": fg.value,
                    "classification": fg.classification,
                    "trend": fg.trend,
                    "signal_source": fg.signal_source,
                },
            })
    except Exception as e:
        logger.debug("Crypto Fear & Greed fetch failed: %s", e)
    return signals


def fetch_btc_dominance(coingecko_client: Any) -> list:
    """Fetch BTC dominance from CoinGecko and emit crypto rotation signals.

    BTC.D > 60% → bearish for alts (capital concentrated in BTC).
    BTC.D < 55% → bullish for alts (alt season starting).
    BTC.D falling >2pp in 7 days → bullish for alts (rotation in progress).

    Uses a simple file-based state store to track prior dominance.
    """
    if coingecko_client is None:
        return []
    import json
    from pathlib import Path

    signals = []
    try:
        global_data = coingecko_client.get_market_data()
        if global_data is None:
            return []

        # Load prior dominance from state file (simple persistence)
        state_path = Path("data/market/btc_dominance_state.json")
        prior_dominance = None
        try:
            if state_path.exists():
                state = json.loads(state_path.read_text())
                prior_dominance = state.get("dominance_7d_ago")
                # Rotate: if we have a 7-day-old reading, use it; then store current as new prior
                # We store daily readings and use the one from 7 days ago
                readings = state.get("daily_readings", [])
                from datetime import datetime, timedelta
                cutoff_7d = (datetime.now() - timedelta(days=7)).date().isoformat()
                cutoff_14d = (datetime.now() - timedelta(days=14)).date().isoformat()
                # Find reading from 7 days ago (nearest available)
                candidates = [r for r in readings if r.get("date", "") >= cutoff_7d]
                older = [r for r in readings if cutoff_14d <= r.get("date", "") < cutoff_7d]
                if older:
                    prior_dominance = older[-1].get("dominance")
                # Keep only last 14 days of readings to avoid file bloat
                readings = [r for r in readings if r.get("date", "") >= cutoff_14d]
            else:
                from datetime import datetime
                readings = []
        except Exception:
            readings = []
            from datetime import datetime

        # Append today's reading
        today = datetime.now().date().isoformat()
        current_dom = float(global_data.btc_dominance)
        # Update only once per day (avoid duplicates)
        if not readings or readings[-1].get("date") != today:
            readings.append({"date": today, "dominance": current_dom})

        # Save updated state
        try:
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(json.dumps({
                "dominance_7d_ago": prior_dominance,
                "daily_readings": readings,
                "last_updated": datetime.now().isoformat(),
            }))
        except Exception:
            pass

        from mae_core.market.signal_adapters.crypto_rotation import from_btc_dominance
        sig = from_btc_dominance(global_data, prior_dominance=prior_dominance)
        if sig is not None:
            signals.append(sig)

    except Exception as e:
        logger.debug("BTC dominance fetch failed: %s", e)
    return signals


def fetch_binance_derivatives(binance_derivatives_client: Any) -> list:
    """Fetch Binance derivatives crowded trade scores (OI + L/S ratio + funding composite)."""
    if binance_derivatives_client is None:
        return []
    signals = []
    try:
        scores = binance_derivatives_client.get_multi_crowded_scores()
        from mae_core.market.signal_adapters.crypto_rotation import from_crowded_trade_score
        for score in scores:
            try:
                sig = from_crowded_trade_score(score)
                if sig is not None:
                    signals.append(sig)
            except Exception:
                pass
    except Exception as e:
        logger.debug("Binance derivatives fetch failed: %s", e)
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
