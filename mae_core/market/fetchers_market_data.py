"""Market data fetch functions — FRED, COT, VIX, EIA, Massive/Polygon.

Covers macroeconomic indicators, futures positioning, fear gauge,
energy inventories, and OHLCV snapshots.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger("midge.market.sensing")


def fetch_fred(fred: Any, converter: Callable) -> list:
    """Fetch FRED macroeconomic indicators."""
    if fred is None:
        return []

    signals = []
    try:
        snapshot = fred.get_macro_snapshot()
        for indicator in snapshot:
            try:
                signals.append(converter(indicator))
            except Exception:
                pass
    except Exception as e:
        logger.debug("FRED macro fetch failed: %s", e)
    return signals


def fetch_fred_yields(fred: Any, converter: Callable) -> list:
    """Fetch forex-critical FRED yield curve and dollar index series."""
    if fred is None:
        return []

    signals = []
    try:
        snapshot = fred.get_yield_curves_and_dollar()
        for indicator in snapshot:
            try:
                signals.append(converter(indicator))
            except Exception:
                pass
    except Exception as e:
        logger.debug("FRED yields fetch failed: %s", e)
    return signals


def fetch_usda(usda_client: Any, converter: Callable) -> list:
    """Fetch USDA agricultural supply/demand indicators for key commodities.

    Queries the USDA PSD (Production, Supply and Distribution) free API for
    wheat, corn, soybeans, and cotton. Returns one MarketSignal per commodity
    with a stocks-to-use ratio as the direction driver.

    Cadence: registered in SOURCE_ROTATION — fetched every cycle but the
    client's 6-hour cache ensures API calls happen at most once per day.
    """
    if usda_client is None:
        return []

    signals = []
    try:
        snapshot = usda_client.get_agricultural_snapshot()
        for indicator in snapshot:
            try:
                signals.append(converter(indicator))
            except Exception:
                pass
    except Exception as e:
        logger.debug("USDA agriculture fetch failed: %s", e)
    return signals


def fetch_cot(cot_client: Any, watchlist: dict, converter: Callable) -> list:
    """Fetch CFTC Commitments of Traders for all mapped contracts (FIX 6: full sweep).

    Previously limited to watchlist futures — now fetches ALL TICKER_TO_CFTC mapped
    contracts so no CFTC positioning data is missed. The client caches for 24h so
    this is cheap to call.
    """
    if cot_client is None:
        return []

    signals = []
    try:
        positions = cot_client.get_latest_positions(symbols=None)  # FIX 6: all mapped contracts
        for pos in positions:
            try:
                signals.append(converter(pos))
            except Exception:
                pass
    except Exception as e:
        logger.debug("COT fetch failed: %s", e)
    return signals


def fetch_vix(vix_client: Any, converter: Callable) -> list:
    """Fetch CBOE VIX term structure."""
    if vix_client is None:
        return []

    signals = []
    try:
        vix = vix_client.get_vix_structure()
        if vix is not None:
            signals.append(converter(vix))
    except Exception as e:
        logger.debug("VIX fetch failed: %s", e)
    return signals


def fetch_eia(eia_client: Any, converter: Callable) -> list:
    """Fetch EIA energy inventory/production indicators."""
    if eia_client is None:
        return []

    signals = []
    try:
        snapshot = eia_client.get_energy_snapshot()
        for indicator in snapshot:
            try:
                signals.append(converter(indicator))
            except Exception:
                pass
    except Exception as e:
        logger.debug("EIA energy fetch failed: %s", e)
    return signals


def fetch_massive_snapshot(
    massive_client: Any,
    watchlist: dict,
    converter: Callable,
) -> list:
    """Fetch Massive/Polygon grouped daily data and generate signals.

    Uses ONE grouped daily call for ALL tickers OHLCV, then fetches 20-day
    historical bars per ticker for accurate volume_ratio calculation.
    Free tier: 5 calls/min — one grouped daily + up to 4 per-ticker bars
    within the rate window. Caps at 4 tickers per cycle to stay under limit.
    """
    if massive_client is None:
        return []

    signals = []
    tickers = watchlist.get("tickers", [])
    if not tickers:
        return []

    # Fetch 20-day historical bars for up to 4 tickers to populate volume_ratio.
    # Capped at 4 because: 1 grouped-daily call + 4 per-ticker = 5 total (free tier limit).
    bars_by_ticker = {}
    for ticker in tickers[:4]:
        try:
            hist = massive_client.get_ticker_bars(ticker, days=20)
            if hist:
                bars_by_ticker[ticker.upper()] = hist
        except Exception as e:
            logger.debug("Massive historical bars failed for %s: %s", ticker, e)

    try:
        snapshots = massive_client.build_snapshots(tickers, bars_by_ticker=bars_by_ticker)
        for snap in snapshots:
            try:
                signals.append(converter(snap))
            except Exception:
                pass
    except Exception as e:
        logger.debug("Massive snapshot fetch failed: %s", e)
    return signals


def fetch_edgar_xbrl(edgar_xbrl_client: Any, watchlist: dict) -> list:
    """Fetch SEC EDGAR XBRL fundamental signals for watchlist tickers."""
    if edgar_xbrl_client is None:
        return []
    from datetime import datetime, timezone
    signals = []
    tickers = watchlist.get("tickers", [])[:10]  # Cap at 10 per cycle (rate limit)
    try:
        results = edgar_xbrl_client.scan_watchlist(tickers)
        for sig in results:
            signals.append({
                "source": "edgar_xbrl",
                "symbol": sig.ticker,
                "asset_class": "stock",
                "domain": "fundamental",
                "direction": sig.direction,
                "strength": sig.strength,
                "confidence": sig.confidence,
                "decay_rate": 0.02,  # Slow decay — fundamentals change quarterly
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "outcome_window_days": 30,
                "metadata": {
                    "revenue_growth_yoy": getattr(sig, "revenue_growth_yoy", None),
                    "debt_cash_ratio": getattr(sig, "debt_cash_ratio", None),
                    "cf_margin": getattr(sig, "cf_margin", None),
                    "reasons": sig.reasons if hasattr(sig, "reasons") else [],
                    "signal_source": "edgar_xbrl",
                },
            })
    except Exception as e:
        logger.debug("EDGAR XBRL fetch failed: %s", e)
    return signals
