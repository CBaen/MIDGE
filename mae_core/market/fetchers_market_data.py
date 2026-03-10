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


def fetch_cot(cot_client: Any, watchlist: dict, converter: Callable) -> list:
    """Fetch CFTC Commitments of Traders for futures watchlist."""
    if cot_client is None:
        return []

    signals = []
    # COT is futures-only — use futures tickers from watchlist or defaults
    futures_tickers = [t for t in watchlist.get("tickers", []) if t.endswith("=F")]
    if not futures_tickers:
        futures_tickers = ["ES=F", "NQ=F", "GC=F", "CL=F"]

    try:
        positions = cot_client.get_latest_positions(futures_tickers)
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
