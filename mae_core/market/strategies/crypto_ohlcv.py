"""
crypto_ohlcv.py - Thin OHLCV wrapper for the strategy layer.

Converts PriceFetcher.get_daily_history() output into a pandas DataFrame
with a proper DatetimeIndex and capitalised OHLCV columns.  A module-level
5-minute TTL cache avoids hammering yfinance when multiple strategies run
against the same symbol in the same daemon step.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger("midge.market.strategies")

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False
    logger.warning("pandas not installed — crypto_ohlcv.get_ohlcv() will always return None")

from mae_core.market.apis.price_fetcher import PriceFetcher

# ---------------------------------------------------------------------------
# Module-level TTL cache
# ---------------------------------------------------------------------------
_CACHE: dict[str, tuple[object, float]] = {}   # key -> (DataFrame, expires_at)
_TTL_SECONDS = 300                              # 5 minutes


def _cache_key(symbol: str, days: int) -> str:
    return f"{symbol}:{days}"


def _get_cached(key: str) -> Optional[object]:
    entry = _CACHE.get(key)
    if entry is None:
        return None
    df, expires_at = entry
    if time.monotonic() > expires_at:
        del _CACHE[key]
        return None
    return df


def _set_cached(key: str, df: object) -> None:
    _CACHE[key] = (df, time.monotonic() + _TTL_SECONDS)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_ohlcv(symbol: str, days: int = 90) -> Optional[object]:  # returns pd.DataFrame
    """Return a pandas DataFrame of daily OHLCV bars for *symbol*.

    Columns: Open, High, Low, Close, Volume (capitalised).
    Index: pandas DatetimeIndex (UTC-naive, date precision).

    Uses PriceFetcher.get_daily_history() which pulls from yfinance.
    Works for crypto tickers (BTC-USD, ETH-USD, SOL-USD) and equities.

    Results are cached for 5 minutes per (symbol, days) pair to avoid
    redundant API calls when multiple strategies evaluate the same symbol
    in the same daemon step.

    Parameters
    ----------
    symbol : str
        Yahoo Finance ticker, e.g. "BTC-USD".
    days : int
        Calendar days of history to request (default 90).

    Returns
    -------
    pd.DataFrame or None
        None if yfinance is unavailable, returns fewer than 15 bars,
        or encounters any fetch error.
    """
    if not _PANDAS_AVAILABLE:
        return None

    key = _cache_key(symbol, days)
    cached = _get_cached(key)
    if cached is not None:
        return cached

    fetcher = PriceFetcher(alpha_vantage_key=None)
    records = fetcher.get_daily_history(symbol, days=days)

    if len(records) < 15:
        logger.debug("get_ohlcv(%s, %d): only %d bars — returning None", symbol, days, len(records))
        return None

    try:
        df = pd.DataFrame(
            {
                "Open":   [float(r.open) if r.open is not None else float('nan') for r in records],
                "High":   [float(r.high) if r.high is not None else float('nan') for r in records],
                "Low":    [float(r.low) if r.low is not None else float('nan') for r in records],
                "Close":  [float(r.price) if r.price is not None else float('nan') for r in records],
                "Volume": [float(r.volume) if r.volume is not None else 0.0 for r in records],
            },
            index=pd.to_datetime([r.timestamp for r in records]),
        )
        df.index.name = "Date"
        # Drop rows with NaN in OHLC (incomplete bars from yfinance)
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        if len(df) < 15:
            return None
        _set_cached(key, df)
        return df
    except Exception as exc:
        logger.warning("get_ohlcv(%s): DataFrame construction failed: %s", symbol, exc)
        return None


def clear_cache() -> None:
    """Evict all cached DataFrames (useful in tests)."""
    _CACHE.clear()
