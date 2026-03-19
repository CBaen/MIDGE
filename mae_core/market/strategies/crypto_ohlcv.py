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
_TTL_SECONDS = 120                              # 2 minutes — fast enough for 5m bars


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

def get_ohlcv(symbol: str, days: int = 90, interval: str = "1d") -> Optional[object]:
    """Return a pandas DataFrame of OHLCV bars for *symbol*.

    Columns: Open, High, Low, Close, Volume (capitalised).
    Index: pandas DatetimeIndex.

    Supports any yfinance interval: 1m, 2m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo.
    For intraday intervals (5m, 15m, etc.), yfinance provides up to 60 days.

    Results are cached for 2 minutes per (symbol, days, interval) triple.

    Parameters
    ----------
    symbol : str
        Yahoo Finance ticker, e.g. "BTC-USD".
    days : int
        Calendar days of history to request (default 90).
    interval : str
        Bar interval — "5m", "15m", "1h", "1d" etc. (default "1d").

    Returns
    -------
    pd.DataFrame or None
        None if yfinance is unavailable, returns fewer than 15 bars,
        or encounters any fetch error.
    """
    if not _PANDAS_AVAILABLE:
        return None

    key = _cache_key(symbol, days) + f":{interval}"
    cached = _get_cached(key)
    if cached is not None:
        return cached

    try:
        import yfinance as yf
        period = f"{days}d"
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df is None or len(df) < 15:
            logger.debug("get_ohlcv(%s, %s, %s): only %d bars", symbol, period, interval, len(df) if df is not None else 0)
            return None
        # Normalize columns (yfinance returns capitalized + extras)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
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
