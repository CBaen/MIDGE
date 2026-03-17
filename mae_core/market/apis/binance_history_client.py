"""Binance History Client — Free historical OHLCV data.

Fetches daily, hourly, and 4H candles from Binance's public REST API.
No authentication required. Covers 600+ coins back to 2017.

Rate limit: 1,200 requests/minute (IP-based). We use a conservative
2-second delay between requests to stay well under limits.

Cache: 1-hour TTL to avoid re-fetching data that hasn't changed.
Symbol conversion: "BTC/USD" → "BTCUSDT", "ETH/USD" → "ETHUSDT"
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger("midge.market.binance_history")

_SPOT_BASE_URL = "https://api.binance.com"
_REQUEST_DELAY = 2.0  # seconds between requests (conservative)
_CACHE_TTL = 3600.0  # 1 hour


def _to_binance_symbol(symbol: str) -> str:
    """Convert MIDGE format to Binance format.

    "BTC/USD" → "BTCUSDT"
    "ETH/USDT" → "ETHUSDT"
    "BTCUSDT" → "BTCUSDT" (pass-through)
    """
    # Already in Binance format
    if "/" not in symbol:
        return symbol.upper()
    base, quote = symbol.split("/", 1)
    # Normalize quote
    if quote.upper() in ("USD", "USDT"):
        return base.upper() + "USDT"
    return base.upper() + quote.upper()


def _parse_klines(raw: list) -> List[dict]:
    """Parse raw Binance kline arrays into OHLCV dicts.

    Binance kline format:
    [open_time, open, high, low, close, volume, close_time, quote_volume,
     num_trades, taker_buy_base, taker_buy_quote, ignore]
    """
    results = []
    for k in raw:
        try:
            results.append({
                "date": datetime.fromtimestamp(int(k[0]) / 1000, tz=timezone.utc).isoformat(),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "quote_volume": float(k[7]),
                "num_trades": int(k[8]),
            })
        except (IndexError, TypeError, ValueError):
            continue
    return results


class BinanceCryptoHistoryClient:
    """Fetches free historical OHLCV data from Binance public API.

    Endpoint: GET https://api.binance.com/api/v3/klines
    No auth required. 1,200 requests/minute. 1,000 candles per request.
    Covers 600+ coins going back to 2017.

    Usage:
        client = BinanceCryptoHistoryClient()
        candles = client.get_daily_history("BTC/USD", days=365)
        for c in candles:
            print(c["date"], c["close"])
    """

    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "MIDGE Trading Research",
            "Accept": "application/json",
        })
        self._last_request: float = 0.0
        # cache: key → (data, fetched_at)
        self._cache: Dict[str, Tuple[list, float]] = {}

    def _rate_limit(self) -> None:
        """Enforce 2-second minimum between requests."""
        elapsed = time.time() - self._last_request
        if elapsed < _REQUEST_DELAY:
            time.sleep(_REQUEST_DELAY - elapsed)
        self._last_request = time.time()

    def _get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 1000,
        start_ms: Optional[int] = None,
    ) -> list:
        """Fetch klines from Binance REST API.

        Args:
            symbol: Binance format, e.g. "BTCUSDT"
            interval: Binance interval string: "1d", "1h", "4h", etc.
            limit: Max candles per request (Binance cap: 1000)
            start_ms: Optional start timestamp in milliseconds

        Returns:
            Raw kline arrays from Binance, or [] on failure.
        """
        cache_key = f"{symbol}:{interval}:{limit}:{start_ms}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            data, fetched_at = cached
            if time.time() - fetched_at < _CACHE_TTL:
                return data

        self._rate_limit()
        params: dict = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_ms is not None:
            params["startTime"] = start_ms

        try:
            resp = self._session.get(
                f"{_SPOT_BASE_URL}/api/v3/klines",
                params=params,
                timeout=15,
            )
            if resp.status_code != 200:
                logger.warning(
                    "Binance klines HTTP %s for %s %s", resp.status_code, symbol, interval
                )
                return []
            data = resp.json()
            if not isinstance(data, list):
                logger.warning("Binance klines unexpected response type for %s", symbol)
                return []
            self._cache[cache_key] = (data, time.time())
            logger.debug(
                "Binance klines: %d candles fetched for %s %s", len(data), symbol, interval
            )
            return data
        except requests.RequestException as exc:
            logger.warning("Binance klines request failed for %s %s: %s", symbol, interval, exc)
            return []

    def get_daily_history(self, symbol: str, days: int = 365) -> List[dict]:
        """Fetch daily OHLCV candles.

        Args:
            symbol: MIDGE format ("BTC/USD") or Binance format ("BTCUSDT")
            days: How many days of history (max 1000 per request)

        Returns:
            List of dicts with: date, open, high, low, close, volume,
            quote_volume, num_trades. Sorted oldest→newest.
        """
        binance_sym = _to_binance_symbol(symbol)
        limit = min(days, 1000)

        # If more than 1000 days needed, calculate start time
        start_ms = None
        if days > 1000:
            start_dt = datetime.now(tz=timezone.utc) - timedelta(days=days)
            start_ms = int(start_dt.timestamp() * 1000)
            limit = 1000

        raw = self._get_klines(binance_sym, "1d", limit=limit, start_ms=start_ms)
        result = _parse_klines(raw)
        logger.info(
            "Binance daily history: %d candles for %s (%d days requested)",
            len(result), binance_sym, days,
        )
        return result

    def get_hourly_history(self, symbol: str, hours: int = 168) -> List[dict]:
        """Fetch 1H candles for swing trading analysis.

        Args:
            symbol: MIDGE or Binance format
            hours: How many hours of history (max 1000 per request = ~41 days)

        Returns:
            List of OHLCV dicts, sorted oldest→newest.
        """
        binance_sym = _to_binance_symbol(symbol)
        limit = min(hours, 1000)
        raw = self._get_klines(binance_sym, "1h", limit=limit)
        result = _parse_klines(raw)
        logger.info(
            "Binance hourly history: %d candles for %s (%d hours requested)",
            len(result), binance_sym, hours,
        )
        return result

    def get_4h_history(self, symbol: str, bars: int = 500) -> List[dict]:
        """Fetch 4H candles — the sweet spot for crypto swing trading.

        4H candles provide enough granularity to see intraday structure
        without the noise of shorter timeframes. 500 bars = ~83 days.

        Args:
            symbol: MIDGE or Binance format
            bars: Number of 4H bars (max 1000)

        Returns:
            List of OHLCV dicts, sorted oldest→newest.
        """
        binance_sym = _to_binance_symbol(symbol)
        limit = min(bars, 1000)
        raw = self._get_klines(binance_sym, "4h", limit=limit)
        result = _parse_klines(raw)
        logger.info(
            "Binance 4H history: %d candles for %s (%d bars requested)",
            len(result), binance_sym, bars,
        )
        return result

    def get_available_symbols(self) -> List[str]:
        """Return all USDT trading pairs on Binance.

        Cached for 1 hour — the symbol list changes rarely.
        Returns list of Binance-format symbols (e.g., "BTCUSDT").
        """
        cache_key = "_available_symbols"
        cached = self._cache.get(cache_key)
        if cached is not None:
            data, fetched_at = cached
            if time.time() - fetched_at < _CACHE_TTL:
                return data

        self._rate_limit()
        try:
            resp = self._session.get(
                f"{_SPOT_BASE_URL}/api/v3/exchangeInfo",
                timeout=20,
            )
            if resp.status_code != 200:
                logger.warning("Binance exchangeInfo HTTP %s", resp.status_code)
                return []
            info = resp.json()
            symbols = [
                s["symbol"]
                for s in info.get("symbols", [])
                if s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING"
            ]
            self._cache[cache_key] = (symbols, time.time())
            logger.info("Binance available symbols: %d USDT pairs", len(symbols))
            return symbols
        except requests.RequestException as exc:
            logger.warning("Binance exchangeInfo failed: %s", exc)
            return []

    def get_recent_closes(self, symbol: str, days: int = 30) -> List[float]:
        """Convenience method: return only closing prices for the last N days."""
        candles = self.get_daily_history(symbol, days=days)
        return [c["close"] for c in candles]
