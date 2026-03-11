"""Polygon.io Bulk Historical Fetcher — fast excavation via paid API.

Uses Polygon's per-ticker aggregate endpoint to fetch daily OHLCV history.
One API call per symbol returns all daily bars (up to 50,000). With a paid
plan (no rate throttle), 3,000+ symbols complete in ~10-15 minutes vs
yfinance's 8-10 hours.

Implements the same get_daily_history(symbol, days) interface as PriceFetcher,
so it's a drop-in replacement for the ExcavationDaemon.

Batch mode: get_daily_history_batch() uses aiohttp for concurrent fetches.
20 concurrent requests → ~85s sequential becomes ~5s. Call from sync code.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import requests

from mae_core.market.apis.price_fetcher import PriceData

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.polygon.io"


def _parse_bars(symbol: str, data: dict) -> List[PriceData]:
    """Parse Polygon.io aggregate response into PriceData list."""
    results: List[PriceData] = []
    for r in data.get("results", []):
        ts_ms = r.get("t", 0)
        if not ts_ms:
            continue

        try:
            bar_date = datetime.fromtimestamp(
                ts_ms / 1000, tz=None
            ).strftime("%Y-%m-%d")
        except (ValueError, OSError):
            continue

        open_price = float(r.get("o", 0))
        close_price = float(r.get("c", 0))
        change_pct = (
            ((close_price - open_price) / open_price * 100)
            if open_price != 0
            else 0.0
        )

        results.append(PriceData(
            symbol=symbol,
            price=close_price,
            timestamp=bar_date,
            source="polygon_history",
            open=open_price,
            high=float(r.get("h", 0)),
            low=float(r.get("l", 0)),
            volume=int(r.get("v", 0)),
            change_pct=change_pct,
        ))

    return results


class PolygonBulkFetcher:
    """Bulk historical price fetcher using Polygon.io paid API.

    Drop-in replacement for PriceFetcher when doing bulk excavation.
    Requires MASSIVE_API_KEY env var (Polygon.io / Massive API key).

    Usage:
        fetcher = PolygonBulkFetcher()
        history = fetcher.get_daily_history("NVDA", days=2000)

        # Batch mode (20x faster for many symbols):
        batch = fetcher.get_daily_history_batch(["AAPL", "NVDA", "MSFT"])
    """

    def __init__(self, api_key: str = None, calls_per_minute: int = 0):
        """
        Args:
            api_key: Polygon.io API key. Falls back to MASSIVE_API_KEY env var.
            calls_per_minute: Rate limit. 0 = no limit (paid plan). Set to 5
                for free tier.
        """
        self._api_key = api_key or os.environ.get("MASSIVE_API_KEY", "")
        self._calls_per_minute = calls_per_minute
        self._session = requests.Session()
        self._session.headers["Accept"] = "application/json"
        self._last_call_time = 0.0
        self._call_count = 0
        self._error_count = 0

    def _build_url(self, symbol: str, days: int) -> tuple[str, dict]:
        """Build Polygon URL and params for a symbol."""
        end = date.today()
        start = end - timedelta(days=days)
        url = (
            f"{_BASE_URL}/v2/aggs/ticker/{symbol}/range/1/day"
            f"/{start.isoformat()}/{end.isoformat()}"
        )
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": "50000",
            "apiKey": self._api_key,
        }
        return url, params

    def get_daily_history(self, symbol: str, days: int = 2000) -> List[PriceData]:
        """Fetch daily OHLCV history via Polygon.io aggregate endpoint.

        Returns List[PriceData] compatible with PriceFetcher interface.
        One API call per symbol — returns all bars in the date range.

        Args:
            symbol: Stock ticker (e.g. "AAPL", "NVDA")
            days: Calendar days of history to fetch

        Returns:
            List of PriceData sorted oldest-first. Empty on failure.
        """
        if not self._api_key:
            return []

        self._rate_limit()

        try:
            url, params = self._build_url(symbol, days)
            resp = self._session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            self._call_count += 1

        except requests.RequestException as e:
            self._error_count += 1
            logger.debug("Polygon fetch failed for %s: %s", symbol, e)
            return []

        return _parse_bars(symbol, data)

    def get_daily_history_batch(
        self,
        symbols: List[str],
        days: int = 2000,
        concurrency: int = 20,
    ) -> Dict[str, List[PriceData]]:
        """Fetch daily history for many symbols concurrently via aiohttp.

        20 concurrent requests turns ~85s sequential into ~5s. Safe to call
        from synchronous code — creates its own event loop if needed.

        Args:
            symbols: List of tickers to fetch.
            days: Calendar days of history per symbol.
            concurrency: Max simultaneous HTTP requests (default 20).

        Returns:
            Dict mapping symbol → List[PriceData]. Missing symbols omitted.
        """
        if not self._api_key or not symbols:
            return {}

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # Already in an async context — run in a thread to avoid nesting
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                future = pool.submit(asyncio.run, self._batch_async(symbols, days, concurrency))
                return future.result()
        else:
            return asyncio.run(self._batch_async(symbols, days, concurrency))

    async def _batch_async(
        self,
        symbols: List[str],
        days: int,
        concurrency: int,
    ) -> Dict[str, List[PriceData]]:
        """Internal async batch fetcher."""
        import aiohttp

        semaphore = asyncio.Semaphore(concurrency)
        results: Dict[str, List[PriceData]] = {}

        async def fetch_one(session: aiohttp.ClientSession, symbol: str):
            async with semaphore:
                url, params = self._build_url(symbol, days)
                try:
                    async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                        if resp.status != 200:
                            self._error_count += 1
                            return
                        data = await resp.json()
                        self._call_count += 1
                        bars = _parse_bars(symbol, data)
                        if bars:
                            results[symbol] = bars
                except Exception as e:
                    self._error_count += 1
                    logger.debug("Polygon async fetch failed for %s: %s", symbol, e)

        async with aiohttp.ClientSession(headers={"Accept": "application/json"}) as session:
            tasks = [fetch_one(session, s) for s in symbols]
            await asyncio.gather(*tasks)

        logger.info(
            "Polygon batch: %d/%d symbols fetched (%d errors)",
            len(results), len(symbols), self._error_count,
        )
        return results

    def get_statistics(self) -> dict:
        """Return fetch statistics."""
        return {
            "calls_made": self._call_count,
            "errors": self._error_count,
            "source": "polygon.io",
        }

    def _rate_limit(self) -> None:
        """Enforce rate limit if configured."""
        if self._calls_per_minute <= 0:
            return

        now = time.time()
        min_interval = 60.0 / self._calls_per_minute
        elapsed = now - self._last_call_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_call_time = time.time()
