"""Massive (formerly Polygon.io) Client — Institutional-grade market data.

Free tier: 5 API calls/minute, 15-minute delayed, 2 years historical.
Uses REST API directly (no SDK dependency) to match codebase conventions.

Primary endpoint: grouped daily bars — ALL tickers' OHLCV in ONE call.
Generates signals from volume anomalies, significant price moves, and gaps.

Part of Always-On MIDGE data enrichment.
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import requests

logger = logging.getLogger("midge.market.massive")

_BASE_URL = "https://api.polygon.io"
_RATE_LIMIT = 5  # calls per minute (free tier)


@dataclass
class TickerDetails:
    """Reference metadata for a single ticker from Polygon /v3/reference/tickers/{ticker}."""
    ticker: str
    name: str
    market: str          # stocks, crypto, fx, otc, indices
    locale: str          # us, global
    primary_exchange: str
    type: str            # CS (common stock), ETF, FUND, etc.
    active: bool
    currency_name: str
    market_cap: float
    sic_code: str
    sic_description: str


@dataclass
class TickerBar:
    """Single day's OHLCV for a ticker."""
    ticker: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float
    transactions: int
    date: str  # YYYY-MM-DD


@dataclass
class TickerSnapshot:
    """Processed snapshot with derived signal metrics."""
    ticker: str
    close: float
    prev_close: float
    volume: float
    avg_volume_20d: float
    daily_return_pct: float
    gap_pct: float  # open vs prev_close
    volume_ratio: float  # volume / avg_volume_20d
    high: float
    low: float
    open: float
    vwap: float
    bar_date: str


class MassiveClient:
    """Massive/Polygon.io REST client for free-tier market data.

    Strategy: Use the grouped daily endpoint (1 call = ALL tickers) for
    the most recent trading day, then compare against historical bars for
    volume and price anomaly detection.
    """

    def __init__(self, api_key: str = None, raw_store=None):
        self._api_key = api_key or os.environ.get("MASSIVE_API_KEY", "")
        self._raw_store = raw_store
        self._session = requests.Session()
        self._session.headers["Accept"] = "application/json"
        self._call_times: deque = deque(maxlen=_RATE_LIMIT)
        # Cache: grouped daily results keyed by date string
        self._grouped_cache: Dict[str, List[dict]] = {}

    def _rate_limit(self):
        """Enforce 5 calls/min rate limit."""
        now = time.time()
        if len(self._call_times) >= _RATE_LIMIT:
            oldest = self._call_times[0]
            elapsed = now - oldest
            if elapsed < 60:
                sleep_time = 60 - elapsed + 0.1
                logger.debug("Massive rate limit: sleeping %.1fs", sleep_time)
                time.sleep(sleep_time)
        self._call_times.append(time.time())

    def _get(self, path: str, params: dict = None) -> dict:
        """Make a rate-limited GET request. Returns empty dict on failure."""
        if not self._api_key:
            logger.debug("Massive: no API key configured")
            return {}
        self._rate_limit()
        url = f"{_BASE_URL}{path}"
        all_params = {"apiKey": self._api_key}
        if params:
            all_params.update(params)
        try:
            resp = self._session.get(url, params=all_params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            logger.debug("Massive request failed: %s", path, exc_info=True)
            return {}

    def get_grouped_daily(self, for_date: date = None) -> List[TickerBar]:
        """Get ALL tickers' daily OHLCV for a given date. ONE API call.

        Returns bars for every US stock traded on that date. The caller
        filters to the watchlist.

        Default: most recent business day.
        """
        if for_date is None:
            for_date = self._last_business_day()

        date_str = for_date.strftime("%Y-%m-%d")

        # Check cache
        if date_str in self._grouped_cache:
            return self._parse_bars(self._grouped_cache[date_str], date_str)

        data = self._get(
            f"/v2/aggs/grouped/locale/us/market/stocks/{date_str}",
            {"adjusted": "true"},
        )
        results = data.get("results", [])
        if results:
            self._grouped_cache[date_str] = results
        bars = self._parse_bars(results, date_str)
        if self._raw_store is not None and bars:
            try:
                self._raw_store.store_massive_bars(bars)
            except Exception:
                pass
        return bars

    def get_previous_close(self, ticker: str) -> Optional[TickerBar]:
        """Get previous day's OHLCV for a single ticker. ONE API call."""
        data = self._get(
            f"/v2/aggs/ticker/{ticker}/prev",
            {"adjusted": "true"},
        )
        results = data.get("results", [])
        if not results:
            return None
        r = results[0]
        return TickerBar(
            ticker=r.get("T", ticker),
            open=float(r.get("o", 0)),
            high=float(r.get("h", 0)),
            low=float(r.get("l", 0)),
            close=float(r.get("c", 0)),
            volume=float(r.get("v", 0)),
            vwap=float(r.get("vw", 0)),
            transactions=int(r.get("n", 0)),
            date=self._ts_to_date(r.get("t", 0)),
        )

    def get_ticker_bars(self, ticker: str, days: int = 20) -> List[TickerBar]:
        """Get daily bars for a single ticker over N days. ONE API call.

        Used for computing rolling averages (volume baseline, volatility).
        """
        end = date.today()
        start = end - timedelta(days=days + 5)  # buffer for weekends/holidays
        data = self._get(
            f"/v2/aggs/ticker/{ticker}/range/1/day/{start.isoformat()}/{end.isoformat()}",
            {"adjusted": "true", "sort": "asc", "limit": str(days + 5)},
        )
        results = data.get("results", [])
        bars = []
        for r in results:
            bars.append(TickerBar(
                ticker=r.get("T", ticker),
                open=float(r.get("o", 0)),
                high=float(r.get("h", 0)),
                low=float(r.get("l", 0)),
                close=float(r.get("c", 0)),
                volume=float(r.get("v", 0)),
                vwap=float(r.get("vw", 0)),
                transactions=int(r.get("n", 0)),
                date=self._ts_to_date(r.get("t", 0)),
            ))
        return bars[-days:]  # trim to requested window

    def get_ticker_details(self, ticker: str) -> Optional[TickerDetails]:
        """Get reference metadata for a single ticker. ONE API call.

        Uses Polygon /v3/reference/tickers/{ticker} (free tier, no daily bar
        quota consumed). Returns sector/industry/SIC/market-cap context useful
        for domain routing and excavation enrichment.

        Does NOT participate in the rotation system — call on-demand during
        excavation or when enriching a ticker for the first time.
        """
        data = self._get(f"/v3/reference/tickers/{ticker.upper()}")
        result = data.get("results")
        if not result:
            return None
        try:
            details = TickerDetails(
                ticker=result.get("ticker", ticker),
                name=result.get("name", ""),
                market=result.get("market", ""),
                locale=result.get("locale", ""),
                primary_exchange=result.get("primary_exchange", ""),
                type=result.get("type", ""),
                active=bool(result.get("active", True)),
                currency_name=result.get("currency_name", ""),
                market_cap=float(result.get("market_cap") or 0.0),
                sic_code=str(result.get("sic_code") or ""),
                sic_description=result.get("sic_description", ""),
            )
        except (TypeError, ValueError):
            logger.debug("Massive: failed to parse ticker details for %s", ticker)
            return None
        if self._raw_store is not None:
            try:
                self._raw_store.store_polygon_ticker_details([details])
            except Exception:
                pass
        return details

    def build_snapshots(
        self, tickers: List[str], bars_by_ticker: Dict[str, List[TickerBar]] = None,
    ) -> List[TickerSnapshot]:
        """Build enriched snapshots from grouped daily + historical context.

        If bars_by_ticker is None, uses only the grouped daily data with
        no historical context (volume_ratio and avg_volume will be zero).
        """
        # Get grouped daily for most recent trading day
        grouped = self.get_grouped_daily()
        ticker_set = {t.upper() for t in tickers}

        # Index grouped by ticker
        latest: Dict[str, TickerBar] = {}
        for bar in grouped:
            if bar.ticker in ticker_set:
                latest[bar.ticker] = bar

        # Also try previous day for gap calculation
        prev_date = self._last_business_day(self._last_business_day() - timedelta(days=1))
        grouped_prev = self.get_grouped_daily(prev_date)
        prev_bars: Dict[str, TickerBar] = {}
        for bar in grouped_prev:
            if bar.ticker in ticker_set:
                prev_bars[bar.ticker] = bar

        snapshots = []
        for ticker in ticker_set:
            bar = latest.get(ticker)
            if bar is None:
                continue

            prev_bar = prev_bars.get(ticker)
            prev_close = prev_bar.close if prev_bar else bar.open

            # Historical volume average
            avg_vol = 0.0
            if bars_by_ticker and ticker in bars_by_ticker:
                hist = bars_by_ticker[ticker]
                if hist:
                    avg_vol = sum(b.volume for b in hist) / len(hist)

            volume_ratio = bar.volume / avg_vol if avg_vol > 0 else 0.0
            daily_return = ((bar.close - prev_close) / prev_close * 100) if prev_close else 0.0
            gap_pct = ((bar.open - prev_close) / prev_close * 100) if prev_close else 0.0

            snapshots.append(TickerSnapshot(
                ticker=ticker,
                close=bar.close,
                prev_close=prev_close,
                volume=bar.volume,
                avg_volume_20d=avg_vol,
                daily_return_pct=daily_return,
                gap_pct=gap_pct,
                volume_ratio=volume_ratio,
                high=bar.high,
                low=bar.low,
                open=bar.open,
                vwap=bar.vwap,
                bar_date=bar.date,
            ))

        return snapshots

    # --- Helpers ---

    @staticmethod
    def _last_business_day(ref: date = None) -> date:
        """Most recent business day (Mon-Fri) before or on ref date."""
        d = ref or date.today() - timedelta(days=1)
        while d.weekday() >= 5:  # Sat=5, Sun=6
            d -= timedelta(days=1)
        return d

    @staticmethod
    def _ts_to_date(ts_ms: int) -> str:
        """Convert Unix timestamp (ms) to YYYY-MM-DD string."""
        if not ts_ms:
            return ""
        try:
            return datetime.fromtimestamp(ts_ms / 1000, tz=None).strftime("%Y-%m-%d")
        except (ValueError, OSError):
            return ""

    @staticmethod
    def _parse_bars(results: list, date_str: str) -> List[TickerBar]:
        """Parse raw API results into TickerBar objects."""
        bars = []
        for r in results:
            try:
                bars.append(TickerBar(
                    ticker=r.get("T", ""),
                    open=float(r.get("o", 0)),
                    high=float(r.get("h", 0)),
                    low=float(r.get("l", 0)),
                    close=float(r.get("c", 0)),
                    volume=float(r.get("v", 0)),
                    vwap=float(r.get("vw", 0)),
                    transactions=int(r.get("n", 0)),
                    date=date_str,
                ))
            except (TypeError, ValueError):
                continue
        return bars
