#!/usr/bin/env python3
"""
finnhub_client.py - Finnhub Market Data Client

Fetches news sentiment, earnings calendar, and social buzz via the Finnhub
free-tier REST API (60 calls/min).

API base: https://finnhub.io/api/v1
Auth: ?token={API_KEY} query parameter (env var: MAE_FINNHUB_API_KEY)

Signals produced:
  - NewsSentiment: bullish/bearish percentages + buzz score for a ticker
  - EarningsEvent:  upcoming or recently-reported earnings with EPS/revenue

Sub-modules:
  finnhub_models.py  — dataclasses (NewsSentiment, EconomicEvent, AnalystRec, EarningsEvent)
  finnhub_parsers.py — static parser functions for each Finnhub endpoint
"""

import os
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import requests

# Re-export models so all existing imports continue to work
from mae_core.market.apis.finnhub_models import (
    NewsSentiment,
    EconomicEvent,
    AnalystRec,
    EarningsEvent,
)

# Re-export parsers (used internally and by tests)
from mae_core.market.apis.finnhub_parsers import (
    parse_sentiment as _parse_sentiment_fn,
    parse_earnings_calendar as _parse_earnings_calendar_fn,
    parse_economic_calendar as _parse_economic_calendar_fn,
    parse_analyst_recommendations as _parse_analyst_recommendations_fn,
)

logger = logging.getLogger(__name__)

# Finnhub free tier: 60 calls/min → 1 per second, be conservative
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
REQUEST_DELAY = 1.0  # seconds between calls

# Cache TTLs
SENTIMENT_CACHE_TTL = 15 * 60   # 15 minutes — news changes quickly
EARNINGS_CACHE_TTL  = 60 * 60   # 1 hour — calendar updates infrequently


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class FinnhubClient:
    """
    Client for the Finnhub free-tier REST API.

    Fetches:
      - News sentiment per ticker    (/news-sentiment)
      - Earnings calendar            (/calendar/earnings)

    Rate limit: 1 request/second (conservative for 60/min free tier).
    Caching:    15 min for sentiment, 1 hour for earnings.

    Usage:
        client = FinnhubClient()           # reads MAE_FINNHUB_API_KEY from env
        sentiment = client.get_news_sentiment("AAPL")
        upcoming = client.get_upcoming_earnings(days=14)
    """

    def __init__(self, api_key: Optional[str] = None, provider=None, raw_store=None):
        """
        Initialize the Finnhub client.

        Args:
            api_key:  Finnhub API key. Falls back to MAE_FINNHUB_API_KEY env var.
            provider: Optional MarketDataProvider for gateway routing.
            raw_store: Optional RawStore for persisting all API data.
        """
        self._api_key = api_key or os.environ.get("MAE_FINNHUB_API_KEY")
        self._provider = provider
        self._raw_store = raw_store

        if not self._api_key:
            logger.warning(
                "No Finnhub API key — set MAE_FINNHUB_API_KEY or pass api_key=. "
                "All requests will fail."
            )

        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "MIDGE Trading Research"})

        self._last_request_time: float = 0.0

        # Per-endpoint caches: { cache_key: (data, fetched_at_unix) }
        self._sentiment_cache: Dict[str, Tuple[NewsSentiment, float]] = {}
        self._earnings_cache: Dict[str, Tuple[List[EarningsEvent], float]] = {}

        # Blocked endpoints: { endpoint: blocked_until_unix }
        # Endpoints returning 403 are blocked for 1 hour to prevent spam
        self._blocked_endpoints: Dict[str, float] = {}
        self._block_duration: float = 60 * 60  # 1 hour

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_news_sentiment(self, symbol: str) -> Optional[NewsSentiment]:
        """
        Get news sentiment for a single ticker.

        Args:
            symbol: Stock ticker (e.g. "AAPL")

        Returns:
            NewsSentiment or None if unavailable / no API key.
        """
        key = symbol.upper()

        # Check cache
        if key in self._sentiment_cache:
            cached, fetched_at = self._sentiment_cache[key]
            if time.time() - fetched_at < SENTIMENT_CACHE_TTL:
                logger.debug("Sentiment cache hit: %s", key)
                return cached

        data = self._get("/news-sentiment", params={"symbol": key})
        if data is None:
            return None

        # Store full sentiment blob before extracting
        if self._raw_store:
            try:
                self._raw_store.store_finnhub_sentiment(key, data)
            except Exception:
                pass

        result = self._parse_sentiment(key, data)
        if result is not None:
            self._sentiment_cache[key] = (result, time.time())
        return result

    def get_upcoming_earnings(self, days: int = 14) -> List[EarningsEvent]:
        """
        Get earnings calendar for the next N days.

        Args:
            days: How many days ahead to look (default 14).

        Returns:
            List of EarningsEvent objects sorted by date ascending.
        """
        today = datetime.now(timezone.utc).date()
        from_date = today.strftime("%Y-%m-%d")
        to_date = (today + timedelta(days=days)).strftime("%Y-%m-%d")
        cache_key = f"upcoming_{from_date}_{to_date}"

        if cache_key in self._earnings_cache:
            cached, fetched_at = self._earnings_cache[cache_key]
            if time.time() - fetched_at < EARNINGS_CACHE_TTL:
                logger.debug("Earnings cache hit: %s", cache_key)
                return cached

        data = self._get("/calendar/earnings", params={"from": from_date, "to": to_date})
        if data is None:
            return []

        # Store ALL earnings data (including quarter/year that get discarded)
        if self._raw_store:
            try:
                self._raw_store.store_finnhub_earnings(data.get("earningsCalendar") or [])
            except Exception:
                pass

        events = self._parse_earnings_calendar(data)
        # Upcoming: exclude entries that already have actual results
        events = [e for e in events if not e.is_reported()]
        events.sort(key=lambda e: e.date)
        self._earnings_cache[cache_key] = (events, time.time())
        return events

    def get_recent_earnings_surprises(self, days: int = 7) -> List[EarningsEvent]:
        """
        Get earnings that have already reported (eps_actual is not None).

        Args:
            days: How many days back to look (default 7).

        Returns:
            List of EarningsEvent objects where actual results are available,
            sorted by date descending (most recent first).
        """
        today = datetime.now(timezone.utc).date()
        from_date = (today - timedelta(days=days)).strftime("%Y-%m-%d")
        to_date = today.strftime("%Y-%m-%d")
        cache_key = f"reported_{from_date}_{to_date}"

        if cache_key in self._earnings_cache:
            cached, fetched_at = self._earnings_cache[cache_key]
            if time.time() - fetched_at < EARNINGS_CACHE_TTL:
                logger.debug("Earnings (reported) cache hit: %s", cache_key)
                return cached

        data = self._get("/calendar/earnings", params={"from": from_date, "to": to_date})
        if data is None:
            return []

        if self._raw_store:
            try:
                self._raw_store.store_finnhub_earnings(data.get("earningsCalendar") or [])
            except Exception:
                pass

        events = self._parse_earnings_calendar(data)
        # Reported only
        reported = [e for e in events if e.is_reported()]
        reported.sort(key=lambda e: e.date, reverse=True)
        self._earnings_cache[cache_key] = (reported, time.time())
        return reported

    def get_watchlist_sentiment(self, tickers: List[str]) -> List[NewsSentiment]:
        """
        Get sentiment for multiple tickers with rate limiting between calls.

        Args:
            tickers: List of stock ticker symbols.

        Returns:
            List of NewsSentiment objects (only for tickers where data
            was successfully retrieved).
        """
        results: List[NewsSentiment] = []
        for i, ticker in enumerate(tickers):
            sentiment = self.get_news_sentiment(ticker)
            if sentiment is not None:
                results.append(sentiment)
            # Rate-limit gap between requests (already enforced inside _get,
            # but we also pause explicitly after each ticker for safety when
            # iterating a watchlist in quick succession).
            if i < len(tickers) - 1:
                time.sleep(REQUEST_DELAY)
        return results

    def get_economic_calendar(self, days: int = 14) -> List[EconomicEvent]:
        """
        Get upcoming economic events (FOMC, CPI, NFP, etc.).

        Args:
            days: How many days ahead to look (default 14).

        Returns:
            List of EconomicEvent objects sorted by date ascending.
        """
        today = datetime.now(timezone.utc).date()
        from_date = today.strftime("%Y-%m-%d")
        to_date = (today + timedelta(days=days)).strftime("%Y-%m-%d")

        data = self._get("/calendar/economic", params={"from": from_date, "to": to_date})
        if data is None:
            return []

        # Store ALL countries before filtering to US-only
        if self._raw_store:
            try:
                self._raw_store.store_finnhub_economic(data.get("economicCalendar") or [])
            except Exception:
                pass

        return self._parse_economic_calendar(data)

    def get_analyst_recommendations(self, symbol: str) -> List[AnalystRec]:
        """
        Get analyst recommendation trends for a ticker.

        Args:
            symbol: Stock ticker (e.g. "AAPL")

        Returns:
            List of AnalystRec objects (most recent first), typically 4 quarters.
        """
        data = self._get("/stock/recommendation", params={"symbol": symbol.upper()})
        if data is None or not isinstance(data, list):
            return []

        return self._parse_analyst_recommendations(data)

    def get_earnings_calendar(self, days: int = 14) -> List[EarningsEvent]:
        """
        Get earnings calendar including both upcoming and recently reported.

        Args:
            days: How many days ahead AND back to look.

        Returns:
            List of EarningsEvent objects sorted by date.
        """
        today = datetime.now(timezone.utc).date()
        from_date = (today - timedelta(days=days)).strftime("%Y-%m-%d")
        to_date = (today + timedelta(days=days)).strftime("%Y-%m-%d")

        data = self._get("/calendar/earnings", params={"from": from_date, "to": to_date})
        if data is None:
            return []

        if self._raw_store:
            try:
                self._raw_store.store_finnhub_earnings(data.get("earningsCalendar") or [])
            except Exception:
                pass

        events = self._parse_earnings_calendar(data)
        events.sort(key=lambda e: e.date)
        return events

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rate_limit(self) -> None:
        """Block until at least REQUEST_DELAY seconds since the last call."""
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()

    def _get(self, endpoint: str, params: Optional[dict] = None) -> Optional[dict]:
        """
        Make a rate-limited GET request to the Finnhub API.

        Routes through MarketDataProvider when a provider is configured,
        otherwise uses requests.Session directly.

        Args:
            endpoint: Path relative to FINNHUB_BASE_URL (e.g. "/news-sentiment").
            params:   Additional query parameters (token is added automatically).

        Returns:
            Parsed JSON dict, or None on error / missing key.
        """
        if not self._api_key:
            logger.warning("Finnhub request skipped — no API key configured.")
            return None

        # Check if endpoint is blocked (403 cooldown)
        if endpoint in self._blocked_endpoints:
            if time.time() < self._blocked_endpoints[endpoint]:
                return None  # Silently skip — already logged on first 403
            else:
                del self._blocked_endpoints[endpoint]
                logger.info("Finnhub endpoint %s unblocked — retrying", endpoint)

        all_params = dict(params or {})
        all_params["token"] = self._api_key

        url = f"{FINNHUB_BASE_URL}{endpoint}"

        self._rate_limit()

        if self._provider is not None:
            from mae_core.market.apis.market_data_provider import market_request
            from mae_core.external.api_client import ApiResponseStatus

            resp = market_request(
                self._provider,
                url,
                params=all_params,
                source_name="finnhub",
                timeout_ms=10000.0,
            )
            if resp.status == ApiResponseStatus.SUCCESS:
                return resp.payload
            # Detect 403 in error message and block the endpoint
            if resp.error_message and "403" in str(resp.error_message):
                self._blocked_endpoints[endpoint] = time.time() + self._block_duration
                logger.warning(
                    "Finnhub 403 on %s — blocked for %.0f min (upgrade API tier to access)",
                    endpoint, self._block_duration / 60,
                )
                return None
            logger.warning("Finnhub provider error [%s]: %s", endpoint, resp.error_message)
            return None

        try:
            response = self._session.get(url, params=all_params, timeout=10)
            if response.status_code == 200:
                return response.json()
            if response.status_code == 403:
                self._blocked_endpoints[endpoint] = time.time() + self._block_duration
                logger.warning(
                    "Finnhub 403 on %s — blocked for %.0f min (upgrade API tier to access)",
                    endpoint, self._block_duration / 60,
                )
                return None
            if response.status_code == 429:
                logger.warning("Finnhub rate limited (429) — back off and retry.")
            else:
                logger.warning(
                    "Finnhub HTTP %d for %s", response.status_code, endpoint
                )
            return None
        except Exception as exc:
            logger.error("Finnhub request failed [%s]: %s", endpoint, exc)
            return None

    # ------------------------------------------------------------------
    # Parser delegates — thin wrappers so internal call sites are unchanged
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_sentiment(symbol: str, data: dict) -> Optional[NewsSentiment]:
        return _parse_sentiment_fn(symbol, data)

    @staticmethod
    def _parse_earnings_calendar(data: dict) -> List[EarningsEvent]:
        return _parse_earnings_calendar_fn(data)

    @staticmethod
    def _parse_economic_calendar(data: dict) -> List[EconomicEvent]:
        return _parse_economic_calendar_fn(data)

    @staticmethod
    def _parse_analyst_recommendations(data: list) -> List[AnalystRec]:
        return _parse_analyst_recommendations_fn(data)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def get_news_sentiment(symbol: str) -> Optional[NewsSentiment]:
    """Convenience wrapper — one-shot sentiment fetch for a ticker."""
    client = FinnhubClient()
    return client.get_news_sentiment(symbol)


def get_upcoming_earnings(days: int = 14) -> List[EarningsEvent]:
    """Convenience wrapper — upcoming earnings calendar."""
    client = FinnhubClient()
    return client.get_upcoming_earnings(days=days)


# ---------------------------------------------------------------------------
# Manual test block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    api_key = os.environ.get("MAE_FINNHUB_API_KEY")
    if not api_key:
        print("ERROR: MAE_FINNHUB_API_KEY not set.")
        print("       export MAE_FINNHUB_API_KEY=<your_key>  (Linux/Mac)")
        print("       set MAE_FINNHUB_API_KEY=<your_key>     (Windows CMD)")
        sys.exit(1)

    client = FinnhubClient(api_key=api_key)

    # ---- News sentiment ------------------------------------------------
    print("\n--- News Sentiment ---")
    test_tickers = ["AAPL", "MSFT", "NVDA"]
    for ticker in test_tickers:
        result = client.get_news_sentiment(ticker)
        if result:
            print(f"  {result.to_plain_language()}")
        else:
            print(f"  {ticker}: no data returned")

    # ---- Watchlist batch -----------------------------------------------
    print("\n--- Watchlist Sentiment (rate-limited) ---")
    watchlist = ["AMZN", "META"]
    sentiments = client.get_watchlist_sentiment(watchlist)
    for s in sentiments:
        print(f"  {s.to_plain_language()}")
    if not sentiments:
        print("  (none returned)")

    # ---- Upcoming earnings ---------------------------------------------
    print("\n--- Upcoming Earnings (next 14 days) ---")
    upcoming = client.get_upcoming_earnings(days=14)
    print(f"  Found {len(upcoming)} events")
    for event in upcoming[:10]:
        print(f"  {event.to_plain_language()}")

    # ---- Recent earnings surprises -------------------------------------
    print("\n--- Recent Earnings Surprises (last 7 days) ---")
    reported = client.get_recent_earnings_surprises(days=7)
    print(f"  Found {len(reported)} reported earnings")
    for event in reported[:10]:
        print(f"  {event.to_plain_language()}")

    print("\nDone.")
