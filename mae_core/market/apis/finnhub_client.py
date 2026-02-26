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
"""

import os
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# Finnhub free tier: 60 calls/min → 1 per second, be conservative
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
REQUEST_DELAY = 1.0  # seconds between calls

# Cache TTLs
SENTIMENT_CACHE_TTL = 15 * 60   # 15 minutes — news changes quickly
EARNINGS_CACHE_TTL  = 60 * 60   # 1 hour — calendar updates infrequently


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class NewsSentiment:
    """
    News sentiment snapshot for a single ticker from Finnhub.

    buzz_score = articles_in_last_week / weekly_average
    Values > 1.0 mean above-average coverage volume.
    """
    ticker: str
    bullish_pct: float          # 0.0 – 1.0  (e.g. 0.65 = 65% bullish)
    bearish_pct: float          # 0.0 – 1.0
    news_score: float           # Finnhub's companyNewsScore (0–1)
    buzz_score: float           # articles_in_last_week / weekly_average
    detected_at: str            # ISO-8601 timestamp when we fetched this
    signal_source: str = "finnhub_news"
    decay_rate: float = 0.20    # ~3-day half-life; sentiment fades fast
    confidence: float = 0.50    # Base confidence before Bayesian update

    def to_plain_language(self) -> str:
        """Format for dashboard display."""
        buzz_label = "above-average" if self.buzz_score > 1.0 else "below-average"
        sentiment_label = "bullish" if self.bullish_pct >= 0.5 else "bearish"
        return (
            f"{self.ticker}: {sentiment_label} ({self.bullish_pct:.0%} bullish, "
            f"{self.bearish_pct:.0%} bearish), {buzz_label} buzz "
            f"(score {self.buzz_score:.2f}), news score {self.news_score:.2f}"
        )


@dataclass
class EarningsEvent:
    """
    Single earnings calendar entry from Finnhub.

    eps_actual and revenue_actual are None when the report has not
    yet been released.  Once reported they are floats.
    """
    symbol: str
    date: str                           # YYYY-MM-DD
    eps_estimate: Optional[float]
    eps_actual: Optional[float]         # None if not yet reported
    revenue_estimate: Optional[float]
    revenue_actual: Optional[float]     # None if not yet reported
    hour: str                           # "bmo" (before market open) | "amc" (after market close)
    signal_source: str = "finnhub_earnings"
    decay_rate: float = 0.25            # Earnings are priced in fast
    confidence: float = 0.65           # Higher base; hard calendar events

    def is_reported(self) -> bool:
        """True when actual results have come in."""
        return self.eps_actual is not None

    def eps_surprise_pct(self) -> Optional[float]:
        """
        Percentage EPS surprise vs estimate.

        Returns None when either value is missing or estimate is zero.
        """
        if self.eps_actual is None or self.eps_estimate is None:
            return None
        if self.eps_estimate == 0:
            return None
        return (self.eps_actual - self.eps_estimate) / abs(self.eps_estimate)

    def to_plain_language(self) -> str:
        """Format for dashboard display."""
        timing = "before open" if self.hour == "bmo" else "after close"
        if self.is_reported():
            surprise = self.eps_surprise_pct()
            surprise_str = (
                f" — EPS surprise {surprise:+.1%}" if surprise is not None
                else " — EPS reported"
            )
            return f"{self.symbol} reported on {self.date}{surprise_str}"
        else:
            est_str = f"${self.eps_estimate:.2f}" if self.eps_estimate is not None else "N/A"
            return (
                f"{self.symbol} earnings on {self.date} {timing}, "
                f"EPS estimate {est_str}"
            )


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

    def __init__(self, api_key: Optional[str] = None, provider=None):
        """
        Initialize the Finnhub client.

        Args:
            api_key:  Finnhub API key. Falls back to MAE_FINNHUB_API_KEY env var.
            provider: Optional MarketDataProvider for gateway routing.
        """
        self._api_key = api_key or os.environ.get("MAE_FINNHUB_API_KEY")
        self._provider = provider

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

    @staticmethod
    def _parse_sentiment(symbol: str, data: dict) -> Optional[NewsSentiment]:
        """
        Parse /news-sentiment response into a NewsSentiment dataclass.

        Expected keys:
            buzz.articlesInLastWeek, buzz.weeklyAverage,
            companyNewsScore, sentiment.bullishPercent, sentiment.bearishPercent
        """
        try:
            buzz_block = data.get("buzz", {})
            articles_this_week = float(buzz_block.get("articlesInLastWeek", 0))
            weekly_average     = float(buzz_block.get("weeklyAverage", 1) or 1)
            buzz_score         = articles_this_week / weekly_average

            sentiment_block = data.get("sentiment", {})
            bullish_pct = float(sentiment_block.get("bullishPercent", 0.5))
            bearish_pct = float(sentiment_block.get("bearishPercent", 0.5))
            news_score  = float(data.get("companyNewsScore", 0.0))

            return NewsSentiment(
                ticker=symbol,
                bullish_pct=bullish_pct,
                bearish_pct=bearish_pct,
                news_score=news_score,
                buzz_score=buzz_score,
                detected_at=datetime.now(timezone.utc).isoformat(),
            )
        except Exception as exc:
            logger.warning("Failed to parse sentiment for %s: %s", symbol, exc)
            return None

    @staticmethod
    def _parse_earnings_calendar(data: dict) -> List[EarningsEvent]:
        """
        Parse /calendar/earnings response into EarningsEvent objects.

        Expected structure: {"earningsCalendar": [...]}
        Each entry has: symbol, date, epsActual, epsEstimate, hour,
                        quarter, revenueActual, revenueEstimate, year
        """
        events: List[EarningsEvent] = []
        calendar = data.get("earningsCalendar") or []

        for entry in calendar:
            try:
                symbol = (entry.get("symbol") or "").upper()
                if not symbol:
                    continue

                date = entry.get("date") or ""

                # Actual fields are null until the report is released
                eps_actual = entry.get("epsActual")
                eps_estimate = entry.get("epsEstimate")
                rev_actual = entry.get("revenueActual")
                rev_estimate = entry.get("revenueEstimate")

                # Cast non-null numerics; keep None as-is
                eps_actual    = float(eps_actual)    if eps_actual    is not None else None
                eps_estimate  = float(eps_estimate)  if eps_estimate  is not None else None
                rev_actual    = float(rev_actual)    if rev_actual    is not None else None
                rev_estimate  = float(rev_estimate)  if rev_estimate  is not None else None

                hour = (entry.get("hour") or "").lower()  # "bmo" or "amc"

                events.append(EarningsEvent(
                    symbol=symbol,
                    date=date,
                    eps_estimate=eps_estimate,
                    eps_actual=eps_actual,
                    revenue_estimate=rev_estimate,
                    revenue_actual=rev_actual,
                    hour=hour,
                ))
            except Exception as exc:
                logger.debug("Skipping malformed earnings entry: %s | %s", entry, exc)
                continue

        return events


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
