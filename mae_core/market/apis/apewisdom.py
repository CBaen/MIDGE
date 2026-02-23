#!/usr/bin/env python3
"""
apewisdom.py - Reddit/WSB Social Sentiment Tracker

Fetches ticker mention velocity from the free ApeWisdom API (no key required).
Tracks how frequently stocks are mentioned on Reddit, primarily WallStreetBets.

Social sentiment is a noisy-but-real signal:
- Sudden acceleration (mention_change >> 1.0) often precedes short squeezes
- Sustained high mention counts can indicate retail accumulation
- Useful as confirmation layer for insider/congressional signals, not standalone
- Data updates every 30 minutes on ApeWisdom's side

Data source: https://apewisdom.io/api/v1.0/filter/all-stocks/page/1
No API key required. Free tier only. Be respectful with rate limits.
"""

import time
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict

import requests

logger = logging.getLogger(__name__)


# API endpoint
APEWISDOM_API = "https://apewisdom.io/api/v1.0/filter/all-stocks/page/1"

# Rate limiting: 1 request per 5 seconds (respectful to a free service)
REQUEST_DELAY = 5.0

# Cache duration: 30 minutes (matches ApeWisdom's own update cadence)
CACHE_DURATION = 1800


@dataclass
class SocialSentiment:
    """Reddit mention data for a single ticker."""

    ticker: str
    mentions_24h: int
    mentions_prior_24h: int
    upvotes: int
    rank: int

    # Calculated: how much has mention velocity changed?
    # > 1.0 means accelerating. 2.0 = doubled. 5.0 = strong momentum signal.
    mention_change: float

    # Metadata for MIDGE signal routing
    source_subreddit: str = "wallstreetbets"
    detected_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    signal_source: str = "social_sentiment"

    # Social sentiment decays fast — ~5 day half-life (Reddit attention is ephemeral)
    decay_rate: float = 0.15

    # Social data is noisy: meaningful for momentum confirmation, not entry signals
    confidence: float = 0.45

    def to_plain_language(self) -> str:
        """Format for dashboard display."""
        direction = "accelerating" if self.mention_change >= 2.0 else "steady"
        return (
            f"#{self.rank} {self.ticker}: {self.mentions_24h:,} mentions "
            f"({self.mention_change:.1f}x prior period, {direction})"
        )


class ApeWisdomClient:
    """
    Client for Reddit/WSB social sentiment data via ApeWisdom.

    Tracks mention velocity to detect retail momentum building before
    it fully manifests in price action. Works best as a confirmation
    signal layered with insider or congressional data.

    No API key required. Rate-limited to 1 req/5s. 30-minute cache.
    """

    def __init__(self, provider=None):
        """
        Initialize the ApeWisdom client.

        Args:
            provider: Optional MarketDataProvider for gateway routing.
                      If None, makes direct requests via requests.Session.
        """
        self._provider = provider
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "MIDGE Trading Research"
        })

        self._last_request_time = 0.0
        self._cache: Optional[List[dict]] = None
        self._cache_time: float = 0.0

    def _rate_limit(self) -> None:
        """Enforce minimum 5-second gap between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_DELAY:
            sleep_for = REQUEST_DELAY - elapsed
            logger.debug(f"ApeWisdom rate limit: sleeping {sleep_for:.1f}s")
            time.sleep(sleep_for)
        self._last_request_time = time.time()

    def _request(self, url: str) -> Optional[dict]:
        """Make a GET request through provider or direct session. Returns parsed JSON or None."""
        if self._provider is not None:
            from mae_core.market.apis.market_data_provider import market_request
            from mae_core.external.api_client import ApiResponseStatus
            resp = market_request(
                self._provider,
                url,
                headers={"User-Agent": "MIDGE Trading Research"},
                source_name="apewisdom",
                timeout_ms=30000.0,
            )
            if resp.status == ApiResponseStatus.SUCCESS:
                return resp.payload
            logger.warning(f"ApeWisdom provider request failed: {resp.error_message}")
            return None

        try:
            response = self.session.get(url, timeout=30)
            if response.status_code == 200:
                return response.json()
            logger.warning(f"ApeWisdom HTTP {response.status_code}")
            return None
        except Exception as exc:
            logger.warning(f"ApeWisdom request error: {exc}")
            return None

    def _fetch_raw(self, use_cache: bool = True) -> List[dict]:
        """
        Fetch raw results list from ApeWisdom. Uses 30-minute cache.

        Returns:
            List of raw result dicts from the API, or empty list on failure.
        """
        if use_cache and self._cache is not None:
            if time.time() - self._cache_time < CACHE_DURATION:
                logger.debug("ApeWisdom: returning cached data")
                return self._cache

        self._rate_limit()

        data = self._request(APEWISDOM_API)
        if data is None:
            logger.warning("ApeWisdom: fetch failed, returning empty")
            return []

        results = data.get("results", [])
        if not results:
            logger.warning("ApeWisdom: response had no 'results' array")
            return []

        self._cache = results
        self._cache_time = time.time()
        logger.debug(f"ApeWisdom: fetched {len(results)} tickers")
        return results

    def _parse_result(self, item: dict) -> Optional[SocialSentiment]:
        """
        Parse a single ApeWisdom result dict into a SocialSentiment dataclass.

        Returns None if the item is missing required fields.
        """
        ticker = item.get("ticker", "").strip().upper()
        if not ticker:
            return None

        mentions_24h = int(item.get("mentions", 0))
        mentions_prior = int(item.get("mentions_24h_ago", 0))
        upvotes = int(item.get("upvotes", 0))
        rank = int(item.get("rank", 0))

        # Guard against division by zero — prior period floor is 1
        mention_change = mentions_24h / max(1, mentions_prior)

        return SocialSentiment(
            ticker=ticker,
            mentions_24h=mentions_24h,
            mentions_prior_24h=mentions_prior,
            upvotes=upvotes,
            rank=rank,
            mention_change=round(mention_change, 4),
            detected_at=datetime.utcnow().isoformat(),
        )

    def get_trending_tickers(self, limit: int = 50) -> List[SocialSentiment]:
        """
        Get the top-mentioned tickers from Reddit, ordered by rank.

        Args:
            limit: Maximum number of results to return (ApeWisdom returns up to 25
                   per page; this caps the returned slice).

        Returns:
            List of SocialSentiment objects, ordered by rank (1 = most mentioned).
        """
        raw = self._fetch_raw()
        results = []

        for item in raw:
            parsed = self._parse_result(item)
            if parsed is not None:
                results.append(parsed)
            if len(results) >= limit:
                break

        return results

    def get_accelerating_tickers(
        self,
        min_change: float = 2.0,
        limit: int = 20,
    ) -> List[SocialSentiment]:
        """
        Get tickers where mention velocity is accelerating above a threshold.

        These are the signal: when a ticker suddenly attracts 2x or more
        the mention volume compared to the prior 24h period, retail momentum
        is building. Use this as a confirmation or early-warning filter.

        Args:
            min_change: Minimum mention_change ratio (2.0 = doubled mentions).
                        Set higher (5.0+) for stronger conviction signals only.
            limit: Maximum number of results to return.

        Returns:
            List of SocialSentiment objects with mention_change >= min_change,
            sorted by mention_change descending (strongest acceleration first).
        """
        all_tickers = self._fetch_raw()
        accelerating = []

        for item in all_tickers:
            parsed = self._parse_result(item)
            if parsed is not None and parsed.mention_change >= min_change:
                accelerating.append(parsed)

        # Strongest acceleration first
        accelerating.sort(key=lambda s: s.mention_change, reverse=True)
        return accelerating[:limit]

    def get_by_ticker(self, ticker: str) -> Optional[SocialSentiment]:
        """
        Look up social sentiment for a specific ticker.

        Args:
            ticker: Stock symbol (e.g., "GME", "AAPL")

        Returns:
            SocialSentiment if found in current trending data, else None.
        """
        ticker_upper = ticker.strip().upper()
        raw = self._fetch_raw()

        for item in raw:
            if item.get("ticker", "").strip().upper() == ticker_upper:
                return self._parse_result(item)

        return None


def get_trending_tickers(limit: int = 50) -> List[SocialSentiment]:
    """
    Convenience function to get top-mentioned Reddit tickers.

    Args:
        limit: Maximum tickers to return

    Returns:
        List of SocialSentiment objects
    """
    client = ApeWisdomClient()
    return client.get_trending_tickers(limit)


def get_accelerating_tickers(
    min_change: float = 2.0,
    limit: int = 20,
) -> List[SocialSentiment]:
    """
    Convenience function to get tickers with accelerating mention velocity.

    Args:
        min_change: Minimum mention_change ratio threshold
        limit: Maximum tickers to return

    Returns:
        List of SocialSentiment objects where velocity is accelerating
    """
    client = ApeWisdomClient()
    return client.get_accelerating_tickers(min_change=min_change, limit=limit)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("Testing ApeWisdom Social Sentiment API...")
    print()

    client = ApeWisdomClient()

    # Top trending tickers
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    print(f"Fetching top {limit} trending tickers on Reddit/WSB...")

    trending = client.get_trending_tickers(limit=limit)

    if trending:
        print(f"\nTop {len(trending)} most-mentioned tickers:")
        for s in trending:
            print(f"  {s.to_plain_language()}")

        # Show accelerating tickers
        min_change = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
        print(f"\nTickers with mention acceleration >= {min_change}x:")
        accelerating = client.get_accelerating_tickers(min_change=min_change)
        if accelerating:
            for s in accelerating:
                print(
                    f"  {s.ticker}: {s.mention_change:.1f}x change "
                    f"({s.mentions_24h:,} mentions now vs {s.mentions_prior_24h:,} prior)"
                )
        else:
            print(f"  None found above {min_change}x threshold")

        # Look up specific ticker if provided
        if len(sys.argv) > 3:
            lookup = sys.argv[3].upper()
            print(f"\nLooking up {lookup}...")
            found = client.get_by_ticker(lookup)
            if found:
                print(f"  {found.to_plain_language()}")
                print(f"  Confidence: {found.confidence} | Decay rate: {found.decay_rate}")
            else:
                print(f"  {lookup} not in current trending data")
    else:
        print("No data returned (API may be down or rate-limited)")

    print("\nDone.")
