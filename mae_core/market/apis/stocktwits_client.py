"""
stocktwits_client.py - StockTwits Social Sentiment Client

Fetches bull/bear sentiment ratios from the StockTwits public API.
No authentication required. Complements ApeWisdom (Reddit) with
StockTwits-specific trader sentiment.

When 70%+ of messages are one direction, it's a crowd sentiment signal.
Useful as a contrarian indicator — extreme bullishness at resistance
or extreme bearishness at support can signal reversals.

API: https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json
Rate limit: generous but undocumented. We use 1s delay to be safe.
"""

import time
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

API_BASE = "https://api.stocktwits.com/api/2"
REQUEST_DELAY = 1.0
CACHE_DURATION = 1800  # 30 minutes


@dataclass
class StockTwitsSentiment:
    """Bull/bear sentiment data from StockTwits for a single ticker."""

    ticker: str
    bull_count: int
    bear_count: int
    bull_ratio: float       # bull / (bull + bear), 0.0-1.0
    total_messages: int
    trending: bool

    detected_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    signal_source: str = "stocktwits_sentiment"
    decay_rate: float = 0.50    # Fast decay — social data is ephemeral
    confidence: float = 0.50    # Low prior — crowd is often wrong

    def to_plain_language(self) -> str:
        stance = "bullish" if self.bull_ratio >= 0.6 else "bearish" if self.bull_ratio <= 0.4 else "mixed"
        trending_tag = " [TRENDING]" if self.trending else ""
        return (
            f"{self.ticker}: {stance} ({self.bull_ratio:.0%} bulls, "
            f"{self.total_messages} messages){trending_tag}"
        )


class StockTwitsClient:
    """
    Client for StockTwits public sentiment API.

    No API key required. Fetches recent messages for a ticker and
    calculates bull/bear sentiment ratio from labeled messages.

    Rate limited to 1 req/s. 30-minute cache per ticker.
    """

    def __init__(self, provider=None):
        self._provider = provider
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "MIDGE Trading Research"})
        self._last_request_time: float = 0.0
        self._cache: Dict[str, tuple] = {}  # {ticker: (StockTwitsSentiment, timestamp)}

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()

    def _request(self, url: str, params: Optional[dict] = None) -> Optional[dict]:
        if self._provider is not None:
            from mae_core.market.apis.market_data_provider import market_request
            from mae_core.external.api_client import ApiResponseStatus
            resp = market_request(
                self._provider, url, params=params,
                source_name="stocktwits", timeout_ms=15000.0,
            )
            if resp.status == ApiResponseStatus.SUCCESS:
                return resp.payload
            return None

        try:
            r = self.session.get(url, params=params, timeout=15)
            if r.status_code == 200:
                return r.json()
            logger.warning("StockTwits HTTP %d for %s", r.status_code, url)
            return None
        except Exception as exc:
            logger.warning("StockTwits request error: %s", exc)
            return None

    def get_sentiment(self, tickers: List[str]) -> List[StockTwitsSentiment]:
        """
        Get bull/bear sentiment for multiple tickers.

        Args:
            tickers: List of stock symbols (e.g. ["AAPL", "TSLA"]).

        Returns:
            List of StockTwitsSentiment objects (empty list on failure).
        """
        results = []
        for ticker in tickers:
            sentiment = self._get_ticker_sentiment(ticker.upper())
            if sentiment is not None:
                results.append(sentiment)
        return results

    def _get_ticker_sentiment(self, ticker: str) -> Optional[StockTwitsSentiment]:
        """Fetch sentiment for a single ticker with caching."""
        # Check cache
        if ticker in self._cache:
            cached, cached_at = self._cache[ticker]
            if time.time() - cached_at < CACHE_DURATION:
                return cached

        self._rate_limit()

        url = f"{API_BASE}/streams/symbol/{ticker}.json"
        data = self._request(url)
        if not data:
            return None

        try:
            messages = data.get("messages", [])
            if not messages:
                return None

            # Count bull/bear from labeled messages
            bull = 0
            bear = 0
            for msg in messages:
                entities = msg.get("entities", {})
                sentiment = entities.get("sentiment")
                if sentiment:
                    basic = sentiment.get("basic")
                    if basic == "Bullish":
                        bull += 1
                    elif basic == "Bearish":
                        bear += 1

            total = bull + bear
            if total == 0:
                return None

            bull_ratio = bull / total

            # Check if trending
            symbol_data = data.get("symbol", {})
            trending = bool(symbol_data.get("is_following", False)) or len(messages) >= 25

            result = StockTwitsSentiment(
                ticker=ticker,
                bull_count=bull,
                bear_count=bear,
                bull_ratio=round(bull_ratio, 4),
                total_messages=len(messages),
                trending=trending,
                detected_at=datetime.utcnow().isoformat(),
            )

            self._cache[ticker] = (result, time.time())
            return result

        except Exception as exc:
            logger.warning("StockTwits parse error for %s: %s", ticker, exc)
            return None


def get_sentiment(tickers: List[str]) -> List[StockTwitsSentiment]:
    """Convenience function for one-shot sentiment fetch."""
    return StockTwitsClient().get_sentiment(tickers)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print("Testing StockTwits Sentiment API...")

    client = StockTwitsClient()
    sentiments = client.get_sentiment(["AAPL", "TSLA", "SPY"])

    if sentiments:
        for s in sentiments:
            print(f"  {s.to_plain_language()}")
    else:
        print("  No data returned")

    print("\nDone.")
