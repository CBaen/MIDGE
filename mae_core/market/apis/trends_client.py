"""
trends_client.py - Google Trends Client for Retail Attention Signals

Fetches search interest data via the pytrends library.
Spikes in retail search volume for a ticker often precede volatility.
Interest in "recession", "market crash", "fed rate" = macro fear signal.

Google rate-limits aggressively — use long cache TTL and delay.
No API key required, but pytrends uses informal scraping.
"""

import time
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

REQUEST_DELAY = 10.0        # Google rate-limits hard
CACHE_DURATION = 21600      # 6 hours — trends update slowly

# Default keywords to monitor (financial tickers + macro fear terms)
DEFAULT_KEYWORDS = [
    "SPY", "QQQ", "AAPL", "TSLA", "NVDA",
    "recession", "market crash", "fed rate",
    "inflation", "unemployment",
]


@dataclass
class TrendsSignal:
    """Google Trends interest data for a single keyword."""

    keyword: str
    interest_score: int         # 0-100 (Google's normalized score)
    interest_delta_7d: float    # Change vs 7 days ago (positive = rising)
    is_breakout: bool           # True if score > 75 (unusual attention)
    related_queries: List[str] = field(default_factory=list)

    detected_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    signal_source: str = "google_trends"
    decay_rate: float = 0.50    # Fast decay — attention fades quickly
    confidence: float = 0.45    # Low prior — Google Trends is noisy

    def to_plain_language(self) -> str:
        trend = "rising" if self.interest_delta_7d > 10 else "falling" if self.interest_delta_7d < -10 else "stable"
        breakout_tag = " [BREAKOUT]" if self.is_breakout else ""
        return (
            f"{self.keyword}: interest {self.interest_score}/100 "
            f"({trend}, delta {self.interest_delta_7d:+.0f}){breakout_tag}"
        )


class TrendsClient:
    """
    Client for Google Trends data via pytrends.

    Monitors search interest for financial keywords to detect
    retail attention spikes. No API key required.

    Rate limited to 1 req/10s. 6-hour cache.
    """

    def __init__(self, provider=None, raw_store=None):
        self._provider = provider  # Not used for pytrends (library, not HTTP)
        self._raw_store = raw_store
        self._last_request_time: float = 0.0
        self._cache: Optional[List[TrendsSignal]] = None
        self._cache_time: float = 0.0

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()

    def get_interest(
        self, keywords: Optional[List[str]] = None
    ) -> List[TrendsSignal]:
        """
        Get search interest for financial keywords.

        Args:
            keywords: List of search terms. If None, uses DEFAULT_KEYWORDS.
                      Google Trends allows max 5 keywords per request.

        Returns:
            List of TrendsSignal objects.
        """
        if self._cache is not None and time.time() - self._cache_time < CACHE_DURATION:
            if keywords:
                return [s for s in self._cache if s.keyword in keywords]
            return self._cache

        try:
            from pytrends.request import TrendReq
        except ImportError:
            logger.warning("pytrends not installed. pip install pytrends")
            return []

        all_keywords = keywords or DEFAULT_KEYWORDS
        results: List[TrendsSignal] = []

        # Process in batches of 5 (Google Trends API limit)
        for i in range(0, len(all_keywords), 5):
            batch = all_keywords[i:i + 5]
            batch_results = self._fetch_batch(batch)
            results.extend(batch_results)

        self._cache = results
        self._cache_time = time.time()
        return results

    def _fetch_batch(self, keywords: List[str]) -> List[TrendsSignal]:
        """Fetch interest data for a batch of up to 5 keywords."""
        self._rate_limit()

        try:
            from pytrends.request import TrendReq

            pytrends = TrendReq(hl="en-US", tz=300)  # EST timezone
            pytrends.build_payload(keywords, cat=0, timeframe="now 7-d")

            # Get interest over time
            df = pytrends.interest_over_time()

            if df is None or df.empty:
                return []

            if self._raw_store:
                try:
                    for kw in keywords:
                        if kw in df.columns:
                            trend_rows = [
                                {"timestamp": str(idx), "interest": int(df.loc[idx, kw])}
                                for idx in df.index
                            ]
                            self._raw_store.store_trends(kw, trend_rows)
                except Exception as exc:
                    logger.debug("RawStore Trends write failed: %s", exc)

            signals = []
            for kw in keywords:
                if kw not in df.columns:
                    continue

                series = df[kw]
                current = int(series.iloc[-1]) if len(series) > 0 else 0

                # Calculate 7-day delta
                if len(series) >= 2:
                    first_val = int(series.iloc[0])
                    delta = current - first_val
                else:
                    delta = 0.0

                is_breakout = current > 75

                # Try to get related queries
                related = []
                try:
                    related_df = pytrends.related_queries()
                    if kw in related_df and related_df[kw].get("top") is not None:
                        top_queries = related_df[kw]["top"]
                        if top_queries is not None and not top_queries.empty:
                            related = top_queries["query"].head(5).tolist()
                except Exception:
                    pass

                signals.append(TrendsSignal(
                    keyword=kw,
                    interest_score=current,
                    interest_delta_7d=float(delta),
                    is_breakout=is_breakout,
                    related_queries=related,
                    detected_at=datetime.utcnow().isoformat(),
                ))

            return signals

        except Exception as exc:
            logger.warning("Google Trends fetch failed for %s: %s", keywords, exc)
            return []


def get_interest(keywords: Optional[List[str]] = None) -> List[TrendsSignal]:
    """Convenience function for one-shot trends fetch."""
    return TrendsClient().get_interest(keywords)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print("Testing Google Trends...")

    client = TrendsClient()
    signals = client.get_interest(["SPY", "recession", "NVDA"])

    if signals:
        for s in signals:
            print(f"  {s.to_plain_language()}")
    else:
        print("  No data returned")

    print("\nDone.")
