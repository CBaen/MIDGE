"""
trends_client.py - Google Trends Client for Retail Attention Signals

Fetches search interest data via the pytrends library.
Spikes in retail search volume for a ticker often precede volatility.
Interest in "recession", "market crash", "fed rate" = macro fear signal.

Google rate-limits aggressively — use long cache TTL and delay.
No API key required, but pytrends uses informal scraping.

Keyword Discovery:
  After each fetch, rising related queries are harvested and stored in
  data/market/discovered_keywords.json.  On the next fetch cycle, up to
  MAX_DISCOVERED_PER_CYCLE discovered keywords are mixed in automatically.
  This creates a self-expanding cultural antenna: MIDGE starts with 10
  seeds and grows toward the edges of whatever the market is talking about.
"""

import json
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

REQUEST_DELAY = 10.0        # Google rate-limits hard
CACHE_DURATION = 21600      # 6 hours — trends update slowly

# Keyword discovery config
DISCOVERED_KEYWORDS_FILE = Path("data/market/discovered_keywords.json")
MAX_DISCOVERED_POOL = 30        # Max pool size before evicting oldest/lowest-score
MAX_DISCOVERED_PER_CYCLE = 3    # How many discovered keywords to inject per fetch

# Terms that are too generic to be useful as discovered keywords
_STOP_TERMS = {
    "the", "a", "an", "is", "are", "was", "be", "and", "or", "of", "to", "in",
    "for", "on", "with", "at", "by", "from", "stock", "stocks", "share", "shares",
    "price", "prices", "market", "markets", "trading", "trade", "today", "now",
    "how", "what", "when", "where", "why", "who", "which",
}

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

    Keyword discovery: after each batch fetch, harvests rising related
    queries and persists them to DISCOVERED_KEYWORDS_FILE. On subsequent
    calls get_interest() automatically mixes in discovered keywords.
    """

    def __init__(self, provider=None, raw_store=None):
        self._provider = provider  # Not used for pytrends (library, not HTTP)
        self._raw_store = raw_store
        self._last_request_time: float = 0.0
        self._cache: Optional[List[TrendsSignal]] = None
        self._cache_time: float = 0.0
        self._discovered: Dict[str, dict] = {}   # keyword -> {score, source, date}
        self._load_discovered()

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()

    # ------------------------------------------------------------------
    # Keyword discovery persistence
    # ------------------------------------------------------------------

    def _load_discovered(self) -> None:
        """Load previously discovered keywords from disk."""
        try:
            if DISCOVERED_KEYWORDS_FILE.exists():
                with open(DISCOVERED_KEYWORDS_FILE, "r", encoding="utf-8") as f:
                    self._discovered = json.load(f)
        except Exception as exc:
            logger.debug("Could not load discovered keywords: %s", exc)
            self._discovered = {}

    def _save_discovered(self) -> None:
        """Persist discovered keyword pool to disk atomically."""
        try:
            DISCOVERED_KEYWORDS_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp = DISCOVERED_KEYWORDS_FILE.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._discovered, f, indent=2)
            tmp.replace(DISCOVERED_KEYWORDS_FILE)
        except Exception as exc:
            logger.debug("Could not save discovered keywords: %s", exc)

    def _is_valid_keyword(self, kw: str) -> bool:
        """Return True if a keyword is worth tracking."""
        if not kw or len(kw) < 3 or len(kw) > 60:
            return False
        low = kw.lower()
        if low in _STOP_TERMS:
            return False
        # Skip pure numbers
        if kw.replace(".", "").replace(",", "").isnumeric():
            return False
        return True

    def _harvest_rising_queries(
        self, parent_keyword: str, related_df_entry: dict
    ) -> None:
        """Extract rising queries from pytrends related_queries result and add to pool."""
        if related_df_entry is None:
            return
        rising_df = related_df_entry.get("rising")
        if rising_df is None or rising_df.empty:
            return

        now_iso = datetime.now(timezone.utc).isoformat()
        for _, row in rising_df.head(10).iterrows():
            query = str(row.get("query", "")).strip()
            value = row.get("value", 0)

            if not self._is_valid_keyword(query):
                continue

            # "Breakout" value from pytrends is either a % or 5000 (= new term)
            score = float(value) if value != 5000 else 500.0

            existing = self._discovered.get(query)
            if existing is None or score > existing.get("score", 0):
                self._discovered[query] = {
                    "keyword": query,
                    "discovered_from": parent_keyword,
                    "discovery_date": now_iso,
                    "score": score,
                }

        # Evict if over pool limit — drop oldest then lowest-scoring
        if len(self._discovered) > MAX_DISCOVERED_POOL:
            # Sort by score descending; keep top MAX_DISCOVERED_POOL
            ranked = sorted(
                self._discovered.values(),
                key=lambda x: x.get("score", 0),
                reverse=True,
            )
            keep = {entry["keyword"]: entry for entry in ranked[:MAX_DISCOVERED_POOL]}
            self._discovered = keep

        self._save_discovered()

    def get_discovered_keywords(self) -> List[dict]:
        """Return current discovered keywords with their scores, sorted by score desc."""
        return sorted(
            self._discovered.values(),
            key=lambda x: x.get("score", 0),
            reverse=True,
        )

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

        seed_keywords = keywords or DEFAULT_KEYWORDS

        # Mix in top discovered keywords (up to MAX_DISCOVERED_PER_CYCLE new ones)
        discovered_top = [
            entry["keyword"]
            for entry in self.get_discovered_keywords()
            if entry["keyword"] not in seed_keywords
        ][:MAX_DISCOVERED_PER_CYCLE]

        all_keywords = list(seed_keywords) + discovered_top
        if discovered_top:
            logger.debug("Trends: injecting %d discovered keywords: %s", len(discovered_top), discovered_top)

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

            # Fetch related queries once per batch (covers all keywords in it)
            related_data: dict = {}
            try:
                related_data = pytrends.related_queries() or {}
            except Exception as exc:
                logger.debug("Trends related_queries failed: %s", exc)

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

                # Extract top related queries for the signal metadata
                related = []
                kw_related = related_data.get(kw) or {}
                try:
                    top_df = kw_related.get("top")
                    if top_df is not None and not top_df.empty:
                        related = top_df["query"].head(5).tolist()
                except Exception:
                    pass

                # Harvest rising queries for keyword discovery
                try:
                    self._harvest_rising_queries(kw, kw_related)
                except Exception as exc:
                    logger.debug("Rising query harvest failed for %s: %s", kw, exc)

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
