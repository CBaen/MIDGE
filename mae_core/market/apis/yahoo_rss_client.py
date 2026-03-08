"""
yahoo_rss_client.py — Yahoo Finance RSS Headline Client

Fetches per-ticker breaking headlines from Yahoo Finance's RSS feed.
No API key required. Free. Zero dependencies beyond feedparser.

Signal logic:
- Headline velocity (count in last N hours) detects news acceleration.
- Sentiment keywords in titles (surge/crash/beats/FDA/etc.) add directional color.
- A sudden spike in headlines is a signal regardless of direction —
  it means something is happening and traders should pay attention.

URL pattern: https://finance.yahoo.com/rss/headline?s=TICKER
Rate limit: 1 request per ticker per 5 minutes (in-memory dict cache).
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

RSS_BASE = "https://finance.yahoo.com/rss/headline"
CACHE_DURATION = 300   # 5 minutes per ticker
VELOCITY_WINDOW_HOURS = 6  # Count headlines in last N hours

# Keywords that carry directional signal in a headline title.
# Grouped by sentiment polarity. Each key is a lowercase search token.
BULLISH_KEYWORDS = frozenset({
    "surge", "surges", "surging", "rally", "rallies", "rallying",
    "beats", "beat", "beat expectations", "raises guidance",
    "upgrade", "upgraded", "buy", "outperform",
    "acquisition", "acquires", "merger", "deal",
    "fda approval", "fda approves", "approved", "clearance",
    "record", "all-time high", "breakout", "gains",
    "earnings beat", "profit", "revenue beat",
})

BEARISH_KEYWORDS = frozenset({
    "crash", "crashes", "plunge", "plunges", "plunging",
    "misses", "miss", "missed estimates", "cuts guidance",
    "downgrade", "downgraded", "sell", "underperform",
    "lawsuit", "sued", "investigation", "subpoena", "probe",
    "layoff", "layoffs", "job cuts", "cuts jobs",
    "recall", "fda rejects", "rejected", "warning",
    "earnings miss", "loss", "decline", "slump",
    "bankruptcy", "default",
})


@dataclass
class YahooHeadlineSignal:
    """Headline velocity + sentiment signal for a single ticker."""

    ticker: str
    headline_count: int          # Total headlines in last VELOCITY_WINDOW_HOURS
    velocity_change: float       # ratio vs prior window (2.0 = doubled)
    sentiment_keywords: List[str]
    latest_headline: str
    latest_published: str        # ISO 8601 string

    # Net keyword polarity: positive = bullish, negative = bearish, 0 = mixed/none
    keyword_polarity: int

    detected_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    signal_source: str = "yahoo_rss"
    decay_rate: float = 0.70     # News is very ephemeral
    confidence: float = 0.45     # Moderate prior — noise is high

    def to_plain_language(self) -> str:
        direction = "bullish" if self.keyword_polarity > 0 else (
            "bearish" if self.keyword_polarity < 0 else "neutral"
        )
        kw = ", ".join(self.sentiment_keywords[:3]) if self.sentiment_keywords else "none"
        return (
            f"{self.ticker}: {self.headline_count} headlines/{VELOCITY_WINDOW_HOURS}h "
            f"({self.velocity_change:.1f}x velocity), {direction} keywords [{kw}]. "
            f"Latest: {self.latest_headline[:80]}"
        )


def _extract_keywords(title: str) -> List[str]:
    """Find sentiment keywords present in a headline title."""
    lower = title.lower()
    found = []
    for kw in BULLISH_KEYWORDS:
        if kw in lower:
            found.append(kw)
    for kw in BEARISH_KEYWORDS:
        if kw in lower:
            found.append(kw)
    return found


def _keyword_polarity(keywords: List[str]) -> int:
    """Return +1 (bullish), -1 (bearish), or 0 (mixed/none)."""
    bull = sum(1 for k in keywords if k in BULLISH_KEYWORDS)
    bear = sum(1 for k in keywords if k in BEARISH_KEYWORDS)
    if bull > bear:
        return 1
    if bear > bull:
        return -1
    return 0


def _parse_rss_date(date_str: str) -> Optional[datetime]:
    """Parse an RSS date string to timezone-aware datetime. Returns None on failure."""
    if not date_str:
        return None
    try:
        return parsedate_to_datetime(date_str)
    except Exception:
        pass
    # Fallback: try ISO format
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


class YahooRSSClient:
    """
    Client for Yahoo Finance RSS headline feeds.

    No API key required. Fetches per-ticker breaking headlines via
    feedparser. Detects headline velocity (news acceleration) and
    extracts directional sentiment keywords from titles.

    Rate limited to 1 request per ticker per 5 minutes (in-memory cache).
    Suitable for watchlist scanning: call get_headlines() with a list
    of tickers, receive velocity-annotated signal objects.
    """

    def __init__(self, provider=None, raw_store=None):
        """
        Args:
            provider: Optional MarketDataProvider for gateway routing.
                      If None, feedparser makes direct requests.
            raw_store: Optional RawStore for SQLite persistence.
        """
        self._provider = provider
        self._raw_store = raw_store
        # Cache: {ticker: (entries_list, fetch_timestamp)}
        self._cache: Dict[str, tuple] = {}

    def _fetch_feed(self, ticker: str) -> List[dict]:
        """
        Fetch raw RSS entries for a single ticker.

        Uses in-memory cache to avoid hammering Yahoo within 5 minutes.
        Returns list of feedparser entry dicts, or empty list on failure.
        """
        ticker = ticker.upper().strip()
        now = time.time()

        # Serve from cache if fresh
        if ticker in self._cache:
            entries, fetched_at = self._cache[ticker]
            if now - fetched_at < CACHE_DURATION:
                return entries

        url = f"{RSS_BASE}?s={ticker}"

        try:
            import feedparser
            if self._provider is not None:
                # Route through BoundaryMembrane (get raw text, then parse)
                from mae_core.market.apis.market_data_provider import market_request
                from mae_core.external.api_client import ApiResponseStatus
                resp = market_request(
                    self._provider, url,
                    source_name="yahoo_rss", timeout_ms=15000.0,
                )
                if resp.status != ApiResponseStatus.SUCCESS:
                    logger.debug("YahooRSS: provider fetch failed for %s", ticker)
                    return []
                # Provider returns parsed JSON — but RSS is text. Fall back to direct.
                feed = feedparser.parse(url)
            else:
                feed = feedparser.parse(url)

            entries = list(getattr(feed, "entries", []))
            self._cache[ticker] = (entries, now)
            logger.debug("YahooRSS: fetched %d entries for %s", len(entries), ticker)
            return entries

        except Exception as exc:
            logger.warning("YahooRSS: feed fetch error for %s: %s", ticker, exc)
            return []

    def _analyze_entries(self, ticker: str, entries: List) -> Optional[YahooHeadlineSignal]:
        """
        Convert raw feed entries into a YahooHeadlineSignal.

        Counts headlines in the last VELOCITY_WINDOW_HOURS and compares
        to the prior window to produce a velocity_change ratio.
        """
        if not entries:
            return None

        now_utc = datetime.now(timezone.utc)
        window_start = now_utc - timedelta(hours=VELOCITY_WINDOW_HOURS)
        prior_start = window_start - timedelta(hours=VELOCITY_WINDOW_HOURS)

        recent_entries = []
        prior_entries = []
        all_keywords: List[str] = []
        latest_headline = ""
        latest_published = ""
        latest_dt: Optional[datetime] = None

        for entry in entries:
            title = getattr(entry, "title", "") or ""
            pub_str = getattr(entry, "published", "") or ""
            pub_dt = _parse_rss_date(pub_str)

            # Track latest headline
            if title and (latest_dt is None or (pub_dt and pub_dt > latest_dt)):
                latest_headline = title
                latest_published = pub_str
                latest_dt = pub_dt

            # Bucket into windows
            if pub_dt is not None:
                if pub_dt >= window_start:
                    recent_entries.append(entry)
                    all_keywords.extend(_extract_keywords(title))
                elif pub_dt >= prior_start:
                    prior_entries.append(entry)

        # Headline count = recent window
        headline_count = len(recent_entries)
        if headline_count == 0:
            # No recent headlines — not a signal
            return None

        # Velocity change: recent vs prior (floor prior at 1 to avoid division by zero)
        velocity_change = headline_count / max(1, len(prior_entries))

        # Deduplicate keywords, preserve order
        seen = set()
        unique_keywords = []
        for kw in all_keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)

        polarity = _keyword_polarity(unique_keywords)

        return YahooHeadlineSignal(
            ticker=ticker,
            headline_count=headline_count,
            velocity_change=round(velocity_change, 3),
            sentiment_keywords=unique_keywords,
            latest_headline=latest_headline,
            latest_published=latest_published,
            keyword_polarity=polarity,
            detected_at=now_utc.isoformat(),
        )

    def get_headlines(self, tickers: List[str]) -> List[YahooHeadlineSignal]:
        """
        Fetch and analyze headlines for a list of tickers.

        Only returns signals where at least 1 recent headline was found.
        Each ticker is rate-limited to 1 fetch per 5 minutes (cached).

        Args:
            tickers: List of ticker symbols (e.g. ["AAPL", "TSLA"]).

        Returns:
            List of YahooHeadlineSignal objects, one per ticker with data.
        """
        results = []
        for ticker in tickers:
            try:
                entries = self._fetch_feed(ticker)

                # Store raw headlines before processing
                if self._raw_store is not None and entries:
                    try:
                        self._raw_store.store_yahoo_headlines(ticker, entries)
                    except Exception:
                        pass  # Never break signal pipeline

                signal = self._analyze_entries(ticker, entries)
                if signal is not None:
                    results.append(signal)
            except Exception as exc:
                logger.debug("YahooRSS: analysis failed for %s: %s", ticker, exc)

        return results

    def get_accelerating(
        self,
        tickers: List[str],
        min_velocity: float = 2.0,
    ) -> List[YahooHeadlineSignal]:
        """
        Return only tickers with headline velocity >= min_velocity.

        Sorted by velocity_change descending (fastest news flow first).
        Useful for detecting sudden news bursts on watchlist names.

        Args:
            tickers: Ticker symbols to scan.
            min_velocity: Minimum velocity_change ratio (2.0 = doubled headline rate).

        Returns:
            Filtered, sorted list of YahooHeadlineSignal objects.
        """
        all_signals = self.get_headlines(tickers)
        accelerating = [s for s in all_signals if s.velocity_change >= min_velocity]
        accelerating.sort(key=lambda s: s.velocity_change, reverse=True)
        return accelerating


def get_headlines(tickers: List[str]) -> List[YahooHeadlineSignal]:
    """Convenience function — one-shot headline fetch."""
    return YahooRSSClient().get_headlines(tickers)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    symbols = sys.argv[1:] if len(sys.argv) > 1 else ["AAPL", "TSLA", "NVDA"]
    print(f"Fetching Yahoo Finance RSS for: {symbols}")
    client = YahooRSSClient()
    signals = client.get_headlines(symbols)
    if signals:
        for s in signals:
            print(f"  {s.to_plain_language()}")
    else:
        print("  No recent headlines found")
    print("Done.")
