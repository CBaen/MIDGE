"""
crypto_news_client.py — CoinDesk + Cointelegraph RSS headline velocity.
No API key. httpx + stdlib XML. 5+ headlines/24h → convergence signal.
"""

import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Dict, List, Optional

import httpx

logger = logging.getLogger("midge.market.apis.crypto_news_client")

CACHE_DURATION = 300  # 5 minutes (both feeds share one cache slot per source)
VELOCITY_WINDOW_HOURS = 24
VELOCITY_THRESHOLD = 5   # minimum headlines to emit a signal
STRENGTH_MIN_COUNT = 5
STRENGTH_MAX_COUNT = 20

FEEDS = {
    "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "cointelegraph": "https://cointelegraph.com/rss",
}

# Ticker → aliases (lowercase) used for headline matching
TICKER_ALIASES: Dict[str, List[str]] = {
    "BTC": ["btc", "bitcoin"],
    "ETH": ["eth", "ethereum"],
    "SOL": ["sol", "solana"],
    "XRP": ["xrp"],
    "ADA": ["ada", "cardano"],
    "DOGE": ["doge", "dogecoin"],
}

BEARISH_KEYWORDS = frozenset({
    "crash", "plunge", "hack", "hacked", "exploit", "regulation",
    "ban", "banned", "sec", "lawsuit", "fraud", "scam", "collapse",
    "sell-off", "selloff", "losses", "dump", "rug pull",
})

BULLISH_KEYWORDS = frozenset({
    "surge", "rally", "adoption", "etf", "approval", "approved",
    "institutional", "record", "all-time high", "breakout", "gains",
    "partnership", "launch", "milestone", "bullish", "accumulation",
})


@dataclass
class CryptoHeadline:
    """A single headline from a crypto news RSS feed."""
    title: str
    source: str               # "coindesk" or "cointelegraph"
    published_at: datetime
    categories: List[str]
    url: str


@dataclass
class CryptoNewsSignal:
    """Convergence signal derived from news velocity for a crypto ticker."""
    symbol: str
    domain: str = "news"
    direction: str = "neutral"
    strength: float = 0.3
    confidence: float = 0.45
    headline_count: int = 0
    sample_titles: List[str] = field(default_factory=list)
    sentiment_keywords_found: List[str] = field(default_factory=list)
    detected_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    signal_source: str = "crypto_news_rss"
    decay_rate: float = 0.70


def _parse_date(date_str: str) -> Optional[datetime]:
    """Parse RSS pubDate string to timezone-aware datetime."""
    if not date_str:
        return None
    try:
        return parsedate_to_datetime(date_str.strip())
    except Exception:
        pass
    try:
        dt = datetime.fromisoformat(date_str.strip().replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _scale_strength(count: int) -> float:
    """Scale headline count [5, 20] → strength [0.3, 1.0]."""
    clamped = max(STRENGTH_MIN_COUNT, min(STRENGTH_MAX_COUNT, count))
    ratio = (clamped - STRENGTH_MIN_COUNT) / (STRENGTH_MAX_COUNT - STRENGTH_MIN_COUNT)
    return round(0.3 + ratio * 0.7, 3)


def _direction_from_keywords(titles: List[str]) -> tuple:
    """Return (direction, keywords_found) from a list of headline titles."""
    lower_all = " ".join(t.lower() for t in titles)
    bull_hits = [kw for kw in BULLISH_KEYWORDS if kw in lower_all]
    bear_hits = [kw for kw in BEARISH_KEYWORDS if kw in lower_all]
    if len(bull_hits) > len(bear_hits):
        direction = "bullish"
    elif len(bear_hits) > len(bull_hits):
        direction = "bearish"
    else:
        direction = "neutral"
    return direction, sorted(bull_hits + bear_hits)


class CryptoNewsClient:
    """
    Fetches and analyzes crypto headlines from CoinDesk and Cointelegraph RSS.

    No API key required. Uses httpx for HTTP, stdlib XML for parsing.
    Both feeds are fetched and merged into a unified headline pool.
    Cache: 5 minutes per feed source. Two public feeds → ~0.5s cold fetch.
    """

    def __init__(self, raw_store=None):
        self._raw_store = raw_store
        # {source_name: (List[CryptoHeadline], fetch_timestamp)}
        self._cache: Dict[str, tuple] = {}
        self._stats = {"fetches": 0, "errors": 0, "headlines_parsed": 0, "signals_emitted": 0}

    def _fetch_feed(self, source: str, url: str) -> List[CryptoHeadline]:
        """Fetch one RSS feed, parse items, return CryptoHeadline list."""
        now = time.time()
        if source in self._cache:
            items, fetched_at = self._cache[source]
            if now - fetched_at < CACHE_DURATION:
                return items

        headlines: List[CryptoHeadline] = []
        try:
            resp = httpx.get(url, timeout=15, follow_redirects=True,
                             headers={"User-Agent": "MIDGE Trading Research"})
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            channel = root.find("channel")
            items_el = channel.findall("item") if channel is not None else root.findall(".//item")
            for item in items_el:
                title = (item.findtext("title") or "").strip()
                pub_str = (item.findtext("pubDate") or "").strip()
                link = (item.findtext("link") or "").strip()
                cats = [c.text.strip() for c in item.findall("category") if c.text]
                pub_dt = _parse_date(pub_str)
                if pub_dt is None:
                    pub_dt = datetime.now(timezone.utc)
                headlines.append(CryptoHeadline(
                    title=title, source=source,
                    published_at=pub_dt, categories=cats, url=link,
                ))
            self._stats["fetches"] += 1
            self._stats["headlines_parsed"] += len(headlines)
            logger.debug("crypto_news: %s → %d items", source, len(headlines))
        except Exception as exc:
            self._stats["errors"] += 1
            logger.warning("crypto_news: fetch error [%s]: %s", source, exc)

        self._cache[source] = (headlines, now)
        return headlines

    def get_recent_headlines(self, hours: int = 24) -> List[CryptoHeadline]:
        """Fetch both feeds, return headlines published in last `hours` hours."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        all_headlines: List[CryptoHeadline] = []
        for source, url in FEEDS.items():
            for h in self._fetch_feed(source, url):
                if h.published_at >= cutoff:
                    all_headlines.append(h)
        return all_headlines

    def get_news_signals(self) -> List[dict]:
        """
        Analyze headlines and return convergence signal dicts.

        One signal per ticker that reaches the velocity threshold (5+ headlines
        in 24h). Direction derived from bullish/bearish keyword majority.
        """
        headlines = self.get_recent_headlines(hours=VELOCITY_WINDOW_HOURS)
        signals: List[dict] = []

        for ticker, aliases in TICKER_ALIASES.items():
            matched = [
                h for h in headlines
                if any(alias in h.title.lower() for alias in aliases)
            ]
            if len(matched) < VELOCITY_THRESHOLD:
                continue

            titles = [h.title for h in matched]
            direction, kw_found = _direction_from_keywords(titles)
            strength = _scale_strength(len(matched))

            sig = CryptoNewsSignal(
                symbol=ticker,
                direction=direction,
                strength=strength,
                headline_count=len(matched),
                sample_titles=titles[:3],
                sentiment_keywords_found=kw_found,
            )
            self._stats["signals_emitted"] += 1
            signals.append({
                "symbol": sig.symbol,
                "domain": sig.domain,
                "direction": sig.direction,
                "strength": sig.strength,
                "confidence": sig.confidence,
                "signal_source": sig.signal_source,
                "decay_rate": sig.decay_rate,
                "detected_at": sig.detected_at,
                "metadata": {
                    "headline_count": sig.headline_count,
                    "sample_titles": sig.sample_titles,
                    "sentiment_keywords_found": sig.sentiment_keywords_found,
                },
            })

        return signals

    def get_statistics(self) -> dict:
        """Return cumulative fetch and signal statistics."""
        cached_counts = {src: len(items) for src, (items, _) in self._cache.items()}
        return {**self._stats, "cached_headline_counts": cached_counts}


def get_news_signals() -> List[dict]:
    """Convenience function — one-shot signal fetch."""
    return CryptoNewsClient().get_news_signals()


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    client = CryptoNewsClient()
    headlines = client.get_recent_headlines(hours=24)
    print(f"Fetched {len(headlines)} headlines in last 24h")
    signals = client.get_news_signals()
    if signals:
        for s in signals:
            m = s["metadata"]
            print(f"  {s['symbol']}: {m['headline_count']} headlines → "
                  f"{s['direction']} {s['strength']:.2f} | {m['sample_titles'][0][:60]}")
    else:
        print("  No tickers above velocity threshold")
    print(f"Stats: {client.get_statistics()}")
