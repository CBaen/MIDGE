"""reddit_crypto_client.py — Reddit crypto sentiment from public RSS feeds.

Free, no auth, no API key. Uses Reddit's public RSS endpoints (.rss suffix)
to track post velocity and sentiment on r/cryptocurrency and r/bitcoin.

Signal: When a crypto ticker is mentioned in 3+ posts with consistent
sentiment (bullish keywords vs bearish keywords), emit a signal.
"""

from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("midge.market.apis.reddit_crypto")

_FEEDS = {
    "r_cryptocurrency": "https://www.reddit.com/r/CryptoCurrency/new/.rss?limit=50",
    "r_bitcoin": "https://www.reddit.com/r/Bitcoin/new/.rss?limit=50",
    "r_ethereum": "https://www.reddit.com/r/ethereum/new/.rss?limit=50",
    "r_solana": "https://www.reddit.com/r/solana/new/.rss?limit=25",
}

_TICKER_KEYWORDS = {
    "BTC": ["bitcoin", "btc", "satoshi"],
    "ETH": ["ethereum", "eth", "vitalik"],
    "SOL": ["solana", "sol"],
    "XRP": ["xrp", "ripple"],
    "ADA": ["cardano", "ada"],
    "DOGE": ["dogecoin", "doge"],
    "AVAX": ["avalanche", "avax"],
}

_BULLISH = {"moon", "pump", "bull", "breakout", "ath", "rally", "surge", "buy",
            "accumulate", "bullish", "green", "rocket", "adoption", "institutional",
            "etf", "approval", "upgrade", "milestone"}
_BEARISH = {"dump", "crash", "bear", "rug", "scam", "hack", "sec", "lawsuit",
            "ban", "regulation", "selloff", "bearish", "red", "plunge", "fraud",
            "exploit", "bankruptcy", "collapse"}

_CACHE_TTL = 300  # 5 minutes
_ATOM_NS = "{http://www.w3.org/2005/Atom}"


@dataclass
class RedditPost:
    title: str
    subreddit: str
    published_at: datetime
    url: str = ""


class RedditCryptoClient:
    """Reddit crypto sentiment client using public RSS feeds."""

    def __init__(self) -> None:
        self._client = httpx.Client(
            timeout=15,
            follow_redirects=True,
            headers={"User-Agent": "MIDGE/1.0 (market intelligence)"},
        )
        self._cache: Dict[str, tuple] = {}  # feed_key → (posts, timestamp)
        self._call_count = 0
        self._error_count = 0
        logger.info("RedditCryptoClient initialized (free RSS, no auth)")

    def get_recent_posts(self, hours: int = 6) -> List[RedditPost]:
        """Fetch recent posts from all crypto subreddits."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        posts = []
        for feed_key, url in _FEEDS.items():
            cached = self._cache.get(feed_key)
            if cached and (time.time() - cached[1]) < _CACHE_TTL:
                posts.extend(cached[0])
                continue
            try:
                self._call_count += 1
                resp = self._client.get(url)
                if resp.status_code != 200:
                    continue
                feed_posts = []
                root = ET.fromstring(resp.text)
                for entry in root.findall(f"{_ATOM_NS}entry"):
                    title_el = entry.find(f"{_ATOM_NS}title")
                    updated_el = entry.find(f"{_ATOM_NS}updated")
                    link_el = entry.find(f"{_ATOM_NS}link")
                    if title_el is None or updated_el is None:
                        continue
                    try:
                        pub = datetime.fromisoformat(updated_el.text.replace("Z", "+00:00"))
                    except (ValueError, AttributeError):
                        continue
                    if pub < cutoff:
                        continue
                    feed_posts.append(RedditPost(
                        title=title_el.text or "",
                        subreddit=feed_key,
                        published_at=pub,
                        url=link_el.attrib.get("href", "") if link_el is not None else "",
                    ))
                self._cache[feed_key] = (feed_posts, time.time())
                posts.extend(feed_posts)
            except Exception as e:
                self._error_count += 1
                logger.debug("Reddit RSS fetch failed for %s: %s", feed_key, e)
        return posts

    def get_sentiment_signals(self) -> List[Dict[str, Any]]:
        """Analyze Reddit posts and return convergence-ready signals."""
        posts = self.get_recent_posts(hours=6)
        if not posts:
            return []

        signals = []
        now = datetime.now().isoformat()

        for ticker, keywords in _TICKER_KEYWORDS.items():
            matching = [p for p in posts
                        if any(kw in p.title.lower() for kw in keywords)]
            if len(matching) < 3:
                continue

            # Count bullish vs bearish keywords across matching posts
            bull_count = 0
            bear_count = 0
            for p in matching:
                title_lower = p.title.lower()
                bull_count += sum(1 for w in _BULLISH if w in title_lower)
                bear_count += sum(1 for w in _BEARISH if w in title_lower)

            if bull_count == bear_count == 0:
                continue

            direction = "bullish" if bull_count > bear_count else "bearish"
            post_count = len(matching)
            strength = min(1.0, (post_count - 3) / 15)  # 3 posts = 0.0, 18+ = 1.0
            strength = max(0.3, strength)  # floor at 0.3

            sample_titles = [p.title[:80] for p in matching[:3]]

            signals.append({
                "source": "reddit_crypto",
                "symbol": ticker,
                "asset_class": "crypto",
                "domain": "sentiment",
                "direction": direction,
                "strength": round(strength, 3),
                "confidence": 0.40,  # Reddit is noisy
                "timestamp": now,
                "metadata": {
                    "symbol": ticker,
                    "post_count": post_count,
                    "bull_keywords": bull_count,
                    "bear_keywords": bear_count,
                    "sample_titles": sample_titles,
                    "subreddits": list(set(p.subreddit for p in matching)),
                    "enrichment_group": f"reddit_crypto_{datetime.now().strftime('%Y%m%d%H')}",
                },
            })

        return signals

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "calls_made": self._call_count,
            "errors": self._error_count,
            "cached_feeds": len(self._cache),
        }
