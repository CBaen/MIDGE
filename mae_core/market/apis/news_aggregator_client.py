"""
news_aggregator_client.py — Free Financial News Headline Aggregator

Aggregates financial news headlines from multiple free RSS feeds and
the SEC EDGAR public search endpoint. No API keys required. Complements
per-ticker Yahoo RSS with broad market news awareness — major macro
events, Fed releases, and cross-market signals that no single ticker
feed would surface.

Sources (configured in news_aggregator_constants.py):
  - Reuters Business RSS
  - CNBC Top News RSS
  - MarketWatch Top Stories RSS
  - Bloomberg Markets RSS (may return 403 — handled gracefully)
  - Federal Reserve Press Releases RSS
  - SEC EDGAR Full-Text Search (8-K material events — same-day)

Caching: 30 minutes per source.
Rate limiting: 5 seconds between source fetches.

Design contract:
  - Never crash on a single source failure. All sources are independent.
  - Return whatever succeeds. Partial results are always useful.
  - Every NewsHeadline is self-contained: source, sentiment, tickers, keywords.
  - raw_store is optional — always works without it.
"""

import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Dict, List, Optional, Set, Tuple

import httpx

from .news_aggregator_constants import (
    EDGAR_8K_URL,
    FINANCIAL_KEYWORDS,
    NEGATIVE_KEYWORDS,
    POSITIVE_KEYWORDS,
    RSS_SOURCES,
    SP500_TICKERS,
)

logger = logging.getLogger(__name__)

CACHE_DURATION = 1800       # 30 minutes per source
INTER_SOURCE_DELAY = 5.0    # Seconds between source requests
HTTP_TIMEOUT = 15.0         # Per-request timeout

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class NewsHeadline:
    """
    A single headline from any aggregated news source.

    Carries extracted context — tickers, sentiment, financial keywords —
    so the convergence engine can use it without re-parsing.
    """

    title: str
    source: str                  # "reuters", "cnbc", "marketwatch", etc.
    url: str
    published: str               # ISO 8601 timestamp string
    tickers: List[str]           # Ticker symbols found in title
    sentiment: str               # "positive", "negative", "neutral"
    keywords: List[str]          # Financial keywords detected in title
    domain: str = "events"
    signal_source: str = "news_aggregator"

    def to_plain_language(self) -> str:
        tickers_str = ", ".join(self.tickers) if self.tickers else "no tickers"
        kw_str = ", ".join(self.keywords[:3]) if self.keywords else "none"
        return (
            f"[{self.source.upper()}] {self.sentiment.upper()} — {self.title[:100]} "
            f"| tickers: {tickers_str} | keywords: {kw_str}"
        )


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_date(date_str: str) -> str:
    """
    Normalise any date string to ISO 8601 UTC.

    Tries RFC 2822 (standard RSS pubDate), then ISO 8601 variants.
    Returns the input string unchanged on complete failure so the
    headline is never silently dropped for a bad date.
    """
    if not date_str:
        return datetime.now(timezone.utc).isoformat()
    try:
        return parsedate_to_datetime(date_str).astimezone(timezone.utc).isoformat()
    except Exception:
        pass
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat()
        except Exception:
            pass
    return date_str


def _detect_sentiment(title: str) -> str:
    """
    Classify headline sentiment as positive/negative/neutral using
    keyword matching only. No LLM or external service required.
    """
    lower = title.lower()
    neg = sum(1 for kw in NEGATIVE_KEYWORDS if kw in lower)
    pos = sum(1 for kw in POSITIVE_KEYWORDS if kw in lower)
    if neg > pos:
        return "negative"
    if pos > neg:
        return "positive"
    return "neutral"


def _extract_financial_keywords(title: str) -> List[str]:
    """Return financial domain keywords found in a headline title."""
    lower = title.lower()
    return [kw for kw in FINANCIAL_KEYWORDS if kw in lower]


def _extract_tickers(title: str) -> List[str]:
    """
    Extract ticker symbols from a headline title via four strategies:

    1. Explicit $TICKER notation  — $AAPL, $TSLA
    2. Parenthetical notation     — (AAPL), (NYSE: AAPL)
    3. Colon notation             — AAPL:
    4. Standalone word match      — known S&P 500 tickers as whole words
       (capped at 3 to suppress false positives from common uppercase words)
    """
    found: List[str] = []
    seen: Set[str] = set()

    for m in re.finditer(r'\$([A-Z]{1,5})\b', title):
        t = m.group(1)
        if t not in seen:
            seen.add(t)
            found.append(t)

    for m in re.finditer(r'\((?:[A-Z]+:\s*)?([A-Z]{1,5})\)', title):
        t = m.group(1)
        if t not in seen:
            seen.add(t)
            found.append(t)

    for m in re.finditer(r'\b([A-Z]{1,5}):', title):
        t = m.group(1)
        if t not in seen:
            seen.add(t)
            found.append(t)

    words = set(re.findall(r'\b([A-Z]{1,5})\b', title))
    standalone = [t for t in words if t in SP500_TICKERS and t not in seen]
    found.extend(standalone[:3])

    return found


def _parse_rss_xml(xml_text: str, source_name: str) -> List[Tuple[str, str, str]]:
    """
    Parse raw RSS 2.0 or Atom 1.0 XML into (title, link, pubDate) tuples.

    Returns an empty list rather than raising on malformed XML.
    """
    results: List[Tuple[str, str, str]] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.debug("NewsAggregator: XML parse error for %s: %s", source_name, exc)
        return results

    ns_atom = "http://www.w3.org/2005/Atom"

    # RSS 2.0: <rss><channel><item>
    items = root.findall(".//item")
    if items:
        for item in items:
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub = (item.findtext("pubDate") or "").strip()
            if title:
                results.append((title, link, pub))
        return results

    # Atom 1.0: <feed><entry>
    # ElementTree element truth value is based on child count, not identity.
    # Must use explicit `is not None` checks to avoid the Python deprecation
    # trap where `elem_a or elem_b` silently picks elem_b when elem_a exists.
    entries = root.findall(f"{{{ns_atom}}}entry")
    if not entries:
        entries = root.findall("entry")

    for entry in entries:
        title_el = entry.find(f"{{{ns_atom}}}title")
        if title_el is None:
            title_el = entry.find("title")
        title = (title_el.text or "").strip() if title_el is not None else ""

        link_el = entry.find(f"{{{ns_atom}}}link")
        if link_el is None:
            link_el = entry.find("link")
        if link_el is not None:
            link = link_el.get("href", link_el.text or "").strip()
        else:
            link = ""

        updated_el = entry.find(f"{{{ns_atom}}}updated")
        if updated_el is None:
            updated_el = entry.find("updated")
        pub = (updated_el.text or "").strip() if updated_el is not None else ""

        if title:
            results.append((title, link, pub))

    return results


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------

class NewsAggregatorClient:
    """
    Aggregates free financial news headlines from multiple RSS feeds
    and the SEC EDGAR public search endpoint.

    No API keys required. Uses httpx for HTTP, xml.etree.ElementTree
    (stdlib) for RSS parsing — no feedparser dependency.

    Cache: 30 minutes per source. Sources fetched with a 5-second
    inter-source delay to avoid rate bans.

    Failure policy: every source has its own try/except. A single
    blocked, timed-out, or malformed source never affects the others.

    Usage:
        client = NewsAggregatorClient(raw_store=store)
        headlines = client.get_headlines(max_per_source=10)
        aapl_news = client.get_ticker_mentions("AAPL")
    """

    _HTTP_HEADERS = {
        "User-Agent": "MIDGE/1.0 Financial Research Bot (midge@example.com)",
        "Accept": "application/rss+xml, application/xml, text/xml, */*",
    }

    def __init__(self, raw_store=None):
        """
        Args:
            raw_store: Optional RawStore instance for SQLite persistence.
                       Always works without it — just no persistence.
        """
        self._raw_store = raw_store
        # {source_name: (headlines_list, fetch_epoch)}
        self._cache: Dict[str, Tuple[List[NewsHeadline], float]] = {}
        self._last_fetch_time: float = 0.0
        self._http: Optional[httpx.Client] = None

    # ------------------------------------------------------------------
    # HTTP layer
    # ------------------------------------------------------------------

    def _get_http(self) -> httpx.Client:
        if self._http is None or self._http.is_closed:
            self._http = httpx.Client(
                headers=self._HTTP_HEADERS,
                timeout=HTTP_TIMEOUT,
                follow_redirects=True,
            )
        return self._http

    def _fetch_url(self, url: str) -> Optional[str]:
        """
        Fetch URL text with inter-source rate limiting.

        Returns response body as str, or None on any failure
        (timeout, 4xx, 5xx, DNS error). Never raises.
        """
        elapsed = time.time() - self._last_fetch_time
        if elapsed < INTER_SOURCE_DELAY:
            time.sleep(INTER_SOURCE_DELAY - elapsed)
        self._last_fetch_time = time.time()

        try:
            resp = self._get_http().get(url)
            if resp.status_code == 200:
                return resp.text
            if resp.status_code == 403:
                logger.debug("NewsAggregator: 403 Forbidden for %s (paywall?)", url)
            else:
                logger.debug("NewsAggregator: HTTP %d for %s", resp.status_code, url)
            return None
        except httpx.TimeoutException:
            logger.debug("NewsAggregator: timeout fetching %s", url)
        except httpx.ConnectError:
            logger.debug("NewsAggregator: DNS/connect error for %s", url)
        except Exception as exc:
            logger.debug("NewsAggregator: fetch error for %s: %s", url, exc)
        return None

    # ------------------------------------------------------------------
    # RSS source fetcher
    # ------------------------------------------------------------------

    def _fetch_rss_source(self, source: Dict, max_items: int) -> List[NewsHeadline]:
        """Fetch one RSS source, serve from cache when fresh."""
        name = source["name"]
        now = time.time()

        if name in self._cache:
            cached, cached_at = self._cache[name]
            if now - cached_at < CACHE_DURATION:
                return cached[:max_items]

        xml_text = self._fetch_url(source["url"])
        if not xml_text:
            return []

        raw_items = _parse_rss_xml(xml_text, name)
        if not raw_items:
            return []

        headlines = [
            NewsHeadline(
                title=title,
                source=name,
                url=link,
                published=_parse_date(pub_raw),
                tickers=_extract_tickers(title),
                sentiment=_detect_sentiment(title),
                keywords=_extract_financial_keywords(title),
            )
            for title, link, pub_raw in raw_items[:max_items]
            if title
        ]

        if self._raw_store is not None and headlines:
            try:
                self._store_raw_headlines(name, headlines)
            except Exception as exc:
                logger.debug("NewsAggregator: raw_store write failed for %s: %s", name, exc)

        self._cache[name] = (headlines, now)
        logger.debug("NewsAggregator: fetched %d headlines from %s", len(headlines), name)
        return headlines

    # ------------------------------------------------------------------
    # SEC EDGAR 8-K source
    # ------------------------------------------------------------------

    def _fetch_edgar_8k(self, max_items: int) -> List[NewsHeadline]:
        """
        Fetch today's 8-K filings from EDGAR full-text search.
        No API key required. Returns neutral NewsHeadline objects.
        """
        name = "sec_edgar_8k"
        now = time.time()

        if name in self._cache:
            cached, cached_at = self._cache[name]
            if now - cached_at < CACHE_DURATION:
                return cached[:max_items]

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        text = self._fetch_url(EDGAR_8K_URL.format(date=today))
        if not text:
            return []

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.debug("NewsAggregator: EDGAR JSON parse error: %s", exc)
            return []

        headlines: List[NewsHeadline] = []
        for hit in data.get("hits", {}).get("hits", [])[:max_items]:
            src = hit.get("_source", {})
            entity = src.get("entity_name", "Unknown entity")
            period = src.get("period_of_report", today)
            form_type = src.get("form_type", "8-K")
            filing_id = hit.get("_id", "")

            title = f"{entity} filed {form_type} ({period})"
            url = (
                f"https://www.sec.gov/cgi-bin/browse-edgar"
                f"?action=getcompany&filenum={filing_id}"
            ) if filing_id else "https://efts.sec.gov/LATEST/search-index"

            headlines.append(NewsHeadline(
                title=title,
                source=name,
                url=url,
                published=_parse_date(src.get("file_date", today)),
                tickers=_extract_tickers(entity.upper()),
                sentiment="neutral",
                keywords=["sec", "filing", "8-k"],
            ))

        if self._raw_store is not None and headlines:
            try:
                self._store_raw_headlines(name, headlines)
            except Exception as exc:
                logger.debug("NewsAggregator: raw_store write failed for edgar_8k: %s", exc)

        self._cache[name] = (headlines, now)
        logger.debug("NewsAggregator: fetched %d EDGAR 8-K filings", len(headlines))
        return headlines

    # ------------------------------------------------------------------
    # Raw store persistence
    # ------------------------------------------------------------------

    def _store_raw_headlines(self, source_name: str, headlines: List[NewsHeadline]) -> None:
        """Persist headlines to a unified `news_headlines` SQLite table."""
        if self._raw_store is None:
            return
        conn = self._raw_store._get_conn("news_aggregator")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS news_headlines (
                source      TEXT,
                title       TEXT,
                url         TEXT,
                published   TEXT,
                tickers     TEXT,
                sentiment   TEXT,
                keywords    TEXT,
                ingested_at TEXT,
                PRIMARY KEY (source, title, published)
            )
        """)
        now_iso = datetime.now(timezone.utc).isoformat()
        rows = [
            (
                h.source, h.title[:500], h.url[:500], h.published,
                json.dumps(h.tickers), h.sentiment,
                json.dumps(h.keywords), now_iso,
            )
            for h in headlines
        ]
        if rows:
            conn.executemany(
                "INSERT OR REPLACE INTO news_headlines "
                "(source, title, url, published, tickers, sentiment, keywords, ingested_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_headlines(self, max_per_source: int = 10) -> List[NewsHeadline]:
        """
        Fetch headlines from all configured sources.

        Each source is attempted independently. Failures do not
        propagate. Results are sorted newest-first by ISO timestamp.

        Args:
            max_per_source: Upper bound per source.

        Returns:
            List of NewsHeadline, newest-first. Empty if all fail.
        """
        all_headlines: List[NewsHeadline] = []

        for source in RSS_SOURCES:
            try:
                all_headlines.extend(self._fetch_rss_source(source, max_per_source))
            except Exception as exc:
                logger.warning("NewsAggregator: source '%s' failed: %s", source["name"], exc)

        try:
            all_headlines.extend(self._fetch_edgar_8k(max_per_source))
        except Exception as exc:
            logger.warning("NewsAggregator: EDGAR 8-K fetch failed: %s", exc)

        all_headlines.sort(key=lambda h: h.published, reverse=True)
        logger.info(
            "NewsAggregator: aggregated %d headlines from %d sources",
            len(all_headlines), len(RSS_SOURCES) + 1,
        )
        return all_headlines

    def get_ticker_mentions(self, ticker: str) -> List[NewsHeadline]:
        """
        Return cached headlines that mention a specific ticker.

        Searches extracted tickers list first, then falls back to
        word-boundary scan of the raw title. Does NOT trigger a fresh
        fetch — call get_headlines() first if you need current data.

        Args:
            ticker: Symbol to search for, e.g. "AAPL".

        Returns:
            Deduplicated list sorted newest-first.
        """
        ticker_upper = ticker.upper().strip()
        matches: List[NewsHeadline] = []

        for _name, (cached, _at) in self._cache.items():
            for h in cached:
                if ticker_upper in h.tickers:
                    matches.append(h)
                    continue
                if re.search(rf'\b{re.escape(ticker_upper)}\b', h.title):
                    matches.append(h)

        seen: Set[Tuple[str, str]] = set()
        unique: List[NewsHeadline] = []
        for h in matches:
            key = (h.source, h.title)
            if key not in seen:
                seen.add(key)
                unique.append(h)

        unique.sort(key=lambda h: h.published, reverse=True)
        return unique

    def clear_cache(self) -> None:
        """Force-expire all cached headlines. Next call fetches fresh data."""
        self._cache.clear()
        logger.debug("NewsAggregator: cache cleared")

    def close(self) -> None:
        """Close the underlying httpx client. Safe to call multiple times."""
        if self._http is not None and not self._http.is_closed:
            self._http.close()


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

def get_headlines(max_per_source: int = 10) -> List[NewsHeadline]:
    """One-shot convenience: create a client, fetch, close, return."""
    client = NewsAggregatorClient()
    try:
        return client.get_headlines(max_per_source=max_per_source)
    finally:
        client.close()


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    max_items = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    print(f"Fetching financial news headlines (max {max_items} per source)...")

    client = NewsAggregatorClient()
    headlines = client.get_headlines(max_per_source=max_items)

    if headlines:
        by_source: Dict[str, List[NewsHeadline]] = {}
        for h in headlines:
            by_source.setdefault(h.source, []).append(h)
        for src_name, items in sorted(by_source.items()):
            print(f"\n  [{src_name.upper()}] ({len(items)} headlines)")
            for h in items[:3]:
                print(f"    {h.to_plain_language()}")
    else:
        print("  No headlines returned from any source.")

    print("\n  AAPL mentions:")
    aapl = client.get_ticker_mentions("AAPL")
    if aapl:
        for h in aapl[:3]:
            print(f"    {h.to_plain_language()}")
    else:
        print("    None found in current cache.")

    client.close()
    print("\nDone.")
