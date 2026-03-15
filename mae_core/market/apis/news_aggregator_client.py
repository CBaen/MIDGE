"""
news_aggregator_client.py — Free Financial News Headline Aggregator

Aggregates financial news headlines from multiple free RSS feeds and
public APIs. No API keys required. Complements per-ticker Yahoo RSS
with broad market news awareness — major macro events, Fed releases,
and cross-market signals that no single ticker feed would surface.

Sources:
  - Reuters Business RSS (top business/finance stories)
  - CNBC Top News RSS (US market-moving headlines)
  - MarketWatch Top Stories RSS (equity + macro)
  - Bloomberg Markets RSS (may return 403 — handled gracefully)
  - Federal Reserve Press Releases RSS (Fed policy signals)
  - SEC EDGAR Full-Text Search (8-K material events — same-day)

Caching: 30 minutes per source (headlines change frequently but
hammering RSS feeds every step is inconsiderate and gets you blocked).

Rate limiting: 5 seconds between source fetches to be a polite citizen.

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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Dict, List, Optional, Set, Tuple

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Timing constants
# ---------------------------------------------------------------------------
CACHE_DURATION = 1800           # 30 minutes per source
INTER_SOURCE_DELAY = 5.0        # Seconds between source requests
HTTP_TIMEOUT = 15.0             # Per-request timeout

# ---------------------------------------------------------------------------
# Sentiment keyword sets  (keyword-based, no LLM)
# ---------------------------------------------------------------------------
NEGATIVE_KEYWORDS: frozenset = frozenset({
    "crash", "crashes", "plunge", "plunges", "plunging",
    "layoff", "layoffs", "job cuts", "cuts jobs", "mass layoff",
    "recall", "recalled",
    "default", "defaulted",
    "bankruptcy", "bankrupt", "chapter 11", "chapter 7",
    "investigation", "investigated", "probe", "subpoena",
    "fraud", "fraudulent", "ponzi",
    "warning", "warns", "warning sign",
    "downgrade", "downgraded",
    "miss", "misses", "missed estimates",
    "decline", "declines", "declining",
    "loss", "losses",
    "slump", "slumps",
    "shutdown", "shuttered",
    "inflation", "stagflation",
    "recession", "contraction",
    "rate hike", "rate hikes", "tightening",
    "sanction", "sanctions",
    "tariff", "tariffs",
})

POSITIVE_KEYWORDS: frozenset = frozenset({
    "surge", "surges", "surging",
    "record", "record high", "all-time high",
    "upgrade", "upgraded",
    "beat", "beats", "beat expectations", "earnings beat",
    "approval", "approved", "fda approves", "fda approval",
    "launch", "launches", "launched",
    "expansion", "expands", "expanding",
    "partnership", "partnerships",
    "acquisition", "acquires", "merger", "deal",
    "profit", "profits",
    "rally", "rallies", "rallying",
    "breakout",
    "rate cut", "rate cuts", "easing",
    "stimulus",
    "growth", "growing",
    "hiring", "jobs added",
})

# ---------------------------------------------------------------------------
# Financial keywords that add analytical context
# ---------------------------------------------------------------------------
FINANCIAL_KEYWORDS: frozenset = frozenset({
    "fed", "federal reserve", "fomc", "interest rate", "rate hike", "rate cut",
    "inflation", "cpi", "pce", "gdp", "jobs report", "nfp", "unemployment",
    "earnings", "revenue", "guidance", "forecast",
    "ipo", "spac", "merger", "acquisition",
    "sec", "filing", "8-k", "10-k", "10-q",
    "congress", "senate", "legislation", "regulation",
    "oil", "crude", "energy", "natural gas",
    "gold", "silver", "copper",
    "yield", "treasury", "bond", "debt ceiling",
    "bitcoin", "crypto", "cryptocurrency",
    "china", "trade war", "tariff",
    "recession", "contraction", "expansion",
    "stock", "shares", "equities",
    "market", "nasdaq", "s&p", "dow",
    "bank", "banking", "credit",
    "dollar", "yen", "euro", "forex",
})

# ---------------------------------------------------------------------------
# S&P 500 ticker set for standalone-word extraction (top 100 by recognition)
# ---------------------------------------------------------------------------
_SP500_TICKERS: Set[str] = {
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "META", "TSLA", "BRK",
    "UNH", "LLY", "JPM", "V", "AVGO", "XOM", "PG", "MA", "JNJ", "HD",
    "MRK", "ABBV", "CVX", "CRM", "BAC", "COST", "NFLX", "AMD", "PEP", "KO",
    "TMO", "ADBE", "WMT", "ACN", "MCD", "CSCO", "LIN", "ABT", "DHR", "TXN",
    "CMCSA", "VZ", "NEE", "WFC", "PM", "MS", "BMY", "ORCL", "INTC", "T",
    "RTX", "AMGN", "GE", "HON", "QCOM", "IBM", "CAT", "UPS", "COP", "GS",
    "LOW", "SPGI", "DE", "INTU", "ELV", "SBUX", "MDT", "AXP", "ISRG", "PLD",
    "GILD", "CVS", "CI", "SYK", "ADP", "BLK", "TJX", "NOW", "REGN", "ZTS",
    "MO", "DUK", "SO", "PNC", "USB", "MMC", "VRTX", "CB", "SCHW", "ADI",
    "LRCX", "KLAC", "PANW", "SNPS", "CDNS", "MELI", "SHW", "ITW", "APD",
    "GM", "F", "BA", "DAL", "UAL", "AAL", "LUV", "X", "NUE", "CLF",
    "SPY", "QQQ", "IWM", "GLD", "SLV", "USO", "TLT", "HYG",
}

# ---------------------------------------------------------------------------
# RSS feed definitions
# ---------------------------------------------------------------------------
_RSS_SOURCES: List[Dict] = [
    {
        "name": "reuters",
        "url": "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best",
        "label": "Reuters Business",
    },
    {
        "name": "cnbc",
        "url": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
        "label": "CNBC Top News",
    },
    {
        "name": "marketwatch",
        "url": "https://feeds.marketwatch.com/marketwatch/topstories/",
        "label": "MarketWatch",
    },
    {
        "name": "bloomberg",
        "url": "https://feeds.bloomberg.com/markets/news.rss",
        "label": "Bloomberg Markets",
    },
    {
        "name": "federal_reserve",
        "url": "https://www.federalreserve.gov/feeds/press_all.xml",
        "label": "Federal Reserve",
    },
]

# ---------------------------------------------------------------------------
# SEC EDGAR full-text search for 8-K filings (no API key needed)
# ---------------------------------------------------------------------------
_EDGAR_8K_URL = "https://efts.sec.gov/LATEST/search-index?q=%228-K%22&forms=8-K&dateRange=custom&startdt={date}&enddt={date}&hits.hits._source=period_of_report,entity_name,file_date,form_type,biz_location,inc_states"
_EDGAR_FULL_TEXT_URL = "https://efts.sec.gov/LATEST/search-index?q=%22material+event%22&forms=8-K&dateRange=custom&startdt={date}&enddt={date}"

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
# Helpers
# ---------------------------------------------------------------------------

def _parse_date(date_str: str) -> str:
    """
    Normalise any date string to ISO 8601 UTC.

    Tries RFC 2822 (RSS), then ISO 8601, then gives up and returns
    the input string so the headline is never discarded for a bad date.
    """
    if not date_str:
        return datetime.now(timezone.utc).isoformat()
    # RFC 2822 (standard RSS pubDate)
    try:
        dt = parsedate_to_datetime(date_str)
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        pass
    # ISO 8601 variants
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat()
        except Exception:
            pass
    return date_str  # Return as-is rather than discard the headline


def _detect_sentiment(title: str) -> str:
    """
    Classify headline sentiment as positive/negative/neutral using
    keyword matching only. No LLM or external service required.
    """
    lower = title.lower()
    neg_hits = sum(1 for kw in NEGATIVE_KEYWORDS if kw in lower)
    pos_hits = sum(1 for kw in POSITIVE_KEYWORDS if kw in lower)
    if neg_hits > pos_hits:
        return "negative"
    if pos_hits > neg_hits:
        return "positive"
    return "neutral"


def _extract_financial_keywords(title: str) -> List[str]:
    """Return financial keywords found in a headline title."""
    lower = title.lower()
    return [kw for kw in FINANCIAL_KEYWORDS if kw in lower]


def _extract_tickers(title: str) -> List[str]:
    """
    Extract ticker symbols from a headline title via three strategies:

    1. Explicit $TICKER notation  — $AAPL, $TSLA
    2. Parenthetical notation     — (AAPL), (NYSE: AAPL)
    3. Colon notation             — AAPL:
    4. Standalone word match      — known S&P 500 tickers appearing as words
       (capped at 3 tickers to avoid false positives from common words)
    """
    found: List[str] = []
    seen: Set[str] = set()

    # Strategy 1: $TICKER
    for m in re.finditer(r'\$([A-Z]{1,5})\b', title):
        t = m.group(1)
        if t not in seen:
            seen.add(t)
            found.append(t)

    # Strategy 2: (TICKER) or (EXCHANGE: TICKER)
    for m in re.finditer(r'\((?:[A-Z]+:\s*)?([A-Z]{1,5})\)', title):
        t = m.group(1)
        if t not in seen:
            seen.add(t)
            found.append(t)

    # Strategy 3: TICKER: at start of word boundary
    for m in re.finditer(r'\b([A-Z]{1,5}):', title):
        t = m.group(1)
        if t not in seen:
            seen.add(t)
            found.append(t)

    # Strategy 4: known S&P 500 tickers as standalone words (limit noise)
    words = set(re.findall(r'\b([A-Z]{1,5})\b', title))
    standalone = [t for t in words if t in _SP500_TICKERS and t not in seen]
    found.extend(standalone[:3])  # Cap at 3 to suppress false positives

    return found


def _parse_rss_xml(xml_text: str, source_name: str) -> List[Tuple[str, str, str]]:
    """
    Parse raw RSS/Atom XML into (title, link, pubDate) tuples.

    Handles both RSS 2.0 and Atom 1.0 gracefully. Returns an empty
    list rather than raising on malformed XML.
    """
    results: List[Tuple[str, str, str]] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.debug("NewsAggregator: XML parse error for %s: %s", source_name, exc)
        return results

    # Detect namespace (Atom uses {http://www.w3.org/2005/Atom})
    ns_atom = "http://www.w3.org/2005/Atom"

    # --- RSS 2.0 path ---
    # <rss><channel><item><title>/<link>/<pubDate>
    items = root.findall(".//item")
    if items:
        for item in items:
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub = (item.findtext("pubDate") or "").strip()
            if title:
                results.append((title, link, pub))
        return results

    # --- Atom 1.0 path ---
    # <feed><entry><title>/<link href="">/<updated>
    entries = root.findall(f"{{{ns_atom}}}entry")
    if not entries:
        entries = root.findall("entry")  # Some feeds omit namespace

    for entry in entries:
        title_el = entry.find(f"{{{ns_atom}}}title") or entry.find("title")
        title = (title_el.text or "").strip() if title_el is not None else ""

        link_el = entry.find(f"{{{ns_atom}}}link") or entry.find("link")
        if link_el is not None:
            link = link_el.get("href", link_el.text or "").strip()
        else:
            link = ""

        updated_el = entry.find(f"{{{ns_atom}}}updated") or entry.find("updated")
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

    No API keys required. Uses httpx for HTTP. Parses RSS with
    xml.etree.ElementTree (stdlib, no feedparser dependency).

    Cache: 30 minutes per source. Sources are fetched sequentially
    with a 5-second inter-source delay to avoid rate bans.

    Failure policy: every source is wrapped in its own try/except.
    A single blocked, timed-out, or malformed source never affects
    the others. Partial results are always returned.

    Usage:
        client = NewsAggregatorClient(raw_store=store)
        headlines = client.get_headlines(max_per_source=10)
        aapl_news = client.get_ticker_mentions("AAPL")
    """

    # Reusable httpx client — connection pooling + consistent headers
    _HTTP_HEADERS = {
        "User-Agent": "MIDGE/1.0 Financial Research Bot (midge@example.com)",
        "Accept": "application/rss+xml, application/xml, text/xml, */*",
    }

    def __init__(self, raw_store=None):
        """
        Args:
            raw_store: Optional RawStore instance. When provided, raw
                       headlines are persisted to SQLite before processing.
                       Always works without it — just no persistence.
        """
        self._raw_store = raw_store

        # Per-source cache: {source_name: (headlines_list, fetch_epoch)}
        self._cache: Dict[str, Tuple[List[NewsHeadline], float]] = {}

        # Track last fetch time for inter-source rate limiting
        self._last_fetch_time: float = 0.0

        # Lazy-initialised httpx client
        self._http: Optional[httpx.Client] = None

    # ------------------------------------------------------------------
    # HTTP layer
    # ------------------------------------------------------------------

    def _get_http(self) -> httpx.Client:
        """Return (or create) the shared httpx client."""
        if self._http is None or self._http.is_closed:
            self._http = httpx.Client(
                headers=self._HTTP_HEADERS,
                timeout=HTTP_TIMEOUT,
                follow_redirects=True,
            )
        return self._http

    def _fetch_url(self, url: str) -> Optional[str]:
        """
        Fetch URL text with rate limiting and graceful error handling.

        Returns the response body as a string, or None on any failure
        (timeout, 4xx, 5xx, DNS error, etc.).
        """
        # Enforce inter-source delay
        elapsed = time.time() - self._last_fetch_time
        if elapsed < INTER_SOURCE_DELAY:
            time.sleep(INTER_SOURCE_DELAY - elapsed)
        self._last_fetch_time = time.time()

        try:
            http = self._get_http()
            resp = http.get(url)
            if resp.status_code == 200:
                return resp.text
            if resp.status_code == 403:
                logger.debug(
                    "NewsAggregator: 403 Forbidden for %s (likely paywall)", url
                )
            else:
                logger.debug(
                    "NewsAggregator: HTTP %d for %s", resp.status_code, url
                )
            return None
        except httpx.TimeoutException:
            logger.debug("NewsAggregator: timeout fetching %s", url)
            return None
        except httpx.ConnectError:
            logger.debug("NewsAggregator: DNS/connect error for %s", url)
            return None
        except Exception as exc:
            logger.debug("NewsAggregator: fetch error for %s: %s", url, exc)
            return None

    # ------------------------------------------------------------------
    # RSS source fetchers
    # ------------------------------------------------------------------

    def _fetch_rss_source(
        self, source: Dict, max_items: int
    ) -> List[NewsHeadline]:
        """
        Fetch a single RSS source and convert to NewsHeadline objects.

        Checks cache first. Stores raw data via raw_store when available.
        Returns empty list on any failure — never raises.
        """
        name = source["name"]
        url = source["url"]
        now = time.time()

        # Serve from cache if fresh
        if name in self._cache:
            cached_headlines, cached_at = self._cache[name]
            if now - cached_at < CACHE_DURATION:
                logger.debug(
                    "NewsAggregator: serving %s from cache (%d items)",
                    name, len(cached_headlines),
                )
                return cached_headlines[:max_items]

        xml_text = self._fetch_url(url)
        if not xml_text:
            return []

        raw_items = _parse_rss_xml(xml_text, name)
        if not raw_items:
            logger.debug("NewsAggregator: no items parsed from %s", name)
            return []

        headlines: List[NewsHeadline] = []
        for title, link, pub_raw in raw_items[:max_items]:
            if not title:
                continue
            published = _parse_date(pub_raw)
            tickers = _extract_tickers(title)
            sentiment = _detect_sentiment(title)
            keywords = _extract_financial_keywords(title)

            headlines.append(NewsHeadline(
                title=title,
                source=name,
                url=link,
                published=published,
                tickers=tickers,
                sentiment=sentiment,
                keywords=keywords,
            ))

        # Persist raw data
        if self._raw_store is not None and headlines:
            try:
                self._store_raw_headlines(name, headlines)
            except Exception as exc:
                logger.debug(
                    "NewsAggregator: raw_store write failed for %s: %s", name, exc
                )

        self._cache[name] = (headlines, now)
        logger.debug(
            "NewsAggregator: fetched %d headlines from %s", len(headlines), name
        )
        return headlines

    # ------------------------------------------------------------------
    # SEC EDGAR 8-K source
    # ------------------------------------------------------------------

    def _fetch_edgar_8k(self, max_items: int) -> List[NewsHeadline]:
        """
        Fetch today's SEC 8-K filings from EDGAR full-text search.

        Uses the public efts.sec.gov endpoint — no API key required.
        Returns NewsHeadline objects with source="sec_edgar_8k".
        """
        name = "sec_edgar_8k"
        now = time.time()

        if name in self._cache:
            cached_headlines, cached_at = self._cache[name]
            if now - cached_at < CACHE_DURATION:
                return cached_headlines[:max_items]

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        url = _EDGAR_8K_URL.format(date=today)

        text = self._fetch_url(url)
        if not text:
            return []

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.debug("NewsAggregator: EDGAR JSON parse error: %s", exc)
            return []

        hits = data.get("hits", {}).get("hits", [])
        headlines: List[NewsHeadline] = []

        for hit in hits[:max_items]:
            source_data = hit.get("_source", {})
            entity = source_data.get("entity_name", "Unknown entity")
            period = source_data.get("period_of_report", today)
            form_type = source_data.get("form_type", "8-K")
            filing_id = hit.get("_id", "")

            title = f"{entity} filed {form_type} ({period})"
            url_hit = (
                f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
                f"&filenum={filing_id}"
            ) if filing_id else "https://efts.sec.gov/LATEST/search-index"

            # Tickers: EDGAR doesn't return them in this endpoint — skip extraction
            # to avoid false positives from company name fragments
            tickers = _extract_tickers(entity.upper())

            headlines.append(NewsHeadline(
                title=title,
                source=name,
                url=url_hit,
                published=_parse_date(source_data.get("file_date", today)),
                tickers=tickers,
                sentiment="neutral",    # 8-K filing is neutral until context is known
                keywords=["sec", "filing", "8-k"],
            ))

        if self._raw_store is not None and headlines:
            try:
                self._store_raw_headlines(name, headlines)
            except Exception as exc:
                logger.debug(
                    "NewsAggregator: raw_store write failed for edgar_8k: %s", exc
                )

        self._cache[name] = (headlines, now)
        logger.debug(
            "NewsAggregator: fetched %d EDGAR 8-K filings", len(headlines)
        )
        return headlines

    # ------------------------------------------------------------------
    # Raw store persistence
    # ------------------------------------------------------------------

    def _store_raw_headlines(
        self, source_name: str, headlines: List[NewsHeadline]
    ) -> None:
        """
        Persist headlines to raw_store SQLite.

        Creates a unified `news_headlines` table across all sources.
        Deduplicates by (source, title, published) so re-fetches are safe.
        """
        raw_store = self._raw_store
        if raw_store is None:
            return

        # Use the _get_conn pattern from RawStoreBase
        conn = raw_store._get_conn("news_aggregator")
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
        rows = []
        for h in headlines:
            rows.append((
                h.source,
                h.title[:500],
                h.url[:500],
                h.published,
                json.dumps(h.tickers),
                h.sentiment,
                json.dumps(h.keywords),
                now_iso,
            ))

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

        Each source is attempted independently. A failure in one source
        does not affect the others. Results are sorted newest-first.

        Args:
            max_per_source: Maximum headlines to return per source.
                            Total upper bound = sources × max_per_source.

        Returns:
            List of NewsHeadline objects, sorted by published timestamp
            descending. Empty list if all sources fail.
        """
        all_headlines: List[NewsHeadline] = []

        # RSS sources
        for source in _RSS_SOURCES:
            try:
                items = self._fetch_rss_source(source, max_per_source)
                all_headlines.extend(items)
            except Exception as exc:
                logger.warning(
                    "NewsAggregator: source '%s' failed unexpectedly: %s",
                    source["name"], exc,
                )

        # SEC EDGAR 8-K source
        try:
            edgar_items = self._fetch_edgar_8k(max_per_source)
            all_headlines.extend(edgar_items)
        except Exception as exc:
            logger.warning("NewsAggregator: EDGAR 8-K fetch failed: %s", exc)

        # Sort newest-first. Use published string for ISO-sortable comparison;
        # fall back to stable ordering for non-parseable dates.
        all_headlines.sort(key=lambda h: h.published, reverse=True)

        logger.info(
            "NewsAggregator: aggregated %d headlines from %d sources",
            len(all_headlines), len(_RSS_SOURCES) + 1,
        )
        return all_headlines

    def get_ticker_mentions(self, ticker: str) -> List[NewsHeadline]:
        """
        Return all cached headlines that mention a specific ticker.

        Searches across both extracted tickers and the headline title
        itself (to catch cases where the extraction missed a mention).

        Does NOT trigger a fresh fetch — uses whatever is in cache
        from the last get_headlines() call. Call get_headlines() first
        if you need fresh data.

        Args:
            ticker: Ticker symbol to search for, e.g. "AAPL".

        Returns:
            List of NewsHeadline objects mentioning the ticker.
            Sorted newest-first.
        """
        ticker_upper = ticker.upper().strip()
        matches: List[NewsHeadline] = []

        for _source_name, (cached_headlines, _cached_at) in self._cache.items():
            for h in cached_headlines:
                # Check extracted tickers list first (fast path)
                if ticker_upper in h.tickers:
                    matches.append(h)
                    continue
                # Fallback: scan title for standalone mention
                # Word-boundary check prevents "CAT" matching "CATALYST"
                if re.search(rf'\b{re.escape(ticker_upper)}\b', h.title):
                    matches.append(h)

        # Deduplicate by (source, title) in case a headline appeared in
        # multiple passes without a cache refresh
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
        # Group by source for readability
        by_source: Dict[str, List[NewsHeadline]] = {}
        for h in headlines:
            by_source.setdefault(h.source, []).append(h)
        for source_name, items in sorted(by_source.items()):
            print(f"\n  [{source_name.upper()}] ({len(items)} headlines)")
            for h in items[:3]:
                print(f"    {h.to_plain_language()}")
    else:
        print("  No headlines returned from any source.")

    # Demonstrate ticker mentions
    print("\n  AAPL mentions:")
    aapl = client.get_ticker_mentions("AAPL")
    if aapl:
        for h in aapl[:3]:
            print(f"    {h.to_plain_language()}")
    else:
        print("    None found in current cache.")

    client.close()
    print("\nDone.")
