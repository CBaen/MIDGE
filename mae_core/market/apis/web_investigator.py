"""
web_investigator.py — Web Investigation Engine for MIDGE

When MIDGE notices something interesting — insider buying cluster, unusual
convergence, cross-market anomaly — she can crawl the open web to find out
WHY. This module is the rabbit-hole follower.

Sources crawled (all free, no API key required):
  - Yahoo Finance news page (per-ticker full article text)
  - Google News RSS (query-based, returns recent headlines + links)
  - SEC EDGAR full-text search (EFTS — filings mentioning a query)
  - Reddit search JSON API (recent community discussion)
  - Federal Register API (regulatory actions by keyword)

Design rules:
  - 2-second delay between outbound HTTP requests (courtesy rate limiting)
  - 1-hour in-memory result cache per (query, source) pair
  - Never crash on a single source failure — partial results are valuable
  - raw_store is optional — works standalone
  - trafilatura extracts clean text from HTML; feedparser handles RSS

Data model:
  WebFinding — one piece of investigative evidence from one web source.
  Domain is always "investigation" so it integrates with ConvergenceAlerter.
"""

from __future__ import annotations

import logging
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate limiting + caching
# ---------------------------------------------------------------------------

_REQUEST_DELAY = 2.0          # seconds between HTTP fetches
_CACHE_TTL = 3600             # 1-hour result cache per (cache_key)
_LAST_REQUEST_TIME: list[float] = [0.0]

# Cache: cache_key → (list[WebFinding], fetched_at_timestamp)
_RESULT_CACHE: dict[str, tuple[list, float]] = {}

# Common English words mistaken for tickers — skip them in extraction
_COMMON_WORDS = frozenset({
    "A", "I", "AN", "AS", "AT", "BE", "BY", "DO", "GO", "IF", "IN",
    "IS", "IT", "MY", "NO", "OF", "ON", "OR", "SO", "TO", "UP", "US",
    "WE", "AI", "ALL", "AND", "ARE", "BUT", "CAN", "FOR", "GET", "HAD",
    "HAS", "HIM", "HIS", "HOW", "ITS", "MAY", "NEW", "NOT", "NOW", "OUR",
    "OUT", "SAY", "SHE", "THE", "TOO", "TWO", "WAS", "WAY", "WHO", "WHY",
    "WITH", "WILL", "THIS", "THAT", "HAVE", "FROM", "THEY", "BEEN", "YOUR",
    "ALSO", "INTO", "SAID", "OVER", "THAN", "THEN", "WHAT", "WHEN", "WHO",
    "MORE", "MUCH", "SOME", "SUCH", "EACH", "BOTH", "MANY", "GOOD", "WELL",
    "ONLY", "LIKE", "JUST", "BACK", "LONG", "TIME", "HIGH", "NEXT", "EVEN",
    "MOST", "LAST", "DOWN", "NEED", "PLAN", "MAKE", "TAKE", "YEAR", "SAYS",
    "CALL", "COME", "CAME", "DOES", "DONE", "HELD", "HOLD", "HURT", "KEEP",
    "LESS", "LOOK", "MADE", "MEAN", "MOVE", "MUST", "NEAR", "PART", "PAST",
    "SELL", "STAY", "TELL", "THEM", "VERY", "WANT", "WERE", "FIND",
})

# Financial key-phrase indicators for relevance scoring
_FINANCIAL_SIGNALS = frozenset({
    "earnings", "revenue", "guidance", "acquisition", "merger", "buyout",
    "fda", "sec", "ceo", "cfo", "insider", "shares", "stock", "trade",
    "lawsuit", "investigation", "subpoena", "probe", "recall", "layoff",
    "layoffs", "job cuts", "contract", "agreement", "partnership", "deal",
    "quarter", "profit", "loss", "forecast", "outlook", "upgrade", "downgrade",
    "short", "squeeze", "catalyst", "breakout", "bankruptcy", "default",
    "dividend", "buyback", "offering", "dilution", "patent", "approval",
})


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class WebFinding:
    """One piece of investigative evidence from a web source.

    Carries enough context for the convergence engine to use:
    - domain="investigation" integrates with ConvergenceAlerter
    - tickers_mentioned enables cross-ticker correlation
    - key_phrases drives keyword matching
    - relevance 0-1 for confidence weighting

    Text summary is truncated at 500 chars — we want signal density,
    not full article storage. Full text goes to raw_store if wired.
    """

    url: str
    title: str
    source: str               # "sec_edgar", "yahoo_news", "google_news", "reddit", "federal_register"
    text_summary: str         # First 500 chars of extracted clean text
    relevance: float          # 0.0–1.0 keyword overlap score
    published: str            # ISO 8601 or empty string
    tickers_mentioned: list   # Other ticker symbols found in text
    key_phrases: list         # Important phrases extracted from text
    domain: str = "investigation"

    # Investigation metadata
    query_ticker: str = ""    # Ticker this finding is about (if applicable)
    query_topic: str = ""     # Topic this finding is about (if applicable)
    depth: int = 0            # 0 = primary find, 1+ = rabbit-hole depth

    found_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "title": self.title,
            "source": self.source,
            "text_summary": self.text_summary,
            "relevance": round(self.relevance, 4),
            "published": self.published,
            "tickers_mentioned": self.tickers_mentioned,
            "key_phrases": self.key_phrases[:10],
            "domain": self.domain,
            "query_ticker": self.query_ticker,
            "query_topic": self.query_topic,
            "depth": self.depth,
            "found_at": self.found_at,
        }

    def to_plain_language(self) -> str:
        kw = ", ".join(self.key_phrases[:4]) if self.key_phrases else "none"
        tickers = ", ".join(self.tickers_mentioned[:5]) if self.tickers_mentioned else "none"
        return (
            f"[{self.source}] {self.title[:80]} "
            f"(relevance={self.relevance:.2f}, tickers={tickers}, phrases={kw})"
        )


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _rate_limited_get(url: str, headers: Optional[dict] = None, timeout: float = 15.0) -> Optional[str]:
    """Fetch URL with rate limiting. Returns raw text or None on failure."""
    try:
        import httpx

        # Enforce rate limit
        now = time.time()
        wait = _REQUEST_DELAY - (now - _LAST_REQUEST_TIME[0])
        if wait > 0:
            time.sleep(wait)

        _LAST_REQUEST_TIME[0] = time.time()

        _headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; MIDGE/1.0; market intelligence research; "
                "contact: research@midge.local)"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        if headers:
            _headers.update(headers)

        with httpx.Client(follow_redirects=True, timeout=timeout) as client:
            resp = client.get(url, headers=_headers)
            if resp.status_code == 200:
                return resp.text
            logger.debug("web_investigator: HTTP %d for %s", resp.status_code, url)
            return None
    except Exception as exc:
        logger.debug("web_investigator: fetch failed for %s: %s", url, exc)
        return None


def _extract_text(html: str, url: str = "") -> str:
    """Extract clean article text from HTML using trafilatura."""
    if not html:
        return ""
    try:
        import trafilatura
        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
            favor_precision=True,
        )
        return (text or "").strip()
    except Exception:
        # Fallback: strip tags crudely
        clean = re.sub(r"<[^>]+>", " ", html)
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean[:1000]


def _truncate(text: str, max_chars: int = 500) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"


# ---------------------------------------------------------------------------
# Entity extraction helpers
# ---------------------------------------------------------------------------

_TICKER_RE = re.compile(r"\b([A-Z]{1,5})\b")
_PHRASE_STOPWORDS = frozenset({
    "the", "and", "for", "with", "this", "that", "from", "have", "been",
    "will", "they", "said", "its", "their", "which", "about", "more", "also",
    "into", "over", "some", "than", "when", "what", "other", "after",
    "market", "stock", "share", "shares", "company", "companies",
})


def _extract_tickers(text: str) -> list[str]:
    """Pull plausible ticker symbols from text. Excludes common English words."""
    found = []
    for m in _TICKER_RE.finditer(text):
        tok = m.group(1)
        if tok not in _COMMON_WORDS and len(tok) >= 2:
            found.append(tok)
    # Deduplicate preserving order
    seen: set[str] = set()
    unique = []
    for t in found:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique[:15]


def _extract_key_phrases(text: str, query_terms: list[str]) -> list[str]:
    """Extract 2-word phrases from text that overlap with query terms or financial signals."""
    if not text:
        return []
    words = re.findall(r"\b[a-z]+\b", text.lower())
    phrases = []
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]
        if w1 in _PHRASE_STOPWORDS or w2 in _PHRASE_STOPWORDS:
            continue
        phrase = f"{w1} {w2}"
        # Include if either word is a query term or financial signal
        q_lower = {q.lower() for q in query_terms}
        if (w1 in q_lower or w2 in q_lower or
                w1 in _FINANCIAL_SIGNALS or w2 in _FINANCIAL_SIGNALS):
            phrases.append(phrase)
    # Deduplicate
    seen: set[str] = set()
    unique = []
    for p in phrases:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique[:20]


def _score_relevance(text: str, query_terms: list[str]) -> float:
    """Score 0-1: how many query terms appear in text."""
    if not text or not query_terms:
        return 0.0
    lower = text.lower()
    hits = sum(1 for t in query_terms if t.lower() in lower)
    base = hits / max(len(query_terms), 1)
    # Bonus: financial signal words present
    fin_hits = sum(1 for kw in _FINANCIAL_SIGNALS if kw in lower)
    bonus = min(0.3, fin_hits * 0.03)
    return min(1.0, base + bonus)


def _cache_key(prefix: str, query: str) -> str:
    return f"{prefix}:{query.lower().strip()}"


def _from_cache(key: str) -> Optional[list]:
    entry = _RESULT_CACHE.get(key)
    if entry is None:
        return None
    findings, fetched_at = entry
    if time.time() - fetched_at > _CACHE_TTL:
        del _RESULT_CACHE[key]
        return None
    return findings


def _to_cache(key: str, findings: list) -> None:
    _RESULT_CACHE[key] = (findings, time.time())
    # Evict oldest entries if cache grows large
    if len(_RESULT_CACHE) > 500:
        oldest_key = min(_RESULT_CACHE, key=lambda k: _RESULT_CACHE[k][1])
        del _RESULT_CACHE[oldest_key]


# ---------------------------------------------------------------------------
# Source crawlers
# ---------------------------------------------------------------------------

def _crawl_google_news_rss(query: str, query_terms: list[str]) -> list[WebFinding]:
    """Google News RSS: recent headlines mentioning the query."""
    findings = []
    encoded = urllib.parse.quote_plus(query)
    url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"

    text = _rate_limited_get(url, headers={"Accept": "application/rss+xml, text/xml, */*"})
    if not text:
        return findings

    try:
        root = ET.fromstring(text)
        channel = root.find("channel")
        if channel is None:
            return findings
        items = channel.findall("item")[:10]
        for item in items:
            title_el = item.find("title")
            link_el = item.find("link")
            pub_el = item.find("pubDate")

            title = (title_el.text or "").strip() if title_el is not None else ""
            link = (link_el.text or "").strip() if link_el is not None else ""
            published = (pub_el.text or "").strip() if pub_el is not None else ""

            if not title:
                continue

            # Score based on title alone (no crawl of article — saves requests)
            relevance = _score_relevance(title, query_terms)
            key_phrases = _extract_key_phrases(title, query_terms)
            tickers = _extract_tickers(title)

            findings.append(WebFinding(
                url=link,
                title=title,
                source="google_news",
                text_summary=_truncate(title, 500),
                relevance=relevance,
                published=published,
                tickers_mentioned=tickers,
                key_phrases=key_phrases,
                query_topic=query,
            ))
    except ET.ParseError as exc:
        logger.debug("google_news RSS parse error for %r: %s", query, exc)

    return findings


def _crawl_yahoo_news(ticker: str, query_terms: list[str]) -> list[WebFinding]:
    """Yahoo Finance news page: fetch HTML, extract article links + text."""
    findings = []
    url = f"https://finance.yahoo.com/quote/{ticker}/news"

    html = _rate_limited_get(url)
    if not html:
        return findings

    # Extract article text from the full page
    text = _extract_text(html, url)
    if not text:
        return findings

    relevance = _score_relevance(text, query_terms)
    key_phrases = _extract_key_phrases(text, query_terms)
    tickers = [t for t in _extract_tickers(text) if t != ticker]

    # Pull headlines from the text (each line that looks like a news title)
    lines = [ln.strip() for ln in text.split("\n") if len(ln.strip()) > 20]
    title = lines[0] if lines else f"{ticker} news"

    findings.append(WebFinding(
        url=url,
        title=title[:200],
        source="yahoo_news",
        text_summary=_truncate(text, 500),
        relevance=relevance,
        published="",
        tickers_mentioned=tickers,
        key_phrases=key_phrases,
        query_ticker=ticker,
        query_topic=ticker,
    ))
    return findings


def _crawl_sec_edgar_efts(query: str, query_terms: list[str]) -> list[WebFinding]:
    """SEC EDGAR full-text search (EFTS): recent filings mentioning the query.

    Uses the public EFTS JSON API — no key required.
    Returns up to 5 most recent filing summaries.
    """
    findings = []
    from datetime import timedelta
    start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    encoded = urllib.parse.quote_plus(query)
    url = (
        f"https://efts.sec.gov/LATEST/search-index?q=%22{encoded}%22"
        f"&dateRange=custom&startdt={start_date}&hits.hits._source=period_of_report"
        f"&hits.hits._source=entity_name&hits.hits._source=file_date"
        f"&hits.hits._source=form_type&hits.hits.total.value=true"
        f"&hits.hits.hits.total=5"
    )

    # EFTS has its own rate limit — use a friendlier endpoint
    api_url = (
        f"https://efts.sec.gov/LATEST/search-index?q=%22{encoded}%22"
        f"&dateRange=custom&startdt={start_date}"
    )

    text = _rate_limited_get(
        api_url,
        headers={"Accept": "application/json"},
    )
    if not text:
        return findings

    try:
        import json as _json
        data = _json.loads(text)
        hits = data.get("hits", {}).get("hits", [])[:5]
        for hit in hits:
            src = hit.get("_source", {})
            entity = src.get("entity_name", "Unknown entity")
            form_type = src.get("form_type", "filing")
            file_date = src.get("file_date", "")
            period = src.get("period_of_report", "")
            accession = hit.get("_id", "").replace(":", "/")
            filing_url = f"https://www.sec.gov/Archives/edgar/data/{accession}" if accession else ""

            description = (
                f"{entity} filed {form_type}"
                + (f" (period: {period})" if period else "")
                + f" — query match: {query}"
            )
            relevance = _score_relevance(description, query_terms)
            key_phrases = _extract_key_phrases(description, query_terms)
            tickers = _extract_tickers(entity)

            findings.append(WebFinding(
                url=filing_url or api_url,
                title=f"SEC {form_type}: {entity}"[:200],
                source="sec_edgar",
                text_summary=_truncate(description, 500),
                relevance=relevance,
                published=file_date,
                tickers_mentioned=tickers,
                key_phrases=key_phrases,
                query_topic=query,
            ))
    except Exception as exc:
        logger.debug("SEC EFTS parse error for %r: %s", query, exc)

    return findings


def _crawl_reddit(query: str, query_terms: list[str]) -> list[WebFinding]:
    """Reddit search JSON API: recent community discussion about a query."""
    findings = []
    encoded = urllib.parse.quote_plus(query)
    url = f"https://www.reddit.com/search.json?q={encoded}&sort=new&limit=10&type=link"

    text = _rate_limited_get(
        url,
        headers={
            "User-Agent": "MIDGE:market-research-bot:v1.0 (by /u/midge_research)"
        },
    )
    if not text:
        return findings

    try:
        import json as _json
        data = _json.loads(text)
        posts = data.get("data", {}).get("children", [])[:8]
        for post in posts:
            pdata = post.get("data", {})
            title = pdata.get("title", "").strip()
            permalink = pdata.get("permalink", "")
            subreddit = pdata.get("subreddit", "")
            selftext = pdata.get("selftext", "")[:300]
            created_utc = pdata.get("created_utc", 0)
            score = pdata.get("score", 0)
            num_comments = pdata.get("num_comments", 0)

            if not title:
                continue

            # Skip low-engagement posts (spam filter)
            if score < 1 and num_comments < 1:
                continue

            full_text = f"{title} {selftext}".strip()
            published = (
                datetime.fromtimestamp(created_utc, tz=timezone.utc).isoformat(timespec="seconds")
                if created_utc else ""
            )
            relevance = _score_relevance(full_text, query_terms)
            key_phrases = _extract_key_phrases(full_text, query_terms)
            tickers = _extract_tickers(full_text)
            post_url = f"https://www.reddit.com{permalink}" if permalink else ""
            context = f"r/{subreddit} • {score} pts • {num_comments} comments"

            findings.append(WebFinding(
                url=post_url,
                title=title[:200],
                source="reddit",
                text_summary=_truncate(f"{context}: {full_text}", 500),
                relevance=relevance,
                published=published,
                tickers_mentioned=tickers,
                key_phrases=key_phrases,
                query_topic=query,
            ))
    except Exception as exc:
        logger.debug("Reddit JSON parse error for %r: %s", query, exc)

    return findings


def _crawl_federal_register(query: str, query_terms: list[str]) -> list[WebFinding]:
    """Federal Register API: recent regulatory actions matching a keyword."""
    findings = []
    encoded = urllib.parse.quote_plus(query)
    url = (
        f"https://www.federalregister.gov/api/v1/documents.json"
        f"?conditions[term]={encoded}&per_page=5&order=newest"
        f"&fields[]=title&fields[]=abstract&fields[]=publication_date"
        f"&fields[]=html_url&fields[]=agencies"
    )

    text = _rate_limited_get(url, headers={"Accept": "application/json"})
    if not text:
        return findings

    try:
        import json as _json
        data = _json.loads(text)
        results = data.get("results", [])[:5]
        for doc in results:
            title = doc.get("title", "").strip()
            abstract = doc.get("abstract", "").strip()
            pub_date = doc.get("publication_date", "")
            html_url = doc.get("html_url", "")
            agencies = doc.get("agencies", [])
            agency_names = ", ".join(
                a.get("name", "") for a in agencies[:3] if isinstance(a, dict)
            )

            if not title:
                continue

            full_text = f"{title} {abstract} {agency_names}"
            relevance = _score_relevance(full_text, query_terms)
            key_phrases = _extract_key_phrases(full_text, query_terms)
            tickers = _extract_tickers(title)

            findings.append(WebFinding(
                url=html_url,
                title=title[:200],
                source="federal_register",
                text_summary=_truncate(f"Agency: {agency_names}. {abstract}", 500),
                relevance=relevance,
                published=pub_date,
                tickers_mentioned=tickers,
                key_phrases=key_phrases,
                query_topic=query,
            ))
    except Exception as exc:
        logger.debug("Federal Register parse error for %r: %s", query, exc)

    return findings


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class WebInvestigator:
    """Web investigation engine.

    When MIDGE notices something interesting (insider buying cluster, unusual
    convergence, cross-market anomaly), she calls this to crawl the open web
    and find out WHY — following the rabbit hole.

    All results are cached for 1 hour per (query, source) pair.
    Rate limited to 2 seconds between outbound HTTP requests.
    """

    def __init__(self, raw_store=None) -> None:
        """
        Args:
            raw_store: Optional RawStore for SQLite persistence of findings.
                       If None, findings are returned in-memory only.
        """
        self._raw_store = raw_store

    # ── Primary investigation methods ────────────────────────────────────────

    def investigate_ticker(self, ticker: str, context: str = "") -> list[WebFinding]:
        """When MIDGE notices something about a ticker, investigate WHY.

        Crawls Yahoo Finance news, Google News RSS, SEC EDGAR, and Reddit
        to surface narrative context behind the signal.

        Args:
            ticker: Ticker symbol (e.g. "AAPL", "LMT").
            context: Optional context phrase (e.g. "insider buying cluster",
                     "unusual convergence with government contracts").

        Returns:
            List of WebFinding objects sorted by relevance descending.
        """
        ticker = ticker.upper().strip()
        cache_key = _cache_key("ticker", f"{ticker}:{context}")
        cached = _from_cache(cache_key)
        if cached is not None:
            logger.debug("WebInvestigator: cache hit for ticker=%s", ticker)
            return cached

        logger.info("WebInvestigator: investigating ticker=%s context=%r", ticker, context)

        query_terms = [ticker] + (context.lower().split() if context else [])
        query = f"{ticker} {context}".strip() if context else ticker

        all_findings: list[WebFinding] = []

        # Source 1: Yahoo Finance news page
        try:
            found = _crawl_yahoo_news(ticker, query_terms)
            for f in found:
                f.query_ticker = ticker
            all_findings.extend(found)
        except Exception as exc:
            logger.debug("WebInvestigator: yahoo_news failed for %s: %s", ticker, exc)

        # Source 2: Google News RSS
        try:
            found = _crawl_google_news_rss(query, query_terms)
            for f in found:
                f.query_ticker = ticker
            all_findings.extend(found)
        except Exception as exc:
            logger.debug("WebInvestigator: google_news failed for %s: %s", ticker, exc)

        # Source 3: SEC EDGAR full-text search
        try:
            found = _crawl_sec_edgar_efts(ticker, query_terms)
            for f in found:
                f.query_ticker = ticker
            all_findings.extend(found)
        except Exception as exc:
            logger.debug("WebInvestigator: sec_edgar failed for %s: %s", ticker, exc)

        # Source 4: Reddit
        try:
            reddit_query = f"{ticker} stock" + (f" {context}" if context else "")
            found = _crawl_reddit(reddit_query, query_terms)
            for f in found:
                f.query_ticker = ticker
            all_findings.extend(found)
        except Exception as exc:
            logger.debug("WebInvestigator: reddit failed for %s: %s", ticker, exc)

        # Sort by relevance descending, limit to top 20
        all_findings.sort(key=lambda f: f.relevance, reverse=True)
        results = all_findings[:20]

        _to_cache(cache_key, results)
        self._persist_findings(results, ticker=ticker, context=context)
        return results

    def investigate_event(
        self,
        event_description: str,
        tickers: Optional[list] = None,
    ) -> list[WebFinding]:
        """When MIDGE sees a cross-market anomaly or cascade, investigate the cause.

        Crawls Google News RSS, SEC EDGAR, and Federal Register to surface
        the regulatory/economic/political drivers behind an observed anomaly.

        Args:
            event_description: Human-readable event description
                               (e.g. "EIA crude oil inventory surprise",
                                "congressional cluster buying defense stocks").
            tickers: Optional list of affected tickers for ticker-level context.

        Returns:
            List of WebFinding objects sorted by relevance descending.
        """
        cache_key = _cache_key("event", event_description)
        cached = _from_cache(cache_key)
        if cached is not None:
            return cached

        logger.info("WebInvestigator: investigating event=%r", event_description)

        query_terms = event_description.lower().split()
        all_findings: list[WebFinding] = []

        # Source 1: Google News RSS for the event topic
        try:
            found = _crawl_google_news_rss(event_description, query_terms)
            all_findings.extend(found)
        except Exception as exc:
            logger.debug("WebInvestigator: google_news failed for event: %s", exc)

        # Source 2: SEC EDGAR — regulatory filings that might explain the event
        try:
            found = _crawl_sec_edgar_efts(event_description, query_terms)
            all_findings.extend(found)
        except Exception as exc:
            logger.debug("WebInvestigator: sec_edgar failed for event: %s", exc)

        # Source 3: Federal Register — government regulatory actions
        try:
            found = _crawl_federal_register(event_description, query_terms)
            all_findings.extend(found)
        except Exception as exc:
            logger.debug("WebInvestigator: federal_register failed for event: %s", exc)

        # Source 4: Google News RSS for each affected ticker (capped to 3)
        if tickers:
            for ticker in tickers[:3]:
                try:
                    ticker_query = f"{ticker} {event_description}"
                    found = _crawl_google_news_rss(ticker_query, query_terms + [ticker])
                    for f in found:
                        f.query_ticker = ticker
                    all_findings.extend(found)
                except Exception as exc:
                    logger.debug(
                        "WebInvestigator: google_news ticker=%s failed for event: %s",
                        ticker, exc
                    )

        all_findings.sort(key=lambda f: f.relevance, reverse=True)
        results = all_findings[:20]

        _to_cache(cache_key, results)
        self._persist_findings(results, topic=event_description)
        return results

    def follow_rabbit_hole(
        self,
        finding: WebFinding,
        depth: int = 2,
    ) -> list[WebFinding]:
        """When a finding mentions something interesting, follow it deeper.

        Extracts tickers and key phrases from the finding, then searches
        for those. Limited to `depth` levels to prevent infinite crawling.

        Args:
            finding: The WebFinding to follow.
            depth: Maximum rabbit-hole depth (1 = follow once, 2 = follow twice).
                   Current finding's depth must be < depth to continue.

        Returns:
            List of deeper WebFinding objects.
        """
        if finding.depth >= depth:
            return []

        new_depth = finding.depth + 1
        all_deeper: list[WebFinding] = []

        # Follow each mentioned ticker that isn't the original query ticker
        for ticker in finding.tickers_mentioned[:5]:
            if ticker == finding.query_ticker:
                continue
            if ticker in _COMMON_WORDS:
                continue
            cache_key = _cache_key(f"rabbit_{new_depth}", ticker)
            cached = _from_cache(cache_key)
            if cached is not None:
                all_deeper.extend(cached)
                continue
            try:
                # Use key phrases from the parent finding as context
                context = " ".join(finding.key_phrases[:3])
                found = self.investigate_ticker(ticker, context)
                for f in found:
                    f.depth = new_depth
                _to_cache(cache_key, found)
                all_deeper.extend(found)
            except Exception as exc:
                logger.debug(
                    "WebInvestigator: rabbit hole ticker=%s depth=%d failed: %s",
                    ticker, new_depth, exc
                )

        # Also follow key phrases as event searches (first 2 only)
        for phrase in finding.key_phrases[:2]:
            if len(phrase) < 5:
                continue
            cache_key = _cache_key(f"rabbit_phrase_{new_depth}", phrase)
            cached = _from_cache(cache_key)
            if cached is not None:
                all_deeper.extend(cached)
                continue
            try:
                found = self.investigate_event(phrase)
                for f in found:
                    f.depth = new_depth
                    f.query_topic = phrase
                _to_cache(cache_key, found)
                all_deeper.extend(found)
            except Exception as exc:
                logger.debug(
                    "WebInvestigator: rabbit hole phrase=%r depth=%d failed: %s",
                    phrase, new_depth, exc
                )

        all_deeper.sort(key=lambda f: f.relevance, reverse=True)
        return all_deeper[:10]

    # ── Persistence ───────────────────────────────────────────────────────────

    def _persist_findings(
        self,
        findings: list[WebFinding],
        ticker: str = "",
        context: str = "",
        topic: str = "",
    ) -> None:
        """Append findings to JSONL + optionally to raw_store."""
        if not findings:
            return

        import json as _json
        from pathlib import Path

        out_path = Path("data/midge/investigations.jsonl")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(out_path, "a", encoding="utf-8") as fh:
                for f in findings:
                    record = f.to_dict()
                    if ticker:
                        record["triggered_by_ticker"] = ticker
                    if context:
                        record["triggered_by_context"] = context
                    if topic:
                        record["triggered_by_topic"] = topic
                    fh.write(_json.dumps(record) + "\n")
        except Exception as exc:
            logger.debug("WebInvestigator: JSONL write failed: %s", exc)

        # Optional raw_store persistence
        if self._raw_store is not None:
            try:
                store_fn = getattr(self._raw_store, "store_web_investigation", None)
                if store_fn is not None:
                    store_fn(ticker or topic, [f.to_dict() for f in findings])
            except Exception as exc:
                logger.debug("WebInvestigator: raw_store write failed: %s", exc)
