"""
web_investigator.py — Web Investigation Engine for MIDGE

When MIDGE notices something interesting (insider buying cluster, unusual
convergence, cross-market anomaly), she crawls the open web to find out WHY.

Sources (all free, no API key):  Yahoo Finance news, Google News RSS,
SEC EDGAR EFTS, Reddit, Federal Register.

Source crawlers live in web_investigator_crawlers.py.
Rate limit: 2 s between requests. Cache: 1 hour per query.
"""

from __future__ import annotations

import logging
import re
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate limiting + caching
# ---------------------------------------------------------------------------

_REQUEST_DELAY = 2.0          # seconds between HTTP fetches
_CACHE_TTL = 3600             # 1-hour result cache
_LAST_REQUEST_TIME: list[float] = [0.0]
_RESULT_CACHE: dict[str, tuple] = {}    # key → (findings, timestamp)

# Common English words mistaken for tickers — skip in extraction
_COMMON_WORDS = frozenset(
    "A I AN AS AT BE BY DO GO IF IN IS IT MY NO OF ON OR SO TO UP US WE "
    "AI ALL AND ARE BUT CAN FOR GET HAD HAS HIM HIS HOW ITS MAY NEW NOT "
    "NOW OUR OUT SAY SHE THE TOO TWO WAS WAY WHO WHY WITH WILL THIS THAT "
    "HAVE FROM THEY BEEN YOUR ALSO INTO SAID OVER THAN THEN WHAT WHEN MORE "
    "MUCH SOME SUCH EACH BOTH MANY GOOD WELL ONLY LIKE JUST BACK LONG TIME "
    "HIGH NEXT EVEN MOST LAST DOWN NEED PLAN MAKE TAKE YEAR SAYS CALL COME "
    "CAME DOES DONE HELD HOLD HURT KEEP LESS LOOK MADE MEAN MOVE MUST NEAR "
    "PART PAST SELL STAY TELL THEM VERY WANT WERE FIND".split()
)

_FINANCIAL_SIGNALS = frozenset(
    "earnings revenue guidance acquisition merger buyout fda sec ceo cfo "
    "insider shares stock trade lawsuit investigation subpoena probe recall "
    "layoff contract agreement partnership deal quarter profit loss forecast "
    "outlook upgrade downgrade short squeeze catalyst breakout bankruptcy "
    "default dividend buyback offering dilution patent approval".split()
)

_PHRASE_STOPWORDS = frozenset(
    "the and for with this that from have been will they said its their "
    "which about more also into over some than when what other after "
    "market stock share shares company companies".split()
)

_TICKER_RE = re.compile(r"\b([A-Z]{1,5})\b")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class WebFinding:
    """One piece of investigative evidence from a web source.

    domain="investigation" integrates with ConvergenceAlerter as a synthetic
    domain — one independent vote that compounds with insider/macro/technical
    signals to push a ticker toward the convergence threshold.
    """

    url: str
    title: str
    source: str             # "sec_edgar", "yahoo_news", "google_news", "reddit", etc.
    text_summary: str       # First 500 chars of extracted clean text
    relevance: float        # 0.0–1.0 keyword overlap score
    published: str          # ISO 8601 or empty string
    tickers_mentioned: list
    key_phrases: list
    domain: str = "investigation"

    query_ticker: str = ""  # Ticker this finding is about (if applicable)
    query_topic: str = ""   # Topic/event this finding is about
    depth: int = 0          # 0 = primary find, 1+ = rabbit-hole depth

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
# Shared HTTP + text helpers (passed to crawlers as a dict)
# ---------------------------------------------------------------------------

def _rate_limited_get(url: str, headers: Optional[dict] = None, timeout: float = 15.0) -> Optional[str]:
    """Fetch URL with 2-second courtesy rate limit. Returns text or None."""
    try:
        import httpx
        now = time.time()
        wait = _REQUEST_DELAY - (now - _LAST_REQUEST_TIME[0])
        if wait > 0:
            time.sleep(wait)
        _LAST_REQUEST_TIME[0] = time.time()

        _h = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; MIDGE/1.0; market research; contact: research@midge.local)"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        if headers:
            _h.update(headers)

        with httpx.Client(follow_redirects=True, timeout=timeout) as client:
            resp = client.get(url, headers=_h)
            return resp.text if resp.status_code == 200 else None
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
            html, include_comments=False, include_tables=False,
            no_fallback=False, favor_precision=True,
        )
        return (text or "").strip()
    except Exception:
        clean = re.sub(r"<[^>]+>", " ", html)
        return re.sub(r"\s+", " ", clean).strip()[:1000]


def _extract_tickers(text: str) -> list[str]:
    """Pull plausible ticker symbols from text. Excludes common English words."""
    seen: set[str] = set()
    unique = []
    for m in _TICKER_RE.finditer(text):
        tok = m.group(1)
        if tok not in _COMMON_WORDS and len(tok) >= 2 and tok not in seen:
            seen.add(tok)
            unique.append(tok)
    return unique[:15]


def _extract_key_phrases(text: str, query_terms: list[str]) -> list[str]:
    """Extract 2-word phrases that overlap with query or financial signals."""
    if not text:
        return []
    words = re.findall(r"\b[a-z]+\b", text.lower())
    q_lower = {q.lower() for q in query_terms}
    seen: set[str] = set()
    phrases = []
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]
        if w1 in _PHRASE_STOPWORDS or w2 in _PHRASE_STOPWORDS:
            continue
        phrase = f"{w1} {w2}"
        if (phrase not in seen and
                (w1 in q_lower or w2 in q_lower or
                 w1 in _FINANCIAL_SIGNALS or w2 in _FINANCIAL_SIGNALS)):
            seen.add(phrase)
            phrases.append(phrase)
    return phrases[:20]


def _score_relevance(text: str, query_terms: list[str]) -> float:
    """Score 0-1: how many query terms appear in text + financial signal bonus."""
    if not text or not query_terms:
        return 0.0
    lower = text.lower()
    hits = sum(1 for t in query_terms if t.lower() in lower)
    base = hits / max(len(query_terms), 1)
    fin_hits = sum(1 for kw in _FINANCIAL_SIGNALS if kw in lower)
    return min(1.0, base + min(0.3, fin_hits * 0.03))


def _truncate(text: str, max_chars: int = 500) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"


def _make_helpers() -> dict:
    """Pack shared utility functions into a dict for passing to crawlers."""
    return {
        "rate_limited_get": _rate_limited_get,
        "extract_text": _extract_text,
        "extract_tickers": _extract_tickers,
        "extract_key_phrases": _extract_key_phrases,
        "score_relevance": _score_relevance,
        "truncate": _truncate,
    }


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

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
    if len(_RESULT_CACHE) > 500:
        oldest = min(_RESULT_CACHE, key=lambda k: _RESULT_CACHE[k][1])
        del _RESULT_CACHE[oldest]


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class WebInvestigator:
    """Web investigation engine.

    When MIDGE notices something interesting (insider buying cluster,
    unusual convergence, cross-market anomaly), she calls this engine
    to crawl the open web and find out WHY — following the rabbit hole.

    Usage:
        investigator = WebInvestigator(raw_store=ctx.raw_store)
        findings = investigator.investigate_ticker("LMT", "insider buying cluster")
        event_findings = investigator.investigate_event("crude oil inventory surprise")
        deeper = investigator.follow_rabbit_hole(findings[0], depth=2)
    """

    def __init__(self, raw_store=None) -> None:
        self._raw_store = raw_store

    def investigate_ticker(self, ticker: str, context: str = "") -> list[WebFinding]:
        """When MIDGE notices something about a ticker, investigate WHY.

        Crawls Yahoo Finance news, Google News RSS, SEC EDGAR, and Reddit.

        Args:
            ticker: Ticker symbol (e.g. "AAPL", "LMT").
            context: Optional context phrase added to search queries
                     (e.g. "insider buying cluster").

        Returns:
            List of WebFinding objects sorted by relevance descending (top 20).
        """
        from .web_investigator_crawlers import (
            _crawl_google_news_rss,
            _crawl_yahoo_news,
            _crawl_sec_edgar_efts,
            _crawl_reddit,
        )

        ticker = ticker.upper().strip()
        cache_key = _cache_key("ticker", f"{ticker}:{context}")
        cached = _from_cache(cache_key)
        if cached is not None:
            return cached

        logger.info("WebInvestigator: investigating ticker=%s context=%r", ticker, context)

        query_terms = [ticker] + (context.lower().split() if context else [])
        query = f"{ticker} {context}".strip() if context else ticker
        helpers = _make_helpers()
        all_findings: list[WebFinding] = []

        for crawl_fn, args, label in [
            (_crawl_yahoo_news,     (ticker, query_terms, helpers),            "yahoo_news"),
            (_crawl_google_news_rss, (query, query_terms, helpers),            "google_news"),
            (_crawl_sec_edgar_efts, (ticker, query_terms, helpers),            "sec_edgar"),
            (_crawl_reddit,         (f"{ticker} stock {context}".strip(), query_terms, helpers), "reddit"),
        ]:
            try:
                found = crawl_fn(*args)
                for f in found:
                    f.query_ticker = ticker
                all_findings.extend(found)
            except Exception as exc:
                logger.debug("WebInvestigator: %s failed for %s: %s", label, ticker, exc)

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

        Crawls Google News RSS, SEC EDGAR, and Federal Register.

        Args:
            event_description: Human-readable event (e.g. "EIA crude oil
                               inventory surprise", "congressional defense cluster").
            tickers: Optional affected tickers for per-ticker context queries.

        Returns:
            List of WebFinding objects sorted by relevance descending (top 20).
        """
        from .web_investigator_crawlers import (
            _crawl_google_news_rss,
            _crawl_sec_edgar_efts,
            _crawl_federal_register,
        )

        cache_key = _cache_key("event", event_description)
        cached = _from_cache(cache_key)
        if cached is not None:
            return cached

        logger.info("WebInvestigator: investigating event=%r", event_description)

        query_terms = event_description.lower().split()
        helpers = _make_helpers()
        all_findings: list[WebFinding] = []

        for crawl_fn, args, label in [
            (_crawl_google_news_rss,   (event_description, query_terms, helpers),    "google_news"),
            (_crawl_sec_edgar_efts,    (event_description, query_terms, helpers),    "sec_edgar"),
            (_crawl_federal_register,  (event_description, query_terms, helpers),    "federal_register"),
        ]:
            try:
                all_findings.extend(crawl_fn(*args))
            except Exception as exc:
                logger.debug("WebInvestigator: %s failed for event: %s", label, exc)

        # Per-ticker context (capped to 3 tickers to respect rate limits)
        if tickers:
            for ticker in tickers[:3]:
                try:
                    found = _crawl_google_news_rss(
                        f"{ticker} {event_description}", query_terms + [ticker], helpers
                    )
                    for f in found:
                        f.query_ticker = ticker
                    all_findings.extend(found)
                except Exception as exc:
                    logger.debug(
                        "WebInvestigator: google_news ticker=%s event failed: %s", ticker, exc
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

        Extracts tickers and key phrases from the finding, searches for those.
        Limited to `depth` levels to prevent infinite crawling.

        Args:
            finding: The WebFinding to follow.
            depth: Maximum rabbit-hole depth. Finding's current depth must be
                   less than this to continue.

        Returns:
            List of deeper WebFinding objects (top 10 by relevance).
        """
        if finding.depth >= depth:
            return []

        new_depth = finding.depth + 1
        all_deeper: list[WebFinding] = []

        # Follow each mentioned ticker (skip the query ticker and common words)
        for ticker in finding.tickers_mentioned[:5]:
            if ticker == finding.query_ticker or ticker in _COMMON_WORDS:
                continue
            cache_key = _cache_key(f"rabbit_{new_depth}", ticker)
            cached = _from_cache(cache_key)
            if cached is not None:
                all_deeper.extend(cached)
                continue
            try:
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

        # Follow key phrases as event searches (first 2 only, to respect rate limits)
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
        """Append findings to data/midge/investigations.jsonl."""
        if not findings:
            return
        import json as _json
        from pathlib import Path
        out_path = Path("data/midge/investigations.jsonl")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(out_path, "a", encoding="utf-8") as fh:
                for f in findings:
                    rec = f.to_dict()
                    if ticker:
                        rec["triggered_by_ticker"] = ticker
                    if context:
                        rec["triggered_by_context"] = context
                    if topic:
                        rec["triggered_by_topic"] = topic
                    fh.write(_json.dumps(rec) + "\n")
        except Exception as exc:
            logger.debug("WebInvestigator: JSONL write failed: %s", exc)

        if self._raw_store is not None:
            try:
                fn = getattr(self._raw_store, "store_web_investigation", None)
                if fn is not None:
                    fn(ticker or topic, [f.to_dict() for f in findings])
            except Exception as exc:
                logger.debug("WebInvestigator: raw_store write failed: %s", exc)
