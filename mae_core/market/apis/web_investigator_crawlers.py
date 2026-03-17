"""
web_investigator_crawlers.py — Source-specific crawl functions for WebInvestigator.

Extracted from web_investigator.py to keep each file under the 500-line cap.
Each function is a standalone crawler that:
  1. Fetches one free web source
  2. Returns a list of WebFinding objects
  3. Never raises — all exceptions are logged and empty list returned

Sources:
  _crawl_google_news_rss()    — Google News RSS feed (query-based)
  _crawl_yahoo_news()         — Yahoo Finance news page (per-ticker)
  _crawl_sec_edgar_efts()     — SEC EDGAR full-text search API
  _crawl_reddit()             — Reddit search JSON API
  _crawl_federal_register()   — Federal Register free JSON API

All functions depend on:
  - _rate_limited_get()       from web_investigator (shared HTTP helper)
  - _extract_text()           from web_investigator (trafilatura wrapper)
  - _extract_tickers()        from web_investigator
  - _extract_key_phrases()    from web_investigator
  - _score_relevance()        from web_investigator
  - WebFinding                from web_investigator
"""

from __future__ import annotations

import logging
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .web_investigator import WebFinding

logger = logging.getLogger(__name__)


def _crawl_google_news_rss(
    query: str,
    query_terms: list[str],
    _helpers: dict,
) -> list:
    """Google News RSS: recent headlines mentioning the query.

    Returns up to 10 WebFinding objects, one per headline.
    Title is used as text summary (no article fetch — saves rate-limit budget).
    """
    from .web_investigator import WebFinding

    _rate_limited_get = _helpers["rate_limited_get"]
    _score_relevance = _helpers["score_relevance"]
    _extract_key_phrases = _helpers["extract_key_phrases"]
    _extract_tickers = _helpers["extract_tickers"]
    _truncate = _helpers["truncate"]

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
        for item in channel.findall("item")[:10]:
            title_el = item.find("title")
            link_el = item.find("link")
            pub_el = item.find("pubDate")

            title = (title_el.text or "").strip() if title_el is not None else ""
            link = (link_el.text or "").strip() if link_el is not None else ""
            published = (pub_el.text or "").strip() if pub_el is not None else ""

            if not title:
                continue

            relevance = _score_relevance(title, query_terms)
            findings.append(WebFinding(
                url=link,
                title=title,
                source="google_news",
                text_summary=_truncate(title, 500),
                relevance=relevance,
                published=published,
                tickers_mentioned=_extract_tickers(title),
                key_phrases=_extract_key_phrases(title, query_terms),
                query_topic=query,
            ))
    except ET.ParseError as exc:
        logger.debug("google_news RSS parse error for %r: %s", query, exc)

    return findings


def _crawl_yahoo_news(
    ticker: str,
    query_terms: list[str],
    _helpers: dict,
) -> list:
    """Yahoo Finance news page: fetch HTML, extract clean text.

    Returns a single WebFinding with the extracted article text.
    Uses trafilatura for boilerplate removal.
    """
    from .web_investigator import WebFinding

    _rate_limited_get = _helpers["rate_limited_get"]
    _extract_text = _helpers["extract_text"]
    _score_relevance = _helpers["score_relevance"]
    _extract_key_phrases = _helpers["extract_key_phrases"]
    _extract_tickers = _helpers["extract_tickers"]
    _truncate = _helpers["truncate"]

    findings = []
    url = f"https://finance.yahoo.com/quote/{ticker}/news"

    html = _rate_limited_get(url)
    if not html:
        return findings

    text = _extract_text(html, url)
    if not text:
        return findings

    lines = [ln.strip() for ln in text.split("\n") if len(ln.strip()) > 20]
    title = lines[0][:200] if lines else f"{ticker} news"
    tickers = [t for t in _extract_tickers(text) if t != ticker]

    findings.append(WebFinding(
        url=url,
        title=title,
        source="yahoo_news",
        text_summary=_truncate(text, 500),
        relevance=_score_relevance(text, query_terms),
        published="",
        tickers_mentioned=tickers,
        key_phrases=_extract_key_phrases(text, query_terms),
        query_ticker=ticker,
        query_topic=ticker,
    ))
    return findings


def _crawl_sec_edgar_efts(
    query: str,
    query_terms: list[str],
    _helpers: dict,
) -> list:
    """SEC EDGAR full-text search (EFTS): recent filings mentioning the query.

    Uses the public EFTS JSON search endpoint — no API key required.
    Returns up to 5 most recent filing summaries (90-day window).
    """
    import json as _json
    from .web_investigator import WebFinding

    _rate_limited_get = _helpers["rate_limited_get"]
    _score_relevance = _helpers["score_relevance"]
    _extract_key_phrases = _helpers["extract_key_phrases"]
    _extract_tickers = _helpers["extract_tickers"]
    _truncate = _helpers["truncate"]

    findings = []
    start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    encoded = urllib.parse.quote_plus(query)
    api_url = (
        f"https://efts.sec.gov/LATEST/search-index?q=%22{encoded}%22"
        f"&dateRange=custom&startdt={start_date}"
    )

    text = _rate_limited_get(api_url, headers={"Accept": "application/json"})
    if not text:
        return findings

    try:
        data = _json.loads(text)
        hits = data.get("hits", {}).get("hits", [])[:5]
        for hit in hits:
            src = hit.get("_source", {})
            entity = src.get("entity_name", "Unknown entity")
            form_type = src.get("form_type", "filing")
            file_date = src.get("file_date", "")
            period = src.get("period_of_report", "")
            accession = hit.get("_id", "").replace(":", "/")
            filing_url = (
                f"https://www.sec.gov/Archives/edgar/data/{accession}"
                if accession else api_url
            )
            description = (
                f"{entity} filed {form_type}"
                + (f" (period: {period})" if period else "")
                + f" — query match: {query}"
            )
            findings.append(WebFinding(
                url=filing_url,
                title=f"SEC {form_type}: {entity}"[:200],
                source="sec_edgar",
                text_summary=_truncate(description, 500),
                relevance=_score_relevance(description, query_terms),
                published=file_date,
                tickers_mentioned=_extract_tickers(entity),
                key_phrases=_extract_key_phrases(description, query_terms),
                query_topic=query,
            ))
    except Exception as exc:
        logger.debug("SEC EFTS parse error for %r: %s", query, exc)

    return findings


def _crawl_reddit(
    query: str,
    query_terms: list[str],
    _helpers: dict,
) -> list:
    """Reddit search JSON API: recent community discussion about a query.

    Uses the public Reddit search endpoint (no OAuth required for read-only).
    Returns up to 8 posts, filtered for minimum engagement (score >= 1 or
    at least 1 comment).
    """
    import json as _json
    from .web_investigator import WebFinding

    _rate_limited_get = _helpers["rate_limited_get"]
    _score_relevance = _helpers["score_relevance"]
    _extract_key_phrases = _helpers["extract_key_phrases"]
    _extract_tickers = _helpers["extract_tickers"]
    _truncate = _helpers["truncate"]

    findings = []
    encoded = urllib.parse.quote_plus(query)
    url = f"https://www.reddit.com/search.json?q={encoded}&sort=new&limit=10&type=link"

    text = _rate_limited_get(
        url,
        headers={"User-Agent": "MIDGE:market-research-bot:v1.0 (by /u/midge_research)"},
    )
    if not text:
        return findings

    try:
        data = _json.loads(text)
        posts = data.get("data", {}).get("children", [])[:8]
        for post in posts:
            pdata = post.get("data", {})
            title = pdata.get("title", "").strip()
            if not title:
                continue
            score = pdata.get("score", 0)
            num_comments = pdata.get("num_comments", 0)
            if score < 1 and num_comments < 1:
                continue  # Skip zero-engagement posts (likely spam)

            permalink = pdata.get("permalink", "")
            subreddit = pdata.get("subreddit", "")
            selftext = pdata.get("selftext", "")[:300]
            created_utc = pdata.get("created_utc", 0)

            full_text = f"{title} {selftext}".strip()
            published = (
                datetime.fromtimestamp(created_utc, tz=timezone.utc).isoformat(timespec="seconds")
                if created_utc else ""
            )
            context = f"r/{subreddit} • {score} pts • {num_comments} comments"

            findings.append(WebFinding(
                url=f"https://www.reddit.com{permalink}" if permalink else "",
                title=title[:200],
                source="reddit",
                text_summary=_truncate(f"{context}: {full_text}", 500),
                relevance=_score_relevance(full_text, query_terms),
                published=published,
                tickers_mentioned=_extract_tickers(full_text),
                key_phrases=_extract_key_phrases(full_text, query_terms),
                query_topic=query,
            ))
    except Exception as exc:
        logger.debug("Reddit JSON parse error for %r: %s", query, exc)

    return findings


def _crawl_federal_register(
    query: str,
    query_terms: list[str],
    _helpers: dict,
) -> list:
    """Federal Register API: recent regulatory actions matching a keyword.

    Free JSON API — no key required. Returns up to 5 most recent documents
    that mention the query term.
    """
    import json as _json
    from .web_investigator import WebFinding

    _rate_limited_get = _helpers["rate_limited_get"]
    _score_relevance = _helpers["score_relevance"]
    _extract_key_phrases = _helpers["extract_key_phrases"]
    _extract_tickers = _helpers["extract_tickers"]
    _truncate = _helpers["truncate"]

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
        data = _json.loads(text)
        for doc in data.get("results", [])[:5]:
            title = doc.get("title", "").strip()
            if not title:
                continue
            abstract = doc.get("abstract", "").strip()
            agencies = doc.get("agencies", [])
            agency_names = ", ".join(
                a.get("name", "") for a in agencies[:3] if isinstance(a, dict)
            )
            full_text = f"{title} {abstract} {agency_names}"

            findings.append(WebFinding(
                url=doc.get("html_url", ""),
                title=title[:200],
                source="federal_register",
                text_summary=_truncate(f"Agency: {agency_names}. {abstract}", 500),
                relevance=_score_relevance(full_text, query_terms),
                published=doc.get("publication_date", ""),
                tickers_mentioned=_extract_tickers(title),
                key_phrases=_extract_key_phrases(full_text, query_terms),
                query_topic=query,
            ))
    except Exception as exc:
        logger.debug("Federal Register parse error for %r: %s", query, exc)

    return findings
