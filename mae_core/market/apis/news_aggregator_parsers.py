"""
news_aggregator_parsers.py — Parsing helpers for NewsAggregatorClient.

Contains pure functions for: RSS/Atom XML parsing, date normalisation,
sentiment detection, ticker extraction, and financial keyword tagging.
Kept separate from the client class to stay under the 500-line cap.
"""

import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import List, Optional, Set, Tuple

from .news_aggregator_constants import (
    FINANCIAL_KEYWORDS,
    NEGATIVE_KEYWORDS,
    POSITIVE_KEYWORDS,
    SP500_TICKERS,
)

logger = logging.getLogger(__name__)


def parse_date(date_str: str) -> str:
    """
    Normalise any date string to ISO 8601 UTC.

    Tries RFC 2822 (standard RSS pubDate), then ISO 8601 variants.
    Returns the input string unchanged on complete failure so the
    headline is never silently dropped for an unparseable date.
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


def detect_sentiment(title: str) -> str:
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


def extract_financial_keywords(title: str) -> List[str]:
    """Return financial domain keywords found in a headline title."""
    lower = title.lower()
    return [kw for kw in FINANCIAL_KEYWORDS if kw in lower]


def extract_tickers(title: str) -> List[str]:
    """
    Extract ticker symbols from a headline title via four strategies:

    1. Explicit $TICKER notation  — $AAPL, $TSLA
    2. Parenthetical notation     — (AAPL), (NYSE: AAPL)
    3. Colon notation             — AAPL:
    4. Standalone word match      — known S&P 500 tickers as whole words
       (capped at 3 to suppress false positives)
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


def parse_rss_xml(xml_text: str, source_name: str) -> List[Tuple[str, str, str]]:
    """
    Parse raw RSS 2.0 or Atom 1.0 XML into (title, link, pubDate) tuples.

    Returns an empty list rather than raising on malformed XML.
    Handles both feed formats. Uses explicit `is not None` checks for
    ElementTree elements to avoid the Python truthiness deprecation trap
    (ET elements evaluate their child count, not identity).
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
        link = link_el.get("href", link_el.text or "").strip() if link_el is not None else ""

        updated_el = entry.find(f"{{{ns_atom}}}updated")
        if updated_el is None:
            updated_el = entry.find("updated")
        pub = (updated_el.text or "").strip() if updated_el is not None else ""

        if title:
            results.append((title, link, pub))

    return results
