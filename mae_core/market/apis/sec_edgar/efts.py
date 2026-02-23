#!/usr/bin/env python3
"""
efts.py - SEC EDGAR Full-Text Search (EFTS) Client

Uses SEC EDGAR's full-text search system to find filings containing
high-signal keywords such as merger announcements, going-concern warnings,
and material weaknesses.

No API key required. Follows the same SEC EDGAR rate limits as client.py
(10 requests/second max, required User-Agent header with contact info).

API endpoint: https://efts.sec.gov/LATEST/search-index
"""

import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple
import requests

logger = logging.getLogger(__name__)

# EFTS endpoint (full-text search index, distinct from data.sec.gov)
EFTS_BASE_URL = "https://efts.sec.gov/LATEST/search-index"

# SEC requires a User-Agent with real contact info — matches client.py
SEC_USER_AGENT = "MIDGE Trading Research admin@example.com"

# Rate limiting — SEC policy is 10 req/sec max; 0.2s gives comfortable headroom
REQUEST_DELAY = 0.20  # 200ms between requests

# Cache duration in seconds (30 minutes)
CACHE_DURATION = 1800


@dataclass
class FilingKeywordHit:
    """
    A filing that matched a high-signal keyword in SEC EFTS full-text search.

    Keyword hits are event-driven signals — a company filing that mentions
    "merger agreement" or "going concern" has disclosed a material change.
    Fast decay (0.20) reflects that markets price these events quickly.
    """
    ticker: str          # First ticker from tickers list, or "" if none
    company_name: str    # First display_name from display_names list, or ""
    filing_date: str     # ISO date string (YYYY-MM-DD)
    form_type: str       # e.g. "8-K", "10-K"
    keyword_matched: str # Which keyword triggered this hit
    description: str     # file_description from EFTS response
    signal_source: str = "sec_efts"
    decay_rate: float = 0.20   # Fast decay — event-driven, market prices quickly
    confidence: float = 0.60   # Moderate base confidence (keyword match, not parsed)

    def to_plain_language(self) -> str:
        """Format for dashboard display."""
        label = self.ticker or self.company_name or "Unknown"
        return (
            f"[EFTS] {label} ({self.form_type}) matched '{self.keyword_matched}' "
            f"on {self.filing_date}: {self.description[:80]}"
        )


class SECFullTextSearchClient:
    """
    Client for SEC EDGAR Full-Text Search (EFTS).

    Searches the full text of SEC filings for high-signal keywords.
    Useful for catching merger announcements, going-concern disclosures,
    and other material events the moment they hit EDGAR.

    No API key required. Respects SEC rate limits.
    """

    # Bullish keywords — suggest positive catalysts
    BULLISH_KEYWORDS = [
        "tender offer",
        "merger agreement",
        "acquisition",
        "strategic alternative",
    ]

    # Bearish keywords — suggest distress or negative catalysts
    BEARISH_KEYWORDS = [
        "going concern",
        "material weakness",
        "restatement",
        "default",
        "delisted",
    ]

    def __init__(self, provider=None):
        self._provider = provider
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": SEC_USER_AGENT,
            "Accept-Encoding": "gzip, deflate",
        })
        self._last_request_time = 0

        # Cache: maps (keyword, forms, startdt) -> (timestamp, List[FilingKeywordHit])
        self._cache: Dict[Tuple, Tuple[float, List[FilingKeywordHit]]] = {}

    def _rate_limit(self) -> None:
        """Enforce SEC rate limiting (10 req/sec max)."""
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()

    def _get(self, params: dict) -> Optional[dict]:
        """
        Make a rate-limited GET request to the EFTS endpoint.

        Routes through provider if configured, otherwise uses requests.Session.
        Returns parsed JSON dict or None on failure.
        """
        self._rate_limit()

        if self._provider is not None:
            from mae_core.market.apis.market_data_provider import market_request
            from mae_core.external.api_client import ApiResponseStatus
            resp = market_request(
                self._provider, EFTS_BASE_URL,
                headers={
                    "User-Agent": SEC_USER_AGENT,
                    "Accept-Encoding": "gzip, deflate",
                },
                params=params, source_name="sec_efts", timeout_ms=30000.0,
            )
            if resp.status == ApiResponseStatus.SUCCESS:
                return resp.payload
            logger.warning("SEC EFTS error via provider: %s", resp.error_message)
            return None

        try:
            response = self.session.get(EFTS_BASE_URL, params=params, timeout=30)
            if response.status_code == 200:
                return response.json()
            logger.warning("SEC EFTS error %s for params: %s", response.status_code, params)
            return None
        except Exception as e:
            logger.error("SEC EFTS request failed: %s", e)
            return None

    def _parse_hits(self, data: dict, keyword: str) -> List[FilingKeywordHit]:
        """
        Parse the EFTS JSON response into FilingKeywordHit objects.

        EFTS returns: {"hits": {"hits": [...], "total": {...}}}
        Each hit has a "_source" object with filing metadata.
        """
        hits = []
        try:
            raw_hits = data.get("hits", {}).get("hits", [])
            for item in raw_hits:
                source = item.get("_source", {})

                tickers = source.get("tickers", [])
                ticker = tickers[0] if tickers else ""

                display_names = source.get("display_names", [])
                company_name = display_names[0] if display_names else ""

                hits.append(FilingKeywordHit(
                    ticker=ticker,
                    company_name=company_name,
                    filing_date=source.get("file_date", ""),
                    form_type=source.get("form_type", ""),
                    keyword_matched=keyword,
                    description=source.get("file_description", ""),
                ))
        except Exception as e:
            logger.error("Error parsing EFTS hits for keyword '%s': %s", keyword, e)
        return hits

    def search_filings(
        self,
        keyword: str,
        forms: str = "8-K,10-K",
        days: int = 7,
    ) -> List[FilingKeywordHit]:
        """
        Search for a keyword in recent SEC filings.

        Args:
            keyword: Search term (e.g. "going concern", "merger agreement")
            forms: Comma-separated form types to filter (e.g. "8-K,10-K")
            days: Number of days to look back from today

        Returns:
            List of FilingKeywordHit objects matching the keyword
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        startdt = start_date.strftime("%Y-%m-%d")
        enddt = end_date.strftime("%Y-%m-%d")

        cache_key = (keyword, forms, startdt)
        cached = self._cache.get(cache_key)
        if cached is not None:
            cached_time, cached_results = cached
            if time.time() - cached_time < CACHE_DURATION:
                logger.debug("EFTS cache hit for keyword '%s'", keyword)
                return cached_results

        params = {
            "q": f'"{keyword}"',
            "dateRange": "custom",
            "startdt": startdt,
            "enddt": enddt,
            "forms": forms,
        }

        logger.debug("EFTS search: keyword='%s' forms='%s' days=%d", keyword, forms, days)
        data = self._get(params)

        if data is None:
            return []

        results = self._parse_hits(data, keyword)
        logger.debug("EFTS found %d hits for '%s'", len(results), keyword)

        self._cache[cache_key] = (time.time(), results)
        return results

    def scan_all_keywords(self, days: int = 3) -> List[FilingKeywordHit]:
        """
        Search all BULLISH_KEYWORDS and BEARISH_KEYWORDS in recent filings.

        Issues one request per keyword. Results are deduplicated by
        (ticker, filing_date, keyword_matched) — a filing can appear once
        per keyword it matched.

        Args:
            days: Number of days to look back (default 3 for recency)

        Returns:
            Combined list of FilingKeywordHit objects across all keywords
        """
        all_keywords = self.BULLISH_KEYWORDS + self.BEARISH_KEYWORDS
        all_hits: List[FilingKeywordHit] = []
        seen = set()

        for keyword in all_keywords:
            hits = self.search_filings(keyword, days=days)
            for hit in hits:
                dedup_key = (hit.ticker, hit.filing_date, hit.keyword_matched)
                if dedup_key not in seen:
                    seen.add(dedup_key)
                    all_hits.append(hit)

        logger.info(
            "EFTS scan_all_keywords: %d total hits across %d keywords (last %d days)",
            len(all_hits), len(all_keywords), days,
        )
        return all_hits

    def get_merger_signals(self, days: int = 7) -> List[FilingKeywordHit]:
        """
        Convenience method: search only merger and acquisition keywords.

        Covers: "tender offer", "merger agreement", "acquisition",
        "strategic alternative".

        Args:
            days: Number of days to look back

        Returns:
            List of FilingKeywordHit objects for M&A-related filings
        """
        all_hits: List[FilingKeywordHit] = []
        seen = set()

        for keyword in self.BULLISH_KEYWORDS:
            hits = self.search_filings(keyword, days=days)
            for hit in hits:
                dedup_key = (hit.ticker, hit.filing_date, hit.keyword_matched)
                if dedup_key not in seen:
                    seen.add(dedup_key)
                    all_hits.append(hit)

        logger.info(
            "EFTS get_merger_signals: %d hits (last %d days)", len(all_hits), days
        )
        return all_hits


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    print("Testing SEC EFTS Full-Text Search Client...")
    print()

    client = SECFullTextSearchClient()

    days = int(sys.argv[1]) if len(sys.argv) > 1 else 7

    # Test single keyword
    keyword = sys.argv[2] if len(sys.argv) > 2 else "going concern"
    print(f"Searching for '{keyword}' in last {days} days...")
    hits = client.search_filings(keyword, days=days)
    if hits:
        print(f"Found {len(hits)} hits:")
        for hit in hits[:5]:
            print(f"  {hit.to_plain_language()}")
    else:
        print("  No hits found (EFTS may be unavailable or no recent filings)")

    print()

    # Test merger signals
    print(f"Fetching merger signals (last {days} days)...")
    merger_hits = client.get_merger_signals(days=days)
    if merger_hits:
        print(f"Found {len(merger_hits)} merger/acquisition signals:")
        for hit in merger_hits[:5]:
            print(f"  {hit.to_plain_language()}")
    else:
        print("  No merger signals found")

    print()

    # Test full keyword scan (short window to stay fast)
    print("Running scan_all_keywords (last 3 days)...")
    all_hits = client.scan_all_keywords(days=3)
    print(f"Total hits across all keywords: {len(all_hits)}")

    # Group by keyword for summary
    by_keyword: Dict[str, int] = {}
    for hit in all_hits:
        by_keyword[hit.keyword_matched] = by_keyword.get(hit.keyword_matched, 0) + 1
    for kw, count in sorted(by_keyword.items(), key=lambda x: -x[1]):
        print(f"  '{kw}': {count} filings")

    print()
    print("Done.")
