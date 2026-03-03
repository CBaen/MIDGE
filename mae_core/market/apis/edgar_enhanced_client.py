"""Enhanced SEC EDGAR Client — 13F institutional holdings + SC 13D activist filings.

Uses SEC EDGAR's free XBRL/JSON APIs for:
- 13F: Quarterly institutional portfolio disclosures (45-day lag)
- SC 13D: Activist investor >5% stake acquisitions (real-time filings)

Free, no API key needed. Requires User-Agent with contact info per SEC policy.
Rate limit: 10 requests/second (SEC standard).
Part of WP-F2: Always-On MIDGE Wave 3.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import requests

logger = logging.getLogger("midge.market.edgar_enhanced")

_BASE_URL = "https://efts.sec.gov/LATEST"
_EDGAR_URL = "https://www.sec.gov/cgi-bin/browse-edgar"
_RATE_LIMIT = 10  # per second


@dataclass
class InstitutionalHolding:
    """A single holding from a 13F filing."""
    filer_name: str
    filer_cik: str
    ticker: str
    company_name: str
    shares: int
    value_usd: float  # In thousands (as reported)
    filing_date: str
    period_of_report: str


@dataclass
class ActivistFiling:
    """An SC 13D filing (>5% stake acquisition)."""
    filer_name: str
    filer_cik: str
    subject_company: str
    subject_ticker: str
    filing_date: str
    form_type: str  # SC 13D, SC 13D/A
    percent_owned: float  # >5% by definition
    purpose: str  # Purpose of transaction (if extractable)


class EdgarEnhancedClient:
    def __init__(self):
        self._session = requests.Session()
        self._session.headers["User-Agent"] = "MIDGE/1.0 market-research@example.com"
        self._session.headers["Accept"] = "application/json"
        self._call_times: deque = deque(maxlen=_RATE_LIMIT)

    def _rate_limit(self):
        """SEC allows 10 req/sec."""
        now = time.time()
        if len(self._call_times) >= _RATE_LIMIT:
            oldest = self._call_times[0]
            if now - oldest < 1.0:
                time.sleep(1.0 - (now - oldest) + 0.01)
        self._call_times.append(time.time())

    def _get(self, url: str, params: dict = None) -> dict:
        self._rate_limit()
        try:
            resp = self._session.get(url, params=params or {}, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            logger.debug("EDGAR request failed: %s", url, exc_info=True)
            return {}
        except ValueError:
            logger.debug("EDGAR response not JSON: %s", url)
            return {}

    def get_13d_filings(self, days: int = 30) -> List[ActivistFiling]:
        """Get recent SC 13D filings (activist investors taking >5% stake).

        Uses SEC EFTS full-text search for recent 13D filings.
        """
        # Search EFTS for recent SC 13D filings
        data = self._get(f"{_BASE_URL}/search-index", {
            "q": "\"SC 13D\"",
            "dateRange": "custom",
            "startdt": "",  # Let the API handle recent
            "forms": "SC 13D,SC 13D/A",
            "from": "0",
            "size": "20",
        })

        if not data:
            # Fallback: try the simpler EFTS search
            data = self._get("https://efts.sec.gov/LATEST/search-index", {
                "q": "\"SC 13D\"",
                "forms": "SC 13D",
            })

        filings = []
        hits = data.get("hits", {}).get("hits", [])
        if not hits:
            hits = data.get("hits", [])

        for hit in hits[:20]:
            source = hit.get("_source", hit)
            try:
                filings.append(ActivistFiling(
                    filer_name=source.get("display_names", [source.get("entity_name", "Unknown")])[0] if source.get("display_names") else source.get("entity_name", "Unknown"),
                    filer_cik=str(source.get("entity_id", source.get("ciks", [""])[0] if source.get("ciks") else "")),
                    subject_company=source.get("display_names", [""])[1] if len(source.get("display_names", [])) > 1 else "",
                    subject_ticker="",  # EFTS doesn't always include ticker
                    filing_date=source.get("file_date", source.get("filed", "")),
                    form_type=source.get("form_type", source.get("forms", "SC 13D")),
                    percent_owned=0.0,  # Would need to parse the actual filing
                    purpose="Activist position (>5% ownership)",
                ))
            except (TypeError, ValueError, IndexError):
                continue
        return filings

    def get_recent_13f_filers(self, days: int = 30) -> List[Dict]:
        """Get list of recent 13F filers (institutional investors).

        Returns basic filing metadata. Full holdings parsing requires
        individual filing downloads (expensive — use sparingly).
        """
        data = self._get(f"{_BASE_URL}/search-index", {
            "forms": "13F-HR",
            "from": "0",
            "size": "20",
        })

        filers = []
        hits = data.get("hits", {}).get("hits", [])
        if not hits:
            hits = data.get("hits", [])

        for hit in hits[:20]:
            source = hit.get("_source", hit)
            try:
                filers.append({
                    "filer_name": source.get("display_names", [source.get("entity_name", "")])[0] if source.get("display_names") else source.get("entity_name", ""),
                    "cik": str(source.get("entity_id", "")),
                    "filing_date": source.get("file_date", source.get("filed", "")),
                    "form_type": source.get("form_type", "13F-HR"),
                    "period_of_report": source.get("period_of_report", ""),
                })
            except (TypeError, ValueError, IndexError):
                continue
        return filers

    def search_filings(self, query: str, forms: str = "", limit: int = 10) -> List[Dict]:
        """General EDGAR EFTS search."""
        params = {"q": query, "from": "0", "size": str(limit)}
        if forms:
            params["forms"] = forms
        data = self._get(f"{_BASE_URL}/search-index", params)
        results = []
        hits = data.get("hits", {}).get("hits", [])
        if not hits:
            hits = data.get("hits", [])
        for hit in hits[:limit]:
            source = hit.get("_source", hit)
            results.append(source)
        return results
