#!/usr/bin/env python3
"""
job_tracker.py - Detect hiring blitzes that precede contract wins

When a company starts hiring aggressively BEFORE a contract announcement,
they likely know they're winning. This is a leading indicator.

Data sources (RapidAPI):
- active-jobs-db: Real-time jobs posted in last 1h/24h
- jsearch: Job search by company
- glassdoor-real-time: Company jobs via Glassdoor

Requires RAPIDAPI_KEY environment variable.
"""

import os
import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import requests

logger = logging.getLogger(__name__)


# RapidAPI endpoints
RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY", "")

# Active Jobs DB - real-time job postings
ACTIVE_JOBS_HOST = "active-jobs-db.p.rapidapi.com"
ACTIVE_JOBS_BASE = f"https://{ACTIVE_JOBS_HOST}"

# JSearch - job search by company
JSEARCH_HOST = "jsearch.p.rapidapi.com"
JSEARCH_BASE = f"https://{JSEARCH_HOST}"

# Glassdoor - company jobs
GLASSDOOR_HOST = "glassdoor-real-time.p.rapidapi.com"
GLASSDOOR_BASE = f"https://{GLASSDOOR_HOST}"

# Rate limiting
REQUEST_DELAY = 1.0


@dataclass
class HiringSignal:
    """Signal derived from job posting activity."""
    company_name: str
    ticker: Optional[str] = None  # If publicly traded

    # Job metrics
    jobs_24h: int = 0  # Jobs posted in last 24 hours
    jobs_7d: int = 0   # Jobs posted in last 7 days
    jobs_30d: int = 0  # Jobs posted in last 30 days

    # Spike detection
    is_spike: bool = False
    spike_ratio: float = 0.0  # Current vs historical average

    # Job categories (for contract correlation)
    engineering_jobs: int = 0
    cleared_jobs: int = 0  # Security clearance required
    contract_related_jobs: int = 0  # Mentions "contract", "government", etc.

    # Signal metadata
    signal_source: str = "hiring_tracker"
    confidence: float = 0.5
    detected_at: str = ""
    decay_rate: float = 0.015  # ~46 day half-life (hiring leads contract by 60-120 days)

    def __post_init__(self):
        if not self.detected_at:
            self.detected_at = datetime.now().isoformat()

    def to_plain_language(self) -> str:
        """Format for dashboard."""
        spike_note = f" (SPIKE: {self.spike_ratio:.1f}x normal)" if self.is_spike else ""
        return (
            f"{self.company_name}: {self.jobs_24h} jobs in 24h, "
            f"{self.jobs_7d} in 7d{spike_note}. "
            f"Cleared jobs: {self.cleared_jobs}. "
            f"Confidence: {self.confidence:.0%}"
        )


class JobTracker:
    """
    Tracks job postings to detect hiring blitzes.

    A "hiring blitz" is when a company's job postings spike significantly
    above their historical average - often a sign they've won or expect
    to win a major contract.
    """

    def __init__(self, rapidapi_key: Optional[str] = None, provider=None, raw_store=None):
        self._provider = provider
        self._raw_store = raw_store
        self.session = requests.Session()
        self.rapidapi_key = rapidapi_key or RAPIDAPI_KEY
        self._last_request_time = 0

        if not self.rapidapi_key:
            logger.warning("No RAPIDAPI_KEY set. Job tracking will not work.")

    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()

    def _get_headers(self, host: str) -> Dict[str, str]:
        """Get RapidAPI headers for a specific host."""
        return {
            "x-rapidapi-host": host,
            "x-rapidapi-key": self.rapidapi_key
        }

    def _request(self, url: str, headers: Optional[Dict] = None,
                 params: Optional[Dict] = None) -> Optional[dict]:
        """Make a GET request through provider or session. Returns parsed JSON or None."""
        if self._provider is not None:
            from mae_core.market.apis.market_data_provider import market_request
            from mae_core.external.api_client import ApiResponseStatus
            resp = market_request(
                self._provider, url, headers=headers,
                params=params, source_name="rapidapi", timeout_ms=30000.0,
            )
            if resp.status == ApiResponseStatus.SUCCESS:
                return resp.payload
            return None

        try:
            response = self.session.get(url, headers=headers, params=params, timeout=30)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None

    def get_recent_jobs_by_company(
        self,
        company_name: str,
        country: str = "us"
    ) -> List[dict]:
        """
        Search for recent job postings by company name.

        Uses JSearch API.
        """
        if not self.rapidapi_key:
            return []

        self._rate_limit()

        data = self._request(
            f"{JSEARCH_BASE}/search",
            headers=self._get_headers(JSEARCH_HOST),
            params={
                "query": f"{company_name} jobs",
                "page": 1,
                "num_pages": 1,
                "country": country,
                "date_posted": "week",
            },
        )
        if data is not None:
            return data.get("data", [])
        return []

    def get_jobs_last_24h(
        self,
        title_filter: Optional[str] = None,
        location_filter: str = "United States"
    ) -> List[dict]:
        """
        Get jobs posted in the last 24 hours.

        Uses Active Jobs DB API.
        """
        if not self.rapidapi_key:
            return []

        self._rate_limit()

        try:
            params = {
                "limit": 100,
                "offset": 0,
                "location_filter": f'"{location_filter}"',
                "description_type": "text"
            }

            if title_filter:
                params["title_filter"] = f'"{title_filter}"'

            data = self._request(
                f"{ACTIVE_JOBS_BASE}/active-ats-24h",
                headers=self._get_headers(ACTIVE_JOBS_HOST),
                params=params,
            )
            if data is not None:
                return data if isinstance(data, list) else data.get("data", [])
            return []

        except Exception as e:
            logger.warning(f"Active Jobs DB error: {str(e)[:50]}")
            return []

    def get_glassdoor_company_jobs(
        self,
        company_id: str,
        max_age_days: int = 30
    ) -> List[dict]:
        """
        Get job postings for a specific company via Glassdoor.

        Note: Requires Glassdoor company_id (not ticker).
        """
        if not self.rapidapi_key:
            return []

        self._rate_limit()

        try:
            data = self._request(
                f"{GLASSDOOR_BASE}/company-jobs",
                headers=self._get_headers(GLASSDOOR_HOST),
                params={
                    "company_id": company_id,
                    "page": 1,
                    "sort": "MOST_RELEVANT",
                    "max_age_days": max_age_days,
                    "domain": "www.glassdoor.com",
                },
            )
            if data is not None:
                return data.get("jobs", data.get("data", []))
            return []

        except Exception as e:
            logger.warning(f"Glassdoor error: {str(e)[:50]}")
            return []

    def analyze_hiring_activity(
        self,
        company_name: str,
        ticker: Optional[str] = None
    ) -> HiringSignal:
        """
        Analyze a company's hiring activity for blitz signals.

        Args:
            company_name: Company name to search
            ticker: Stock ticker if publicly traded

        Returns:
            HiringSignal with analysis results
        """
        signal = HiringSignal(
            company_name=company_name,
            ticker=ticker
        )

        # Get recent job postings
        jobs = self.get_recent_jobs_by_company(company_name)

        if not jobs:
            return signal

        # Analyze jobs
        now = datetime.now()

        for job in jobs:
            # Parse job date
            posted_at = job.get("job_posted_at_datetime_utc", "")
            if posted_at:
                try:
                    job_date = datetime.fromisoformat(posted_at.replace("Z", ""))
                    days_ago = (now - job_date).days

                    if days_ago <= 1:
                        signal.jobs_24h += 1
                    if days_ago <= 7:
                        signal.jobs_7d += 1
                    if days_ago <= 30:
                        signal.jobs_30d += 1
                except Exception:
                    signal.jobs_7d += 1  # Assume within week if can't parse

            # Check job content
            title = job.get("job_title", "").lower()
            description = job.get("job_description", "").lower()

            # Engineering/technical roles
            if any(kw in title for kw in ["engineer", "developer", "architect", "scientist"]):
                signal.engineering_jobs += 1

            # Security clearance (government contract indicator)
            if any(kw in description for kw in ["clearance", "secret", "ts/sci", "classified"]):
                signal.cleared_jobs += 1

            # Contract-related keywords
            if any(kw in description for kw in ["government", "federal", "contract", "dod", "defense"]):
                signal.contract_related_jobs += 1

        # Detect spike
        # Rule: 24h jobs > 10% of 7d average per day = spike
        if signal.jobs_7d > 0:
            daily_avg = signal.jobs_7d / 7
            if daily_avg > 0:
                signal.spike_ratio = signal.jobs_24h / daily_avg
                signal.is_spike = signal.spike_ratio >= 2.0  # 2x normal = spike

        # Calculate confidence
        base_confidence = 0.5
        if signal.is_spike:
            base_confidence += 0.15
        if signal.cleared_jobs > 0:
            base_confidence += 0.10
        if signal.contract_related_jobs > 3:
            base_confidence += 0.10
        signal.confidence = min(0.90, base_confidence)

        # Persist raw posting signal before returning — store ALL, not just spikes
        if self._raw_store is not None and signal.jobs_7d > 0:
            try:
                self._raw_store.store_job_postings([signal])
            except Exception:
                pass  # Never let raw storage failure block signal processing

        return signal

    def detect_hiring_blitz(
        self,
        companies: List[str],
        spike_threshold: float = 2.0
    ) -> List[HiringSignal]:
        """
        Scan multiple companies for hiring blitzes.

        Args:
            companies: List of company names to check
            spike_threshold: Minimum spike ratio to flag

        Returns:
            List of HiringSignal objects for companies with spikes
        """
        blitzes = []

        for company in companies:
            signal = self.analyze_hiring_activity(company)
            if signal.is_spike and signal.spike_ratio >= spike_threshold:
                blitzes.append(signal)

        # Sort by spike ratio (highest first)
        blitzes.sort(key=lambda s: s.spike_ratio, reverse=True)
        if self._raw_store is not None and blitzes:
            try:
                self._raw_store.store_job_postings(blitzes)
            except Exception:
                pass
        return blitzes


# Defense contractor tickers for quick reference
DEFENSE_CONTRACTORS = {
    "Lockheed Martin": "LMT",
    "Raytheon": "RTX",
    "Northrop Grumman": "NOC",
    "General Dynamics": "GD",
    "Boeing": "BA",
    "L3Harris": "LHX",
    "BAE Systems": "BAESY",
    "Leidos": "LDOS",
    "SAIC": "SAIC",
    "Booz Allen Hamilton": "BAH",
}


def detect_defense_hiring_blitz() -> List[HiringSignal]:
    """
    Convenience function to scan defense contractors for hiring blitzes.

    Returns:
        List of HiringSignal objects for contractors with spikes
    """
    tracker = JobTracker()

    signals = []
    for company, ticker in DEFENSE_CONTRACTORS.items():
        signal = tracker.analyze_hiring_activity(company, ticker)
        signals.append(signal)

    # Return only spikes, sorted by ratio
    blitzes = [s for s in signals if s.is_spike]
    blitzes.sort(key=lambda s: s.spike_ratio, reverse=True)
    return blitzes


if __name__ == "__main__":
    import sys

    print("Job Tracker - Hiring Blitz Detection")
    print()

    tracker = JobTracker()

    if not tracker.rapidapi_key:
        print("ERROR: Set RAPIDAPI_KEY environment variable")
        print("  export RAPIDAPI_KEY='your-key-here'")
        sys.exit(1)

    # Test with a specific company or default to defense contractors
    if len(sys.argv) > 1:
        company = " ".join(sys.argv[1:])
        print(f"Analyzing: {company}")
        signal = tracker.analyze_hiring_activity(company)
        print(f"  {signal.to_plain_language()}")
    else:
        print("Scanning defense contractors for hiring blitzes...")
        for company, ticker in list(DEFENSE_CONTRACTORS.items())[:3]:  # Test first 3
            print(f"\n{company} ({ticker}):")
            signal = tracker.analyze_hiring_activity(company, ticker)
            print(f"  {signal.to_plain_language()}")

    print("\nDone.")
