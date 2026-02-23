#!/usr/bin/env python3
"""
sam_gov.py - Track federal contract opportunities from SAM.gov

SAM.gov (System for Award Management) is where federal contracts are posted.
Companies bidding on contracts can be tracked BEFORE awards are announced.

The pattern:
1. Contract solicitation posted on SAM.gov
2. Multiple companies submit bids
3. One company starts hiring blitz (they know they're winning)
4. Contract award announced
5. Stock moves

We want to detect step 3 BEFORE step 4.

API: api.sam.gov (free, requires registration at sam.gov)
Set SAM_GOV_API_KEY environment variable.

Note: If you don't have a SAM.gov API key yet:
1. Register at sam.gov (free)
2. Request API access in your account settings
3. Set SAM_GOV_API_KEY environment variable
"""

import os
import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import requests

logger = logging.getLogger(__name__)


# SAM.gov API
SAM_GOV_API_KEY = os.environ.get("SAM_GOV_API_KEY", "")
SAM_GOV_BASE = "https://api.sam.gov/opportunities/v2"

# Rate limiting
REQUEST_DELAY = 1.0


@dataclass
class ContractOpportunity:
    """A federal contract opportunity from SAM.gov."""
    # Identification
    notice_id: str
    title: str
    solicitation_number: str = ""

    # Agency info
    department: str = ""  # e.g., "Department of Defense"
    agency: str = ""      # e.g., "Army"
    office: str = ""      # e.g., "Army Contracting Command"

    # Contract details
    naics_code: str = ""  # Industry classification
    set_aside: str = ""   # Small business set-asides
    type_of_contract: str = ""
    place_of_performance: str = ""

    # Value
    estimated_value: float = 0.0
    award_date: str = ""

    # Dates
    posted_date: str = ""
    response_deadline: str = ""

    # Status
    active: bool = True
    contract_type: str = ""  # "Solicitation", "Award Notice", etc.

    # Links
    url: str = ""

    # Signal metadata
    signal_source: str = "sam_gov"

    def to_plain_language(self) -> str:
        """Format for dashboard."""
        value_str = f"${self.estimated_value:,.0f}" if self.estimated_value else "TBD"
        return (
            f"{self.title[:60]}... ({self.agency}). "
            f"Value: {value_str}. "
            f"Deadline: {self.response_deadline}"
        )


@dataclass
class ContractAward:
    """A contract award from SAM.gov or USASpending."""
    # Identification
    award_id: str
    title: str
    contract_number: str = ""

    # Parties
    awarding_agency: str = ""
    recipient_name: str = ""  # The winner
    recipient_ticker: Optional[str] = None

    # Value
    award_amount: float = 0.0
    potential_value: float = 0.0  # Including options

    # Dates
    award_date: str = ""
    start_date: str = ""
    end_date: str = ""

    # Classification
    naics_code: str = ""
    product_service_code: str = ""

    # Signal metadata
    signal_source: str = "contract_award"

    def to_plain_language(self) -> str:
        """Format for dashboard."""
        return (
            f"{self.recipient_name} won ${self.award_amount:,.0f} contract "
            f"from {self.awarding_agency}: {self.title[:50]}..."
        )


class SAMGovClient:
    """
    Client for SAM.gov Opportunities API.

    Tracks federal contract solicitations and awards.
    """

    def __init__(self, api_key: Optional[str] = None, provider=None):
        self._provider = provider
        self.session = requests.Session()
        self.api_key = api_key or SAM_GOV_API_KEY
        self._last_request_time = 0

        if not self.api_key:
            logger.warning("No SAM_GOV_API_KEY set. Using public endpoints only.")

    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()

    def search_opportunities(
        self,
        keywords: Optional[str] = None,
        naics: Optional[str] = None,
        posted_from: Optional[str] = None,
        posted_to: Optional[str] = None,
        response_deadline_from: Optional[str] = None,
        limit: int = 100
    ) -> List[ContractOpportunity]:
        """
        Search for contract opportunities.

        Args:
            keywords: Search terms
            naics: NAICS code filter (e.g., "541330" for engineering)
            posted_from: Start date (MM/DD/YYYY)
            posted_to: End date (MM/DD/YYYY)
            response_deadline_from: Filter by response deadline
            limit: Max results

        Returns:
            List of ContractOpportunity objects
        """
        self._rate_limit()

        params = {
            "api_key": self.api_key,
            "limit": limit,
            "postedFrom": posted_from or (datetime.now() - timedelta(days=30)).strftime("%m/%d/%Y"),
            "postedTo": posted_to or datetime.now().strftime("%m/%d/%Y"),
        }

        if keywords:
            params["title"] = keywords
        if naics:
            params["ncode"] = naics
        if response_deadline_from:
            params["rdlfrom"] = response_deadline_from

        url = f"{SAM_GOV_BASE}/search"

        if self._provider is not None:
            from mae_core.market.apis.market_data_provider import market_request
            from mae_core.external.api_client import ApiResponseStatus
            resp = market_request(
                self._provider, url, params=params,
                source_name="sam_gov", timeout_ms=30000.0,
            )
            if resp.status == ApiResponseStatus.SUCCESS:
                data = resp.payload or {}
                opportunities = data.get("opportunitiesData", [])
                return [self._parse_opportunity(opp) for opp in opportunities]
            logger.warning("SAM.gov search failed via provider: %s", resp.error_message)
            return []

        try:
            response = self.session.get(url, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()
                opportunities = data.get("opportunitiesData", [])
                return [self._parse_opportunity(opp) for opp in opportunities]
            else:
                logger.warning(f"SAM.gov search failed ({response.status_code}): {response.text[:100]}")
                return []

        except Exception as e:
            logger.warning(f"SAM.gov error: {str(e)[:50]}")
            return []

    def _parse_opportunity(self, data: dict) -> ContractOpportunity:
        """Parse SAM.gov opportunity data into dataclass."""
        return ContractOpportunity(
            notice_id=data.get("noticeId", ""),
            title=data.get("title", ""),
            solicitation_number=data.get("solicitationNumber", ""),
            department=data.get("department", ""),
            agency=data.get("subtierAgency", data.get("agency", "")),
            office=data.get("office", ""),
            naics_code=data.get("naicsCode", ""),
            set_aside=data.get("typeOfSetAside", ""),
            type_of_contract=data.get("typeOfContract", ""),
            place_of_performance=data.get("placeOfPerformance", {}).get("state", ""),
            posted_date=data.get("postedDate", ""),
            response_deadline=data.get("responseDeadLine", ""),
            active=data.get("active", "Yes") == "Yes",
            contract_type=data.get("type", ""),
            url=data.get("uiLink", "")
        )

    def get_defense_opportunities(
        self,
        days_back: int = 30
    ) -> List[ContractOpportunity]:
        """
        Get recent Department of Defense contract opportunities.

        These are the contracts defense contractors bid on.
        """
        posted_from = (datetime.now() - timedelta(days=days_back)).strftime("%m/%d/%Y")

        opportunities = self.search_opportunities(
            posted_from=posted_from,
            limit=100
        )

        # Filter to DoD
        dod_keywords = ["defense", "army", "navy", "air force", "dod", "military"]
        dod_opps = []

        for opp in opportunities:
            combined = f"{opp.department} {opp.agency} {opp.title}".lower()
            if any(kw in combined for kw in dod_keywords):
                dod_opps.append(opp)

        return dod_opps

    def get_large_contracts(
        self,
        min_value: float = 10_000_000,
        days_back: int = 30
    ) -> List[ContractOpportunity]:
        """
        Get contract opportunities above a minimum value.

        Large contracts move stocks more.
        """
        # Note: SAM.gov doesn't always include estimated values
        # This is a best-effort filter
        opportunities = self.search_opportunities(
            posted_from=(datetime.now() - timedelta(days=days_back)).strftime("%m/%d/%Y"),
            limit=200
        )

        return [opp for opp in opportunities if opp.estimated_value >= min_value]


# NAICS codes for key industries
DEFENSE_NAICS = {
    "336411": "Aircraft Manufacturing",
    "336414": "Guided Missile and Space Vehicle Manufacturing",
    "336415": "Space Vehicle Propulsion Units",
    "541330": "Engineering Services",
    "541512": "Computer Systems Design Services",
    "541519": "Other Computer Related Services",
    "541715": "Research and Development (Physical Sciences)",
    "561210": "Facilities Support Services",
}


def get_defense_contracts(days_back: int = 30) -> List[ContractOpportunity]:
    """
    Convenience function to get recent defense contract opportunities.
    """
    client = SAMGovClient()
    return client.get_defense_opportunities(days_back)


if __name__ == "__main__":
    import sys

    print("SAM.gov Contract Tracker")
    print()

    client = SAMGovClient()

    if not client.api_key:
        print("NOTE: No SAM_GOV_API_KEY set.")
        print("Register at sam.gov and request API access.")
        print()
        print("Testing with public search...")

    # Test search
    print("Searching for defense opportunities...")
    opportunities = client.get_defense_opportunities(days_back=7)

    if opportunities:
        print(f"\nFound {len(opportunities)} defense opportunities:")
        for opp in opportunities[:5]:
            print(f"  {opp.to_plain_language()}")
    else:
        print("  No opportunities found (may need API key)")

    print("\nDone.")
