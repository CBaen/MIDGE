#!/usr/bin/env python3
"""
usa_spending.py - USASpending.gov Government Contract Scraper

Scrapes federal contract award data from USASpending.gov API.
Free data source - no API key required.

Data Points:
- Award amount and date
- Recipient company name
- Awarding agency
- Contract description
- Period of performance

Use Case:
- Track which companies receive government contracts
- Correlate with congressional committee oversight
- Identify potential trading signals (contract award -> stock movement)
"""

import time
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import requests

# USASpending.gov API base URL
USASPENDING_API = "https://api.usaspending.gov/api/v2"

# Rate limiting (USASpending is generous but let's be respectful)
REQUEST_DELAY = 0.25


@dataclass
class GovernmentContract:
    """A federal government contract award."""
    # Recipient Info
    recipient_name: str
    recipient_duns: str  # DUNS/UEI number
    recipient_location: str

    # Award Info
    award_id: str
    award_amount: float
    award_date: str
    award_type: str  # Contract, Grant, Loan, etc.

    # Contract Details
    description: str
    naics_code: str  # Industry code
    naics_description: str

    # Agency Info
    awarding_agency: str
    awarding_sub_agency: str
    funding_agency: str

    # Period
    start_date: str
    end_date: str

    # Signal metadata for MIDGE
    signal_source: str = "contract"
    decay_rate: float = 0.02  # ~35 day half-life

    def to_plain_language(self) -> str:
        """Format for Guiding Light's dashboard."""
        return (
            f"{self.awarding_agency} awarded ${self.award_amount:,.0f} to {self.recipient_name} "
            f"for {self.description[:100]}..."
        )


class USASpendingClient:
    """
    Client for USASpending.gov API.

    API documentation: https://api.usaspending.gov/
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json"
        })
        self._last_request_time = 0

    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()

    def _post(self, endpoint: str, data: dict) -> Optional[dict]:
        """Make rate-limited POST request."""
        self._rate_limit()
        try:
            response = self.session.post(
                f"{USASPENDING_API}/{endpoint}",
                json=data,
                timeout=60
            )
            if response.status_code == 200:
                return response.json()
            else:
                print(f"USASpending API error: {response.status_code} - {response.text[:200]}")
                return None
        except Exception as e:
            print(f"USASpending request failed: {e}")
            return None

    def search_contracts(self,
                        keyword: str = None,
                        recipient_name: str = None,
                        awarding_agency: str = None,
                        start_date: str = None,
                        end_date: str = None,
                        min_amount: float = None,
                        max_amount: float = None,
                        limit: int = 100) -> List[GovernmentContract]:
        """
        Search for government contracts.

        Args:
            keyword: Search in award descriptions
            recipient_name: Filter by recipient company name
            awarding_agency: Filter by agency name
            start_date: Filter by award date (YYYY-MM-DD)
            end_date: Filter by award date (YYYY-MM-DD)
            min_amount: Minimum award amount
            max_amount: Maximum award amount
            limit: Max results to return

        Returns:
            List of GovernmentContract objects
        """
        # Build filters
        filters = {
            "award_type_codes": ["A", "B", "C", "D"],  # Contract types
        }

        if keyword:
            filters["keywords"] = [keyword]

        if recipient_name:
            filters["recipient_search_text"] = [recipient_name]

        if awarding_agency:
            filters["agencies"] = [{
                "type": "awarding",
                "tier": "toptier",
                "name": awarding_agency
            }]

        if start_date or end_date:
            filters["time_period"] = [{
                "start_date": start_date or "2020-01-01",
                "end_date": end_date or datetime.now().strftime("%Y-%m-%d")
            }]

        if min_amount is not None or max_amount is not None:
            filters["award_amounts"] = [{
                "lower_bound": min_amount or 0,
                "upper_bound": max_amount or 999999999999
            }]

        # Make the API request
        # Fields to retrieve (required parameter)
        fields = [
            "Award ID",
            "Recipient Name",
            "Start Date",
            "End Date",
            "Award Amount",
            "Awarding Agency",
            "Awarding Sub Agency",
            "Award Type",
            "Funding Agency",
            "Description",
            "NAICS Code",
            "NAICS Description",
            "Recipient State Name",
            "Recipient Country"
        ]

        data = {
            "filters": filters,
            "fields": fields,
            "limit": limit,
            "page": 1,
            "sort": "Award Amount",
            "order": "desc"
        }

        result = self._post("search/spending_by_award/", data)

        if not result:
            return []

        contracts = []
        for award in result.get("results", []):
            try:
                contract = GovernmentContract(
                    recipient_name=award.get("Recipient Name", "Unknown"),
                    recipient_duns=award.get("recipient_id", ""),
                    recipient_location=f"{award.get('Recipient State Name', '')}, {award.get('Recipient Country', '')}",
                    award_id=award.get("internal_id", "") or award.get("Award ID", ""),
                    award_amount=float(award.get("Award Amount", 0) or 0),
                    award_date=award.get("Start Date", "") or award.get("Award Date", ""),
                    award_type=award.get("Award Type", "Contract"),
                    description=award.get("Description", ""),
                    naics_code=award.get("NAICS Code", "") or "",
                    naics_description=award.get("NAICS Description", "") or "",
                    awarding_agency=award.get("Awarding Agency", "") or award.get("awarding_toptier_agency_name", ""),
                    awarding_sub_agency=award.get("Awarding Sub Agency", "") or "",
                    funding_agency=award.get("Funding Agency", "") or "",
                    start_date=award.get("Start Date", ""),
                    end_date=award.get("End Date", "")
                )
                contracts.append(contract)
            except Exception as e:
                print(f"Error parsing contract: {e}")
                continue

        return contracts

    def get_recent_large_contracts(self,
                                   days: int = 30,
                                   min_amount: float = 1000000) -> List[GovernmentContract]:
        """
        Get recent large contract awards.

        Args:
            days: Number of days to look back
            min_amount: Minimum award amount (default $1M)

        Returns:
            List of GovernmentContract objects
        """
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        return self.search_contracts(
            start_date=start_date,
            end_date=end_date,
            min_amount=min_amount,
            limit=100
        )

    def search_by_company(self, company_name: str, days: int = 365) -> List[GovernmentContract]:
        """
        Find contracts awarded to a specific company.

        Args:
            company_name: Company name to search (e.g., "Lockheed", "Boeing")
            days: Number of days to look back

        Returns:
            List of GovernmentContract objects
        """
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        return self.search_contracts(
            recipient_name=company_name,
            start_date=start_date,
            end_date=end_date,
            limit=50
        )

    def search_by_agency(self, agency_name: str, days: int = 90) -> List[GovernmentContract]:
        """
        Find contracts awarded by a specific agency.

        Args:
            agency_name: Agency name (e.g., "Department of Defense")
            days: Number of days to look back

        Returns:
            List of GovernmentContract objects
        """
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        return self.search_contracts(
            awarding_agency=agency_name,
            start_date=start_date,
            end_date=end_date,
            limit=50
        )


# Mapping of agencies to congressional oversight committees
# This is used for the politician/contract correlation
AGENCY_COMMITTEE_MAP = {
    "Department of Defense": [
        "House Armed Services Committee",
        "Senate Armed Services Committee",
        "House Appropriations Committee (Defense Subcommittee)",
        "Senate Appropriations Committee (Defense Subcommittee)"
    ],
    "Department of Health and Human Services": [
        "House Energy and Commerce Committee",
        "Senate Health, Education, Labor, and Pensions Committee",
        "House Ways and Means Committee"
    ],
    "Department of Homeland Security": [
        "House Homeland Security Committee",
        "Senate Homeland Security and Governmental Affairs Committee"
    ],
    "Department of Energy": [
        "House Energy and Commerce Committee",
        "Senate Energy and Natural Resources Committee"
    ],
    "Department of Veterans Affairs": [
        "House Veterans' Affairs Committee",
        "Senate Veterans' Affairs Committee"
    ],
    "National Aeronautics and Space Administration": [
        "House Science, Space, and Technology Committee",
        "Senate Commerce, Science, and Transportation Committee"
    ],
    "Department of Transportation": [
        "House Transportation and Infrastructure Committee",
        "Senate Commerce, Science, and Transportation Committee"
    ]
}


def get_oversight_committees(agency_name: str) -> List[str]:
    """
    Get congressional committees that oversee a given agency.

    Args:
        agency_name: Federal agency name

    Returns:
        List of committee names
    """
    # Try exact match first
    if agency_name in AGENCY_COMMITTEE_MAP:
        return AGENCY_COMMITTEE_MAP[agency_name]

    # Try partial match
    for agency, committees in AGENCY_COMMITTEE_MAP.items():
        if agency.lower() in agency_name.lower() or agency_name.lower() in agency.lower():
            return committees

    return []


def get_recent_contracts(company: str, days: int = 30) -> List[GovernmentContract]:
    """
    Convenience function to get recent contracts for a company.

    Args:
        company: Company name (partial match supported)
        days: Days to look back

    Returns:
        List of GovernmentContract objects
    """
    client = USASpendingClient()
    return client.search_by_company(company, days)


def get_large_defense_contracts(days: int = 30, min_amount: float = 10000000) -> List[GovernmentContract]:
    """
    Get recent large defense contracts (for defense sector trading signals).

    Args:
        days: Days to look back
        min_amount: Minimum award amount (default $10M)

    Returns:
        List of GovernmentContract objects
    """
    client = USASpendingClient()
    return client.search_contracts(
        awarding_agency="Department of Defense",
        start_date=(datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d"),
        end_date=datetime.now().strftime("%Y-%m-%d"),
        min_amount=min_amount,
        limit=50
    )


if __name__ == "__main__":
    print("Testing USASpending.gov API...")
    print()

    client = USASpendingClient()

    # Test 1: Recent large contracts
    print("Recent large contracts (>$10M):")
    contracts = client.get_recent_large_contracts(days=30, min_amount=10000000)
    print(f"Found {len(contracts)} contracts")

    for c in contracts[:3]:
        print(f"  {c.to_plain_language()}")
        committees = get_oversight_committees(c.awarding_agency)
        if committees:
            print(f"    Oversight: {committees[0]}")

    print()

    # Test 2: Company search
    print("Contracts for 'Lockheed':")
    lockheed = client.search_by_company("Lockheed", days=180)
    print(f"Found {len(lockheed)} contracts")

    for c in lockheed[:2]:
        print(f"  ${c.award_amount:,.0f} - {c.awarding_agency}")

    print()
    print("Done.")
