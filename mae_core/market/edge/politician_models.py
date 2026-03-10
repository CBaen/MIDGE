"""politician_models.py - Data models and member loader for PoliticianTracker.

Extracted from politician_tracker.py to keep each file under 500 lines.

Public API:
  PoliticianProfile     — dataclass for a politician + committee memberships
  CorrelationSignal     — detected correlation between insider trade and contract
  _load_congress_members() — loads profiles from congress_members.json or fallback
  KNOWN_POLITICIANS     — module-level dict populated at import time
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)

from mae_core.market.apis.usa_spending import (
    GovernmentContract,
    get_oversight_committees,
)
from mae_core.market.apis.sec_edgar import InsiderTrade


@dataclass
class PoliticianProfile:
    """A politician and their committee memberships."""
    name: str
    committees: List[str] = field(default_factory=list)
    role: str = ""  # e.g., "Chair", "Member", "Ranking Member"
    party: str = ""
    state: str = ""

    def oversees_agency(self, agency_name: str) -> bool:
        """Check if this politician's committees oversee a given agency."""
        agency_committees = get_oversight_committees(agency_name)
        return any(
            self._committee_match(c, ac)
            for c in self.committees
            for ac in agency_committees
        )

    def _committee_match(self, committee1: str, committee2: str) -> bool:
        """Fuzzy match committee names."""
        c1 = committee1.lower().replace("committee", "").strip()
        c2 = committee2.lower().replace("committee", "").strip()
        return c1 in c2 or c2 in c1


@dataclass
class CorrelationSignal:
    """A detected correlation between insider trade and contract."""
    # The insider trade
    trade: InsiderTrade
    trader_name: str
    symbol: str
    trade_date: str
    trade_type: str  # "buy" or "sell"
    shares: int
    value: float

    # The related contract (if found)
    contract: Optional[GovernmentContract] = None
    contract_value: float = 0.0
    awarding_agency: str = ""
    contract_date: str = ""

    # Correlation details
    correlation_type: str = ""  # "politician_contract", "insider_preannouncement", etc.
    days_between: int = 0
    confidence: float = 0.0

    # Committee connection
    committee: str = ""
    oversight_match: bool = False

    def to_plain_language(self) -> str:
        """Format for Guiding Light's dashboard."""
        if self.correlation_type == "politician_contract":
            return (
                f"{self.trader_name} ({self.committee}) bought ${self.value:,.0f} of {self.symbol} "
                f"({self.trade_date}). {self.days_between} days later, "
                f"{self.awarding_agency} awarded ${self.contract_value:,.0f} contract."
            )
        elif self.correlation_type == "insider_preannouncement":
            return (
                f"Insider {self.trader_name} {self.trade_type} ${self.value:,.0f} of {self.symbol} "
                f"on {self.trade_date}. Pattern suggests upcoming announcement."
            )
        else:
            return f"{self.trader_name}: {self.trade_type} {self.symbol} (${self.value:,.0f})"


def _load_congress_members() -> Dict[str, PoliticianProfile]:
    """Load Congress members from data file, falling back to minimal set."""
    import json
    from pathlib import Path

    data_path = Path(__file__).resolve().parents[3] / "data" / "market" / "congress_members.json"

    if data_path.exists():
        try:
            with open(data_path, "r") as f:
                data = json.load(f)

            profiles = {}
            for key, member in data.get("members", {}).items():
                profiles[key] = PoliticianProfile(
                    name=member.get("name", key),
                    committees=member.get("committees", []),
                    role="Member",
                    party=member.get("party", ""),
                    state=member.get("state", ""),
                )
            logger.info("Loaded %d Congress members from %s", len(profiles), data_path.name)
            return profiles
        except Exception as e:
            logger.warning("Failed to load congress_members.json: %s", e)

    # Fallback: minimal set of frequently-flagged stock traders
    return {
        "PELOSI": PoliticianProfile(
            name="Nancy Pelosi",
            committees=["House Financial Services", "House Appropriations"],
            role="Former Speaker", party="D", state="CA"
        ),
        "TUBERVILLE": PoliticianProfile(
            name="Tommy Tuberville",
            committees=["Senate Armed Services", "Senate Agriculture"],
            role="Member", party="R", state="AL"
        ),
        "OSSOFF": PoliticianProfile(
            name="Jon Ossoff",
            committees=["Senate Judiciary", "Senate Homeland Security"],
            role="Member", party="D", state="GA"
        ),
        "SULLIVAN": PoliticianProfile(
            name="Dan Sullivan",
            committees=["Senate Armed Services", "Senate Commerce"],
            role="Member", party="R", state="AK"
        ),
    }


KNOWN_POLITICIANS: Dict[str, PoliticianProfile] = _load_congress_members()
