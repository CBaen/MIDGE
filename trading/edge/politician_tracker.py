#!/usr/bin/env python3
"""
politician_tracker.py - Correlation Engine for Insider/Contract Patterns

Connects the dots:
1. Committee member buys stock in Company X
2. Agency (overseen by that committee) awards contract to Company X
3. Signal: high confidence buy

This is the "edge" - finding patterns before they become obvious.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple

# Import our scrapers
from trading.apis.sec_edgar import SECEdgarClient, InsiderTrade, get_recent_form4s
from trading.apis.usa_spending import (
    USASpendingClient, GovernmentContract,
    get_oversight_committees, AGENCY_COMMITTEE_MAP
)


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


# Known congressional stock traders (frequently flagged in news)
# This is a subset - would need to be expanded from congress.gov data
KNOWN_POLITICIANS = {
    "PELOSI": PoliticianProfile(
        name="Nancy Pelosi",
        committees=["House Financial Services", "House Appropriations"],
        role="Former Speaker",
        party="D",
        state="CA"
    ),
    "TUBERVILLE": PoliticianProfile(
        name="Tommy Tuberville",
        committees=["Senate Armed Services", "Senate Agriculture"],
        role="Member",
        party="R",
        state="AL"
    ),
    "OSSOFF": PoliticianProfile(
        name="Jon Ossoff",
        committees=["Senate Judiciary", "Senate Homeland Security"],
        role="Member",
        party="D",
        state="GA"
    ),
    "SULLIVAN": PoliticianProfile(
        name="Dan Sullivan",
        committees=["Senate Armed Services", "Senate Commerce"],
        role="Member",
        party="R",
        state="AK"
    ),
}


class PoliticianTracker:
    """
    Correlation engine for politician trades and government contracts.

    Pattern detection:
    1. Find insider trades by known politicians (or their spouses)
    2. Find government contracts to the same companies
    3. Check if the politician sits on a committee that oversees the awarding agency
    4. Score confidence based on timing, amount, and committee relevance
    """

    def __init__(self):
        self.sec_client = SECEdgarClient()
        self.usa_client = USASpendingClient()
        self._correlation_cache: List[CorrelationSignal] = []

    def find_correlations(self,
                         symbols: List[str] = None,
                         days_lookback: int = 90,
                         min_trade_value: float = 10000) -> List[CorrelationSignal]:
        """
        Find correlations between insider trades and contracts.

        Args:
            symbols: List of stock symbols to check (None = search all recent)
            days_lookback: How far back to search
            min_trade_value: Minimum trade value to consider

        Returns:
            List of CorrelationSignal objects, sorted by confidence
        """
        correlations = []

        if symbols is None:
            # Default to major defense contractors and tech companies
            symbols = ["LMT", "RTX", "BA", "NOC", "GD", "MSFT", "AMZN", "GOOGL"]

        for symbol in symbols:
            try:
                # Get insider trades using the module function
                trades = get_recent_form4s(symbol, days=days_lookback)

                for trade in trades:
                    # Skip small trades
                    if trade.total_value < min_trade_value:
                        continue

                    # Check if this is a known politician or matches pattern
                    politician = self._identify_politician(trade.filer_name)

                    if politician:
                        # This is a politician trade - look for contract correlation
                        correlation = self._check_contract_correlation(
                            trade, politician, symbol, days_lookback
                        )
                        if correlation:
                            correlations.append(correlation)
                    else:
                        # Regular insider - check for unusual patterns
                        correlation = self._check_insider_pattern(trade, symbol)
                        if correlation:
                            correlations.append(correlation)

            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                continue

        # Sort by confidence
        correlations.sort(key=lambda x: x.confidence, reverse=True)
        self._correlation_cache = correlations

        return correlations

    def _identify_politician(self, filer_name: str) -> Optional[PoliticianProfile]:
        """Check if a filer name matches a known politician."""
        filer_upper = filer_name.upper()

        for key, profile in KNOWN_POLITICIANS.items():
            if key in filer_upper:
                return profile
            # Also check last name match
            last_name = profile.name.split()[-1].upper()
            if last_name in filer_upper:
                return profile

        return None

    def _check_contract_correlation(self,
                                    trade: InsiderTrade,
                                    politician: PoliticianProfile,
                                    symbol: str,
                                    days_lookback: int) -> Optional[CorrelationSignal]:
        """
        Check if there's a contract correlation for a politician trade.

        Pattern: Politician on Committee X buys stock in Company Y,
                 then Agency (overseen by Committee X) awards contract to Company Y.
        """
        try:
            # Get company name from symbol (simplified - would need mapping)
            company_name = self._symbol_to_company(symbol)

            # Search for contracts to this company
            contracts = self.usa_client.search_by_company(company_name, days=days_lookback * 2)

            if not contracts:
                return None

            # Find contracts within a relevant time window
            trade_date = self._parse_date(trade.transaction_date)
            if not trade_date:
                return None

            for contract in contracts:
                contract_date = self._parse_date(contract.award_date)
                if not contract_date:
                    continue

                days_diff = (contract_date - trade_date).days

                # Trade before contract (predictive) = positive signal
                # Trade after contract (reactive) = weaker signal
                if -30 <= days_diff <= 90:  # Trade 30 days before to 90 days after
                    # Check committee oversight
                    oversight_match = politician.oversees_agency(contract.awarding_agency)

                    # Calculate confidence
                    confidence = self._calculate_confidence(
                        trade_value=trade.total_value,
                        contract_value=contract.award_amount,
                        days_between=days_diff,
                        oversight_match=oversight_match,
                        is_buy=trade.transaction_type.lower() in ["p", "buy", "purchase"]
                    )

                    if confidence > 0.5:  # Only return significant correlations
                        return CorrelationSignal(
                            trade=trade,
                            trader_name=politician.name,
                            symbol=symbol,
                            trade_date=trade.transaction_date,
                            trade_type="buy" if trade.transaction_type.lower() in ["p", "buy", "purchase"] else "sell",
                            shares=trade.shares_traded,
                            value=trade.total_value,
                            contract=contract,
                            contract_value=contract.award_amount,
                            awarding_agency=contract.awarding_agency,
                            contract_date=contract.award_date,
                            correlation_type="politician_contract",
                            days_between=days_diff,
                            confidence=confidence,
                            committee=", ".join(politician.committees[:2]),
                            oversight_match=oversight_match
                        )

            return None

        except Exception as e:
            print(f"Contract correlation error: {e}")
            return None

    def _check_insider_pattern(self, trade: InsiderTrade, symbol: str) -> Optional[CorrelationSignal]:
        """
        Check for unusual insider trading patterns.

        Patterns to detect:
        - Cluster buying (multiple insiders buying together)
        - Large trades relative to history
        - Executive buying before known announcements
        """
        # For now, just flag large insider buys
        if trade.total_value > 100000 and trade.transaction_type.lower() in ["p", "buy", "purchase"]:
            # Check if this is a significant purchase
            confidence = min(0.3 + (trade.total_value / 1000000) * 0.2, 0.7)

            return CorrelationSignal(
                trade=trade,
                trader_name=trade.filer_name,
                symbol=symbol,
                trade_date=trade.transaction_date,
                trade_type="buy",
                shares=trade.shares_traded,
                value=trade.total_value,
                correlation_type="insider_preannouncement",
                confidence=confidence
            )

        return None

    def _calculate_confidence(self,
                             trade_value: float,
                             contract_value: float,
                             days_between: int,
                             oversight_match: bool,
                             is_buy: bool) -> float:
        """
        Calculate confidence score for a correlation.

        Factors:
        - Trade timing (before contract = higher confidence)
        - Committee oversight match (big bonus)
        - Trade size relative to contract
        - Direction (buy before contract = bullish signal)
        """
        confidence = 0.3  # Base confidence

        # Timing bonus: Trade before contract is more predictive
        if days_between < 0:  # Trade came before contract
            timing_bonus = min(0.3, abs(days_between) / 30 * 0.1)  # Up to 0.3 for trades 30+ days before
            confidence += timing_bonus
        else:
            # Trade after contract - still informative but less predictive
            confidence += 0.05

        # Oversight match is the strongest signal
        if oversight_match:
            confidence += 0.25

        # Trade size matters
        if trade_value > 100000:
            confidence += 0.1
        if trade_value > 500000:
            confidence += 0.1

        # Direction consistency
        if is_buy:
            confidence += 0.05  # Buying before contract award = bullish

        # Contract size
        if contract_value > 10000000:  # $10M+
            confidence += 0.1
        if contract_value > 100000000:  # $100M+
            confidence += 0.1

        return min(confidence, 1.0)

    def _symbol_to_company(self, symbol: str) -> str:
        """Map stock symbol to company name for contract search."""
        # Basic mapping - would need to be expanded
        mappings = {
            "LMT": "Lockheed",
            "RTX": "Raytheon",
            "BA": "Boeing",
            "NOC": "Northrop",
            "GD": "General Dynamics",
            "MSFT": "Microsoft",
            "AMZN": "Amazon",
            "GOOGL": "Google",
            "AAPL": "Apple",
            "META": "Meta",
            "NVDA": "NVIDIA",
        }
        return mappings.get(symbol, symbol)

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse various date formats."""
        if not date_str:
            return None

        formats = ["%Y-%m-%d", "%m/%d/%Y", "%Y%m%d"]
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        return None

    def get_daily_alerts(self, min_confidence: float = 0.6) -> List[Dict]:
        """
        Generate daily alerts for Guiding Light.

        Returns list of alerts with:
        - Level: STRONG, MEDIUM, WATCH
        - Plain language description
        - Confidence score
        - Recommended action (informational only)
        """
        if not self._correlation_cache:
            self.find_correlations()

        alerts = []

        for corr in self._correlation_cache:
            if corr.confidence < min_confidence:
                continue

            # Determine alert level
            if corr.confidence >= 0.8:
                level = "STRONG"
            elif corr.confidence >= 0.7:
                level = "MEDIUM"
            else:
                level = "WATCH"

            alerts.append({
                "level": level,
                "symbol": corr.symbol,
                "description": corr.to_plain_language(),
                "confidence": corr.confidence,
                "correlation_type": corr.correlation_type,
                "trade_value": corr.value,
                "contract_value": corr.contract_value if corr.contract else 0,
                "trader": corr.trader_name,
                "oversight_match": corr.oversight_match
            })

        return alerts


def find_correlations(symbols: List[str] = None, days: int = 90) -> List[CorrelationSignal]:
    """Convenience function for quick correlation check."""
    tracker = PoliticianTracker()
    return tracker.find_correlations(symbols=symbols, days_lookback=days)


def get_daily_alerts(min_confidence: float = 0.6) -> List[Dict]:
    """Convenience function for daily alert generation."""
    tracker = PoliticianTracker()
    tracker.find_correlations()
    return tracker.get_daily_alerts(min_confidence=min_confidence)


if __name__ == "__main__":
    print("Politician Tracker - Correlation Engine")
    print("=" * 50)

    tracker = PoliticianTracker()

    # Check defense contractors
    print("\nChecking defense contractors...")
    correlations = tracker.find_correlations(
        symbols=["LMT", "RTX", "BA"],
        days_lookback=90,
        min_trade_value=50000
    )

    print(f"\nFound {len(correlations)} correlations")

    for corr in correlations[:5]:
        print(f"\n[{corr.confidence:.2f}] {corr.correlation_type}")
        print(f"  {corr.to_plain_language()}")
        if corr.oversight_match:
            print(f"  ** OVERSIGHT MATCH: {corr.committee}")

    # Generate alerts
    print("\n" + "=" * 50)
    print("DAILY ALERTS")
    print("=" * 50)

    alerts = tracker.get_daily_alerts(min_confidence=0.5)
    for alert in alerts[:5]:
        print(f"\n[{alert['level']}] {alert['symbol']}")
        print(f"  {alert['description']}")
        print(f"  Confidence: {alert['confidence']:.2f}")
