#!/usr/bin/env python3
"""
contract_predictor.py - Predict contract winners BEFORE announcements

The Pattern (from Guiding Light's insight):
"I was looking at one contract in Alabama. There were three companies
up for it... they started hiring blitz even before they won the contract..."

Bidder + Hiring Blitz + Insider Buying = WINNER BEFORE ANNOUNCEMENT

Data sources:
1. SAM.gov - Who's bidding on contracts
2. Job APIs - Who's hiring aggressively
3. Insider trades - Who's buying their own stock
4. USASpending - Historical win patterns

This is MIDGE's most speculative edge - but also potentially the most valuable.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from datetime import datetime, timedelta
import requests

# Import our data sources
from mae_core.market.apis.job_tracker import JobTracker, HiringSignal, DEFENSE_CONTRACTORS
from mae_core.market.apis.sam_gov import SAMGovClient, ContractOpportunity
from mae_core.market.apis.sec_edgar import get_recent_form4s


QDRANT_URL = "http://localhost:6333"


@dataclass
class ContractPrediction:
    """
    Prediction that a company will win a specific contract.

    Confidence based on signal convergence:
    - Bidder on active contract: +0.20
    - Hiring blitz detected: +0.25
    - Insider buying: +0.20
    - Historical win pattern: +0.15
    """
    # The prediction
    predicted_winner: str
    predicted_ticker: Optional[str] = None
    contract_title: str = ""
    contract_value: float = 0.0

    # Signals that formed this prediction
    is_active_bidder: bool = False
    hiring_blitz_detected: bool = False
    insider_buying_detected: bool = False
    historical_winner: bool = False  # Have they won similar contracts?

    # Signal details
    hiring_spike_ratio: float = 0.0
    insider_buy_count: int = 0
    insider_buy_value: float = 0.0

    # Confidence
    confidence: float = 0.0
    confidence_breakdown: Dict[str, float] = field(default_factory=dict)

    # Timing
    predicted_at: str = ""
    contract_deadline: str = ""
    expected_award_date: str = ""

    # Metadata
    signal_source: str = "contract_prediction"
    decay_rate: float = 0.03  # Slow decay - contract timelines are long

    def __post_init__(self):
        if not self.predicted_at:
            self.predicted_at = datetime.now().isoformat()
        self._calculate_confidence()

    def _calculate_confidence(self):
        """Calculate confidence from component signals."""
        breakdown = {}

        if self.is_active_bidder:
            breakdown["active_bidder"] = 0.20

        if self.hiring_blitz_detected:
            # Scale by spike intensity
            base = 0.15
            if self.hiring_spike_ratio >= 3.0:
                base = 0.25
            elif self.hiring_spike_ratio >= 2.0:
                base = 0.20
            breakdown["hiring_blitz"] = base

        if self.insider_buying_detected:
            # Scale by buy value
            base = 0.10
            if self.insider_buy_value >= 500000:
                base = 0.20
            elif self.insider_buy_value >= 100000:
                base = 0.15
            breakdown["insider_buying"] = base

        if self.historical_winner:
            breakdown["historical_pattern"] = 0.15

        self.confidence_breakdown = breakdown
        self.confidence = min(0.95, sum(breakdown.values()))

    def to_plain_language(self) -> str:
        """Format for dashboard."""
        signals = []
        if self.hiring_blitz_detected:
            signals.append(f"hiring spike {self.hiring_spike_ratio:.1f}x")
        if self.insider_buying_detected:
            signals.append(f"insider buys ${self.insider_buy_value:,.0f}")
        if self.historical_winner:
            signals.append("past winner")

        signal_str = ", ".join(signals) if signals else "bidder only"

        return (
            f"PREDICTION: {self.predicted_winner} ({self.predicted_ticker or 'private'}) "
            f"to win {self.contract_title[:40]}... "
            f"[{signal_str}] "
            f"Confidence: {self.confidence:.0%}"
        )


class ContractPredictor:
    """
    Predicts contract winners by correlating multiple signals.

    The edge: Companies know they're winning before announcements.
    They signal this through:
    1. Hiring for the contract work
    2. Insiders buying stock
    3. Board connections to the agency

    We detect these signals and predict winners.
    """

    def __init__(self):
        self.job_tracker = JobTracker()
        self.sam_client = SAMGovClient()
        self.qdrant_url = QDRANT_URL

    def analyze_contract(
        self,
        contract: ContractOpportunity,
        potential_bidders: List[Tuple[str, str]]  # (company_name, ticker)
    ) -> List[ContractPrediction]:
        """
        Analyze a contract opportunity and predict the winner.

        Args:
            contract: The contract opportunity from SAM.gov
            potential_bidders: List of (company_name, ticker) tuples

        Returns:
            List of predictions, sorted by confidence
        """
        predictions = []

        for company_name, ticker in potential_bidders:
            prediction = self._analyze_bidder(contract, company_name, ticker)
            if prediction.confidence > 0:
                predictions.append(prediction)

        # Sort by confidence
        predictions.sort(key=lambda p: p.confidence, reverse=True)
        return predictions

    def _analyze_bidder(
        self,
        contract: ContractOpportunity,
        company_name: str,
        ticker: str
    ) -> ContractPrediction:
        """Analyze a single potential bidder."""
        prediction = ContractPrediction(
            predicted_winner=company_name,
            predicted_ticker=ticker,
            contract_title=contract.title,
            contract_value=contract.estimated_value,
            contract_deadline=contract.response_deadline,
            is_active_bidder=True  # Assume they're bidding if we're checking
        )

        # Check hiring activity
        hiring_signal = self.job_tracker.analyze_hiring_activity(company_name, ticker)
        if hiring_signal.is_spike:
            prediction.hiring_blitz_detected = True
            prediction.hiring_spike_ratio = hiring_signal.spike_ratio

        # Check insider buying
        insider_data = self._check_insider_buying(ticker)
        if insider_data["has_buying"]:
            prediction.insider_buying_detected = True
            prediction.insider_buy_count = insider_data["buy_count"]
            prediction.insider_buy_value = insider_data["buy_value"]

        # Check historical wins (via Qdrant or USASpending)
        prediction.historical_winner = self._check_historical_wins(
            company_name, contract.naics_code
        )

        # Recalculate confidence with all signals
        prediction._calculate_confidence()

        return prediction

    def _check_insider_buying(self, ticker: str, days_back: int = 30) -> Dict:
        """Check for recent insider buying activity."""
        result = {
            "has_buying": False,
            "buy_count": 0,
            "buy_value": 0.0
        }

        if not ticker:
            return result

        try:
            # Use SEC EDGAR function
            trades = get_recent_form4s(ticker, days_back)

            for trade in trades:
                if trade.is_purchase and trade.transaction_code == "P":
                    result["buy_count"] += 1
                    result["buy_value"] += trade.total_value

            result["has_buying"] = result["buy_count"] > 0

        except Exception as e:
            print(f"Insider check error for {ticker}: {e}")

        return result

    def _check_historical_wins(self, company_name: str, naics_code: str) -> bool:
        """
        Check if company has won similar contracts before.

        Uses Qdrant to search for historical contract awards.
        """
        try:
            response = requests.post(
                f"{self.qdrant_url}/collections/midge_signals/points/scroll",
                json={
                    "filter": {
                        "must": [
                            {"key": "signal_source", "match": {"value": "contract_award"}},
                            {"key": "recipient_name", "match": {"value": company_name}}
                        ]
                    },
                    "limit": 10,
                    "with_payload": True
                },
                timeout=10
            )

            if response.status_code == 200:
                points = response.json().get("result", {}).get("points", [])
                return len(points) > 0

        except Exception:
            pass

        return False

    def scan_defense_contracts(self) -> List[ContractPrediction]:
        """
        Scan active defense contracts and predict winners.

        Uses known defense contractors as potential bidders.
        """
        # Get active defense opportunities
        opportunities = self.sam_client.get_defense_opportunities(days_back=14)

        if not opportunities:
            print("No active defense opportunities found")
            return []

        # Convert defense contractors to bidder format
        bidders = [(name, ticker) for name, ticker in DEFENSE_CONTRACTORS.items()]

        all_predictions = []

        for contract in opportunities[:5]:  # Limit to top 5 for now
            print(f"Analyzing: {contract.title[:50]}...")
            predictions = self.analyze_contract(contract, bidders)

            # Keep top prediction for each contract
            if predictions and predictions[0].confidence >= 0.30:
                all_predictions.append(predictions[0])

        # Sort all predictions by confidence
        all_predictions.sort(key=lambda p: p.confidence, reverse=True)
        return all_predictions

    def quick_scan(self, company_name: str, ticker: str) -> ContractPrediction:
        """
        Quick scan of a single company for contract signals.

        Doesn't require a specific contract - just checks if the company
        shows signs of expecting to win something.
        """
        prediction = ContractPrediction(
            predicted_winner=company_name,
            predicted_ticker=ticker,
            contract_title="Unknown (signal-based detection)",
            is_active_bidder=False  # Don't know specific contract
        )

        # Check hiring
        hiring_signal = self.job_tracker.analyze_hiring_activity(company_name, ticker)
        if hiring_signal.is_spike:
            prediction.hiring_blitz_detected = True
            prediction.hiring_spike_ratio = hiring_signal.spike_ratio
            prediction.is_active_bidder = True  # Assume bidding if hiring

        # Check insiders
        insider_data = self._check_insider_buying(ticker)
        if insider_data["has_buying"]:
            prediction.insider_buying_detected = True
            prediction.insider_buy_count = insider_data["buy_count"]
            prediction.insider_buy_value = insider_data["buy_value"]

        prediction._calculate_confidence()
        return prediction


def predict_contract_winners() -> List[ContractPrediction]:
    """
    Convenience function to run full contract prediction scan.
    """
    predictor = ContractPredictor()
    return predictor.scan_defense_contracts()


def quick_scan_company(company_name: str, ticker: str) -> ContractPrediction:
    """
    Quick scan a single company for contract win signals.
    """
    predictor = ContractPredictor()
    return predictor.quick_scan(company_name, ticker)


if __name__ == "__main__":
    import sys

    print("Contract Predictor - Win Before Announcement")
    print()

    predictor = ContractPredictor()

    if len(sys.argv) > 2:
        # Quick scan specific company
        company = sys.argv[1]
        ticker = sys.argv[2]
        print(f"Quick scan: {company} ({ticker})")
        prediction = predictor.quick_scan(company, ticker)
        print(f"  {prediction.to_plain_language()}")
        print(f"  Breakdown: {prediction.confidence_breakdown}")

    elif len(sys.argv) > 1 and sys.argv[1] == "defense":
        # Full defense scan
        print("Scanning defense contracts for predicted winners...")
        predictions = predictor.scan_defense_contracts()

        if predictions:
            print(f"\nTop predictions:")
            for p in predictions[:5]:
                print(f"  {p.to_plain_language()}")
        else:
            print("  No high-confidence predictions")

    else:
        # Default: quick scan top defense contractors
        print("Quick scan of defense contractors...")
        for company, ticker in list(DEFENSE_CONTRACTORS.items())[:3]:
            prediction = predictor.quick_scan(company, ticker)
            if prediction.confidence > 0:
                print(f"  {prediction.to_plain_language()}")

    print("\nDone.")
