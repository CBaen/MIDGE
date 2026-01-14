#!/usr/bin/env python3
"""
ingest_contracts.py - Continuous government contract ingestion worker

Fetches contract awards from USASpending.gov and stores them to Qdrant.
Runs continuously with configurable intervals.

Usage:
    python scripts/ingest_contracts.py                    # Run once
    python scripts/ingest_contracts.py --continuous       # Run forever
    python scripts/ingest_contracts.py --interval 7200    # Custom interval (seconds)
"""

import sys
import time
import json
import requests
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from trading.apis.usa_spending import USASpendingClient, get_oversight_committees

QDRANT_URL = "http://localhost:6333"
COLLECTION = "midge_signals"
OLLAMA_URL = "http://localhost:11434"

# Companies to track - map stock symbols to company names for USASpending search
TRACKED_COMPANIES = {
    "LMT": ["Lockheed Martin"],
    "RTX": ["Raytheon", "RTX Corporation"],
    "BA": ["Boeing"],
    "NOC": ["Northrop Grumman"],
    "GD": ["General Dynamics"],
    "LHX": ["L3Harris"],
    "MSFT": ["Microsoft"],
    "AMZN": ["Amazon Web Services", "AWS"],
    "GOOGL": ["Google", "Alphabet"],
    "ORCL": ["Oracle"],
    "IBM": ["IBM", "International Business Machines"],
    "PLTR": ["Palantir"],
    "UNH": ["UnitedHealth"],
    "JNJ": ["Johnson & Johnson"],
    "PFE": ["Pfizer"],
}

# USASpending API base
USA_SPENDING_API = "https://api.usaspending.gov/api/v2"


@dataclass
class ContractSignal:
    """A trading signal from government contract."""
    signal_id: str
    signal_source: str
    symbol: str
    timestamp: str
    direction: str  # "bullish" for contract awards
    confidence: float
    details: dict
    raw_data: dict


def get_embedding(text: str) -> Optional[List[float]]:
    """Get embedding from Ollama."""
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": text},
            timeout=30
        )
        if response.status_code == 200:
            return response.json().get("embedding")
    except:
        pass
    return None


def store_signal(signal: ContractSignal) -> bool:
    """Store a signal to Qdrant."""

    text = f"{signal.signal_source} signal: {signal.symbol} {signal.direction} "
    text += f"confidence {signal.confidence:.0%}. {json.dumps(signal.details)}"

    embedding = get_embedding(text)
    if not embedding:
        print(f"  [WARN] Could not get embedding for {signal.symbol}")
        return False

    payload = {
        "signal_id": signal.signal_id,
        "signal_source": signal.signal_source,
        "symbol": signal.symbol,
        "timestamp": signal.timestamp,
        "direction": signal.direction,
        "confidence": signal.confidence,
        "details": signal.details,
        "text": text,
        "ingested_at": datetime.now().isoformat(),
        "decayed": False
    }

    try:
        response = requests.put(
            f"{QDRANT_URL}/collections/{COLLECTION}/points",
            json={
                "points": [{
                    "id": abs(hash(signal.signal_id)) % (10**18),
                    "vector": embedding,
                    "payload": payload
                }]
            }
        )
        return response.status_code in (200, 201)
    except Exception as e:
        print(f"  [ERROR] Failed to store: {e}")
        return False


def search_contracts(recipient_name: str, days: int = 30) -> List[Dict]:
    """Search for recent contracts awarded to a recipient."""
    contracts = []

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    try:
        response = requests.post(
            f"{USA_SPENDING_API}/search/spending_by_award/",
            json={
                "filters": {
                    "recipient_search_text": [recipient_name],
                    "time_period": [{
                        "start_date": start_date.strftime("%Y-%m-%d"),
                        "end_date": end_date.strftime("%Y-%m-%d")
                    }],
                    "award_type_codes": ["A", "B", "C", "D"]  # Contract types
                },
                "fields": [
                    "Award ID",
                    "Recipient Name",
                    "Award Amount",
                    "Awarding Agency",
                    "Start Date",
                    "Description"
                ],
                "page": 1,
                "limit": 50,
                "sort": "Award Amount",
                "order": "desc"
            },
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            for result in data.get("results", []):
                contracts.append({
                    "award_id": result.get("Award ID"),
                    "recipient": result.get("Recipient Name"),
                    "amount": result.get("Award Amount"),
                    "agency": result.get("Awarding Agency"),
                    "start_date": result.get("Start Date"),
                    "description": result.get("Description", "")[:500]
                })

    except Exception as e:
        print(f"  [ERROR] USASpending API error: {e}")

    return contracts


def fetch_and_store_contracts(symbol: str, company_names: List[str], days: int = 30) -> int:
    """Fetch contracts for a company and store as signals."""
    stored = 0

    for company_name in company_names:
        contracts = search_contracts(company_name, days=days)

        for contract in contracts:
            if not contract.get("amount") or contract["amount"] <= 0:
                continue

            # Create signal ID
            signal_id = hashlib.md5(
                f"{symbol}_{contract['award_id']}_{contract['amount']}".encode()
            ).hexdigest()

            # Calculate confidence based on contract size
            amount = contract["amount"]
            base_confidence = 0.5
            if amount > 10_000_000:  # $10M+
                base_confidence = 0.65
            if amount > 50_000_000:  # $50M+
                base_confidence = 0.75
            if amount > 100_000_000:  # $100M+
                base_confidence = 0.85

            # Check if awarding agency has congressional oversight connection
            agency = contract.get("agency", "")
            oversight = get_oversight_committees(agency)
            if oversight:
                base_confidence += 0.05  # Boost for tracked oversight

            signal = ContractSignal(
                signal_id=signal_id,
                signal_source="usa_spending",
                symbol=symbol,
                timestamp=contract.get("start_date") or datetime.now().isoformat(),
                direction="bullish",  # Contract awards are bullish
                confidence=min(0.95, base_confidence),
                details={
                    "award_id": contract["award_id"],
                    "amount": amount,
                    "agency": agency,
                    "recipient": contract["recipient"],
                    "description": contract["description"][:200],
                    "oversight_committees": oversight
                },
                raw_data=contract
            )

            if store_signal(signal):
                stored += 1

    return stored


def run_ingestion_cycle():
    """Run one complete contract ingestion cycle."""
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting contract ingestion cycle...")

    total_stored = 0
    total_companies = len(TRACKED_COMPANIES)

    for i, (symbol, names) in enumerate(TRACKED_COMPANIES.items(), 1):
        print(f"  [{i}/{total_companies}] Fetching {symbol} ({names[0]})...", end=" ")
        stored = fetch_and_store_contracts(symbol, names)
        print(f"stored {stored} signals")
        total_stored += stored

        # Brief pause between companies
        time.sleep(0.5)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Cycle complete. Stored {total_stored} signals.")
    return total_stored


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Government Contract Ingestion Worker")
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=7200, help="Seconds between cycles (default: 7200)")
    parser.add_argument("--days", type=int, default=30, help="Days of history to fetch (default: 30)")

    args = parser.parse_args()

    print("="*60)
    print("MIDGE Government Contract Ingestion Worker")
    print("="*60)
    print(f"Qdrant: {QDRANT_URL}")
    print(f"Collection: {COLLECTION}")
    print(f"Tracking {len(TRACKED_COMPANIES)} companies")
    print(f"Mode: {'Continuous' if args.continuous else 'Single run'}")
    if args.continuous:
        print(f"Interval: {args.interval} seconds")

    if args.continuous:
        print("\nPress Ctrl+C to stop.\n")
        try:
            while True:
                run_ingestion_cycle()
                print(f"Sleeping {args.interval}s until next cycle...")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped by user.")
    else:
        run_ingestion_cycle()


if __name__ == "__main__":
    main()
