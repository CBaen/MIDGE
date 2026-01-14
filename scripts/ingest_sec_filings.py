#!/usr/bin/env python3
"""
ingest_sec_filings.py - Continuous SEC Form 4 ingestion worker

Fetches insider trading filings from SEC EDGAR and stores them to Qdrant.
Runs continuously with configurable intervals.

Usage:
    python scripts/ingest_sec_filings.py                    # Run once
    python scripts/ingest_sec_filings.py --continuous       # Run forever
    python scripts/ingest_sec_filings.py --interval 3600    # Custom interval (seconds)
"""

import sys
import time
import json
import requests
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from trading.apis.sec_edgar import SECEdgarClient, get_recent_form4s

QDRANT_URL = "http://localhost:6333"
COLLECTION = "midge_signals"
OLLAMA_URL = "http://localhost:11434"

# Symbols to track - defense contractors and tech giants with government contracts
TRACKED_SYMBOLS = [
    # Defense
    "LMT", "RTX", "BA", "NOC", "GD", "LHX",
    # Tech with government contracts
    "MSFT", "AMZN", "GOOGL", "ORCL", "IBM", "PLTR",
    # Healthcare
    "UNH", "JNJ", "PFE", "MRK",
    # Finance
    "JPM", "GS", "BAC",
]

# Rate limit: SEC asks for max 10 requests/second
SEC_RATE_LIMIT_DELAY = 0.15  # seconds between requests


@dataclass
class IngestedSignal:
    """A trading signal ingested from SEC filings."""
    signal_id: str
    signal_source: str
    symbol: str
    timestamp: str
    direction: str  # "bullish" or "bearish"
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


def store_signal(signal: IngestedSignal) -> bool:
    """Store a signal to Qdrant."""

    # Create searchable text
    text = f"{signal.signal_source} signal: {signal.symbol} {signal.direction} "
    text += f"confidence {signal.confidence:.0%}. {json.dumps(signal.details)}"

    # Get embedding
    embedding = get_embedding(text)
    if not embedding:
        print(f"  [WARN] Could not get embedding for {signal.symbol}")
        return False

    # Store to Qdrant
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


def fetch_and_store_form4s(symbol: str, days: int = 7) -> int:
    """Fetch Form 4 filings for a symbol and store them."""
    stored = 0

    try:
        trades = get_recent_form4s(symbol, days=days)

        for trade in trades:
            # Create signal ID from trade details
            signal_id = hashlib.md5(
                f"{trade.symbol}_{trade.filer_name}_{trade.transaction_date}_{trade.shares}".encode()
            ).hexdigest()

            # Determine direction
            direction = "bullish" if trade.transaction_type == "buy" else "bearish"

            # Calculate confidence based on trade size and insider role
            base_confidence = 0.5
            if trade.shares * (trade.price_per_share or 100) > 100000:
                base_confidence += 0.15  # Large trade
            if trade.shares * (trade.price_per_share or 100) > 500000:
                base_confidence += 0.15  # Very large trade
            # Insider role boost would go here if we had that data

            signal = IngestedSignal(
                signal_id=signal_id,
                signal_source="sec_form4",
                symbol=trade.symbol,
                timestamp=trade.transaction_date or datetime.now().isoformat(),
                direction=direction,
                confidence=min(0.95, base_confidence),
                details={
                    "filer": trade.filer_name,
                    "shares": trade.shares,
                    "price": trade.price_per_share,
                    "type": trade.transaction_type,
                    "value": trade.shares * (trade.price_per_share or 0)
                },
                raw_data=asdict(trade)
            )

            if store_signal(signal):
                stored += 1

    except Exception as e:
        print(f"  [ERROR] Failed to fetch {symbol}: {e}")

    return stored


def run_ingestion_cycle():
    """Run one complete ingestion cycle."""
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting SEC ingestion cycle...")

    total_stored = 0
    total_symbols = len(TRACKED_SYMBOLS)

    for i, symbol in enumerate(TRACKED_SYMBOLS, 1):
        print(f"  [{i}/{total_symbols}] Fetching {symbol}...", end=" ")
        stored = fetch_and_store_form4s(symbol)
        print(f"stored {stored} signals")
        total_stored += stored

        # Rate limiting
        time.sleep(SEC_RATE_LIMIT_DELAY)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Cycle complete. Stored {total_stored} signals.")
    return total_stored


def main():
    import argparse

    parser = argparse.ArgumentParser(description="SEC Form 4 Ingestion Worker")
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=3600, help="Seconds between cycles (default: 3600)")
    parser.add_argument("--symbols", nargs="+", help="Override symbols to track")

    args = parser.parse_args()

    if args.symbols:
        global TRACKED_SYMBOLS
        TRACKED_SYMBOLS = args.symbols

    print("="*60)
    print("MIDGE SEC Form 4 Ingestion Worker")
    print("="*60)
    print(f"Qdrant: {QDRANT_URL}")
    print(f"Collection: {COLLECTION}")
    print(f"Tracking {len(TRACKED_SYMBOLS)} symbols")
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
