#!/usr/bin/env python3
"""backfill_archives.py — Populate MIDGE signal archives with historical data.

Downloads 90 days of market signals from free data sources (SEC EDGAR,
Congressional trades, USASpending, EFTS keywords), converts them through
the same signal.py adapters the live system uses, and writes them to
data/midge/signals/YYYY-MM-DD.jsonl — one file per day.

This feeds the lag-correlation analyzer, Thompson calibrator, and Kelly
position sizer with enough temporal data to produce meaningful results.

Usage:
    python backfill_archives.py                    # Default: 90 days, all sources
    python backfill_archives.py --days 180         # Override lookback
    python backfill_archives.py --sources sec,congress  # Subset of sources
    python backfill_archives.py --dry-run          # Count signals, don't write
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-30s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("midge.backfill")

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
SIGNALS_DIR = PROJECT_ROOT / "data" / "midge" / "signals"
WATCHLIST_PATH = PROJECT_ROOT / "data" / "midge" / "watchlist.json"

# ── Ticker list for SEC Form 4 backfill ───────────────────────────────────────

BACKFILL_TICKERS = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "CRM", "ORCL",
    # Defense
    "LMT", "RTX", "NOC", "GD", "BA",
    # Finance
    "JPM", "GS", "BAC", "WFC",
    # Pharma
    "JNJ", "PFE", "UNH",
    # Energy
    "XOM", "CVX",
    # Semiconductors
    "INTC", "AMD", "MU", "AVGO",
    # Retail
    "WMT", "COST",
    # Cybersecurity
    "PANW", "CRWD",
]

ALL_SOURCES = ["sec", "congress", "contracts", "efts"]


# ── Signal serialization (matches sensing_hook._store_signals) ────────────────

def _serialize_signal(sig) -> dict:
    """Convert MarketSignal to the JSONL record format the archive reader expects."""
    return {
        "signal_id": sig.signal_id,
        "source": sig.source,
        "symbol": sig.symbol,
        "domain": sig.domain,
        "direction": sig.direction,
        "strength": sig.strength,
        "confidence": sig.confidence,
        "velocity": sig.velocity,
        "timestamp": sig.timestamp.isoformat(),
        "received_at": sig.received_at.isoformat(),
        "metadata": sig.metadata,
    }


# ── Phase 1: Fetch ───────────────────────────────────────────────────────────

def fetch_sec_form4(tickers: List[str], days: int) -> list:
    """Fetch Form 4 insider trades for all tickers."""
    try:
        from mae_core.market.apis.sec_edgar import get_recent_form4s
    except ImportError:
        logger.warning("SEC EDGAR client not available, skipping Form 4")
        return []

    all_trades = []
    for i, ticker in enumerate(tickers):
        logger.info("  SEC Form 4: %s (%d/%d)", ticker, i + 1, len(tickers))
        try:
            trades = get_recent_form4s(ticker, days=days)
            all_trades.extend(trades)
            logger.info("    -> %d trades", len(trades))
        except Exception as e:
            logger.warning("    -> FAILED: %s", e)
    return all_trades


def fetch_house_trades(days: int) -> list:
    """Fetch House congressional trades."""
    try:
        from mae_core.market.apis.house_stock_watcher import HouseStockWatcherClient
    except ImportError:
        logger.warning("House Stock Watcher client not available, skipping")
        return []

    try:
        client = HouseStockWatcherClient()
        trades = client.get_recent_trades(days=days)
        logger.info("  House trades: %d", len(trades))
        return trades
    except Exception as e:
        logger.warning("  House trades FAILED: %s", e)
        return []


def fetch_senate_trades(days: int) -> list:
    """Fetch Senate stock trades."""
    try:
        from mae_core.market.apis.senate_stock_watcher import SenateStockWatcherClient
    except ImportError:
        logger.warning("Senate Stock Watcher client not available, skipping")
        return []

    try:
        client = SenateStockWatcherClient()
        trades = client.get_recent_trades(days=days)
        logger.info("  Senate trades: %d", len(trades))
        return trades
    except Exception as e:
        logger.warning("  Senate trades FAILED: %s", e)
        return []


def fetch_contracts(days: int) -> list:
    """Fetch large government contracts from USASpending.gov."""
    try:
        from mae_core.market.apis.usa_spending import USASpendingClient
    except ImportError:
        logger.warning("USASpending client not available, skipping")
        return []

    try:
        client = USASpendingClient()
        contracts = client.get_recent_large_contracts(days=days)
        logger.info("  Government contracts: %d", len(contracts))
        return contracts
    except Exception as e:
        logger.warning("  Government contracts FAILED: %s", e)
        return []


def fetch_efts_keywords(days: int) -> list:
    """Fetch SEC EFTS keyword filing hits."""
    try:
        from mae_core.market.apis.sec_edgar.efts import SECFullTextSearchClient
    except ImportError:
        logger.warning("SEC EFTS client not available, skipping")
        return []

    try:
        client = SECFullTextSearchClient()
        hits = client.scan_all_keywords(days=days)
        logger.info("  EFTS keyword hits: %d", len(hits))
        return hits
    except Exception as e:
        logger.warning("  EFTS keyword scan FAILED: %s", e)
        return []


# ── Phase 2: Convert ─────────────────────────────────────────────────────────

def convert_all(
    form4_trades: list,
    house_trades: list,
    senate_trades: list,
    contracts: list,
    efts_hits: list,
) -> list:
    """Convert raw API results to MarketSignal objects, deduped by signal_id."""
    from mae_core.market.signal import (
        from_insider_trade,
        from_congressional_trade,
        from_senate_trade,
        from_government_contract,
        from_filing_keyword,
    )

    signals = []
    seen_ids: Set[str] = set()

    def _add(sig):
        if sig.signal_id not in seen_ids:
            seen_ids.add(sig.signal_id)
            signals.append(sig)

    for trade in form4_trades:
        try:
            _add(from_insider_trade(trade))
        except Exception:
            pass

    for trade in house_trades:
        try:
            _add(from_congressional_trade(trade))
        except Exception:
            pass

    for trade in senate_trades:
        try:
            _add(from_senate_trade(trade))
        except Exception:
            pass

    for contract in contracts:
        try:
            _add(from_government_contract(contract))
        except Exception:
            pass

    for hit in efts_hits:
        try:
            _add(from_filing_keyword(hit))
        except Exception:
            pass

    return signals


# ── Phase 3: Write ───────────────────────────────────────────────────────────

def load_existing_ids(path: Path) -> Set[str]:
    """Load signal_ids already in a JSONL file for dedup."""
    ids = set()
    if path.exists():
        try:
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        sid = record.get("signal_id")
                        if sid:
                            ids.add(sid)
                    except json.JSONDecodeError:
                        pass
        except OSError:
            pass
    return ids


def write_signals(signals: list, dry_run: bool = False) -> Dict[str, int]:
    """Write signals to daily JSONL files. Returns {date_str: count} written."""
    # Group by event date
    by_date: Dict[str, list] = defaultdict(list)
    for sig in signals:
        day_str = sig.timestamp.strftime("%Y-%m-%d")
        by_date[day_str].append(sig)

    if dry_run:
        logger.info("DRY RUN — would write %d signals across %d days",
                     len(signals), len(by_date))
        return {d: len(sigs) for d, sigs in by_date.items()}

    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
    written_counts: Dict[str, int] = {}

    for day_str in sorted(by_date.keys()):
        day_signals = by_date[day_str]
        path = SIGNALS_DIR / f"{day_str}.jsonl"

        # Load existing IDs for idempotent append
        existing_ids = load_existing_ids(path)
        new_signals = [s for s in day_signals if s.signal_id not in existing_ids]

        if not new_signals:
            continue

        with open(path, "a") as f:
            for sig in new_signals:
                record = _serialize_signal(sig)
                f.write(json.dumps(record) + "\n")

        written_counts[day_str] = len(new_signals)

    return written_counts


# ── Main ──────────────────────────────────────────────────────────────────────

def get_tickers() -> List[str]:
    """Load watchlist tickers and merge with BACKFILL_TICKERS."""
    tickers = list(BACKFILL_TICKERS)
    if WATCHLIST_PATH.exists():
        try:
            watchlist = json.loads(WATCHLIST_PATH.read_text())
            for t in watchlist.get("tickers", []):
                if t not in tickers:
                    tickers.append(t)
        except Exception:
            pass
    return tickers


def main():
    parser = argparse.ArgumentParser(description="Backfill MIDGE signal archives")
    parser.add_argument("--days", type=int, default=90, help="Days of history (default: 90)")
    parser.add_argument("--sources", type=str, default="all",
                        help="Comma-separated sources: sec,congress,contracts,efts (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Count signals without writing")
    args = parser.parse_args()

    sources = ALL_SOURCES if args.sources == "all" else [s.strip() for s in args.sources.split(",")]

    logger.info("=" * 60)
    logger.info("MIDGE Archive Backfill — %d days, sources: %s", args.days, sources)
    logger.info("=" * 60)

    tickers = get_tickers()
    logger.info("Tickers for SEC Form 4: %d (%s...)", len(tickers), ", ".join(tickers[:5]))

    # Phase 1: Fetch
    logger.info("")
    logger.info("Phase 1: Fetching from %d sources...", len(sources))

    form4_trades = []
    house_trades = []
    senate_trades = []
    contracts = []
    efts_hits = []

    if "sec" in sources:
        form4_trades = fetch_sec_form4(tickers, args.days)

    if "congress" in sources:
        house_trades = fetch_house_trades(args.days)
        senate_trades = fetch_senate_trades(args.days)

    if "contracts" in sources:
        contracts = fetch_contracts(args.days)

    if "efts" in sources:
        efts_hits = fetch_efts_keywords(args.days)

    raw_total = len(form4_trades) + len(house_trades) + len(senate_trades) + len(contracts) + len(efts_hits)
    logger.info("")
    logger.info("Phase 1 complete: %d raw records fetched", raw_total)

    if raw_total == 0:
        logger.warning("No data fetched from any source. Check network connectivity.")
        return

    # Phase 2: Convert
    logger.info("")
    logger.info("Phase 2: Converting to MarketSignal format...")
    signals = convert_all(form4_trades, house_trades, senate_trades, contracts, efts_hits)
    logger.info("Phase 2 complete: %d unique signals (deduped from %d raw)", len(signals), raw_total)

    # Source breakdown
    source_counts: Dict[str, int] = defaultdict(int)
    for sig in signals:
        source_counts[sig.source] += 1
    for src, count in sorted(source_counts.items()):
        logger.info("  %s: %d signals", src, count)

    # Phase 3: Write
    logger.info("")
    logger.info("Phase 3: Writing to daily JSONL files...")
    written = write_signals(signals, dry_run=args.dry_run)

    total_written = sum(written.values())
    logger.info("")
    logger.info("=" * 60)
    logger.info("Backfill complete: %d signals written across %d days", total_written, len(written))
    logger.info("Archive directory: %s", SIGNALS_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
