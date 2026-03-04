#!/usr/bin/env python3
"""Excavate — offline CLI for mining historical price moves into pattern fingerprints.

Usage:
    python excavate.py --symbols NVDA,AAPL,MSFT --lookback 30 --min-move 5
    python excavate.py --all-archived --lookback 30 --min-move 7
    python excavate.py --symbols NVDA --lookback 30 --min-move 10 --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from mae_core.market.archaeology.dig_sites import DigSite, find_dig_sites
from mae_core.market.archaeology.excavator import Excavator
from mae_core.market.archaeology.fingerprint import MoveFingerprint
from mae_core.market.apis.price_fetcher import PriceFetcher

logger = logging.getLogger("midge.excavate")
SIGNAL_DIR = Path(__file__).parent / "data" / "midge" / "signals"
LIBRARY_PATH = Path(__file__).parent / "data" / "market" / "pattern_library.jsonl"


def get_archived_symbols() -> list[str]:
    """Extract unique symbols from signal archive filenames and contents."""
    symbols: set[str] = set()
    signal_dir = SIGNAL_DIR
    if not signal_dir.exists():
        return []

    # Sample a few files to extract symbols
    files = sorted(signal_dir.glob("*.jsonl"))
    sample_files = files[-30:] if len(files) > 30 else files  # Recent files

    for f in sample_files:
        try:
            with open(f, "r") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        sym = record.get("symbol", "")
                        if sym and len(sym) <= 6 and sym.isalpha():
                            symbols.add(sym)
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue

    return sorted(symbols)


def run_excavation(
    symbols: list[str],
    lookback_days: int = 30,
    min_move_pct: float = 5.0,
    price_history_days: int = 5000,
    dry_run: bool = False,
) -> list[MoveFingerprint]:
    """Run the full excavation pipeline for a list of symbols."""
    fetcher = PriceFetcher()
    excavator = Excavator()
    all_fingerprints: list[MoveFingerprint] = []

    for symbol in symbols:
        logger.info("=" * 60)
        logger.info("Excavating %s (lookback=%dd, min_move=%.1f%%)", symbol, lookback_days, min_move_pct)
        logger.info("=" * 60)

        # Step 1: Get price history
        try:
            history = fetcher.get_daily_history(symbol, days=price_history_days)
        except Exception as e:
            logger.warning("Could not fetch price history for %s: %s", symbol, e)
            continue

        if len(history) < 30:
            logger.warning("Insufficient price history for %s (%d points)", symbol, len(history))
            continue

        # Step 2: Find dig sites
        sites = find_dig_sites(history, symbol, min_single_day_pct=min_move_pct)
        logger.info("Found %d dig sites for %s", len(sites), symbol)

        if dry_run:
            for site in sites:
                logger.info(
                    "  DIG SITE: %s %s on %s (%.1f%% over %d day(s))",
                    site.symbol, site.direction, site.move_date,
                    site.move_pct, site.move_days,
                )
            continue

        # Step 3: Excavate each site
        fingerprints = excavator.excavate_batch(sites, lookback_days)
        all_fingerprints.extend(fingerprints)

        for fp in fingerprints:
            logger.info(
                "  FINGERPRINT: %s %s %s | %.1f%% | %d precursors | %s",
                fp.symbol, fp.direction, fp.move_date,
                fp.move_pct, len(fp.precursor_signals), fp.domain_signature,
            )

        # Clear cache between symbols to manage memory
        excavator.clear_cache()

    return all_fingerprints


def save_fingerprints(fingerprints: list[MoveFingerprint], output_path: Path) -> None:
    """Append fingerprints to the pattern library JSONL file."""
    # Load existing IDs for dedup
    existing_ids: set[str] = set()
    if output_path.exists():
        with open(output_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        existing_ids.add(record.get("fingerprint_id", ""))
                    except json.JSONDecodeError:
                        continue

    new_count = 0
    with open(output_path, "a") as f:
        for fp in fingerprints:
            if fp.fingerprint_id in existing_ids:
                logger.debug("Skipping duplicate fingerprint: %s", fp.fingerprint_id)
                continue
            f.write(fp.to_json() + "\n")
            existing_ids.add(fp.fingerprint_id)
            new_count += 1

    logger.info("Saved %d new fingerprints to %s (%d duplicates skipped)",
                new_count, output_path, len(fingerprints) - new_count)


def main():
    parser = argparse.ArgumentParser(description="Excavate historical price moves into pattern fingerprints")
    parser.add_argument("--symbols", type=str, help="Comma-separated symbols (e.g. NVDA,AAPL,MSFT)")
    parser.add_argument("--all-archived", action="store_true", help="Excavate all symbols found in signal archive")
    parser.add_argument("--lookback", type=int, default=30, help="Days to look back for precursor signals (default: 30)")
    parser.add_argument("--min-move", type=float, default=5.0, help="Minimum move percentage for dig sites (default: 5.0)")
    parser.add_argument("--history-days", type=int, default=5000, help="Days of price history to fetch (default: 5000)")
    parser.add_argument("--output", type=str, default=str(LIBRARY_PATH), help="Output JSONL path")
    parser.add_argument("--dry-run", action="store_true", help="Show dig sites without excavating")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)-35s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Determine symbols
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    elif args.all_archived:
        symbols = get_archived_symbols()
        logger.info("Found %d symbols in signal archive", len(symbols))
    else:
        parser.error("Specify --symbols or --all-archived")
        return

    if not symbols:
        logger.error("No symbols to excavate")
        return

    logger.info("=" * 60)
    logger.info("MIDGE Pattern Excavation")
    logger.info("  Symbols: %s", ", ".join(symbols[:10]) + (f" (+{len(symbols)-10} more)" if len(symbols) > 10 else ""))
    logger.info("  Lookback: %d days", args.lookback)
    logger.info("  Min move: %.1f%%", args.min_move)
    logger.info("  Dry run: %s", args.dry_run)
    logger.info("=" * 60)

    fingerprints = run_excavation(
        symbols=symbols,
        lookback_days=args.lookback,
        min_move_pct=args.min_move,
        price_history_days=args.history_days,
        dry_run=args.dry_run,
    )

    if fingerprints and not args.dry_run:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_fingerprints(fingerprints, output_path)

    # Summary
    if not args.dry_run:
        logger.info("=" * 60)
        logger.info("EXCAVATION COMPLETE")
        logger.info("  Total fingerprints: %d", len(fingerprints))
        if fingerprints:
            bullish = sum(1 for fp in fingerprints if fp.direction == "bullish")
            bearish = len(fingerprints) - bullish
            avg_precursors = sum(len(fp.precursor_signals) for fp in fingerprints) / len(fingerprints)
            logger.info("  Bullish: %d | Bearish: %d", bullish, bearish)
            logger.info("  Avg precursor signals: %.1f", avg_precursors)
            # Domain signature distribution
            sigs: dict[str, int] = {}
            for fp in fingerprints:
                sigs[fp.domain_signature] = sigs.get(fp.domain_signature, 0) + 1
            for sig, count in sorted(sigs.items(), key=lambda x: -x[1])[:10]:
                logger.info("  %s: %d", sig, count)
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
