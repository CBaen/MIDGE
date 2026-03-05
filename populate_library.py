#!/usr/bin/env python3
"""Populate the Pattern Library — first excavation pass.

Runs the ExcavationDaemon in a tight loop (no step-hook cadence) to
excavate all symbols in the signal archive.

Auto-detects Polygon.io paid API (MASSIVE_API_KEY) for fast bulk
excavation (~10-15 min for 3,000+ symbols). Falls back to yfinance
(~8-10 hours) when no API key is set.

Usage:
    python populate_library.py                    # Auto-detect source, all symbols
    python populate_library.py --batch-size 50    # Larger batches (faster)
    python populate_library.py --source polygon   # Force Polygon.io
    python populate_library.py --source yfinance  # Force yfinance
    python populate_library.py --symbols NVDA,AAPL,MSFT  # Specific symbols
    python populate_library.py --max-symbols 50   # Cap for testing
    python populate_library.py --reset            # Restart from scratch
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Load .env file for API keys
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _key, _, _val = _line.partition("=")
            os.environ.setdefault(_key.strip(), _val.strip())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("populate_library")


def main():
    parser = argparse.ArgumentParser(description="Populate MIDGE Pattern Library")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Symbols per batch (default: 10)")
    parser.add_argument("--symbols", type=str, default="",
                        help="Comma-separated symbol list (default: all from archive)")
    parser.add_argument("--max-symbols", type=int, default=0,
                        help="Max symbols to process (0 = unlimited)")
    parser.add_argument("--min-move", type=float, default=5.0,
                        help="Minimum %% move to qualify as dig site (default: 5.0)")
    parser.add_argument("--lookback", type=int, default=30,
                        help="Days to look back for precursor signals (default: 30)")
    parser.add_argument("--history-days", type=int, default=2000,
                        help="Days of price history to fetch (default: 2000)")
    parser.add_argument("--reset", action="store_true",
                        help="Reset progress and start from scratch")
    parser.add_argument("--source", type=str, default="auto",
                        choices=["auto", "polygon", "yfinance"],
                        help="Price data source (default: auto-detect)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be excavated without doing it")
    args = parser.parse_args()

    from mae_core.market.archaeology.pattern_library import PatternLibrary
    from mae_core.market.archaeology.excavator import Excavator
    from mae_core.market.archaeology.historical_fetcher import HistoricalDataFetcher
    from mae_core.market.archaeology.excavation_daemon import ExcavationDaemon
    from mae_core.market.apis.price_fetcher import PriceFetcher

    logger.info("Initializing excavation components...")

    library = PatternLibrary()
    fetcher = HistoricalDataFetcher()

    # Pre-load entire signal archive into memory (eliminates per-dig-site I/O)
    n_files = fetcher.preload_archive()
    logger.info("Signal archive: %d files pre-loaded into memory", n_files)

    excavator = Excavator(fetcher=fetcher)

    # Select price data source
    use_polygon = False
    if args.source == "polygon":
        use_polygon = True
    elif args.source == "auto" and os.environ.get("MASSIVE_API_KEY"):
        use_polygon = True

    if use_polygon:
        from mae_core.market.archaeology.polygon_bulk_fetcher import PolygonBulkFetcher
        price_fetcher = PolygonBulkFetcher()
        logger.info("Using Polygon.io paid API for bulk excavation (fast)")
    else:
        price_fetcher = PriceFetcher()
        logger.info("Using yfinance for excavation (slower, no API key needed)")

    daemon = ExcavationDaemon(
        library=library,
        excavator=excavator,
        price_fetcher=price_fetcher,
        batch_size=args.batch_size,
        min_move_pct=args.min_move,
        lookback_days=args.lookback,
        price_history_days=args.history_days,
    )

    if args.reset:
        daemon.reset_progress()
        logger.info("Progress reset — starting from scratch")

    # Override queue if specific symbols requested
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        daemon._queue_initialized = True
        daemon._symbols_queue = symbols
        daemon._symbols_done = set()
        logger.info("Targeting %d specific symbols: %s", len(symbols), ", ".join(symbols))

    # Dry run — show symbol count and exit
    if args.dry_run:
        daemon._initialize_queue() if not daemon._queue_initialized else None
        logger.info(
            "Dry run: %d symbols queued, %d already done, batch_size=%d",
            len(daemon._symbols_queue), len(daemon._symbols_done), args.batch_size,
        )
        if daemon._symbols_queue:
            preview = daemon._symbols_queue[:20]
            logger.info("First 20: %s", ", ".join(preview))
        return

    logger.info(
        "Starting excavation (batch_size=%d, min_move=%.1f%%, lookback=%dd, history=%dd)",
        args.batch_size, args.min_move, args.lookback, args.history_days,
    )

    start_time = time.time()
    total_fingerprints = 0
    total_templates = 0
    cycles = 0
    symbols_processed = 0

    while True:
        summary = daemon.step()

        if summary.get("status") == "idle":
            logger.info("Excavation complete — all symbols processed")
            break

        cycles += 1
        batch_fps = summary.get("fingerprints_found", 0)
        batch_templates = summary.get("new_templates", 0)
        total_fingerprints += batch_fps
        total_templates = summary.get("total_templates", total_templates)
        symbols_processed = summary.get("symbols_done", symbols_processed)
        remaining = summary.get("symbols_remaining", 0)

        if batch_fps > 0:
            logger.info(
                "Batch %d: +%d fingerprints, +%d templates | "
                "Total: %d fps, %d templates | %d/%d symbols",
                cycles, batch_fps, batch_templates,
                total_fingerprints, total_templates,
                symbols_processed, symbols_processed + remaining,
            )

        if args.max_symbols and symbols_processed >= args.max_symbols:
            logger.info("Reached max-symbols limit (%d)", args.max_symbols)
            break

    elapsed = time.time() - start_time
    logger.info(
        "Excavation finished in %.0fs: %d symbols, %d fingerprints, %d templates",
        elapsed, symbols_processed, total_fingerprints, total_templates,
    )
    logger.info("Library: %s", library._path)
    logger.info("Templates: %s", library._templates_path)


if __name__ == "__main__":
    main()
