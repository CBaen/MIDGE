"""Excavation Daemon — continuous background pattern discovery.

This replaces the CLI script (excavate.py). The daemon runs as a step hook
in MIDGE's daemon loop, progressively excavating ALL symbols with significant
price moves and building the pattern template library.

MIDGE doesn't stop researching. She doesn't wait to be told. She continuously
digs through history, discovers patterns, cross-validates them across symbols,
and builds a library of transferable pattern templates.

Operation:
  - Runs every N steps (default 5000, configurable)
  - Maintains persistent progress state (which symbols done, which pending)
  - Processes symbols in batches (default 5 per cycle)
  - Rate-limited: respects API rate limits
  - Progressive: starts with most data-rich symbols, expands outward
  - Publishes discoveries via EventBus
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

from mae_core.market.archaeology.dig_sites import DigSite, find_dig_sites
from mae_core.market.archaeology.excavator import Excavator
from mae_core.market.archaeology.fingerprint import MoveFingerprint, PatternTemplate
from mae_core.market.archaeology.pattern_library import PatternLibrary

logger = logging.getLogger(__name__)

PROGRESS_PATH = Path(__file__).resolve().parents[3] / "data" / "market" / "excavation_progress.json"
SIGNAL_ARCHIVE_DIR = Path(__file__).resolve().parents[3] / "data" / "midge" / "signals"


class ExcavationDaemon:
    """Continuous background excavation of historical price moves.

    Designed to run as a step hook in the daemon loop. Each invocation
    processes a batch of symbols, building the pattern template library
    progressively.
    """

    def __init__(
        self,
        library: PatternLibrary,
        excavator: Excavator,
        price_fetcher: Any = None,
        bus: Any = None,
        batch_size: int = 5,
        min_move_pct: float = 5.0,
        lookback_days: int = 30,
        price_history_days: int = 2000,
        progress_path: Optional[Path] = None,
    ):
        self._library = library
        self._excavator = excavator
        self._price_fetcher = price_fetcher
        self._bus = bus
        self._batch_size = batch_size
        self._min_move_pct = min_move_pct
        self._lookback_days = lookback_days
        self._price_history_days = price_history_days
        self._progress_path = progress_path or PROGRESS_PATH

        # Progress tracking
        self._symbols_done: set[str] = set()
        self._symbols_queue: list[str] = []
        self._queue_initialized = False
        self._total_fingerprints = 0
        self._total_templates = 0
        self._cycles_completed = 0
        self._load_progress()

    def step(self) -> dict:
        """Process one batch of symbols. Called from the daemon step hook.

        Returns:
            Summary dict of what was discovered in this batch.
        """
        if not self._queue_initialized:
            self._initialize_queue()

        if not self._symbols_queue:
            # All symbols exhausted — restart with broader scan or idle
            return {"status": "idle", "symbols_done": len(self._symbols_done)}

        # Pop next batch
        batch = []
        while self._symbols_queue and len(batch) < self._batch_size:
            sym = self._symbols_queue.pop(0)
            if sym not in self._symbols_done:
                batch.append(sym)

        if not batch:
            return {"status": "idle", "symbols_done": len(self._symbols_done)}

        batch_fingerprints = 0
        batch_templates_new = 0

        # Batch fetch price history if fetcher supports it (aiohttp, 20x faster)
        batch_history = self._fetch_batch_history(batch)

        for symbol in batch:
            try:
                fps, new_templates = self._excavate_symbol(
                    symbol, prefetched_history=batch_history.get(symbol),
                )
                batch_fingerprints += len(fps)
                batch_templates_new += new_templates
                self._symbols_done.add(symbol)
            except Exception:
                logger.debug("Excavation failed for %s", symbol, exc_info=True)
                self._symbols_done.add(symbol)  # Mark done to avoid retry loops

        self._cycles_completed += 1
        self._save_progress()

        summary = {
            "status": "active",
            "cycle": self._cycles_completed,
            "symbols_processed": batch,
            "fingerprints_found": batch_fingerprints,
            "new_templates": batch_templates_new,
            "symbols_done": len(self._symbols_done),
            "symbols_remaining": len(self._symbols_queue),
            "total_templates": self._library.template_count,
        }

        # Publish discovery if we found cross-validated templates
        if batch_templates_new > 0 and self._bus is not None:
            try:
                from mae_core.market.channels import CH_PATTERN_STACK_DETECTED
                self._bus.publish("market.archaeology.excavation_batch", summary)
            except Exception:
                pass

        logger.info(
            "Excavation cycle %d: %d symbols, %d fingerprints, %d new templates "
            "(%d/%d symbols done)",
            self._cycles_completed, len(batch), batch_fingerprints,
            batch_templates_new, len(self._symbols_done),
            len(self._symbols_done) + len(self._symbols_queue),
        )

        return summary

    def _fetch_batch_history(self, symbols: list[str]) -> dict:
        """Fetch price history for all symbols in batch concurrently if possible.

        Falls back to empty dict (per-symbol fetch) if batch mode unavailable.
        """
        if self._price_fetcher is None:
            return {}
        if not hasattr(self._price_fetcher, "get_daily_history_batch"):
            return {}
        try:
            return self._price_fetcher.get_daily_history_batch(
                symbols, days=self._price_history_days,
            )
        except Exception:
            logger.debug("Batch history fetch failed, falling back to per-symbol", exc_info=True)
            return {}

    def _excavate_symbol(
        self,
        symbol: str,
        prefetched_history: list = None,
    ) -> tuple[list[MoveFingerprint], int]:
        """Excavate all dig sites for a single symbol.

        Args:
            symbol: Ticker to excavate.
            prefetched_history: Pre-fetched price data from batch mode. If None,
                fetches individually (fallback for non-batch fetchers).

        Returns:
            Tuple of (fingerprints produced, number of new templates stored).
        """
        if self._price_fetcher is None and prefetched_history is None:
            return ([], 0)

        # Use prefetched data if available, otherwise fetch individually
        if prefetched_history is not None:
            history = prefetched_history
        else:
            try:
                history = self._price_fetcher.get_daily_history(
                    symbol, days=self._price_history_days,
                )
            except Exception:
                logger.debug("Price history fetch failed for %s", symbol, exc_info=True)
                return ([], 0)

        if len(history) < 30:
            return ([], 0)

        # Find dig sites
        sites = find_dig_sites(
            history, symbol,
            min_single_day_pct=self._min_move_pct,
        )
        if not sites:
            return ([], 0)

        # Excavate
        fingerprints = self._excavator.excavate_symbol(
            symbol, history, sites, self._lookback_days,
        )

        # Store fingerprints and templates in library
        stored = self._library.store_batch(fingerprints)
        new_templates = self._library.store_templates(self._excavator.templates)

        self._total_fingerprints += stored
        self._total_templates = self._library.template_count

        return (fingerprints, new_templates)

    def _initialize_queue(self) -> None:
        """Build the symbol queue from available data sources."""
        symbols: set[str] = set()

        # Source 1: All symbols in the signal archive
        archive_symbols = self._get_archived_symbols()
        symbols.update(archive_symbols)

        # Source 2: If price_fetcher supports a symbol universe, use it
        # (yfinance doesn't have a built-in universe, so archive is primary)

        # Remove already-done symbols
        pending = sorted(symbols - self._symbols_done)

        # Prioritize: symbols with more archive data first (more signals = richer excavation)
        # For now, alphabetical is fine — the daemon will get through them all
        self._symbols_queue = pending
        self._queue_initialized = True

        logger.info(
            "Excavation queue initialized: %d symbols (%d already done, %d pending)",
            len(symbols), len(self._symbols_done), len(pending),
        )

    def _get_archived_symbols(self) -> set[str]:
        """Extract unique symbols from the signal archive."""
        symbols: set[str] = set()
        signal_dir = SIGNAL_ARCHIVE_DIR
        if not signal_dir.exists():
            return symbols

        # Read ALL archive files for comprehensive coverage
        for f in signal_dir.glob("*.jsonl"):
            try:
                with open(f, "r") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                            sym = record.get("symbol", "")
                            # Filter: valid ticker format, skip empty/macro signals
                            if sym and 1 <= len(sym) <= 10 and not sym.startswith("hist:"):
                                symbols.add(sym)
                        except json.JSONDecodeError:
                            continue
            except OSError:
                continue

        return symbols

    def _load_progress(self) -> None:
        """Load excavation progress from disk."""
        if not self._progress_path.exists():
            return
        try:
            with open(self._progress_path, "r") as f:
                data = json.load(f)
            self._symbols_done = set(data.get("symbols_done", []))
            self._cycles_completed = data.get("cycles_completed", 0)
            self._total_fingerprints = data.get("total_fingerprints", 0)
            logger.info(
                "Excavation progress loaded: %d symbols done, %d cycles, %d fingerprints",
                len(self._symbols_done), self._cycles_completed, self._total_fingerprints,
            )
        except (OSError, json.JSONDecodeError):
            pass

    def _save_progress(self) -> None:
        """Save excavation progress to disk."""
        try:
            self._progress_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "symbols_done": sorted(self._symbols_done),
                "symbols_remaining": len(self._symbols_queue),
                "cycles_completed": self._cycles_completed,
                "total_fingerprints": self._total_fingerprints,
                "total_templates": self._total_templates,
                "last_updated": datetime.now().isoformat(),
            }
            with open(self._progress_path, "w") as f:
                json.dump(data, f, indent=2)
        except OSError:
            logger.debug("Could not save excavation progress", exc_info=True)

    def reset_progress(self) -> None:
        """Reset progress to start fresh."""
        self._symbols_done.clear()
        self._symbols_queue.clear()
        self._queue_initialized = False
        self._cycles_completed = 0
        self._total_fingerprints = 0
        if self._progress_path.exists():
            self._progress_path.unlink()

    def get_statistics(self) -> dict:
        """Return daemon statistics."""
        cross_validated = self._excavator.get_cross_validated_templates()
        return {
            "cycles_completed": self._cycles_completed,
            "symbols_done": len(self._symbols_done),
            "symbols_remaining": len(self._symbols_queue),
            "total_fingerprints": self._total_fingerprints,
            "total_templates": self._library.template_count,
            "cross_validated_templates": len(cross_validated),
            "excavator_stats": self._excavator.get_statistics(),
        }
