"""SignalArchiveReader — indexed read interface over data/midge/signals/*.jsonl.

The write side lives in sensing_hook.py:_store_signals(). This module is
the read side: loads date-partitioned JSONL files, builds an in-memory
index by (source, symbol), and provides query methods used by
LagCorrelationAnalyzer and ThompsonCalibrator.

Does NOT write anything. Does NOT duplicate _store_signals.

Biological analogy: episodic memory recall — the organism remembering
what it sensed on a particular day, indexed for rapid retrieval.
"""

import json
import logging
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ARCHIVE_DIR = Path(__file__).resolve().parents[3] / "data" / "midge" / "signals"


def _parse_dt(s: str) -> datetime:
    """Parse ISO timestamp, tolerant of various formats."""
    if not s:
        return datetime.min
    try:
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return datetime.min


class ArchiveRecord:
    """Lightweight parsed signal record from JSONL archive."""
    __slots__ = (
        "signal_id", "source", "symbol", "domain", "direction",
        "strength", "confidence", "velocity", "timestamp", "metadata",
    )

    def __init__(self, d: dict):
        self.signal_id = d.get("signal_id", "")
        self.source = d.get("source", "unknown")
        self.symbol = d.get("symbol", "")
        self.domain = d.get("domain", "")
        self.direction = d.get("direction", "neutral")
        self.strength = float(d.get("strength", 0.0))
        self.confidence = float(d.get("confidence", 0.5))
        self.velocity = float(d.get("velocity", 0.0))
        self.timestamp = _parse_dt(d.get("timestamp", ""))
        self.metadata = d.get("metadata", {})


class SignalArchiveReader:
    """
    Read interface over data/midge/signals/*.jsonl.

    Loads archive files on first access. Index is built in memory:
      _by_source: source -> list of ArchiveRecord
      _by_symbol: symbol -> list of ArchiveRecord
      _loaded_dates: set of date strings already loaded

    Lazy loading: call load_range(start, end) before querying.
    """

    def __init__(self, archive_dir: Optional[Path] = None):
        self._archive_dir = archive_dir or ARCHIVE_DIR
        self._by_source: Dict[str, List[ArchiveRecord]] = defaultdict(list)
        self._by_symbol: Dict[str, List[ArchiveRecord]] = defaultdict(list)
        self._loaded_dates: set = set()
        self._total_records: int = 0

    def load_range(self, start: date, end: date) -> int:
        """Load all archive files covering [start, end] into memory.

        Returns count of new records loaded. Idempotent — skips
        already-loaded dates.
        """
        loaded = 0
        current = start
        while current <= end:
            date_str = current.isoformat()
            if date_str not in self._loaded_dates:
                fname = self._archive_dir / f"{date_str}.jsonl"
                if fname.exists():
                    loaded += self._load_file(fname, date_str)
                else:
                    # Mark as loaded even if file doesn't exist to avoid re-checking
                    self._loaded_dates.add(date_str)
            current += timedelta(days=1)
        self._total_records += loaded
        return loaded

    def _load_file(self, path: Path, date_str: str) -> int:
        """Load a single JSONL file and index its records."""
        count = 0
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = ArchiveRecord(json.loads(line))
                        self._by_source[rec.source].append(rec)
                        if rec.symbol:
                            self._by_symbol[rec.symbol].append(rec)
                        count += 1
                    except (json.JSONDecodeError, KeyError, TypeError):
                        pass
            self._loaded_dates.add(date_str)
        except Exception as e:
            logger.warning("Could not load archive file %s: %s", path, e)
        return count

    def query_source(
        self,
        source: str,
        start: Optional[date] = None,
        end: Optional[date] = None,
        symbol: Optional[str] = None,
    ) -> List[ArchiveRecord]:
        """Return records for a source, optionally filtered by date range and symbol.

        Args:
            source: Signal source key ("sec_form4", "congressional", etc.)
            start: Inclusive start date
            end: Inclusive end date
            symbol: Optional ticker filter

        Returns:
            List of ArchiveRecord sorted by timestamp
        """
        records = self._by_source.get(source, [])
        if start:
            records = [r for r in records if r.timestamp.date() >= start]
        if end:
            records = [r for r in records if r.timestamp.date() <= end]
        if symbol:
            records = [r for r in records if r.symbol == symbol]
        return sorted(records, key=lambda r: r.timestamp)

    def query_symbol(
        self,
        symbol: str,
        start: Optional[date] = None,
        end: Optional[date] = None,
    ) -> List[ArchiveRecord]:
        """Return all records for a symbol across all sources."""
        records = self._by_symbol.get(symbol, [])
        if start:
            records = [r for r in records if r.timestamp.date() >= start]
        if end:
            records = [r for r in records if r.timestamp.date() <= end]
        return sorted(records, key=lambda r: r.timestamp)

    def get_timeseries(
        self,
        source: str,
        field: str = "strength",
        start: Optional[date] = None,
        end: Optional[date] = None,
    ) -> List[Tuple[datetime, float]]:
        """Return (timestamp, value) pairs for a source's field over time.

        Used by LagCorrelationAnalyzer to build time series for cross-correlation.

        Args:
            source: Signal source key
            field: "strength", "confidence", or "velocity"
            start: Inclusive start date
            end: Inclusive end date
        """
        records = self.query_source(source, start=start, end=end)
        result = []
        for r in records:
            val = getattr(r, field, 0.0)
            result.append((r.timestamp, float(val)))
        return result

    def available_sources(self) -> List[str]:
        """List all sources present in loaded archives."""
        return sorted(self._by_source.keys())

    def available_symbols(self) -> List[str]:
        """List all symbols present in loaded archives."""
        return sorted(self._by_symbol.keys())

    def get_statistics(self) -> dict:
        """HolonProxy.sense() delegation."""
        return {
            "total_records": self._total_records,
            "loaded_dates": len(self._loaded_dates),
            "sources": {src: len(recs) for src, recs in self._by_source.items()},
            "symbols": len(self._by_symbol),
        }

    def save(self):
        """No-op: archive reader has no mutable state to persist."""
        pass
