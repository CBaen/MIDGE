#!/usr/bin/env python3
"""Tests for mae_core.market.intelligence.signal_archive_reader.

All tests are self-contained — no real archive files are read.
Synthetic JSONL files are written to a temporary directory and
SignalArchiveReader is constructed with that directory.
"""

import json
import os
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path

from mae_core.market.intelligence.signal_archive_reader import (
    ArchiveRecord,
    SignalArchiveReader,
)


# ---------------------------------------------------------------------------
# Synthetic fixture builder
# ---------------------------------------------------------------------------

def _make_record(
    signal_id="sig1",
    source="sec_form4",
    symbol="AAPL",
    domain="insider",
    direction="bullish",
    strength=0.8,
    confidence=0.7,
    velocity=0.1,
    timestamp="2026-02-20T10:00:00",
    metadata=None,
):
    return {
        "signal_id": signal_id,
        "source": source,
        "symbol": symbol,
        "domain": domain,
        "direction": direction,
        "strength": strength,
        "confidence": confidence,
        "velocity": velocity,
        "timestamp": timestamp,
        "metadata": metadata or {},
    }


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSignalArchiveReader(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.archive_dir = Path(self.tmpdir)

        # Day 1: three records — two sec_form4, one congressional
        day1_records = [
            _make_record("sig1", "sec_form4", "AAPL", timestamp="2026-02-20T10:00:00", strength=0.8),
            _make_record("sig2", "congressional", "MSFT", domain="government", strength=0.6,
                         confidence=0.5, velocity=0.0, timestamp="2026-02-20T11:00:00"),
            _make_record("sig3", "sec_form4", "GOOGL", direction="bearish", strength=0.3,
                         confidence=0.4, velocity=0.0, timestamp="2026-02-20T12:00:00"),
        ]
        _write_jsonl(self.archive_dir / "2026-02-20.jsonl", day1_records)

        # Day 2: one record — sec_form4 on AAPL
        day2_records = [
            _make_record("sig4", "sec_form4", "AAPL", strength=0.9, confidence=0.8,
                         velocity=0.2, timestamp="2026-02-21T09:00:00"),
        ]
        _write_jsonl(self.archive_dir / "2026-02-21.jsonl", day2_records)

    def _make_reader(self):
        return SignalArchiveReader(archive_dir=self.archive_dir)

    # ------------------------------------------------------------------
    # load_range
    # ------------------------------------------------------------------

    def test_load_range_empty_dir(self):
        """load_range on a dir with no matching files returns 0."""
        empty_dir = Path(tempfile.mkdtemp())
        reader = SignalArchiveReader(archive_dir=empty_dir)
        count = reader.load_range(date(2026, 2, 20), date(2026, 2, 21))
        self.assertEqual(count, 0)

    def test_load_range_loads_records(self):
        """Loading two days that have files returns the total record count."""
        reader = self._make_reader()
        count = reader.load_range(date(2026, 2, 20), date(2026, 2, 21))
        self.assertEqual(count, 4)  # 3 on day1 + 1 on day2

    def test_load_range_idempotent(self):
        """Loading the same range a second time returns 0 — no duplicates."""
        reader = self._make_reader()
        reader.load_range(date(2026, 2, 20), date(2026, 2, 21))
        second_load = reader.load_range(date(2026, 2, 20), date(2026, 2, 21))
        self.assertEqual(second_load, 0)

    def test_load_range_partial_overlap(self):
        """A partially overlapping second load only reads the new day."""
        reader = self._make_reader()
        reader.load_range(date(2026, 2, 20), date(2026, 2, 20))  # day1 only
        second = reader.load_range(date(2026, 2, 20), date(2026, 2, 21))  # adds day2
        self.assertEqual(second, 1)  # Only the day2 record is new

    # ------------------------------------------------------------------
    # query_source
    # ------------------------------------------------------------------

    def test_query_source_returns_matching(self):
        """Querying by source returns all records for that source."""
        reader = self._make_reader()
        reader.load_range(date(2026, 2, 20), date(2026, 2, 21))
        results = reader.query_source("sec_form4")
        self.assertEqual(len(results), 3)  # sig1, sig3, sig4

    def test_query_source_with_symbol_filter(self):
        """Source + symbol filter returns only matching records."""
        reader = self._make_reader()
        reader.load_range(date(2026, 2, 20), date(2026, 2, 21))
        results = reader.query_source("sec_form4", symbol="AAPL")
        self.assertEqual(len(results), 2)  # sig1 and sig4
        for r in results:
            self.assertEqual(r.symbol, "AAPL")
            self.assertEqual(r.source, "sec_form4")

    def test_query_source_with_date_filter(self):
        """Date-range filter excludes records outside the range."""
        reader = self._make_reader()
        reader.load_range(date(2026, 2, 20), date(2026, 2, 21))
        # Only day1 range — should return sig1 and sig3 (both sec_form4 on 2026-02-20)
        results = reader.query_source(
            "sec_form4",
            start=date(2026, 2, 20),
            end=date(2026, 2, 20),
        )
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertEqual(r.timestamp.date(), date(2026, 2, 20))

    def test_query_source_unknown_returns_empty(self):
        """Querying a source that does not exist returns an empty list."""
        reader = self._make_reader()
        reader.load_range(date(2026, 2, 20), date(2026, 2, 21))
        results = reader.query_source("nonexistent_source")
        self.assertEqual(results, [])

    # ------------------------------------------------------------------
    # query_symbol
    # ------------------------------------------------------------------

    def test_query_symbol_returns_cross_source(self):
        """query_symbol returns records from all sources for that ticker."""
        reader = self._make_reader()
        reader.load_range(date(2026, 2, 20), date(2026, 2, 21))
        results = reader.query_symbol("AAPL")
        self.assertEqual(len(results), 2)  # sig1 + sig4
        for r in results:
            self.assertEqual(r.symbol, "AAPL")

    def test_query_symbol_sorted_by_timestamp(self):
        """Results from query_symbol are sorted ascending by timestamp."""
        reader = self._make_reader()
        reader.load_range(date(2026, 2, 20), date(2026, 2, 21))
        results = reader.query_symbol("AAPL")
        timestamps = [r.timestamp for r in results]
        self.assertEqual(timestamps, sorted(timestamps))

    # ------------------------------------------------------------------
    # get_timeseries
    # ------------------------------------------------------------------

    def test_get_timeseries_sorted(self):
        """get_timeseries returns (datetime, float) pairs sorted by timestamp."""
        reader = self._make_reader()
        reader.load_range(date(2026, 2, 20), date(2026, 2, 21))
        ts = reader.get_timeseries("sec_form4", field="strength")
        self.assertEqual(len(ts), 3)
        # Each element is a (datetime, float) tuple
        for item in ts:
            self.assertIsInstance(item[0], datetime)
            self.assertIsInstance(item[1], float)
        # Ascending order
        timestamps = [item[0] for item in ts]
        self.assertEqual(timestamps, sorted(timestamps))

    def test_get_timeseries_values_correct(self):
        """Strength values in timeseries match the written records."""
        reader = self._make_reader()
        reader.load_range(date(2026, 2, 20), date(2026, 2, 21))
        ts = reader.get_timeseries("sec_form4", field="strength")
        strengths = [item[1] for item in ts]
        # sig1=0.8, sig3=0.3, sig4=0.9 (sorted by timestamp)
        self.assertAlmostEqual(strengths[0], 0.8)
        self.assertAlmostEqual(strengths[1], 0.3)
        self.assertAlmostEqual(strengths[2], 0.9)

    # ------------------------------------------------------------------
    # available_sources / available_symbols
    # ------------------------------------------------------------------

    def test_available_sources(self):
        """available_sources returns sorted list of all loaded sources."""
        reader = self._make_reader()
        reader.load_range(date(2026, 2, 20), date(2026, 2, 21))
        sources = reader.available_sources()
        self.assertEqual(sources, ["congressional", "sec_form4"])

    def test_available_symbols(self):
        """available_symbols returns sorted list of all loaded symbols."""
        reader = self._make_reader()
        reader.load_range(date(2026, 2, 20), date(2026, 2, 21))
        symbols = reader.available_symbols()
        self.assertIn("AAPL", symbols)
        self.assertIn("MSFT", symbols)
        self.assertIn("GOOGL", symbols)
        self.assertEqual(symbols, sorted(symbols))

    # ------------------------------------------------------------------
    # get_statistics
    # ------------------------------------------------------------------

    def test_get_statistics_structure(self):
        """get_statistics returns a dict with the expected top-level keys."""
        reader = self._make_reader()
        reader.load_range(date(2026, 2, 20), date(2026, 2, 21))
        stats = reader.get_statistics()
        self.assertIn("total_records", stats)
        self.assertIn("loaded_dates", stats)
        self.assertIn("sources", stats)
        self.assertIn("symbols", stats)

    def test_get_statistics_counts(self):
        """get_statistics reflects the actual number of loaded records."""
        reader = self._make_reader()
        reader.load_range(date(2026, 2, 20), date(2026, 2, 21))
        stats = reader.get_statistics()
        self.assertEqual(stats["total_records"], 4)
        self.assertEqual(stats["loaded_dates"], 2)
        self.assertEqual(stats["sources"]["sec_form4"], 3)
        self.assertEqual(stats["sources"]["congressional"], 1)

    # ------------------------------------------------------------------
    # Malformed input handling
    # ------------------------------------------------------------------

    def test_malformed_json_line_is_skipped(self):
        """A JSONL file with one bad line still loads the valid records."""
        mixed_path = self.archive_dir / "2026-02-22.jsonl"
        with open(mixed_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(_make_record("sig5", "sec_form4", "TSLA",
                                            timestamp="2026-02-22T08:00:00")) + "\n")
            f.write("THIS IS NOT JSON\n")  # malformed
            f.write(json.dumps(_make_record("sig6", "congressional", "NVDA",
                                            timestamp="2026-02-22T09:00:00",
                                            domain="government")) + "\n")

        reader = self._make_reader()
        # Load only the new day to isolate the test
        count = reader.load_range(date(2026, 2, 22), date(2026, 2, 22))
        self.assertEqual(count, 2)  # Bad line is silently skipped

    def test_save_is_noop(self):
        """save() should not raise — it is a no-op on the reader."""
        reader = self._make_reader()
        reader.load_range(date(2026, 2, 20), date(2026, 2, 20))
        reader.save()  # Must not raise


if __name__ == "__main__":
    unittest.main()
