"""Tests for Pattern Library — storage, querying, and stat tracking."""

import json
import pytest
from pathlib import Path

from mae_core.market.archaeology.fingerprint import MoveFingerprint, PrecursorSignal
from mae_core.market.archaeology.pattern_library import PatternLibrary


def _make_fingerprint(
    symbol="NVDA", direction="bullish", move_date="2024-01-15",
    move_pct=10.0, sources=None,
):
    if sources is None:
        sources = [("sec_form4", "insider"), ("fred_macro", "macro"), ("ta_rsi", "price")]
    return MoveFingerprint(
        fingerprint_id="",
        symbol=symbol,
        direction=direction,
        move_date=move_date,
        move_pct=move_pct,
        precursor_signals=[
            PrecursorSignal(src, dom, direction, 0.5, 3, "short")
            for src, dom in sources
        ],
    )


class TestPatternLibraryStorage:
    def test_store_and_retrieve(self, tmp_path):
        lib_path = tmp_path / "library.jsonl"
        lib = PatternLibrary(library_path=lib_path)
        fp = _make_fingerprint()
        assert lib.store(fp) is True
        assert lib.size == 1
        assert lib.get(fp.fingerprint_id) is not None

    def test_dedup(self, tmp_path):
        lib_path = tmp_path / "library.jsonl"
        lib = PatternLibrary(library_path=lib_path)
        fp = _make_fingerprint()
        assert lib.store(fp) is True
        assert lib.store(fp) is False  # Duplicate
        assert lib.size == 1

    def test_store_batch(self, tmp_path):
        lib_path = tmp_path / "library.jsonl"
        lib = PatternLibrary(library_path=lib_path)
        fps = [
            _make_fingerprint(move_date="2024-01-15"),
            _make_fingerprint(move_date="2024-02-15"),
            _make_fingerprint(move_date="2024-01-15"),  # Duplicate of first
        ]
        stored = lib.store_batch(fps)
        assert stored == 2
        assert lib.size == 2

    def test_persistence_across_loads(self, tmp_path):
        lib_path = tmp_path / "library.jsonl"
        lib1 = PatternLibrary(library_path=lib_path)
        fp = _make_fingerprint()
        lib1.store(fp)

        lib2 = PatternLibrary(library_path=lib_path)
        assert lib2.size == 1
        assert lib2.get(fp.fingerprint_id) is not None

    def test_empty_library(self, tmp_path):
        lib_path = tmp_path / "nonexistent.jsonl"
        lib = PatternLibrary(library_path=lib_path)
        assert lib.size == 0

    def test_update_outcome(self, tmp_path):
        lib_path = tmp_path / "library.jsonl"
        lib = PatternLibrary(library_path=lib_path)
        fp = _make_fingerprint()
        lib.store(fp)

        lib.update_outcome(fp.fingerprint_id, won=True)
        lib.update_outcome(fp.fingerprint_id, won=True)
        lib.update_outcome(fp.fingerprint_id, won=False)

        updated = lib.get(fp.fingerprint_id)
        assert updated.wins == 2
        assert updated.losses == 1
        assert updated.total_activations == 3


class TestPatternLibraryQuery:
    def test_query_matching_fingerprint(self, tmp_path):
        lib_path = tmp_path / "library.jsonl"
        lib = PatternLibrary(library_path=lib_path, match_threshold=0.5)
        fp = _make_fingerprint(sources=[
            ("sec_form4", "insider"),
            ("fred_macro", "macro"),
            ("ta_rsi", "price"),
        ])
        lib.store(fp)

        # All 3 sources present → should match
        matches = lib.query_similar(
            live_sources={"sec_form4", "fred_macro", "ta_rsi"},
            symbol="NVDA",
            direction="bullish",
        )
        assert len(matches) == 1
        assert matches[0].match_score >= 0.5

    def test_query_partial_match(self, tmp_path):
        lib_path = tmp_path / "library.jsonl"
        lib = PatternLibrary(library_path=lib_path, match_threshold=0.5)
        fp = _make_fingerprint(sources=[
            ("sec_form4", "insider"),
            ("fred_macro", "macro"),
            ("ta_rsi", "price"),
        ])
        lib.store(fp)

        # 2 of 3 sources present
        matches = lib.query_similar(
            live_sources={"sec_form4", "fred_macro"},
            symbol="NVDA",
            direction="bullish",
        )
        assert len(matches) == 1  # Should still match at 0.5+ threshold

    def test_query_no_match_wrong_direction(self, tmp_path):
        lib_path = tmp_path / "library.jsonl"
        lib = PatternLibrary(library_path=lib_path, match_threshold=0.5)
        fp = _make_fingerprint(direction="bullish")
        lib.store(fp)

        matches = lib.query_similar(
            live_sources={"sec_form4", "fred_macro", "ta_rsi"},
            symbol="NVDA",
            direction="bearish",  # Wrong direction
        )
        assert len(matches) == 0

    def test_query_no_match_wrong_symbol(self, tmp_path):
        lib_path = tmp_path / "library.jsonl"
        lib = PatternLibrary(library_path=lib_path, match_threshold=0.5)
        fp = _make_fingerprint(symbol="NVDA")
        lib.store(fp)

        matches = lib.query_similar(
            live_sources={"sec_form4", "fred_macro", "ta_rsi"},
            symbol="AAPL",  # Wrong symbol
            direction="bullish",
        )
        assert len(matches) == 0

    def test_query_below_threshold(self, tmp_path):
        lib_path = tmp_path / "library.jsonl"
        lib = PatternLibrary(library_path=lib_path, match_threshold=0.8)
        fp = _make_fingerprint(sources=[
            ("sec_form4", "insider"),
            ("fred_macro", "macro"),
            ("ta_rsi", "price"),
            ("ta_macd", "price"),
            ("congressional", "government"),
        ])
        lib.store(fp)

        # Only 1 of 5 sources present
        matches = lib.query_similar(
            live_sources={"sec_form4"},
            symbol="NVDA",
            direction="bullish",
        )
        assert len(matches) == 0  # Below 80% threshold


class TestPatternLibraryStats:
    def test_statistics(self, tmp_path):
        lib_path = tmp_path / "library.jsonl"
        lib = PatternLibrary(library_path=lib_path)
        lib.store(_make_fingerprint(direction="bullish", move_date="2024-01-15"))
        lib.store(_make_fingerprint(direction="bearish", move_date="2024-02-15"))

        stats = lib.get_statistics()
        assert stats["total"] == 2
        assert stats["bullish"] == 1
        assert stats["bearish"] == 1

    def test_clopper_pearson_ci(self, tmp_path):
        lib_path = tmp_path / "library.jsonl"
        lib = PatternLibrary(library_path=lib_path)
        fp = _make_fingerprint()
        lib.store(fp)

        # No data yet
        lo, hi = lib.clopper_pearson_ci(fp.fingerprint_id)
        assert lo == 0.0
        assert hi == 1.0

        # Add some outcomes
        for _ in range(7):
            lib.update_outcome(fp.fingerprint_id, won=True)
        for _ in range(3):
            lib.update_outcome(fp.fingerprint_id, won=False)

        lo, hi = lib.clopper_pearson_ci(fp.fingerprint_id)
        assert 0.3 < lo < 0.7  # 70% WR, CI should be reasonable
        assert 0.7 < hi < 1.0
