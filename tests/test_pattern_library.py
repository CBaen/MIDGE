"""Tests for Pattern Library — storage, querying, and stat tracking."""

import json
import pytest
from pathlib import Path

from mae_core.market.archaeology.fingerprint import MoveFingerprint, PrecursorSignal, PatternTemplate
from mae_core.market.archaeology.pattern_library import PatternLibrary


def _make_fingerprint(
    symbol="NVDA", direction="bullish", move_date="2024-01-15",
    move_pct=10.0, sources=None,
):
    """Helper to create fingerprints with domain-consistent source mappings.

    Uses domains matching PatternLibrary._SOURCE_DOMAIN_MAP:
      sec_form4 → insider, fred_macro → macro, ta_rsi → technical
    """
    if sources is None:
        sources = [("sec_form4", "insider"), ("fred_macro", "macro"), ("ta_rsi", "technical")]
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
        lib = PatternLibrary(library_path=lib_path, templates_path=tmp_path / "templates.jsonl")
        fp = _make_fingerprint()
        assert lib.store(fp) is True
        assert lib.size == 1
        assert lib.get(fp.fingerprint_id) is not None

    def test_dedup(self, tmp_path):
        lib_path = tmp_path / "library.jsonl"
        lib = PatternLibrary(library_path=lib_path, templates_path=tmp_path / "templates.jsonl")
        fp = _make_fingerprint()
        assert lib.store(fp) is True
        assert lib.store(fp) is False  # Duplicate
        assert lib.size == 1

    def test_store_batch(self, tmp_path):
        lib_path = tmp_path / "library.jsonl"
        lib = PatternLibrary(library_path=lib_path, templates_path=tmp_path / "templates.jsonl")
        fps = [
            _make_fingerprint(move_date="2024-01-15"),
            _make_fingerprint(move_date="2024-02-15"),
            _make_fingerprint(move_date="2024-01-15"),  # Duplicate of first
        ]
        stored = lib.store_batch(fps)
        assert stored == 2
        assert lib.size == 2

    def test_store_creates_template(self, tmp_path):
        """Storing a fingerprint auto-creates a PatternTemplate."""
        lib = PatternLibrary(
            library_path=tmp_path / "library.jsonl",
            templates_path=tmp_path / "templates.jsonl",
        )
        fp = _make_fingerprint()
        lib.store(fp)
        assert lib.template_count == 1

    def test_same_domains_share_template(self, tmp_path):
        """Fingerprints with same direction+domains share one template."""
        lib = PatternLibrary(
            library_path=tmp_path / "library.jsonl",
            templates_path=tmp_path / "templates.jsonl",
        )
        fp1 = _make_fingerprint(symbol="NVDA", move_date="2024-01-15")
        fp2 = _make_fingerprint(symbol="AAPL", move_date="2024-02-15")
        lib.store(fp1)
        lib.store(fp2)
        assert lib.size == 2
        assert lib.template_count == 1  # Same domains → same template

    def test_persistence_across_loads(self, tmp_path):
        lib_path = tmp_path / "library.jsonl"
        tmpl_path = tmp_path / "templates.jsonl"
        lib1 = PatternLibrary(library_path=lib_path, templates_path=tmpl_path)
        fp = _make_fingerprint()
        lib1.store(fp)

        lib2 = PatternLibrary(library_path=lib_path, templates_path=tmpl_path)
        assert lib2.size == 1
        assert lib2.template_count == 1
        assert lib2.get(fp.fingerprint_id) is not None

    def test_empty_library(self, tmp_path):
        lib_path = tmp_path / "nonexistent.jsonl"
        lib = PatternLibrary(library_path=lib_path, templates_path=tmp_path / "templates.jsonl")
        assert lib.size == 0
        assert lib.template_count == 0

    def test_update_outcome_fingerprint(self, tmp_path):
        lib = PatternLibrary(
            library_path=tmp_path / "library.jsonl",
            templates_path=tmp_path / "templates.jsonl",
        )
        fp = _make_fingerprint()
        lib.store(fp)

        lib.update_outcome(fingerprint_id=fp.fingerprint_id, won=True)
        lib.update_outcome(fingerprint_id=fp.fingerprint_id, won=True)
        lib.update_outcome(fingerprint_id=fp.fingerprint_id, won=False)

        updated = lib.get(fp.fingerprint_id)
        assert updated.wins == 2
        assert updated.losses == 1
        assert updated.total_activations == 3


class TestPatternLibraryQuery:
    """Query tests — templates are symbol-AGNOSTIC, matched by domain overlap."""

    def test_query_full_domain_match(self, tmp_path):
        """All template domains present in live sources → high match score."""
        lib = PatternLibrary(
            library_path=tmp_path / "library.jsonl",
            templates_path=tmp_path / "templates.jsonl",
            match_threshold=0.5,
        )
        fp = _make_fingerprint(sources=[
            ("sec_form4", "insider"),
            ("fred_macro", "macro"),
            ("ta_rsi", "technical"),
        ])
        lib.store(fp)

        # All 3 source domains present → 3/3 = 1.0 match
        matches = lib.query_similar(
            live_sources={"sec_form4", "fred_macro", "ta_rsi"},
            direction="bullish",
        )
        assert len(matches) == 1
        assert matches[0].match_score >= 0.9

    def test_query_partial_domain_match(self, tmp_path):
        """2 of 3 template domains present → ~0.67 match score."""
        lib = PatternLibrary(
            library_path=tmp_path / "library.jsonl",
            templates_path=tmp_path / "templates.jsonl",
            match_threshold=0.5,
        )
        fp = _make_fingerprint(sources=[
            ("sec_form4", "insider"),
            ("fred_macro", "macro"),
            ("ta_rsi", "technical"),
        ])
        lib.store(fp)

        # 2 of 3 domains present (insider + macro)
        matches = lib.query_similar(
            live_sources={"sec_form4", "fred_macro"},
            direction="bullish",
        )
        assert len(matches) == 1
        assert matches[0].match_score >= 0.5

    def test_query_cross_symbol_match(self, tmp_path):
        """Templates are symbol-agnostic: pattern from NVDA matches on AAPL."""
        lib = PatternLibrary(
            library_path=tmp_path / "library.jsonl",
            templates_path=tmp_path / "templates.jsonl",
            match_threshold=0.5,
        )
        fp = _make_fingerprint(symbol="NVDA")
        lib.store(fp)

        # Query for AAPL with same sources → template still matches
        matches = lib.query_similar(
            live_sources={"sec_form4", "fred_macro", "ta_rsi"},
            symbol="AAPL",
            direction="bullish",
        )
        assert len(matches) == 1  # Template is symbol-agnostic

    def test_query_no_match_wrong_direction(self, tmp_path):
        lib = PatternLibrary(
            library_path=tmp_path / "library.jsonl",
            templates_path=tmp_path / "templates.jsonl",
            match_threshold=0.5,
        )
        fp = _make_fingerprint(direction="bullish")
        lib.store(fp)

        matches = lib.query_similar(
            live_sources={"sec_form4", "fred_macro", "ta_rsi"},
            direction="bearish",  # Wrong direction
        )
        assert len(matches) == 0

    def test_query_below_threshold(self, tmp_path):
        lib = PatternLibrary(
            library_path=tmp_path / "library.jsonl",
            templates_path=tmp_path / "templates.jsonl",
            match_threshold=0.8,
        )
        fp = _make_fingerprint(sources=[
            ("sec_form4", "insider"),
            ("fred_macro", "macro"),
            ("ta_rsi", "technical"),
            ("ta_macd", "technical"),
            ("congressional", "government"),
        ])
        lib.store(fp)

        # Only 1 of 4 unique domains present (insider from sec_form4)
        matches = lib.query_similar(
            live_sources={"sec_form4"},
            direction="bullish",
        )
        assert len(matches) == 0  # 0.25 < 0.8 threshold

    def test_query_returns_matched_domains(self, tmp_path):
        """PatternMatch reports which domains matched and which are missing."""
        lib = PatternLibrary(
            library_path=tmp_path / "library.jsonl",
            templates_path=tmp_path / "templates.jsonl",
            match_threshold=0.5,
        )
        fp = _make_fingerprint(sources=[
            ("sec_form4", "insider"),
            ("fred_macro", "macro"),
            ("ta_rsi", "technical"),
        ])
        lib.store(fp)

        matches = lib.query_similar(
            live_sources={"sec_form4", "fred_macro"},
            direction="bullish",
        )
        assert len(matches) == 1
        assert "insider" in matches[0].matched_domains
        assert "macro" in matches[0].matched_domains
        assert "technical" in matches[0].missing_domains


class TestPatternLibraryStats:
    def test_statistics(self, tmp_path):
        lib = PatternLibrary(
            library_path=tmp_path / "library.jsonl",
            templates_path=tmp_path / "templates.jsonl",
        )
        lib.store(_make_fingerprint(direction="bullish", move_date="2024-01-15"))
        lib.store(_make_fingerprint(direction="bearish", move_date="2024-02-15"))

        stats = lib.get_statistics()
        assert stats["total_fingerprints"] == 2
        assert stats["total_templates"] == 2  # Different directions → different templates

    def test_clopper_pearson_ci(self, tmp_path):
        lib = PatternLibrary(
            library_path=tmp_path / "library.jsonl",
            templates_path=tmp_path / "templates.jsonl",
        )
        fp = _make_fingerprint()
        lib.store(fp)

        # No outcome data yet
        lo, hi = lib.clopper_pearson_ci(fingerprint_id=fp.fingerprint_id)
        assert lo == 0.0
        assert hi == 1.0

        # Add some outcomes
        for _ in range(7):
            lib.update_outcome(fingerprint_id=fp.fingerprint_id, won=True)
        for _ in range(3):
            lib.update_outcome(fingerprint_id=fp.fingerprint_id, won=False)

        lo, hi = lib.clopper_pearson_ci(fingerprint_id=fp.fingerprint_id)
        assert 0.3 < lo < 0.7  # 70% WR, CI should be reasonable
        assert 0.7 < hi < 1.0
