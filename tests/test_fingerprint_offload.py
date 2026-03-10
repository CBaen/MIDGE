"""Tests for lazy fingerprint loading in PatternLibrary.

Verifies that:
- Fingerprints are NOT loaded into RAM at startup (lazy offload)
- Templates ARE loaded at startup (always in RAM)
- Fingerprint count (size) is accurate without loading objects
- store() and store_batch() work without in-RAM dict
- get() fetches a single fingerprint via disk scan
- rebuild_templates() loads fingerprints on-demand then releases
- update_outcome() loads fingerprints on-demand for mutation
- Dedup (no double-stores) still works correctly
- A library with pre-existing JSONL data loads count/IDs without loading objects
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from mae_core.market.archaeology.fingerprint import MoveFingerprint, PrecursorSignal, PatternTemplate
from mae_core.market.archaeology.pattern_library import PatternLibrary


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_fp(
    symbol: str = "NVDA",
    direction: str = "bullish",
    move_date: str = "2024-01-15",
    move_pct: float = 10.0,
    sources: list | None = None,
) -> MoveFingerprint:
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


def _write_fingerprints_to_jsonl(path: Path, fps: list[MoveFingerprint]) -> None:
    """Write fingerprints directly to a JSONL file (simulating a pre-existing library)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for fp in fps:
            f.write(fp.to_json() + "\n")


# ── Core lazy-loading tests ───────────────────────────────────────────────────

class TestLazyFingerprints:
    def test_fingerprints_not_in_ram_at_startup(self, tmp_path):
        """PatternLibrary must NOT have a populated _fingerprints dict after init."""
        lib_path = tmp_path / "fp.jsonl"
        tmpl_path = tmp_path / "tmpl.jsonl"

        # Pre-populate file with 3 fingerprints
        fps = [_make_fp(move_date=f"2024-0{i+1}-15") for i in range(3)]
        _write_fingerprints_to_jsonl(lib_path, fps)

        lib = PatternLibrary(library_path=lib_path, templates_path=tmpl_path)

        # The old _fingerprints dict should NOT exist or be empty
        assert not hasattr(lib, "_fingerprints") or lib._fingerprints == {}, (
            "Fingerprints were loaded into RAM at startup — lazy loading not active"
        )

    def test_fingerprint_count_accurate_without_ram(self, tmp_path):
        """size property returns correct count from disk scan, no RAM objects needed."""
        lib_path = tmp_path / "fp.jsonl"
        tmpl_path = tmp_path / "tmpl.jsonl"

        fps = [_make_fp(move_date=f"2024-0{i+1}-15") for i in range(5)]
        _write_fingerprints_to_jsonl(lib_path, fps)

        lib = PatternLibrary(library_path=lib_path, templates_path=tmpl_path)

        assert lib.size == 5

    def test_fingerprint_ids_set_populated(self, tmp_path):
        """_fingerprint_ids set must be populated for dedup to work."""
        lib_path = tmp_path / "fp.jsonl"
        tmpl_path = tmp_path / "tmpl.jsonl"

        fps = [_make_fp(move_date=f"2024-0{i+1}-15") for i in range(3)]
        _write_fingerprints_to_jsonl(lib_path, fps)

        lib = PatternLibrary(library_path=lib_path, templates_path=tmpl_path)

        assert hasattr(lib, "_fingerprint_ids")
        assert len(lib._fingerprint_ids) == 3
        for fp in fps:
            assert fp.fingerprint_id in lib._fingerprint_ids

    def test_templates_always_loaded(self, tmp_path):
        """Templates must always be in RAM after init."""
        lib_path = tmp_path / "fp.jsonl"
        tmpl_path = tmp_path / "tmpl.jsonl"

        # Build a library with one template stored
        setup_lib = PatternLibrary(library_path=lib_path, templates_path=tmpl_path)
        fp = _make_fp()
        setup_lib.store(fp)  # stores fingerprint AND creates template

        # Load a fresh library — templates should be present
        lib = PatternLibrary(library_path=lib_path, templates_path=tmpl_path)
        assert lib.template_count == 1
        assert len(lib._templates) == 1

    def test_empty_library_size_zero(self, tmp_path):
        lib = PatternLibrary(
            library_path=tmp_path / "fp.jsonl",
            templates_path=tmp_path / "tmpl.jsonl",
        )
        assert lib.size == 0


# ── store() and store_batch() ─────────────────────────────────────────────────

class TestStoreWithoutRam:
    def test_store_increments_count(self, tmp_path):
        lib = PatternLibrary(
            library_path=tmp_path / "fp.jsonl",
            templates_path=tmp_path / "tmpl.jsonl",
        )
        fp = _make_fp()
        assert lib.store(fp) is True
        assert lib.size == 1

    def test_store_persists_to_disk(self, tmp_path):
        lib_path = tmp_path / "fp.jsonl"
        lib = PatternLibrary(library_path=lib_path, templates_path=tmp_path / "tmpl.jsonl")
        fp = _make_fp()
        lib.store(fp)

        assert lib_path.exists()
        lines = [l for l in lib_path.read_text().splitlines() if l.strip()]
        assert len(lines) == 1

    def test_store_dedup_no_ram(self, tmp_path):
        """Duplicate detection works via _fingerprint_ids set, not in-RAM dict."""
        lib = PatternLibrary(
            library_path=tmp_path / "fp.jsonl",
            templates_path=tmp_path / "tmpl.jsonl",
        )
        fp = _make_fp()
        assert lib.store(fp) is True
        assert lib.store(fp) is False  # Duplicate
        assert lib.size == 1

    def test_store_batch_dedup(self, tmp_path):
        lib = PatternLibrary(
            library_path=tmp_path / "fp.jsonl",
            templates_path=tmp_path / "tmpl.jsonl",
        )
        fps = [
            _make_fp(move_date="2024-01-15"),
            _make_fp(move_date="2024-02-15"),
            _make_fp(move_date="2024-01-15"),  # Duplicate of first
        ]
        stored = lib.store_batch(fps)
        assert stored == 2
        assert lib.size == 2

    def test_store_updates_ids_set(self, tmp_path):
        lib = PatternLibrary(
            library_path=tmp_path / "fp.jsonl",
            templates_path=tmp_path / "tmpl.jsonl",
        )
        fp = _make_fp()
        lib.store(fp)
        assert fp.fingerprint_id in lib._fingerprint_ids

    def test_dedup_across_reload(self, tmp_path):
        """After reload, fingerprints stored in prior session are still deduped."""
        lib_path = tmp_path / "fp.jsonl"
        tmpl_path = tmp_path / "tmpl.jsonl"

        lib1 = PatternLibrary(library_path=lib_path, templates_path=tmpl_path)
        fp = _make_fp()
        lib1.store(fp)

        # Reload
        lib2 = PatternLibrary(library_path=lib_path, templates_path=tmpl_path)
        assert lib2.store(fp) is False  # Should be deduped via ID scan at load


# ── get() — targeted disk scan ────────────────────────────────────────────────

class TestGetFingerprint:
    def test_get_returns_fingerprint(self, tmp_path):
        lib = PatternLibrary(
            library_path=tmp_path / "fp.jsonl",
            templates_path=tmp_path / "tmpl.jsonl",
        )
        fp = _make_fp()
        lib.store(fp)

        result = lib.get(fp.fingerprint_id)
        assert result is not None
        assert result.fingerprint_id == fp.fingerprint_id
        assert result.symbol == "NVDA"

    def test_get_missing_returns_none(self, tmp_path):
        lib = PatternLibrary(
            library_path=tmp_path / "fp.jsonl",
            templates_path=tmp_path / "tmpl.jsonl",
        )
        assert lib.get("nonexistent-id") is None

    def test_get_does_not_pollute_ram(self, tmp_path):
        """get() must return an object without populating any in-RAM dict."""
        lib = PatternLibrary(
            library_path=tmp_path / "fp.jsonl",
            templates_path=tmp_path / "tmpl.jsonl",
        )
        fp = _make_fp()
        lib.store(fp)
        lib.get(fp.fingerprint_id)

        # Should still not have a loaded _fingerprints dict
        assert not hasattr(lib, "_fingerprints") or lib._fingerprints == {}


# ── rebuild_templates() — on-demand load ──────────────────────────────────────

class TestRebuildTemplates:
    def test_rebuild_from_pre_existing_fingerprints(self, tmp_path):
        """rebuild_templates() loads fingerprints from disk, builds templates, releases."""
        lib_path = tmp_path / "fp.jsonl"
        tmpl_path = tmp_path / "tmpl.jsonl"

        # Write fingerprints without creating templates
        fps = [
            _make_fp(move_date="2024-01-15"),
            _make_fp(move_date="2024-02-15"),
            _make_fp(symbol="AAPL", move_date="2024-01-15"),
        ]
        _write_fingerprints_to_jsonl(lib_path, fps)

        lib = PatternLibrary(library_path=lib_path, templates_path=tmpl_path)
        assert lib.template_count == 0  # No templates file

        n = lib.rebuild_templates()
        assert n >= 1
        assert lib.template_count >= 1

    def test_rebuild_does_not_leave_fingerprints_in_ram(self, tmp_path):
        lib_path = tmp_path / "fp.jsonl"
        tmpl_path = tmp_path / "tmpl.jsonl"

        fps = [_make_fp(move_date=f"2024-0{i+1}-15") for i in range(3)]
        _write_fingerprints_to_jsonl(lib_path, fps)

        lib = PatternLibrary(library_path=lib_path, templates_path=tmpl_path)
        lib.rebuild_templates()

        # After rebuild, fingerprints should NOT be in RAM
        assert not hasattr(lib, "_fingerprints") or lib._fingerprints == {}


# ── update_outcome() ──────────────────────────────────────────────────────────

class TestUpdateOutcome:
    def test_update_template_outcome(self, tmp_path):
        lib = PatternLibrary(
            library_path=tmp_path / "fp.jsonl",
            templates_path=tmp_path / "tmpl.jsonl",
        )
        fp = _make_fp()
        lib.store(fp)

        tmpl_id = list(lib._templates.keys())[0]
        lib.update_outcome(template_id=tmpl_id, won=True)

        t = lib.get_template(tmpl_id)
        assert t.wins == 1
        assert t.losses == 0

    def test_update_fingerprint_outcome(self, tmp_path):
        lib = PatternLibrary(
            library_path=tmp_path / "fp.jsonl",
            templates_path=tmp_path / "tmpl.jsonl",
        )
        fp = _make_fp()
        lib.store(fp)

        lib.update_outcome(fingerprint_id=fp.fingerprint_id, won=True)

        # Verify via disk (re-load the fingerprint)
        result = lib.get(fp.fingerprint_id)
        assert result is not None
        assert result.wins == 1


# ── query_similar() — templates only ─────────────────────────────────────────

class TestQuerySimilar:
    def test_query_uses_templates_not_fingerprints(self, tmp_path):
        lib = PatternLibrary(
            library_path=tmp_path / "fp.jsonl",
            templates_path=tmp_path / "tmpl.jsonl",
        )
        fp = _make_fp()
        lib.store(fp)

        matches = lib.query_similar(
            live_sources={"sec_form4", "fred_macro", "ta_rsi"},
            direction="bullish",
        )
        assert len(matches) >= 1
        assert matches[0].match_score > 0

    def test_query_returns_nothing_when_no_templates(self, tmp_path):
        """Pre-existing fingerprints without templates file → query returns empty."""
        lib_path = tmp_path / "fp.jsonl"
        tmpl_path = tmp_path / "tmpl.jsonl"

        fps = [_make_fp()]
        _write_fingerprints_to_jsonl(lib_path, fps)

        lib = PatternLibrary(library_path=lib_path, templates_path=tmpl_path)
        assert lib.template_count == 0

        matches = lib.query_similar(
            live_sources={"sec_form4", "fred_macro", "ta_rsi"},
            direction="bullish",
        )
        assert matches == []


# ── Memory savings smoke test ─────────────────────────────────────────────────

class TestMemorySavings:
    def test_large_library_does_not_exhaust_memory(self, tmp_path):
        """Write 1000 fingerprints and verify load is fast and light (no full parse)."""
        lib_path = tmp_path / "fp.jsonl"
        tmpl_path = tmp_path / "tmpl.jsonl"

        # Write 1000 fingerprints directly
        fps = [
            _make_fp(
                symbol=f"SYM{i:04d}",
                move_date=f"2024-01-{(i % 28) + 1:02d}",
            )
            for i in range(1000)
        ]
        _write_fingerprints_to_jsonl(lib_path, fps)

        lib = PatternLibrary(library_path=lib_path, templates_path=tmpl_path)

        # size should be accurate
        assert lib.size == 1000

        # IDs set should have 1000 entries
        assert len(lib._fingerprint_ids) == 1000

        # No full fingerprint objects in RAM
        assert not hasattr(lib, "_fingerprints") or lib._fingerprints == {}

    def test_new_store_after_large_preload(self, tmp_path):
        """After loading a large pre-existing library, new stores still work."""
        lib_path = tmp_path / "fp.jsonl"
        tmpl_path = tmp_path / "tmpl.jsonl"

        fps = [_make_fp(move_date=f"2024-01-{(i % 28) + 1:02d}", symbol=f"X{i:03d}") for i in range(100)]
        _write_fingerprints_to_jsonl(lib_path, fps)

        lib = PatternLibrary(library_path=lib_path, templates_path=tmpl_path)

        new_fp = _make_fp(symbol="NEW", move_date="2025-01-01")
        assert lib.store(new_fp) is True
        assert lib.size == 101
