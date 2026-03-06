"""Tests for PatternLibrary template persistence fixes.

Covers three safety properties introduced to prevent crash-induced data loss:

1. Atomic write  — templates are written to a .tmp file then renamed into place.
   A pre-existing corrupt .tmp file must not prevent a clean write.

2. Empty guard  — _persist_templates() returns early when self._templates is
   empty, so an accidental call can never silently wipe a valid templates file.

3. Batch persistence  — store_batch() calls _persist_templates() ONCE at the
   end of the loop, not once per fingerprint.

4. Single store persists immediately  — store() (single fingerprint) persists
   templates on every call.

5. Rebuild from fingerprints  — rebuild_templates() correctly reconstructs
   templates and persists them.
"""

import json
import unittest.mock as mock
from pathlib import Path

import pytest

from mae_core.market.archaeology.fingerprint import MoveFingerprint, PatternTemplate
from mae_core.market.archaeology.pattern_library import PatternLibrary


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_fp(
    fp_id: str,
    symbol: str,
    direction: str = "bullish",
    domain_signature: str = "insider+macro+technical",
    move_date: str = "2025-01-15",
    move_pct: float = 8.5,
) -> MoveFingerprint:
    """Return a minimal MoveFingerprint with an explicit domain_signature.

    We set domain_signature directly so no precursor_signals are required —
    keeping the construction free of external dependencies.
    """
    fp = MoveFingerprint(
        fingerprint_id=fp_id,
        symbol=symbol,
        direction=direction,
        move_date=move_date,
        move_pct=move_pct,
        domain_signature=domain_signature,
        lag_profile={"short": 2, "medium": 1},
        precursor_signals=[],
    )
    return fp


def _make_library(tmp_path: Path) -> PatternLibrary:
    """Return a PatternLibrary backed by files in tmp_path (no production I/O)."""
    return PatternLibrary(
        library_path=tmp_path / "fingerprints.jsonl",
        templates_path=tmp_path / "templates.jsonl",
    )


# ── 1. Atomic write ───────────────────────────────────────────────────────────


def test_atomic_write_tmp_file_is_gone_after_persist(tmp_path):
    """After a successful _persist_templates() call the .tmp file must not exist.

    The implementation writes to <templates_path>.tmp then renames it into place.
    The .tmp file must be gone (renamed) once persist returns successfully.
    """
    lib = _make_library(tmp_path)
    fp = _make_fp("fp_001", "AAPL")
    lib.store(fp)

    tmp_file = (tmp_path / "templates.jsonl").with_suffix(".tmp")
    assert not tmp_file.exists(), (
        ".tmp file should have been renamed into the real path, not left behind"
    )
    assert (tmp_path / "templates.jsonl").exists(), (
        "templates.jsonl must exist after store()"
    )


def test_atomic_write_stale_tmp_does_not_corrupt_real_file(tmp_path):
    """A stale/corrupt .tmp file left by a previous crash must not prevent a
    subsequent successful write from landing in the real file.

    Scenario:
      1. Pre-seed the real templates file with valid data.
      2. Leave a corrupt .tmp file next to it (simulates a prior crash mid-write).
      3. Call _persist_templates() — it must overwrite the .tmp and rename it,
         leaving the real file intact and the .tmp gone.
    """
    lib = _make_library(tmp_path)

    # Pre-seed with one fingerprint so templates are non-empty
    fp = _make_fp("fp_001", "AAPL")
    lib.store(fp)
    real_path = tmp_path / "templates.jsonl"
    original_content = real_path.read_text()

    # Leave a corrupt .tmp to simulate a prior crash
    tmp_file = real_path.with_suffix(".tmp")
    tmp_file.write_text("!!this is corrupt json!!")

    # Trigger another persist (store a new fingerprint to force a rewrite)
    fp2 = _make_fp("fp_002", "MSFT")
    lib.store(fp2)

    # .tmp must be gone (renamed into place)
    assert not tmp_file.exists(), ".tmp must be cleaned up after successful persist"

    # Real file must be valid JSON-lines (not the corrupt content)
    lines = [l for l in real_path.read_text().splitlines() if l.strip()]
    assert len(lines) >= 1
    for line in lines:
        json.loads(line)  # must not raise


# ── 2. Empty guard ────────────────────────────────────────────────────────────


def test_empty_guard_does_not_overwrite_existing_file(tmp_path):
    """_persist_templates() must return early when self._templates is empty.

    Scenario:
      1. Write a valid templates file manually.
      2. Create a PatternLibrary whose _templates dict is cleared after load.
      3. Call _persist_templates() directly.
      4. The file must still contain the original data.
    """
    lib = _make_library(tmp_path)
    fp = _make_fp("fp_001", "AAPL")
    lib.store(fp)

    templates_path = tmp_path / "templates.jsonl"
    original_content = templates_path.read_text()
    assert original_content.strip(), "Pre-condition: file must contain data"

    # Clear templates in-memory to simulate crash state
    lib._templates.clear()

    # This call must be a no-op — empty guard should prevent the write
    lib._persist_templates()

    assert templates_path.read_text() == original_content, (
        "_persist_templates() must not overwrite file when _templates is empty"
    )


def test_empty_guard_does_not_create_file_when_no_templates(tmp_path):
    """_persist_templates() must not create the templates file at all when
    _templates is empty and the file does not yet exist.
    """
    lib = _make_library(tmp_path)
    templates_path = tmp_path / "templates.jsonl"

    # Sanity: no fingerprints stored yet, so _templates should be empty
    assert lib.template_count == 0

    lib._persist_templates()

    assert not templates_path.exists(), (
        "_persist_templates() must not create the file when _templates is empty"
    )


# ── 3. Batch persistence — single call at end ─────────────────────────────────


def test_store_batch_persists_templates_once(tmp_path):
    """store_batch() must call _persist_templates() ONCE at the end, not once
    per fingerprint.  We patch _persist_templates with a counter to verify.
    """
    lib = _make_library(tmp_path)
    fps = [
        _make_fp(f"fp_{i:03d}", f"SYM{i}", move_date=f"2025-01-{i+1:02d}")
        for i in range(5)
    ]

    with mock.patch.object(lib, "_persist_templates", wraps=lib._persist_templates) as patched:
        stored = lib.store_batch(fps)

    assert stored == 5, "All 5 fingerprints should have been stored"
    assert patched.call_count == 1, (
        f"_persist_templates() must be called exactly once by store_batch(), "
        f"got {patched.call_count}"
    )


def test_store_batch_result_is_on_disk(tmp_path):
    """After store_batch(), the templates file must exist and contain all
    distinct domain+direction combinations from the stored fingerprints.
    """
    lib = _make_library(tmp_path)
    fps = [
        _make_fp("fp_001", "AAPL", domain_signature="insider+macro+technical"),
        _make_fp("fp_002", "MSFT", domain_signature="insider+macro+technical"),
        _make_fp("fp_003", "NVDA", domain_signature="events+macro"),
    ]
    lib.store_batch(fps)

    templates_path = tmp_path / "templates.jsonl"
    assert templates_path.exists(), "templates.jsonl must exist after store_batch()"

    lines = [l for l in templates_path.read_text().splitlines() if l.strip()]
    # Two distinct domain_signature values → two templates
    assert len(lines) == 2

    templates_on_disk = [json.loads(l) for l in lines]
    sigs = {t["domain_signature"] for t in templates_on_disk}
    assert "insider+macro+technical" in sigs
    assert "events+macro" in sigs


# ── 4. Single store persists immediately ─────────────────────────────────────


def test_single_store_persists_templates_immediately(tmp_path):
    """store() must persist templates after every call (unlike store_batch).
    Verified by checking the file exists and is readable after each store().
    """
    lib = _make_library(tmp_path)
    templates_path = tmp_path / "templates.jsonl"

    fp1 = _make_fp("fp_001", "AAPL")
    lib.store(fp1)
    assert templates_path.exists(), "templates.jsonl must exist after first store()"
    lines_after_1 = [l for l in templates_path.read_text().splitlines() if l.strip()]
    assert len(lines_after_1) == 1

    fp2 = _make_fp("fp_002", "MSFT", domain_signature="events+macro")
    lib.store(fp2)
    lines_after_2 = [l for l in templates_path.read_text().splitlines() if l.strip()]
    assert len(lines_after_2) == 2, (
        "A second store() with a new domain_signature must add a second template"
    )


def test_single_store_calls_persist_once_per_call(tmp_path):
    """Each store() call must invoke _persist_templates() exactly once."""
    lib = _make_library(tmp_path)
    fp = _make_fp("fp_001", "AAPL")

    with mock.patch.object(lib, "_persist_templates", wraps=lib._persist_templates) as patched:
        lib.store(fp)

    assert patched.call_count == 1, (
        f"store() must call _persist_templates() once, got {patched.call_count}"
    )


# ── 5. Rebuild from fingerprints ──────────────────────────────────────────────


def test_rebuild_templates_reconstructs_correct_count(tmp_path):
    """rebuild_templates() must produce one template per unique
    direction+domain_signature combination found in the stored fingerprints.
    """
    lib = _make_library(tmp_path)
    fps = [
        # Three fingerprints sharing the same template
        _make_fp("fp_001", "AAPL", domain_signature="insider+macro+technical"),
        _make_fp("fp_002", "MSFT", domain_signature="insider+macro+technical"),
        _make_fp("fp_003", "NVDA", domain_signature="insider+macro+technical"),
        # One fingerprint in a different template
        _make_fp("fp_004", "GOOGL", domain_signature="events+macro"),
    ]
    lib.store_batch(fps)

    # Simulate corruption: wipe in-memory templates
    lib._templates.clear()
    assert lib.template_count == 0, "Pre-condition: templates must be cleared"

    rebuilt = lib.rebuild_templates()

    assert rebuilt == 2, f"Expected 2 templates rebuilt, got {rebuilt}"
    assert lib.template_count == 2


def test_rebuild_templates_accumulates_instances(tmp_path):
    """After rebuild_templates(), the template for a domain signature that had
    multiple fingerprints must show n_instances == number of those fingerprints.
    """
    lib = _make_library(tmp_path)
    fps = [
        _make_fp("fp_001", "AAPL", domain_signature="insider+macro+technical"),
        _make_fp("fp_002", "MSFT", domain_signature="insider+macro+technical"),
        _make_fp("fp_003", "NVDA", domain_signature="insider+macro+technical"),
    ]
    lib.store_batch(fps)
    lib._templates.clear()
    lib.rebuild_templates()

    # Find the template for our domain signature
    matching = [
        t for t in lib._templates.values()
        if t.domain_signature == "insider+macro+technical"
    ]
    assert len(matching) == 1
    template = matching[0]
    assert template.n_instances == 3, (
        f"Template should have 3 instances, got {template.n_instances}"
    )
    assert set(template.unique_symbols) == {"AAPL", "MSFT", "NVDA"}


def test_rebuild_templates_persists_to_disk(tmp_path):
    """rebuild_templates() must write the rebuilt templates to the templates file."""
    lib = _make_library(tmp_path)
    fp = _make_fp("fp_001", "AAPL")
    lib.store(fp)

    # Corrupt the file and clear in-memory state
    templates_path = tmp_path / "templates.jsonl"
    templates_path.write_text("")  # empty / corrupt
    lib._templates.clear()

    lib.rebuild_templates()

    assert templates_path.exists()
    lines = [l for l in templates_path.read_text().splitlines() if l.strip()]
    assert len(lines) == 1, "Rebuilt templates must be persisted to disk"
    t = json.loads(lines[0])
    assert t["domain_signature"] == "insider+macro+technical"
