"""Tests for ConsolidationEngine (Gift 6 — Sleep Cycle / Memory Consolidation).

Verifies Thompson pruning, hypothesis archiving, and discovery log
compression. No real data files — uses mocks/stubs for determinism.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mae_core.market.intelligence.consolidation_engine import (
    ConsolidationEngine,
    ConsolidationReport,
)


def _make_dist(samples=10, last_update=None):
    """Create a mock Thompson distribution."""
    if last_update is None:
        last_update = datetime.now().isoformat()
    return SimpleNamespace(samples=samples, last_update=last_update)


def _make_hypothesis(status="RETIRED", retired_at=None, **kwargs):
    """Create a mock hypothesis."""
    if retired_at is None:
        retired_at = datetime.now().isoformat()
    return SimpleNamespace(
        status=status,
        retired_at=retired_at,
        trigger_source="src_a",
        target_source="src_b",
        lag_days=3,
        win_rate=0.55,
        observations=25,
        **kwargs,
    )


class TestConstruction:
    """Basic construction and interface."""

    def test_constructs_without_args(self):
        ce = ConsolidationEngine()
        assert ce is not None

    def test_constructs_with_all_args(self):
        ce = ConsolidationEngine(
            thompson_sampler=object(),
            hypothesis_registry=object(),
            hypothesis_engine=object(),
            event_bus=object(),
        )
        assert ce._thompson_sampler is not None

    def test_statistics_on_fresh_engine(self):
        ce = ConsolidationEngine()
        stats = ce.get_statistics()
        assert stats["total_consolidations"] == 0
        assert stats["last_consolidation"] is None


class TestShouldConsolidate:
    """Test cadence checking."""

    def test_first_step_does_not_trigger(self):
        """Step 0 - last_consolidation 0 = 0, which is < cadence."""
        ce = ConsolidationEngine()
        assert ce.should_consolidate(0) is False

    def test_early_step_does_not_trigger(self):
        ce = ConsolidationEngine()
        ce._last_consolidation = 0
        assert ce.should_consolidate(100) is False

    def test_at_cadence_triggers(self):
        ce = ConsolidationEngine()
        ce._last_consolidation = 0
        assert ce.should_consolidate(5000) is True

    def test_updates_last_consolidation(self):
        ce = ConsolidationEngine()
        ce.should_consolidate(5000)
        assert ce._last_consolidation == 5000


class TestThompsonPruning:
    """Test pruning of stale, thin Thompson distributions."""

    def test_prunes_thin_stale_distributions(self):
        old_date = (datetime.now() - timedelta(days=60)).isoformat()
        sampler = SimpleNamespace(
            _distributions={
                "good_key": _make_dist(samples=20),
                "thin_stale": _make_dist(samples=2, last_update=old_date),
            }
        )
        ce = ConsolidationEngine(thompson_sampler=sampler)
        pruned = ce._prune_thompson()
        assert pruned == 1
        assert "good_key" in sampler._distributions
        assert "thin_stale" not in sampler._distributions

    def test_keeps_thin_but_recent_distributions(self):
        recent_date = datetime.now().isoformat()
        sampler = SimpleNamespace(
            _distributions={
                "thin_recent": _make_dist(samples=2, last_update=recent_date),
            }
        )
        ce = ConsolidationEngine(thompson_sampler=sampler)
        pruned = ce._prune_thompson()
        assert pruned == 0
        assert "thin_recent" in sampler._distributions

    def test_keeps_well_sampled_distributions(self):
        old_date = (datetime.now() - timedelta(days=60)).isoformat()
        sampler = SimpleNamespace(
            _distributions={
                "well_sampled": _make_dist(samples=50, last_update=old_date),
            }
        )
        ce = ConsolidationEngine(thompson_sampler=sampler)
        pruned = ce._prune_thompson()
        assert pruned == 0

    def test_handles_no_sampler(self):
        ce = ConsolidationEngine(thompson_sampler=None)
        assert ce._prune_thompson() == 0

    def test_handles_sampler_without_distributions(self):
        ce = ConsolidationEngine(thompson_sampler=object())
        assert ce._prune_thompson() == 0

    def test_prunes_dist_with_no_last_update(self):
        """Distributions with no last_update and thin data are pruned."""
        sampler = SimpleNamespace(
            _distributions={
                "no_date": SimpleNamespace(samples=1, last_update=None),
            }
        )
        ce = ConsolidationEngine(thompson_sampler=sampler)
        pruned = ce._prune_thompson()
        assert pruned == 1


class TestHypothesisArchiving:
    """Test archiving of old RETIRED hypotheses."""

    def test_archives_old_retired_hypotheses(self, tmp_path, monkeypatch):
        old_date = (datetime.now() - timedelta(days=30)).isoformat()
        registry = SimpleNamespace(
            _hypotheses={
                "old_retired": _make_hypothesis(status="RETIRED", retired_at=old_date),
                "active_one": _make_hypothesis(status="ACTIVE", retired_at=None),
            }
        )
        # Redirect data dir to tmp
        monkeypatch.setattr(
            "mae_core.market.intelligence.consolidation_engine._DATA_DIR",
            tmp_path,
        )
        ce = ConsolidationEngine(hypothesis_registry=registry)
        archived = ce._archive_hypotheses()
        assert archived == 1
        assert "old_retired" not in registry._hypotheses
        assert "active_one" in registry._hypotheses
        # Verify archive file
        archive_path = tmp_path / "hypotheses_archive.jsonl"
        assert archive_path.exists()

    def test_keeps_recently_retired(self):
        recent_date = datetime.now().isoformat()
        registry = SimpleNamespace(
            _hypotheses={
                "recent_retired": _make_hypothesis(
                    status="RETIRED", retired_at=recent_date
                ),
            }
        )
        ce = ConsolidationEngine(hypothesis_registry=registry)
        archived = ce._archive_hypotheses()
        assert archived == 0
        assert "recent_retired" in registry._hypotheses

    def test_handles_no_registry(self):
        ce = ConsolidationEngine(hypothesis_registry=None)
        assert ce._archive_hypotheses() == 0

    def test_handles_registry_without_hypotheses(self):
        ce = ConsolidationEngine(hypothesis_registry=object())
        assert ce._archive_hypotheses() == 0


class TestDiscoveryLogCompression:
    """Test compression of old discovery log entries."""

    def test_compresses_old_entries(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "mae_core.market.intelligence.consolidation_engine._DATA_DIR",
            tmp_path,
        )
        log_path = tmp_path / "discovery_log.jsonl"

        # Write entries: some old, some recent
        old_ts = (datetime.now() - timedelta(days=120)).isoformat()
        recent_ts = datetime.now().isoformat()

        with open(log_path, "w") as f:
            f.write(json.dumps({"timestamp": old_ts, "finding": "old"}) + "\n")
            f.write(json.dumps({"timestamp": old_ts, "finding": "old2"}) + "\n")
            f.write(json.dumps({"timestamp": recent_ts, "finding": "recent"}) + "\n")

        ce = ConsolidationEngine()
        compressed = ce._compress_discovery_log()
        assert compressed == 2

        # Verify only recent entry remains
        with open(log_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 1
        assert json.loads(lines[0])["finding"] == "recent"

    def test_no_compression_when_all_recent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "mae_core.market.intelligence.consolidation_engine._DATA_DIR",
            tmp_path,
        )
        log_path = tmp_path / "discovery_log.jsonl"
        recent_ts = datetime.now().isoformat()
        with open(log_path, "w") as f:
            f.write(json.dumps({"timestamp": recent_ts, "finding": "ok"}) + "\n")

        ce = ConsolidationEngine()
        compressed = ce._compress_discovery_log()
        assert compressed == 0

    def test_handles_missing_log_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "mae_core.market.intelligence.consolidation_engine._DATA_DIR",
            tmp_path,
        )
        ce = ConsolidationEngine()
        assert ce._compress_discovery_log() == 0


class TestFullConsolidation:
    """Test the full consolidation cycle."""

    def test_consolidate_produces_report(self):
        ce = ConsolidationEngine()
        report = ce.consolidate()
        assert isinstance(report, ConsolidationReport)
        assert report.timestamp != ""
        assert report.duration_ms >= 0

    def test_consolidate_updates_statistics(self):
        ce = ConsolidationEngine()
        ce.consolidate()
        stats = ce.get_statistics()
        assert stats["total_consolidations"] == 1
        assert stats["last_consolidation"] is not None

    def test_consolidate_publishes_event(self):
        bus = MagicMock()
        ce = ConsolidationEngine(event_bus=bus)
        ce.consolidate()
        bus.publish.assert_called_once()

    def test_multiple_consolidations_tracked(self):
        ce = ConsolidationEngine()
        ce.consolidate()
        ce.consolidate()
        ce.consolidate()
        stats = ce.get_statistics()
        assert stats["total_consolidations"] == 3

    def test_report_history_capped(self):
        ce = ConsolidationEngine()
        for _ in range(25):
            ce.consolidate()
        assert len(ce._reports) == 20  # Max 20 reports kept
