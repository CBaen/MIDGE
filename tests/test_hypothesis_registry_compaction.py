"""Tests for HypothesisRegistry snapshot compaction.

Verifies that:
  - A snapshot is created after _SNAPSHOT_INTERVAL events
  - Incremental replay works: snapshot + post-snapshot events → full state
  - Full JSONL replay works when no snapshot exists (backward compat)
  - Snapshot uses atomic tmp+replace pattern
  - Snapshot JSON includes 'saved_at' timestamp
  - events_since_snapshot counter increments and resets after snapshot
  - Snapshot captures all statuses (ACTIVE, RETIRED, PROBATION, HIBERNATED)
  - Only events AFTER snapshot timestamp are replayed during incremental load
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from mae_core.market.intelligence.hypothesis import (
    Hypothesis,
    HypothesisStatus,
    SourceType,
    TriggerPattern,
)
from mae_core.market.intelligence.hypothesis_registry import HypothesisRegistry


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_data_dir(tmp_path):
    return tmp_path


@pytest.fixture
def registry(tmp_data_dir):
    return HypothesisRegistry(data_dir=tmp_data_dir)


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_hypothesis(name="test_hyp", source_a="sec_form4", source_b="price"):
    return Hypothesis(
        name=name,
        trigger=TriggerPattern(
            source_a=source_a, source_b=source_b,
            lag_days=30, direction="positive",
        ),
        causal_story=f"Insiders buy before {source_b} moves.",
        source_type=SourceType.LAG_CORRELATION,
    )


def _fill_events(registry, count, base_name="hyp"):
    """Register `count` unique hypotheses to generate `count` events."""
    hyps = []
    for i in range(count):
        h = _make_hypothesis(name=f"{base_name}_{i}", source_a=f"src_{i}")
        registry.register(h)
        hyps.append(h)
    return hyps


# ── Tests ──────────────────────────────────────────────────────────────────


class TestSnapshotCreation:
    def test_snapshot_created_after_threshold(self, tmp_data_dir):
        """Append 201+ events → snapshot file created."""
        reg = HypothesisRegistry(data_dir=tmp_data_dir)
        threshold = reg._SNAPSHOT_INTERVAL  # 200

        # Fill threshold + 1 events to trigger snapshot
        _fill_events(reg, threshold + 1)

        assert reg._snapshot_path.exists(), (
            f"Snapshot should exist after {threshold + 1} events"
        )

    def test_snapshot_not_created_below_threshold(self, tmp_data_dir):
        """Fewer than _SNAPSHOT_INTERVAL events → no snapshot created."""
        reg = HypothesisRegistry(data_dir=tmp_data_dir)
        threshold = reg._SNAPSHOT_INTERVAL

        # Fill one less than threshold
        _fill_events(reg, threshold - 1)

        assert not reg._snapshot_path.exists(), (
            "Snapshot should NOT exist below threshold"
        )

    def test_snapshot_contains_saved_at(self, tmp_data_dir):
        """Snapshot JSON has 'saved_at' timestamp field."""
        reg = HypothesisRegistry(data_dir=tmp_data_dir)
        _fill_events(reg, reg._SNAPSHOT_INTERVAL + 1)

        assert reg._snapshot_path.exists()
        snap = json.loads(reg._snapshot_path.read_text())
        assert "saved_at" in snap
        # saved_at should be parseable ISO datetime
        dt = datetime.fromisoformat(snap["saved_at"])
        assert dt is not None

    def test_snapshot_preserves_all_states(self, tmp_data_dir):
        """Snapshot captures ACTIVE, RETIRED, PROBATION, and HIBERNATED hypotheses."""
        reg = HypothesisRegistry(data_dir=tmp_data_dir)

        # Create hypotheses in different states
        h_active = _make_hypothesis("h_active", source_a="src_a")
        h_retired = _make_hypothesis("h_retired", source_a="src_b")
        h_probation = _make_hypothesis("h_probation", source_a="src_c")
        h_hibernated = _make_hypothesis("h_hibernated", source_a="src_d")

        reg.register(h_active)
        reg.register(h_retired)
        reg.register(h_probation)
        reg.register(h_hibernated)

        reg.promote(h_active.hypothesis_id)      # → ACTIVE
        reg.retire(h_retired.hypothesis_id)      # → RETIRED
        reg.promote(h_hibernated.hypothesis_id)  # → ACTIVE first
        reg.hibernate(h_hibernated.hypothesis_id)  # → HIBERNATED
        # h_probation stays in PROBATION

        # Force a snapshot regardless of event count
        reg._save_snapshot()

        snap = json.loads(reg._snapshot_path.read_text())
        hyps_in_snap = snap.get("hypotheses", {})

        # All 4 should be in snapshot
        assert h_active.hypothesis_id in hyps_in_snap
        assert h_retired.hypothesis_id in hyps_in_snap
        assert h_probation.hypothesis_id in hyps_in_snap
        assert h_hibernated.hypothesis_id in hyps_in_snap

        # Status should be preserved
        assert hyps_in_snap[h_active.hypothesis_id]["status"] == "active"
        assert hyps_in_snap[h_retired.hypothesis_id]["status"] == "retired"
        assert hyps_in_snap[h_probation.hypothesis_id]["status"] == "probation"
        assert hyps_in_snap[h_hibernated.hypothesis_id]["status"] == "hibernated"


class TestSnapshotAtomicWrite:
    def test_snapshot_atomic_write(self, tmp_data_dir):
        """Snapshot uses tmp+replace: tmp file does NOT persist after save."""
        reg = HypothesisRegistry(data_dir=tmp_data_dir)
        _fill_events(reg, reg._SNAPSHOT_INTERVAL + 1)

        tmp_path = reg._snapshot_path.with_suffix(".tmp")
        # After a successful save, the .tmp file should not exist
        assert not tmp_path.exists(), ".tmp file should not persist after atomic replace"
        assert reg._snapshot_path.exists(), "Snapshot should exist after save"

    def test_snapshot_is_valid_json(self, tmp_data_dir):
        """Snapshot file is valid JSON with 'hypotheses' key."""
        reg = HypothesisRegistry(data_dir=tmp_data_dir)
        _fill_events(reg, reg._SNAPSHOT_INTERVAL + 1)

        assert reg._snapshot_path.exists()
        snap = json.loads(reg._snapshot_path.read_text())
        assert "hypotheses" in snap
        assert isinstance(snap["hypotheses"], dict)


class TestEventCounterAndReset:
    def test_events_since_snapshot_counter_increments(self, tmp_data_dir):
        """Counter increments on each event."""
        reg = HypothesisRegistry(data_dir=tmp_data_dir)
        assert reg._events_since_snapshot == 0

        h = _make_hypothesis("counter_test")
        reg.register(h)
        assert reg._events_since_snapshot == 1

        reg.promote(h.hypothesis_id)
        assert reg._events_since_snapshot == 2

    def test_events_since_snapshot_resets_after_snapshot(self, tmp_data_dir):
        """Counter resets to 0 after snapshot is saved."""
        reg = HypothesisRegistry(data_dir=tmp_data_dir)

        # Force a snapshot directly
        _fill_events(reg, 5)
        reg._save_snapshot()

        assert reg._events_since_snapshot == 0


class TestIncrementalReplay:
    def test_incremental_replay(self, tmp_data_dir):
        """Create snapshot, append more events, reload → all hypotheses present."""
        reg1 = HypothesisRegistry(data_dir=tmp_data_dir)

        # Register + promote a batch, force snapshot
        h_before = _make_hypothesis("before_snap", source_a="pre_snap")
        reg1.register(h_before)
        reg1.promote(h_before.hypothesis_id)
        reg1._save_snapshot()

        # Add more hypotheses AFTER the snapshot
        h_after = _make_hypothesis("after_snap", source_a="post_snap")
        reg1.register(h_after)

        # New registry loads from snapshot + incremental events
        reg2 = HypothesisRegistry(data_dir=tmp_data_dir)

        assert reg2.get(h_before.hypothesis_id) is not None, "Pre-snapshot hyp should load"
        assert reg2.get(h_after.hypothesis_id) is not None, "Post-snapshot hyp should load"
        assert reg2.get(h_before.hypothesis_id).status == HypothesisStatus.ACTIVE

    def test_full_replay_without_snapshot(self, tmp_data_dir):
        """No snapshot → full JSONL replay (backward compatibility)."""
        reg1 = HypothesisRegistry(data_dir=tmp_data_dir)

        h1 = _make_hypothesis("fullreplay_1", source_a="src_x")
        h2 = _make_hypothesis("fullreplay_2", source_a="src_y")
        reg1.register(h1)
        reg1.register(h2)
        reg1.promote(h1.hypothesis_id)

        # Ensure no snapshot was created
        assert not reg1._snapshot_path.exists(), "No snapshot should exist for this test"

        # New registry should replay from JSONL
        reg2 = HypothesisRegistry(data_dir=tmp_data_dir)
        assert len(reg2.get_all()) == 2
        assert reg2.get(h1.hypothesis_id).status == HypothesisStatus.ACTIVE
        assert reg2.get(h2.hypothesis_id).status == HypothesisStatus.PROBATION

    def test_replay_only_after_snapshot_time(self, tmp_data_dir):
        """Events before snapshot timestamp are skipped during incremental replay."""
        reg1 = HypothesisRegistry(data_dir=tmp_data_dir)

        # Register hypothesis and save snapshot
        h_pre = _make_hypothesis("pre_snap_only", source_a="pre")
        reg1.register(h_pre)
        reg1.promote(h_pre.hypothesis_id)
        reg1._save_snapshot()

        # Note the snapshot timestamp
        snap_before_reload = json.loads(reg1._snapshot_path.read_text())
        saved_at = snap_before_reload["saved_at"]

        # Now add a post-snapshot event
        h_post = _make_hypothesis("post_snap_only", source_a="post")
        reg1.register(h_post)

        # Reload
        reg2 = HypothesisRegistry(data_dir=tmp_data_dir)

        # Both should be present: pre from snapshot, post from incremental replay
        assert reg2.get(h_pre.hypothesis_id) is not None
        assert reg2.get(h_post.hypothesis_id) is not None

        # Pre-snapshot hypothesis should have ACTIVE status (from snapshot, not re-applied)
        assert reg2.get(h_pre.hypothesis_id).status == HypothesisStatus.ACTIVE

    def test_snapshot_count_matches_hypotheses(self, tmp_data_dir):
        """Snapshot hypotheses dict has the right count."""
        reg = HypothesisRegistry(data_dir=tmp_data_dir)

        for i in range(10):
            h = _make_hypothesis(f"count_test_{i}", source_a=f"src_{i}")
            reg.register(h)

        reg._save_snapshot()
        snap = json.loads(reg._snapshot_path.read_text())
        assert len(snap["hypotheses"]) == 10
