"""Tests for Layer 7 — Persistence layer roundtrips.

Covers save/load for:
  - LEARNING_CONFIG snapshots (learning_config.py)
  - _deep_merge semantics (learning_config.py)
  - HypothesisEngine retirement window (hypothesis_engine.py)
  - HypothesisGenerator pair outcomes (hypothesis_generator.py)
  - StepTimer session stats (step_timer.py)
  - Warm-start via load_snapshot() modifying LEARNING_CONFIG in-place
"""

import copy
import json

import pytest
from unittest.mock import MagicMock

from mae_core.market.intelligence import learning_config as lc
from mae_core.market.intelligence.learning_config import (
    LEARNING_CONFIG,
    _deep_merge,
    load_snapshot,
    save_snapshot,
)
from mae_core.market.intelligence.hypothesis_registry import HypothesisRegistry
from mae_core.market.intelligence.hypothesis_generator import HypothesisGenerator
from mae_core.market.intelligence.hypothesis_engine import HypothesisEngine
from mae_core.market.step_timer import StepTimer


# ── Shared fixtures ──────────────────────────────────────────────────


@pytest.fixture
def registry(tmp_path):
    """Isolated HypothesisRegistry that writes to tmp_path."""
    return HypothesisRegistry(data_dir=tmp_path)


@pytest.fixture
def generator(registry):
    """HypothesisGenerator wired to the isolated registry."""
    return HypothesisGenerator(registry=registry)


@pytest.fixture
def engine(registry, generator):
    """Minimal HypothesisEngine with mock validator — no live I/O."""
    return HypothesisEngine(
        registry=registry,
        generator=generator,
        validator=MagicMock(generate=MagicMock(return_value=[])),
    )


# ── 1. Config snapshot roundtrip ────────────────────────────────────


def test_config_snapshot_roundtrip(tmp_path):
    """save_snapshot() writes JSON; load_snapshot() restores values into the
    live LEARNING_CONFIG dict.  A nested leaf value (decay_rates.news) must
    survive the round-trip unchanged.
    """
    snapshot_path = tmp_path / "config_snapshot.json"

    # Record the original value so we can restore it afterward
    original_news_decay = LEARNING_CONFIG["decay_rates"]["news"]
    original_version = LEARNING_CONFIG["version"]

    # Mutate a nested leaf to a distinctive sentinel
    LEARNING_CONFIG["decay_rates"]["news"] = 0.999

    try:
        save_snapshot(path=snapshot_path)

        assert snapshot_path.exists(), "save_snapshot must create the file"

        # Verify raw JSON contains the sentinel
        raw = json.loads(snapshot_path.read_text())
        assert raw["decay_rates"]["news"] == 0.999

        # Reset the live dict back to original so load_snapshot has work to do
        LEARNING_CONFIG["decay_rates"]["news"] = original_news_decay

        result = load_snapshot(path=snapshot_path)

        assert result is True, "load_snapshot must return True when file is found"
        assert LEARNING_CONFIG["decay_rates"]["news"] == 0.999
    finally:
        # Always restore so other tests see the original state
        LEARNING_CONFIG["decay_rates"]["news"] = original_news_decay
        LEARNING_CONFIG["version"] = original_version


# ── 2. _deep_merge semantics ─────────────────────────────────────────


def test_deep_merge_preserves_structure():
    """_deep_merge must:
    - Update existing leaf values recursively
    - Ignore keys that are not present in base (forward-compat gate)
    - Leave unmentioned existing keys untouched
    """
    base = {
        "learning_rate": 0.1,
        "decay_rates": {
            "news": 0.5,
            "technical": 0.1,
        },
    }
    update = {
        "learning_rate": 0.99,
        "decay_rates": {
            "news": 0.77,          # existing key — should update
            "brand_new_domain": 0.3,  # absent from base — must be ignored
        },
        "completely_new_key": "should_be_ignored",  # absent from base
    }

    _deep_merge(base, update)

    # Existing leaf updated
    assert base["learning_rate"] == 0.99
    # Nested existing leaf updated
    assert base["decay_rates"]["news"] == 0.77
    # Nested untouched leaf preserved
    assert base["decay_rates"]["technical"] == 0.1
    # New key at root ignored
    assert "completely_new_key" not in base
    # New key inside nested dict ignored
    assert "brand_new_domain" not in base["decay_rates"]


# ── 3. Retirement window save/load ───────────────────────────────────


def test_retirement_window_save_load(tmp_path):
    """Items appended to _retirement_window must be present after save/load
    into a fresh engine instance pointing at the same tmp_path.
    """
    reg = HypothesisRegistry(data_dir=tmp_path)
    gen = HypothesisGenerator(registry=reg)
    eng1 = HypothesisEngine(
        registry=reg,
        generator=gen,
        validator=MagicMock(generate=MagicMock(return_value=[])),
    )

    # Directly manipulate the retirement window and persist
    eng1._retirement_window = ["promoted", "retired", "promoted"]
    eng1._meta_promoted_total = 2
    eng1._meta_retired_after_active = 1
    eng1._save_retirement_window()

    # Verify the file was written
    window_path = tmp_path / "retirement_window.json"
    assert window_path.exists(), "_save_retirement_window must create the file"

    # Create a second engine pointing at the same directory
    reg2 = HypothesisRegistry(data_dir=tmp_path)
    gen2 = HypothesisGenerator(registry=reg2)
    eng2 = HypothesisEngine(
        registry=reg2,
        generator=gen2,
        validator=MagicMock(generate=MagicMock(return_value=[])),
    )

    assert eng2._retirement_window == ["promoted", "retired", "promoted"]
    assert eng2._meta_promoted_total == 2
    assert eng2._meta_retired_after_active == 1


# ── 4. Retirement window trims to max on load ────────────────────────


def test_retirement_window_trims_to_max(tmp_path):
    """When the saved window has more entries than _retirement_window_max, only
    the most recent max entries are kept after load.
    """
    reg = HypothesisRegistry(data_dir=tmp_path)
    gen = HypothesisGenerator(registry=reg)
    eng1 = HypothesisEngine(
        registry=reg,
        generator=gen,
        validator=MagicMock(generate=MagicMock(return_value=[])),
    )

    # Write 60 entries directly to the file (bypassing the list guard)
    window_path = tmp_path / "retirement_window.json"
    oversized = ["promoted"] * 35 + ["retired"] * 25  # 60 total
    window_path.write_text(json.dumps({
        "saved_at": "2026-01-01T00:00:00",
        "retirement_window": oversized,
        "meta_promoted_total": 35,
        "meta_retired_after_active": 10,
    }))

    # Load into a fresh engine whose max is 50
    reg2 = HypothesisRegistry(data_dir=tmp_path)
    gen2 = HypothesisGenerator(registry=reg2)
    eng2 = HypothesisEngine(
        registry=reg2,
        generator=gen2,
        validator=MagicMock(generate=MagicMock(return_value=[])),
    )

    assert len(eng2._retirement_window) == 50
    # Should be the *last* 50 entries (tail of the list)
    assert eng2._retirement_window == oversized[-50:]


# ── 5. Pair outcomes save/load ───────────────────────────────────────


def test_pair_outcomes_save_load(tmp_path):
    """record_outcome() accumulates counts, save_pair_outcomes() writes them,
    and a fresh HypothesisGenerator instance restores them correctly.
    """
    reg = HypothesisRegistry(data_dir=tmp_path)
    gen1 = HypothesisGenerator(registry=reg)

    gen1.record_outcome("sec_form4", "finnhub_earnings", "promoted")
    gen1.record_outcome("sec_form4", "finnhub_earnings", "promoted")
    gen1.record_outcome("sec_form4", "finnhub_earnings", "retired")
    gen1.record_outcome("congressional", "finra_short", "retired")

    # Verify the file exists after record_outcome (it calls save internally)
    pair_path = tmp_path / "pair_outcomes.json"
    assert pair_path.exists(), "save_pair_outcomes must create the file"

    # Load into a new generator
    reg2 = HypothesisRegistry(data_dir=tmp_path)
    gen2 = HypothesisGenerator(registry=reg2)

    key1 = ("sec_form4", "finnhub_earnings")
    key2 = ("congressional", "finra_short")

    assert key1 in gen2._pair_outcomes, "pair 1 must survive roundtrip"
    assert gen2._pair_outcomes[key1]["promoted"] == 2
    assert gen2._pair_outcomes[key1]["retired"] == 1

    assert key2 in gen2._pair_outcomes, "pair 2 must survive roundtrip"
    assert gen2._pair_outcomes[key2]["retired"] == 1
    assert gen2._pair_outcomes[key2]["promoted"] == 0


# ── 6. StepTimer session stats snapshot ─────────────────────────────


def test_step_timer_snapshot(tmp_path):
    """save_session_stats() must write a JSON file with 'saved_at' and 'stats'
    keys; each tracked operation must appear in 'stats' with p50/p95/max/count.
    """
    timer = StepTimer()

    with timer.track("convergence_check"):
        pass  # near-zero duration

    with timer.track("convergence_check"):
        pass

    with timer.track("thompson_update"):
        pass

    out_path = tmp_path / "step_timer.json"
    timer.save_session_stats(out_path)

    assert out_path.exists(), "save_session_stats must create the file"

    payload = json.loads(out_path.read_text())

    assert "saved_at" in payload, "payload must include saved_at timestamp"
    assert "stats" in payload, "payload must include stats dict"

    stats = payload["stats"]
    assert "convergence_check" in stats
    assert "thompson_update" in stats

    cc = stats["convergence_check"]
    for field in ("p50_ms", "p95_ms", "max_ms", "count"):
        assert field in cc, f"convergence_check must have field '{field}'"

    assert cc["count"] == 2


# ── 7. load_snapshot modifies LEARNING_CONFIG in-place ───────────────


def test_warm_start_loads_before_systems(tmp_path):
    """load_snapshot() must modify the live LEARNING_CONFIG dict in-place,
    not replace it.  This is the contract that market_systems.py depends on:
    other modules hold a reference to the same dict object and will see
    the restored values without re-importing.

    Procedure:
      1. Capture the object identity of LEARNING_CONFIG.
      2. Write a snapshot with a modified value.
      3. Manually reset that value in the live dict.
      4. Call load_snapshot().
      5. Assert the same dict object now carries the restored value.
    """
    snapshot_path = tmp_path / "warm_start_snapshot.json"

    original_rate = LEARNING_CONFIG["learning_rate"]
    original_version = LEARNING_CONFIG["version"]

    LEARNING_CONFIG["learning_rate"] = 0.42
    save_snapshot(path=snapshot_path)

    # Capture identity before load
    dict_id_before = id(LEARNING_CONFIG)

    # Reset to something obviously different
    LEARNING_CONFIG["learning_rate"] = 0.01

    try:
        result = load_snapshot(path=snapshot_path)

        assert result is True
        # Identity must be preserved — same object, not a replacement
        assert id(LEARNING_CONFIG) == dict_id_before, (
            "load_snapshot must modify LEARNING_CONFIG in-place, not replace it"
        )
        # Value must be restored from the snapshot
        assert LEARNING_CONFIG["learning_rate"] == 0.42, (
            "load_snapshot must restore the saved value into the live dict"
        )
    finally:
        LEARNING_CONFIG["learning_rate"] = original_rate
        LEARNING_CONFIG["version"] = original_version
