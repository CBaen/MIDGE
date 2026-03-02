"""Tests for OutcomeCollector registered signal pruning.

Verifies that:
  - _registered is a dict (not a set)
  - Entries older than 90 days are pruned on the next register call
  - Recent entries survive pruning
  - Dedup still works correctly (same signal_id not registered twice)
  - Backward compat: old list-format data is converted to dict on load
  - Prune runs automatically when register_signals is called

OutcomeTracker is imported inside OutcomeCollector.__init__ with a local import,
so we patch it at its source module: mae_core.market.outcome_tracker.OutcomeTracker
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def mock_price_fetcher():
    return MagicMock()


@pytest.fixture
def mock_thompson():
    sampler = MagicMock()
    sampler.get_distribution.return_value = MagicMock(mean=0.5, samples=10)
    return sampler


@pytest.fixture
def tmp_data_dir(tmp_path):
    return tmp_path


# ── Factory ────────────────────────────────────────────────────────────────


def _make_collector(tmp_data_dir, mock_price_fetcher, mock_thompson):
    """Create OutcomeCollector with OutcomeTracker mocked at its source module.

    OutcomeTracker is imported inside __init__ via:
      from mae_core.market.outcome_tracker import OutcomeTracker
    so we patch it at 'mae_core.market.outcome_tracker.OutcomeTracker'.
    """
    from mae_core.market.intelligence.outcome_collector import OutcomeCollector

    with patch("mae_core.market.outcome_tracker.OutcomeTracker") as MockTracker:
        mock_tracker = MagicMock()
        mock_tracker.check_pending_outcomes.return_value = 0
        mock_tracker.get_statistics.return_value = {
            "pending_predictions": 0,
            "total_evaluated": 0,
        }
        MockTracker.return_value = mock_tracker
        c = OutcomeCollector(
            price_fetcher=mock_price_fetcher,
            thompson_sampler=mock_thompson,
            data_dir=tmp_data_dir,
        )
    return c


@pytest.fixture
def collector(tmp_data_dir, mock_price_fetcher, mock_thompson):
    """OutcomeCollector backed by a temporary data directory."""
    return _make_collector(tmp_data_dir, mock_price_fetcher, mock_thompson)


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_signal(signal_id="sig-001", symbol="AAPL", source="sec_form4",
                 direction="bullish"):
    """Create a minimal MarketSignal-like object for testing."""
    sig = SimpleNamespace(
        signal_id=signal_id,
        symbol=symbol,
        source=source,
        direction=direction,
        timestamp=datetime.now(),
    )
    return sig


# ── Tests ──────────────────────────────────────────────────────────────────


class TestRegisteredIsDict:
    def test_registered_is_dict(self, collector):
        """_registered is a dict (signal_id -> datetime), not a set."""
        assert isinstance(collector._registered, dict)

    def test_registered_values_are_datetimes(self, collector):
        """After registering a signal, the stored value is a datetime."""
        sig = _make_signal()

        with patch.object(collector.tracker, "record_prediction"):
            collector.register_signals([sig])

        assert isinstance(collector._registered.get("sig-001"), datetime)


class TestPrune90Days:
    def test_registered_prune_90_days(self, tmp_data_dir, mock_price_fetcher, mock_thompson):
        """Signal registered 91 days ago is pruned on the next register call."""
        c = _make_collector(tmp_data_dir, mock_price_fetcher, mock_thompson)

        # Inject a stale entry (91 days old)
        c._registered["stale-sig"] = datetime.now() - timedelta(days=91)
        assert "stale-sig" in c._registered

        # Fresh signal to trigger prune
        fresh_sig = _make_signal(signal_id="fresh-001")
        with patch.object(c.tracker, "record_prediction"):
            c.register_signals([fresh_sig])

        assert "stale-sig" not in c._registered, "91-day old entry should be pruned"
        assert "fresh-001" in c._registered

    def test_registered_keeps_recent(self, tmp_data_dir, mock_price_fetcher, mock_thompson):
        """Signal registered 89 days ago survives pruning (within 90-day window)."""
        c = _make_collector(tmp_data_dir, mock_price_fetcher, mock_thompson)

        c._registered["recent-sig"] = datetime.now() - timedelta(days=89)

        fresh_sig = _make_signal(signal_id="fresh-002")
        with patch.object(c.tracker, "record_prediction"):
            c.register_signals([fresh_sig])

        assert "recent-sig" in c._registered, "89-day old entry should survive"

    def test_prune_runs_on_register(self, tmp_data_dir, mock_price_fetcher, mock_thompson):
        """Prune executes automatically each time register_signals is called."""
        c = _make_collector(tmp_data_dir, mock_price_fetcher, mock_thompson)

        # Inject multiple stale entries
        for i in range(5):
            c._registered[f"stale-{i}"] = datetime.now() - timedelta(days=95)

        fresh_sig = _make_signal(signal_id="trigger-prune")
        with patch.object(c.tracker, "record_prediction"):
            c.register_signals([fresh_sig])

        stale_remaining = [k for k in c._registered if k.startswith("stale-")]
        assert len(stale_remaining) == 0, "All stale entries should be pruned"


class TestRegisteredDedup:
    def test_registered_dedup_still_works(self, collector):
        """Same signal_id registered twice → only one entry, one record_prediction call."""
        sig = _make_signal(signal_id="dedup-001")

        with patch.object(collector.tracker, "record_prediction") as mock_record:
            collector.register_signals([sig])
            collector.register_signals([sig])

        assert mock_record.call_count == 1, "record_prediction should only be called once"
        assert len([k for k in collector._registered if k == "dedup-001"]) == 1

    def test_registered_no_symbol_skipped(self, collector):
        """Signal without symbol is NOT registered."""
        sig = _make_signal(signal_id="no-symbol", symbol="")

        with patch.object(collector.tracker, "record_prediction") as mock_record:
            collector.register_signals([sig])

        assert mock_record.call_count == 0
        assert "no-symbol" not in collector._registered


class TestBackwardCompat:
    def test_registered_backward_compat_list(self, tmp_data_dir, mock_price_fetcher,
                                              mock_thompson):
        """If persisted data is a list (old format), it converts to dict on load."""
        # Write old-format list data before creating the collector
        reg_path = tmp_data_dir / "registered_signals.json"
        reg_path.write_text(json.dumps(["old-sig-1", "old-sig-2"]))

        c = _make_collector(tmp_data_dir, mock_price_fetcher, mock_thompson)

        assert isinstance(c._registered, dict), "List format should be converted to dict"
        assert "old-sig-1" in c._registered
        assert "old-sig-2" in c._registered
        # Values should be datetime objects (migrated with fallback timestamp)
        for v in c._registered.values():
            assert isinstance(v, datetime)
