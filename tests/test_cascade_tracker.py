"""Tests for CascadeTracker — domino confirmation + WorldModel feedback."""

import time
from unittest.mock import MagicMock

import pytest

from mae_core.market.intelligence.cascade_tracker import CascadeTracker
from mae_core.market.intelligence.world_model import WorldModel


def _ripples():
    return [
        {"ticker": "XLE", "direction": "bullish", "strength": 0.8, "lag_days": 1},
        {"ticker": "DAL", "direction": "bearish", "strength": 0.6, "lag_days": 5},
        {"ticker": "AAL", "direction": "bearish", "strength": 0.5, "lag_days": 5},
    ]


class TestRegistration:
    def test_register_cascade(self):
        ct = CascadeTracker()
        assert ct.register_cascade("A1", "crude_price_spike", _ripples(), "bullish")
        assert "A1" in ct.get_active_chains()

    def test_duplicate_rejected(self):
        ct = CascadeTracker()
        ct.register_cascade("A1", "t", _ripples(), "bullish")
        assert not ct.register_cascade("A1", "t", _ripples(), "bullish")

    def test_empty_ripples_rejected(self):
        ct = CascadeTracker()
        assert not ct.register_cascade("A1", "t", [], "bullish")

    def test_evicts_oldest_over_cap(self):
        ct = CascadeTracker(max_chains=2)
        ct.register_cascade("A1", "t1", _ripples(), "bullish")
        ct.register_cascade("A2", "t2", _ripples(), "bullish")
        ct.register_cascade("A3", "t3", _ripples(), "bullish")
        assert len(ct.get_active_chains()) == 2
        assert "A1" not in ct.get_active_chains()


class TestSignalConfirmation:
    def test_confirm_matching_signal(self):
        ct = CascadeTracker()
        ct.register_cascade("A1", "crude_price_spike", _ripples(), "bullish")
        results = ct.check_signal("XLE", "bullish")
        assert len(results) == 1
        assert results[0]["confirmed_ticker"] == "XLE"
        assert results[0]["confirmed_count"] == 1

    def test_no_match_wrong_direction(self):
        ct = CascadeTracker()
        ct.register_cascade("A1", "crude_price_spike", _ripples(), "bullish")
        assert ct.check_signal("XLE", "bearish") == []

    def test_no_match_wrong_ticker(self):
        ct = CascadeTracker()
        ct.register_cascade("A1", "crude_price_spike", _ripples(), "bullish")
        assert ct.check_signal("AAPL", "bullish") == []

    def test_double_confirm_ignored(self):
        ct = CascadeTracker()
        ct.register_cascade("A1", "crude_price_spike", _ripples(), "bullish")
        ct.check_signal("XLE", "bullish")
        results = ct.check_signal("XLE", "bullish")
        assert results == []

    def test_multiple_confirmations_tracked(self):
        ct = CascadeTracker()
        ct.register_cascade("A1", "crude_price_spike", _ripples(), "bullish")
        ct.check_signal("XLE", "bullish")
        results = ct.check_signal("DAL", "bearish")
        assert results[0]["confirmed_count"] == 2
        assert len(results[0]["remaining"]) == 1

    def test_confirmation_emits_event(self):
        bus = MagicMock()
        ct = CascadeTracker(event_bus=bus)
        ct.register_cascade("A1", "crude_price_spike", _ripples(), "bullish")
        ct.check_signal("XLE", "bullish")
        bus.publish.assert_called_once()
        assert bus.publish.call_args[0][0] == "market.intel.cascade_confirmed"


class TestWorldModelFeedback:
    def test_confirmation_calls_record_outcome_true(self):
        wm = MagicMock()
        ct = CascadeTracker(world_model=wm)
        ct.register_cascade("A1", "crude_price_spike", _ripples(), "bullish")
        ct.check_signal("XLE", "bullish")
        wm.record_outcome.assert_called_once_with("crude_price_spike", "XLE", was_correct=True)

    def test_expiry_calls_record_outcome_false(self):
        wm = MagicMock()
        ct = CascadeTracker(world_model=wm)
        ct.register_cascade("A1", "crude_price_spike", _ripples(), "bullish")
        # Force age past threshold
        ct._active_chains["A1"]["registered_at"] = time.time() - 86400 * 31
        ct.expire_stale(max_age_days=30)
        # 3 pending links → 3 calls with was_correct=False
        assert wm.record_outcome.call_count == 3

    def test_confirmed_links_not_expired_as_miss(self):
        wm = MagicMock()
        ct = CascadeTracker(world_model=wm)
        ct.register_cascade("A1", "crude_price_spike", _ripples(), "bullish")
        ct.check_signal("XLE", "bullish")  # confirm 1
        wm.reset_mock()
        ct._active_chains["A1"]["registered_at"] = time.time() - 86400 * 31
        ct.expire_stale(max_age_days=30)
        # Only 2 remaining pending links should get False
        assert wm.record_outcome.call_count == 2
        for call in wm.record_outcome.call_args_list:
            assert call.kwargs["was_correct"] is False

    def test_real_world_model_feedback(self):
        wm = WorldModel()
        ct = CascadeTracker(world_model=wm)
        ct.register_cascade("A1", "crude_price_spike", _ripples(), "bullish")
        initial = wm._graph.edges["crude_price_spike", "XLE"]["strength"]
        ct.check_signal("XLE", "bullish")
        assert wm._graph.edges["crude_price_spike", "XLE"]["strength"] > initial


class TestStatistics:
    def test_empty_stats(self):
        ct = CascadeTracker()
        s = ct.get_statistics()
        assert s["active_chains"] == 0
        assert s["confirmation_rate"] == 0

    def test_stats_after_activity(self):
        ct = CascadeTracker()
        ct.register_cascade("A1", "crude_price_spike", _ripples(), "bullish")
        ct.check_signal("XLE", "bullish")
        s = ct.get_statistics()
        assert s["active_chains"] == 1
        assert s["confirmed_links"] == 1
        assert s["pending_links"] == 2
        assert s["total_links"] == 3
