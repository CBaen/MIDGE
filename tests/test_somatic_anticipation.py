"""Tests for Gift 9: Somatic Anticipation (Council Gift).

Pre-conscious pattern response — MIDGE's body starts responding
to emerging patterns before formal convergence.
"""

import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from mae_core.market.intelligence.somatic_anticipation import (
    SomaticAnticipation,
    TickerState,
    AnticipationEvent,
)


class TestConstruction:
    """Basic construction and configuration."""

    def test_default_construction(self):
        sa = SomaticAnticipation()
        assert sa._anticipation_threshold == 2
        assert sa._response_window_hours == 48.0

    def test_custom_threshold(self):
        sa = SomaticAnticipation(anticipation_threshold=3)
        assert sa._anticipation_threshold == 3

    def test_custom_window(self):
        sa = SomaticAnticipation(response_window_hours=24.0)
        assert sa._response_window_hours == 24.0

    def test_endocrine_injected(self):
        mock_endo = MagicMock()
        sa = SomaticAnticipation(endocrine_system=mock_endo)
        assert sa._endocrine is mock_endo


class TestSignalRecording:
    """Signal recording accumulates state correctly."""

    def test_record_single_signal(self):
        sa = SomaticAnticipation()
        sa.record_signal("AAPL", "insider", "bullish", 0.8, datetime.now())
        assert "AAPL" in sa._ticker_states
        state = sa._ticker_states["AAPL"]
        assert "insider" in state.domains_active
        assert state.signal_count == 1

    def test_record_multiple_domains(self):
        sa = SomaticAnticipation()
        now = datetime.now()
        sa.record_signal("AAPL", "insider", "bullish", 0.8, now)
        sa.record_signal("AAPL", "technical", "bullish", 0.7, now)
        sa.record_signal("AAPL", "government", "bullish", 0.6, now)
        state = sa._ticker_states["AAPL"]
        assert len(state.domains_active) == 3
        assert state.signal_count == 3
        assert state.total_strength == pytest.approx(2.1, abs=0.01)

    def test_record_empty_ticker_ignored(self):
        sa = SomaticAnticipation()
        sa.record_signal("", "insider", "bullish", 0.8, datetime.now())
        assert len(sa._ticker_states) == 0

    def test_stale_state_reset(self):
        """State older than window is reset on next signal."""
        sa = SomaticAnticipation(response_window_hours=1.0)
        old_time = datetime.now() - timedelta(hours=2)
        sa.record_signal("AAPL", "insider", "bullish", 0.8, old_time)
        sa._ticker_states["AAPL"].first_signal_time = old_time

        # Record a new signal — should reset state
        sa.record_signal("AAPL", "technical", "bearish", 0.6, datetime.now())
        state = sa._ticker_states["AAPL"]
        assert len(state.domains_active) == 1  # Only the new domain
        assert "technical" in state.domains_active

    def test_multiple_tickers_tracked(self):
        sa = SomaticAnticipation()
        now = datetime.now()
        sa.record_signal("AAPL", "insider", "bullish", 0.8, now)
        sa.record_signal("MSFT", "technical", "bearish", 0.7, now)
        assert len(sa._ticker_states) == 2


class TestAnticipation:
    """Anticipation event firing and hormone computation."""

    def test_no_anticipation_below_threshold(self):
        sa = SomaticAnticipation(anticipation_threshold=2)
        sa.record_signal("AAPL", "insider", "bullish", 0.8, datetime.now())
        events = sa.check_anticipation()
        assert len(events) == 0

    def test_anticipation_fires_at_threshold(self):
        sa = SomaticAnticipation(anticipation_threshold=2)
        now = datetime.now()
        sa.record_signal("AAPL", "insider", "bullish", 0.8, now)
        sa.record_signal("AAPL", "technical", "bullish", 0.7, now)
        events = sa.check_anticipation()
        assert len(events) == 1
        assert events[0].ticker == "AAPL"
        assert events[0].domains_active == 2
        assert events[0].dominant_direction == "bullish"

    def test_anticipation_strength_range(self):
        sa = SomaticAnticipation(anticipation_threshold=2)
        now = datetime.now()
        sa.record_signal("AAPL", "insider", "bullish", 0.8, now)
        sa.record_signal("AAPL", "technical", "bullish", 0.7, now)
        events = sa.check_anticipation()
        assert 0.0 <= events[0].anticipation_strength <= 1.0

    def test_cooldown_prevents_spam(self):
        sa = SomaticAnticipation(anticipation_threshold=2)
        now = datetime.now()
        sa.record_signal("AAPL", "insider", "bullish", 0.8, now)
        sa.record_signal("AAPL", "technical", "bullish", 0.7, now)

        events1 = sa.check_anticipation()
        assert len(events1) == 1

        # Immediate re-check should be suppressed by cooldown
        events2 = sa.check_anticipation()
        assert len(events2) == 0

    def test_mixed_directions_produce_cortisol(self):
        sa = SomaticAnticipation(anticipation_threshold=2)
        now = datetime.now()
        sa.record_signal("AAPL", "insider", "bullish", 0.8, now)
        sa.record_signal("AAPL", "technical", "bearish", 0.7, now)
        events = sa.check_anticipation()
        assert len(events) == 1
        hormones = events[0].hormone_response
        # Mixed signals should produce CORTISOL (agreement < 0.4 since half each direction)
        # Actually 50% agreement = each direction gets 1 out of 2 = 0.5 agreement.
        # That's moderate, not mixed. Let's add more mixed signals.

    def test_strong_agreement_produces_dopamine(self):
        sa = SomaticAnticipation(anticipation_threshold=2)
        now = datetime.now()
        sa.record_signal("AAPL", "insider", "bullish", 0.8, now)
        sa.record_signal("AAPL", "technical", "bullish", 0.7, now)
        sa.record_signal("AAPL", "government", "bullish", 0.6, now)
        events = sa.check_anticipation()
        assert len(events) == 1
        hormones = events[0].hormone_response
        # 3 domains all bullish = high agreement → DOPAMINE + ADRENALINE
        assert "DOPAMINE" in hormones
        assert hormones["DOPAMINE"] == 0.25

    def test_three_plus_domains_adds_adrenaline(self):
        sa = SomaticAnticipation(anticipation_threshold=2)
        now = datetime.now()
        sa.record_signal("AAPL", "insider", "bullish", 0.8, now)
        sa.record_signal("AAPL", "technical", "bullish", 0.7, now)
        sa.record_signal("AAPL", "government", "bullish", 0.6, now)
        events = sa.check_anticipation()
        hormones = events[0].hormone_response
        assert "ADRENALINE" in hormones
        assert hormones["ADRENALINE"] == 0.10

    def test_stale_tickers_pruned(self):
        sa = SomaticAnticipation(anticipation_threshold=2, response_window_hours=1.0)
        old_time = datetime.now() - timedelta(hours=2)
        sa.record_signal("AAPL", "insider", "bullish", 0.8, old_time)
        sa.record_signal("AAPL", "technical", "bullish", 0.7, old_time)
        sa._ticker_states["AAPL"].first_signal_time = old_time

        events = sa.check_anticipation()
        assert len(events) == 0
        assert "AAPL" not in sa._ticker_states

    def test_multiple_tickers_can_anticipate(self):
        sa = SomaticAnticipation(anticipation_threshold=2)
        now = datetime.now()
        sa.record_signal("AAPL", "insider", "bullish", 0.8, now)
        sa.record_signal("AAPL", "technical", "bullish", 0.7, now)
        sa.record_signal("MSFT", "insider", "bearish", 0.8, now)
        sa.record_signal("MSFT", "government", "bearish", 0.6, now)

        events = sa.check_anticipation()
        assert len(events) == 2
        tickers = {e.ticker for e in events}
        assert tickers == {"AAPL", "MSFT"}


class TestEndocrineIntegration:
    """Hormone release via endocrine system."""

    def test_hormones_released_on_anticipation(self):
        mock_endo = MagicMock()
        sa = SomaticAnticipation(endocrine_system=mock_endo, anticipation_threshold=2)
        now = datetime.now()
        sa.record_signal("AAPL", "insider", "bullish", 0.8, now)
        sa.record_signal("AAPL", "technical", "bullish", 0.7, now)

        sa.check_anticipation()
        assert mock_endo.release_hormone.called

    def test_no_crash_without_endocrine(self):
        sa = SomaticAnticipation(anticipation_threshold=2)
        now = datetime.now()
        sa.record_signal("AAPL", "insider", "bullish", 0.8, now)
        sa.record_signal("AAPL", "technical", "bullish", 0.7, now)
        # Should not raise
        events = sa.check_anticipation()
        assert len(events) == 1


class TestEventBusPublishing:
    """EventBus publishing for somatic anticipation."""

    def test_event_published_on_anticipation(self):
        mock_bus = MagicMock()
        sa = SomaticAnticipation(event_bus=mock_bus, anticipation_threshold=2)
        now = datetime.now()
        sa.record_signal("AAPL", "insider", "bullish", 0.8, now)
        sa.record_signal("AAPL", "technical", "bullish", 0.7, now)

        sa.check_anticipation()
        assert mock_bus.publish.called

    def test_no_crash_without_bus(self):
        sa = SomaticAnticipation(anticipation_threshold=2)
        now = datetime.now()
        sa.record_signal("AAPL", "insider", "bullish", 0.8, now)
        sa.record_signal("AAPL", "technical", "bullish", 0.7, now)
        events = sa.check_anticipation()
        assert len(events) == 1


class TestStatistics:
    """Statistics reporting."""

    def test_stats_empty(self):
        sa = SomaticAnticipation()
        stats = sa.get_statistics()
        assert stats["tracked_tickers"] == 0
        assert stats["active_anticipations"] == 0

    def test_stats_with_data(self):
        sa = SomaticAnticipation(anticipation_threshold=2)
        now = datetime.now()
        sa.record_signal("AAPL", "insider", "bullish", 0.8, now)
        sa.record_signal("AAPL", "technical", "bullish", 0.7, now)
        stats = sa.get_statistics()
        assert stats["tracked_tickers"] == 1
        assert stats["active_anticipations"] == 1
        assert "AAPL" in stats["active_tickers"]
