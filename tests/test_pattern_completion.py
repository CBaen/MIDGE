"""Tests for Gift 10: Pattern Completion Engine (Council Gift).

Active pattern seeking — hunts for missing archetype signals.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, PropertyMock
from types import SimpleNamespace

import pytest

from mae_core.market.intelligence.pattern_completion import (
    PatternCompletionEngine,
    CompletionHunt,
    CompletionEvent,
)


def _make_partial_match(archetype_id, ticker, match_score, missing, matched=None):
    """Create a mock partial match from archetype engine."""
    return SimpleNamespace(
        archetype_id=archetype_id,
        ticker=ticker,
        match_score=match_score,
        missing_signals=list(missing),
        matched_signals=list(matched or []),
    )


def _make_signal(ticker, domain, source="", strength=0.7, direction="bullish"):
    """Create a mock signal."""
    return SimpleNamespace(
        symbol=ticker,
        domain=domain,
        source=source or domain,
        strength=strength,
        direction=direction,
        metadata={"symbol": ticker},
    )


class TestConstruction:
    """Basic construction and configuration."""

    def test_default_construction(self):
        engine = PatternCompletionEngine()
        assert engine._max_concurrent_hunts == 5
        assert engine._hunt_ttl_hours == 72.0

    def test_custom_limits(self):
        engine = PatternCompletionEngine(max_concurrent_hunts=3, hunt_ttl_hours=24.0)
        assert engine._max_concurrent_hunts == 3
        assert engine._hunt_ttl_hours == 24.0

    def test_no_archetype_engine_returns_empty(self):
        engine = PatternCompletionEngine()
        hunts = engine.review_partial_matches()
        assert hunts == []


class TestHuntCreation:
    """Hunt creation from partial archetype matches."""

    def test_creates_hunt_from_partial_match(self):
        mock_ae = MagicMock()
        mock_ae.get_partial_matches.return_value = {
            "AAPL": _make_partial_match(
                "accumulation", "AAPL", 0.5,
                missing=["technical", "institutional"],
                matched=["insider"],
            ),
        }
        engine = PatternCompletionEngine(archetype_engine=mock_ae)
        hunts = engine.review_partial_matches()
        assert len(hunts) == 1
        assert hunts[0].archetype_id == "accumulation"
        assert hunts[0].ticker == "AAPL"
        assert "technical" in hunts[0].missing_signals
        assert "institutional" in hunts[0].missing_signals

    def test_skips_low_score_partials(self):
        mock_ae = MagicMock()
        mock_ae.get_partial_matches.return_value = {
            "AAPL": _make_partial_match(
                "squeeze", "AAPL", 0.2,  # Below _MIN_PARTIAL_SCORE (0.3)
                missing=["technical"],
            ),
        }
        engine = PatternCompletionEngine(archetype_engine=mock_ae)
        hunts = engine.review_partial_matches()
        assert len(hunts) == 0

    def test_skips_fully_matched(self):
        mock_ae = MagicMock()
        mock_ae.get_partial_matches.return_value = {
            "AAPL": _make_partial_match(
                "accumulation", "AAPL", 1.0,
                missing=[],
                matched=["insider", "technical"],
            ),
        }
        engine = PatternCompletionEngine(archetype_engine=mock_ae)
        hunts = engine.review_partial_matches()
        assert len(hunts) == 0

    def test_respects_max_concurrent_hunts(self):
        mock_ae = MagicMock()
        mock_ae.get_partial_matches.return_value = {
            f"TICK{i}": _make_partial_match(
                "squeeze", f"TICK{i}", 0.5,
                missing=["technical"],
            )
            for i in range(10)
        }
        engine = PatternCompletionEngine(archetype_engine=mock_ae, max_concurrent_hunts=3)
        hunts = engine.review_partial_matches()
        assert len(hunts) == 3

    def test_no_duplicate_hunts_for_same_ticker_archetype(self):
        mock_ae = MagicMock()
        mock_ae.get_partial_matches.return_value = {
            "AAPL": _make_partial_match(
                "accumulation", "AAPL", 0.5,
                missing=["technical"],
            ),
        }
        engine = PatternCompletionEngine(archetype_engine=mock_ae)
        hunts1 = engine.review_partial_matches()
        assert len(hunts1) == 1

        # Second call should not create duplicate
        hunts2 = engine.review_partial_matches()
        assert len(hunts2) == 0


class TestCompletionChecking:
    """Check completions against active hunts."""

    def test_signal_completes_hunt(self):
        mock_ae = MagicMock()
        mock_ae.get_partial_matches.return_value = {
            "AAPL": _make_partial_match(
                "accumulation", "AAPL", 0.5,
                missing=["technical"],
                matched=["insider"],
            ),
        }
        engine = PatternCompletionEngine(archetype_engine=mock_ae)
        engine.review_partial_matches()

        # Signal arrives that matches the missing piece
        signals = [_make_signal("AAPL", "technical")]
        events = engine.check_completions(signals)
        assert len(events) == 1
        assert events[0].archetype_id == "accumulation"
        assert events[0].ticker == "AAPL"
        assert events[0].completing_signal_domain == "technical"
        assert events[0].confidence_boost == 0.15

    def test_wrong_ticker_no_completion(self):
        mock_ae = MagicMock()
        mock_ae.get_partial_matches.return_value = {
            "AAPL": _make_partial_match(
                "accumulation", "AAPL", 0.5,
                missing=["technical"],
                matched=["insider"],
            ),
        }
        engine = PatternCompletionEngine(archetype_engine=mock_ae)
        engine.review_partial_matches()

        signals = [_make_signal("MSFT", "technical")]
        events = engine.check_completions(signals)
        assert len(events) == 0

    def test_wrong_domain_no_completion(self):
        mock_ae = MagicMock()
        mock_ae.get_partial_matches.return_value = {
            "AAPL": _make_partial_match(
                "accumulation", "AAPL", 0.5,
                missing=["technical"],
                matched=["insider"],
            ),
        }
        engine = PatternCompletionEngine(archetype_engine=mock_ae)
        engine.review_partial_matches()

        signals = [_make_signal("AAPL", "government")]
        events = engine.check_completions(signals)
        assert len(events) == 0

    def test_partial_completion_updates_hunt(self):
        """Hunt with 2 missing signals — completing 1 should update, not remove."""
        mock_ae = MagicMock()
        mock_ae.get_partial_matches.return_value = {
            "AAPL": _make_partial_match(
                "accumulation", "AAPL", 0.4,
                missing=["technical", "institutional"],
                matched=["insider"],
            ),
        }
        engine = PatternCompletionEngine(archetype_engine=mock_ae)
        engine.review_partial_matches()

        signals = [_make_signal("AAPL", "technical")]
        events = engine.check_completions(signals)
        assert len(events) == 1
        # Hunt should still be active (1 more signal needed)
        assert len(engine._active_hunts) == 1
        hunt = list(engine._active_hunts.values())[0]
        assert "institutional" in hunt.missing_signals
        assert "technical" in hunt.matched_signals

    def test_full_completion_removes_hunt(self):
        """Hunt with 1 missing signal — completing it removes the hunt."""
        mock_ae = MagicMock()
        mock_ae.get_partial_matches.return_value = {
            "AAPL": _make_partial_match(
                "squeeze", "AAPL", 0.7,
                missing=["technical"],
                matched=["insider", "institutional"],
            ),
        }
        engine = PatternCompletionEngine(archetype_engine=mock_ae)
        engine.review_partial_matches()
        assert len(engine._active_hunts) == 1

        signals = [_make_signal("AAPL", "technical")]
        events = engine.check_completions(signals)
        assert len(events) == 1
        assert len(engine._active_hunts) == 0  # Hunt removed after full completion

    def test_empty_signals_no_events(self):
        engine = PatternCompletionEngine()
        events = engine.check_completions([])
        assert events == []

    def test_no_hunts_no_events(self):
        engine = PatternCompletionEngine()
        signals = [_make_signal("AAPL", "technical")]
        events = engine.check_completions(signals)
        assert events == []

    def test_source_match_also_works(self):
        """Missing signal can match on source name too, not just domain."""
        mock_ae = MagicMock()
        mock_ae.get_partial_matches.return_value = {
            "AAPL": _make_partial_match(
                "momentum_ignition", "AAPL", 0.5,
                missing=["order_flow"],
                matched=["insider"],
            ),
        }
        engine = PatternCompletionEngine(archetype_engine=mock_ae)
        engine.review_partial_matches()

        # Signal with source matching the missing
        signals = [_make_signal("AAPL", "technical", source="order_flow")]
        events = engine.check_completions(signals)
        assert len(events) == 1


class TestHuntExpiration:
    """Expired hunts are pruned."""

    def test_expired_hunts_pruned(self):
        mock_ae = MagicMock()
        mock_ae.get_partial_matches.return_value = {
            "AAPL": _make_partial_match(
                "accumulation", "AAPL", 0.5,
                missing=["technical"],
            ),
        }
        engine = PatternCompletionEngine(archetype_engine=mock_ae, hunt_ttl_hours=0.001)
        engine.review_partial_matches()
        assert len(engine._active_hunts) == 1

        # Force expiration by setting expires_at to past
        for h in engine._active_hunts.values():
            h.expires_at = datetime.now() - timedelta(hours=1)

        pruned = engine._prune_expired()
        assert pruned == 1
        assert len(engine._active_hunts) == 0

    def test_non_expired_hunts_kept(self):
        mock_ae = MagicMock()
        mock_ae.get_partial_matches.return_value = {
            "AAPL": _make_partial_match(
                "accumulation", "AAPL", 0.5,
                missing=["technical"],
            ),
        }
        engine = PatternCompletionEngine(archetype_engine=mock_ae, hunt_ttl_hours=72.0)
        engine.review_partial_matches()

        pruned = engine._prune_expired()
        assert pruned == 0
        assert len(engine._active_hunts) == 1


class TestEventBusPublishing:
    """EventBus publishing on completion."""

    def test_publishes_on_completion(self):
        mock_bus = MagicMock()
        mock_ae = MagicMock()
        mock_ae.get_partial_matches.return_value = {
            "AAPL": _make_partial_match(
                "accumulation", "AAPL", 0.5,
                missing=["technical"],
                matched=["insider"],
            ),
        }
        engine = PatternCompletionEngine(archetype_engine=mock_ae, event_bus=mock_bus)
        engine.review_partial_matches()

        signals = [_make_signal("AAPL", "technical")]
        engine.check_completions(signals)
        assert mock_bus.publish.called

    def test_no_crash_without_bus(self):
        mock_ae = MagicMock()
        mock_ae.get_partial_matches.return_value = {
            "AAPL": _make_partial_match(
                "accumulation", "AAPL", 0.5,
                missing=["technical"],
                matched=["insider"],
            ),
        }
        engine = PatternCompletionEngine(archetype_engine=mock_ae)
        engine.review_partial_matches()

        signals = [_make_signal("AAPL", "technical")]
        events = engine.check_completions(signals)
        assert len(events) == 1


class TestStatistics:
    """Statistics reporting."""

    def test_stats_empty(self):
        engine = PatternCompletionEngine()
        stats = engine.get_statistics()
        assert stats["active_hunts"] == 0

    def test_stats_with_hunts(self):
        mock_ae = MagicMock()
        mock_ae.get_partial_matches.return_value = {
            "AAPL": _make_partial_match(
                "accumulation", "AAPL", 0.5,
                missing=["technical"],
            ),
        }
        engine = PatternCompletionEngine(archetype_engine=mock_ae)
        engine.review_partial_matches()
        stats = engine.get_statistics()
        assert stats["active_hunts"] == 1
        assert len(stats["hunts"]) == 1
        assert stats["hunts"][0]["ticker"] == "AAPL"

    def test_active_hunts_accessor(self):
        mock_ae = MagicMock()
        mock_ae.get_partial_matches.return_value = {
            "AAPL": _make_partial_match(
                "accumulation", "AAPL", 0.5,
                missing=["technical"],
            ),
        }
        engine = PatternCompletionEngine(archetype_engine=mock_ae)
        engine.review_partial_matches()
        hunts = engine.get_active_hunts()
        assert len(hunts) == 1
        assert hunts[0].ticker == "AAPL"
