"""Tests for dynamic outcome windows — lag profile aggregation and window computation."""

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from pathlib import Path

from mae_core.market.archaeology.fingerprint import (
    MoveFingerprint,
    PatternTemplate,
    PrecursorSignal,
    TemplateInstance,
)


def _make_fingerprint(symbol="NVDA", direction="bullish", move_date="2024-01-15",
                       move_pct=10.0, lag_profile=None, precursors=None):
    """Helper to build a fingerprint with specific lag data."""
    if precursors is None:
        precursors = [
            PrecursorSignal("ta_rsi", "technical", direction, 0.7, 5, "short"),
            PrecursorSignal("ta_macd", "technical", direction, 0.6, 15, "long"),
        ]
    fp = MoveFingerprint(
        fingerprint_id="",
        symbol=symbol,
        direction=direction,
        move_date=move_date,
        move_pct=move_pct,
        precursor_signals=precursors,
    )
    if lag_profile is not None:
        fp.lag_profile = lag_profile
    return fp


# ── Lag Profile Aggregation ──────────────────────────────────────────


class TestLagProfileAggregation:

    def test_single_fingerprint_populates_raw(self):
        fp = _make_fingerprint(lag_profile={"short": 3, "medium": 1})
        t = PatternTemplate.from_fingerprint(fp)
        assert t.lag_profile_raw == {"short": 3, "medium": 1}

    def test_add_instance_accumulates_raw(self):
        fp1 = _make_fingerprint(symbol="NVDA", lag_profile={"short": 3, "medium": 1},
                                 move_date="2024-01-15")
        fp2 = _make_fingerprint(symbol="AAPL", lag_profile={"immediate": 2, "short": 1},
                                 move_date="2024-02-15")
        t = PatternTemplate.from_fingerprint(fp1)
        t.add_instance(fp2)
        assert t.lag_profile_raw == {"short": 4, "medium": 1, "immediate": 2}

    def test_normalized_updated_after_add(self):
        fp1 = _make_fingerprint(symbol="NVDA", lag_profile={"short": 3, "medium": 1},
                                 move_date="2024-01-15")
        fp2 = _make_fingerprint(symbol="AAPL", lag_profile={"immediate": 2, "short": 1},
                                 move_date="2024-02-15")
        t = PatternTemplate.from_fingerprint(fp1)
        t.add_instance(fp2)
        total = 4 + 1 + 2  # short=4, medium=1, immediate=2
        assert abs(t.lag_profile_normalized["short"] - 4 / total) < 0.001
        assert abs(t.lag_profile_normalized["medium"] - 1 / total) < 0.001
        assert abs(t.lag_profile_normalized["immediate"] - 2 / total) < 0.001

    def test_no_double_counting_in_from_fingerprint(self):
        """from_fingerprint() should NOT pre-populate lag_profile_raw."""
        fp = _make_fingerprint(lag_profile={"long": 5})
        t = PatternTemplate.from_fingerprint(fp)
        # Should be exactly 5, not 10 (from double-counting)
        assert t.lag_profile_raw == {"long": 5}


# ── Expected Move Window ─────────────────────────────────────────────


class TestExpectedMoveWindow:

    def test_fallback_when_no_lag_data(self):
        t = PatternTemplate(
            template_id="test", direction="bullish",
            domain_signature="technical",
        )
        assert t.expected_move_window_days == 14

    def test_immediate_signals_give_short_window(self):
        t = PatternTemplate(
            template_id="test", direction="bullish",
            domain_signature="technical",
            lag_profile_normalized={"immediate": 1.0},
        )
        # mean=1.0, buffer=int(1.2)+1=2, clamped to 3
        assert t.expected_move_window_days == 3

    def test_medium_signals(self):
        t = PatternTemplate(
            template_id="test", direction="bullish",
            domain_signature="technical",
            lag_profile_normalized={"medium": 1.0},
        )
        # mean=8.0, buffer=int(9.6)+1=10
        assert t.expected_move_window_days == 10

    def test_long_signals(self):
        t = PatternTemplate(
            template_id="test", direction="bullish",
            domain_signature="technical",
            lag_profile_normalized={"long": 1.0},
        )
        # mean=15.0, buffer=int(18.0)+1=19
        assert t.expected_move_window_days == 19

    def test_extended_clamped_to_30(self):
        t = PatternTemplate(
            template_id="test", direction="bullish",
            domain_signature="technical",
            lag_profile_normalized={"extended": 1.0},
        )
        # mean=25.0, buffer=int(30.0)+1=31, clamped to 30
        assert t.expected_move_window_days == 30

    def test_mixed_profile_weighted(self):
        t = PatternTemplate(
            template_id="test", direction="bullish",
            domain_signature="technical",
            lag_profile_normalized={"short": 0.7, "long": 0.3},
        )
        # mean = 4.0*0.7 + 15.0*0.3 = 2.8 + 4.5 = 7.3
        # buffer = int(8.76) + 1 = 9
        assert t.expected_move_window_days == 9

    def test_from_fingerprint_computes_window(self):
        """Templates built from fingerprints should have working window computation."""
        fp = _make_fingerprint(lag_profile={"short": 7, "medium": 3})
        t = PatternTemplate.from_fingerprint(fp)
        assert t.expected_move_window_days > 0
        assert t.expected_move_window_days != 14  # not the fallback


# ── Serialization Round-Trip ─────────────────────────────────────────


class TestSerialization:

    def test_lag_profile_raw_survives_round_trip(self):
        fp = _make_fingerprint(lag_profile={"short": 5, "long": 3})
        t = PatternTemplate.from_fingerprint(fp)
        d = t.to_dict()
        t2 = PatternTemplate.from_dict(d)
        assert t2.lag_profile_raw == {"short": 5, "long": 3}
        assert abs(t2.lag_profile_normalized["short"] - 5 / 8) < 0.001

    def test_missing_lag_profile_raw_backward_compat(self):
        """Old template dicts without lag_profile_raw should still load."""
        d = {
            "template_id": "abc",
            "direction": "bullish",
            "domain_signature": "technical",
            "lag_profile_normalized": {"long": 0.6, "extended": 0.4},
        }
        t = PatternTemplate.from_dict(d)
        assert t.lag_profile_raw == {}
        # Should still use the existing normalized profile for window
        assert t.expected_move_window_days > 0

    def test_expected_window_from_old_format(self):
        """Templates loaded from old format use lag_profile_normalized directly."""
        d = {
            "template_id": "abc",
            "direction": "bullish",
            "domain_signature": "technical",
            "lag_profile_normalized": {"immediate": 0.5, "short": 0.5},
        }
        t = PatternTemplate.from_dict(d)
        # mean = 1.0*0.5 + 4.0*0.5 = 2.5, buffer = int(3.0)+1 = 4
        assert t.expected_move_window_days == 4


# ── Dynamic Registration ─────────────────────────────────────────────


class TestDynamicRegistration:

    def test_dynamic_window_used(self, tmp_path):
        """register_pattern_stack uses template timing data when available."""
        from mae_core.market.intelligence.outcome_collector import OutcomeCollector

        oc = OutcomeCollector(
            price_fetcher=MagicMock(),
            thompson_sampler=MagicMock(),
            data_dir=tmp_path,
        )

        # Build a mock stack with templates that have lag data
        tmpl = PatternTemplate(
            template_id="t1", direction="bullish",
            domain_signature="insider+technical",
            domains=["insider", "technical"],
            lag_profile_normalized={"short": 0.8, "medium": 0.2},
        )
        act = SimpleNamespace(template=tmpl, match_score=0.8)
        stack = SimpleNamespace(
            activations=[act],
            direction="bullish",
            tier="medium",
            stack_confidence=0.7,
            independent_pairs=1,
            created_at="2024-01-15T12:00:00",
        )

        oc.register_pattern_stack(stack, "NVDA")

        call_args = oc.tracker.record_prediction.call_args
        assert call_args is not None
        # Should NOT be 14 (the fallback)
        window = call_args.kwargs.get("outcome_window_days",
                                       call_args[1].get("outcome_window_days", 14))
        assert window == tmpl.expected_move_window_days
        metadata = call_args.kwargs.get("metadata", call_args[1].get("metadata", {}))
        assert metadata["outcome_window_source"] == "dynamic"

    def test_fallback_when_no_lag_data(self, tmp_path):
        """Falls back to 14 when templates have no expected_move_window_days."""
        from mae_core.market.intelligence.outcome_collector import OutcomeCollector

        oc = OutcomeCollector(
            thompson_sampler=MagicMock(),
            predictions_path=tmp_path / "predictions.jsonl",
            outcomes_path=tmp_path / "outcomes.jsonl",
            registered_path=tmp_path / "registered.json",
        )

        # Mock template without the property
        tmpl = SimpleNamespace(
            template_id="t1", domains=["technical"],
        )
        act = SimpleNamespace(template=tmpl, match_score=0.8)
        stack = SimpleNamespace(
            activations=[act],
            direction="bullish",
            tier="low",
            stack_confidence=0.5,
            independent_pairs=0,
            created_at="2024-01-15T12:00:00",
        )

        oc.register_pattern_stack(stack, "AAPL")

        call_args = oc.tracker.record_prediction.call_args
        window = call_args.kwargs.get("outcome_window_days",
                                       call_args[1].get("outcome_window_days", 0))
        assert window == 14
        metadata = call_args.kwargs.get("metadata", call_args[1].get("metadata", {}))
        assert metadata["outcome_window_source"] == "fallback"
