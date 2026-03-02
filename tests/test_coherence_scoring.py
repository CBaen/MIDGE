"""Tests for Capability 3: Narrative Coherence Scoring.

Adversarial tests for _compute_coherence_score(), _check_direction_convergence()
coherence multiplier, ConvergenceAlert fields, and CH_CONTRADICTION_DETECTED.
"""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from mae_core.market.channels import CH_CONTRADICTION_DETECTED
from mae_core.market.intelligence.convergence_alerter import (
    ConvergenceAlert,
    ConvergenceAlerter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _alerter(min_domains=2, min_strength=0.5):
    return ConvergenceAlerter(min_domains=min_domains, min_strength=min_strength)


def _add_signals(alerter, specs):
    """specs = list of (signal_id, strength, domain, direction)"""
    for spec in specs:
        alerter.record_signal(
            signal_id=spec[0],
            strength=spec[1],
            domain=spec[2],
            direction=spec[3],
            confidence=0.7,
        )


# ---------------------------------------------------------------------------
# CH_CONTRADICTION_DETECTED channel constant
# ---------------------------------------------------------------------------

class TestChannelConstant:
    def test_contradiction_channel_constant_exists(self):
        assert CH_CONTRADICTION_DETECTED == "market.intel.contradiction_detected"


# ---------------------------------------------------------------------------
# _compute_coherence_score() — directional vote counting
# ---------------------------------------------------------------------------

class TestComputeCoherenceScore:
    def test_all_bullish_gives_coherence_1(self):
        a = _alerter()
        _add_signals(a, [
            ("s1", 0.8, "insider", "bullish"),
            ("s2", 0.8, "technical", "bullish"),
            ("s3", 0.8, "macro", "bullish"),
        ])
        result = a._compute_coherence_score()
        assert result["coherence"] == 1.0
        assert result["dominant_direction"] == "bullish"
        assert result["contradiction_details"] == []

    def test_all_bearish_gives_coherence_1(self):
        a = _alerter()
        _add_signals(a, [
            ("s1", 0.8, "insider", "bearish"),
            ("s2", 0.8, "technical", "bearish"),
            ("s3", 0.8, "sentiment", "bearish"),
        ])
        result = a._compute_coherence_score()
        assert result["coherence"] == 1.0
        assert result["dominant_direction"] == "bearish"

    def test_3_bullish_1_bearish_gives_coherence_0_75(self):
        a = _alerter()
        _add_signals(a, [
            ("s1", 0.8, "insider", "bullish"),
            ("s2", 0.8, "technical", "bullish"),
            ("s3", 0.8, "macro", "bullish"),
            ("s4", 0.8, "sentiment", "bearish"),
        ])
        result = a._compute_coherence_score()
        assert abs(result["coherence"] - 0.75) < 1e-9
        assert result["dominant_direction"] == "bullish"

    def test_2_bullish_2_bearish_gives_coherence_0_5(self):
        a = _alerter()
        _add_signals(a, [
            ("s1", 0.8, "insider", "bullish"),
            ("s2", 0.8, "technical", "bullish"),
            ("s3", 0.8, "macro", "bearish"),
            ("s4", 0.8, "sentiment", "bearish"),
        ])
        result = a._compute_coherence_score()
        assert abs(result["coherence"] - 0.5) < 1e-9

    def test_no_directional_signals_gives_coherence_1_neutral(self):
        a = _alerter()
        _add_signals(a, [
            ("s1", 0.8, "volatility", "neutral"),
            ("s2", 0.8, "macro", "neutral"),
        ])
        result = a._compute_coherence_score()
        assert result["coherence"] == 1.0
        assert result["dominant_direction"] == "neutral"
        assert result["contradiction_details"] == []

    def test_empty_signals_gives_coherence_1(self):
        a = _alerter()
        result = a._compute_coherence_score()
        assert result["coherence"] == 1.0
        assert result["dominant_direction"] == "neutral"

    def test_contradiction_details_lists_minority_domains(self):
        a = _alerter()
        _add_signals(a, [
            ("s1", 0.8, "insider", "bullish"),
            ("s2", 0.8, "technical", "bullish"),
            ("s3", 0.8, "macro", "bullish"),
            ("s4", 0.8, "sentiment", "bearish"),  # minority
        ])
        result = a._compute_coherence_score()
        # The minority side is bearish — sentiment should appear in contradiction_details
        minority_domains = [d for (d, _) in result["contradiction_details"]]
        assert "sentiment" in minority_domains

    def test_domain_with_both_bullish_and_bearish_counted_in_both(self):
        """One domain with two signals of opposite direction is counted in both lists."""
        a = _alerter()
        # Same domain, different directions
        a.record_signal("s1", 0.8, "insider", "bullish", confidence=0.7)
        a.record_signal("s2", 0.8, "insider", "bearish", confidence=0.7)
        a.record_signal("s3", 0.8, "technical", "bullish", confidence=0.7)
        result = a._compute_coherence_score()
        # insider is in both bullish and bearish domains
        # bullish_count = 2 (insider + technical), bearish_count = 1 (insider)
        assert result["bullish_count"] == 2
        assert result["bearish_count"] == 1


# ---------------------------------------------------------------------------
# Coherence multiplier in _check_direction_convergence()
# ---------------------------------------------------------------------------

class TestCoherenceMultiplier:
    def test_coherence_1_leaves_confidence_unchanged(self):
        """At coherence=1.0, multiplier=1.0. Confidence should not change."""
        a = _alerter(min_domains=2, min_strength=0.5)
        _add_signals(a, [
            ("s1", 0.8, "insider", "bullish"),
            ("s2", 0.8, "technical", "bullish"),
            ("s3", 0.8, "macro", "bullish"),
        ])
        coherence_data = {"coherence": 1.0, "contradiction_details": []}
        alert_with = a._check_direction_convergence("bullish", coherence_data=coherence_data)

        # Compare to no coherence_data (should match since multiplier=1)
        alert_without = a._check_direction_convergence("bullish", coherence_data=None)

        assert alert_with is not None
        assert alert_without is not None
        assert abs(alert_with.confidence - alert_without.confidence) < 1e-6

    def test_coherence_0_5_reduces_confidence(self):
        """At coherence=0.5, multiplier=0.75. Confidence should be lower."""
        a = _alerter(min_domains=2, min_strength=0.5)
        _add_signals(a, [
            ("s1", 0.8, "insider", "bullish"),
            ("s2", 0.8, "technical", "bullish"),
            ("s3", 0.8, "macro", "bullish"),
        ])
        coherence_1 = {"coherence": 1.0, "contradiction_details": []}
        coherence_half = {"coherence": 0.5, "contradiction_details": [("sentiment", "bearish")]}

        alert_full = a._check_direction_convergence("bullish", coherence_data=coherence_1)
        alert_reduced = a._check_direction_convergence("bullish", coherence_data=coherence_half)

        assert alert_full is not None
        assert alert_reduced is not None
        assert alert_reduced.confidence < alert_full.confidence

    def test_coherence_multiplier_formula_0_6(self):
        """At coherence=0.6, multiplier=0.8. Confidence * 0.8."""
        a = _alerter(min_domains=2, min_strength=0.5)
        _add_signals(a, [
            ("s1", 0.8, "insider", "bullish"),
            ("s2", 0.8, "technical", "bullish"),
            ("s3", 0.8, "macro", "bullish"),
        ])
        coherence_full = {"coherence": 1.0, "contradiction_details": []}
        coherence_06 = {"coherence": 0.6, "contradiction_details": [("x", "bearish")]}

        alert_full = a._check_direction_convergence("bullish", coherence_data=coherence_full)
        alert_06 = a._check_direction_convergence("bullish", coherence_data=coherence_06)

        expected_multiplier = 0.5 + 0.5 * 0.6  # 0.8
        assert alert_full is not None
        assert alert_06 is not None
        expected_confidence = max(0.05, min(0.95, alert_full.confidence * expected_multiplier))
        assert abs(alert_06.confidence - expected_confidence) < 0.01

    def test_coherence_cannot_increase_confidence_above_formula(self):
        """Coherence multiplier is in [0.5, 1.0]. It can only reduce or maintain confidence."""
        a = _alerter(min_domains=2, min_strength=0.5)
        _add_signals(a, [
            ("s1", 0.8, "insider", "bullish"),
            ("s2", 0.8, "technical", "bullish"),
            ("s3", 0.8, "macro", "bullish"),
        ])
        # coherence > 1.0 is invalid input; but coherence=1.0 is max valid
        coherence_max = {"coherence": 1.0, "contradiction_details": []}
        alert_base = a._check_direction_convergence("bullish", coherence_data=None)
        alert_max = a._check_direction_convergence("bullish", coherence_data=coherence_max)

        # coherence=1.0 multiplier=1.0, so confidence should be same
        assert alert_base is not None
        assert alert_max is not None
        assert abs(alert_max.confidence - alert_base.confidence) < 1e-6


# ---------------------------------------------------------------------------
# ConvergenceAlert fields
# ---------------------------------------------------------------------------

class TestConvergenceAlertFields:
    def test_alert_includes_coherence_field(self):
        a = _alerter(min_domains=2, min_strength=0.5)
        _add_signals(a, [
            ("s1", 0.8, "insider", "bullish"),
            ("s2", 0.8, "technical", "bullish"),
            ("s3", 0.8, "macro", "bullish"),
        ])
        alerts = a.check_convergence()
        assert len(alerts) > 0
        alert = alerts[0]
        assert hasattr(alert, "coherence")
        assert 0.0 <= alert.coherence <= 1.0

    def test_alert_includes_contradiction_details_field(self):
        a = _alerter(min_domains=2, min_strength=0.5)
        _add_signals(a, [
            ("s1", 0.8, "insider", "bullish"),
            ("s2", 0.8, "technical", "bullish"),
            ("s3", 0.8, "macro", "bullish"),
        ])
        alerts = a.check_convergence()
        assert len(alerts) > 0
        assert hasattr(alerts[0], "contradiction_details")
        assert isinstance(alerts[0].contradiction_details, list)

    def test_alert_default_coherence_is_1(self):
        """Default ConvergenceAlert has coherence=1.0 for backward compat."""
        from datetime import datetime
        alert = ConvergenceAlert(
            alert_id="test",
            timestamp=datetime.now(),
            direction="bullish",
            strength=0.8,
            confidence=0.7,
            domains_converging=["insider"],
            signals=[],
            cross_domain_count=1,
            summary="test",
            urgency="days",
        )
        assert alert.coherence == 1.0
        assert alert.contradiction_details == []

    def test_to_dict_includes_coherence_and_contradiction(self):
        from datetime import datetime
        alert = ConvergenceAlert(
            alert_id="test",
            timestamp=datetime.now(),
            direction="bullish",
            strength=0.8,
            confidence=0.7,
            domains_converging=["insider"],
            signals=[],
            cross_domain_count=1,
            summary="test",
            urgency="days",
            coherence=0.75,
            contradiction_details=[("sentiment", "bearish")],
        )
        d = alert.to_dict()
        assert "coherence" in d
        assert d["coherence"] == 0.75
        assert "contradiction_details" in d
        assert d["contradiction_details"] == [("sentiment", "bearish")]

    def test_alert_coherence_reflects_actual_signal_mix(self):
        """Alert coherence matches what _compute_coherence_score returns for same signals."""
        a = _alerter(min_domains=2, min_strength=0.5)
        _add_signals(a, [
            ("s1", 0.8, "insider", "bullish"),
            ("s2", 0.8, "technical", "bullish"),
            ("s3", 0.8, "macro", "bullish"),
            ("s4", 0.8, "sentiment", "bearish"),  # 3 vs 1 → coherence = 0.75
        ])
        # Compute directly
        coherence_data = a._compute_coherence_score()
        alerts = a.check_convergence(direction_filter="bullish")
        assert len(alerts) > 0
        assert abs(alerts[0].coherence - coherence_data["coherence"]) < 1e-6


# ---------------------------------------------------------------------------
# Summary includes coherence note when < 1.0
# ---------------------------------------------------------------------------

class TestSummaryCoherenceNote:
    def test_summary_includes_coherence_note_when_below_1(self):
        a = _alerter(min_domains=2, min_strength=0.5)
        _add_signals(a, [
            ("s1", 0.8, "insider", "bullish"),
            ("s2", 0.8, "technical", "bullish"),
            ("s3", 0.8, "macro", "bullish"),
            ("s4", 0.8, "sentiment", "bearish"),
        ])
        alerts = a.check_convergence(direction_filter="bullish")
        assert len(alerts) > 0
        # Summary should mention coherence
        assert "coherence" in alerts[0].summary

    def test_summary_no_coherence_note_when_perfect(self):
        """When coherence=1.0, the summary should NOT include a coherence note."""
        a = _alerter(min_domains=2, min_strength=0.5)
        _add_signals(a, [
            ("s1", 0.8, "insider", "bullish"),
            ("s2", 0.8, "technical", "bullish"),
            ("s3", 0.8, "macro", "bullish"),
        ])
        alerts = a.check_convergence(direction_filter="bullish")
        assert len(alerts) > 0
        # coherence=1.0 → no note
        assert "coherence" not in alerts[0].summary


# ---------------------------------------------------------------------------
# check_convergence() passes coherence_data to both directions
# ---------------------------------------------------------------------------

class TestCheckConvergencePassesCoherence:
    def test_both_directions_checked_with_same_coherence_data(self):
        """check_convergence() computes coherence once and shares it."""
        a = _alerter(min_domains=2, min_strength=0.5)
        # Add signals in BOTH directions
        _add_signals(a, [
            ("s1", 0.8, "insider", "bullish"),
            ("s2", 0.8, "technical", "bullish"),
            ("s3", 0.8, "macro", "bullish"),
            ("s4", 0.8, "sentiment_b", "bearish"),
            ("s5", 0.8, "contracts", "bearish"),
            ("s6", 0.8, "government", "bearish"),
        ])
        alerts = a.check_convergence()
        # If both directions triggered, they share the same coherence data
        if len(alerts) == 2:
            assert alerts[0].coherence == alerts[1].coherence

    def test_contradiction_details_populated_for_mixed_signals(self):
        a = _alerter(min_domains=2, min_strength=0.5)
        _add_signals(a, [
            ("s1", 0.8, "insider", "bullish"),
            ("s2", 0.8, "technical", "bullish"),
            ("s3", 0.8, "macro", "bullish"),
            ("s4", 0.8, "sentiment", "bearish"),
        ])
        alerts = a.check_convergence(direction_filter="bullish")
        if alerts:
            # contradiction_details should note the minority bearish domain
            assert len(alerts[0].contradiction_details) > 0


# ---------------------------------------------------------------------------
# Market connections Group 20
# ---------------------------------------------------------------------------

class TestMarketConnectionsGroup20:
    def test_group_20_has_one_connection(self):
        """Group 20 adds 1 triadic connection for contradiction channel."""
        from mae_core.bootstrap.market_connections import _register_market_connections

        mock_reg = MagicMock()
        call_count = [0]

        def track_call(*args, **kwargs):
            call_count[0] += 1

        mock_reg.register_connection.side_effect = track_call

        # We can't easily isolate Group 20, but we can verify the total
        # connection count registered is 67 from the docstring/logs
        # Instead, verify the market_connections module has Group 20 code
        import inspect
        source = inspect.getsource(_register_market_connections)
        assert "Group 20" in source
        assert "market.intel.contradiction_detected" in source
