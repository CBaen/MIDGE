"""Tests for domain independence correction in ConvergenceAlerter.

Round 1 Corrector build — 2026-03-05.

Verifies:
- Correlated domains (macro+technical) produce LOWER confidence than
  uncorrelated domains (insider+government) when correlation data is present.
- CorrelationTracker=None falls back gracefully (backward compat).
- _compute_effective_domain_count with 3 domains where 2 are correlated → < 3.0.
- _compute_effective_domain_count with 3 fully independent domains → 3.0.
- seed_from_lag_data correctly populates correlation state.
"""

import json
import math
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mae_core.market.intelligence.convergence_alerter import ConvergenceAlerter, Signal
from mae_core.market.intelligence.correlation_tracker import CorrelationTracker, CorrelationPair


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal(
    domain: str,
    confidence: float = 0.7,
    strength: float = 0.7,
    source: str = "",
    direction: str = "bullish",
) -> Signal:
    return Signal(
        signal_id=f"{domain}_signal",
        strength=strength,
        domain=domain,
        direction=direction,
        timestamp=datetime.now(),
        confidence=confidence,
        source=source,
    )


def _alerter_with_tracker(tracker) -> ConvergenceAlerter:
    """Build a minimal ConvergenceAlerter with a given CorrelationTracker."""
    return ConvergenceAlerter(
        min_domains=2,
        correlation_tracker=tracker,
    )


def _tracker_with_pair(src_a: str, src_b: str, corr: float) -> CorrelationTracker:
    """Build a CorrelationTracker with a single seeded correlation pair."""
    ct = CorrelationTracker()
    key = (src_a, src_b) if src_a < src_b else (src_b, src_a)
    ct.correlations[key] = CorrelationPair(
        signal_a=key[0],
        signal_b=key[1],
        current_correlation=corr,
        observation_count=5,
    )
    return ct


# ---------------------------------------------------------------------------
# Task 5 — Backward compatibility: correlation_tracker=None
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_no_correlation_tracker_constructs_ok(self):
        """ConvergenceAlerter must construct without correlation_tracker."""
        alerter = ConvergenceAlerter()
        assert alerter._correlation_tracker is None

    def test_no_correlation_tracker_compute_confidence_returns_float(self):
        """_compute_confidence must work even without a CorrelationTracker."""
        alerter = ConvergenceAlerter()
        signals = [
            _make_signal("macro", confidence=0.8),
            _make_signal("technical", confidence=0.7),
            _make_signal("insider", confidence=0.75),
        ]
        result = alerter._compute_confidence(signals, cross_domain_count=3)
        assert isinstance(result, float)
        assert 0.05 <= result <= 0.95

    def test_no_correlation_tracker_uses_raw_domain_count(self):
        """Without CorrelationTracker, effective count equals raw domain count."""
        alerter_no_tracker = ConvergenceAlerter()
        alerter_with_tracker = ConvergenceAlerter(
            correlation_tracker=CorrelationTracker()  # empty tracker — no correlations
        )
        signals = [
            _make_signal("macro", confidence=0.7),
            _make_signal("insider", confidence=0.7),
            _make_signal("government", confidence=0.7),
        ]
        # Both should produce the same result: no correlation data available either way
        conf_no_tracker = alerter_no_tracker._compute_confidence(signals, cross_domain_count=3)
        conf_empty_tracker = alerter_with_tracker._compute_confidence(signals, cross_domain_count=3)
        # With empty tracker all domains are independent → same as no tracker
        assert abs(conf_no_tracker - conf_empty_tracker) < 0.01


# ---------------------------------------------------------------------------
# Task 6 — Effective domain count
# ---------------------------------------------------------------------------

class TestComputeEffectiveDomainCount:
    def test_single_domain_returns_1(self):
        """A single domain always contributes a full count of 1.0."""
        alerter = _alerter_with_tracker(CorrelationTracker())
        result = alerter._compute_effective_domain_count(["macro"])
        assert result == 1.0

    def test_empty_domains_returns_0(self):
        """Empty domain list returns 0.0."""
        alerter = _alerter_with_tracker(CorrelationTracker())
        result = alerter._compute_effective_domain_count([])
        assert result == 0.0

    def test_three_independent_domains_return_3(self):
        """Three domains with no correlation data → effective count = 3.0 (full credit each)."""
        # Empty tracker: no correlations → all treated as independent
        alerter = _alerter_with_tracker(CorrelationTracker())
        result = alerter._compute_effective_domain_count(["insider", "government", "positioning"])
        assert result == 3.0

    def test_two_strongly_correlated_domains_half_credit(self):
        """When two domains are strongly correlated (|r|>0.5), second gets 0.5 credit.

        macro uses fred_macro as a source; technical uses yfinance_price.
        We seed |r|=0.73 for that pair.
        """
        tracker = _tracker_with_pair("fred_macro", "yfinance_price", 0.73)
        alerter = _alerter_with_tracker(tracker)
        result = alerter._compute_effective_domain_count(["macro", "technical"])
        # First domain: 1.0, second strongly correlated: +0.5 → total 1.5
        assert abs(result - 1.5) < 1e-9

    def test_two_moderately_correlated_domains_partial_credit(self):
        """When |r| is 0.31-0.50, second domain gets 0.7 credit."""
        tracker = _tracker_with_pair("fred_macro", "yfinance_price", 0.4)
        alerter = _alerter_with_tracker(tracker)
        result = alerter._compute_effective_domain_count(["macro", "technical"])
        # First: 1.0, second moderately correlated: +0.7 → total 1.7
        assert abs(result - 1.7) < 1e-9

    def test_three_domains_two_correlated_less_than_3(self):
        """3 domains where 2 are strongly correlated → effective count < 3.0."""
        # macro and technical are strongly correlated; insider is independent of both
        tracker = _tracker_with_pair("fred_macro", "yfinance_price", 0.73)
        alerter = _alerter_with_tracker(tracker)
        # domains[0]=macro → 1.0
        # domains[1]=technical → correlated with macro (|r|=0.73 > 0.5) → +0.5
        # domains[2]=insider → no data for insider sources vs macro/technical → +1.0
        result = alerter._compute_effective_domain_count(["macro", "technical", "insider"])
        assert result < 3.0
        assert abs(result - 2.5) < 1e-9

    def test_three_fully_independent_domains_return_3(self):
        """3 domains with no correlation data all get full credit → 3.0."""
        # Use domains whose sources don't appear in lag_correlations at all
        tracker = CorrelationTracker()  # completely empty
        alerter = _alerter_with_tracker(tracker)
        result = alerter._compute_effective_domain_count(["crypto", "positioning", "volatility"])
        assert result == 3.0


# ---------------------------------------------------------------------------
# Task 6 — Confidence comparison: correlated vs independent
# ---------------------------------------------------------------------------

class TestCorrelatedVsIndependentConfidence:
    def test_correlated_domains_produce_lower_confidence(self):
        """macro+technical (correlated) should yield lower confidence than insider+government (independent).

        Same per-signal confidence, same number of domains — the only difference
        is the correlation between domains.
        """
        # Seed high correlation between fred_macro and yfinance_price (macro ↔ technical)
        tracker = _tracker_with_pair("fred_macro", "yfinance_price", 0.73)

        alerter = _alerter_with_tracker(tracker)

        # Correlated pair: macro + technical
        correlated_signals = [
            _make_signal("macro", confidence=0.75),
            _make_signal("technical", confidence=0.75),
        ]
        conf_correlated = alerter._compute_confidence(correlated_signals, cross_domain_count=2)

        # Independent pair: insider + government — no correlation data → full credit
        independent_signals = [
            _make_signal("insider", confidence=0.75),
            _make_signal("government", confidence=0.75),
        ]
        conf_independent = alerter._compute_confidence(independent_signals, cross_domain_count=2)

        assert conf_correlated < conf_independent, (
            f"Correlated confidence {conf_correlated:.4f} should be less than "
            f"independent confidence {conf_independent:.4f}"
        )

    def test_confidence_without_tracker_equals_full_credit_baseline(self):
        """Without CorrelationTracker, all domains treated as independent (full credit).

        This verifies the fallback produces the same result as wiring a tracker
        with no correlation data (the no-tracker case should match all-independent).
        """
        alerter_no_tracker = ConvergenceAlerter()
        alerter_empty_tracker = _alerter_with_tracker(CorrelationTracker())

        signals = [
            _make_signal("macro", confidence=0.7),
            _make_signal("technical", confidence=0.7),
        ]

        conf_no_tracker = alerter_no_tracker._compute_confidence(signals, cross_domain_count=2)
        conf_empty_tracker = alerter_empty_tracker._compute_confidence(signals, cross_domain_count=2)

        assert abs(conf_no_tracker - conf_empty_tracker) < 0.01, (
            f"No-tracker ({conf_no_tracker:.4f}) should ~equal "
            f"empty-tracker ({conf_empty_tracker:.4f})"
        )


# ---------------------------------------------------------------------------
# Task 7 — seed_from_lag_data
# ---------------------------------------------------------------------------

class TestSeedFromLagData:
    def test_seed_returns_correct_count(self, tmp_path):
        """seed_from_lag_data returns the number of unique pairs seeded."""
        lag_data = [
            {"source_a": "fred_macro", "source_b": "yfinance_price",
             "correlation": 0.73, "lag_days": 7},
            {"source_a": "sec_form4", "source_b": "finra_short",
             "correlation": -0.58, "lag_days": 5},
        ]
        lag_file = tmp_path / "lag_correlations.json"
        lag_file.write_text(json.dumps(lag_data))

        ct = CorrelationTracker()
        n = ct.seed_from_lag_data(str(lag_file))
        assert n == 2

    def test_seed_populates_correlations(self, tmp_path):
        """Seeded pairs are accessible via get_correlation()."""
        lag_data = [
            {"source_a": "fred_macro", "source_b": "yfinance_price",
             "correlation": 0.73, "lag_days": 7},
        ]
        lag_file = tmp_path / "lag_correlations.json"
        lag_file.write_text(json.dumps(lag_data))

        ct = CorrelationTracker()
        ct.seed_from_lag_data(str(lag_file))

        corr = ct.get_correlation("fred_macro", "yfinance_price")
        assert corr is not None
        # Stored as positive absolute value
        assert abs(corr - 0.73) < 1e-9

    def test_seed_deduplicates_by_max_abs_correlation(self, tmp_path):
        """Multiple entries for same pair → only max |correlation| is stored."""
        lag_data = [
            {"source_a": "fred_macro", "source_b": "yfinance_price",
             "correlation": 0.45, "lag_days": 7},
            {"source_a": "fred_macro", "source_b": "yfinance_price",
             "correlation": -0.73, "lag_days": 14},  # |r|=0.73 wins
            {"source_a": "yfinance_price", "source_b": "fred_macro",
             "correlation": 0.31, "lag_days": 3},    # canonical key same pair
        ]
        lag_file = tmp_path / "lag_correlations.json"
        lag_file.write_text(json.dumps(lag_data))

        ct = CorrelationTracker()
        n = ct.seed_from_lag_data(str(lag_file))
        assert n == 1  # just one unique canonical pair

        corr = ct.get_correlation("fred_macro", "yfinance_price")
        assert abs(corr - 0.73) < 1e-9

    def test_seed_does_not_overwrite_live_data(self, tmp_path):
        """If a pair already exists in correlations, seed leaves it unchanged."""
        ct = CorrelationTracker()
        # Pre-populate with live data (higher quality)
        pair_key = ("fred_macro", "yfinance_price")
        ct.correlations[pair_key] = CorrelationPair(
            signal_a="fred_macro",
            signal_b="yfinance_price",
            current_correlation=0.90,
            observation_count=50,
        )

        lag_data = [
            {"source_a": "fred_macro", "source_b": "yfinance_price",
             "correlation": 0.73, "lag_days": 7},
        ]
        lag_file = tmp_path / "lag_correlations.json"
        lag_file.write_text(json.dumps(lag_data))

        ct.seed_from_lag_data(str(lag_file))
        # Live value must be preserved
        assert abs(ct.correlations[pair_key].current_correlation - 0.90) < 1e-9

    def test_seed_returns_zero_on_missing_file(self, tmp_path):
        """seed_from_lag_data returns 0 when file does not exist."""
        ct = CorrelationTracker()
        n = ct.seed_from_lag_data(str(tmp_path / "nonexistent.json"))
        assert n == 0

    def test_seed_returns_zero_on_empty_list(self, tmp_path):
        """seed_from_lag_data returns 0 for an empty JSON array."""
        lag_file = tmp_path / "lag_correlations.json"
        lag_file.write_text("[]")
        ct = CorrelationTracker()
        n = ct.seed_from_lag_data(str(lag_file))
        assert n == 0

    def test_seed_skips_self_pairs(self, tmp_path):
        """Entries where source_a == source_b are skipped."""
        lag_data = [
            {"source_a": "fred_macro", "source_b": "fred_macro",
             "correlation": 1.0, "lag_days": 0},
        ]
        lag_file = tmp_path / "lag_correlations.json"
        lag_file.write_text(json.dumps(lag_data))
        ct = CorrelationTracker()
        n = ct.seed_from_lag_data(str(lag_file))
        assert n == 0

    def test_seed_skips_entries_with_missing_fields(self, tmp_path):
        """Entries with empty source_a or source_b are gracefully skipped."""
        lag_data = [
            {"source_a": "", "source_b": "yfinance_price", "correlation": 0.5},
            {"source_a": "fred_macro", "source_b": "", "correlation": 0.5},
        ]
        lag_file = tmp_path / "lag_correlations.json"
        lag_file.write_text(json.dumps(lag_data))
        ct = CorrelationTracker()
        n = ct.seed_from_lag_data(str(lag_file))
        assert n == 0


# ---------------------------------------------------------------------------
# Integration: effective count flows through to confidence correctly
# ---------------------------------------------------------------------------

class TestEffectiveDomainCountIntegration:
    def test_confidence_scales_with_effective_count(self):
        """Higher effective domain count → higher diversity bonus → higher confidence.

        We compare 3 independent domains vs 3 where 2 are strongly correlated.
        The 3-independent case should produce higher (or equal) confidence.
        """
        tracker_correlated = _tracker_with_pair("fred_macro", "yfinance_price", 0.73)
        tracker_independent = CorrelationTracker()  # empty: all treated as independent

        alerter_corr = _alerter_with_tracker(tracker_correlated)
        alerter_indep = _alerter_with_tracker(tracker_independent)

        signals = [
            _make_signal("macro", confidence=0.7),
            _make_signal("technical", confidence=0.7),
            _make_signal("insider", confidence=0.7),
        ]

        conf_corr = alerter_corr._compute_confidence(signals, cross_domain_count=3)
        conf_indep = alerter_indep._compute_confidence(signals, cross_domain_count=3)

        # Independent domains should get higher diversity bonus
        assert conf_indep >= conf_corr

    def test_max_domain_correlation_with_no_data_returns_0(self):
        """_max_domain_correlation returns 0.0 when tracker has no data for the pair."""
        alerter = _alerter_with_tracker(CorrelationTracker())
        result = alerter._max_domain_correlation("macro", ["insider"])
        assert result == 0.0

    def test_max_domain_correlation_finds_max_across_source_pairs(self):
        """_max_domain_correlation returns the highest |r| across all source pairs."""
        # macro domain → fred_macro is a source
        # technical domain → yfinance_price, ta_rsi are sources
        ct = CorrelationTracker()
        # Seed two source pairs: fred_macro↔yfinance_price at 0.73, fred_macro↔ta_rsi at 0.30
        key1 = ("fred_macro", "yfinance_price")
        ct.correlations[key1] = CorrelationPair(
            signal_a="fred_macro", signal_b="yfinance_price",
            current_correlation=0.73, observation_count=5
        )
        key2 = ("fred_macro", "ta_rsi")
        ct.correlations[key2] = CorrelationPair(
            signal_a="fred_macro", signal_b="ta_rsi",
            current_correlation=0.30, observation_count=5
        )

        alerter = _alerter_with_tracker(ct)
        result = alerter._max_domain_correlation("macro", ["technical"])
        # Should return max(0.73, 0.30) = 0.73
        assert abs(result - 0.73) < 1e-9
