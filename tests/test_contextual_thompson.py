"""Tests for Capability 2: Contextual Thompson.

Adversarial tests for the contextual cascade in _resolve_thompson_key()
and the 15 contextual priors in learning_config.py.
"""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from mae_core.market.intelligence.convergence_alerter import (
    ConvergenceAlerter,
    Signal,
)
from mae_core.market.intelligence.learning_config import LEARNING_CONFIG


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal(
    source="sec_form4",
    direction="bullish",
    metadata=None,
):
    return Signal(
        signal_id="test-sig",
        strength=0.8,
        domain="insider",
        direction=direction,
        timestamp=datetime.now(),
        metadata=metadata or {},
        source=source,
    )


def _make_thompson(distributions: dict):
    """Create a mock ThompsonSampler where get_distribution returns fake dists."""
    mock = MagicMock()

    def get_dist(key, regime="default"):
        if key in distributions:
            d = MagicMock()
            d.samples = distributions[key]["samples"]
            d.mean = distributions[key]["mean"]
            return d
        raise KeyError(f"No distribution for {key}")

    mock.get_distribution.side_effect = get_dist
    return mock


# ---------------------------------------------------------------------------
# Static map fallback (no metadata or no Thompson)
# ---------------------------------------------------------------------------

class TestStaticMapFallback:
    def test_no_metadata_falls_through_to_static_map(self):
        alerter = ConvergenceAlerter()
        sig = _make_signal(source="sec_form4", metadata={})
        key = alerter._resolve_thompson_key(sig)
        assert key == "sec_form4"

    def test_no_thompson_returns_static_map_key(self):
        alerter = ConvergenceAlerter(thompson_sampler=None)
        sig = _make_signal(source="finra_short", metadata={"role": "large_change"})
        key = alerter._resolve_thompson_key(sig)
        # No Thompson → static map fallback always
        assert key == "finra_short"

    def test_unknown_source_returns_source_itself(self):
        alerter = ConvergenceAlerter(thompson_sampler=None)
        sig = _make_signal(source="totally_unknown_source", metadata={})
        key = alerter._resolve_thompson_key(sig)
        assert key == "totally_unknown_source"

    def test_sweep_sources_still_use_bridge2_not_contextual(self):
        """session_sweep and session_sweep_ifvg must use Bridge 2 cascade, not contextual."""
        thompson = _make_thompson({
            "sweep_bt:ES=F:bullish": {"samples": 10, "mean": 0.6},
        })
        alerter = ConvergenceAlerter(thompson_sampler=thompson)
        sig = _make_signal(
            source="session_sweep",
            direction="bullish",
            metadata={"symbol": "ES=F", "role": "officer"},  # role present but irrelevant
        )
        key = alerter._resolve_thompson_key(sig)
        # Must resolve via Bridge 2, NOT contextual {source}:{role}
        assert key == "sweep_bt:ES=F:bullish"

    def test_session_sweep_ifvg_uses_bridge2_not_contextual(self):
        thompson = _make_thompson({
            "sweep_bt:NQ=F": {"samples": 8, "mean": 0.55},
        })
        alerter = ConvergenceAlerter(thompson_sampler=thompson)
        sig = _make_signal(
            source="session_sweep_ifvg",
            direction="bearish",
            metadata={"symbol": "NQ=F", "role": "officer"},
        )
        key = alerter._resolve_thompson_key(sig)
        assert key == "sweep_bt:NQ=F"


# ---------------------------------------------------------------------------
# Contextual cascade — role-based
# ---------------------------------------------------------------------------

class TestContextualCascade:
    def test_role_key_selected_when_mature(self):
        thompson = _make_thompson({
            "sec_form4:officer": {"samples": 7, "mean": 0.65},
        })
        alerter = ConvergenceAlerter(thompson_sampler=thompson)
        sig = _make_signal(source="sec_form4", metadata={"role": "officer"})
        key = alerter._resolve_thompson_key(sig)
        assert key == "sec_form4:officer"

    def test_role_key_skipped_when_immature(self):
        """Only picks contextual key if dist.samples >= 5."""
        thompson = _make_thompson({
            "sec_form4:officer": {"samples": 3, "mean": 0.65},
            # No fallback key in our mock → will raise KeyError → falls to static map
        })
        alerter = ConvergenceAlerter(thompson_sampler=thompson)
        sig = _make_signal(source="sec_form4", metadata={"role": "officer"})
        key = alerter._resolve_thompson_key(sig)
        # 3 < 5 → immature → static map fallback
        assert key == "sec_form4"

    def test_most_specific_wins_role_sector_size(self):
        """Most specific key with >= 5 samples wins."""
        thompson = _make_thompson({
            "sec_form4:officer:tech:large": {"samples": 6, "mean": 0.70},
            "sec_form4:officer:tech": {"samples": 8, "mean": 0.65},
            "sec_form4:officer:large": {"samples": 7, "mean": 0.62},
            "sec_form4:officer": {"samples": 10, "mean": 0.60},
        })
        alerter = ConvergenceAlerter(thompson_sampler=thompson)
        sig = _make_signal(
            source="sec_form4",
            metadata={"role": "officer", "sector": "tech", "transaction_value": 1_000_000},
        )
        key = alerter._resolve_thompson_key(sig)
        assert key == "sec_form4:officer:tech:large"

    def test_falls_through_to_role_when_sector_immature(self):
        thompson = _make_thompson({
            "sec_form4:officer:tech:large": {"samples": 2, "mean": 0.70},
            "sec_form4:officer:tech": {"samples": 1, "mean": 0.65},
            "sec_form4:officer:large": {"samples": 3, "mean": 0.62},
            "sec_form4:officer": {"samples": 6, "mean": 0.60},
        })
        alerter = ConvergenceAlerter(thompson_sampler=thompson)
        sig = _make_signal(
            source="sec_form4",
            metadata={"role": "officer", "sector": "tech", "transaction_value": 1_000_000},
        )
        key = alerter._resolve_thompson_key(sig)
        assert key == "sec_form4:officer"

    def test_falls_through_to_static_when_all_contextual_immature(self):
        thompson = _make_thompson({
            "sec_form4:officer": {"samples": 2, "mean": 0.65},
        })
        alerter = ConvergenceAlerter(thompson_sampler=thompson)
        sig = _make_signal(source="sec_form4", metadata={"role": "officer"})
        key = alerter._resolve_thompson_key(sig)
        assert key == "sec_form4"


# ---------------------------------------------------------------------------
# Size derivation
# ---------------------------------------------------------------------------

class TestSizeDerivedFromTransactionValue:
    def test_large_when_tx_over_500k(self):
        thompson = _make_thompson({
            "sec_form4:officer:large": {"samples": 8, "mean": 0.65},
        })
        alerter = ConvergenceAlerter(thompson_sampler=thompson)
        sig = _make_signal(
            source="sec_form4",
            metadata={"role": "officer", "transaction_value": 600_000},
        )
        key = alerter._resolve_thompson_key(sig)
        assert key == "sec_form4:officer:large"

    def test_small_when_tx_between_0_and_500k(self):
        thompson = _make_thompson({
            "sec_form4:officer:small": {"samples": 8, "mean": 0.50},
        })
        alerter = ConvergenceAlerter(thompson_sampler=thompson)
        sig = _make_signal(
            source="sec_form4",
            metadata={"role": "officer", "transaction_value": 100_000},
        )
        key = alerter._resolve_thompson_key(sig)
        assert key == "sec_form4:officer:small"

    def test_empty_size_when_no_transaction_value(self):
        thompson = _make_thompson({
            "sec_form4:officer": {"samples": 8, "mean": 0.60},
        })
        alerter = ConvergenceAlerter(thompson_sampler=thompson)
        sig = _make_signal(
            source="sec_form4",
            metadata={"role": "officer"},  # no transaction_value
        )
        key = alerter._resolve_thompson_key(sig)
        # Without size, only role-level key should resolve
        assert key == "sec_form4:officer"

    def test_tx_exactly_500k_is_small_not_large(self):
        """Boundary: > 500K is large; == 500K should be small."""
        thompson = _make_thompson({
            "sec_form4:officer:small": {"samples": 8, "mean": 0.50},
        })
        alerter = ConvergenceAlerter(thompson_sampler=thompson)
        sig = _make_signal(
            source="sec_form4",
            metadata={"role": "officer", "transaction_value": 500_000},
        )
        key = alerter._resolve_thompson_key(sig)
        # 500_000 is NOT > 500_000, so size = "small"
        assert key == "sec_form4:officer:small"

    def test_tx_zero_produces_empty_size(self):
        """tx_value == 0 → size = "" → no size dimension in key."""
        thompson = _make_thompson({
            "sec_form4:officer": {"samples": 8, "mean": 0.60},
        })
        alerter = ConvergenceAlerter(thompson_sampler=thompson)
        sig = _make_signal(
            source="sec_form4",
            metadata={"role": "officer", "transaction_value": 0},
        )
        key = alerter._resolve_thompson_key(sig)
        assert key == "sec_form4:officer"


# ---------------------------------------------------------------------------
# Multiple signals, different metadata → different keys
# ---------------------------------------------------------------------------

class TestMultipleSignalsDifferentContexts:
    def test_two_signals_same_source_different_roles_resolve_differently(self):
        thompson = _make_thompson({
            "sec_form4:officer": {"samples": 6, "mean": 0.65},
            "sec_form4:director": {"samples": 7, "mean": 0.40},
        })
        alerter = ConvergenceAlerter(thompson_sampler=thompson)

        sig_officer = _make_signal(source="sec_form4", metadata={"role": "officer"})
        sig_director = _make_signal(source="sec_form4", metadata={"role": "director"})

        key_officer = alerter._resolve_thompson_key(sig_officer)
        key_director = alerter._resolve_thompson_key(sig_director)

        assert key_officer == "sec_form4:officer"
        assert key_director == "sec_form4:director"
        assert key_officer != key_director


# ---------------------------------------------------------------------------
# learning_config has the 15 contextual priors
# ---------------------------------------------------------------------------

class TestLearningConfigContextualPriors:
    def test_contextual_priors_present_in_source_reliability(self):
        sr = LEARNING_CONFIG["source_reliability"]
        expected_contextual_keys = [
            "sec_form4:officer",
            "sec_form4:officer:large",
            "sec_form4:director",
            "sec_form4:10pct_owner",
            "congressional:senate",
            "congressional:house",
            "finra_short:large_change",
            "finra_short:small_change",
            "finnhub_earnings:beat",
            "finnhub_earnings:miss",
            "contract_award:defense",
            "contract_award:tech",
            "ta_rsi:oversold",
            "ta_rsi:overbought",
            "ta_macd:crossover",
        ]
        for key in expected_contextual_keys:
            assert key in sr, f"Missing contextual prior: {key}"

    def test_contextual_priors_are_valid_probabilities(self):
        sr = LEARNING_CONFIG["source_reliability"]
        contextual_keys = [k for k in sr if ":" in k]
        for key in contextual_keys:
            val = sr[key]
            assert 0.0 <= val <= 1.0, f"Contextual prior {key}={val} out of [0,1]"

    def test_fifteen_contextual_priors(self):
        sr = LEARNING_CONFIG["source_reliability"]
        contextual_keys = [k for k in sr if ":" in k]
        assert len(contextual_keys) >= 15, (
            f"Expected at least 15 contextual priors, found {len(contextual_keys)}: {contextual_keys}"
        )


# ---------------------------------------------------------------------------
# Thompson weight uses contextual distribution
# ---------------------------------------------------------------------------

class TestThompsonWeightContextual:
    def test_weight_uses_contextual_distribution_when_mature(self):
        """With a mature contextual dist, weight should differ from base weight."""
        base_thompson = _make_thompson({
            "sec_form4": {"samples": 20, "mean": 0.36},
            "sec_form4:officer": {"samples": 10, "mean": 0.65},
        })
        alerter = ConvergenceAlerter(thompson_sampler=base_thompson)

        sig_with_role = _make_signal(source="sec_form4", metadata={"role": "officer"})
        sig_no_role = _make_signal(source="sec_form4", metadata={})

        regime = "default"
        weight_contextual = alerter._get_thompson_weight(sig_with_role, regime)
        weight_base = alerter._get_thompson_weight(sig_no_role, regime)

        # Contextual dist has mean=0.65 → weight 0.5+0.65=1.15
        # Base dist has mean=0.36 → weight 0.5+0.36=0.86
        assert weight_contextual > weight_base
