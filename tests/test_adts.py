"""
test_adts.py - Adaptive Dynamic Thompson Sampling tests

Verifies that regime-aware forgetting applies the correct decay rate
for each market regime and that the fallback/default path works.
"""

import pytest
from pathlib import Path
import tempfile

from mae_core.market.intelligence.thompson_sampler import (
    ThompsonSampler,
    REGIME_DECAY_RATES,
)


@pytest.fixture
def sampler(tmp_path):
    """Fresh ThompsonSampler backed by a temp file."""
    return ThompsonSampler(
        persistence_path=tmp_path / "thompson.json",
        seed_from_reliability=False,
    )


# ---------------------------------------------------------------------------
# REGIME_DECAY_RATES dict
# ---------------------------------------------------------------------------

def test_all_regimes_present():
    required = {"volatile", "bear", "bull", "sideways", "default"}
    assert required == set(REGIME_DECAY_RATES.keys())


def test_volatile_fastest():
    assert REGIME_DECAY_RATES["volatile"] < REGIME_DECAY_RATES["bear"]
    assert REGIME_DECAY_RATES["volatile"] < REGIME_DECAY_RATES["bull"]
    assert REGIME_DECAY_RATES["volatile"] < REGIME_DECAY_RATES["sideways"]


def test_sideways_slowest_non_default():
    assert REGIME_DECAY_RATES["sideways"] > REGIME_DECAY_RATES["bull"]
    assert REGIME_DECAY_RATES["sideways"] > REGIME_DECAY_RATES["bear"]
    assert REGIME_DECAY_RATES["sideways"] > REGIME_DECAY_RATES["volatile"]


def test_all_rates_between_zero_and_one():
    for regime, rate in REGIME_DECAY_RATES.items():
        assert 0.0 < rate < 1.0, f"Rate for {regime!r} out of range: {rate}"


# ---------------------------------------------------------------------------
# regime_aware_forget() method
# ---------------------------------------------------------------------------

def test_regime_aware_forget_returns_count(sampler):
    sampler.update("sig_a", success=True)
    sampler.update("sig_a", success=False)
    count = sampler.regime_aware_forget("bull")
    assert count >= 1


def test_regime_aware_forget_volatile_decays_more(tmp_path):
    """Volatile should reduce alpha/beta more than sideways in one forgetting call."""
    def _make_sampler(regime):
        s = ThompsonSampler(
            persistence_path=tmp_path / f"thompson_{regime}.json",
            seed_from_reliability=False,
        )
        # Give it some evidence so decay is visible
        for _ in range(5):
            s.update("sig", success=True)
        for _ in range(5):
            s.update("sig", success=False)
        before = s.get_distribution("sig").alpha
        s.regime_aware_forget(regime)
        after = s.get_distribution("sig").alpha
        return before, after

    before_v, after_v = _make_sampler("volatile")
    before_s, after_s = _make_sampler("sideways")

    decay_volatile = after_v / before_v
    decay_sideways = after_s / before_s

    assert decay_volatile < decay_sideways, (
        f"Volatile decay ratio {decay_volatile:.4f} should be lower than "
        f"sideways {decay_sideways:.4f}"
    )


def test_regime_aware_forget_unknown_regime_uses_default(sampler):
    """Unrecognised regime string should fall back to 'default' rate without error."""
    sampler.update("sig", success=True)
    count = sampler.regime_aware_forget("unknown_regime_xyz")
    assert count >= 1


def test_regime_aware_forget_all_known_regimes(sampler):
    """Calling with every defined regime should not raise."""
    sampler.update("sig", success=True)
    for regime in REGIME_DECAY_RATES:
        count = sampler.regime_aware_forget(regime)
        assert count >= 1


def test_floor_respected_after_aggressive_forgetting(tmp_path):
    """After many volatile forgetting cycles, alpha/beta must not drop below 2.0."""
    s = ThompsonSampler(
        persistence_path=tmp_path / "thompson_floor.json",
        seed_from_reliability=False,
    )
    s.update("sig", success=True)
    for _ in range(50):
        s.regime_aware_forget("volatile")
    dist = s.get_distribution("sig")
    assert dist.alpha >= 2.0
    assert dist.beta >= 2.0
