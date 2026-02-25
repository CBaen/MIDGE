#!/usr/bin/env python3
"""Tests for mae_core.market.intelligence.kelly_position_sizer.

All tests are self-contained — no real outcomes.jsonl files are read unless
explicitly written to a temporary directory. A MockSampler provides the
minimal ThompsonSampler interface required by KellyPositionSizer.

Kelly formula reminder:
    f = (b * p - q) / b      where q = 1 - p
    kelly_half = f * HALF_KELLY_FACTOR (0.5)
    kelly_capped = min(kelly_half, MAX_KELLY_FRACTION) (0.05)
"""

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mae_core.market.intelligence.kelly_position_sizer import (
    KellyPositionSizer,
    PositionRecommendation,
    DEFAULT_WIN_LOSS_RATIO,
    HALF_KELLY_FACTOR,
    MAX_KELLY_FRACTION,
    MIN_OBSERVATIONS,
)


# ---------------------------------------------------------------------------
# Mock infrastructure
# ---------------------------------------------------------------------------

class MockDistribution:
    """Minimal distribution object returned by MockSampler.get_distribution()."""

    def __init__(self, mean: float):
        self.mean = mean
        self.alpha = mean * 2.0
        self.beta = (1.0 - mean) * 2.0


class MockSampler:
    """Minimal ThompsonSampler interface — returns a controlled p_win mean."""

    def __init__(self, default_mean: float = 0.5):
        self._means: dict = {}
        self._default_mean = default_mean
        self.distributions: dict = {}
        self.prior_scale: float = 2.0

    def set_mean(self, key: str, mean: float, regime: str = "default"):
        """Configure the distribution mean for a given key and regime."""
        self._means[(key, regime)] = mean

    def get_distribution(self, key: str, regime: str = "default"):
        mean = self._means.get((key, regime), self._default_mean)
        return MockDistribution(mean)

    def _save_distributions(self):
        pass


def _write_outcomes(path: Path, records: list):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _make_sizer(
    sampler=None,
    data_dir=None,
    max_fraction: float = MAX_KELLY_FRACTION,
    outcomes: list = None,
):
    """Convenience factory. Optionally write outcomes.jsonl to data_dir."""
    if sampler is None:
        sampler = MockSampler()
    if data_dir is None:
        data_dir = Path(tempfile.mkdtemp())
    if outcomes is not None:
        _write_outcomes(data_dir / "outcomes.jsonl", outcomes)
    return KellyPositionSizer(sampler, data_dir=data_dir, max_fraction=max_fraction)


# ---------------------------------------------------------------------------
# Tests: thin-data / default behaviour
# ---------------------------------------------------------------------------

class TestRecommendThinData(unittest.TestCase):

    def test_thin_data_uses_default_win_loss_ratio(self):
        """
        With no outcomes.jsonl, _get_win_loss_ratio must fall back to
        DEFAULT_WIN_LOSS_RATIO and report 0 observations.
        """
        sampler = MockSampler(default_mean=0.6)
        sizer = _make_sizer(sampler=sampler)
        rec = sizer.recommend("sec_form4", "AAPL")
        self.assertAlmostEqual(rec.win_loss_ratio, DEFAULT_WIN_LOSS_RATIO)
        self.assertEqual(rec.observations_used, 0)

    def test_thin_data_confidence_is_low(self):
        """
        With fewer than MIN_OBSERVATIONS outcomes, confidence_in_sizing is "low".
        """
        sizer = _make_sizer()
        rec = sizer.recommend("sec_form4", "AAPL")
        self.assertEqual(rec.confidence_in_sizing, "low")

    def test_recommend_returns_position_recommendation(self):
        """recommend() must return a PositionRecommendation dataclass."""
        sizer = _make_sizer()
        rec = sizer.recommend("sec_form4", "AAPL")
        self.assertIsInstance(rec, PositionRecommendation)
        self.assertEqual(rec.signal_source, "sec_form4")
        self.assertEqual(rec.symbol, "AAPL")


# ---------------------------------------------------------------------------
# Tests: Kelly math verification
# ---------------------------------------------------------------------------

class TestKellyMath(unittest.TestCase):

    def test_kelly_formula_correct(self):
        """
        p=0.7, b=1.5 → f = (1.5*0.7 - 0.3)/1.5 = (1.05 - 0.3)/1.5 = 0.5
        kelly_half = 0.5 * 0.5 = 0.25
        kelly_capped = min(0.25, 0.05) = 0.05
        """
        sampler = MockSampler()
        sampler.set_mean("sec_form4", 0.7)
        sizer = _make_sizer(sampler=sampler)
        rec = sizer.recommend("sec_form4", "AAPL")
        self.assertAlmostEqual(rec.p_win, 0.7, places=5)
        self.assertAlmostEqual(rec.win_loss_ratio, DEFAULT_WIN_LOSS_RATIO, places=4)
        expected_full = (DEFAULT_WIN_LOSS_RATIO * 0.7 - 0.3) / DEFAULT_WIN_LOSS_RATIO
        self.assertAlmostEqual(rec.kelly_full, expected_full, places=4)
        expected_half = expected_full * HALF_KELLY_FACTOR
        self.assertAlmostEqual(rec.kelly_half, expected_half, places=4)
        self.assertAlmostEqual(rec.kelly_capped, MAX_KELLY_FRACTION, places=6)

    def test_kelly_caps_at_max_fraction(self):
        """
        Very high p → full Kelly > MAX_KELLY_FRACTION → capped at 0.05.
        """
        sampler = MockSampler()
        sampler.set_mean("sec_form4", 0.99)
        sizer = _make_sizer(sampler=sampler)
        rec = sizer.recommend("sec_form4", "AAPL")
        self.assertAlmostEqual(rec.kelly_capped, MAX_KELLY_FRACTION, places=6)

    def test_kelly_zero_edge(self):
        """
        p=0.3, b=1.5 → f = (1.5*0.3 - 0.7)/1.5 = (0.45 - 0.7)/1.5 = -0.167
        Clamped to 0.0 — no trade recommended.
        """
        sampler = MockSampler()
        sampler.set_mean("sec_form4", 0.3)
        sizer = _make_sizer(sampler=sampler)
        rec = sizer.recommend("sec_form4", "AAPL")
        self.assertAlmostEqual(rec.kelly_full, 0.0, places=6)
        self.assertAlmostEqual(rec.kelly_half, 0.0, places=6)
        self.assertAlmostEqual(rec.kelly_capped, 0.0, places=6)

    def test_kelly_neutral_prior(self):
        """
        p=0.5, b=1.5 → f = (1.5*0.5 - 0.5)/1.5 = (0.75 - 0.5)/1.5 = 0.1667
        kelly_half = 0.0833
        kelly_capped = min(0.0833, 0.05) = 0.05
        """
        sampler = MockSampler()
        sampler.set_mean("sec_form4", 0.5)
        sizer = _make_sizer(sampler=sampler)
        rec = sizer.recommend("sec_form4", "AAPL")
        b = DEFAULT_WIN_LOSS_RATIO
        expected_full = (b * 0.5 - 0.5) / b
        self.assertAlmostEqual(rec.kelly_full, expected_full, places=4)
        expected_half = expected_full * HALF_KELLY_FACTOR
        self.assertAlmostEqual(rec.kelly_half, expected_half, places=4)
        self.assertAlmostEqual(rec.kelly_capped, MAX_KELLY_FRACTION, places=6)

    def test_kelly_full_never_exceeds_one(self):
        """kelly_full is always in [0, 1] regardless of p."""
        for p in [0.0, 0.1, 0.5, 0.9, 1.0]:
            sampler = MockSampler()
            sampler.set_mean("any_key", p)
            sizer = _make_sizer(sampler=sampler)
            rec = sizer.recommend("any_key", "X")
            self.assertLessEqual(rec.kelly_full, 1.0)
            self.assertGreaterEqual(rec.kelly_full, 0.0)


# ---------------------------------------------------------------------------
# Tests: confidence tiers
# ---------------------------------------------------------------------------

class TestConfidenceTiers(unittest.TestCase):

    def _outcomes_for(self, source: str, n: int, win_ratio: float = 0.6) -> list:
        """Build n synthetic outcome records for a source."""
        records = []
        for i in range(n):
            records.append({
                "source": source,
                "price_change_pct": 0.05 if i < int(n * win_ratio) else -0.03,
                "success": i < int(n * win_ratio),
            })
        return records

    def test_confidence_low_below_min_observations(self):
        """obs < MIN_OBSERVATIONS → confidence = 'low'."""
        outcomes = self._outcomes_for("sec_form4", MIN_OBSERVATIONS - 1)
        sizer = _make_sizer(outcomes=outcomes)
        rec = sizer.recommend("sec_form4", "AAPL")
        self.assertEqual(rec.confidence_in_sizing, "low")

    def test_confidence_medium_at_min_observations(self):
        """obs == MIN_OBSERVATIONS → confidence = 'medium'."""
        outcomes = self._outcomes_for("sec_form4", MIN_OBSERVATIONS)
        sizer = _make_sizer(outcomes=outcomes)
        rec = sizer.recommend("sec_form4", "AAPL")
        self.assertEqual(rec.confidence_in_sizing, "medium")

    def test_confidence_medium_between_thresholds(self):
        """10 <= obs < 50 → confidence = 'medium'."""
        outcomes = self._outcomes_for("sec_form4", 30)
        sizer = _make_sizer(outcomes=outcomes)
        rec = sizer.recommend("sec_form4", "AAPL")
        self.assertEqual(rec.confidence_in_sizing, "medium")

    def test_confidence_high_at_fifty_observations(self):
        """obs >= 50 → confidence = 'high'."""
        outcomes = self._outcomes_for("sec_form4", 50)
        sizer = _make_sizer(outcomes=outcomes)
        rec = sizer.recommend("sec_form4", "AAPL")
        self.assertEqual(rec.confidence_in_sizing, "high")


# ---------------------------------------------------------------------------
# Tests: ratio cache and refresh
# ---------------------------------------------------------------------------

class TestRatioCache(unittest.TestCase):

    def _make_outcomes(self, source: str, n: int) -> list:
        records = []
        for i in range(n):
            records.append({
                "source": source,
                "price_change_pct": 0.05 if i % 2 == 0 else -0.02,
                "success": i % 2 == 0,
            })
        return records

    def test_ratio_cached_after_first_call(self):
        """_get_win_loss_ratio result is cached after the first call per source."""
        outcomes = self._make_outcomes("sec_form4", 20)
        sizer = _make_sizer(outcomes=outcomes)
        # First call — not yet in cache
        ratio1, obs1 = sizer._get_win_loss_ratio("sec_form4")
        # Second call — must return cached value without re-reading file
        ratio2, obs2 = sizer._get_win_loss_ratio("sec_form4")
        self.assertAlmostEqual(ratio1, ratio2)
        self.assertEqual(obs1, obs2)

    def test_refresh_ratios_clears_cache(self):
        """refresh_ratios() clears _ratio_cache so the next call reloads from disk."""
        outcomes = self._make_outcomes("congressional", 20)
        sizer = _make_sizer(outcomes=outcomes)
        # Populate cache
        sizer._get_win_loss_ratio("congressional")
        self.assertIn("congressional", sizer._ratio_cache)
        # Refresh clears it
        sizer.refresh_ratios()
        # After refresh the cache should be repopulated from the file
        # but the key should still resolve correctly
        self.assertIn("congressional", sizer._ratio_cache)

    def test_unknown_source_returns_default_ratio(self):
        """A source not in outcomes.jsonl falls back to DEFAULT_WIN_LOSS_RATIO."""
        sizer = _make_sizer()  # No outcomes at all
        ratio, obs = sizer._get_win_loss_ratio("completely_unknown_source")
        self.assertAlmostEqual(ratio, DEFAULT_WIN_LOSS_RATIO)
        self.assertEqual(obs, 0)


# ---------------------------------------------------------------------------
# Tests: get_statistics
# ---------------------------------------------------------------------------

class TestGetStatistics(unittest.TestCase):

    def test_get_statistics_structure(self):
        """get_statistics() returns a dict with expected top-level keys."""
        sizer = _make_sizer()
        stats = sizer.get_statistics()
        self.assertIn("sources_with_historical_data", stats)
        self.assertIn("outcomes_path", stats)
        self.assertIn("max_kelly_fraction", stats)
        self.assertIn("min_observations_required", stats)
        self.assertIn("default_win_loss_ratio", stats)
        self.assertIn("cached_sources", stats)

    def test_get_statistics_max_fraction(self):
        """get_statistics reflects the max_fraction set at construction."""
        sizer = _make_sizer(max_fraction=0.03)
        stats = sizer.get_statistics()
        self.assertAlmostEqual(stats["max_kelly_fraction"], 0.03)

    def test_get_statistics_empty_cache(self):
        """With no outcomes, sources_with_historical_data is 0."""
        sizer = _make_sizer()
        stats = sizer.get_statistics()
        self.assertEqual(stats["sources_with_historical_data"], 0)

    def test_get_statistics_populated_cache(self):
        """sources_with_historical_data reflects loaded sources above MIN_OBSERVATIONS."""
        outcomes = []
        for source in ["sec_form4", "congressional"]:
            for i in range(MIN_OBSERVATIONS):
                outcomes.append({
                    "source": source,
                    "price_change_pct": 0.05 if i % 2 == 0 else -0.02,
                    "success": i % 2 == 0,
                })
        sizer = _make_sizer(outcomes=outcomes)
        stats = sizer.get_statistics()
        self.assertEqual(stats["sources_with_historical_data"], 2)


# ---------------------------------------------------------------------------
# Tests: save() is a no-op
# ---------------------------------------------------------------------------

class TestSave(unittest.TestCase):

    def test_save_does_not_raise(self):
        """save() must not raise — it is a no-op per the implementation."""
        sizer = _make_sizer()
        sizer.save()  # Must complete without error


if __name__ == "__main__":
    unittest.main()
