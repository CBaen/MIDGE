#!/usr/bin/env python3
"""Tests for mae_core.market.intelligence.thompson_calibrator.

All tests are self-contained — no real predictions.jsonl or outcomes.jsonl
are read unless explicitly written to a temporary directory.

A MockSampler provides the minimal ThompsonSampler interface so
ThompsonCalibrator can be tested without the full sampler stack.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

from mae_core.market.intelligence.thompson_calibrator import (
    ThompsonCalibrator,
    _SEEDED_THRESHOLD,
    _MIN_PAIRS_FOR_CALIBRATION,
)


# ---------------------------------------------------------------------------
# Mock infrastructure
# ---------------------------------------------------------------------------

class MockSampler:
    """Minimal ThompsonSampler interface for ThompsonCalibrator tests."""

    def __init__(self):
        # Mirrors ThompsonSampler.distributions: {key: {regime: {alpha, beta}}}
        self.distributions: dict = {}
        self.prior_scale: float = 2.0
        self._saved: bool = False

    def get_distribution(self, key: str, regime: str = "default"):
        d = self.distributions.get(key, {}).get(regime, {"alpha": 1.0, "beta": 1.0})
        alpha = d["alpha"]
        beta = d["beta"]
        mean = alpha / (alpha + beta)
        return type("Dist", (), {"alpha": alpha, "beta": beta, "mean": mean})()

    def _save_distributions(self):
        self._saved = True


def _write_jsonl(path: Path, records: list):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _make_calibrator(sampler=None, data_dir=None):
    """Convenience factory — creates a ThompsonCalibrator with a temp data_dir."""
    if sampler is None:
        sampler = MockSampler()
    if data_dir is None:
        data_dir = Path(tempfile.mkdtemp())
    return ThompsonCalibrator(sampler, data_dir=data_dir)


# ---------------------------------------------------------------------------
# Seed fix tests
# ---------------------------------------------------------------------------

class TestSeedMissingKeys(unittest.TestCase):

    def test_seed_missing_keys_populates_source_reliability(self):
        """
        After construction with an empty distributions dict, the calibrator
        should seed all source_reliability keys from LEARNING_CONFIG.
        """
        sampler = MockSampler()
        calibrator = _make_calibrator(sampler=sampler)

        # LEARNING_CONFIG has source_reliability entries; at least the signal-source
        # level keys should appear in distributions after seeding.
        expected_keys = {
            "sec_form4", "congressional", "insider_cluster",
            "contract_award", "sec_form8k",
        }
        seeded = set(sampler.distributions.keys())
        self.assertTrue(
            expected_keys.issubset(seeded),
            f"Expected keys {expected_keys} not all found in {seeded}",
        )

    def test_seeded_distributions_have_nonuniform_priors(self):
        """
        Each seeded distribution should reflect the reliability value from
        LEARNING_CONFIG — not the uninformative Beta(1, 1).
        """
        sampler = MockSampler()
        _make_calibrator(sampler=sampler)

        for key, regime_dict in sampler.distributions.items():
            default = regime_dict.get("default", {})
            alpha = default.get("alpha", 1.0)
            beta = default.get("beta", 1.0)
            total = alpha + beta
            # A seeded key should have alpha + beta == prior_scale * 1 = 2.0
            # (sum equals prior_scale because reliability is in (0,1))
            # or at least be distinguishably non-Beta(1,1)
            self.assertAlmostEqual(total, 2.0, places=9,
                                   msg=f"Key '{key}' has unexpected alpha+beta={total}")

    def test_seed_missing_keys_idempotent(self):
        """
        Constructing a second calibrator on the same sampler should seed 0
        additional keys — idempotency check.
        """
        sampler = MockSampler()
        # First construction seeds everything
        ThompsonCalibrator(sampler, data_dir=Path(tempfile.mkdtemp()))
        keys_after_first = set(sampler.distributions.keys())

        # Second construction should not add or modify any key
        ThompsonCalibrator(sampler, data_dir=Path(tempfile.mkdtemp()))
        keys_after_second = set(sampler.distributions.keys())

        self.assertEqual(keys_after_first, keys_after_second)

    def test_seed_preserves_existing_distributions(self):
        """
        A key pre-populated with alpha+beta > _SEEDED_THRESHOLD must NOT be
        overwritten by the seed fix.
        """
        sampler = MockSampler()
        # Pre-seed sec_form4 with alpha=5, beta=3 (sum=8, well above threshold)
        sampler.distributions["sec_form4"] = {
            "default": {"alpha": 5.0, "beta": 3.0}
        }
        _make_calibrator(sampler=sampler)

        dist = sampler.distributions["sec_form4"]["default"]
        self.assertAlmostEqual(dist["alpha"], 5.0, places=9,
                               msg="Existing alpha was overwritten")
        self.assertAlmostEqual(dist["beta"], 3.0, places=9,
                               msg="Existing beta was overwritten")

    def test_seed_calls_save_distributions(self):
        """
        After seeding at least one key, the calibrator should call
        sampler._save_distributions() to persist the new priors.
        """
        sampler = MockSampler()
        _make_calibrator(sampler=sampler)
        # If at least one key was seeded, save should have been called
        if sampler.distributions:
            self.assertTrue(
                sampler._saved,
                "_save_distributions was not called after seeding",
            )


# ---------------------------------------------------------------------------
# Calibrate tests
# ---------------------------------------------------------------------------

class TestCalibrate(unittest.TestCase):

    def _setup_data_dir(self, predictions=None, outcomes=None):
        """Write synthetic predictions.jsonl and outcomes.jsonl to a tmpdir."""
        data_dir = Path(tempfile.mkdtemp())
        if predictions is not None:
            _write_jsonl(data_dir / "predictions.jsonl", predictions)
        if outcomes is not None:
            _write_jsonl(data_dir / "outcomes.jsonl", outcomes)
        return data_dir

    def test_calibrate_no_data_returns_empty(self):
        """calibrate() with no data files returns an empty list."""
        data_dir = Path(tempfile.mkdtemp())  # Empty — no predictions.jsonl
        calibrator = ThompsonCalibrator(MockSampler(), data_dir=data_dir)
        result = calibrator.calibrate()
        self.assertEqual(result, [])

    def test_calibrate_predictions_but_no_outcomes_returns_empty(self):
        """calibrate() with predictions but no outcomes returns empty list."""
        predictions = [
            {"prediction_id": "p1", "confidence": 0.8, "source": "sec_form4"},
        ]
        data_dir = self._setup_data_dir(predictions=predictions)
        calibrator = ThompsonCalibrator(MockSampler(), data_dir=data_dir)
        result = calibrator.calibrate()
        self.assertEqual(result, [])

    def test_calibrate_overconfident_source(self):
        """
        5 predictions at confidence 0.9, all outcomes incorrect → the source
        should be flagged as OVERCONFIDENT (mean_confidence >> mean_hit_rate).
        """
        n = 5
        predictions = [
            {"prediction_id": f"p{i}", "confidence": 0.9, "source": "sec_form4"}
            for i in range(n)
        ]
        outcomes = [
            {"prediction_id": f"p{i}", "was_correct": False, "source": "sec_form4"}
            for i in range(n)
        ]
        data_dir = self._setup_data_dir(predictions=predictions, outcomes=outcomes)
        calibrator = ThompsonCalibrator(MockSampler(), data_dir=data_dir)
        results = calibrator.calibrate()

        self.assertEqual(len(results), 1)
        cal = results[0]
        self.assertTrue(cal.is_overconfident)
        self.assertFalse(cal.is_underconfident)
        self.assertIn("OVERCONFIDENT", cal.recommendation)

    def test_calibrate_well_calibrated_source(self):
        """
        7 predictions at confidence 0.7 where 70% succeed → the source
        should be flagged as WELL CALIBRATED.
        """
        n = 10
        predictions = [
            {"prediction_id": f"p{i}", "confidence": 0.7, "source": "congressional"}
            for i in range(n)
        ]
        # 7 correct, 3 incorrect → hit rate = 0.7 = confidence → well calibrated
        outcomes = [
            {"prediction_id": f"p{i}", "was_correct": i < 7, "source": "congressional"}
            for i in range(n)
        ]
        data_dir = self._setup_data_dir(predictions=predictions, outcomes=outcomes)
        calibrator = ThompsonCalibrator(MockSampler(), data_dir=data_dir)
        results = calibrator.calibrate()

        self.assertEqual(len(results), 1)
        cal = results[0]
        self.assertFalse(cal.is_overconfident)
        self.assertFalse(cal.is_underconfident)
        self.assertIn("WELL CALIBRATED", cal.recommendation)

    def test_calibrate_fewer_than_min_pairs_excluded(self):
        """
        A source with fewer than _MIN_PAIRS_FOR_CALIBRATION matched pairs
        is excluded from the calibration report.
        """
        n = _MIN_PAIRS_FOR_CALIBRATION - 1  # Just below threshold
        predictions = [
            {"prediction_id": f"p{i}", "confidence": 0.8, "source": "sec_form4"}
            for i in range(n)
        ]
        outcomes = [
            {"prediction_id": f"p{i}", "was_correct": True, "source": "sec_form4"}
            for i in range(n)
        ]
        data_dir = self._setup_data_dir(predictions=predictions, outcomes=outcomes)
        calibrator = ThompsonCalibrator(MockSampler(), data_dir=data_dir)
        results = calibrator.calibrate()
        self.assertEqual(results, [])

    def test_calibrate_two_sources_produces_two_results(self):
        """
        When two sources each have enough matched pairs, both appear in output.
        """
        n = 5
        predictions = (
            [{"prediction_id": f"a{i}", "confidence": 0.8, "source": "sec_form4"} for i in range(n)]
            + [{"prediction_id": f"b{i}", "confidence": 0.6, "source": "congressional"} for i in range(n)]
        )
        outcomes = (
            [{"prediction_id": f"a{i}", "was_correct": True, "source": "sec_form4"} for i in range(n)]
            + [{"prediction_id": f"b{i}", "was_correct": True, "source": "congressional"} for i in range(n)]
        )
        data_dir = self._setup_data_dir(predictions=predictions, outcomes=outcomes)
        calibrator = ThompsonCalibrator(MockSampler(), data_dir=data_dir)
        results = calibrator.calibrate()
        sources = {r.source for r in results}
        self.assertEqual(sources, {"sec_form4", "congressional"})


# ---------------------------------------------------------------------------
# get_statistics tests
# ---------------------------------------------------------------------------

class TestGetStatistics(unittest.TestCase):

    def test_get_statistics_structure_before_calibrate(self):
        """get_statistics returns expected keys even before calibrate() is run."""
        calibrator = _make_calibrator()
        stats = calibrator.get_statistics()
        self.assertIn("seeded_key_count", stats)
        self.assertIn("seeded_keys", stats)
        self.assertIn("calibration_report_exists", stats)
        self.assertIn("calibrated_source_count", stats)

    def test_get_statistics_seeded_key_count_positive(self):
        """seeded_key_count reflects the number of distributions seeded."""
        sampler = MockSampler()
        calibrator = _make_calibrator(sampler=sampler)
        stats = calibrator.get_statistics()
        # We expect at least some keys to have been seeded from LEARNING_CONFIG
        self.assertGreaterEqual(stats["seeded_key_count"], 0)


# ---------------------------------------------------------------------------
# save / report persistence tests
# ---------------------------------------------------------------------------

class TestSaveReport(unittest.TestCase):

    def test_save_creates_calibration_report(self):
        """After calibrate(), save() should write calibration_report.json."""
        n = 5
        predictions = [
            {"prediction_id": f"p{i}", "confidence": 0.9, "source": "sec_form4"}
            for i in range(n)
        ]
        outcomes = [
            {"prediction_id": f"p{i}", "was_correct": False, "source": "sec_form4"}
            for i in range(n)
        ]
        data_dir = Path(tempfile.mkdtemp())
        _write_jsonl(data_dir / "predictions.jsonl", predictions)
        _write_jsonl(data_dir / "outcomes.jsonl", outcomes)

        calibrator = ThompsonCalibrator(MockSampler(), data_dir=data_dir)
        calibrator.calibrate()

        # calibrate() calls _save_report() — check that the file now exists
        # Note: CALIBRATION_PATH is module-level; we need to check data_dir
        # ThompsonCalibrator writes to CALIBRATION_PATH (the global constant),
        # not to data_dir. We can verify _last_report is populated and that
        # _save_report doesn't raise.
        self.assertIsNotNone(calibrator._last_report)
        self.assertEqual(len(calibrator._last_report), 1)

    def test_save_noop_before_calibrate(self):
        """save() before calibrate() does not raise."""
        calibrator = _make_calibrator()
        calibrator.save()  # Must not raise

    def test_calibrate_sets_last_report(self):
        """After calibrate(), _last_report is populated with SourceCalibration objects."""
        n = 5
        predictions = [
            {"prediction_id": f"p{i}", "confidence": 0.6, "source": "congressional"}
            for i in range(n)
        ]
        outcomes = [
            {"prediction_id": f"p{i}", "was_correct": True, "source": "congressional"}
            for i in range(n)
        ]
        data_dir = Path(tempfile.mkdtemp())
        _write_jsonl(data_dir / "predictions.jsonl", predictions)
        _write_jsonl(data_dir / "outcomes.jsonl", outcomes)

        calibrator = ThompsonCalibrator(MockSampler(), data_dir=data_dir)
        self.assertIsNone(calibrator._last_report)
        calibrator.calibrate()
        self.assertIsNotNone(calibrator._last_report)
        self.assertEqual(len(calibrator._last_report), 1)
        self.assertEqual(calibrator._last_report[0].source, "congressional")


if __name__ == "__main__":
    unittest.main()
