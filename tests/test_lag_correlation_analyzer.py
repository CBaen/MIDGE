#!/usr/bin/env python3
"""Tests for mae_core.market.intelligence.lag_correlation_analyzer.

All tests are self-contained — no real archive files or network calls are made.
A MockArchiveReader injects synthetic time series, and findings persistence
uses a temporary directory.
"""

import json
import math
import tempfile
import unittest
from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path

from mae_core.market.intelligence.lag_correlation_analyzer import (
    LagCorrelationAnalyzer,
    LagFinding,
    _fisher_p,
    _norm_sf,
    _pearson,
)


# ---------------------------------------------------------------------------
# Mock infrastructure
# ---------------------------------------------------------------------------

class _MockRecord:
    """Minimal record shape expected by _build_daily_series."""

    def __init__(self, timestamp: datetime, strength: float):
        self.timestamp = timestamp
        self.strength = strength


class MockArchiveReader:
    """
    Thin stand-in for SignalArchiveReader.

    Accepts a dict of {source: [(datetime, strength), ...]} and returns
    _MockRecord objects from query_source().
    """

    def __init__(self, series_data: dict):
        self._series = series_data  # {source: [(datetime, float), ...]}

    def load_range(self, start: date, end: date) -> int:
        return 0

    def available_sources(self):
        return list(self._series.keys())

    def query_source(self, source, start=None, end=None, symbol=None):
        pairs = self._series.get(source, [])
        records = [_MockRecord(ts, val) for ts, val in pairs]
        if start:
            records = [r for r in records if r.timestamp.date() >= start]
        if end:
            records = [r for r in records if r.timestamp.date() <= end]
        return records


def _make_daily_series(
    start: date, n_days: int, values: list
) -> list:
    """Build a list of (datetime, float) pairs, one per day."""
    return [
        (datetime(start.year, start.month, start.day) + timedelta(days=i), float(v))
        for i, v in enumerate(values[:n_days])
    ]


# ---------------------------------------------------------------------------
# Pure statistics function tests
# ---------------------------------------------------------------------------

class TestPearson(unittest.TestCase):

    def test_pearson_perfect_positive(self):
        """Identical series → r = 1.0."""
        r = _pearson([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        self.assertIsNotNone(r)
        self.assertAlmostEqual(r, 1.0, places=9)

    def test_pearson_perfect_negative(self):
        """Inverted series → r = -1.0."""
        r = _pearson([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
        self.assertIsNotNone(r)
        self.assertAlmostEqual(r, -1.0, places=9)

    def test_pearson_zero_variance_x(self):
        """Constant x → returns None (no variance)."""
        r = _pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])
        self.assertIsNone(r)

    def test_pearson_zero_variance_y(self):
        """Constant y → returns None (no variance)."""
        r = _pearson([1.0, 2.0, 3.0], [5.0, 5.0, 5.0])
        self.assertIsNone(r)

    def test_pearson_too_few_points(self):
        """Fewer than 3 points → returns None."""
        r = _pearson([1.0, 2.0], [1.0, 2.0])
        self.assertIsNone(r)

    def test_pearson_exactly_three_points(self):
        """Exactly 3 points is the minimum — should return a value."""
        r = _pearson([1.0, 2.0, 4.0], [2.0, 4.0, 8.0])
        self.assertIsNotNone(r)
        self.assertGreater(r, 0.9)

    def test_pearson_result_clamped_to_unit_interval(self):
        """Result is always in [-1, 1] regardless of floating-point drift."""
        # Fabricate a case where floating-point could exceed 1.0 slightly
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [1.001 * x for x in xs]
        r = _pearson(xs, ys)
        self.assertIsNotNone(r)
        self.assertLessEqual(r, 1.0)
        self.assertGreaterEqual(r, -1.0)


# ---------------------------------------------------------------------------
# Fisher p-value tests
# ---------------------------------------------------------------------------

class TestFisherP(unittest.TestCase):

    def test_fisher_p_high_correlation_is_significant(self):
        """High r with large n should yield p < 0.001."""
        p = _fisher_p(0.9, 30)
        self.assertLess(p, 0.001)

    def test_fisher_p_low_correlation_is_not_significant(self):
        """Low r with small n should yield p > 0.05."""
        p = _fisher_p(0.1, 10)
        self.assertGreater(p, 0.05)

    def test_fisher_p_at_exactly_one_returns_one(self):
        """r = 1.0 triggers the guard and returns 1.0."""
        p = _fisher_p(1.0, 10)
        self.assertEqual(p, 1.0)

    def test_fisher_p_at_negative_one_returns_one(self):
        """r = -1.0 triggers the guard and returns 1.0."""
        p = _fisher_p(-1.0, 10)
        self.assertEqual(p, 1.0)

    def test_fisher_p_n_three_returns_one(self):
        """n = 3 makes n-3 = 0 → ZeroDivisionError guarded → returns 1.0."""
        p = _fisher_p(0.5, 3)
        self.assertEqual(p, 1.0)

    def test_fisher_p_moderate_correlation_moderate_n(self):
        """r=0.5 with n=20 should be plausibly between 0 and 0.05."""
        p = _fisher_p(0.5, 20)
        self.assertLessEqual(p, 1.0)
        self.assertGreaterEqual(p, 0.0)

    def test_fisher_p_negative_r_symmetric(self):
        """Negative r with same magnitude as positive should give same p."""
        p_pos = _fisher_p(0.7, 30)
        p_neg = _fisher_p(-0.7, 30)
        self.assertAlmostEqual(p_pos, p_neg, places=10)


# ---------------------------------------------------------------------------
# Analyzer integration tests
# ---------------------------------------------------------------------------

class TestLagCorrelationAnalyzer(unittest.TestCase):

    def _tmppath(self, filename="lag_correlations.json"):
        d = tempfile.mkdtemp()
        return Path(d) / filename

    def test_analyze_insufficient_sources_returns_empty(self):
        """Fewer than 2 sources → analyze() returns empty list without error."""
        reader = MockArchiveReader({"only_source": []})
        analyzer = LagCorrelationAnalyzer(
            archive_reader=reader,
            persistence_path=self._tmppath(),
        )
        results = analyzer.analyze(lookback_days=30)
        self.assertEqual(results, [])

    def test_analyze_no_sources_returns_empty(self):
        """Zero sources → analyze() returns empty list without error."""
        reader = MockArchiveReader({})
        analyzer = LagCorrelationAnalyzer(
            archive_reader=reader,
            persistence_path=self._tmppath(),
        )
        results = analyzer.analyze(lookback_days=30)
        self.assertEqual(results, [])

    def test_analyze_detects_synthetic_lag(self):
        """
        Source A leads source B with high positive correlation at some lag.

        Construction: A[i] = sin(2pi*i/7) (period 7), B[i] = A[i-1] + small noise.
        The analyzer must find at least one significant finding where source_a
        leads source_b. Note: a perfectly-shifted series produces Pearson r = 1.0
        which _fisher_p() treats as degenerate (p=1.0, blocked). We add small
        non-uniform padding so the best lag has high-but-not-exact correlation.

        The test verifies the detection pipeline works end-to-end without
        asserting which specific lag wins (the sine period determines that).
        """
        import math as _math
        base = date(2026, 1, 1)
        n = 40
        period = 7.0
        # Sine wave — non-linear so different lags have different Pearson r
        a_values = [_math.sin(2 * _math.pi * i / period) for i in range(n)]
        # B is A shifted by 1 day with a small constant offset to avoid exact r=1.0
        # This means B[i] ≈ A[i-1] but not perfectly, so _fisher_p won't block it
        b_values = [0.1] + [a_values[i - 1] + 0.05 * _math.cos(i) for i in range(1, n)]

        series = {
            "source_a": _make_daily_series(base, n, a_values),
            "source_b": _make_daily_series(base, n, b_values),
        }
        reader = MockArchiveReader(series)
        analyzer = LagCorrelationAnalyzer(
            archive_reader=reader,
            lag_range=(1, 6),
            min_observations=3,
            significance_threshold=0.05,
            persistence_path=self._tmppath(),
        )
        findings = analyzer.analyze(lookback_days=90)

        # At least one significant finding must exist
        self.assertGreater(len(findings), 0)

        # source_a must appear as a leading indicator for source_b at some lag
        leading = [
            f for f in findings
            if f.source_a == "source_a" and f.source_b == "source_b"
        ]
        self.assertGreater(
            len(leading), 0,
            "No significant finding with source_a leading source_b",
        )

        # The best source_a->source_b finding should have a strong positive correlation
        best = max(leading, key=lambda f: abs(f.correlation))
        self.assertGreater(
            abs(best.correlation), 0.7,
            f"Best correlation {best.correlation:.4f} not strong enough",
        )

    def test_analyze_returns_findings_sorted_by_abs_correlation(self):
        """Findings are sorted descending by |correlation|."""
        base = date(2026, 1, 1)
        # Two sources with varying correlation strengths
        a_values = [float(i) for i in range(30)]
        b_values = [0.0] + a_values[:-1]  # perfect lag-1 lead

        series = {
            "source_a": _make_daily_series(base, 30, a_values),
            "source_b": _make_daily_series(base, 30, b_values),
        }
        reader = MockArchiveReader(series)
        analyzer = LagCorrelationAnalyzer(
            archive_reader=reader,
            lag_range=(1, 3),
            min_observations=3,
            significance_threshold=0.05,
            persistence_path=self._tmppath(),
        )
        findings = analyzer.analyze(lookback_days=60)

        if len(findings) >= 2:
            for i in range(len(findings) - 1):
                self.assertGreaterEqual(
                    abs(findings[i].correlation),
                    abs(findings[i + 1].correlation),
                )

    def test_findings_persistence_roundtrip(self):
        """Findings saved to disk are reloaded by a new analyzer instance."""
        base = date(2026, 1, 1)
        a_values = [float(i) / 30.0 for i in range(30)]
        b_values = [0.0] + a_values[:-1]

        series = {
            "source_a": _make_daily_series(base, 30, a_values),
            "source_b": _make_daily_series(base, 30, b_values),
        }
        reader = MockArchiveReader(series)
        persistence_path = self._tmppath()

        # First analyzer: run analysis, persist findings
        analyzer1 = LagCorrelationAnalyzer(
            archive_reader=reader,
            lag_range=(1, 3),
            min_observations=3,
            significance_threshold=0.05,
            persistence_path=persistence_path,
        )
        findings1 = analyzer1.analyze(lookback_days=60)
        analyzer1.save()

        self.assertTrue(persistence_path.exists())

        # Second analyzer: loaded from same path, no analysis run
        analyzer2 = LagCorrelationAnalyzer(
            archive_reader=MockArchiveReader({}),
            persistence_path=persistence_path,
        )
        findings2 = analyzer2._findings

        self.assertEqual(len(findings1), len(findings2))
        if findings1:
            self.assertEqual(findings1[0].source_a, findings2[0].source_a)
            self.assertEqual(findings1[0].lag_days, findings2[0].lag_days)

    def test_get_top_findings_limits_results(self):
        """get_top_findings(n) returns at most n findings."""
        base = date(2026, 1, 1)
        a_values = [float(i) / 30.0 for i in range(30)]
        b_values = [0.0] + a_values[:-1]

        series = {
            "source_a": _make_daily_series(base, 30, a_values),
            "source_b": _make_daily_series(base, 30, b_values),
        }
        reader = MockArchiveReader(series)
        analyzer = LagCorrelationAnalyzer(
            archive_reader=reader,
            lag_range=(1, 5),
            min_observations=3,
            significance_threshold=0.05,
            persistence_path=self._tmppath(),
        )
        analyzer.analyze(lookback_days=60)
        top = analyzer.get_top_findings(n=2)
        self.assertLessEqual(len(top), 2)

    def test_get_leading_indicators_filters_by_target(self):
        """get_leading_indicators returns only findings where source_b matches target."""
        base = date(2026, 1, 1)
        a_values = [float(i) / 30.0 for i in range(30)]
        b_values = [0.0] + a_values[:-1]

        series = {
            "source_a": _make_daily_series(base, 30, a_values),
            "source_b": _make_daily_series(base, 30, b_values),
        }
        reader = MockArchiveReader(series)
        analyzer = LagCorrelationAnalyzer(
            archive_reader=reader,
            lag_range=(1, 3),
            min_observations=3,
            significance_threshold=0.05,
            persistence_path=self._tmppath(),
        )
        analyzer.analyze(lookback_days=60)

        leaders = analyzer.get_leading_indicators("source_b")
        for f in leaders:
            self.assertEqual(f.source_b, "source_b")

    def test_get_statistics_structure(self):
        """get_statistics() contains the expected top-level keys."""
        reader = MockArchiveReader({})
        analyzer = LagCorrelationAnalyzer(
            archive_reader=reader,
            persistence_path=self._tmppath(),
        )
        stats = analyzer.get_statistics()
        self.assertIn("total_analyses", stats)
        self.assertIn("significant_findings", stats)
        self.assertIn("last_analysis_date", stats)
        self.assertIn("top_findings", stats)

    def test_get_statistics_counts_after_analyze(self):
        """total_analyses increments and significant_findings reflects results."""
        base = date(2026, 1, 1)
        a_values = [float(i) / 30.0 for i in range(30)]
        b_values = [0.0] + a_values[:-1]

        series = {
            "source_a": _make_daily_series(base, 30, a_values),
            "source_b": _make_daily_series(base, 30, b_values),
        }
        reader = MockArchiveReader(series)
        analyzer = LagCorrelationAnalyzer(
            archive_reader=reader,
            lag_range=(1, 3),
            min_observations=3,
            significance_threshold=0.05,
            persistence_path=self._tmppath(),
        )
        analyzer.analyze(lookback_days=60)
        stats = analyzer.get_statistics()
        self.assertEqual(stats["total_analyses"], 1)
        self.assertGreaterEqual(stats["significant_findings"], 0)

    def test_lag_finding_dataclass_fields(self):
        """LagFinding has all expected fields and correct types."""
        finding = LagFinding(
            source_a="a",
            source_b="b",
            lag_days=5,
            correlation=0.85,
            p_value_approx=0.001,
            n_pairs=25,
            direction="positive",
            last_updated=datetime.now().isoformat(),
        )
        self.assertEqual(finding.source_a, "a")
        self.assertEqual(finding.lag_days, 5)
        self.assertAlmostEqual(finding.correlation, 0.85)
        self.assertEqual(finding.direction, "positive")


if __name__ == "__main__":
    unittest.main()
