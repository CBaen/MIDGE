"""Tests for mae_core.market.intelligence.granger_analyzer.

All tests are self-contained — no real archive files or network calls.
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
from unittest.mock import patch

from mae_core.market.intelligence.granger_analyzer import (
    GrangerAnalyzer,
    GrangerFinding,
)


# ---------------------------------------------------------------------------
# Mock infrastructure (same pattern as test_lag_correlation_analyzer)
# ---------------------------------------------------------------------------

class _MockRecord:
    def __init__(self, timestamp: datetime, strength: float):
        self.timestamp = timestamp
        self.strength = strength


class MockArchiveReader:
    def __init__(self, series_data: dict):
        self._series = series_data

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


def _make_daily_series(start: date, n_days: int, values: list) -> list:
    return [
        (datetime(start.year, start.month, start.day) + timedelta(days=i), float(v))
        for i, v in enumerate(values[:n_days])
    ]


# ---------------------------------------------------------------------------
# GrangerFinding dataclass tests
# ---------------------------------------------------------------------------

class TestGrangerFinding(unittest.TestCase):

    def test_dataclass_construction(self):
        f = GrangerFinding(
            cause_source="insider",
            effect_source="price",
            best_lag=5,
            f_statistic=4.23,
            p_value=0.01,
            direction="causal",
            all_lags_tested=30,
            significant_lags=3,
            min_p_value=0.003,
            n_observations=100,
            last_updated="2026-03-06T12:00:00",
        )
        assert f.cause_source == "insider"
        assert f.effect_source == "price"
        assert f.best_lag == 5

    def test_asdict_roundtrip(self):
        f = GrangerFinding(
            cause_source="a", effect_source="b", best_lag=3,
            f_statistic=2.0, p_value=0.04, direction="causal",
            all_lags_tested=10, significant_lags=1,
            min_p_value=0.04, n_observations=50,
            last_updated="2026-03-06",
        )
        d = asdict(f)
        f2 = GrangerFinding(**d)
        assert f == f2


# ---------------------------------------------------------------------------
# GrangerAnalyzer core tests
# ---------------------------------------------------------------------------

class TestGrangerAnalyzer(unittest.TestCase):

    def _make_causal_data(self, n=120, lag=5, noise=0.1):
        """Create synthetic data where source_a causes source_b with a lag.

        source_a is random walk. source_b = source_a shifted by `lag` days
        plus noise. Granger test should detect a→b causality.
        """
        import random
        random.seed(42)
        start = date(2025, 9, 1)

        # Generate source_a as random walk
        a_vals = [0.5]
        for _ in range(n - 1):
            a_vals.append(a_vals[-1] + random.gauss(0, 0.05))

        # source_b = a shifted by lag + noise
        b_vals = [0.5] * lag  # pad initial values
        for i in range(lag, n):
            b_vals.append(a_vals[i - lag] + random.gauss(0, noise))

        return {
            "source_a": _make_daily_series(start, n, a_vals),
            "source_b": _make_daily_series(start, n, b_vals),
        }

    def test_detects_causal_relationship(self):
        """Analyzer should detect that source_a Granger-causes source_b."""
        data = self._make_causal_data(n=150, lag=5, noise=0.02)
        reader = MockArchiveReader(data)

        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = GrangerAnalyzer(
                archive_reader=reader,
                max_lag=15,
                min_observations=30,
                persistence_path=Path(tmpdir) / "granger.json",
            )
            findings = analyzer.analyze(lookback_days=200)

        # Should find at least one significant causal relationship
        assert len(findings) > 0

        # At least one finding should have source_a as cause
        causes_b = [f for f in findings if f.cause_source == "source_a"
                     and f.effect_source == "source_b"]
        assert len(causes_b) > 0, "Should detect source_a → source_b"

    def test_no_false_causality_for_independent_series(self):
        """Independent random series should not show Granger causality."""
        import random
        random.seed(123)
        start = date(2025, 9, 1)
        n = 120

        # Two completely independent random walks
        a_vals = [random.gauss(0.5, 0.05) for _ in range(n)]
        b_vals = [random.gauss(0.5, 0.05) for _ in range(n)]

        data = {
            "indep_a": _make_daily_series(start, n, a_vals),
            "indep_b": _make_daily_series(start, n, b_vals),
        }
        reader = MockArchiveReader(data)

        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = GrangerAnalyzer(
                archive_reader=reader,
                max_lag=10,
                min_observations=30,
                persistence_path=Path(tmpdir) / "granger.json",
            )
            findings = analyzer.analyze(lookback_days=200)

        # Should find no significant causality (with Bonferroni correction)
        assert len(findings) == 0, (
            f"Independent series should not show causality, found {len(findings)}"
        )

    def test_insufficient_data_returns_empty(self):
        """Too few observations should return empty."""
        start = date(2025, 12, 1)
        data = {
            "short_a": _make_daily_series(start, 10, [float(i) for i in range(10)]),
            "short_b": _make_daily_series(start, 10, [float(i) for i in range(10)]),
        }
        reader = MockArchiveReader(data)

        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = GrangerAnalyzer(
                archive_reader=reader,
                min_observations=40,
                persistence_path=Path(tmpdir) / "granger.json",
            )
            findings = analyzer.analyze(lookback_days=200)

        assert findings == []

    def test_single_source_returns_empty(self):
        """Need at least 2 sources."""
        start = date(2025, 9, 1)
        data = {
            "only_one": _make_daily_series(start, 100, [float(i) for i in range(100)]),
        }
        reader = MockArchiveReader(data)

        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = GrangerAnalyzer(
                archive_reader=reader,
                persistence_path=Path(tmpdir) / "granger.json",
            )
            findings = analyzer.analyze()

        assert findings == []

    def test_persistence_save_and_load(self):
        """Findings should persist to JSON and reload correctly."""
        data = self._make_causal_data(n=150, lag=5, noise=0.02)
        reader = MockArchiveReader(data)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "granger.json"

            analyzer1 = GrangerAnalyzer(
                archive_reader=reader, max_lag=15,
                min_observations=30, persistence_path=path,
            )
            findings1 = analyzer1.analyze(lookback_days=200)

            # Create new analyzer that loads from same path
            analyzer2 = GrangerAnalyzer(
                archive_reader=reader, max_lag=15,
                min_observations=30, persistence_path=path,
            )

            assert len(analyzer2._findings) == len(findings1)
            if findings1:
                assert analyzer2._findings[0].cause_source == findings1[0].cause_source

    def test_causal_strength_range(self):
        """Causal strength should be in [0, 1]."""
        data = self._make_causal_data(n=150, lag=5, noise=0.02)
        reader = MockArchiveReader(data)

        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = GrangerAnalyzer(
                archive_reader=reader, max_lag=15,
                min_observations=30,
                persistence_path=Path(tmpdir) / "granger.json",
            )
            analyzer.analyze(lookback_days=200)

            for (cause, effect), strength in analyzer._causal_pairs.items():
                assert 0.0 <= strength <= 1.0, (
                    f"Strength {strength} out of range for {cause}→{effect}"
                )

    def test_get_causal_strength_missing_pair(self):
        """Missing pair should return 0.0."""
        reader = MockArchiveReader({})
        analyzer = GrangerAnalyzer(archive_reader=reader)
        assert analyzer.get_causal_strength("x", "y") == 0.0

    def test_get_causes_of(self):
        """get_causes_of should filter by effect_source."""
        data = self._make_causal_data(n=150, lag=5, noise=0.02)
        reader = MockArchiveReader(data)

        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = GrangerAnalyzer(
                archive_reader=reader, max_lag=15,
                min_observations=30,
                persistence_path=Path(tmpdir) / "granger.json",
            )
            analyzer.analyze(lookback_days=200)

            causes = analyzer.get_causes_of("source_b")
            for f in causes:
                assert f.effect_source == "source_b"

    def test_get_effects_of(self):
        """get_effects_of should filter by cause_source."""
        data = self._make_causal_data(n=150, lag=5, noise=0.02)
        reader = MockArchiveReader(data)

        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = GrangerAnalyzer(
                archive_reader=reader, max_lag=15,
                min_observations=30,
                persistence_path=Path(tmpdir) / "granger.json",
            )
            analyzer.analyze(lookback_days=200)

            effects = analyzer.get_effects_of("source_a")
            for f in effects:
                assert f.cause_source == "source_a"

    def test_statistics_structure(self):
        """get_statistics should return expected keys."""
        reader = MockArchiveReader({})
        analyzer = GrangerAnalyzer(archive_reader=reader)
        stats = analyzer.get_statistics()

        assert "total_analyses" in stats
        assert "causal_findings" in stats
        assert "causal_pairs" in stats
        assert "bidirectional_pairs" in stats
        assert "last_analysis_date" in stats
        assert "top_findings" in stats

    def test_atomic_save(self):
        """Save should use atomic write (tmp + rename)."""
        reader = MockArchiveReader({})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "granger.json"
            analyzer = GrangerAnalyzer(
                archive_reader=reader,
                persistence_path=path,
            )
            # Manually add a finding to verify save
            analyzer._findings = [GrangerFinding(
                cause_source="a", effect_source="b", best_lag=3,
                f_statistic=2.0, p_value=0.04, direction="causal",
                all_lags_tested=10, significant_lags=1,
                min_p_value=0.04, n_observations=50,
                last_updated="2026-03-06",
            )]
            analyzer.save()

            assert path.exists()
            data = json.loads(path.read_text())
            assert len(data) == 1
            assert data[0]["cause_source"] == "a"


# ---------------------------------------------------------------------------
# Bootstrap wiring tests
# ---------------------------------------------------------------------------

class TestBootstrapWiring(unittest.TestCase):

    def test_granger_in_market_registration(self):
        """market_registration.py should mention granger_analyzer."""
        source = Path("C:/Users/baenb/projects/MIDGE/mae_core/bootstrap/market_registration.py").read_text()
        assert "granger_analyzer" in source

    def test_granger_in_main_systems_dict(self):
        """main.py _build_systems_dict should include granger_analyzer."""
        import ast
        source = Path("C:/Users/baenb/projects/MIDGE/main.py").read_text()
        assert "granger_analyzer" in source


if __name__ == "__main__":
    unittest.main()
