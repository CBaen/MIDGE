"""Tests for HypothesisValidator — DSR and promote/retire recommendations."""

import json
from pathlib import Path

import pytest

from mae_core.market.intelligence.hypothesis import (
    Hypothesis,
    HypothesisStats,
    TriggerPattern,
)
from mae_core.market.intelligence.hypothesis_validator import (
    HypothesisValidator,
    MIN_OBSERVATIONS,
    PROMOTE_WIN_RATE,
    PROMOTE_DSR,
    RETIRE_WIN_RATE,
)


@pytest.fixture
def tmp_data_dir(tmp_path):
    return tmp_path


@pytest.fixture
def signals_dir(tmp_path):
    d = tmp_path / "signals"
    d.mkdir()
    return d


def _make_hypothesis(source_a="sec_form4", source_b="finnhub_earnings", lag=45):
    return Hypothesis(
        name=f"{source_a}→{source_b}",
        trigger=TriggerPattern(
            source_a=source_a, source_b=source_b,
            lag_days=lag, direction="positive",
        ),
        causal_story="Test story.",
    )


def _write_signals(signals_dir, records):
    path = signals_dir / "2026-01-01.jsonl"
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _write_outcomes(data_dir, outcomes):
    path = data_dir / "outcomes.jsonl"
    with open(path, "w") as f:
        for o in outcomes:
            f.write(json.dumps(o) + "\n")


class TestDSRComputation:
    def test_dsr_penalizes_many_trials(self, tmp_data_dir, signals_dir):
        """More trials tested → lower DSR for the same Sharpe."""
        v1 = HypothesisValidator(
            signals_dir=signals_dir,
            data_dir=tmp_data_dir,
        )
        # Low trial count
        v1._dsr_trials_tracked = 2
        dsr_low = v1._compute_dsr(sharpe=1.0, n_obs=30)

        # High trial count
        v1._dsr_trials_tracked = 100
        dsr_high = v1._compute_dsr(sharpe=1.0, n_obs=30)

        assert dsr_low > dsr_high, "DSR should decrease with more trials"

    def test_dsr_zero_for_zero_sharpe(self, tmp_data_dir, signals_dir):
        v = HypothesisValidator(signals_dir=signals_dir, data_dir=tmp_data_dir)
        v._dsr_trials_tracked = 5
        dsr = v._compute_dsr(sharpe=0.0, n_obs=30)
        assert dsr < 0

    def test_sharpe_computation(self, tmp_data_dir, signals_dir):
        v = HypothesisValidator(signals_dir=signals_dir, data_dir=tmp_data_dir)
        returns = [5.0, 3.0, 4.0, 6.0, 2.0]
        sharpe = v._compute_sharpe(returns)
        assert sharpe > 0, "Positive returns should give positive Sharpe"


class TestValidation:
    def test_validate_empty_archive(self, tmp_data_dir, signals_dir):
        v = HypothesisValidator(signals_dir=signals_dir, data_dir=tmp_data_dir)
        hyp = _make_hypothesis()
        result = v.validate(hyp)
        assert result.total_observations == 0
        assert not result.recommend_promote
        assert not result.recommend_retire

    def test_validate_with_signals_and_outcomes(self, tmp_data_dir, signals_dir):
        # Write trigger signals (source_a = sec_form4)
        signals = [
            {"source": "sec_form4", "symbol": "AAPL",
             "timestamp": "2025-09-01T10:00:00", "strength": 0.8},
            {"source": "sec_form4", "symbol": "MSFT",
             "timestamp": "2025-09-15T10:00:00", "strength": 0.7},
        ]
        _write_signals(signals_dir, signals)

        # Write outcomes (source_b = finnhub_earnings, matching lag window)
        outcomes = [
            {"source": "finnhub_earnings", "symbol": "AAPL",
             "predicted_at": "2025-10-15T10:00:00",
             "price_change_pct": 8.5, "success": True},
            {"source": "finnhub_earnings", "symbol": "MSFT",
             "predicted_at": "2025-10-30T10:00:00",
             "price_change_pct": -3.0, "success": False},
        ]
        _write_outcomes(tmp_data_dir, outcomes)

        v = HypothesisValidator(signals_dir=signals_dir, data_dir=tmp_data_dir)
        hyp = _make_hypothesis()
        result = v.validate(hyp, lookback_days=365)
        assert result.total_observations >= 1  # At least one match

    def test_dsr_trials_persisted(self, tmp_data_dir, signals_dir):
        v1 = HypothesisValidator(signals_dir=signals_dir, data_dir=tmp_data_dir)
        hyp = _make_hypothesis()
        v1.validate(hyp)
        assert v1._dsr_trials_tracked >= 1

        # New validator from same dir should load the counter
        v2 = HypothesisValidator(signals_dir=signals_dir, data_dir=tmp_data_dir)
        assert v2._dsr_trials_tracked >= 1

    def test_statistics(self, tmp_data_dir, signals_dir):
        v = HypothesisValidator(signals_dir=signals_dir, data_dir=tmp_data_dir)
        v._dsr_trials_tracked = 42
        stats = v.get_statistics()
        assert stats["dsr_trials_tracked"] == 42
