"""Tests for HypothesisValidator — DSR and promote/retire recommendations."""

import json
from pathlib import Path

import pytest

from mae_core.market.intelligence.hypothesis import (
    Hypothesis,
    HypothesisStats,
    SourceType,
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


class TestBacktestDerivedValidation:
    """Tests for BACKTEST_DERIVED hypotheses — pre-populated stats path."""

    def _make_backtest_hypothesis(self, wins=15, losses=10, sharpe=1.5,
                                  causal_story="ICT session sweep pattern."):
        return Hypothesis(
            name="Sweep:CL=F:bearish (60% win, n=25)",
            trigger=TriggerPattern(
                source_a="session_sweep", source_b="price_outcome",
                lag_days=0, direction="bearish",
                domain_filter="CL=F:bearish",
            ),
            stats=HypothesisStats(
                wins=wins,
                losses=losses,
                total_observations=wins + losses,
                sharpe_ratio=sharpe,
            ),
            causal_story=causal_story,
            source_type=SourceType.BACKTEST_DERIVED,
        )

    def test_precomputed_path_used(self, tmp_data_dir, signals_dir):
        """BACKTEST_DERIVED skips archive scanning — uses pre-populated stats."""
        v = HypothesisValidator(signals_dir=signals_dir, data_dir=tmp_data_dir)
        hyp = self._make_backtest_hypothesis(wins=15, losses=10, sharpe=1.5)
        result = v.validate(hyp)
        assert result.wins == 15
        assert result.losses == 10
        assert result.total_observations == 25

    def test_dsr_computed_from_global_counter(self, tmp_data_dir, signals_dir):
        """DSR uses the global trial counter even for BACKTEST_DERIVED."""
        v = HypothesisValidator(signals_dir=signals_dir, data_dir=tmp_data_dir)
        v._dsr_trials_tracked = 5
        hyp = self._make_backtest_hypothesis(wins=15, losses=10, sharpe=1.5)
        result = v.validate(hyp)
        assert result.deflated_sharpe_ratio != 0.0
        # Counter should have incremented
        assert v._dsr_trials_tracked == 6

    def test_promotes_strong_backtest(self, tmp_data_dir, signals_dir):
        """High win rate + positive DSR + real story → promote."""
        v = HypothesisValidator(signals_dir=signals_dir, data_dir=tmp_data_dir)
        v._dsr_trials_tracked = 1  # Low penalty = easy promotion
        hyp = self._make_backtest_hypothesis(wins=18, losses=7, sharpe=3.0)
        result = v.validate(hyp)
        assert result.recommend_promote is True

    def test_retires_weak_backtest(self, tmp_data_dir, signals_dir):
        """Low win rate → retire."""
        v = HypothesisValidator(signals_dir=signals_dir, data_dir=tmp_data_dir)
        hyp = self._make_backtest_hypothesis(wins=5, losses=20, sharpe=0.1)
        result = v.validate(hyp)
        assert result.recommend_retire is True
        assert "win rate" in result.retire_reason.lower() or "Win rate" in result.retire_reason

    def test_no_archive_scan(self, tmp_data_dir, signals_dir):
        """Even with signals in the archive, BACKTEST_DERIVED doesn't scan them."""
        # Write some signals that would match if scanning happened
        sig_file = signals_dir / "2025-12-01.jsonl"
        sig_file.write_text(json.dumps({
            "source": "session_sweep", "symbol": "CL=F",
            "timestamp": "2025-12-01T10:00:00",
        }) + "\n")
        v = HypothesisValidator(signals_dir=signals_dir, data_dir=tmp_data_dir)
        hyp = self._make_backtest_hypothesis(wins=0, losses=0, sharpe=0.0)
        hyp.stats.total_observations = 0
        result = v.validate(hyp)
        # Zero observations because precomputed stats had 0
        assert result.total_observations == 0
