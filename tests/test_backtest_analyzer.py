"""Tests for BacktestAnalyzer — Bridge 1: backtest results → hypothesis engine."""

import json
import math
from pathlib import Path

import pytest

from mae_core.market.intelligence.backtest_analyzer import (
    BacktestAnalyzer,
    MIN_AGGREGATE_N,
    _build_story,
)
from mae_core.market.intelligence.hypothesis import (
    Hypothesis,
    HypothesisStatus,
    SourceType,
    TriggerPattern,
)
from mae_core.market.intelligence.hypothesis_registry import HypothesisRegistry


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def tmp_data_dir(tmp_path):
    return tmp_path


@pytest.fixture
def registry(tmp_data_dir):
    return HypothesisRegistry(data_dir=tmp_data_dir)


def _make_trade(symbol="CL=F", direction="bearish", session="asia",
                result="win_2r", r_captured=2.0):
    return {
        "symbol": symbol,
        "direction": direction,
        "session_swept": session,
        "result": result,
        "r_captured": r_captured,
        "entry_price": 70.0,
        "stop_price": 69.0,
        "target_2r": 72.0,
        "entry_time": "2025-12-01 10:00:00-05:00",
        "exit_time": "2025-12-01 11:00:00-05:00",
        "exit_price": 72.0,
    }


def _write_backtest(path, trades, run_time="2026-02-26T10:00:00"):
    data = {"run_time": run_time, "trades": trades}
    path.write_text(json.dumps(data))


def _make_trades(n_wins, n_losses, n_timeouts=0, **kwargs):
    """Generate n trades with specified win/loss/timeout counts."""
    trades = []
    for _ in range(n_wins):
        trades.append(_make_trade(result="win_2r", r_captured=2.0, **kwargs))
    for _ in range(n_losses):
        trades.append(_make_trade(result="loss", r_captured=-1.0, **kwargs))
    for _ in range(n_timeouts):
        trades.append(_make_trade(result="timeout", r_captured=0.2, **kwargs))
    return trades


# ── TestBacktestLoader ────────────────────────────────────────────────

class TestBacktestLoader:
    def test_loads_valid_file(self, registry, tmp_data_dir):
        path = tmp_data_dir / "backtest.json"
        trades = _make_trades(10, 10, symbol="CL=F")
        _write_backtest(path, trades)
        analyzer = BacktestAnalyzer(registry=registry, backtest_path=path)
        data = analyzer._load_backtest()
        assert "trades" in data
        assert len(data["trades"]) == 20

    def test_empty_on_missing_file(self, registry, tmp_data_dir):
        path = tmp_data_dir / "nonexistent.json"
        analyzer = BacktestAnalyzer(registry=registry, backtest_path=path)
        assert analyzer._load_backtest() == {}

    def test_empty_on_malformed_json(self, registry, tmp_data_dir):
        path = tmp_data_dir / "bad.json"
        path.write_text("{not valid json")
        analyzer = BacktestAnalyzer(registry=registry, backtest_path=path)
        assert analyzer._load_backtest() == {}

    def test_empty_on_missing_trades_key(self, registry, tmp_data_dir):
        path = tmp_data_dir / "no_trades.json"
        path.write_text(json.dumps({"run_time": "2026-01-01", "other": []}))
        analyzer = BacktestAnalyzer(registry=registry, backtest_path=path)
        assert analyzer._load_backtest() == {}


# ── TestAggregateBuilding ────────────────────────────────────────────

class TestAggregateBuilding:
    def test_symbol_aggregates(self, registry, tmp_data_dir):
        trades = (
            _make_trades(15, 10, symbol="CL=F")
            + _make_trades(12, 13, symbol="GC=F")
        )
        path = tmp_data_dir / "bt.json"
        _write_backtest(path, trades)
        analyzer = BacktestAnalyzer(registry=registry, backtest_path=path)
        aggs = analyzer._build_aggregates(trades)
        symbol_aggs = [a for a in aggs if a["dimension"] == "symbol"]
        assert len(symbol_aggs) == 2
        symbols = {a["symbol"] for a in symbol_aggs}
        assert symbols == {"CL=F", "GC=F"}

    def test_direction_aggregates(self, registry, tmp_data_dir):
        trades = (
            _make_trades(15, 10, direction="bearish")
            + _make_trades(12, 13, direction="bullish")
        )
        path = tmp_data_dir / "bt.json"
        _write_backtest(path, trades)
        analyzer = BacktestAnalyzer(registry=registry, backtest_path=path)
        aggs = analyzer._build_aggregates(trades)
        dir_aggs = [a for a in aggs if a["dimension"] == "direction"]
        assert len(dir_aggs) == 2

    def test_session_aggregates(self, registry, tmp_data_dir):
        trades = (
            _make_trades(12, 10, session="asia")
            + _make_trades(15, 13, session="london")
        )
        path = tmp_data_dir / "bt.json"
        _write_backtest(path, trades)
        analyzer = BacktestAnalyzer(registry=registry, backtest_path=path)
        aggs = analyzer._build_aggregates(trades)
        sess_aggs = [a for a in aggs if a["dimension"] == "session"]
        assert len(sess_aggs) == 2

    def test_symbol_direction_combos(self, registry, tmp_data_dir):
        trades = (
            _make_trades(12, 10, symbol="CL=F", direction="bearish")
            + _make_trades(11, 12, symbol="CL=F", direction="bullish")
        )
        path = tmp_data_dir / "bt.json"
        _write_backtest(path, trades)
        analyzer = BacktestAnalyzer(registry=registry, backtest_path=path)
        aggs = analyzer._build_aggregates(trades)
        combo_aggs = [a for a in aggs if a["dimension"] == "symbol+direction"]
        assert len(combo_aggs) == 2

    def test_wins_count_matches(self, registry, tmp_data_dir):
        trades = _make_trades(13, 9, 3, symbol="CL=F")  # 13 wins, 9 loss, 3 timeout
        path = tmp_data_dir / "bt.json"
        _write_backtest(path, trades)
        analyzer = BacktestAnalyzer(registry=registry, backtest_path=path)
        aggs = analyzer._build_aggregates(trades)
        sym_agg = [a for a in aggs if a["dimension"] == "symbol"][0]
        assert sym_agg["wins"] == 13
        assert sym_agg["losses"] == 12  # 9 loss + 3 timeout
        assert sym_agg["timeouts"] == 3
        assert sym_agg["total"] == 25


# ── TestFiltering ────────────────────────────────────────────────────

class TestFiltering:
    def test_skips_below_min_n(self, registry, tmp_data_dir):
        # Only 10 trades — below MIN_AGGREGATE_N (20)
        trades = _make_trades(5, 5, symbol="CL=F")
        path = tmp_data_dir / "bt.json"
        _write_backtest(path, trades)
        analyzer = BacktestAnalyzer(registry=registry, backtest_path=path)
        result = analyzer.analyze()
        assert len(result) == 0

    def test_generates_from_qualifying(self, registry, tmp_data_dir):
        # 25 trades — above threshold
        trades = _make_trades(15, 10, symbol="CL=F")
        path = tmp_data_dir / "bt.json"
        _write_backtest(path, trades)
        analyzer = BacktestAnalyzer(registry=registry, backtest_path=path)
        result = analyzer.analyze()
        # Should produce at least symbol + direction + session aggregates
        assert len(result) >= 1


# ── TestHypothesisConstruction ──────────────────────────────────────

class TestHypothesisConstruction:
    @pytest.fixture
    def sample_hypothesis(self, registry, tmp_data_dir):
        trades = _make_trades(15, 10, symbol="CL=F", direction="bearish", session="asia")
        path = tmp_data_dir / "bt.json"
        _write_backtest(path, trades)
        analyzer = BacktestAnalyzer(registry=registry, backtest_path=path)
        hypotheses = analyzer.analyze()
        return hypotheses

    def test_source_type_is_backtest_derived(self, sample_hypothesis):
        for hyp in sample_hypothesis:
            assert hyp.source_type == SourceType.BACKTEST_DERIVED

    def test_status_is_probation(self, sample_hypothesis):
        for hyp in sample_hypothesis:
            assert hyp.status == HypothesisStatus.PROBATION

    def test_trigger_source_a(self, sample_hypothesis):
        for hyp in sample_hypothesis:
            assert hyp.trigger.source_a == "session_sweep"

    def test_trigger_source_b(self, sample_hypothesis):
        for hyp in sample_hypothesis:
            assert hyp.trigger.source_b == "price_outcome"

    def test_trigger_lag_days_zero(self, sample_hypothesis):
        for hyp in sample_hypothesis:
            assert hyp.trigger.lag_days == 0

    def test_domain_filter_encoding(self, registry, tmp_data_dir):
        trades = _make_trades(15, 10, symbol="CL=F", direction="bearish", session="asia")
        path = tmp_data_dir / "bt.json"
        _write_backtest(path, trades)
        analyzer = BacktestAnalyzer(registry=registry, backtest_path=path)
        hypotheses = analyzer.analyze()
        domain_filters = {h.trigger.domain_filter for h in hypotheses}
        # Should include combo filter
        assert "CL=F:bearish" in domain_filters

    def test_stats_pre_populated(self, sample_hypothesis):
        for hyp in sample_hypothesis:
            assert hyp.stats.total_observations > 0
            assert hyp.stats.wins + hyp.stats.losses == hyp.stats.total_observations

    def test_causal_story_is_real(self, sample_hypothesis):
        for hyp in sample_hypothesis:
            assert hyp.causal_story
            assert "REQUIRES MANUAL REVIEW" not in hyp.causal_story

    def test_parent_lag_finding_populated(self, sample_hypothesis):
        for hyp in sample_hypothesis:
            assert hyp.parent_lag_finding
            assert "dimension" in hyp.parent_lag_finding
            assert "total" in hyp.parent_lag_finding


# ── TestDedup ────────────────────────────────────────────────────────

class TestDedup:
    def test_second_analyze_produces_nothing(self, registry, tmp_data_dir):
        trades = _make_trades(15, 10, symbol="CL=F")
        path = tmp_data_dir / "bt.json"
        _write_backtest(path, trades)
        analyzer = BacktestAnalyzer(registry=registry, backtest_path=path)
        first = analyzer.analyze()
        assert len(first) >= 1
        second = analyzer.analyze()
        assert len(second) == 0

    def test_dedup_checks_domain_filter(self, registry, tmp_data_dir):
        trades = _make_trades(15, 10, symbol="CL=F")
        path = tmp_data_dir / "bt.json"
        _write_backtest(path, trades)
        analyzer = BacktestAnalyzer(registry=registry, backtest_path=path)
        analyzer.analyze()

        # Manually check that duplicate detection works
        trigger = TriggerPattern(
            source_a="session_sweep", source_b="price_outcome",
            lag_days=0, domain_filter="CL=F",
        )
        assert analyzer._is_duplicate(trigger) is True

        # New domain filter should not be duplicate
        trigger_new = TriggerPattern(
            source_a="session_sweep", source_b="price_outcome",
            lag_days=0, domain_filter="AAPL",
        )
        assert analyzer._is_duplicate(trigger_new) is False


# ── TestSharpe ───────────────────────────────────────────────────────

class TestSharpe:
    def test_sharpe_uses_r_captured(self, registry, tmp_data_dir):
        path = tmp_data_dir / "bt.json"
        _write_backtest(path, [])
        analyzer = BacktestAnalyzer(registry=registry, backtest_path=path)
        r_values = [2.0, 2.0, -1.0, -1.0, 0.2]
        sharpe = analyzer._compute_sharpe(r_values)
        assert isinstance(sharpe, float)

    def test_sharpe_positive_for_winning_trades(self, registry, tmp_data_dir):
        path = tmp_data_dir / "bt.json"
        _write_backtest(path, [])
        analyzer = BacktestAnalyzer(registry=registry, backtest_path=path)
        r_values = [2.0, 2.0, 2.0, -1.0]  # net positive
        sharpe = analyzer._compute_sharpe(r_values)
        assert sharpe > 0

    def test_sharpe_empty_returns_zero(self, registry, tmp_data_dir):
        path = tmp_data_dir / "bt.json"
        _write_backtest(path, [])
        analyzer = BacktestAnalyzer(registry=registry, backtest_path=path)
        assert analyzer._compute_sharpe([]) == 0.0
        assert analyzer._compute_sharpe([1.0]) == 0.0

    def test_timeout_trades_included(self, registry, tmp_data_dir):
        path = tmp_data_dir / "bt.json"
        _write_backtest(path, [])
        analyzer = BacktestAnalyzer(registry=registry, backtest_path=path)
        # All timeouts at +0.2R
        r_values = [0.2, 0.2, 0.2, 0.2]
        sharpe = analyzer._compute_sharpe(r_values)
        # Std dev is 0 → returns 0.0
        assert sharpe == 0.0


# ── TestCausalStory ──────────────────────────────────────────────────

class TestCausalStory:
    def test_symbol_story(self):
        agg = {"symbol": "CL=F", "direction": None, "session": None,
               "win_rate": 0.52, "total": 44, "avg_r": 0.3}
        story = _build_story(agg)
        assert "crude" in story.lower() or "oil" in story.lower()

    def test_direction_story(self):
        agg = {"symbol": None, "direction": "bearish", "session": None,
               "win_rate": 0.50, "total": 169, "avg_r": 0.1}
        story = _build_story(agg)
        assert "bearish" in story.lower() or "buy-side" in story.lower() or "stop" in story.lower()

    def test_combo_story_has_both(self):
        agg = {"symbol": "GC=F", "direction": "bullish", "session": None,
               "win_rate": 0.55, "total": 30, "avg_r": 0.5}
        story = _build_story(agg)
        assert "gold" in story.lower() or "bullion" in story.lower()
        assert "bullish" in story.lower() or "sell-side" in story.lower() or "stop" in story.lower()

    def test_story_includes_backtest_evidence(self):
        agg = {"symbol": "CL=F", "direction": None, "session": None,
               "win_rate": 0.52, "total": 44, "avg_r": 0.3}
        story = _build_story(agg)
        assert "52.0%" in story or "44 trades" in story


# ── TestStatistics ───────────────────────────────────────────────────

class TestAnalyzerStatistics:
    def test_get_statistics(self, registry, tmp_data_dir):
        path = tmp_data_dir / "bt.json"
        _write_backtest(path, [])
        analyzer = BacktestAnalyzer(registry=registry, backtest_path=path)
        stats = analyzer.get_statistics()
        assert "aggregates_built" in stats
        assert "hypotheses_created" in stats
        assert "backtest_file_exists" in stats
