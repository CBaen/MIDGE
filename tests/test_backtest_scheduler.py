"""Tests for BacktestScheduler — Bridge 3: autonomous backtest rerun scheduling."""

import json
from concurrent.futures import Future
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mae_core.market.intelligence.backtest_scheduler import BacktestScheduler
from mae_core.market.intelligence.hypothesis import (
    Hypothesis,
    HypothesisStats,
    HypothesisStatus,
    SourceType,
    TriggerPattern,
)
from mae_core.market.intelligence.hypothesis_registry import HypothesisRegistry


# ── Helpers ───────────────────────────────────────────────────────────

def _make_mock_trade(**overrides):
    """Create a mock trade with all required attributes."""
    defaults = dict(
        result="win_2r", symbol="ES=F", direction="bearish",
        session_swept="asia", entry_price=5000, stop_price=4990,
        target_2r=5020, entry_time="2026-01-01 10:00",
        exit_time="2026-01-01 11:00", exit_price=5020,
        r_captured=2.0, hit_1r=True, risk_pts=10,
        displacement_score=0.7, fvg_atr_ratio=1.2,
        kill_zone_score=0.9, quality_score=0.8,
    )
    defaults.update(overrides)
    trade = MagicMock()
    for k, v in defaults.items():
        setattr(trade, k, v)
    return trade


def _write_results(path, run_time=None, trades=None):
    """Write a minimal results file with the given run_time."""
    if run_time is None:
        run_time = datetime.now().isoformat()
    data = {
        "run_time": run_time,
        "config": {"interval": "5m", "days": 59, "symbols": ["ES=F"]},
        "summary": {"total_trades": 10, "wins": 6, "losses": 4,
                     "win_rate": 60.0, "avg_r": 0.8, "hit_1r_count": 7},
        "trades": trades or [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def _make_trigger(domain_filter="CL=F"):
    """Create a TriggerPattern matching backtest_analyzer's format."""
    return TriggerPattern(
        source_a="session_sweep", source_b="price_outcome",
        lag_days=0, domain_filter=domain_filter,
    )


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def tmp_results(tmp_path):
    return tmp_path / "sweep_backtest_results.json"


@pytest.fixture
def mock_analyzer():
    analyzer = MagicMock()
    analyzer.refresh_probation.return_value = 3
    analyzer.analyze.return_value = [MagicMock(), MagicMock()]
    return analyzer


@pytest.fixture
def scheduler(mock_analyzer, tmp_results):
    return BacktestScheduler(
        backtest_analyzer=mock_analyzer,
        results_path=tmp_results,
        stale_threshold_hours=24.0,
        symbols=["ES=F", "NQ=F"],
    )


# ── TestStalenessDetection ────────────────────────────────────────────

class TestStalenessDetection:
    def test_stale_when_file_missing(self, scheduler, tmp_results):
        assert not tmp_results.exists()
        assert scheduler._is_stale() is True

    def test_stale_when_older_than_threshold(self, scheduler, tmp_results):
        old_time = (datetime.now() - timedelta(hours=25)).isoformat()
        _write_results(tmp_results, run_time=old_time)
        assert scheduler._is_stale() is True

    def test_fresh_when_recent(self, scheduler, tmp_results):
        recent = (datetime.now() - timedelta(hours=1)).isoformat()
        _write_results(tmp_results, run_time=recent)
        assert scheduler._is_stale() is False

    def test_stale_on_malformed_json(self, scheduler, tmp_results):
        tmp_results.parent.mkdir(parents=True, exist_ok=True)
        tmp_results.write_text("{not valid json")
        assert scheduler._is_stale() is True

    def test_stale_on_missing_run_time(self, scheduler, tmp_results):
        tmp_results.parent.mkdir(parents=True, exist_ok=True)
        tmp_results.write_text(json.dumps({"trades": []}))
        assert scheduler._is_stale() is True

    def test_reads_run_time_correctly(self, scheduler, tmp_results):
        exactly_23h = (datetime.now() - timedelta(hours=23)).isoformat()
        _write_results(tmp_results, run_time=exactly_23h)
        assert scheduler._is_stale() is False

    def test_exactly_at_threshold_is_stale(self, scheduler, tmp_results):
        exactly_24h = (datetime.now() - timedelta(hours=24, seconds=1)).isoformat()
        _write_results(tmp_results, run_time=exactly_24h)
        assert scheduler._is_stale() is True


# ── TestCheckAndSchedule ──────────────────────────────────────────────

class TestCheckAndSchedule:
    def test_skips_when_fresh(self, scheduler, tmp_results):
        _write_results(tmp_results)
        scheduler.check_and_schedule()
        assert scheduler._runs_scheduled == 0

    def test_launches_when_stale(self, scheduler, tmp_results):
        with patch.object(scheduler, "_run_backtest_and_refresh",
                          return_value={"trades": 10, "refreshed": 2}):
            scheduler.check_and_schedule()
        assert scheduler._runs_scheduled == 1

    def test_skips_when_busy(self, scheduler, tmp_results):
        fake_future = Future()
        scheduler._pending_future = fake_future
        scheduler.check_and_schedule()
        assert scheduler._runs_scheduled == 0

    def test_does_not_double_submit(self, scheduler, tmp_results):
        # First call: launch. Second call: future still pending → skip.
        fake_future = Future()  # never completed
        scheduler._pending_future = fake_future
        scheduler._runs_scheduled = 1  # simulate first submit already happened
        scheduler.check_and_schedule()  # should skip — busy
        assert scheduler._runs_scheduled == 1

    def test_collects_completed_future(self, scheduler, tmp_results):
        done_future = Future()
        done_future.set_result({"trades": 8, "refreshed": 2})
        scheduler._pending_future = done_future

        _write_results(tmp_results)  # fresh — won't re-launch
        scheduler.check_and_schedule()

        assert scheduler._pending_future is None
        assert scheduler._runs_completed == 1


# ── TestRunBacktestAndRefresh ─────────────────────────────────────────

class TestRunBacktestAndRefresh:
    def test_calls_analyzer_refresh_and_analyze(self, scheduler, mock_analyzer):
        mock_trade = _make_mock_trade()
        with patch(
            "mae_core.market.edge.sweep_backtest.SweepBacktester"
        ) as MockBT:
            MockBT.return_value.run.return_value = [mock_trade]
            result = scheduler._run_backtest_and_refresh()

        assert result["trades"] == 1
        assert result["refreshed"] == 3
        mock_analyzer.refresh_probation.assert_called_once()
        mock_analyzer.analyze.assert_called_once()

    def test_handles_empty_trades(self, scheduler, mock_analyzer):
        with patch(
            "mae_core.market.edge.sweep_backtest.SweepBacktester"
        ) as MockBT:
            MockBT.return_value.run.return_value = []
            result = scheduler._run_backtest_and_refresh()

        assert result["trades"] == 0
        mock_analyzer.refresh_probation.assert_not_called()

    def test_handles_backtest_exception(self, scheduler, mock_analyzer):
        with patch(
            "mae_core.market.edge.sweep_backtest.SweepBacktester"
        ) as MockBT:
            MockBT.return_value.run.side_effect = RuntimeError("network error")
            result = scheduler._run_backtest_and_refresh()

        assert result.get("error") is True
        assert result["trades"] == 0

    def test_publishes_event_when_bus_present(self, scheduler, mock_analyzer):
        bus = MagicMock()
        scheduler._bus = bus

        mock_trade = _make_mock_trade()
        with patch(
            "mae_core.market.edge.sweep_backtest.SweepBacktester"
        ) as MockBT:
            MockBT.return_value.run.return_value = [mock_trade]
            scheduler._run_backtest_and_refresh()

        bus.publish.assert_called_once()
        call_args = bus.publish.call_args
        assert call_args[0][0] == "market.intel.backtest_refreshed"

    def test_no_publish_when_bus_is_none(self, scheduler, mock_analyzer):
        scheduler._bus = None
        mock_trade = _make_mock_trade()
        with patch(
            "mae_core.market.edge.sweep_backtest.SweepBacktester"
        ) as MockBT:
            MockBT.return_value.run.return_value = [mock_trade]
            result = scheduler._run_backtest_and_refresh()

        assert result["trades"] == 1  # still succeeds


# ── TestRefreshProbation ──────────────────────────────────────────────

class TestRefreshProbation:
    """Tests for BacktestAnalyzer.refresh_probation() — called by scheduler."""

    @pytest.fixture
    def registry(self, tmp_path):
        return HypothesisRegistry(data_dir=tmp_path)

    def _make_hypothesis(self, registry, status, source_type, domain_filter="CL=F"):
        from mae_core.market.intelligence.backtest_analyzer import BacktestAnalyzer
        trigger = _make_trigger(domain_filter)
        hyp = Hypothesis(
            trigger=trigger,
            causal_story="Test story",
            source_type=source_type,
        )
        hyp_id = registry.register(hyp)
        if status == HypothesisStatus.ACTIVE:
            registry.promote(hyp_id)
        return hyp_id

    def test_retires_probation_backtest_derived(self, registry, tmp_path):
        from mae_core.market.intelligence.backtest_analyzer import BacktestAnalyzer
        self._make_hypothesis(registry, HypothesisStatus.PROBATION,
                              SourceType.BACKTEST_DERIVED, "CL=F")
        self._make_hypothesis(registry, HypothesisStatus.PROBATION,
                              SourceType.BACKTEST_DERIVED, "GC=F")

        path = tmp_path / "bt.json"
        path.write_text(json.dumps({"run_time": "2026-01-01", "trades": []}))
        analyzer = BacktestAnalyzer(registry=registry, backtest_path=path)
        count = analyzer.refresh_probation()
        assert count == 2

    def test_skips_active_hypotheses(self, registry, tmp_path):
        from mae_core.market.intelligence.backtest_analyzer import BacktestAnalyzer
        self._make_hypothesis(registry, HypothesisStatus.ACTIVE,
                              SourceType.BACKTEST_DERIVED, "CL=F")

        path = tmp_path / "bt.json"
        path.write_text(json.dumps({"run_time": "2026-01-01", "trades": []}))
        analyzer = BacktestAnalyzer(registry=registry, backtest_path=path)
        count = analyzer.refresh_probation()
        assert count == 0

    def test_skips_non_backtest_derived(self, registry, tmp_path):
        from mae_core.market.intelligence.backtest_analyzer import BacktestAnalyzer
        self._make_hypothesis(registry, HypothesisStatus.PROBATION,
                              SourceType.LAG_CORRELATION, "CL=F")

        path = tmp_path / "bt.json"
        path.write_text(json.dumps({"run_time": "2026-01-01", "trades": []}))
        analyzer = BacktestAnalyzer(registry=registry, backtest_path=path)
        count = analyzer.refresh_probation()
        assert count == 0

    def test_returns_count(self, registry, tmp_path):
        from mae_core.market.intelligence.backtest_analyzer import BacktestAnalyzer
        for i in range(5):
            self._make_hypothesis(registry, HypothesisStatus.PROBATION,
                                  SourceType.BACKTEST_DERIVED, f"SYM{i}")

        path = tmp_path / "bt.json"
        path.write_text(json.dumps({"run_time": "2026-01-01", "trades": []}))
        analyzer = BacktestAnalyzer(registry=registry, backtest_path=path)
        assert analyzer.refresh_probation() == 5


# ── TestDedupWithRetired ──────────────────────────────────────────────

class TestDedupWithRetired:
    """Tests for _is_duplicate() skipping RETIRED hypotheses."""

    @pytest.fixture
    def registry(self, tmp_path):
        return HypothesisRegistry(data_dir=tmp_path)

    def test_is_duplicate_skips_retired(self, registry, tmp_path):
        from mae_core.market.intelligence.backtest_analyzer import BacktestAnalyzer
        trigger = _make_trigger("CL=F")
        hyp = Hypothesis(
            trigger=trigger,
            causal_story="Test story",
            source_type=SourceType.BACKTEST_DERIVED,
        )
        hyp_id = registry.register(hyp)
        registry.retire(hyp_id, reason="test_retire")

        path = tmp_path / "bt.json"
        path.write_text(json.dumps({"run_time": "2026-01-01", "trades": []}))
        analyzer = BacktestAnalyzer(registry=registry, backtest_path=path)

        assert analyzer._is_duplicate(trigger) is False

    def test_is_duplicate_catches_probation(self, registry, tmp_path):
        from mae_core.market.intelligence.backtest_analyzer import BacktestAnalyzer
        trigger = _make_trigger("CL=F")
        hyp = Hypothesis(
            trigger=trigger,
            causal_story="Test story",
            source_type=SourceType.BACKTEST_DERIVED,
        )
        registry.register(hyp)  # returns ID, don't need it

        path = tmp_path / "bt.json"
        path.write_text(json.dumps({"run_time": "2026-01-01", "trades": []}))
        analyzer = BacktestAnalyzer(registry=registry, backtest_path=path)

        assert analyzer._is_duplicate(trigger) is True

    def test_purge_recreate_cycle(self, registry, tmp_path):
        """End-to-end: refresh_probation retires, then _is_duplicate allows recreation."""
        from mae_core.market.intelligence.backtest_analyzer import BacktestAnalyzer
        trigger = _make_trigger("CL=F")
        hyp = Hypothesis(
            trigger=trigger,
            causal_story="Test story",
            source_type=SourceType.BACKTEST_DERIVED,
        )
        registry.register(hyp)

        path = tmp_path / "bt.json"
        path.write_text(json.dumps({"run_time": "2026-01-01", "trades": []}))
        analyzer = BacktestAnalyzer(registry=registry, backtest_path=path)

        # Before refresh: duplicate detected
        assert analyzer._is_duplicate(trigger) is True

        # Refresh retires it
        count = analyzer.refresh_probation()
        assert count == 1

        # After refresh: no longer duplicate
        assert analyzer._is_duplicate(trigger) is False


# ── TestGetStatistics ─────────────────────────────────────────────────

class TestGetStatistics:
    def test_initial_statistics(self, scheduler):
        stats = scheduler.get_statistics()
        assert stats["runs_scheduled"] == 0
        assert stats["runs_completed"] == 0
        assert stats["last_run_time"] == ""
        assert stats["hypotheses_refreshed"] == 0
        assert stats["stale_threshold_hours"] == 24.0
        assert stats["symbols_count"] == 2
        assert stats["is_running"] is False

    def test_updates_after_run(self, scheduler, tmp_results):
        with patch.object(scheduler, "_run_backtest_and_refresh",
                          return_value={"trades": 10, "refreshed": 3}):
            scheduler.check_and_schedule()

        stats = scheduler.get_statistics()
        assert stats["runs_scheduled"] == 1

    def test_is_running_while_future_pending(self, scheduler):
        fake_future = Future()
        scheduler._pending_future = fake_future
        stats = scheduler.get_statistics()
        assert stats["is_running"] is True


# ── TestWriteResults ──────────────────────────────────────────────────

class TestWriteResults:
    def test_writes_valid_json(self, scheduler, tmp_results):
        mock_trade = _make_mock_trade()
        scheduler._write_results([mock_trade])

        assert tmp_results.exists()
        data = json.loads(tmp_results.read_text())
        assert "run_time" in data
        assert data["summary"]["total_trades"] == 1
        assert len(data["trades"]) == 1

    def test_creates_parent_directory(self, mock_analyzer, tmp_path):
        nested = tmp_path / "deep" / "nested" / "results.json"
        sched = BacktestScheduler(
            backtest_analyzer=mock_analyzer,
            results_path=nested,
        )
        sched._write_results([_make_mock_trade()])
        assert nested.exists()
