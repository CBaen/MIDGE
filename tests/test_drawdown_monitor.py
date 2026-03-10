"""Tests for DrawdownMonitor — equity curve tracking and drawdown circuit-breaker.

What is under test:
  - mae_core/market/intelligence/drawdown_monitor.py (DrawdownMonitor)

Design decisions validated here:
  - equity tracks as starting_capital + sum of realized P&L
  - Peak equity only updates on new highs (never decreases)
  - get_current_drawdown() = (peak - current) / peak, floor at 0.0
  - Circuit-breaker trips when drawdown >= max_drawdown_pct
  - Warning fires once when drawdown >= 80% of max_drawdown_pct
  - Trading resumes and halted=False when drawdown recovers below max
  - Warning flag clears when drawdown recovers below warning threshold
  - Persistence round-trip: save_state / load_state are inverse operations
  - Thread safety: concurrent record_trade_result calls don't corrupt state
  - Graceful degradation: EventBus=None never raises
  - get_statistics returns all required HolonProxy keys
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from mae_core.market.intelligence.drawdown_monitor import (
    DrawdownMonitor,
    CH_DRAWDOWN_WARNING,
    CH_TRADING_HALTED,
    CH_TRADING_RESUMED,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_monitor(**kwargs) -> DrawdownMonitor:
    """Construct a DrawdownMonitor with sensible test defaults."""
    defaults = dict(
        event_bus=None,
        starting_capital=10_000.0,
        max_drawdown_pct=0.40,
        data_dir="data/market",
    )
    defaults.update(kwargs)
    return DrawdownMonitor(**defaults)


def _make_bus() -> MagicMock:
    """Return a mock EventBus."""
    bus = MagicMock()
    bus.publish = MagicMock()
    return bus


# ---------------------------------------------------------------------------
# Equity tracking — positive and negative trades
# ---------------------------------------------------------------------------

class TestEquityTracking:

    def test_initial_state(self):
        m = _make_monitor(starting_capital=10_000.0)
        assert m._current_equity == 10_000.0
        assert m._peak_equity == 10_000.0
        assert m._realized_pnl == 0.0
        assert m._trade_count == 0

    def test_profit_trade_increases_equity(self):
        m = _make_monitor(starting_capital=10_000.0)
        m.record_trade_result("AAPL", realized_pnl=500.0)
        assert m._current_equity == pytest.approx(10_500.0)
        assert m._realized_pnl == pytest.approx(500.0)
        assert m._trade_count == 1

    def test_loss_trade_decreases_equity(self):
        m = _make_monitor(starting_capital=10_000.0)
        m.record_trade_result("AAPL", realized_pnl=-300.0)
        assert m._current_equity == pytest.approx(9_700.0)
        assert m._realized_pnl == pytest.approx(-300.0)

    def test_multiple_trades_accumulate(self):
        m = _make_monitor(starting_capital=10_000.0)
        m.record_trade_result("AAPL", realized_pnl=500.0)
        m.record_trade_result("TSLA", realized_pnl=-200.0)
        m.record_trade_result("MSFT", realized_pnl=100.0)
        assert m._current_equity == pytest.approx(10_400.0)
        assert m._realized_pnl == pytest.approx(400.0)
        assert m._trade_count == 3

    def test_equity_history_grows(self):
        m = _make_monitor()
        assert len(m._equity_history) == 0
        m.record_trade_result("AAPL", realized_pnl=100.0)
        assert len(m._equity_history) == 1
        m.record_trade_result("TSLA", realized_pnl=-50.0)
        assert len(m._equity_history) == 2

    def test_equity_history_stores_ticker_and_pnl(self):
        m = _make_monitor()
        m.record_trade_result("AAPL", realized_pnl=123.45, direction="buy")
        record = m._equity_history[-1]
        assert record["ticker"] == "AAPL"
        assert record["pnl_delta"] == pytest.approx(123.45)
        assert record["direction"] == "buy"
        assert "ts" in record
        assert "equity" in record

    def test_history_maxlen_respected(self):
        """Verify deque rolls over at 10,000 — no unbounded growth."""
        from mae_core.market.intelligence.drawdown_monitor import _HISTORY_MAXLEN
        m = _make_monitor()
        for i in range(_HISTORY_MAXLEN + 100):
            m.record_trade_result("X", realized_pnl=0.01)
        assert len(m._equity_history) == _HISTORY_MAXLEN


# ---------------------------------------------------------------------------
# Peak equity — only updates on new highs
# ---------------------------------------------------------------------------

class TestPeakEquity:

    def test_peak_updates_on_profit(self):
        m = _make_monitor(starting_capital=10_000.0)
        m.record_trade_result("AAPL", realized_pnl=1_000.0)
        assert m._peak_equity == pytest.approx(11_000.0)

    def test_peak_does_not_retreat_on_loss(self):
        m = _make_monitor(starting_capital=10_000.0)
        m.record_trade_result("AAPL", realized_pnl=1_000.0)  # peak = 11,000
        m.record_trade_result("TSLA", realized_pnl=-500.0)   # equity = 10,500 but peak stays
        assert m._peak_equity == pytest.approx(11_000.0)
        assert m._current_equity == pytest.approx(10_500.0)

    def test_peak_updates_to_successive_highs(self):
        m = _make_monitor(starting_capital=10_000.0)
        m.record_trade_result("AAPL", realized_pnl=500.0)   # peak = 10,500
        m.record_trade_result("TSLA", realized_pnl=500.0)   # peak = 11,000
        m.record_trade_result("MSFT", realized_pnl=-200.0)  # peak stays 11,000
        m.record_trade_result("GOOG", realized_pnl=1_000.0) # peak = 11,800
        assert m._peak_equity == pytest.approx(11_800.0)


# ---------------------------------------------------------------------------
# Drawdown calculation
# ---------------------------------------------------------------------------

class TestDrawdownCalculation:

    def test_no_drawdown_at_peak(self):
        m = _make_monitor(starting_capital=10_000.0)
        m.record_trade_result("AAPL", realized_pnl=1_000.0)
        assert m.get_current_drawdown() == pytest.approx(0.0)

    def test_drawdown_after_loss_from_peak(self):
        m = _make_monitor(starting_capital=10_000.0)
        m.record_trade_result("AAPL", realized_pnl=1_000.0)  # peak = 11,000
        m.record_trade_result("TSLA", realized_pnl=-1_100.0) # equity = 9,900
        # drawdown = (11000 - 9900) / 11000 = 1100/11000 = 0.1
        assert m.get_current_drawdown() == pytest.approx(0.1)

    def test_drawdown_from_starting_capital(self):
        """Loss with no prior profit: drawdown from starting_capital."""
        m = _make_monitor(starting_capital=10_000.0)
        m.record_trade_result("AAPL", realized_pnl=-2_000.0)
        # drawdown = (10000 - 8000) / 10000 = 0.20
        assert m.get_current_drawdown() == pytest.approx(0.20)

    def test_drawdown_floor_at_zero(self):
        """get_current_drawdown never returns negative."""
        m = _make_monitor(starting_capital=10_000.0)
        # No trades — equity == peak
        assert m.get_current_drawdown() == 0.0
        m.record_trade_result("AAPL", realized_pnl=500.0)
        assert m.get_current_drawdown() == 0.0

    def test_forty_percent_drawdown(self):
        m = _make_monitor(starting_capital=10_000.0, max_drawdown_pct=0.40)
        m.record_trade_result("AAPL", realized_pnl=-4_000.0)
        # drawdown = (10000 - 6000) / 10000 = 0.40
        assert m.get_current_drawdown() == pytest.approx(0.40)


# ---------------------------------------------------------------------------
# Circuit-breaker: halt and resume
# ---------------------------------------------------------------------------

class TestCircuitBreaker:

    def test_not_halted_initially(self):
        m = _make_monitor()
        assert not m.is_trading_halted()

    def test_not_halted_below_threshold(self):
        m = _make_monitor(starting_capital=10_000.0, max_drawdown_pct=0.40)
        m.record_trade_result("AAPL", realized_pnl=-3_900.0)  # 39% drawdown
        assert not m.is_trading_halted()

    def test_halted_at_exact_threshold(self):
        m = _make_monitor(starting_capital=10_000.0, max_drawdown_pct=0.40)
        m.record_trade_result("AAPL", realized_pnl=-4_000.0)  # exactly 40%
        assert m.is_trading_halted()

    def test_halted_above_threshold(self):
        m = _make_monitor(starting_capital=10_000.0, max_drawdown_pct=0.40)
        m.record_trade_result("AAPL", realized_pnl=-5_000.0)  # 50% drawdown
        assert m.is_trading_halted()

    def test_trading_resumes_after_recovery(self):
        m = _make_monitor(starting_capital=10_000.0, max_drawdown_pct=0.40)
        m.record_trade_result("AAPL", realized_pnl=-4_500.0)  # 45% — halted
        assert m.is_trading_halted()
        # Recover with a big profit — drawdown recovers
        m.record_trade_result("AAPL", realized_pnl=3_000.0)
        # equity = 8500, peak = 10000, drawdown = 15% — below 40% halt
        assert not m.is_trading_halted()

    def test_halted_event_published(self):
        bus = _make_bus()
        m = _make_monitor(event_bus=bus, starting_capital=10_000.0, max_drawdown_pct=0.40)
        m.record_trade_result("AAPL", realized_pnl=-4_000.0)
        calls = [c[0][0] for c in bus.publish.call_args_list]
        assert CH_TRADING_HALTED in calls

    def test_halted_event_published_only_once_per_trip(self):
        """Circuit-breaker should not fire again if already halted."""
        bus = _make_bus()
        m = _make_monitor(event_bus=bus, starting_capital=10_000.0, max_drawdown_pct=0.40)
        m.record_trade_result("AAPL", realized_pnl=-4_000.0)
        m.record_trade_result("AAPL", realized_pnl=-500.0)  # still halted, deeper
        halt_calls = [c for c in bus.publish.call_args_list if c[0][0] == CH_TRADING_HALTED]
        assert len(halt_calls) == 1

    def test_resumed_event_published(self):
        bus = _make_bus()
        m = _make_monitor(event_bus=bus, starting_capital=10_000.0, max_drawdown_pct=0.40)
        m.record_trade_result("AAPL", realized_pnl=-4_500.0)  # halted
        bus.publish.reset_mock()
        m.record_trade_result("AAPL", realized_pnl=3_000.0)   # recover
        calls = [c[0][0] for c in bus.publish.call_args_list]
        assert CH_TRADING_RESUMED in calls
        assert not m.is_trading_halted()


# ---------------------------------------------------------------------------
# Warning threshold (80% of max)
# ---------------------------------------------------------------------------

class TestWarningThreshold:

    def test_warning_fires_at_80pct_of_max(self):
        """With max=40%, warning should fire at 32% drawdown."""
        bus = _make_bus()
        m = _make_monitor(event_bus=bus, starting_capital=10_000.0, max_drawdown_pct=0.40)
        # 32% drawdown (exactly 80% of 40%)
        m.record_trade_result("AAPL", realized_pnl=-3_200.0)
        calls = [c[0][0] for c in bus.publish.call_args_list]
        assert CH_DRAWDOWN_WARNING in calls
        assert CH_TRADING_HALTED not in calls

    def test_warning_does_not_fire_below_80pct(self):
        bus = _make_bus()
        m = _make_monitor(event_bus=bus, starting_capital=10_000.0, max_drawdown_pct=0.40)
        # 31% drawdown — just below warning threshold of 32%
        m.record_trade_result("AAPL", realized_pnl=-3_100.0)
        calls = [c[0][0] for c in bus.publish.call_args_list]
        assert CH_DRAWDOWN_WARNING not in calls

    def test_warning_fires_only_once_per_zone_entry(self):
        """Warning should not spam once already in warning zone."""
        bus = _make_bus()
        m = _make_monitor(event_bus=bus, starting_capital=10_000.0, max_drawdown_pct=0.40)
        m.record_trade_result("AAPL", realized_pnl=-3_200.0)  # enters warning
        m.record_trade_result("AAPL", realized_pnl=-100.0)    # still in warning
        warning_calls = [c for c in bus.publish.call_args_list if c[0][0] == CH_DRAWDOWN_WARNING]
        assert len(warning_calls) == 1

    def test_warning_clears_on_recovery(self):
        bus = _make_bus()
        m = _make_monitor(event_bus=bus, starting_capital=10_000.0, max_drawdown_pct=0.40)
        m.record_trade_result("AAPL", realized_pnl=-3_200.0)  # warning active
        assert m._warning_active
        # Recover fully — big profit to push above starting capital
        m.record_trade_result("AAPL", realized_pnl=5_000.0)
        assert not m._warning_active

    def test_warning_refires_after_recovery_and_reentry(self):
        """After recovering and then falling again, warning should fire again."""
        bus = _make_bus()
        m = _make_monitor(event_bus=bus, starting_capital=10_000.0, max_drawdown_pct=0.40)
        m.record_trade_result("AAPL", realized_pnl=-3_200.0)  # 1st warning
        m.record_trade_result("AAPL", realized_pnl=5_000.0)   # recover
        m.record_trade_result("AAPL", realized_pnl=-4_500.0)  # 2nd warning + halt
        warning_calls = [c for c in bus.publish.call_args_list if c[0][0] == CH_DRAWDOWN_WARNING]
        assert len(warning_calls) == 2


# ---------------------------------------------------------------------------
# Graceful degradation (event_bus=None)
# ---------------------------------------------------------------------------

class TestGracefulDegradation:

    def test_no_exception_without_bus(self):
        m = _make_monitor(event_bus=None, starting_capital=10_000.0, max_drawdown_pct=0.40)
        # Should not raise even when circuit-breaker trips
        m.record_trade_result("AAPL", realized_pnl=-4_000.0)
        assert m.is_trading_halted()

    def test_state_correct_without_bus(self):
        m = _make_monitor(event_bus=None, starting_capital=10_000.0, max_drawdown_pct=0.40)
        m.record_trade_result("AAPL", realized_pnl=-4_000.0)
        assert m._trading_halted is True
        assert m._warning_active is True
        m.record_trade_result("AAPL", realized_pnl=5_000.0)
        assert m._trading_halted is False


# ---------------------------------------------------------------------------
# get_statistics
# ---------------------------------------------------------------------------

class TestGetStatistics:

    def test_returns_required_keys(self):
        m = _make_monitor(starting_capital=10_000.0)
        stats = m.get_statistics()
        required_keys = {
            "peak_equity", "current_equity", "drawdown_pct",
            "realized_pnl", "trading_halted", "trade_count",
        }
        assert required_keys <= set(stats.keys()), (
            f"Missing keys: {required_keys - set(stats.keys())}"
        )

    def test_initial_stats_values(self):
        m = _make_monitor(starting_capital=10_000.0)
        stats = m.get_statistics()
        assert stats["peak_equity"] == pytest.approx(10_000.0)
        assert stats["current_equity"] == pytest.approx(10_000.0)
        assert stats["drawdown_pct"] == pytest.approx(0.0)
        assert stats["realized_pnl"] == pytest.approx(0.0)
        assert stats["trading_halted"] is False
        assert stats["trade_count"] == 0

    def test_stats_reflect_trades(self):
        m = _make_monitor(starting_capital=10_000.0, max_drawdown_pct=0.40)
        m.record_trade_result("AAPL", realized_pnl=2_000.0)  # peak = 12,000
        m.record_trade_result("TSLA", realized_pnl=-4_800.0) # equity = 9,200, drawdown = 7200/12000 = 60%
        stats = m.get_statistics()
        assert stats["peak_equity"] == pytest.approx(12_000.0)
        assert stats["current_equity"] == pytest.approx(9_200.0 - 2_000.0 + 10_000.0)
        # 10,000 + 2,000 - 4,800 = 7,200
        assert stats["current_equity"] == pytest.approx(7_200.0)
        assert stats["trading_halted"] is True
        assert stats["trade_count"] == 2


# ---------------------------------------------------------------------------
# Persistence round-trip
# ---------------------------------------------------------------------------

class TestPersistence:

    def test_save_creates_file(self, tmp_path):
        m = _make_monitor(data_dir=str(tmp_path))
        m.record_trade_result("AAPL", realized_pnl=500.0)
        m.save_state()
        assert (tmp_path / "equity_history.jsonl").exists()

    def test_save_and_load_round_trip(self, tmp_path):
        m = _make_monitor(starting_capital=10_000.0, data_dir=str(tmp_path))
        m.record_trade_result("AAPL", realized_pnl=1_000.0)  # equity = 11,000
        m.record_trade_result("TSLA", realized_pnl=-500.0)   # equity = 10,500
        m.save_state()

        # Create fresh monitor and load
        m2 = _make_monitor(starting_capital=10_000.0, data_dir=str(tmp_path))
        m2.load_state()

        assert m2._current_equity == pytest.approx(m._current_equity)
        assert m2._peak_equity == pytest.approx(m._peak_equity)
        assert m2._realized_pnl == pytest.approx(m._realized_pnl)
        assert m2._trade_count == m._trade_count
        assert m2._trading_halted == m._trading_halted
        assert len(m2._equity_history) == len(m._equity_history)

    def test_load_preserves_halted_state(self, tmp_path):
        m = _make_monitor(starting_capital=10_000.0, max_drawdown_pct=0.40, data_dir=str(tmp_path))
        m.record_trade_result("AAPL", realized_pnl=-4_000.0)  # halted
        assert m.is_trading_halted()
        m.save_state()

        m2 = _make_monitor(starting_capital=10_000.0, max_drawdown_pct=0.40, data_dir=str(tmp_path))
        m2.load_state()
        assert m2.is_trading_halted()

    def test_save_is_atomic_no_partial_file(self, tmp_path):
        """Save uses os.replace — if it fails partway, original is untouched."""
        m = _make_monitor(starting_capital=10_000.0, data_dir=str(tmp_path))
        m.record_trade_result("AAPL", realized_pnl=100.0)
        m.save_state()

        # Record the file size after first save
        first_path = tmp_path / "equity_history.jsonl"
        first_size = first_path.stat().st_size

        # Simulate a failed second save (no error raised, original intact)
        with patch("mae_core.market.intelligence.drawdown_monitor.os.replace", side_effect=OSError("disk full")):
            m.record_trade_result("TSLA", realized_pnl=200.0)
            m.save_state()  # should NOT raise

        # Original file is unchanged (os.replace failed before clobbering)
        assert first_path.stat().st_size == first_size

    def test_load_graceful_missing_file(self, tmp_path):
        """load_state with no file should not raise and leaves state at defaults."""
        m = _make_monitor(starting_capital=10_000.0, data_dir=str(tmp_path))
        m.load_state()
        assert m._current_equity == pytest.approx(10_000.0)
        assert m._peak_equity == pytest.approx(10_000.0)

    def test_load_graceful_malformed_file(self, tmp_path):
        """Malformed JSONL lines are skipped — no crash."""
        bad_path = tmp_path / "equity_history.jsonl"
        bad_path.write_text('{"__meta__": true, "peak_equity": 10500, "current_equity": 10500, "realized_pnl": 500, "trade_count": 1, "trading_halted": false, "warning_active": false}\nnot-json-at-all\n{"ts": 1000, "equity": 10500, "pnl_delta": 500, "ticker": "AAPL", "direction": "buy"}\n')
        m = _make_monitor(starting_capital=10_000.0, data_dir=str(tmp_path))
        m.load_state()  # should not raise
        assert m._current_equity == pytest.approx(10_500.0)
        assert len(m._equity_history) == 1

    def test_explicit_path_override(self, tmp_path):
        custom_path = tmp_path / "custom_equity.jsonl"
        m = _make_monitor(starting_capital=10_000.0)
        m.record_trade_result("AAPL", realized_pnl=100.0)
        m.save_state(path=str(custom_path))
        assert custom_path.exists()

        m2 = _make_monitor(starting_capital=10_000.0)
        m2.load_state(path=str(custom_path))
        assert m2._current_equity == pytest.approx(m._current_equity)


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:

    def test_concurrent_record_trade_result(self):
        """100 concurrent threads each record a $1 profit — final equity must be correct."""
        m = _make_monitor(starting_capital=0.0, max_drawdown_pct=0.99)
        n_threads = 100
        n_per_thread = 10

        def worker():
            for _ in range(n_per_thread):
                m.record_trade_result("AAPL", realized_pnl=1.0)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected_equity = float(n_threads * n_per_thread)
        assert m._current_equity == pytest.approx(expected_equity)
        assert m._trade_count == n_threads * n_per_thread
        assert len(m._equity_history) <= 10_000  # may be capped by maxlen

    def test_concurrent_reads_during_writes(self):
        """get_current_drawdown and is_trading_halted must be safe to call concurrently."""
        m = _make_monitor(starting_capital=10_000.0, max_drawdown_pct=0.40)
        errors = []

        def writer():
            for _ in range(50):
                try:
                    m.record_trade_result("AAPL", realized_pnl=-100.0)
                except Exception as exc:
                    errors.append(exc)

        def reader():
            for _ in range(50):
                try:
                    _ = m.get_current_drawdown()
                    _ = m.is_trading_halted()
                    _ = m.get_statistics()
                except Exception as exc:
                    errors.append(exc)

        threads = (
            [threading.Thread(target=writer) for _ in range(4)] +
            [threading.Thread(target=reader) for _ in range(4)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread safety violation: {errors}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_zero_pnl_trade(self):
        """A trade with zero P&L should not change equity or peak."""
        m = _make_monitor(starting_capital=10_000.0)
        m.record_trade_result("AAPL", realized_pnl=0.0)
        assert m._current_equity == pytest.approx(10_000.0)
        assert m._peak_equity == pytest.approx(10_000.0)
        assert m.get_current_drawdown() == pytest.approx(0.0)

    def test_publish_exception_does_not_propagate(self):
        """EventBus publish raising an exception must not propagate."""
        bus = MagicMock()
        bus.publish.side_effect = RuntimeError("bus exploded")
        m = _make_monitor(event_bus=bus, starting_capital=10_000.0, max_drawdown_pct=0.40)
        # Should not raise
        m.record_trade_result("AAPL", realized_pnl=-4_000.0)
        assert m.is_trading_halted()

    def test_drawdown_calculation_with_fractional_pnl(self):
        m = _make_monitor(starting_capital=10_000.0, max_drawdown_pct=0.40)
        m.record_trade_result("AAPL", realized_pnl=-1_500.55)
        expected_dd = 1_500.55 / 10_000.0
        assert m.get_current_drawdown() == pytest.approx(expected_dd)

    def test_recovery_above_peak_sets_new_peak(self):
        """After a drawdown, recovery to a new all-time high should update peak."""
        m = _make_monitor(starting_capital=10_000.0)
        m.record_trade_result("AAPL", realized_pnl=-2_000.0)  # equity = 8,000
        m.record_trade_result("AAPL", realized_pnl=5_000.0)   # equity = 13,000 — new peak
        assert m._peak_equity == pytest.approx(13_000.0)
        assert m.get_current_drawdown() == pytest.approx(0.0)
