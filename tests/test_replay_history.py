"""Tests for replay_history.py — historical convergence replay harness."""

import json
import os
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# Import the replay module
from replay_history import (
    SimulatedDateTime,
    _extract_symbols,
    _get_date_range,
    _group_into_windows,
    _load_day_signals,
    _parse_timestamp,
    _primary_symbol,
    phase_grade,
    phase_report,
    phase_replay,
    SUCCESS_THRESHOLD_PCT,
)


# ---------------------------------------------------------------
# SimulatedDateTime
# ---------------------------------------------------------------

class TestSimulatedDateTime:
    """Verify the time simulation wrapper."""

    def test_now_returns_frozen_time(self):
        frozen = datetime(2025, 6, 15, 14, 30, 0)
        sim = SimulatedDateTime(frozen)
        assert sim.now() == frozen

    def test_now_with_tz_returns_frozen_with_tz(self):
        from datetime import timezone
        frozen = datetime(2025, 6, 15, 14, 30, 0)
        sim = SimulatedDateTime(frozen)
        result = sim.now(tz=timezone.utc)
        assert result.year == 2025
        assert result.month == 6
        assert result.tzinfo == timezone.utc

    def test_now_is_repeatable(self):
        """Calling now() multiple times returns the same frozen time."""
        frozen = datetime(2025, 3, 1, 9, 0, 0)
        sim = SimulatedDateTime(frozen)
        assert sim.now() == sim.now() == frozen

    def test_delegates_fromisoformat(self):
        """Non-now attributes delegate to real datetime class."""
        frozen = datetime(2025, 1, 1)
        sim = SimulatedDateTime(frozen)
        result = sim.fromisoformat("2025-06-15T10:00:00")
        assert result == datetime(2025, 6, 15, 10, 0, 0)

    def test_delegates_min(self):
        """datetime.min should still work."""
        frozen = datetime(2025, 1, 1)
        sim = SimulatedDateTime(frozen)
        assert sim.min == datetime.min

    def test_constructor_delegation(self):
        """SimulatedDateTime(...)(year, month, day) creates real datetime."""
        frozen = datetime(2025, 1, 1)
        sim = SimulatedDateTime(frozen)
        result = sim(2025, 7, 4, 12, 0, 0)
        assert result == datetime(2025, 7, 4, 12, 0, 0)
        assert isinstance(result, datetime)


# ---------------------------------------------------------------
# Archive loading and parsing
# ---------------------------------------------------------------

class TestArchiveLoading:
    """Test signal archive loading and timestamp parsing."""

    def test_parse_timestamp_valid(self):
        result = _parse_timestamp("2025-06-15T14:30:00")
        assert result == datetime(2025, 6, 15, 14, 30, 0)

    def test_parse_timestamp_empty(self):
        assert _parse_timestamp("") == datetime.min

    def test_parse_timestamp_none(self):
        assert _parse_timestamp(None) == datetime.min

    def test_parse_timestamp_invalid(self):
        assert _parse_timestamp("not-a-date") == datetime.min

    def test_load_day_signals_from_file(self, tmp_path):
        """Loads valid JSONL records from an archive file."""
        day = date(2025, 6, 15)
        archive_file = tmp_path / f"{day.isoformat()}.jsonl"
        records = [
            {"signal_id": "s1", "source": "sec_form4", "symbol": "AAPL",
             "domain": "insider", "direction": "bullish", "strength": 0.8,
             "confidence": 0.7, "velocity": 0.0,
             "timestamp": "2025-06-15T10:00:00", "metadata": {}},
            {"signal_id": "s2", "source": "congressional", "symbol": "MSFT",
             "domain": "congress", "direction": "bullish", "strength": 0.6,
             "confidence": 0.5, "velocity": 0.1,
             "timestamp": "2025-06-15T11:00:00", "metadata": {}},
        ]
        archive_file.write_text(
            "\n".join(json.dumps(r) for r in records) + "\n"
        )

        loaded = _load_day_signals(day, signals_dir=tmp_path)
        assert len(loaded) == 2
        assert loaded[0]["signal_id"] == "s1"
        assert loaded[1]["signal_id"] == "s2"

    def test_load_day_signals_missing_file(self, tmp_path):
        """Returns empty list for non-existent archive file."""
        loaded = _load_day_signals(date(2025, 1, 1), signals_dir=tmp_path)
        assert loaded == []

    def test_load_day_signals_skips_bad_json(self, tmp_path):
        """Skips malformed JSON lines without crashing."""
        day = date(2025, 6, 15)
        archive_file = tmp_path / f"{day.isoformat()}.jsonl"
        archive_file.write_text(
            '{"signal_id": "s1", "source": "test"}\n'
            'not valid json\n'
            '{"signal_id": "s2", "source": "test2"}\n'
        )
        loaded = _load_day_signals(day, signals_dir=tmp_path)
        assert len(loaded) == 2


# ---------------------------------------------------------------
# Window grouping
# ---------------------------------------------------------------

class TestWindowGrouping:
    """Test signal grouping into time windows."""

    def test_empty_input(self):
        assert _group_into_windows([]) == []

    def test_single_signal(self):
        signals = [{"timestamp": "2025-06-15T10:00:00"}]
        windows = _group_into_windows(signals)
        assert len(windows) == 1
        assert len(windows[0]) == 1

    def test_signals_within_window(self):
        """Signals within 4 hours stay in one window."""
        signals = [
            {"timestamp": "2025-06-15T10:00:00"},
            {"timestamp": "2025-06-15T11:00:00"},
            {"timestamp": "2025-06-15T13:30:00"},
        ]
        windows = _group_into_windows(signals, window_hours=4.0)
        assert len(windows) == 1
        assert len(windows[0]) == 3

    def test_signals_across_windows(self):
        """Signals separated by more than window_hours split into groups."""
        signals = [
            {"timestamp": "2025-06-15T08:00:00"},
            {"timestamp": "2025-06-15T09:00:00"},
            {"timestamp": "2025-06-15T14:00:00"},  # 6h gap from start
            {"timestamp": "2025-06-15T15:00:00"},
        ]
        windows = _group_into_windows(signals, window_hours=4.0)
        assert len(windows) == 2
        assert len(windows[0]) == 2
        assert len(windows[1]) == 2


# ---------------------------------------------------------------
# Symbol extraction
# ---------------------------------------------------------------

class TestSymbolExtraction:
    """Test extracting symbols from convergence alerts."""

    def _make_alert_with_signals(self, symbols):
        """Helper: create a mock alert with signals containing given symbols."""
        alert = SimpleNamespace()
        alert.signals = []
        for sym in symbols:
            sig = SimpleNamespace(metadata={"symbol": sym} if sym else {})
            alert.signals.append(sig)
        return alert

    def test_extract_symbols_from_metadata(self):
        alert = self._make_alert_with_signals(["AAPL", "MSFT", "AAPL"])
        assert _extract_symbols(alert) == ["AAPL", "MSFT", "AAPL"]

    def test_extract_symbols_skips_empty(self):
        alert = self._make_alert_with_signals(["AAPL", None, "MSFT"])
        assert _extract_symbols(alert) == ["AAPL", "MSFT"]

    def test_primary_symbol_most_mentioned(self):
        alert = self._make_alert_with_signals(["AAPL", "MSFT", "AAPL", "GOOGL"])
        assert _primary_symbol(alert) == "AAPL"

    def test_primary_symbol_none_when_no_symbols(self):
        alert = self._make_alert_with_signals([None, None])
        assert _primary_symbol(alert) is None


# ---------------------------------------------------------------
# Convergence under simulated time
# ---------------------------------------------------------------

class TestConvergenceReplay:
    """Test that convergence fires correctly under simulated time."""

    def test_three_domains_fire_alert(self):
        """3+ domains with same direction should produce a convergence alert."""
        import mae_core.market.intelligence.convergence_alerter as ca_module
        from mae_core.market.intelligence.convergence_alerter import ConvergenceAlerter

        alerter = ConvergenceAlerter(min_domains=3)
        original_dt = ca_module.datetime

        sim_time = datetime(2025, 6, 15, 14, 0, 0)
        ca_module.datetime = SimulatedDateTime(sim_time)

        try:
            # Feed 3 different domains — all bullish
            alerter.record_signal(
                "s1", 0.8, "insider", direction="bullish",
                confidence=0.7, timestamp=sim_time - timedelta(hours=1),
                metadata={"symbol": "AAPL"}, source="sec_form4",
            )
            alerter.record_signal(
                "s2", 0.7, "congress", direction="bullish",
                confidence=0.6, timestamp=sim_time - timedelta(minutes=30),
                metadata={"symbol": "AAPL"}, source="congressional",
            )
            alerter.record_signal(
                "s3", 0.75, "crypto", direction="bullish",
                confidence=0.65, timestamp=sim_time,
                metadata={"symbol": "BTC"}, source="crypto_coingecko",
            )

            alerts = alerter.check_convergence()
            assert len(alerts) >= 1
            alert = alerts[0]
            assert alert.direction == "bullish"
            assert len(alert.domains_converging) >= 3
        finally:
            ca_module.datetime = original_dt

    def test_two_domains_no_alert(self):
        """Fewer than min_domains should NOT produce an alert."""
        import mae_core.market.intelligence.convergence_alerter as ca_module
        from mae_core.market.intelligence.convergence_alerter import ConvergenceAlerter

        alerter = ConvergenceAlerter(min_domains=3)
        original_dt = ca_module.datetime

        sim_time = datetime(2025, 6, 15, 14, 0, 0)
        ca_module.datetime = SimulatedDateTime(sim_time)

        try:
            alerter.record_signal(
                "s1", 0.8, "insider", direction="bullish",
                timestamp=sim_time - timedelta(hours=1),
            )
            alerter.record_signal(
                "s2", 0.7, "congress", direction="bullish",
                timestamp=sim_time,
            )

            alerts = alerter.check_convergence()
            assert len(alerts) == 0
        finally:
            ca_module.datetime = original_dt

    def test_monkeypatch_restores_cleanly(self):
        """Datetime module is restored after replay context."""
        import mae_core.market.intelligence.convergence_alerter as ca_module

        original = ca_module.datetime
        assert original is datetime  # Should be the real datetime class

        sim = SimulatedDateTime(datetime(2025, 1, 1))
        ca_module.datetime = sim
        assert ca_module.datetime is sim

        # Restore
        ca_module.datetime = original
        assert ca_module.datetime is datetime

    def test_freshness_works_under_simulated_time(self):
        """Signals close to simulated 'now' get high freshness, not 0.3 floor."""
        import mae_core.market.intelligence.convergence_alerter as ca_module
        from mae_core.market.intelligence.convergence_alerter import ConvergenceAlerter

        alerter = ConvergenceAlerter(min_domains=3)
        original_dt = ca_module.datetime

        sim_time = datetime(2025, 6, 15, 14, 0, 0)
        ca_module.datetime = SimulatedDateTime(sim_time)

        try:
            # Signal from 1 hour ago (under simulated time)
            sig_time = sim_time - timedelta(hours=1)
            alerter.record_signal(
                "s1", 0.8, "insider", direction="bullish",
                timestamp=sig_time,
            )

            # Freshness should be high (~0.88) not the 0.3 floor
            freshness = alerter._compute_freshness(
                alerter.signals["insider"][0], "insider"
            )
            assert freshness > 0.5, f"Freshness {freshness} should be > 0.5 for 1h-old signal"
        finally:
            ca_module.datetime = original_dt

    def test_dedup_reset_between_days(self):
        """Clearing _last_alert_times allows alerts on consecutive days."""
        import mae_core.market.intelligence.convergence_alerter as ca_module
        from mae_core.market.intelligence.convergence_alerter import ConvergenceAlerter

        alerter = ConvergenceAlerter(min_domains=3)
        original_dt = ca_module.datetime

        try:
            # Day 1 — produce an alert
            day1_time = datetime(2025, 6, 15, 14, 0, 0)
            ca_module.datetime = SimulatedDateTime(day1_time)

            for i, domain in enumerate(["insider", "congress", "crypto"]):
                alerter.record_signal(
                    f"d1_{i}", 0.8, domain, direction="bullish",
                    timestamp=day1_time - timedelta(minutes=30 * i),
                )

            alerts_day1 = alerter.check_convergence()
            assert len(alerts_day1) >= 1

            # Reset dedup (as the replay harness does between days)
            alerter._last_alert_times.clear()

            # Day 2 — should also produce alert (new signals + reset dedup)
            day2_time = datetime(2025, 6, 16, 14, 0, 0)
            ca_module.datetime = SimulatedDateTime(day2_time)

            for i, domain in enumerate(["insider", "congress", "technical"]):
                alerter.record_signal(
                    f"d2_{i}", 0.8, domain, direction="bearish",
                    timestamp=day2_time - timedelta(minutes=30 * i),
                )

            alerts_day2 = alerter.check_convergence()
            assert len(alerts_day2) >= 1
        finally:
            ca_module.datetime = original_dt


# ---------------------------------------------------------------
# Grading logic
# ---------------------------------------------------------------

class TestGrading:
    """Test alert grading against price movement."""

    def _make_alert_dict(self, direction, symbol="AAPL"):
        return {
            "alert_id": "CONV-20250615-0001",
            "timestamp": "2025-06-15T14:00:00",
            "direction": direction,
            "strength": 0.8,
            "confidence": 0.7,
            "domains": ["insider", "congress", "crypto"],
            "primary_symbol": symbol,
        }

    @patch("replay_history.RESULTS_PATH")
    def test_bullish_success(self, mock_path):
        """Bullish alert + price up >= 5% = success."""
        alert = self._make_alert_dict("bullish")

        mock_pf = MagicMock()
        mock_pf.get_historical_price.side_effect = [
            SimpleNamespace(close=100.0),   # entry
            SimpleNamespace(close=106.0),   # exit (+6%)
        ]

        with patch("replay_history.time_module.sleep"):
            with patch("mae_core.market.apis.price_fetcher.PriceFetcher", return_value=mock_pf):
                result = phase_grade([alert], outcome_window_days=14)

        assert result[0]["grade"] == "success"
        assert result[0]["pct_change"] == 6.0

    @patch("replay_history.RESULTS_PATH")
    def test_bullish_failure(self, mock_path):
        """Bullish alert + price down = failure."""
        alert = self._make_alert_dict("bullish")

        mock_pf = MagicMock()
        mock_pf.get_historical_price.side_effect = [
            SimpleNamespace(close=100.0),
            SimpleNamespace(close=97.0),    # -3%
        ]

        with patch("replay_history.time_module.sleep"):
            with patch("mae_core.market.apis.price_fetcher.PriceFetcher", return_value=mock_pf):
                result = phase_grade([alert], outcome_window_days=14)

        assert result[0]["grade"] == "failure"
        assert result[0]["pct_change"] == -3.0

    @patch("replay_history.RESULTS_PATH")
    def test_bearish_success(self, mock_path):
        """Bearish alert + price down >= 5% = success."""
        alert = self._make_alert_dict("bearish")

        mock_pf = MagicMock()
        mock_pf.get_historical_price.side_effect = [
            SimpleNamespace(close=100.0),
            SimpleNamespace(close=93.0),    # -7%
        ]

        with patch("replay_history.time_module.sleep"):
            with patch("mae_core.market.apis.price_fetcher.PriceFetcher", return_value=mock_pf):
                result = phase_grade([alert], outcome_window_days=14)

        assert result[0]["grade"] == "success"
        assert result[0]["pct_change"] == -7.0

    def test_no_symbol_ungraded(self):
        """Alert without a primary symbol is marked ungraded."""
        alert = self._make_alert_dict("bullish", symbol=None)
        alert["primary_symbol"] = None

        result = phase_grade([alert])
        assert result[0]["grade"] == "ungraded"
        assert result[0]["grade_reason"] == "no_symbol"

    @patch("replay_history.RESULTS_PATH")
    def test_no_price_data_ungraded(self, mock_path):
        """Alert where price lookup fails is marked ungraded."""
        alert = self._make_alert_dict("bullish")

        mock_pf = MagicMock()
        mock_pf.get_historical_price.return_value = None

        with patch("replay_history.time_module.sleep"):
            with patch("mae_core.market.apis.price_fetcher.PriceFetcher", return_value=mock_pf):
                result = phase_grade([alert], outcome_window_days=14)

        assert result[0]["grade"] == "ungraded"
        assert result[0]["grade_reason"] == "no_price_data"


# ---------------------------------------------------------------
# Report
# ---------------------------------------------------------------

class TestReport:
    """Test report generation."""

    def test_report_has_expected_keys(self):
        alerts = [
            {
                "alert_id": "CONV-001", "timestamp": "2025-06-15T14:00:00",
                "direction": "bullish", "strength": 0.8, "confidence": 0.7,
                "domains": ["insider", "congress", "crypto"],
                "grade": "success", "pct_change": 6.0,
                "primary_symbol": "AAPL",
            },
            {
                "alert_id": "CONV-002", "timestamp": "2025-07-01T10:00:00",
                "direction": "bearish", "strength": 0.6, "confidence": 0.5,
                "domains": ["insider", "technical", "volume"],
                "grade": "failure", "pct_change": 2.0,
                "primary_symbol": "MSFT",
            },
        ]

        report = phase_report(alerts)

        assert "total_alerts" in report
        assert "graded" in report
        assert "win_rate_pct" in report
        assert "domain_combos" in report
        assert "top_10_alerts" in report
        assert "monthly_distribution" in report
        assert report["total_alerts"] == 2
        assert report["graded"] == 2
        assert report["successes"] == 1
        assert report["failures"] == 1
        assert report["win_rate_pct"] == 50.0

    def test_report_empty_alerts(self):
        report = phase_report([])
        assert report["total_alerts"] == 0
        assert report["graded"] == 0
        assert report["win_rate_pct"] == 0

    def test_report_direction_breakdown(self):
        alerts = [
            {"direction": "bullish", "grade": "success", "strength": 0.7,
             "confidence": 0.6, "domains": ["a", "b", "c"],
             "timestamp": "2025-06-15T10:00:00"},
            {"direction": "bullish", "grade": "failure", "strength": 0.5,
             "confidence": 0.4, "domains": ["a", "b", "c"],
             "timestamp": "2025-06-16T10:00:00"},
            {"direction": "bearish", "grade": "success", "strength": 0.8,
             "confidence": 0.7, "domains": ["x", "y", "z"],
             "timestamp": "2025-06-17T10:00:00"},
        ]
        report = phase_report(alerts)
        assert report["win_rate_bullish_pct"] == 50.0
        assert report["win_rate_bearish_pct"] == 100.0
        assert report["count_bullish"] == 2
        assert report["count_bearish"] == 1


# ---------------------------------------------------------------
# Date range discovery
# ---------------------------------------------------------------

class TestDateRange:
    """Test archive date range discovery."""

    def test_get_date_range(self, tmp_path):
        (tmp_path / "2025-01-15.jsonl").touch()
        (tmp_path / "2025-06-01.jsonl").touch()
        (tmp_path / "2025-12-31.jsonl").touch()
        (tmp_path / "not-a-date.jsonl").touch()

        start, end = _get_date_range(signals_dir=tmp_path)
        assert start == date(2025, 1, 15)
        assert end == date(2025, 12, 31)

    def test_get_date_range_empty(self, tmp_path):
        start, end = _get_date_range(signals_dir=tmp_path)
        assert start is None
        assert end is None


# ---------------------------------------------------------------
# Full replay integration (small scale)
# ---------------------------------------------------------------

class TestReplayIntegration:
    """Integration test with a small synthetic archive."""

    def test_replay_with_three_domain_day(self, tmp_path):
        """A day with 3+ domains aligned should produce at least one alert."""
        day = date(2025, 6, 15)
        base_time = datetime(2025, 6, 15, 10, 0, 0)

        records = [
            {"signal_id": "s1", "source": "sec_form4", "symbol": "AAPL",
             "domain": "insider", "direction": "bullish", "strength": 0.8,
             "confidence": 0.7, "velocity": 0.0,
             "timestamp": (base_time).isoformat(), "metadata": {}},
            {"signal_id": "s2", "source": "congressional", "symbol": "AAPL",
             "domain": "congress", "direction": "bullish", "strength": 0.7,
             "confidence": 0.6, "velocity": 0.1,
             "timestamp": (base_time + timedelta(minutes=30)).isoformat(),
             "metadata": {}},
            {"signal_id": "s3", "source": "hiring_tracker", "symbol": "AAPL",
             "domain": "contracts", "direction": "bullish", "strength": 0.65,
             "confidence": 0.55, "velocity": 0.0,
             "timestamp": (base_time + timedelta(hours=1)).isoformat(),
             "metadata": {}},
        ]

        archive_file = tmp_path / f"{day.isoformat()}.jsonl"
        archive_file.write_text(
            "\n".join(json.dumps(r) for r in records) + "\n"
        )

        alerts = phase_replay(
            start_date=day,
            end_date=day,
            min_domains=3,
            use_thompson=False,
            signals_dir=tmp_path,
        )

        assert len(alerts) >= 1
        assert alerts[0]["direction"] == "bullish"

    def test_replay_empty_day_no_alerts(self, tmp_path):
        """A day with no signals produces no alerts."""
        day = date(2025, 6, 15)
        # No archive file for this day

        alerts = phase_replay(
            start_date=day,
            end_date=day,
            min_domains=3,
            use_thompson=False,
            signals_dir=tmp_path,
        )

        assert len(alerts) == 0
