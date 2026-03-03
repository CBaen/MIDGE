"""Tests for WP-C: Daemon Mode components.

Covers DaemonMonitor (heartbeat writing, statistics) and
_daemon_persistence_flush (flush orchestration, fault isolation).

Does NOT test run_daemon() itself — that requires full organism creation.
"""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mae_core.market.daemon_monitor import DaemonMonitor
from main import _daemon_persistence_flush


# ---------------------------------------------------------------------------
# DaemonMonitor tests
# ---------------------------------------------------------------------------


class TestDaemonMonitorHeartbeat:
    """Heartbeat write cadence and content."""

    def test_heartbeat_write_on_cadence(self, tmp_path):
        """After exactly heartbeat_cadence steps, the file must exist."""
        hb = tmp_path / "heartbeat.json"
        monitor = DaemonMonitor(heartbeat_path=hb, heartbeat_cadence=100)
        for step in range(1, 101):
            monitor.record_step(step, {})
        assert hb.exists(), "Heartbeat file should be written at step 100"

    def test_heartbeat_not_written_off_cadence(self, tmp_path):
        """Fewer than heartbeat_cadence steps must NOT produce a file."""
        hb = tmp_path / "heartbeat.json"
        monitor = DaemonMonitor(heartbeat_path=hb, heartbeat_cadence=100)
        for step in range(1, 100):
            monitor.record_step(step, {})
        assert not hb.exists(), "Heartbeat file must not appear before cadence is reached"

    def test_heartbeat_json_structure(self, tmp_path):
        """Written file must contain all ten expected fields."""
        hb = tmp_path / "heartbeat.json"
        monitor = DaemonMonitor(heartbeat_path=hb, heartbeat_cadence=10)
        for step in range(1, 11):
            monitor.record_step(step, {})
        data = json.loads(hb.read_text(encoding="utf-8"))
        expected_keys = {
            "pid",
            "started_at",
            "uptime_seconds",
            "current_step",
            "total_daemon_steps",
            "daemon_round",
            "step_rate_per_second",
            "active_agents",
            "last_fetch_source",
            "heartbeat_at",
        }
        assert expected_keys == set(data.keys()), f"Missing or extra keys: {set(data.keys()) ^ expected_keys}"

    def test_heartbeat_atomic_write(self, tmp_path):
        """No .tmp file should remain after the heartbeat write completes."""
        hb = tmp_path / "heartbeat.json"
        monitor = DaemonMonitor(heartbeat_path=hb, heartbeat_cadence=5)
        for step in range(1, 6):
            monitor.record_step(step, {})
        tmp_file = hb.with_suffix(".json.tmp")
        assert not tmp_file.exists(), ".tmp file must not persist after atomic rename"
        assert hb.exists(), "Final heartbeat file must exist after write"

    def test_custom_heartbeat_path(self, tmp_path):
        """Passing a custom path writes the heartbeat there, not the default."""
        custom = tmp_path / "subdir" / "custom_hb.json"
        monitor = DaemonMonitor(heartbeat_path=custom, heartbeat_cadence=10)
        for step in range(1, 11):
            monitor.record_step(step, {})
        assert custom.exists(), "Heartbeat should be written at the custom path"

    def test_custom_cadence(self, tmp_path):
        """A cadence of 50 should trigger a write after exactly 50 steps."""
        hb = tmp_path / "heartbeat.json"
        monitor = DaemonMonitor(heartbeat_path=hb, heartbeat_cadence=50)
        # 49 steps — no file yet
        for step in range(1, 50):
            monitor.record_step(step, {})
        assert not hb.exists()
        # 50th step — file appears
        monitor.record_step(50, {})
        assert hb.exists()

    def test_begin_round_updates_round_num(self, tmp_path):
        """daemon_round in the heartbeat must reflect the last begin_round call."""
        hb = tmp_path / "heartbeat.json"
        monitor = DaemonMonitor(heartbeat_path=hb, heartbeat_cadence=10)
        monitor.begin_round(7)
        for step in range(1, 11):
            monitor.record_step(step, {})
        data = json.loads(hb.read_text(encoding="utf-8"))
        assert data["daemon_round"] == 7

    def test_active_agents_counted_from_systems(self, tmp_path):
        """active_agents in the heartbeat must equal the length of systems['agents']."""
        hb = tmp_path / "heartbeat.json"
        monitor = DaemonMonitor(heartbeat_path=hb, heartbeat_cadence=10)
        fake_agents = [object(), object(), object()]
        systems = {"agents": fake_agents}
        for step in range(1, 11):
            monitor.record_step(step, systems)
        data = json.loads(hb.read_text(encoding="utf-8"))
        assert data["active_agents"] == 3

    def test_heartbeat_pid_matches_process(self, tmp_path):
        """pid field in heartbeat must match the current process PID."""
        import os
        hb = tmp_path / "heartbeat.json"
        monitor = DaemonMonitor(heartbeat_path=hb, heartbeat_cadence=10)
        for step in range(1, 11):
            monitor.record_step(step, {})
        data = json.loads(hb.read_text(encoding="utf-8"))
        assert data["pid"] == os.getpid()


class TestDaemonMonitorStatistics:
    """get_uptime() and get_statistics() correctness."""

    def test_uptime_increases(self):
        """get_uptime() must return a larger value after time has passed."""
        monitor = DaemonMonitor()
        t0 = monitor.get_uptime()
        time.sleep(0.05)
        t1 = monitor.get_uptime()
        assert t1 > t0, "Uptime should increase monotonically"

    def test_uptime_is_non_negative(self):
        """get_uptime() must never return a negative value."""
        monitor = DaemonMonitor()
        assert monitor.get_uptime() >= 0.0

    def test_statistics_dict_has_all_keys(self):
        """get_statistics() must contain the five documented keys."""
        monitor = DaemonMonitor()
        stats = monitor.get_statistics()
        expected = {"pid", "uptime_seconds", "total_daemon_steps", "daemon_round", "heartbeat_cadence"}
        assert expected.issubset(set(stats.keys())), f"Missing keys: {expected - set(stats.keys())}"

    def test_statistics_reflects_begin_round(self):
        """daemon_round in get_statistics() must match begin_round argument."""
        monitor = DaemonMonitor()
        monitor.begin_round(42)
        stats = monitor.get_statistics()
        assert stats["daemon_round"] == 42

    def test_statistics_counts_total_steps(self, tmp_path):
        """total_daemon_steps must equal the number of record_step calls made."""
        hb = tmp_path / "heartbeat.json"
        monitor = DaemonMonitor(heartbeat_path=hb, heartbeat_cadence=1000)
        for step in range(1, 26):
            monitor.record_step(step, {})
        stats = monitor.get_statistics()
        assert stats["total_daemon_steps"] == 25

    def test_step_rate_calculation(self, tmp_path):
        """step_rate_per_second in the heartbeat must be a non-negative number."""
        hb = tmp_path / "heartbeat.json"
        monitor = DaemonMonitor(heartbeat_path=hb, heartbeat_cadence=10)
        for step in range(1, 11):
            monitor.record_step(step, {})
        data = json.loads(hb.read_text(encoding="utf-8"))
        rate = data["step_rate_per_second"]
        assert isinstance(rate, (int, float))
        assert rate >= 0.0, "Step rate must be non-negative"

    def test_statistics_heartbeat_cadence_matches_constructor(self):
        """heartbeat_cadence in get_statistics() must equal the value passed to constructor."""
        monitor = DaemonMonitor(heartbeat_cadence=250)
        assert monitor.get_statistics()["heartbeat_cadence"] == 250


# ---------------------------------------------------------------------------
# _daemon_persistence_flush tests
# ---------------------------------------------------------------------------


class TestDaemonPersistenceFlush:
    """_daemon_persistence_flush: orchestration, fault isolation, and edge cases."""

    def test_flush_with_empty_systems(self):
        """Empty systems dict must not raise."""
        _daemon_persistence_flush({})  # Should not raise

    def test_flush_with_none_values(self):
        """Systems dict containing None values must not raise."""
        systems = {
            "hypothesis_engine": None,
            "hypothesis_generator": None,
            "convergence_alerter": None,
            "somatic_anticipation": None,
            "deception_detector": None,
        }
        _daemon_persistence_flush(systems)  # Should not raise

    def test_flush_calls_hypothesis_engine_saver(self):
        """hypothesis_engine._save_retirement_window() must be called on flush."""
        mock_engine = MagicMock()
        _daemon_persistence_flush({"hypothesis_engine": mock_engine})
        mock_engine._save_retirement_window.assert_called_once()

    def test_flush_calls_hypothesis_generator_saver(self):
        """hypothesis_generator.save_pair_outcomes() must be called on flush."""
        mock_gen = MagicMock()
        _daemon_persistence_flush({"hypothesis_generator": mock_gen})
        mock_gen.save_pair_outcomes.assert_called_once()

    def test_flush_calls_convergence_alerter_saver(self):
        """convergence_alerter.save_signal_buffer() must be called when method exists."""
        mock_alerter = MagicMock()
        mock_alerter.save_signal_buffer = MagicMock()
        _daemon_persistence_flush({"convergence_alerter": mock_alerter})
        mock_alerter.save_signal_buffer.assert_called_once()

    def test_flush_calls_somatic_anticipation_saver(self):
        """somatic_anticipation.save_state() must be called when method exists."""
        mock_sa = MagicMock()
        mock_sa.save_state = MagicMock()
        _daemon_persistence_flush({"somatic_anticipation": mock_sa})
        mock_sa.save_state.assert_called_once()

    def test_flush_calls_deception_detector_saver(self):
        """deception_detector.save_state() must be called when method exists."""
        mock_dd = MagicMock()
        mock_dd.save_state = MagicMock()
        _daemon_persistence_flush({"deception_detector": mock_dd})
        mock_dd.save_state.assert_called_once()

    def test_flush_continues_on_individual_failure(self):
        """If one saver raises, subsequent savers must still be called."""
        mock_engine = MagicMock()
        mock_engine._save_retirement_window.side_effect = RuntimeError("disk full")

        mock_gen = MagicMock()

        _daemon_persistence_flush({
            "hypothesis_engine": mock_engine,
            "hypothesis_generator": mock_gen,
        })

        # hypothesis_generator.save_pair_outcomes must still be called despite the engine failure
        mock_gen.save_pair_outcomes.assert_called_once()

    def test_flush_skips_save_state_when_method_missing(self):
        """Systems without save_state must not cause AttributeError."""
        # Object has no save_state attribute — flush must silently skip it
        class NoSaveState:
            pass

        systems = {
            "somatic_anticipation": NoSaveState(),
            "deception_detector": NoSaveState(),
        }
        _daemon_persistence_flush(systems)  # Should not raise

    def test_flush_calls_learning_config_save_snapshot(self):
        """save_snapshot() from learning_config must be attempted during flush."""
        with patch("mae_core.market.intelligence.learning_config.save_snapshot") as mock_snap:
            _daemon_persistence_flush({})
            mock_snap.assert_called_once()
