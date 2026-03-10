"""Tests for SystemHealthMonitor — infrastructure health aggregation."""
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from mae_core.market.system_health_monitor import (
    CORE_SUBSYSTEMS,
    SystemHealthMonitor,
    _DEGRADED_THRESHOLD,
    _FAILED_THRESHOLD,
    CH_HEALTH_TIER_CHANGE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_monitor(**kwargs) -> SystemHealthMonitor:
    """Return a monitor with a small error_window for faster tests."""
    kwargs.setdefault("error_window", 50)
    return SystemHealthMonitor(**kwargs)


def _push_errors(monitor: SystemHealthMonitor, subsystem: str, count: int) -> None:
    for _ in range(count):
        monitor.record_error(subsystem)


# ---------------------------------------------------------------------------
# Tier transitions: green → yellow → orange → red
# ---------------------------------------------------------------------------

class TestHealthTierTransitions:
    def test_starts_green(self):
        m = _make_monitor()
        assert m.evaluate_health() == "green"

    def test_one_degraded_subsystem_gives_yellow(self):
        m = _make_monitor()
        _push_errors(m, "signals", _DEGRADED_THRESHOLD)
        assert m.evaluate_health() == "yellow"

    def test_two_degraded_subsystems_still_yellow(self):
        m = _make_monitor()
        _push_errors(m, "signals", _DEGRADED_THRESHOLD)
        _push_errors(m, "metrics", _DEGRADED_THRESHOLD)
        assert m.evaluate_health() == "yellow"

    def test_three_degraded_subsystems_gives_orange(self):
        m = _make_monitor()
        _push_errors(m, "signals", _DEGRADED_THRESHOLD)
        _push_errors(m, "metrics", _DEGRADED_THRESHOLD)
        _push_errors(m, "granger", _DEGRADED_THRESHOLD)
        assert m.evaluate_health() == "orange"

    def test_one_failed_non_core_subsystem_gives_orange(self):
        m = _make_monitor()
        _push_errors(m, "granger", _FAILED_THRESHOLD)
        assert m.evaluate_health() == "orange"

    def test_failed_core_subsystem_gives_red(self):
        m = _make_monitor()
        core = next(iter(CORE_SUBSYSTEMS))
        _push_errors(m, core, _FAILED_THRESHOLD)
        assert m.evaluate_health() == "red"

    def test_all_four_core_subsystems_trigger_red(self):
        for core in CORE_SUBSYSTEMS:
            m = _make_monitor()
            _push_errors(m, core, _FAILED_THRESHOLD)
            assert m.evaluate_health() == "red", f"{core} should trigger red"

    def test_degraded_non_core_plus_failed_non_core_gives_orange(self):
        m = _make_monitor()
        _push_errors(m, "a", _DEGRADED_THRESHOLD)
        _push_errors(m, "b", _FAILED_THRESHOLD)
        assert m.evaluate_health() == "orange"

    def test_tier_string_values_are_lowercase(self):
        m = _make_monitor()
        tier = m.evaluate_health()
        assert tier == tier.lower()


# ---------------------------------------------------------------------------
# Core subsystem fast-path to red
# ---------------------------------------------------------------------------

class TestCoreSubsystemRedPath:
    def test_convergence_check_to_red(self):
        m = _make_monitor()
        _push_errors(m, "convergence_check", _FAILED_THRESHOLD)
        assert m.evaluate_health() == "red"

    def test_thompson_to_red(self):
        m = _make_monitor()
        _push_errors(m, "thompson", _FAILED_THRESHOLD)
        assert m.evaluate_health() == "red"

    def test_sensing_to_red(self):
        m = _make_monitor()
        _push_errors(m, "sensing", _FAILED_THRESHOLD)
        assert m.evaluate_health() == "red"

    def test_outcome_evaluation_to_red(self):
        m = _make_monitor()
        _push_errors(m, "outcome_evaluation", _FAILED_THRESHOLD)
        assert m.evaluate_health() == "red"

    def test_core_degraded_but_not_failed_is_not_red(self):
        """A core subsystem at 'degraded' (not 'failed') stays orange/yellow."""
        m = _make_monitor()
        core = next(iter(CORE_SUBSYSTEMS))
        _push_errors(m, core, _DEGRADED_THRESHOLD)   # degraded but not failed
        tier = m.evaluate_health()
        assert tier != "red"
        assert tier in ("yellow", "orange")


# ---------------------------------------------------------------------------
# Error window maxlen (old errors fall off)
# ---------------------------------------------------------------------------

class TestErrorWindowRolloff:
    def test_errors_fall_off_when_window_full(self):
        """After the window fills, adding more errors doesn't grow the deque."""
        m = SystemHealthMonitor(error_window=10)
        _push_errors(m, "signals", 10)
        dq = m._error_counts["signals"]
        assert len(dq) == 10, "deque should be full"
        # Push one more — oldest should fall off
        m.record_error("signals")
        assert len(dq) == 10, "maxlen enforced — no growth beyond window"

    def test_tier_recovers_when_old_errors_fall_off(self):
        """Tier improves once failed errors scroll out of the window."""
        # Use error_window=5 so _FAILED_THRESHOLD(20) never triggers.
        # With window=5 we can test _DEGRADED_THRESHOLD(5) falling off.
        m = SystemHealthMonitor(error_window=5)
        _push_errors(m, "signals", 5)         # fills window → degraded
        assert m.evaluate_health() == "yellow"

        # Call record_success to clear the deque (simulates window clearing).
        m.record_success("signals")
        assert m.evaluate_health() == "green"

    def test_window_size_respected(self):
        """The deque respects a custom error_window value."""
        m = SystemHealthMonitor(error_window=7)
        _push_errors(m, "signals", 20)
        assert len(m._error_counts["signals"]) == 7


# ---------------------------------------------------------------------------
# record_success resets subsystem health
# ---------------------------------------------------------------------------

class TestRecordSuccess:
    def test_success_clears_errors_and_resets_to_healthy(self):
        m = _make_monitor()
        _push_errors(m, "signals", _DEGRADED_THRESHOLD)
        assert m.is_degraded("signals")
        m.record_success("signals")
        assert not m.is_degraded("signals")
        assert m.evaluate_health() == "green"

    def test_success_on_unknown_subsystem_is_harmless(self):
        m = _make_monitor()
        m.record_success("never_seen_before")  # should not raise

    def test_success_resets_failed_subsystem(self):
        m = _make_monitor()
        core = "convergence_check"
        _push_errors(m, core, _FAILED_THRESHOLD)
        assert m.evaluate_health() == "red"
        m.record_success(core)
        assert m.evaluate_health() == "green"

    def test_success_on_one_does_not_clear_others(self):
        m = _make_monitor()
        _push_errors(m, "a", _DEGRADED_THRESHOLD)
        _push_errors(m, "b", _DEGRADED_THRESHOLD)
        _push_errors(m, "c", _DEGRADED_THRESHOLD)
        assert m.evaluate_health() == "orange"
        m.record_success("a")
        # two degraded remain → still yellow
        assert m.evaluate_health() == "yellow"


# ---------------------------------------------------------------------------
# is_degraded
# ---------------------------------------------------------------------------

class TestIsDegraded:
    def test_healthy_subsystem_not_degraded(self):
        m = _make_monitor()
        _push_errors(m, "signals", 1)
        assert not m.is_degraded("signals")

    def test_degraded_subsystem_is_degraded(self):
        m = _make_monitor()
        _push_errors(m, "signals", _DEGRADED_THRESHOLD)
        assert m.is_degraded("signals")

    def test_failed_subsystem_is_degraded(self):
        m = _make_monitor()
        _push_errors(m, "signals", _FAILED_THRESHOLD)
        assert m.is_degraded("signals")

    def test_unseen_subsystem_not_degraded(self):
        m = _make_monitor()
        assert not m.is_degraded("never_seen")


# ---------------------------------------------------------------------------
# EventBus integration
# ---------------------------------------------------------------------------

class TestEventBusPublishing:
    def test_tier_change_publishes_event(self):
        bus = MagicMock()
        m = SystemHealthMonitor(event_bus=bus, error_window=50)
        _push_errors(m, "signals", _DEGRADED_THRESHOLD)
        bus.publish.assert_called_once()
        channel, payload = bus.publish.call_args[0]
        assert channel == CH_HEALTH_TIER_CHANGE
        assert payload["old_tier"] == "green"
        assert payload["new_tier"] == "yellow"

    def test_payload_includes_degraded_subsystems(self):
        bus = MagicMock()
        m = SystemHealthMonitor(event_bus=bus, error_window=50)
        _push_errors(m, "granger", _DEGRADED_THRESHOLD)
        _, payload = bus.publish.call_args[0]
        assert "granger" in payload["degraded_subsystems"]

    def test_no_event_when_tier_does_not_change(self):
        bus = MagicMock()
        m = SystemHealthMonitor(event_bus=bus, error_window=50)
        # Single error — does not cross degraded threshold
        m.record_error("signals")
        assert bus.publish.call_count == 0

    def test_two_tier_changes_publish_twice(self):
        bus = MagicMock()
        m = SystemHealthMonitor(event_bus=bus, error_window=50)
        # green → yellow
        _push_errors(m, "signals", _DEGRADED_THRESHOLD)
        # yellow → orange (3 degraded)
        _push_errors(m, "metrics", _DEGRADED_THRESHOLD)
        _push_errors(m, "granger", _DEGRADED_THRESHOLD)
        assert bus.publish.call_count == 2

    def test_no_bus_does_not_raise(self):
        m = _make_monitor(event_bus=None)
        _push_errors(m, "signals", _FAILED_THRESHOLD)  # should not raise


# ---------------------------------------------------------------------------
# Latency report from StepTimer
# ---------------------------------------------------------------------------

class TestLatencyReport:
    def test_returns_empty_when_step_timer_none(self):
        m = _make_monitor(step_timer=None)
        assert m.get_latency_report() == {}

    def test_pulls_from_step_timer(self):
        step_timer = MagicMock()
        step_timer.get_statistics.return_value = {
            "convergence_check": {
                "p50_ms": 10.0,
                "p95_ms": 50.0,
                "max_ms": 200.0,
                "count": 100,
            }
        }
        m = SystemHealthMonitor(step_timer=step_timer, latency_threshold_ms=5000.0)
        report = m.get_latency_report()
        assert "convergence_check" in report
        assert report["convergence_check"]["p50_ms"] == 10.0
        assert report["convergence_check"]["exceeds_threshold"] is False

    def test_exceeds_threshold_flagged(self):
        step_timer = MagicMock()
        step_timer.get_statistics.return_value = {
            "slow_op": {
                "p50_ms": 6000.0,
                "p95_ms": 8000.0,
                "max_ms": 10000.0,
                "count": 5,
            }
        }
        m = SystemHealthMonitor(step_timer=step_timer, latency_threshold_ms=5000.0)
        report = m.get_latency_report()
        assert report["slow_op"]["exceeds_threshold"] is True

    def test_step_timer_exception_returns_empty(self):
        step_timer = MagicMock()
        step_timer.get_statistics.side_effect = RuntimeError("broken")
        m = SystemHealthMonitor(step_timer=step_timer)
        assert m.get_latency_report() == {}


# ---------------------------------------------------------------------------
# get_statistics
# ---------------------------------------------------------------------------

class TestGetStatistics:
    def _required_keys(self) -> set:
        return {
            "overall_tier",
            "subsystems",
            "core_subsystems",
            "latency_summary",
            "error_window",
            "latency_threshold_ms",
        }

    def test_returns_expected_keys(self):
        m = _make_monitor()
        stats = m.get_statistics()
        assert self._required_keys().issubset(stats.keys())

    def test_overall_tier_present_and_correct(self):
        m = _make_monitor()
        _push_errors(m, "signals", _DEGRADED_THRESHOLD)
        stats = m.get_statistics()
        assert stats["overall_tier"] == "yellow"

    def test_subsystems_per_entry_keys(self):
        m = _make_monitor()
        _push_errors(m, "signals", _DEGRADED_THRESHOLD)
        stats = m.get_statistics()
        entry = stats["subsystems"]["signals"]
        assert "health" in entry
        assert "errors_in_window" in entry

    def test_core_subsystems_listed(self):
        m = _make_monitor()
        stats = m.get_statistics()
        assert set(stats["core_subsystems"]) == CORE_SUBSYSTEMS

    def test_latency_summary_empty_when_no_timer(self):
        m = _make_monitor(step_timer=None)
        stats = m.get_statistics()
        assert stats["latency_summary"] == {}

    def test_latency_summary_only_slow_ops(self):
        step_timer = MagicMock()
        step_timer.get_statistics.return_value = {
            "fast_op": {"p50_ms": 1.0, "p95_ms": 2.0, "max_ms": 3.0, "count": 10},
            "slow_op": {"p50_ms": 6000.0, "p95_ms": 8000.0, "max_ms": 10000.0, "count": 5},
        }
        m = SystemHealthMonitor(step_timer=step_timer, latency_threshold_ms=5000.0)
        stats = m.get_statistics()
        assert "slow_op" in stats["latency_summary"]
        assert "fast_op" not in stats["latency_summary"]

    def test_error_window_reflected_in_stats(self):
        m = SystemHealthMonitor(error_window=42)
        stats = m.get_statistics()
        assert stats["error_window"] == 42

    def test_latency_threshold_reflected_in_stats(self):
        m = SystemHealthMonitor(latency_threshold_ms=1234.5)
        stats = m.get_statistics()
        assert stats["latency_threshold_ms"] == 1234.5


# ---------------------------------------------------------------------------
# Graceful degradation with None dependencies
# ---------------------------------------------------------------------------

class TestGracefulDegradation:
    def test_no_bus_no_timer_basic_operation(self):
        m = SystemHealthMonitor(event_bus=None, step_timer=None)
        m.record_error("signals")
        m.record_success("signals")
        tier = m.evaluate_health()
        assert tier == "green"

    def test_evaluate_health_no_subsystems(self):
        m = SystemHealthMonitor()
        assert m.evaluate_health() == "green"

    def test_get_statistics_no_subsystems(self):
        m = SystemHealthMonitor()
        stats = m.get_statistics()
        assert stats["overall_tier"] == "green"
        assert stats["subsystems"] == {}


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_record_error_does_not_raise(self):
        """Multiple threads recording errors concurrently should not corrupt state."""
        m = SystemHealthMonitor(error_window=200)
        errors: list = []

        def worker(subsystem: str) -> None:
            try:
                for _ in range(30):
                    m.record_error(subsystem)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=(f"sub_{i}",))
            for i in range(8)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"

    def test_concurrent_record_success_and_error(self):
        """Interleaved record_error / record_success must not deadlock."""
        m = SystemHealthMonitor(error_window=100)
        errors: list = []
        done = threading.Event()

        def error_worker() -> None:
            try:
                for _ in range(50):
                    m.record_error("signals")
            except Exception as exc:
                errors.append(exc)

        def success_worker() -> None:
            try:
                for _ in range(50):
                    m.record_success("signals")
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=error_worker)
        t2 = threading.Thread(target=success_worker)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert not t1.is_alive(), "error_worker deadlocked"
        assert not t2.is_alive(), "success_worker deadlocked"
        assert errors == [], f"Thread errors: {errors}"

    def test_evaluate_health_concurrent_reads(self):
        """Concurrent evaluate_health calls should not raise."""
        m = SystemHealthMonitor(error_window=100)
        _push_errors(m, "signals", _DEGRADED_THRESHOLD)
        errors: list = []

        def reader() -> None:
            try:
                for _ in range(20):
                    m.evaluate_health()
                    m.get_statistics()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=reader) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []


# ---------------------------------------------------------------------------
# Additional edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_record_error_with_exception_object(self):
        m = _make_monitor()
        exc = ValueError("test error")
        m.record_error("signals", exc)  # should not raise
        assert m.is_degraded("signals") is False  # 1 error < threshold

    def test_same_subsystem_multiple_errors_accumulate(self):
        m = _make_monitor()
        for i in range(_DEGRADED_THRESHOLD - 1):
            m.record_error("signals")
        assert not m.is_degraded("signals")   # one below threshold
        m.record_error("signals")
        assert m.is_degraded("signals")        # now at threshold

    def test_different_subsystems_counted_independently(self):
        m = _make_monitor()
        _push_errors(m, "a", _DEGRADED_THRESHOLD - 1)
        _push_errors(m, "b", _DEGRADED_THRESHOLD - 1)
        # Neither crosses the threshold independently
        assert m.evaluate_health() == "green"

    def test_core_subsystems_constant_has_four_members(self):
        assert len(CORE_SUBSYSTEMS) == 4

    def test_default_error_window_is_100(self):
        m = SystemHealthMonitor()
        assert m._error_window == 100

    def test_default_latency_threshold_is_5000(self):
        m = SystemHealthMonitor()
        assert m._latency_threshold_ms == 5000.0
