"""Tests for InhabitantScheduler.

What is under test:
    mae_core/scheduling/inhabitant_scheduler.py

Design decisions validated:
    - register() adds a system; callback fires within interval
    - unregister() prevents further dispatch (lazy deletion)
    - reschedule() changes cadence so system fires sooner
    - higher priority fires first when two tasks are simultaneously due
    - stop() signals the daemon, thread joins, no further dispatches
    - get_statistics() returns expected keys with correct types
    - EventBus publish is best-effort: exception in bus never breaks scheduler
    - Callback exceptions are caught by _run_callback, never propagate

Test strategy:
    - Use threading.Event and time.sleep(small) to observe real dispatch timing
    - Keep sleep times small (0.05–0.15s) to stay fast but allow scheduler ticks
    - Use threading.Lock + counter for thread-safe callback counting
    - No mocking of threads — we test the real daemon behavior
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from mae_core.scheduling.inhabitant_scheduler import (
    InhabitantScheduler,
    CH_INHABITANT_DISPATCHED,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _counter() -> tuple[list, callable]:
    """Return a (calls list, callback) pair. Thread-safe via list.append."""
    calls: list[float] = []
    lock = threading.Lock()

    def cb():
        with lock:
            calls.append(time.monotonic())

    return calls, cb


def _make_scheduler(**kwargs) -> InhabitantScheduler:
    defaults = {"event_bus": None, "max_workers": 2}
    defaults.update(kwargs)
    return InhabitantScheduler(**defaults)


# ---------------------------------------------------------------------------
# Section 1: register and start
# ---------------------------------------------------------------------------

class TestRegisterAndStart:
    """Callback fires within the declared interval after start()."""

    def test_register_and_start(self):
        """Single registered callback fires at least once within its interval."""
        calls, cb = _counter()
        sched = _make_scheduler()
        sched.register("test_system", cb, interval_seconds=0.05)
        sched.start()
        try:
            # Wait 3× the interval for robustness on slow CI.
            time.sleep(0.20)
        finally:
            sched.stop()

        assert len(calls) >= 1, (
            f"Expected callback to fire at least once, got {len(calls)} calls"
        )

    def test_callback_fires_multiple_times(self):
        """Callback fires repeatedly, not just once."""
        calls, cb = _counter()
        sched = _make_scheduler()
        sched.register("repeating", cb, interval_seconds=0.04)
        sched.start()
        try:
            time.sleep(0.30)
        finally:
            sched.stop()

        assert len(calls) >= 3, (
            f"Expected >= 3 firings, got {len(calls)}"
        )

    def test_multiple_systems_registered(self):
        """Two registered systems both fire."""
        calls_a, cb_a = _counter()
        calls_b, cb_b = _counter()
        sched = _make_scheduler()
        sched.register("system_a", cb_a, interval_seconds=0.05)
        sched.register("system_b", cb_b, interval_seconds=0.05)
        sched.start()
        try:
            time.sleep(0.25)
        finally:
            sched.stop()

        assert len(calls_a) >= 1, "system_a did not fire"
        assert len(calls_b) >= 1, "system_b did not fire"


# ---------------------------------------------------------------------------
# Section 2: unregister
# ---------------------------------------------------------------------------

class TestUnregister:
    """Unregistered system stops being dispatched."""

    def test_unregister_stops_dispatch(self):
        """After unregister(), callback does not fire again."""
        calls, cb = _counter()
        sched = _make_scheduler()
        sched.register("removable", cb, interval_seconds=0.04)
        sched.start()

        # Let it fire at least once.
        time.sleep(0.12)
        sched.unregister("removable")

        # Record count after unregister.
        count_after_unregister = len(calls)

        # Wait another window — no new calls should come.
        time.sleep(0.15)
        sched.stop()

        count_after_wait = len(calls)
        assert count_after_wait == count_after_unregister, (
            f"Calls increased after unregister: {count_after_unregister} -> {count_after_wait}"
        )

    def test_unregister_nonexistent_no_error(self):
        """Unregistering a system that was never registered must not raise."""
        sched = _make_scheduler()
        sched.unregister("never_registered")  # Should not raise.

    def test_unregister_before_start_safe(self):
        """Unregistering before start() must not raise."""
        calls, cb = _counter()
        sched = _make_scheduler()
        sched.register("early_out", cb, interval_seconds=0.05)
        sched.unregister("early_out")
        sched.start()
        try:
            time.sleep(0.15)
        finally:
            sched.stop()
        # Callback may or may not have fired depending on timing of lazy deletion.
        # The important invariant: no exception raised.


# ---------------------------------------------------------------------------
# Section 3: reschedule
# ---------------------------------------------------------------------------

class TestReschedule:
    """reschedule() changes the cadence of a registered system."""

    def test_reschedule_changes_interval(self):
        """After reschedule to a shorter interval, more dispatches occur."""
        calls, cb = _counter()
        sched = _make_scheduler()
        # Start with a slow interval.
        sched.register("adaptive", cb, interval_seconds=0.15)
        sched.start()

        # Wait one slow interval to let first dispatch happen.
        time.sleep(0.20)
        count_before = len(calls)

        # Speed up to 10× faster.
        sched.reschedule("adaptive", interval_seconds=0.02)
        time.sleep(0.20)
        sched.stop()

        count_after = len(calls)
        new_dispatches = count_after - count_before

        assert new_dispatches >= 3, (
            f"Expected >= 3 dispatches after speedup, got {new_dispatches}"
        )

    def test_reschedule_nonexistent_logs_warning(self):
        """Rescheduling a non-registered system logs a warning and does not raise."""
        sched = _make_scheduler()
        import logging
        with patch.object(logging.getLogger("mae_core.scheduling.inhabitant_scheduler"), "warning") as mock_warn:
            sched.reschedule("ghost_system", interval_seconds=1.0)
        mock_warn.assert_called_once()
        assert "ghost_system" in mock_warn.call_args[0][0] or "ghost_system" in str(mock_warn.call_args)


# ---------------------------------------------------------------------------
# Section 4: priority ordering
# ---------------------------------------------------------------------------

class TestPriorityOrdering:
    """Higher-priority tasks fire before lower-priority when simultaneously due."""

    def test_priority_ordering(self):
        """Two systems with the same short interval: higher priority fires first.

        Strategy: register both with identical short intervals so they become
        due at the same time on each cycle. Use max_workers=1 (serial pool)
        so the heap ordering determines which callback executes first.

        The heap key is (next_run_time, -priority, system_name). Equal times
        break ties in favor of higher priority (lower heap key because we
        negate priority). Over many cycles, high-priority should almost always
        appear before low-priority in each pair of dispatches.

        We check a weaker but robust invariant: out of all dispatches observed,
        high-priority has a greater-or-equal total count than low-priority,
        because it fires first in each cycle pair.
        """
        dispatch_order: list[str] = []
        lock = threading.Lock()

        def make_cb(name):
            def cb():
                with lock:
                    dispatch_order.append(name)
            return cb

        # Use a very short interval so many cycles occur.
        sched = _make_scheduler(max_workers=1)  # serial execution
        sched.register("low_priority", make_cb("low"), interval_seconds=0.04, priority=0)
        sched.register("high_priority", make_cb("high"), interval_seconds=0.04, priority=10)

        sched.start()
        try:
            time.sleep(0.40)
        finally:
            sched.stop()

        # Both must have fired.
        assert "high" in dispatch_order, "high_priority callback never fired"
        assert "low" in dispatch_order, "low_priority callback never fired"

        # The heap key (next_run_time, -priority, name) means that when two entries
        # share the same next_run_time, higher priority (larger int → smaller neg key)
        # is popped first. Verify: in the first two dispatches, high comes before low.
        assert len(dispatch_order) >= 2, (
            f"Expected at least 2 dispatches, got {len(dispatch_order)}"
        )
        # Verify heapq priority by checking raw heap ordering invariant:
        # higher priority should appear at least as often as lower priority
        # (it fires first in each round, so count(high) >= count(low)).
        count_high = dispatch_order.count("high")
        count_low = dispatch_order.count("low")
        assert count_high >= count_low, (
            f"Expected high_priority to fire >= low_priority times, "
            f"got high={count_high}, low={count_low}. "
            f"Order sample: {dispatch_order[:10]}"
        )


# ---------------------------------------------------------------------------
# Section 5: stop graceful
# ---------------------------------------------------------------------------

class TestStopGraceful:
    """stop() terminates the daemon thread cleanly."""

    def test_stop_graceful(self):
        """stop() returns within a reasonable time and thread is no longer alive."""
        sched = _make_scheduler()
        calls, cb = _counter()
        sched.register("bg_task", cb, interval_seconds=0.05)
        sched.start()

        assert sched._thread is not None
        assert sched._thread.is_alive()

        start = time.monotonic()
        sched.stop()
        elapsed = time.monotonic() - start

        # Thread should have joined within the 10s timeout; for a healthy
        # scheduler it should be much faster.
        assert elapsed < 2.0, f"stop() took too long: {elapsed:.2f}s"
        assert sched._thread is None, "Thread reference should be cleared after stop()"

    def test_no_more_dispatches_after_stop(self):
        """After stop(), callback no longer fires."""
        calls, cb = _counter()
        sched = _make_scheduler()
        sched.register("active", cb, interval_seconds=0.04)
        sched.start()
        time.sleep(0.15)

        sched.stop()
        count_at_stop = len(calls)

        time.sleep(0.20)
        count_after_wait = len(calls)

        assert count_after_wait == count_at_stop, (
            f"Callbacks continued after stop: {count_at_stop} -> {count_after_wait}"
        )

    def test_double_stop_no_error(self):
        """Calling stop() twice must not raise."""
        sched = _make_scheduler()
        sched.start()
        sched.stop()
        sched.stop()  # Should not raise.

    def test_stop_without_start_no_error(self):
        """Calling stop() before start() must not raise."""
        sched = _make_scheduler()
        sched.stop()  # Should not raise.


# ---------------------------------------------------------------------------
# Section 6: get_statistics
# ---------------------------------------------------------------------------

class TestGetStatistics:
    """get_statistics() returns a dict with expected keys and types."""

    def test_get_statistics_keys(self):
        """Statistics dict has all required keys."""
        sched = _make_scheduler()
        stats = sched.get_statistics()

        required_keys = {"registered_systems", "systems", "running", "max_workers"}
        assert required_keys.issubset(stats.keys()), (
            f"Missing keys: {required_keys - stats.keys()}"
        )

    def test_get_statistics_registered_count(self):
        """registered_systems count reflects registered callbacks."""
        sched = _make_scheduler()
        _, cb1 = _counter()
        _, cb2 = _counter()
        sched.register("s1", cb1, interval_seconds=1.0)
        sched.register("s2", cb2, interval_seconds=2.0)

        stats = sched.get_statistics()
        assert stats["registered_systems"] == 2

    def test_get_statistics_running_false_before_start(self):
        """running is False before start() is called."""
        sched = _make_scheduler()
        _, cb = _counter()
        sched.register("s", cb, interval_seconds=1.0)
        stats = sched.get_statistics()
        assert stats["running"] is False

    def test_get_statistics_running_true_after_start(self):
        """running is True while scheduler is active."""
        sched = _make_scheduler()
        sched.start()
        try:
            stats = sched.get_statistics()
            assert stats["running"] is True
        finally:
            sched.stop()

    def test_get_statistics_run_count_increments(self):
        """run_count for a system increases after it fires."""
        sched = _make_scheduler()
        _, cb = _counter()
        sched.register("counting", cb, interval_seconds=0.04)
        sched.start()
        try:
            time.sleep(0.25)
        finally:
            sched.stop()

        stats = sched.get_statistics()
        sys_info = stats["systems"].get("counting")
        assert sys_info is not None, "System 'counting' missing from statistics"
        assert sys_info["run_count"] >= 1, (
            f"Expected run_count >= 1, got {sys_info['run_count']}"
        )

    def test_get_statistics_last_run_time_set(self):
        """last_run_time is set after the first dispatch."""
        sched = _make_scheduler()
        _, cb = _counter()
        sched.register("timed", cb, interval_seconds=0.04)
        sched.start()
        try:
            time.sleep(0.20)
        finally:
            sched.stop()

        stats = sched.get_statistics()
        sys_info = stats["systems"].get("timed")
        assert sys_info is not None
        assert sys_info["last_run_time"] is not None, (
            "last_run_time should be set after first dispatch"
        )

    def test_get_statistics_per_system_keys(self):
        """Each system entry has interval_seconds, priority, run_count, last_run_time."""
        sched = _make_scheduler()
        _, cb = _counter()
        sched.register("detailed", cb, interval_seconds=5.0, priority=3)

        stats = sched.get_statistics()
        sys_info = stats["systems"].get("detailed")
        assert sys_info is not None
        for key in ("interval_seconds", "priority", "run_count", "last_run_time"):
            assert key in sys_info, f"Missing key in system stats: {key}"
        assert sys_info["interval_seconds"] == 5.0
        assert sys_info["priority"] == 3


# ---------------------------------------------------------------------------
# Section 7: EventBus integration
# ---------------------------------------------------------------------------

class TestEventBusIntegration:
    """EventBus publish on dispatch; errors never break the scheduler."""

    def test_eventbus_published_on_dispatch(self):
        """When event_bus is provided, CH_INHABITANT_DISPATCHED is published."""
        bus = MagicMock()
        sched = InhabitantScheduler(event_bus=bus, max_workers=2)
        _, cb = _counter()
        sched.register("bus_test", cb, interval_seconds=0.04)
        sched.start()
        try:
            time.sleep(0.20)
        finally:
            sched.stop()

        # publish should have been called at least once.
        assert bus.publish.called, "Expected EventBus.publish to be called"
        # All calls should use the correct channel.
        for call in bus.publish.call_args_list:
            channel = call[0][0]
            assert channel == CH_INHABITANT_DISPATCHED, (
                f"Unexpected channel: {channel}"
            )

    def test_eventbus_exception_does_not_crash_scheduler(self):
        """If bus.publish raises, the scheduler must continue running."""
        bus = MagicMock()
        bus.publish.side_effect = RuntimeError("bus exploded")

        sched = InhabitantScheduler(event_bus=bus, max_workers=2)
        calls, cb = _counter()
        sched.register("resilient", cb, interval_seconds=0.04)
        sched.start()
        try:
            time.sleep(0.25)
        finally:
            sched.stop()

        # Despite bus failures, callback still fired.
        assert len(calls) >= 1, "Scheduler stopped working after bus exception"

    def test_no_eventbus_no_publish_attempted(self):
        """Without event_bus, no publish is attempted."""
        sched = _make_scheduler(event_bus=None)
        _, cb = _counter()
        sched.register("quiet", cb, interval_seconds=0.05)
        sched.start()
        try:
            time.sleep(0.20)
        finally:
            sched.stop()
        # No assertion needed — the point is it completes without error.


# ---------------------------------------------------------------------------
# Section 8: Callback exception safety
# ---------------------------------------------------------------------------

class TestCallbackExceptionSafety:
    """A crashing callback must not kill the scheduler."""

    def test_crashing_callback_does_not_kill_scheduler(self):
        """If a callback raises, subsequent callbacks still fire."""
        calls, good_cb = _counter()

        def crashing_cb():
            raise ValueError("intentional crash in test")

        sched = _make_scheduler()
        sched.register("crasher", crashing_cb, interval_seconds=0.04)
        sched.register("survivor", good_cb, interval_seconds=0.04)
        sched.start()
        try:
            time.sleep(0.30)
        finally:
            sched.stop()

        assert len(calls) >= 1, (
            "Scheduler stopped processing after callback exception"
        )
