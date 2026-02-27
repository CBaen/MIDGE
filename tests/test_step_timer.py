"""Tests for StepTimer — per-operation step latency tracking."""

import time

from mae_core.market.step_timer import StepTimer


class TestStepTimer:
    """Basic functionality."""

    def test_track_records_timing(self):
        timer = StepTimer()
        with timer.track("test_op"):
            time.sleep(0.01)
        stats = timer.get_statistics()
        assert "test_op" in stats
        assert stats["test_op"]["count"] == 1
        assert stats["test_op"]["p50_ms"] >= 5  # at least 5ms

    def test_multiple_operations(self):
        timer = StepTimer()
        with timer.track("op_a"):
            pass
        with timer.track("op_b"):
            pass
        stats = timer.get_statistics()
        assert "op_a" in stats
        assert "op_b" in stats

    def test_multiple_samples(self):
        timer = StepTimer()
        for _ in range(10):
            with timer.track("op"):
                pass
        stats = timer.get_statistics()
        assert stats["op"]["count"] == 10

    def test_percentiles_ordered(self):
        timer = StepTimer()
        for _ in range(100):
            with timer.track("op"):
                pass
        stats = timer.get_statistics()
        assert stats["op"]["p50_ms"] <= stats["op"]["p95_ms"]
        assert stats["op"]["p95_ms"] <= stats["op"]["max_ms"]

    def test_empty_statistics(self):
        timer = StepTimer()
        assert timer.get_statistics() == {}

    def test_reset_clears_data(self):
        timer = StepTimer()
        with timer.track("op"):
            pass
        timer.reset()
        assert timer.get_statistics() == {}

    def test_max_samples_cap(self):
        timer = StepTimer(max_samples=5)
        for _ in range(20):
            with timer.track("op"):
                pass
        stats = timer.get_statistics()
        assert stats["op"]["count"] == 5

    def test_exception_still_records(self):
        timer = StepTimer()
        try:
            with timer.track("fail_op"):
                raise ValueError("test")
        except ValueError:
            pass
        stats = timer.get_statistics()
        assert "fail_op" in stats
        assert stats["fail_op"]["count"] == 1
