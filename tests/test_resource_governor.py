"""Tests for ResourceGovernor — internal budget governance."""
import time
from unittest.mock import MagicMock

import pytest

from mae_core.market.resource_governor import ResourceGovernor


class TestResourceGovernor:
    def test_unregistered_source_always_allowed(self):
        gov = ResourceGovernor()
        assert gov.can_call("unknown_source") is True

    def test_registered_source_allowed_under_limit(self):
        gov = ResourceGovernor()
        gov.register_source("sec_edgar", hourly_limit=100)
        assert gov.can_call("sec_edgar") is True

    def test_record_and_count(self):
        gov = ResourceGovernor()
        gov.register_source("sec_edgar", hourly_limit=100)
        gov.record_call("sec_edgar")
        gov.record_call("sec_edgar")
        usage = gov.get_usage("sec_edgar")
        assert usage["calls_last_hour"] == 2
        assert usage["total_calls"] == 2

    def test_throttled_at_limit(self):
        gov = ResourceGovernor()
        gov.register_source("test_src", hourly_limit=5)
        for _ in range(5):
            gov.record_call("test_src")
        assert gov.can_call("test_src") is False

    def test_under_limit_still_allowed(self):
        gov = ResourceGovernor()
        gov.register_source("test_src", hourly_limit=10)
        for _ in range(9):
            gov.record_call("test_src")
        assert gov.can_call("test_src") is True

    def test_global_limit_enforced(self):
        gov = ResourceGovernor(global_hourly_limit=3)
        gov.register_source("a", hourly_limit=100)
        gov.register_source("b", hourly_limit=100)
        gov.record_call("a")
        gov.record_call("a")
        gov.record_call("b")
        # Global limit of 3 reached
        assert gov.can_call("a") is False

    def test_throttle_event_published(self):
        bus = MagicMock()
        gov = ResourceGovernor(event_bus=bus)
        gov.register_source("test_src", hourly_limit=2)
        gov.record_call("test_src")
        gov.record_call("test_src")
        gov.can_call("test_src")  # should trigger throttle
        bus.publish.assert_called()
        call_args = bus.publish.call_args[0]
        assert call_args[0] == "market.resource.throttle"

    def test_warning_at_threshold(self):
        bus = MagicMock()
        gov = ResourceGovernor(event_bus=bus)
        gov.register_source("test_src", hourly_limit=10, warn_at=0.5)
        for _ in range(5):
            gov.record_call("test_src")
        gov.can_call("test_src")  # 5/10 = 50%, should warn
        # Find the warning publish call
        warning_calls = [
            c for c in bus.publish.call_args_list
            if c[0][0] == "market.resource.budget_warning"
        ]
        assert len(warning_calls) > 0

    def test_statistics(self):
        gov = ResourceGovernor()
        gov.register_source("a", hourly_limit=100)
        gov.register_source("b", hourly_limit=200)
        gov.record_call("a")
        gov.record_call("b")
        gov.record_call("b")
        stats = gov.get_statistics()
        assert stats["sources_registered"] == 2
        assert stats["sources"]["a"]["calls_last_hour"] == 1
        assert stats["sources"]["b"]["calls_last_hour"] == 2

    def test_unregistered_usage_returns_not_registered(self):
        gov = ResourceGovernor()
        usage = gov.get_usage("nonexistent")
        assert usage["registered"] is False

    def test_total_throttled_counter(self):
        gov = ResourceGovernor()
        gov.register_source("test_src", hourly_limit=1)
        gov.record_call("test_src")
        gov.can_call("test_src")  # throttled
        gov.can_call("test_src")  # throttled again
        usage = gov.get_usage("test_src")
        assert usage["total_throttled"] == 2
