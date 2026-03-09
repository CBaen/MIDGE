"""Tests for ResourceGovernor — internal budget governance."""
import time
from unittest.mock import MagicMock

import pytest

from mae_core.market.resource_governor import ResourceGovernor, SourceTier


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


class TestSourceTiers:
    """Tests for DEB priority tier logic."""

    def test_maintenance_tier_never_throttled(self):
        gov = ResourceGovernor()
        gov.register_source("heartbeat", hourly_limit=2, tier=SourceTier.MAINTENANCE)
        gov.record_call("heartbeat")
        gov.record_call("heartbeat")
        # Over limit but MAINTENANCE — always allowed
        assert gov.can_call("heartbeat") is True

    def test_active_tier_gets_1_5x_multiplier(self):
        gov = ResourceGovernor()
        gov.register_source("sec_edgar", hourly_limit=10, tier=SourceTier.ACTIVE)
        for _ in range(10):
            gov.record_call("sec_edgar")
        # At nominal limit but ACTIVE gets 1.5x (15) — still allowed
        assert gov.can_call("sec_edgar") is True
        for _ in range(5):
            gov.record_call("sec_edgar")
        # Now at 15 = effective limit — throttled
        assert gov.can_call("sec_edgar") is False

    def test_explore_tier_uses_standard_limit(self):
        gov = ResourceGovernor()
        gov.register_source("test", hourly_limit=5, tier=SourceTier.EXPLORE)
        for _ in range(5):
            gov.record_call("test")
        assert gov.can_call("test") is False

    def test_set_source_tier(self):
        gov = ResourceGovernor()
        gov.register_source("src", hourly_limit=5)
        assert gov.get_usage("src")["tier"] == "explore"  # default
        gov.set_source_tier("src", SourceTier.ACTIVE)
        assert gov.get_usage("src")["tier"] == "active"

    def test_set_source_tier_unregistered_is_noop(self):
        gov = ResourceGovernor()
        gov.set_source_tier("nonexistent", SourceTier.MAINTENANCE)  # no crash

    def test_tighten_only_affects_explore(self):
        gov = ResourceGovernor()
        gov.register_source("active_src", hourly_limit=100, tier=SourceTier.ACTIVE)
        gov.register_source("explore_src", hourly_limit=100, tier=SourceTier.EXPLORE)
        gov.register_source("maint_src", hourly_limit=100, tier=SourceTier.MAINTENANCE)
        gov.tighten_budgets(0.5)
        assert gov.get_usage("explore_src")["hourly_limit"] == 50
        assert gov.get_usage("active_src")["hourly_limit"] == 100
        assert gov.get_usage("maint_src")["hourly_limit"] == 100

    def test_relax_only_affects_explore(self):
        gov = ResourceGovernor()
        gov.register_source("explore_src", hourly_limit=100, tier=SourceTier.EXPLORE)
        gov.register_source("active_src", hourly_limit=100, tier=SourceTier.ACTIVE)
        gov.tighten_budgets(0.5)  # explore_src → 50
        gov.relax_budgets(1.5)    # explore_src → 75 (capped at original 100)
        assert gov.get_usage("explore_src")["hourly_limit"] == 75
        assert gov.get_usage("active_src")["hourly_limit"] == 100

    def test_relax_capped_at_original(self):
        gov = ResourceGovernor()
        gov.register_source("src", hourly_limit=100, tier=SourceTier.EXPLORE)
        gov.relax_budgets(2.0)  # Would be 200 but capped at original 100
        assert gov.get_usage("src")["hourly_limit"] == 100

    def test_tighten_rejects_factor_above_one(self):
        gov = ResourceGovernor()
        gov.register_source("src", hourly_limit=100, tier=SourceTier.EXPLORE)
        gov.tighten_budgets(1.5)  # Should be rejected
        assert gov.get_usage("src")["hourly_limit"] == 100

    def test_relax_rejects_factor_below_one(self):
        gov = ResourceGovernor()
        gov.register_source("src", hourly_limit=100, tier=SourceTier.EXPLORE)
        gov.relax_budgets(0.5)  # Should be rejected
        assert gov.get_usage("src")["hourly_limit"] == 100

    def test_tighten_floor_at_one(self):
        gov = ResourceGovernor()
        gov.register_source("src", hourly_limit=2, tier=SourceTier.EXPLORE)
        gov.tighten_budgets(0.1)  # 2 * 0.1 = 0.2 → max(1, 0) = 1
        assert gov.get_usage("src")["hourly_limit"] == 1
