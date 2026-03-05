"""Tests for active tracking system — continuous monitoring of predicted assets."""

import json
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

from mae_core.market.archaeology.active_tracker import (
    ActiveTracker,
    TrackedAsset,
    MAX_TRACKED,
    PRICE_COOLDOWN_SECONDS,
    CONFIRM_THRESHOLD_FRACTION,
    FAIL_THRESHOLD_PCT,
)
from mae_core.market.archaeology.fingerprint import PatternTemplate


def _make_stack(symbol="NVDA", direction="bullish", n_patterns=2, tier="medium",
                confidence=0.72, window=7, avg_move=10.0):
    """Build a mock PatternStack for testing."""
    templates = []
    for i in range(n_patterns):
        tmpl = PatternTemplate(
            template_id=f"t{i}", direction=direction,
            domain_signature="insider+technical",
            domains=["insider", "technical"],
            lag_profile_normalized={"short": 0.6, "medium": 0.4},
            n_instances=25,
            symbols_seen=["NVDA", "AAPL", "MSFT"],
            avg_move_pct=avg_move,
        )
        templates.append(tmpl)

    activations = [
        SimpleNamespace(
            template=tmpl, match_score=0.8,
            matched_domains=["insider", "technical"],
            missing_domains=[],
            matched_sources=["sec_form4", "ta_rsi"],
            missing_sources=[],
        )
        for tmpl in templates
    ]

    return SimpleNamespace(
        symbol=symbol,
        direction=direction,
        activations=activations,
        stack_confidence=confidence,
        tier=tier,
        independent_pairs=1,
        created_at=datetime.now().isoformat(),
    )


def _make_price_fetcher(prices: dict):
    """Build a mock price fetcher that returns specified prices."""
    def get_current_price(symbol):
        p = prices.get(symbol)
        if p is None:
            return None
        return SimpleNamespace(price=p, symbol=symbol)
    fetcher = MagicMock()
    fetcher.get_current_price = MagicMock(side_effect=get_current_price)
    return fetcher


# ── Registration Tests ────────────────────────────────────────────


class TestRegistration:

    def test_register_creates_tracked_asset(self, tmp_path):
        at = ActiveTracker(data_dir=tmp_path)
        stack = _make_stack()
        result = at.register(stack, entry_price=150.0)
        assert result is not None
        assert result.symbol == "NVDA"
        assert result.direction == "bullish"
        assert result.entry_price == 150.0
        assert result.status == "tracking"
        assert at.count == 1

    def test_register_dedup(self, tmp_path):
        at = ActiveTracker(data_dir=tmp_path)
        stack = _make_stack()
        at.register(stack, entry_price=150.0)
        result = at.register(stack, entry_price=155.0)
        assert result is None  # duplicate
        assert at.count == 1

    def test_register_different_direction(self, tmp_path):
        at = ActiveTracker(data_dir=tmp_path)
        stack_bull = _make_stack(direction="bullish")
        stack_bear = _make_stack(direction="bearish")
        at.register(stack_bull, entry_price=150.0)
        at.register(stack_bear, entry_price=150.0)
        assert at.count == 2

    def test_register_zero_price_rejected(self, tmp_path):
        at = ActiveTracker(data_dir=tmp_path)
        stack = _make_stack()
        result = at.register(stack, entry_price=0.0)
        assert result is None

    def test_capacity_limit(self, tmp_path):
        at = ActiveTracker(data_dir=tmp_path)
        for i in range(MAX_TRACKED + 5):
            stack = _make_stack(symbol=f"SYM{i}")
            at.register(stack, entry_price=100.0 + i)
        assert at.count <= MAX_TRACKED

    def test_expected_window_from_templates(self, tmp_path):
        at = ActiveTracker(data_dir=tmp_path)
        stack = _make_stack()
        result = at.register(stack, entry_price=150.0)
        # Templates have lag_profile_normalized so should compute dynamic window
        assert result.expected_window_days > 0


# ── Status Transition Tests ──────────────────────────────────────


class TestStatusTransitions:

    def test_confirming_at_half_expected(self, tmp_path):
        pf = _make_price_fetcher({"NVDA": 157.5})  # 5% up from 150
        at = ActiveTracker(price_fetcher=pf, data_dir=tmp_path)
        stack = _make_stack(avg_move=10.0)  # expected 10%, confirm at 5%
        at.register(stack, entry_price=150.0)
        events = at.check_prices()
        statuses = [e["new_status"] for e in events]
        assert "confirming" in statuses

    def test_confirmed_at_full_magnitude(self, tmp_path):
        pf = _make_price_fetcher({"NVDA": 165.0})  # 10% up from 150
        at = ActiveTracker(price_fetcher=pf, data_dir=tmp_path)
        stack = _make_stack(avg_move=10.0)
        at.register(stack, entry_price=150.0)
        events = at.check_prices()
        statuses = [e["new_status"] for e in events]
        assert "confirmed" in statuses

    def test_confirmed_at_5pct_regardless(self, tmp_path):
        pf = _make_price_fetcher({"NVDA": 157.6})  # 5.07% up
        at = ActiveTracker(price_fetcher=pf, data_dir=tmp_path)
        stack = _make_stack(avg_move=20.0)  # high expected, but 5% is enough
        at.register(stack, entry_price=150.0)
        events = at.check_prices()
        statuses = [e["new_status"] for e in events]
        assert "confirmed" in statuses

    def test_failed_on_wrong_direction(self, tmp_path):
        pf = _make_price_fetcher({"NVDA": 145.0})  # -3.3% for bullish = failed
        at = ActiveTracker(price_fetcher=pf, data_dir=tmp_path)
        stack = _make_stack(direction="bullish")
        at.register(stack, entry_price=150.0)
        events = at.check_prices()
        statuses = [e["new_status"] for e in events]
        assert "failed" in statuses

    def test_bearish_confirmed_on_price_drop(self, tmp_path):
        pf = _make_price_fetcher({"NVDA": 142.5})  # -5% = good for bearish
        at = ActiveTracker(price_fetcher=pf, data_dir=tmp_path)
        stack = _make_stack(direction="bearish")
        at.register(stack, entry_price=150.0)
        events = at.check_prices()
        statuses = [e["new_status"] for e in events]
        assert "confirmed" in statuses

    def test_expired_after_window(self, tmp_path):
        pf = _make_price_fetcher({"NVDA": 151.0})  # small move, not enough
        at = ActiveTracker(price_fetcher=pf, data_dir=tmp_path)
        stack = _make_stack()
        asset = at.register(stack, entry_price=150.0)
        # Backdate creation to trigger expiry
        asset.created_at = (datetime.now() - timedelta(days=30)).isoformat()
        events = at.check_prices()
        statuses = [e["new_status"] for e in events]
        assert "expired" in statuses


# ── MFE/MAE Tracking Tests ──────────────────────────────────────


class TestMFEMAE:

    def test_mfe_tracked(self, tmp_path):
        pf = _make_price_fetcher({"NVDA": 157.0})  # +4.67%
        at = ActiveTracker(price_fetcher=pf, data_dir=tmp_path)
        stack = _make_stack(avg_move=15.0)  # high expected so stays tracking
        at.register(stack, entry_price=150.0)
        at.check_prices()
        asset = list(at._tracked.values())[0]
        assert asset.mfe_pct > 4.0

    def test_mae_tracked(self, tmp_path):
        pf = _make_price_fetcher({"NVDA": 148.0})  # -1.33% (not enough to fail)
        at = ActiveTracker(price_fetcher=pf, data_dir=tmp_path)
        stack = _make_stack(avg_move=15.0)
        at.register(stack, entry_price=150.0)
        at.check_prices()
        asset = list(at._tracked.values())[0]
        assert asset.mae_pct < 0


# ── Persistence Tests ────────────────────────────────────────────


class TestPersistence:

    def test_save_and_restore(self, tmp_path):
        at = ActiveTracker(data_dir=tmp_path)
        stack = _make_stack()
        at.register(stack, entry_price=150.0)

        # Create new tracker from same path
        at2 = ActiveTracker(data_dir=tmp_path)
        assert at2.count == 1
        asset = list(at2._tracked.values())[0]
        assert asset.symbol == "NVDA"
        assert asset.entry_price == 150.0

    def test_empty_path_creates_clean(self, tmp_path):
        at = ActiveTracker(data_dir=tmp_path)
        assert at.count == 0


# ── Rate Limiting Tests ──────────────────────────────────────────


class TestRateLimiting:

    def test_cooldown_skips_recent_check(self, tmp_path):
        pf = _make_price_fetcher({"NVDA": 155.0})
        at = ActiveTracker(price_fetcher=pf, data_dir=tmp_path)
        stack = _make_stack(avg_move=15.0)
        at.register(stack, entry_price=150.0)

        # First check should work
        at.check_prices()
        assert pf.get_current_price.call_count == 1

        # Second check within cooldown should skip
        at.check_prices()
        assert pf.get_current_price.call_count == 1  # not called again


# ── Statistics Tests ─────────────────────────────────────────────


class TestStatistics:

    def test_get_statistics(self, tmp_path):
        at = ActiveTracker(data_dir=tmp_path)
        stack = _make_stack()
        at.register(stack, entry_price=150.0)
        stats = at.get_statistics()
        assert stats["total_tracked"] == 1
        assert "tracking" in stats["by_status"]

    def test_get_status_summary(self, tmp_path):
        at = ActiveTracker(data_dir=tmp_path)
        stack = _make_stack()
        at.register(stack, entry_price=150.0)
        summary = at.get_status_summary()
        assert len(summary) == 1
        assert summary[0]["symbol"] == "NVDA"
        assert summary[0]["entry_price"] == 150.0
