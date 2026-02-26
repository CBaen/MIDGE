"""Tests for Bridge 2 — Thompson cascade routing for sweep signals."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from mae_core.market.intelligence.convergence_alerter import (
    ConvergenceAlerter,
    Signal,
)
from mae_core.market.intelligence.thompson_sampler import BetaDistribution


def _make_signal(source="session_sweep", symbol="CL=F", direction="bearish"):
    """Create a sweep signal with metadata."""
    return Signal(
        signal_id="test",
        strength=0.7,
        domain="technical",
        direction=direction,
        timestamp=datetime.now(),
        metadata={"symbol": symbol},
        source=source,
    )


def _mock_thompson(keys_with_params=None):
    """Create a mock ThompsonSampler that returns specified distributions.

    keys_with_params: dict mapping key -> (alpha, beta) tuples.
    Unregistered keys get Beta(1,1) — same as real auto-creation.
    """
    keys_with_params = keys_with_params or {}
    sampler = MagicMock()

    def get_distribution(key, regime="default"):
        if key in keys_with_params:
            a, b = keys_with_params[key]
            return BetaDistribution(alpha=a, beta=b)
        # Auto-create with uninformative prior (samples=0)
        return BetaDistribution(alpha=1.0, beta=1.0)

    sampler.get_distribution = MagicMock(side_effect=get_distribution)
    return sampler


class TestSweepCascade:
    """Tests for _resolve_thompson_key cascade routing."""

    def test_sweep_cascade_uses_combo_key(self):
        """Most specific key (symbol+direction) wins when mature."""
        thompson = _mock_thompson({
            "sweep_bt:CL=F:bearish": (14.1, 9.9),  # samples=22
        })
        alerter = ConvergenceAlerter(thompson_sampler=thompson)
        signal = _make_signal(symbol="CL=F", direction="bearish")
        key = alerter._resolve_thompson_key(signal)
        assert key == "sweep_bt:CL=F:bearish"

    def test_sweep_cascade_falls_to_symbol(self):
        """Missing combo → falls to symbol-only key."""
        thompson = _mock_thompson({
            "sweep_bt:CL=F": (12.0, 10.0),  # samples=20
        })
        alerter = ConvergenceAlerter(thompson_sampler=thompson)
        signal = _make_signal(symbol="CL=F", direction="bearish")
        key = alerter._resolve_thompson_key(signal)
        assert key == "sweep_bt:CL=F"

    def test_sweep_cascade_falls_to_direction(self):
        """Missing combo + symbol → falls to direction-only key."""
        thompson = _mock_thompson({
            "sweep_bt:bearish": (10.0, 8.0),  # samples=16
        })
        alerter = ConvergenceAlerter(thompson_sampler=thompson)
        signal = _make_signal(symbol="CL=F", direction="bearish")
        key = alerter._resolve_thompson_key(signal)
        assert key == "sweep_bt:bearish"

    def test_sweep_cascade_falls_to_generic(self):
        """No mature granular keys → falls to generic session_sweep."""
        thompson = _mock_thompson({})  # All keys auto-create as Beta(1,1)
        alerter = ConvergenceAlerter(thompson_sampler=thompson)
        signal = _make_signal(symbol="CL=F", direction="bearish")
        key = alerter._resolve_thompson_key(signal)
        assert key == "session_sweep"

    def test_sweep_cascade_skips_immature(self):
        """Key with <5 samples is skipped in cascade."""
        thompson = _mock_thompson({
            "sweep_bt:CL=F:bearish": (3.0, 2.0),  # samples=3, immature
            "sweep_bt:CL=F": (12.0, 10.0),  # samples=20, mature
        })
        alerter = ConvergenceAlerter(thompson_sampler=thompson)
        signal = _make_signal(symbol="CL=F", direction="bearish")
        key = alerter._resolve_thompson_key(signal)
        assert key == "sweep_bt:CL=F"

    def test_non_sweep_source_unchanged(self):
        """Non-sweep sources use static _SOURCE_TO_THOMPSON_KEY map."""
        thompson = _mock_thompson({
            "sweep_bt:AAPL:bullish": (20.0, 10.0),
        })
        alerter = ConvergenceAlerter(thompson_sampler=thompson)
        signal = _make_signal(source="sec_form4", symbol="AAPL", direction="bullish")
        key = alerter._resolve_thompson_key(signal)
        assert key == "sec_form4"

    def test_ifvg_uses_same_cascade(self):
        """session_sweep_ifvg triggers the same cascade as session_sweep."""
        thompson = _mock_thompson({
            "sweep_bt:CL=F:bearish": (14.1, 9.9),  # samples=22
        })
        alerter = ConvergenceAlerter(thompson_sampler=thompson)
        signal = _make_signal(source="session_sweep_ifvg", symbol="CL=F",
                              direction="bearish")
        key = alerter._resolve_thompson_key(signal)
        assert key == "sweep_bt:CL=F:bearish"

    def test_sweep_no_thompson_uses_static(self):
        """Without ThompsonSampler, falls back to static map."""
        alerter = ConvergenceAlerter(thompson_sampler=None)
        signal = _make_signal(symbol="CL=F", direction="bearish")
        key = alerter._resolve_thompson_key(signal)
        assert key == "session_sweep"

    def test_cascade_priority_order(self):
        """When all keys are mature, most specific wins."""
        thompson = _mock_thompson({
            "sweep_bt:CL=F:bearish": (10.0, 8.0),  # samples=16
            "sweep_bt:CL=F": (12.0, 10.0),  # samples=20
            "sweep_bt:bearish": (15.0, 12.0),  # samples=25
        })
        alerter = ConvergenceAlerter(thompson_sampler=thompson)
        signal = _make_signal(symbol="CL=F", direction="bearish")
        key = alerter._resolve_thompson_key(signal)
        assert key == "sweep_bt:CL=F:bearish"

    def test_no_symbol_skips_combo_and_symbol(self):
        """Signal without symbol skips combo and symbol-only keys."""
        thompson = _mock_thompson({
            "sweep_bt:bearish": (10.0, 8.0),  # samples=16
        })
        alerter = ConvergenceAlerter(thompson_sampler=thompson)
        signal = _make_signal(symbol="", direction="bearish")
        key = alerter._resolve_thompson_key(signal)
        assert key == "sweep_bt:bearish"

    def test_no_direction_skips_combo_and_direction(self):
        """Signal without direction skips combo and direction-only keys."""
        thompson = _mock_thompson({
            "sweep_bt:CL=F": (12.0, 10.0),  # samples=20
        })
        alerter = ConvergenceAlerter(thompson_sampler=thompson)
        signal = _make_signal(symbol="CL=F", direction="")
        key = alerter._resolve_thompson_key(signal)
        assert key == "sweep_bt:CL=F"


class TestTierRoutingFix:
    """Tests for the session_sweep_ifvg TIER_ROUTING bugfix."""

    def test_ifvg_in_tier_routing(self):
        from mae_core.market.sensing_hook import TIER_ROUTING
        assert TIER_ROUTING.get("session_sweep_ifvg") == "tactical"

    def test_ifvg_matches_sweep_tier(self):
        from mae_core.market.sensing_hook import TIER_ROUTING
        assert TIER_ROUTING["session_sweep_ifvg"] == TIER_ROUTING["session_sweep"]
