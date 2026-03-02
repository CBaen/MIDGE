"""Tests for per-domain convergence windows in ConvergenceAlerter.

Verifies that slow-moving data sources (positioning=14d, government=7d,
contracts=7d) survive pruning longer than the default 72h window, and
that slow signals can still contribute to convergence with fresh domains.
"""

from datetime import datetime, timedelta

import pytest

from mae_core.market.intelligence.convergence_alerter import (
    ConvergenceAlerter,
    Signal,
)


# ── Helpers ────────────────────────────────────────────────────────────────


def _record_signal(alerter, signal_id, domain, direction="bullish",
                   strength=0.7, age_hours=0, age_days=0):
    """Record a signal backdated by age_hours + age_days."""
    age = timedelta(hours=age_hours, days=age_days)
    ts = datetime.now() - age
    alerter.record_signal(
        signal_id=signal_id,
        strength=strength,
        domain=domain,
        direction=direction,
        confidence=0.6,
        timestamp=ts,
    )


# ── Tests ──────────────────────────────────────────────────────────────────


class TestDomainWindows:
    def test_positioning_14day_window(self):
        """Positioning signal 10 days old survives prune (14-day window)."""
        alerter = ConvergenceAlerter(min_domains=1, convergence_window_hours=72)
        _record_signal(alerter, "cot-1", "positioning", age_days=10)
        alerter._prune_old_signals()
        assert "positioning" in alerter.signals
        assert len(alerter.signals["positioning"]) == 1

    def test_positioning_15day_prune(self):
        """Positioning signal 15 days old gets pruned (outside 14-day window)."""
        alerter = ConvergenceAlerter(min_domains=1, convergence_window_hours=72)
        _record_signal(alerter, "cot-2", "positioning", age_days=15)
        alerter._prune_old_signals()
        assert "positioning" not in alerter.signals

    def test_government_7day_window(self):
        """Government signal 6 days old survives (7-day window)."""
        alerter = ConvergenceAlerter(min_domains=1, convergence_window_hours=72)
        _record_signal(alerter, "gov-1", "government", age_days=6)
        alerter._prune_old_signals()
        assert "government" in alerter.signals
        assert len(alerter.signals["government"]) == 1

    def test_government_8day_prune(self):
        """Government signal 8 days old gets pruned (outside 7-day window)."""
        alerter = ConvergenceAlerter(min_domains=1, convergence_window_hours=72)
        _record_signal(alerter, "gov-2", "government", age_days=8)
        alerter._prune_old_signals()
        assert "government" not in alerter.signals

    def test_contracts_7day_window(self):
        """Contracts signal 6 days old survives (7-day window)."""
        alerter = ConvergenceAlerter(min_domains=1, convergence_window_hours=72)
        _record_signal(alerter, "con-1", "contracts", age_days=6)
        alerter._prune_old_signals()
        assert "contracts" in alerter.signals
        assert len(alerter.signals["contracts"]) == 1

    def test_default_72h_window_survives(self):
        """Technical signal 70h old survives the default 72h window."""
        alerter = ConvergenceAlerter(min_domains=1, convergence_window_hours=72)
        _record_signal(alerter, "tech-1", "technical", age_hours=70)
        alerter._prune_old_signals()
        assert "technical" in alerter.signals

    def test_default_72h_window_pruned(self):
        """Technical signal 73h old gets pruned by the default 72h window."""
        alerter = ConvergenceAlerter(min_domains=1, convergence_window_hours=72)
        _record_signal(alerter, "tech-2", "technical", age_hours=73)
        alerter._prune_old_signals()
        assert "technical" not in alerter.signals

    def test_slow_signal_contributes_to_convergence(self):
        """Positioning signal 10 days old + 2 fresh directional domains → convergence fires."""
        alerter = ConvergenceAlerter(min_domains=3, convergence_window_hours=72)

        # Slow positioning signal (neutral is fine — counts toward domain presence)
        _record_signal(alerter, "cot-3", "positioning", direction="neutral",
                       strength=0.7, age_days=10)

        # Two fresh directional signals
        _record_signal(alerter, "ins-1", "insider", direction="bullish",
                       strength=0.75, age_hours=1)
        _record_signal(alerter, "gov-3", "government", direction="bullish",
                       strength=0.70, age_hours=2)

        alerts = alerter.check_convergence(direction_filter="bullish")
        assert len(alerts) == 1, "Should fire: 3 domains (2 directional + 1 neutral positioning)"

    def test_domain_windows_backward_compat(self):
        """Domains not in DOMAIN_WINDOWS use the global convergence_window (72h default)."""
        alerter = ConvergenceAlerter(min_domains=1, convergence_window_hours=72)

        # 'sentiment' is not in _domain_windows — uses global 72h
        _record_signal(alerter, "sent-1", "sentiment", age_hours=70)  # survives
        _record_signal(alerter, "sent-2", "sentiment", age_hours=73)  # pruned
        alerter._prune_old_signals()

        # Only the 70h signal should survive
        remaining = alerter.signals.get("sentiment", [])
        assert len(remaining) == 1
        assert remaining[0].signal_id == "sent-1"

    def test_domain_windows_dict_exists(self):
        """ConvergenceAlerter exposes _domain_windows with the three extended domains."""
        alerter = ConvergenceAlerter()
        assert "positioning" in alerter._domain_windows
        assert "government" in alerter._domain_windows
        assert "contracts" in alerter._domain_windows
        # Positioning gets 14 days
        assert alerter._domain_windows["positioning"] == timedelta(days=14)
        # Government and contracts get 7 days
        assert alerter._domain_windows["government"] == timedelta(days=7)
        assert alerter._domain_windows["contracts"] == timedelta(days=7)
