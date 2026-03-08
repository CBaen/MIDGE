"""Tests for quorum contributor count as confidence multiplier on convergence alerts.

B4 Quorum & Convergence — Round 1 build.
Verifies that ConvergenceAlerter._apply_quorum_boost() correctly amplifies
confidence when multiple independent agents deposit signals for the same
ticker+direction in QuorumSpace.
"""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from mae_core.market.intelligence.convergence_alerter import (
    ConvergenceAlerter,
    Signal,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_quorum_space(signal_key: str, contributor_count: int):
    """Return a mock QuorumSpace whose get_contributor_count() returns
    the specified count for the given signal_key, and 0 for everything else.
    """
    qs = MagicMock()
    qs.get_contributor_count = MagicMock(
        side_effect=lambda k: contributor_count if k == signal_key else 0
    )
    return qs


def _make_alerter(quorum_space=None) -> ConvergenceAlerter:
    """Create a minimal ConvergenceAlerter with optional quorum_space."""
    return ConvergenceAlerter(quorum_space=quorum_space)


def _make_signals(n: int, direction: str = "bullish", ticker: str = "AAPL") -> list:
    """Create n minimal Signal objects pointing at ticker."""
    return [
        Signal(
            signal_id=f"sig-{i}",
            strength=0.7,
            domain=f"domain{i}",
            direction=direction,
            timestamp=datetime.now(),
            metadata={"symbol": ticker},
            source=f"source{i}",
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestQuorumBoost:
    """Unit tests for _apply_quorum_boost()."""

    def test_quorum_boost_at_3_contributors(self):
        """3 contributors → 1.1× multiplier applied to confidence."""
        qs = _make_quorum_space("bullish:AAPL", 3)
        alerter = _make_alerter(quorum_space=qs)

        base_confidence = 0.60
        result = alerter._apply_quorum_boost(base_confidence, "AAPL", "bullish")

        expected = min(1.0, base_confidence * 1.1)
        assert abs(result - expected) < 1e-9, (
            f"Expected {expected:.6f}, got {result:.6f}"
        )

    def test_quorum_boost_at_5_contributors(self):
        """5 contributors → 1.3× multiplier (cap) applied to confidence."""
        qs = _make_quorum_space("bullish:TSLA", 5)
        alerter = _make_alerter(quorum_space=qs)

        base_confidence = 0.50
        result = alerter._apply_quorum_boost(base_confidence, "TSLA", "bullish")

        expected = min(1.0, base_confidence * 1.3)
        assert abs(result - expected) < 1e-9, (
            f"Expected {expected:.6f}, got {result:.6f}"
        )

    def test_quorum_boost_at_4_contributors(self):
        """4 contributors → 1.2× multiplier applied to confidence."""
        qs = _make_quorum_space("bearish:SPY", 4)
        alerter = _make_alerter(quorum_space=qs)

        base_confidence = 0.55
        result = alerter._apply_quorum_boost(base_confidence, "SPY", "bearish")

        expected = min(1.0, base_confidence * 1.2)
        assert abs(result - expected) < 1e-9, (
            f"Expected {expected:.6f}, got {result:.6f}"
        )

    def test_quorum_boost_capped_at_1(self):
        """High base confidence + quorum boost must not exceed 1.0."""
        qs = _make_quorum_space("bullish:NVDA", 5)
        alerter = _make_alerter(quorum_space=qs)

        # 0.95 * 1.3 = 1.235 — must be capped at 1.0
        base_confidence = 0.95
        result = alerter._apply_quorum_boost(base_confidence, "NVDA", "bullish")

        assert result <= 1.0, f"Confidence exceeded 1.0: {result}"
        assert result == 1.0, (
            f"Expected 1.0 (capped), got {result:.6f}"
        )

    def test_no_quorum_space_no_boost(self):
        """quorum_space=None → confidence unchanged."""
        alerter = _make_alerter(quorum_space=None)

        base_confidence = 0.70
        result = alerter._apply_quorum_boost(base_confidence, "AAPL", "bullish")

        assert result == base_confidence, (
            f"Expected {base_confidence}, got {result}"
        )

    def test_quorum_below_threshold_no_boost(self):
        """1 contributor → no boost (count < 3 threshold)."""
        qs = _make_quorum_space("bullish:AAPL", 1)
        alerter = _make_alerter(quorum_space=qs)

        base_confidence = 0.60
        result = alerter._apply_quorum_boost(base_confidence, "AAPL", "bullish")

        assert result == base_confidence, (
            f"Expected no change ({base_confidence}), got {result}"
        )

    def test_quorum_2_contributors_no_boost(self):
        """2 contributors → still no boost (threshold is 3)."""
        qs = _make_quorum_space("bullish:AAPL", 2)
        alerter = _make_alerter(quorum_space=qs)

        base_confidence = 0.60
        result = alerter._apply_quorum_boost(base_confidence, "AAPL", "bullish")

        assert result == base_confidence, (
            f"Expected no change ({base_confidence}), got {result}"
        )

    def test_quorum_signal_key_format(self):
        """Signal key passed to QuorumSpace must be '{direction}:{ticker}'."""
        recorded_keys = []

        qs = MagicMock()
        def capture_key(k):
            recorded_keys.append(k)
            return 5  # always return 5 contributors so boost fires

        qs.get_contributor_count = MagicMock(side_effect=capture_key)
        alerter = _make_alerter(quorum_space=qs)

        alerter._apply_quorum_boost(0.60, "MSFT", "bearish")

        assert len(recorded_keys) == 1, "Expected exactly one key lookup"
        assert recorded_keys[0] == "bearish:MSFT", (
            f"Expected 'bearish:MSFT', got '{recorded_keys[0]}'"
        )

    def test_quorum_large_contributor_count_capped_multiplier(self):
        """100 contributors → multiplier capped at 1.3× (not unbounded)."""
        qs = _make_quorum_space("bullish:BTC-USD", 100)
        alerter = _make_alerter(quorum_space=qs)

        base_confidence = 0.50
        result = alerter._apply_quorum_boost(base_confidence, "BTC-USD", "bullish")

        # max multiplier = 1.3, so result should be exactly 0.50 * 1.3 = 0.65
        expected = min(1.0, base_confidence * 1.3)
        assert abs(result - expected) < 1e-9, (
            f"Multiplier not capped at 1.3×: expected {expected:.6f}, got {result:.6f}"
        )

    def test_quorum_empty_ticker_no_boost(self):
        """Empty ticker string → no boost (guard condition)."""
        qs = MagicMock()
        qs.get_contributor_count = MagicMock(return_value=5)
        alerter = _make_alerter(quorum_space=qs)

        base_confidence = 0.60
        result = alerter._apply_quorum_boost(base_confidence, "", "bullish")

        assert result == base_confidence, (
            f"Expected no change for empty ticker, got {result}"
        )
        qs.get_contributor_count.assert_not_called()

    def test_quorum_space_exception_is_swallowed(self):
        """QuorumSpace raising an exception → confidence returned unchanged (graceful)."""
        qs = MagicMock()
        qs.get_contributor_count = MagicMock(side_effect=RuntimeError("db offline"))
        alerter = _make_alerter(quorum_space=qs)

        base_confidence = 0.70
        result = alerter._apply_quorum_boost(base_confidence, "AAPL", "bullish")

        assert result == base_confidence, (
            f"Exception should be swallowed: expected {base_confidence}, got {result}"
        )
