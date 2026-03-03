"""Tests for FractalResonanceDetector (Gift 7 — Fractal Market Resonance).

Council Gift — detects self-similar price patterns across multiple
timeframes (daily, weekly, monthly). No network calls — uses mock
price fetcher with deterministic data.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Optional

import pytest

from mae_core.market.edge.fractal_resonance import (
    FractalResonanceDetector,
    FractalResonanceResult,
    TimeframePattern,
)


@dataclass
class MockPriceBar:
    """Minimal price bar for testing."""

    open: float
    high: float
    low: float
    close: float


def _make_bullish_bars(n: int = 40, start: float = 100.0) -> List[MockPriceBar]:
    """Create bars with bullish market structure (HH + HL).

    Large oscillation (±15) with gentle upward drift (0.3/bar).
    The oscillation dominates so swing points are clearly detected,
    while the drift ensures each successive peak/trough is higher
    than the prior one (HH + HL = bullish structure).
    """
    import math
    bars = []
    for i in range(n):
        cycle = math.sin(i * math.pi / 4) * 15.0  # Large 8-bar oscillation
        drift = i * 0.3  # Gentle upward drift
        mid = start + drift + cycle
        bars.append(MockPriceBar(open=mid - 1, high=mid + 3, low=mid - 3, close=mid + 1))
    return bars


def _make_bearish_bars(n: int = 40, start: float = 200.0) -> List[MockPriceBar]:
    """Create bars with bearish market structure (LL + LH).

    Large oscillation with gentle downward drift.
    """
    import math
    bars = []
    for i in range(n):
        cycle = math.sin(i * math.pi / 4) * 15.0
        drift = -i * 0.3  # Gentle downward drift
        mid = start + drift + cycle
        bars.append(MockPriceBar(open=mid + 1, high=mid + 3, low=mid - 3, close=mid - 1))
    return bars


def _make_sideways_bars(n: int = 30, center: float = 150.0) -> List[MockPriceBar]:
    """Create sideways/neutral bars."""
    bars = []
    for i in range(n):
        offset = (i % 3 - 1) * 1.0
        o = center + offset
        h = center + offset + 2.0
        l = center + offset - 2.0
        c = center + offset + 0.5
        bars.append(MockPriceBar(open=o, high=h, low=l, close=c))
    return bars


class _MockPriceFetcher:
    """Mock price fetcher that returns deterministic data per timeframe."""

    def __init__(
        self,
        daily=None,
        weekly=None,
        monthly=None,
    ):
        self._daily = daily
        self._weekly = weekly
        self._monthly = monthly

    def get_daily_history(self, ticker, days=90):
        return self._daily

    def get_weekly_history(self, ticker, weeks=52):
        return self._weekly

    def get_monthly_history(self, ticker, months=24):
        return self._monthly


class TestConstruction:
    """Basic construction and interface."""

    def test_constructs_without_args(self):
        frd = FractalResonanceDetector()
        assert frd is not None

    def test_constructs_with_price_fetcher(self):
        frd = FractalResonanceDetector(price_fetcher=object())
        assert frd._price_fetcher is not None

    def test_statistics(self):
        frd = FractalResonanceDetector()
        stats = frd.get_statistics()
        assert stats["timeframes_analyzed"] == 3
        assert stats["has_price_fetcher"] is False

    def test_returns_none_without_price_fetcher(self):
        frd = FractalResonanceDetector(price_fetcher=None)
        result = frd.detect_resonance("AAPL")
        assert result is None


class TestResonanceDetection:
    """Test resonance detection with mock price data."""

    def test_all_bullish_produces_high_resonance(self):
        pf = _MockPriceFetcher(
            daily=_make_bullish_bars(),
            weekly=_make_bullish_bars(),
            monthly=_make_bullish_bars(),
        )
        frd = FractalResonanceDetector(price_fetcher=pf)
        result = frd.detect_resonance("AAPL")
        assert result is not None
        assert result.resonance_score >= 0.7
        assert result.dominant_direction == "bullish"

    def test_all_bearish_produces_high_resonance(self):
        pf = _MockPriceFetcher(
            daily=_make_bearish_bars(),
            weekly=_make_bearish_bars(),
            monthly=_make_bearish_bars(),
        )
        frd = FractalResonanceDetector(price_fetcher=pf)
        result = frd.detect_resonance("AAPL")
        assert result is not None
        assert result.resonance_score >= 0.7
        assert result.dominant_direction == "bearish"

    def test_mixed_directions_lower_resonance(self):
        pf = _MockPriceFetcher(
            daily=_make_bullish_bars(),
            weekly=_make_bearish_bars(),
            monthly=_make_bullish_bars(),
        )
        frd = FractalResonanceDetector(price_fetcher=pf)
        result = frd.detect_resonance("AAPL")
        assert result is not None
        # Mixed should have lower resonance than all-aligned
        assert result.resonance_score <= 0.8

    def test_sideways_produces_neutral(self):
        pf = _MockPriceFetcher(
            daily=_make_sideways_bars(),
            weekly=_make_sideways_bars(),
            monthly=_make_sideways_bars(),
        )
        frd = FractalResonanceDetector(price_fetcher=pf)
        result = frd.detect_resonance("AAPL")
        assert result is not None
        assert result.dominant_direction == "neutral"

    def test_two_timeframes_sufficient(self):
        """Only 2 of 3 timeframes available should still produce a result."""
        pf = _MockPriceFetcher(
            daily=_make_bullish_bars(),
            weekly=_make_bullish_bars(),
            monthly=None,
        )
        frd = FractalResonanceDetector(price_fetcher=pf)
        result = frd.detect_resonance("AAPL")
        assert result is not None
        assert len(result.patterns_by_timeframe) == 2

    def test_one_timeframe_insufficient(self):
        """Only 1 timeframe available should return None."""
        pf = _MockPriceFetcher(
            daily=_make_bullish_bars(),
            weekly=None,
            monthly=None,
        )
        frd = FractalResonanceDetector(price_fetcher=pf)
        result = frd.detect_resonance("AAPL")
        assert result is None

    def test_insufficient_data_returns_none(self):
        """Too few bars should return None."""
        pf = _MockPriceFetcher(
            daily=[MockPriceBar(100, 101, 99, 100)] * 5,
            weekly=[MockPriceBar(100, 101, 99, 100)] * 5,
            monthly=None,
        )
        frd = FractalResonanceDetector(price_fetcher=pf)
        result = frd.detect_resonance("AAPL")
        # 5 bars is below the minimum (10)
        assert result is None


class TestTimeframePattern:
    """Test TimeframePattern dataclass."""

    def test_fields_accessible(self):
        p = TimeframePattern(
            timeframe="1d",
            structure="bullish",
            pattern_type="breakout",
            strength=0.8,
            swing_points=6,
        )
        assert p.timeframe == "1d"
        assert p.structure == "bullish"
        assert p.strength == 0.8


class TestFractalResonanceResult:
    """Test FractalResonanceResult dataclass."""

    def test_fields_accessible(self):
        r = FractalResonanceResult(
            ticker="AAPL",
            resonance_score=0.85,
            aligned_timeframes=3,
            dominant_direction="bullish",
        )
        assert r.ticker == "AAPL"
        assert r.resonance_score == 0.85

    def test_default_patterns_dict(self):
        r = FractalResonanceResult(
            ticker="X",
            resonance_score=0.0,
            aligned_timeframes=0,
            dominant_direction="neutral",
        )
        assert r.patterns_by_timeframe == {}


class TestSwingPointDetection:
    """Test the static swing point detection helper."""

    def test_finds_swing_highs(self):
        values = [10, 12, 15, 12, 10, 8, 11, 14, 11, 9, 7, 10, 13, 10, 8]
        highs = FractalResonanceDetector._find_swing_points(
            values, is_high=True, lookback=2
        )
        assert len(highs) > 0

    def test_finds_swing_lows(self):
        values = [10, 8, 6, 8, 10, 12, 9, 6, 9, 11, 13, 10, 7, 10, 12]
        lows = FractalResonanceDetector._find_swing_points(
            values, is_high=False, lookback=2
        )
        assert len(lows) > 0

    def test_empty_values_returns_empty(self):
        highs = FractalResonanceDetector._find_swing_points(
            [], is_high=True, lookback=2
        )
        assert highs == []


class TestStructureClassification:
    """Test market structure classification logic."""

    def test_higher_highs_higher_lows_bullish(self):
        result = FractalResonanceDetector._classify_structure(
            swing_highs=[100, 105, 110],
            swing_lows=[90, 93, 96],
        )
        assert result == "bullish"

    def test_lower_lows_lower_highs_bearish(self):
        result = FractalResonanceDetector._classify_structure(
            swing_highs=[110, 105, 100],
            swing_lows=[96, 93, 90],
        )
        assert result == "bearish"

    def test_insufficient_swings_neutral(self):
        result = FractalResonanceDetector._classify_structure(
            swing_highs=[100],
            swing_lows=[90],
        )
        assert result == "neutral"
