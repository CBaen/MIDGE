"""Tests for OrderFlowDetector — intraday order flow imbalance detection.

All tests are self-contained. No real yfinance calls are made.
price_fetcher is mocked to return synthetic DataFrames with known patterns.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import List
from unittest.mock import MagicMock

import pandas as pd
import pytest

from mae_core.market.edge.order_flow_detector import (
    OrderFlowDetector,
    OrderFlowSignal,
    _std,
)
from mae_core.market.apis.price_fetcher import PriceData


# ── DataFrame helpers ──────────────────────────────────────────────────────────

def _make_df(bars: list, base_dt=None) -> pd.DataFrame:
    """
    Build a 5-min OHLCV DataFrame from a list of (open, high, low, close, volume) tuples.

    index is a UTC DatetimeIndex spaced 5 minutes apart.
    Column names are lowercase to match price_fetcher convention.
    """
    if base_dt is None:
        base_dt = datetime(2026, 3, 2, 9, 30)
    idx = [base_dt + timedelta(minutes=5 * i) for i in range(len(bars))]
    data = {
        "open":   [b[0] for b in bars],
        "high":   [b[1] for b in bars],
        "low":    [b[2] for b in bars],
        "close":  [b[3] for b in bars],
        "volume": [b[4] for b in bars],
    }
    return pd.DataFrame(data, index=pd.DatetimeIndex(idx))


def _flat_bars(n: int, price: float = 100.0, volume: int = 1000) -> list:
    """
    n bars with stable price and slightly varying volume (baseline filler).

    Volume alternates between 90% and 110% of the nominal value so that the
    rolling standard deviation is non-zero. The detector correctly skips
    degenerate windows where std < 1, which a perfectly uniform baseline would
    trigger — realistic market data always has volume variation.
    """
    result = []
    for i in range(n):
        vol = int(volume * (0.90 if i % 2 == 0 else 1.10))
        result.append((price, price + 0.10, price - 0.10, price, vol))
    return result


def _mock_fetcher_from_df(df: pd.DataFrame, daily_history=None) -> MagicMock:
    """Return a price_fetcher mock whose get_intraday_candles returns df."""
    fetcher = MagicMock()
    fetcher.get_intraday_candles.return_value = df
    if daily_history is not None:
        fetcher.get_daily_history.return_value = daily_history
    else:
        fetcher.get_daily_history.return_value = []
    return fetcher


def _make_daily_history(n: int = 25, volume: int = 500_000) -> list:
    """Synthetic daily history with uniform volume for baseline tests."""
    records = []
    for i in range(n):
        pd_mock = MagicMock()
        pd_mock.volume = volume
        records.append(pd_mock)
    return records


# ── TestVolumeBaseline ─────────────────────────────────────────────────────────

class TestVolumeBaseline:
    def test_baseline_set_from_daily_history(self):
        fetcher = MagicMock()
        fetcher.get_daily_history.return_value = _make_daily_history(25, volume=400_000)
        fetcher.get_intraday_candles.return_value = pd.DataFrame()

        detector = OrderFlowDetector(price_fetcher=fetcher)
        detector.update_baselines("AAPL")
        assert "AAPL" in detector._volume_baselines
        assert abs(detector._volume_baselines["AAPL"] - 400_000) < 1.0

    def test_baseline_uses_last_20_days(self):
        """25 records provided; only last 20 should be averaged."""
        records = _make_daily_history(5, volume=100_000) + _make_daily_history(20, volume=600_000)
        fetcher = MagicMock()
        fetcher.get_daily_history.return_value = records
        detector = OrderFlowDetector(price_fetcher=fetcher)
        detector.update_baselines("AAPL")
        # Average of last 20 entries (600k each)
        assert abs(detector._volume_baselines["AAPL"] - 600_000) < 1.0

    def test_baseline_skips_zero_volume(self):
        """Records with volume=0 should be excluded from average."""
        records = _make_daily_history(20, volume=0) + _make_daily_history(5, volume=300_000)
        fetcher = MagicMock()
        fetcher.get_daily_history.return_value = records
        detector = OrderFlowDetector(price_fetcher=fetcher)
        detector.update_baselines("AAPL")
        # Only the 5 non-zero records enter the average
        assert detector._volume_baselines.get("AAPL", 0.0) == 300_000.0

    def test_baseline_no_fetcher_does_not_crash(self):
        detector = OrderFlowDetector(price_fetcher=None)
        detector.update_baselines("AAPL")  # Must not raise
        assert "AAPL" not in detector._volume_baselines

    def test_baseline_empty_history_not_stored(self):
        fetcher = MagicMock()
        fetcher.get_daily_history.return_value = []
        detector = OrderFlowDetector(price_fetcher=fetcher)
        detector.update_baselines("AAPL")
        assert "AAPL" not in detector._volume_baselines


# ── TestImbalanceDetection ─────────────────────────────────────────────────────

class TestImbalanceDetection:
    def _detector_with_bars(self, bars: list) -> "tuple[OrderFlowDetector, MagicMock]":
        df = _make_df(bars)
        fetcher = _mock_fetcher_from_df(df)
        detector = OrderFlowDetector(price_fetcher=fetcher)
        return detector, fetcher

    def test_buying_pressure_detected(self):
        """High volume + close near high = bullish signal."""
        baseline = _flat_bars(20, price=100.0, volume=1000)
        # Spike bar: close at top of range, volume at 5x baseline (sigma >> 2)
        spike = [(100.0, 101.0, 99.0, 100.9, 5000)]  # close_pos ≈ 0.95
        bars = baseline + spike
        df = _make_df(bars)
        fetcher = _mock_fetcher_from_df(df)
        detector = OrderFlowDetector(price_fetcher=fetcher)
        signals = detector.detect_imbalance("AAPL")
        assert any(s.direction == "bullish" for s in signals)

    def test_selling_pressure_detected(self):
        """High volume + close near low = bearish signal."""
        baseline = _flat_bars(20, price=100.0, volume=1000)
        # Spike bar: close at bottom of range
        spike = [(100.0, 101.0, 99.0, 99.05, 5000)]  # close_pos ≈ 0.025
        bars = baseline + spike
        df = _make_df(bars)
        fetcher = _mock_fetcher_from_df(df)
        detector = OrderFlowDetector(price_fetcher=fetcher)
        signals = detector.detect_imbalance("AAPL")
        assert any(s.direction == "bearish" for s in signals)

    def test_no_signal_below_volume_threshold(self):
        """Normal volume bars produce no signals."""
        bars = _flat_bars(30, price=100.0, volume=1000)
        df = _make_df(bars)
        fetcher = _mock_fetcher_from_df(df)
        detector = OrderFlowDetector(price_fetcher=fetcher)
        signals = detector.detect_imbalance("AAPL")
        assert signals == []

    def test_mid_range_close_suppressed(self):
        """High volume but close in mid-range → no directional signal."""
        baseline = _flat_bars(20, price=100.0, volume=1000)
        # Mid-range close: (close - low) / (high - low) = 0.5
        spike = [(100.0, 101.0, 99.0, 100.0, 5000)]  # close_pos = 0.50
        bars = baseline + spike
        df = _make_df(bars)
        fetcher = _mock_fetcher_from_df(df)
        detector = OrderFlowDetector(price_fetcher=fetcher)
        signals = detector.detect_imbalance("AAPL")
        assert signals == []

    def test_no_fetcher_returns_empty(self):
        detector = OrderFlowDetector(price_fetcher=None)
        signals = detector.detect_imbalance("AAPL")
        assert signals == []


# ── TestStrengthCalculation ────────────────────────────────────────────────────

class TestStrengthCalculation:
    def _strength_for_zscore(self, zscore_multiplier: float) -> float:
        """
        Build bars where the spike bar has a known volume z-score.
        baseline: 20 bars at volume=1000 (mean=1000, std≈0 but add variance).
        Use varying volumes to get non-zero std, then place spike.
        """
        # baseline with slight variance: volumes 800-1200, mean=1000
        baseline_vols = [800, 900, 950, 1000, 1000, 1000, 1000, 1050, 1100, 1200] * 2
        baseline = [
            (100.0, 100.5, 99.5, 100.4, v) for v in baseline_vols
        ]
        vol_mean = sum(baseline_vols) / len(baseline_vols)
        vol_std = _std(baseline_vols)
        spike_vol = vol_mean + zscore_multiplier * vol_std
        spike = [(100.0, 101.0, 99.0, 100.9, int(spike_vol))]
        bars = baseline + spike
        df = _make_df(bars)
        fetcher = _mock_fetcher_from_df(df)
        detector = OrderFlowDetector(price_fetcher=fetcher)
        signals = detector.detect_imbalance("AAPL")
        return signals[0].strength if signals else 0.0

    def test_strength_two_sigma(self):
        strength = self._strength_for_zscore(2.0)
        # 2/4 = 0.5
        assert abs(strength - 0.5) < 0.15  # allow tolerance for rolling window edge effects

    def test_strength_four_sigma_caps_at_one(self):
        strength = self._strength_for_zscore(4.0)
        assert strength <= 1.0
        assert strength >= 0.8

    def test_strength_below_threshold_no_signal(self):
        strength = self._strength_for_zscore(1.5)
        # Below default threshold of 2.0 → no signal
        assert strength == 0.0

    def test_signal_strength_always_bounded(self):
        """Whatever the volume spike, strength must stay in [0, 1]."""
        baseline = _flat_bars(20, price=100.0, volume=1000)
        spike = [(100.0, 101.0, 99.0, 100.9, 999_999)]
        df = _make_df(baseline + spike)
        fetcher = _mock_fetcher_from_df(df)
        detector = OrderFlowDetector(price_fetcher=fetcher)
        signals = detector.detect_imbalance("AAPL")
        for sig in signals:
            assert 0.0 <= sig.strength <= 1.0


# ── TestClosePosition ──────────────────────────────────────────────────────────

class TestClosePosition:
    def test_close_near_high_bullish(self):
        """close_position > 0.70 + high volume → bullish."""
        baseline = _flat_bars(20, price=100.0, volume=1000)
        # close at 100.9 on 100-101 range → close_pos = (100.9-99)/2 = 0.95
        spike = [(100.0, 101.0, 99.0, 100.9, 6000)]
        df = _make_df(baseline + spike)
        fetcher = _mock_fetcher_from_df(df)
        detector = OrderFlowDetector(price_fetcher=fetcher)
        signals = detector.detect_imbalance("AAPL")
        assert any(s.direction == "bullish" for s in signals)
        bullish = next(s for s in signals if s.direction == "bullish")
        assert bullish.close_position_in_range > 0.70

    def test_close_near_low_bearish(self):
        """close_position < 0.30 + high volume → bearish."""
        baseline = _flat_bars(20, price=100.0, volume=1000)
        spike = [(100.0, 101.0, 99.0, 99.1, 6000)]  # close_pos = (99.1-99)/2 = 0.05
        df = _make_df(baseline + spike)
        fetcher = _mock_fetcher_from_df(df)
        detector = OrderFlowDetector(price_fetcher=fetcher)
        signals = detector.detect_imbalance("AAPL")
        assert any(s.direction == "bearish" for s in signals)

    def test_mid_range_close_no_signal(self):
        """close_position ≈ 0.50 → no directional signal even with high volume."""
        baseline = _flat_bars(20, price=100.0, volume=1000)
        spike = [(100.0, 101.0, 99.0, 100.0, 6000)]  # close_pos = 0.50
        df = _make_df(baseline + spike)
        fetcher = _mock_fetcher_from_df(df)
        detector = OrderFlowDetector(price_fetcher=fetcher)
        signals = detector.detect_imbalance("AAPL")
        assert signals == []

    def test_close_position_recorded_accurately(self):
        baseline = _flat_bars(20, price=100.0, volume=1000)
        spike = [(100.0, 101.0, 99.0, 100.9, 6000)]
        df = _make_df(baseline + spike)
        fetcher = _mock_fetcher_from_df(df)
        detector = OrderFlowDetector(price_fetcher=fetcher)
        signals = detector.detect_imbalance("AAPL")
        bullish = next((s for s in signals if s.direction == "bullish"), None)
        if bullish:
            expected = (100.9 - 99.0) / (101.0 - 99.0)
            assert abs(bullish.close_position_in_range - expected) < 0.01

    def test_zero_range_bar_skipped(self):
        """Bar with high == low (gap at open) must not produce ZeroDivisionError."""
        baseline = _flat_bars(20, price=100.0, volume=1000)
        zero_range = [(100.0, 100.0, 100.0, 100.0, 5000)]
        df = _make_df(baseline + zero_range)
        fetcher = _mock_fetcher_from_df(df)
        detector = OrderFlowDetector(price_fetcher=fetcher)
        signals = detector.detect_imbalance("AAPL")
        # Should not raise; zero-range bar produces no signal
        assert isinstance(signals, list)


# ── TestGracefulDegradation ────────────────────────────────────────────────────

class TestGracefulDegradation:
    def test_price_fetcher_none(self):
        detector = OrderFlowDetector(price_fetcher=None)
        assert detector.detect_imbalance("AAPL") == []

    def test_empty_candles_returns_empty(self):
        fetcher = MagicMock()
        fetcher.get_intraday_candles.return_value = pd.DataFrame()
        detector = OrderFlowDetector(price_fetcher=fetcher)
        assert detector.detect_imbalance("AAPL") == []

    def test_none_candles_returns_empty(self):
        fetcher = MagicMock()
        fetcher.get_intraday_candles.return_value = None
        detector = OrderFlowDetector(price_fetcher=fetcher)
        assert detector.detect_imbalance("AAPL") == []

    def test_insufficient_bars_returns_empty(self):
        """Fewer than min_bars candles — not enough for a z-score."""
        bars = _flat_bars(5, price=100.0, volume=1000)  # only 5 < default min_bars=10
        df = _make_df(bars)
        fetcher = _mock_fetcher_from_df(df)
        detector = OrderFlowDetector(price_fetcher=fetcher)
        assert detector.detect_imbalance("AAPL") == []

    def test_fetcher_exception_returns_empty(self):
        fetcher = MagicMock()
        fetcher.get_intraday_candles.side_effect = RuntimeError("network error")
        detector = OrderFlowDetector(price_fetcher=fetcher)
        assert detector.detect_imbalance("AAPL") == []

    def test_update_baselines_no_fetcher(self):
        detector = OrderFlowDetector(price_fetcher=None)
        detector.update_baselines("AAPL")  # Must not raise
        assert "AAPL" not in detector._volume_baselines

    def test_signal_has_ticker(self):
        baseline = _flat_bars(20, price=100.0, volume=1000)
        spike = [(100.0, 101.0, 99.0, 100.9, 6000)]
        df = _make_df(baseline + spike)
        fetcher = _mock_fetcher_from_df(df)
        detector = OrderFlowDetector(price_fetcher=fetcher)
        signals = detector.detect_imbalance("AAPL")
        for sig in signals:
            assert sig.ticker == "AAPL"


# ── TestStdUtility ─────────────────────────────────────────────────────────────

class TestStdUtility:
    """Unit tests for the internal _std() function."""

    def test_empty_list(self):
        assert _std([]) == 0.0

    def test_single_element(self):
        assert _std([42.0]) == 0.0

    def test_known_std(self):
        # Population std of [2, 4, 4, 4, 5, 5, 7, 9] = 2.0
        result = _std([2, 4, 4, 4, 5, 5, 7, 9])
        assert abs(result - 2.0) < 0.001

    def test_uniform_values(self):
        assert _std([5, 5, 5, 5]) == 0.0
