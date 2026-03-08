"""Tests for vectorized TA indicator implementation.

Verifies numerical correctness and edge-case safety after numpy/pandas
vectorization. All tests compare against known expected values or assert
invariants that must hold regardless of implementation strategy.
"""

import math
import time
from datetime import datetime, timedelta

import numpy as np
import pytest

from mae_core.market.apis.price_fetcher import PriceData
from mae_core.market.edge.ta_indicators import (
    RSISignal,
    BollingerSignal,
    MACDSignal,
    compute_all,
    compute_bollinger,
    compute_macd,
    compute_rsi,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_price(
    day_offset: int,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: int = 100_000,
) -> PriceData:
    ts = (datetime(2026, 1, 1) + timedelta(days=day_offset)).strftime("%Y-%m-%d")
    return PriceData(
        symbol="TEST",
        price=close,
        timestamp=ts,
        source="test",
        open=open_,
        high=high,
        low=low,
        volume=volume,
        change_pct=((close - open_) / open_ * 100) if open_ != 0 else 0.0,
    )


def constant_prices(value: float, n: int) -> list:
    """N bars all at exactly the same price."""
    return [make_price(i, value, value, value, value) for i in range(n)]


def step_prices(low_val: float, high_val: float, step_at: int, total: int) -> list:
    """Price stays at low_val, then jumps to high_val at step_at."""
    prices = []
    for i in range(total):
        v = high_val if i >= step_at else low_val
        prices.append(make_price(i, v * 0.999, v * 1.001, v * 0.999, v))
    return prices


def linear_prices(start: float, end: float, n: int) -> list:
    return [
        make_price(
            i,
            start + (end - start) * i / max(n - 1, 1) - 0.01,
            start + (end - start) * i / max(n - 1, 1) + 0.5,
            start + (end - start) * i / max(n - 1, 1) - 0.5,
            start + (end - start) * i / max(n - 1, 1),
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# RSI numerical tests
# ---------------------------------------------------------------------------

class TestRSIVectorized:
    def test_rsi_all_up_days_approaches_100(self):
        """Constant up moves → RSI should be very high (near 100)."""
        prices = [make_price(i, 100 + i, 101 + i, 99 + i, 100 + i) for i in range(30)]
        signal = compute_rsi("TEST", prices)
        # All gains, no losses → RSI at or near 100 → overbought signal
        assert signal is not None
        assert signal.direction == "bearish"
        assert signal.value > 90

    def test_rsi_all_down_days_approaches_0(self):
        """Constant down moves → RSI should be very low (near 0)."""
        prices = [make_price(i, 100 - i, 101 - i, 99 - i, 100 - i) for i in range(30)]
        signal = compute_rsi("TEST", prices)
        assert signal is not None
        assert signal.direction == "bullish"
        assert signal.value < 10

    def test_rsi_value_in_valid_range(self):
        """RSI must always be in [0, 100]."""
        rng = np.random.default_rng(0)
        raw = rng.uniform(50, 150, 60).tolist()
        prices = [make_price(i, v * 0.99, v * 1.01, v * 0.99, v) for i, v in enumerate(raw)]
        signal = compute_rsi("TEST", prices)
        if signal is not None:
            assert 0 <= signal.value <= 100

    def test_rsi_exact_period_boundary(self):
        """Exactly period+1 prices is the minimum — should not raise."""
        prices = linear_prices(100, 60, 15)  # 15 bars, period=14 → just enough
        result = compute_rsi("TEST", prices, period=14)
        # May or may not produce a signal; must not raise
        assert result is None or isinstance(result, RSISignal)

    def test_rsi_metadata_keys_present(self):
        prices = linear_prices(100, 60, 30)
        signal = compute_rsi("TEST", prices)
        if signal is not None:
            assert "period" in signal.metadata
            assert "avg_gain" in signal.metadata
            assert "avg_loss" in signal.metadata


# ---------------------------------------------------------------------------
# Bollinger Band numerical tests
# ---------------------------------------------------------------------------

class TestBollingerVectorized:
    def test_constant_price_zero_std(self):
        """All-same prices → zero std → bands collapse → band_range == 0 → None."""
        prices = constant_prices(100.0, 25)
        result = compute_bollinger("TEST", prices)
        # band_range == 0 should return None (guard in code)
        assert result is None

    def test_upper_band_gt_middle_gt_lower(self):
        """Band ordering invariant must hold whenever a signal is returned."""
        # Stable prices then spike to trigger signal
        stable = [make_price(i, 100, 101, 99, 100) for i in range(25)]
        spike = [make_price(25 + i, 108 + i * 2, 110 + i * 2, 107 + i * 2, 109 + i * 2) for i in range(3)]
        signal = compute_bollinger("TEST", stable + spike)
        if signal is not None:
            assert signal.lower_band < signal.middle_band < signal.upper_band

    def test_band_position_clamped_0_to_1(self):
        """band_position must always be in [0, 1]."""
        # Extreme drop well below lower band
        stable = [make_price(i, 100, 101, 99, 100) for i in range(25)]
        crash = [make_price(25 + i, 70 - i * 5, 71 - i * 5, 69 - i * 5, 70 - i * 5) for i in range(4)]
        signal = compute_bollinger("TEST", stable + crash)
        if signal is not None:
            assert 0.0 <= signal.band_position <= 1.0

    def test_bandwidth_positive(self):
        """Bandwidth must be a positive number."""
        stable = [make_price(i, 100 + (i % 3) * 0.5, 101, 99, 100) for i in range(25)]
        signal = compute_bollinger("TEST", stable)
        if signal is not None:
            assert signal.bandwidth > 0

    def test_bollinger_std_matches_population_std(self):
        """The std_dev in metadata must match the population std of the last 20 closes."""
        closes = [100.0 + (i % 5) * 2.5 for i in range(25)]
        prices = [make_price(i, c - 0.5, c + 0.5, c - 0.5, c) for i, c in enumerate(closes)]
        signal = compute_bollinger("TEST", prices)
        if signal is not None:
            window = closes[-20:]
            mean = sum(window) / 20
            expected_std = math.sqrt(sum((x - mean) ** 2 for x in window) / 20)
            assert abs(signal.metadata["std_dev"] - round(expected_std, 4)) < 1e-3


# ---------------------------------------------------------------------------
# MACD numerical tests
# ---------------------------------------------------------------------------

class TestMACDVectorized:
    def test_macd_histogram_sign_matches_direction(self):
        """Histogram at crossover: positive → bullish, negative → bearish."""
        down = linear_prices(120, 80, 20)
        up = linear_prices(80, 120, 20)
        signal = compute_macd("TEST", down + up)
        if signal is not None:
            if signal.direction == "bullish":
                assert signal.histogram > 0
            else:
                assert signal.histogram < 0

    def test_macd_signal_fields_are_finite(self):
        """macd_value and signal_value must be finite floats."""
        prices = linear_prices(80, 130, 60)
        signal = compute_macd("TEST", prices)
        if signal is not None:
            assert math.isfinite(signal.macd_value)
            assert math.isfinite(signal.signal_value)

    def test_macd_strength_in_range(self):
        """Strength must be in [0, 1]."""
        down = linear_prices(120, 80, 20)
        up = linear_prices(80, 120, 20)
        signal = compute_macd("TEST", down + up)
        if signal is not None:
            assert 0.0 <= signal.strength <= 1.0


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_price_returns_none_all_indicators(self):
        prices = [make_price(0, 100, 101, 99, 100)]
        assert compute_rsi("TEST", prices) is None
        assert compute_macd("TEST", prices) is None
        assert compute_bollinger("TEST", prices) is None

    def test_empty_list_returns_none_all_indicators(self):
        assert compute_rsi("TEST", []) is None
        assert compute_macd("TEST", []) is None
        assert compute_bollinger("TEST", []) is None

    def test_compute_all_empty_returns_empty_list(self):
        assert compute_all("TEST", []) == []

    def test_all_same_price_no_rsi_crash(self):
        """Constant price: all gains = all losses = 0. Must not raise ZeroDivisionError."""
        prices = constant_prices(100.0, 30)
        result = compute_rsi("TEST", prices)
        # avg_loss == 0 → RSI = 100 → overbought signal or None; must not raise
        assert result is None or isinstance(result, RSISignal)

    def test_very_long_series_no_crash(self):
        """1000-bar series must run without error for all three indicators."""
        rng = np.random.default_rng(7)
        raw = rng.uniform(90, 110, 1000).tolist()
        prices = [make_price(i, v - 0.5, v + 0.5, v - 0.5, v) for i, v in enumerate(raw)]
        # Should not raise
        r = compute_rsi("TEST", prices)
        m = compute_macd("TEST", prices)
        b = compute_bollinger("TEST", prices)
        assert r is None or isinstance(r, RSISignal)
        assert m is None or isinstance(m, MACDSignal)
        assert b is None or isinstance(b, BollingerSignal)


# ---------------------------------------------------------------------------
# Performance smoke test
# ---------------------------------------------------------------------------

class TestPerformance:
    def test_100_iterations_under_5_seconds(self):
        """100 compute_rsi + compute_bollinger + compute_macd calls must finish in < 5s."""
        rng = np.random.default_rng(42)
        raw = rng.uniform(100, 200, 1000).tolist()
        prices = [make_price(i, v - 0.5, v + 0.5, v - 0.5, v) for i, v in enumerate(raw)]

        start = time.perf_counter()
        for _ in range(100):
            compute_rsi("BENCH", prices, 14)
            compute_bollinger("BENCH", prices, 20, 2.0)
            compute_macd("BENCH", prices, 12, 26, 9)
        elapsed = time.perf_counter() - start

        assert elapsed < 5.0, f"100 iterations took {elapsed:.2f}s — vectorization may not be working"
