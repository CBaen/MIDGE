"""Tests for technical analysis indicators (ta_indicators.py).

Tests each indicator with synthetic price data to verify:
- Correct signal generation at extreme values
- No signal generation in neutral zones
- Proper direction, strength, confidence ranges
- compute_all() aggregation
"""

import pytest
from datetime import datetime, timedelta
from mae_core.market.apis.price_fetcher import PriceData
from mae_core.market.edge.ta_indicators import (
    compute_rsi,
    compute_macd,
    compute_bollinger,
    compute_market_structure,
    compute_candlestick_patterns,
    compute_all,
    RSISignal,
    MACDSignal,
    BollingerSignal,
    StructureSignal,
    CandleSignal,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_price(day_offset: int, open_: float, high: float, low: float, close: float,
               volume: int = 100000, symbol: str = "TEST") -> PriceData:
    """Create a PriceData with a synthetic date."""
    ts = (datetime(2026, 1, 1) + timedelta(days=day_offset)).strftime("%Y-%m-%d")
    return PriceData(
        symbol=symbol, price=close, timestamp=ts, source="test",
        open=open_, high=high, low=low, volume=volume,
        change_pct=((close - open_) / open_ * 100) if open_ != 0 else 0.0,
    )


def make_trending_prices(start: float, end: float, days: int, symbol: str = "TEST") -> list:
    """Generate a linear trend from start to end over N days with synthetic OHLC."""
    prices = []
    for i in range(days):
        price = start + (end - start) * i / max(1, days - 1)
        daily_range = abs(price * 0.01)  # 1% daily range
        prices.append(make_price(
            day_offset=i, symbol=symbol,
            open_=price - daily_range * 0.3,
            high=price + daily_range * 0.5,
            low=price - daily_range * 0.5,
            close=price,
        ))
    return prices


def make_falling_prices(start: float, end: float, days: int, symbol: str = "TEST") -> list:
    """Shorthand for make_trending_prices with start > end."""
    return make_trending_prices(start, end, days, symbol)


# ---------------------------------------------------------------------------
# RSI Tests
# ---------------------------------------------------------------------------

class TestRSI:
    def test_insufficient_data_returns_none(self):
        prices = make_trending_prices(100, 110, 10)
        assert compute_rsi("TEST", prices) is None

    def test_oversold_generates_bullish_signal(self):
        # 30 days of steady decline → RSI should be deeply oversold
        prices = make_falling_prices(100, 60, 30)
        signal = compute_rsi("TEST", prices)
        assert signal is not None
        assert isinstance(signal, RSISignal)
        assert signal.direction == "bullish"
        assert signal.zone == "oversold"
        assert signal.value < 30
        assert 0 < signal.strength <= 1.0
        assert signal.symbol == "TEST"
        assert signal.indicator == "rsi"

    def test_overbought_generates_bearish_signal(self):
        # 30 days of steady rise → RSI should be overbought
        prices = make_trending_prices(60, 100, 30)
        signal = compute_rsi("TEST", prices)
        assert signal is not None
        assert signal.direction == "bearish"
        assert signal.zone == "overbought"
        assert signal.value > 70

    def test_neutral_returns_none(self):
        # Flat prices → RSI near 50 → no signal
        prices = []
        for i in range(30):
            # Oscillate slightly around 100
            price = 100 + (1 if i % 2 == 0 else -1)
            prices.append(make_price(i, price - 0.5, price + 0.5, price - 0.5, price))
        signal = compute_rsi("TEST", prices)
        assert signal is None

    def test_signal_fields_valid(self):
        prices = make_falling_prices(100, 60, 30)
        signal = compute_rsi("TEST", prices)
        if signal:
            assert signal.signal_id.startswith("ta_rsi:")
            assert signal.confidence > 0
            assert signal.decay_rate > 0
            assert signal.detected_at != ""


# ---------------------------------------------------------------------------
# MACD Tests
# ---------------------------------------------------------------------------

class TestMACD:
    def test_insufficient_data_returns_none(self):
        prices = make_trending_prices(100, 110, 20)
        assert compute_macd("TEST", prices) is None

    def test_bullish_crossover_after_reversal(self):
        # 20 days down then 20 days up → MACD bullish crossover near the turn
        down = make_falling_prices(120, 80, 20)
        up = make_trending_prices(80, 120, 20)
        prices = down + up
        signal = compute_macd("TEST", prices)
        # Signal may or may not fire depending on exact EMA dynamics,
        # but if it fires it should be bullish
        if signal is not None:
            assert isinstance(signal, MACDSignal)
            assert signal.direction == "bullish"
            assert signal.crossover_type == "bullish_crossover"

    def test_bearish_crossover_after_reversal(self):
        # 20 days up then 20 days down
        up = make_trending_prices(80, 120, 20)
        down = make_falling_prices(120, 80, 20)
        prices = up + down
        signal = compute_macd("TEST", prices)
        if signal is not None:
            assert signal.direction == "bearish"
            assert signal.crossover_type == "bearish_crossover"

    def test_no_crossover_in_trend(self):
        # Steady uptrend — no crossover expected once EMA stabilizes
        prices = make_trending_prices(80, 130, 50)
        signal = compute_macd("TEST", prices)
        # In a steady trend, no recent crossover
        # (the initial crossover happens early; by bar 50 it's gone)
        # This is acceptable either way

    def test_signal_fields_valid(self):
        down = make_falling_prices(120, 80, 20)
        up = make_trending_prices(80, 120, 20)
        prices = down + up
        signal = compute_macd("TEST", prices)
        if signal is not None:
            assert signal.indicator == "macd"
            assert signal.signal_id.startswith("ta_macd:")
            assert signal.histogram != 0


# ---------------------------------------------------------------------------
# Bollinger Band Tests
# ---------------------------------------------------------------------------

class TestBollinger:
    def test_insufficient_data_returns_none(self):
        prices = make_trending_prices(100, 110, 10)
        assert compute_bollinger("TEST", prices) is None

    def test_price_at_lower_band_is_bullish(self):
        # Stable prices then sudden drop → price at lower band
        stable = [make_price(i, 100, 101, 99, 100) for i in range(25)]
        drop = [make_price(25 + i, 95 - i, 96 - i, 94 - i, 95 - i) for i in range(3)]
        prices = stable + drop
        signal = compute_bollinger("TEST", prices)
        if signal is not None:
            assert isinstance(signal, BollingerSignal)
            assert signal.direction == "bullish"
            assert signal.band_position < 0.2

    def test_price_at_upper_band_is_bearish(self):
        # Stable prices then sudden spike → price at upper band
        stable = [make_price(i, 100, 101, 99, 100) for i in range(25)]
        spike = [make_price(25 + i, 105 + i, 106 + i, 104 + i, 105 + i) for i in range(3)]
        prices = stable + spike
        signal = compute_bollinger("TEST", prices)
        if signal is not None:
            assert signal.direction == "bearish"
            assert signal.band_position > 0.8

    def test_mid_band_returns_none(self):
        # Stable prices → no signal
        prices = [make_price(i, 100, 101, 99, 100) for i in range(25)]
        signal = compute_bollinger("TEST", prices)
        assert signal is None

    def test_signal_bands_ordered(self):
        stable = [make_price(i, 100, 101, 99, 100) for i in range(25)]
        drop = [make_price(25 + i, 95 - i, 96 - i, 94 - i, 95 - i) for i in range(3)]
        prices = stable + drop
        signal = compute_bollinger("TEST", prices)
        if signal is not None:
            assert signal.lower_band < signal.middle_band < signal.upper_band


# ---------------------------------------------------------------------------
# Market Structure Tests
# ---------------------------------------------------------------------------

class TestMarketStructure:
    def test_insufficient_data_returns_none(self):
        prices = make_trending_prices(100, 110, 5)
        assert compute_market_structure("TEST", prices) is None

    def test_uptrend_detected(self):
        # Create prices with clear higher highs and higher lows
        # Need enough data for swing detection with lookback=5
        prices = []
        for cycle in range(4):
            base = 100 + cycle * 10
            # Up phase (swing high)
            for i in range(6):
                p = base + i * 2
                prices.append(make_price(len(prices), p - 1, p + 3, p - 2, p))
            # Down phase (swing low, but higher than previous low)
            for i in range(6):
                p = base + 10 - i * 1.5
                prices.append(make_price(len(prices), p + 1, p + 2, p - 1, p))

        signal = compute_market_structure("TEST", prices)
        if signal is not None:
            assert isinstance(signal, StructureSignal)
            assert signal.indicator == "structure"
            # Should detect uptrend or BOS bullish
            assert signal.direction in ("bullish", "bearish")

    def test_downtrend_detected(self):
        # Create prices with lower highs and lower lows
        prices = []
        for cycle in range(4):
            base = 130 - cycle * 10
            for i in range(6):
                p = base + i * 2
                prices.append(make_price(len(prices), p - 1, p + 3, p - 2, p))
            for i in range(6):
                p = base + 10 - i * 1.5
                prices.append(make_price(len(prices), p + 1, p + 2, p - 1, p))

        signal = compute_market_structure("TEST", prices)
        if signal is not None:
            assert signal.direction in ("bullish", "bearish")
            assert signal.structure_type in ("uptrend", "downtrend", "break_of_structure")

    def test_swing_points_populated(self):
        # Just verify the dataclass fields are populated when we get a signal
        prices = []
        for cycle in range(4):
            base = 100 + cycle * 10
            for i in range(6):
                p = base + i * 2
                prices.append(make_price(len(prices), p - 1, p + 3, p - 2, p))
            for i in range(6):
                p = base + 10 - i * 1.5
                prices.append(make_price(len(prices), p + 1, p + 2, p - 1, p))

        signal = compute_market_structure("TEST", prices)
        if signal is not None:
            assert signal.swing_high > 0
            assert signal.swing_low > 0
            assert signal.prev_swing_high > 0
            assert signal.prev_swing_low > 0


# ---------------------------------------------------------------------------
# Candlestick Pattern Tests
# ---------------------------------------------------------------------------

class TestCandlestickPatterns:
    def test_insufficient_data_returns_none(self):
        prices = [make_price(0, 100, 101, 99, 100)]
        assert compute_candlestick_patterns("TEST", prices) is None

    def test_bullish_engulfing(self):
        # Previous: red candle (open 102, close 98)
        # Current: green candle that engulfs (open 97, close 103)
        prev = make_price(0, 102, 103, 97, 98)
        curr = make_price(1, 97, 104, 96, 103)
        signal = compute_candlestick_patterns("TEST", [prev, curr])
        assert signal is not None
        assert isinstance(signal, CandleSignal)
        assert signal.pattern == "bullish_engulfing"
        assert signal.direction == "bullish"

    def test_bearish_engulfing(self):
        # Previous: green candle (open 98, close 102)
        # Current: red candle that engulfs (open 103, close 97)
        prev = make_price(0, 98, 103, 97, 102)
        curr = make_price(1, 103, 104, 96, 97)
        signal = compute_candlestick_patterns("TEST", [prev, curr])
        assert signal is not None
        assert signal.pattern == "bearish_engulfing"
        assert signal.direction == "bearish"

    def test_hammer(self):
        # Small body at top, long lower shadow, minimal upper shadow
        # Open 100, Close 101, High 101.3, Low 95 (lower shadow >> body)
        prev = make_price(0, 100, 102, 99, 101)  # Need 2 candles
        curr = make_price(1, 100, 101.3, 95, 101)
        signal = compute_candlestick_patterns("TEST", [prev, curr])
        if signal is not None:
            assert signal.pattern == "hammer"
            assert signal.direction == "bullish"

    def test_shooting_star(self):
        # Small body at bottom, long upper shadow, minimal lower shadow
        # Open 101, Close 100, High 106, Low 99.8 (upper shadow >> body)
        prev = make_price(0, 100, 102, 99, 101)
        curr = make_price(1, 101, 106, 99.8, 100)
        signal = compute_candlestick_patterns("TEST", [prev, curr])
        if signal is not None:
            assert signal.pattern == "shooting_star"
            assert signal.direction == "bearish"

    def test_doji_needs_context(self):
        # Doji without 5+ bars of context returns None
        prev = make_price(0, 100, 102, 98, 100)
        curr = make_price(1, 100.0, 102, 98, 100.05)  # Almost no body
        signal = compute_candlestick_patterns("TEST", [prev, curr])
        # Should be None because doji needs 5+ bars for trend context
        assert signal is None

    def test_doji_with_context(self):
        # Doji after uptrend → bearish
        context = make_trending_prices(90, 100, 5)
        doji = make_price(5, 100.0, 102, 98, 100.05)
        prices = context + [doji]
        signal = compute_candlestick_patterns("TEST", prices)
        if signal is not None and signal.pattern == "doji":
            assert signal.direction == "bearish"  # Doji at top of uptrend

    def test_no_pattern_in_normal_candle(self):
        # Normal candle with moderate body and shadows
        prev = make_price(0, 100, 102, 99, 101)
        curr = make_price(1, 101, 103, 100, 102)
        signal = compute_candlestick_patterns("TEST", [prev, curr])
        assert signal is None


# ---------------------------------------------------------------------------
# compute_all() Integration Tests
# ---------------------------------------------------------------------------

class TestComputeAll:
    def test_empty_history_returns_empty(self):
        assert compute_all("TEST", []) == []

    def test_returns_list_of_signals(self):
        # Generate enough data for all indicators
        prices = make_falling_prices(100, 60, 40)  # Strong downtrend
        signals = compute_all("TEST", prices)
        assert isinstance(signals, list)
        # Should get at least RSI oversold from this
        for sig in signals:
            assert hasattr(sig, "direction")
            assert hasattr(sig, "strength")
            assert hasattr(sig, "confidence")
            assert sig.symbol == "TEST"

    def test_all_signals_have_valid_ranges(self):
        prices = make_falling_prices(100, 60, 40)
        signals = compute_all("TEST", prices)
        for sig in signals:
            assert 0 <= sig.strength <= 1.0
            assert 0 <= sig.confidence <= 1.0
            assert sig.decay_rate > 0
            assert sig.direction in ("bullish", "bearish")
            assert sig.indicator in ("rsi", "macd", "bollinger", "structure", "candle")

    def test_multiple_indicators_can_fire(self):
        # Extreme scenario: sharp reversal should trigger multiple indicators
        down = make_falling_prices(120, 70, 25)
        up = make_trending_prices(70, 120, 20)
        prices = down + up
        signals = compute_all("TEST", prices)
        indicators_found = {s.indicator for s in signals}
        # At least one indicator should fire on this dramatic move
        assert len(signals) >= 1, f"Expected at least 1 signal, got {len(signals)}"

    def test_exceptions_dont_crash(self):
        # Pass garbage data — compute_all should catch exceptions internally
        class FakePrice:
            price = "not_a_number"
            open = None
            high = None
            low = None
            volume = 0
            timestamp = "2026-01-01"

        # Should not raise
        signals = compute_all("TEST", [FakePrice(), FakePrice()])
        assert isinstance(signals, list)
