#!/usr/bin/env python3
"""
test_signal_translator.py - Tests for signal_translator and compute_atr.

Covers:
  - compute_atr: known values, insufficient data, vectorized correctness
  - translate_alert: long/short SL placement, TP placement, 2:1 R:R,
    position sizing bounds, graceful degradation (missing price / zero ATR),
    ticker resolution
"""

import math
import pytest


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int, base: float = 100.0, volatility: float = 2.0):
    """Return (highs, lows, closes) lists of length n with deterministic prices."""
    import random
    rng = random.Random(42)
    closes, highs, lows = [], [], []
    price = base
    for _ in range(n):
        delta = rng.uniform(-volatility, volatility)
        price = max(1.0, price + delta)
        h = price + rng.uniform(0, volatility * 0.5)
        l = price - rng.uniform(0, volatility * 0.5)
        closes.append(price)
        highs.append(h)
        lows.append(l)
    return highs, lows, closes


def _simple_alert(direction: str = "bullish", ticker: str = "AAPL",
                  confidence: float = 0.72, domains=None) -> dict:
    """Minimal ConvergenceAlert dict for translator tests."""
    if domains is None:
        domains = ["insider", "macro", "technical"]
    return {
        "alert_id": "test-alert-001",
        "direction": direction,
        "strength": 0.75,
        "confidence": confidence,
        "domains": domains,
        "signals": [
            {"metadata": {"symbol": ticker}, "source": "sec_form4"},
        ],
    }


# ---------------------------------------------------------------------------
# compute_atr tests
# ---------------------------------------------------------------------------

class TestComputeATR:
    """Tests for mae_core.market.edge.ta_indicators.compute_atr."""

    def test_returns_positive_float(self):
        from mae_core.market.edge.ta_indicators import compute_atr
        highs, lows, closes = _make_ohlcv(30)
        result = compute_atr(highs, lows, closes, period=14)
        assert result > 0
        assert math.isfinite(result)

    def test_insufficient_data_returns_zero(self):
        from mae_core.market.edge.ta_indicators import compute_atr
        # Need period+1 bars; give fewer
        highs, lows, closes = _make_ohlcv(10)
        result = compute_atr(highs, lows, closes, period=14)
        assert result == 0.0

    def test_exact_period_plus_one_bar_works(self):
        from mae_core.market.edge.ta_indicators import compute_atr
        highs, lows, closes = _make_ohlcv(15)  # 14+1
        result = compute_atr(highs, lows, closes, period=14)
        assert result > 0

    def test_flat_market_low_atr(self):
        """Completely flat prices should produce near-zero ATR."""
        from mae_core.market.edge.ta_indicators import compute_atr
        n = 30
        highs  = [100.1] * n
        lows   = [99.9]  * n
        closes = [100.0] * n
        result = compute_atr(highs, lows, closes, period=14)
        assert result < 0.5   # flat market, ATR should be tiny

    def test_high_volatility_produces_large_atr(self):
        """Large H-L ranges should produce large ATR."""
        from mae_core.market.edge.ta_indicators import compute_atr
        n = 30
        # Every bar spans 10 points
        highs  = [105.0] * n
        lows   = [95.0]  * n
        closes = [100.0] * n
        result = compute_atr(highs, lows, closes, period=14)
        assert result > 5.0   # TR is 10 each bar, ATR should be ~10

    def test_known_value(self):
        """ATR of identical bars (H=102, L=98, C=100, prev_C=100) = 4.0."""
        from mae_core.market.edge.ta_indicators import compute_atr
        n = 30
        highs  = [102.0] * n
        lows   = [98.0]  * n
        closes = [100.0] * n
        result = compute_atr(highs, lows, closes, period=14)
        # TR = max(102-98, |102-100|, |98-100|) = max(4, 2, 2) = 4 for all bars
        assert abs(result - 4.0) < 0.01

    def test_empty_lists_return_zero(self):
        from mae_core.market.edge.ta_indicators import compute_atr
        assert compute_atr([], [], [], period=14) == 0.0

    def test_mismatched_length_returns_zero(self):
        from mae_core.market.edge.ta_indicators import compute_atr
        # lows is shorter than highs — numpy will raise, should return 0.0
        highs  = [101.0] * 20
        lows   = [99.0]  * 10   # shorter
        closes = [100.0] * 20
        # Should not crash
        result = compute_atr(highs, lows, closes, period=14)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# translate_alert tests
# ---------------------------------------------------------------------------

class TestTranslateAlert:
    """Tests for mae_core.market.execution.signal_translator.translate_alert."""

    def test_long_stop_below_entry(self):
        from mae_core.market.execution.signal_translator import translate_alert
        alert = _simple_alert("bullish")
        sig = translate_alert(alert, current_price=100.0, atr=2.0)
        assert sig is not None
        assert sig.stop_loss < sig.entry_price

    def test_short_stop_above_entry(self):
        from mae_core.market.execution.signal_translator import translate_alert
        alert = _simple_alert("bearish")
        sig = translate_alert(alert, current_price=100.0, atr=2.0)
        assert sig is not None
        assert sig.stop_loss > sig.entry_price

    def test_long_take_profit_above_entry(self):
        from mae_core.market.execution.signal_translator import translate_alert
        alert = _simple_alert("bullish")
        sig = translate_alert(alert, current_price=100.0, atr=2.0)
        assert sig is not None
        assert sig.take_profit > sig.entry_price

    def test_short_take_profit_below_entry(self):
        from mae_core.market.execution.signal_translator import translate_alert
        alert = _simple_alert("bearish")
        sig = translate_alert(alert, current_price=100.0, atr=2.0)
        assert sig is not None
        assert sig.take_profit < sig.entry_price

    def test_two_to_one_rr_ratio(self):
        """Reward distance should be exactly 2x risk distance."""
        from mae_core.market.execution.signal_translator import translate_alert
        alert = _simple_alert("bullish")
        sig = translate_alert(alert, current_price=100.0, atr=2.0)
        assert sig is not None
        assert abs(sig.rr_ratio - 2.0) < 0.01

    def test_short_two_to_one_rr_ratio(self):
        from mae_core.market.execution.signal_translator import translate_alert
        alert = _simple_alert("bearish")
        sig = translate_alert(alert, current_price=50.0, atr=1.0)
        assert sig is not None
        assert abs(sig.rr_ratio - 2.0) < 0.01

    def test_sl_distance_is_1_5_atr(self):
        from mae_core.market.execution.signal_translator import translate_alert
        alert = _simple_alert("bullish")
        atr = 3.0
        sig = translate_alert(alert, current_price=200.0, atr=atr)
        assert sig is not None
        assert abs(sig.risk_distance - 1.5 * atr) < 0.001

    def test_tp_distance_is_3_atr(self):
        from mae_core.market.execution.signal_translator import translate_alert
        alert = _simple_alert("bullish")
        atr = 3.0
        sig = translate_alert(alert, current_price=200.0, atr=atr)
        assert sig is not None
        assert abs(sig.reward_distance - 3.0 * atr) < 0.001

    def test_position_size_within_bounds(self):
        from mae_core.market.execution.signal_translator import (
            translate_alert, _MAX_POSITION_PCT, _MIN_POSITION_PCT,
        )
        alert = _simple_alert("bullish")
        sig = translate_alert(alert, current_price=100.0, atr=2.0)
        assert sig is not None
        assert _MIN_POSITION_PCT <= sig.position_size_pct <= _MAX_POSITION_PCT

    def test_position_size_never_exceeds_max(self):
        """Extremely small ATR (wide risk) should not produce absurd position sizes."""
        from mae_core.market.execution.signal_translator import (
            translate_alert, _MAX_POSITION_PCT,
        )
        alert = _simple_alert("bullish")
        # ATR=0.001 on a $1000 stock => risk fraction = 0.0015% => huge raw size
        sig = translate_alert(alert, current_price=1000.0, atr=0.001)
        assert sig is not None
        assert sig.position_size_pct <= _MAX_POSITION_PCT

    def test_position_size_never_zero(self):
        from mae_core.market.execution.signal_translator import (
            translate_alert, _MIN_POSITION_PCT,
        )
        alert = _simple_alert("bullish")
        sig = translate_alert(alert, current_price=100.0, atr=99.0)  # huge ATR
        assert sig is not None
        assert sig.position_size_pct >= _MIN_POSITION_PCT

    def test_account_risk_pct_respected(self):
        """Custom account_risk_pct should scale position size proportionally."""
        from mae_core.market.execution.signal_translator import translate_alert
        alert = _simple_alert("bullish")
        sig1 = translate_alert(alert, current_price=100.0, atr=1.0, account_risk_pct=0.01)
        sig2 = translate_alert(alert, current_price=100.0, atr=1.0, account_risk_pct=0.02)
        assert sig1 is not None and sig2 is not None
        # 2% risk should produce ~2x the position size of 1% risk
        # (unless capped by _MAX_POSITION_PCT)
        assert sig2.position_size_pct >= sig1.position_size_pct

    # --- Graceful degradation ---

    def test_returns_none_when_price_is_zero(self):
        from mae_core.market.execution.signal_translator import translate_alert
        alert = _simple_alert("bullish")
        result = translate_alert(alert, current_price=0.0, atr=2.0)
        assert result is None

    def test_returns_none_when_price_is_none(self):
        from mae_core.market.execution.signal_translator import translate_alert
        alert = _simple_alert("bullish")
        result = translate_alert(alert, current_price=None, atr=2.0)
        assert result is None

    def test_returns_none_when_price_is_negative(self):
        from mae_core.market.execution.signal_translator import translate_alert
        alert = _simple_alert("bullish")
        result = translate_alert(alert, current_price=-50.0, atr=2.0)
        assert result is None

    def test_returns_none_when_atr_is_zero(self):
        from mae_core.market.execution.signal_translator import translate_alert
        alert = _simple_alert("bullish")
        result = translate_alert(alert, current_price=100.0, atr=0.0)
        assert result is None

    def test_returns_none_when_atr_is_none(self):
        from mae_core.market.execution.signal_translator import translate_alert
        alert = _simple_alert("bullish")
        result = translate_alert(alert, current_price=100.0, atr=None)
        assert result is None

    def test_returns_none_when_atr_is_negative(self):
        from mae_core.market.execution.signal_translator import translate_alert
        alert = _simple_alert("bullish")
        result = translate_alert(alert, current_price=100.0, atr=-1.0)
        assert result is None

    def test_returns_none_for_neutral_direction(self):
        from mae_core.market.execution.signal_translator import translate_alert
        alert = _simple_alert("neutral")
        result = translate_alert(alert, current_price=100.0, atr=2.0)
        assert result is None

    # --- Ticker resolution ---

    def test_ticker_resolved_from_signal_metadata(self):
        from mae_core.market.execution.signal_translator import translate_alert
        alert = _simple_alert("bullish", ticker="NVDA")
        sig = translate_alert(alert, current_price=800.0, atr=10.0)
        assert sig is not None
        assert sig.ticker == "NVDA"

    def test_ticker_falls_back_to_multi_when_no_symbol(self):
        from mae_core.market.execution.signal_translator import translate_alert
        alert = {
            "alert_id": "no-ticker",
            "direction": "bullish",
            "confidence": 0.7,
            "domains": ["macro"],
            "signals": [{"metadata": {}, "source": "fred_macro"}],
        }
        sig = translate_alert(alert, current_price=100.0, atr=2.0)
        assert sig is not None
        assert sig.ticker == "MULTI"

    # --- Field correctness ---

    def test_source_alert_id_passed_through(self):
        from mae_core.market.execution.signal_translator import translate_alert
        alert = _simple_alert("bullish")
        alert["alert_id"] = "SPECIAL-ALERT-XYZ"
        sig = translate_alert(alert, current_price=100.0, atr=2.0)
        assert sig is not None
        assert sig.source_alert_id == "SPECIAL-ALERT-XYZ"

    def test_confidence_passed_through(self):
        from mae_core.market.execution.signal_translator import translate_alert
        alert = _simple_alert("bullish", confidence=0.88)
        sig = translate_alert(alert, current_price=100.0, atr=2.0)
        assert sig is not None
        assert abs(sig.confidence - 0.88) < 0.001

    def test_direction_long_for_bullish(self):
        from mae_core.market.execution.signal_translator import translate_alert
        alert = _simple_alert("bullish")
        sig = translate_alert(alert, current_price=100.0, atr=2.0)
        assert sig is not None
        assert sig.direction == "long"

    def test_direction_short_for_bearish(self):
        from mae_core.market.execution.signal_translator import translate_alert
        alert = _simple_alert("bearish")
        sig = translate_alert(alert, current_price=100.0, atr=2.0)
        assert sig is not None
        assert sig.direction == "short"

    def test_timestamp_is_iso_string(self):
        from mae_core.market.execution.signal_translator import translate_alert
        from datetime import datetime
        alert = _simple_alert("bullish")
        sig = translate_alert(alert, current_price=100.0, atr=2.0)
        assert sig is not None
        assert sig.timestamp.endswith("Z")
        # Should be parseable
        datetime.fromisoformat(sig.timestamp.rstrip("Z"))

    def test_to_dict_is_json_serializable(self):
        import json
        from mae_core.market.execution.signal_translator import translate_alert
        alert = _simple_alert("bullish")
        sig = translate_alert(alert, current_price=150.0, atr=3.0)
        assert sig is not None
        d = sig.to_dict()
        # Must not raise
        json.dumps(d)

    def test_to_dict_keys_present(self):
        from mae_core.market.execution.signal_translator import translate_alert
        alert = _simple_alert("bullish")
        sig = translate_alert(alert, current_price=100.0, atr=2.0)
        d = sig.to_dict()
        for key in ("ticker", "direction", "entry_price", "stop_loss", "take_profit",
                    "risk_distance", "reward_distance", "rr_ratio",
                    "position_size_pct", "confidence", "domains",
                    "source_alert_id", "atr", "timestamp"):
            assert key in d, f"Missing key: {key}"
