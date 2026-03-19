"""strategy_library.py - 25 math-first strategies for Crypto Pattern Convergence.

Six families: RSI (5), MACD (4), Bollinger (4), Structure (4), Volume (4), MA (4).
Min 50 bars; 200-EMA strategies need 210. confidence=0.55 is prior (backtester overwrites).
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Optional

import numpy as np
import pandas as pd

from mae_core.market.strategies.models import StrategyResult

_DEFAULT_CONFIDENCE = 0.55
_MIN_BARS = 50


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _atr_stops(
    df: pd.DataFrame,
    direction: str,
    atr_sl_mult: float = 1.5,
    atr_tp_mult: float = 3.0,
) -> tuple[float, float]:
    """Compute ATR-based stop loss and take profit from the last close."""
    from mae_core.market.edge.ta_structure import compute_atr

    if len(df) < 15:
        return 0.0, 0.0
    highs = df["High"].tolist()
    lows = df["Low"].tolist()
    closes = df["Close"].tolist()
    atr = compute_atr(highs, lows, closes, period=14)
    if atr <= 0:
        return 0.0, 0.0
    last_close = closes[-1]
    if direction == "bullish":
        return last_close - atr * atr_sl_mult, last_close + atr * atr_tp_mult
    else:
        return last_close + atr * atr_sl_mult, last_close - atr * atr_tp_mult


def _rsi_series(closes: pd.Series, period: int) -> pd.Series:
    """Wilder RSI via ewm — identical method to ta_indicators.py."""
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _now() -> str:
    return datetime.now().isoformat()


# ---------------------------------------------------------------------------
# RSI Family
# ---------------------------------------------------------------------------

def rsi_oversold_14(symbol: str, df: pd.DataFrame) -> Optional[StrategyResult]:
    """RSI(14) < 20 → bullish (extreme oversold on the standard period)."""
    if len(df) < _MIN_BARS:
        return None
    rsi = _rsi_series(df["Close"], 14)
    last = rsi.iloc[-1]
    if pd.isna(last) or last >= 20:
        return None
    strength = round((20 - last) / 20, 4)
    sl, tp = _atr_stops(df, "bullish")
    return StrategyResult(
        strategy_name="rsi_oversold_14",
        symbol=symbol,
        direction="bullish",
        signal=1,
        stop_loss=sl,
        take_profit=tp,
        strength=strength,
        confidence=_DEFAULT_CONFIDENCE,
        detected_at=_now(),
        metadata={"rsi": round(last, 2), "threshold": 20},
    )


def rsi_oversold_2(symbol: str, df: pd.DataFrame) -> Optional[StrategyResult]:
    """RSI(2) < 10 → bullish (short-term mean reversion on the 2-period RSI)."""
    if len(df) < _MIN_BARS:
        return None
    rsi = _rsi_series(df["Close"], 2)
    last = rsi.iloc[-1]
    if pd.isna(last) or last >= 10:
        return None
    strength = round((10 - last) / 10, 4)
    sl, tp = _atr_stops(df, "bullish")
    return StrategyResult(
        strategy_name="rsi_oversold_2",
        symbol=symbol,
        direction="bullish",
        signal=1,
        stop_loss=sl,
        take_profit=tp,
        strength=strength,
        confidence=_DEFAULT_CONFIDENCE,
        detected_at=_now(),
        metadata={"rsi2": round(last, 2), "threshold": 10},
    )


def rsi_overbought_14(symbol: str, df: pd.DataFrame) -> Optional[StrategyResult]:
    """RSI(14) > 80 → bearish (extreme overbought on the standard period)."""
    if len(df) < _MIN_BARS:
        return None
    rsi = _rsi_series(df["Close"], 14)
    last = rsi.iloc[-1]
    if pd.isna(last) or last <= 80:
        return None
    strength = round((last - 80) / 20, 4)
    sl, tp = _atr_stops(df, "bearish")
    return StrategyResult(
        strategy_name="rsi_overbought_14",
        symbol=symbol,
        direction="bearish",
        signal=-1,
        stop_loss=sl,
        take_profit=tp,
        strength=strength,
        confidence=_DEFAULT_CONFIDENCE,
        detected_at=_now(),
        metadata={"rsi": round(last, 2), "threshold": 80},
    )


def rsi_momentum_shift(symbol: str, df: pd.DataFrame) -> Optional[StrategyResult]:
    """RSI(14) was below 30, now rising above 40 → bullish momentum restoration."""
    if len(df) < _MIN_BARS:
        return None
    rsi = _rsi_series(df["Close"], 14).dropna()
    if len(rsi) < 5:
        return None
    prev = rsi.iloc[-2]
    last = rsi.iloc[-1]
    # Must have been below 30 at some point in the last 10 bars, now above 40
    recent_low = rsi.iloc[-10:].min()
    if not (recent_low < 30 and prev < 40 and last >= 40):
        return None
    strength = round(min((last - 40) / 30, 1.0), 4)
    sl, tp = _atr_stops(df, "bullish")
    return StrategyResult(
        strategy_name="rsi_momentum_shift",
        symbol=symbol,
        direction="bullish",
        signal=1,
        stop_loss=sl,
        take_profit=tp,
        strength=strength,
        confidence=_DEFAULT_CONFIDENCE,
        detected_at=_now(),
        metadata={"rsi_now": round(last, 2), "rsi_recent_low": round(recent_low, 2)},
    )


def rsi_divergence(symbol: str, df: pd.DataFrame) -> Optional[StrategyResult]:
    """Bullish RSI divergence: price makes new 14-bar low, RSI does not."""
    if len(df) < _MIN_BARS:
        return None
    closes = df["Close"]
    rsi = _rsi_series(closes, 14).dropna()
    if len(rsi) < 14:
        return None
    # Compare last bar to 14 bars ago
    price_low_now = closes.iloc[-1]
    price_low_prev = closes.iloc[-14:-1].min()
    rsi_now = rsi.iloc[-1]
    rsi_prev = rsi.iloc[-14:-1].min()
    # Price lower low but RSI higher low
    if not (price_low_now < price_low_prev and rsi_now > rsi_prev):
        return None
    price_diff = (price_low_prev - price_low_now) / (price_low_prev + 1e-9)
    rsi_improvement = rsi_now - rsi_prev
    strength = round(min(price_diff * 5 + rsi_improvement / 30, 1.0), 4)
    sl, tp = _atr_stops(df, "bullish")
    return StrategyResult(
        strategy_name="rsi_divergence",
        symbol=symbol,
        direction="bullish",
        signal=1,
        stop_loss=sl,
        take_profit=tp,
        strength=strength,
        confidence=_DEFAULT_CONFIDENCE,
        detected_at=_now(),
        metadata={
            "price_low_now": round(price_low_now, 4),
            "price_low_prev": round(price_low_prev, 4),
            "rsi_now": round(rsi_now, 2),
            "rsi_prev_low": round(rsi_prev, 2),
        },
    )


# ---------------------------------------------------------------------------
# MACD Family
# ---------------------------------------------------------------------------

def _macd_lines(closes: pd.Series, fast: int, slow: int, signal: int):
    """Return (macd_line, signal_line, histogram) as pd.Series."""
    ema_fast = closes.ewm(span=fast, adjust=False).mean()
    ema_slow = closes.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist


def macd_crossover_std(symbol: str, df: pd.DataFrame) -> Optional[StrategyResult]:
    """MACD(12,26,9) signal line cross. Bullish when MACD crosses above signal."""
    if len(df) < _MIN_BARS:
        return None
    macd, sig, _ = _macd_lines(df["Close"], 12, 26, 9)
    if len(macd) < 2:
        return None
    prev_diff = macd.iloc[-2] - sig.iloc[-2]
    curr_diff = macd.iloc[-1] - sig.iloc[-1]
    if not (prev_diff < 0 and curr_diff >= 0):
        return None
    strength = round(min(abs(curr_diff) / (df["Close"].iloc[-1] * 0.001 + 1e-9), 1.0), 4)
    sl, tp = _atr_stops(df, "bullish")
    return StrategyResult(
        strategy_name="macd_crossover_std",
        symbol=symbol,
        direction="bullish",
        signal=1,
        stop_loss=sl,
        take_profit=tp,
        strength=strength,
        confidence=_DEFAULT_CONFIDENCE,
        detected_at=_now(),
        metadata={"macd": round(macd.iloc[-1], 6), "signal_line": round(sig.iloc[-1], 6)},
    )


def macd_crossover_fast(symbol: str, df: pd.DataFrame) -> Optional[StrategyResult]:
    """MACD(5,13,5) bullish crossover — faster response tuned for crypto volatility."""
    if len(df) < _MIN_BARS:
        return None
    macd, sig, _ = _macd_lines(df["Close"], 5, 13, 5)
    if len(macd) < 2:
        return None
    prev_diff = macd.iloc[-2] - sig.iloc[-2]
    curr_diff = macd.iloc[-1] - sig.iloc[-1]
    if not (prev_diff < 0 and curr_diff >= 0):
        return None
    strength = round(min(abs(curr_diff) / (df["Close"].iloc[-1] * 0.001 + 1e-9), 1.0), 4)
    sl, tp = _atr_stops(df, "bullish")
    return StrategyResult(
        strategy_name="macd_crossover_fast",
        symbol=symbol,
        direction="bullish",
        signal=1,
        stop_loss=sl,
        take_profit=tp,
        strength=strength,
        confidence=_DEFAULT_CONFIDENCE,
        detected_at=_now(),
        metadata={"macd_fast": round(macd.iloc[-1], 6), "signal_line": round(sig.iloc[-1], 6)},
    )


def macd_zero_cross(symbol: str, df: pd.DataFrame) -> Optional[StrategyResult]:
    """MACD(12,26,9) line crosses zero from below → bullish trend confirmation."""
    if len(df) < _MIN_BARS:
        return None
    macd, _, _ = _macd_lines(df["Close"], 12, 26, 9)
    if len(macd) < 2:
        return None
    if not (macd.iloc[-2] < 0 and macd.iloc[-1] >= 0):
        return None
    strength = round(min(macd.iloc[-1] / (df["Close"].iloc[-1] * 0.005 + 1e-9), 1.0), 4)
    sl, tp = _atr_stops(df, "bullish")
    return StrategyResult(
        strategy_name="macd_zero_cross",
        symbol=symbol,
        direction="bullish",
        signal=1,
        stop_loss=sl,
        take_profit=tp,
        strength=strength,
        confidence=_DEFAULT_CONFIDENCE,
        detected_at=_now(),
        metadata={"macd_prev": round(macd.iloc[-2], 6), "macd_now": round(macd.iloc[-1], 6)},
    )


def macd_histogram_reversal(symbol: str, df: pd.DataFrame) -> Optional[StrategyResult]:
    """Histogram was negative and declining, now starts rising → bullish momentum shift."""
    if len(df) < _MIN_BARS:
        return None
    _, _, hist = _macd_lines(df["Close"], 12, 26, 9)
    if len(hist) < 3:
        return None
    h0 = hist.iloc[-3]
    h1 = hist.iloc[-2]
    h2 = hist.iloc[-1]
    # All three negative; h1 was the trough; h2 > h1 (reversal starting)
    if not (h0 < 0 and h1 < 0 and h2 < 0 and h1 < h0 and h2 > h1):
        return None
    strength = round(min(abs(h2 - h1) / (abs(h1) + 1e-9), 1.0), 4)
    sl, tp = _atr_stops(df, "bullish")
    return StrategyResult(
        strategy_name="macd_histogram_reversal",
        symbol=symbol,
        direction="bullish",
        signal=1,
        stop_loss=sl,
        take_profit=tp,
        strength=strength,
        confidence=_DEFAULT_CONFIDENCE,
        detected_at=_now(),
        metadata={"hist_t2": round(h0, 6), "hist_t1": round(h1, 6), "hist_t0": round(h2, 6)},
    )


# ---------------------------------------------------------------------------
# Bollinger Band Family
# ---------------------------------------------------------------------------

def _bollinger(closes: pd.Series, period: int = 20, n_std: float = 2.0):
    """Return (mid, upper, lower) Bollinger Bands."""
    mid = closes.rolling(period).mean()
    std = closes.rolling(period).std(ddof=0)
    return mid, mid + n_std * std, mid - n_std * std


def bollinger_lower_touch(symbol: str, df: pd.DataFrame) -> Optional[StrategyResult]:
    """Price at or below the lower Bollinger Band (20, 2) → bullish mean reversion."""
    if len(df) < _MIN_BARS:
        return None
    _, _, lower = _bollinger(df["Close"])
    last_close = df["Close"].iloc[-1]
    last_lower = lower.iloc[-1]
    if pd.isna(last_lower) or last_close > last_lower:
        return None
    penetration = (last_lower - last_close) / (last_lower + 1e-9)
    strength = round(min(1 + penetration * 10, 1.0), 4)
    sl, tp = _atr_stops(df, "bullish")
    return StrategyResult(
        strategy_name="bollinger_lower_touch",
        symbol=symbol,
        direction="bullish",
        signal=1,
        stop_loss=sl,
        take_profit=tp,
        strength=strength,
        confidence=_DEFAULT_CONFIDENCE,
        detected_at=_now(),
        metadata={"close": round(last_close, 4), "lower_band": round(last_lower, 4)},
    )


def bollinger_upper_touch(symbol: str, df: pd.DataFrame) -> Optional[StrategyResult]:
    """Price at or above the upper Bollinger Band (20, 2) → bearish mean reversion."""
    if len(df) < _MIN_BARS:
        return None
    _, upper, _ = _bollinger(df["Close"])
    last_close = df["Close"].iloc[-1]
    last_upper = upper.iloc[-1]
    if pd.isna(last_upper) or last_close < last_upper:
        return None
    penetration = (last_close - last_upper) / (last_upper + 1e-9)
    strength = round(min(1 + penetration * 10, 1.0), 4)
    sl, tp = _atr_stops(df, "bearish")
    return StrategyResult(
        strategy_name="bollinger_upper_touch",
        symbol=symbol,
        direction="bearish",
        signal=-1,
        stop_loss=sl,
        take_profit=tp,
        strength=strength,
        confidence=_DEFAULT_CONFIDENCE,
        detected_at=_now(),
        metadata={"close": round(last_close, 4), "upper_band": round(last_upper, 4)},
    )


def bollinger_squeeze_break(symbol: str, df: pd.DataFrame) -> Optional[StrategyResult]:
    """Bandwidth below 80th percentile of last 100 bars, then price closes outside band with elevated volume."""
    if len(df) < _MIN_BARS:
        return None
    closes = df["Close"]
    vol = df["Volume"]
    mid, upper, lower = _bollinger(closes)
    bw = (upper - lower) / (mid.replace(0, np.nan))
    if len(bw.dropna()) < 20:
        return None
    lookback = bw.iloc[-100:] if len(bw) >= 100 else bw
    pct80 = lookback.quantile(0.80)
    last_bw = bw.iloc[-1]
    if pd.isna(last_bw) or last_bw > pct80:
        return None  # Not in squeeze
    last_close = closes.iloc[-1]
    last_upper = upper.iloc[-1]
    last_lower = lower.iloc[-1]
    above = last_close > last_upper
    below = last_close < last_lower
    if not (above or below):
        return None
    avg_vol = vol.iloc[-20:].mean()
    last_vol = vol.iloc[-1]
    if last_vol < avg_vol:
        return None
    direction = "bullish" if above else "bearish"
    vol_ratio = last_vol / (avg_vol + 1e-9)
    strength = round(min((vol_ratio - 1) * 0.5, 1.0), 4)
    sl, tp = _atr_stops(df, direction)
    return StrategyResult(
        strategy_name="bollinger_squeeze_break",
        symbol=symbol,
        direction=direction,
        signal=1 if above else -1,
        stop_loss=sl,
        take_profit=tp,
        strength=strength,
        confidence=_DEFAULT_CONFIDENCE,
        detected_at=_now(),
        metadata={
            "bandwidth": round(last_bw, 6),
            "bandwidth_p80": round(pct80, 6),
            "volume_ratio": round(vol_ratio, 2),
        },
    )


def bollinger_mean_reversion(symbol: str, df: pd.DataFrame) -> Optional[StrategyResult]:
    """Price > 2.5 std deviations from 20-period mean → expect reversion."""
    if len(df) < _MIN_BARS:
        return None
    closes = df["Close"]
    roll_mean = closes.rolling(20).mean()
    roll_std = closes.rolling(20).std(ddof=0)
    last_close = closes.iloc[-1]
    mean = roll_mean.iloc[-1]
    std = roll_std.iloc[-1]
    if pd.isna(mean) or std <= 0:
        return None
    z = (last_close - mean) / std
    if abs(z) < 2.5:
        return None
    direction = "bearish" if z > 0 else "bullish"
    strength = round(min((abs(z) - 2.5) / 1.5, 1.0), 4)
    sl, tp = _atr_stops(df, direction)
    return StrategyResult(
        strategy_name="bollinger_mean_reversion",
        symbol=symbol,
        direction=direction,
        signal=-1 if z > 0 else 1,
        stop_loss=sl,
        take_profit=tp,
        strength=strength,
        confidence=_DEFAULT_CONFIDENCE,
        detected_at=_now(),
        metadata={"z_score": round(z, 3), "mean": round(mean, 4), "std": round(std, 4)},
    )


# ---------------------------------------------------------------------------
# Structure Family
# ---------------------------------------------------------------------------

def _swing_points(series: pd.Series, lookback: int = 5):
    """Return (swing_highs_values, swing_lows_values) as lists of floats using vectorized rolling."""
    highs_arr = series.values
    n = len(highs_arr)
    swing_highs = []
    swing_lows = []
    for i in range(lookback, n - lookback):
        window = highs_arr[i - lookback: i + lookback + 1]
        val = highs_arr[i]
        if val == window.max():
            swing_highs.append(val)
        if val == window.min():
            swing_lows.append(val)
    return swing_highs, swing_lows


def structure_higher_high(symbol: str, df: pd.DataFrame) -> Optional[StrategyResult]:
    """Last two swing highs ascending AND last two swing lows ascending → bullish trend."""
    if len(df) < _MIN_BARS:
        return None
    swing_highs, _ = _swing_points(df["High"], lookback=5)
    _, swing_lows = _swing_points(df["Low"], lookback=5)
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return None
    hh = swing_highs[-1] > swing_highs[-2]
    hl = swing_lows[-1] > swing_lows[-2]
    if not (hh and hl):
        return None
    hh_strength = (swing_highs[-1] - swing_highs[-2]) / (swing_highs[-2] + 1e-9)
    strength = round(min(hh_strength * 20, 1.0), 4)
    sl, tp = _atr_stops(df, "bullish")
    return StrategyResult(
        strategy_name="structure_higher_high",
        symbol=symbol,
        direction="bullish",
        signal=1,
        stop_loss=sl,
        take_profit=tp,
        strength=strength,
        confidence=_DEFAULT_CONFIDENCE,
        detected_at=_now(),
        metadata={
            "swing_high_prev": round(swing_highs[-2], 4),
            "swing_high_last": round(swing_highs[-1], 4),
            "swing_low_prev": round(swing_lows[-2], 4),
            "swing_low_last": round(swing_lows[-1], 4),
        },
    )


def structure_bos_bull(symbol: str, df: pd.DataFrame) -> Optional[StrategyResult]:
    """Current close > highest swing high of last 20 bars → bullish break of structure."""
    if len(df) < _MIN_BARS:
        return None
    closes = df["Close"]
    highs = df["High"]
    last_close = closes.iloc[-1]
    prior_high = highs.iloc[-21:-1].max()
    if last_close <= prior_high:
        return None
    breakout_pct = (last_close - prior_high) / (prior_high + 1e-9)
    strength = round(min(breakout_pct * 20, 1.0), 4)
    sl, tp = _atr_stops(df, "bullish")
    return StrategyResult(
        strategy_name="structure_bos_bull",
        symbol=symbol,
        direction="bullish",
        signal=1,
        stop_loss=sl,
        take_profit=tp,
        strength=strength,
        confidence=_DEFAULT_CONFIDENCE,
        detected_at=_now(),
        metadata={"close": round(last_close, 4), "prior_20bar_high": round(prior_high, 4)},
    )


def structure_bos_bear(symbol: str, df: pd.DataFrame) -> Optional[StrategyResult]:
    """Current close < lowest swing low of last 20 bars → bearish break of structure."""
    if len(df) < _MIN_BARS:
        return None
    closes = df["Close"]
    lows = df["Low"]
    last_close = closes.iloc[-1]
    prior_low = lows.iloc[-21:-1].min()
    if last_close >= prior_low:
        return None
    breakdown_pct = (prior_low - last_close) / (prior_low + 1e-9)
    strength = round(min(breakdown_pct * 20, 1.0), 4)
    sl, tp = _atr_stops(df, "bearish")
    return StrategyResult(
        strategy_name="structure_bos_bear",
        symbol=symbol,
        direction="bearish",
        signal=-1,
        stop_loss=sl,
        take_profit=tp,
        strength=strength,
        confidence=_DEFAULT_CONFIDENCE,
        detected_at=_now(),
        metadata={"close": round(last_close, 4), "prior_20bar_low": round(prior_low, 4)},
    )


def structure_support_retest(symbol: str, df: pd.DataFrame) -> Optional[StrategyResult]:
    """Price dropped to within 0.5% of prior swing low, held, and bounced → bullish."""
    if len(df) < _MIN_BARS:
        return None
    lows = df["Low"]
    closes = df["Close"]
    # Find the prior swing low in bars -20 to -3
    prior_swing_low = lows.iloc[-20:-3].min()
    last_low = lows.iloc[-1]
    last_close = closes.iloc[-1]
    proximity = abs(last_low - prior_swing_low) / (prior_swing_low + 1e-9)
    # Held near support (within 0.5%)
    if proximity > 0.005:
        return None
    # Bounced: close is above the low by at least 0.3% of price
    if last_close <= last_low * 1.003:
        return None
    bounce_pct = (last_close - last_low) / (last_low + 1e-9)
    strength = round(min(bounce_pct * 50, 1.0), 4)
    sl, tp = _atr_stops(df, "bullish")
    return StrategyResult(
        strategy_name="structure_support_retest",
        symbol=symbol,
        direction="bullish",
        signal=1,
        stop_loss=sl,
        take_profit=tp,
        strength=strength,
        confidence=_DEFAULT_CONFIDENCE,
        detected_at=_now(),
        metadata={
            "prior_swing_low": round(prior_swing_low, 4),
            "last_low": round(last_low, 4),
            "last_close": round(last_close, 4),
            "proximity_pct": round(proximity * 100, 3),
        },
    )


# ---------------------------------------------------------------------------
# Volume Family
# ---------------------------------------------------------------------------

def volume_climax_bull(symbol: str, df: pd.DataFrame) -> Optional[StrategyResult]:
    """Volume > 2x 20-bar avg, red candle, close in bottom 25% of range → exhaustion, bullish."""
    if len(df) < _MIN_BARS:
        return None
    opens = df["Open"]
    closes = df["Close"]
    highs = df["High"]
    lows = df["Low"]
    vol = df["Volume"]
    avg_vol = vol.iloc[-21:-1].mean()
    last_vol = vol.iloc[-1]
    if last_vol < avg_vol * 2:
        return None
    last_open = opens.iloc[-1]
    last_close = closes.iloc[-1]
    last_high = highs.iloc[-1]
    last_low = lows.iloc[-1]
    is_red = last_close < last_open
    candle_range = last_high - last_low
    if candle_range <= 0 or not is_red:
        return None
    close_position = (last_close - last_low) / candle_range
    if close_position > 0.25:
        return None
    vol_ratio = last_vol / (avg_vol + 1e-9)
    strength = round(min((vol_ratio - 2) * 0.3 + (0.25 - close_position), 1.0), 4)
    sl, tp = _atr_stops(df, "bullish")
    return StrategyResult(
        strategy_name="volume_climax_bull",
        symbol=symbol,
        direction="bullish",
        signal=1,
        stop_loss=sl,
        take_profit=tp,
        strength=strength,
        confidence=_DEFAULT_CONFIDENCE,
        detected_at=_now(),
        metadata={
            "volume_ratio": round(vol_ratio, 2),
            "close_position_in_range": round(close_position, 3),
        },
    )


def volume_climax_bear(symbol: str, df: pd.DataFrame) -> Optional[StrategyResult]:
    """Volume > 2x 20-bar avg, green candle, close in top 25% of range → exhaustion, bearish."""
    if len(df) < _MIN_BARS:
        return None
    opens = df["Open"]
    closes = df["Close"]
    highs = df["High"]
    lows = df["Low"]
    vol = df["Volume"]
    avg_vol = vol.iloc[-21:-1].mean()
    last_vol = vol.iloc[-1]
    if last_vol < avg_vol * 2:
        return None
    last_open = opens.iloc[-1]
    last_close = closes.iloc[-1]
    last_high = highs.iloc[-1]
    last_low = lows.iloc[-1]
    is_green = last_close > last_open
    candle_range = last_high - last_low
    if candle_range <= 0 or not is_green:
        return None
    close_position = (last_close - last_low) / candle_range
    if close_position < 0.75:
        return None
    vol_ratio = last_vol / (avg_vol + 1e-9)
    strength = round(min((vol_ratio - 2) * 0.3 + (close_position - 0.75), 1.0), 4)
    sl, tp = _atr_stops(df, "bearish")
    return StrategyResult(
        strategy_name="volume_climax_bear",
        symbol=symbol,
        direction="bearish",
        signal=-1,
        stop_loss=sl,
        take_profit=tp,
        strength=strength,
        confidence=_DEFAULT_CONFIDENCE,
        detected_at=_now(),
        metadata={
            "volume_ratio": round(vol_ratio, 2),
            "close_position_in_range": round(close_position, 3),
        },
    )


def volume_accumulation(symbol: str, df: pd.DataFrame) -> Optional[StrategyResult]:
    """Last 5 bars: price trending up AND volume trending up → institutional accumulation."""
    if len(df) < _MIN_BARS:
        return None
    closes = df["Close"].iloc[-5:]
    vol = df["Volume"].iloc[-5:]
    # Both must have positive slope (use correlation with range index)
    idx = np.arange(5)
    price_slope = np.polyfit(idx, closes.values, 1)[0]
    vol_slope = np.polyfit(idx, vol.values, 1)[0]
    if price_slope <= 0 or vol_slope <= 0:
        return None
    # Normalize slopes for strength
    price_chg = (closes.iloc[-1] - closes.iloc[0]) / (closes.iloc[0] + 1e-9)
    vol_chg = (vol.iloc[-1] - vol.iloc[0]) / (vol.iloc[0] + 1e-9)
    strength = round(min((price_chg + vol_chg * 0.5) * 5, 1.0), 4)
    sl, tp = _atr_stops(df, "bullish")
    return StrategyResult(
        strategy_name="volume_accumulation",
        symbol=symbol,
        direction="bullish",
        signal=1,
        stop_loss=sl,
        take_profit=tp,
        strength=strength,
        confidence=_DEFAULT_CONFIDENCE,
        detected_at=_now(),
        metadata={
            "price_5bar_chg_pct": round(price_chg * 100, 2),
            "volume_5bar_chg_pct": round(vol_chg * 100, 2),
        },
    )


def volume_dry_up(symbol: str, df: pd.DataFrame) -> Optional[StrategyResult]:
    """Volume < 50% of 20-bar avg for 3+ consecutive bars → coil; direction from prior trend."""
    if len(df) < _MIN_BARS:
        return None
    vol = df["Volume"]
    closes = df["Close"]
    avg_vol = vol.iloc[-24:-3].mean()
    # Check last 3 bars for dry up
    last3_vol = vol.iloc[-3:]
    if not (last3_vol < avg_vol * 0.5).all():
        return None
    # Prior trend: compare close 20 bars ago vs 10 bars ago (pre-dry-up)
    trend_start = closes.iloc[-20]
    trend_end = closes.iloc[-4]
    prior_trend_up = trend_end > trend_start
    direction = "bullish" if prior_trend_up else "bearish"
    dry_ratio = last3_vol.mean() / (avg_vol + 1e-9)
    strength = round(min((0.5 - dry_ratio) * 2, 1.0), 4)
    sl, tp = _atr_stops(df, direction)
    return StrategyResult(
        strategy_name="volume_dry_up",
        symbol=symbol,
        direction=direction,
        signal=1 if prior_trend_up else -1,
        stop_loss=sl,
        take_profit=tp,
        strength=strength,
        confidence=_DEFAULT_CONFIDENCE,
        detected_at=_now(),
        metadata={
            "avg_volume": round(avg_vol, 0),
            "dry_up_avg": round(last3_vol.mean(), 0),
            "dry_ratio": round(dry_ratio, 3),
            "prior_trend": "up" if prior_trend_up else "down",
        },
    )


# ---------------------------------------------------------------------------
# Moving Average Family
# ---------------------------------------------------------------------------

def ema_cross_9_21(symbol: str, df: pd.DataFrame) -> Optional[StrategyResult]:
    """9 EMA crosses above 21 EMA → bullish short-term trend shift."""
    if len(df) < _MIN_BARS:
        return None
    closes = df["Close"]
    ema9 = closes.ewm(span=9, adjust=False).mean()
    ema21 = closes.ewm(span=21, adjust=False).mean()
    prev_diff = ema9.iloc[-2] - ema21.iloc[-2]
    curr_diff = ema9.iloc[-1] - ema21.iloc[-1]
    if not (prev_diff < 0 and curr_diff >= 0):
        return None
    separation = curr_diff / (closes.iloc[-1] + 1e-9)
    strength = round(min(separation * 200, 1.0), 4)
    sl, tp = _atr_stops(df, "bullish")
    return StrategyResult(
        strategy_name="ema_cross_9_21",
        symbol=symbol,
        direction="bullish",
        signal=1,
        stop_loss=sl,
        take_profit=tp,
        strength=strength,
        confidence=_DEFAULT_CONFIDENCE,
        detected_at=_now(),
        metadata={"ema9": round(ema9.iloc[-1], 4), "ema21": round(ema21.iloc[-1], 4)},
    )


def ema_cross_50_200(symbol: str, df: pd.DataFrame) -> Optional[StrategyResult]:
    """50 EMA crosses above 200 EMA → bullish golden cross (major trend signal)."""
    if len(df) < 210:
        return None
    closes = df["Close"]
    ema50 = closes.ewm(span=50, adjust=False).mean()
    ema200 = closes.ewm(span=200, adjust=False).mean()
    prev_diff = ema50.iloc[-2] - ema200.iloc[-2]
    curr_diff = ema50.iloc[-1] - ema200.iloc[-1]
    if not (prev_diff < 0 and curr_diff >= 0):
        return None
    separation = curr_diff / (closes.iloc[-1] + 1e-9)
    strength = round(min(separation * 500, 1.0), 4)
    sl, tp = _atr_stops(df, "bullish")
    return StrategyResult(
        strategy_name="ema_cross_50_200",
        symbol=symbol,
        direction="bullish",
        signal=1,
        stop_loss=sl,
        take_profit=tp,
        strength=strength,
        confidence=_DEFAULT_CONFIDENCE,
        detected_at=_now(),
        metadata={"ema50": round(ema50.iloc[-1], 4), "ema200": round(ema200.iloc[-1], 4)},
    )


def price_above_200ema(symbol: str, df: pd.DataFrame) -> Optional[StrategyResult]:
    """Price was below 200 EMA, now closes above → bullish regime change."""
    if len(df) < 210:
        return None
    closes = df["Close"]
    ema200 = closes.ewm(span=200, adjust=False).mean()
    if not (closes.iloc[-2] < ema200.iloc[-2] and closes.iloc[-1] >= ema200.iloc[-1]):
        return None
    cross_pct = (closes.iloc[-1] - ema200.iloc[-1]) / (ema200.iloc[-1] + 1e-9)
    strength = round(min(cross_pct * 100, 1.0), 4)
    sl, tp = _atr_stops(df, "bullish")
    return StrategyResult(
        strategy_name="price_above_200ema",
        symbol=symbol,
        direction="bullish",
        signal=1,
        stop_loss=sl,
        take_profit=tp,
        strength=strength,
        confidence=_DEFAULT_CONFIDENCE,
        detected_at=_now(),
        metadata={
            "close": round(closes.iloc[-1], 4),
            "ema200": round(ema200.iloc[-1], 4),
            "cross_pct": round(cross_pct * 100, 3),
        },
    )


def ma_ribbon_expand(symbol: str, df: pd.DataFrame) -> Optional[StrategyResult]:
    """EMAs 8,13,21,34,55 in perfect order AND spreading apart → strong trend confirmation."""
    if len(df) < _MIN_BARS:
        return None
    closes = df["Close"]
    spans = [8, 13, 21, 34, 55]
    emas = [closes.ewm(span=s, adjust=False).mean().iloc[-1] for s in spans]
    e8, e13, e21, e34, e55 = emas
    bullish_order = e8 > e13 > e21 > e34 > e55
    bearish_order = e8 < e13 < e21 < e34 < e55
    if not (bullish_order or bearish_order):
        return None
    direction = "bullish" if bullish_order else "bearish"
    # Spreading: compare current spread to 5 bars ago
    emas_prev = [closes.ewm(span=s, adjust=False).mean().iloc[-6] for s in spans]
    spread_now = emas[0] - emas[-1]
    spread_prev = emas_prev[0] - emas_prev[-1]
    is_spreading = abs(spread_now) > abs(spread_prev) if spread_prev != 0 else False
    if not is_spreading:
        return None
    spread_pct = abs(spread_now) / (closes.iloc[-1] + 1e-9)
    strength = round(min(spread_pct * 20, 1.0), 4)
    sl, tp = _atr_stops(df, direction)
    return StrategyResult(
        strategy_name="ma_ribbon_expand",
        symbol=symbol,
        direction=direction,
        signal=1 if bullish_order else -1,
        stop_loss=sl,
        take_profit=tp,
        strength=strength,
        confidence=_DEFAULT_CONFIDENCE,
        detected_at=_now(),
        metadata={
            "ema8": round(e8, 4),
            "ema13": round(e13, 4),
            "ema21": round(e21, 4),
            "ema34": round(e34, 4),
            "ema55": round(e55, 4),
            "spread_pct": round(spread_pct * 100, 3),
        },
    )


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

ALL_STRATEGIES: list[tuple[str, Callable]] = [
    ("rsi_oversold_14", rsi_oversold_14),
    ("rsi_oversold_2", rsi_oversold_2),
    ("rsi_overbought_14", rsi_overbought_14),
    ("rsi_momentum_shift", rsi_momentum_shift),
    ("rsi_divergence", rsi_divergence),
    ("macd_crossover_std", macd_crossover_std),
    ("macd_crossover_fast", macd_crossover_fast),
    ("macd_zero_cross", macd_zero_cross),
    ("macd_histogram_reversal", macd_histogram_reversal),
    ("bollinger_lower_touch", bollinger_lower_touch),
    ("bollinger_upper_touch", bollinger_upper_touch),
    ("bollinger_squeeze_break", bollinger_squeeze_break),
    ("bollinger_mean_reversion", bollinger_mean_reversion),
    ("structure_higher_high", structure_higher_high),
    ("structure_bos_bull", structure_bos_bull),
    ("structure_bos_bear", structure_bos_bear),
    ("structure_support_retest", structure_support_retest),
    ("volume_climax_bull", volume_climax_bull),
    ("volume_climax_bear", volume_climax_bear),
    ("volume_accumulation", volume_accumulation),
    ("volume_dry_up", volume_dry_up),
    ("ema_cross_9_21", ema_cross_9_21),
    ("ema_cross_50_200", ema_cross_50_200),
    ("price_above_200ema", price_above_200ema),
    ("ma_ribbon_expand", ma_ribbon_expand),
]
