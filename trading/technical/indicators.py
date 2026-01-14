#!/usr/bin/env python3
"""
indicators.py - Technical Indicator Calculations for MIDGE

Pure Python implementation of standard technical indicators.
No external TA libraries required - all formulas explicit for understanding.

Indicators implemented:
- Trend: SMA, EMA, MACD, Parabolic SAR
- Momentum: RSI, Stochastic, ADX, CCI, Williams %R
- Volatility: Bollinger Bands, ATR, Keltner Channels
- Volume: VWAP, OBV, Accumulation/Distribution, Chaikin Money Flow
- Support/Resistance: Pivot Points, Fibonacci Retracement

All functions accept lists of OHLCV data and return calculated values.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import math


@dataclass
class OHLCV:
    """Single candlestick data point."""
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: Optional[str] = None


# =============================================================================
# TREND INDICATORS
# =============================================================================

def sma(closes: List[float], period: int) -> List[Optional[float]]:
    """
    Simple Moving Average

    Formula: SMA = sum(close[i] for i in range(period)) / period

    Common periods: 20 (short), 50 (medium), 200 (long-term trend)

    Args:
        closes: List of closing prices
        period: Number of periods to average

    Returns:
        List of SMA values (None for first period-1 values)
    """
    result = [None] * (period - 1)
    for i in range(period - 1, len(closes)):
        window = closes[i - period + 1:i + 1]
        result.append(sum(window) / period)
    return result


def ema(closes: List[float], period: int) -> List[Optional[float]]:
    """
    Exponential Moving Average

    Formula:
        Multiplier (k) = 2 / (period + 1)
        EMA = close * k + EMA_prev * (1 - k)

    First EMA = SMA of first 'period' values

    Args:
        closes: List of closing prices
        period: EMA period (common: 12, 26 for MACD)

    Returns:
        List of EMA values (None for first period-1 values)
    """
    if len(closes) < period:
        return [None] * len(closes)

    k = 2 / (period + 1)
    result = [None] * (period - 1)

    # First EMA is SMA of first period values
    first_sma = sum(closes[:period]) / period
    result.append(first_sma)

    # Subsequent EMAs
    for i in range(period, len(closes)):
        ema_val = closes[i] * k + result[-1] * (1 - k)
        result.append(ema_val)

    return result


def macd(closes: List[float],
         fast_period: int = 12,
         slow_period: int = 26,
         signal_period: int = 9) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    """
    Moving Average Convergence Divergence

    Formula:
        MACD Line = EMA(fast) - EMA(slow)
        Signal Line = EMA(MACD Line, signal_period)
        Histogram = MACD Line - Signal Line

    Standard settings: 12, 26, 9

    Signals:
        - MACD crosses above Signal = bullish
        - MACD crosses below Signal = bearish
        - Histogram increasing = momentum building
        - Divergence with price = potential reversal

    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    ema_fast = ema(closes, fast_period)
    ema_slow = ema(closes, slow_period)

    # MACD Line = EMA(fast) - EMA(slow)
    macd_line = []
    for i in range(len(closes)):
        if ema_fast[i] is None or ema_slow[i] is None:
            macd_line.append(None)
        else:
            macd_line.append(ema_fast[i] - ema_slow[i])

    # Signal Line = EMA of MACD Line
    # Find first valid MACD value
    valid_macd = [v for v in macd_line if v is not None]
    if len(valid_macd) < signal_period:
        return macd_line, [None] * len(closes), [None] * len(closes)

    signal_ema = ema(valid_macd, signal_period)

    # Map signal back to full length
    signal_line = []
    valid_idx = 0
    for i in range(len(closes)):
        if macd_line[i] is None:
            signal_line.append(None)
        else:
            if valid_idx < len(signal_ema):
                signal_line.append(signal_ema[valid_idx])
                valid_idx += 1
            else:
                signal_line.append(None)

    # Histogram = MACD - Signal
    histogram = []
    for i in range(len(closes)):
        if macd_line[i] is None or signal_line[i] is None:
            histogram.append(None)
        else:
            histogram.append(macd_line[i] - signal_line[i])

    return macd_line, signal_line, histogram


# =============================================================================
# MOMENTUM INDICATORS
# =============================================================================

def rsi(closes: List[float], period: int = 14) -> List[Optional[float]]:
    """
    Relative Strength Index

    Formula:
        Change = close - close_prev
        Gain = change if change > 0 else 0
        Loss = abs(change) if change < 0 else 0

        First Avg Gain = sum(gains[:period]) / period
        First Avg Loss = sum(losses[:period]) / period

        Subsequent:
        Avg Gain = (prev_avg_gain * (period-1) + current_gain) / period
        Avg Loss = (prev_avg_loss * (period-1) + current_loss) / period

        RS = Avg Gain / Avg Loss
        RSI = 100 - (100 / (1 + RS))

    Interpretation:
        RSI > 70 = Overbought (potential sell)
        RSI < 30 = Oversold (potential buy)
        Divergence = RSI direction differs from price direction

    Args:
        closes: List of closing prices
        period: RSI period (standard: 14)

    Returns:
        List of RSI values (0-100 scale)
    """
    if len(closes) < period + 1:
        return [None] * len(closes)

    # Calculate gains and losses
    gains = []
    losses = []
    for i in range(1, len(closes)):
        change = closes[i] - closes[i - 1]
        gains.append(change if change > 0 else 0)
        losses.append(abs(change) if change < 0 else 0)

    result = [None] * period

    # First average
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    # First RSI
    if avg_loss == 0:
        result.append(100.0)
    else:
        rs = avg_gain / avg_loss
        result.append(100 - (100 / (1 + rs)))

    # Subsequent RSI values (smoothed)
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            result.append(100.0)
        else:
            rs = avg_gain / avg_loss
            result.append(100 - (100 / (1 + rs)))

    return result


def stochastic(highs: List[float],
               lows: List[float],
               closes: List[float],
               k_period: int = 14,
               d_period: int = 3) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    """
    Stochastic Oscillator

    Formula:
        %K = 100 * (close - lowest_low) / (highest_high - lowest_low)
        %D = SMA(%K, d_period)  # Signal line

    Interpretation:
        %K > 80 = Overbought
        %K < 20 = Oversold
        %K crosses above %D = bullish signal
        %K crosses below %D = bearish signal

    Returns:
        Tuple of (%K, %D)
    """
    if len(closes) < k_period:
        return [None] * len(closes), [None] * len(closes)

    k_values = [None] * (k_period - 1)

    for i in range(k_period - 1, len(closes)):
        window_highs = highs[i - k_period + 1:i + 1]
        window_lows = lows[i - k_period + 1:i + 1]

        highest_high = max(window_highs)
        lowest_low = min(window_lows)

        if highest_high == lowest_low:
            k_values.append(50.0)  # Neutral when no range
        else:
            k = 100 * (closes[i] - lowest_low) / (highest_high - lowest_low)
            k_values.append(k)

    # %D is SMA of %K
    valid_k = [v for v in k_values if v is not None]
    d_sma = sma(valid_k, d_period)

    # Map D back
    d_values = [None] * (k_period - 1)
    d_values.extend(d_sma)

    return k_values, d_values


def adx(highs: List[float],
        lows: List[float],
        closes: List[float],
        period: int = 14) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    """
    Average Directional Index

    Measures trend strength (not direction).

    Formula:
        +DM = high - high_prev if (high - high_prev) > (low_prev - low) else 0
        -DM = low_prev - low if (low_prev - low) > (high - high_prev) else 0

        TR = max(high - low, abs(high - close_prev), abs(low - close_prev))

        +DI = 100 * smoothed(+DM) / smoothed(TR)
        -DI = 100 * smoothed(-DM) / smoothed(TR)

        DX = 100 * abs(+DI - -DI) / (+DI + -DI)
        ADX = smoothed(DX)

    Interpretation:
        ADX > 25 = Strong trend
        ADX < 20 = Weak/no trend
        +DI > -DI = Uptrend
        +DI < -DI = Downtrend

    Returns:
        Tuple of (ADX, +DI, -DI)
    """
    if len(closes) < period + 1:
        none_list = [None] * len(closes)
        return none_list, none_list, none_list

    # Calculate +DM, -DM, TR
    plus_dm = []
    minus_dm = []
    tr = []

    for i in range(1, len(closes)):
        high_diff = highs[i] - highs[i - 1]
        low_diff = lows[i - 1] - lows[i]

        if high_diff > low_diff and high_diff > 0:
            plus_dm.append(high_diff)
        else:
            plus_dm.append(0)

        if low_diff > high_diff and low_diff > 0:
            minus_dm.append(low_diff)
        else:
            minus_dm.append(0)

        # True Range
        tr1 = highs[i] - lows[i]
        tr2 = abs(highs[i] - closes[i - 1])
        tr3 = abs(lows[i] - closes[i - 1])
        tr.append(max(tr1, tr2, tr3))

    # Smooth using Wilder's method (similar to EMA but different smoothing)
    def wilder_smooth(values: List[float], period: int) -> List[float]:
        result = []
        if len(values) < period:
            return [None] * len(values)

        # First value is sum of first period
        first = sum(values[:period])
        result.extend([None] * (period - 1))
        result.append(first)

        for i in range(period, len(values)):
            smoothed = result[-1] - (result[-1] / period) + values[i]
            result.append(smoothed)

        return result

    smoothed_plus_dm = wilder_smooth(plus_dm, period)
    smoothed_minus_dm = wilder_smooth(minus_dm, period)
    smoothed_tr = wilder_smooth(tr, period)

    # Calculate +DI and -DI
    plus_di = []
    minus_di = []
    dx = []

    for i in range(len(smoothed_tr)):
        if smoothed_tr[i] is None or smoothed_tr[i] == 0:
            plus_di.append(None)
            minus_di.append(None)
            dx.append(None)
        else:
            pdi = 100 * smoothed_plus_dm[i] / smoothed_tr[i]
            mdi = 100 * smoothed_minus_dm[i] / smoothed_tr[i]
            plus_di.append(pdi)
            minus_di.append(mdi)

            if pdi + mdi == 0:
                dx.append(0)
            else:
                dx.append(100 * abs(pdi - mdi) / (pdi + mdi))

    # ADX is smoothed DX
    valid_dx = [v for v in dx if v is not None]
    adx_smooth = wilder_smooth(valid_dx, period) if len(valid_dx) >= period else [None] * len(valid_dx)

    # Map back to full length
    adx_values = [None] * (len(closes) - len(valid_dx))
    for val in adx_smooth:
        if val is not None:
            adx_values.append(val / period)  # Normalize
        else:
            adx_values.append(None)

    # Prepend None for first value (no previous to compare)
    plus_di = [None] + plus_di
    minus_di = [None] + minus_di

    # Ensure all lists are same length
    while len(adx_values) < len(closes):
        adx_values.insert(0, None)
    while len(plus_di) < len(closes):
        plus_di.insert(0, None)
    while len(minus_di) < len(closes):
        minus_di.insert(0, None)

    return adx_values, plus_di, minus_di


def williams_r(highs: List[float],
               lows: List[float],
               closes: List[float],
               period: int = 14) -> List[Optional[float]]:
    """
    Williams %R

    Similar to Stochastic but inverted scale.

    Formula:
        %R = -100 * (highest_high - close) / (highest_high - lowest_low)

    Interpretation:
        %R > -20 = Overbought
        %R < -80 = Oversold

    Returns:
        List of Williams %R values (-100 to 0 scale)
    """
    if len(closes) < period:
        return [None] * len(closes)

    result = [None] * (period - 1)

    for i in range(period - 1, len(closes)):
        window_highs = highs[i - period + 1:i + 1]
        window_lows = lows[i - period + 1:i + 1]

        highest = max(window_highs)
        lowest = min(window_lows)

        if highest == lowest:
            result.append(-50.0)
        else:
            wr = -100 * (highest - closes[i]) / (highest - lowest)
            result.append(wr)

    return result


def cci(highs: List[float],
        lows: List[float],
        closes: List[float],
        period: int = 20) -> List[Optional[float]]:
    """
    Commodity Channel Index

    Formula:
        Typical Price (TP) = (high + low + close) / 3
        SMA_TP = SMA(TP, period)
        Mean Deviation = avg(abs(TP - SMA_TP))
        CCI = (TP - SMA_TP) / (0.015 * Mean Deviation)

    Interpretation:
        CCI > 100 = Strong uptrend / overbought
        CCI < -100 = Strong downtrend / oversold

    Returns:
        List of CCI values
    """
    if len(closes) < period:
        return [None] * len(closes)

    # Calculate typical prices
    tp = [(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)]

    # SMA of TP
    tp_sma = sma(tp, period)

    result = []
    for i in range(len(closes)):
        if tp_sma[i] is None:
            result.append(None)
        else:
            # Mean deviation
            window_tp = tp[i - period + 1:i + 1]
            mean_dev = sum(abs(t - tp_sma[i]) for t in window_tp) / period

            if mean_dev == 0:
                result.append(0)
            else:
                cci_val = (tp[i] - tp_sma[i]) / (0.015 * mean_dev)
                result.append(cci_val)

    return result


# =============================================================================
# VOLATILITY INDICATORS
# =============================================================================

def bollinger_bands(closes: List[float],
                    period: int = 20,
                    num_std: float = 2.0) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    """
    Bollinger Bands

    Formula:
        Middle Band = SMA(close, period)
        Standard Deviation = sqrt(sum((close - SMA)^2) / period)
        Upper Band = Middle + (num_std * StdDev)
        Lower Band = Middle - (num_std * StdDev)

    Squeeze Detection:
        Band width = (Upper - Lower) / Middle
        Squeeze = Band width at multi-period low

    Interpretation:
        Price at upper band = potentially overbought
        Price at lower band = potentially oversold
        Squeeze followed by expansion = volatility breakout

    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    if len(closes) < period:
        none_list = [None] * len(closes)
        return none_list, none_list, none_list

    middle = sma(closes, period)
    upper = []
    lower = []

    for i in range(len(closes)):
        if middle[i] is None:
            upper.append(None)
            lower.append(None)
        else:
            # Calculate standard deviation
            window = closes[i - period + 1:i + 1]
            variance = sum((c - middle[i]) ** 2 for c in window) / period
            std_dev = math.sqrt(variance)

            upper.append(middle[i] + num_std * std_dev)
            lower.append(middle[i] - num_std * std_dev)

    return upper, middle, lower


def atr(highs: List[float],
        lows: List[float],
        closes: List[float],
        period: int = 14) -> List[Optional[float]]:
    """
    Average True Range

    Measures volatility.

    Formula:
        True Range = max(
            high - low,
            abs(high - close_prev),
            abs(low - close_prev)
        )
        ATR = EMA(TR, period) or Wilder's smoothing

    Interpretation:
        Higher ATR = More volatility
        Can be used for stop-loss placement (e.g., 2x ATR from entry)

    Returns:
        List of ATR values
    """
    if len(closes) < 2:
        return [None] * len(closes)

    # Calculate True Range
    tr = [highs[0] - lows[0]]  # First TR is just high-low

    for i in range(1, len(closes)):
        tr1 = highs[i] - lows[i]
        tr2 = abs(highs[i] - closes[i - 1])
        tr3 = abs(lows[i] - closes[i - 1])
        tr.append(max(tr1, tr2, tr3))

    # Use EMA for smoothing
    return ema(tr, period)


def keltner_channels(highs: List[float],
                     lows: List[float],
                     closes: List[float],
                     ema_period: int = 20,
                     atr_period: int = 10,
                     multiplier: float = 2.0) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    """
    Keltner Channels

    Similar to Bollinger Bands but uses ATR instead of standard deviation.

    Formula:
        Middle = EMA(close, ema_period)
        Upper = Middle + (multiplier * ATR)
        Lower = Middle - (multiplier * ATR)

    Returns:
        Tuple of (upper, middle, lower)
    """
    middle = ema(closes, ema_period)
    atr_values = atr(highs, lows, closes, atr_period)

    upper = []
    lower = []

    for i in range(len(closes)):
        if middle[i] is None or atr_values[i] is None:
            upper.append(None)
            lower.append(None)
        else:
            upper.append(middle[i] + multiplier * atr_values[i])
            lower.append(middle[i] - multiplier * atr_values[i])

    return upper, middle, lower


# =============================================================================
# VOLUME INDICATORS
# =============================================================================

def vwap(highs: List[float],
         lows: List[float],
         closes: List[float],
         volumes: List[float]) -> List[Optional[float]]:
    """
    Volume Weighted Average Price

    Intraday indicator - resets each day.
    This implementation is cumulative (for single-day data).

    Formula:
        Typical Price = (high + low + close) / 3
        VWAP = cumsum(TP * volume) / cumsum(volume)

    Interpretation:
        Price above VWAP = bullish (buyers in control)
        Price below VWAP = bearish (sellers in control)
        Often acts as support/resistance

    Returns:
        List of VWAP values
    """
    if len(closes) == 0:
        return []

    result = []
    cumulative_tpv = 0
    cumulative_volume = 0

    for i in range(len(closes)):
        tp = (highs[i] + lows[i] + closes[i]) / 3
        cumulative_tpv += tp * volumes[i]
        cumulative_volume += volumes[i]

        if cumulative_volume == 0:
            result.append(None)
        else:
            result.append(cumulative_tpv / cumulative_volume)

    return result


def obv(closes: List[float], volumes: List[float]) -> List[float]:
    """
    On-Balance Volume

    Cumulative volume indicator.

    Formula:
        If close > close_prev: OBV = OBV_prev + volume
        If close < close_prev: OBV = OBV_prev - volume
        If close == close_prev: OBV = OBV_prev

    Interpretation:
        Rising OBV + rising price = uptrend confirmed
        Falling OBV + falling price = downtrend confirmed
        Divergence (OBV direction differs from price) = potential reversal

    Returns:
        List of OBV values
    """
    if len(closes) == 0:
        return []

    result = [volumes[0]]  # First OBV is first volume

    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            result.append(result[-1] + volumes[i])
        elif closes[i] < closes[i - 1]:
            result.append(result[-1] - volumes[i])
        else:
            result.append(result[-1])

    return result


def accumulation_distribution(highs: List[float],
                               lows: List[float],
                               closes: List[float],
                               volumes: List[float]) -> List[float]:
    """
    Accumulation/Distribution Line

    Formula:
        Money Flow Multiplier = ((close - low) - (high - close)) / (high - low)
        Money Flow Volume = MFM * volume
        A/D = cumsum(Money Flow Volume)

    Interpretation:
        Rising A/D = accumulation (buying pressure)
        Falling A/D = distribution (selling pressure)
        Divergence with price = potential reversal

    Returns:
        List of A/D values
    """
    if len(closes) == 0:
        return []

    result = []
    cumulative = 0

    for i in range(len(closes)):
        if highs[i] == lows[i]:
            mfm = 0
        else:
            mfm = ((closes[i] - lows[i]) - (highs[i] - closes[i])) / (highs[i] - lows[i])

        mfv = mfm * volumes[i]
        cumulative += mfv
        result.append(cumulative)

    return result


def chaikin_money_flow(highs: List[float],
                       lows: List[float],
                       closes: List[float],
                       volumes: List[float],
                       period: int = 20) -> List[Optional[float]]:
    """
    Chaikin Money Flow

    Formula:
        Money Flow Multiplier = ((close - low) - (high - close)) / (high - low)
        Money Flow Volume = MFM * volume
        CMF = sum(MFV, period) / sum(volume, period)

    Interpretation:
        CMF > 0 = buying pressure (accumulation)
        CMF < 0 = selling pressure (distribution)
        CMF > 0.25 = strong buying
        CMF < -0.25 = strong selling

    Returns:
        List of CMF values (-1 to 1 scale)
    """
    if len(closes) < period:
        return [None] * len(closes)

    # Calculate Money Flow Volume
    mfv = []
    for i in range(len(closes)):
        if highs[i] == lows[i]:
            mfv.append(0)
        else:
            mfm = ((closes[i] - lows[i]) - (highs[i] - closes[i])) / (highs[i] - lows[i])
            mfv.append(mfm * volumes[i])

    result = [None] * (period - 1)

    for i in range(period - 1, len(closes)):
        sum_mfv = sum(mfv[i - period + 1:i + 1])
        sum_vol = sum(volumes[i - period + 1:i + 1])

        if sum_vol == 0:
            result.append(0)
        else:
            result.append(sum_mfv / sum_vol)

    return result


# =============================================================================
# SUPPORT/RESISTANCE
# =============================================================================

def pivot_points(high: float, low: float, close: float) -> dict:
    """
    Classic Pivot Points

    Calculate daily support/resistance levels from previous day's OHLC.

    Formula:
        Pivot (P) = (high + low + close) / 3
        R1 = 2*P - low
        S1 = 2*P - high
        R2 = P + (high - low)
        S2 = P - (high - low)
        R3 = high + 2*(P - low)
        S3 = low - 2*(high - P)

    Args:
        high: Previous day's high
        low: Previous day's low
        close: Previous day's close

    Returns:
        Dict with P, R1, R2, R3, S1, S2, S3
    """
    pivot = (high + low + close) / 3

    return {
        "P": pivot,
        "R1": 2 * pivot - low,
        "R2": pivot + (high - low),
        "R3": high + 2 * (pivot - low),
        "S1": 2 * pivot - high,
        "S2": pivot - (high - low),
        "S3": low - 2 * (high - pivot)
    }


def fibonacci_retracement(swing_high: float, swing_low: float) -> dict:
    """
    Fibonacci Retracement Levels

    Calculate key retracement levels between a swing high and low.

    Key levels: 23.6%, 38.2%, 50%, 61.8%, 78.6%

    Args:
        swing_high: Recent swing high price
        swing_low: Recent swing low price

    Returns:
        Dict with retracement levels (for downtrend from high to low)
    """
    diff = swing_high - swing_low

    return {
        "high": swing_high,
        "low": swing_low,
        "23.6%": swing_high - 0.236 * diff,
        "38.2%": swing_high - 0.382 * diff,
        "50%": swing_high - 0.5 * diff,
        "61.8%": swing_high - 0.618 * diff,
        "78.6%": swing_high - 0.786 * diff
    }


# =============================================================================
# ICC MARKET STRUCTURE (Trades By Sci methodology)
# =============================================================================

def detect_swing_points(highs: List[float],
                        lows: List[float],
                        lookback: int = 5) -> Tuple[List[int], List[int]]:
    """
    Detect swing highs and swing lows for ICC analysis.

    A swing high is a high that is higher than 'lookback' bars on each side.
    A swing low is a low that is lower than 'lookback' bars on each side.

    Args:
        highs: List of high prices
        lows: List of low prices
        lookback: Bars to check on each side (default 5)

    Returns:
        Tuple of (swing_high_indices, swing_low_indices)
    """
    swing_highs = []
    swing_lows = []

    for i in range(lookback, len(highs) - lookback):
        # Check for swing high
        is_swing_high = True
        for j in range(1, lookback + 1):
            if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                is_swing_high = False
                break
        if is_swing_high:
            swing_highs.append(i)

        # Check for swing low
        is_swing_low = True
        for j in range(1, lookback + 1):
            if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                is_swing_low = False
                break
        if is_swing_low:
            swing_lows.append(i)

    return swing_highs, swing_lows


def detect_market_structure(highs: List[float],
                            lows: List[float],
                            closes: List[float],
                            lookback: int = 5) -> List[dict]:
    """
    Detect market structure for ICC methodology.

    Identifies:
    - Higher highs (HH) and higher lows (HL) = uptrend
    - Lower highs (LH) and lower lows (LL) = downtrend
    - Structure breaks = potential reversals (ICC Indication phase)

    Args:
        highs: List of high prices
        lows: List of low prices
        closes: List of close prices
        lookback: Bars for swing detection

    Returns:
        List of structure events with type, index, and price
    """
    swing_high_idx, swing_low_idx = detect_swing_points(highs, lows, lookback)

    events = []

    # Track consecutive swing highs
    for i in range(1, len(swing_high_idx)):
        prev_idx = swing_high_idx[i - 1]
        curr_idx = swing_high_idx[i]

        if highs[curr_idx] > highs[prev_idx]:
            events.append({
                "type": "HH",  # Higher High
                "index": curr_idx,
                "price": highs[curr_idx],
                "prev_price": highs[prev_idx]
            })
        else:
            events.append({
                "type": "LH",  # Lower High
                "index": curr_idx,
                "price": highs[curr_idx],
                "prev_price": highs[prev_idx]
            })

    # Track consecutive swing lows
    for i in range(1, len(swing_low_idx)):
        prev_idx = swing_low_idx[i - 1]
        curr_idx = swing_low_idx[i]

        if lows[curr_idx] > lows[prev_idx]:
            events.append({
                "type": "HL",  # Higher Low
                "index": curr_idx,
                "price": lows[curr_idx],
                "prev_price": lows[prev_idx]
            })
        else:
            events.append({
                "type": "LL",  # Lower Low
                "index": curr_idx,
                "price": lows[curr_idx],
                "prev_price": lows[prev_idx]
            })

    # Sort by index
    events.sort(key=lambda x: x["index"])

    return events


def identify_icc_phases(highs: List[float],
                        lows: List[float],
                        closes: List[float],
                        lookback: int = 5) -> List[dict]:
    """
    Identify ICC (Indication, Correction, Continuation) phases.

    ICC Methodology (Trades By Sci):
    1. INDICATION: Price breaks structure (new HH or LL)
    2. CORRECTION: Counter-trend pullback (liquidity grab)
    3. CONTINUATION: Trend resumes - this is the entry point

    Args:
        highs: List of high prices
        lows: List of low prices
        closes: List of close prices
        lookback: Bars for swing detection

    Returns:
        List of ICC signals with phase, direction, and entry level
    """
    structure = detect_market_structure(highs, lows, closes, lookback)

    if len(structure) < 3:
        return []

    signals = []

    for i in range(2, len(structure)):
        prev2 = structure[i - 2]
        prev1 = structure[i - 1]
        curr = structure[i]

        # Bullish ICC: HH (indication) -> HL (correction) -> HH (continuation)
        if prev2["type"] == "HH" and prev1["type"] == "HL" and curr["type"] == "HH":
            signals.append({
                "phase": "continuation",
                "direction": "bullish",
                "entry_index": curr["index"],
                "entry_price": curr["price"],
                "stop_loss": prev1["price"],  # Below the higher low
                "indication_price": prev2["price"],
                "correction_price": prev1["price"]
            })

        # Bearish ICC: LL (indication) -> LH (correction) -> LL (continuation)
        if prev2["type"] == "LL" and prev1["type"] == "LH" and curr["type"] == "LL":
            signals.append({
                "phase": "continuation",
                "direction": "bearish",
                "entry_index": curr["index"],
                "entry_price": curr["price"],
                "stop_loss": prev1["price"],  # Above the lower high
                "indication_price": prev2["price"],
                "correction_price": prev1["price"]
            })

    return signals


# =============================================================================
# WRAPPER CLASS
# =============================================================================

class TechnicalIndicators:
    """
    Convenience wrapper for all technical indicators.

    Usage:
        ti = TechnicalIndicators(closes, highs, lows, volumes)

        # Individual indicators
        ti.sma(20)
        ti.rsi()
        ti.macd()

        # All indicators at once
        all_indicators = ti.calculate_all()
    """

    def __init__(self,
                 closes: List[float],
                 highs: List[float] = None,
                 lows: List[float] = None,
                 volumes: List[float] = None):
        self.closes = closes
        self.highs = highs or closes
        self.lows = lows or closes
        self.volumes = volumes or [1] * len(closes)

    def sma(self, period: int = 20) -> List[Optional[float]]:
        return sma(self.closes, period)

    def ema(self, period: int = 20) -> List[Optional[float]]:
        return ema(self.closes, period)

    def macd(self, fast: int = 12, slow: int = 26, signal: int = 9):
        return macd(self.closes, fast, slow, signal)

    def rsi(self, period: int = 14) -> List[Optional[float]]:
        return rsi(self.closes, period)

    def stochastic(self, k_period: int = 14, d_period: int = 3):
        return stochastic(self.highs, self.lows, self.closes, k_period, d_period)

    def adx(self, period: int = 14):
        return adx(self.highs, self.lows, self.closes, period)

    def bollinger_bands(self, period: int = 20, num_std: float = 2.0):
        return bollinger_bands(self.closes, period, num_std)

    def atr(self, period: int = 14) -> List[Optional[float]]:
        return atr(self.highs, self.lows, self.closes, period)

    def vwap(self) -> List[Optional[float]]:
        return vwap(self.highs, self.lows, self.closes, self.volumes)

    def obv(self) -> List[float]:
        return obv(self.closes, self.volumes)

    def williams_r(self, period: int = 14) -> List[Optional[float]]:
        return williams_r(self.highs, self.lows, self.closes, period)

    def cci(self, period: int = 20) -> List[Optional[float]]:
        return cci(self.highs, self.lows, self.closes, period)

    def chaikin_money_flow(self, period: int = 20) -> List[Optional[float]]:
        return chaikin_money_flow(self.highs, self.lows, self.closes, self.volumes, period)

    def icc_signals(self, lookback: int = 5) -> List[dict]:
        """Get ICC trading signals."""
        return identify_icc_phases(self.highs, self.lows, self.closes, lookback)

    def market_structure(self, lookback: int = 5) -> List[dict]:
        """Get market structure events."""
        return detect_market_structure(self.highs, self.lows, self.closes, lookback)

    def calculate_all(self) -> dict:
        """
        Calculate all indicators and return as dict.

        Returns:
            Dict with all indicator values
        """
        macd_line, signal_line, histogram = self.macd()
        upper_bb, middle_bb, lower_bb = self.bollinger_bands()
        stoch_k, stoch_d = self.stochastic()
        adx_val, plus_di, minus_di = self.adx()

        return {
            "sma_20": self.sma(20),
            "sma_50": self.sma(50),
            "sma_200": self.sma(200),
            "ema_12": self.ema(12),
            "ema_26": self.ema(26),
            "macd_line": macd_line,
            "macd_signal": signal_line,
            "macd_histogram": histogram,
            "rsi": self.rsi(),
            "stoch_k": stoch_k,
            "stoch_d": stoch_d,
            "adx": adx_val,
            "plus_di": plus_di,
            "minus_di": minus_di,
            "bb_upper": upper_bb,
            "bb_middle": middle_bb,
            "bb_lower": lower_bb,
            "atr": self.atr(),
            "vwap": self.vwap(),
            "obv": self.obv(),
            "williams_r": self.williams_r(),
            "cci": self.cci(),
            "cmf": self.chaikin_money_flow(),
            "icc_signals": self.icc_signals(),
            "market_structure": self.market_structure()
        }


if __name__ == "__main__":
    # Test with sample data
    test_closes = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109,
                   110, 108, 106, 104, 102, 103, 105, 107, 109, 111,
                   113, 115, 114, 116, 118]
    test_highs = [c + 1 for c in test_closes]
    test_lows = [c - 1 for c in test_closes]
    test_volumes = [1000000] * len(test_closes)

    ti = TechnicalIndicators(test_closes, test_highs, test_lows, test_volumes)

    print("SMA(5):", ti.sma(5)[-5:])
    print("RSI:", [f"{v:.1f}" if v else "None" for v in ti.rsi()[-5:]])
    print("VWAP:", [f"{v:.2f}" if v else "None" for v in ti.vwap()[-5:]])

    # Test ICC
    signals = ti.icc_signals()
    print(f"ICC Signals found: {len(signals)}")
    for s in signals:
        print(f"  {s['direction'].upper()} at index {s['entry_index']}")
