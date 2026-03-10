#!/usr/bin/env python3
"""
ta_indicators.py - Technical Analysis Indicators for MIDGE

Computes RSI, MACD, Bollinger Bands, Market Structure, and Candlestick
Patterns from OHLCV price history. Pure computation — no side effects,
no EventBus, no external calls.

Input: List[PriceData] from price_fetcher.get_daily_history()
Output: List of typed signal dataclasses (one per actionable indicator reading)

These signals flow through the same convergence pipeline as all other
MIDGE signals — Thompson Sampling learns which indicators actually work.

Research backing (Trades by Sci / MIDGE session eb00919a):
- RSI oversold in high volatility = 62.5% win rate
- MACD crossover alone = 53.5%; combined with RSI = higher
- Combination signals outperform singles consistently
"""

import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal dataclasses — one per indicator type
# ---------------------------------------------------------------------------

@dataclass
class TASignalBase:
    """Base for all TA signals. Matches ClusterSignal/SessionSweepSignal pattern."""
    signal_id: str = ""
    symbol: str = ""
    indicator: str = ""       # "rsi", "macd", "bollinger", "structure", "candle"
    direction: str = ""       # "bullish", "bearish"
    strength: float = 0.0     # 0.0-1.0
    confidence: float = 0.50
    decay_rate: float = 0.05  # ~14 day half-life (daily TA signals)
    detected_at: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class RSISignal(TASignalBase):
    """RSI indicator signal."""
    indicator: str = "rsi"
    value: float = 50.0       # RSI value (0-100)
    zone: str = "neutral"     # "oversold", "overbought", "neutral"
    decay_rate: float = 0.05


@dataclass
class MACDSignal(TASignalBase):
    """MACD crossover signal."""
    indicator: str = "macd"
    crossover_type: str = ""  # "bullish_crossover", "bearish_crossover"
    macd_value: float = 0.0
    signal_value: float = 0.0
    histogram: float = 0.0
    histogram_slope: float = 0.0  # Change in histogram (momentum)
    decay_rate: float = 0.05


@dataclass
class BollingerSignal(TASignalBase):
    """Bollinger Band signal."""
    indicator: str = "bollinger"
    band_position: float = 0.5  # 0.0=lower band, 0.5=middle, 1.0=upper band
    bandwidth: float = 0.0      # Band width as % of middle band
    squeeze: bool = False        # Bandwidth narrowing = breakout imminent
    price: float = 0.0
    upper_band: float = 0.0
    middle_band: float = 0.0
    lower_band: float = 0.0
    decay_rate: float = 0.05


@dataclass
class StructureSignal(TASignalBase):
    """Market structure signal (higher highs / lower lows)."""
    indicator: str = "structure"
    structure_type: str = ""  # "uptrend", "downtrend", "break_of_structure"
    swing_high: float = 0.0
    swing_low: float = 0.0
    prev_swing_high: float = 0.0
    prev_swing_low: float = 0.0
    decay_rate: float = 0.07  # ~10 day half-life (structure signals last)


@dataclass
class CandleSignal(TASignalBase):
    """Candlestick pattern signal."""
    indicator: str = "candle"
    pattern: str = ""  # "bullish_engulfing", "bearish_engulfing", "hammer", etc.
    decay_rate: float = 0.10  # ~7 day half-life (candle patterns are short-lived)


# ---------------------------------------------------------------------------
# TA computation functions — pure math on price arrays
# ---------------------------------------------------------------------------

def _extract_ohlcv(price_history) -> Tuple[
    List[float], List[float], List[float], List[float], List[int], List[str]
]:
    """Extract OHLCV arrays from PriceData list."""
    opens = [p.open for p in price_history]
    highs = [p.high for p in price_history]
    lows = [p.low for p in price_history]
    closes = [p.price for p in price_history]
    volumes = [p.volume for p in price_history]
    timestamps = [p.timestamp for p in price_history]
    return opens, highs, lows, closes, volumes, timestamps


def compute_rsi(symbol: str, price_history, period: int = 14) -> Optional[RSISignal]:
    """Compute RSI using Wilder's smoothed moving average (vectorized).

    Standard 14-period RSI. Only returns a signal when RSI is in
    oversold (<30) or overbought (>70) territory.

    Args:
        symbol: Ticker symbol
        price_history: List of PriceData (oldest first)
        period: RSI period (default 14)

    Returns:
        RSISignal if actionable, None if neutral
    """
    if len(price_history) < period + 1:
        return None

    closes = np.array([p.price for p in price_history], dtype=float)
    deltas = np.diff(closes)

    if len(deltas) < period:
        return None

    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Wilder smoothing = EMA with alpha = 1/period (com = period - 1)
    avg_gain_series = pd.Series(gains).ewm(com=period - 1, min_periods=period).mean()
    avg_loss_series = pd.Series(losses).ewm(com=period - 1, min_periods=period).mean()

    avg_gain = float(avg_gain_series.iloc[-1])
    avg_loss = float(avg_loss_series.iloc[-1])

    if avg_loss == 0:
        rsi = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

    # Determine zone and direction
    if rsi < 30:
        zone = "oversold"
        direction = "bullish"
        # Strength: how deeply oversold (RSI 20 stronger than RSI 29)
        strength = min(1.0, (30 - rsi) / 30)
        confidence = 0.60
    elif rsi > 70:
        zone = "overbought"
        direction = "bearish"
        strength = min(1.0, (rsi - 70) / 30)
        confidence = 0.55
    else:
        return None  # Neutral — no signal

    now = datetime.now().isoformat()
    return RSISignal(
        signal_id=f"ta_rsi:{symbol}:{now}",
        symbol=symbol,
        direction=direction,
        strength=strength,
        confidence=confidence,
        detected_at=now,
        value=round(rsi, 2),
        zone=zone,
        metadata={"period": period, "avg_gain": round(avg_gain, 6), "avg_loss": round(avg_loss, 6)},
    )


def compute_macd(
    symbol: str,
    price_history,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> Optional[MACDSignal]:
    """Compute MACD and detect crossover events (vectorized).

    Standard 12/26/9 EMA configuration. Only returns a signal when a
    crossover occurred in the most recent bar.

    Args:
        symbol: Ticker symbol
        price_history: List of PriceData (oldest first)
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal_period: Signal line EMA period (default 9)

    Returns:
        MACDSignal if crossover detected, None otherwise
    """
    if len(price_history) < slow + signal_period:
        return None

    closes_series = pd.Series([p.price for p in price_history], dtype=float)

    # Vectorized EMA via pandas ewm (span = period, adjust=False matches iterative EMA)
    fast_ema = closes_series.ewm(span=fast, adjust=False).mean()
    slow_ema = closes_series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line

    if len(histogram) < 2:
        return None

    # Detect crossover: sign change in last two bars
    prev_hist = float(histogram.iloc[-2])
    curr_hist = float(histogram.iloc[-1])

    if prev_hist <= 0 and curr_hist > 0:
        crossover_type = "bullish_crossover"
        direction = "bullish"
    elif prev_hist >= 0 and curr_hist < 0:
        crossover_type = "bearish_crossover"
        direction = "bearish"
    else:
        return None  # No crossover

    # Strength from histogram magnitude relative to price
    price = float(closes_series.iloc[-1])
    strength = min(1.0, abs(curr_hist) / (price * 0.02)) if price > 0 else 0.5

    # Histogram slope (momentum confirmation)
    hist_slope = curr_hist - prev_hist

    now = datetime.now().isoformat()
    return MACDSignal(
        signal_id=f"ta_macd:{symbol}:{now}",
        symbol=symbol,
        direction=direction,
        strength=strength,
        confidence=0.53,
        detected_at=now,
        crossover_type=crossover_type,
        macd_value=round(float(macd_line.iloc[-1]), 4),
        signal_value=round(float(signal_line.iloc[-1]), 4),
        histogram=round(curr_hist, 4),
        histogram_slope=round(hist_slope, 4),
        metadata={"fast": fast, "slow": slow, "signal_period": signal_period},
    )


def compute_bollinger(
    symbol: str,
    price_history,
    period: int = 20,
    num_std: float = 2.0,
) -> Optional[BollingerSignal]:
    """Compute Bollinger Bands and detect band touches or squeezes (vectorized).

    20-period SMA +/- 2 standard deviations. Signals when price touches
    or exceeds bands, or when bandwidth contracts (squeeze = breakout imminent).

    Args:
        symbol: Ticker symbol
        price_history: List of PriceData (oldest first)
        period: SMA period (default 20)
        num_std: Number of standard deviations (default 2.0)

    Returns:
        BollingerSignal if actionable, None if price is mid-band and no squeeze
    """
    if len(price_history) < period:
        return None

    closes_arr = np.array([p.price for p in price_history], dtype=float)
    closes_s = pd.Series(closes_arr)

    # Vectorized rolling SMA and population std (ddof=0 matches original)
    rolling_mean = closes_s.rolling(period).mean()
    rolling_std = closes_s.rolling(period).std(ddof=0)

    middle = float(rolling_mean.iloc[-1])
    std_dev = float(rolling_std.iloc[-1])

    if middle == 0 or np.isnan(middle) or np.isnan(std_dev):
        return None

    upper = middle + num_std * std_dev
    lower = middle - num_std * std_dev

    # Band width as percentage of middle band
    bandwidth = (upper - lower) / middle

    # Band position: 0.0 = at lower band, 0.5 = middle, 1.0 = at upper band
    band_range = upper - lower
    if band_range == 0:
        return None
    current_price = float(closes_arr[-1])
    band_position = (current_price - lower) / band_range
    band_position = max(0.0, min(1.0, band_position))

    # Squeeze detection: compare current bandwidth to 120-day average (vectorized)
    squeeze = False
    if len(closes_arr) >= 120:
        # Rolling bandwidth over last 120 bars — fully vectorized
        hist_mean = rolling_mean.iloc[-(101):-(1)]   # 100 prior bars' SMA
        hist_std = rolling_std.iloc[-(101):-(1)]
        valid = hist_mean > 0
        if valid.any():
            hist_bw = (2 * num_std * hist_std[valid]) / hist_mean[valid]
            avg_bw = float(hist_bw.mean())
            squeeze = bandwidth < avg_bw * 0.75

    # Signal conditions
    if band_position <= 0.05:
        # Price at or below lower band = oversold bounce potential
        direction = "bullish"
        strength = min(1.0, (0.05 - band_position) / 0.05 + 0.5)
        confidence = 0.55
    elif band_position >= 0.95:
        # Price at or above upper band = overbought reversal potential
        direction = "bearish"
        strength = min(1.0, (band_position - 0.95) / 0.05 + 0.5)
        confidence = 0.55
    elif squeeze:
        # Squeeze = breakout imminent, direction unknown
        # Use recent momentum to guess direction
        if len(closes_arr) >= 5:
            recent_momentum = float(closes_arr[-1]) - float(closes_arr[-5])
            direction = "bullish" if recent_momentum > 0 else "bearish"
        else:
            direction = "bullish"
        strength = 0.4
        confidence = 0.45
    else:
        return None  # Mid-band, no squeeze — no signal

    now = datetime.now().isoformat()
    return BollingerSignal(
        signal_id=f"ta_bollinger:{symbol}:{now}",
        symbol=symbol,
        direction=direction,
        strength=strength,
        confidence=confidence,
        detected_at=now,
        band_position=round(band_position, 4),
        bandwidth=round(bandwidth, 4),
        squeeze=squeeze,
        price=current_price,
        upper_band=round(upper, 2),
        middle_band=round(middle, 2),
        lower_band=round(lower, 2),
        metadata={"period": period, "num_std": num_std, "std_dev": round(std_dev, 4)},
    )


def compute_market_structure(
    symbol: str,
    price_history,
    lookback: int = 5,
) -> Optional[StructureSignal]:
    """Detect market structure: uptrend (HH+HL), downtrend (LH+LL), break of structure.

    Uses swing point detection with a lookback window. A swing high is a
    bar whose high is higher than the `lookback` bars on each side. Same
    for swing lows. Then compares the two most recent swing highs and
    two most recent swing lows to determine structure.

    Break of structure (BOS) is the strongest signal — it's a trend reversal
    where the prior swing point is violated.

    Args:
        symbol: Ticker symbol
        price_history: List of PriceData (oldest first)
        lookback: Bars on each side for swing detection (default 5)

    Returns:
        StructureSignal if structure is clear, None if insufficient data
    """
    if len(price_history) < lookback * 2 + 3:
        return None

    highs = [p.high for p in price_history]
    lows = [p.low for p in price_history]

    # Find swing highs and lows
    swing_highs: List[Tuple[int, float]] = []
    swing_lows: List[Tuple[int, float]] = []

    for i in range(lookback, len(highs) - lookback):
        # Swing high: higher than all neighbors in lookback window
        if all(highs[i] >= highs[i - j] for j in range(1, lookback + 1)) and \
           all(highs[i] >= highs[i + j] for j in range(1, lookback + 1)):
            swing_highs.append((i, highs[i]))

        # Swing low: lower than all neighbors in lookback window
        if all(lows[i] <= lows[i - j] for j in range(1, lookback + 1)) and \
           all(lows[i] <= lows[i + j] for j in range(1, lookback + 1)):
            swing_lows.append((i, lows[i]))

    # Need at least 2 swing highs and 2 swing lows for structure analysis
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return None

    # Most recent two swings
    sh1_idx, sh1 = swing_highs[-2]
    sh2_idx, sh2 = swing_highs[-1]
    sl1_idx, sl1 = swing_lows[-2]
    sl2_idx, sl2 = swing_lows[-1]

    current_price = price_history[-1].price

    # Determine structure
    higher_high = sh2 > sh1
    higher_low = sl2 > sl1
    lower_high = sh2 < sh1
    lower_low = sl2 < sl1

    if higher_high and higher_low:
        structure_type = "uptrend"
        direction = "bullish"
        strength = 0.5
        confidence = 0.55
    elif lower_high and lower_low:
        structure_type = "downtrend"
        direction = "bearish"
        strength = 0.5
        confidence = 0.55
    else:
        # Mixed — check for break of structure
        # BOS bullish: was making lower lows, now price breaks above last swing high
        if lower_low and current_price > sh2:
            structure_type = "break_of_structure"
            direction = "bullish"
            strength = 0.75
            confidence = 0.60
        # BOS bearish: was making higher highs, now price breaks below last swing low
        elif higher_high and current_price < sl2:
            structure_type = "break_of_structure"
            direction = "bearish"
            strength = 0.75
            confidence = 0.60
        else:
            return None  # No clear structure

    now = datetime.now().isoformat()
    return StructureSignal(
        signal_id=f"ta_structure:{symbol}:{now}",
        symbol=symbol,
        direction=direction,
        strength=strength,
        confidence=confidence,
        detected_at=now,
        structure_type=structure_type,
        swing_high=sh2,
        swing_low=sl2,
        prev_swing_high=sh1,
        prev_swing_low=sl1,
        metadata={
            "lookback": lookback,
            "num_swing_highs": len(swing_highs),
            "num_swing_lows": len(swing_lows),
        },
    )


def compute_candlestick_patterns(
    symbol: str,
    price_history,
) -> Optional[CandleSignal]:
    """Detect candlestick patterns in the most recent 2 bars.

    Patterns detected:
    - Bullish/bearish engulfing (2-candle reversal)
    - Hammer (1-candle, long lower shadow)
    - Shooting star (1-candle, long upper shadow)
    - Doji (1-candle, open ~ close)

    Only returns the strongest pattern found. Returns None if no pattern.

    Args:
        symbol: Ticker symbol
        price_history: List of PriceData (oldest first)

    Returns:
        CandleSignal if pattern detected, None otherwise
    """
    if len(price_history) < 2:
        return None

    curr = price_history[-1]
    prev = price_history[-2]

    curr_body = abs(curr.price - curr.open)
    prev_body = abs(prev.price - prev.open)
    curr_range = curr.high - curr.low
    prev_range = prev.high - prev.low

    if curr_range == 0:
        return None

    curr_is_green = curr.price > curr.open
    prev_is_green = prev.price > prev.open

    curr_real_body_top = max(curr.open, curr.price)
    curr_real_body_bottom = min(curr.open, curr.price)
    prev_real_body_top = max(prev.open, prev.price)
    prev_real_body_bottom = min(prev.open, prev.price)

    upper_shadow = curr.high - curr_real_body_top
    lower_shadow = curr_real_body_bottom - curr.low

    pattern = None
    direction = ""
    strength = 0.0
    confidence = 0.0

    # --- Bullish engulfing ---
    # Previous red candle fully engulfed by current green candle
    if (not prev_is_green and curr_is_green and
            curr_real_body_bottom <= prev_real_body_bottom and
            curr_real_body_top >= prev_real_body_top and
            curr_body > prev_body * 0.5):
        pattern = "bullish_engulfing"
        direction = "bullish"
        strength = min(1.0, curr_body / curr_range)
        confidence = 0.58

    # --- Bearish engulfing ---
    # Previous green candle fully engulfed by current red candle
    elif (prev_is_green and not curr_is_green and
            curr_real_body_bottom <= prev_real_body_bottom and
            curr_real_body_top >= prev_real_body_top and
            curr_body > prev_body * 0.5):
        pattern = "bearish_engulfing"
        direction = "bearish"
        strength = min(1.0, curr_body / curr_range)
        confidence = 0.58

    # --- Hammer (bullish reversal) ---
    # Small body at top, long lower shadow (2x+ body), little upper shadow
    elif (curr_body > 0 and
            lower_shadow >= curr_body * 2 and
            upper_shadow <= curr_body * 0.5):
        pattern = "hammer"
        direction = "bullish"
        strength = min(1.0, lower_shadow / curr_range)
        confidence = 0.55

    # --- Shooting star (bearish reversal) ---
    # Small body at bottom, long upper shadow (2x+ body), little lower shadow
    elif (curr_body > 0 and
            upper_shadow >= curr_body * 2 and
            lower_shadow <= curr_body * 0.5):
        pattern = "shooting_star"
        direction = "bearish"
        strength = min(1.0, upper_shadow / curr_range)
        confidence = 0.55

    # --- Doji (indecision, reversal possible) ---
    # Body is very small relative to range
    elif curr_range > 0 and curr_body / curr_range < 0.1:
        pattern = "doji"
        # Direction from context: doji after uptrend = bearish, after downtrend = bullish
        if len(price_history) >= 5:
            trend = price_history[-1].price - price_history[-5].price
            direction = "bearish" if trend > 0 else "bullish"
        else:
            return None  # Doji without context is noise
        strength = 0.3
        confidence = 0.45

    if pattern is None:
        return None

    now = datetime.now().isoformat()
    return CandleSignal(
        signal_id=f"ta_candle:{symbol}:{now}",
        symbol=symbol,
        direction=direction,
        strength=strength,
        confidence=confidence,
        detected_at=now,
        pattern=pattern,
        metadata={
            "curr_open": curr.open,
            "curr_close": curr.price,
            "curr_high": curr.high,
            "curr_low": curr.low,
            "prev_open": prev.open,
            "prev_close": prev.price,
        },
    )


def compute_atr(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    period: int = 14,
) -> float:
    """Compute Average True Range (ATR) using Wilder's smoothed moving average (vectorized).

    ATR measures volatility as the average of True Range over `period` bars.
    True Range = max(high-low, |high-prev_close|, |low-prev_close|).

    Used by the signal translator to set ATR-proportional stop-loss and
    take-profit levels that adapt to the instrument's current volatility.

    Args:
        highs:  List of bar high prices (oldest first, length >= period+1)
        lows:   List of bar low prices  (oldest first, length >= period+1)
        closes: List of bar close/last prices (oldest first, same length)
        period: ATR smoothing period (default 14)

    Returns:
        ATR value as a positive float, or 0.0 when insufficient data.
    """
    if len(highs) < period + 1 or len(lows) < period + 1 or len(closes) < period + 1:
        return 0.0

    h = np.array(highs, dtype=float)
    l = np.array(lows, dtype=float)
    c = np.array(closes, dtype=float)

    # True Range for each bar (starting from bar 1 so we have prev_close)
    prev_c = c[:-1]
    curr_h = h[1:]
    curr_l = l[1:]

    hl   = curr_h - curr_l
    h_pc = np.abs(curr_h - prev_c)
    l_pc = np.abs(curr_l - prev_c)

    tr_series = pd.Series(np.maximum(hl, np.maximum(h_pc, l_pc)))

    # Wilder smoothing = EMA with alpha=1/period (com = period-1), same as compute_rsi
    atr_series = tr_series.ewm(com=period - 1, min_periods=period).mean()
    val = float(atr_series.iloc[-1])
    return val if not math.isnan(val) else 0.0


# ---------------------------------------------------------------------------
# Top-level computation function
# ---------------------------------------------------------------------------

def compute_all(symbol: str, price_history) -> List[TASignalBase]:
    """Run all TA indicators and return actionable signals.

    Args:
        symbol: Ticker symbol
        price_history: List of PriceData (oldest first, from price_fetcher.get_daily_history)

    Returns:
        List of TASignalBase subclass instances (only actionable, not neutral)
    """
    if not price_history:
        return []

    signals: List[TASignalBase] = []

    # RSI
    try:
        rsi = compute_rsi(symbol, price_history)
        if rsi is not None:
            signals.append(rsi)
    except Exception:
        logger.debug("RSI computation failed for %s", symbol, exc_info=True)

    # MACD
    try:
        macd = compute_macd(symbol, price_history)
        if macd is not None:
            signals.append(macd)
    except Exception:
        logger.debug("MACD computation failed for %s", symbol, exc_info=True)

    # Bollinger Bands
    try:
        bollinger = compute_bollinger(symbol, price_history)
        if bollinger is not None:
            signals.append(bollinger)
    except Exception:
        logger.debug("Bollinger computation failed for %s", symbol, exc_info=True)

    # Market Structure
    try:
        structure = compute_market_structure(symbol, price_history)
        if structure is not None:
            signals.append(structure)
    except Exception:
        logger.debug("Market structure computation failed for %s", symbol, exc_info=True)

    # Candlestick Patterns
    try:
        candle = compute_candlestick_patterns(symbol, price_history)
        if candle is not None:
            signals.append(candle)
    except Exception:
        logger.debug("Candlestick pattern detection failed for %s", symbol, exc_info=True)

    return signals


# ---------------------------------------------------------------------------
# Performance benchmark (run directly: python -m mae_core.market.edge.ta_indicators)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    class _FakePrice:
        """Minimal stand-in for PriceData for benchmark purposes."""
        __slots__ = ("price", "open", "high", "low", "volume", "timestamp")

        def __init__(self, price: float, i: int):
            self.price = price
            self.open = price * 0.999
            self.high = price * 1.005
            self.low = price * 0.995
            self.volume = 1_000_000
            self.timestamp = f"2025-{(i // 30) % 12 + 1:02d}-{i % 28 + 1:02d}"

    rng = np.random.default_rng(42)
    raw_prices = rng.uniform(100, 200, 1000).tolist()
    fake_history = [_FakePrice(p, i) for i, p in enumerate(raw_prices)]

    # Warm-up
    compute_rsi("BENCH", fake_history)
    compute_bollinger("BENCH", fake_history)
    compute_macd("BENCH", fake_history)

    iterations = 100
    start = time.perf_counter()
    for _ in range(iterations):
        compute_rsi("BENCH", fake_history, 14)
        compute_bollinger("BENCH", fake_history, 20, 2.0)
        compute_macd("BENCH", fake_history, 12, 26, 9)
    elapsed = time.perf_counter() - start
    print(f"Vectorized TA — {iterations} iterations: {elapsed:.3f}s  ({elapsed / iterations * 1000:.1f}ms each)")
