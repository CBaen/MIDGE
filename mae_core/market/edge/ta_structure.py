"""
ta_structure.py - Market structure, candlestick patterns, and ATR (vectorized)

compute_market_structure, compute_candlestick_patterns, compute_atr.
Pure computation — no side effects.
"""

import logging
import math
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from mae_core.market.edge.ta_models import StructureSignal, CandleSignal

logger = logging.getLogger(__name__)


def compute_market_structure(
    symbol: str,
    price_history,
    lookback: int = 5,
) -> Optional[StructureSignal]:
    """Detect market structure: uptrend (HH+HL), downtrend (LH+LL), break of structure.

    Uses swing point detection with a lookback window.
    """
    if len(price_history) < lookback * 2 + 3:
        return None

    highs = [p.high for p in price_history]
    lows = [p.low for p in price_history]

    swing_highs: List[Tuple[int, float]] = []
    swing_lows: List[Tuple[int, float]] = []

    for i in range(lookback, len(highs) - lookback):
        if all(highs[i] >= highs[i - j] for j in range(1, lookback + 1)) and \
           all(highs[i] >= highs[i + j] for j in range(1, lookback + 1)):
            swing_highs.append((i, highs[i]))

        if all(lows[i] <= lows[i - j] for j in range(1, lookback + 1)) and \
           all(lows[i] <= lows[i + j] for j in range(1, lookback + 1)):
            swing_lows.append((i, lows[i]))

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return None

    sh1_idx, sh1 = swing_highs[-2]
    sh2_idx, sh2 = swing_highs[-1]
    sl1_idx, sl1 = swing_lows[-2]
    sl2_idx, sl2 = swing_lows[-1]

    current_price = price_history[-1].price

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
        if lower_low and current_price > sh2:
            structure_type = "break_of_structure"
            direction = "bullish"
            strength = 0.75
            confidence = 0.60
        elif higher_high and current_price < sl2:
            structure_type = "break_of_structure"
            direction = "bearish"
            strength = 0.75
            confidence = 0.60
        else:
            return None

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

    Patterns: bullish/bearish engulfing, hammer, shooting star, doji.
    Returns None if no pattern detected.
    """
    if len(price_history) < 2:
        return None

    curr = price_history[-1]
    prev = price_history[-2]

    curr_body = abs(curr.price - curr.open)
    prev_body = abs(prev.price - prev.open)
    curr_range = curr.high - curr.low

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

    if (not prev_is_green and curr_is_green and
            curr_real_body_bottom <= prev_real_body_bottom and
            curr_real_body_top >= prev_real_body_top and
            curr_body > prev_body * 0.5):
        pattern = "bullish_engulfing"
        direction = "bullish"
        strength = min(1.0, curr_body / curr_range)
        confidence = 0.58

    elif (prev_is_green and not curr_is_green and
            curr_real_body_bottom <= prev_real_body_bottom and
            curr_real_body_top >= prev_real_body_top and
            curr_body > prev_body * 0.5):
        pattern = "bearish_engulfing"
        direction = "bearish"
        strength = min(1.0, curr_body / curr_range)
        confidence = 0.58

    elif (curr_body > 0 and
            lower_shadow >= curr_body * 2 and
            upper_shadow <= curr_body * 0.5):
        pattern = "hammer"
        direction = "bullish"
        strength = min(1.0, lower_shadow / curr_range)
        confidence = 0.55

    elif (curr_body > 0 and
            upper_shadow >= curr_body * 2 and
            lower_shadow <= curr_body * 0.5):
        pattern = "shooting_star"
        direction = "bearish"
        strength = min(1.0, upper_shadow / curr_range)
        confidence = 0.55

    elif curr_range > 0 and curr_body / curr_range < 0.1:
        pattern = "doji"
        if len(price_history) >= 5:
            trend = price_history[-1].price - price_history[-5].price
            direction = "bearish" if trend > 0 else "bullish"
        else:
            return None
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
    """
    if len(highs) < period + 1 or len(lows) < period + 1 or len(closes) < period + 1:
        return 0.0

    h = np.array(highs, dtype=float)
    l = np.array(lows, dtype=float)
    c = np.array(closes, dtype=float)

    prev_c = c[:-1]
    curr_h = h[1:]
    curr_l = l[1:]

    hl   = curr_h - curr_l
    h_pc = np.abs(curr_h - prev_c)
    l_pc = np.abs(curr_l - prev_c)

    tr_series = pd.Series(np.maximum(hl, np.maximum(h_pc, l_pc)))
    atr_series = tr_series.ewm(com=period - 1, min_periods=period).mean()
    val = float(atr_series.iloc[-1])
    return val if not math.isnan(val) else 0.0
