"""
ta_oscillators.py - RSI, MACD, and Bollinger Band computations (vectorized)

compute_rsi, compute_macd, compute_bollinger — pure math on price arrays.
Research backing (Trades by Sci / MIDGE session eb00919a):
  RSI oversold in high volatility = 62.5% win rate
  MACD crossover alone = 53.5%; combined with RSI = higher
"""

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from mae_core.market.edge.ta_models import RSISignal, MACDSignal, BollingerSignal

logger = logging.getLogger(__name__)


def compute_rsi(symbol: str, price_history, period: int = 14) -> Optional[RSISignal]:
    """Compute RSI using Wilder's smoothed moving average (vectorized).

    Standard 14-period RSI. Only returns a signal when RSI is in
    oversold (<30) or overbought (>70) territory.
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

    if rsi < 30:
        zone = "oversold"
        direction = "bullish"
        strength = min(1.0, (30 - rsi) / 30)
        confidence = 0.60
    elif rsi > 70:
        zone = "overbought"
        direction = "bearish"
        strength = min(1.0, (rsi - 70) / 30)
        confidence = 0.55
    else:
        return None

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
    """
    if len(price_history) < slow + signal_period:
        return None

    closes_series = pd.Series([p.price for p in price_history], dtype=float)

    fast_ema = closes_series.ewm(span=fast, adjust=False).mean()
    slow_ema = closes_series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line

    if len(histogram) < 2:
        return None

    prev_hist = float(histogram.iloc[-2])
    curr_hist = float(histogram.iloc[-1])

    if prev_hist <= 0 and curr_hist > 0:
        crossover_type = "bullish_crossover"
        direction = "bullish"
    elif prev_hist >= 0 and curr_hist < 0:
        crossover_type = "bearish_crossover"
        direction = "bearish"
    else:
        return None

    price = float(closes_series.iloc[-1])
    strength = min(1.0, abs(curr_hist) / (price * 0.02)) if price > 0 else 0.5
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
    """
    if len(price_history) < period:
        return None

    closes_arr = np.array([p.price for p in price_history], dtype=float)
    closes_s = pd.Series(closes_arr)

    rolling_mean = closes_s.rolling(period).mean()
    rolling_std = closes_s.rolling(period).std(ddof=0)

    middle = float(rolling_mean.iloc[-1])
    std_dev = float(rolling_std.iloc[-1])

    if middle == 0 or np.isnan(middle) or np.isnan(std_dev):
        return None

    upper = middle + num_std * std_dev
    lower = middle - num_std * std_dev

    bandwidth = (upper - lower) / middle

    band_range = upper - lower
    if band_range == 0:
        return None
    current_price = float(closes_arr[-1])
    band_position = (current_price - lower) / band_range
    band_position = max(0.0, min(1.0, band_position))

    squeeze = False
    if len(closes_arr) >= 120:
        hist_mean = rolling_mean.iloc[-(101):-(1)]
        hist_std = rolling_std.iloc[-(101):-(1)]
        valid = hist_mean > 0
        if valid.any():
            hist_bw = (2 * num_std * hist_std[valid]) / hist_mean[valid]
            avg_bw = float(hist_bw.mean())
            squeeze = bandwidth < avg_bw * 0.75

    if band_position <= 0.05:
        direction = "bullish"
        strength = min(1.0, (0.05 - band_position) / 0.05 + 0.5)
        confidence = 0.55
    elif band_position >= 0.95:
        direction = "bearish"
        strength = min(1.0, (band_position - 0.95) / 0.05 + 0.5)
        confidence = 0.55
    elif squeeze:
        if len(closes_arr) >= 5:
            recent_momentum = float(closes_arr[-1]) - float(closes_arr[-5])
            direction = "bullish" if recent_momentum > 0 else "bearish"
        else:
            direction = "bullish"
        strength = 0.4
        confidence = 0.45
    else:
        return None

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
