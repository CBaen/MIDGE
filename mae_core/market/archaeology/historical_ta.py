"""historical_ta.py - TA signal computation for Pattern Archaeology.

Retroactively computes RSI, MACD, Bollinger Band, and volume signals from
price history. These are Tier 1 signals (zero API calls) for the
HistoricalDataFetcher.

All functions are pure — they take price history and date windows and return
lists of signal dicts in archive format, identical to live monitoring output.
"""

from __future__ import annotations

import math
import logging
from datetime import date, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


def _make_signal(
    signal_id: str, source: str, symbol: str, domain: str,
    direction: str, strength: float, date_str: str,
    metadata: Optional[dict] = None,
) -> dict:
    """Create a signal dict in archive format."""
    return {
        "signal_id": signal_id,
        "source": source,
        "symbol": symbol,
        "domain": domain,
        "direction": direction,
        "strength": strength,
        "confidence": 0.5,
        "velocity": 0.0,
        "timestamp": f"{date_str}T00:00:00",
        "received_at": f"{date_str}T00:00:00",
        "metadata": metadata or {},
    }


def compute_ta_signals(
    symbol: str,
    price_history: list,
    start_date: date,
    end_date: date,
) -> list[dict]:
    """Compute TA indicators retroactively from price history.

    For each day in [start_date, end_date), generates signal records for:
    - RSI extreme (oversold < 30, overbought > 70)
    - MACD crossover
    - Bollinger squeeze / breakout
    - Volume anomaly (> 2x 20-day average)
    """
    if len(price_history) < 35:
        return []

    signals: list[dict] = []

    prices_by_date: dict[str, int] = {}
    for idx, p in enumerate(price_history):
        ts = getattr(p, "timestamp", "")[:10]
        if ts:
            prices_by_date[ts] = idx

    signals.extend(compute_rsi_series(symbol, price_history, prices_by_date, start_date, end_date))
    signals.extend(compute_macd_series(symbol, price_history, prices_by_date, start_date, end_date))
    signals.extend(compute_bollinger_series(symbol, price_history, prices_by_date, start_date, end_date))
    signals.extend(compute_volume_signals(symbol, price_history, prices_by_date, start_date, end_date))

    return signals


def compute_rsi_series(
    symbol: str, history: list, date_idx: dict, start: date, end: date,
) -> list[dict]:
    """Compute RSI for all dates and flag extremes in the window."""
    closes = [p.price for p in history]
    if len(closes) < 15:
        return []

    period = 14
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(d, 0) for d in deltas]
    losses_arr = [abs(min(d, 0)) for d in deltas]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses_arr[:period]) / period

    rsi_values: list[Optional[float]] = [None] * (period + 1)
    if avg_loss == 0:
        rsi_values.append(100.0)
    else:
        rs = avg_gain / avg_loss
        rsi_values.append(round(100 - (100 / (1 + rs)), 2))

    for i in range(period + 1, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses_arr[i]) / period
        if avg_loss == 0:
            rsi_values.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsi_values.append(round(100 - (100 / (1 + rs)), 2))

    signals: list[dict] = []
    for idx in range(len(rsi_values)):
        rsi = rsi_values[idx]
        if rsi is None:
            continue
        if idx >= len(history):
            continue
        ts = getattr(history[idx], "timestamp", "")[:10]
        if not ts:
            continue
        try:
            sig_date = date.fromisoformat(ts)
        except ValueError:
            continue
        if sig_date < start or sig_date >= end:
            continue

        if rsi < 30:
            signals.append(_make_signal(
                f"hist:ta_rsi:oversold:{symbol}:{ts}",
                "ta_rsi", symbol, "technical", "bullish",
                round(min(1.0, (30 - rsi) / 30), 2), ts,
                {"rsi_value": rsi, "condition": "oversold"},
            ))
        elif rsi > 70:
            signals.append(_make_signal(
                f"hist:ta_rsi:overbought:{symbol}:{ts}",
                "ta_rsi", symbol, "technical", "bearish",
                round(min(1.0, (rsi - 70) / 30), 2), ts,
                {"rsi_value": rsi, "condition": "overbought"},
            ))

    return signals


def compute_macd_series(
    symbol: str, history: list, date_idx: dict, start: date, end: date,
) -> list[dict]:
    """Detect MACD crossovers in the window."""
    closes = [p.price for p in history]
    if len(closes) < 35:
        return []

    fast, slow, sig_period = 12, 26, 9

    def ema(data: list[float], period: int) -> list[float]:
        result = [data[0]]
        mult = 2 / (period + 1)
        for val in data[1:]:
            result.append(val * mult + result[-1] * (1 - mult))
        return result

    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)
    macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
    signal_line = ema(macd_line[slow - 1:], sig_period)

    offset = slow - 1
    signals: list[dict] = []

    for i in range(1, len(signal_line)):
        abs_idx = offset + i
        if abs_idx >= len(history):
            continue

        ts = getattr(history[abs_idx], "timestamp", "")[:10]
        if not ts:
            continue
        try:
            sig_date = date.fromisoformat(ts)
        except ValueError:
            continue
        if sig_date < start or sig_date >= end:
            continue

        prev_macd = macd_line[offset + i - 1]
        curr_macd = macd_line[offset + i]
        prev_sig = signal_line[i - 1]
        curr_sig = signal_line[i]

        if prev_macd <= prev_sig and curr_macd > curr_sig:
            signals.append(_make_signal(
                f"hist:ta_macd:crossover_bull:{symbol}:{ts}",
                "ta_macd", symbol, "technical", "bullish",
                round(min(1.0, abs(curr_macd - curr_sig) * 10), 2), ts,
                {"crossover": "bullish", "macd": round(curr_macd, 4), "signal": round(curr_sig, 4)},
            ))
        elif prev_macd >= prev_sig and curr_macd < curr_sig:
            signals.append(_make_signal(
                f"hist:ta_macd:crossover_bear:{symbol}:{ts}",
                "ta_macd", symbol, "technical", "bearish",
                round(min(1.0, abs(curr_macd - curr_sig) * 10), 2), ts,
                {"crossover": "bearish", "macd": round(curr_macd, 4), "signal": round(curr_sig, 4)},
            ))

    return signals


def compute_bollinger_series(
    symbol: str, history: list, date_idx: dict, start: date, end: date,
) -> list[dict]:
    """Detect Bollinger Band squeezes and breakouts in the window."""
    closes = [p.price for p in history]
    period = 20
    if len(closes) < period + 1:
        return []

    signals: list[dict] = []

    for i in range(period, len(closes)):
        if i >= len(history):
            break
        ts = getattr(history[i], "timestamp", "")[:10]
        if not ts:
            continue
        try:
            sig_date = date.fromisoformat(ts)
        except ValueError:
            continue
        if sig_date < start or sig_date >= end:
            continue

        window = closes[i - period:i]
        sma = sum(window) / period
        variance = sum((x - sma) ** 2 for x in window) / period
        std = math.sqrt(variance) if variance > 0 else 0.001
        upper = sma + 2 * std
        lower = sma - 2 * std
        bandwidth = (upper - lower) / sma if sma > 0 else 0

        price = closes[i]

        if price > upper:
            signals.append(_make_signal(
                f"hist:ta_bollinger:upper_break:{symbol}:{ts}",
                "ta_bollinger", symbol, "technical", "bullish",
                round(min(1.0, (price - upper) / (upper - sma) if upper != sma else 0.5), 2), ts,
                {"condition": "upper_breakout", "bandwidth": round(bandwidth, 4)},
            ))
        elif price < lower:
            signals.append(_make_signal(
                f"hist:ta_bollinger:lower_break:{symbol}:{ts}",
                "ta_bollinger", symbol, "technical", "bearish",
                round(min(1.0, (lower - price) / (sma - lower) if sma != lower else 0.5), 2), ts,
                {"condition": "lower_breakout", "bandwidth": round(bandwidth, 4)},
            ))
        elif bandwidth < 0.04 and i > period + 10:
            prev_window = closes[i - period - 1:i - 1]
            prev_sma = sum(prev_window) / period
            prev_var = sum((x - prev_sma) ** 2 for x in prev_window) / period
            prev_std = math.sqrt(prev_var) if prev_var > 0 else 0.001
            prev_bw = (2 * prev_std * 2) / prev_sma if prev_sma > 0 else 0
            if prev_bw >= 0.04:
                signals.append(_make_signal(
                    f"hist:ta_bollinger:squeeze:{symbol}:{ts}",
                    "ta_bollinger", symbol, "technical", "neutral",
                    0.5, ts,
                    {"condition": "squeeze_entry", "bandwidth": round(bandwidth, 4)},
                ))

    return signals


def compute_volume_signals(
    symbol: str, history: list, date_idx: dict, start: date, end: date,
) -> list[dict]:
    """Detect unusual volume (> 2x 20-day average) in the window."""
    if len(history) < 21:
        return []

    signals: list[dict] = []
    for i in range(20, len(history)):
        ts = getattr(history[i], "timestamp", "")[:10]
        if not ts:
            continue
        try:
            sig_date = date.fromisoformat(ts)
        except ValueError:
            continue
        if sig_date < start or sig_date >= end:
            continue

        vol = getattr(history[i], "volume", 0) or 0
        avg_vol = sum(getattr(history[j], "volume", 0) or 0 for j in range(i - 20, i)) / 20
        if avg_vol <= 0 or vol <= 0:
            continue

        ratio = vol / avg_vol
        if ratio >= 2.0:
            prev_price = history[i - 1].price
            curr_price = history[i].price
            direction = "bullish" if curr_price > prev_price else "bearish"
            signals.append(_make_signal(
                f"hist:volume_anomaly:{symbol}:{ts}",
                "finviz_unusual_volume", symbol, "technical", direction,
                round(min(1.0, (ratio - 2) / 3), 2), ts,
                {"volume_ratio": round(ratio, 2), "volume": vol, "avg_volume": round(avg_vol, 0)},
            ))

    return signals
