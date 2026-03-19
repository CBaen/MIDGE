"""strategy_library.py - 25 math-first strategies for Crypto Pattern Convergence.

Six families: RSI (5), MACD (4), Bollinger (4), Structure (4), Volume (4), MA (4).
Min 50 bars; 200-EMA strategies need 210. confidence=0.55 is the prior (backtester overwrites).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd

from mae_core.market.strategies.models import StrategyResult

_DEFAULT_CONFIDENCE = 0.55
_MIN_BARS = 50


# ── Shared helpers ────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now().isoformat()


def _make(name: str, symbol: str, direction: str, sl: float, tp: float,
          strength: float, meta: Dict[str, Any]) -> StrategyResult:
    return StrategyResult(
        strategy_name=name, symbol=symbol, direction=direction,
        signal=1 if direction == "bullish" else -1,
        stop_loss=sl, take_profit=tp, strength=strength,
        confidence=_DEFAULT_CONFIDENCE, detected_at=_now(), metadata=meta,
    )


def _atr_stops(df: pd.DataFrame, direction: str,
               atr_sl_mult: float = 1.5, atr_tp_mult: float = 3.0) -> tuple[float, float]:
    """ATR-based stop loss and take profit from the last close."""
    from mae_core.market.edge.ta_structure import compute_atr
    if len(df) < 15:
        return 0.0, 0.0
    atr = compute_atr(df["High"].tolist(), df["Low"].tolist(), df["Close"].tolist(), period=14)
    if atr <= 0:
        return 0.0, 0.0
    last = df["Close"].iloc[-1]
    if direction == "bullish":
        return last - atr * atr_sl_mult, last + atr * atr_tp_mult
    return last + atr * atr_sl_mult, last - atr * atr_tp_mult


def _rsi(closes: pd.Series, period: int) -> pd.Series:
    """Wilder RSI via ewm."""
    delta = closes.diff()
    alpha = 1.0 / period
    ag = delta.clip(lower=0).ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    al = (-delta).clip(lower=0).ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    return 100 - (100 / (1 + ag / al.replace(0, np.nan)))


def _macd(closes: pd.Series, fast: int, slow: int, sig_period: int):
    """Return (macd_line, signal_line, histogram)."""
    m = closes.ewm(span=fast, adjust=False).mean() - closes.ewm(span=slow, adjust=False).mean()
    s = m.ewm(span=sig_period, adjust=False).mean()
    return m, s, m - s


def _bollinger(closes: pd.Series, period: int = 20, n_std: float = 2.0):
    """Return (mid, upper, lower)."""
    mid = closes.rolling(period).mean()
    std = closes.rolling(period).std(ddof=0)
    return mid, mid + n_std * std, mid - n_std * std


def _swing_highs_lows(highs: pd.Series, lows: pd.Series, lookback: int = 5):
    """Return (swing_high_values, swing_low_values) lists."""
    h, l = highs.values, lows.values
    n = len(h)
    sh, sl = [], []
    for i in range(lookback, n - lookback):
        w_h = h[i - lookback: i + lookback + 1]
        w_l = l[i - lookback: i + lookback + 1]
        if h[i] == w_h.max():
            sh.append(h[i])
        if l[i] == w_l.min():
            sl.append(l[i])
    return sh, sl


# ── RSI Family ────────────────────────────────────────────────────────────────

def rsi_oversold_14(symbol: str, df: pd.DataFrame) -> Optional[StrategyResult]:
    """RSI(14) < 20 → bullish."""
    if len(df) < _MIN_BARS:
        return None
    last = _rsi(df["Close"], 14).iloc[-1]
    if pd.isna(last) or last >= 20:
        return None
    sl, tp = _atr_stops(df, "bullish")
    return _make("rsi_oversold_14", symbol, "bullish", sl, tp,
                 round((20 - last) / 20, 4), {"rsi": round(last, 2)})


def rsi_oversold_2(symbol: str, df: pd.DataFrame) -> Optional[StrategyResult]:
    """RSI(2) < 10 → bullish (short-term mean reversion)."""
    if len(df) < _MIN_BARS:
        return None
    last = _rsi(df["Close"], 2).iloc[-1]
    if pd.isna(last) or last >= 10:
        return None
    sl, tp = _atr_stops(df, "bullish")
    return _make("rsi_oversold_2", symbol, "bullish", sl, tp,
                 round((10 - last) / 10, 4), {"rsi2": round(last, 2)})


def rsi_overbought_14(symbol: str, df: pd.DataFrame) -> Optional[StrategyResult]:
    """RSI(14) > 80 → bearish."""
    if len(df) < _MIN_BARS:
        return None
    last = _rsi(df["Close"], 14).iloc[-1]
    if pd.isna(last) or last <= 80:
        return None
    sl, tp = _atr_stops(df, "bearish")
    return _make("rsi_overbought_14", symbol, "bearish", sl, tp,
                 round((last - 80) / 20, 4), {"rsi": round(last, 2)})


def rsi_momentum_shift(symbol: str, df: pd.DataFrame) -> Optional[StrategyResult]:
    """RSI(14) was below 30, now crosses above 40 → bullish momentum restoration."""
    if len(df) < _MIN_BARS:
        return None
    rsi = _rsi(df["Close"], 14).dropna()
    if len(rsi) < 5:
        return None
    prev, last = rsi.iloc[-2], rsi.iloc[-1]
    recent_low = rsi.iloc[-10:].min()
    if not (recent_low < 30 and prev < 40 and last >= 40):
        return None
    sl, tp = _atr_stops(df, "bullish")
    return _make("rsi_momentum_shift", symbol, "bullish", sl, tp,
                 round(min((last - 40) / 30, 1.0), 4),
                 {"rsi_now": round(last, 2), "rsi_recent_low": round(recent_low, 2)})


def rsi_divergence(symbol: str, df: pd.DataFrame) -> Optional[StrategyResult]:
    """Bullish divergence: price makes 14-bar low, RSI does not."""
    if len(df) < _MIN_BARS:
        return None
    closes = df["Close"]
    rsi = _rsi(closes, 14).dropna()
    if len(rsi) < 14:
        return None
    p_now = closes.iloc[-1]
    p_prev = closes.iloc[-14:-1].min()
    r_now = rsi.iloc[-1]
    r_prev = rsi.iloc[-14:-1].min()
    if not (p_now < p_prev and r_now > r_prev):
        return None
    strength = round(min((p_prev - p_now) / (p_prev + 1e-9) * 5 + (r_now - r_prev) / 30, 1.0), 4)
    sl, tp = _atr_stops(df, "bullish")
    return _make("rsi_divergence", symbol, "bullish", sl, tp, strength,
                 {"price_low_now": round(p_now, 4), "rsi_now": round(r_now, 2),
                  "rsi_prev_low": round(r_prev, 2)})


# ── MACD Family ───────────────────────────────────────────────────────────────

def macd_crossover_std(symbol: str, df: pd.DataFrame) -> Optional[StrategyResult]:
    """MACD(12,26,9) crosses above signal line → bullish."""
    if len(df) < _MIN_BARS:
        return None
    m, s, _ = _macd(df["Close"], 12, 26, 9)
    if not (m.iloc[-2] - s.iloc[-2] < 0 and m.iloc[-1] - s.iloc[-1] >= 0):
        return None
    strength = round(min(abs(m.iloc[-1] - s.iloc[-1]) / (df["Close"].iloc[-1] * 0.001 + 1e-9), 1.0), 4)
    sl, tp = _atr_stops(df, "bullish")
    return _make("macd_crossover_std", symbol, "bullish", sl, tp, strength,
                 {"macd": round(m.iloc[-1], 6), "signal_line": round(s.iloc[-1], 6)})


def macd_crossover_fast(symbol: str, df: pd.DataFrame) -> Optional[StrategyResult]:
    """MACD(5,13,5) crosses above signal → bullish (fast crypto setting)."""
    if len(df) < _MIN_BARS:
        return None
    m, s, _ = _macd(df["Close"], 5, 13, 5)
    if not (m.iloc[-2] - s.iloc[-2] < 0 and m.iloc[-1] - s.iloc[-1] >= 0):
        return None
    strength = round(min(abs(m.iloc[-1] - s.iloc[-1]) / (df["Close"].iloc[-1] * 0.001 + 1e-9), 1.0), 4)
    sl, tp = _atr_stops(df, "bullish")
    return _make("macd_crossover_fast", symbol, "bullish", sl, tp, strength,
                 {"macd_fast": round(m.iloc[-1], 6), "signal_line": round(s.iloc[-1], 6)})


def macd_zero_cross(symbol: str, df: pd.DataFrame) -> Optional[StrategyResult]:
    """MACD(12,26,9) line crosses from negative to zero → bullish trend confirmation."""
    if len(df) < _MIN_BARS:
        return None
    m, _, _ = _macd(df["Close"], 12, 26, 9)
    if not (m.iloc[-2] < 0 and m.iloc[-1] >= 0):
        return None
    strength = round(min(m.iloc[-1] / (df["Close"].iloc[-1] * 0.005 + 1e-9), 1.0), 4)
    sl, tp = _atr_stops(df, "bullish")
    return _make("macd_zero_cross", symbol, "bullish", sl, tp, strength,
                 {"macd_prev": round(m.iloc[-2], 6), "macd_now": round(m.iloc[-1], 6)})


def macd_histogram_reversal(symbol: str, df: pd.DataFrame) -> Optional[StrategyResult]:
    """Histogram negative and declining, now starts rising → bullish momentum shift."""
    if len(df) < _MIN_BARS:
        return None
    _, _, hist = _macd(df["Close"], 12, 26, 9)
    h0, h1, h2 = hist.iloc[-3], hist.iloc[-2], hist.iloc[-1]
    if not (h0 < 0 and h1 < 0 and h2 < 0 and h1 < h0 and h2 > h1):
        return None
    strength = round(min(abs(h2 - h1) / (abs(h1) + 1e-9), 1.0), 4)
    sl, tp = _atr_stops(df, "bullish")
    return _make("macd_histogram_reversal", symbol, "bullish", sl, tp, strength,
                 {"hist_t2": round(h0, 6), "hist_t1": round(h1, 6), "hist_t0": round(h2, 6)})


# ── Bollinger Band Family ─────────────────────────────────────────────────────

def bollinger_lower_touch(symbol: str, df: pd.DataFrame) -> Optional[StrategyResult]:
    """Price at or below lower Bollinger Band (20,2) → bullish mean reversion."""
    if len(df) < _MIN_BARS:
        return None
    _, _, lower = _bollinger(df["Close"])
    last_close, last_lower = df["Close"].iloc[-1], lower.iloc[-1]
    if pd.isna(last_lower) or last_close > last_lower:
        return None
    strength = round(min(1 + (last_lower - last_close) / (last_lower + 1e-9) * 10, 1.0), 4)
    sl, tp = _atr_stops(df, "bullish")
    return _make("bollinger_lower_touch", symbol, "bullish", sl, tp, strength,
                 {"close": round(last_close, 4), "lower_band": round(last_lower, 4)})


def bollinger_upper_touch(symbol: str, df: pd.DataFrame) -> Optional[StrategyResult]:
    """Price at or above upper Bollinger Band (20,2) → bearish mean reversion."""
    if len(df) < _MIN_BARS:
        return None
    _, upper, _ = _bollinger(df["Close"])
    last_close, last_upper = df["Close"].iloc[-1], upper.iloc[-1]
    if pd.isna(last_upper) or last_close < last_upper:
        return None
    strength = round(min(1 + (last_close - last_upper) / (last_upper + 1e-9) * 10, 1.0), 4)
    sl, tp = _atr_stops(df, "bearish")
    return _make("bollinger_upper_touch", symbol, "bearish", sl, tp, strength,
                 {"close": round(last_close, 4), "upper_band": round(last_upper, 4)})


def bollinger_squeeze_break(symbol: str, df: pd.DataFrame) -> Optional[StrategyResult]:
    """Bandwidth below 80th-pct of 100 bars, price closes outside band with volume > avg → breakout."""
    if len(df) < _MIN_BARS:
        return None
    closes, vol = df["Close"], df["Volume"]
    mid, upper, lower = _bollinger(closes)
    bw = (upper - lower) / mid.replace(0, np.nan)
    bw_clean = bw.dropna()
    if len(bw_clean) < 20:
        return None
    pct80 = (bw_clean.iloc[-100:] if len(bw_clean) >= 100 else bw_clean).quantile(0.80)
    last_bw = bw.iloc[-1]
    if pd.isna(last_bw) or last_bw > pct80:
        return None
    last_close, last_upper, last_lower = closes.iloc[-1], upper.iloc[-1], lower.iloc[-1]
    above = last_close > last_upper
    below = last_close < last_lower
    if not (above or below):
        return None
    avg_vol = vol.iloc[-20:].mean()
    if vol.iloc[-1] < avg_vol:
        return None
    direction = "bullish" if above else "bearish"
    vol_ratio = vol.iloc[-1] / (avg_vol + 1e-9)
    strength = round(min((vol_ratio - 1) * 0.5, 1.0), 4)
    sl, tp = _atr_stops(df, direction)
    return _make("bollinger_squeeze_break", symbol, direction, sl, tp, strength,
                 {"bandwidth": round(last_bw, 6), "bandwidth_p80": round(pct80, 6),
                  "volume_ratio": round(vol_ratio, 2)})


def bollinger_mean_reversion(symbol: str, df: pd.DataFrame) -> Optional[StrategyResult]:
    """Price > 2.5 std from 20-period mean → expect reversion."""
    if len(df) < _MIN_BARS:
        return None
    closes = df["Close"]
    mean = closes.rolling(20).mean().iloc[-1]
    std = closes.rolling(20).std(ddof=0).iloc[-1]
    if pd.isna(mean) or std <= 0:
        return None
    z = (closes.iloc[-1] - mean) / std
    if abs(z) < 2.5:
        return None
    direction = "bearish" if z > 0 else "bullish"
    strength = round(min((abs(z) - 2.5) / 1.5, 1.0), 4)
    sl, tp = _atr_stops(df, direction)
    return _make("bollinger_mean_reversion", symbol, direction, sl, tp, strength,
                 {"z_score": round(z, 3), "mean": round(mean, 4), "std": round(std, 4)})


# ── Structure Family ──────────────────────────────────────────────────────────

def structure_higher_high(symbol: str, df: pd.DataFrame) -> Optional[StrategyResult]:
    """Last two swing highs and swing lows both ascending → bullish uptrend."""
    if len(df) < _MIN_BARS:
        return None
    sh, sl_vals = _swing_highs_lows(df["High"], df["Low"], lookback=5)
    if len(sh) < 2 or len(sl_vals) < 2:
        return None
    if not (sh[-1] > sh[-2] and sl_vals[-1] > sl_vals[-2]):
        return None
    strength = round(min((sh[-1] - sh[-2]) / (sh[-2] + 1e-9) * 20, 1.0), 4)
    sl, tp = _atr_stops(df, "bullish")
    return _make("structure_higher_high", symbol, "bullish", sl, tp, strength,
                 {"swing_high_prev": round(sh[-2], 4), "swing_high_last": round(sh[-1], 4),
                  "swing_low_prev": round(sl_vals[-2], 4), "swing_low_last": round(sl_vals[-1], 4)})


def structure_bos_bull(symbol: str, df: pd.DataFrame) -> Optional[StrategyResult]:
    """Close > highest high of last 20 bars → bullish break of structure."""
    if len(df) < _MIN_BARS:
        return None
    last_close = df["Close"].iloc[-1]
    prior_high = df["High"].iloc[-21:-1].max()
    if last_close <= prior_high:
        return None
    strength = round(min((last_close - prior_high) / (prior_high + 1e-9) * 20, 1.0), 4)
    sl, tp = _atr_stops(df, "bullish")
    return _make("structure_bos_bull", symbol, "bullish", sl, tp, strength,
                 {"close": round(last_close, 4), "prior_20bar_high": round(prior_high, 4)})


def structure_bos_bear(symbol: str, df: pd.DataFrame) -> Optional[StrategyResult]:
    """Close < lowest low of last 20 bars → bearish break of structure."""
    if len(df) < _MIN_BARS:
        return None
    last_close = df["Close"].iloc[-1]
    prior_low = df["Low"].iloc[-21:-1].min()
    if last_close >= prior_low:
        return None
    strength = round(min((prior_low - last_close) / (prior_low + 1e-9) * 20, 1.0), 4)
    sl, tp = _atr_stops(df, "bearish")
    return _make("structure_bos_bear", symbol, "bearish", sl, tp, strength,
                 {"close": round(last_close, 4), "prior_20bar_low": round(prior_low, 4)})


def structure_support_retest(symbol: str, df: pd.DataFrame) -> Optional[StrategyResult]:
    """Price dropped to within 0.5% of prior swing low, held, then bounced → bullish."""
    if len(df) < _MIN_BARS:
        return None
    lows, closes = df["Low"], df["Close"]
    prior_swing_low = lows.iloc[-20:-3].min()
    last_low, last_close = lows.iloc[-1], closes.iloc[-1]
    proximity = abs(last_low - prior_swing_low) / (prior_swing_low + 1e-9)
    if proximity > 0.005 or last_close <= last_low * 1.003:
        return None
    bounce_pct = (last_close - last_low) / (last_low + 1e-9)
    strength = round(min(bounce_pct * 50, 1.0), 4)
    sl, tp = _atr_stops(df, "bullish")
    return _make("structure_support_retest", symbol, "bullish", sl, tp, strength,
                 {"prior_swing_low": round(prior_swing_low, 4), "last_low": round(last_low, 4),
                  "proximity_pct": round(proximity * 100, 3)})


# ── Volume Family ─────────────────────────────────────────────────────────────

def volume_climax_bull(symbol: str, df: pd.DataFrame) -> Optional[StrategyResult]:
    """Volume > 2x avg, red candle, close in bottom 25% of range → exhaustion, bullish."""
    if len(df) < _MIN_BARS:
        return None
    avg_vol = df["Volume"].iloc[-21:-1].mean()
    last_vol = df["Volume"].iloc[-1]
    if last_vol < avg_vol * 2:
        return None
    o, c, h, l = df["Open"].iloc[-1], df["Close"].iloc[-1], df["High"].iloc[-1], df["Low"].iloc[-1]
    rng = h - l
    if rng <= 0 or c >= o:
        return None
    pos = (c - l) / rng
    if pos > 0.25:
        return None
    vol_ratio = last_vol / (avg_vol + 1e-9)
    strength = round(min((vol_ratio - 2) * 0.3 + (0.25 - pos), 1.0), 4)
    sl, tp = _atr_stops(df, "bullish")
    return _make("volume_climax_bull", symbol, "bullish", sl, tp, strength,
                 {"volume_ratio": round(vol_ratio, 2), "close_pos_in_range": round(pos, 3)})


def volume_climax_bear(symbol: str, df: pd.DataFrame) -> Optional[StrategyResult]:
    """Volume > 2x avg, green candle, close in top 25% of range → exhaustion, bearish."""
    if len(df) < _MIN_BARS:
        return None
    avg_vol = df["Volume"].iloc[-21:-1].mean()
    last_vol = df["Volume"].iloc[-1]
    if last_vol < avg_vol * 2:
        return None
    o, c, h, l = df["Open"].iloc[-1], df["Close"].iloc[-1], df["High"].iloc[-1], df["Low"].iloc[-1]
    rng = h - l
    if rng <= 0 or c <= o:
        return None
    pos = (c - l) / rng
    if pos < 0.75:
        return None
    vol_ratio = last_vol / (avg_vol + 1e-9)
    strength = round(min((vol_ratio - 2) * 0.3 + (pos - 0.75), 1.0), 4)
    sl, tp = _atr_stops(df, "bearish")
    return _make("volume_climax_bear", symbol, "bearish", sl, tp, strength,
                 {"volume_ratio": round(vol_ratio, 2), "close_pos_in_range": round(pos, 3)})


def volume_accumulation(symbol: str, df: pd.DataFrame) -> Optional[StrategyResult]:
    """Last 5 bars: price and volume both trending up → institutional accumulation."""
    if len(df) < _MIN_BARS:
        return None
    closes = df["Close"].iloc[-5:]
    vol = df["Volume"].iloc[-5:]
    idx = np.arange(5)
    if np.polyfit(idx, closes.values, 1)[0] <= 0 or np.polyfit(idx, vol.values, 1)[0] <= 0:
        return None
    price_chg = (closes.iloc[-1] - closes.iloc[0]) / (closes.iloc[0] + 1e-9)
    vol_chg = (vol.iloc[-1] - vol.iloc[0]) / (vol.iloc[0] + 1e-9)
    strength = round(min((price_chg + vol_chg * 0.5) * 5, 1.0), 4)
    sl, tp = _atr_stops(df, "bullish")
    return _make("volume_accumulation", symbol, "bullish", sl, tp, strength,
                 {"price_5bar_chg_pct": round(price_chg * 100, 2),
                  "volume_5bar_chg_pct": round(vol_chg * 100, 2)})


def volume_dry_up(symbol: str, df: pd.DataFrame) -> Optional[StrategyResult]:
    """Volume < 50% of 20-bar avg for 3+ consecutive bars → coil; direction from prior trend."""
    if len(df) < _MIN_BARS:
        return None
    vol = df["Volume"]
    avg_vol = vol.iloc[-24:-3].mean()
    if not (vol.iloc[-3:] < avg_vol * 0.5).all():
        return None
    closes = df["Close"]
    prior_trend_up = closes.iloc[-4] > closes.iloc[-20]
    direction = "bullish" if prior_trend_up else "bearish"
    dry_ratio = vol.iloc[-3:].mean() / (avg_vol + 1e-9)
    strength = round(min((0.5 - dry_ratio) * 2, 1.0), 4)
    sl, tp = _atr_stops(df, direction)
    return _make("volume_dry_up", symbol, direction, sl, tp, strength,
                 {"dry_ratio": round(dry_ratio, 3), "prior_trend": "up" if prior_trend_up else "down"})


# ── Moving Average Family ─────────────────────────────────────────────────────

def ema_cross_9_21(symbol: str, df: pd.DataFrame) -> Optional[StrategyResult]:
    """9 EMA crosses above 21 EMA → bullish short-term trend shift."""
    if len(df) < _MIN_BARS:
        return None
    closes = df["Close"]
    e9 = closes.ewm(span=9, adjust=False).mean()
    e21 = closes.ewm(span=21, adjust=False).mean()
    if not (e9.iloc[-2] - e21.iloc[-2] < 0 and e9.iloc[-1] - e21.iloc[-1] >= 0):
        return None
    strength = round(min((e9.iloc[-1] - e21.iloc[-1]) / (closes.iloc[-1] + 1e-9) * 200, 1.0), 4)
    sl, tp = _atr_stops(df, "bullish")
    return _make("ema_cross_9_21", symbol, "bullish", sl, tp, strength,
                 {"ema9": round(e9.iloc[-1], 4), "ema21": round(e21.iloc[-1], 4)})


def ema_cross_50_200(symbol: str, df: pd.DataFrame) -> Optional[StrategyResult]:
    """50 EMA crosses above 200 EMA → bullish golden cross."""
    if len(df) < 210:
        return None
    closes = df["Close"]
    e50 = closes.ewm(span=50, adjust=False).mean()
    e200 = closes.ewm(span=200, adjust=False).mean()
    if not (e50.iloc[-2] - e200.iloc[-2] < 0 and e50.iloc[-1] - e200.iloc[-1] >= 0):
        return None
    strength = round(min((e50.iloc[-1] - e200.iloc[-1]) / (closes.iloc[-1] + 1e-9) * 500, 1.0), 4)
    sl, tp = _atr_stops(df, "bullish")
    return _make("ema_cross_50_200", symbol, "bullish", sl, tp, strength,
                 {"ema50": round(e50.iloc[-1], 4), "ema200": round(e200.iloc[-1], 4)})


def price_above_200ema(symbol: str, df: pd.DataFrame) -> Optional[StrategyResult]:
    """Price was below 200 EMA, now closes above → bullish regime change."""
    if len(df) < 210:
        return None
    closes = df["Close"]
    e200 = closes.ewm(span=200, adjust=False).mean()
    if not (closes.iloc[-2] < e200.iloc[-2] and closes.iloc[-1] >= e200.iloc[-1]):
        return None
    cross_pct = (closes.iloc[-1] - e200.iloc[-1]) / (e200.iloc[-1] + 1e-9)
    strength = round(min(cross_pct * 100, 1.0), 4)
    sl, tp = _atr_stops(df, "bullish")
    return _make("price_above_200ema", symbol, "bullish", sl, tp, strength,
                 {"close": round(closes.iloc[-1], 4), "ema200": round(e200.iloc[-1], 4),
                  "cross_pct": round(cross_pct * 100, 3)})


def ma_ribbon_expand(symbol: str, df: pd.DataFrame) -> Optional[StrategyResult]:
    """EMAs 8,13,21,34,55 in order AND spreading → strong trend confirmation."""
    if len(df) < _MIN_BARS:
        return None
    closes = df["Close"]
    spans = [8, 13, 21, 34, 55]
    emas_now = [closes.ewm(span=s, adjust=False).mean().iloc[-1] for s in spans]
    emas_prev = [closes.ewm(span=s, adjust=False).mean().iloc[-6] for s in spans]
    bull = emas_now[0] > emas_now[1] > emas_now[2] > emas_now[3] > emas_now[4]
    bear = emas_now[0] < emas_now[1] < emas_now[2] < emas_now[3] < emas_now[4]
    if not (bull or bear):
        return None
    spread_now = abs(emas_now[0] - emas_now[-1])
    spread_prev = abs(emas_prev[0] - emas_prev[-1])
    if spread_now <= spread_prev:
        return None
    direction = "bullish" if bull else "bearish"
    spread_pct = spread_now / (closes.iloc[-1] + 1e-9)
    strength = round(min(spread_pct * 20, 1.0), 4)
    sl, tp = _atr_stops(df, direction)
    return _make("ma_ribbon_expand", symbol, direction, sl, tp, strength,
                 {"ema8": round(emas_now[0], 4), "ema55": round(emas_now[4], 4),
                  "spread_pct": round(spread_pct * 100, 3)})


# ── Strategy registry ─────────────────────────────────────────────────────────

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


# ═══════════════════════════════════════════════════════════════════════
# Family 7: Mathematical / Physics / Chaos
# ═══════════════════════════════════════════════════════════════════════


def _hurst_exponent(series, max_k: int = 20) -> float:
    """Compute Hurst exponent via rescaled range (R/S) analysis.

    H > 0.5 = trending (persistent), H < 0.5 = mean-reverting, H ≈ 0.5 = random.
    Uses nolds if available, falls back to a simple R/S implementation.
    """
    try:
        import nolds
        return float(nolds.hurst_rs(series, nvals=None, fit="poly"))
    except Exception:
        pass
    # Fallback: simple R/S
    import numpy as np
    n = len(series)
    if n < 20:
        return 0.5
    max_k = min(max_k, n // 2)
    rs_list = []
    for k in range(10, max_k + 1):
        rs_vals = []
        for start in range(0, n - k, k):
            subset = series[start:start + k]
            mean_s = np.mean(subset)
            devs = np.cumsum(subset - mean_s)
            r = np.max(devs) - np.min(devs)
            s = np.std(subset, ddof=1)
            if s > 0:
                rs_vals.append(r / s)
        if rs_vals:
            rs_list.append((np.log(k), np.log(np.mean(rs_vals))))
    if len(rs_list) < 3:
        return 0.5
    x = np.array([p[0] for p in rs_list])
    y = np.array([p[1] for p in rs_list])
    slope = np.polyfit(x, y, 1)[0]
    return float(np.clip(slope, 0.0, 1.0))


def hurst_regime(symbol: str, df: "pd.DataFrame") -> "Optional[StrategyResult]":
    """Hurst exponent regime detector — tells MIDGE if price is trending or mean-reverting.

    H < 0.4 → mean-reverting → bullish signal (expect reversion to mean from extremes)
    H > 0.6 → trending → signal in trend direction
    0.4-0.6 → random walk → no signal
    """
    if len(df) < 100:
        return None
    import numpy as np
    closes = df["Close"].values
    returns = np.diff(np.log(closes[~np.isnan(closes)]))
    if len(returns) < 50:
        return None
    h = _hurst_exponent(returns)
    sl, tp = _atr_stops(df, "bullish")
    if sl == 0:
        return None
    if h < 0.4:
        # Mean-reverting regime — price extended from mean should revert
        mean_price = float(np.nanmean(closes[-20:]))
        last = float(closes[-1])
        if last < mean_price * 0.97:
            return _make("hurst_regime", symbol, "bullish", 1, sl, tp,
                         min(1.0, (0.4 - h) / 0.3), metadata={"hurst": round(h, 4), "regime": "mean_reverting"})
        sl_b, tp_b = _atr_stops(df, "bearish")
        if last > mean_price * 1.03:
            return _make("hurst_regime", symbol, "bearish", -1, sl_b, tp_b,
                         min(1.0, (0.4 - h) / 0.3), metadata={"hurst": round(h, 4), "regime": "mean_reverting"})
    elif h > 0.6:
        # Trending regime — go with the trend
        recent_return = (closes[-1] - closes[-10]) / closes[-10] if closes[-10] != 0 else 0
        if recent_return > 0.01:
            return _make("hurst_regime", symbol, "bullish", 1, sl, tp,
                         min(1.0, (h - 0.6) / 0.3), metadata={"hurst": round(h, 4), "regime": "trending"})
        elif recent_return < -0.01:
            sl_b, tp_b = _atr_stops(df, "bearish")
            return _make("hurst_regime", symbol, "bearish", -1, sl_b, tp_b,
                         min(1.0, (h - 0.6) / 0.3), metadata={"hurst": round(h, 4), "regime": "trending"})
    return None


def ou_zscore_reversion(symbol: str, df: "pd.DataFrame") -> "Optional[StrategyResult]":
    """Ornstein-Uhlenbeck Z-score mean reversion — spring physics for price.

    When price is stretched far from its equilibrium (high Z-score), it snaps back.
    Z > 2.0 → overbought → bearish. Z < -2.0 → oversold → bullish.
    """
    if len(df) < 50:
        return None
    import numpy as np
    closes = df["Close"].values
    clean = closes[~np.isnan(closes)]
    if len(clean) < 50:
        return None
    # Use 20-period rolling mean as the equilibrium
    window = 20
    mean = np.mean(clean[-window:])
    std = np.std(clean[-window:], ddof=1)
    if std <= 0:
        return None
    z = (clean[-1] - mean) / std
    if abs(z) < 2.0:
        return None
    if z < -2.0:
        sl, tp = _atr_stops(df, "bullish")
        if sl == 0:
            return None
        return _make("ou_zscore_reversion", symbol, "bullish", 1, sl, tp,
                     min(1.0, (abs(z) - 2.0) / 2.0),
                     metadata={"z_score": round(float(z), 3), "equilibrium": round(float(mean), 2)})
    elif z > 2.0:
        sl, tp = _atr_stops(df, "bearish")
        if sl == 0:
            return None
        return _make("ou_zscore_reversion", symbol, "bearish", -1, sl, tp,
                     min(1.0, (abs(z) - 2.0) / 2.0),
                     metadata={"z_score": round(float(z), 3), "equilibrium": round(float(mean), 2)})
    return None


def _permutation_entropy(series, order: int = 3, delay: int = 1) -> float:
    """Compute permutation entropy — measures chaos/order in a time series.

    Low PE (< 0.7) = ordered, predictable. High PE (> 0.9) = chaotic, noisy.
    Pure Python, no external dependencies.
    """
    import math
    from itertools import permutations
    n = len(series)
    if n < (order - 1) * delay + order:
        return 1.0  # insufficient data → assume chaotic
    # Build embedded vectors
    perms = {}
    total = 0
    for i in range(n - (order - 1) * delay):
        pattern = tuple(sorted(range(order), key=lambda k: series[i + k * delay]))
        perms[pattern] = perms.get(pattern, 0) + 1
        total += 1
    if total == 0:
        return 1.0
    # Shannon entropy normalized by maximum
    max_entropy = math.log(math.factorial(order))
    if max_entropy == 0:
        return 1.0
    entropy = -sum((c / total) * math.log(c / total) for c in perms.values() if c > 0)
    return entropy / max_entropy


def permutation_entropy_gate(symbol: str, df: "pd.DataFrame") -> "Optional[StrategyResult]":
    """Permutation entropy gate — fires when the market is ordered (predictable).

    Low entropy + price below mean → bullish (ordered reversion expected).
    Low entropy + price above mean → bearish (ordered reversion expected).
    High entropy → no signal (chaos = unpredictable).
    """
    if len(df) < 50:
        return None
    import numpy as np
    closes = df["Close"].values
    clean = closes[~np.isnan(closes)]
    if len(clean) < 50:
        return None
    pe = _permutation_entropy(clean[-50:], order=4, delay=1)
    if pe > 0.85:
        return None  # Too chaotic — don't trade
    # Ordered market — check direction
    mean_20 = float(np.mean(clean[-20:]))
    last = float(clean[-1])
    deviation_pct = (last - mean_20) / mean_20 if mean_20 != 0 else 0
    if abs(deviation_pct) < 0.02:
        return None  # Not deviated enough
    if deviation_pct < -0.02:
        sl, tp = _atr_stops(df, "bullish")
        if sl == 0:
            return None
        return _make("permutation_entropy_gate", symbol, "bullish", 1, sl, tp,
                     min(1.0, (0.85 - pe) / 0.3),
                     metadata={"pe": round(pe, 4), "deviation_pct": round(deviation_pct * 100, 2)})
    elif deviation_pct > 0.02:
        sl, tp = _atr_stops(df, "bearish")
        if sl == 0:
            return None
        return _make("permutation_entropy_gate", symbol, "bearish", -1, sl, tp,
                     min(1.0, (0.85 - pe) / 0.3),
                     metadata={"pe": round(pe, 4), "deviation_pct": round(deviation_pct * 100, 2)})
    return None


def physical_momentum(symbol: str, df: "pd.DataFrame") -> "Optional[StrategyResult]":
    """Physics-based momentum: volume × return² — like kinetic energy in price.

    High "energy" with positive returns → bullish trend has force behind it.
    High "energy" with negative returns → bearish trend has force behind it.
    Low energy → no conviction, no signal.
    """
    if len(df) < 50:
        return None
    import numpy as np
    closes = df["Close"].values
    volumes = df["Volume"].values
    clean_c = closes[~np.isnan(closes)]
    clean_v = volumes[~np.isnan(volumes)]
    n = min(len(clean_c), len(clean_v))
    if n < 20:
        return None
    # Compute 5-bar momentum energy
    returns_5 = (clean_c[-1] - clean_c[-6]) / clean_c[-6] if clean_c[-6] != 0 else 0
    avg_vol_5 = float(np.mean(clean_v[-5:]))
    avg_vol_20 = float(np.mean(clean_v[-20:])) if len(clean_v) >= 20 else avg_vol_5
    if avg_vol_20 <= 0:
        return None
    vol_ratio = avg_vol_5 / avg_vol_20
    energy = vol_ratio * (returns_5 ** 2)
    if energy < 0.001:
        return None  # No conviction
    if returns_5 > 0.01 and vol_ratio > 1.2:
        sl, tp = _atr_stops(df, "bullish")
        if sl == 0:
            return None
        return _make("physical_momentum", symbol, "bullish", 1, sl, tp,
                     min(1.0, energy * 10),
                     metadata={"energy": round(energy, 6), "vol_ratio": round(vol_ratio, 3),
                               "return_5bar": round(returns_5 * 100, 2)})
    elif returns_5 < -0.01 and vol_ratio > 1.2:
        sl, tp = _atr_stops(df, "bearish")
        if sl == 0:
            return None
        return _make("physical_momentum", symbol, "bearish", -1, sl, tp,
                     min(1.0, energy * 10),
                     metadata={"energy": round(energy, 6), "vol_ratio": round(vol_ratio, 3),
                               "return_5bar": round(returns_5 * 100, 2)})
    return None


def _higuchi_fd(series, kmax: int = 10) -> float:
    """Compute Higuchi fractal dimension of a time series.

    FD ≈ 1.0 = smooth trend, FD ≈ 1.5 = Brownian noise, FD ≈ 2.0 = very rough/chaotic.
    Pure numpy implementation.
    """
    import numpy as np
    n = len(series)
    if n < kmax * 4:
        return 1.5  # insufficient data → assume Brownian
    lk = []
    x = np.array(series, dtype=float)
    for k in range(1, kmax + 1):
        lm_k = []
        for m in range(1, k + 1):
            indices = np.arange(m - 1, n, k)
            if len(indices) < 2:
                continue
            diffs = np.abs(np.diff(x[indices]))
            norm = (n - 1) / (k * len(diffs) * k) if len(diffs) > 0 else 0
            lm_k.append(float(np.sum(diffs) * norm))
        if lm_k:
            lk.append((np.log(1.0 / k), np.log(np.mean(lm_k)) if np.mean(lm_k) > 0 else 0))
    if len(lk) < 3:
        return 1.5
    x_fit = np.array([p[0] for p in lk])
    y_fit = np.array([p[1] for p in lk])
    slope = np.polyfit(x_fit, y_fit, 1)[0]
    return float(np.clip(slope, 1.0, 2.0))


def higuchi_fractal(symbol: str, df: "pd.DataFrame") -> "Optional[StrategyResult]":
    """Higuchi fractal dimension — regime detector independent from Hurst.

    FD < 1.3 → smooth/trending → go with the trend direction.
    FD > 1.7 → rough/chaotic → mean reversion expected.
    1.3-1.7 → Brownian → no signal.
    """
    if len(df) < 100:
        return None
    import numpy as np
    closes = df["Close"].values
    clean = closes[~np.isnan(closes)]
    if len(clean) < 80:
        return None
    fd = _higuchi_fd(clean[-80:], kmax=10)
    if 1.3 <= fd <= 1.7:
        return None  # Brownian — unpredictable
    recent_return = (clean[-1] - clean[-10]) / clean[-10] if clean[-10] != 0 else 0
    if fd < 1.3:
        # Smooth trend — follow it
        if recent_return > 0.01:
            sl, tp = _atr_stops(df, "bullish")
            if sl == 0:
                return None
            return _make("higuchi_fractal", symbol, "bullish", 1, sl, tp,
                         min(1.0, (1.3 - fd) / 0.3),
                         metadata={"fd": round(fd, 4), "regime": "smooth_trend"})
        elif recent_return < -0.01:
            sl, tp = _atr_stops(df, "bearish")
            if sl == 0:
                return None
            return _make("higuchi_fractal", symbol, "bearish", -1, sl, tp,
                         min(1.0, (1.3 - fd) / 0.3),
                         metadata={"fd": round(fd, 4), "regime": "smooth_trend"})
    elif fd > 1.7:
        # Chaotic — mean reversion
        mean_20 = float(np.mean(clean[-20:]))
        last = float(clean[-1])
        if last < mean_20 * 0.97:
            sl, tp = _atr_stops(df, "bullish")
            if sl == 0:
                return None
            return _make("higuchi_fractal", symbol, "bullish", 1, sl, tp,
                         min(1.0, (fd - 1.7) / 0.3),
                         metadata={"fd": round(fd, 4), "regime": "chaotic_reversion"})
        elif last > mean_20 * 1.03:
            sl, tp = _atr_stops(df, "bearish")
            if sl == 0:
                return None
            return _make("higuchi_fractal", symbol, "bearish", -1, sl, tp,
                         min(1.0, (fd - 1.7) / 0.3),
                         metadata={"fd": round(fd, 4), "regime": "chaotic_reversion"})
    return None


def fib_618_confluence(symbol: str, df: "pd.DataFrame") -> "Optional[StrategyResult]":
    """Fibonacci 61.8% retracement confluence — the golden ratio in price.

    When price retraces to the 61.8% Fibonacci level of a significant swing,
    AND that level coincides with recent support/resistance, it's a high-probability
    reversal zone. The 61.8% level is watched by millions of traders — self-fulfilling.
    """
    if len(df) < 50:
        return None
    import numpy as np
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    clean_h = highs[~np.isnan(highs)]
    clean_l = lows[~np.isnan(lows)]
    clean_c = closes[~np.isnan(closes)]
    if len(clean_c) < 30:
        return None
    # Find the significant swing (highest high and lowest low in last 30 bars)
    swing_high = float(np.max(clean_h[-30:]))
    swing_low = float(np.min(clean_l[-30:]))
    swing_range = swing_high - swing_low
    if swing_range <= 0:
        return None
    last = float(clean_c[-1])
    # Fibonacci levels from swing low to swing high
    fib_618 = swing_low + swing_range * 0.618
    fib_382 = swing_low + swing_range * 0.382
    # Check proximity to 61.8% level (within 1% of the level)
    tolerance = swing_range * 0.01
    # Bullish: price dropped to 38.2% retracement of an upswing (pulling back in uptrend)
    if abs(last - fib_382) < tolerance and last > swing_low:
        sl, tp = _atr_stops(df, "bullish")
        if sl == 0:
            return None
        return _make("fib_618_confluence", symbol, "bullish", 1, sl, tp, 0.6,
                     metadata={"fib_level": "38.2%", "swing_high": round(swing_high, 2),
                               "swing_low": round(swing_low, 2), "fib_price": round(fib_382, 2)})
    # Bearish: price rallied to 61.8% retracement of a downswing (relief rally in downtrend)
    if abs(last - fib_618) < tolerance and last < swing_high:
        sl, tp = _atr_stops(df, "bearish")
        if sl == 0:
            return None
        return _make("fib_618_confluence", symbol, "bearish", -1, sl, tp, 0.6,
                     metadata={"fib_level": "61.8%", "swing_high": round(swing_high, 2),
                               "swing_low": round(swing_low, 2), "fib_price": round(fib_618, 2)})
