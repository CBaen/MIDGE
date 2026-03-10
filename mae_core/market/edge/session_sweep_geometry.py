"""
session_sweep_geometry.py - Session level marking and FVG geometry for ICT Sweep Detector

Contains: _mark_session_levels, _extract_session_levels, _find_fvg,
_find_ifvg, _score_displacement, _compute_atr.

These are standalone functions that take a DataFrame and index arguments.
SessionSweepDetector uses them as methods via the mixin pattern by importing
them and calling with self as first arg — or they can be called directly.
"""

import logging
from datetime import datetime, time, timedelta
from typing import List, Optional, Tuple
from zoneinfo import ZoneInfo

from mae_core.market.edge.session_sweep_models import SessionLevel, FairValueGap

logger = logging.getLogger(__name__)

EASTERN = ZoneInfo("America/New_York")

# Full session windows — used to record highs/lows
SESSION_WINDOWS = {
    "asia":     {"start": time(20, 0), "end": time(0, 0)},   # crosses midnight
    "london":   {"start": time(2, 0),  "end": time(5, 0)},
    "new_york": {"start": time(7, 0),  "end": time(10, 0)},
}


def mark_session_levels(ohlc, min_fvg_pct: float = 0.01) -> List[SessionLevel]:
    """Mark session highs/lows from the last 2 completed sessions of each type."""
    levels = []
    now_et = datetime.now(EASTERN)

    for session_name, window in SESSION_WINDOWS.items():
        session_levels = extract_session_levels(ohlc, session_name, window, now_et)
        levels.extend(session_levels)

    return levels


def extract_session_levels(
    ohlc,
    session_name: str,
    window: dict,
    now_et: datetime,
) -> List[SessionLevel]:
    """Extract session high/low for last 2 completed sessions of a type."""
    start_t = window["start"]
    end_t = window["end"]
    crosses_midnight = start_t > end_t

    results = []
    for days_back in range(5):
        ref_date = (now_et - timedelta(days=days_back)).date()

        if crosses_midnight:
            start_dt = datetime.combine(ref_date, start_t, tzinfo=EASTERN)
            if end_t == time(0, 0):
                end_dt = datetime.combine(
                    ref_date + timedelta(days=1), time(0, 0), tzinfo=EASTERN,
                )
            else:
                end_dt = datetime.combine(
                    ref_date + timedelta(days=1), end_t, tzinfo=EASTERN,
                )
        else:
            start_dt = datetime.combine(ref_date, start_t, tzinfo=EASTERN)
            end_dt = datetime.combine(ref_date, end_t, tzinfo=EASTERN)

        if end_dt >= now_et:
            continue

        mask = (ohlc.index >= start_dt) & (ohlc.index < end_dt)
        session_candles = ohlc[mask]

        if len(session_candles) < 5:
            continue

        high_val = session_candles["high"].max()
        low_val = session_candles["low"].min()
        high_idx = session_candles["high"].idxmax()
        low_idx = session_candles["low"].idxmin()

        results.append(SessionLevel(
            session=session_name,
            session_date=ref_date.isoformat(),
            session_high=float(high_val),
            session_low=float(low_val),
            high_time=str(high_idx),
            low_time=str(low_idx),
            candles_in_session=len(session_candles),
        ))

        if len(results) >= 2:
            break

    return results


def find_fvg(
    ohlc,
    from_idx: int,
    direction: str,
    min_fvg_pct: float = 0.01,
) -> Optional[FairValueGap]:
    """Find a Fair Value Gap after the sweep candle (up to 20 candles forward).

    Bullish FVG: candle[i+2].low > candle[i].high (gap up)
    Bearish FVG: candle[i+2].high < candle[i].low (gap down)
    """
    scan_end = min(from_idx + 20, len(ohlc) - 2)

    for i in range(from_idx, scan_end):
        c0 = ohlc.iloc[i]
        c2 = ohlc.iloc[i + 2]
        c1 = ohlc.iloc[i + 1]

        if direction == "bullish" and c2["low"] > c0["high"]:
            top = float(c2["low"])
            bottom = float(c0["high"])
            gap_size = top - bottom
            mid_price = float(c1["close"])
            size_pct = (gap_size / mid_price) * 100 if mid_price > 0 else 0

            if size_pct < min_fvg_pct:
                continue

            return FairValueGap(
                top=top, bottom=bottom, midpoint=(top + bottom) / 2,
                direction="bullish", formed_at=str(ohlc.index[i + 1]),
                candle_index=i + 1, size_pct=round(size_pct, 4),
            )

        elif direction == "bearish" and c2["high"] < c0["low"]:
            top = float(c0["low"])
            bottom = float(c2["high"])
            gap_size = top - bottom
            mid_price = float(c1["close"])
            size_pct = (gap_size / mid_price) * 100 if mid_price > 0 else 0

            if size_pct < min_fvg_pct:
                continue

            return FairValueGap(
                top=top, bottom=bottom, midpoint=(top + bottom) / 2,
                direction="bearish", formed_at=str(ohlc.index[i + 1]),
                candle_index=i + 1, size_pct=round(size_pct, 4),
            )

    return None


def find_ifvg(ohlc, sweep_idx: int, direction: str) -> bool:
    """Check if a prior-trend FVG was mitigated post-sweep (IFVG).

    Returns True if any prior-trend FVG was mitigated within 80 candles.
    """
    prior_dir = "bearish" if direction == "bullish" else "bullish"
    lookback = min(200, sweep_idx)
    fvg_start = max(0, sweep_idx - lookback)

    prior_fvgs = []
    for i in range(fvg_start, sweep_idx - 1):
        if i + 2 >= len(ohlc):
            break
        c0 = ohlc.iloc[i]
        c2 = ohlc.iloc[i + 2]
        c1 = ohlc.iloc[i + 1]
        mid_price = float(c1["close"])
        if mid_price <= 0:
            continue

        if prior_dir == "bearish" and c2["high"] < c0["low"]:
            top = float(c0["low"])
            bottom = float(c2["high"])
            if (top - bottom) / mid_price >= 0.0005:
                prior_fvgs.append((top, bottom, i + 1))

        elif prior_dir == "bullish" and c2["low"] > c0["high"]:
            top = float(c2["low"])
            bottom = float(c0["high"])
            if (top - bottom) / mid_price >= 0.0005:
                prior_fvgs.append((top, bottom, i + 1))

    if not prior_fvgs:
        return False

    fill_end = min(sweep_idx + 80, len(ohlc))
    for top, bottom, _ in reversed(prior_fvgs):
        for j in range(sweep_idx + 1, fill_end):
            close = ohlc.iloc[j]["close"]
            if prior_dir == "bearish" and close > top:
                return True
            if prior_dir == "bullish" and close < bottom:
                return True

    return False


def score_displacement(ohlc, sweep_idx: int, direction: str) -> float:
    """Measure reversal quality via body-to-range ratio of post-sweep candles."""
    ratios = []
    for i in range(sweep_idx + 1, min(sweep_idx + 6, len(ohlc))):
        candle = ohlc.iloc[i]
        bar_range = candle["high"] - candle["low"]
        if bar_range <= 0:
            continue
        body = abs(candle["close"] - candle["open"])
        if direction == "bullish" and candle["close"] > candle["open"]:
            ratios.append(body / bar_range)
        elif direction == "bearish" and candle["close"] < candle["open"]:
            ratios.append(body / bar_range)
    return sum(ratios) / len(ratios) if ratios else 0.0


def compute_atr(ohlc, idx: int, period: int = 14) -> float:
    """14-period Average True Range at a given index."""
    start = max(1, idx - period + 1)
    trs = []
    for i in range(start, idx + 1):
        high = ohlc.iloc[i]["high"]
        low = ohlc.iloc[i]["low"]
        prev_close = ohlc.iloc[i - 1]["close"]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    return sum(trs) / len(trs) if trs else 0.0
