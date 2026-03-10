"""
session_sweep_scoring.py - Kill zone and quality scoring for ICT Session Sweep Detector

Contains: get_kill_zone_score, compute_quality, is_kill_zone.

These are standalone functions used by SessionSweepDetector.
"""

from datetime import time
from typing import Optional
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")

# Kill zones — high-probability windows within sessions
KILL_ZONES = {
    "asia_kz":     {"start": time(20, 0), "end": time(22, 0)},
    "london_kz":   {"start": time(2, 0),  "end": time(5, 0)},
    "new_york_kz": {"start": time(7, 0),  "end": time(10, 0)},
}


def get_kill_zone_score(dt) -> float:
    """Tiered kill zone score (1.0=NY, 0.85=London, 0.70=Asia, 0.0=outside)."""
    try:
        t = dt.time() if hasattr(dt, "time") else dt
    except Exception:
        return 0.0
    if time(7, 0) <= t < time(10, 0):
        return 1.0
    if time(2, 0) <= t < time(5, 0):
        return 0.85
    if t >= time(20, 0) or (t < time(22, 0) and t >= time(20, 0)):
        return 0.70
    return 0.0


def compute_quality(displacement: float, fvg_atr: float, kz: float) -> float:
    """Composite quality: 40% displacement + 35% FVG/ATR + 25% kill zone."""
    fvg_atr_score = min(1.0, fvg_atr / 1.5)
    return displacement * 0.4 + fvg_atr_score * 0.35 + kz * 0.25


def is_kill_zone(dt) -> bool:
    """Check if a datetime falls within an ICT kill zone window."""
    try:
        if hasattr(dt, "time"):
            t = dt.time()
        else:
            return False
    except Exception:
        return False

    for kz_name, window in KILL_ZONES.items():
        start = window["start"]
        end = window["end"]

        if start <= end:
            if start <= t <= end:
                return True
        else:
            if t >= start or t <= end:
                return True

    return False


def classify_sweep_quality(
    kill_zone: bool,
    fvg_size_pct: float,
    side: str,
    sweep_candle,
    sweep_level: float,
) -> float:
    """Score sweep quality into a confidence value (0.0-1.0).

    Base: 0.45 (sweep + FVG confirmed)
    +0.15 if during kill zone
    +0.10 if FVG is sizable (>0.03%)
    +0.10 if clean single-candle rejection (wick vs body ratio)
    Cap: 0.90
    """
    conf = 0.45

    if kill_zone:
        conf += 0.15

    if fvg_size_pct > 0.03:
        conf += 0.10

    body = abs(float(sweep_candle["close"]) - float(sweep_candle["open"]))
    full_range = float(sweep_candle["high"]) - float(sweep_candle["low"])
    if full_range > 0:
        wick_ratio = 1.0 - (body / full_range)
        if wick_ratio > 0.60:
            conf += 0.10

    return min(0.90, conf)
