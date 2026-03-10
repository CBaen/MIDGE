"""
sweep_backtest_models.py - Dataclasses for ICT Session Sweep + IFVG Backtest

Level, FVGZone, SweepEvent, Trade dataclasses used by the sweep backtest engine.
"""

from dataclasses import dataclass, field
from datetime import date


@dataclass
class Level:
    """A price level (session high or low) that can be swept."""
    session: str       # "asia", "london", "prev_day"
    level_date: date
    high: float
    low: float


@dataclass
class FVGZone:
    """A Fair Value Gap zone."""
    top: float
    bottom: float
    direction: str     # "bullish" (gap up) or "bearish" (gap down)
    formed_idx: int    # Positional index in DataFrame
    midpoint: float = 0.0

    def __post_init__(self):
        self.midpoint = (self.top + self.bottom) / 2


@dataclass
class SweepEvent:
    """A detected sweep of a session level."""
    symbol: str
    direction: str        # "bullish" (low swept) or "bearish" (high swept)
    session: str
    sweep_level: float
    sweep_idx: int        # Positional index of sweep candle
    sweep_extreme: float  # Wick extreme of sweep candle
    sweep_time: str


@dataclass
class Trade:
    """A simulated trade."""
    symbol: str
    direction: str
    entry_price: float
    stop_price: float
    target_1r: float
    target_2r: float
    entry_time: str
    exit_time: str = ""
    exit_price: float = 0.0
    result: str = ""       # "win_2r", "loss", "timeout"
    r_captured: float = 0.0
    hit_1r: bool = False   # Did price reach 1R (even if 2R wasn't hit)?
    session_swept: str = ""
    sweep_level: float = 0.0
    ifvg_top: float = 0.0
    ifvg_bottom: float = 0.0
    risk_pts: float = 0.0
    displacement_score: float = 0.0
    fvg_atr_ratio: float = 0.0
    kill_zone_score: float = 0.0
    quality_score: float = 0.0
