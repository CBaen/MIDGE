"""
session_sweep_models.py - Dataclasses for ICT Session Sweep Detector

SessionLevel, FairValueGap, SessionSweepSignal dataclasses.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict


@dataclass
class SessionLevel:
    """A session's high and low price levels (liquidity pools)."""
    session: str           # "asia", "london", "new_york"
    session_date: str      # Date string (YYYY-MM-DD) the session started
    session_high: float
    session_low: float
    high_time: str         # ISO timestamp of the high candle
    low_time: str          # ISO timestamp of the low candle
    candles_in_session: int


@dataclass
class FairValueGap:
    """A 3-candle Fair Value Gap (price imbalance zone)."""
    top: float
    bottom: float
    midpoint: float
    direction: str         # "bullish" or "bearish"
    formed_at: str         # ISO timestamp of the middle candle
    candle_index: int
    size_pct: float        # Gap size as % of price
    is_mitigated: bool = False


@dataclass
class SessionSweepSignal:
    """A detected session liquidity sweep with FVG entry zone."""
    # Identity
    sweep_id: str = ""
    symbol: str = ""

    # Sweep details
    sweep_type: str = ""         # "high_sweep" or "low_sweep"
    direction: str = ""          # "bullish" (low swept) or "bearish" (high swept)
    session_swept: str = ""      # "asia", "london", "new_york"
    sweep_level: float = 0.0     # The session high/low that was swept
    sweep_candle_high: float = 0.0
    sweep_candle_low: float = 0.0
    reversal_confirmed: bool = False

    # FVG entry zone
    fvg_top: float = 0.0
    fvg_bottom: float = 0.0
    fvg_midpoint: float = 0.0
    fvg_size_pct: float = 0.0

    # Trade parameters
    entry_zone_top: float = 0.0
    entry_zone_bottom: float = 0.0
    stop_level: float = 0.0
    target_level: float = 0.0
    rr_ratio: float = 2.0

    # Signal quality
    confidence: float = 0.50
    strength: float = 0.50
    kill_zone: bool = False

    # IFVG + pattern stacking scores
    is_ifvg: bool = False
    displacement_score: float = 0.0
    fvg_atr_ratio: float = 0.0
    quality_score: float = 0.0

    # Metadata
    signal_source: str = "session_sweep"
    decay_rate: float = 0.85     # Hourly-scale decay (~18h half-life)
    detected_at: str = ""
    metadata: dict = field(default_factory=dict)

    def to_plain_language(self) -> str:
        """Human-readable summary of the sweep signal."""
        kz_tag = " [KILL ZONE]" if self.kill_zone else ""
        return (
            f"{self.symbol} {self.direction.upper()} — "
            f"{self.session_swept} {self.sweep_type.replace('_', ' ')} "
            f"at {self.sweep_level:.2f}{kz_tag}. "
            f"FVG zone: {self.fvg_bottom:.2f}-{self.fvg_top:.2f}, "
            f"stop {self.stop_level:.2f}, target {self.target_level:.2f} "
            f"({self.rr_ratio:.1f}R). "
            f"Confidence: {self.confidence:.0%}"
        )
