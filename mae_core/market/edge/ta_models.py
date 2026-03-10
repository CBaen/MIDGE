"""
ta_models.py - Signal dataclasses and OHLCV extraction for TA indicators

TASignalBase, RSISignal, MACDSignal, BollingerSignal, StructureSignal,
CandleSignal dataclasses + _extract_ohlcv helper.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


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
