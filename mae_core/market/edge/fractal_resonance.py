"""Fractal Market Resonance — Gift 7 (Council Gift).

Detects self-similar price patterns at multiple timeframes. When the
daily chart shows a bullish structure AND the weekly chart shows the same
AND the monthly confirms, the signal is fractal-reinforced.

This is Law 4 (Fractal Self-Similarity) applied to market data. The
same geometry repeating at every scale. A breakout on the daily inside
a breakout on the weekly inside a breakout on the monthly is the
strongest possible confirmation.

Reuses ta_indicators.py Market Structure logic (HH/HL = bullish,
LL/LH = bearish) at each timeframe.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_TIMEFRAMES = ["1d", "1wk", "1mo"]
_TIMEFRAME_PERIODS = {"1d": 90, "1wk": 52, "1mo": 24}


@dataclass
class TimeframePattern:
    """Pattern analysis for a single timeframe."""

    timeframe: str
    structure: str  # "bullish", "bearish", "neutral"
    pattern_type: str  # "breakout", "breakdown", "consolidation", "trend"
    strength: float  # 0.0 to 1.0
    swing_points: int = 0  # Number of swing points identified


@dataclass
class FractalResonanceResult:
    """Result of fractal resonance analysis across timeframes."""

    ticker: str
    resonance_score: float  # 0.0 to 1.0
    aligned_timeframes: int  # How many timeframes agree
    dominant_direction: str  # "bullish", "bearish", "neutral"
    patterns_by_timeframe: Dict[str, TimeframePattern] = field(
        default_factory=dict
    )


class FractalResonanceDetector:
    """Detect self-similar price patterns across multiple timeframes.

    Uses Market Structure analysis (higher-highs/higher-lows for bullish,
    lower-lows/lower-highs for bearish) at daily, weekly, and monthly
    scales. When all three align, the resonance score approaches 1.0.
    """

    def __init__(self, price_fetcher=None):
        self._price_fetcher = price_fetcher
        self._timeframes = list(_TIMEFRAMES)

    def detect_resonance(
        self, ticker: str
    ) -> Optional[FractalResonanceResult]:
        """Analyze a ticker for fractal resonance across timeframes.

        Returns None if insufficient data is available.
        """
        if self._price_fetcher is None:
            return None

        patterns: Dict[str, TimeframePattern] = {}

        for tf in self._timeframes:
            try:
                pattern = self._analyze_timeframe(ticker, tf)
                if pattern is not None:
                    patterns[tf] = pattern
            except Exception:
                logger.debug(
                    "Fractal analysis failed for %s at %s", ticker, tf,
                    exc_info=True,
                )

        if len(patterns) < 2:
            return None  # Need at least 2 timeframes

        resonance_score = self._compute_resonance_score(
            list(patterns.values())
        )
        aligned = self._count_aligned(list(patterns.values()))
        dominant = self._determine_dominant_direction(list(patterns.values()))

        return FractalResonanceResult(
            ticker=ticker,
            resonance_score=resonance_score,
            aligned_timeframes=aligned,
            dominant_direction=dominant,
            patterns_by_timeframe=patterns,
        )

    def _analyze_timeframe(
        self, ticker: str, interval: str
    ) -> Optional[TimeframePattern]:
        """Analyze price structure for a single timeframe."""
        period = _TIMEFRAME_PERIODS.get(interval, 90)

        # Get price history at the right timeframe
        history = self._get_history(ticker, interval, period)
        if not history or len(history) < 10:
            return None

        # Extract highs and lows for structure analysis
        highs = [getattr(bar, "high", 0) for bar in history]
        lows = [getattr(bar, "low", 0) for bar in history]
        closes = [getattr(bar, "close", 0) for bar in history]

        if not highs or not lows or not closes:
            return None

        # Compute swing points and structure
        swing_highs = self._find_swing_points(highs, is_high=True)
        swing_lows = self._find_swing_points(lows, is_high=False)
        total_swings = len(swing_highs) + len(swing_lows)

        structure = self._classify_structure(swing_highs, swing_lows)
        pattern_type = self._classify_pattern_type(
            closes, swing_highs, swing_lows
        )
        strength = self._compute_pattern_strength(
            closes, swing_highs, swing_lows, structure
        )

        return TimeframePattern(
            timeframe=interval,
            structure=structure,
            pattern_type=pattern_type,
            strength=strength,
            swing_points=total_swings,
        )

    def _get_history(self, ticker: str, interval: str, period: int) -> list:
        """Get price history using the appropriate fetcher method."""
        pf = self._price_fetcher
        try:
            if interval == "1wk" and hasattr(pf, "get_weekly_history"):
                return pf.get_weekly_history(ticker, weeks=period) or []
            elif interval == "1mo" and hasattr(pf, "get_monthly_history"):
                return pf.get_monthly_history(ticker, months=period) or []
            elif interval == "1d" and hasattr(pf, "get_daily_history"):
                return pf.get_daily_history(ticker, period) or []
        except Exception:
            logger.debug("Price fetch failed for %s/%s", ticker, interval)
        return []

    @staticmethod
    def _find_swing_points(
        values: List[float], is_high: bool, lookback: int = 5
    ) -> List[float]:
        """Identify swing highs or swing lows from a price series."""
        swings = []
        for i in range(lookback, len(values) - lookback):
            window = values[i - lookback: i + lookback + 1]
            if is_high and values[i] == max(window):
                swings.append(values[i])
            elif not is_high and values[i] == min(window):
                swings.append(values[i])
        return swings

    @staticmethod
    def _classify_structure(
        swing_highs: List[float], swing_lows: List[float]
    ) -> str:
        """Classify market structure from swing points.

        HH + HL = bullish, LL + LH = bearish, else neutral.
        """
        if len(swing_highs) < 2 and len(swing_lows) < 2:
            return "neutral"

        hh = False
        hl = False
        ll = False
        lh = False

        if len(swing_highs) >= 2:
            hh = swing_highs[-1] > swing_highs[-2]
            lh = swing_highs[-1] < swing_highs[-2]

        if len(swing_lows) >= 2:
            hl = swing_lows[-1] > swing_lows[-2]
            ll = swing_lows[-1] < swing_lows[-2]

        if hh and hl:
            return "bullish"
        if ll and lh:
            return "bearish"
        return "neutral"

    @staticmethod
    def _classify_pattern_type(
        closes: List[float],
        swing_highs: List[float],
        swing_lows: List[float],
    ) -> str:
        """Classify the pattern type (breakout, breakdown, consolidation, trend)."""
        if len(closes) < 5:
            return "consolidation"

        recent_close = closes[-1]
        avg_close = sum(closes[-20:]) / max(len(closes[-20:]), 1)

        # Check for breakout/breakdown relative to recent range
        if swing_highs and recent_close > max(swing_highs):
            return "breakout"
        if swing_lows and recent_close < min(swing_lows):
            return "breakdown"

        # Check trend vs consolidation
        pct_change = (closes[-1] - closes[0]) / closes[0] if closes[0] != 0 else 0
        if abs(pct_change) > 0.10:
            return "trend"

        return "consolidation"

    @staticmethod
    def _compute_pattern_strength(
        closes: List[float],
        swing_highs: List[float],
        swing_lows: List[float],
        structure: str,
    ) -> float:
        """Compute pattern strength from 0.0 to 1.0."""
        if structure == "neutral":
            return 0.3

        # Strength based on consistency of swing sequence
        strength = 0.5

        # More swing points = more confidence in structure
        total_swings = len(swing_highs) + len(swing_lows)
        if total_swings >= 6:
            strength += 0.15
        elif total_swings >= 4:
            strength += 0.10

        # Trend magnitude adds strength
        if len(closes) >= 5:
            pct_change = abs(closes[-1] - closes[0]) / closes[0] if closes[0] != 0 else 0
            if pct_change > 0.20:
                strength += 0.15
            elif pct_change > 0.10:
                strength += 0.10

        return min(1.0, strength)

    def _compute_resonance_score(
        self, patterns: List[TimeframePattern]
    ) -> float:
        """Compute resonance score from aligned timeframe patterns.

        2/3 aligned = 0.7, 3/3 aligned = 1.0, conflicting = 0.3.
        """
        if not patterns:
            return 0.0

        aligned = self._count_aligned(patterns)
        total = len(patterns)

        if total >= 3 and aligned == 3:
            return 1.0
        elif aligned >= 2:
            return 0.7
        elif aligned == 1:
            return 0.4
        return 0.3

    @staticmethod
    def _count_aligned(patterns: List[TimeframePattern]) -> int:
        """Count how many timeframes share the dominant direction."""
        if not patterns:
            return 0

        directions = [p.structure for p in patterns if p.structure != "neutral"]
        if not directions:
            return 0

        bullish = directions.count("bullish")
        bearish = directions.count("bearish")
        return max(bullish, bearish)

    @staticmethod
    def _determine_dominant_direction(
        patterns: List[TimeframePattern],
    ) -> str:
        """Determine the dominant direction across timeframes."""
        bullish = sum(1 for p in patterns if p.structure == "bullish")
        bearish = sum(1 for p in patterns if p.structure == "bearish")

        if bullish > bearish:
            return "bullish"
        if bearish > bullish:
            return "bearish"
        return "neutral"

    def get_statistics(self) -> dict:
        """Return detector statistics for holon protocol."""
        return {
            "timeframes_analyzed": len(self._timeframes),
            "has_price_fetcher": self._price_fetcher is not None,
        }
