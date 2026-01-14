#!/usr/bin/env python3
"""
signals.py - Signal Generation from Technical Indicators

Converts raw indicator values into actionable trading signals.
Designed for Guiding Light's dashboard - plain language, clear confidence levels.

Signal Types:
- Trend signals (moving average crossovers, MACD)
- Momentum signals (RSI, Stochastic)
- Volatility signals (Bollinger squeeze, ATR breakouts)
- Volume signals (OBV divergence, CMF)
- ICC signals (market structure)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal
from datetime import datetime
from enum import Enum

from .indicators import TechnicalIndicators


class SignalDirection(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class SignalStrength(Enum):
    STRONG = "strong"      # Multiple confirmations
    MODERATE = "moderate"  # Single clear signal
    WEAK = "weak"          # Early indication


@dataclass
class TradingSignal:
    """
    A single trading signal with explanation.

    Designed to be readable by non-traders.
    """
    # Core signal info
    direction: SignalDirection
    strength: SignalStrength
    confidence: float  # 0.0 to 1.0

    # Human-readable explanation
    indicator: str           # Which indicator generated this
    headline: str            # One-line summary ("RSI shows oversold")
    explanation: str         # Plain language explanation
    action: str              # What this suggests ("Consider buying")

    # Technical details (for the dashboard, hidden by default)
    value: float             # The indicator value
    threshold: float         # The threshold that triggered
    timestamp: str = ""

    # Context
    symbol: Optional[str] = None
    timeframe: str = "daily"

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_plain_language(self) -> str:
        """Format for Guiding Light's dashboard."""
        strength_emoji = {
            SignalStrength.STRONG: "[STRONG]",
            SignalStrength.MODERATE: "[MEDIUM]",
            SignalStrength.WEAK: "[WATCH]"
        }

        return f"""{strength_emoji[self.strength]} {self.headline}
         {self.explanation}
         Confidence: {self.confidence:.0%}"""


@dataclass
class SignalSummary:
    """
    Aggregated view of all signals for a symbol.
    """
    symbol: str
    overall_direction: SignalDirection
    overall_confidence: float
    signals: List[TradingSignal] = field(default_factory=list)

    # Counts
    bullish_count: int = 0
    bearish_count: int = 0
    neutral_count: int = 0

    def add_signal(self, signal: TradingSignal):
        self.signals.append(signal)
        if signal.direction == SignalDirection.BULLISH:
            self.bullish_count += 1
        elif signal.direction == SignalDirection.BEARISH:
            self.bearish_count += 1
        else:
            self.neutral_count += 1

        # Recalculate overall
        self._update_overall()

    def _update_overall(self):
        if not self.signals:
            self.overall_direction = SignalDirection.NEUTRAL
            self.overall_confidence = 0.0
            return

        # Weight by confidence
        bullish_weight = sum(s.confidence for s in self.signals if s.direction == SignalDirection.BULLISH)
        bearish_weight = sum(s.confidence for s in self.signals if s.direction == SignalDirection.BEARISH)

        total = bullish_weight + bearish_weight
        if total == 0:
            self.overall_direction = SignalDirection.NEUTRAL
            self.overall_confidence = 0.0
        elif bullish_weight > bearish_weight:
            self.overall_direction = SignalDirection.BULLISH
            self.overall_confidence = bullish_weight / total
        else:
            self.overall_direction = SignalDirection.BEARISH
            self.overall_confidence = bearish_weight / total


class SignalGenerator:
    """
    Generate trading signals from price data.

    Usage:
        gen = SignalGenerator(closes, highs, lows, volumes)
        signals = gen.generate_all()

        # Or individual signal types
        rsi_signals = gen.rsi_signals()
        macd_signals = gen.macd_signals()
    """

    def __init__(self,
                 closes: List[float],
                 highs: List[float] = None,
                 lows: List[float] = None,
                 volumes: List[float] = None,
                 symbol: str = None):
        self.closes = closes
        self.highs = highs or closes
        self.lows = lows or closes
        self.volumes = volumes or [1] * len(closes)
        self.symbol = symbol

        self.ti = TechnicalIndicators(closes, highs, lows, volumes)
        self._indicators = None

    def _get_indicators(self) -> dict:
        """Calculate all indicators (cached)."""
        if self._indicators is None:
            self._indicators = self.ti.calculate_all()
        return self._indicators

    def _latest_value(self, indicator: List[Optional[float]]) -> Optional[float]:
        """Get the most recent non-None value."""
        for val in reversed(indicator):
            if val is not None:
                return val
        return None

    # =========================================================================
    # RSI SIGNALS
    # =========================================================================

    def rsi_signals(self) -> List[TradingSignal]:
        """
        Generate RSI-based signals.

        Rules:
        - RSI < 30: Oversold (bullish)
        - RSI > 70: Overbought (bearish)
        - RSI crossing 30 from below: Strong buy
        - RSI crossing 70 from above: Strong sell
        """
        signals = []
        rsi_values = self._get_indicators()["rsi"]

        if len(rsi_values) < 2:
            return signals

        current = rsi_values[-1]
        previous = rsi_values[-2]

        if current is None:
            return signals

        # Oversold
        if current < 30:
            strength = SignalStrength.STRONG if previous and previous >= 30 else SignalStrength.MODERATE
            signals.append(TradingSignal(
                direction=SignalDirection.BULLISH,
                strength=strength,
                confidence=0.7 if strength == SignalStrength.STRONG else 0.5,
                indicator="RSI",
                headline="RSI shows oversold conditions",
                explanation=f"RSI at {current:.1f} is below 30, suggesting the price may have fallen too far too fast. This often precedes a bounce.",
                action="Consider buying - sellers may be exhausted",
                value=current,
                threshold=30,
                symbol=self.symbol
            ))

        # Overbought
        elif current > 70:
            strength = SignalStrength.STRONG if previous and previous <= 70 else SignalStrength.MODERATE
            signals.append(TradingSignal(
                direction=SignalDirection.BEARISH,
                strength=strength,
                confidence=0.7 if strength == SignalStrength.STRONG else 0.5,
                indicator="RSI",
                headline="RSI shows overbought conditions",
                explanation=f"RSI at {current:.1f} is above 70, suggesting the price may have risen too far too fast. This often precedes a pullback.",
                action="Consider selling or taking profits",
                value=current,
                threshold=70,
                symbol=self.symbol
            ))

        # Crossing back from extremes
        elif previous and previous < 30 and current >= 30:
            signals.append(TradingSignal(
                direction=SignalDirection.BULLISH,
                strength=SignalStrength.WEAK,
                confidence=0.4,
                indicator="RSI",
                headline="RSI recovering from oversold",
                explanation=f"RSI crossed back above 30 (now {current:.1f}), suggesting the oversold condition may be ending.",
                action="Watch for continuation of upward momentum",
                value=current,
                threshold=30,
                symbol=self.symbol
            ))

        elif previous and previous > 70 and current <= 70:
            signals.append(TradingSignal(
                direction=SignalDirection.BEARISH,
                strength=SignalStrength.WEAK,
                confidence=0.4,
                indicator="RSI",
                headline="RSI falling from overbought",
                explanation=f"RSI crossed back below 70 (now {current:.1f}), suggesting the overbought condition may be ending.",
                action="Watch for continuation of downward momentum",
                value=current,
                threshold=70,
                symbol=self.symbol
            ))

        return signals

    # =========================================================================
    # MACD SIGNALS
    # =========================================================================

    def macd_signals(self) -> List[TradingSignal]:
        """
        Generate MACD-based signals.

        Rules:
        - MACD crosses above Signal: Bullish
        - MACD crosses below Signal: Bearish
        - Histogram increasing: Momentum building
        - Histogram decreasing: Momentum fading
        """
        signals = []
        indicators = self._get_indicators()
        macd_line = indicators["macd_line"]
        signal_line = indicators["macd_signal"]
        histogram = indicators["macd_histogram"]

        if len(macd_line) < 2:
            return signals

        current_macd = macd_line[-1]
        current_signal = signal_line[-1]
        prev_macd = macd_line[-2]
        prev_signal = signal_line[-2]
        current_hist = histogram[-1]
        prev_hist = histogram[-2]

        if current_macd is None or current_signal is None:
            return signals

        # Crossover signals
        if prev_macd and prev_signal:
            # Bullish crossover: MACD crosses above Signal
            if prev_macd <= prev_signal and current_macd > current_signal:
                signals.append(TradingSignal(
                    direction=SignalDirection.BULLISH,
                    strength=SignalStrength.STRONG,
                    confidence=0.65,
                    indicator="MACD",
                    headline="MACD bullish crossover",
                    explanation="The MACD line crossed above the signal line, a classic buy signal indicating momentum is shifting upward.",
                    action="Consider buying - momentum turning positive",
                    value=current_macd,
                    threshold=current_signal,
                    symbol=self.symbol
                ))

            # Bearish crossover: MACD crosses below Signal
            elif prev_macd >= prev_signal and current_macd < current_signal:
                signals.append(TradingSignal(
                    direction=SignalDirection.BEARISH,
                    strength=SignalStrength.STRONG,
                    confidence=0.65,
                    indicator="MACD",
                    headline="MACD bearish crossover",
                    explanation="The MACD line crossed below the signal line, a classic sell signal indicating momentum is shifting downward.",
                    action="Consider selling - momentum turning negative",
                    value=current_macd,
                    threshold=current_signal,
                    symbol=self.symbol
                ))

        # Histogram momentum
        if current_hist and prev_hist:
            if current_hist > prev_hist and current_hist > 0:
                signals.append(TradingSignal(
                    direction=SignalDirection.BULLISH,
                    strength=SignalStrength.WEAK,
                    confidence=0.35,
                    indicator="MACD Histogram",
                    headline="Bullish momentum building",
                    explanation="The MACD histogram is increasing, showing that bullish momentum is strengthening.",
                    action="Watch for continued strength",
                    value=current_hist,
                    threshold=0,
                    symbol=self.symbol
                ))
            elif current_hist < prev_hist and current_hist < 0:
                signals.append(TradingSignal(
                    direction=SignalDirection.BEARISH,
                    strength=SignalStrength.WEAK,
                    confidence=0.35,
                    indicator="MACD Histogram",
                    headline="Bearish momentum building",
                    explanation="The MACD histogram is decreasing, showing that bearish momentum is strengthening.",
                    action="Watch for continued weakness",
                    value=current_hist,
                    threshold=0,
                    symbol=self.symbol
                ))

        return signals

    # =========================================================================
    # MOVING AVERAGE SIGNALS
    # =========================================================================

    def ma_signals(self) -> List[TradingSignal]:
        """
        Generate moving average signals.

        Rules:
        - Price above 200 SMA: Long-term uptrend
        - Price below 200 SMA: Long-term downtrend
        - 50 SMA crosses 200 SMA (Golden Cross/Death Cross)
        - Price bouncing off SMA: Support/resistance
        """
        signals = []
        indicators = self._get_indicators()
        sma_50 = indicators["sma_50"]
        sma_200 = indicators["sma_200"]
        current_price = self.closes[-1]

        # Price vs 200 SMA (long-term trend)
        sma200_current = self._latest_value(sma_200)
        if sma200_current:
            if current_price > sma200_current * 1.02:  # 2% above
                signals.append(TradingSignal(
                    direction=SignalDirection.BULLISH,
                    strength=SignalStrength.WEAK,
                    confidence=0.4,
                    indicator="SMA 200",
                    headline="Price above long-term average",
                    explanation=f"Price is trading above the 200-day moving average, indicating the long-term trend is up.",
                    action="Long-term bias is bullish",
                    value=current_price,
                    threshold=sma200_current,
                    symbol=self.symbol
                ))
            elif current_price < sma200_current * 0.98:  # 2% below
                signals.append(TradingSignal(
                    direction=SignalDirection.BEARISH,
                    strength=SignalStrength.WEAK,
                    confidence=0.4,
                    indicator="SMA 200",
                    headline="Price below long-term average",
                    explanation=f"Price is trading below the 200-day moving average, indicating the long-term trend is down.",
                    action="Long-term bias is bearish",
                    value=current_price,
                    threshold=sma200_current,
                    symbol=self.symbol
                ))

        # Golden Cross / Death Cross
        if len(sma_50) >= 2 and len(sma_200) >= 2:
            sma50_current = sma_50[-1]
            sma50_prev = sma_50[-2]
            sma200_prev = sma_200[-2]

            if all(v is not None for v in [sma50_current, sma50_prev, sma200_current, sma200_prev]):
                # Golden Cross: 50 SMA crosses above 200 SMA
                if sma50_prev <= sma200_prev and sma50_current > sma200_current:
                    signals.append(TradingSignal(
                        direction=SignalDirection.BULLISH,
                        strength=SignalStrength.STRONG,
                        confidence=0.7,
                        indicator="Golden Cross",
                        headline="Golden Cross detected",
                        explanation="The 50-day moving average crossed above the 200-day moving average. This is a major bullish signal that often precedes extended uptrends.",
                        action="Strong buy signal - consider adding to positions",
                        value=sma50_current,
                        threshold=sma200_current,
                        symbol=self.symbol
                    ))

                # Death Cross: 50 SMA crosses below 200 SMA
                elif sma50_prev >= sma200_prev and sma50_current < sma200_current:
                    signals.append(TradingSignal(
                        direction=SignalDirection.BEARISH,
                        strength=SignalStrength.STRONG,
                        confidence=0.7,
                        indicator="Death Cross",
                        headline="Death Cross detected",
                        explanation="The 50-day moving average crossed below the 200-day moving average. This is a major bearish signal that often precedes extended downtrends.",
                        action="Strong sell signal - consider reducing positions",
                        value=sma50_current,
                        threshold=sma200_current,
                        symbol=self.symbol
                    ))

        return signals

    # =========================================================================
    # BOLLINGER BANDS SIGNALS
    # =========================================================================

    def bollinger_signals(self) -> List[TradingSignal]:
        """
        Generate Bollinger Bands signals.

        Rules:
        - Price touches lower band: Potential oversold
        - Price touches upper band: Potential overbought
        - Squeeze (bands narrow): Volatility expansion coming
        - Band breakout: Momentum signal
        """
        signals = []
        indicators = self._get_indicators()
        upper = indicators["bb_upper"]
        middle = indicators["bb_middle"]
        lower = indicators["bb_lower"]
        current_price = self.closes[-1]

        upper_val = self._latest_value(upper)
        middle_val = self._latest_value(middle)
        lower_val = self._latest_value(lower)

        if not all([upper_val, middle_val, lower_val]):
            return signals

        # Price at bands
        if current_price <= lower_val:
            signals.append(TradingSignal(
                direction=SignalDirection.BULLISH,
                strength=SignalStrength.MODERATE,
                confidence=0.55,
                indicator="Bollinger Bands",
                headline="Price at lower Bollinger Band",
                explanation="Price has touched the lower Bollinger Band, which often acts as support. This can indicate oversold conditions.",
                action="Watch for a bounce off support",
                value=current_price,
                threshold=lower_val,
                symbol=self.symbol
            ))
        elif current_price >= upper_val:
            signals.append(TradingSignal(
                direction=SignalDirection.BEARISH,
                strength=SignalStrength.MODERATE,
                confidence=0.55,
                indicator="Bollinger Bands",
                headline="Price at upper Bollinger Band",
                explanation="Price has touched the upper Bollinger Band, which often acts as resistance. This can indicate overbought conditions.",
                action="Watch for a pullback from resistance",
                value=current_price,
                threshold=upper_val,
                symbol=self.symbol
            ))

        # Squeeze detection (compare current width to recent average)
        if len(upper) >= 20 and len(lower) >= 20:
            widths = []
            for i in range(-20, 0):
                if upper[i] and lower[i] and middle[i] and middle[i] != 0:
                    widths.append((upper[i] - lower[i]) / middle[i])

            if widths:
                current_width = (upper_val - lower_val) / middle_val
                avg_width = sum(widths) / len(widths)

                if current_width < avg_width * 0.7:  # 30% narrower than average
                    signals.append(TradingSignal(
                        direction=SignalDirection.NEUTRAL,
                        strength=SignalStrength.WEAK,
                        confidence=0.45,
                        indicator="Bollinger Squeeze",
                        headline="Bollinger Bands squeezing",
                        explanation="The Bollinger Bands are unusually narrow, indicating low volatility. This often precedes a significant price move in either direction.",
                        action="Prepare for volatility expansion",
                        value=current_width,
                        threshold=avg_width,
                        symbol=self.symbol
                    ))

        return signals

    # =========================================================================
    # VOLUME SIGNALS
    # =========================================================================

    def volume_signals(self) -> List[TradingSignal]:
        """
        Generate volume-based signals.

        Rules:
        - OBV divergence from price: Potential reversal
        - CMF > 0.25: Strong accumulation
        - CMF < -0.25: Strong distribution
        """
        signals = []
        indicators = self._get_indicators()
        obv_values = indicators["obv"]
        cmf_values = indicators["cmf"]

        # CMF signals
        cmf_current = self._latest_value(cmf_values)
        if cmf_current:
            if cmf_current > 0.25:
                signals.append(TradingSignal(
                    direction=SignalDirection.BULLISH,
                    strength=SignalStrength.MODERATE,
                    confidence=0.6,
                    indicator="Chaikin Money Flow",
                    headline="Strong buying pressure detected",
                    explanation=f"Chaikin Money Flow at {cmf_current:.2f} shows significant accumulation. Large players appear to be buying.",
                    action="Follow the smart money - bullish",
                    value=cmf_current,
                    threshold=0.25,
                    symbol=self.symbol
                ))
            elif cmf_current < -0.25:
                signals.append(TradingSignal(
                    direction=SignalDirection.BEARISH,
                    strength=SignalStrength.MODERATE,
                    confidence=0.6,
                    indicator="Chaikin Money Flow",
                    headline="Strong selling pressure detected",
                    explanation=f"Chaikin Money Flow at {cmf_current:.2f} shows significant distribution. Large players appear to be selling.",
                    action="Follow the smart money - bearish",
                    value=cmf_current,
                    threshold=-0.25,
                    symbol=self.symbol
                ))

        # OBV divergence (simplified - check if OBV trend differs from price trend)
        if len(obv_values) >= 10 and len(self.closes) >= 10:
            price_change = (self.closes[-1] - self.closes[-10]) / self.closes[-10]
            obv_change = (obv_values[-1] - obv_values[-10]) / abs(obv_values[-10]) if obv_values[-10] != 0 else 0

            # Bearish divergence: price up, OBV down
            if price_change > 0.05 and obv_change < -0.05:
                signals.append(TradingSignal(
                    direction=SignalDirection.BEARISH,
                    strength=SignalStrength.MODERATE,
                    confidence=0.55,
                    indicator="OBV Divergence",
                    headline="Volume not confirming price rise",
                    explanation="Price is rising but On-Balance Volume is falling. This divergence often precedes a price reversal downward.",
                    action="Be cautious of the uptrend",
                    value=obv_change,
                    threshold=price_change,
                    symbol=self.symbol
                ))

            # Bullish divergence: price down, OBV up
            elif price_change < -0.05 and obv_change > 0.05:
                signals.append(TradingSignal(
                    direction=SignalDirection.BULLISH,
                    strength=SignalStrength.MODERATE,
                    confidence=0.55,
                    indicator="OBV Divergence",
                    headline="Volume not confirming price drop",
                    explanation="Price is falling but On-Balance Volume is rising. This divergence often precedes a price reversal upward.",
                    action="Watch for bottom formation",
                    value=obv_change,
                    threshold=price_change,
                    symbol=self.symbol
                ))

        return signals

    # =========================================================================
    # ICC SIGNALS (Trades By Sci methodology)
    # =========================================================================

    def icc_signals(self) -> List[TradingSignal]:
        """
        Generate ICC (Indication, Correction, Continuation) signals.

        The ICC methodology from Trades By Sci focuses on market structure.
        Entry is on the CONTINUATION phase, not the first break (Indication).
        """
        signals = []
        icc_data = self._get_indicators()["icc_signals"]

        for icc in icc_data:
            if icc["direction"] == "bullish":
                signals.append(TradingSignal(
                    direction=SignalDirection.BULLISH,
                    strength=SignalStrength.STRONG,
                    confidence=0.7,
                    indicator="ICC Pattern",
                    headline="Bullish ICC continuation pattern",
                    explanation=f"Market structure shows: Higher High (indication) -> Higher Low (correction) -> Higher High (continuation). This is an entry signal.",
                    action=f"Buy entry. Stop loss below {icc['stop_loss']:.2f}",
                    value=icc["entry_price"],
                    threshold=icc["stop_loss"],
                    symbol=self.symbol
                ))
            elif icc["direction"] == "bearish":
                signals.append(TradingSignal(
                    direction=SignalDirection.BEARISH,
                    strength=SignalStrength.STRONG,
                    confidence=0.7,
                    indicator="ICC Pattern",
                    headline="Bearish ICC continuation pattern",
                    explanation=f"Market structure shows: Lower Low (indication) -> Lower High (correction) -> Lower Low (continuation). This is a short entry signal.",
                    action=f"Sell entry. Stop loss above {icc['stop_loss']:.2f}",
                    value=icc["entry_price"],
                    threshold=icc["stop_loss"],
                    symbol=self.symbol
                ))

        return signals

    # =========================================================================
    # AGGREGATE ALL SIGNALS
    # =========================================================================

    def generate_all(self) -> SignalSummary:
        """
        Generate all signals and return a summary.

        This is the main entry point for the dashboard.
        """
        summary = SignalSummary(
            symbol=self.symbol or "UNKNOWN",
            overall_direction=SignalDirection.NEUTRAL,
            overall_confidence=0.0
        )

        # Collect all signals
        all_signals = []
        all_signals.extend(self.rsi_signals())
        all_signals.extend(self.macd_signals())
        all_signals.extend(self.ma_signals())
        all_signals.extend(self.bollinger_signals())
        all_signals.extend(self.volume_signals())
        all_signals.extend(self.icc_signals())

        # Add to summary
        for signal in all_signals:
            summary.add_signal(signal)

        return summary

    def generate_alerts(self) -> str:
        """
        Generate plain-language alert text for Guiding Light's dashboard.
        """
        summary = self.generate_all()

        if not summary.signals:
            return f"No significant signals detected for {self.symbol or 'this asset'}."

        lines = [f"SIGNALS FOR {self.symbol or 'ASSET'}"]
        lines.append("=" * 40)
        lines.append(f"Overall: {summary.overall_direction.value.upper()} ({summary.overall_confidence:.0%} confidence)")
        lines.append(f"Bullish signals: {summary.bullish_count}")
        lines.append(f"Bearish signals: {summary.bearish_count}")
        lines.append("")

        # Sort by confidence
        sorted_signals = sorted(summary.signals, key=lambda s: s.confidence, reverse=True)

        for signal in sorted_signals:
            lines.append(signal.to_plain_language())
            lines.append("")

        return "\n".join(lines)


if __name__ == "__main__":
    # Test with sample data
    test_closes = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109,
                   110, 108, 106, 104, 102, 103, 105, 107, 109, 111,
                   113, 115, 114, 116, 118, 117, 119, 121, 120, 122]
    test_highs = [c + 1 for c in test_closes]
    test_lows = [c - 1 for c in test_closes]
    test_volumes = [1000000] * len(test_closes)

    gen = SignalGenerator(test_closes, test_highs, test_lows, test_volumes, symbol="TEST")
    print(gen.generate_alerts())
