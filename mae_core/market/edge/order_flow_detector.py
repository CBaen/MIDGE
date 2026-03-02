#!/usr/bin/env python3
"""
order_flow_detector.py - Intraday Order Flow Imbalance Detector

Detects abnormal buying or selling pressure by analysing intraday 5-minute candles.
High volume + close near the top of the bar range = buying pressure.
High volume + close near the bottom of the bar range = selling pressure.

Algorithm per bar:
  1. Compute volume z-score vs the rolling 20-bar mean/std.
  2. Compute close-position = (close - low) / (high - low) — 0=low, 1=high.
  3. Bullish: close_position > 0.70 AND volume_zscore > imbalance_threshold.
  4. Bearish: close_position < 0.30 AND volume_zscore > imbalance_threshold.
  5. Strength = min(1.0, volume_zscore / 4.0).

Baseline update:
  Pulls 20+ days of daily volume from price_fetcher.get_daily_history() and
  stores the 20-day rolling average per ticker. Falls back to in-session
  volume mean when no baseline exists.

Follows cluster_detector.py pattern: standalone class, returns dataclass list,
no EventBus publishing. The sensing hook adapter is written separately.

Graceful degradation: returns empty list when price_fetcher is None or
candle fetching fails.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_DEFAULT_IMBALANCE_THRESHOLD = 2.0   # sigma — bars below this are ignored
_DEFAULT_MIN_BARS = 10               # minimum bars needed to compute a z-score
_BASELINE_DAYS = 20                  # rolling window for daily volume baseline
_INTRADAY_PERIOD = "2d"              # look back 2 days of 5-min candles
_INTRADAY_INTERVAL = "5m"           # 5-minute candles
_RECENT_BARS = 24                    # ~2 hours of 5-min bars to produce signals on
_CLOSE_POS_BULLISH = 0.70            # close must be this far up the range
_CLOSE_POS_BEARISH = 0.30            # close must be this far down the range
_STRENGTH_SATURATION = 4.0           # z-score at which strength reaches 1.0


# ── Dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class OrderFlowSignal:
    """A detected order-flow imbalance bar."""
    ticker: str
    direction: str                    # "bullish" or "bearish"
    strength: float                   # 0-1
    volume_zscore: float
    close_position_in_range: float    # 0=low end, 1=high end
    consecutive_directional: int      # bars in the same direction before this one
    relative_volume: float            # current bar volume / 20-day daily avg volume per bar
    timestamp: datetime


# ── OrderFlowDetector ──────────────────────────────────────────────────────────

class OrderFlowDetector:
    """
    Detects intraday order flow imbalances using volume z-scores and close position.

    Usage:
        detector = OrderFlowDetector(price_fetcher=fetcher)
        detector.update_baselines("AAPL")
        signals = detector.detect_imbalance("AAPL")
    """

    def __init__(self, price_fetcher=None) -> None:
        """
        Args:
            price_fetcher: Optional PriceFetcher instance. If None, all detect
                           calls return empty lists (graceful degradation).
        """
        self._price_fetcher = price_fetcher
        self._volume_baselines: Dict[str, float] = {}  # ticker -> 20-day avg daily volume
        self._imbalance_threshold: float = _DEFAULT_IMBALANCE_THRESHOLD
        self._min_bars: int = _DEFAULT_MIN_BARS

    # ── Public API ─────────────────────────────────────────────────────────────

    def detect_imbalance(self, ticker: str) -> List[OrderFlowSignal]:
        """
        Scan the most recent ~2 hours of 5-min bars for order flow imbalances.

        Args:
            ticker: Symbol to analyse (e.g. "AAPL", "ES=F").

        Returns:
            List of OrderFlowSignal, one per qualifying bar. Empty on failure
            or when no bars exceed the imbalance threshold.
        """
        if self._price_fetcher is None:
            logger.debug("OrderFlowDetector: no price_fetcher — returning empty")
            return []

        candles = self._fetch_candles(ticker)
        if candles is None or len(candles) < self._min_bars:
            logger.debug(
                "OrderFlowDetector: insufficient candles for %s (%d)",
                ticker, 0 if candles is None else len(candles),
            )
            return []

        return self._scan_bars(ticker, candles)

    def update_baselines(self, ticker: str) -> None:
        """
        Fetch 20+ days of daily history and store the mean daily volume.

        The baseline is used to compute relative_volume — how today's bar
        compares to the average daily volume divided across the trading day.
        Falls back silently if the fetcher is unavailable.
        """
        if self._price_fetcher is None:
            return

        try:
            history = self._price_fetcher.get_daily_history(ticker, _BASELINE_DAYS + 10)
            if not history:
                return

            volumes = [
                float(getattr(pd, "volume", 0))
                for pd in history
                if getattr(pd, "volume", 0) > 0
            ]
            if not volumes:
                return

            # Store the rolling 20-day mean (use last 20 entries)
            recent = volumes[-_BASELINE_DAYS:]
            self._volume_baselines[ticker] = sum(recent) / len(recent)
            logger.debug(
                "OrderFlowDetector: baseline for %s = %.0f", ticker, self._volume_baselines[ticker]
            )
        except Exception:
            logger.debug("OrderFlowDetector: baseline update failed for %s", ticker, exc_info=True)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _fetch_candles(self, ticker: str):
        """Return a list of OHLCV row dicts, or None on failure."""
        try:
            df = self._price_fetcher.get_intraday_candles(
                ticker, interval=_INTRADAY_INTERVAL, period=_INTRADAY_PERIOD,
            )
            if df is None or len(df) == 0:
                return None
            # Convert to a list of dicts for dependency-free processing
            rows = []
            for idx, row in df.iterrows():
                try:
                    rows.append({
                        "timestamp": idx,
                        "open": float(row.get("open", 0)),
                        "high": float(row.get("high", 0)),
                        "low": float(row.get("low", 0)),
                        "close": float(row.get("close", 0)),
                        "volume": float(row.get("volume", 0)),
                    })
                except Exception:
                    continue
            return rows if rows else None
        except Exception:
            logger.debug("OrderFlowDetector: candle fetch failed for %s", ticker, exc_info=True)
            return None

    def _scan_bars(self, ticker: str, candles: list) -> List[OrderFlowSignal]:
        """
        Walk through the recent bars and produce an OrderFlowSignal for each
        that meets the imbalance threshold.

        Only the last _RECENT_BARS bars are evaluated for signals (older bars
        are used to build the rolling volume statistics).
        """
        if len(candles) < self._min_bars:
            return []

        volumes = [c["volume"] for c in candles]
        daily_baseline = self._volume_baselines.get(ticker, 0.0)

        signals: List[OrderFlowSignal] = []
        # We need at least min_bars to compute a rolling mean/std
        start_idx = max(self._min_bars, len(candles) - _RECENT_BARS)

        for i in range(start_idx, len(candles)):
            bar = candles[i]
            bar_volume = bar["volume"]

            # Rolling window: use all bars up to (not including) bar i for baseline
            window_vols = volumes[max(0, i - self._min_bars):i]
            if not window_vols:
                continue

            vol_mean = sum(window_vols) / len(window_vols)
            vol_std = _std(window_vols)
            if vol_std < 1.0:
                continue  # Degenerate distribution — skip

            zscore = (bar_volume - vol_mean) / vol_std
            if zscore < self._imbalance_threshold:
                continue  # Below detection threshold

            bar_range = bar["high"] - bar["low"]
            if bar_range <= 0:
                continue

            close_pos = (bar["close"] - bar["low"]) / bar_range

            # Directional classification
            if close_pos >= _CLOSE_POS_BULLISH:
                direction = "bullish"
            elif close_pos <= _CLOSE_POS_BEARISH:
                direction = "bearish"
            else:
                continue  # Close is mid-range — no strong directional read

            strength = min(1.0, zscore / _STRENGTH_SATURATION)
            consecutive = self._count_consecutive(candles, i, direction)
            relative_vol = self._compute_relative_volume(bar_volume, daily_baseline)

            ts = bar["timestamp"]
            if not isinstance(ts, datetime):
                try:
                    ts = ts.to_pydatetime()
                    # Strip tz if needed for uniformity — keep naive UTC equivalent
                    if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
                        ts = ts.replace(tzinfo=None)
                except Exception:
                    ts = datetime.now()

            signals.append(OrderFlowSignal(
                ticker=ticker,
                direction=direction,
                strength=round(strength, 4),
                volume_zscore=round(zscore, 4),
                close_position_in_range=round(close_pos, 4),
                consecutive_directional=consecutive,
                relative_volume=round(relative_vol, 4),
                timestamp=ts,
            ))

        return signals

    def _count_consecutive(self, candles: list, current_idx: int, direction: str) -> int:
        """
        Count how many bars immediately before current_idx move in the same
        directional sense (close > open for bullish, close < open for bearish).
        """
        count = 0
        for j in range(current_idx - 1, -1, -1):
            bar = candles[j]
            if direction == "bullish" and bar["close"] > bar["open"]:
                count += 1
            elif direction == "bearish" and bar["close"] < bar["open"]:
                count += 1
            else:
                break
        return count

    def _compute_relative_volume(self, bar_volume: float, daily_baseline: float) -> float:
        """
        Relative volume: bar volume vs average per-bar daily volume.

        daily_baseline is total daily volume; divide by ~78 (6.5h * 12 bars/h)
        to get average 5-min bar volume. Falls back to 1.0 when baseline is zero.
        """
        if daily_baseline <= 0:
            return 1.0
        avg_bar_volume = daily_baseline / 78.0  # 78 five-min bars per regular session
        return bar_volume / avg_bar_volume if avg_bar_volume > 0 else 1.0


# ── Utility ────────────────────────────────────────────────────────────────────

def _std(values: list) -> float:
    """Population standard deviation. Returns 0.0 for empty or single-element list."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return math.sqrt(variance)
