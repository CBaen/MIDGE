"""
sweep_backtest_engine.py - ICT Session Sweep + IFVG Backtest Engine Core

Fetch, session level detection, sweep detection, FVG/IFVG detection,
and quality scoring helpers. Pure computation — no reporting.
"""

import logging
from datetime import datetime, time, timedelta, date
from typing import List, Optional, Tuple
from zoneinfo import ZoneInfo

from mae_core.market.edge.sweep_backtest_models import Level, FVGZone, SweepEvent, Trade

logger = logging.getLogger(__name__)

try:
    import pandas as pd
    import numpy as np
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

ET = ZoneInfo("America/New_York")


class SweepBacktestEngine:
    """Core engine: fetch, detect, score. Used by SweepBacktester."""

    def __init__(
        self,
        interval: str = "5m",
        days: int = 59,
        min_fvg_pct: float = 0.0005,
        sweep_confirm_candles: int = 3,
        fvg_lookback: int = 200,
        fill_lookforward: int = 80,
        entry_timeout: int = 100,
        trade_timeout: int = 200,
        min_quality: float = 0.0,
    ):
        self.interval = interval
        self.days = days
        self.min_fvg_pct = min_fvg_pct
        self.sweep_confirm = sweep_confirm_candles
        self.fvg_lookback = fvg_lookback
        self.fill_lookforward = fill_lookforward
        self.entry_timeout = entry_timeout
        self.trade_timeout = trade_timeout
        self.min_quality = min_quality

    # ── Data fetch ────────────────────────────────────────

    def fetch_candles(self, symbol: str) -> Optional["pd.DataFrame"]:
        """Fetch intraday candles from yfinance."""
        if not HAS_YF or not HAS_DEPS:
            return None
        try:
            end = datetime.now()
            start = end - timedelta(days=self.days)
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval=self.interval,
            )
            if df.empty:
                return None
            df.columns = [c.lower() for c in df.columns]
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            df.index = df.index.tz_convert(ET)
            df = df.dropna(subset=["open", "high", "low", "close"])
            return df
        except Exception as e:
            logger.warning("Fetch failed for %s: %s", symbol, e)
            return None

    # ── Session levels ────────────────────────────────────

    def get_session_levels(
        self, df: "pd.DataFrame", ref_date: date, is_futures: bool,
    ) -> List[Level]:
        """Get session highs/lows to use as liquidity levels."""
        levels = []

        if is_futures:
            asia_start = datetime.combine(
                ref_date - timedelta(days=1), time(18, 0), tzinfo=ET,
            )
            asia_end = datetime.combine(ref_date, time(0, 0), tzinfo=ET)
            asia = df[(df.index >= asia_start) & (df.index < asia_end)]
            if len(asia) >= 3:
                levels.append(Level(
                    session="asia", level_date=ref_date,
                    high=float(asia["high"].max()),
                    low=float(asia["low"].min()),
                ))

            lon_start = datetime.combine(ref_date, time(2, 0), tzinfo=ET)
            lon_end = datetime.combine(ref_date, time(5, 0), tzinfo=ET)
            lon = df[(df.index >= lon_start) & (df.index < lon_end)]
            if len(lon) >= 3:
                levels.append(Level(
                    session="london", level_date=ref_date,
                    high=float(lon["high"].max()),
                    low=float(lon["low"].min()),
                ))
        else:
            for offset in range(1, 5):
                prev = ref_date - timedelta(days=offset)
                day_start = datetime.combine(prev, time(9, 30), tzinfo=ET)
                day_end = datetime.combine(prev, time(16, 0), tzinfo=ET)
                prev_day = df[(df.index >= day_start) & (df.index < day_end)]
                if len(prev_day) >= 10:
                    levels.append(Level(
                        session="prev_day", level_date=ref_date,
                        high=float(prev_day["high"].max()),
                        low=float(prev_day["low"].min()),
                    ))
                    break

        return levels

    # ── Sweep detection ───────────────────────────────────

    def detect_sweeps(
        self, df: "pd.DataFrame", levels: List[Level],
        scan_start: datetime, scan_end: datetime, symbol: str,
    ) -> List[SweepEvent]:
        """Find sweeps of session levels within a scan window."""
        scan = df[(df.index >= scan_start) & (df.index < scan_end)]
        if len(scan) < 5:
            return []

        sweeps = []
        for level in levels:
            for side in ("high", "low"):
                target = level.high if side == "high" else level.low
                sweep = self._find_sweep(df, scan, target, side, level, symbol)
                if sweep:
                    sweeps.append(sweep)
        return sweeps

    def _find_sweep(
        self, full_df, scan_df, target, side, level, symbol,
    ) -> Optional[SweepEvent]:
        """Check if a price level was swept (breached then reversed)."""
        for i in range(len(scan_df)):
            candle = scan_df.iloc[i]

            breached = (
                (side == "high" and candle["high"] > target)
                or (side == "low" and candle["low"] < target)
            )
            if not breached:
                continue

            for off in range(self.sweep_confirm):
                j = i + off
                if j >= len(scan_df):
                    break
                close = scan_df.iloc[j]["close"]
                reversed_back = (
                    (side == "high" and close < target)
                    or (side == "low" and close > target)
                )
                if reversed_back:
                    abs_idx = full_df.index.get_loc(scan_df.index[i])
                    direction = "bullish" if side == "low" else "bearish"
                    extreme = (
                        float(candle["low"]) if side == "low"
                        else float(candle["high"])
                    )
                    return SweepEvent(
                        symbol=symbol,
                        direction=direction,
                        session=level.session,
                        sweep_level=target,
                        sweep_idx=abs_idx,
                        sweep_extreme=extreme,
                        sweep_time=str(scan_df.index[i]),
                    )
        return None

    # ── FVG detection ─────────────────────────────────────

    def find_fvgs(
        self, df: "pd.DataFrame", start_idx: int, end_idx: int,
        direction: Optional[str] = None,
    ) -> List[FVGZone]:
        """Find Fair Value Gaps (3-candle gap patterns) in a range."""
        fvgs = []
        end_idx = min(end_idx, len(df) - 2)

        for i in range(max(0, start_idx), end_idx):
            c0 = df.iloc[i]
            c1 = df.iloc[i + 1]
            c2 = df.iloc[i + 2]
            mid_price = float(c1["close"])
            if mid_price <= 0:
                continue

            if c2["low"] > c0["high"]:
                top = float(c2["low"])
                bottom = float(c0["high"])
                if (top - bottom) / mid_price >= self.min_fvg_pct:
                    if direction is None or direction == "bullish":
                        fvgs.append(FVGZone(
                            top=top, bottom=bottom,
                            direction="bullish", formed_idx=i + 1,
                        ))

            if c2["high"] < c0["low"]:
                top = float(c0["low"])
                bottom = float(c2["high"])
                if (top - bottom) / mid_price >= self.min_fvg_pct:
                    if direction is None or direction == "bearish":
                        fvgs.append(FVGZone(
                            top=top, bottom=bottom,
                            direction="bearish", formed_idx=i + 1,
                        ))

        return fvgs

    # ── IFVG detection ────────────────────────────────────

    def find_ifvg(
        self, df: "pd.DataFrame", sweep: SweepEvent,
    ) -> Optional[Tuple[FVGZone, int]]:
        """Find an Inverse FVG after a sweep.

        Returns:
            (fvg_zone_now_inverted, candle_index_where_mitigated) or None
        """
        prior_dir = "bearish" if sweep.direction == "bullish" else "bullish"

        fvg_start = max(0, sweep.sweep_idx - self.fvg_lookback)
        prior_fvgs = self.find_fvgs(df, fvg_start, sweep.sweep_idx, prior_dir)
        if not prior_fvgs:
            return None

        fill_end = min(sweep.sweep_idx + self.fill_lookforward, len(df))

        for fvg in reversed(prior_fvgs):
            for j in range(sweep.sweep_idx + 1, fill_end):
                close = df.iloc[j]["close"]

                if prior_dir == "bearish":
                    if close > fvg.top:
                        return (fvg, j)
                else:
                    if close < fvg.bottom:
                        return (fvg, j)

        return None

    # ── Scoring helpers ────────────────────────────────────

    def _score_displacement(
        self, df: "pd.DataFrame", sweep_idx: int, direction: str, n: int = 5,
    ) -> float:
        """Measure reversal quality after a sweep (mean body ratio, 0-1)."""
        ratios = []
        for i in range(sweep_idx + 1, min(sweep_idx + 1 + n, len(df))):
            candle = df.iloc[i]
            bar_range = candle["high"] - candle["low"]
            if bar_range <= 0:
                continue
            body = abs(candle["close"] - candle["open"])
            if direction == "bullish" and candle["close"] > candle["open"]:
                ratios.append(body / bar_range)
            elif direction == "bearish" and candle["close"] < candle["open"]:
                ratios.append(body / bar_range)
        if not ratios:
            return 0.0
        return sum(ratios) / len(ratios)

    def _compute_atr(
        self, df: "pd.DataFrame", idx: int, period: int = 14,
    ) -> float:
        """Compute 14-period Average True Range at a given index."""
        start = max(1, idx - period + 1)
        trs = []
        for i in range(start, idx + 1):
            high = df.iloc[i]["high"]
            low = df.iloc[i]["low"]
            prev_close = df.iloc[i - 1]["close"]
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            trs.append(tr)
        if not trs:
            return 0.0
        return sum(trs) / len(trs)

    def _get_kill_zone_score(self, dt: datetime) -> float:
        """Tiered kill zone score (1.0=NY, 0.85=London, 0.70=Asia, 0.0=outside)."""
        if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
            dt_et = dt.astimezone(ET)
        else:
            dt_et = dt
        t = dt_et.time()

        if time(7, 0) <= t < time(10, 0):
            return 1.0
        if time(2, 0) <= t < time(5, 0):
            return 0.85
        if t >= time(20, 0) or t < time(22, 0) and t >= time(20, 0):
            return 0.70
        return 0.0

    def _compute_quality(
        self, displacement: float, fvg_atr: float, kz: float,
    ) -> float:
        """Composite quality: 40% displacement + 35% FVG/ATR + 25% kill zone."""
        fvg_atr_score = min(1.0, fvg_atr / 1.5)
        return displacement * 0.4 + fvg_atr_score * 0.35 + kz * 0.25
