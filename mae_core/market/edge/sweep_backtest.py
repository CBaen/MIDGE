#!/usr/bin/env python3
"""
sweep_backtest.py - ICT Session Sweep + IFVG Backtest Engine

Backtests: session level sweep -> IFVG entry -> 2R target.
Data: yfinance 5-minute candles (60 days free, no API key needed).

Strategy (Trades by Sci / ICT):
1. Mark session highs/lows as liquidity pools (Asia, London for futures;
   previous-day high/low for equities)
2. Detect sweep: price takes the level then closes back inside
3. Find Inverse FVG (IFVG): a prior-trend FVG that gets filled (mitigated)
   during the post-sweep displacement. Once filled, the zone flips polarity.
4. Enter at IFVG zone, stop at sweep extreme, target at 2R

IFVG explained:
- Down-trend creates bearish FVGs (gap-down zones)
- Price sweeps session low, then reverses upward (displacement)
- The displacement fills prior bearish FVGs from below
- Filled bearish FVG = bullish IFVG (support zone)
- Price pulls back to this zone = long entry
- Mirror logic for high sweeps / bearish IFVGs

Usage:
    python -m mae_core.market.edge.sweep_backtest
    python -m mae_core.market.edge.sweep_backtest --symbols ES=F NQ=F
    python -m mae_core.market.edge.sweep_backtest --futures-only
    python -m mae_core.market.edge.sweep_backtest --equities-only --interval 15m
"""

import logging
import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, date
from pathlib import Path
from typing import List, Optional, Tuple
from zoneinfo import ZoneInfo

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

# ── Symbol lists ──────────────────────────────────────────────

FUTURES = ["ES=F", "NQ=F", "YM=F", "RTY=F", "GC=F", "CL=F"]
EQUITIES = [
    "SPY", "QQQ", "IWM", "DIA",
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META",
    "AMD", "NFLX", "JPM", "V", "MA", "COST", "CRM", "AVGO",
]


# ── Dataclasses ───────────────────────────────────────────────

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


# ── Backtest Engine ───────────────────────────────────────────

class SweepBacktester:
    """ICT Session Sweep + IFVG backtester.

    Pure computation. Fetches data from yfinance, runs the strategy,
    returns Trade results. No side effects.
    """

    def __init__(
        self,
        interval: str = "5m",
        days: int = 59,
        min_fvg_pct: float = 0.005,  # Min FVG size as % of price
        sweep_confirm_candles: int = 3,
        fvg_lookback: int = 100,     # How far back to search for prior FVGs
        fill_lookforward: int = 50,  # How far forward to check for FVG mitigation
        entry_timeout: int = 80,     # Max candles to wait for pullback to IFVG
        trade_timeout: int = 200,    # Max candles for trade to resolve
    ):
        self.interval = interval
        self.days = days
        self.min_fvg_pct = min_fvg_pct
        self.sweep_confirm = sweep_confirm_candles
        self.fvg_lookback = fvg_lookback
        self.fill_lookforward = fill_lookforward
        self.entry_timeout = entry_timeout
        self.trade_timeout = trade_timeout

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
            # Asia: 18:00 previous day → 00:00 ref_date
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

            # London: 02:00-05:00 on ref_date
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
            # Equities: previous trading day's high/low
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

            # Confirm reversal: close back on the other side within N candles
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

            # Bullish FVG: candle 2's low > candle 0's high (gap up)
            if c2["low"] > c0["high"]:
                top = float(c2["low"])
                bottom = float(c0["high"])
                if (top - bottom) / mid_price >= self.min_fvg_pct:
                    if direction is None or direction == "bullish":
                        fvgs.append(FVGZone(
                            top=top, bottom=bottom,
                            direction="bullish", formed_idx=i + 1,
                        ))

            # Bearish FVG: candle 2's high < candle 0's low (gap down)
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

        An IFVG is a prior-trend FVG that gets mitigated (filled through)
        during the post-sweep displacement. Once filled, the zone flips
        polarity and becomes an entry zone.

        For bullish setup (low swept):
          - Prior trend was bearish -> find bearish FVGs before sweep
          - After sweep, check if displacement fills them from below
          - Filled bearish FVG = bullish IFVG (support)

        Returns:
            (fvg_zone_now_inverted, candle_index_where_mitigated) or None
        """
        # Prior FVGs: from the trend that CREATED the swept level
        prior_dir = "bearish" if sweep.direction == "bullish" else "bullish"

        fvg_start = max(0, sweep.sweep_idx - self.fvg_lookback)
        prior_fvgs = self.find_fvgs(df, fvg_start, sweep.sweep_idx, prior_dir)
        if not prior_fvgs:
            return None

        # Check most recent FVGs first (closest to sweep = most relevant)
        fill_end = min(sweep.sweep_idx + self.fill_lookforward, len(df))

        for fvg in reversed(prior_fvgs):
            for j in range(sweep.sweep_idx + 1, fill_end):
                close = df.iloc[j]["close"]

                if prior_dir == "bearish":
                    # Bearish FVG filled when price closes ABOVE its top
                    if close > fvg.top:
                        return (fvg, j)
                else:
                    # Bullish FVG filled when price closes BELOW its bottom
                    if close < fvg.bottom:
                        return (fvg, j)

        return None

    # ── Trade simulation ──────────────────────────────────

    def simulate_trade(
        self, df: "pd.DataFrame", sweep: SweepEvent,
        ifvg: FVGZone, mitigated_idx: int,
    ) -> Optional[Trade]:
        """Simulate a trade from IFVG entry to stop/target resolution.

        1. After IFVG mitigation, wait for price to pull back to the zone
        2. Enter at IFVG midpoint (limit order simulation)
        3. Stop at sweep extreme
        4. Target at 2R from entry
        5. Walk forward: does target or stop hit first?

        Conservative rule: if a candle contains both stop and target,
        assume stop was hit (prevents overstating performance).
        """
        # Wait for pullback to IFVG zone
        entry_end = min(mitigated_idx + self.entry_timeout, len(df))
        entry_idx = None
        entry_price = ifvg.midpoint

        for j in range(mitigated_idx + 1, entry_end):
            candle = df.iloc[j]
            if sweep.direction == "bullish":
                # Long: price pulls back down into IFVG zone
                if candle["low"] <= ifvg.top:
                    entry_idx = j
                    break
            else:
                # Short: price pulls back up into IFVG zone
                if candle["high"] >= ifvg.bottom:
                    entry_idx = j
                    break

        if entry_idx is None:
            return None

        # Stop and targets
        stop = sweep.sweep_extreme
        if sweep.direction == "bullish":
            risk = entry_price - stop
            if risk <= 0:
                return None
            target_1r = entry_price + risk
            target_2r = entry_price + (2.0 * risk)
        else:
            risk = stop - entry_price
            if risk <= 0:
                return None
            target_1r = entry_price - risk
            target_2r = entry_price - (2.0 * risk)

        # Walk forward
        trade_end = min(entry_idx + self.trade_timeout, len(df))
        result = "timeout"
        exit_price = float(df.iloc[min(trade_end - 1, len(df) - 1)]["close"])
        exit_time = str(df.index[min(trade_end - 1, len(df) - 1)])
        r_captured = 0.0
        hit_1r = False

        for j in range(entry_idx + 1, trade_end):
            candle = df.iloc[j]

            if sweep.direction == "bullish":
                # Check stop first (conservative)
                if candle["low"] <= stop:
                    result = "loss"
                    exit_price = stop
                    exit_time = str(df.index[j])
                    r_captured = -1.0
                    break
                if candle["high"] >= target_1r:
                    hit_1r = True
                if candle["high"] >= target_2r:
                    result = "win_2r"
                    exit_price = target_2r
                    exit_time = str(df.index[j])
                    r_captured = 2.0
                    break
            else:
                if candle["high"] >= stop:
                    result = "loss"
                    exit_price = stop
                    exit_time = str(df.index[j])
                    r_captured = -1.0
                    break
                if candle["low"] <= target_1r:
                    hit_1r = True
                if candle["low"] <= target_2r:
                    result = "win_2r"
                    exit_price = target_2r
                    exit_time = str(df.index[j])
                    r_captured = 2.0
                    break

        # If timed out, calculate actual R
        if result == "timeout":
            if sweep.direction == "bullish":
                r_captured = round((exit_price - entry_price) / risk, 2)
            else:
                r_captured = round((entry_price - exit_price) / risk, 2)

        return Trade(
            symbol=sweep.symbol,
            direction=sweep.direction,
            entry_price=round(entry_price, 2),
            stop_price=round(stop, 2),
            target_1r=round(target_1r, 2),
            target_2r=round(target_2r, 2),
            entry_time=str(df.index[entry_idx]),
            exit_time=exit_time,
            exit_price=round(exit_price, 2),
            result=result,
            r_captured=r_captured,
            hit_1r=hit_1r,
            session_swept=sweep.session,
            sweep_level=round(sweep.sweep_level, 2),
            ifvg_top=round(ifvg.top, 2),
            ifvg_bottom=round(ifvg.bottom, 2),
            risk_pts=round(risk, 2),
        )

    # ── Full pipeline ─────────────────────────────────────

    def backtest_symbol(self, symbol: str) -> List[Trade]:
        """Run the full backtest pipeline for one symbol."""
        print(f"  {symbol:8s}", end=" ", flush=True)
        df = self.fetch_candles(symbol)
        if df is None or len(df) < 100:
            print("-- insufficient data")
            return []

        is_futures = symbol.endswith("=F")
        dates = sorted(set(df.index.date))
        trades = []
        sweep_count = 0
        ifvg_count = 0

        for ref_date in dates:
            levels = self.get_session_levels(df, ref_date, is_futures)
            if not levels:
                continue

            # Scan window for sweeps
            if is_futures:
                scan_start = datetime.combine(ref_date, time(7, 0), tzinfo=ET)
                scan_end = datetime.combine(ref_date, time(16, 0), tzinfo=ET)
            else:
                scan_start = datetime.combine(ref_date, time(10, 0), tzinfo=ET)
                scan_end = datetime.combine(ref_date, time(15, 30), tzinfo=ET)

            sweeps = self.detect_sweeps(df, levels, scan_start, scan_end, symbol)
            sweep_count += len(sweeps)

            for sweep in sweeps:
                result = self.find_ifvg(df, sweep)
                if result is None:
                    continue
                ifvg, mitigated_idx = result
                ifvg_count += 1

                trade = self.simulate_trade(df, sweep, ifvg, mitigated_idx)
                if trade:
                    trades.append(trade)

        print(
            f"{len(df):6d} candles | {len(dates):3d} days | "
            f"{sweep_count:3d} sweeps | {ifvg_count:3d} IFVGs | "
            f"{len(trades):3d} trades"
        )
        return trades

    def run(self, symbols: List[str]) -> List[Trade]:
        """Run backtest across all symbols."""
        print(f"\nICT Sweep + IFVG Backtest")
        print(f"Config: {self.interval} candles, {self.days} days lookback")
        print(f"Symbols: {len(symbols)}")
        print("=" * 80)
        print(f"  {'Symbol':8s} {'Candles':>8s} | {'Days':>4s} | "
              f"{'Sweeps':>6s} | {'IFVGs':>5s} | {'Trades':>6s}")
        print("  " + "-" * 70)

        all_trades = []
        for symbol in symbols:
            try:
                trades = self.backtest_symbol(symbol)
                all_trades.extend(trades)
            except Exception as e:
                print(f"  {symbol:8s} ERROR: {e}")

        return all_trades


# ── Reporting ─────────────────────────────────────────────────

def report(trades: List[Trade]) -> str:
    """Generate comprehensive backtest statistics."""
    if not trades:
        return "\nNo trades found. Strategy produced zero setups in this data."

    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("  ICT SESSION SWEEP + IFVG BACKTEST RESULTS")
    lines.append("=" * 70)

    total = len(trades)
    wins = [t for t in trades if t.result == "win_2r"]
    losses = [t for t in trades if t.result == "loss"]
    timeouts = [t for t in trades if t.result == "timeout"]
    hit_1r = [t for t in trades if t.hit_1r]

    win_rate = len(wins) / total * 100
    avg_r = sum(t.r_captured for t in trades) / total

    # Expectancy
    avg_win_r = sum(t.r_captured for t in wins) / len(wins) if wins else 0
    avg_loss_r = (
        abs(sum(t.r_captured for t in losses)) / len(losses)
        if losses else 0
    )
    expectancy = (
        avg_win_r * (len(wins) / total)
        - avg_loss_r * (len(losses) / total)
    )

    # 1R analysis (what if we targeted 1R instead?)
    # Trades that hit 1R = wins + any losses/timeouts that touched 1R
    would_win_1r = len(hit_1r)
    win_rate_1r = would_win_1r / total * 100
    expectancy_1r = (
        1.0 * (would_win_1r / total)
        - 1.0 * ((total - would_win_1r) / total)
    )

    lines.append(f"\n--- Overall (targeting 2R) ---")
    lines.append(f"  Total trades:     {total}")
    lines.append(f"  Wins (2R):        {len(wins)} ({win_rate:.1f}%)")
    lines.append(f"  Losses:           {len(losses)} ({len(losses)/total*100:.1f}%)")
    lines.append(f"  Timeouts:         {len(timeouts)} ({len(timeouts)/total*100:.1f}%)")
    lines.append(f"  Avg R captured:   {avg_r:+.3f}R")
    lines.append(f"  Expectancy:       {expectancy:+.3f}R per trade")

    lines.append(f"\n--- What if targeting 1R instead? ---")
    lines.append(f"  Reached 1R:       {would_win_1r} ({win_rate_1r:.1f}%)")
    lines.append(f"  1R expectancy:    {expectancy_1r:+.3f}R per trade")

    # Profit factor
    gross_profit = sum(t.r_captured for t in trades if t.r_captured > 0)
    gross_loss = abs(sum(t.r_captured for t in trades if t.r_captured < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    lines.append(f"\n--- Risk metrics ---")
    lines.append(f"  Profit factor:    {profit_factor:.2f}")
    lines.append(f"  Gross profit:     {gross_profit:+.1f}R")
    lines.append(f"  Gross loss:       {-gross_loss:.1f}R")
    lines.append(f"  Net R:            {gross_profit - gross_loss:+.1f}R")

    # Max consecutive losses
    max_consec_loss = 0
    current_streak = 0
    for t in trades:
        if t.result == "loss":
            current_streak += 1
            max_consec_loss = max(max_consec_loss, current_streak)
        else:
            current_streak = 0
    lines.append(f"  Max consec. loss: {max_consec_loss}")

    # By direction
    lines.append(f"\n--- By direction ---")
    for dir_name in ("bullish", "bearish"):
        dir_trades = [t for t in trades if t.direction == dir_name]
        if not dir_trades:
            continue
        dir_wins = [t for t in dir_trades if t.result == "win_2r"]
        dir_wr = len(dir_wins) / len(dir_trades) * 100
        dir_avg = sum(t.r_captured for t in dir_trades) / len(dir_trades)
        lines.append(
            f"  {dir_name:10s}: {len(dir_trades):3d} trades, "
            f"{dir_wr:.1f}% WR, {dir_avg:+.3f}R avg"
        )

    # By session swept
    lines.append(f"\n--- By session swept ---")
    sessions = sorted(set(t.session_swept for t in trades))
    for sess in sessions:
        s_trades = [t for t in trades if t.session_swept == sess]
        s_wins = [t for t in s_trades if t.result == "win_2r"]
        s_wr = len(s_wins) / len(s_trades) * 100
        s_avg = sum(t.r_captured for t in s_trades) / len(s_trades)
        lines.append(
            f"  {sess:10s}: {len(s_trades):3d} trades, "
            f"{s_wr:.1f}% WR, {s_avg:+.3f}R avg"
        )

    # By symbol
    lines.append(f"\n--- By symbol ---")
    symbols = sorted(set(t.symbol for t in trades))
    for sym in symbols:
        sym_trades = [t for t in trades if t.symbol == sym]
        sym_wins = [t for t in sym_trades if t.result == "win_2r"]
        sym_wr = len(sym_wins) / len(sym_trades) * 100
        sym_avg = sum(t.r_captured for t in sym_trades) / len(sym_trades)
        sym_net = sum(t.r_captured for t in sym_trades)
        lines.append(
            f"  {sym:8s}: {len(sym_trades):3d} trades, "
            f"{sym_wr:.1f}% WR, {sym_avg:+.3f}R avg, {sym_net:+.1f}R net"
        )

    # Trade log (last 30)
    lines.append(f"\n--- Recent trades (last 30) ---")
    header = (
        f"  {'Symbol':8s} {'Dir':8s} {'Sess':8s} "
        f"{'Entry':>9s} {'Stop':>9s} {'Target':>9s} "
        f"{'Result':8s} {'R':>6s} {'1R?':>4s}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for t in trades[-30:]:
        lines.append(
            f"  {t.symbol:8s} {t.direction:8s} {t.session_swept:8s} "
            f"{t.entry_price:9.2f} {t.stop_price:9.2f} {t.target_2r:9.2f} "
            f"{t.result:8s} {t.r_captured:+5.1f}R "
            f"{'Y' if t.hit_1r else 'N':>3s}"
        )

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ICT Session Sweep + IFVG Backtest",
    )
    parser.add_argument(
        "--symbols", nargs="+", default=None,
        help="Specific symbols to test",
    )
    parser.add_argument(
        "--futures-only", action="store_true",
        help="Only test futures",
    )
    parser.add_argument(
        "--equities-only", action="store_true",
        help="Only test equities",
    )
    parser.add_argument(
        "--interval", default="5m",
        help="Candle interval (default: 5m)",
    )
    parser.add_argument(
        "--days", type=int, default=59,
        help="Days of history (default: 59)",
    )
    args = parser.parse_args()

    if not HAS_DEPS or not HAS_YF:
        print("Missing dependencies: pip install yfinance pandas numpy")
        return

    if args.symbols:
        symbols = args.symbols
    elif args.futures_only:
        symbols = FUTURES
    elif args.equities_only:
        symbols = EQUITIES
    else:
        symbols = FUTURES + EQUITIES

    bt = SweepBacktester(interval=args.interval, days=args.days)
    trades = bt.run(symbols)
    output = report(trades)
    print(output)

    # Save results to JSON
    if trades:
        results_path = Path("data/market/sweep_backtest_results.json")
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results = {
            "run_time": datetime.now().isoformat(),
            "config": {
                "interval": args.interval,
                "days": args.days,
                "symbols": [t.symbol for t in trades],
            },
            "summary": {
                "total_trades": len(trades),
                "wins": len([t for t in trades if t.result == "win_2r"]),
                "losses": len([t for t in trades if t.result == "loss"]),
                "win_rate": (
                    len([t for t in trades if t.result == "win_2r"])
                    / len(trades) * 100
                ),
                "avg_r": sum(t.r_captured for t in trades) / len(trades),
                "hit_1r_count": len([t for t in trades if t.hit_1r]),
            },
            "trades": [
                {
                    "symbol": t.symbol,
                    "direction": t.direction,
                    "session_swept": t.session_swept,
                    "entry_price": t.entry_price,
                    "stop_price": t.stop_price,
                    "target_2r": t.target_2r,
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                    "exit_price": t.exit_price,
                    "result": t.result,
                    "r_captured": t.r_captured,
                    "hit_1r": t.hit_1r,
                    "risk_pts": t.risk_pts,
                }
                for t in trades
            ],
        }
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
