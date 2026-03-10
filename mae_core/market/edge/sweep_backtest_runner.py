"""
sweep_backtest_runner.py - Trade simulation and backtest runner for ICT Sweep + IFVG

simulate_trade, backtest_symbol, and run — the execution layer that
orchestrates the engine and collects Trade results.
"""

import logging
from datetime import datetime, time
from typing import List, Optional
from zoneinfo import ZoneInfo

from mae_core.market.edge.sweep_backtest_models import FVGZone, SweepEvent, Trade
from mae_core.market.edge.sweep_backtest_engine import SweepBacktestEngine

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")


class SweepBacktester(SweepBacktestEngine):
    """ICT Session Sweep + IFVG backtester.

    Extends SweepBacktestEngine with trade simulation and full pipeline.
    """

    # ── Trade simulation ──────────────────────────────────

    def simulate_trade(
        self, df, sweep: SweepEvent,
        ifvg: FVGZone, mitigated_idx: int,
    ) -> Optional[Trade]:
        """Simulate a trade from IFVG entry to stop/target resolution.

        Conservative rule: if a candle contains both stop and target,
        assume stop was hit.
        """
        entry_end = min(mitigated_idx + self.entry_timeout, len(df))
        entry_idx = None
        entry_price = ifvg.midpoint

        for j in range(mitigated_idx + 1, entry_end):
            candle = df.iloc[j]
            if sweep.direction == "bullish":
                if candle["low"] <= ifvg.top:
                    entry_idx = j
                    break
            else:
                if candle["high"] >= ifvg.bottom:
                    entry_idx = j
                    break

        if entry_idx is None:
            return None

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

        trade_end = min(entry_idx + self.trade_timeout, len(df))
        result = "timeout"
        exit_price = float(df.iloc[min(trade_end - 1, len(df) - 1)]["close"])
        exit_time = str(df.index[min(trade_end - 1, len(df) - 1)])
        r_captured = 0.0
        hit_1r = False

        for j in range(entry_idx + 1, trade_end):
            candle = df.iloc[j]

            if sweep.direction == "bullish":
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
        filtered_count = 0
        sweep_count = 0
        ifvg_count = 0

        for ref_date in dates:
            levels = self.get_session_levels(df, ref_date, is_futures)
            if not levels:
                continue

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

                disp = self._score_displacement(df, sweep.sweep_idx, sweep.direction)
                atr = self._compute_atr(df, sweep.sweep_idx)
                fvg_atr = (ifvg.top - ifvg.bottom) / atr if atr > 0 else 0.0
                kz = self._get_kill_zone_score(df.index[sweep.sweep_idx])
                quality = self._compute_quality(disp, fvg_atr, kz)

                if self.min_quality > 0 and quality < self.min_quality:
                    filtered_count += 1
                    continue

                trade = self.simulate_trade(df, sweep, ifvg, mitigated_idx)
                if trade:
                    trade.displacement_score = round(disp, 3)
                    trade.fvg_atr_ratio = round(fvg_atr, 3)
                    trade.kill_zone_score = round(kz, 2)
                    trade.quality_score = round(quality, 3)
                    trades.append(trade)

        filt_str = f" | {filtered_count:3d} filtered" if filtered_count else ""
        print(
            f"{len(df):6d} candles | {len(dates):3d} days | "
            f"{sweep_count:3d} sweeps | {ifvg_count:3d} IFVGs | "
            f"{len(trades):3d} trades{filt_str}"
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
