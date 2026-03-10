"""FTMO challenge backtester adapted for MIDGE.

Simulates trading against historical data while enforcing prop firm constraints:
- Max daily loss (5% of initial balance)
- Max total drawdown (10% of initial balance)
- Profit target (10% Phase 1, 5% Phase 2)
- Minimum trading days

Originally built by a sibling instance at project_cameron/trading/engine.py.
Ported into MIDGE's execution package with multi-asset support and
ConvergenceAlert integration.

Usage:
    from mae_core.market.execution.ftmo_engine import FTMOBacktester
    from mae_core.market.execution.ftmo_config import FTMO_PHASE_1, EQUITY_BACKTEST

    engine = FTMOBacktester(EQUITY_BACKTEST)
    result = engine.run(ohlcv_df, signals_df)
    print(f"Passed: {result.passed_challenge}, Return: {result.total_return_pct:.1f}%")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

from mae_core.market.execution.ftmo_config import FTMOConfig

logger = logging.getLogger(__name__)


class Direction(Enum):
    LONG = 1
    SHORT = -1


@dataclass
class Trade:
    """A single completed or open trade."""
    entry_date: pd.Timestamp
    direction: Direction
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    ticker: str = ""
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    exit_reason: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Full backtest output with challenge pass/fail determination."""
    trades: List[Trade]
    equity_curve: List[Dict]
    final_balance: float
    max_drawdown: float
    max_daily_loss: float
    total_return_pct: float
    win_rate: float
    profit_factor: float
    total_trades: int
    trading_days: int
    passed_challenge: bool
    fail_reason: Optional[str] = None
    sharpe_ratio: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0

    def summary(self) -> str:
        """One-line summary for logging."""
        status = "PASS" if self.passed_challenge else f"FAIL ({self.fail_reason})"
        return (
            f"{status} | {self.total_trades} trades | "
            f"WR={self.win_rate:.1%} | PF={self.profit_factor:.2f} | "
            f"Return={self.total_return_pct:.1f}% | "
            f"MaxDD=${self.max_drawdown:.2f} | Sharpe={self.sharpe_ratio:.2f}"
        )


class FTMOBacktester:
    """Simulate an FTMO challenge with MIDGE signals.

    Enforces all prop firm constraints: daily loss, total drawdown, profit
    target, minimum trading days. Candle-exit simulation uses conservative
    open-relative heuristic when both SL and TP could trigger in the same bar.
    """

    def __init__(self, config: Optional[FTMOConfig] = None):
        self.config = config or FTMOConfig()

    def calculate_position_size(self, balance: float, stop_loss_pips: float) -> float:
        """Position size based on risk percentage and stop-loss distance."""
        if stop_loss_pips <= 0:
            return 0.0
        risk_amount = balance * self.config.risk_per_trade_pct
        lots = risk_amount / (stop_loss_pips * self.config.pip_value_per_lot)
        return round(lots, 2)

    def simulate_candle_exit(self, trade: Trade, candle) -> tuple:
        """Check if SL or TP is hit within a candle.

        When both could trigger in the same bar, uses an open-relative
        heuristic: whichever level is closer to the open price is assumed
        to have been hit first. Conservative for backtesting accuracy.

        Returns (exit_price, exit_reason) or (None, None).
        """
        high = candle["High"]
        low = candle["Low"]
        open_price = candle["Open"]

        if trade.direction == Direction.LONG:
            sl_hit = low <= trade.stop_loss
            tp_hit = high >= trade.take_profit

            if sl_hit and tp_hit:
                if open_price <= trade.stop_loss:
                    return trade.stop_loss, "stop_loss"
                dist_to_sl = open_price - trade.stop_loss
                dist_to_tp = trade.take_profit - open_price
                if dist_to_sl < dist_to_tp:
                    return trade.stop_loss, "stop_loss"
                return trade.take_profit, "take_profit"
            elif sl_hit:
                return trade.stop_loss, "stop_loss"
            elif tp_hit:
                return trade.take_profit, "take_profit"

        else:  # SHORT
            sl_hit = high >= trade.stop_loss
            tp_hit = low <= trade.take_profit

            if sl_hit and tp_hit:
                if open_price >= trade.stop_loss:
                    return trade.stop_loss, "stop_loss"
                dist_to_sl = trade.stop_loss - open_price
                dist_to_tp = open_price - trade.take_profit
                if dist_to_sl < dist_to_tp:
                    return trade.stop_loss, "stop_loss"
                return trade.take_profit, "take_profit"
            elif sl_hit:
                return trade.stop_loss, "stop_loss"
            elif tp_hit:
                return trade.take_profit, "take_profit"

        return None, None

    def run(self, data: pd.DataFrame, signals: pd.DataFrame) -> BacktestResult:
        """Run backtest with FTMO constraints.

        Args:
            data: OHLCV DataFrame with DatetimeIndex. Must have columns:
                  Open, High, Low, Close, Volume.
            signals: DataFrame aligned to same index with columns:
                  signal (1=long, -1=short, 0=flat),
                  stop_loss (absolute price),
                  take_profit (absolute price).

        Returns:
            BacktestResult with full trade list, equity curve, and pass/fail.
        """
        cfg = self.config
        balance = cfg.initial_balance
        high_water_mark = balance
        equity_curve: List[Dict] = []
        trades: List[Trade] = []
        open_trade: Optional[Trade] = None
        trading_days = set()
        daily_start_balance = balance
        current_day = None
        max_daily_loss = 0.0
        max_drawdown = 0.0
        challenge_failed = False
        fail_reason = None

        profit_target = cfg.initial_balance * cfg.profit_target_pct
        max_daily_loss_limit = cfg.initial_balance * cfg.max_daily_loss_pct
        max_drawdown_limit = cfg.initial_balance * cfg.max_total_drawdown_pct

        for i in range(len(data)):
            row = data.iloc[i]
            date = data.index[i]
            day = date.date() if hasattr(date, "date") else date

            # Track daily boundaries
            if day != current_day:
                current_day = day
                daily_start_balance = balance

            # Check for exit on open trade
            if open_trade is not None:
                exit_price, exit_reason = self.simulate_candle_exit(open_trade, row)

                if exit_price is not None:
                    if open_trade.direction == Direction.LONG:
                        pips = (exit_price - open_trade.entry_price) / cfg.pip_size
                    else:
                        pips = (open_trade.entry_price - exit_price) / cfg.pip_size

                    pnl = pips * cfg.pip_value_per_lot * open_trade.position_size
                    open_trade.exit_date = date
                    open_trade.exit_price = exit_price
                    open_trade.pnl = pnl
                    open_trade.exit_reason = exit_reason
                    trades.append(open_trade)
                    balance += pnl
                    open_trade = None

            # Check FTMO constraints
            daily_loss = daily_start_balance - balance
            drawdown = high_water_mark - balance

            if daily_loss > max_daily_loss:
                max_daily_loss = daily_loss
            if drawdown > max_drawdown:
                max_drawdown = drawdown

            if daily_loss >= max_daily_loss_limit:
                challenge_failed = True
                fail_reason = (
                    f"Daily loss limit: ${daily_loss:.2f} >= ${max_daily_loss_limit:.2f}"
                )
                break

            if drawdown >= max_drawdown_limit:
                challenge_failed = True
                fail_reason = (
                    f"Max drawdown: ${drawdown:.2f} >= ${max_drawdown_limit:.2f}"
                )
                break

            # Check profit target
            if balance - cfg.initial_balance >= profit_target:
                break

            # Update high water mark
            if balance > high_water_mark:
                high_water_mark = balance

            # New entry signal (only if no open trade)
            if open_trade is None and i < len(signals):
                sig = signals.iloc[i]
                signal_val = sig.get("signal", 0)

                if signal_val != 0 and not pd.isna(sig.get("stop_loss", np.nan)):
                    direction = Direction.LONG if signal_val == 1 else Direction.SHORT
                    entry_price = row["Close"]
                    sl = sig["stop_loss"]
                    tp = sig["take_profit"]

                    sl_pips = abs(entry_price - sl) / cfg.pip_size
                    if sl_pips > 0:
                        pos_size = self.calculate_position_size(balance, sl_pips)
                        if pos_size > 0:
                            open_trade = Trade(
                                entry_date=date,
                                direction=direction,
                                entry_price=entry_price,
                                stop_loss=sl,
                                take_profit=tp,
                                position_size=pos_size,
                                ticker=sig.get("ticker", ""),
                            )
                            trading_days.add(day)

            equity_curve.append(
                {"date": date, "balance": balance, "drawdown": high_water_mark - balance}
            )

        # Close remaining open trade at last close
        if open_trade is not None:
            last_row = data.iloc[-1]
            last_date = data.index[-1]
            exit_price = last_row["Close"]

            if open_trade.direction == Direction.LONG:
                pips = (exit_price - open_trade.entry_price) / cfg.pip_size
            else:
                pips = (open_trade.entry_price - exit_price) / cfg.pip_size

            pnl = pips * cfg.pip_value_per_lot * open_trade.position_size
            open_trade.exit_date = last_date
            open_trade.exit_price = exit_price
            open_trade.pnl = pnl
            open_trade.exit_reason = "end_of_data"
            trades.append(open_trade)
            balance += pnl

        # Statistics
        total_trades = len(trades)
        if total_trades > 0:
            wins = [t for t in trades if t.pnl and t.pnl > 0]
            losses = [t for t in trades if t.pnl and t.pnl <= 0]
            win_rate = len(wins) / total_trades
            gross_profit = sum(t.pnl for t in wins) if wins else 0
            gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
            avg_win = gross_profit / len(wins) if wins else 0
            avg_loss = gross_loss / len(losses) if losses else 0

            daily_returns = (
                pd.Series([e["balance"] for e in equity_curve]).pct_change().dropna()
            )
            if len(daily_returns) > 1 and daily_returns.std() > 0:
                sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
            else:
                sharpe = 0.0
        else:
            win_rate = profit_factor = avg_win = avg_loss = sharpe = 0.0

        total_return_pct = (
            (balance - cfg.initial_balance) / cfg.initial_balance
        ) * 100

        passed = (
            not challenge_failed
            and total_return_pct >= cfg.profit_target_pct * 100
            and len(trading_days) >= cfg.min_trading_days
        )

        if not passed and not challenge_failed:
            if total_return_pct < cfg.profit_target_pct * 100:
                fail_reason = (
                    f"Profit target not reached: {total_return_pct:.1f}% "
                    f"< {cfg.profit_target_pct * 100:.0f}%"
                )
            elif len(trading_days) < cfg.min_trading_days:
                fail_reason = (
                    f"Not enough trading days: {len(trading_days)} "
                    f"< {cfg.min_trading_days}"
                )

        result = BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            final_balance=balance,
            max_drawdown=max_drawdown,
            max_daily_loss=max_daily_loss,
            total_return_pct=total_return_pct,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            trading_days=len(trading_days),
            passed_challenge=passed,
            fail_reason=fail_reason,
            sharpe_ratio=sharpe,
            avg_win=avg_win,
            avg_loss=avg_loss,
        )

        logger.info("FTMO backtest: %s", result.summary())
        return result
