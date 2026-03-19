"""
strategy_backtester.py - Walk-forward backtester for the strategy layer.

Tests a single strategy function against a single symbol's price history
using a strict walk-forward approach: at every bar the strategy only sees
data up to that point, eliminating lookahead bias.

The backtester delegates all P&L simulation to FTMOBacktester (using the
CRYPTO_BACKTEST config), which handles position sizing, stop-loss/take-profit
exit simulation, and all performance statistics.

Only strategies that clear both gates — win_rate > WIN_RATE_THRESHOLD and
total_trades >= MIN_TRADES — are marked validated=True.  Validated strategies
are the only ones that can contribute votes to PatternConvergenceAlerts.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable, Optional

logger = logging.getLogger("midge.market.strategies")

try:
    import numpy as np
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False
    logger.warning("pandas/numpy not installed — StrategyBacktester will always return None")


class StrategyBacktester:
    """Backtests a single strategy against crypto history using FTMOBacktester.

    Walk-forward protocol
    ---------------------
    For each bar i >= MIN_WINDOW the strategy is called with df.iloc[:i] —
    only the data that was observable at that moment.  The returned
    StrategyResult (or None) provides signal, stop_loss, and take_profit for
    bar i.  Those values are assembled into a signals DataFrame aligned to the
    full OHLCV index, then handed to FTMOBacktester.run() for P&L simulation.

    This guarantees no future data leaks into any signal decision.

    Validation gates
    ----------------
    win_rate > WIN_RATE_THRESHOLD  AND  total_trades >= MIN_TRADES
    Both must be satisfied for validated=True.
    """

    MIN_WINDOW = 50             # Minimum bars before strategy evaluation starts
    MIN_TRADES = 10             # Ignore results with fewer trades
    WIN_RATE_THRESHOLD = 0.25   # Low bar — stacking creates edge, not individual WR

    def __init__(self, ftmo_config=None) -> None:
        from mae_core.market.execution.ftmo_config import CRYPTO_BACKTEST
        self._config = ftmo_config or CRYPTO_BACKTEST

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def backtest(
        self,
        strategy_fn: Callable,
        strategy_name: str,
        symbol: str,
        days: int = 730,
    ) -> Optional[object]:  # returns StrategyBacktestRecord or None
        """Walk-forward backtest of *strategy_fn* against *symbol*.

        Parameters
        ----------
        strategy_fn : Callable
            A strategy function with signature (symbol: str, df: pd.DataFrame)
            -> Optional[StrategyResult].
        strategy_name : str
            Identifier stored in the resulting StrategyBacktestRecord.
        symbol : str
            Yahoo Finance ticker, e.g. "BTC-USD".
        days : int
            Calendar days of OHLCV history to fetch (default 730 = 2 years).

        Returns
        -------
        StrategyBacktestRecord or None
            None when data is unavailable, too short, or fewer than MIN_TRADES
            signals fire across the entire history.
        """
        if not _PANDAS_AVAILABLE:
            logger.warning("backtest(%s, %s): pandas unavailable", strategy_name, symbol)
            return None

        from mae_core.market.strategies.crypto_ohlcv import get_ohlcv
        from mae_core.market.strategies.models import StrategyBacktestRecord
        from mae_core.market.execution.ftmo_engine import FTMOBacktester

        # 1. Fetch OHLCV
        df = get_ohlcv(symbol, days=days)
        if df is None or len(df) < self.MIN_WINDOW + 1:
            logger.debug(
                "backtest(%s, %s): insufficient data (got %s bars)",
                strategy_name, symbol,
                len(df) if df is not None else 0,
            )
            return None

        logger.info(
            "backtest(%s, %s): %d bars — running walk-forward",
            strategy_name, symbol, len(df),
        )

        # 2. Walk-forward signal generation
        signals_df = self._build_signals(strategy_fn, strategy_name, symbol, df)

        signal_count = int((signals_df["signal"] != 0).sum())
        if signal_count == 0:
            logger.debug(
                "backtest(%s, %s): strategy never fired — skipping",
                strategy_name, symbol,
            )
            return None

        logger.debug(
            "backtest(%s, %s): %d signals fired across %d bars",
            strategy_name, symbol, signal_count, len(df),
        )

        # 3. Simulate with FTMOBacktester
        engine = FTMOBacktester(self._config)
        try:
            result = engine.run(data=df, signals=signals_df)
        except Exception as exc:
            logger.warning(
                "backtest(%s, %s): FTMOBacktester.run() failed: %s",
                strategy_name, symbol, exc,
            )
            return None

        # 4. Apply validation gates
        if result.total_trades < self.MIN_TRADES:
            logger.debug(
                "backtest(%s, %s): only %d trades (< MIN_TRADES=%d) — skipping",
                strategy_name, symbol, result.total_trades, self.MIN_TRADES,
            )
            return None

        validated = (
            result.win_rate > self.WIN_RATE_THRESHOLD
            and result.total_trades >= self.MIN_TRADES
        )

        # max_drawdown from FTMOBacktester is in currency units; convert to fraction
        max_dd_fraction = (
            result.max_drawdown / self._config.initial_balance
            if self._config.initial_balance > 0
            else 0.0
        )

        record = StrategyBacktestRecord(
            strategy_name=strategy_name,
            symbol=symbol,
            win_rate=round(result.win_rate, 4),
            sharpe_ratio=round(result.sharpe_ratio, 4),
            profit_factor=round(result.profit_factor, 4),
            max_drawdown=round(max_dd_fraction, 4),
            total_trades=result.total_trades,
            days_tested=days,
            validated=validated,
            backtested_at=datetime.now().isoformat(),
        )

        logger.info(
            "backtest(%s, %s): trades=%d win_rate=%.3f sharpe=%.2f validated=%s",
            strategy_name, symbol,
            result.total_trades, result.win_rate, result.sharpe_ratio, validated,
        )

        return record

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_signals(
        self,
        strategy_fn: Callable,
        strategy_name: str,
        symbol: str,
        df: "pd.DataFrame",
    ) -> "pd.DataFrame":
        """Run the strategy walk-forward and return a signals DataFrame.

        The returned DataFrame has the same DatetimeIndex as *df* with columns:
          signal     : int   — 1 (long), -1 (short), or 0 (flat)
          stop_loss  : float — absolute stop-loss price (NaN when no signal)
          take_profit: float — absolute take-profit price (NaN when no signal)

        Strategy is called with df.iloc[:i] for each bar i >= MIN_WINDOW.
        A new signal is only recorded when no position is currently open
        (to avoid piling signals on top of each other).
        """
        n = len(df)
        signals = pd.DataFrame(
            {
                "signal": np.zeros(n, dtype=int),
                "stop_loss": np.full(n, float("nan")),
                "take_profit": np.full(n, float("nan")),
            },
            index=df.index,
        )

        in_position = False   # naive one-trade-at-a-time tracker

        for i in range(self.MIN_WINDOW, n):
            window = df.iloc[:i]
            try:
                result = strategy_fn(symbol, window)
            except Exception as exc:
                logger.debug(
                    "_build_signals(%s, %s) bar %d: strategy raised %s",
                    strategy_name, symbol, i, exc,
                )
                result = None

            if result is None:
                # No signal at this bar — check if the hypothetical open position
                # would have closed (we track this via a simple next-bar heuristic:
                # once the strategy goes silent for 1 bar, the trade is "done")
                if in_position:
                    in_position = False
                continue

            if in_position:
                # Already in a trade — FTMOBacktester handles the exit; skip re-entry
                continue

            signals.at[df.index[i], "signal"] = result.signal
            signals.at[df.index[i], "stop_loss"] = result.stop_loss
            signals.at[df.index[i], "take_profit"] = result.take_profit
            in_position = True

        return signals
