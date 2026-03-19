"""
models.py - Data models for the Crypto Pattern Convergence strategy layer.

Three types:
  StrategyResult           — a single strategy's live signal on a symbol
  StrategyBacktestRecord   — backtest metrics that determine whether to trust a strategy
  PatternConvergenceAlert  — multiple validated strategies agreeing on the same move
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger("midge.market.strategies")


@dataclass
class StrategyResult:
    """Output of a single strategy evaluation.

    Fields
    ------
    strategy_name : str
        Identifier matching the function name, e.g. "rsi_oversold_14".
    symbol : str
        Ticker symbol this result applies to, e.g. "BTC-USD".
    direction : str
        "bullish" or "bearish".
    signal : int
        1 for long, -1 for short, 0 for flat.
    stop_loss : float
        ATR-based stop-loss price level (absolute, not distance).
    take_profit : float
        ATR-based take-profit price level (absolute, not distance).
    strength : float
        0.0–1.0 raw signal strength from the strategy's own math
        (e.g., how far RSI is from the threshold).
    confidence : float
        0.0–1.0 posterior confidence. Defaults to 0.55; overwritten by
        the backtester once a win-rate is established.
    detected_at : str
        ISO-8601 timestamp of detection (datetime.now().isoformat()).
    metadata : dict
        Strategy-specific diagnostic values for debugging and
        the post-mortem reviewer.
    """

    strategy_name: str
    symbol: str
    direction: str
    signal: int
    stop_loss: float
    take_profit: float
    strength: float
    confidence: float
    detected_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyBacktestRecord:
    """Historical performance record for a strategy on a specific symbol.

    validated is True when the strategy has demonstrated statistical reliability:
    win_rate > WIN_RATE_THRESHOLD (0.55) AND total_trades >= MIN_TRADES (10).
    Only validated strategies contribute to PatternConvergenceAlerts.
    """

    strategy_name: str
    symbol: str
    win_rate: float          # fraction of profitable trades 0.0–1.0
    sharpe_ratio: float      # risk-adjusted return
    profit_factor: float     # gross profit / gross loss
    max_drawdown: float      # peak-to-trough decline as a fraction (e.g. 0.15 = 15%)
    total_trades: int        # number of trades in the test window
    days_tested: int         # calendar days covered by the backtest
    validated: bool          # win_rate > 0.55 and total_trades >= 10
    backtested_at: str       # ISO-8601 timestamp of when the backtest ran


@dataclass
class PatternConvergenceAlert:
    """Multiple validated strategies independently agreeing on the same trade.

    This is the strategy-layer equivalent of ConvergenceAlerter: instead of
    cross-domain data signals converging, independent algorithmic strategies
    converge on the same ticker, direction, and timing.

    confidence and strength are aggregates across contributing strategies.
    stop_loss uses the widest (most conservative) level across all strategies;
    take_profit uses the nearest (most conservative) level — ensuring any single
    strategy's risk parameters are satisfied by the consensus trade.
    """

    alert_id: str
    symbol: str
    direction: str           # "bullish" | "bearish"
    strategy_names: List[str]  # which validated strategies agreed
    confidence: float        # 0.0–1.0 aggregate
    strength: float          # 0.0–1.0 aggregate
    stop_loss: float         # absolute price (widest / most conservative)
    take_profit: float       # absolute price (nearest / most conservative)
    timestamp: str           # ISO-8601
    metadata: Dict[str, Any] = field(default_factory=dict)
