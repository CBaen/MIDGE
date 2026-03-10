"""FTMO challenge configurations.

Phase 1: $10K account, 10% target, 5% daily loss, 10% max DD, min 4 trading days.
Phase 2: $10K account, 5% target, same loss limits, min 4 trading days.
Funded: Same loss limits, no profit target.

Adapted from sibling instance's work at project_cameron/trading/engine.py.
"""
from dataclasses import dataclass


@dataclass
class FTMOConfig:
    """Configuration for an FTMO challenge simulation."""

    initial_balance: float = 10_000.0
    profit_target_pct: float = 0.10
    max_daily_loss_pct: float = 0.05
    max_total_drawdown_pct: float = 0.10
    min_trading_days: int = 4
    risk_per_trade_pct: float = 0.01
    # For forex: pip_value_per_lot=10.0, pip_size=0.0001 (JPY: 0.01)
    # For equities: pip_value_per_lot=1.0, pip_size=0.01
    pip_value_per_lot: float = 10.0
    pip_size: float = 0.0001


# Pre-built configurations
FTMO_PHASE_1 = FTMOConfig()

FTMO_PHASE_2 = FTMOConfig(profit_target_pct=0.05)

FTMO_FUNDED = FTMOConfig(profit_target_pct=1.0)  # Effectively no target

EQUITY_BACKTEST = FTMOConfig(
    pip_value_per_lot=1.0,
    pip_size=0.01,
    risk_per_trade_pct=0.02,
)
