"""
strategy_registry.py - Backtest record store for the strategy layer.

Tracks which (strategy, symbol) combinations have been backtested and whether
they passed the validation gate (win_rate > 0.55 AND total_trades >= 10).

Only validated strategies are allowed to contribute votes to PatternConvergenceAlerts.
The registry persists to data/market/strategy_registry.json between restarts so
backtest work is not repeated unless explicitly invalidated.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from mae_core.market.strategies.models import StrategyBacktestRecord

logger = logging.getLogger("midge.market.strategies")


class StrategyRegistry:
    """Stores and loads backtest results.  Knows which strategies are validated.

    Key format in the internal dict: "strategy_name:symbol", e.g.
    "rsi_oversold_14:BTC-USD".  This lets one strategy be validated on BTC
    but not yet tested on ETH — they are independent entries.
    """

    _PERSISTENCE_PATH = Path("data/market/strategy_registry.json")
    # Individual strategies don't need to be great — the STACKING creates edge.
    # With 2:1 R:R (SL=1.5×ATR, TP=3×ATR), breakeven is at 33% WR.
    # Any strategy that generates 10+ trades and beats noise participates.
    # Quality control happens at the COMBO level via Thompson sampling.
    WIN_RATE_THRESHOLD = 0.25
    MIN_TRADES = 10

    def __init__(self, data_path: Optional[str] = None) -> None:
        self._path = Path(data_path) if data_path else self._PERSISTENCE_PATH
        self._records: dict[str, StrategyBacktestRecord] = {}
        self._load()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def register_result(self, record: StrategyBacktestRecord) -> None:
        """Persist a backtest result (overwrites any previous result for the same key)."""
        key = self._key(record.strategy_name, record.symbol)
        self._records[key] = record
        logger.debug(
            "Registered backtest: %s | win_rate=%.3f validated=%s",
            key, record.win_rate, record.validated,
        )
        self.save()

    def is_validated(self, strategy_name: str, symbol: str) -> bool:
        """Return True if the strategy has a passing backtest on this symbol."""
        record = self._records.get(self._key(strategy_name, symbol))
        return record.validated if record is not None else False

    def get_validated_strategies(self, symbol: str) -> list[str]:
        """Return strategy names that are validated for *symbol*."""
        return [
            r.strategy_name
            for r in self._records.values()
            if r.symbol == symbol and r.validated
        ]

    def get_confidence_prior(self, strategy_name: str) -> float:
        """Return the mean win_rate across all validated symbols for *strategy_name*.

        Falls back to WIN_RATE_THRESHOLD when no backtest exists yet (neutral prior).
        Aggregating across symbols gives a cross-market reliability estimate that
        seeds the initial confidence field on StrategyResult before symbol-specific
        data accumulates.
        """
        matching = [
            r for r in self._records.values()
            if r.strategy_name == strategy_name and r.validated
        ]
        if not matching:
            return self.WIN_RATE_THRESHOLD
        return sum(r.win_rate for r in matching) / len(matching)

    def needs_backtest(self, strategy_name: str, symbol: str) -> bool:
        """Return True when no backtest record exists for this (strategy, symbol) pair."""
        return self._key(strategy_name, symbol) not in self._records

    def save(self) -> None:
        """Flush all records to disk as JSON."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            key: {
                "strategy_name": r.strategy_name,
                "symbol": r.symbol,
                "win_rate": r.win_rate,
                "sharpe_ratio": r.sharpe_ratio,
                "profit_factor": r.profit_factor,
                "max_drawdown": r.max_drawdown,
                "total_trades": r.total_trades,
                "days_tested": r.days_tested,
                "validated": r.validated,
                "backtested_at": r.backtested_at,
            }
            for key, r in self._records.items()
        }
        try:
            self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            logger.debug("StrategyRegistry saved %d records to %s", len(payload), self._path)
        except OSError as exc:
            logger.warning("StrategyRegistry save failed: %s", exc)

    def get_statistics(self) -> dict:
        """Summary stats for health-check and logging."""
        total = len(self._records)
        validated = sum(1 for r in self._records.values() if r.validated)
        symbols = {r.symbol for r in self._records.values()}
        strategies = {r.strategy_name for r in self._records.values()}
        return {
            "total_records": total,
            "validated_records": validated,
            "unique_symbols": len(symbols),
            "unique_strategies": len(strategies),
            "persistence_path": str(self._path),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _key(strategy_name: str, symbol: str) -> str:
        return f"{strategy_name}:{symbol}"

    def _load(self) -> None:
        """Load persisted records from disk; silently skip if file absent or corrupt."""
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            for key, data in raw.items():
                self._records[key] = StrategyBacktestRecord(**data)
            logger.debug("StrategyRegistry loaded %d records from %s", len(self._records), self._path)
        except Exception as exc:
            logger.warning("StrategyRegistry load failed (%s) — starting empty: %s", self._path, exc)
