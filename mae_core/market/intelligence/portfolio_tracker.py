#!/usr/bin/env python3
"""
portfolio_tracker.py - MIDGE Paper Position Tracker

Reads paper_trades.jsonl (written by market_hooks.py) and maintains a live
view of open paper positions: current mark-to-market value, unrealized P&L,
and exit signals when stop-loss / take-profit / time-decay thresholds are hit.

This does NOT write trades — market_hooks.py owns that path.
This is a read-only consumer of paper_trades.jsonl, plus a mark-to-market
updater using price_fetcher.

Exit rules (conservative defaults, can be tuned):
  - Stop loss  : position falls -5% from entry
  - Take profit: position gains +15% from entry
  - Time decay : position held > 30 calendar days

All external calls (price_fetcher) are wrapped in try/except — if live
pricing is unavailable, positions are still tracked but P&L shows as 0.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Default thresholds ─────────────────────────────────────────────────────────

_DEFAULT_STOP_LOSS_PCT = -5.0
_DEFAULT_TAKE_PROFIT_PCT = 15.0
_DEFAULT_MAX_HOLD_DAYS = 30
_DEFAULT_MAX_POSITIONS = 10
_DEFAULT_MAX_SINGLE_POSITION_PCT = 20.0  # % of portfolio value


# ── Dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class Position:
    """A live paper position, enriched with current mark-to-market data."""
    ticker: str
    direction: str          # "buy" or "sell"
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    size_usd: float = 0.0
    kelly_fraction: float = 0.0

    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized P&L as a percentage of entry price."""
        if self.entry_price <= 0:
            return 0.0
        raw = (self.current_price - self.entry_price) / self.entry_price * 100.0
        return raw if self.direction == "buy" else -raw

    @property
    def age_days(self) -> float:
        """Calendar days since entry."""
        return (datetime.now() - self.entry_time).total_seconds() / 86400.0


@dataclass
class ClosedTrade:
    """A position that has been exited, with full P&L accounting."""
    ticker: str
    direction: str
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    realized_pnl: float
    reason: str             # "stop_loss", "take_profit", "time_decay"


@dataclass
class PortfolioState:
    """Snapshot of the entire paper portfolio at a point in time."""
    total_value: float          # paper_account_value from learning_config
    unrealized_pnl: float
    position_count: int
    largest_position_pct: float
    net_exposure: float         # +1.0 = all long, -1.0 = all short


@dataclass
class ExitSignal:
    """An actionable exit recommendation for an open position."""
    ticker: str
    direction: str          # opposite of position direction — the exit trade direction
    reason: str             # "stop_loss", "take_profit", "time_decay"
    urgency: str            # "immediate", "hours", "days"
    strength: float         # 0-1
    current_pnl_pct: float


# ── PortfolioTracker ───────────────────────────────────────────────────────────

class PortfolioTracker:
    """
    Reads paper_trades.jsonl and tracks open paper positions.

    Usage:
        tracker = PortfolioTracker(price_fetcher=fetcher)
        tracker.load_positions()
        tracker.update_prices()
        exits = tracker.check_exits()
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        price_fetcher=None,
    ) -> None:
        """
        Args:
            data_dir: Directory containing paper_trades.jsonl.
                      Defaults to <project_root>/data/midge.
            price_fetcher: Optional PriceFetcher instance for mark-to-market.
                           If None, unrealized P&L will be zero for all positions.
        """
        if data_dir is None:
            data_dir = Path(__file__).resolve().parents[3] / "data" / "midge"
        self._data_dir = Path(data_dir)
        self._paper_trades_path = self._data_dir / "paper_trades.jsonl"
        self._price_fetcher = price_fetcher

        self._positions: Dict[str, Position] = {}
        self._closed: List[ClosedTrade] = []

        # Exit thresholds
        self._stop_loss_pct = _DEFAULT_STOP_LOSS_PCT
        self._take_profit_pct = _DEFAULT_TAKE_PROFIT_PCT
        self._max_hold_days = _DEFAULT_MAX_HOLD_DAYS
        self._max_positions = _DEFAULT_MAX_POSITIONS
        self._max_single_position_pct = _DEFAULT_MAX_SINGLE_POSITION_PCT

    # ── Position loading ───────────────────────────────────────────────────────

    def load_positions(self) -> int:
        """
        Read paper_trades.jsonl and (re)build the position map.

        Dedup rule: last entry per ticker wins (most recent trade replaces
        an older one for the same ticker). Entry price is fetched from
        price_fetcher at the time the position was written; if unavailable,
        falls back to zero so the position is still tracked.

        Returns:
            Number of open positions loaded.
        """
        if not self._paper_trades_path.exists():
            logger.debug("paper_trades.jsonl not found at %s", self._paper_trades_path)
            return 0

        raw: Dict[str, dict] = {}
        try:
            with open(self._paper_trades_path, encoding="utf-8") as fh:
                for line_no, line in enumerate(fh, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        logger.debug("Malformed JSON on line %d, skipping", line_no)
                        continue
                    ticker = record.get("ticker", "")
                    if not ticker:
                        continue
                    raw[ticker] = record  # last entry wins
        except OSError:
            logger.warning("Could not read paper_trades.jsonl", exc_info=True)
            return 0

        self._positions.clear()
        for ticker, record in raw.items():
            position = self._record_to_position(ticker, record)
            if position is not None:
                self._positions[ticker] = position

        logger.debug("Loaded %d paper positions from %s", len(self._positions), self._paper_trades_path)
        return len(self._positions)

    def _record_to_position(self, ticker: str, record: dict) -> Optional[Position]:
        """Parse a single paper_trades.jsonl record into a Position."""
        try:
            direction = record.get("direction", "")
            if direction not in ("buy", "sell"):
                return None

            ts_raw = record.get("timestamp", "")
            try:
                entry_time = datetime.fromisoformat(ts_raw)
            except (ValueError, TypeError):
                entry_time = datetime.now()

            entry_price = self._fetch_entry_price(ticker, record, ts_raw)

            return Position(
                ticker=ticker,
                direction=direction,
                entry_price=entry_price,
                entry_time=entry_time,
                current_price=entry_price,
                unrealized_pnl=0.0,
                size_usd=float(record.get("position_size_usd", 0.0)),
                kelly_fraction=float(record.get("kelly_fraction", 0.0)),
            )
        except Exception:
            logger.debug("Failed to parse paper trade record for %s", ticker, exc_info=True)
            return None

    def _fetch_entry_price(self, ticker: str, record: dict, ts_raw: str) -> float:
        """
        Attempt to resolve the entry price.

        Priority:
          1. record["entry_price"] if present
          2. price_fetcher.get_current_price() (approximation)
          3. 0.0 (unknown — position tracked but P&L will be zero)
        """
        if "entry_price" in record:
            return float(record["entry_price"])

        if self._price_fetcher is not None:
            try:
                price_data = self._price_fetcher.get_current_price(ticker)
                if price_data is not None:
                    return float(price_data.price)
            except Exception:
                logger.debug("Price fetch failed for entry price of %s", ticker, exc_info=True)

        return 0.0

    # ── Mark-to-market ─────────────────────────────────────────────────────────

    def update_prices(self) -> None:
        """
        Refresh current_price and unrealized_pnl for all open positions.

        If price_fetcher is None or a fetch fails, the existing current_price
        and unrealized_pnl are left unchanged (they default to 0 on first load).
        """
        if self._price_fetcher is None:
            logger.debug("No price_fetcher — skipping mark-to-market update")
            return

        for ticker, position in self._positions.items():
            try:
                price_data = self._price_fetcher.get_current_price(ticker)
                if price_data is None:
                    continue
                current = float(price_data.price)
                position.current_price = current
                if position.entry_price > 0:
                    raw_pnl_pct = (current - position.entry_price) / position.entry_price * 100.0
                    sign = 1.0 if position.direction == "buy" else -1.0
                    position.unrealized_pnl = (
                        sign * raw_pnl_pct / 100.0 * position.size_usd
                    )
            except Exception:
                logger.debug("Failed to update price for %s", ticker, exc_info=True)

    # ── Accessors ──────────────────────────────────────────────────────────────

    def get_position(self, ticker: str) -> Optional[Position]:
        """Return the open Position for ticker, or None if not held."""
        return self._positions.get(ticker)

    def has_position(self, ticker: str) -> bool:
        """True if there is an open paper position for this ticker."""
        return ticker in self._positions

    def is_max_exposed(self, ticker: str) -> bool:
        """
        True if adding a new position for ticker would exceed risk limits.

        Limits checked:
          - Total position count >= _max_positions
          - Existing position for ticker already at or above _max_single_position_pct
        """
        if len(self._positions) >= self._max_positions:
            return True

        if ticker in self._positions:
            position = self._positions[ticker]
            portfolio_value = self._get_paper_account_value()
            if portfolio_value > 0:
                pct = (position.size_usd / portfolio_value) * 100.0
                if pct >= self._max_single_position_pct:
                    return True

        return False

    # ── Portfolio state ────────────────────────────────────────────────────────

    def get_portfolio_state(self) -> PortfolioState:
        """
        Compute a snapshot of the full portfolio.

        total_value comes from learning_config.paper_account_value.
        net_exposure is the directional skew: (long_usd - short_usd) / total_value.
        """
        total_value = self._get_paper_account_value()
        unrealized_pnl = sum(p.unrealized_pnl for p in self._positions.values())

        long_usd = sum(p.size_usd for p in self._positions.values() if p.direction == "buy")
        short_usd = sum(p.size_usd for p in self._positions.values() if p.direction == "sell")

        net_exposure = 0.0
        if total_value > 0:
            net_exposure = (long_usd - short_usd) / total_value

        largest_pct = 0.0
        if self._positions and total_value > 0:
            largest_usd = max(p.size_usd for p in self._positions.values())
            largest_pct = (largest_usd / total_value) * 100.0

        return PortfolioState(
            total_value=total_value,
            unrealized_pnl=unrealized_pnl,
            position_count=len(self._positions),
            largest_position_pct=largest_pct,
            net_exposure=net_exposure,
        )

    def _get_paper_account_value(self) -> float:
        """Read paper_account_value from learning_config. Falls back to 50000."""
        try:
            from mae_core.market.intelligence.learning_config import LEARNING_CONFIG
            return float(LEARNING_CONFIG.get("paper_account_value", 50000))
        except Exception:
            return 50000.0

    # ── Exit signal generation ─────────────────────────────────────────────────

    def check_exits(self) -> List[ExitSignal]:
        """
        Scan all open positions and return ExitSignals for those that breach
        stop-loss, take-profit, or time-decay thresholds.

        Positions within all bounds return no signal.
        Multiple positions may produce signals simultaneously.
        """
        signals: List[ExitSignal] = []

        for ticker, position in self._positions.items():
            pnl_pct = position.unrealized_pnl_pct

            # ── Stop loss ──────────────────────────────────────────────────────
            if pnl_pct <= self._stop_loss_pct:
                exit_direction = "sell" if position.direction == "buy" else "buy"
                signals.append(ExitSignal(
                    ticker=ticker,
                    direction=exit_direction,
                    reason="stop_loss",
                    urgency="immediate",
                    strength=min(1.0, abs(pnl_pct) / abs(self._stop_loss_pct) * 0.8 + 0.2),
                    current_pnl_pct=pnl_pct,
                ))
                continue  # Only one signal per position per check

            # ── Take profit ────────────────────────────────────────────────────
            if pnl_pct >= self._take_profit_pct:
                exit_direction = "sell" if position.direction == "buy" else "buy"
                signals.append(ExitSignal(
                    ticker=ticker,
                    direction=exit_direction,
                    reason="take_profit",
                    urgency="hours",
                    strength=min(1.0, pnl_pct / self._take_profit_pct * 0.9),
                    current_pnl_pct=pnl_pct,
                ))
                continue

            # ── Time decay ─────────────────────────────────────────────────────
            if position.age_days >= self._max_hold_days:
                exit_direction = "sell" if position.direction == "buy" else "buy"
                signals.append(ExitSignal(
                    ticker=ticker,
                    direction=exit_direction,
                    reason="time_decay",
                    urgency="days",
                    strength=0.5,
                    current_pnl_pct=pnl_pct,
                ))

        return signals

    # ── Statistics ─────────────────────────────────────────────────────────────

    def get_statistics(self) -> dict:
        """Summary statistics for HolonProxy.sense() delegation."""
        state = self.get_portfolio_state()
        return {
            "position_count": state.position_count,
            "unrealized_pnl": round(state.unrealized_pnl, 2),
            "net_exposure": round(state.net_exposure, 3),
            "largest_position_pct": round(state.largest_position_pct, 2),
            "total_value": state.total_value,
            "closed_count": len(self._closed),
        }
