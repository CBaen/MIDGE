"""DrawdownMonitor — equity curve tracking and drawdown circuit-breaker.

Tracks realized P&L from closed paper trades, maintains a high-water mark
(peak equity), calculates drawdown from peak, and trips a circuit-breaker
when drawdown exceeds the configured maximum.

Three alert levels published on EventBus:
  - CH_DRAWDOWN_WARNING  : drawdown >= 80% of max threshold (early warning)
  - CH_TRADING_HALTED    : drawdown >= max threshold (circuit-breaker tripped)
  - CH_TRADING_RESUMED   : drawdown falls back below max threshold (recovery)

Design decisions:
  - threading.RLock on all state mutations (same pattern as ResourceGovernor)
  - equity_history capped at 10,000 entries (rolling — no unbounded growth)
  - EventBus publish is best-effort: exceptions must not propagate
  - Persisted as JSONL (one JSON record per equity snapshot) with atomic write
  - Gracefully degrades when event_bus is None — monitoring still works,
    events simply aren't published

Usage:
    monitor = DrawdownMonitor(event_bus=ctx.bus, data_dir="data/market")
    monitor.load_state()

    # When a position closes:
    monitor.record_trade_result("AAPL", realized_pnl=-1200.0, direction="buy")

    if monitor.is_trading_halted():
        # skip paper trade
        ...
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Channel constants (will be imported from channels.py once Wiring Builder adds them)
CH_DRAWDOWN_WARNING = "market.risk.drawdown_warning"
CH_TRADING_HALTED = "market.risk.trading_halted"
CH_TRADING_RESUMED = "market.risk.trading_resumed"

_EQUITY_HISTORY_FILENAME = "equity_history.jsonl"
_HISTORY_MAXLEN = 10_000


class DrawdownMonitor:
    """Equity curve tracker with peak-drawdown circuit-breaker.

    All state mutations are protected by threading.RLock. EventBus
    publishing is fire-and-forget — exceptions are swallowed so the
    monitor never blocks the step loop.

    Args:
        event_bus:         EventBus instance (optional — monitor works without it)
        starting_capital:  Initial account value. Default 50,000 matches
                           LEARNING_CONFIG["paper_account_value"].
        max_drawdown_pct:  Fractional drawdown that trips the circuit-breaker.
                           0.40 = halt when account falls 40% from peak.
        data_dir:          Directory for equity_history.jsonl persistence.
    """

    def __init__(
        self,
        event_bus: Any = None,
        starting_capital: float = 50_000.0,
        max_drawdown_pct: float = 0.40,
        data_dir: str = "data/market",
    ) -> None:
        self._bus = event_bus
        self._starting_capital = float(starting_capital)
        self._max_drawdown_pct = float(max_drawdown_pct)
        self._data_dir = Path(data_dir)
        self._state_path = self._data_dir / _EQUITY_HISTORY_FILENAME

        self._lock = threading.RLock()

        # Core equity state
        self._current_equity: float = self._starting_capital
        self._peak_equity: float = self._starting_capital
        self._realized_pnl: float = 0.0
        self._trade_count: int = 0

        # Equity history: deque of {"ts": float, "equity": float, "pnl_delta": float}
        self._equity_history: deque = deque(maxlen=_HISTORY_MAXLEN)

        # Circuit-breaker
        self._trading_halted: bool = False

        # Warning tracking — avoids spamming the same warning on every trade
        # once already in the warning zone. Resets when equity recovers.
        self._warning_active: bool = False

    # ── Public API ──────────────────────────────────────────────────────────────

    def record_trade_result(
        self,
        ticker: str,
        realized_pnl: float,
        direction: str = "",
    ) -> None:
        """Record a closed position's realized P&L.

        Updates equity, high-water mark, equity history, and checks the
        circuit-breaker. Publishes EventBus events as thresholds are crossed.

        Args:
            ticker:       Instrument symbol (for event payload only — not stored)
            realized_pnl: Dollar gain/loss from this trade (positive = profit,
                          negative = loss)
            direction:    "buy" or "sell" (for event payload only)
        """
        with self._lock:
            self._realized_pnl += realized_pnl
            self._current_equity = self._starting_capital + self._realized_pnl
            self._trade_count += 1

            # Update high-water mark
            if self._current_equity > self._peak_equity:
                self._peak_equity = self._current_equity

            # Record equity snapshot
            self._equity_history.append({
                "ts": time.time(),
                "equity": self._current_equity,
                "pnl_delta": realized_pnl,
                "ticker": ticker,
                "direction": direction,
            })

            # Evaluate circuit-breaker state
            self._check_circuit_breaker(ticker, realized_pnl, direction)

        logger.debug(
            "DrawdownMonitor: trade recorded ticker=%s pnl=%.2f "
            "equity=%.2f peak=%.2f drawdown=%.2f%%",
            ticker,
            realized_pnl,
            self._current_equity,
            self._peak_equity,
            self.get_current_drawdown() * 100,
        )

    def get_current_drawdown(self) -> float:
        """Return current drawdown as a fraction of peak equity.

        Returns:
            0.0  = at or above peak (no drawdown)
            0.40 = 40% below peak
        """
        with self._lock:
            if self._peak_equity <= 0:
                return 0.0
            dd = (self._peak_equity - self._current_equity) / self._peak_equity
            return max(0.0, dd)

    def is_trading_halted(self) -> bool:
        """True if the circuit-breaker has been tripped (drawdown >= max)."""
        with self._lock:
            return self._trading_halted

    def get_statistics(self) -> dict[str, Any]:
        """Summary statistics for HolonProxy/SomaticMap integration.

        Returns:
            dict with keys: peak_equity, current_equity, drawdown_pct,
            realized_pnl, trading_halted, trade_count
        """
        with self._lock:
            return {
                "peak_equity": round(self._peak_equity, 2),
                "current_equity": round(self._current_equity, 2),
                "drawdown_pct": round(self.get_current_drawdown(), 6),
                "realized_pnl": round(self._realized_pnl, 2),
                "trading_halted": self._trading_halted,
                "trade_count": self._trade_count,
                "max_drawdown_pct": self._max_drawdown_pct,
                "warning_active": self._warning_active,
                "history_length": len(self._equity_history),
            }

    def save_state(self, path: str | None = None) -> None:
        """Persist equity history to JSONL via atomic write.

        Writes to path if provided, else to self._state_path.
        One JSON record per line. Uses os.replace() for atomicity.
        """
        target = Path(path) if path else self._state_path
        tmp = target.with_suffix(".tmp")
        with self._lock:
            snapshot = list(self._equity_history)
            meta = {
                "peak_equity": self._peak_equity,
                "current_equity": self._current_equity,
                "realized_pnl": self._realized_pnl,
                "trade_count": self._trade_count,
                "trading_halted": self._trading_halted,
                "warning_active": self._warning_active,
            }
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as fh:
                # First line: metadata record
                fh.write(json.dumps({"__meta__": True, **meta}) + "\n")
                for record in snapshot:
                    fh.write(json.dumps(record) + "\n")
            os.replace(tmp, target)
            logger.debug(
                "DrawdownMonitor: saved %d equity history records to %s",
                len(snapshot),
                target,
            )
        except Exception as exc:
            logger.warning("DrawdownMonitor: failed to save state: %s", exc)
            try:
                if tmp.exists():
                    tmp.unlink(missing_ok=True)
            except Exception:
                pass

    def load_state(self, path: str | None = None) -> None:
        """Restore equity history and running totals from JSONL.

        Gracefully handles missing file, empty file, and malformed lines.
        If the file is empty or absent, starts from initial state.
        """
        target = Path(path) if path else self._state_path
        if not target.exists():
            logger.debug("DrawdownMonitor: no state file at %s, starting fresh", target)
            return

        records = []
        meta = None
        try:
            with open(target, encoding="utf-8") as fh:
                for line_no, line in enumerate(fh, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        logger.debug(
                            "DrawdownMonitor: malformed JSON on line %d, skipping", line_no
                        )
                        continue
                    if record.get("__meta__"):
                        meta = record
                    else:
                        records.append(record)
        except OSError as exc:
            logger.warning("DrawdownMonitor: failed to read state file: %s", exc)
            return

        with self._lock:
            if meta:
                self._peak_equity = float(meta.get("peak_equity", self._starting_capital))
                self._current_equity = float(meta.get("current_equity", self._starting_capital))
                self._realized_pnl = float(meta.get("realized_pnl", 0.0))
                self._trade_count = int(meta.get("trade_count", 0))
                self._trading_halted = bool(meta.get("trading_halted", False))
                self._warning_active = bool(meta.get("warning_active", False))
            # Restore history (respects maxlen automatically)
            self._equity_history.clear()
            for r in records[-_HISTORY_MAXLEN:]:
                self._equity_history.append(r)

        logger.debug(
            "DrawdownMonitor: loaded %d equity history records from %s "
            "(equity=%.2f peak=%.2f halted=%s)",
            len(records),
            target,
            self._current_equity,
            self._peak_equity,
            self._trading_halted,
        )

    # ── Private helpers ─────────────────────────────────────────────────────────

    def _check_circuit_breaker(
        self,
        ticker: str,
        realized_pnl: float,
        direction: str,
    ) -> None:
        """Evaluate drawdown thresholds and publish EventBus events.

        Must be called while holding self._lock.

        Three transitions managed:
          1. Normal → Warning      (drawdown crosses 80% of max)
          2. Warning → Halted      (drawdown crosses max)
          3. Halted/Warning → OK   (drawdown recovers below max)
        """
        if self._peak_equity <= 0:
            return

        drawdown = (self._peak_equity - self._current_equity) / self._peak_equity
        drawdown = max(0.0, drawdown)

        warning_threshold = self._max_drawdown_pct * 0.80

        payload_base = {
            "ticker": ticker,
            "direction": direction,
            "realized_pnl": realized_pnl,
            "current_equity": self._current_equity,
            "peak_equity": self._peak_equity,
            "drawdown_pct": round(drawdown, 6),
            "max_drawdown_pct": self._max_drawdown_pct,
            "ts": time.time(),
        }

        if drawdown >= self._max_drawdown_pct:
            # Circuit-breaker trip
            if not self._trading_halted:
                self._trading_halted = True
                self._warning_active = True  # warning is also active when halted
                self._publish(CH_TRADING_HALTED, {
                    **payload_base,
                    "reason": f"Drawdown {drawdown:.1%} exceeded max {self._max_drawdown_pct:.1%}",
                })
                logger.warning(
                    "DrawdownMonitor: TRADING HALTED — drawdown %.2f%% exceeds %.2f%% max "
                    "(equity=%.2f peak=%.2f)",
                    drawdown * 100,
                    self._max_drawdown_pct * 100,
                    self._current_equity,
                    self._peak_equity,
                )
        elif drawdown >= warning_threshold:
            # Warning zone — not yet halted (or re-entered warning after partial recovery)
            if self._trading_halted:
                # Partial recovery: was halted, now below halt threshold but still in warning
                self._trading_halted = False
                self._publish(CH_TRADING_RESUMED, {
                    **payload_base,
                    "reason": f"Drawdown {drawdown:.1%} recovered below halt threshold {self._max_drawdown_pct:.1%}",
                })
                logger.info(
                    "DrawdownMonitor: trading RESUMED — drawdown %.2f%% below halt threshold %.2f%%",
                    drawdown * 100,
                    self._max_drawdown_pct * 100,
                )
            if not self._warning_active:
                # First time entering warning zone
                self._warning_active = True
                self._publish(CH_DRAWDOWN_WARNING, {
                    **payload_base,
                    "warning_threshold": warning_threshold,
                    "reason": f"Drawdown {drawdown:.1%} approaching max {self._max_drawdown_pct:.1%}",
                })
                logger.warning(
                    "DrawdownMonitor: DRAWDOWN WARNING — %.2f%% drawdown (warning at %.2f%%, halt at %.2f%%)",
                    drawdown * 100,
                    warning_threshold * 100,
                    self._max_drawdown_pct * 100,
                )
        else:
            # Below warning threshold — healthy
            if self._trading_halted:
                self._trading_halted = False
                self._publish(CH_TRADING_RESUMED, {
                    **payload_base,
                    "reason": f"Drawdown {drawdown:.1%} fully recovered below warning threshold",
                })
                logger.info(
                    "DrawdownMonitor: trading RESUMED — drawdown %.2f%% fully recovered",
                    drawdown * 100,
                )
            if self._warning_active:
                self._warning_active = False
                logger.info(
                    "DrawdownMonitor: warning cleared — drawdown %.2f%% below warning threshold %.2f%%",
                    drawdown * 100,
                    warning_threshold * 100,
                )

    def _publish(self, channel: str, payload: dict) -> None:
        """Best-effort EventBus publish. Exceptions are swallowed."""
        if self._bus is None:
            return
        try:
            self._bus.publish(channel, payload)
        except Exception as exc:
            logger.debug("DrawdownMonitor: failed to publish %s: %s", channel, exc)
