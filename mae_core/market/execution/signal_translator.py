#!/usr/bin/env python3
"""
signal_translator.py - ConvergenceAlert → ExecutableSignal translator.

Converts MIDGE ConvergenceAlerts into broker-ready trade parameters:
  - Entry price   (current market price)
  - Stop loss     (entry ∓ 1.5 × ATR, against direction)
  - Take profit   (entry ± 3.0 × ATR, 2:1 reward/risk)
  - Position size (% of account to risk, derived from SL distance)

The 1.5×ATR stop gives the trade room to breathe without being so wide
that expected losses are ruinous.  The 3.0×ATR target locks in a 2:1
reward-to-risk ratio, consistent with the payoff profile MIDGE's replay
harness measured (3.34:1 actual, 2:1 minimum required for profitability
at convergence win rates).

Graceful degradation:
  - If current_price is 0 or None  → returns None (caller skips)
  - If ATR is 0 or None            → returns None with warning (no SL/TP)
  - All arithmetic guarded against divide-by-zero

Ticker resolution follows the same logic as _write_paper_trade in
market_hooks.py: walk alert.signals looking for metadata["symbol"].
"""

import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger(__name__)

# ATR multipliers — calibrated to 2:1 R:R minimum
_SL_ATR_MULTIPLE = 1.5   # Stop loss distance = 1.5 × ATR
_TP_ATR_MULTIPLE = 3.0   # Take profit distance = 3.0 × ATR  → 2:1 R:R

# Hard position size caps (fraction of account)
_MAX_POSITION_PCT = 0.10   # Never risk more than 10% in one trade
_MIN_POSITION_PCT = 0.001  # Floor — prevents rounding to zero


@dataclass
class ExecutableSignal:
    """A ConvergenceAlert translated into concrete broker order parameters.

    Fields:
        ticker          Instrument symbol (resolved from alert signals)
        direction       "long" or "short"
        entry_price     Current market price at signal generation
        stop_loss       Absolute price level for stop-loss order
        take_profit     Absolute price level for take-profit order
        risk_distance   |entry - stop_loss| in price units
        reward_distance |take_profit - entry| in price units
        rr_ratio        reward_distance / risk_distance (target ≥ 2.0)
        position_size_pct  Fraction of account to risk (capped at _MAX_POSITION_PCT)
        confidence      ConvergenceAlert.confidence passed through
        domains         ConvergenceAlert.domains_converging passed through
        source_alert_id ConvergenceAlert.alert_id for back-tracking
        atr             ATR value used to size stop and target
        timestamp       ISO-8601 UTC string at translation time
    """

    ticker: str
    direction: str               # "long" or "short"
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_distance: float         # |entry - stop_loss|
    reward_distance: float       # |take_profit - entry|
    rr_ratio: float              # reward / risk (should be ~2.0)
    position_size_pct: float     # fraction of account equity to risk
    confidence: float
    domains: List[str] = field(default_factory=list)
    source_alert_id: str = ""
    atr: float = 0.0
    timestamp: str = ""

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict for JSONL logging."""
        return {
            "ticker": self.ticker,
            "direction": self.direction,
            "entry_price": round(self.entry_price, 4),
            "stop_loss": round(self.stop_loss, 4),
            "take_profit": round(self.take_profit, 4),
            "risk_distance": round(self.risk_distance, 4),
            "reward_distance": round(self.reward_distance, 4),
            "rr_ratio": round(self.rr_ratio, 3),
            "position_size_pct": round(self.position_size_pct, 4),
            "confidence": round(self.confidence, 4),
            "domains": self.domains,
            "source_alert_id": self.source_alert_id,
            "atr": round(self.atr, 4),
            "timestamp": self.timestamp,
        }


def _resolve_ticker(alert_dict: dict) -> str:
    """Walk alert signals to find the best ticker.

    Mirrors the resolution in _write_paper_trade (market_hooks.py):
    checks metadata["symbol"] on each signal dict, falls back to "MULTI".
    """
    for sig in alert_dict.get("signals", []):
        # Signals may be dicts (from to_dict()) or Signal objects
        if isinstance(sig, dict):
            sym = sig.get("metadata", {}).get("symbol", "")
        else:
            sym = getattr(sig, "metadata", {}).get("symbol", "")
        if sym:
            return sym
    return "MULTI"


def translate_alert(
    alert_dict: dict,
    current_price: float,
    atr: float,
    account_risk_pct: float = 0.02,
) -> Optional["ExecutableSignal"]:
    """Convert a ConvergenceAlert dict into an ExecutableSignal.

    Args:
        alert_dict:       ConvergenceAlert.to_dict() output (or equivalent dict)
        current_price:    Current market price for the instrument (entry proxy)
        atr:              14-period ATR for the instrument
        account_risk_pct: Fraction of account equity to risk (default 2%)

    Returns:
        ExecutableSignal if translation succeeds, None on invalid inputs.

    Stop / Target math:
        Long:  stop_loss  = entry - 1.5 × ATR
               take_profit = entry + 3.0 × ATR   → 2:1 R:R
        Short: stop_loss  = entry + 1.5 × ATR
               take_profit = entry - 3.0 × ATR   → 2:1 R:R

    Position sizing:
        risk_distance = |entry - stop_loss| = 1.5 × ATR
        risk_fraction = risk_distance / entry_price   (as % of entry)
        position_pct  = account_risk_pct / risk_fraction
        Capped at _MAX_POSITION_PCT to prevent over-sizing on low-ATR instruments.
    """
    # ── Guard: price must be positive ──────────────────────────────────────────
    if not current_price or not math.isfinite(current_price) or current_price <= 0:
        logger.debug(
            "translate_alert: invalid current_price=%s for %s — skipping",
            current_price, alert_dict.get("alert_id", "?"),
        )
        return None

    # ── Guard: ATR must be positive ─────────────────────────────────────────────
    if not atr or not math.isfinite(atr) or atr <= 0:
        logger.warning(
            "translate_alert: ATR=%s for %s — cannot size SL/TP, skipping",
            atr, alert_dict.get("alert_id", "?"),
        )
        return None

    # ── Direction ───────────────────────────────────────────────────────────────
    raw_direction = alert_dict.get("direction", "neutral")
    if raw_direction == "bullish":
        direction = "long"
    elif raw_direction == "bearish":
        direction = "short"
    else:
        logger.debug(
            "translate_alert: neutral/unknown direction '%s' — not actionable",
            raw_direction,
        )
        return None

    # ── Levels ──────────────────────────────────────────────────────────────────
    sl_distance = _SL_ATR_MULTIPLE * atr
    tp_distance = _TP_ATR_MULTIPLE * atr

    if direction == "long":
        stop_loss   = current_price - sl_distance
        take_profit = current_price + tp_distance
    else:  # short
        stop_loss   = current_price + sl_distance
        take_profit = current_price - tp_distance

    risk_distance   = abs(current_price - stop_loss)    # = sl_distance
    reward_distance = abs(take_profit - current_price)  # = tp_distance
    rr_ratio = reward_distance / risk_distance if risk_distance > 0 else 0.0

    # ── Position sizing ─────────────────────────────────────────────────────────
    # risk_fraction = what % of entry price we lose if stopped out
    risk_fraction = risk_distance / current_price
    if risk_fraction <= 0:
        position_size_pct = _MIN_POSITION_PCT
    else:
        position_size_pct = account_risk_pct / risk_fraction
        position_size_pct = max(_MIN_POSITION_PCT, min(_MAX_POSITION_PCT, position_size_pct))

    # ── Metadata ────────────────────────────────────────────────────────────────
    ticker = _resolve_ticker(alert_dict)

    return ExecutableSignal(
        ticker=ticker,
        direction=direction,
        entry_price=current_price,
        stop_loss=round(stop_loss, 4),
        take_profit=round(take_profit, 4),
        risk_distance=round(risk_distance, 4),
        reward_distance=round(reward_distance, 4),
        rr_ratio=round(rr_ratio, 3),
        position_size_pct=round(position_size_pct, 6),
        confidence=float(alert_dict.get("confidence", 0.0)),
        domains=list(alert_dict.get("domains", [])),
        source_alert_id=str(alert_dict.get("alert_id", "")),
        atr=round(atr, 4),
        timestamp=datetime.utcnow().isoformat() + "Z",
    )
