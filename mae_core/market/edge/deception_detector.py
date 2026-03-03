"""Deception Immune Response — Gift 5.

Detects coordinated manipulation patterns: pump-and-dump, wash trading
signatures, retail traps. When social sentiment explodes but insider
activity is absent or inverse, that's a manufactured signal, not organic
convergence. MIDGE needs an immune system.

Detection rules:
- Pump-and-dump: social surge (>3 signals/24h) + zero insider/institutional
  signals + low volume_zscore → deception_prob 0.7+
- Retail trap: price breakout + high social + insider selling → 0.6+
- Wash trading proxy: volume spike >5σ + no news/insider catalyst → flag
"""

from __future__ import annotations

import json
import logging
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Maximum signal history per ticker
_MAX_HISTORY = 100
# Window for pattern detection
_DETECTION_WINDOW_HOURS = 48.0
# Persistence directory
_STATE_DIR = Path(__file__).resolve().parents[3] / "data" / "market"


@dataclass
class DeceptionAssessment:
    """Result of assessing signal authenticity for a ticker."""

    ticker: str
    deception_probability: float  # 0.0 to 1.0
    pattern_type: str  # "pump_dump", "retail_trap", "wash_trading", "none"
    evidence: List[str] = field(default_factory=list)


@dataclass
class _SignalRecord:
    """Internal record of a signal for history tracking."""

    ticker: str
    domain: str
    direction: str
    strength: float
    timestamp: datetime


class DeceptionDetector:
    """Detect coordinated manipulation patterns in signal flow.

    Monitors signal history per ticker and flags suspicious patterns
    where signal composition suggests manufactured rather than organic
    market interest.
    """

    def __init__(self, event_bus=None):
        self._event_bus = event_bus
        self._signal_history: Dict[str, deque] = {}

    def save_state(self) -> None:
        """Persist signal history to disk as JSON (atomic write)."""
        payload: Dict[str, list] = {}
        for ticker, dq in self._signal_history.items():
            payload[ticker] = [
                {
                    "ticker": r.ticker,
                    "domain": r.domain,
                    "direction": r.direction,
                    "strength": r.strength,
                    "timestamp": r.timestamp.isoformat(),
                }
                for r in dq
            ]

        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        target = _STATE_DIR / "deception_state.json"
        tmp = target.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            os.replace(tmp, target)
            total = sum(len(v) for v in payload.values())
            logger.debug("DeceptionDetector: saved %d records for %d tickers", total, len(payload))
        except Exception as exc:
            logger.warning("DeceptionDetector: save_state failed: %s", exc)
            tmp.unlink(missing_ok=True)

    def load_state(self) -> None:
        """Restore signal history from disk, dropping expired entries."""
        state_file = _STATE_DIR / "deception_state.json"
        if not state_file.exists():
            return

        try:
            raw = json.loads(state_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("DeceptionDetector: load_state failed to parse JSON: %s", exc)
            return

        cutoff = datetime.now() - timedelta(hours=_DETECTION_WINDOW_HOURS)
        loaded = 0
        for ticker, records in raw.items():
            dq: deque = deque(maxlen=_MAX_HISTORY)
            for entry in records:
                try:
                    ts = datetime.fromisoformat(entry["timestamp"])
                    if ts < cutoff:
                        continue
                    dq.append(
                        _SignalRecord(
                            ticker=entry["ticker"],
                            domain=entry["domain"],
                            direction=entry["direction"],
                            strength=float(entry["strength"]),
                            timestamp=ts,
                        )
                    )
                    loaded += 1
                except Exception:
                    continue
            if dq:
                self._signal_history[ticker] = dq

        logger.debug("DeceptionDetector: loaded %d records for %d tickers", loaded, len(self._signal_history))

    def record_signal(
        self,
        ticker: str,
        domain: str,
        direction: str,
        strength: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Record a signal for deception pattern analysis."""
        if not ticker:
            return

        ts = timestamp or datetime.now()
        record = _SignalRecord(
            ticker=ticker,
            domain=domain,
            direction=direction,
            strength=strength,
            timestamp=ts,
        )

        if ticker not in self._signal_history:
            self._signal_history[ticker] = deque(maxlen=_MAX_HISTORY)
        self._signal_history[ticker].append(record)

    def assess_signal_authenticity(self, ticker: str) -> DeceptionAssessment:
        """Assess whether recent signals on a ticker appear authentic.

        Examines signal composition within the detection window and checks
        for known manipulation signatures.
        """
        if ticker not in self._signal_history:
            return DeceptionAssessment(
                ticker=ticker, deception_probability=0.0, pattern_type="none"
            )

        cutoff = datetime.now() - timedelta(hours=_DETECTION_WINDOW_HOURS)
        recent = [
            r for r in self._signal_history[ticker] if r.timestamp >= cutoff
        ]

        if len(recent) < 2:
            return DeceptionAssessment(
                ticker=ticker, deception_probability=0.0, pattern_type="none"
            )

        # Categorize signals by domain
        domain_counts: Dict[str, int] = {}
        domain_directions: Dict[str, List[str]] = {}
        for r in recent:
            domain_counts[r.domain] = domain_counts.get(r.domain, 0) + 1
            if r.domain not in domain_directions:
                domain_directions[r.domain] = []
            domain_directions[r.domain].append(r.direction)

        social_count = domain_counts.get("sentiment", 0) + domain_counts.get(
            "social", 0
        )
        insider_count = domain_counts.get("insider", 0) + domain_counts.get(
            "regulatory", 0
        )
        technical_count = domain_counts.get("technical", 0)
        institutional_count = domain_counts.get("institutional", 0)

        # Check for pump-and-dump pattern
        pump_dump = self._check_pump_dump(
            ticker, social_count, insider_count, institutional_count, recent
        )
        if pump_dump.deception_probability > 0.0:
            return pump_dump

        # Check for retail trap pattern
        retail_trap = self._check_retail_trap(
            ticker, social_count, insider_count, technical_count, recent,
            domain_directions,
        )
        if retail_trap.deception_probability > 0.0:
            return retail_trap

        # Check for wash trading proxy
        wash = self._check_wash_trading(
            ticker, technical_count, insider_count, social_count, recent
        )
        if wash.deception_probability > 0.0:
            return wash

        return DeceptionAssessment(
            ticker=ticker, deception_probability=0.0, pattern_type="none"
        )

    def _check_pump_dump(
        self,
        ticker: str,
        social_count: int,
        insider_count: int,
        institutional_count: int,
        recent: List[_SignalRecord],
    ) -> DeceptionAssessment:
        """Social sentiment surge + zero insider/institutional activity."""
        evidence = []

        if social_count < 3:
            return DeceptionAssessment(
                ticker=ticker, deception_probability=0.0, pattern_type="none"
            )

        evidence.append(f"Social signal surge: {social_count} signals in window")

        if insider_count > 0 or institutional_count > 0:
            # Some insider/institutional activity — less suspicious
            return DeceptionAssessment(
                ticker=ticker, deception_probability=0.0, pattern_type="none"
            )

        evidence.append("Zero insider/institutional signals during social surge")

        # Check if social signals are uni-directional (coordinated)
        social_signals = [
            r for r in recent if r.domain in ("sentiment", "social")
        ]
        directions = [r.direction for r in social_signals]
        dominant_pct = max(
            directions.count("bullish"), directions.count("bearish")
        ) / len(directions) if directions else 0.0

        if dominant_pct > 0.8:
            evidence.append(
                f"Uni-directional social signals: {dominant_pct:.0%} same direction"
            )

        # Compute probability
        base_prob = 0.50
        if social_count >= 5:
            base_prob += 0.15
        if dominant_pct > 0.8:
            base_prob += 0.10
        if social_count >= 3 and insider_count == 0 and institutional_count == 0:
            base_prob += 0.05

        prob = min(1.0, base_prob)

        return DeceptionAssessment(
            ticker=ticker,
            deception_probability=prob,
            pattern_type="pump_dump",
            evidence=evidence,
        )

    def _check_retail_trap(
        self,
        ticker: str,
        social_count: int,
        insider_count: int,
        technical_count: int,
        recent: List[_SignalRecord],
        domain_directions: Dict[str, List[str]],
    ) -> DeceptionAssessment:
        """Price breakout + high social + insider selling = trap."""
        evidence = []

        if social_count < 2 or insider_count < 1:
            return DeceptionAssessment(
                ticker=ticker, deception_probability=0.0, pattern_type="none"
            )

        # Check for directional conflict: social bullish + insider bearish
        social_dirs = domain_directions.get("sentiment", []) + domain_directions.get(
            "social", []
        )
        insider_dirs = domain_directions.get("insider", []) + domain_directions.get(
            "regulatory", []
        )

        social_bullish = social_dirs.count("bullish") > social_dirs.count("bearish")
        insider_bearish = insider_dirs.count("bearish") > insider_dirs.count("bullish")

        if not (social_bullish and insider_bearish):
            return DeceptionAssessment(
                ticker=ticker, deception_probability=0.0, pattern_type="none"
            )

        evidence.append("Social sentiment bullish while insiders selling")
        evidence.append(f"Social signals: {social_count}, Insider signals: {insider_count}")

        # Technical breakout amplifies the trap
        if technical_count > 0:
            evidence.append("Technical breakout present — classic trap setup")

        base_prob = 0.55
        if technical_count > 0:
            base_prob += 0.10
        if social_count >= 4:
            base_prob += 0.05

        prob = min(1.0, base_prob)

        return DeceptionAssessment(
            ticker=ticker,
            deception_probability=prob,
            pattern_type="retail_trap",
            evidence=evidence,
        )

    def _check_wash_trading(
        self,
        ticker: str,
        technical_count: int,
        insider_count: int,
        social_count: int,
        recent: List[_SignalRecord],
    ) -> DeceptionAssessment:
        """Volume spike with no news/insider/social catalyst."""
        evidence = []

        # Need technical signals (volume-based) but no other catalysts
        volume_signals = [
            r for r in recent
            if r.domain == "technical" and r.strength > 0.7
        ]

        if len(volume_signals) < 2:
            return DeceptionAssessment(
                ticker=ticker, deception_probability=0.0, pattern_type="none"
            )

        if insider_count > 0 or social_count > 0:
            # Has catalysts — less suspicious
            return DeceptionAssessment(
                ticker=ticker, deception_probability=0.0, pattern_type="none"
            )

        evidence.append(
            f"High-strength technical signals ({len(volume_signals)}) with no catalysts"
        )
        evidence.append("Zero insider and social signals during volume activity")

        prob = min(1.0, 0.40 + 0.05 * len(volume_signals))

        return DeceptionAssessment(
            ticker=ticker,
            deception_probability=prob,
            pattern_type="wash_trading",
            evidence=evidence,
        )

    def get_statistics(self) -> dict:
        """Return detector statistics for holon protocol."""
        total_tickers = len(self._signal_history)
        total_records = sum(len(q) for q in self._signal_history.values())
        return {
            "tracked_tickers": total_tickers,
            "total_records": total_records,
        }
