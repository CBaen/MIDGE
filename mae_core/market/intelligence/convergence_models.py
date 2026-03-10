"""Convergence models — Signal, ConvergenceAlert dataclasses and read_discoveries.

Extracted from convergence_alerter.py to keep it under the 500-line cap.
All public names are re-exported by convergence_alerter.py for backward compatibility.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "market"
_DISCOVERY_LOG = _DATA_DIR / "discovery_log.jsonl"


@dataclass
class Signal:
    """Single signal observation."""
    signal_id: str
    strength: float  # 0-1 normalized
    domain: str
    direction: str  # bullish, bearish, neutral
    timestamp: datetime
    metadata: dict = field(default_factory=dict)
    velocity: float = 0.0  # Rate of change
    confidence: float = 0.5  # Reliability estimate
    source: str = ""  # Original source type for Thompson key lookup


@dataclass
class ConvergenceAlert:
    """Alert generated when multiple domains converge."""
    alert_id: str
    timestamp: datetime
    direction: str  # bullish, bearish
    strength: float  # Overall convergence strength 0-1
    confidence: float  # Reliability estimate 0-1
    domains_converging: List[str]
    signals: List[Signal]
    cross_domain_count: int
    summary: str
    urgency: str  # immediate, hours, days
    # Coherence fields — Capability 3 (narrative conflict detection).
    # Defaults preserve backward compatibility with existing callers.
    coherence: float = 1.0            # 1.0 = all signals agree, 0.5 = evenly split
    contradiction_details: list = field(default_factory=list)  # [(domain, direction), ...]
    combo_key: str = ""  # Thompson key for this domain combination (e.g. "combo:events+macro+price")
    # Temporal ordering fields — domains sorted by when they fired (earliest first).
    # sequence_score > 1.0 = ordering matches known lag relationships (boost).
    # sequence_score < 1.0 = ordering reversed vs. known lags (discount).
    # Default 1.0 = neutral (no lag data or single domain).
    domain_sequence: List[str] = field(default_factory=list)
    sequence_score: float = 1.0
    # Causal cascade — WorldModel ripple effects predicting downstream dominoes.
    # Each entry: {ticker, direction, strength, lag_days, path, confidence}
    ripple_effects: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp.isoformat(),
            "direction": self.direction,
            "strength": round(self.strength, 3),
            "confidence": round(self.confidence, 3),
            "domains": self.domains_converging,
            "signal_count": len(self.signals),
            "cross_domain_count": self.cross_domain_count,
            "summary": self.summary,
            "urgency": self.urgency,
            "coherence": round(self.coherence, 3),
            "contradiction_details": self.contradiction_details,
            "combo_key": self.combo_key,
            "domain_sequence": self.domain_sequence,
            "sequence_score": round(self.sequence_score, 3),
            "ripple_effects": self.ripple_effects,
        }


def read_discoveries(
    max_entries: int = 100,
    min_strength: float = 0.0,
    direction: str = None,
) -> List[dict]:
    """
    Read recent discoveries from discovery_log.jsonl.

    Useful for:
    - Surfacing novel patterns to a dashboard
    - Seeding Thompson distributions with discovered signal combinations
    - Auditing what convergence patterns MIDGE has detected

    Args:
        max_entries: Maximum entries to return (most recent first)
        min_strength: Only return discoveries above this strength
        direction: Filter by "bullish" or "bearish" (None = all)

    Returns:
        List of discovery dicts, most recent first
    """
    if not _DISCOVERY_LOG.exists():
        return []

    entries = []
    try:
        with open(_DISCOVERY_LOG, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("strength", 0) < min_strength:
                        continue
                    if direction and entry.get("direction") != direction:
                        continue
                    entries.append(entry)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.debug("Failed to read discovery log: %s", e)

    # Most recent first, capped
    return entries[-max_entries:][::-1]
