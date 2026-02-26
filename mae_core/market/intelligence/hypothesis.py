"""Hypothesis dataclass — the fundamental unit of MIDGE's RSI Layer 2.

A Hypothesis is a formalized, testable pattern: "When signal A fires,
price moves in direction D within W days." Every hypothesis carries:
  - Trigger pattern (sources, domains, lag_days, direction)
  - Lifecycle status (probation → active → hibernated → retired)
  - Statistical tracking (wins, losses, DSR, IC)
  - Causal story (required — anti-overfitting gate)
  - Provenance (how it was discovered)

Law 2 compliance: Hypotheses form triads with their generator (creator)
and validator (adversary). No hypothesis exists without a witness.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class HypothesisStatus(Enum):
    """Lifecycle stages. One-way flow except hibernation (reversible)."""
    PROBATION = "probation"    # Just discovered, awaiting validation
    ACTIVE = "active"          # Validated, promoted, being monitored
    HIBERNATED = "hibernated"  # Temporarily deactivated (regime mismatch)
    RETIRED = "retired"        # Permanently deactivated (failed validation)


class SourceType(Enum):
    """How the hypothesis was discovered."""
    LAG_CORRELATION = "lag_correlation"     # From lag_correlation_analyzer findings
    MANUAL = "manual"                      # Human-specified
    AGENT_DISCOVERED = "agent_discovered"  # Discovered by HYPOTHESIS_EXPLORER agent


@dataclass
class TriggerPattern:
    """What fires the hypothesis — the specific signal pattern it watches for."""
    source_a: str               # Leading signal source (e.g., "sec_form4")
    source_b: str               # Lagging signal source (e.g., "finnhub_earnings")
    lag_days: int               # Expected lag between A and B
    direction: str = ""         # "positive" or "negative" correlation
    min_strength: float = 0.0   # Minimum signal strength to trigger
    domain_filter: str = ""     # Optional domain restriction


@dataclass
class HypothesisStats:
    """Running statistical record for the hypothesis."""
    wins: int = 0
    losses: int = 0
    total_observations: int = 0
    cumulative_return_pct: float = 0.0
    information_coefficient: float = 0.0  # IC: correlation of prediction vs outcome
    sharpe_ratio: float = 0.0
    deflated_sharpe_ratio: float = 0.0    # DSR: the anti-overfitting metric
    last_evaluated: Optional[datetime] = None

    @property
    def win_rate(self) -> float:
        """Win rate as fraction [0, 1]."""
        if self.total_observations == 0:
            return 0.0
        return self.wins / self.total_observations


@dataclass
class Hypothesis:
    """A formalized, testable market pattern with full lifecycle tracking.

    Every hypothesis must have a causal_story — this is the anti-overfitting
    gate. If you can't explain WHY the pattern works, it's curve-fitting.
    """
    hypothesis_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    trigger: TriggerPattern = field(default_factory=lambda: TriggerPattern("", ""))
    status: HypothesisStatus = HypothesisStatus.PROBATION
    stats: HypothesisStats = field(default_factory=HypothesisStats)
    causal_story: str = ""      # Required before promotion — WHY does this work?
    source_type: SourceType = SourceType.LAG_CORRELATION
    parent_lag_finding: dict = field(default_factory=dict)  # Original lag finding data
    created_at: datetime = field(default_factory=datetime.now)
    promoted_at: Optional[datetime] = None
    retired_at: Optional[datetime] = None
    retired_reason: str = ""
    regime_created_in: str = "default"

    def to_dict(self) -> dict:
        """Serialize for JSONL persistence."""
        return {
            "hypothesis_id": self.hypothesis_id,
            "name": self.name,
            "trigger": {
                "source_a": self.trigger.source_a,
                "source_b": self.trigger.source_b,
                "lag_days": self.trigger.lag_days,
                "direction": self.trigger.direction,
                "min_strength": self.trigger.min_strength,
                "domain_filter": self.trigger.domain_filter,
            },
            "status": self.status.value,
            "stats": {
                "wins": self.stats.wins,
                "losses": self.stats.losses,
                "total_observations": self.stats.total_observations,
                "cumulative_return_pct": self.stats.cumulative_return_pct,
                "information_coefficient": self.stats.information_coefficient,
                "sharpe_ratio": self.stats.sharpe_ratio,
                "deflated_sharpe_ratio": self.stats.deflated_sharpe_ratio,
                "last_evaluated": (
                    self.stats.last_evaluated.isoformat()
                    if self.stats.last_evaluated else None
                ),
            },
            "causal_story": self.causal_story,
            "source_type": self.source_type.value,
            "parent_lag_finding": self.parent_lag_finding,
            "created_at": self.created_at.isoformat(),
            "promoted_at": self.promoted_at.isoformat() if self.promoted_at else None,
            "retired_at": self.retired_at.isoformat() if self.retired_at else None,
            "retired_reason": self.retired_reason,
            "regime_created_in": self.regime_created_in,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Hypothesis:
        """Deserialize from JSONL record."""
        trigger_data = data.get("trigger", {})
        stats_data = data.get("stats", {})

        last_eval = stats_data.get("last_evaluated")
        if last_eval and isinstance(last_eval, str):
            last_eval = datetime.fromisoformat(last_eval)

        created = data.get("created_at", "")
        promoted = data.get("promoted_at")
        retired = data.get("retired_at")

        return cls(
            hypothesis_id=data.get("hypothesis_id", str(uuid.uuid4())),
            name=data.get("name", ""),
            trigger=TriggerPattern(
                source_a=trigger_data.get("source_a", ""),
                source_b=trigger_data.get("source_b", ""),
                lag_days=trigger_data.get("lag_days", 0),
                direction=trigger_data.get("direction", ""),
                min_strength=trigger_data.get("min_strength", 0.0),
                domain_filter=trigger_data.get("domain_filter", ""),
            ),
            status=HypothesisStatus(data.get("status", "probation")),
            stats=HypothesisStats(
                wins=stats_data.get("wins", 0),
                losses=stats_data.get("losses", 0),
                total_observations=stats_data.get("total_observations", 0),
                cumulative_return_pct=stats_data.get("cumulative_return_pct", 0.0),
                information_coefficient=stats_data.get("information_coefficient", 0.0),
                sharpe_ratio=stats_data.get("sharpe_ratio", 0.0),
                deflated_sharpe_ratio=stats_data.get("deflated_sharpe_ratio", 0.0),
                last_evaluated=last_eval,
            ),
            causal_story=data.get("causal_story", ""),
            source_type=SourceType(data.get("source_type", "lag_correlation")),
            parent_lag_finding=data.get("parent_lag_finding", {}),
            created_at=(
                datetime.fromisoformat(created) if created else datetime.now()
            ),
            promoted_at=(
                datetime.fromisoformat(promoted) if promoted else None
            ),
            retired_at=(
                datetime.fromisoformat(retired) if retired else None
            ),
            retired_reason=data.get("retired_reason", ""),
            regime_created_in=data.get("regime_created_in", "default"),
        )
