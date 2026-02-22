"""Quorum Sensor - Agent-level quorum sensing interface.

Each agent gets a QuorumSensor to deposit signals, sense concentrations,
and query collective insights. Includes rate limiting and history tracking.

Biological analogy: Individual bacterium's autoinducer receptor.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Optional

from .consensus_metrics import ConfidenceInterval, ConsensusConfidenceCalculator
from .message_aggregator import AggregatedMessage, MessageAggregator, MessagePriority
from .quorum_signal import SenseEvent
from .quorum_space import QuorumSpace

logger = logging.getLogger(__name__)

MIN_VOTES = 3  # Minimum votes for statistical significance


@dataclass
class DepositRecord:
    """Record of a signal deposit by this agent."""

    timestamp: float = field(default_factory=time.time)
    signal_type: str = ""
    strength: float = 1.0
    concentration_after: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class TokenBucketRateLimiter:
    """Simple token bucket for deposit rate limiting."""

    def __init__(self, rate_per_second: float = 1.0, burst_size: int = 10) -> None:
        self._rate = rate_per_second
        self._burst = burst_size
        self._tokens = float(burst_size)
        self._last_refill = time.time()

    def try_consume(self) -> bool:
        now = time.time()
        elapsed = now - self._last_refill
        self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
        self._last_refill = now

        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False


class QuorumSensor:
    """Agent-level quorum sensing interface.

    Provides the interface expected by CollectiveConsensusMixin:
    sense(), get_collective_insights(), get_consensus_strength(), get_statistics().
    """

    def __init__(
        self,
        agent_id: str,
        quorum_space: QuorumSpace,
        message_aggregator: MessageAggregator | None = None,
        confidence_calculator: ConsensusConfidenceCalculator | None = None,
        max_deposits_per_minute: int = 60,
        max_history: int = 1000,
    ) -> None:
        self._agent_id = agent_id
        self._space = quorum_space
        self._aggregator = message_aggregator
        self._confidence = confidence_calculator or ConsensusConfidenceCalculator()
        self._limiter = TokenBucketRateLimiter(
            rate_per_second=max_deposits_per_minute / 60.0,
            burst_size=min(10, max_deposits_per_minute),
        )

        self._deposit_history: deque[DepositRecord] = deque(maxlen=max_history)
        self._sensing_history: deque[SenseEvent] = deque(maxlen=max_history)

        # Counters
        self._total_deposits = 0
        self._total_senses = 0
        self._rate_limit_violations = 0
        self._deposits_by_type: dict[str, int] = defaultdict(int)

    # --- Core operations ---

    def deposit(
        self,
        signal_type: str,
        strength: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Deposit a signal into the quorum space."""
        if not self._limiter.try_consume():
            self._rate_limit_violations += 1
            return False

        success = self._space.deposit_signal(
            signal_type, self._agent_id, strength, metadata
        )

        if success:
            concentration = self._space.get_concentration(signal_type)
            self._deposit_history.append(DepositRecord(
                signal_type=signal_type,
                strength=strength or 1.0,
                concentration_after=concentration,
                metadata=metadata or {},
            ))
            self._total_deposits += 1
            self._deposits_by_type[signal_type] += 1

            # Also submit to message aggregator if available
            if self._aggregator:
                self._aggregator.submit_message(
                    self._agent_id, signal_type, metadata or {}
                )

        return success

    # Alias
    deposit_signal = deposit

    def sense(self, signal_type: str) -> float:
        """Sense current concentration of a signal type."""
        concentration = self._space.get_concentration(signal_type)
        self._total_senses += 1

        self._sensing_history.append(SenseEvent(
            event_type="sense",
            signal_type=signal_type,
            agent_id=self._agent_id,
            concentration=concentration,
        ))

        return concentration

    def sense_all(self) -> dict[str, float]:
        """Sense all signal concentrations."""
        return self._space.get_all_concentrations()

    def has_contributed(self, signal_type: str) -> bool:
        return self._agent_id in self._space.get_contributors(signal_type)

    def get_contributor_fraction(self, signal_type: str, total_agents: int) -> float:
        if total_agents <= 0:
            return 0.0
        return self._space.get_contributor_count(signal_type) / total_agents

    # --- Phase 3: Consensus-weighted memory priority ---

    def get_consensus_priority(
        self, signal_type: str | None = None, metadata: dict[str, Any] | None = None
    ) -> float:
        """Get consensus-weighted priority for memory storage. Returns [0, 1]."""
        if not signal_type:
            return 0.5

        concentration = self._space.get_concentration(signal_type)
        contributors = self._space.get_contributor_count(signal_type)

        # Priority = concentration * sqrt(contributors) / scaling
        priority = concentration * (contributors ** 0.5) / 5.0
        return max(0.0, min(1.0, priority))

    def get_collective_insights(
        self,
        count: int = 5,
        min_priority: MessagePriority | None = None,
    ) -> list[AggregatedMessage]:
        """Get top collective insights from message aggregator."""
        if not self._aggregator:
            return []
        return self._aggregator.get_top_messages(count, mark_read_by=self._agent_id)

    def get_consensus_strength(
        self, signal_type: str, min_votes: int = MIN_VOTES
    ) -> float:
        """How strong is consensus for this signal? Returns [0, 1]."""
        if self._aggregator:
            return self._aggregator.get_consensus_strength(signal_type, min_votes)
        # Fallback: use raw concentration
        return self._space.get_concentration(signal_type)

    # --- Phase 4: Confidence metrics ---

    def get_message_confidence(
        self,
        signal_type: str,
        metadata: dict[str, Any],
        total_agents: int,
    ) -> ConfidenceInterval | None:
        """Get statistical confidence for a consensus measurement."""
        supporters = self._space.get_contributor_count(signal_type)
        if supporters < MIN_VOTES:
            return None
        return self._confidence.calculate(supporters, total_agents)

    def is_consensus_confident(
        self,
        signal_type: str,
        total_agents: int,
        min_confidence: float = 0.5,
    ) -> bool:
        """Quick check: is this signal's consensus statistically reliable?"""
        supporters = self._space.get_contributor_count(signal_type)
        ci = self._confidence.calculate(supporters, total_agents)
        return ci.is_significant and ci.lower_bound >= min_confidence

    # --- Statistics ---

    def get_statistics(self) -> dict[str, Any]:
        return {
            "agent_id": self._agent_id,
            "total_deposits": self._total_deposits,
            "total_senses": self._total_senses,
            "rate_limit_violations": self._rate_limit_violations,
            "deposits_by_type": dict(self._deposits_by_type),
            "deposit_history_size": len(self._deposit_history),
            "sensing_history_size": len(self._sensing_history),
        }

    def get_recent_deposits(
        self, signal_type: str | None = None, limit: int = 10
    ) -> list[DepositRecord]:
        records = list(self._deposit_history)
        if signal_type:
            records = [r for r in records if r.signal_type == signal_type]
        return records[-limit:]

    def __repr__(self) -> str:
        return f"QuorumSensor(agent={self._agent_id}, deposits={self._total_deposits})"
