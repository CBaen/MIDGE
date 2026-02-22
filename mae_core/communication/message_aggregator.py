"""Message Aggregator - Deduplication and priority-based message handling.

Deduplicates messages by content hash. Same content from multiple agents
increases urgency (voting). Automatic TTL-based cleanup.

Biological analogy: Chemical signal concentration from multiple sources.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class MessagePriority(Enum):
    """Message priority levels."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


@dataclass
class AggregatedMessage:
    """A deduplicated message with vote count and urgency."""

    content_hash: str
    message_type: str
    content: dict[str, Any]
    supporters: set[str] = field(default_factory=set)
    vote_count: int = 0
    urgency_score: float = 0.0  # [0, 1]
    priority: MessagePriority = MessagePriority.NORMAL
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    times_read: int = 0
    agents_who_read: set[str] = field(default_factory=set)


class MessageAggregator:
    """Deduplicates and prioritizes messages from multiple agents.

    Same content from different agents = same message with higher urgency.
    Provides anti-spam and voting consensus.
    """

    def __init__(
        self,
        max_messages: int = 10000,
        max_message_age: float = 600.0,
        decay_interval: float = 60.0,
    ) -> None:
        self._max_messages = max_messages
        self._max_age = max_message_age
        self._decay_interval = decay_interval

        self._messages: OrderedDict[str, AggregatedMessage] = OrderedDict()
        self._lock = threading.RLock()
        self._last_decay = time.time()

        # Statistics
        self._total_submitted = 0
        self._total_deduplicated = 0

    def submit_message(
        self,
        agent_id: str,
        message_type: str,
        content: dict[str, Any],
        allow_duplicate_vote: bool = False,
    ) -> tuple[str, bool, AggregatedMessage]:
        """Submit a message. Returns (content_hash, is_new_vote, aggregated_msg)."""
        content_hash = self._hash_content(message_type, content)

        with self._lock:
            self._maybe_decay()
            self._total_submitted += 1

            if content_hash in self._messages:
                msg = self._messages[content_hash]
                is_new = agent_id not in msg.supporters or allow_duplicate_vote

                if is_new:
                    msg.supporters.add(agent_id)
                    msg.vote_count += 1
                    msg.last_seen = time.time()
                    self._total_deduplicated += 1

                    # Update urgency: blend of recency and consensus
                    recency = max(0.0, 1.0 - (time.time() - msg.first_seen) / self._max_age)
                    consensus = min(1.0, msg.vote_count / 10.0)
                    msg.urgency_score = 0.4 * recency + 0.6 * consensus

                    # Upgrade priority at vote thresholds
                    if msg.vote_count >= 10:
                        msg.priority = MessagePriority.CRITICAL
                    elif msg.vote_count >= 5:
                        msg.priority = MessagePriority.URGENT
                    elif msg.vote_count >= 3:
                        msg.priority = MessagePriority.HIGH

                return content_hash, is_new, msg

            # New message
            msg = AggregatedMessage(
                content_hash=content_hash,
                message_type=message_type,
                content=content,
                supporters={agent_id},
                vote_count=1,
                urgency_score=0.1,
            )
            self._messages[content_hash] = msg

            # Evict oldest if at capacity
            while len(self._messages) > self._max_messages:
                self._messages.popitem(last=False)

            return content_hash, True, msg

    def get_messages(
        self,
        message_type: str | None = None,
        min_priority: MessagePriority | None = None,
        min_votes: int = 1,
        max_results: int = 50,
        mark_read_by: str | None = None,
    ) -> list[AggregatedMessage]:
        """Get messages matching criteria, sorted by urgency."""
        with self._lock:
            results = []
            for msg in self._messages.values():
                if message_type and msg.message_type != message_type:
                    continue
                if min_priority and msg.priority.value < min_priority.value:
                    continue
                if msg.vote_count < min_votes:
                    continue
                results.append(msg)

            results.sort(key=lambda m: m.urgency_score, reverse=True)
            results = results[:max_results]

            if mark_read_by:
                for msg in results:
                    msg.times_read += 1
                    msg.agents_who_read.add(mark_read_by)

            return results

    def get_top_messages(
        self, count: int = 5, mark_read_by: str | None = None
    ) -> list[AggregatedMessage]:
        """Get top-N messages by urgency."""
        return self.get_messages(max_results=count, mark_read_by=mark_read_by)

    def get_message_by_hash(self, content_hash: str) -> AggregatedMessage | None:
        return self._messages.get(content_hash)

    def get_consensus_strength(
        self, message_type: str, min_votes: int = 3
    ) -> float:
        """Overall consensus strength for a message type. Returns [0, 1]."""
        with self._lock:
            matching = [
                m for m in self._messages.values()
                if m.message_type == message_type and m.vote_count >= min_votes
            ]
            if not matching:
                return 0.0
            return max(m.urgency_score for m in matching)

    def get_statistics(self) -> dict[str, Any]:
        with self._lock:
            return {
                "active_messages": len(self._messages),
                "total_submitted": self._total_submitted,
                "total_deduplicated": self._total_deduplicated,
                "dedup_rate": (
                    self._total_deduplicated / self._total_submitted
                    if self._total_submitted > 0 else 0.0
                ),
                "max_messages": self._max_messages,
            }

    def _hash_content(self, message_type: str, content: dict[str, Any]) -> str:
        """Create deterministic hash for message deduplication."""
        key = f"{message_type}:{sorted(content.items())}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def _maybe_decay(self) -> None:
        """Remove old messages periodically."""
        now = time.time()
        if now - self._last_decay < self._decay_interval:
            return

        self._last_decay = now
        cutoff = now - self._max_age
        to_remove = [
            h for h, m in self._messages.items() if m.last_seen < cutoff
        ]
        for h in to_remove:
            del self._messages[h]

    def __repr__(self) -> str:
        return f"MessageAggregator(messages={len(self._messages)}, submitted={self._total_submitted})"
