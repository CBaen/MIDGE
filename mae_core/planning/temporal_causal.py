"""Temporal causal discovery mixin — causal link detection, pattern checking,
chain tracing, common cause finding, and next-event prediction.

Extracted from temporal_memory.py to respect the 500-line cap.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .temporal_memory import CausalChain, EventType, FourDEvent

logger = logging.getLogger(__name__)


class TemporalCausalMixin:
    """Causal discovery and pattern detection for TemporalMemory."""

    def _discover_causal_links(self, event: "FourDEvent") -> None:
        """Discover potential causal links from recent events to this one.

        Heuristic: If event B occurs within causal_window after event A,
        and they share the same entity OR are spatially close, there's
        a potential causal link A -> B.
        """
        cutoff = event.timestamp - self._causal_window
        for other in reversed(self._events):
            if other.event_id == event.event_id:
                continue
            if other.timestamp < cutoff:
                break
            if other.timestamp >= event.timestamp:
                continue  # Must be before this event

            # Same entity chain - likely causal
            if other.entity_id == event.entity_id:
                if other.event_id not in event.causal_predecessors:
                    event.causal_predecessors.append(other.event_id)
                if event.event_id not in other.causal_successors:
                    other.causal_successors.append(event.event_id)
                    self._causal_links_discovered += 1

                    # Feed causal engine
                    if self._causal:
                        self._causal.observe_correlation(
                            other.event_type.value,
                            event.event_type.value,
                            correlation_strength=0.6,
                        )

    def _check_patterns(self, event: "FourDEvent") -> None:
        """Check if this event completes or extends a temporal pattern."""
        entity_events = self._events_by_entity.get(event.entity_id, [])
        if len(entity_events) < 3:
            return

        # Look at the last N event types for this entity
        recent_ids = entity_events[-5:]
        recent_types = []
        for eid in recent_ids:
            e = self._events_by_id.get(eid)
            if e:
                recent_types.append(e.event_type)

        if len(recent_types) < 3:
            return

        from .temporal_memory import TemporalPattern, CH_TEMPORAL_PATTERN_DETECTED

        # Check for recurring 2-3 event sequences
        for seq_len in (2, 3):
            if len(recent_types) >= seq_len * 2:
                recent_seq = tuple(recent_types[-seq_len:])
                prior_seq = tuple(recent_types[-seq_len * 2 : -seq_len])
                if recent_seq == prior_seq:
                    pattern_key = "|".join(t.value for t in recent_seq)
                    if pattern_key not in self._patterns:
                        self._pattern_counter += 1
                        self._patterns[pattern_key] = TemporalPattern(
                            pattern_id=f"pat-{self._pattern_counter}",
                            event_sequence=list(recent_seq),
                            avg_interval=0.0,
                            occurrence_count=2,
                            confidence=0.4,
                            entity_ids=[event.entity_id],
                        )
                    else:
                        p = self._patterns[pattern_key]
                        p.occurrence_count += 1
                        p.last_seen = event.timestamp
                        p.confidence = min(1.0, p.occurrence_count / 10.0)
                        if event.entity_id not in p.entity_ids:
                            p.entity_ids.append(event.entity_id)

                        if (
                            p.occurrence_count >= self._pattern_min
                            and self._bus
                        ):
                            self._bus.publish(CH_TEMPORAL_PATTERN_DETECTED, {
                                "pattern_id": p.pattern_id,
                                "sequence": [t.value for t in p.event_sequence],
                                "occurrences": p.occurrence_count,
                                "confidence": p.confidence,
                            })

    def trace_causal_chain(
        self, event_id: str, direction: str = "backward", max_depth: int = 10
    ) -> "CausalChain":
        """Trace a causal chain forward or backward from an event.

        Like following a chain of dominoes - each event caused the next.
        """
        from .temporal_memory import CausalChain

        with self._lock:
            visited = set()
            chain_events = []
            queue = deque([event_id])

            while queue and len(chain_events) < max_depth:
                current_id = queue.popleft()
                if current_id in visited:
                    continue
                visited.add(current_id)

                event = self._events_by_id.get(current_id)
                if not event:
                    continue

                chain_events.append(event)

                if direction == "backward":
                    for pred_id in event.causal_predecessors:
                        if pred_id not in visited:
                            queue.append(pred_id)
                else:
                    for succ_id in event.causal_successors:
                        if succ_id not in visited:
                            queue.append(succ_id)

            # Sort by timestamp
            chain_events.sort(key=lambda e: e.timestamp)

            return CausalChain(
                chain_id=f"chain-{event_id}",
                events=chain_events,
            )

    def find_common_causes(
        self, event_id_a: str, event_id_b: str
    ) -> list["FourDEvent"]:
        """Find events that are causal predecessors of BOTH A and B."""
        chain_a = self.trace_causal_chain(event_id_a, direction="backward")
        chain_b = self.trace_causal_chain(event_id_b, direction="backward")

        ids_a = {e.event_id for e in chain_a.events}
        ids_b = {e.event_id for e in chain_b.events}
        common_ids = ids_a & ids_b

        return [
            self._events_by_id[eid]
            for eid in common_ids
            if eid in self._events_by_id
        ]

    def predict_next_event_type(self, entity_id: str) -> "EventType | None":
        """Predict what type of event will happen next for an entity."""
        with self._lock:
            entity_events = self._events_by_entity.get(entity_id, [])
            if not entity_events:
                return None

            recent_ids = entity_events[-3:]
            recent_types = []
            for eid in recent_ids:
                e = self._events_by_id.get(eid)
                if e:
                    recent_types.append(e.event_type)

            if not recent_types:
                return None

            # Check patterns for matching prefix
            best_match = None
            best_confidence = 0.0

            for pattern in self._patterns.values():
                seq = pattern.event_sequence
                for i in range(1, len(seq)):
                    prefix = seq[:i]
                    suffix = recent_types[-len(prefix):]
                    if len(suffix) == len(prefix) and all(
                        a == b for a, b in zip(suffix, prefix)
                    ):
                        if i < len(seq) and pattern.confidence > best_confidence:
                            best_match = seq[i]
                            best_confidence = pattern.confidence

            return best_match
