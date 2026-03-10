"""4D Temporal Memory - Timeline of causally-linked events across spacetime.

Biological analogy: Hippocampal time cells + place cells combined.
Time cells fire at specific moments during an experience. Place cells
fire at spatial locations. Together they create a 4D "where + when"
map of events. This gives Mae a spatiotemporal memory that standard
episodic memory lacks.

Key insight: Events don't just happen, they CAUSE other events.
4D Temporal Memory tracks these causal chains, enabling Mae to
reason about "what caused what" and "what will happen next."

Connection points:
- EventBus publishes temporal.event_recorded for each stored event
- CausalEngine receives temporal causal links (A happened before B)
- EpisodicMemory receives events as experiences via bridge
- WorldModel uses temporal patterns for prediction improvement
- DecisionRouter uses temporal context for deliberation
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

# EventBus channels
CH_TEMPORAL_EVENT_RECORDED = "temporal.event_recorded"
CH_TEMPORAL_CAUSAL_LINK = "temporal.causal_link_discovered"
CH_TEMPORAL_PATTERN_DETECTED = "temporal.pattern_detected"


class EventType(Enum):
    """Categories of temporal events."""

    ACTION = "action"  # Agent took an action
    OBSERVATION = "observation"  # Agent observed something
    COMMUNICATION = "communication"  # Signal sent/received
    STATE_CHANGE = "state_change"  # Environment state changed
    HEALING = "healing"  # Auto-healer activated
    THREAT = "threat"  # Threat detected
    GROWTH = "growth"  # Morphogenesis event
    DECISION = "decision"  # Decision router choice
    CONSOLIDATION = "consolidation"  # Memory consolidation
    CUSTOM = "custom"  # User-defined


@dataclass
class FourDEvent:
    """A spatiotemporal event with causal links.

    The 4D: (x, y, z) spatial coordinates + t (time).
    Like a neuron firing in a specific place at a specific moment,
    connected to other firings that caused it or that it caused.
    """

    event_id: str
    entity_id: str  # Agent or system that generated this event
    event_type: EventType
    timestamp: float = field(default_factory=time.time)

    # Spatial coordinates (4D: position + time)
    position: tuple[float, ...] = (0.0, 0.0)  # (x, y) or (x, y, z)

    # Event data
    data: dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5  # [0, 1] - affects retention priority

    # Causal links - what caused this and what this caused
    causal_predecessors: list[str] = field(default_factory=list)  # event_ids
    causal_successors: list[str] = field(default_factory=list)  # event_ids

    # Temporal neighbors (events close in time, not necessarily causal)
    temporal_neighbors: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.importance = max(0.0, min(1.0, self.importance))


@dataclass
class TemporalPattern:
    """A recurring pattern detected in the temporal stream.

    Like recognizing that "every time cortisol spikes, agents
    redistribute load within 5 steps" - a temporal regularity.
    """

    pattern_id: str
    event_sequence: list[EventType]  # Ordered event types in pattern
    avg_interval: float  # Average time between events in pattern
    occurrence_count: int = 0
    confidence: float = 0.0  # [0, 1]
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    entity_ids: list[str] = field(default_factory=list)  # Entities involved


@dataclass
class CausalChain:
    """A chain of causally linked events through time."""

    chain_id: str
    events: list[FourDEvent]  # Ordered by timestamp
    total_duration: float = 0.0
    root_cause: str = ""  # event_id of the first event

    def __post_init__(self) -> None:
        if self.events and not self.total_duration:
            self.total_duration = self.events[-1].timestamp - self.events[0].timestamp
        if self.events and not self.root_cause:
            self.root_cause = self.events[0].event_id


from .temporal_causal import TemporalCausalMixin


class TemporalMemory(TemporalCausalMixin):
    """4D Temporal Memory - spatiotemporal event timeline with causal reasoning.

    Stores events in a timeline, discovers causal chains, detects
    temporal patterns, and bridges to other memory systems.

    Like the hippocampus maintaining a running timeline of experience,
    with time cells encoding WHEN and place cells encoding WHERE.
    """

    def __init__(
        self,
        event_bus: Any = None,
        causal_engine: Any = None,
        max_events: int = 50_000,
        temporal_window: float = 5.0,
        causal_window: float = 10.0,
        pattern_min_occurrences: int = 3,
    ) -> None:
        self._bus = event_bus
        self._causal = causal_engine
        self._max_events = max_events
        self._temporal_window = temporal_window  # Seconds for temporal neighbor search
        self._causal_window = causal_window  # Seconds for causal link detection
        self._pattern_min = pattern_min_occurrences

        # Primary storage - ordered by timestamp
        self._events: deque[FourDEvent] = deque(maxlen=max_events)
        self._events_by_id: dict[str, FourDEvent] = {}

        # Indices for fast lookup
        self._events_by_entity: dict[str, list[str]] = defaultdict(list)
        self._events_by_type: dict[EventType, list[str]] = defaultdict(list)

        # Detected patterns
        self._patterns: dict[str, TemporalPattern] = {}
        self._pattern_counter = 0

        # Statistics
        self._total_recorded = 0
        self._causal_links_discovered = 0

        self._lock = threading.RLock()

        logger.info(
            "TemporalMemory initialized (max_events=%d, causal_window=%.1fs)",
            max_events,
            causal_window,
        )

    # =========================================================================
    # Event Recording
    # =========================================================================

    def record_event(self, event: FourDEvent) -> FourDEvent:
        """Record a new 4D event in the temporal memory.

        Automatically:
        - Links temporal neighbors (events within temporal_window)
        - Discovers potential causal links
        - Feeds causal engine if available
        - Publishes on EventBus
        """
        with self._lock:
            # Store
            self._events.append(event)
            self._events_by_id[event.event_id] = event
            self._events_by_entity[event.entity_id].append(event.event_id)
            self._events_by_type[event.event_type].append(event.event_id)
            self._total_recorded += 1

            # Find temporal neighbors
            self._link_temporal_neighbors(event)

            # Discover causal links
            self._discover_causal_links(event)

            # Detect patterns
            self._check_patterns(event)

            # Evict old entries from lookup indices if deque rolled over
            if len(self._events) == self._max_events:
                self._evict_oldest()

        # Publish event recorded
        if self._bus:
            self._bus.publish(CH_TEMPORAL_EVENT_RECORDED, {
                "event_id": event.event_id,
                "entity_id": event.entity_id,
                "event_type": event.event_type.value,
                "timestamp": event.timestamp,
                "importance": event.importance,
            })

        return event

    def _link_temporal_neighbors(self, event: FourDEvent) -> None:
        """Find events within temporal_window and link as neighbors."""
        cutoff = event.timestamp - self._temporal_window
        for other in reversed(self._events):
            if other.event_id == event.event_id:
                continue
            if other.timestamp < cutoff:
                break
            if other.event_id not in event.temporal_neighbors:
                event.temporal_neighbors.append(other.event_id)
            if event.event_id not in other.temporal_neighbors:
                other.temporal_neighbors.append(event.event_id)

    def _discover_causal_links(self, event: FourDEvent) -> None:
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

    def _check_patterns(self, event: FourDEvent) -> None:
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

    def _evict_oldest(self) -> None:
        """Clean up indices when deque evicts oldest events."""
        current_ids = {e.event_id for e in self._events}
        stale = [eid for eid in self._events_by_id if eid not in current_ids]
        for eid in stale:
            event = self._events_by_id.pop(eid, None)
            if event:
                elist = self._events_by_entity.get(event.entity_id, [])
                if eid in elist:
                    elist.remove(eid)
                tlist = self._events_by_type.get(event.event_type, [])
                if eid in tlist:
                    tlist.remove(eid)

    # =========================================================================
    # Querying
    # =========================================================================

    def get_event(self, event_id: str) -> FourDEvent | None:
        """Get a specific event by ID."""
        with self._lock:
            return self._events_by_id.get(event_id)

    def query_by_time_range(
        self, start: float, end: float, event_type: EventType | None = None
    ) -> list[FourDEvent]:
        """Get all events within a time range, optionally filtered by type."""
        with self._lock:
            results = []
            for event in self._events:
                if event.timestamp < start:
                    continue
                if event.timestamp > end:
                    break
                if event_type is None or event.event_type == event_type:
                    results.append(event)
            return results

    def query_by_entity(
        self, entity_id: str, limit: int = 50
    ) -> list[FourDEvent]:
        """Get recent events for an entity."""
        with self._lock:
            event_ids = self._events_by_entity.get(entity_id, [])[-limit:]
            return [
                self._events_by_id[eid]
                for eid in event_ids
                if eid in self._events_by_id
            ]

    def query_by_position(
        self, position: tuple[float, ...], radius: float
    ) -> list[FourDEvent]:
        """Get events near a spatial position."""
        with self._lock:
            results = []
            pos = np.array(position)
            for event in self._events:
                event_pos = np.array(event.position[: len(position)])
                if np.linalg.norm(pos - event_pos) <= radius:
                    results.append(event)
            return results

    def get_recent(self, count: int = 20) -> list[FourDEvent]:
        """Get the N most recent events."""
        with self._lock:
            return list(self._events)[-count:]

    # =========================================================================
    # Causal Chain Discovery
    # =========================================================================

    def trace_causal_chain(
        self, event_id: str, direction: str = "backward", max_depth: int = 10
    ) -> CausalChain:
        """Trace a causal chain forward or backward from an event.

        Like following a chain of dominoes - each event caused the next.

        Args:
            event_id: Starting event.
            direction: "backward" (find causes) or "forward" (find effects).
            max_depth: Maximum chain length to prevent infinite loops.
        """
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
    ) -> list[FourDEvent]:
        """Find events that are causal predecessors of BOTH A and B.

        Like finding "what caused both the fever and the cough"
        (the virus is the common cause).
        """
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

    # =========================================================================
    # Pattern Access
    # =========================================================================

    def get_patterns(self, min_confidence: float = 0.3) -> list[TemporalPattern]:
        """Get detected temporal patterns above a confidence threshold."""
        with self._lock:
            return [
                p
                for p in self._patterns.values()
                if p.confidence >= min_confidence
            ]

    def predict_next_event_type(self, entity_id: str) -> EventType | None:
        """Predict what type of event will happen next for an entity.

        Uses detected patterns to predict. If the entity's recent
        events match the start of a known pattern, predict the next.
        """
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

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_statistics(self) -> dict[str, Any]:
        """Get temporal memory statistics."""
        with self._lock:
            return {
                "total_recorded": self._total_recorded,
                "current_size": len(self._events),
                "max_events": self._max_events,
                "entities_tracked": len(self._events_by_entity),
                "event_types": {
                    t.value: len(ids)
                    for t, ids in self._events_by_type.items()
                },
                "causal_links_discovered": self._causal_links_discovered,
                "patterns_detected": len(self._patterns),
                "patterns_confident": len([
                    p for p in self._patterns.values()
                    if p.confidence >= 0.5
                ]),
            }

    def get_timeline(
        self, entity_id: str | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Get a human-readable timeline of recent events."""
        with self._lock:
            if entity_id:
                event_ids = self._events_by_entity.get(entity_id, [])[-limit:]
                events = [
                    self._events_by_id[eid]
                    for eid in event_ids
                    if eid in self._events_by_id
                ]
            else:
                events = list(self._events)[-limit:]

            return [
                {
                    "event_id": e.event_id,
                    "entity": e.entity_id,
                    "type": e.event_type.value,
                    "time": e.timestamp,
                    "position": e.position,
                    "importance": e.importance,
                    "causes": len(e.causal_predecessors),
                    "effects": len(e.causal_successors),
                    "neighbors": len(e.temporal_neighbors),
                }
                for e in events
            ]
