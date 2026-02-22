"""Working Memory - Human-like 7+/-2 capacity constraint (Miller's Law).

Limited-capacity active store with activation decay and rehearsal.
Information fades without active rehearsal. Higher-importance items
resist decay and displacement longer.

Biological analogy: Prefrontal cortex working memory buffer.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class WorkingMemorySlot:
    """Individual working memory slot with activation decay."""

    slot_id: str
    content: Any
    activation_level: float  # 0.0-1.0, decays over time
    created_at: float
    last_accessed: float
    access_count: int
    importance: float  # 0.0-1.0, for retention priority


class WorkingMemory:
    """Biological working memory with 7+/-2 capacity constraint.

    Args:
        capacity: Number of slots (5-9, default 7).
        decay_rate: Base activation decay per tick.
        rehearsal_boost: Activation boost from rehearsal.
        min_activation: Below this threshold, slot is cleared.
    """

    def __init__(
        self,
        capacity: int = 7,
        decay_rate: float = 0.1,
        rehearsal_boost: float = 0.3,
        min_activation: float = 0.2,
    ) -> None:
        if not 5 <= capacity <= 9:
            raise ValueError(f"Capacity must be 5-9 (Miller's Law), got {capacity}")

        self.capacity = capacity
        self.decay_rate = decay_rate
        self.rehearsal_boost = rehearsal_boost
        self.min_activation = min_activation

        self._slots: dict[str, WorkingMemorySlot] = {}
        self._lock = threading.RLock()

        # Counters (reset periodically to avoid unbounded growth)
        self._total_stored = 0
        self._items_decayed = 0
        self._successful_retrievals = 0
        self._failed_retrievals = 0

    # -- public API --

    def store(self, item_id: str, content: Any, importance: float = 0.5) -> bool:
        """Store item. Returns True if stored, False if full and couldn't displace."""
        importance = max(0.0, min(1.0, importance))
        now = time.time()

        with self._lock:
            # Update if already exists
            if item_id in self._slots:
                slot = self._slots[item_id]
                slot.content = content
                slot.activation_level = min(1.0, slot.activation_level + 0.1)
                slot.last_accessed = now
                slot.access_count += 1
                slot.importance = max(slot.importance, importance)
                return True

            # Free slot available
            if len(self._slots) < self.capacity:
                self._slots[item_id] = WorkingMemorySlot(
                    slot_id=item_id,
                    content=content,
                    activation_level=1.0,
                    created_at=now,
                    last_accessed=now,
                    access_count=1,
                    importance=importance,
                )
                self._total_stored += 1
                return True

            # At capacity - displace lowest if new item is more important
            lowest = self._find_lowest_slot()
            if lowest is not None and importance > lowest.importance:
                del self._slots[lowest.slot_id]
                self._slots[item_id] = WorkingMemorySlot(
                    slot_id=item_id,
                    content=content,
                    activation_level=1.0,
                    created_at=now,
                    last_accessed=now,
                    access_count=1,
                    importance=importance,
                )
                self._total_stored += 1
                return True

            return False

    def retrieve(self, item_id: str) -> Optional[Any]:
        """Retrieve item. Boosts activation on access. Returns None if absent/decayed."""
        with self._lock:
            slot = self._slots.get(item_id)
            if slot is None:
                self._failed_retrievals += 1
                return None

            if slot.activation_level < self.min_activation:
                del self._slots[item_id]
                self._items_decayed += 1
                self._failed_retrievals += 1
                return None

            slot.activation_level = min(1.0, slot.activation_level + 0.15)
            slot.last_accessed = time.time()
            slot.access_count += 1
            self._successful_retrievals += 1
            return slot.content

    def rehearse(self, item_ids: list[str] | None = None) -> None:
        """Rehearse items to prevent decay. None = rehearse all."""
        with self._lock:
            now = time.time()
            ids = item_ids if item_ids is not None else list(self._slots.keys())
            for item_id in ids:
                slot = self._slots.get(item_id)
                if slot is not None:
                    slot.activation_level = min(1.0, slot.activation_level + self.rehearsal_boost)
                    slot.last_accessed = now
                    slot.access_count += 1

    def decay_tick(self) -> int:
        """Apply time-based decay. Returns number of items removed."""
        with self._lock:
            now = time.time()
            to_remove: list[str] = []

            for item_id, slot in self._slots.items():
                elapsed = now - slot.last_accessed
                # Higher importance resists decay
                factor = self.decay_rate * (1.0 - slot.importance * 0.5)
                loss = factor * elapsed / 10.0
                slot.activation_level = max(0.0, slot.activation_level - loss)

                if slot.activation_level < self.min_activation:
                    to_remove.append(item_id)

            for item_id in to_remove:
                del self._slots[item_id]
                self._items_decayed += 1

            return len(to_remove)

    def get_active_items(self) -> list[WorkingMemorySlot]:
        """All current items sorted by activation (highest first)."""
        with self._lock:
            items = list(self._slots.values())
            items.sort(key=lambda s: s.activation_level, reverse=True)
            return items

    @property
    def available_slots(self) -> int:
        return self.capacity - len(self._slots)

    def clear(self) -> None:
        """Remove all items."""
        with self._lock:
            self._slots.clear()

    # -- internals --

    def _find_lowest_slot(self) -> Optional[WorkingMemorySlot]:
        """Find slot with lowest retention score (activation weighted by importance)."""
        if not self._slots:
            return None
        return min(
            self._slots.values(),
            key=lambda s: s.activation_level * (1.0 - s.importance * 0.3),
        )

    def __len__(self) -> int:
        return len(self._slots)

    def __repr__(self) -> str:
        return (
            f"WorkingMemory(capacity={self.capacity}, "
            f"used={len(self._slots)}, "
            f"available={self.available_slots})"
        )
