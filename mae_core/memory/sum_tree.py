"""SumTree - Binary sum tree for O(log N) prioritized sampling.

Each leaf stores an experience priority. Internal nodes store the sum
of their children. Enables proportional sampling and O(log N) updates.

Based on: Schaul et al. (2016) "Prioritized Experience Replay"
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class SumTree:
    """Binary sum tree stored as a flat array.

    Array layout: [root, left, right, ll, lr, rl, rr, ...]
    Leaf nodes start at index (capacity - 1).
    Parent of node i: (i-1) // 2
    Children of node i: 2*i+1 (left), 2*i+2 (right)
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")

        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data: list[Any] = [None] * capacity
        self.write_idx = 0
        self.n_entries = 0

        # Periodic resync to combat floating-point drift
        self._update_count = 0
        self._resync_interval = max(10_000, capacity // 10)

    def total(self) -> float:
        """Sum of all priorities (root node)."""
        return float(self.tree[0])

    def add(self, priority: float, data: Any) -> None:
        """Add data with given priority at next circular position."""
        if priority < 0:
            raise ValueError(f"Priority must be non-negative, got {priority}")

        idx = self.write_idx
        self.data[idx] = data
        self._update_leaf(idx, priority)
        self.write_idx = (self.write_idx + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, data_idx: int, priority: float) -> None:
        """Update priority at data_idx and propagate to root."""
        if priority < 0:
            raise ValueError(f"Priority must be non-negative, got {priority}")
        if not 0 <= data_idx < self.capacity:
            raise IndexError(f"Index {data_idx} out of range [0, {self.capacity})")

        self._update_leaf(data_idx, priority)

    def get(self, value: float) -> tuple[int, float, Any]:
        """Sample by value in [0, total). Returns (data_idx, priority, data)."""
        tree_idx = self._retrieve(value)
        data_idx = tree_idx - self.capacity + 1
        return data_idx, float(self.tree[tree_idx]), self.data[data_idx]

    def get_priority(self, data_idx: int) -> float:
        """Get priority at data_idx."""
        return float(self.tree[data_idx + self.capacity - 1])

    def get_max_priority(self) -> float:
        """Max priority across all entries (1.0 if empty)."""
        if self.n_entries == 0:
            return 1.0
        leaf_start = self.capacity - 1
        return float(np.max(self.tree[leaf_start : leaf_start + self.n_entries]))

    # -- internals --

    def _update_leaf(self, data_idx: int, priority: float) -> None:
        tree_idx = data_idx + self.capacity - 1
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        if change != 0:
            self._propagate(tree_idx, change)

        self._update_count += 1
        if self._update_count >= self._resync_interval:
            self._resync()
            self._update_count = 0

    def _propagate(self, idx: int, change: float) -> None:
        while idx > 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def _retrieve(self, value: float) -> int:
        idx = 0
        while True:
            left = 2 * idx + 1
            if left >= len(self.tree):
                return idx
            left_val = self.tree[left]
            if left_val == 0 or value >= left_val:
                value -= left_val
                idx = left + 1  # right child
            else:
                idx = left

    def _resync(self) -> None:
        """Rebuild internal nodes from leaves to fix float drift."""
        for i in range(self.capacity - 2, -1, -1):
            left = 2 * i + 1
            right = 2 * i + 2
            self.tree[i] = self.tree[left] + (
                self.tree[right] if right < len(self.tree) else 0.0
            )

    def __len__(self) -> int:
        return self.n_entries

    def __repr__(self) -> str:
        return f"SumTree(capacity={self.capacity}, entries={self.n_entries}, total={self.total():.2f})"
