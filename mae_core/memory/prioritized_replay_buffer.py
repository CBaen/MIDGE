"""Prioritized Experience Replay Buffer.

Priority-based sampling using a SumTree for O(log N) operations.

    P(i) = priority_i^alpha / sum(priority_k^alpha)
    w_i  = (N * P(i))^(-beta) / max(w)

Based on: Schaul et al. (2016) "Prioritized Experience Replay", ICLR
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

import numpy as np

from .experience import Experience
from .sum_tree import SumTree

logger = logging.getLogger(__name__)


class PrioritizedReplayBuffer:
    """Prioritized replay buffer backed by SumTree.

    Experiences are stored directly in the SumTree's data array
    (circular buffer, capacity-bounded). No separate storage needed.

    Args:
        capacity: Max experiences to store.
        alpha: Prioritization exponent (0=uniform, 1=greedy). Default 0.6.
        beta: Importance-sampling correction (0=none, 1=full). Default 0.4.
        epsilon: Small constant added to priorities. Default 1e-6.
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        epsilon: float = 1e-6,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")
        if not 0 <= alpha <= 1:
            raise ValueError(f"Alpha must be in [0, 1], got {alpha}")
        if not 0 <= beta <= 1:
            raise ValueError(f"Beta must be in [0, 1], got {beta}")

        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self._lock = threading.RLock()

        self._tree = SumTree(capacity)
        self._max_priority = 1.0

    # -- public API --

    def add(self, experience: Experience, priority: float | None = None) -> None:
        """Add experience with given priority (defaults to max seen so far)."""
        if priority is None:
            priority = self._max_priority

        p = self._priority_to_alpha(priority)

        with self._lock:
            self._tree.add(p, experience)
            self._max_priority = max(self._max_priority, priority)

    def sample(
        self, batch_size: int, beta: float | None = None
    ) -> tuple[list[Experience], list[int], np.ndarray]:
        """Sample batch via stratified priority sampling.

        Returns:
            (experiences, data_indices, importance_weights)
        """
        if beta is None:
            beta = self.beta

        with self._lock:
            size = len(self._tree)
            if batch_size > size:
                raise ValueError(
                    f"Cannot sample {batch_size} from buffer of size {size}"
                )

            total = self._tree.total()
            if total <= 0:
                raise ValueError("Tree has zero total priority")

            segment = total / batch_size
            experiences: list[Experience] = []
            indices: list[int] = []
            priorities = np.empty(batch_size, dtype=np.float64)

            for i in range(batch_size):
                lo = segment * i
                hi = segment * (i + 1)
                value = np.random.uniform(lo, hi)

                data_idx, priority, exp = self._tree.get(value)
                experiences.append(exp)
                indices.append(data_idx)
                priorities[i] = priority

            # Importance-sampling weights: w_i = (N * P(i))^(-beta) / max(w)
            probs = priorities / total
            weights = (size * probs) ** (-beta)
            weights /= weights.max()

        return experiences, indices, weights

    def update_priorities(
        self, indices: list[int], td_errors: np.ndarray
    ) -> None:
        """Update priorities from TD errors: priority = |td_error| + epsilon."""
        if len(indices) != len(td_errors):
            raise ValueError(
                f"indices ({len(indices)}) and td_errors ({len(td_errors)}) "
                f"must have same length"
            )

        with self._lock:
            for idx, td in zip(indices, td_errors):
                priority = float(abs(td))
                p = self._priority_to_alpha(priority)
                self._tree.update(idx, p)
                self._max_priority = max(self._max_priority, priority)

    def update_beta(self, beta: float) -> None:
        """Anneal beta (typically 0.4 -> 1.0 over training)."""
        if not 0 <= beta <= 1:
            raise ValueError(f"Beta must be in [0, 1], got {beta}")
        self.beta = beta

    @property
    def max_priority(self) -> float:
        return self._max_priority

    def clear(self) -> None:
        """Remove all experiences."""
        with self._lock:
            self._tree = SumTree(self.capacity)
            self._max_priority = 1.0

    # -- internals --

    def _priority_to_alpha(self, priority: float) -> float:
        """Convert raw priority to alpha-adjusted tree priority."""
        return (abs(priority) + self.epsilon) ** self.alpha

    def __len__(self) -> int:
        return len(self._tree)

    def __repr__(self) -> str:
        return (
            f"PrioritizedReplayBuffer(capacity={self.capacity}, "
            f"size={len(self)}, alpha={self.alpha}, beta={self.beta})"
        )
