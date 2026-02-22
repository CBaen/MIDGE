"""Episodic Memory - Hippocampus-inspired memory with prioritized replay.

Hub connecting PrioritizedReplayBuffer (fast sampling), SemanticRetriever
(similarity search), and WorkingMemory (attention gating). Stores agent
experiences and enables prioritized replay for efficient learning.

Based on: Schaul et al. (2016) "Prioritized Experience Replay"
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

import numpy as np

from .experience import Experience
from .prioritized_replay_buffer import PrioritizedReplayBuffer
from .semantic_retriever import SemanticRetriever

logger = logging.getLogger(__name__)


class EpisodicMemory:
    """Episodic memory with prioritized experience replay.

    Core storage uses PrioritizedReplayBuffer (SumTree-backed, O(log N)).
    Optional semantic index enables similarity-based retrieval via FAISS.

    Args:
        capacity: Max experiences (default 100_000).
        alpha: PER prioritization exponent (0=uniform, 1=greedy).
        beta: Importance-sampling correction (0=none, 1=full).
        use_semantic_index: Enable FAISS-backed similarity search.
        semantic_retriever: Existing SemanticRetriever, or auto-created.
    """

    def __init__(
        self,
        capacity: int = 100_000,
        alpha: float = 0.6,
        beta: float = 0.4,
        epsilon: float = 1e-6,
        use_semantic_index: bool = False,
        semantic_retriever: SemanticRetriever | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")

        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta

        self._buffer = PrioritizedReplayBuffer(
            capacity=capacity, alpha=alpha, beta=beta, epsilon=epsilon
        )

        # Semantic index (optional)
        self.use_semantic_index = use_semantic_index
        if use_semantic_index and semantic_retriever is None:
            self._semantic = SemanticRetriever(embedding_dim=128)
            logger.info("Auto-created SemanticRetriever (128-dim)")
        else:
            self._semantic = semantic_retriever

        # Thread safety for compound operations (buffer + semantic)
        self._lock = threading.RLock()

        # Counters
        self.total_stored = 0
        self.total_sampled = 0

    # -- public API --

    def store(
        self,
        state: np.ndarray,
        action: Any,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: dict[str, Any] | None = None,
        priority: float | None = None,
    ) -> None:
        """Store a new experience."""
        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            info=info or {},
            timestamp=time.time(),
            priority=priority,
        )

        with self._lock:
            self._buffer.add(experience, priority=priority)

            if self.use_semantic_index and self._semantic is not None:
                self._semantic.add(experience)

            self.total_stored += 1

    def add(self, experience: Experience, priority: float | None = None) -> None:
        """Store a pre-built Experience object."""
        with self._lock:
            self._buffer.add(experience, priority=priority)

            if self.use_semantic_index and self._semantic is not None:
                self._semantic.add(experience)

            self.total_stored += 1

    def sample(
        self, batch_size: int = 32, beta: float | None = None
    ) -> tuple[list[Experience], list[int], np.ndarray]:
        """Sample a prioritized batch.

        Returns:
            (experiences, buffer_indices, importance_weights)
        """
        if len(self._buffer) == 0:
            raise ValueError("Cannot sample from empty memory")
        if batch_size > len(self._buffer):
            raise ValueError(
                f"Cannot sample {batch_size} from {len(self._buffer)} experiences"
            )

        batch, indices, weights = self._buffer.sample(batch_size, beta=beta)
        self.total_sampled += batch_size
        return batch, indices, weights

    def update_priorities(
        self, indices: list[int], td_errors: np.ndarray
    ) -> None:
        """Update priorities from TD errors (|td_error| + epsilon)."""
        self._buffer.update_priorities(indices, td_errors)

    def update_beta(self, beta: float) -> None:
        """Anneal beta (typically 0.4 -> 1.0 over training)."""
        self.beta = beta
        self._buffer.update_beta(beta)

    def get_similar_experiences(
        self, state: np.ndarray, k: int = 5
    ) -> list[Experience]:
        """Retrieve k most similar experiences by state (semantic search)."""
        if not self.use_semantic_index or self._semantic is None:
            return []

        with self._lock:
            result = self._semantic.search_by_state(state, k=k)
        return result.experiences

    def get_counterfactual_experiences(
        self, state: np.ndarray, action: Any, k: int = 5
    ) -> list[Experience]:
        """Find similar states where a different action was taken."""
        if not self.use_semantic_index or self._semantic is None:
            return []

        with self._lock:
            result = self._semantic.get_counterfactual_experiences(state, action, k=k)
        return result.experiences

    def set_semantic_retriever(self, retriever: SemanticRetriever) -> None:
        """Attach a semantic retriever after construction."""
        self._semantic = retriever
        self.use_semantic_index = True

    @property
    def max_priority(self) -> float:
        return self._buffer.max_priority

    def clear(self) -> None:
        """Remove all experiences."""
        with self._lock:
            self._buffer.clear()
            if self._semantic is not None:
                self._semantic.clear()
            self.total_stored = 0
            self.total_sampled = 0

    # -- persistence --

    def serialize(self, persist_dir: "Path") -> dict:
        """Save episodic memory state to disk. Returns JSON-safe metadata."""
        import pickle
        from pathlib import Path

        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        # Save SumTree state (tree array, data, write_idx, n_entries)
        tree = self._buffer._tree
        tree_state = {
            "tree_array": tree.tree,
            "data": tree.data,
            "write_idx": tree.write_idx,
            "n_entries": tree.n_entries,
            "capacity": tree.capacity,
        }
        buf_state = {
            "tree_state": tree_state,
            "max_priority": self._buffer._max_priority,
            "alpha": self._buffer.alpha,
            "beta": self._buffer.beta,
            "epsilon": self._buffer.epsilon,
        }
        with open(persist_dir / "episodic_memory.pkl", "wb") as f:
            pickle.dump(buf_state, f, protocol=pickle.HIGHEST_PROTOCOL)

        return {
            "total_stored": self.total_stored,
            "total_sampled": self.total_sampled,
            "capacity": self.capacity,
            "alpha": self.alpha,
            "beta": self.beta,
            "size": len(self._buffer),
        }

    def restore(self, persist_dir: "Path", metadata: dict) -> None:
        """Restore episodic memory state from disk."""
        import pickle
        from pathlib import Path

        persist_dir = Path(persist_dir)
        path = persist_dir / "episodic_memory.pkl"
        if not path.exists():
            logger.warning("No episodic memory file at %s, starting fresh", path)
            return

        try:
            with open(path, "rb") as f:
                buf_state = pickle.load(f)

            # Restore SumTree
            tree = self._buffer._tree
            ts = buf_state["tree_state"]
            if ts["capacity"] == tree.capacity:
                tree.tree[:] = ts["tree_array"]
                tree.data[:] = ts["data"]
                tree.write_idx = ts["write_idx"]
                tree.n_entries = ts["n_entries"]
            else:
                logger.warning(
                    "Capacity mismatch (%d vs %d), skipping restore",
                    ts["capacity"], tree.capacity,
                )
                return

            self._buffer._max_priority = buf_state.get("max_priority", 1.0)
            self.total_stored = metadata.get("total_stored", 0)
            self.total_sampled = metadata.get("total_sampled", 0)
            logger.info("Restored episodic memory (%d entries)", tree.n_entries)
        except Exception:
            logger.warning("Failed to restore episodic memory", exc_info=True)

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return (
            f"EpisodicMemory(capacity={self.capacity}, size={len(self)}, "
            f"alpha={self.alpha}, beta={self.beta}, "
            f"semantic={'on' if self.use_semantic_index else 'off'})"
        )
