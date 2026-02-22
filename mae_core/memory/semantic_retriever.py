"""Semantic Retriever - Vector similarity search over experience embeddings.

Replaces the v5-pivot ChromaDB-based retriever with VectorStore (FAISS backbone).
Encodes experiences as fixed-dim vectors and supports state-based search,
counterfactual retrieval, and nearest-neighbor queries.

Biological analogy: Neocortical semantic memory - gist-based retrieval
by similarity rather than exact match.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from ..backbone.vector_store import VectorStore
from .experience import Experience

logger = logging.getLogger(__name__)


@dataclass
class SemanticQuery:
    """Result of a semantic search."""

    experiences: list[Experience] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    query_time: float = 0.0


class SemanticRetriever:
    """Vector similarity search over experience embeddings.

    Uses VectorStore (FAISS/numpy) for nearest-neighbor search.
    Experiences are encoded as fixed-dimension vectors combining
    state features, action, and reward signals.

    Args:
        vector_store: Shared VectorStore instance (or None to create one).
        embedding_dim: Dimension of experience embeddings. Default 128.
        agent_id: ID of the owning agent (for multi-agent filtering).
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        embedding_dim: int = 128,
        agent_id: str = "default",
    ) -> None:
        self._embedding_dim = embedding_dim
        self._agent_id = agent_id

        if vector_store is not None:
            self._store = vector_store
        else:
            self._store = VectorStore(
                embedding_dim=embedding_dim, metric="cosine"
            )

        # Local experience cache: policy_id -> Experience
        self._experiences: dict[str, Experience] = {}
        self._n_indexed = 0

    # -- public API --

    def add(self, experience: Experience, embedding: np.ndarray | None = None) -> str:
        """Index an experience for similarity search.

        Returns the assigned experience ID.
        """
        exp_id = f"exp-{self._n_indexed:08d}-{uuid.uuid4().hex[:6]}"

        if embedding is None:
            embedding = self.encode_experience(experience)

        self._store.add_policy_embedding(
            policy_id=exp_id,
            agent_id=self._agent_id,
            embedding=embedding,
            metadata={
                "reward": float(experience.reward),
                "done": experience.done,
                "timestamp": experience.timestamp,
                "action": int(experience.action) if isinstance(experience.action, (int, float)) else hash(str(experience.action)) % 2**31,
            },
        )

        self._experiences[exp_id] = experience
        self._n_indexed += 1
        return exp_id

    def search_by_state(
        self, state: np.ndarray, k: int = 5
    ) -> SemanticQuery:
        """Find k experiences most similar to a query state."""
        t0 = time.time()
        query_vec = self.encode_state(state)
        results = self._store.search_similar_policies(
            query_embedding=query_vec,
            top_k=k,
            filter_criteria={"agent_id": self._agent_id},
        )

        experiences = []
        scores = []
        for r in results:
            exp = self._experiences.get(r.policy_id)
            if exp is not None:
                experiences.append(exp)
                scores.append(r.similarity_score)

        return SemanticQuery(
            experiences=experiences,
            scores=scores,
            query_time=time.time() - t0,
        )

    def get_counterfactual_experiences(
        self, state: np.ndarray, action: Any, k: int = 5
    ) -> SemanticQuery:
        """Find similar states where a *different* action was taken.

        Useful for off-policy learning and exploration analysis.
        """
        t0 = time.time()
        query_vec = self.encode_state(state)
        # Fetch more than k so we can filter
        results = self._store.search_similar_policies(
            query_embedding=query_vec,
            top_k=k * 3,
            filter_criteria={"agent_id": self._agent_id},
        )

        experiences = []
        scores = []
        for r in results:
            exp = self._experiences.get(r.policy_id)
            if exp is None:
                continue
            # Keep only experiences with different actions
            if np.isscalar(exp.action) and np.isscalar(action):
                if exp.action == action:
                    continue
            elif isinstance(exp.action, np.ndarray) and isinstance(action, np.ndarray):
                if np.allclose(exp.action, action):
                    continue

            experiences.append(exp)
            scores.append(r.similarity_score)
            if len(experiences) >= k:
                break

        return SemanticQuery(
            experiences=experiences,
            scores=scores,
            query_time=time.time() - t0,
        )

    def encode_experience(self, experience: Experience) -> np.ndarray:
        """Encode a full experience into a fixed-dim vector."""
        state_part = self._truncate_pad(experience.state.flatten(), 64)
        next_state_part = self._truncate_pad(experience.next_state.flatten(), 48)

        raw_action = experience.action
        if isinstance(raw_action, (int, float)):
            action_val = [float(raw_action)]
        elif isinstance(raw_action, (str, dict)):
            # Hash non-numeric actions into a numeric fingerprint
            action_val = [float(hash(str(raw_action)) % 2**31) / 2**31]
        elif hasattr(raw_action, '__iter__'):
            try:
                action_val = [float(x) for x in raw_action]
            except (TypeError, ValueError):
                action_val = [float(hash(str(raw_action)) % 2**31) / 2**31]
        else:
            action_val = [0.0]
        action_part = self._truncate_pad(np.asarray(action_val, dtype=np.float64).flatten(), 8)

        scalar_part = np.array(
            [experience.reward, float(experience.done)], dtype=np.float64
        )

        # Pad remaining dims to reach embedding_dim
        combined = np.concatenate([state_part, next_state_part, action_part, scalar_part])
        vec = self._truncate_pad(combined, self._embedding_dim)

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32)

    def encode_state(self, state: np.ndarray) -> np.ndarray:
        """Encode a state vector for search (query-side encoding)."""
        state_part = self._truncate_pad(state.flatten(), 64)
        # Zero-pad the remaining dims to match experience encoding layout
        vec = self._truncate_pad(state_part, self._embedding_dim)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32)

    def clear(self) -> None:
        """Remove all indexed experiences."""
        self._store.clear_collection()
        self._experiences.clear()
        self._n_indexed = 0

    @property
    def n_indexed(self) -> int:
        return self._n_indexed

    # -- internals --

    @staticmethod
    def _truncate_pad(arr: np.ndarray, length: int) -> np.ndarray:
        """Truncate or zero-pad array to exact length."""
        arr = np.asarray(arr, dtype=np.float64).flatten()
        if len(arr) >= length:
            return arr[:length]
        return np.pad(arr, (0, length - len(arr)))

    # -- persistence --

    def serialize(self, persist_dir: "Path") -> dict:
        """Save semantic retriever state to disk. Returns JSON-safe metadata."""
        import pickle
        from pathlib import Path

        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "experiences": self._experiences,
            "n_indexed": self._n_indexed,
            "embedding_dim": self._embedding_dim,
            "agent_id": self._agent_id,
        }
        with open(persist_dir / "semantic_retriever.pkl", "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Also persist the underlying VectorStore
        self._store.save(persist_dir / "semantic_vector_store.pkl")

        return {
            "n_indexed": self._n_indexed,
            "embedding_dim": self._embedding_dim,
            "agent_id": self._agent_id,
        }

    def restore(self, persist_dir: "Path", metadata: dict) -> None:
        """Restore semantic retriever state from disk."""
        import pickle
        from pathlib import Path

        persist_dir = Path(persist_dir)
        path = persist_dir / "semantic_retriever.pkl"
        if not path.exists():
            logger.warning("No semantic retriever file at %s, starting fresh", path)
            return

        try:
            with open(path, "rb") as f:
                state = pickle.load(f)

            self._experiences = state["experiences"]
            self._n_indexed = state["n_indexed"]

            # Restore VectorStore if file exists
            vs_path = persist_dir / "semantic_vector_store.pkl"
            if vs_path.exists():
                self._store.load(vs_path)

            logger.info("Restored semantic retriever (%d indexed)", self._n_indexed)
        except Exception:
            logger.warning("Failed to restore semantic retriever", exc_info=True)

    def __len__(self) -> int:
        return self._n_indexed

    def __repr__(self) -> str:
        return (
            f"SemanticRetriever(dim={self._embedding_dim}, "
            f"indexed={self._n_indexed}, agent={self._agent_id})"
        )
