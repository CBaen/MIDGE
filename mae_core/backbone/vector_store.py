"""VectorStore - FAISS-backed vector search replacing ChromaDB.

Provides similarity search for policy embeddings and experience
memories using FAISS (in-memory, no server required).

Implements the VectorDBInterface from the original codebase.
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# FAISS is optional - fall back to numpy if not installed
try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.info("FAISS not installed, using numpy fallback for vector search")


@dataclass
class PolicyEmbedding:
    """A stored embedding with metadata."""

    policy_id: str
    agent_id: str
    embedding: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    performance: float = 0.0


@dataclass
class SearchResult:
    """A search result with similarity score."""

    policy_id: str
    agent_id: str
    similarity_score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None


class VectorStore:
    """FAISS-backed vector store with numpy fallback.

    Stores policy embeddings and supports similarity search.
    No external server required - everything in-memory with
    optional persistence via pickle.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        metric: str = "cosine",
        persist_path: Optional[Path] = None,
    ) -> None:
        self._embedding_dim = embedding_dim
        self._metric = metric
        self._persist_path = persist_path

        # Storage
        self._policies: dict[str, PolicyEmbedding] = {}
        self._agent_index: dict[str, set[str]] = {}  # agent_id -> set of policy_ids

        # FAISS index (or None for numpy fallback)
        self._index: Any = None
        self._index_ids: list[str] = []  # Maps FAISS row -> policy_id
        self._index_dirty = True

        self._initialize_index()

    def _initialize_index(self) -> None:
        """Create the FAISS index."""
        if FAISS_AVAILABLE:
            if self._metric == "cosine":
                self._index = faiss.IndexFlatIP(self._embedding_dim)
            else:
                self._index = faiss.IndexFlatL2(self._embedding_dim)

    def _rebuild_index(self) -> None:
        """Rebuild FAISS index from current policies."""
        if not FAISS_AVAILABLE or not self._policies:
            self._index_dirty = False
            return

        self._initialize_index()
        self._index_ids = []

        embeddings = []
        for pid, policy in self._policies.items():
            vec = self._normalize(policy.embedding)
            embeddings.append(vec)
            self._index_ids.append(pid)

        if embeddings:
            matrix = np.array(embeddings, dtype=np.float32)
            self._index.add(matrix)

        self._index_dirty = False

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        """Normalize vector for cosine similarity."""
        vec = np.array(vec, dtype=np.float32).flatten()

        # Pad or truncate to embedding_dim
        if len(vec) < self._embedding_dim:
            vec = np.pad(vec, (0, self._embedding_dim - len(vec)))
        elif len(vec) > self._embedding_dim:
            vec = vec[: self._embedding_dim]

        if self._metric == "cosine":
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm

        return vec

    # --- Core Operations ---

    def add_policy_embedding(
        self,
        policy_id: str,
        agent_id: str,
        embedding: np.ndarray,
        metadata: Optional[dict[str, Any]] = None,
        performance: float = 0.0,
    ) -> bool:
        """Store a policy embedding."""
        policy = PolicyEmbedding(
            policy_id=policy_id,
            agent_id=agent_id,
            embedding=np.array(embedding, dtype=np.float32),
            metadata=metadata or {},
            timestamp=time.time(),
            performance=performance,
        )

        self._policies[policy_id] = policy

        if agent_id not in self._agent_index:
            self._agent_index[agent_id] = set()
        self._agent_index[agent_id].add(policy_id)

        self._index_dirty = True
        return True

    def add_policy_embeddings_batch(self, embeddings: list[PolicyEmbedding]) -> int:
        """Store multiple policy embeddings. Returns count added."""
        for emb in embeddings:
            self.add_policy_embedding(
                policy_id=emb.policy_id,
                agent_id=emb.agent_id,
                embedding=emb.embedding,
                metadata=emb.metadata,
                performance=emb.performance,
            )
        return len(embeddings)

    def search_similar_policies(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_criteria: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """Find the most similar policies to a query vector."""
        if not self._policies:
            return []

        query = self._normalize(query_embedding)

        if FAISS_AVAILABLE and self._index is not None:
            return self._search_faiss(query, top_k, filter_criteria)
        return self._search_numpy(query, top_k, filter_criteria)

    def _search_faiss(
        self,
        query: np.ndarray,
        top_k: int,
        filter_criteria: Optional[dict[str, Any]],
    ) -> list[SearchResult]:
        """Search using FAISS index."""
        if self._index_dirty:
            self._rebuild_index()

        if not self._index_ids:
            return []

        # Search more than needed if filtering
        search_k = min(top_k * 3, len(self._index_ids)) if filter_criteria else top_k
        search_k = min(search_k, len(self._index_ids))

        query_matrix = query.reshape(1, -1).astype(np.float32)
        distances, indices = self._index.search(query_matrix, search_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._index_ids):
                continue

            pid = self._index_ids[idx]
            policy = self._policies.get(pid)
            if policy is None:
                continue

            if filter_criteria and not self._matches_filter(policy, filter_criteria):
                continue

            score = float(dist) if self._metric == "cosine" else 1.0 / (1.0 + float(dist))
            results.append(SearchResult(
                policy_id=pid,
                agent_id=policy.agent_id,
                similarity_score=score,
                metadata=policy.metadata,
            ))

            if len(results) >= top_k:
                break

        return results

    def _search_numpy(
        self,
        query: np.ndarray,
        top_k: int,
        filter_criteria: Optional[dict[str, Any]],
    ) -> list[SearchResult]:
        """Fallback search using numpy (no FAISS)."""
        scored = []

        for pid, policy in self._policies.items():
            if filter_criteria and not self._matches_filter(policy, filter_criteria):
                continue

            vec = self._normalize(policy.embedding)

            if self._metric == "cosine":
                score = float(np.dot(query, vec))
            else:
                score = 1.0 / (1.0 + float(np.linalg.norm(query - vec)))

            scored.append((score, pid, policy))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            SearchResult(
                policy_id=pid,
                agent_id=policy.agent_id,
                similarity_score=score,
                metadata=policy.metadata,
            )
            for score, pid, policy in scored[:top_k]
        ]

    @staticmethod
    def _matches_filter(policy: PolicyEmbedding, criteria: dict[str, Any]) -> bool:
        """Check if a policy matches filter criteria."""
        for key, value in criteria.items():
            if key == "agent_id" and policy.agent_id != value:
                return False
            if key in policy.metadata and policy.metadata[key] != value:
                return False
        return True

    # --- Retrieval ---

    def get_policy_embedding(self, policy_id: str) -> Optional[PolicyEmbedding]:
        """Get a specific policy by ID."""
        return self._policies.get(policy_id)

    def get_agent_policies(
        self, agent_id: str, limit: int = 100
    ) -> list[PolicyEmbedding]:
        """Get all policies for a specific agent."""
        policy_ids = self._agent_index.get(agent_id, set())
        policies = [self._policies[pid] for pid in policy_ids if pid in self._policies]
        return policies[:limit]

    def update_policy_metadata(
        self, policy_id: str, metadata: dict[str, Any]
    ) -> bool:
        """Update metadata for an existing policy."""
        policy = self._policies.get(policy_id)
        if policy is None:
            return False
        policy.metadata.update(metadata)
        return True

    # --- Deletion ---

    def delete_policy(self, policy_id: str) -> bool:
        """Delete a specific policy."""
        policy = self._policies.pop(policy_id, None)
        if policy is None:
            return False

        agent_pids = self._agent_index.get(policy.agent_id)
        if agent_pids:
            agent_pids.discard(policy_id)

        self._index_dirty = True
        return True

    def delete_agent_policies(self, agent_id: str) -> int:
        """Delete all policies for an agent."""
        policy_ids = list(self._agent_index.get(agent_id, set()))
        for pid in policy_ids:
            self._policies.pop(pid, None)

        self._agent_index.pop(agent_id, None)
        self._index_dirty = True
        return len(policy_ids)

    # --- Analysis ---

    def cluster_policies(
        self,
        num_clusters: int = 10,
        filter_criteria: Optional[dict[str, Any]] = None,
    ) -> dict[int, list[str]]:
        """Cluster policies using K-means."""
        from sklearn.cluster import KMeans

        policies = list(self._policies.values())
        if filter_criteria:
            policies = [p for p in policies if self._matches_filter(p, filter_criteria)]

        if len(policies) < num_clusters:
            return {0: [p.policy_id for p in policies]}

        matrix = np.array(
            [self._normalize(p.embedding) for p in policies], dtype=np.float32
        )

        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(matrix)

        clusters: dict[int, list[str]] = {}
        for label, policy in zip(labels, policies):
            clusters.setdefault(int(label), []).append(policy.policy_id)

        return clusters

    def find_policy_patterns(
        self,
        min_cluster_size: int = 5,
        similarity_threshold: float = 0.8,
    ) -> list[dict[str, Any]]:
        """Find recurring patterns in stored policies."""
        if len(self._policies) < min_cluster_size:
            return []

        num_clusters = max(2, len(self._policies) // min_cluster_size)
        clusters = self.cluster_policies(num_clusters=num_clusters)

        patterns = []
        for cluster_id, policy_ids in clusters.items():
            if len(policy_ids) >= min_cluster_size:
                patterns.append({
                    "cluster_id": cluster_id,
                    "size": len(policy_ids),
                    "policy_ids": policy_ids,
                })

        return patterns

    def get_collection_stats(self) -> dict[str, Any]:
        """Get collection statistics."""
        return {
            "total_policies": len(self._policies),
            "total_agents": len(self._agent_index),
            "embedding_dim": self._embedding_dim,
            "metric": self._metric,
            "faiss_available": FAISS_AVAILABLE,
            "index_dirty": self._index_dirty,
        }

    def clear_collection(self) -> bool:
        """Remove all policies."""
        self._policies.clear()
        self._agent_index.clear()
        self._index_ids.clear()
        self._index_dirty = True
        self._initialize_index()
        return True

    # --- Persistence ---

    def save(self, path: Optional[Path] = None) -> None:
        """Save all data to disk via pickle."""
        target = path or self._persist_path
        if target is None:
            return

        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "policies": self._policies,
            "agent_index": {k: list(v) for k, v in self._agent_index.items()},
            "embedding_dim": self._embedding_dim,
            "metric": self._metric,
        }

        with open(target, "wb") as f:
            pickle.dump(data, f)

        logger.info("VectorStore saved to %s (%d policies)", target, len(self._policies))

    def load(self, path: Optional[Path] = None) -> None:
        """Load data from disk."""
        target = path or self._persist_path
        if target is None:
            return

        target = Path(target)
        if not target.exists():
            logger.info("No vector store at %s, starting fresh", target)
            return

        with open(target, "rb") as f:
            data = pickle.load(f)  # noqa: S301

        self._policies = data["policies"]
        self._agent_index = {k: set(v) for k, v in data["agent_index"].items()}
        self._embedding_dim = data.get("embedding_dim", self._embedding_dim)
        self._metric = data.get("metric", self._metric)
        self._index_dirty = True

        logger.info("VectorStore loaded from %s (%d policies)", target, len(self._policies))

    def close(self) -> None:
        """Save and clean up."""
        self.save()
        self._policies.clear()
        self._agent_index.clear()
