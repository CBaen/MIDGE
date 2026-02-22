"""Deep Memory - Qdrant-backed permanent memory layer.

Provides the bridge between Mae's numerical experiences and
Qdrant's text-embedding search. Handles collection management,
embedding via Ollama, and triadic storage verification.

Biological analogy: Long-term memory consolidation from
hippocampus to neocortex during sleep.

Three Qdrant collections (Rule of Three):
  - mae_narrative: Individual agent episode narratives
  - mae_ancestral: Distilled cross-agent patterns
  - mae_meta: Memory about memory (strange loop)
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Collection names
COLLECTION_NARRATIVE = "mae_narrative"
COLLECTION_ANCESTRAL = "mae_ancestral"
COLLECTION_META = "mae_meta"

ALL_COLLECTIONS = [COLLECTION_NARRATIVE, COLLECTION_ANCESTRAL, COLLECTION_META]

# Payload index definitions per collection
_NARRATIVE_INDEXES = [
    ("agent_id", "keyword"),
    ("agent_role", "keyword"),
    ("reward_sign", "keyword"),
    ("circadian_phase", "keyword"),
    ("organ_id", "keyword"),
    ("type", "keyword"),
]

_ANCESTRAL_INDEXES = [
    ("pattern_type", "keyword"),
    ("domain", "keyword"),
    ("type", "keyword"),
]

_META_INDEXES = [
    ("describes_collection", "keyword"),
    ("type", "keyword"),
]

_COLLECTION_INDEXES = {
    COLLECTION_NARRATIVE: _NARRATIVE_INDEXES,
    COLLECTION_ANCESTRAL: _ANCESTRAL_INDEXES,
    COLLECTION_META: _META_INDEXES,
}


@dataclass(frozen=True)
class QdrantConfig:
    """Configuration for Qdrant connection."""

    url: str = "http://localhost:6335"
    ollama_url: str = "http://localhost:11434"
    embedding_model: str = "mxbai-embed-large"
    embedding_dim: int = 1024
    batch_size: int = 32
    max_workers: int = 16
    timeout: int = 60


@dataclass
class SearchResult:
    """Single result from a Qdrant search."""

    point_id: str
    score: float
    payload: dict[str, Any]
    text: str = ""


class DeepMemoryStore:
    """Manages Qdrant collections for Mae's deep memory layer.

    Handles:
    - Collection creation and schema management (V2 hybrid)
    - Embedding generation via Ollama (1024-dim mxbai-embed-large)
    - Batch storage with parallel embedding
    - Hybrid search (dense + sparse)
    - Graceful degradation (empty results when Qdrant unavailable)
    """

    def __init__(
        self,
        config: QdrantConfig | None = None,
    ) -> None:
        self._config = config or QdrantConfig()
        self._available: bool | None = None  # lazy-checked on first call
        self._sparse_module: Any = None

    # -- Availability --

    def ping(self) -> bool:
        """Check if Qdrant is reachable."""
        try:
            resp = requests.get(
                f"{self._config.url}/collections",
                timeout=5,
            )
            return resp.status_code == 200
        except Exception:
            return False

    def is_available(self) -> bool:
        """Check Qdrant availability (cached after first check)."""
        if self._available is None:
            self._available = self.ping()
            if self._available:
                logger.info("Deep memory: Qdrant available at %s", self._config.url)
            else:
                logger.warning(
                    "Deep memory: Qdrant unavailable at %s, running without deep memory",
                    self._config.url,
                )
        return self._available

    # -- Collection Management --

    def ensure_collections(self) -> dict[str, bool]:
        """Create all Mae memory collections if they don't exist.

        Returns dict of collection_name -> created_or_exists.
        """
        if not self.is_available():
            return {c: False for c in ALL_COLLECTIONS}

        results: dict[str, bool] = {}
        for collection in ALL_COLLECTIONS:
            try:
                # Check if exists
                resp = requests.get(
                    f"{self._config.url}/collections/{collection}",
                    timeout=self._config.timeout,
                )
                if resp.status_code == 200:
                    results[collection] = True
                    continue

                # Create V2 collection (named vectors: dense + sparse)
                create_resp = requests.put(
                    f"{self._config.url}/collections/{collection}",
                    json={
                        "vectors": {
                            "dense": {
                                "size": self._config.embedding_dim,
                                "distance": "Cosine",
                            }
                        },
                        "sparse_vectors": {"sparse": {}},
                        "on_disk_payload": True,
                    },
                    timeout=self._config.timeout,
                )
                created = create_resp.status_code == 200
                results[collection] = created

                # Create payload indexes
                if created:
                    self._ensure_indexes(collection)

                logger.info(
                    "Deep memory: collection %s %s",
                    collection,
                    "created" if created else "FAILED",
                )
            except Exception:
                logger.exception("Failed to ensure collection %s", collection)
                results[collection] = False

        return results

    def _ensure_indexes(self, collection: str) -> None:
        """Create payload indexes for a collection."""
        indexes = _COLLECTION_INDEXES.get(collection, [])
        for field_name, field_type in indexes:
            try:
                requests.put(
                    f"{self._config.url}/collections/{collection}/index",
                    json={
                        "field_name": field_name,
                        "field_schema": field_type,
                    },
                    timeout=self._config.timeout,
                )
            except Exception:
                logger.debug("Index creation warning for %s.%s", collection, field_name)

    def get_collection_stats(self, collection: str) -> dict[str, Any]:
        """Get statistics for a collection."""
        if not self.is_available():
            return {}


        try:
            resp = requests.get(
                f"{self._config.url}/collections/{collection}",
                timeout=self._config.timeout,
            )
            if resp.status_code == 200:
                data = resp.json().get("result", {})
                return {
                    "points_count": data.get("points_count", 0),
                    "vectors_count": data.get("vectors_count", 0),
                    "status": data.get("status", "unknown"),
                }
        except Exception:
            pass
        return {}

    # -- Embedding --

    def embed_text(self, text: str) -> np.ndarray | None:
        """Get dense embedding from Ollama."""

        try:
            resp = requests.post(
                f"{self._config.ollama_url}/api/embeddings",
                json={
                    "model": self._config.embedding_model,
                    "prompt": text[:8000],
                },
                timeout=self._config.timeout,
            )
            if resp.status_code == 200:
                embedding = resp.json().get("embedding", [])
                if embedding and len(embedding) == self._config.embedding_dim:
                    return np.array(embedding, dtype=np.float32)
        except Exception:
            logger.debug("Ollama embedding failed", exc_info=True)
        return None

    def embed_texts_batch(
        self, texts: list[str]
    ) -> list[np.ndarray | None]:
        """Get dense embeddings for multiple texts in parallel."""
        if not texts:
            return []

        if len(texts) <= 2:
            return [self.embed_text(t) for t in texts]

        results: list[np.ndarray | None] = [None] * len(texts)

        def _embed_single(idx_text: tuple[int, str]) -> tuple[int, np.ndarray | None]:
            idx, text = idx_text
            return idx, self.embed_text(text)

        workers = min(len(texts), self._config.max_workers)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_embed_single, (i, t)): i
                for i, t in enumerate(texts)
            }
            for future in as_completed(futures):
                try:
                    idx, embedding = future.result()
                    results[idx] = embedding
                except Exception:
                    logger.debug("Batch embedding error", exc_info=True)

        return results

    def embed_sparse(self, text: str) -> tuple[list[int], list[float]] | None:
        """Get sparse embedding for hybrid search (TF-IDF fallback)."""
        try:
            indices, values = _sparse_embed(text)
            if indices and values:
                return indices, values
        except Exception:
            logger.debug("Sparse embedding failed", exc_info=True)
        return None

    # -- Storage --

    def store_point(
        self,
        collection: str,
        text: str,
        payload: dict[str, Any],
        point_id: str | None = None,
    ) -> str | None:
        """Store a single point with embedding.

        Returns point_id on success, None on failure.
        """
        if not self.is_available():
            return None


        pid = point_id or str(uuid.uuid4())

        # Get dense embedding
        dense = self.embed_text(text)
        if dense is None:
            logger.warning("Deep memory: embedding failed for point %s", pid)
            return None

        # Build vector data
        vector_data: dict[str, Any] = {"dense": dense.tolist()}

        # Try sparse embedding
        sparse = self.embed_sparse(text)
        if sparse is not None:
            indices, values = sparse
            vector_data["sparse"] = {"indices": indices, "values": values}

        # Store payload text for retrieval
        payload["text"] = text

        try:
            resp = requests.put(
                f"{self._config.url}/collections/{collection}/points",
                json={
                    "points": [
                        {
                            "id": pid,
                            "vector": vector_data,
                            "payload": _sanitize_payload(payload),
                        }
                    ]
                },
                timeout=self._config.timeout,
            )
            if resp.status_code == 200:
                return pid
            logger.warning("Deep memory: store failed (%d)", resp.status_code)
        except Exception:
            logger.debug("Deep memory: store error", exc_info=True)
        return None

    def store_points_batch(
        self,
        collection: str,
        items: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Store multiple points with parallel embedding.

        Each item must have 'text' and 'payload' keys. Optional 'point_id'.

        Returns stats dict: {stored, failed, total}.
        """
        if not self.is_available() or not items:
            return {"stored": 0, "failed": len(items), "total": len(items)}


        # 1. Batch embed all texts
        texts = [item["text"] for item in items]
        embeddings = self.embed_texts_batch(texts)

        # 2. Build points (skip items where embedding failed)
        points = []
        failed = 0
        for i, item in enumerate(items):
            dense = embeddings[i]
            if dense is None:
                failed += 1
                continue

            pid = item.get("point_id", str(uuid.uuid4()))
            vector_data: dict[str, Any] = {"dense": dense.tolist()}

            sparse = self.embed_sparse(item["text"])
            if sparse is not None:
                indices, values = sparse
                vector_data["sparse"] = {"indices": indices, "values": values}

            payload = dict(item.get("payload", {}))
            payload["text"] = item["text"]

            points.append({
                "id": pid,
                "vector": vector_data,
                "payload": _sanitize_payload(payload),
            })

        if not points:
            return {"stored": 0, "failed": failed, "total": len(items)}

        # 3. Store in batches
        stored = 0
        batch_size = self._config.batch_size
        for start in range(0, len(points), batch_size):
            batch = points[start : start + batch_size]
            try:
                resp = requests.put(
                    f"{self._config.url}/collections/{collection}/points",
                    json={"points": batch},
                    timeout=self._config.timeout,
                )
                if resp.status_code == 200:
                    stored += len(batch)
                else:
                    failed += len(batch)
            except Exception:
                failed += len(batch)

        return {"stored": stored, "failed": failed, "total": len(items)}

    # -- Search --

    def search(
        self,
        collection: str,
        query_text: str,
        limit: int = 5,
        score_threshold: float = 0.3,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search Qdrant using hybrid (dense + sparse) vectors.

        Returns list of SearchResult ordered by relevance.
        """
        if not self.is_available():
            return []


        # Embed query
        dense = self.embed_text(query_text)
        if dense is None:
            return []

        # Build search request
        sparse = self.embed_sparse(query_text)
        if sparse is not None:
            indices, values = sparse
            # Hybrid search with RRF fusion
            request_body: dict[str, Any] = {
                "prefetch": [
                    {
                        "query": dense.tolist(),
                        "using": "dense",
                        "limit": limit * 3,
                    },
                    {
                        "query": {
                            "indices": indices,
                            "values": values,
                        },
                        "using": "sparse",
                        "limit": limit * 3,
                    },
                ],
                "query": {"fusion": "rrf"},
                "limit": limit,
                "with_payload": True,
            }
        else:
            # Dense-only fallback
            request_body = {
                "vector": {"name": "dense", "vector": dense.tolist()},
                "limit": limit,
                "score_threshold": score_threshold,
                "with_payload": True,
            }

        if filters:
            request_body["filter"] = filters

        try:
            resp = requests.post(
                f"{self._config.url}/collections/{collection}/points/query",
                json=request_body,
                timeout=self._config.timeout,
            )
            if resp.status_code == 200:
                points = resp.json().get("result", {}).get("points", [])
                return [
                    SearchResult(
                        point_id=str(p.get("id", "")),
                        score=float(p.get("score", 0.0)),
                        payload=p.get("payload", {}),
                        text=p.get("payload", {}).get("text", ""),
                    )
                    for p in points
                ]
        except Exception:
            logger.debug("Deep memory: search error", exc_info=True)
        return []

    def search_with_filter(
        self,
        collection: str,
        query_text: str,
        must_conditions: list[dict[str, Any]] | None = None,
        limit: int = 5,
    ) -> list[SearchResult]:
        """Search with Qdrant filter conditions.

        Example must_conditions:
            [{"key": "agent_id", "match": {"value": "agent-3"}}]
        """
        filters = None
        if must_conditions:
            filters = {"must": must_conditions}
        return self.search(collection, query_text, limit=limit, filters=filters)

    # -- Witness Hash --

    @staticmethod
    def compute_witness_hash(
        state: np.ndarray,
        next_state: np.ndarray,
        action: Any,
        reward: float,
    ) -> str:
        """Compute SHA-256 witness hash for triadic verification."""
        data = (
            state.tobytes()
            + next_state.tobytes()
            + str(action).encode()
            + str(reward).encode()
        )
        return hashlib.sha256(data).hexdigest()

    # -- Repr --

    def __repr__(self) -> str:
        avail = "connected" if self._available else "unavailable" if self._available is False else "unchecked"
        return f"DeepMemoryStore({avail}, url={self._config.url})"


# -- Module-level helpers --


def _sanitize_payload(obj: Any) -> Any:
    """Recursively sanitize payloads for UTF-8 safety and JSON compatibility."""
    if isinstance(obj, str):
        return obj.encode("utf-8", errors="replace").decode("utf-8")
    if isinstance(obj, dict):
        return {k: _sanitize_payload(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_payload(item) for item in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


def _sparse_embed(text: str) -> tuple[list[int], list[float]]:
    """Generate sparse embedding using TF-IDF hash-based approach.

    Lightweight reimplementation of the sparse embedding strategy
    from the lineage infrastructure, avoiding external dependencies.
    """
    import math
    import re
    from collections import Counter

    VOCAB_SIZE = 30000
    STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
        "be", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "must", "shall", "can", "need",
        "this", "that", "these", "those", "it", "its", "they", "them", "their",
        "we", "our", "you", "your", "i", "my", "he", "she", "him", "her", "his",
        "what", "which", "who", "whom", "when", "where", "why", "how",
        "all", "each", "every", "both", "few", "more", "most", "other", "some",
        "such", "no", "not", "only", "same", "so", "than", "too", "very",
    }

    # Tokenize
    text_lower = re.sub(r"[^\w\s]", " ", text.lower())
    tokens = [t for t in text_lower.split() if len(t) > 2 and t not in STOPWORDS]

    if not tokens:
        return [], []

    # Hash tokens to vocabulary indices
    def hash_token(token: str) -> int:
        h = 0
        for c in token:
            h = (h * 31 + ord(c)) & 0xFFFFFFFF
        return h % VOCAB_SIZE

    counts = Counter(hash_token(t) for t in tokens)
    max_count = max(counts.values())

    # TF-IDF-style scoring
    indices = sorted(counts.keys())
    values = [
        (0.5 + 0.5 * counts[idx] / max_count) * math.log(1 + VOCAB_SIZE / (1 + counts[idx]))
        for idx in indices
    ]

    return indices, values
