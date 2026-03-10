"""
deep_memory_search.py - Search and embedding helpers for DeepMemoryStore.

Extracted from deep_memory.py. Contains:
  - embed_text / embed_texts_batch / embed_sparse: Ollama + TF-IDF embedding
  - search / search_with_filter: Qdrant hybrid search
  - _sparse_embed: module-level sparse embedding helper
  - _sanitize_payload: payload sanitization helper
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

import numpy as np

try:
    import requests as _requests
except ImportError:
    _requests = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from mae_core.memory.deep_memory import DeepMemoryStore, SearchResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers re-exported for backward compat
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Embedding methods (called as free functions, passing store as first arg)
# ---------------------------------------------------------------------------

def embed_text(store: DeepMemoryStore, text: str) -> np.ndarray | None:
    """Get dense embedding from Ollama."""
    if _requests is None:
        return None
    try:
        resp = _requests.post(
            f"{store._config.ollama_url}/api/embeddings",
            json={
                "model": store._config.embedding_model,
                "prompt": text[:8000],
            },
            timeout=store._config.timeout,
        )
        if resp.status_code == 200:
            embedding = resp.json().get("embedding", [])
            if embedding and len(embedding) == store._config.embedding_dim:
                return np.array(embedding, dtype=np.float32)
    except Exception:
        logger.debug("Ollama embedding failed", exc_info=True)
    return None


def embed_texts_batch(
    store: DeepMemoryStore,
    texts: list[str],
) -> list[np.ndarray | None]:
    """Get dense embeddings for multiple texts in parallel."""
    if not texts:
        return []

    if len(texts) <= 2:
        return [embed_text(store, t) for t in texts]

    results: list[np.ndarray | None] = [None] * len(texts)

    def _embed_single(idx_text: tuple[int, str]) -> tuple[int, np.ndarray | None]:
        idx, text = idx_text
        return idx, embed_text(store, text)

    workers = min(len(texts), store._config.max_workers)
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


def embed_sparse(store: DeepMemoryStore, text: str) -> tuple[list[int], list[float]] | None:
    """Get sparse embedding for hybrid search (TF-IDF fallback)."""
    try:
        indices, values = _sparse_embed(text)
        if indices and values:
            return indices, values
    except Exception:
        logger.debug("Sparse embedding failed", exc_info=True)
    return None


# ---------------------------------------------------------------------------
# Search methods
# ---------------------------------------------------------------------------

def search(
    store: DeepMemoryStore,
    collection: str,
    query_text: str,
    limit: int = 5,
    score_threshold: float = 0.3,
    filters: dict[str, Any] | None = None,
) -> list[SearchResult]:
    """Search Qdrant using hybrid (dense + sparse) vectors.

    Returns list of SearchResult ordered by relevance.
    """
    from mae_core.memory.deep_memory import SearchResult as SR

    if not store.is_available() or _requests is None:
        return []

    # Embed query
    dense = embed_text(store, query_text)
    if dense is None:
        return []

    # Build search request
    sparse = embed_sparse(store, query_text)
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
        resp = _requests.post(
            f"{store._config.url}/collections/{collection}/points/query",
            json=request_body,
            timeout=store._config.timeout,
        )
        if resp.status_code == 200:
            points = resp.json().get("result", {}).get("points", [])
            return [
                SR(
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
    store: DeepMemoryStore,
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
    return search(store, collection, query_text, limit=limit, filters=filters)
