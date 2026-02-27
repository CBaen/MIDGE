"""Tests for DeepMemoryStore — Qdrant-backed permanent memory layer.

All Qdrant and Ollama calls are mocked. No real services required.
"""

import json
from unittest.mock import MagicMock, patch, call
import numpy as np
import pytest

from mae_core.memory.deep_memory import (
    ALL_COLLECTIONS,
    COLLECTION_ANCESTRAL,
    COLLECTION_META,
    COLLECTION_NARRATIVE,
    DeepMemoryStore,
    QdrantConfig,
    SearchResult,
    _sanitize_payload,
    _sparse_embed,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config():
    return QdrantConfig(url="http://test:6333", ollama_url="http://test:11434")


@pytest.fixture
def store(config):
    s = DeepMemoryStore(config=config)
    s._available = True  # skip real ping
    return s


@pytest.fixture
def mock_embedding():
    """Return a valid 1024-dim embedding."""
    return list(np.random.randn(1024).astype(float))


# ---------------------------------------------------------------------------
# QdrantConfig
# ---------------------------------------------------------------------------


class TestQdrantConfig:
    def test_defaults(self):
        c = QdrantConfig()
        assert c.url == "http://localhost:6335"
        assert c.ollama_url == "http://localhost:11434"
        assert c.embedding_model == "mxbai-embed-large"
        assert c.embedding_dim == 1024
        assert c.batch_size == 32
        assert c.max_workers == 16

    def test_frozen(self):
        c = QdrantConfig()
        with pytest.raises(AttributeError):
            c.url = "other"


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


class TestAvailability:
    @patch("mae_core.memory.deep_memory.requests")
    def test_ping_success(self, mock_req, config):
        mock_req.get.return_value = MagicMock(status_code=200)
        store = DeepMemoryStore(config=config)
        assert store.ping() is True

    @patch("mae_core.memory.deep_memory.requests")
    def test_ping_failure(self, mock_req, config):
        mock_req.get.side_effect = ConnectionError("refused")
        store = DeepMemoryStore(config=config)
        assert store.ping() is False

    @patch("mae_core.memory.deep_memory.requests")
    def test_is_available_caches(self, mock_req, config):
        mock_req.get.return_value = MagicMock(status_code=200)
        store = DeepMemoryStore(config=config)
        assert store.is_available() is True
        # Second call doesn't ping again
        mock_req.get.reset_mock()
        assert store.is_available() is True
        mock_req.get.assert_not_called()

    def test_graceful_degradation_when_unavailable(self, config):
        store = DeepMemoryStore(config=config)
        store._available = False
        assert store.search("midge_narrative", "test") == []
        assert store.store_point("midge_narrative", "test", {}) is None
        assert store.store_points_batch("midge_narrative", [{"text": "t", "payload": {}}]) == {
            "stored": 0, "failed": 1, "total": 1,
        }


# ---------------------------------------------------------------------------
# Collection Management
# ---------------------------------------------------------------------------


class TestCollectionManagement:
    @patch("mae_core.memory.deep_memory.requests")
    def test_ensure_collections_creates_three(self, mock_req, store):
        # First GET returns 404 (collection doesn't exist), PUT creates it
        resp_404 = MagicMock(status_code=404)
        resp_200 = MagicMock(status_code=200)

        def side_effect(url, **kwargs):
            if "/index" in url:
                return resp_200
            return resp_404

        mock_req.get.side_effect = side_effect
        mock_req.put.return_value = resp_200

        results = store.ensure_collections()
        assert len(results) == 3
        assert COLLECTION_NARRATIVE in results
        assert COLLECTION_ANCESTRAL in results
        assert COLLECTION_META in results
        assert all(v is True for v in results.values())

    @patch("mae_core.memory.deep_memory.requests")
    def test_ensure_collections_skips_existing(self, mock_req, store):
        resp_200 = MagicMock(status_code=200)
        mock_req.get.return_value = resp_200

        results = store.ensure_collections()
        # Should not call PUT since collections already exist
        put_calls = [c for c in mock_req.put.call_args_list if "/index" not in str(c)]
        assert len(put_calls) == 0

    @patch("mae_core.memory.deep_memory.requests")
    def test_get_collection_stats(self, mock_req, store):
        mock_req.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "result": {
                    "points_count": 42,
                    "vectors_count": 42,
                    "status": "green",
                }
            },
        )
        stats = store.get_collection_stats(COLLECTION_NARRATIVE)
        assert stats["points_count"] == 42

    def test_ensure_collections_unavailable(self, config):
        store = DeepMemoryStore(config=config)
        store._available = False
        results = store.ensure_collections()
        assert all(v is False for v in results.values())


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


class TestEmbedding:
    @patch("mae_core.memory.deep_memory.requests")
    def test_embed_text_success(self, mock_req, store, mock_embedding):
        mock_req.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"embedding": mock_embedding},
        )
        result = store.embed_text("hello world")
        assert result is not None
        assert result.shape == (1024,)

    @patch("mae_core.memory.deep_memory.requests")
    def test_embed_text_failure(self, mock_req, store):
        mock_req.post.side_effect = ConnectionError("refused")
        result = store.embed_text("hello world")
        assert result is None

    @patch("mae_core.memory.deep_memory.requests")
    def test_embed_texts_batch(self, mock_req, store, mock_embedding):
        mock_req.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"embedding": mock_embedding},
        )
        results = store.embed_texts_batch(["a", "b", "c"])
        assert len(results) == 3
        assert all(r is not None for r in results)

    @patch("mae_core.memory.deep_memory.requests")
    def test_embed_texts_batch_empty(self, mock_req, store):
        assert store.embed_texts_batch([]) == []

    @patch("mae_core.memory.deep_memory.requests")
    def test_embed_texts_batch_small(self, mock_req, store, mock_embedding):
        """Small batches (<=2) skip ThreadPoolExecutor."""
        mock_req.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"embedding": mock_embedding},
        )
        results = store.embed_texts_batch(["a"])
        assert len(results) == 1

    def test_sparse_embed(self):
        indices, values = _sparse_embed("The agent explored the environment and found resources")
        assert len(indices) > 0
        assert len(indices) == len(values)
        assert all(isinstance(i, int) for i in indices)
        assert all(isinstance(v, float) for v in values)
        assert indices == sorted(indices)  # sorted by index

    def test_sparse_embed_empty(self):
        indices, values = _sparse_embed("")
        assert indices == []
        assert values == []


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


class TestStorage:
    @patch("mae_core.memory.deep_memory.requests")
    def test_store_point(self, mock_req, store, mock_embedding):
        # Mock embedding
        mock_req.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"embedding": mock_embedding},
        )
        # Mock Qdrant PUT
        mock_req.put.return_value = MagicMock(status_code=200)

        result = store.store_point(
            collection=COLLECTION_NARRATIVE,
            text="Agent explored and found reward",
            payload={"agent_id": "agent-3", "type": "episode"},
        )
        assert result is not None  # returns point_id

    @patch("mae_core.memory.deep_memory.requests")
    def test_store_point_custom_id(self, mock_req, store, mock_embedding):
        mock_req.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"embedding": mock_embedding},
        )
        mock_req.put.return_value = MagicMock(status_code=200)

        result = store.store_point(
            collection=COLLECTION_NARRATIVE,
            text="test",
            payload={},
            point_id="my-custom-id",
        )
        assert result == "my-custom-id"

    @patch("mae_core.memory.deep_memory.requests")
    def test_store_point_embedding_failure(self, mock_req, store):
        mock_req.post.side_effect = ConnectionError("ollama down")
        result = store.store_point(COLLECTION_NARRATIVE, "test", {})
        assert result is None

    @patch("mae_core.memory.deep_memory.requests")
    def test_store_points_batch(self, mock_req, store, mock_embedding):
        mock_req.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"embedding": mock_embedding},
        )
        mock_req.put.return_value = MagicMock(status_code=200)

        items = [
            {"text": f"experience {i}", "payload": {"agent_id": f"agent-{i}"}}
            for i in range(5)
        ]
        result = store.store_points_batch(COLLECTION_NARRATIVE, items)
        assert result["stored"] == 5
        assert result["failed"] == 0
        assert result["total"] == 5

    @patch("mae_core.memory.deep_memory.requests")
    def test_store_points_batch_partial_failure(self, mock_req, store, mock_embedding):
        # First 3 embeddings succeed, last 2 fail
        call_count = [0]

        def embed_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 3:
                return MagicMock(
                    status_code=200,
                    json=lambda: {"embedding": mock_embedding},
                )
            return MagicMock(status_code=500)

        mock_req.post.side_effect = embed_side_effect
        mock_req.put.return_value = MagicMock(status_code=200)

        items = [{"text": f"exp {i}", "payload": {}} for i in range(5)]
        result = store.store_points_batch(COLLECTION_NARRATIVE, items)
        assert result["total"] == 5
        assert result["stored"] == 3
        assert result["failed"] == 2


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class TestSearch:
    @patch("mae_core.memory.deep_memory.requests")
    def test_search_returns_results(self, mock_req, store, mock_embedding):
        # Mock embedding call
        mock_req.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"embedding": mock_embedding},
        )
        # Mock search response
        search_resp = MagicMock(
            status_code=200,
            json=lambda: {
                "result": {
                    "points": [
                        {
                            "id": "point-1",
                            "score": 0.95,
                            "payload": {
                                "text": "Agent explored",
                                "agent_id": "agent-3",
                            },
                        },
                        {
                            "id": "point-2",
                            "score": 0.82,
                            "payload": {
                                "text": "Agent rested",
                                "agent_id": "agent-1",
                            },
                        },
                    ]
                }
            },
        )
        # post is used for both embedding and search
        mock_req.post.side_effect = [
            MagicMock(status_code=200, json=lambda: {"embedding": mock_embedding}),
            search_resp,
        ]

        results = store.search(COLLECTION_NARRATIVE, "agent exploring", limit=5)
        assert len(results) == 2
        assert isinstance(results[0], SearchResult)
        assert results[0].score == 0.95
        assert results[0].text == "Agent explored"

    @patch("mae_core.memory.deep_memory.requests")
    def test_search_with_filter(self, mock_req, store, mock_embedding):
        mock_req.post.side_effect = [
            MagicMock(status_code=200, json=lambda: {"embedding": mock_embedding}),
            MagicMock(status_code=200, json=lambda: {"result": {"points": []}}),
        ]

        results = store.search_with_filter(
            COLLECTION_NARRATIVE,
            "test",
            must_conditions=[{"key": "agent_id", "match": {"value": "agent-3"}}],
        )
        assert results == []

    @patch("mae_core.memory.deep_memory.requests")
    def test_search_embedding_failure(self, mock_req, store):
        mock_req.post.side_effect = ConnectionError("ollama down")
        results = store.search(COLLECTION_NARRATIVE, "test")
        assert results == []


# ---------------------------------------------------------------------------
# Witness Hash
# ---------------------------------------------------------------------------


class TestWitnessHash:
    def test_deterministic(self):
        state = np.array([1.0, 2.0, 3.0])
        next_state = np.array([4.0, 5.0, 6.0])
        h1 = DeepMemoryStore.compute_witness_hash(state, next_state, 2, 0.5)
        h2 = DeepMemoryStore.compute_witness_hash(state, next_state, 2, 0.5)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_different_inputs(self):
        state = np.array([1.0, 2.0, 3.0])
        h1 = DeepMemoryStore.compute_witness_hash(state, state, 0, 1.0)
        h2 = DeepMemoryStore.compute_witness_hash(state, state, 1, 1.0)
        assert h1 != h2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_sanitize_payload_strings(self):
        assert _sanitize_payload("hello") == "hello"

    def test_sanitize_payload_nested(self):
        payload = {"a": {"b": [1, "test", np.float32(3.14)]}}
        result = _sanitize_payload(payload)
        assert isinstance(result["a"]["b"][2], float)

    def test_sanitize_numpy_array(self):
        arr = np.array([1, 2, 3])
        result = _sanitize_payload(arr)
        assert result == [1, 2, 3]

    def test_repr(self, config):
        store = DeepMemoryStore(config=config)
        assert "unchecked" in repr(store)
        store._available = True
        assert "connected" in repr(store)
        store._available = False
        assert "unavailable" in repr(store)
