"""Tests for recall witness hash verification (Law 1 — bare dyad closure).

MemoryBridge._verify_results() checks that recalled experiences carry
valid witness hashes, tags results with witness_verified, publishes
verification events on EventBus, and tracks stats.
"""

from unittest.mock import MagicMock, call
import hashlib

import numpy as np
import pytest

from mae_core.memory.deep_memory import (
    COLLECTION_ANCESTRAL,
    COLLECTION_NARRATIVE,
    DeepMemoryStore,
    SearchResult,
)
from mae_core.memory.experience_narrator import ExperienceNarrator
from mae_core.memory.memory_bridge import MemoryBridge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_witness_hash():
    """Generate a valid 64-char SHA-256 hex digest."""
    data = np.random.rand(8).tobytes() + b"test"
    return hashlib.sha256(data).hexdigest()


def _result_with_hash(point_id="p1", score=0.9, witness_hash=None):
    """Create a SearchResult with optional witness_hash in payload."""
    payload = {
        "type": "episode",
        "agent_id": "agent-1",
        "mean_reward": 0.5,
    }
    if witness_hash is not None:
        payload["witness_hash"] = witness_hash
    return SearchResult(point_id=point_id, score=score, payload=payload, text="test")


def _result_ancestral(point_id="a1", score=0.8):
    """Create a SearchResult mimicking an ancestral pattern (no witness_hash)."""
    return SearchResult(
        point_id=point_id,
        score=score,
        payload={
            "type": "pattern",
            "pattern_type": "behavioral",
            "domain": "general",
            "confidence": 0.7,
        },
        text="ancestral pattern",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_store():
    store = MagicMock(spec=DeepMemoryStore)
    store.is_available.return_value = True
    store.get_collection_stats.return_value = {"points_count": 10, "status": "green"}
    return store


@pytest.fixture
def mock_bus():
    return MagicMock()


@pytest.fixture
def mock_haven():
    haven = MagicMock()
    haven.is_agent_isolated.return_value = False
    return haven


@pytest.fixture
def narrator():
    return ExperienceNarrator()


@pytest.fixture
def bridge(mock_store, narrator, mock_bus):
    return MemoryBridge(
        deep_store=mock_store, narrator=narrator,
        event_bus=mock_bus,
    )


@pytest.fixture
def bridge_with_haven(mock_store, narrator, mock_bus, mock_haven):
    return MemoryBridge(
        deep_store=mock_store, narrator=narrator,
        event_bus=mock_bus, haven=mock_haven,
    )


# ---------------------------------------------------------------------------
# _verify_results — Direct Tests
# ---------------------------------------------------------------------------

class TestVerifyResults:
    """Test the _verify_results private method directly."""

    def test_empty_results_returns_empty(self, bridge):
        result = bridge._verify_results([], "test", "agent-1")
        assert result == []
        assert bridge._total_verified == 0
        assert bridge._total_unverified == 0

    def test_result_with_valid_hash_marked_verified(self, bridge):
        wh = _make_witness_hash()
        results = [_result_with_hash(witness_hash=wh)]
        out = bridge._verify_results(results, "test", "agent-1")
        assert len(out) == 1
        assert out[0].payload["witness_verified"] is True
        assert bridge._total_verified == 1
        assert bridge._total_unverified == 0

    def test_result_without_hash_marked_unverified(self, bridge):
        results = [_result_with_hash(witness_hash=None)]
        out = bridge._verify_results(results, "test", "agent-1")
        assert out[0].payload["witness_verified"] is False
        assert bridge._total_unverified == 1

    def test_result_with_empty_hash_marked_unverified(self, bridge):
        results = [_result_with_hash(witness_hash="")]
        out = bridge._verify_results(results, "test", "agent-1")
        assert out[0].payload["witness_verified"] is False

    def test_result_with_short_hash_marked_unverified(self, bridge):
        results = [_result_with_hash(witness_hash="abc123")]
        out = bridge._verify_results(results, "test", "agent-1")
        assert out[0].payload["witness_verified"] is False

    def test_mixed_results_tagged_correctly(self, bridge):
        wh = _make_witness_hash()
        results = [
            _result_with_hash(point_id="p1", witness_hash=wh),
            _result_with_hash(point_id="p2", witness_hash=None),
            _result_with_hash(point_id="p3", witness_hash=wh),
        ]
        out = bridge._verify_results(results, "test", "agent-1")
        assert out[0].payload["witness_verified"] is True
        assert out[1].payload["witness_verified"] is False
        assert out[2].payload["witness_verified"] is True
        assert bridge._total_verified == 2
        assert bridge._total_unverified == 1

    def test_counters_accumulate_across_calls(self, bridge):
        wh = _make_witness_hash()
        bridge._verify_results([_result_with_hash(witness_hash=wh)], "t1", "a1")
        bridge._verify_results([_result_with_hash(witness_hash=None)], "t2", "a2")
        bridge._verify_results([_result_with_hash(witness_hash=wh)], "t3", "a3")
        assert bridge._total_verified == 2
        assert bridge._total_unverified == 1

    def test_ancestral_pattern_marked_unverified(self, bridge):
        results = [_result_ancestral()]
        out = bridge._verify_results(results, "recall_ancestral", "any")
        assert out[0].payload["witness_verified"] is False

    def test_result_with_none_payload_counted_unverified(self, bridge):
        r = SearchResult(point_id="x", score=0.5, payload=None, text="no payload")
        # payload is None — should not crash, counted as unverified
        # Tag doesn't stick on None payload (expected: it's a degenerate case)
        out = bridge._verify_results([r], "test", "agent-1")
        assert bridge._total_unverified == 1
        assert bridge._total_verified == 0


# ---------------------------------------------------------------------------
# EventBus Verification Events
# ---------------------------------------------------------------------------

class TestVerificationEvents:
    """Verify that _verify_results publishes memory.recall_verified events."""

    def test_publishes_verification_event(self, bridge, mock_bus):
        wh = _make_witness_hash()
        bridge._verify_results(
            [_result_with_hash(witness_hash=wh)], "recall_from_deep", "agent-1",
        )
        mock_bus.publish.assert_called_with(
            "memory.recall_verified",
            {
                "recall_type": "recall_from_deep",
                "requester": "agent-1",
                "verified_count": 1,
                "unverified_count": 0,
                "total": 1,
            },
        )

    def test_publishes_mixed_counts(self, bridge, mock_bus):
        wh = _make_witness_hash()
        bridge._verify_results(
            [
                _result_with_hash(witness_hash=wh),
                _result_with_hash(witness_hash=None),
            ],
            "recall_peer_experiences",
            "agent-2",
        )
        mock_bus.publish.assert_called_with(
            "memory.recall_verified",
            {
                "recall_type": "recall_peer_experiences",
                "requester": "agent-2",
                "verified_count": 1,
                "unverified_count": 1,
                "total": 2,
            },
        )

    def test_no_event_without_bus(self, mock_store, narrator):
        bridge = MemoryBridge(deep_store=mock_store, narrator=narrator, event_bus=None)
        wh = _make_witness_hash()
        # Should not crash
        out = bridge._verify_results(
            [_result_with_hash(witness_hash=wh)], "test", "agent-1",
        )
        assert out[0].payload["witness_verified"] is True

    def test_bus_exception_does_not_crash(self, bridge, mock_bus):
        mock_bus.publish.side_effect = RuntimeError("bus down")
        wh = _make_witness_hash()
        out = bridge._verify_results(
            [_result_with_hash(witness_hash=wh)], "test", "agent-1",
        )
        # Still tags the result despite bus failure
        assert out[0].payload["witness_verified"] is True
        assert bridge._total_verified == 1


# ---------------------------------------------------------------------------
# Integration: Recall Methods Call Verification
# ---------------------------------------------------------------------------

class TestRecallFromDeepVerification:
    """recall_from_deep() wires through _verify_results."""

    def test_verified_results_tagged(self, bridge, mock_store):
        wh = _make_witness_hash()
        mock_store.search.return_value = [_result_with_hash(witness_hash=wh)]
        results = bridge.recall_from_deep("test query", agent_id="agent-1")
        assert results[0].payload["witness_verified"] is True
        assert bridge._total_verified == 1

    def test_unverified_results_tagged(self, bridge, mock_store):
        mock_store.search.return_value = [_result_with_hash(witness_hash=None)]
        results = bridge.recall_from_deep("test query")
        assert results[0].payload["witness_verified"] is False
        assert bridge._total_unverified == 1


class TestRecallPeerVerification:
    """recall_peer_experiences() wires through _verify_results."""

    def test_verified_peer_results(self, bridge, mock_store):
        wh = _make_witness_hash()
        mock_store.search.return_value = [_result_with_hash(witness_hash=wh)]
        results = bridge.recall_peer_experiences("query", "agent-1", ["agent-2"])
        assert results[0].payload["witness_verified"] is True

    def test_haven_filter_then_verification(self, bridge_with_haven, mock_store, mock_haven):
        """HAVEN filter runs first, then verification on remaining results."""
        wh = _make_witness_hash()
        mock_store.search.return_value = [
            _result_with_hash(point_id="p1", witness_hash=wh),
            _result_with_hash(point_id="p2", witness_hash=None),
        ]
        # Second result's agent is isolated — HAVEN removes it
        mock_haven.is_agent_isolated.side_effect = lambda aid: aid == "agent-1"
        # Both results have agent_id="agent-1", so both get filtered by HAVEN
        # Let's make them different agents
        r1 = _result_with_hash(point_id="p1", witness_hash=wh)
        r1.payload["agent_id"] = "agent-2"
        r2 = _result_with_hash(point_id="p2", witness_hash=None)
        r2.payload["agent_id"] = "agent-3"
        mock_store.search.return_value = [r1, r2]
        mock_haven.is_agent_isolated.side_effect = lambda aid: aid == "agent-3"

        results = bridge_with_haven.recall_peer_experiences("query", "agent-1")
        # agent-3 filtered by HAVEN, only agent-2 remains
        assert len(results) == 1
        assert results[0].payload["witness_verified"] is True


class TestRecallAncestralVerification:
    """recall_ancestral_patterns() wires through _verify_results."""

    def test_ancestral_all_unverified(self, bridge, mock_store):
        mock_store.search.return_value = [_result_ancestral(), _result_ancestral("a2")]
        results = bridge.recall_ancestral_patterns("test query")
        assert all(r.payload["witness_verified"] is False for r in results)
        assert bridge._total_unverified == 2
        assert bridge._total_verified == 0


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestVerificationStats:
    """get_stats() includes verification counters."""

    def test_stats_include_verification(self, bridge):
        wh = _make_witness_hash()
        bridge._verify_results([_result_with_hash(witness_hash=wh)], "t", "a")
        bridge._verify_results([_result_with_hash(witness_hash=None)], "t", "a")
        stats = bridge.get_stats()
        assert stats["total_verified"] == 1
        assert stats["total_unverified"] == 1

    def test_stats_zero_initially(self, bridge):
        stats = bridge.get_stats()
        assert stats["total_verified"] == 0
        assert stats["total_unverified"] == 0
