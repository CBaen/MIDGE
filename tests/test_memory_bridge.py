"""Tests for MemoryBridge and Coordinator deep memory integration.

All Qdrant/Ollama calls mocked. No real services needed.
"""

from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np
import pytest

from mae_core.memory.deep_memory import (
    COLLECTION_ANCESTRAL,
    COLLECTION_META,
    COLLECTION_NARRATIVE,
    DeepMemoryStore,
    QdrantConfig,
    SearchResult,
)
from mae_core.memory.experience import Experience
from mae_core.memory.experience_narrator import ExperienceNarrator
from mae_core.memory.memory_bridge import MemoryBridge


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_experience(reward=0.5, action=0, done=False):
    return Experience(
        state=np.random.rand(8).astype(np.float32),
        action=action,
        reward=reward,
        next_state=np.random.rand(8).astype(np.float32),
        done=done,
    )


@pytest.fixture
def mock_store():
    store = MagicMock(spec=DeepMemoryStore)
    store.is_available.return_value = True
    store.store_point.return_value = "mock-point-id"
    store.store_points_batch.return_value = {"stored": 5, "failed": 0, "total": 5}
    store.search.return_value = [
        SearchResult(point_id="p1", score=0.9, payload={"text": "found"}, text="found"),
    ]
    store.get_collection_stats.return_value = {"points_count": 42, "status": "green"}
    store.compute_witness_hash = DeepMemoryStore.compute_witness_hash
    return store


@pytest.fixture
def narrator():
    return ExperienceNarrator()


@pytest.fixture
def bridge(mock_store, narrator):
    return MemoryBridge(deep_store=mock_store, narrator=narrator)


@pytest.fixture
def agent_context():
    return {
        "agent_id": "agent-3",
        "agent_role": "EXPLORER",
        "parent_id": "colony",
        "peer_count": 4,
        "circadian_phase": "ACTIVE",
        "step": 1000,
    }


# ---------------------------------------------------------------------------
# MemoryBridge — Upward Flow
# ---------------------------------------------------------------------------


class TestConsolidateToDeep:
    def test_basic_consolidation(self, bridge, mock_store, agent_context):
        experiences = [_make_experience(reward=0.3 * i) for i in range(5)]
        result = bridge.consolidate_to_deep("agent-3", experiences, agent_context, consolidation_id=1)
        assert result["stored"] == 5
        assert result["consolidation_id"] == 1
        # Should also store a summary
        assert mock_store.store_point.called

    def test_consolidation_empty(self, bridge, agent_context):
        result = bridge.consolidate_to_deep("agent-3", [], agent_context)
        assert result["stored"] == 0

    def test_consolidation_unavailable(self, bridge, mock_store, agent_context):
        mock_store.is_available.return_value = False
        experiences = [_make_experience()]
        result = bridge.consolidate_to_deep("agent-3", experiences, agent_context)
        assert result["stored"] == 0
        assert result["failed"] == 1

    def test_consolidation_tracks_total(self, bridge, mock_store, agent_context):
        mock_store.store_points_batch.return_value = {"stored": 3, "failed": 0, "total": 3}
        bridge.consolidate_to_deep("agent-3", [_make_experience()] * 3, agent_context)
        assert bridge._total_consolidated == 3


# ---------------------------------------------------------------------------
# MemoryBridge — Downward Flow
# ---------------------------------------------------------------------------


class TestRecallFromDeep:
    def test_basic_recall(self, bridge, mock_store):
        results = bridge.recall_from_deep("agent exploring", agent_id="agent-3")
        assert len(results) == 1
        assert results[0].score == 0.9
        # Verify filter was passed
        call_args = mock_store.search.call_args
        assert call_args.kwargs.get("filters") is not None

    def test_recall_no_filter(self, bridge, mock_store):
        bridge.recall_from_deep("exploring")
        call_args = mock_store.search.call_args
        assert call_args.kwargs.get("filters") is None

    def test_recall_unavailable(self, bridge, mock_store):
        mock_store.is_available.return_value = False
        results = bridge.recall_from_deep("test")
        assert results == []

    def test_recall_tracks_total(self, bridge, mock_store):
        bridge.recall_from_deep("test")
        assert bridge._total_recalled == 1


class TestRecallPeerExperiences:
    def test_excludes_self(self, bridge, mock_store):
        bridge.recall_peer_experiences("exploring", "agent-3")
        call_args = mock_store.search.call_args
        filters = call_args.kwargs.get("filters", {})
        assert "must_not" in filters
        # Should exclude requesting agent
        excluded = filters["must_not"][0]["key"]
        assert excluded == "agent_id"


# ---------------------------------------------------------------------------
# MemoryBridge — Ancestral Memory
# ---------------------------------------------------------------------------


class TestAncestralMemory:
    def test_store_pattern(self, bridge, mock_store):
        pattern = {
            "pattern_type": "behavioral",
            "domain": "exploration",
            "occurrence_count": 15,
            "confidence": 0.87,
            "description": "Explorers find high reward in safe zones",
            "applicable_roles": ["EXPLORER"],
        }
        pid = bridge.store_ancestral_pattern(pattern, ["agent-0", "agent-3"])
        assert pid == "mock-point-id"
        mock_store.store_point.assert_called_once()
        call_payload = mock_store.store_point.call_args.args[2]
        assert call_payload["type"] == "pattern"
        assert call_payload["pattern_type"] == "behavioral"

    def test_recall_ancestral(self, bridge, mock_store):
        results = bridge.recall_ancestral_patterns("exploration reward")
        assert len(results) == 1

    def test_recall_ancestral_with_role(self, bridge, mock_store):
        bridge.recall_ancestral_patterns("test", applicable_role="EXPLORER")
        call_args = mock_store.search.call_args
        assert call_args.kwargs.get("filters") is not None


# ---------------------------------------------------------------------------
# MemoryBridge — Meta-Memory (Strange Loop)
# ---------------------------------------------------------------------------


class TestMetaMemory:
    def test_update_meta(self, bridge, mock_store):
        pid = bridge.update_meta_memory({"agent_count": 5, "hot_size": 1000})
        assert pid == "mock-point-id"
        call_args = mock_store.store_point.call_args
        assert COLLECTION_META in call_args.args[0]
        payload = call_args.args[2]
        assert payload["type"] == "self_model"
        assert payload["agent_count"] == 5

    def test_meta_unavailable(self, bridge, mock_store):
        mock_store.is_available.return_value = False
        pid = bridge.update_meta_memory({})
        assert pid is None


# ---------------------------------------------------------------------------
# MemoryBridge — Stats
# ---------------------------------------------------------------------------


class TestBridgeStats:
    def test_get_stats(self, bridge, mock_store):
        stats = bridge.get_stats()
        assert stats["available"] is True
        assert stats["total_consolidated"] == 0
        assert "narrative_stats" in stats

    def test_repr(self, bridge):
        r = repr(bridge)
        assert "MemoryBridge" in r


# ---------------------------------------------------------------------------
# Coordinator Integration
# ---------------------------------------------------------------------------


class TestCoordinatorIntegration:
    def test_store_adds_witness_hash(self):
        """When bridge is set, store() adds witness_hash to info."""
        from mae_core.backbone.event_bus import EventBus
        from mae_core.memory.coordinator import MemoryCoordinator

        bus = EventBus()
        mc = MemoryCoordinator(bus, agent_id="test", enable_semantic=False)
        mc._bridge = MagicMock()  # Just needs to be truthy

        state = np.random.rand(8).astype(np.float32)
        next_state = np.random.rand(8).astype(np.float32)
        mc.store(state=state, action=0, reward=0.5, next_state=next_state, done=False)

        # The experience stored in episodic should have witness_hash
        experiences, _, _ = mc.episodic.sample(1)
        assert "witness_hash" in experiences[0].info
        assert len(experiences[0].info["witness_hash"]) == 64

    def test_store_without_bridge_no_hash(self):
        """Without bridge, no witness hash overhead."""
        from mae_core.backbone.event_bus import EventBus
        from mae_core.memory.coordinator import MemoryCoordinator

        bus = EventBus()
        mc = MemoryCoordinator(bus, agent_id="test", enable_semantic=False)

        state = np.random.rand(8).astype(np.float32)
        mc.store(state=state, action=0, reward=0.5, next_state=state, done=False)

        experiences, _, _ = mc.episodic.sample(1)
        assert "witness_hash" not in experiences[0].info

    def test_search_with_deep_fallback(self):
        """Search cascades to deep memory when local results insufficient."""
        from mae_core.backbone.event_bus import EventBus
        from mae_core.memory.coordinator import MemoryCoordinator

        bus = EventBus()
        mc = MemoryCoordinator(bus, agent_id="test", enable_semantic=False)

        mock_bridge = MagicMock()
        mock_bridge.recall_from_deep.return_value = [
            SearchResult(point_id="deep-1", score=0.8, payload={}, text="from deep"),
        ]
        mc._bridge = mock_bridge

        # Search with no local results but with query_text
        results = mc.search(np.random.rand(8), k=5, query_text="exploring")
        assert len(results) == 1
        assert results[0].text == "from deep"
        mock_bridge.recall_from_deep.assert_called_once()

    def test_search_no_deep_without_query_text(self):
        """Deep fallback only fires when query_text provided."""
        from mae_core.backbone.event_bus import EventBus
        from mae_core.memory.coordinator import MemoryCoordinator

        bus = EventBus()
        mc = MemoryCoordinator(bus, agent_id="test", enable_semantic=False)

        mock_bridge = MagicMock()
        mc._bridge = mock_bridge

        mc.search(np.random.rand(8), k=5)
        mock_bridge.recall_from_deep.assert_not_called()

    def test_build_agent_context(self):
        """_build_agent_context extracts info from agent."""
        from mae_core.backbone.event_bus import EventBus
        from mae_core.memory.coordinator import MemoryCoordinator

        bus = EventBus()
        mc = MemoryCoordinator(bus, agent_id="agent-3", enable_semantic=False)

        mock_agent = MagicMock()
        mock_agent.role = "EXPLORER"
        mock_agent.step_count = 500
        mock_agent._holon = None
        mock_agent.circadian = None

        ctx = mc._build_agent_context(mock_agent)
        assert ctx["agent_id"] == "agent-3"
        assert ctx["agent_role"] == "EXPLORER"
        assert ctx["step"] == 500
