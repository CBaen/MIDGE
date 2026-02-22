"""Tests for TriageClassifier and MemoryBridge HAVEN filtering + EventBus recall notifications.

Tests both:
- Task #7: Triadic recall enforcement (MemoryBridge HAVEN filtering + EventBus notifications)
- Task #8: TRIAGE biological urgency classification

All external systems are mocked. No real services required.
"""

import time
from unittest.mock import MagicMock, Mock, patch, call
import pytest

from mae_core.communication.triage_classifier import (
    TriageCategory,
    TriageClassifier,
    TriageResult,
)
from mae_core.communication.signal_priority import (
    SignalPriorityResolver,
    PriorityConfig,
)
from mae_core.communication.signal_bus import Signal, SignalBus
from mae_core.memory.memory_bridge import MemoryBridge
from mae_core.memory.deep_memory import SearchResult
from mae_core.backbone.event_bus import EventBus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_nociception():
    """Mock NociceptionSystem."""
    mock = MagicMock()
    mock.is_in_pain.return_value = False
    mock.get_pain_load.return_value = 0.0
    return mock


@pytest.fixture
def mock_threat_detector():
    """Mock ThreatDetector."""
    mock = MagicMock()
    mock.is_shelled = False
    mock.get_active_threats.return_value = []
    return mock


@pytest.fixture
def mock_endocrine():
    """Mock EndocrineSystem."""
    mock = MagicMock()
    mock.is_stressed.return_value = False
    mock.get_urgency_level.return_value = 0.0
    return mock


@pytest.fixture
def triage_classifier(mock_nociception, mock_threat_detector, mock_endocrine):
    """TriageClassifier with all three biological systems."""
    return TriageClassifier(
        nociception=mock_nociception,
        threat_detector=mock_threat_detector,
        endocrine=mock_endocrine,
    )


@pytest.fixture
def mock_deep_store():
    """Mock DeepMemoryStore."""
    mock = MagicMock()
    mock.is_available.return_value = True
    mock.search.return_value = []
    mock.get_collection_stats.return_value = {"points_count": 0}
    return mock


@pytest.fixture
def mock_narrator():
    """Mock ExperienceNarrator."""
    mock = MagicMock()
    return mock


@pytest.fixture
def mock_haven():
    """Mock HavenRiskCoordinator."""
    mock = MagicMock()
    mock.is_agent_isolated.return_value = False
    return mock


@pytest.fixture
def event_bus():
    """Real EventBus for testing."""
    return EventBus()


@pytest.fixture
def signal_bus(event_bus):
    """Real SignalBus for testing."""
    return SignalBus(event_bus)


@pytest.fixture
def mock_event_bus():
    """Mock EventBus for testing."""
    mock = MagicMock()
    mock.publish = MagicMock()
    return mock


# ---------------------------------------------------------------------------
# TestTriageClassifier - Classification rules and state tracking
# ---------------------------------------------------------------------------


class TestTriageClassifier:
    """Tests for TriageClassifier medical triage rules."""

    def test_initialize_all_systems(self, mock_nociception, mock_threat_detector, mock_endocrine):
        classifier = TriageClassifier(
            nociception=mock_nociception,
            threat_detector=mock_threat_detector,
            endocrine=mock_endocrine,
        )
        assert classifier._nociception is mock_nociception
        assert classifier._threat_detector is mock_threat_detector
        assert classifier._endocrine is mock_endocrine

    def test_initialize_no_systems(self):
        classifier = TriageClassifier()
        assert classifier._nociception is None
        assert classifier._threat_detector is None
        assert classifier._endocrine is None

    def test_immediate_pain_overload(self, triage_classifier, mock_nociception):
        mock_nociception.get_pain_load.return_value = 1.6
        result = triage_classifier.classify("DANGER", 0.5)
        assert result.category == TriageCategory.IMMEDIATE
        assert result.urgency_boost == 0.30
        assert result.reason == "pain_overload"

    def test_immediate_critical_threat_high_priority(self, triage_classifier, mock_threat_detector):
        mock_threat_detector.is_shelled = True
        mock_threat_detector.get_active_threats.return_value = [{"score": 0.9}]
        result = triage_classifier.classify("DANGER", 0.75)
        assert result.category == TriageCategory.IMMEDIATE
        assert result.urgency_boost == 0.30
        assert result.reason == "critical_threat_high_priority"

    def test_immediate_threshold_edge_threat_score(self, triage_classifier, mock_threat_detector):
        mock_threat_detector.is_shelled = False
        mock_threat_detector.get_active_threats.return_value = [{"score": 0.81}]
        result = triage_classifier.classify("DANGER", 0.71)
        assert result.category == TriageCategory.IMMEDIATE

    def test_not_immediate_below_pain_threshold(self, triage_classifier, mock_nociception):
        mock_nociception.get_pain_load.return_value = 1.5
        result = triage_classifier.classify("DANGER", 0.5)
        assert result.category != TriageCategory.IMMEDIATE

    def test_not_immediate_threat_priority_too_low(self, triage_classifier, mock_threat_detector):
        mock_threat_detector.is_shelled = True
        mock_threat_detector.get_active_threats.return_value = [{"score": 0.9}]
        result = triage_classifier.classify("DANGER", 0.69)
        assert result.category != TriageCategory.IMMEDIATE

    def test_urgent_in_pain_moderate_priority(self, triage_classifier, mock_nociception):
        mock_nociception.is_in_pain.return_value = True
        result = triage_classifier.classify("OPPORTUNITY", 0.51)
        assert result.category == TriageCategory.URGENT
        assert result.urgency_boost == 0.15
        assert result.reason == "pain_moderate_priority"

    def test_urgent_active_defense(self, triage_classifier, mock_threat_detector):
        mock_threat_detector.is_shelled = True
        result = triage_classifier.classify("DANGER", 0.41)
        assert result.category == TriageCategory.URGENT
        assert result.reason == "active_defense"

    def test_urgent_stressed_high_priority(self, triage_classifier, mock_endocrine):
        mock_endocrine.is_stressed.return_value = True
        result = triage_classifier.classify("OPPORTUNITY", 0.51)
        assert result.category == TriageCategory.URGENT
        assert result.reason == "stressed_high_priority"

    def test_expectant_low_priority_during_overload(self, triage_classifier, mock_nociception):
        mock_nociception.get_pain_load.return_value = 1.1
        result = triage_classifier.classify("KNOWLEDGE_SHARE", 0.29)
        assert result.category == TriageCategory.EXPECTANT
        assert result.urgency_boost == -0.10
        assert result.reason == "low_priority_during_overload"

    def test_delayed_stable_state(self, triage_classifier):
        result = triage_classifier.classify("OPPORTUNITY", 0.5)
        assert result.category == TriageCategory.DELAYED
        assert result.urgency_boost == 0.0
        assert result.reason == "stable"

    def test_classification_counter_increments(self, triage_classifier):
        stats_before = triage_classifier.get_statistics()
        assert stats_before["total_classified"] == 0
        triage_classifier.classify("DANGER", 0.5)
        triage_classifier.classify("OPPORTUNITY", 0.5)
        stats_after = triage_classifier.get_statistics()
        assert stats_after["total_classified"] == 2

    def test_category_counts_tracking(self, triage_classifier, mock_nociception):
        mock_nociception.get_pain_load.return_value = 1.6
        triage_classifier.classify("DANGER", 0.5)  # IMMEDIATE
        mock_nociception.get_pain_load.return_value = 0.0
        triage_classifier.classify("OPPORTUNITY", 0.5)  # DELAYED
        stats = triage_classifier.get_statistics()
        assert stats["category_counts"]["immediate"] == 1
        assert stats["category_counts"]["delayed"] == 1
        assert stats["category_counts"]["urgent"] == 0
        assert stats["category_counts"]["expectant"] == 0

    def test_get_statistics_availability_flags(self, mock_nociception, mock_threat_detector):
        classifier = TriageClassifier(
            nociception=mock_nociception,
            threat_detector=mock_threat_detector,
            endocrine=None,
        )
        stats = classifier.get_statistics()
        assert stats["nociception_available"] is True
        assert stats["threat_detector_available"] is True
        assert stats["endocrine_available"] is False

    def test_graceful_degradation_nociception_error(self, triage_classifier, mock_nociception):
        mock_nociception.get_pain_load.side_effect = RuntimeError("System failure")
        mock_nociception.is_in_pain.side_effect = RuntimeError("System failure")
        result = triage_classifier.classify("DANGER", 0.5)
        assert result.category == TriageCategory.DELAYED
        assert result.reason == "stable"

    def test_graceful_degradation_threat_error(self, triage_classifier, mock_threat_detector):
        mock_threat_detector.get_active_threats.side_effect = RuntimeError("System failure")
        result = triage_classifier.classify("DANGER", 0.5)
        assert result.category == TriageCategory.DELAYED
        assert result.reason == "stable"

    def test_graceful_degradation_endocrine_error(self, triage_classifier, mock_endocrine):
        mock_endocrine.is_stressed.side_effect = RuntimeError("System failure")
        result = triage_classifier.classify("OPPORTUNITY", 0.51)
        assert result.category == TriageCategory.DELAYED
        assert result.reason == "stable"

    def test_no_systems_available_always_delayed(self):
        classifier = TriageClassifier()
        result = classifier.classify("DANGER", 1.0)
        assert result.category == TriageCategory.DELAYED
        assert result.urgency_boost == 0.0

    def test_threat_without_active_threats(self, triage_classifier, mock_threat_detector):
        mock_threat_detector.is_shelled = True
        mock_threat_detector.get_active_threats.return_value = []
        result = triage_classifier.classify("DANGER", 0.9)
        assert result.category != TriageCategory.IMMEDIATE

    def test_repr_with_all_systems(self, triage_classifier):
        repr_str = repr(triage_classifier)
        assert "TriageClassifier" in repr_str
        assert "noci=yes" in repr_str
        assert "threat=yes" in repr_str
        assert "endo=yes" in repr_str

    def test_repr_with_no_systems(self):
        classifier = TriageClassifier()
        repr_str = repr(classifier)
        assert "TriageClassifier" in repr_str
        assert "noci=no" in repr_str
        assert "threat=no" in repr_str
        assert "endo=no" in repr_str


# ---------------------------------------------------------------------------
# TestTriageIntegration - Integration with SignalPriorityResolver
# ---------------------------------------------------------------------------


class TestTriageIntegration:
    """Tests for triage integration with SignalPriorityResolver."""

    def test_signal_priority_without_triage(self, signal_bus):
        resolver = SignalPriorityResolver(
            agent_id="agent_1",
            signal_bus=signal_bus,
        )
        signal = Signal(
            signal_type="DANGER",
            payload={"data": "test"},
            sender_id="sender_1",
            priority=0.5,
            timestamp=time.time(),
        )
        score = resolver._compute_score(signal)
        assert 0.0 <= score <= 1.0
        assert resolver._triage_classifier is None

    def test_signal_priority_with_triage(self, signal_bus, triage_classifier):
        resolver = SignalPriorityResolver(
            agent_id="agent_1",
            signal_bus=signal_bus,
        )
        resolver._triage_classifier = triage_classifier
        signal = Signal(
            signal_type="DANGER",
            payload={"data": "test"},
            sender_id="sender_1",
            priority=0.5,
            timestamp=time.time(),
        )
        score_with_triage = resolver._compute_score(signal)
        assert 0.0 <= score_with_triage <= 1.0

    def test_urgency_boost_applied_to_score(self, signal_bus, triage_classifier, mock_nociception):
        resolver = SignalPriorityResolver(
            agent_id="agent_1",
            signal_bus=signal_bus,
        )
        resolver._triage_classifier = triage_classifier
        mock_nociception.get_pain_load.return_value = 1.6  # IMMEDIATE
        signal = Signal(
            signal_type="DANGER",
            payload={"data": "test"},
            sender_id="sender_1",
            priority=0.5,
            timestamp=time.time(),
        )
        score = resolver._compute_score(signal)
        assert score > 0.5  # boost should increase score

    def test_triage_classifier_error_graceful(self, signal_bus):
        mock_classifier = MagicMock()
        mock_classifier.classify.side_effect = RuntimeError("Classifier error")
        resolver = SignalPriorityResolver(
            agent_id="agent_1",
            signal_bus=signal_bus,
        )
        resolver._triage_classifier = mock_classifier
        signal = Signal(
            signal_type="DANGER",
            payload={"data": "test"},
            sender_id="sender_1",
            priority=0.5,
            timestamp=time.time(),
        )
        score = resolver._compute_score(signal)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# TestMemoryBridgeHaven - HAVEN isolation filtering in recall
# ---------------------------------------------------------------------------


class TestMemoryBridgeHaven:
    """Tests for MemoryBridge HAVEN filtering of isolated agents."""

    def test_recall_peer_experiences_without_haven(self, mock_deep_store, mock_narrator, mock_event_bus):
        bridge = MemoryBridge(
            deep_store=mock_deep_store,
            narrator=mock_narrator,
            event_bus=mock_event_bus,
            haven=None,
        )
        mock_deep_store.search.return_value = [
            SearchResult(
                point_id="1",
                score=0.9,
                payload={"agent_id": "peer_1"},
                text="Experience 1",
            )
        ]
        results = bridge.recall_peer_experiences(
            query_text="test",
            requesting_agent_id="agent_1",
            peer_agent_ids=["peer_1"],
            limit=5,
        )
        assert len(results) == 1
        assert bridge._total_blocked_by_isolation == 0

    def test_recall_peer_experiences_with_haven_all_isolated(
        self, mock_deep_store, mock_narrator, mock_event_bus, mock_haven
    ):
        mock_haven.is_agent_isolated.return_value = True
        bridge = MemoryBridge(
            deep_store=mock_deep_store,
            narrator=mock_narrator,
            event_bus=mock_event_bus,
            haven=mock_haven,
        )
        mock_deep_store.search.return_value = []
        results = bridge.recall_peer_experiences(
            query_text="test",
            requesting_agent_id="agent_1",
            peer_agent_ids=["peer_1", "peer_2"],
            limit=5,
        )
        assert len(results) == 0
        assert bridge._total_blocked_by_isolation == 2

    def test_recall_peer_experiences_with_haven_some_isolated(
        self, mock_deep_store, mock_narrator, mock_event_bus, mock_haven
    ):
        def is_isolated(agent_id):
            return agent_id == "peer_2"

        mock_haven.is_agent_isolated.side_effect = is_isolated
        bridge = MemoryBridge(
            deep_store=mock_deep_store,
            narrator=mock_narrator,
            event_bus=mock_event_bus,
            haven=mock_haven,
        )
        mock_deep_store.search.return_value = [
            SearchResult(
                point_id="1",
                score=0.9,
                payload={"agent_id": "peer_1"},
                text="Experience from peer_1",
            )
        ]
        results = bridge.recall_peer_experiences(
            query_text="test",
            requesting_agent_id="agent_1",
            peer_agent_ids=["peer_1", "peer_2"],
            limit=5,
        )
        assert len(results) == 1
        assert bridge._total_blocked_by_isolation == 1

    def test_recall_peer_experiences_post_filter_isolated_results(
        self, mock_deep_store, mock_narrator, mock_event_bus, mock_haven
    ):
        def is_isolated(agent_id):
            return agent_id == "peer_2"

        mock_haven.is_agent_isolated.side_effect = is_isolated
        bridge = MemoryBridge(
            deep_store=mock_deep_store,
            narrator=mock_narrator,
            event_bus=mock_event_bus,
            haven=mock_haven,
        )
        mock_deep_store.search.return_value = [
            SearchResult(
                point_id="1",
                score=0.9,
                payload={"agent_id": "peer_1"},
                text="Experience from peer_1",
            ),
            SearchResult(
                point_id="2",
                score=0.8,
                payload={"agent_id": "peer_2"},
                text="Experience from peer_2",
            ),
        ]
        results = bridge.recall_peer_experiences(
            query_text="test",
            requesting_agent_id="agent_1",
            peer_agent_ids=["peer_1", "peer_2"],
            limit=5,
        )
        assert len(results) == 1
        assert results[0].payload["agent_id"] == "peer_1"
        assert bridge._total_blocked_by_isolation == 2

    def test_recall_peer_experiences_empty_peer_list(self, mock_deep_store, mock_narrator, mock_event_bus):
        bridge = MemoryBridge(
            deep_store=mock_deep_store,
            narrator=mock_narrator,
            event_bus=mock_event_bus,
            haven=None,
        )
        results = bridge.recall_peer_experiences(
            query_text="test",
            requesting_agent_id="agent_1",
            peer_agent_ids=[],
            limit=5,
        )
        assert len(results) == 0

    def test_recall_peer_experiences_store_unavailable(self, mock_deep_store, mock_narrator):
        mock_deep_store.is_available.return_value = False
        bridge = MemoryBridge(
            deep_store=mock_deep_store,
            narrator=mock_narrator,
            event_bus=None,
            haven=None,
        )
        results = bridge.recall_peer_experiences(
            query_text="test",
            requesting_agent_id="agent_1",
            peer_agent_ids=["peer_1"],
            limit=5,
        )
        assert len(results) == 0

    def test_isolation_tracking_cumulative(
        self, mock_deep_store, mock_narrator, mock_event_bus, mock_haven
    ):
        mock_haven.is_agent_isolated.return_value = True
        bridge = MemoryBridge(
            deep_store=mock_deep_store,
            narrator=mock_narrator,
            event_bus=mock_event_bus,
            haven=mock_haven,
        )
        bridge.recall_peer_experiences(
            query_text="test",
            requesting_agent_id="agent_1",
            peer_agent_ids=["peer_1"],
            limit=5,
        )
        bridge.recall_peer_experiences(
            query_text="test",
            requesting_agent_id="agent_1",
            peer_agent_ids=["peer_2"],
            limit=5,
        )
        assert bridge._total_blocked_by_isolation == 2


# ---------------------------------------------------------------------------
# TestMemoryBridgeRecallEvents - EventBus recall notifications
# ---------------------------------------------------------------------------


class TestMemoryBridgeRecallEvents:
    """Tests for MemoryBridge EventBus recall event notifications (triadic witnessing)."""

    def test_recall_from_deep_publishes_event(self, mock_deep_store, mock_narrator, mock_event_bus):
        bridge = MemoryBridge(
            deep_store=mock_deep_store,
            narrator=mock_narrator,
            event_bus=mock_event_bus,
            haven=None,
        )
        mock_deep_store.search.return_value = [
            SearchResult(
                point_id="1",
                score=0.9,
                payload={"agent_id": "agent_1"},
                text="Experience",
            )
        ]
        bridge.recall_from_deep(
            query_text="What happened yesterday?",
            agent_id="agent_1",
            limit=5,
        )
        mock_event_bus.publish.assert_called()
        call_args = mock_event_bus.publish.call_args
        assert call_args[0][0] == "memory.recall_completed"
        assert call_args[0][1]["recall_type"] == "recall_from_deep"
        assert call_args[0][1]["requester"] == "agent_1"
        assert call_args[0][1]["result_count"] == 1
        assert call_args[0][1]["query_preview"] == "What happened yesterday?"

    def test_recall_from_deep_query_preview_truncation(self, mock_deep_store, mock_narrator, mock_event_bus):
        bridge = MemoryBridge(
            deep_store=mock_deep_store,
            narrator=mock_narrator,
            event_bus=mock_event_bus,
            haven=None,
        )
        mock_deep_store.search.return_value = []
        long_query = "x" * 100
        bridge.recall_from_deep(
            query_text=long_query,
            agent_id="agent_1",
            limit=5,
        )
        call_args = mock_event_bus.publish.call_args
        assert len(call_args[0][1]["query_preview"]) <= 80

    def test_recall_peer_experiences_publishes_event(self, mock_deep_store, mock_narrator, mock_event_bus):
        bridge = MemoryBridge(
            deep_store=mock_deep_store,
            narrator=mock_narrator,
            event_bus=mock_event_bus,
            haven=None,
        )
        mock_deep_store.search.return_value = [
            SearchResult(
                point_id="1",
                score=0.9,
                payload={"agent_id": "peer_1"},
                text="Peer experience",
            )
        ]
        bridge.recall_peer_experiences(
            query_text="peer strategy",
            requesting_agent_id="agent_1",
            peer_agent_ids=["peer_1"],
            limit=5,
        )
        mock_event_bus.publish.assert_called()
        call_args = mock_event_bus.publish.call_args
        assert call_args[0][0] == "memory.recall_completed"
        assert call_args[0][1]["recall_type"] == "recall_peer_experiences"
        assert call_args[0][1]["requester"] == "agent_1"

    def test_recall_ancestral_patterns_publishes_event(self, mock_deep_store, mock_narrator, mock_event_bus):
        bridge = MemoryBridge(
            deep_store=mock_deep_store,
            narrator=mock_narrator,
            event_bus=mock_event_bus,
            haven=None,
        )
        mock_deep_store.search.return_value = [
            SearchResult(
                point_id="1",
                score=0.95,
                payload={"type": "pattern"},
                text="Pattern",
            )
        ]
        bridge.recall_ancestral_patterns(
            query_text="cooperation pattern",
            applicable_role="NEGOTIATOR",
            limit=5,
        )
        mock_event_bus.publish.assert_called()
        call_args = mock_event_bus.publish.call_args
        assert call_args[0][0] == "memory.recall_completed"
        assert call_args[0][1]["recall_type"] == "recall_ancestral_patterns"
        assert call_args[0][1]["requester"] == "NEGOTIATOR"

    def test_no_event_when_bus_is_none(self, mock_deep_store, mock_narrator):
        bridge = MemoryBridge(
            deep_store=mock_deep_store,
            narrator=mock_narrator,
            event_bus=None,
            haven=None,
        )
        mock_deep_store.search.return_value = []
        result = bridge.recall_from_deep(
            query_text="test",
            agent_id="agent_1",
            limit=5,
        )
        assert result == []

    def test_no_event_when_bus_publish_fails(self, mock_deep_store, mock_narrator, mock_event_bus):
        mock_event_bus.publish.side_effect = RuntimeError("Bus error")
        bridge = MemoryBridge(
            deep_store=mock_deep_store,
            narrator=mock_narrator,
            event_bus=mock_event_bus,
            haven=None,
        )
        mock_deep_store.search.return_value = []
        result = bridge.recall_from_deep(
            query_text="test",
            agent_id="agent_1",
            limit=5,
        )
        assert result == []

    def test_multiple_recalls_all_publish(self, mock_deep_store, mock_narrator, mock_event_bus):
        bridge = MemoryBridge(
            deep_store=mock_deep_store,
            narrator=mock_narrator,
            event_bus=mock_event_bus,
            haven=None,
        )
        mock_deep_store.search.return_value = [
            SearchResult(
                point_id="1",
                score=0.9,
                payload={},
                text="Result",
            )
        ]
        bridge.recall_from_deep(query_text="query1", agent_id="agent_1", limit=5)
        bridge.recall_from_deep(query_text="query2", agent_id="agent_2", limit=5)
        # Each recall publishes 2 events: memory.recall_verified + memory.recall_completed
        assert mock_event_bus.publish.call_count == 4
        event_names = [c.args[0] for c in mock_event_bus.publish.call_args_list]
        assert event_names.count("memory.recall_verified") == 2
        assert event_names.count("memory.recall_completed") == 2


# ---------------------------------------------------------------------------
# TestMemoryBridgeStats - Statistics and integration
# ---------------------------------------------------------------------------


class TestMemoryBridgeStats:
    """Tests for MemoryBridge statistics tracking."""

    def test_stats_include_isolation_count(self, mock_deep_store, mock_narrator, mock_event_bus, mock_haven):
        mock_haven.is_agent_isolated.return_value = True
        bridge = MemoryBridge(
            deep_store=mock_deep_store,
            narrator=mock_narrator,
            event_bus=mock_event_bus,
            haven=mock_haven,
        )
        bridge.recall_peer_experiences(
            query_text="test",
            requesting_agent_id="agent_1",
            peer_agent_ids=["peer_1"],
            limit=5,
        )
        stats = bridge.get_stats()
        assert stats["total_blocked_by_isolation"] == 1

    def test_repr_includes_consolidation_and_recall(self, mock_deep_store, mock_narrator):
        bridge = MemoryBridge(
            deep_store=mock_deep_store,
            narrator=mock_narrator,
            event_bus=None,
            haven=None,
        )
        repr_str = repr(bridge)
        assert "MemoryBridge" in repr_str
        assert "consolidated=0" in repr_str
        assert "recalled=0" in repr_str
