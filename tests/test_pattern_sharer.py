"""Tests for PatternSharer -- triadic pattern communication."""

from __future__ import annotations

from collections import defaultdict
from unittest.mock import MagicMock

import pytest

from mae_core.patterns.pattern_sharer import PatternSharer
from mae_core.patterns.pattern_signal import (
    PatternDomain,
    PatternForm,
    PatternSignal,
)
from mae_core.patterns.translators.triadic import TriadicPatternTranslator


# ── Helpers ──────────────────────────────────────────────────────────

class FakeHolonRegistry:
    """Minimal registry that tracks parent-child relationships."""

    def __init__(self):
        self._parent: dict[str, str] = {}
        self._children: dict[str, set[str]] = defaultdict(set)

    def register(self, holon_id: str, parent_id: str = None):
        if parent_id:
            self._parent[holon_id] = parent_id
            self._children[parent_id].add(holon_id)

    def get_peers(self, holon_id: str) -> list[str]:
        parent = self._parent.get(holon_id)
        if parent is None:
            return []
        return sorted(s for s in self._children[parent] if s != holon_id)

    def get_parent(self, holon_id: str) -> str | None:
        return self._parent.get(holon_id)


class FakeEventBus:
    def __init__(self):
        self.published: list[tuple[str, dict]] = []

    def publish(self, channel: str, data):
        self.published.append((channel, data))


def _make_signal(domain: PatternDomain, salience: float = 0.5) -> PatternSignal:
    return PatternSignal(
        source_system="test",
        domain=domain,
        form=PatternForm.REACTIVE,
        confidence=0.7,
        salience=salience,
        description=f"Test {domain.value}",
    )


def _setup_triad(registry: FakeHolonRegistry) -> tuple[str, str, str]:
    """Create a triad of agents under a shared parent."""
    registry.register("triad-A", parent_id="tissue-1")
    registry.register("agent-0", parent_id="triad-A")
    registry.register("agent-1", parent_id="triad-A")
    registry.register("agent-2", parent_id="triad-A")
    return "agent-0", "agent-1", "agent-2"


# ── Construction ─────────────────────────────────────────────────────

class TestConstruction:
    def test_creates_with_agent_id(self):
        ps = PatternSharer("agent-0", holon_registry=None)
        assert ps.agent_id == "agent-0"

    def test_repr(self):
        ps = PatternSharer("agent-1", holon_registry=None)
        assert "agent-1" in repr(ps)

    def test_statistics_empty(self):
        ps = PatternSharer("agent-0", holon_registry=None)
        stats = ps.get_statistics()
        assert stats["total_shares"] == 0
        assert stats["total_consensus"] == 0


# ── Peer Discovery ───────────────────────────────────────────────────

class TestPeerDiscovery:
    def test_gets_triad_mates(self):
        reg = FakeHolonRegistry()
        _setup_triad(reg)
        ps = PatternSharer("agent-0", holon_registry=reg)

        mates = ps.get_triad_mates()
        assert sorted(mates) == ["agent-1", "agent-2"]

    def test_no_mates_without_registry(self):
        ps = PatternSharer("agent-0", holon_registry=None)
        assert ps.get_triad_mates() == []

    def test_no_mates_when_alone(self):
        reg = FakeHolonRegistry()
        reg.register("agent-lonely", parent_id="island")
        ps = PatternSharer("agent-lonely", holon_registry=reg)
        assert ps.get_triad_mates() == []


# ── Sharing ──────────────────────────────────────────────────────────

class TestSharing:
    def test_share_sends_via_gnn(self):
        reg = FakeHolonRegistry()
        _setup_triad(reg)

        gnn = MagicMock()
        ps = PatternSharer("agent-0", holon_registry=reg, gnn_communicator=gnn)

        signals = [_make_signal(PatternDomain.THREAT)]
        ps.share(signals)

        gnn.send_message.assert_called_once()
        kwargs = gnn.send_message.call_args[1]
        assert kwargs["sender_id"] == "agent-0"
        assert kwargs["message_type"] == "KNOWLEDGE_SHARE"
        assert set(kwargs["target_ids"]) == {"agent-1", "agent-2"}
        payload = kwargs["content"]
        assert payload["type"] == "pattern_share"
        assert payload["sender"] == "agent-0"
        assert "threat" in payload["domains"]

    def test_no_share_on_empty_signals(self):
        reg = FakeHolonRegistry()
        _setup_triad(reg)
        gnn = MagicMock()
        ps = PatternSharer("agent-0", holon_registry=reg, gnn_communicator=gnn)

        ps.share([])
        gnn.send_message.assert_not_called()

    def test_share_increments_counter(self):
        reg = FakeHolonRegistry()
        _setup_triad(reg)
        gnn = MagicMock()
        ps = PatternSharer("agent-0", holon_registry=reg, gnn_communicator=gnn)

        ps.share([_make_signal(PatternDomain.NOVELTY)])
        assert ps.get_statistics()["total_shares"] == 1

    def test_graceful_without_gnn(self):
        reg = FakeHolonRegistry()
        _setup_triad(reg)
        ps = PatternSharer("agent-0", holon_registry=reg, gnn_communicator=None)

        # Should not raise
        ps.share([_make_signal(PatternDomain.THREAT)])


# ── Consensus Detection ──────────────────────────────────────────────

class TestConsensus:
    def test_consensus_on_two_of_three(self):
        reg = FakeHolonRegistry()
        _setup_triad(reg)
        eb = FakeEventBus()
        ps = PatternSharer("agent-0", holon_registry=reg, event_bus=eb)

        # Simulate peer signal from agent-1
        ps.receive_peer_signal("agent-1", {
            "type": "pattern_share",
            "sender": "agent-1",
            "domains": ["threat"],
        })

        # Own signals include THREAT
        own = [_make_signal(PatternDomain.THREAT)]
        result = ps.receive_and_correlate(own)

        assert len(result) == 1
        assert result[0].domain == PatternDomain.THREAT
        assert result[0].form == PatternForm.CORRELATED
        assert "consensus" in result[0].description.lower()

    def test_no_consensus_on_one_of_three(self):
        reg = FakeHolonRegistry()
        _setup_triad(reg)
        ps = PatternSharer("agent-0", holon_registry=reg)

        # Only self reports THREAT (no peer agreement)
        own = [_make_signal(PatternDomain.THREAT)]
        result = ps.receive_and_correlate(own)

        assert len(result) == 0

    def test_consensus_publishes_to_event_bus(self):
        reg = FakeHolonRegistry()
        _setup_triad(reg)
        eb = FakeEventBus()
        ps = PatternSharer("agent-0", holon_registry=reg, event_bus=eb)

        ps.receive_peer_signal("agent-1", {
            "domains": ["novelty"],
        })
        own = [_make_signal(PatternDomain.NOVELTY)]
        ps.receive_and_correlate(own)

        assert len(eb.published) == 1
        channel, data = eb.published[0]
        assert channel == "pattern.triadic_correlation"
        assert data["domain"] == "novelty"

    def test_consensus_clears_inbox(self):
        reg = FakeHolonRegistry()
        _setup_triad(reg)
        ps = PatternSharer("agent-0", holon_registry=reg)

        ps.receive_peer_signal("agent-1", {"domains": ["threat"]})
        ps.receive_and_correlate([_make_signal(PatternDomain.THREAT)])

        # Inbox should be cleared
        assert ps.get_statistics()["inbox_peers"] == 0

    def test_multiple_domains_consensus(self):
        reg = FakeHolonRegistry()
        _setup_triad(reg)
        ps = PatternSharer("agent-0", holon_registry=reg)

        # Peer reports both THREAT and NOVELTY
        ps.receive_peer_signal("agent-1", {
            "domains": ["threat", "novelty"],
        })

        own = [
            _make_signal(PatternDomain.THREAT),
            _make_signal(PatternDomain.NOVELTY),
        ]
        result = ps.receive_and_correlate(own)

        domains = {s.domain for s in result}
        assert PatternDomain.THREAT in domains
        assert PatternDomain.NOVELTY in domains

    def test_consensus_increments_counter(self):
        reg = FakeHolonRegistry()
        _setup_triad(reg)
        ps = PatternSharer("agent-0", holon_registry=reg)

        ps.receive_peer_signal("agent-1", {"domains": ["threat"]})
        ps.receive_and_correlate([_make_signal(PatternDomain.THREAT)])

        assert ps.get_statistics()["total_consensus"] == 1

    def test_source_system_includes_triad_id(self):
        reg = FakeHolonRegistry()
        _setup_triad(reg)
        ps = PatternSharer("agent-0", holon_registry=reg)

        ps.receive_peer_signal("agent-1", {"domains": ["threat"]})
        result = ps.receive_and_correlate([_make_signal(PatternDomain.THREAT)])

        assert result[0].source_system.startswith("triad:")

    def test_graceful_without_holon_registry(self):
        ps = PatternSharer("agent-0", holon_registry=None)
        result = ps.receive_and_correlate([_make_signal(PatternDomain.THREAT)])
        assert result == []


# ── GNN Handler Routing ──────────────────────────────────────────────

class TestGNNHandlerRouting:
    """Verify the GNN handler factory routes KNOWLEDGE_SHARE to receive_peer_signal."""

    def test_handler_routes_pattern_share_to_inbox(self):
        reg = FakeHolonRegistry()
        _setup_triad(reg)
        sharer = PatternSharer("agent-1", holon_registry=reg)

        # Reproduce the handler factory from main.py
        def _make_pattern_handler(sharer_ref):
            def handler(message):
                content = getattr(message, "content", {})
                if isinstance(content, dict) and content.get("type") == "pattern_share":
                    sharer_ref.receive_peer_signal(
                        content.get("sender", "unknown"), content,
                    )
            return handler

        handler = _make_pattern_handler(sharer)

        # Simulate a GNN message object
        msg = MagicMock()
        msg.content = {
            "type": "pattern_share",
            "sender": "agent-0",
            "domains": ["threat"],
            "signals": [],
        }
        handler(msg)

        assert sharer.get_statistics()["inbox_peers"] == 1

    def test_handler_ignores_non_pattern_messages(self):
        sharer = PatternSharer("agent-1", holon_registry=None)

        def _make_pattern_handler(sharer_ref):
            def handler(message):
                content = getattr(message, "content", {})
                if isinstance(content, dict) and content.get("type") == "pattern_share":
                    sharer_ref.receive_peer_signal(
                        content.get("sender", "unknown"), content,
                    )
            return handler

        handler = _make_pattern_handler(sharer)

        msg = MagicMock()
        msg.content = {"type": "capability_announcement", "data": "something"}
        handler(msg)

        assert sharer.get_statistics()["inbox_peers"] == 0


# ── Triadic Translator ───────────────────────────────────────────────

class TestTriadicTranslator:
    def setup_method(self):
        self.t = TriadicPatternTranslator()

    def test_source_name(self):
        assert self.t.source_name == "triadic_consensus"

    def test_channels(self):
        assert "pattern.triadic_correlation" in self.t.channels

    def test_translates_consensus_event(self):
        sig = self.t.translate("pattern.triadic_correlation", {
            "domain": "threat",
            "voters": ["agent-0", "agent-1"],
            "triad_size": 3,
            "agreement_ratio": 0.67,
        })
        assert sig is not None
        assert sig.domain == PatternDomain.THREAT
        assert sig.form == PatternForm.CORRELATED
        assert sig.source_system == "triadic_consensus"

    def test_skips_non_dict(self):
        assert self.t.translate("pattern.triadic_correlation", "string") is None

    def test_skips_missing_domain(self):
        assert self.t.translate("pattern.triadic_correlation", {"voters": []}) is None

    def test_skips_invalid_domain(self):
        assert self.t.translate("pattern.triadic_correlation", {"domain": "nonsense"}) is None

    def test_protocol_compliance(self):
        assert isinstance(self.t.source_name, str)
        assert isinstance(self.t.channels, list)
        assert len(self.t.channels) > 0
        assert callable(self.t.translate)
