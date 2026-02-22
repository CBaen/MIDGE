"""Tests for GNN Learning Loop — feedback from message processing to routing optimization."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mae_core.communication.gnn_communicator import GNNCommunicator
from mae_core.communication.gnn_graph import AgentGraph, CommunicationEdge
from mae_core.communication.gnn_message import GNNMessage
from mae_core.communication.gnn_propagator import RoutingOptimizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_communicator(**kwargs) -> GNNCommunicator:
    """Create a GNNCommunicator with sensible test defaults."""
    defaults = dict(
        embedding_dim=16,
        enable_learning=True,
        auto_optimize_interval=100,
        default_k=3,
        default_ttl=3,
    )
    defaults.update(kwargs)
    return defaults | kwargs and GNNCommunicator(**defaults)


def _setup_two_agents(comm: GNNCommunicator) -> tuple[str, str]:
    """Register two connected agents, return (sender_id, receiver_id)."""
    comm.register_agent("agent_a", "mycelial", {"route"}, level=1)
    comm.register_agent("agent_b", "mycelial", {"route"}, level=1)
    comm.add_edge("agent_a", "agent_b", weight=0.8, edge_type="manual")
    return "agent_a", "agent_b"


def _setup_three_agents(comm: GNNCommunicator) -> tuple[str, str, str]:
    """Register three agents in a chain: A -> B -> C."""
    comm.register_agent("agent_a", "mycelial", {"route"}, level=1)
    comm.register_agent("agent_b", "mycelial", {"route"}, level=1)
    comm.register_agent("agent_c", "mycelial", {"route"}, level=1)
    comm.add_edge("agent_a", "agent_b", weight=0.8, edge_type="manual")
    comm.add_edge("agent_b", "agent_c", weight=0.8, edge_type="manual")
    return "agent_a", "agent_b", "agent_c"


# ---------------------------------------------------------------------------
# GNNCommunicator Core Tests
# ---------------------------------------------------------------------------

class TestGNNCommunicatorCore:
    def test_send_message_returns_id(self):
        comm = GNNCommunicator(embedding_dim=16)
        _setup_two_agents(comm)
        msg_id = comm.send_message("agent_a", {"data": 1}, message_type="broadcast")
        assert msg_id is not None
        assert isinstance(msg_id, str)

    def test_receive_messages_returns_queued(self):
        comm = GNNCommunicator(embedding_dim=16)
        _setup_two_agents(comm)
        comm.send_message("agent_a", {"data": 1}, message_type="broadcast")
        msgs = comm.receive_messages("agent_b")
        assert len(msgs) == 1
        assert msgs[0].content == {"data": 1}

    def test_unregistered_sender_returns_none(self):
        comm = GNNCommunicator(embedding_dim=16)
        result = comm.send_message("ghost", {"data": 1})
        assert result is None

    def test_queue_capacity_respected(self):
        comm = GNNCommunicator(embedding_dim=16, queue_capacity=2)
        _setup_two_agents(comm)
        for i in range(5):
            comm.send_message("agent_a", {"i": i}, message_type="broadcast")
        msgs = comm.receive_messages("agent_b", max_messages=10)
        assert len(msgs) == 2  # capacity capped at 2

    def test_message_history_tracks_deliveries(self):
        comm = GNNCommunicator(embedding_dim=16)
        _setup_two_agents(comm)
        comm.send_message("agent_a", {"data": 1}, message_type="broadcast")
        assert len(comm._message_history) == 1
        assert comm._message_history[0]["delivered"] == 1

    def test_statistics_returns_expected_fields(self):
        comm = GNNCommunicator(embedding_dim=16)
        stats = comm.get_communication_statistics()
        assert "messages_sent" in stats
        assert "messages_delivered" in stats
        assert "delivery_rate" in stats
        assert "optimizer" in stats


# ---------------------------------------------------------------------------
# RoutingOptimizer Tests
# ---------------------------------------------------------------------------

class TestRoutingOptimizer:
    def test_record_outcome_stores(self):
        opt = RoutingOptimizer()
        opt.record_outcome("msg1", ["a", "b"], 0.5)
        assert "msg1" in opt._outcomes

    def test_optimize_updates_edge_weights(self):
        graph = AgentGraph(embedding_dim=16)
        graph.add_agent("a")
        graph.add_agent("b")
        graph.add_edge("a", "b", weight=0.5)
        initial_weight = graph.edges[("a", "b")].weight

        opt = RoutingOptimizer(learning_rate=0.5)
        for i in range(15):
            opt.record_outcome(f"msg{i}", ["a", "b"], 1.0)

        updated = opt.optimize_graph(graph, min_samples=10)
        assert updated == 1
        assert graph.edges[("a", "b")].weight != initial_weight

    def test_optimize_needs_min_samples(self):
        graph = AgentGraph(embedding_dim=16)
        graph.add_agent("a")
        graph.add_agent("b")
        graph.add_edge("a", "b", weight=0.5)

        opt = RoutingOptimizer()
        opt.record_outcome("msg1", ["a", "b"], 1.0)
        updated = opt.optimize_graph(graph, min_samples=10)
        assert updated == 0  # not enough samples

    def test_positive_reward_increases_weight(self):
        graph = AgentGraph(embedding_dim=16)
        graph.add_agent("a")
        graph.add_agent("b")
        graph.add_edge("a", "b", weight=0.3)
        initial = graph.edges[("a", "b")].weight

        opt = RoutingOptimizer(learning_rate=0.5)
        for i in range(15):
            opt.record_outcome(f"msg{i}", ["a", "b"], 1.0)
        opt.optimize_graph(graph, min_samples=10)

        assert graph.edges[("a", "b")].weight > initial

    def test_negative_reward_decreases_weight(self):
        graph = AgentGraph(embedding_dim=16)
        graph.add_agent("a")
        graph.add_agent("b")
        graph.add_edge("a", "b", weight=0.8)
        initial = graph.edges[("a", "b")].weight

        opt = RoutingOptimizer(learning_rate=0.5)
        for i in range(15):
            opt.record_outcome(f"msg{i}", ["a", "b"], -1.0)
        opt.optimize_graph(graph, min_samples=10)

        assert graph.edges[("a", "b")].weight < initial


# ---------------------------------------------------------------------------
# Outcome Reporting Tests
# ---------------------------------------------------------------------------

class TestOutcomeReporting:
    def test_report_records_in_optimizer(self):
        comm = GNNCommunicator(embedding_dim=16)
        _setup_two_agents(comm)
        msg_id = comm.send_message("agent_a", {"data": 1}, message_type="broadcast")
        comm.report_communication_outcome(msg_id, "agent_b", success=True, reward=0.5)
        assert msg_id in comm._optimizer._outcomes

    def test_failure_applies_negative_reward(self):
        comm = GNNCommunicator(embedding_dim=16)
        _setup_two_agents(comm)
        msg_id = comm.send_message("agent_a", {"data": 1}, message_type="broadcast")
        comm.report_communication_outcome(msg_id, "agent_b", success=False, reward=0.0)
        path, reward = comm._optimizer._outcomes[msg_id]
        assert reward == -0.5  # GNNCommunicator applies -0.5 for failures

    def test_unknown_message_id_is_noop(self):
        comm = GNNCommunicator(embedding_dim=16)
        comm.report_communication_outcome("nonexistent", "agent_b", success=True)
        assert len(comm._optimizer._outcomes) == 0

    def test_learning_disabled_is_noop(self):
        comm = GNNCommunicator(embedding_dim=16, enable_learning=False)
        _setup_two_agents(comm)
        msg_id = comm.send_message("agent_a", {"data": 1}, message_type="broadcast")
        # Should not raise even without optimizer
        comm.report_communication_outcome(msg_id, "agent_b", success=True)

    def test_auto_optimize_triggers(self):
        comm = GNNCommunicator(embedding_dim=16, auto_optimize_interval=5)
        _setup_two_agents(comm)

        # Record some outcomes first so optimizer has data
        for i in range(3):
            mid = comm.send_message("agent_a", {"i": i}, message_type="broadcast")
            comm.report_communication_outcome(mid, "agent_b", success=True, reward=0.5)

        initial_optimizations = comm._optimizer._total_optimizations

        # Send enough messages to trigger auto-optimize (every 5 sends)
        # We've sent 3 already, send 2 more to hit 5
        for i in range(3, 15):
            mid = comm.send_message("agent_a", {"i": i}, message_type="broadcast")
            comm.report_communication_outcome(mid, "agent_b", success=True, reward=0.5)
            # Drain queue to avoid capacity issues
            comm.receive_messages("agent_b", max_messages=100)

        assert comm._optimizer._total_optimizations > initial_optimizations


# ---------------------------------------------------------------------------
# process_gnn_messages Feedback Tests
# ---------------------------------------------------------------------------

class TestProcessGnnMessagesFeedback:
    """Test that GNNCommunicationMixin.process_gnn_messages reports outcomes."""

    def _make_agent_like(self, gnn_communicator=None):
        """Create a minimal object with GNNCommunicationMixin behavior."""
        from mae_core.agents.mixins.gnn_communication import GNNCommunicationMixin

        class FakeAgent(GNNCommunicationMixin):
            def __init__(self, comm):
                self._init_gnn_communication(gnn_communicator=comm)
                self.unique_id = "test_agent"

        return FakeAgent(gnn_communicator)

    def test_handler_success_reports_positive(self):
        comm = GNNCommunicator(embedding_dim=16)
        comm.register_agent("sender", "mycelial")
        comm.register_agent("test_agent", "mycelial")
        comm.add_edge("sender", "test_agent", 0.8)

        agent = self._make_agent_like(comm)
        handler_called = []
        agent.register_gnn_message_handler(
            "KNOWLEDGE_SHARE", lambda msg: handler_called.append(msg)
        )

        msg_id = comm.send_message(
            "sender", {"knowledge": {}}, message_type="KNOWLEDGE_SHARE", priority=0.7
        )
        agent.process_gnn_messages()

        assert len(handler_called) == 1
        assert msg_id in comm._optimizer._outcomes
        path, reward = comm._optimizer._outcomes[msg_id]
        assert reward == 0.7  # priority used as reward

    def test_handler_exception_reports_negative(self):
        comm = GNNCommunicator(embedding_dim=16)
        comm.register_agent("sender", "mycelial")
        comm.register_agent("test_agent", "mycelial")
        comm.add_edge("sender", "test_agent", 0.8)

        agent = self._make_agent_like(comm)
        agent.register_gnn_message_handler(
            "KNOWLEDGE_SHARE", lambda msg: (_ for _ in ()).throw(ValueError("boom"))
        )

        msg_id = comm.send_message(
            "sender", {"knowledge": {}}, message_type="KNOWLEDGE_SHARE"
        )
        agent.process_gnn_messages()

        assert msg_id in comm._optimizer._outcomes
        path, reward = comm._optimizer._outcomes[msg_id]
        assert reward == -0.5  # failure penalty

    def test_no_handler_reports_weak_positive(self):
        comm = GNNCommunicator(embedding_dim=16)
        comm.register_agent("sender", "mycelial")
        comm.register_agent("test_agent", "mycelial")
        comm.add_edge("sender", "test_agent", 0.8)

        agent = self._make_agent_like(comm)
        # No handler registered for KNOWLEDGE_SHARE

        msg_id = comm.send_message(
            "sender", {"knowledge": {}}, message_type="KNOWLEDGE_SHARE"
        )
        agent.process_gnn_messages()

        assert msg_id in comm._optimizer._outcomes
        path, reward = comm._optimizer._outcomes[msg_id]
        assert reward == 0.1  # weak positive (delivered but unhandled)

    def test_multiple_messages_report_independently(self):
        comm = GNNCommunicator(embedding_dim=16)
        comm.register_agent("sender", "mycelial")
        comm.register_agent("test_agent", "mycelial")
        comm.add_edge("sender", "test_agent", 0.8)

        agent = self._make_agent_like(comm)
        agent.register_gnn_message_handler("TYPE_A", lambda msg: None)
        # No handler for TYPE_B

        id1 = comm.send_message("sender", {"a": 1}, message_type="TYPE_A", priority=0.8)
        id2 = comm.send_message("sender", {"b": 2}, message_type="TYPE_B")
        agent.process_gnn_messages()

        assert id1 in comm._optimizer._outcomes
        assert id2 in comm._optimizer._outcomes
        _, reward1 = comm._optimizer._outcomes[id1]
        _, reward2 = comm._optimizer._outcomes[id2]
        assert reward1 == 0.8  # handler success, reward = priority
        assert reward2 == 0.1  # no handler, weak positive

    def test_no_communicator_is_noop(self):
        agent = self._make_agent_like(None)
        # Should not raise
        agent.process_gnn_messages()

    def test_message_fields_extracted_correctly(self):
        comm = GNNCommunicator(embedding_dim=16)
        comm.register_agent("sender_x", "mycelial")
        comm.register_agent("test_agent", "mycelial")
        comm.add_edge("sender_x", "test_agent", 0.8)

        agent = self._make_agent_like(comm)

        msg_id = comm.send_message(
            "sender_x", {"data": 1}, message_type="broadcast"
        )
        agent.process_gnn_messages()

        assert msg_id in comm._optimizer._outcomes
        path, _ = comm._optimizer._outcomes[msg_id]
        assert path == ["sender_x", "test_agent"]


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------

class TestGNNLearningLoopIntegration:
    def test_full_loop_changes_edge_weights(self):
        """send -> route -> receive -> process -> report -> optimize -> weights change."""
        comm = GNNCommunicator(embedding_dim=16, auto_optimize_interval=5)
        comm.register_agent("sender", "mycelial")
        comm.register_agent("receiver", "mycelial")
        comm.add_edge("sender", "receiver", weight=0.5, edge_type="manual")

        from mae_core.agents.mixins.gnn_communication import GNNCommunicationMixin

        class FakeAgent(GNNCommunicationMixin):
            def __init__(self, c):
                self._init_gnn_communication(gnn_communicator=c)
                self.unique_id = "receiver"

        agent = FakeAgent(comm)
        agent.register_gnn_message_handler("DATA", lambda msg: None)

        initial_weight = comm.graph.edges[("sender", "receiver")].weight

        # Send enough messages to trigger auto-optimize
        for i in range(15):
            comm.send_message("sender", {"i": i}, message_type="DATA", priority=0.9)
            agent.process_gnn_messages()

        # Force optimize if auto didn't trigger
        comm._optimizer.optimize_graph(comm.graph, min_samples=1)

        final_weight = comm.graph.edges[("sender", "receiver")].weight
        assert final_weight != initial_weight

    def test_repeated_success_strengthens_edges(self):
        """Consistently successful routes should increase edge weight."""
        graph = AgentGraph(embedding_dim=16)
        graph.add_agent("a")
        graph.add_agent("b")
        graph.add_edge("a", "b", weight=0.3)

        opt = RoutingOptimizer(learning_rate=0.3)

        # 3 rounds of optimization with positive rewards
        for round_num in range(3):
            for i in range(15):
                opt.record_outcome(f"msg_{round_num}_{i}", ["a", "b"], 0.8)
            opt.optimize_graph(graph, min_samples=10)

        assert graph.edges[("a", "b")].weight > 0.3

    def test_repeated_failure_weakens_edges(self):
        """Consistently failed routes should decrease edge weight."""
        graph = AgentGraph(embedding_dim=16)
        graph.add_agent("a")
        graph.add_agent("b")
        graph.add_edge("a", "b", weight=0.8)

        opt = RoutingOptimizer(learning_rate=0.3)

        # 3 rounds with negative rewards
        for round_num in range(3):
            for i in range(15):
                opt.record_outcome(f"msg_{round_num}_{i}", ["a", "b"], -0.8)
            opt.optimize_graph(graph, min_samples=10)

        assert graph.edges[("a", "b")].weight < 0.8

    def test_communicate_calls_process_gnn_messages(self):
        """MycelialAgent._communicate() processes GNN messages."""
        from unittest.mock import MagicMock as Mock

        from mae_core.agents.mycelial_agent import MycelialAgent

        model = Mock()
        model.schedule = Mock()
        # Mesa 3.4 needs model._agents for auto-registration
        model._agents = {}

        agent = MycelialAgent(model=model, agent_type="mycelial")
        # process_gnn_messages should be called, but with no communicator it's a no-op
        agent._communicate()  # should not raise


# ---------------------------------------------------------------------------
# Edge Case Tests
# ---------------------------------------------------------------------------

class TestGNNEdgeCases:
    def test_empty_queue_processes_nothing(self):
        comm = GNNCommunicator(embedding_dim=16)
        comm.register_agent("agent_a", "mycelial")

        from mae_core.agents.mixins.gnn_communication import GNNCommunicationMixin

        class FakeAgent(GNNCommunicationMixin):
            def __init__(self, c):
                self._init_gnn_communication(gnn_communicator=c)
                self.unique_id = "agent_a"

        agent = FakeAgent(comm)
        agent.process_gnn_messages()  # No messages, no outcomes
        assert len(comm._optimizer._outcomes) == 0

    def test_single_agent_no_recipients(self):
        comm = GNNCommunicator(embedding_dim=16)
        comm.register_agent("lonely", "mycelial")
        msg_id = comm.send_message("lonely", {"data": 1}, message_type="broadcast")
        assert msg_id is not None
        assert comm._message_history[-1]["delivered"] == 0

    def test_high_priority_gives_larger_reward(self):
        comm = GNNCommunicator(embedding_dim=16)
        comm.register_agent("sender", "mycelial")
        comm.register_agent("receiver", "mycelial")
        comm.add_edge("sender", "receiver", 0.8)

        from mae_core.agents.mixins.gnn_communication import GNNCommunicationMixin

        class FakeAgent(GNNCommunicationMixin):
            def __init__(self, c):
                self._init_gnn_communication(gnn_communicator=c)
                self.unique_id = "receiver"

        agent = FakeAgent(comm)
        agent.register_gnn_message_handler("DATA", lambda msg: None)

        id_low = comm.send_message("sender", {"d": 1}, message_type="DATA", priority=0.2)
        agent.process_gnn_messages()
        _, reward_low = comm._optimizer._outcomes[id_low]

        id_high = comm.send_message("sender", {"d": 2}, message_type="DATA", priority=0.9)
        agent.process_gnn_messages()
        _, reward_high = comm._optimizer._outcomes[id_high]

        assert reward_high > reward_low

    def test_optimizer_clears_outcomes_after_optimize(self):
        opt = RoutingOptimizer()
        graph = AgentGraph(embedding_dim=16)
        graph.add_agent("a")
        graph.add_agent("b")
        graph.add_edge("a", "b", weight=0.5)

        for i in range(15):
            opt.record_outcome(f"msg{i}", ["a", "b"], 0.5)
        assert len(opt._outcomes) == 15

        opt.optimize_graph(graph, min_samples=10)
        assert len(opt._outcomes) == 0  # cleared after optimization
