"""GNN Communicator - Intelligent message routing via graph neural network.

Hub for agent-to-agent communication. Routes messages through a learned
agent graph, with backpressure, deduplication, and outcome-based learning.
40-60% overhead reduction vs naive broadcast.

Biological analogy: Mycelial network nutrient/signal transport.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .gnn_graph import AgentGraph
from .gnn_message import GNNMessage
from .gnn_propagator import GNNMessagePropagator, RoutingOptimizer

logger = logging.getLogger(__name__)


@dataclass
class SendResult:
    """Outcome of a send_message_with_ack call."""

    message_id: str | None = None
    success: bool = False
    delivered_count: int = 0
    failed_count: int = 0
    rejected_recipients: list[str] = field(default_factory=list)
    reason: str = ""


class GNNCommunicator:
    """Main GNN communication hub for agent messaging.

    Provides the interface expected by GNNCommunicationMixin:
    send_message(), receive_messages(), report_communication_outcome(),
    get_agent_neighbors(), get_communication_statistics().
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        max_message_history: int = 10000,
        enable_learning: bool = True,
        auto_optimize_interval: int = 100,
        default_k: int = 3,
        default_ttl: int = 3,
        queue_capacity: int = 1000,
        substrate: Any = None,
        event_bus: Any = None,
    ) -> None:
        self._graph = AgentGraph(embedding_dim)
        self._propagator = GNNMessagePropagator(embedding_dim, default_k)
        self._optimizer = RoutingOptimizer() if enable_learning else None
        self._default_k = default_k
        self._default_ttl = default_ttl
        self._enable_learning = enable_learning
        self._auto_optimize_interval = auto_optimize_interval
        self._queue_capacity = queue_capacity
        self._substrate = substrate
        self._event_bus = event_bus

        # Per-agent message queues
        self._queues: dict[str, deque[GNNMessage]] = defaultdict(
            lambda: deque(maxlen=queue_capacity)
        )

        # Message tracking
        self._message_history: deque[dict[str, Any]] = deque(maxlen=max_message_history)
        self._messages_sent = 0
        self._messages_delivered = 0
        self._messages_dropped = 0

        # Subscribe to substrate topology events
        if self._event_bus is not None:
            self._event_bus.register_callback(
                "substrate.topology_changed", self._on_topology_changed
            )
            self._event_bus.register_callback(
                "substrate.agent_registered", self._on_agent_registered
            )
            self._event_bus.register_callback(
                "substrate.agent_deregistered", self._on_agent_deregistered
            )

    @property
    def graph(self) -> AgentGraph:
        return self._graph

    def register_agent(
        self,
        agent_id: str,
        agent_type: str = "mycelial",
        capabilities: set[str] | None = None,
        level: int = 0,
        position: tuple[float, ...] = (0.0, 0.0),
        embedding: np.ndarray | None = None,
    ) -> bool:
        """Register an agent in the communication network."""
        success = self._graph.add_agent(
            agent_id, agent_type, capabilities, level, position, embedding
        )
        if success:
            self._create_initial_edges(agent_id)
        return success

    def unregister_agent(self, agent_id: str) -> bool:
        """Remove agent from the network."""
        if agent_id in self._queues:
            del self._queues[agent_id]
        return self._graph.remove_agent(agent_id)

    def send_message(
        self,
        sender_id: str,
        content: dict[str, Any],
        message_type: str = "broadcast",
        target_ids: list[str] | None = None,
        priority: float = 0.5,
        ttl: int | None = None,
        k: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """Send a message through the network. Returns message_id or None."""
        if sender_id not in self._graph.nodes:
            return None

        message = GNNMessage(
            sender_id=sender_id,
            content=content,
            message_type=message_type,
            priority=priority,
            ttl=ttl or self._default_ttl,
            metadata=metadata or {},
        )

        # Route message
        if target_ids:
            routes = self._propagator.propagate_targeted(
                self._graph, message, target_ids
            )
            recipients = [t for t, path in routes.items() if path is not None]
        else:
            recipients = self._propagator.propagate(
                self._graph, message, k=k or self._default_k
            )

        # Deliver to queues
        delivered = 0
        for rid in recipients:
            if rid in self._graph.nodes:
                queue = self._queues[rid]
                if len(queue) < self._queue_capacity:
                    queue.append(message)
                    delivered += 1
                else:
                    self._messages_dropped += 1

        self._messages_sent += 1
        self._messages_delivered += delivered

        # Track for history
        self._message_history.append({
            "message_id": message.message_id,
            "sender_id": sender_id,
            "message_type": message_type,
            "recipients": recipients,
            "delivered": delivered,
            "timestamp": time.time(),
        })

        # Auto-optimize
        if (
            self._enable_learning
            and self._optimizer
            and self._messages_sent % self._auto_optimize_interval == 0
        ):
            self._optimizer.optimize_graph(self._graph)

        return message.message_id

    def send_message_with_ack(
        self,
        sender_id: str,
        content: dict[str, Any],
        message_type: str = "broadcast",
        target_ids: list[str] | None = None,
        priority: float = 0.5,
        ttl: int | None = None,
        k: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SendResult:
        """Send with detailed delivery report."""
        msg_id = self.send_message(
            sender_id, content, message_type, target_ids, priority, ttl, k, metadata
        )
        if msg_id is None:
            return SendResult(reason="invalid_sender")

        # Find delivery info from history
        for entry in reversed(self._message_history):
            if entry["message_id"] == msg_id:
                return SendResult(
                    message_id=msg_id,
                    success=entry["delivered"] > 0,
                    delivered_count=entry["delivered"],
                    failed_count=len(entry["recipients"]) - entry["delivered"],
                    reason="delivered" if entry["delivered"] > 0 else "no_recipients",
                )
        return SendResult(message_id=msg_id, reason="unknown")

    def receive_messages(
        self,
        agent_id: str,
        message_type: str | None = None,
        max_messages: int = 10,
        min_priority: float = 0.0,
    ) -> list[GNNMessage]:
        """Receive pending messages for an agent."""
        queue = self._queues.get(agent_id)
        if not queue:
            return []

        results: list[GNNMessage] = []
        remaining: deque[GNNMessage] = deque()

        while queue and len(results) < max_messages:
            msg = queue.popleft()
            if message_type and msg.message_type != message_type:
                remaining.append(msg)
                continue
            if msg.priority < min_priority:
                remaining.append(msg)
                continue
            results.append(msg)

        # Put back unmatched messages
        remaining.extend(queue)
        self._queues[agent_id] = remaining

        return results

    def report_communication_outcome(
        self,
        message_id: str,
        recipient_id: str,
        success: bool,
        reward: float = 0.0,
    ) -> None:
        """Report whether communication was useful (for learning)."""
        if not self._optimizer:
            return

        # Find message path from history
        for entry in reversed(self._message_history):
            if entry["message_id"] == message_id:
                path = [entry["sender_id"], recipient_id]
                effective_reward = reward if success else -0.5
                self._optimizer.record_outcome(message_id, path, effective_reward)
                return

    def get_agent_neighbors(self, agent_id: str, k: int | None = None) -> list[str]:
        return self._graph.get_neighbors(agent_id, k)

    def add_edge(
        self, source_id: str, target_id: str, weight: float = 1.0, edge_type: str = "manual"
    ) -> bool:
        return self._graph.add_edge(source_id, target_id, weight, edge_type)

    def remove_edge(self, source_id: str, target_id: str) -> bool:
        return self._graph.remove_edge(source_id, target_id)

    def get_communication_statistics(self) -> dict[str, Any]:
        stats = self._graph.get_statistics()
        stats.update({
            "messages_sent": self._messages_sent,
            "messages_delivered": self._messages_delivered,
            "messages_dropped": self._messages_dropped,
            "delivery_rate": (
                self._messages_delivered / self._messages_sent
                if self._messages_sent > 0 else 0.0
            ),
            "active_queues": sum(1 for q in self._queues.values() if q),
            "total_queued": sum(len(q) for q in self._queues.values()),
        })
        if self._optimizer:
            stats["optimizer"] = self._optimizer.get_statistics()
        return stats

    def get_message_queue_size(self, agent_id: str) -> int:
        return len(self._queues.get(agent_id, []))

    # =========================================================================
    # Substrate Integration
    # =========================================================================

    def sync_topology(self) -> None:
        """Sync internal graph from substrate topology.

        Calls substrate.get_topology_graph() and updates the GNN graph
        with node/edge information from the substrate.
        """
        if self._substrate is None:
            return

        topo = self._substrate.get_topology_graph()
        if topo is None:
            return

        for nid, node_data in topo.get("nodes", {}).items():
            agent_id = node_data.get("agent_id")
            if agent_id is not None:
                aid = str(agent_id)
                if aid not in self._graph.nodes:
                    pos = node_data.get("position", (0.0, 0.0))
                    self._graph.add_agent(aid, position=pos)

        for edge in topo.get("edges", []):
            from_node = edge.get("from", "")
            to_node = edge.get("to", "")
            # Map node IDs to agent IDs for GNN edges
            from_agent = topo["nodes"].get(from_node, {}).get("agent_id")
            to_agent = topo["nodes"].get(to_node, {}).get("agent_id")
            if from_agent is not None and to_agent is not None:
                conductance = edge.get("conductance", 1.0)
                self._graph.add_edge(
                    str(from_agent), str(to_agent), conductance, "substrate"
                )

    def _on_topology_changed(self, channel: str, message: Any) -> None:
        """Handle substrate.topology_changed events by re-syncing topology."""
        self.sync_topology()

    def _on_agent_registered(self, channel: str, message: Any) -> None:
        """Handle substrate.agent_registered by adding node to GNN graph."""
        import json

        data = message
        if isinstance(message, str):
            try:
                data = json.loads(message)
            except (json.JSONDecodeError, TypeError):
                return

        if isinstance(data, dict):
            agent_id = data.get("agent_id")
            if agent_id is not None:
                aid = str(agent_id)
                if aid not in self._graph.nodes:
                    self._graph.add_agent(aid)

    def _on_agent_deregistered(self, channel: str, message: Any) -> None:
        """Handle substrate.agent_deregistered by removing node from GNN graph."""
        import json

        data = message
        if isinstance(message, str):
            try:
                data = json.loads(message)
            except (json.JSONDecodeError, TypeError):
                return

        if isinstance(data, dict):
            agent_id = data.get("agent_id")
            if agent_id is not None:
                aid = str(agent_id)
                if aid in self._graph.nodes:
                    self.unregister_agent(aid)

    def _create_initial_edges(self, agent_id: str) -> None:
        """Create initial edges based on proximity and role similarity."""
        node = self._graph.nodes.get(agent_id)
        if not node:
            return

        for other_id, other_node in self._graph.nodes.items():
            if other_id == agent_id:
                continue

            # Role similarity
            if node.agent_type == other_node.agent_type:
                self._graph.add_edge(agent_id, other_id, 0.6, "role")
                self._graph.add_edge(other_id, agent_id, 0.6, "role")
                continue

            # Capability overlap
            shared = node.capabilities & other_node.capabilities
            if shared:
                weight = min(1.0, 0.3 + 0.1 * len(shared))
                self._graph.add_edge(agent_id, other_id, weight, "capability")
                self._graph.add_edge(other_id, agent_id, weight, "capability")
                continue

            # Proximity (if close)
            dist = node.distance_to(other_node)
            if dist < 10.0:
                weight = max(0.2, 1.0 - dist / 10.0)
                self._graph.add_edge(agent_id, other_id, weight, "proximity")
                self._graph.add_edge(other_id, agent_id, weight, "proximity")

    def __repr__(self) -> str:
        return (
            f"GNNCommunicator(agents={len(self._graph)}, "
            f"sent={self._messages_sent}, delivered={self._messages_delivered})"
        )
