"""GNN Message Propagator - Intelligent message routing through agent graph.

Routes messages using learned edge weights and relevance scoring.
Multi-hop BFS with top-k selection per hop, cycle prevention via
visited sets. Edge weights learn from communication outcomes.

Biological analogy: Mycelial nutrient routing.
"""

from __future__ import annotations

import heapq
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .gnn_graph import AgentGraph
from .gnn_message import GNNMessage, MessageEncoder

logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    """Record of a routing decision for learning."""

    message_id: str
    sender_id: str
    recipients: list[str]
    scores: dict[str, float] = field(default_factory=dict)
    hop_count: int = 0
    timestamp: float = field(default_factory=time.time)


class GNNMessagePropagator:
    """Routes messages through the agent graph using learned weights."""

    def __init__(
        self,
        embedding_dim: int = 64,
        default_k: int = 3,
        enable_tracking: bool = True,
    ) -> None:
        self._encoder = MessageEncoder(embedding_dim)
        self._default_k = default_k
        self._enable_tracking = enable_tracking
        self._routing_history: list[RoutingDecision] = []
        self._total_propagated = 0

    def propagate(
        self,
        graph: AgentGraph,
        message: GNNMessage,
        max_hops: int | None = None,
        k: int | None = None,
    ) -> list[str]:
        """Route message through graph. Returns list of recipient agent IDs.

        Multi-hop BFS: at each hop, select top-k candidates by relevance
        score. Visited set prevents cycles. TTL limits hop count.
        """
        k = k or self._default_k
        max_hops = max_hops or message.ttl
        if max_hops <= 0:
            max_hops = 3

        # Broadcast: deliver to all connected agents
        if message.message_type == "broadcast":
            all_agents = [
                aid for aid in graph.nodes if aid != message.sender_id
            ]
            self._total_propagated += 1
            if self._enable_tracking:
                self._routing_history.append(RoutingDecision(
                    message_id=message.message_id,
                    sender_id=message.sender_id,
                    recipients=all_agents,
                    hop_count=1,
                ))
            return all_agents

        # Multi-hop routing
        visited: set[str] = {message.sender_id}
        current_hop: list[str] = [message.sender_id]
        all_recipients: list[str] = []
        all_scores: dict[str, float] = {}

        msg_embedding = self._encoder.encode(message)

        for hop in range(max_hops):
            next_hop: list[str] = []

            for agent_id in current_hop:
                candidates = [
                    n for n in graph.get_neighbors(agent_id)
                    if n not in visited
                ]
                if not candidates:
                    continue

                scores = self._compute_relevance(
                    agent_id, candidates, msg_embedding, graph
                )
                selected = self._select_top_k(candidates, scores, k)

                for sel_id in selected:
                    all_recipients.append(sel_id)
                    all_scores[sel_id] = scores.get(sel_id, 0.0)
                    visited.add(sel_id)
                    next_hop.append(sel_id)

            current_hop = next_hop
            if not current_hop:
                break

        self._total_propagated += 1
        if self._enable_tracking:
            self._routing_history.append(RoutingDecision(
                message_id=message.message_id,
                sender_id=message.sender_id,
                recipients=all_recipients,
                scores=all_scores,
                hop_count=min(hop + 1, max_hops),
            ))

        return all_recipients

    def propagate_targeted(
        self,
        graph: AgentGraph,
        message: GNNMessage,
        target_ids: list[str],
        max_hops: int | None = None,
    ) -> dict[str, list[str] | None]:
        """Route message to specific targets. Returns {target: path or None}."""
        max_hops = max_hops or message.ttl or 5
        results: dict[str, list[str] | None] = {}

        for target in target_ids:
            path = self._find_shortest_path(graph, message.sender_id, target, max_hops)
            results[target] = path

        return results

    def _compute_relevance(
        self,
        from_id: str,
        candidate_ids: list[str],
        msg_embedding: np.ndarray,
        graph: AgentGraph,
    ) -> dict[str, float]:
        """Score candidates by relevance to the message."""
        scores: dict[str, float] = {}

        for cid in candidate_ids:
            node = graph.nodes.get(cid)
            if node is None:
                continue

            # Compatibility: cosine similarity between message and agent embeddings
            compatibility = float(np.dot(msg_embedding, node.embedding))
            n1, n2 = np.linalg.norm(msg_embedding), np.linalg.norm(node.embedding)
            if n1 > 0 and n2 > 0:
                compatibility /= (n1 * n2)

            # Edge weight
            edge = graph.edges.get((from_id, cid))
            edge_weight = edge.weight if edge else 0.5

            scores[cid] = compatibility * edge_weight

        return scores

    def _select_top_k(
        self, candidates: list[str], scores: dict[str, float], k: int
    ) -> list[str]:
        """Select top-k candidates by score."""
        scored = [(scores.get(c, 0.0), c) for c in candidates]
        scored.sort(reverse=True)
        return [c for _, c in scored[:k]]

    def _find_shortest_path(
        self,
        graph: AgentGraph,
        source_id: str,
        target_id: str,
        max_hops: int,
    ) -> list[str] | None:
        """Dijkstra shortest path with edge weights (inverted for distance)."""
        if source_id not in graph.nodes or target_id not in graph.nodes:
            return None

        # Priority queue: (cost, node, path)
        pq: list[tuple[float, str, list[str]]] = [(0.0, source_id, [source_id])]
        visited: set[str] = set()

        while pq:
            cost, current, path = heapq.heappop(pq)

            if current == target_id:
                return path

            if current in visited or len(path) > max_hops + 1:
                continue
            visited.add(current)

            for neighbor in graph.get_neighbors(current):
                if neighbor not in visited:
                    edge = graph.edges.get((current, neighbor))
                    edge_cost = 1.0 / max(edge.weight, 0.01) if edge else 10.0
                    heapq.heappush(pq, (cost + edge_cost, neighbor, path + [neighbor]))

        return None

    def get_routing_statistics(self) -> dict[str, Any]:
        return {
            "total_propagated": self._total_propagated,
            "routing_history_size": len(self._routing_history),
        }

    def clear_history(self) -> None:
        self._routing_history.clear()


class RoutingOptimizer:
    """Learns optimal routing by updating edge weights from outcomes."""

    def __init__(self, learning_rate: float = 0.1) -> None:
        self._lr = learning_rate
        self._outcomes: dict[str, tuple[list[str], float]] = {}  # msg_id -> (path, reward)
        self._total_optimizations = 0

    def record_outcome(self, message_id: str, path: list[str], reward: float) -> None:
        self._outcomes[message_id] = (path, reward)

    def optimize_graph(self, graph: AgentGraph, min_samples: int = 10) -> int:
        """Update edge weights from recorded outcomes. Returns edges updated."""
        if len(self._outcomes) < min_samples:
            return 0

        # Group rewards by edge
        edge_rewards: dict[tuple[str, str], list[float]] = defaultdict(list)
        for path, reward in self._outcomes.values():
            for i in range(len(path) - 1):
                edge_rewards[(path[i], path[i + 1])].append(reward)

        updated = 0
        for (src, tgt), rewards in edge_rewards.items():
            edge = graph.edges.get((src, tgt))
            if edge:
                avg_reward = sum(rewards) / len(rewards)
                edge.update_from_outcome(avg_reward, self._lr)
                updated += 1

        self._outcomes.clear()
        self._total_optimizations += 1
        return updated

    def get_statistics(self) -> dict[str, Any]:
        return {
            "pending_outcomes": len(self._outcomes),
            "total_optimizations": self._total_optimizations,
            "learning_rate": self._lr,
        }
