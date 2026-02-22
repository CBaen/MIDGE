"""Agent Graph - Topology for GNN message routing.

Manages agent nodes and communication edges with learned weights.
Edge weights update from communication outcomes (success/failure).

Biological analogy: Mycelial network topology.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class AgentNode:
    """A node in the agent communication graph."""

    agent_id: str
    embedding: np.ndarray  # agent's position in embedding space
    agent_type: str = "mycelial"
    capabilities: set[str] = field(default_factory=set)
    level: int = 0
    position: tuple[float, ...] = (0.0, 0.0)
    last_update: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def update_embedding(self, new_embedding: np.ndarray) -> None:
        self.embedding = new_embedding
        self.last_update = time.time()

    def similarity(self, other: AgentNode) -> float:
        """Cosine similarity to another node."""
        n1, n2 = np.linalg.norm(self.embedding), np.linalg.norm(other.embedding)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.dot(self.embedding, other.embedding) / (n1 * n2))

    def distance_to(self, other: AgentNode) -> float:
        """Euclidean distance in position space."""
        a, b = self.position, other.position
        dims = min(len(a), len(b))
        return float(np.sqrt(sum((a[i] - b[i]) ** 2 for i in range(dims))))


@dataclass
class CommunicationEdge:
    """A directed edge in the communication graph."""

    source_id: str
    target_id: str
    weight: float = 1.0  # [0, 1] learned importance
    message_count: int = 0
    last_message_time: float = 0.0
    success_rate: float = 1.0  # [0, 1]
    edge_type: str = "learned"  # "proximity", "role", "capability", "learned"
    metadata: dict[str, Any] = field(default_factory=dict)

    def update_from_outcome(self, reward: float, learning_rate: float = 0.1) -> None:
        """Update edge weight from communication outcome via EMA."""
        normalized = (reward + 1.0) / 2.0  # [-1,1] -> [0,1]
        self.weight = (1.0 - learning_rate) * self.weight + learning_rate * normalized
        self.weight = max(0.0, min(1.0, self.weight))

    def record_message(self) -> None:
        self.message_count += 1
        self.last_message_time = time.time()

    def is_active(self, timeout: float = 300.0) -> bool:
        return self.last_message_time > 0 and (time.time() - self.last_message_time) < timeout

    def is_strong(self, threshold: float = 0.7) -> bool:
        return self.weight >= threshold


class AgentGraph:
    """Directed graph of agent communication topology."""

    def __init__(self, embedding_dim: int = 64) -> None:
        self._dim = embedding_dim
        self._nodes: dict[str, AgentNode] = {}
        self._edges: dict[tuple[str, str], CommunicationEdge] = {}
        self._adj_out: dict[str, set[str]] = {}
        self._adj_in: dict[str, set[str]] = {}
        self._lock = threading.RLock()

    @property
    def nodes(self) -> dict[str, AgentNode]:
        return self._nodes

    @property
    def edges(self) -> dict[tuple[str, str], CommunicationEdge]:
        return self._edges

    def add_agent(
        self,
        agent_id: str,
        agent_type: str = "mycelial",
        capabilities: set[str] | None = None,
        level: int = 0,
        position: tuple[float, ...] = (0.0, 0.0),
        embedding: np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Add an agent node to the graph."""
        with self._lock:
            if agent_id in self._nodes:
                return False

            if embedding is None:
                embedding = self._initialize_embedding(agent_type, capabilities or set(), level)

            self._nodes[agent_id] = AgentNode(
                agent_id=agent_id,
                embedding=embedding,
                agent_type=agent_type,
                capabilities=capabilities or set(),
                level=level,
                position=position,
                metadata=metadata or {},
            )
            self._adj_out[agent_id] = set()
            self._adj_in[agent_id] = set()
            return True

    def remove_agent(self, agent_id: str) -> bool:
        with self._lock:
            if agent_id not in self._nodes:
                return False

            # Remove all edges involving this agent
            to_remove = [
                k for k in self._edges if k[0] == agent_id or k[1] == agent_id
            ]
            for key in to_remove:
                self._remove_edge_unsafe(key[0], key[1])

            del self._nodes[agent_id]
            del self._adj_out[agent_id]
            del self._adj_in[agent_id]
            return True

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        weight: float = 1.0,
        edge_type: str = "learned",
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        with self._lock:
            if source_id not in self._nodes or target_id not in self._nodes:
                return False
            key = (source_id, target_id)
            if key in self._edges:
                self._edges[key].weight = weight
                return True
            self._edges[key] = CommunicationEdge(
                source_id=source_id,
                target_id=target_id,
                weight=weight,
                edge_type=edge_type,
                metadata=metadata or {},
            )
            self._adj_out[source_id].add(target_id)
            self._adj_in[target_id].add(source_id)
            return True

    def remove_edge(self, source_id: str, target_id: str) -> bool:
        with self._lock:
            return self._remove_edge_unsafe(source_id, target_id)

    def _remove_edge_unsafe(self, source_id: str, target_id: str) -> bool:
        key = (source_id, target_id)
        if key not in self._edges:
            return False
        del self._edges[key]
        self._adj_out.get(source_id, set()).discard(target_id)
        self._adj_in.get(target_id, set()).discard(source_id)
        return True

    def get_neighbors(
        self, agent_id: str, k: int | None = None, direction: str = "out"
    ) -> list[str]:
        """Get neighbors of an agent. direction: 'out', 'in', 'both'."""
        with self._lock:
            if direction == "out":
                neighbors = list(self._adj_out.get(agent_id, set()))
            elif direction == "in":
                neighbors = list(self._adj_in.get(agent_id, set()))
            else:
                neighbors = list(
                    self._adj_out.get(agent_id, set()) | self._adj_in.get(agent_id, set())
                )

            if k is not None:
                # Sort by edge weight (strongest first)
                neighbors.sort(
                    key=lambda n: self._edges.get((agent_id, n), CommunicationEdge("", "")).weight,
                    reverse=True,
                )
                neighbors = neighbors[:k]
            return neighbors

    def prune_weak_edges(self, weight_threshold: float = 0.1) -> int:
        """Remove edges below weight threshold. Returns count removed."""
        with self._lock:
            to_remove = [k for k, e in self._edges.items() if e.weight < weight_threshold]
            for key in to_remove:
                self._remove_edge_unsafe(key[0], key[1])
            return len(to_remove)

    def prune_inactive_edges(self, timeout: float = 300.0) -> int:
        with self._lock:
            to_remove = [k for k, e in self._edges.items() if not e.is_active(timeout)]
            for key in to_remove:
                self._remove_edge_unsafe(key[0], key[1])
            return len(to_remove)

    def get_statistics(self) -> dict[str, Any]:
        with self._lock:
            n_nodes = len(self._nodes)
            n_edges = len(self._edges)
            return {
                "nodes": n_nodes,
                "edges": n_edges,
                "density": n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0,
                "avg_degree": n_edges / n_nodes if n_nodes > 0 else 0.0,
                "avg_edge_weight": (
                    float(np.mean([e.weight for e in self._edges.values()]))
                    if self._edges else 0.0
                ),
            }

    def _initialize_embedding(
        self, agent_type: str, capabilities: set[str], level: int
    ) -> np.ndarray:
        """Deterministic initial embedding from agent properties."""
        rng = np.random.RandomState(hash(agent_type) % (2**31))
        emb = rng.randn(self._dim).astype(np.float32)
        # Encode level
        emb[0] = level / 10.0
        # Encode capabilities
        for i, cap in enumerate(sorted(capabilities)):
            if i + 1 < self._dim:
                emb[i + 1] += hash(cap) % 10 / 10.0
        # Normalize
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb /= norm
        return emb

    def __len__(self) -> int:
        return len(self._nodes)

    def __repr__(self) -> str:
        return f"AgentGraph(nodes={len(self._nodes)}, edges={len(self._edges)})"
