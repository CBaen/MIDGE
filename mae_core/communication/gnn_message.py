"""GNN Message - Message types for graph neural network communication.

Messages are routed through the agent graph with TTL-based
hop limiting and path tracking for cycle prevention.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class MessageType(Enum):
    """Types of messages routed through the GNN."""

    BROADCAST = "broadcast"
    COLLABORATION_REQUEST = "collaboration_request"
    COLLABORATION_RESPONSE = "collaboration_response"
    KNOWLEDGE_SHARE = "knowledge_share"
    CAPABILITY_BROADCAST = "capability_broadcast"
    STATE_UPDATE = "state_update"
    QUERY = "query"
    QUERY_RESPONSE = "query_response"
    TASK_ASSIGNMENT = "task_assignment"
    TASK_COMPLETION = "task_completion"
    VOTE = "vote"
    CONSENSUS = "consensus"


@dataclass
class GNNMessage:
    """A message routed through the agent graph."""

    sender_id: str
    content: dict[str, Any]
    message_type: str = "broadcast"
    priority: float = 0.5  # [0, 1]
    ttl: int = 3  # hops remaining
    message_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    path: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def decrement_ttl(self) -> bool:
        """Decrement TTL. Returns True if message is still alive."""
        self.ttl -= 1
        return self.ttl > 0

    def add_to_path(self, agent_id: str) -> None:
        self.path.append(agent_id)

    def has_visited(self, agent_id: str) -> bool:
        return agent_id in self.path

    @property
    def age(self) -> float:
        return time.time() - self.timestamp

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "content": self.content,
            "message_type": self.message_type,
            "priority": self.priority,
            "ttl": self.ttl,
            "path": self.path,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GNNMessage:
        return cls(
            sender_id=data["sender_id"],
            content=data["content"],
            message_type=data.get("message_type", "broadcast"),
            priority=data.get("priority", 0.5),
            ttl=data.get("ttl", 3),
            message_id=data.get("message_id", uuid.uuid4().hex[:12]),
            path=data.get("path", []),
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
        )


class MessageEncoder:
    """Encodes messages into embedding space for routing decisions."""

    def __init__(self, embedding_dim: int = 64) -> None:
        self._dim = embedding_dim
        self._type_embeddings = self._initialize_type_embeddings()

    def encode(self, message: GNNMessage, sender_embedding: np.ndarray | None = None) -> np.ndarray:
        """Encode a message into embedding space."""
        # Type embedding (first quarter)
        type_emb = self._type_embeddings.get(
            message.message_type, np.zeros(self._dim // 4)
        )

        # Content embedding (second quarter) - hash-based
        content_str = str(sorted(message.content.items()))
        content_hash = hashlib.sha256(content_str.encode()).digest()
        content_emb = np.frombuffer(content_hash[: self._dim // 4], dtype=np.uint8).astype(np.float32)
        content_emb = (content_emb / 255.0) * 2 - 1  # normalize to [-1, 1]

        # Metadata embedding (third quarter) - priority and TTL
        meta_emb = np.zeros(self._dim // 4)
        meta_emb[0] = message.priority
        meta_emb[1] = message.ttl / 10.0
        meta_emb[2] = min(1.0, message.age / 300.0)

        # Sender context (last quarter)
        if sender_embedding is not None:
            sender_part = sender_embedding[: self._dim // 4]
        else:
            sender_part = np.zeros(self._dim // 4)

        embedding = np.concatenate([type_emb, content_emb, meta_emb, sender_part])

        # Pad or truncate to exact dim
        if len(embedding) < self._dim:
            embedding = np.pad(embedding, (0, self._dim - len(embedding)))
        elif len(embedding) > self._dim:
            embedding = embedding[: self._dim]

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Cosine similarity between two embeddings."""
        n1, n2 = np.linalg.norm(emb1), np.linalg.norm(emb2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.dot(emb1, emb2) / (n1 * n2))

    def _initialize_type_embeddings(self) -> dict[str, np.ndarray]:
        """Create deterministic embeddings for each message type."""
        rng = np.random.RandomState(42)
        dim = self._dim // 4
        return {mt.value: rng.randn(dim).astype(np.float32) for mt in MessageType}
