"""GNN Communication Mixin - Intelligent message routing.

Uses Graph Neural Networks to learn optimal communication patterns,
achieving 40-60% overhead reduction vs broadcast.

Ported from v5-pivot base_agent.py Big Rock 7 GNN methods.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Optional

logger = logging.getLogger(__name__)


class GNNCommunicationMixin:
    """Adds GNN-based intelligent message routing to agents."""

    def _init_gnn_communication(
        self,
        gnn_communicator: Any = None,
        agent_config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize GNN communication attributes."""
        config = agent_config or {}
        self.gnn_communicator = gnn_communicator
        self.gnn_message_handlers: dict[str, Callable[..., None]] = {}
        self.capabilities: set[str] = set(config.get("capabilities", []))

    def send_gnn_message(
        self,
        content: dict[str, Any],
        message_type: str = "BROADCAST",
        target_ids: Optional[list[str]] = None,
        priority: float = 0.5,
        ttl: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[str]:
        """Send message through GNN-based routing."""
        if not self.gnn_communicator:
            return None
        return self.gnn_communicator.send_message(
            sender_id=getattr(self, "unique_id", "unknown"),
            content=content,
            message_type=message_type,
            target_ids=target_ids,
            priority=priority,
            ttl=ttl,
            metadata=metadata,
        )

    def receive_gnn_messages(
        self,
        message_type: Optional[str] = None,
        max_messages: int = 10,
        min_priority: float = 0.0,
    ) -> list[Any]:
        """Receive GNN-routed messages."""
        if not self.gnn_communicator:
            return []
        return self.gnn_communicator.receive_messages(
            agent_id=getattr(self, "unique_id", "unknown"),
            message_type=message_type,
            max_messages=max_messages,
            min_priority=min_priority,
        )

    def process_gnn_messages(self) -> None:
        """Process all pending GNN messages and report outcomes for learning."""
        my_id = str(getattr(self, "unique_id", "unknown"))
        for message in self.receive_gnn_messages():
            msg_id = getattr(message, "message_id", None)
            mtype = getattr(message, "message_type", None)
            handler = self.gnn_message_handlers.get(mtype) if mtype else None
            if handler:
                try:
                    handler(message)
                    if msg_id:
                        self.report_communication_outcome(
                            msg_id, my_id, success=True,
                            reward=getattr(message, "priority", 0.5),
                        )
                except Exception:
                    logger.exception("Error handling GNN message")
                    if msg_id:
                        self.report_communication_outcome(
                            msg_id, my_id, success=False, reward=0.0,
                        )
            else:
                if msg_id:
                    self.report_communication_outcome(
                        msg_id, my_id, success=True, reward=0.1,
                    )

    def register_gnn_message_handler(
        self, message_type: str, handler: Callable[..., None]
    ) -> None:
        """Register handler for a specific message type."""
        self.gnn_message_handlers[message_type] = handler

    def report_communication_outcome(
        self, message_id: str, recipient_id: str, success: bool, reward: float = 0.0
    ) -> None:
        """Report outcome for GNN learning (updates edge weights)."""
        if self.gnn_communicator:
            self.gnn_communicator.report_communication_outcome(
                message_id=message_id,
                recipient_id=recipient_id,
                success=success,
                reward=reward,
            )

    def get_gnn_neighbors(self, k: Optional[int] = None) -> list[str]:
        """Get neighboring agents in communication graph."""
        if not self.gnn_communicator:
            return []
        return self.gnn_communicator.get_agent_neighbors(
            agent_id=getattr(self, "unique_id", "unknown"), k=k
        )

    def broadcast_capability(self) -> None:
        """Broadcast agent capabilities to the network."""
        if not self.gnn_communicator or not self.capabilities:
            return
        self.send_gnn_message(
            content={
                "capabilities": list(self.capabilities),
                "level": getattr(self, "agent_level", 1),
            },
            message_type="CAPABILITY_BROADCAST",
            priority=0.3,
        )

    def request_collaboration(
        self,
        target_ids: Optional[list[str]] = None,
        task_description: str = "",
        required_capabilities: Optional[list[str]] = None,
    ) -> Optional[str]:
        """Request collaboration from other agents."""
        return self.send_gnn_message(
            content={
                "task": task_description,
                "required_capabilities": required_capabilities or [],
            },
            message_type="COLLABORATION_REQUEST",
            target_ids=target_ids,
            priority=0.7,
        )

    def share_knowledge(
        self, knowledge: dict[str, Any], target_ids: Optional[list[str]] = None
    ) -> Optional[str]:
        """Share learned knowledge with other agents."""
        return self.send_gnn_message(
            content={"knowledge": knowledge},
            message_type="KNOWLEDGE_SHARE",
            target_ids=target_ids,
            priority=0.5,
        )

    def get_gnn_communication_stats(self) -> Optional[dict[str, Any]]:
        """Get GNN communication statistics."""
        if not self.gnn_communicator:
            return None
        return self.gnn_communicator.get_communication_statistics()

    def _serialize_gnn_communication(self) -> dict:
        return {
            "capabilities": list(self.capabilities) if hasattr(self, "capabilities") else [],
        }

    def _restore_gnn_communication(self, data: dict) -> None:
        if "capabilities" in data and hasattr(self, "capabilities"):
            self.capabilities = set(data["capabilities"])
