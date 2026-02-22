"""Federated Reinforcement Learning - Decentralized policy sharing.

Agents share policy updates via EventBus, aggregate using weighted
averaging, and maintain peer trust scores. Replaces Redis P2P.

Biological analogy: Mycelial nutrient sharing network.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

from ..backbone.event_bus import EventBus

logger = logging.getLogger(__name__)

CH_POLICY_UPDATE = "frl.policy_update"
CH_PEER_DISCOVERY = "frl.peer_discovery"


class PolicyUpdateStrategy(Enum):
    BROADCAST = "broadcast"
    SELECTIVE = "selective"
    PERFORMANCE_BASED = "performance_based"
    NEIGHBORHOOD = "neighborhood"
    RANDOM = "random"


class AggregationMethod(Enum):
    WEIGHTED_AVERAGE = "weighted_average"
    MEDIAN = "median"
    FEDERATED_AVERAGING = "federated_averaging"
    TRUST_WEIGHTED = "trust_weighted"


@dataclass
class PolicyUpdate:
    """A policy update shared between agents."""

    agent_id: str
    policy_state: dict[str, Any]
    performance: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class FederatedLearningEngine:
    """Federated RL with EventBus-based policy sharing.

    Replaces Redis P2P communication with in-process EventBus.
    """

    def __init__(
        self,
        agent_id: str,
        event_bus: EventBus,
        update_strategy: PolicyUpdateStrategy = PolicyUpdateStrategy.SELECTIVE,
        aggregation_method: AggregationMethod = AggregationMethod.WEIGHTED_AVERAGE,
        max_peers: int = 10,
        trust_threshold: float = 0.5,
        share_frequency: int = 10,
        memory_coordinator: Any = None,
    ) -> None:
        self._agent_id = agent_id
        self._bus = event_bus
        self._strategy = update_strategy
        self._aggregation = aggregation_method
        self._max_peers = max_peers
        self._trust_threshold = trust_threshold
        self._share_frequency = share_frequency
        self._memory_coordinator = memory_coordinator

        self._peers: set[str] = set()
        self._peer_trust: dict[str, float] = defaultdict(lambda: 0.5)
        self._peer_performance: dict[str, float] = {}
        self._update_buffer: list[PolicyUpdate] = []
        self._lock = threading.RLock()

        # Statistics
        self._shared_count = 0
        self._received_count = 0
        self._aggregation_count = 0

        # Subscribe to updates
        self._bus.register_callback(CH_POLICY_UPDATE, self._on_policy_update)
        self._bus.register_callback(CH_PEER_DISCOVERY, self._on_peer_discovery)

    def share_policy_update(
        self, policy_state: dict[str, Any], performance: float = 0.0, metadata: dict[str, Any] | None = None
    ) -> int:
        """Share policy with peers via EventBus. Returns peers notified."""
        update = PolicyUpdate(
            agent_id=self._agent_id,
            policy_state=policy_state,
            performance=performance,
            metadata=metadata or {},
        )

        delivered = self._bus.publish(CH_POLICY_UPDATE, {
            "agent_id": self._agent_id,
            "policy_state": policy_state,
            "performance": performance,
            "metadata": metadata or {},
            "timestamp": update.timestamp,
        })

        self._shared_count += 1
        return delivered

    def receive_policy_updates(self, max_updates: int = 10) -> list[PolicyUpdate]:
        """Get buffered policy updates from peers."""
        with self._lock:
            updates = self._update_buffer[:max_updates]
            self._update_buffer = self._update_buffer[max_updates:]
            return updates

    def aggregate_policy_updates(
        self,
        local_policy: dict[str, Any],
        peer_updates: list[PolicyUpdate] | None = None,
    ) -> dict[str, Any]:
        """Combine local policy with peer policies."""
        if peer_updates is None:
            peer_updates = self.receive_policy_updates()

        if not peer_updates:
            return local_policy

        # Filter by trust
        trusted = [u for u in peer_updates if self._peer_trust[u.agent_id] >= self._trust_threshold]
        if not trusted:
            return local_policy

        self._aggregation_count += 1

        if self._aggregation == AggregationMethod.TRUST_WEIGHTED:
            return self._trust_weighted_aggregate(local_policy, trusted)

        # Default: weighted average by performance
        return self._performance_weighted_aggregate(local_policy, trusted)

    def update_peer_trust(
        self, peer_id: str, update_quality: float, performance_delta: float
    ) -> None:
        """Update trust score for a peer based on their contribution."""
        current = self._peer_trust[peer_id]
        # EMA with quality signal
        quality_signal = 0.5 * update_quality + 0.5 * max(0, performance_delta)
        self._peer_trust[peer_id] = 0.9 * current + 0.1 * quality_signal

    def discover_peers(self) -> set[str]:
        """Broadcast discovery and return known peers."""
        self._bus.publish(CH_PEER_DISCOVERY, {
            "agent_id": self._agent_id,
            "timestamp": time.time(),
        })
        return set(self._peers)

    def select_peers_for_sharing(self, num_peers: int = 5) -> list[str]:
        """Select peers based on strategy."""
        if not self._peers:
            return []

        if self._strategy == PolicyUpdateStrategy.BROADCAST:
            return list(self._peers)

        if self._strategy == PolicyUpdateStrategy.PERFORMANCE_BASED:
            sorted_peers = sorted(
                self._peers,
                key=lambda p: self._peer_performance.get(p, 0),
                reverse=True,
            )
            return sorted_peers[:num_peers]

        if self._strategy == PolicyUpdateStrategy.RANDOM:
            peers = list(self._peers)
            np.random.shuffle(peers)
            return peers[:num_peers]

        # SELECTIVE: by trust
        sorted_peers = sorted(
            self._peers, key=lambda p: self._peer_trust[p], reverse=True
        )
        return sorted_peers[:num_peers]

    def pull_replay_batch(self, batch_size: int = 32) -> tuple[list, list[int], Any] | None:
        """Pull a batch of experiences from memory coordinator for training.

        Returns (experiences, indices, weights) or None if unavailable.
        """
        if self._memory_coordinator is None:
            return None
        try:
            return self._memory_coordinator.sample(batch_size)
        except (ValueError, Exception):
            logger.debug("Could not pull replay batch from memory coordinator")
            return None

    def train_from_memory(
        self,
        memory_coordinator: Any = None,
        batch_size: int = 32,
    ) -> tuple[list, list[int], Any] | None:
        """Pull replay batches from a memory coordinator for training.

        Uses the provided memory_coordinator, falling back to the
        stored one. Returns (experiences, indices, weights) or None.
        """
        mc = memory_coordinator or self._memory_coordinator
        if mc is None:
            return None
        try:
            return mc.sample(batch_size)
        except (ValueError, Exception):
            logger.debug("Could not sample from memory coordinator for training")
            return None

    def get_sharing_statistics(self) -> dict[str, Any]:
        return {
            "agent_id": self._agent_id,
            "peers": len(self._peers),
            "shared_count": self._shared_count,
            "received_count": self._received_count,
            "aggregation_count": self._aggregation_count,
            "update_buffer_size": len(self._update_buffer),
            "strategy": self._strategy.value,
            "aggregation": self._aggregation.value,
        }

    def _on_policy_update(self, channel: str, data: Any) -> None:
        """Handle incoming policy update from EventBus."""
        import json
        if isinstance(data, str):
            data = json.loads(data)
        if data.get("agent_id") == self._agent_id:
            return  # Skip own updates

        with self._lock:
            self._update_buffer.append(PolicyUpdate(
                agent_id=data["agent_id"],
                policy_state=data.get("policy_state", {}),
                performance=data.get("performance", 0.0),
                metadata=data.get("metadata", {}),
            ))
            self._received_count += 1
            self._peers.add(data["agent_id"])
            self._peer_performance[data["agent_id"]] = data.get("performance", 0.0)

    def _on_peer_discovery(self, channel: str, data: Any) -> None:
        import json
        if isinstance(data, str):
            data = json.loads(data)
        peer_id = data.get("agent_id", "")
        if peer_id and peer_id != self._agent_id:
            self._peers.add(peer_id)

    def _performance_weighted_aggregate(
        self, local: dict[str, Any], updates: list[PolicyUpdate]
    ) -> dict[str, Any]:
        """Weighted average using performance scores."""
        # Simple: return best performer's policy
        all_options = [(0.0, local)] + [(u.performance, u.policy_state) for u in updates]
        best_perf, best_policy = max(all_options, key=lambda x: x[0])
        return best_policy

    def _trust_weighted_aggregate(
        self, local: dict[str, Any], updates: list[PolicyUpdate]
    ) -> dict[str, Any]:
        """Weighted average using trust scores."""
        best_trust = 1.0  # Trust self fully
        best_policy = local
        for u in updates:
            trust = self._peer_trust[u.agent_id]
            if trust > best_trust:
                best_trust = trust
                best_policy = u.policy_state
        return best_policy

    # -- persistence --

    def serialize(self, persist_dir: "Path") -> dict:
        """Save FRL state to disk. Returns JSON-safe metadata (all JSON-safe)."""
        import json
        from pathlib import Path

        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "peer_trust": dict(self._peer_trust),
            "peer_performance": dict(self._peer_performance),
            "peers": list(self._peers),
            "shared_count": self._shared_count,
            "received_count": self._received_count,
            "aggregation_count": self._aggregation_count,
        }
        with open(persist_dir / "frl.json", "w") as f:
            json.dump(state, f, indent=2)

        return {
            "agent_id": self._agent_id,
            "peers": len(self._peers),
            "shared_count": self._shared_count,
        }

    def restore(self, persist_dir: "Path", metadata: dict) -> None:
        """Restore FRL state from disk."""
        import json
        from pathlib import Path

        persist_dir = Path(persist_dir)
        path = persist_dir / "frl.json"
        if not path.exists():
            logger.warning("No FRL file at %s, starting fresh", path)
            return

        try:
            with open(path, "r") as f:
                state = json.load(f)

            with self._lock:
                for k, v in state.get("peer_trust", {}).items():
                    self._peer_trust[k] = v
                self._peer_performance = state.get("peer_performance", {})
                self._peers = set(state.get("peers", []))
                self._shared_count = state.get("shared_count", 0)
                self._received_count = state.get("received_count", 0)
                self._aggregation_count = state.get("aggregation_count", 0)
            logger.info(
                "Restored FRL (peers=%d, shared=%d)",
                len(self._peers), self._shared_count,
            )
        except Exception:
            logger.warning("Failed to restore FRL", exc_info=True)

    def __repr__(self) -> str:
        return f"FRL(agent={self._agent_id}, peers={len(self._peers)}, shared={self._shared_count})"
