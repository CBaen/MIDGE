"""Knowledge Base - Shared episode and policy storage for learning systems.

Central repository for transfer learning and MAML meta-learning.
Stores successful episodes, policies, value functions, and task metadata.
Supports similarity-based retrieval for knowledge transfer.

Biological analogy: Long-term procedural memory.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TaskDescriptor:
    """Description of a learning task for similarity matching."""

    task_id: str
    name: str = ""
    domain: str = ""
    state_dim: int = 0
    action_dim: int = 0
    embedding: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def similarity(self, other: TaskDescriptor) -> float:
        """Cosine similarity between task embeddings."""
        if self.embedding is None or other.embedding is None:
            # Fallback: domain match + dimension match
            score = 0.0
            if self.domain and self.domain == other.domain:
                score += 0.5
            if self.state_dim == other.state_dim and self.action_dim == other.action_dim:
                score += 0.3
            return score

        n1, n2 = np.linalg.norm(self.embedding), np.linalg.norm(other.embedding)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.dot(self.embedding, other.embedding) / (n1 * n2))


@dataclass
class StoredEpisode:
    """A complete episode stored for transfer or replay."""

    episode_id: str
    task_id: str
    agent_id: str
    transitions: list[dict[str, Any]]
    total_reward: float
    success: bool
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class KnowledgeBase:
    """Shared repository for episodes, policies, and task metadata.

    Used by TransferLearningEngine and MAMLLearner for cross-task
    knowledge retrieval.
    """

    def __init__(
        self,
        max_episodes_per_task: int = 100,
        max_policies_per_task: int = 10,
        event_bus: Any = None,
    ) -> None:
        self._max_episodes = max_episodes_per_task
        self._max_policies = max_policies_per_task

        self._tasks: dict[str, TaskDescriptor] = {}
        self._episodes: dict[str, list[StoredEpisode]] = defaultdict(list)
        self._policies: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._value_functions: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._performance: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.RLock()

        self._episode_counter = 0

        # EventBus: subscribe to policy updates from FRL and successful policies
        self._event_bus = event_bus
        if event_bus is not None:
            event_bus.register_callback(
                "frl.policy_update", self._on_policy_update
            )
            event_bus.register_callback(
                "learning.policy_success", self._on_policy_success
            )

    def register_task(self, descriptor: TaskDescriptor) -> None:
        """Register a task for future retrieval."""
        with self._lock:
            self._tasks[descriptor.task_id] = descriptor

    def store_episode(
        self,
        task_id: str,
        agent_id: str,
        transitions: list[dict[str, Any]],
        total_reward: float,
        success: bool,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store a completed episode. Returns episode_id."""
        with self._lock:
            self._episode_counter += 1
            episode_id = f"ep-{self._episode_counter}"

            episode = StoredEpisode(
                episode_id=episode_id,
                task_id=task_id,
                agent_id=agent_id,
                transitions=transitions,
                total_reward=total_reward,
                success=success,
                metadata=metadata or {},
            )

            episodes = self._episodes[task_id]
            episodes.append(episode)
            # Keep only top episodes by reward
            if len(episodes) > self._max_episodes:
                episodes.sort(key=lambda e: e.total_reward, reverse=True)
                self._episodes[task_id] = episodes[: self._max_episodes]

            self._performance[task_id].append(total_reward)
            return episode_id

    def store_policy(
        self,
        task_id: str,
        agent_id: str,
        policy_state: dict[str, Any],
        performance: float = 0.0,
    ) -> None:
        """Store a policy checkpoint."""
        with self._lock:
            entry = {
                "agent_id": agent_id,
                "policy_state": policy_state,
                "performance": performance,
                "timestamp": time.time(),
            }
            policies = self._policies[task_id]
            policies.append(entry)
            if len(policies) > self._max_policies:
                policies.sort(key=lambda p: p["performance"], reverse=True)
                self._policies[task_id] = policies[: self._max_policies]

    def store_value_function(
        self,
        task_id: str,
        agent_id: str,
        value_state: dict[str, Any],
    ) -> None:
        with self._lock:
            self._value_functions[task_id].append({
                "agent_id": agent_id,
                "value_state": value_state,
                "timestamp": time.time(),
            })

    def retrieve_similar_tasks(
        self,
        target: TaskDescriptor,
        min_similarity: float = 0.3,
        k: int = 5,
    ) -> list[tuple[TaskDescriptor, float]]:
        """Find tasks similar to target. Returns [(task, similarity)]."""
        with self._lock:
            scored = []
            for tid, task in self._tasks.items():
                if tid == target.task_id:
                    continue
                sim = target.similarity(task)
                if sim >= min_similarity:
                    scored.append((task, sim))
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:k]

    def retrieve_successful_episodes(
        self,
        task_id: str,
        min_reward: float | None = None,
        limit: int = 10,
    ) -> list[StoredEpisode]:
        """Get successful episodes for a task."""
        with self._lock:
            episodes = self._episodes.get(task_id, [])
            if min_reward is not None:
                episodes = [e for e in episodes if e.total_reward >= min_reward]
            episodes = sorted(episodes, key=lambda e: e.total_reward, reverse=True)
            return episodes[:limit]

    def retrieve_best_policy(self, task_id: str) -> dict[str, Any] | None:
        """Get the highest-performing policy for a task."""
        with self._lock:
            policies = self._policies.get(task_id, [])
            if not policies:
                return None
            return max(policies, key=lambda p: p["performance"])

    def retrieve_value_function(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            vfs = self._value_functions.get(task_id, [])
            return vfs[-1] if vfs else None

    def get_task_performance(self, task_id: str) -> float:
        """Average performance for a task."""
        with self._lock:
            perf = self._performance.get(task_id, [])
            return float(np.mean(perf)) if perf else 0.0

    def get_statistics(self) -> dict[str, Any]:
        with self._lock:
            return {
                "tasks": len(self._tasks),
                "total_episodes": sum(len(v) for v in self._episodes.values()),
                "total_policies": sum(len(v) for v in self._policies.values()),
                "total_value_functions": sum(len(v) for v in self._value_functions.values()),
            }

    def store_validated_capability(
        self,
        task_id: str,
        agent_id: str,
        policy_state: dict[str, Any],
        performance: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store a validated, successful policy as a known capability.

        Unlike store_policy, this only accepts policies that have
        demonstrated success (performance > 0). Used by the
        learning.policy_success channel.
        """
        if performance <= 0:
            return
        self.store_policy(task_id, agent_id, policy_state, performance)
        logger.debug(
            "Stored validated capability (task=%s, agent=%s, perf=%.2f)",
            task_id, agent_id, performance,
        )

    def _on_policy_success(self, channel: str, data: Any) -> None:
        """Handle successful policy events from learning systems."""
        import json
        if isinstance(data, str):
            data = json.loads(data)
        agent_id = data.get("agent_id", "unknown")
        policy_state = data.get("policy_state", {})
        performance = data.get("performance", 0.0)
        task_id = data.get("task_id", "default")
        metadata = data.get("metadata", {})
        self.store_validated_capability(
            task_id, agent_id, policy_state, performance, metadata
        )

    def _on_policy_update(self, channel: str, data: Any) -> None:
        """Store successful policies automatically when FRL shares them."""
        import json
        if isinstance(data, str):
            data = json.loads(data)
        agent_id = data.get("agent_id", "unknown")
        policy_state = data.get("policy_state", {})
        performance = data.get("performance", 0.0)
        task_id = data.get("metadata", {}).get("task_id", "default")
        self.store_policy(task_id, agent_id, policy_state, performance)
        logger.debug(
            "Stored policy from FRL update (agent=%s, perf=%.2f)",
            agent_id, performance,
        )

    # -- persistence --

    def serialize(self, persist_dir: "Path") -> dict:
        """Save knowledge base state to disk. Returns JSON-safe metadata."""
        import pickle
        from pathlib import Path

        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "tasks": dict(self._tasks),
            "episodes": dict(self._episodes),
            "policies": dict(self._policies),
            "value_functions": dict(self._value_functions),
            "performance": dict(self._performance),
            "episode_counter": self._episode_counter,
        }
        with open(persist_dir / "knowledge_base.pkl", "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

        return {
            "tasks": len(self._tasks),
            "total_episodes": sum(len(v) for v in self._episodes.values()),
            "total_policies": sum(len(v) for v in self._policies.values()),
            "episode_counter": self._episode_counter,
        }

    def restore(self, persist_dir: "Path", metadata: dict) -> None:
        """Restore knowledge base state from disk."""
        import pickle
        from pathlib import Path

        persist_dir = Path(persist_dir)
        path = persist_dir / "knowledge_base.pkl"
        if not path.exists():
            logger.warning("No knowledge base file at %s, starting fresh", path)
            return

        try:
            with open(path, "rb") as f:
                state = pickle.load(f)

            with self._lock:
                self._tasks = state["tasks"]
                self._episodes = defaultdict(list, state["episodes"])
                self._policies = defaultdict(list, state["policies"])
                self._value_functions = defaultdict(list, state["value_functions"])
                self._performance = defaultdict(list, state["performance"])
                self._episode_counter = state["episode_counter"]

            logger.info(
                "Restored knowledge base (tasks=%d, episodes=%d)",
                len(self._tasks),
                sum(len(v) for v in self._episodes.values()),
            )
        except Exception:
            logger.warning("Failed to restore knowledge base", exc_info=True)

    def __repr__(self) -> str:
        return f"KnowledgeBase(tasks={len(self._tasks)}, episodes={sum(len(v) for v in self._episodes.values())})"
