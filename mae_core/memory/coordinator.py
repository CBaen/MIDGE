"""MemoryCoordinator - Orchestrates all memory subsystems.

Wires EpisodicMemory, WorkingMemory, SemanticRetriever,
GenerativeReplayMemory, and MemoryConsolidator into a
unified system. Publishes lifecycle events on EventBus.

Design: Direct method calls for hot-path (store/sample),
EventBus for lifecycle events and cross-system notifications.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import numpy as np

from ..backbone.event_bus import EventBus
from .episodic_memory import EpisodicMemory
from .experience import Experience
from .generative_replay import GenerativeReplayMemory
from .memory_consolidator import ConsolidationResult, ConsolidationStrategy, MemoryConsolidator
from .semantic_retriever import SemanticRetriever
from .working_memory import WorkingMemory

logger = logging.getLogger(__name__)

# EventBus channels
CH_EXPERIENCE_STORED = "memory.experience_stored"
CH_CONSOLIDATION_STARTED = "memory.consolidation_started"
CH_CONSOLIDATION_COMPLETE = "memory.consolidation_complete"
CH_CAPACITY_WARNING = "memory.capacity_warning"
CH_NOVEL_EXPERIENCE = "memory.novel_experience"


class MemoryCoordinator:
    """Orchestrates all memory subsystems for an agent.

    Args:
        event_bus: Shared EventBus for lifecycle events.
        agent_id: Owning agent's ID.
        episodic_capacity: Max experiences in episodic memory.
        state_dim: State vector dimension (for VAE/semantic).
        action_dim: Action space size.
        enable_semantic: Enable FAISS-backed semantic search.
        enable_generative: Enable VAE-based generative replay.
        enable_working: Enable working memory (7+/-2 slots).
        enable_consolidation: Enable offline "sleep" learning.
        novelty_threshold: Semantic distance above this emits novel_experience.
    """

    def __init__(
        self,
        event_bus: EventBus,
        agent_id: str = "default",
        episodic_capacity: int = 100_000,
        state_dim: int = 74,
        action_dim: int = 4,
        enable_semantic: bool = True,
        enable_generative: bool = False,
        enable_working: bool = True,
        enable_consolidation: bool = True,
        novelty_threshold: float = 0.7,
        compression_threshold: int = 200,
    ) -> None:
        self._bus = event_bus
        self._agent_id = agent_id
        self._novelty_threshold = novelty_threshold

        # -- Subsystems --

        # Semantic retriever (used by episodic for similarity search)
        self.semantic: SemanticRetriever | None = None
        if enable_semantic:
            self.semantic = SemanticRetriever(
                embedding_dim=128, agent_id=agent_id
            )

        # Episodic memory (core)
        self.episodic = EpisodicMemory(
            capacity=episodic_capacity,
            use_semantic_index=enable_semantic,
            semantic_retriever=self.semantic,
        )

        # Working memory
        self.working: WorkingMemory | None = None
        if enable_working:
            self.working = WorkingMemory(capacity=7)

        # Generative replay
        self.generative: GenerativeReplayMemory | None = None
        if enable_generative:
            self.generative = GenerativeReplayMemory(
                state_dim=state_dim,
                action_dim=action_dim,
                compression_threshold=compression_threshold,
            )

        # Consolidator
        self.consolidator: MemoryConsolidator | None = None
        if enable_consolidation:
            self.consolidator = MemoryConsolidator(
                episodic_memory=self.episodic,
            )

        # Capacity warning threshold
        self._capacity_warned = False

        # Deep memory bridge (Qdrant-backed, injected post-init)
        self._bridge: Any = None

    # -- Hot-path operations (direct method calls) --

    def store(
        self,
        state: np.ndarray,
        action: Any,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: dict[str, Any] | None = None,
        priority: float | None = None,
    ) -> None:
        """Store experience across all enabled subsystems."""
        info_dict = info or {}

        # Triadic witness hash for deep memory verification
        if self._bridge is not None:
            from .deep_memory import DeepMemoryStore
            info_dict["witness_hash"] = DeepMemoryStore.compute_witness_hash(
                state, next_state, action, reward
            )

        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            info=info_dict,
            timestamp=time.time(),
            priority=priority,
        )

        # 1. Episodic memory (primary storage)
        self.episodic.store(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            info=info_dict,
            priority=priority,
        )

        # 2. Generative replay buffer
        if self.generative is not None:
            self.generative.store(experience)

        # 3. Working memory (high-importance only)
        if self.working is not None and priority is not None and priority > 0.8:
            self.working.store(
                item_id=f"exp-{self.episodic.total_stored}",
                content=experience,
                importance=min(1.0, priority),
            )

        # 4. Novelty detection via semantic distance
        if self.semantic is not None and self.semantic.n_indexed > 10:
            result = self.semantic.search_by_state(state, k=1)
            if result.scores and result.scores[0] < self._novelty_threshold:
                self._bus.publish(CH_NOVEL_EXPERIENCE, {
                    "agent_id": self._agent_id,
                    "novelty_score": 1.0 - result.scores[0],
                    "reward": reward,
                })

        # 5. Lifecycle event
        self._bus.publish(CH_EXPERIENCE_STORED, {
            "agent_id": self._agent_id,
            "total_stored": self.episodic.total_stored,
        })

        # 6. Capacity warning
        utilization = len(self.episodic) / self.episodic.capacity
        if utilization > 0.9 and not self._capacity_warned:
            self._capacity_warned = True
            self._bus.publish(CH_CAPACITY_WARNING, {
                "agent_id": self._agent_id,
                "utilization": utilization,
                "capacity": self.episodic.capacity,
            })

    def sample(
        self, batch_size: int = 32, beta: float | None = None
    ) -> tuple[list[Experience], list[int], np.ndarray]:
        """Sample from episodic memory (prioritized)."""
        return self.episodic.sample(batch_size, beta=beta)

    def update_priorities(
        self, indices: list[int], td_errors: np.ndarray
    ) -> None:
        """Update episodic memory priorities from TD errors."""
        self.episodic.update_priorities(indices, td_errors)

    # -- Search cascade: WM(O(1)) -> Episodic(O(log N)) -> Semantic(embedding) --

    def search(
        self,
        state: np.ndarray,
        k: int = 5,
        item_id: str | None = None,
        query_text: str | None = None,
    ) -> list:
        """Cascading search across memory subsystems.

        1. Check working memory (if item_id given)
        2. Search semantic/episodic for similar experiences
        3. Fall back to deep memory (Qdrant) if insufficient local results
        """
        results: list = []

        # Level 1: Working memory (O(1))
        if item_id is not None and self.working is not None:
            content = self.working.retrieve(item_id)
            if isinstance(content, Experience):
                results.append(content)

        # Level 2: Semantic search (embedding-based, local FAISS)
        similar = self.episodic.get_similar_experiences(state, k=k)
        results.extend(similar)

        # Level 3: Deep memory fallback (Qdrant)
        if len(results) < k and self._bridge is not None and query_text:
            deep_results = self._bridge.recall_from_deep(
                query_text=query_text,
                agent_id=self._agent_id,
                limit=k - len(results),
            )
            results.extend(deep_results)

        return results[:k]

    # -- Consolidation --

    def should_consolidate(self, current_step: int | None = None) -> bool:
        if self.consolidator is None:
            return False
        return self.consolidator.should_consolidate(current_step)

    def consolidate(
        self, agent: Any, num_steps: int | None = None,
        strategy: ConsolidationStrategy | None = None,
    ) -> ConsolidationResult | None:
        """Run a consolidation ("sleep") phase."""
        if self.consolidator is None:
            return None

        self._bus.publish(CH_CONSOLIDATION_STARTED, {
            "agent_id": self._agent_id,
        })

        result = self.consolidator.consolidate(
            agent=agent, num_steps=num_steps, strategy=strategy
        )

        self._bus.publish(CH_CONSOLIDATION_COMPLETE, {
            "agent_id": self._agent_id,
            "loss_reduction": result.loss_reduction,
            "steps": result.steps,
        })

        # Deep memory: write consolidated experiences to Qdrant
        if self._bridge is not None:
            try:
                sample_size = min(100, len(self.episodic))
                if sample_size > 0:
                    recent, _, _ = self.episodic.sample(sample_size)
                    agent_context = self._build_agent_context(agent)
                    self._bridge.consolidate_to_deep(
                        agent_id=self._agent_id,
                        experiences=recent,
                        agent_context=agent_context,
                        consolidation_id=getattr(result, "consolidation_id", 0),
                    )
            except Exception:
                logger.warning("Deep memory consolidation failed", exc_info=True)

        return result

    def trigger_consolidation(
        self, agent: Any, num_steps: int | None = None,
        strategy: ConsolidationStrategy | None = None,
    ) -> ConsolidationResult | None:
        """Public entry point for externally triggered consolidation.

        Intended to be called by circadian rhythm or other scheduling
        systems. Delegates to consolidate() after checking readiness.
        """
        if self.consolidator is None:
            return None
        if len(self.episodic) < self.consolidator._min_size:
            logger.debug("Consolidation skipped: insufficient experiences")
            return None
        return self.consolidate(agent=agent, num_steps=num_steps, strategy=strategy)

    # -- Working memory lifecycle --

    def decay_working_memory(self) -> int:
        """Tick working memory decay. Returns items removed."""
        if self.working is None:
            return 0
        return self.working.decay_tick()

    def rehearse_working_memory(self, item_ids: list[str] | None = None) -> None:
        """Rehearse working memory items to prevent decay."""
        if self.working is not None:
            self.working.rehearse(item_ids)

    # -- persistence --

    def serialize(self, persist_dir: "Path") -> dict:
        """Save all memory subsystem state to disk. Returns JSON-safe metadata."""
        from pathlib import Path

        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        metadata = {"agent_id": self._agent_id}

        # Episodic memory
        metadata["episodic"] = self.episodic.serialize(persist_dir)

        # Semantic retriever
        if self.semantic is not None:
            metadata["semantic"] = self.semantic.serialize(persist_dir)

        # Generative replay
        if self.generative is not None:
            metadata["generative"] = self.generative.serialize(persist_dir)

        # Consolidator
        if self.consolidator is not None:
            metadata["consolidator"] = self.consolidator.serialize(persist_dir)

        return metadata

    def restore(self, persist_dir: "Path", metadata: dict) -> None:
        """Restore all memory subsystem state from disk."""
        from pathlib import Path

        persist_dir = Path(persist_dir)
        if not persist_dir.exists():
            logger.warning("No persist dir at %s, starting fresh", persist_dir)
            return

        # Episodic memory
        if "episodic" in metadata:
            self.episodic.restore(persist_dir, metadata["episodic"])

        # Semantic retriever
        if self.semantic is not None and "semantic" in metadata:
            self.semantic.restore(persist_dir, metadata["semantic"])

        # Generative replay
        if self.generative is not None and "generative" in metadata:
            self.generative.restore(persist_dir, metadata["generative"])

        # Consolidator
        if self.consolidator is not None and "consolidator" in metadata:
            self.consolidator.restore(persist_dir, metadata["consolidator"])

        logger.info("Restored memory coordinator for agent %s", self._agent_id)

    # -- Agent context for deep memory narration --

    def _build_agent_context(self, agent: Any = None) -> dict[str, Any]:
        """Gather agent context for experience narration."""
        context: dict[str, Any] = {"agent_id": self._agent_id}
        if agent is not None:
            context["agent_role"] = getattr(agent, "role", "GENERALIST")
            context["step"] = getattr(agent, "step_count", 0)
            # Holonic context
            holon = getattr(agent, "_holon", None)
            if holon is not None:
                context["parent_id"] = getattr(holon, "parent_id", "colony")
                peers = getattr(holon, "peer_ids", None)
                context["peer_count"] = len(peers) if peers else 0
            # Circadian phase
            circadian = getattr(agent, "circadian", None)
            if circadian is not None:
                phase = getattr(circadian, "current_phase", None)
                if phase is not None:
                    context["circadian_phase"] = str(phase)
        return context

    # -- State --

    def __repr__(self) -> str:
        parts = [f"agent={self._agent_id}"]
        parts.append(f"episodic={len(self.episodic)}")
        if self.working:
            parts.append(f"working={len(self.working)}/{self.working.capacity}")
        if self.semantic:
            parts.append(f"semantic={self.semantic.n_indexed}")
        if self.generative:
            parts.append(f"generative={self.generative.buffer_size}")
        return f"MemoryCoordinator({', '.join(parts)})"
