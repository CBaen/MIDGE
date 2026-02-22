"""Memory Bridge - Cross-agent memory sharing and deep memory integration.

Enables agents to learn from each other's experiences via the deep
memory layer. Provides the upward flow (Hot -> Deep via narration)
and the cross-agent flow (agent A's experiences inform agent B).

Biological analogy: The way cultural knowledge transmits between
organisms. The way mirror neurons let us learn from others' experiences.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import numpy as np

from .deep_memory import (
    COLLECTION_ANCESTRAL,
    COLLECTION_META,
    COLLECTION_NARRATIVE,
    DeepMemoryStore,
    SearchResult,
)
from .experience_narrator import ExperienceNarrator

logger = logging.getLogger(__name__)


class MemoryBridge:
    """Bridges agent memory coordinators with the deep memory layer.

    Handles:
    - Upward flow: Hot experiences -> narrated -> embedded -> stored in Qdrant
    - Downward flow: Qdrant search -> relevant experiences for agent priming
    - Cross-agent: Agent A can find Agent B's narrated experiences
    - Meta-memory: Updates self-model of memory system (strange loop)
    """

    def __init__(
        self,
        deep_store: DeepMemoryStore,
        narrator: ExperienceNarrator,
        event_bus: Any = None,
        haven: Any = None,
    ) -> None:
        self._store = deep_store
        self._narrator = narrator
        self._bus = event_bus
        self._haven = haven  # HAVEN risk coordinator — filters isolated agents
        self._total_consolidated = 0
        self._total_recalled = 0
        self._total_blocked_by_isolation = 0
        self._total_verified = 0
        self._total_unverified = 0

    @property
    def is_available(self) -> bool:
        """Whether the deep memory backend is reachable."""
        return self._store.is_available()

    # -- Upward Flow (Hot -> Deep) --

    def consolidate_to_deep(
        self,
        agent_id: str,
        experiences: list[Any],
        agent_context: dict[str, Any],
        consolidation_id: int = 0,
    ) -> dict[str, Any]:
        """Narrate experiences and store in Qdrant narrative collection.

        Called during consolidation ("sleep") phases.
        Returns stats dict: {stored, failed, total, consolidation_id}.
        """
        if not self._store.is_available() or not experiences:
            return {
                "stored": 0,
                "failed": len(experiences),
                "total": len(experiences),
                "consolidation_id": consolidation_id,
            }

        # Narrate each experience
        narrations = self._narrator.narrate_batch(experiences, agent_context)

        # Build items for batch storage
        items = []
        for exp, text in zip(experiences, narrations):
            witness_hash = DeepMemoryStore.compute_witness_hash(
                np.asarray(exp.state),
                np.asarray(exp.next_state),
                exp.action,
                float(exp.reward),
            )
            payload = self._narrator.build_payload(
                exp,
                agent_context,
                witness_hash=witness_hash,
                consolidation_id=consolidation_id,
            )
            items.append({"text": text, "payload": payload})

        result = self._store.store_points_batch(COLLECTION_NARRATIVE, items)
        self._total_consolidated += result.get("stored", 0)

        # Store a consolidation summary too
        if result.get("stored", 0) > 0:
            summary_text = self._narrator.narrate_consolidation_summary(
                consolidation_id, agent_context, experiences
            )
            summary_payload = {
                "type": "consolidation_summary",
                "agent_id": agent_context.get("agent_id", "unknown"),
                "agent_role": agent_context.get("agent_role", "GENERALIST"),
                "consolidation_id": consolidation_id,
                "experience_count": len(experiences),
                "mean_reward": float(np.mean([e.reward for e in experiences])),
                "timestamp": time.time(),
            }
            self._store.store_point(
                COLLECTION_NARRATIVE, summary_text, summary_payload
            )

        result["consolidation_id"] = consolidation_id
        return result

    # -- Downward Flow (Deep -> Hot) --

    def recall_from_deep(
        self,
        query_text: str,
        agent_id: str | None = None,
        limit: int = 5,
    ) -> list[SearchResult]:
        """Search Qdrant narrative memory for relevant experiences.

        If agent_id is given, filters to that agent's experiences only.
        Publishes memory.recall_completed on EventBus (triadic witnessing).
        """
        if not self._store.is_available():
            return []

        filters = None
        if agent_id is not None:
            filters = {
                "must": [{"key": "agent_id", "match": {"value": agent_id}}]
            }

        results = self._store.search(
            COLLECTION_NARRATIVE,
            query_text,
            limit=limit,
            filters=filters,
        )
        self._total_recalled += len(results)

        # Witness hash verification (Law 1 — close the bare dyad)
        results = self._verify_results(
            results, "recall_from_deep", agent_id or "unknown",
        )

        # Triadic witnessing: notify EventBus of recall (Law 1 compliance)
        self._publish_recall_event(
            "recall_from_deep", agent_id or "unknown", len(results), query_text,
        )
        return results

    def recall_peer_experiences(
        self,
        query_text: str,
        requesting_agent_id: str,
        peer_agent_ids: list[str] | None = None,
        limit: int = 5,
    ) -> list[SearchResult]:
        """Search other agents' narrative memories.

        Excludes the requesting agent's own experiences.
        If peer_agent_ids given, restricts to those peers only.
        HAVEN check: filters out isolated peers (policy contagion defense).
        Publishes memory.recall_completed on EventBus (triadic witnessing).
        """
        if not self._store.is_available():
            return []

        # HAVEN isolation filter: exclude peers that are isolated
        safe_peers = peer_agent_ids
        if self._haven is not None and peer_agent_ids:
            safe_peers = [
                pid for pid in peer_agent_ids
                if not self._haven.is_agent_isolated(pid)
            ]
            blocked = len(peer_agent_ids) - len(safe_peers)
            if blocked > 0:
                self._total_blocked_by_isolation += blocked
                logger.info(
                    "MemoryBridge: %d isolated peers excluded from recall for %s",
                    blocked, requesting_agent_id,
                )
            if not safe_peers:
                # All peers isolated — return empty
                self._publish_recall_event(
                    "recall_peer_experiences", requesting_agent_id, 0, query_text,
                )
                return []

        must_not = [
            {"key": "agent_id", "match": {"value": requesting_agent_id}}
        ]
        must = []
        if safe_peers:
            must.append({
                "key": "agent_id",
                "match": {"any": safe_peers},
            })

        filters: dict[str, Any] = {"must_not": must_not}
        if must:
            filters["must"] = must

        results = self._store.search(
            COLLECTION_NARRATIVE,
            query_text,
            limit=limit,
            filters=filters,
        )

        # Post-search HAVEN filter: check result source agents
        if self._haven is not None and results:
            pre_count = len(results)
            results = [
                r for r in results
                if not self._haven.is_agent_isolated(
                    r.payload.get("agent_id", "") if r.payload else ""
                )
            ]
            blocked = pre_count - len(results)
            if blocked > 0:
                self._total_blocked_by_isolation += blocked

        self._total_recalled += len(results)

        # Witness hash verification (Law 1 — close the bare dyad)
        results = self._verify_results(
            results, "recall_peer_experiences", requesting_agent_id,
        )

        # Triadic witnessing: notify EventBus
        self._publish_recall_event(
            "recall_peer_experiences", requesting_agent_id, len(results), query_text,
        )
        return results

    # -- Ancestral Memory --

    def recall_ancestral_patterns(
        self,
        query_text: str,
        applicable_role: str | None = None,
        limit: int = 5,
    ) -> list[SearchResult]:
        """Search ancestral memory for relevant patterns.

        Publishes memory.recall_completed on EventBus (triadic witnessing).
        """
        if not self._store.is_available():
            return []

        filters = None
        if applicable_role:
            filters = {
                "must": [
                    {"key": "applicable_roles", "match": {"value": applicable_role}}
                ]
            }

        results = self._store.search(
            COLLECTION_ANCESTRAL,
            query_text,
            limit=limit,
            filters=filters,
        )
        self._total_recalled += len(results)

        # Witness hash verification (Law 1 — ancestral patterns expected unverified)
        results = self._verify_results(
            results, "recall_ancestral_patterns", applicable_role or "any",
        )

        # Triadic witnessing: notify EventBus
        self._publish_recall_event(
            "recall_ancestral_patterns", applicable_role or "any", len(results), query_text,
        )
        return results

    def store_ancestral_pattern(
        self,
        pattern: dict[str, Any],
        contributing_agents: list[str],
    ) -> str | None:
        """Store a distilled pattern in ancestral memory."""
        if not self._store.is_available():
            return None

        text = self._narrator.narrate_pattern(pattern, contributing_agents)
        payload = {
            "type": "pattern",
            "pattern_type": pattern.get("pattern_type", "behavioral"),
            "domain": pattern.get("domain", "general"),
            "occurrence_count": pattern.get("occurrence_count", 0),
            "confidence": pattern.get("confidence", 0.0),
            "contributing_agents": contributing_agents,
            "agent_count": len(contributing_agents),
            "applicable_roles": pattern.get("applicable_roles", []),
            "timestamp": time.time(),
            "primary_store": COLLECTION_ANCESTRAL,
            "verification_store": COLLECTION_NARRATIVE,
            "balance_store": COLLECTION_META,
        }
        return self._store.store_point(COLLECTION_ANCESTRAL, text, payload)

    # -- Meta-Memory (Strange Loop) --

    def update_meta_memory(
        self,
        memory_stats: dict[str, Any],
    ) -> str | None:
        """Update Mae's model of her own memory system.

        This is the strange loop: memory recording facts about memory.
        """
        if not self._store.is_available():
            return None

        # Build self-description
        narrative_stats = self._store.get_collection_stats(COLLECTION_NARRATIVE)
        ancestral_stats = self._store.get_collection_stats(COLLECTION_ANCESTRAL)

        text = (
            f"Mae's memory state: "
            f"{narrative_stats.get('points_count', 0)} narrative memories, "
            f"{ancestral_stats.get('points_count', 0)} ancestral patterns. "
            f"Total consolidated: {self._total_consolidated}. "
            f"Total recalled: {self._total_recalled}. "
            f"Agent count: {memory_stats.get('agent_count', 0)}. "
            f"Hot layer: {memory_stats.get('hot_size', 'unknown')} experiences. "
            f"Warm layer: {memory_stats.get('warm_status', 'unknown')}."
        )

        payload = {
            "type": "self_model",
            "describes_collection": "all",
            "total_narrative_points": narrative_stats.get("points_count", 0),
            "total_ancestral_points": ancestral_stats.get("points_count", 0),
            "total_consolidated": self._total_consolidated,
            "total_recalled": self._total_recalled,
            "agent_count": memory_stats.get("agent_count", 0),
            "timestamp": time.time(),
        }

        return self._store.store_point(COLLECTION_META, text, payload)

    # -- Witness Hash Verification --

    def _verify_results(
        self,
        results: list[SearchResult],
        recall_type: str,
        requester: str,
    ) -> list[SearchResult]:
        """Verify witness hashes on recalled results (Law 1 compliance).

        Checks each result for a non-empty witness_hash in payload.
        Tags results with witness_verified=True/False.
        Publishes verification event on EventBus (triadic witnessing).
        """
        if not results:
            return results

        verified = 0
        unverified = 0
        for r in results:
            payload = r.payload if r.payload else {}
            wh = payload.get("witness_hash", "")
            if wh and isinstance(wh, str) and len(wh) >= 64:
                payload["witness_verified"] = True
                verified += 1
            else:
                payload["witness_verified"] = False
                unverified += 1

        self._total_verified += verified
        self._total_unverified += unverified

        if unverified > 0:
            logger.info(
                "MemoryBridge: %d/%d results unverified in %s for %s",
                unverified, len(results), recall_type, requester,
            )

        # Triadic witnessing: EventBus observes the verification
        self._publish_verification_event(
            recall_type, requester, verified, unverified,
        )
        return results

    def _publish_verification_event(
        self,
        recall_type: str,
        requester: str,
        verified: int,
        unverified: int,
    ) -> None:
        """Publish verification event on EventBus (Law 1 — witness C)."""
        if self._bus is None:
            return
        try:
            self._bus.publish(
                "memory.recall_verified",
                {
                    "recall_type": recall_type,
                    "requester": requester,
                    "verified_count": verified,
                    "unverified_count": unverified,
                    "total": verified + unverified,
                },
            )
        except Exception:
            logger.debug("Failed to publish verification event", exc_info=True)

    # -- EventBus Recall Notification (Triadic Witnessing) --

    def _publish_recall_event(
        self,
        recall_type: str,
        requester: str,
        result_count: int,
        query_text: str,
    ) -> None:
        """Publish recall event on EventBus for triadic witnessing (Law 1).

        The EventBus acts as witness C for the recall dyad (requester ↔ deep store).
        """
        if self._bus is None:
            return
        try:
            self._bus.publish(
                "memory.recall_completed",
                {
                    "recall_type": recall_type,
                    "requester": requester,
                    "result_count": result_count,
                    "query_preview": query_text[:80] if query_text else "",
                },
            )
        except Exception:
            logger.debug("Failed to publish recall event", exc_info=True)

    # -- Stats --

    def get_stats(self) -> dict[str, Any]:
        """Get bridge statistics."""
        return {
            "available": self.is_available,
            "total_consolidated": self._total_consolidated,
            "total_recalled": self._total_recalled,
            "total_verified": self._total_verified,
            "total_unverified": self._total_unverified,
            "total_blocked_by_isolation": self._total_blocked_by_isolation,
            "narrative_stats": self._store.get_collection_stats(COLLECTION_NARRATIVE),
            "ancestral_stats": self._store.get_collection_stats(COLLECTION_ANCESTRAL),
            "meta_stats": self._store.get_collection_stats(COLLECTION_META),
        }

    def __repr__(self) -> str:
        return (
            f"MemoryBridge(consolidated={self._total_consolidated}, "
            f"recalled={self._total_recalled}, "
            f"store={self._store})"
        )
