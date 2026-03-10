"""Fractal ACT — SubsystemAction and OrganClusterAction.

Extracted from fractal_act.py for single-responsibility.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from mae_core.backbone.holon_registry import HolonRegistry

logger = logging.getLogger(__name__)


class SubsystemAction:
    """Fractal ACT at the subsystem level.

    A subsystem acts by coordinating its child holons (3 systems).
    It calls act() on each child, aggregates results, and if any
    child failed, attempts repair via the child's heal capability.

    Biological analogy: A tissue coordinating its cells. When a cell
    fails to perform, the tissue signals for repair.
    """

    def __init__(
        self,
        subsystem_id: str,
        registry: HolonRegistry,
    ) -> None:
        self._subsystem_id = subsystem_id
        self._registry = registry
        self._last_result: float = 0.0
        self._act_count: int = 0
        self._total_score: float = 0.0
        self._memory: dict[str, Any] = {}
        self._score_history: list[float] = []

    def act(self) -> float:
        """Coordinate child holons to act collectively.

        1. Call act() on each child's proxy
        2. Aggregate results (mean of child scores)
        3. If any child returned 0.0, attempt heal() on that child
        4. Return aggregate score

        Returns:
            Aggregate action score (0.0-1.0). 0.0 if no children.
        """
        children_ids = self._registry.get_children(self._subsystem_id)
        if not children_ids:
            return 0.0

        scores: list[float] = []
        failed_children: list[str] = []

        for child_id in children_ids:
            proxy = self._registry.get_proxy(child_id)
            # Use act() if available (HolonProxy), else try step on system_ref
            act_fn = getattr(proxy, "act", None)
            if callable(act_fn):
                try:
                    score = act_fn()
                    scores.append(score)
                    if score <= 0.0:
                        failed_children.append(child_id)
                except Exception:
                    scores.append(0.0)
                    failed_children.append(child_id)
            else:
                scores.append(0.0)
                failed_children.append(child_id)

        # Attempt heal on failed children (advisory — never blocks)
        for child_id in failed_children:
            proxy = self._registry.get_proxy(child_id)
            heal_fn = getattr(proxy, "heal", None)
            if callable(heal_fn):
                try:
                    heal_fn()  # Triggers recovery + SomaticMap reporting
                except Exception:
                    pass

        aggregate = sum(scores) / len(scores) if scores else 0.0
        self._last_result = aggregate
        self._act_count += 1
        self._total_score += aggregate
        return aggregate

    def get_statistics(self) -> dict[str, Any]:
        """Subsystem action statistics."""
        return {
            "subsystem_id": self._subsystem_id,
            "act_count": self._act_count,
            "last_result": self._last_result,
            "mean_score": (
                self._total_score / self._act_count
                if self._act_count > 0
                else 0.0
            ),
        }

    def get_state(self) -> dict[str, Any]:
        """Operational state for HolonProxy.sense() delegation."""
        return {
            "subsystem_id": self._subsystem_id,
            "last_result": self._last_result,
            "act_count": self._act_count,
            "children": self._registry.get_children(self._subsystem_id),
        }

    def sense(self) -> dict[str, Any]:
        """Aggregate children's operational state."""
        children_ids = self._registry.get_children(self._subsystem_id)
        states: dict[str, Any] = {}
        for child_id in children_ids:
            proxy = self._registry.get_proxy(child_id)
            try:
                states[child_id] = proxy.sense()
            except Exception:
                states[child_id] = {"error": True}
        return {"subsystem_id": self._subsystem_id, "children_states": states}

    def remember(self, key: str, value: Any = None) -> Any:
        """Store or retrieve from per-subsystem memory."""
        if value is not None:
            self._memory[key] = value
            return value
        return self._memory.get(key)

    def decide(self, stimulus: Any = None) -> list[tuple[str, float]]:
        """Decide which children need attention (lowest health first)."""
        children_ids = self._registry.get_children(self._subsystem_id)
        scores: dict[str, float] = {}
        for child_id in children_ids:
            proxy = self._registry.get_proxy(child_id)
            scores[child_id] = proxy.get_health()
        return sorted(scores.items(), key=lambda x: x[1])

    def learn(self, feedback: Any = None) -> None:
        """Track aggregate performance trends."""
        self._score_history.append(self._last_result)
        if len(self._score_history) > 100:
            self._score_history = self._score_history[-100:]

    def heal(self) -> dict[str, Any]:
        """Detect failing children, call heal() on them."""
        children_ids = self._registry.get_children(self._subsystem_id)
        healed: list[str] = []
        for child_id in children_ids:
            proxy = self._registry.get_proxy(child_id)
            health = proxy.get_health()
            if health <= 0.5:
                try:
                    proxy.heal()
                    healed.append(child_id)
                except Exception:
                    pass
        return {"subsystem_id": self._subsystem_id, "healed": healed}

    def know_self(self) -> dict[str, Any]:
        """Self-model: ID, type, children count, peers, performance."""
        entry = self._registry.get_entry(self._subsystem_id)
        return {
            "holon_id": self._subsystem_id,
            "holon_type": entry.holon_type if entry else "subsystem",
            "children_count": len(self._registry.get_children(self._subsystem_id)),
            "peers_count": len(self._registry.get_peers(self._subsystem_id)),
            "last_result": self._last_result,
        }

    def know_up(self) -> Optional[dict[str, Any]]:
        """Aware of parent context."""
        parent_id = self._registry.get_parent(self._subsystem_id)
        if parent_id is None:
            return None
        entry = self._registry.get_entry(parent_id)
        return {
            "parent_id": parent_id,
            "parent_type": entry.holon_type if entry else None,
        }

    def know_down(self) -> list[dict[str, Any]]:
        """Aware of child components."""
        children: list[dict[str, Any]] = []
        for child_id in self._registry.get_children(self._subsystem_id):
            entry = self._registry.get_entry(child_id)
            children.append({
                "holon_id": child_id,
                "holon_type": entry.holon_type if entry else "system",
            })
        return children

    def know_peers(self) -> list[dict[str, Any]]:
        """Aware of sibling subsystems."""
        peers: list[dict[str, Any]] = []
        for peer_id in self._registry.get_peers(self._subsystem_id):
            entry = self._registry.get_entry(peer_id)
            peers.append({
                "holon_id": peer_id,
                "holon_type": entry.holon_type if entry else "subsystem",
            })
        return peers


class OrganClusterAction:
    """Fractal ACT at the organ cluster level.

    An organ cluster coordinates 2-3 related organs (e.g., metabolic + somatic
    as the vital cluster, or cognitive + sensory as the cognitive cluster).
    It aggregates their performance and triggers healing when clusters degrade.

    Biological analogy: A functional body region (e.g., "circulatory + metabolic
    as cardiopulmonary system") coordinating subsystems across organs.
    """

    def __init__(
        self,
        cluster_id: str,
        organ_actions: dict[str, "OrganAction"],  # type: ignore[name-defined]
        registry: Optional[HolonRegistry] = None,
    ) -> None:
        self._cluster_id = cluster_id
        self._organ_actions = organ_actions
        self._registry = registry
        self._last_result: float = 0.0
        self._act_count: int = 0
        self._total_score: float = 0.0
        self._organ_scores: dict[str, float] = {}
        self._memory: dict[str, Any] = {}
        self._score_history: list[float] = []

    def act(self) -> float:
        """Coordinate organ actions within the cluster.

        1. Call act() on each organ in the cluster
        2. Aggregate results (mean of organ scores)
        3. If any organ failed, attempt heal() on that organ
        4. Return aggregate cluster score

        Returns:
            Aggregate cluster action score (0.0-1.0). 0.0 if no organs.
        """
        if not self._organ_actions:
            return 0.0

        scores: dict[str, float] = {}
        failed_organs: list[str] = []

        for organ_id, organ_action in self._organ_actions.items():
            try:
                score = organ_action.act()
                scores[organ_id] = score
                if score <= 0.0:
                    failed_organs.append(organ_id)
            except Exception:
                logger.debug(
                    "OrganClusterAction: organ %s failed during act()",
                    organ_id, exc_info=True,
                )
                scores[organ_id] = 0.0
                failed_organs.append(organ_id)

        # Attempt heal on failed organs (advisory — never blocks)
        for organ_id in failed_organs:
            try:
                self._organ_actions[organ_id].heal()
            except Exception:
                pass

        aggregate = sum(scores.values()) / len(scores) if scores else 0.0
        self._organ_scores = dict(scores)
        self._last_result = aggregate
        self._act_count += 1
        self._total_score += aggregate
        return aggregate

    def get_statistics(self) -> dict[str, Any]:
        """Organ cluster action statistics."""
        organ_stats = {
            oid: organ.get_statistics()
            for oid, organ in self._organ_actions.items()
        }
        return {
            "cluster_id": self._cluster_id,
            "act_count": self._act_count,
            "last_result": self._last_result,
            "mean_score": (
                self._total_score / self._act_count
                if self._act_count > 0
                else 0.0
            ),
            "organ_count": len(self._organ_actions),
            "organ_scores": dict(self._organ_scores),
            "organ_stats": organ_stats,
        }

    def get_state(self) -> dict[str, Any]:
        """Operational state for HolonProxy.sense() delegation."""
        return {
            "cluster_id": self._cluster_id,
            "last_result": self._last_result,
            "act_count": self._act_count,
            "organs": list(self._organ_actions.keys()),
            "organ_scores": dict(self._organ_scores),
        }

    def sense(self) -> dict[str, Any]:
        """Aggregate organ states within cluster."""
        states: dict[str, Any] = {}
        for organ_id, organ_action in self._organ_actions.items():
            try:
                states[organ_id] = organ_action.sense()
            except Exception:
                states[organ_id] = {"error": True}
        return {"cluster_id": self._cluster_id, "organ_states": states}

    def remember(self, key: str, value: Any = None) -> Any:
        """Store or retrieve from per-cluster memory."""
        if value is not None:
            self._memory[key] = value
            return value
        return self._memory.get(key)

    def decide(self, stimulus: Any = None) -> list[tuple[str, float]]:
        """Decide which organs need attention (lowest score first)."""
        return sorted(self._organ_scores.items(), key=lambda x: x[1])

    def learn(self, feedback: Any = None) -> None:
        """Track aggregate performance trends."""
        self._score_history.append(self._last_result)
        if len(self._score_history) > 100:
            self._score_history = self._score_history[-100:]

    def heal(self) -> dict[str, Any]:
        """Detect failing organs, trigger their heal()."""
        healed: list[str] = []
        for organ_id, organ_action in self._organ_actions.items():
            if organ_action._last_result <= 0.3:
                try:
                    organ_action.heal()
                    healed.append(organ_id)
                except Exception:
                    pass
        return {"cluster_id": self._cluster_id, "healed": healed}

    def know_self(self) -> dict[str, Any]:
        """Self-model: cluster ID, type, organ count, performance."""
        result: dict[str, Any] = {
            "holon_id": self._cluster_id,
            "holon_type": "organ_cluster",
            "organ_count": len(self._organ_actions),
            "last_result": self._last_result,
        }
        if self._registry:
            result["peers_count"] = len(self._registry.get_peers(self._cluster_id))
        return result

    def know_up(self) -> Optional[dict[str, Any]]:
        """Aware of parent context (mae organism)."""
        if self._registry is None:
            return {"parent_id": "mae", "parent_type": "organism"}
        parent_id = self._registry.get_parent(self._cluster_id)
        if parent_id is None:
            return {"parent_id": "mae", "parent_type": "organism"}
        entry = self._registry.get_entry(parent_id)
        return {
            "parent_id": parent_id,
            "parent_type": entry.holon_type if entry else "organism",
        }

    def know_down(self) -> list[dict[str, Any]]:
        """Aware of child organs."""
        return [
            {"holon_id": oid, "holon_type": "organ", "last_result": o._last_result}
            for oid, o in self._organ_actions.items()
        ]

    def know_peers(self) -> list[dict[str, Any]]:
        """Aware of sibling clusters."""
        if self._registry is None:
            return []
        peers: list[dict[str, Any]] = []
        for peer_id in self._registry.get_peers(self._cluster_id):
            entry = self._registry.get_entry(peer_id)
            peers.append({
                "holon_id": peer_id,
                "holon_type": entry.holon_type if entry else "organ_cluster",
            })
        return peers

