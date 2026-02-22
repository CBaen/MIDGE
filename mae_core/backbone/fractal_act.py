"""Fractal ACT — action delegation at every scale (Law 3: Holon Protocol).

Extracted from fractal_generator.py for single-responsibility.

Same pattern at every level:
  Cell (agent):     Execute specific action in TaskPool
  Tissue (subsystem): Coordinate children, aggregate results
  Organ:            Strategic allocation across subsystems
  Organism:         Global homeostasis across all organs

Each level DELEGATES to the level below and AGGREGATES results upward.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from mae_core.backbone.event_bus import EventBus
from mae_core.backbone.fractal_generator import FRACTAL_GROUPING
from mae_core.backbone.holon_protocol import HolonRegistry

logger = logging.getLogger(__name__)

# =====================================================================
# EventBus channel
# =====================================================================
CH_FRACTAL_ACT = "fractal.act"


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
        organ_actions: dict[str, OrganAction],
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


class OrganAction:
    """Fractal ACT at the organ level.

    An organ acts by coordinating its subsystem actions. It checks
    which subsystems are most critical (lowest recent score) and
    prioritizes action on those.

    Biological analogy: An organ coordinating its tissues. A heart
    prioritizes blood flow to the tissue under most stress.
    """

    def __init__(
        self,
        organ_id: str,
        subsystem_actions: dict[str, SubsystemAction],
        registry: Optional[HolonRegistry] = None,
    ) -> None:
        self._organ_id = organ_id
        self._subsystem_actions = subsystem_actions
        self._registry = registry
        self._last_result: float = 0.0
        self._act_count: int = 0
        self._total_score: float = 0.0
        self._memory: dict[str, Any] = {}
        self._score_history: list[float] = []

    def act(self) -> float:
        """Coordinate subsystem actions strategically.

        1. Act on all subsystems
        2. Weight results (lower scores get more attention next time)
        3. Return aggregate organ performance

        Returns:
            Aggregate organ score (0.0-1.0). 0.0 if no subsystems.
        """
        if not self._subsystem_actions:
            return 0.0

        scores: dict[str, float] = {}
        for sub_id, sub_action in self._subsystem_actions.items():
            try:
                scores[sub_id] = sub_action.act()
            except Exception:
                logger.debug(
                    "OrganAction: subsystem %s failed during act()",
                    sub_id, exc_info=True,
                )
                scores[sub_id] = 0.0

        aggregate = (
            sum(scores.values()) / len(scores) if scores else 0.0
        )
        self._last_result = aggregate
        self._act_count += 1
        self._total_score += aggregate
        return aggregate

    def get_statistics(self) -> dict[str, Any]:
        """Organ action statistics."""
        subsystem_stats = {
            sub_id: sub.get_statistics()
            for sub_id, sub in self._subsystem_actions.items()
        }
        return {
            "organ_id": self._organ_id,
            "act_count": self._act_count,
            "last_result": self._last_result,
            "mean_score": (
                self._total_score / self._act_count
                if self._act_count > 0
                else 0.0
            ),
            "subsystem_count": len(self._subsystem_actions),
            "subsystem_stats": subsystem_stats,
        }

    def get_state(self) -> dict[str, Any]:
        """Operational state for HolonProxy.sense() delegation."""
        return {
            "organ_id": self._organ_id,
            "last_result": self._last_result,
            "act_count": self._act_count,
            "subsystems": list(self._subsystem_actions.keys()),
        }

    def sense(self) -> dict[str, Any]:
        """Aggregate subsystem states."""
        states: dict[str, Any] = {}
        for sub_id, sub_action in self._subsystem_actions.items():
            try:
                states[sub_id] = sub_action.sense()
            except Exception:
                states[sub_id] = {"error": True}
        return {"organ_id": self._organ_id, "subsystem_states": states}

    def remember(self, key: str, value: Any = None) -> Any:
        """Store or retrieve from per-organ memory."""
        if value is not None:
            self._memory[key] = value
            return value
        return self._memory.get(key)

    def decide(self, stimulus: Any = None) -> list[tuple[str, float]]:
        """Decide which subsystems need attention (lowest score first)."""
        scores: dict[str, float] = {}
        for sub_id, sub_action in self._subsystem_actions.items():
            scores[sub_id] = sub_action._last_result
        return sorted(scores.items(), key=lambda x: x[1])

    def learn(self, feedback: Any = None) -> None:
        """Track aggregate performance trends."""
        self._score_history.append(self._last_result)
        if len(self._score_history) > 100:
            self._score_history = self._score_history[-100:]

    def heal(self) -> dict[str, Any]:
        """Detect failing subsystems, trigger their heal()."""
        healed: list[str] = []
        for sub_id, sub_action in self._subsystem_actions.items():
            if sub_action._last_result <= 0.3:
                try:
                    sub_action.heal()
                    healed.append(sub_id)
                except Exception:
                    pass
        return {"organ_id": self._organ_id, "healed": healed}

    def know_self(self) -> dict[str, Any]:
        """Self-model: ID, type, subsystem count, performance."""
        result: dict[str, Any] = {
            "holon_id": self._organ_id,
            "holon_type": "organ",
            "subsystem_count": len(self._subsystem_actions),
            "last_result": self._last_result,
        }
        if self._registry:
            result["peers_count"] = len(self._registry.get_peers(self._organ_id))
        return result

    def know_up(self) -> Optional[dict[str, Any]]:
        """Aware of parent context."""
        if self._registry is None:
            return None
        parent_id = self._registry.get_parent(self._organ_id)
        if parent_id is None:
            return None
        entry = self._registry.get_entry(parent_id)
        return {
            "parent_id": parent_id,
            "parent_type": entry.holon_type if entry else None,
        }

    def know_down(self) -> list[dict[str, Any]]:
        """Aware of child subsystems."""
        return [
            {"holon_id": sub_id, "holon_type": "subsystem", "last_result": sub._last_result}
            for sub_id, sub in self._subsystem_actions.items()
        ]

    def know_peers(self) -> list[dict[str, Any]]:
        """Aware of sibling organs."""
        if self._registry is None:
            return []
        peers: list[dict[str, Any]] = []
        for peer_id in self._registry.get_peers(self._organ_id):
            entry = self._registry.get_entry(peer_id)
            peers.append({
                "holon_id": peer_id,
                "holon_type": entry.holon_type if entry else "organ",
            })
        return peers


class OrganismAction:
    """Fractal ACT at the organism level — Mae as a whole.

    The organism acts by coordinating organ clusters and bridge organs.
    Clusters provide triadic structure at organism level:
    - organ-cluster-vital (metabolic + somatic)
    - organ-cluster-cognitive (cognitive + sensory)
    - nervous-system (bridge, coordinates all others)

    After all agents have individually acted (cell-level ACT in TaskPool),
    the organism-level ACT ensures macro-level homeostasis across all systems.

    Biological analogy: The body's integrated response — organ cluster systems
    (cardiopulmonary, nervous, immune) coordinating to maintain viability.
    """

    def __init__(
        self,
        cluster_actions: Optional[dict[str, OrganClusterAction]] = None,
        bridge_organs: Optional[dict[str, OrganAction]] = None,
        event_bus: Optional[EventBus] = None,
        registry: Optional[HolonRegistry] = None,
    ) -> None:
        # Legacy support: if cluster_actions looks like organ_actions dict (all values are OrganAction)
        # and bridge_organs is None, treat it as old-style organ_actions parameter
        if (
            cluster_actions is not None
            and bridge_organs is None
            and isinstance(cluster_actions, dict)
            and all(isinstance(v, OrganAction) for v in cluster_actions.values())
        ):
            # Old-style call: OrganismAction(organ_actions_dict)
            self._cluster_actions = {}
            self._bridge_organs = cluster_actions
            self._organ_scores_legacy = {}  # Track for backward compat
        else:
            self._cluster_actions = cluster_actions or {}
            self._bridge_organs = bridge_organs or {}
            self._organ_scores_legacy = None

        self._bus = event_bus
        self._registry = registry
        self._last_result: float = 0.0
        self._act_count: int = 0
        self._total_score: float = 0.0
        self._cluster_scores: dict[str, float] = {}
        self._bridge_scores: dict[str, float] = {}
        self._memory: dict[str, Any] = {}
        self._score_history: list[float] = []

    # Legacy support: allow organ_actions as direct parameter (converts to clusters/bridge)
    @property
    def _organ_actions(self) -> dict[str, OrganAction]:
        """Legacy property: return all organs (clusters' organs + bridge organs)."""
        result = {}
        for cluster_action in self._cluster_actions.values():
            result.update(cluster_action._organ_actions)
        result.update(self._bridge_organs)
        return result

    @property
    def _organ_scores(self) -> dict[str, float]:
        """Legacy property: return organ scores (bridge organs in legacy mode)."""
        if self._organ_scores_legacy is not None:
            return self._organ_scores_legacy
        # In new mode, aggregate cluster + bridge scores
        return {**self._cluster_scores, **self._bridge_scores}

    @_organ_scores.setter
    def _organ_scores(self, value: dict[str, float]) -> None:
        """Legacy setter: allow tests to set _organ_scores directly."""
        if self._organ_scores_legacy is not None:
            self._organ_scores_legacy = value

    def act(self) -> float:
        """Coordinate clusters and bridge organs for organism-level homeostasis.

        Law 4: Same pattern at every scale — organism delegates to 3 children:
        - organ-cluster-vital (metabolic + somatic)
        - organ-cluster-cognitive (cognitive + sensory)
        - nervous-system (bridge organ, coordinates all others)

        1. Each cluster acts (coordinating its organs)
        2. Each bridge organ acts
        3. Cross-cluster/bridge comparison (detect imbalances)
        4. Compute organism-level health metric
        5. Publish results on EventBus for observability

        Returns:
            Organism-level action score (0.0-1.0).
        """
        all_scores: dict[str, float] = {}

        # Coordinate organ clusters
        for cluster_id, cluster_action in self._cluster_actions.items():
            try:
                score = cluster_action.act()
                all_scores[cluster_id] = score
                self._cluster_scores[cluster_id] = score
            except Exception:
                logger.debug(
                    "OrganismAction: cluster %s failed during act()",
                    cluster_id, exc_info=True,
                )
                all_scores[cluster_id] = 0.0
                self._cluster_scores[cluster_id] = 0.0

        # Coordinate bridge organs
        for bridge_id, bridge_action in self._bridge_organs.items():
            try:
                score = bridge_action.act()
                all_scores[bridge_id] = score
                self._bridge_scores[bridge_id] = score
                # Legacy: also track in _organ_scores_legacy
                if self._organ_scores_legacy is not None:
                    self._organ_scores_legacy[bridge_id] = score
            except Exception:
                logger.debug(
                    "OrganismAction: bridge organ %s failed during act()",
                    bridge_id, exc_info=True,
                )
                all_scores[bridge_id] = 0.0
                self._bridge_scores[bridge_id] = 0.0
                if self._organ_scores_legacy is not None:
                    self._organ_scores_legacy[bridge_id] = 0.0

        aggregate = (
            sum(all_scores.values()) / len(all_scores) if all_scores else 0.0
        )
        self._last_result = aggregate
        self._act_count += 1
        self._total_score += aggregate

        # Publish organism-level action report
        if self._bus is not None:
            try:
                self._bus.publish(CH_FRACTAL_ACT, {
                    "act_count": self._act_count,
                    "organism_score": aggregate,
                    "cluster_scores": dict(self._cluster_scores),
                    "bridge_scores": dict(self._bridge_scores),
                })
            except Exception:
                pass

        return aggregate

    def get_statistics(self) -> dict[str, Any]:
        """Organism action statistics."""
        cluster_stats = {
            cid: cluster.get_statistics()
            for cid, cluster in self._cluster_actions.items()
        }
        bridge_stats = {
            bid: bridge.get_statistics()
            for bid, bridge in self._bridge_organs.items()
        }
        result = {
            "act_count": self._act_count,
            "last_result": self._last_result,
            "mean_score": (
                self._total_score / self._act_count
                if self._act_count > 0
                else 0.0
            ),
            "cluster_count": len(self._cluster_actions),
            "bridge_count": len(self._bridge_organs),
            "last_cluster_scores": dict(self._cluster_scores),
            "last_bridge_scores": dict(self._bridge_scores),
            "cluster_stats": cluster_stats,
            "bridge_stats": bridge_stats,
        }
        # Legacy: organ_count and last_organ_scores
        result["organ_count"] = len(self._organ_actions)
        result["last_organ_scores"] = self._organ_scores
        # Legacy: organ_stats (aggregate from cluster and bridge)
        organ_stats = {}
        for cid, cluster in self._cluster_actions.items():
            for oid, organ in cluster._organ_actions.items():
                organ_stats[oid] = organ.get_statistics()
        for bid, organ in self._bridge_organs.items():
            organ_stats[bid] = organ.get_statistics()
        result["organ_stats"] = organ_stats
        return result

    def get_state(self) -> dict[str, Any]:
        """Operational state for HolonProxy.sense() delegation."""
        result = {
            "organism": "mae",
            "last_result": self._last_result,
            "act_count": self._act_count,
            "clusters": list(self._cluster_actions.keys()),
            "bridge_organs": list(self._bridge_organs.keys()),
            "cluster_scores": dict(self._cluster_scores),
            "bridge_scores": dict(self._bridge_scores),
        }
        # Legacy: organs key
        result["organs"] = list(self._organ_actions.keys())
        result["organ_scores"] = self._organ_scores
        return result

    def sense(self) -> dict[str, Any]:
        """Aggregate cluster and bridge organ states — organism-level awareness."""
        cluster_states: dict[str, Any] = {}
        bridge_states: dict[str, Any] = {}
        organ_states: dict[str, Any] = {}
        for cluster_id, cluster_action in self._cluster_actions.items():
            try:
                cluster_states[cluster_id] = cluster_action.sense()
            except Exception:
                cluster_states[cluster_id] = {"error": True}
        for bridge_id, bridge_action in self._bridge_organs.items():
            try:
                bridge_states[bridge_id] = bridge_action.sense()
                organ_states[bridge_id] = bridge_states[bridge_id]  # Legacy
            except Exception:
                bridge_states[bridge_id] = {"error": True}
                organ_states[bridge_id] = {"error": True}
        result = {
            "organism": "mae",
            "cluster_states": cluster_states,
            "bridge_states": bridge_states,
        }
        # Legacy: organ_states (aggregate from all organs)
        for cluster_id, cluster_action in self._cluster_actions.items():
            for oid, organ in cluster_action._organ_actions.items():
                try:
                    organ_states[oid] = organ.sense()
                except Exception:
                    organ_states[oid] = {"error": True}
        result["organ_states"] = organ_states
        return result

    def remember(self, key: str, value: Any = None) -> Any:
        """Store or retrieve from organism-level memory."""
        if value is not None:
            self._memory[key] = value
            return value
        return self._memory.get(key)

    def decide(self, stimulus: Any = None) -> list[tuple[str, float]]:
        """Decide which clusters/bridges need attention (lowest score first)."""
        all_scores = {**self._cluster_scores, **self._bridge_scores}
        return sorted(all_scores.items(), key=lambda x: x[1])

    def learn(self, feedback: Any = None) -> None:
        """Track organism-level performance trends."""
        self._score_history.append(self._last_result)
        if len(self._score_history) > 100:
            self._score_history = self._score_history[-100:]

    def heal(self) -> dict[str, Any]:
        """Detect failing clusters/bridges, trigger their heal()."""
        healed: list[str] = []
        for cluster_id, cluster_action in self._cluster_actions.items():
            if cluster_action._last_result <= 0.3:
                try:
                    cluster_action.heal()
                    healed.append(cluster_id)
                except Exception:
                    pass
        for bridge_id, bridge_action in self._bridge_organs.items():
            if bridge_action._last_result <= 0.3:
                try:
                    bridge_action.heal()
                    healed.append(bridge_id)
                except Exception:
                    pass
        return {"organism": "mae", "healed": healed}

    def know_self(self) -> dict[str, Any]:
        """Self-model: organism-level identity."""
        return {
            "holon_id": "mae",
            "holon_type": "organism",
            "organ_count": len(self._organ_actions),
            "last_result": self._last_result,
        }

    def know_up(self) -> Optional[dict[str, Any]]:
        """Mae is the root — no parent."""
        return None

    def know_down(self) -> list[dict[str, Any]]:
        """Aware of child clusters and bridge organs (3-level hierarchy)."""
        result: list[dict[str, Any]] = []
        for cid, cluster in self._cluster_actions.items():
            result.append({
                "holon_id": cid,
                "holon_type": "organ_cluster",
                "last_result": cluster._last_result,
            })
        for bid, bridge in self._bridge_organs.items():
            result.append({
                "holon_id": bid,
                "holon_type": "organ",
                "last_result": bridge._last_result,
            })
        return result

    def know_peers(self) -> list[dict[str, Any]]:
        """Mae has no peers — she is the whole organism."""
        return []


def build_fractal_action(
    registry: HolonRegistry,
    grouping: Optional[dict[str, dict[str, list[str]]]] = None,
    organ_grouping: Optional[dict[str, list[str]]] = None,
    event_bus: Optional[EventBus] = None,
) -> OrganismAction:
    """Build the complete fractal action hierarchy from the grouping maps.

    Law 4: Same pattern at every scale. Creates:
    - SubsystemAction: coordinates 3 systems
    - OrganAction: coordinates 3 subsystems/modules
    - OrganClusterAction: coordinates 2 organs
    - OrganismAction: coordinates 3 clusters/bridges (K3)

    Args:
        registry: HolonRegistry with the fractal hierarchy already built
        grouping: Organ->subsystems map (defaults to FRACTAL_GROUPING)
        organ_grouping: Cluster->organs map (defaults to ORGAN_GROUPING)
        event_bus: EventBus for organism-level action events

    Returns:
        OrganismAction with 3 children: 2 clusters + 1 bridge organ.
    """
    from mae_core.backbone.fractal_generator import ORGAN_GROUPING as DEFAULT_ORGAN_GROUPING

    grouping = grouping or FRACTAL_GROUPING
    organ_grouping = organ_grouping or DEFAULT_ORGAN_GROUPING

    # Step 1: Build all organ actions (SubsystemAction -> OrganAction)
    all_organ_actions: dict[str, OrganAction] = {}

    for organ_name, subsystems_spec in grouping.items():
        subsystem_actions: dict[str, SubsystemAction] = {}

        for sub_name in subsystems_spec:
            subsystem_actions[sub_name] = SubsystemAction(
                subsystem_id=sub_name,
                registry=registry,
            )

        all_organ_actions[organ_name] = OrganAction(
            organ_id=organ_name,
            subsystem_actions=subsystem_actions,
            registry=registry,
        )

    # Step 2: Group organs into clusters (OrganClusterAction)
    cluster_actions: dict[str, OrganClusterAction] = {}
    clustered_organs: set[str] = set()

    for cluster_name, organ_names in organ_grouping.items():
        # Only create clusters with organs that exist
        valid_organs = [o for o in organ_names if o in all_organ_actions]
        if len(valid_organs) >= 2:
            cluster_organ_actions = {o: all_organ_actions[o] for o in valid_organs}
            cluster_actions[cluster_name] = OrganClusterAction(
                cluster_id=cluster_name,
                organ_actions=cluster_organ_actions,
                registry=registry,
            )
            clustered_organs.update(valid_organs)

    # Step 3: Identify bridge organs (organs not in any cluster)
    bridge_organs: dict[str, OrganAction] = {}
    for organ_name, organ_action in all_organ_actions.items():
        if organ_name not in clustered_organs:
            bridge_organs[organ_name] = organ_action

    # Step 4: Build organism action (coordinates clusters + bridges)
    return OrganismAction(
        cluster_actions=cluster_actions,
        bridge_organs=bridge_organs,
        event_bus=event_bus,
        registry=registry,
    )
