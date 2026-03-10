"""Fractal ACT — OrganAction.

Extracted from fractal_act.py for single-responsibility.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from mae_core.backbone.holon_registry import HolonRegistry

logger = logging.getLogger(__name__)


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
        subsystem_actions: dict[str, Any],
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
