"""Stigmergy Mixin - Pheromone trail coordination.

Enables indirect agent coordination through environmental markers,
like ant pheromone trails. Agents deposit markers (success, danger,
exploration) and sense nearby markers to guide behavior.

Ported from v5-pivot base_agent.py Big Rock 6 stigmergy methods.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any, Optional

logger = logging.getLogger(__name__)


class StigmergyMixin:
    """Adds stigmergic (pheromone-like) coordination to agents."""

    def _init_stigmergy(
        self, stigmergy_env: Any = None, agent_config: dict[str, Any] | None = None
    ) -> None:
        """Initialize stigmergy attributes."""
        config = agent_config or {}
        self.stigmergy_env = stigmergy_env
        self.stigmergy_position: tuple[float, ...] = (0.0, 0.0)
        self.sensing_radius: float = config.get("sensing_radius", 5.0)
        self.trail_following_enabled: bool = config.get("trail_following", True)

    def deposit_marker(
        self,
        marker_type: str,
        intensity: float = 1.0,
        position: Optional[tuple[float, ...]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[str]:
        """Deposit a stigmergic marker at a position."""
        if self.stigmergy_env is None:
            return None

        pos = position if position is not None else self.stigmergy_position
        return self.stigmergy_env.deposit_marker(
            marker_type=marker_type,
            position=pos,
            depositor_id=str(getattr(self, "unique_id", "unknown")),
            intensity=intensity,
            metadata=metadata,
        )

    def deposit_success_marker(self, reward: float) -> None:
        """Deposit SUCCESS marker scaled by reward."""
        if self.stigmergy_env and reward > 0:
            intensity = min(1.0, reward / 10.0)
            self.deposit_marker("SUCCESS", intensity=intensity, metadata={"reward": reward})

    def deposit_danger_marker(self, risk_level: float) -> None:
        """Deposit DANGER marker if risk is high."""
        if self.stigmergy_env and risk_level > 0.5:
            self.deposit_marker(
                "DANGER",
                intensity=risk_level,
                metadata={"risk_score": getattr(self, "risk_score", 0.0)},
            )

    def deposit_exploration_marker(self) -> None:
        """Mark current position as explored."""
        if self.stigmergy_env:
            self.deposit_marker(
                "EXPLORATION",
                intensity=0.5,
                metadata={"visit_count": 1, "timestamp": time.time()},
            )

    def sense_markers(
        self,
        marker_types: Optional[list[str]] = None,
        radius: Optional[float] = None,
    ) -> list[Any]:
        """Sense markers near current position, sorted by intensity."""
        if self.stigmergy_env is None:
            return []

        r = radius if radius is not None else self.sensing_radius
        return self.stigmergy_env.sense_markers(self.stigmergy_position, r, marker_types)

    def sense_environment(self) -> dict[str, list[Any]]:
        """Sense all markers grouped by type."""
        if self.stigmergy_env is None:
            return {}

        all_markers = self.sense_markers()
        by_type: dict[str, list[Any]] = defaultdict(list)
        for marker in all_markers:
            by_type[marker.marker_type].append(marker)
        return dict(by_type)

    def follow_trail(
        self, marker_type: str, attractive: bool = True
    ) -> tuple[float, ...]:
        """Get direction to follow (or avoid) markers of a type."""
        if self.stigmergy_env is None:
            return tuple(0.0 for _ in range(len(self.stigmergy_position)))

        gradient = self.stigmergy_env.get_gradient(
            self.stigmergy_position,
            marker_type,
            radius=self.sensing_radius,
        )
        if not attractive:
            gradient = tuple(-g for g in gradient)
        return gradient

    def move_in_stigmergy(
        self, direction: tuple[float, ...], step_size: float = 1.0
    ) -> None:
        """Move in stigmergic space."""
        if len(direction) != len(self.stigmergy_position):
            return

        self.stigmergy_position = tuple(
            self.stigmergy_position[i] + direction[i] * step_size
            for i in range(len(self.stigmergy_position))
        )

    def get_strongest_nearby_marker(
        self, marker_type: Optional[str] = None
    ) -> Optional[Any]:
        """Get the strongest marker within sensing radius."""
        if self.stigmergy_env is None:
            return None
        return self.stigmergy_env.get_strongest_marker(
            self.stigmergy_position, marker_type=marker_type, radius=self.sensing_radius
        )

    def set_stigmergy_position(self, position: tuple[float, ...]) -> None:
        """Manually set position in stigmergic space."""
        self.stigmergy_position = position

    def get_stigmergy_statistics(self) -> Optional[dict[str, Any]]:
        """Get stigmergy statistics."""
        if self.stigmergy_env is None:
            return None

        nearby = self.sense_environment()
        return {
            "position": self.stigmergy_position,
            "sensing_radius": self.sensing_radius,
            "nearby_markers": {mtype: len(markers) for mtype, markers in nearby.items()},
            "total_nearby": sum(len(m) for m in nearby.values()),
        }

    def _serialize_stigmergy(self) -> dict:
        return {
            "stigmergy_position": list(self.stigmergy_position) if hasattr(self, "stigmergy_position") else [0.0, 0.0],
        }

    def _restore_stigmergy(self, data: dict) -> None:
        if "stigmergy_position" in data:
            self.stigmergy_position = tuple(data["stigmergy_position"])
