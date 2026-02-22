"""Spatial consensus - location-aware collective decision making.

Biological analogy: Ant colonies make decisions that are spatially
distributed - pheromone concentrations at different locations create
different consensus outcomes. Bees waggle-dance with directional
encoding that integrates spatial information into collective choice.

Spatial consensus extends Mae's quorum sensing with location awareness.
Instead of just "what do agents agree on?" it answers "what do agents
NEAR THIS LOCATION agree on?" This enables:
- Regional specialization (different behaviors in different areas)
- Spatial coordination (nearby agents align)
- Hot spot detection (where is consensus strongest?)
- Gradient-guided behavior (move toward/away from consensus)

Connection points:
- QuorumSensor provides base consensus mechanism
- Substrate provides agent positions
- PredictiveField provides spatial context
- Morphogenesis reads spatial consensus patterns
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class SpatialVote:
    """A vote with spatial location."""

    agent_id: int
    position: tuple[float, float]
    topic: str
    value: Any  # The vote value (bool, float, string, etc.)
    confidence: float = 1.0
    weight: float = 1.0  # Expertise-based weight
    timestamp: int = 0


@dataclass
class SpatialConsensusResult:
    """Result of spatial consensus at a location."""

    topic: str
    position: tuple[float, float]
    radius: float
    votes: int
    consensus_value: Any
    consensus_strength: float  # 0.0 (no agreement) to 1.0 (unanimous)
    participants: list[int]  # Agent IDs that voted


class SpatialConsensusTracker:
    """Tracks votes and computes location-aware consensus.

    Uses a grid-based spatial index for efficient queries.
    Supports weighted voting (expertise-based), distance decay
    (closer votes count more), and temporal decay (older votes
    fade).
    """

    def __init__(
        self,
        grid_resolution: float = 5.0,
        max_votes_per_cell: int = 100,
        temporal_decay: float = 0.1,  # Per-step decay of vote weight
    ) -> None:
        self._grid_resolution = grid_resolution
        self._max_votes = max_votes_per_cell
        self._temporal_decay = temporal_decay

        # Grid-indexed votes: (grid_x, grid_y) -> topic -> [votes]
        self._grid: dict[tuple[int, int], dict[str, list[SpatialVote]]] = defaultdict(
            lambda: defaultdict(list)
        )

        self._step_count = 0
        self._total_votes = 0

    # =========================================================================
    # Vote Management
    # =========================================================================

    def add_vote(
        self,
        agent_id: int,
        position: tuple[float, float],
        topic: str,
        value: Any,
        confidence: float = 1.0,
        weight: float = 1.0,
    ) -> None:
        """Add a spatially-located vote."""
        vote = SpatialVote(
            agent_id=agent_id,
            position=position,
            topic=topic,
            value=value,
            confidence=confidence,
            weight=weight,
            timestamp=self._step_count,
        )

        cell = self._world_to_grid(position)
        topic_votes = self._grid[cell][topic]

        # Replace existing vote from same agent on same topic
        topic_votes[:] = [v for v in topic_votes if v.agent_id != agent_id]
        topic_votes.append(vote)

        # Enforce capacity
        if len(topic_votes) > self._max_votes:
            topic_votes.sort(key=lambda v: v.timestamp)
            self._grid[cell][topic] = topic_votes[-self._max_votes :]

        self._total_votes += 1

    # =========================================================================
    # Consensus Queries
    # =========================================================================

    def get_consensus_at(
        self,
        position: tuple[float, float],
        topic: str,
        radius: float = 10.0,
        min_votes: int = 3,
    ) -> Optional[SpatialConsensusResult]:
        """Get consensus at a location within a radius.

        Votes closer to the query position have higher weight (distance decay).
        Votes from recent steps have higher weight (temporal decay).
        """
        center_cell = self._world_to_grid(position)
        cell_radius = max(1, int(radius / self._grid_resolution))

        # Gather votes from nearby cells
        relevant_votes: list[tuple[SpatialVote, float]] = []  # (vote, effective_weight)

        for dx in range(-cell_radius, cell_radius + 1):
            for dy in range(-cell_radius, cell_radius + 1):
                cell = (center_cell[0] + dx, center_cell[1] + dy)
                if cell not in self._grid or topic not in self._grid[cell]:
                    continue

                for vote in self._grid[cell][topic]:
                    # Distance decay
                    vx, vy = vote.position
                    dist = ((vx - position[0]) ** 2 + (vy - position[1]) ** 2) ** 0.5
                    if dist > radius:
                        continue

                    distance_weight = max(0.0, 1.0 - dist / radius)

                    # Temporal decay
                    age = self._step_count - vote.timestamp
                    temporal_weight = max(0.0, 1.0 - age * self._temporal_decay)

                    effective_weight = (
                        vote.weight * vote.confidence * distance_weight * temporal_weight
                    )
                    if effective_weight > 0.01:
                        relevant_votes.append((vote, effective_weight))

        if len(relevant_votes) < min_votes:
            return None

        # Compute weighted consensus
        return self._compute_consensus(
            topic, position, radius, relevant_votes
        )

    def get_spatial_heatmap(
        self,
        topic: str,
        min_votes: int = 3,
    ) -> dict[tuple[int, int], float]:
        """Get consensus strength across the grid for a topic."""
        heatmap: dict[tuple[int, int], float] = {}

        for cell_pos, topics in self._grid.items():
            if topic not in topics:
                continue
            votes = topics[topic]
            if len(votes) < min_votes:
                continue

            # Simple count-based strength
            heatmap[cell_pos] = float(len(votes))

        return heatmap

    def get_all_topics(self) -> set[str]:
        """Get all topics with active votes."""
        topics: set[str] = set()
        for topics_dict in self._grid.values():
            topics.update(topics_dict.keys())
        return topics

    # =========================================================================
    # Step
    # =========================================================================

    def step(self) -> None:
        """Advance step counter. Old votes naturally decay via temporal weight."""
        self._step_count += 1

        # Clean up very old votes (beyond decay horizon)
        max_age = int(1.0 / self._temporal_decay) + 5 if self._temporal_decay > 0 else 1000
        cells_to_clean = []

        for cell_pos, topics in self._grid.items():
            for topic, votes in list(topics.items()):
                votes[:] = [
                    v
                    for v in votes
                    if self._step_count - v.timestamp <= max_age
                ]
                if not votes:
                    del topics[topic]
            if not topics:
                cells_to_clean.append(cell_pos)

        for cell_pos in cells_to_clean:
            del self._grid[cell_pos]

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_statistics(self) -> dict[str, Any]:
        total_active = sum(
            sum(len(votes) for votes in topics.values())
            for topics in self._grid.values()
        )
        return {
            "active_cells": len(self._grid),
            "active_votes": total_active,
            "total_votes_ever": self._total_votes,
            "topics": list(self.get_all_topics()),
            "step_count": self._step_count,
        }

    # =========================================================================
    # Internal
    # =========================================================================

    def _world_to_grid(self, position: tuple[float, float]) -> tuple[int, int]:
        """Convert world coordinates to grid cell."""
        return (
            int(position[0] / self._grid_resolution),
            int(position[1] / self._grid_resolution),
        )

    def _compute_consensus(
        self,
        topic: str,
        position: tuple[float, float],
        radius: float,
        weighted_votes: list[tuple[SpatialVote, float]],
    ) -> SpatialConsensusResult:
        """Compute weighted consensus from votes."""
        # Group by value
        value_weights: dict[Any, float] = defaultdict(float)
        participants: set[int] = set()

        for vote, weight in weighted_votes:
            # Handle hashable and non-hashable values
            key = vote.value
            if isinstance(key, (list, dict)):
                key = str(key)
            value_weights[key] += weight
            participants.add(vote.agent_id)

        # Find winning value
        total_weight = sum(value_weights.values())
        if total_weight == 0:
            return SpatialConsensusResult(
                topic=topic,
                position=position,
                radius=radius,
                votes=len(weighted_votes),
                consensus_value=None,
                consensus_strength=0.0,
                participants=list(participants),
            )

        winner = max(value_weights.items(), key=lambda x: x[1])
        strength = winner[1] / total_weight

        return SpatialConsensusResult(
            topic=topic,
            position=position,
            radius=radius,
            votes=len(weighted_votes),
            consensus_value=winner[0],
            consensus_strength=strength,
            participants=list(participants),
        )
