"""Predictive field - spatial prediction and coordination.

Biological analogy: Many organisms maintain mental models of where
others will be. Flocking birds predict neighbor trajectories to avoid
collisions. Hunting packs predict prey movement to coordinate flanking.
Even slime molds maintain gradient fields that guide future growth.

Mae's predictive field is a 2D grid where agents broadcast their
intentions and predicted states. Other agents read the field to
anticipate movements, detect collision risks, and find coordination
opportunities.

This creates emergent flock-like behavior without explicit communication
- agents coordinate through the shared field, like fish in a school
sensing pressure waves from neighbors.

Connection points:
- Substrate provides agent positions
- Agents write predictions to the field
- Other agents read predictions for coordination
- GNN communicator uses field gradients for routing
- Morphogenesis reads fields to detect activity patterns
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentPrediction:
    """An agent's predicted future state."""

    agent_id: int
    position: tuple[float, float]
    predicted_position: tuple[float, float]  # Where agent expects to be
    velocity: tuple[float, float] = (0.0, 0.0)
    intention: str = ""  # What the agent plans to do
    confidence: float = 0.5  # How confident in prediction
    timestamp: int = 0  # Step when prediction was made
    horizon: int = 5  # Steps into the future


@dataclass
class FieldCell:
    """A cell in the predictive field grid."""

    predictions: list[AgentPrediction] = field(default_factory=list)
    activity_level: float = 0.0  # Aggregate activity in this cell
    gradient: tuple[float, float] = (0.0, 0.0)  # Direction of increasing activity


class PredictiveField:
    """2D grid where agents broadcast predictions and intentions.

    The field enables:
    1. Collision risk detection (overlapping predictions)
    2. Coordination opportunities (complementary intentions)
    3. Activity pattern recognition (hot spots, cold zones)
    4. Gradient-guided movement (toward or away from activity)
    """

    def __init__(
        self,
        field_size: tuple[float, float] = (10.0, 10.0),
        grid_resolution: float = 1.0,
        prediction_horizon: int = 5,
        decay_rate: float = 0.1,
        substrate: Any = None,
    ) -> None:
        self._field_size = field_size
        self._grid_resolution = grid_resolution
        self._prediction_horizon = prediction_horizon
        self._decay_rate = decay_rate
        self._substrate = substrate

        # Grid dimensions
        self._grid_w = max(1, int(field_size[0] / grid_resolution))
        self._grid_h = max(1, int(field_size[1] / grid_resolution))

        # Grid of field cells
        self._grid: dict[tuple[int, int], FieldCell] = {}

        # Agent states
        self._agent_states: dict[int, AgentPrediction] = {}

        # Step tracking
        self._step_count = 0
        self._collision_detections = 0

        logger.info(
            "PredictiveField: %.0fx%.0f (grid %dx%d, resolution=%.1f)",
            field_size[0],
            field_size[1],
            self._grid_w,
            self._grid_h,
            grid_resolution,
        )

    # =========================================================================
    # Agent State Management
    # =========================================================================

    def update_agent_state(
        self,
        agent_id: int,
        position: tuple[float, float],
        velocity: tuple[float, float] = (0.0, 0.0),
        intention: str = "",
        confidence: float = 0.5,
    ) -> None:
        """Update an agent's current state and predicted future position."""
        # Linear prediction: predicted_pos = pos + velocity * horizon
        predicted = (
            position[0] + velocity[0] * self._prediction_horizon,
            position[1] + velocity[1] * self._prediction_horizon,
        )

        prediction = AgentPrediction(
            agent_id=agent_id,
            position=position,
            predicted_position=predicted,
            velocity=velocity,
            intention=intention,
            confidence=confidence,
            timestamp=self._step_count,
            horizon=self._prediction_horizon,
        )

        self._agent_states[agent_id] = prediction

        # Write prediction to grid
        cell_pos = self._world_to_grid(position)
        pred_cell = self._world_to_grid(predicted)

        self._get_cell(cell_pos).predictions.append(prediction)
        if pred_cell != cell_pos:
            self._get_cell(pred_cell).predictions.append(prediction)

    def broadcast_intention(
        self,
        agent_id: int,
        intention: str,
        target_position: Optional[tuple[float, float]] = None,
    ) -> None:
        """Broadcast an intention to the field."""
        state = self._agent_states.get(agent_id)
        if state is None:
            return

        state.intention = intention
        if target_position:
            state.predicted_position = target_position

    # =========================================================================
    # Substrate Integration
    # =========================================================================

    def sync_positions(self) -> None:
        """Read agent positions from substrate and update field states.

        For agents already tracked, updates their positions. For new agents,
        creates initial state entries.
        """
        if self._substrate is None:
            return

        positions = self._substrate.get_all_agent_positions()
        if positions is None:
            return

        for agent_id, position in positions.items():
            existing = self._agent_states.get(agent_id)
            if existing is not None:
                # Compute velocity from position change
                velocity = (
                    position[0] - existing.position[0],
                    position[1] - existing.position[1],
                )
                self.update_agent_state(
                    agent_id, position, velocity=velocity,
                    intention=existing.intention, confidence=existing.confidence,
                )
            else:
                self.update_agent_state(agent_id, position)

    # =========================================================================
    # Prediction and Detection
    # =========================================================================

    def predict_agent_state(
        self, agent_id: int, steps_ahead: int = 1
    ) -> Optional[tuple[float, float]]:
        """Predict where an agent will be in N steps."""
        state = self._agent_states.get(agent_id)
        if state is None:
            return None

        return (
            state.position[0] + state.velocity[0] * steps_ahead,
            state.position[1] + state.velocity[1] * steps_ahead,
        )

    def detect_collision_risk(
        self,
        agent_id: int,
        threshold: float = 2.0,
    ) -> list[dict[str, Any]]:
        """Detect potential collisions with other agents.

        Returns list of agents whose predicted positions are within
        the threshold distance of this agent's predicted position.
        """
        state = self._agent_states.get(agent_id)
        if state is None:
            return []

        risks = []
        for other_id, other_state in self._agent_states.items():
            if other_id == agent_id:
                continue

            dx = state.predicted_position[0] - other_state.predicted_position[0]
            dy = state.predicted_position[1] - other_state.predicted_position[1]
            dist = (dx * dx + dy * dy) ** 0.5

            if dist < threshold:
                risks.append(
                    {
                        "agent_id": other_id,
                        "distance": dist,
                        "predicted_position": other_state.predicted_position,
                        "intention": other_state.intention,
                    }
                )
                self._collision_detections += 1

        return risks

    def find_coordination_opportunities(
        self,
        agent_id: int,
        max_distance: float = 5.0,
    ) -> list[dict[str, Any]]:
        """Find nearby agents with complementary intentions."""
        state = self._agent_states.get(agent_id)
        if state is None:
            return []

        opportunities = []
        for other_id, other_state in self._agent_states.items():
            if other_id == agent_id:
                continue

            dx = state.position[0] - other_state.position[0]
            dy = state.position[1] - other_state.position[1]
            dist = (dx * dx + dy * dy) ** 0.5

            if dist < max_distance and other_state.intention:
                opportunities.append(
                    {
                        "agent_id": other_id,
                        "distance": dist,
                        "intention": other_state.intention,
                        "confidence": other_state.confidence,
                    }
                )

        return opportunities

    # =========================================================================
    # Field Queries
    # =========================================================================

    def calculate_field_gradient(
        self, position: tuple[float, float]
    ) -> tuple[float, float]:
        """Calculate the activity gradient at a position.

        Returns direction (dx, dy) pointing toward higher activity.
        Agents can follow gradients to find active areas or flee them.
        """
        cell_pos = self._world_to_grid(position)
        gx, gy = 0.0, 0.0

        # Sample neighboring cells
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                neighbor = (cell_pos[0] + dx, cell_pos[1] + dy)
                if 0 <= neighbor[0] < self._grid_w and 0 <= neighbor[1] < self._grid_h:
                    cell = self._grid.get(neighbor)
                    if cell:
                        activity = len(cell.predictions)
                        gx += dx * activity
                        gy += dy * activity

        # Normalize
        mag = (gx * gx + gy * gy) ** 0.5
        if mag > 0:
            gx /= mag
            gy /= mag

        return (gx, gy)

    def get_activity_at(self, position: tuple[float, float]) -> float:
        """Get activity level at a world position."""
        cell_pos = self._world_to_grid(position)
        cell = self._grid.get(cell_pos)
        return float(len(cell.predictions)) if cell else 0.0

    def get_activity_heatmap(self) -> dict[tuple[int, int], float]:
        """Get activity levels for all grid cells."""
        heatmap = {}
        for pos, cell in self._grid.items():
            heatmap[pos] = float(len(cell.predictions))
        return heatmap

    # =========================================================================
    # Step
    # =========================================================================

    def step(self) -> None:
        """Advance the field by one step. Decay old predictions."""
        self._step_count += 1

        # Auto-sync positions from substrate if available
        if self._substrate is not None:
            self.sync_positions()

        # Decay old predictions
        for cell in self._grid.values():
            cell.predictions = [
                p
                for p in cell.predictions
                if self._step_count - p.timestamp <= p.horizon
            ]
            cell.activity_level = float(len(cell.predictions))

        # Remove empty cells to save memory
        empty_cells = [pos for pos, cell in self._grid.items() if not cell.predictions]
        for pos in empty_cells:
            del self._grid[pos]

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_statistics(self) -> dict[str, Any]:
        total_predictions = sum(len(c.predictions) for c in self._grid.values())
        return {
            "agents_tracked": len(self._agent_states),
            "active_cells": len(self._grid),
            "total_predictions": total_predictions,
            "collision_detections": self._collision_detections,
            "step_count": self._step_count,
            "grid_size": (self._grid_w, self._grid_h),
        }

    # =========================================================================
    # Internal
    # =========================================================================

    def _world_to_grid(self, position: tuple[float, float]) -> tuple[int, int]:
        """Convert world coordinates to grid cell coordinates."""
        gx = max(0, min(self._grid_w - 1, int(position[0] / self._grid_resolution)))
        gy = max(0, min(self._grid_h - 1, int(position[1] / self._grid_resolution)))
        return (gx, gy)

    def _get_cell(self, grid_pos: tuple[int, int]) -> FieldCell:
        """Get or create a field cell."""
        if grid_pos not in self._grid:
            self._grid[grid_pos] = FieldCell()
        return self._grid[grid_pos]
