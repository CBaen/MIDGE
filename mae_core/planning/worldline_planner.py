"""Worldline Planner - Plan trajectories through spacetime.

Biological analogy: Prospective memory + predictive coding.
Animals don't just react - they plan routes, time actions,
and imagine future states. A cheetah doesn't chase where the
gazelle IS, it predicts where it WILL BE.

A "worldline" is a trajectory through spacetime: a sequence of
(state, action, time) points that an entity will traverse.
The WorldlinePlanner uses the WorldModel to project future
states, then evaluates and selects optimal worldlines.

Connection points:
- WorldModel provides state predictions (rollout)
- CollectiveDreamPlanner validates worldlines via consensus
- TemporalMemory provides historical patterns for prediction
- CausalEngine identifies which actions cause which outcomes
- EventBus publishes planning lifecycle events
- DecisionRouter uses worldline evaluations for decisions
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# EventBus channels
CH_WORLDLINE_PLANNED = "planning.worldline_planned"
CH_WORLDLINE_SELECTED = "planning.worldline_selected"
CH_WORLDLINE_VALIDATED = "planning.worldline_validated"


class WorldlineStatus(Enum):
    PROJECTED = "projected"  # Imagined but not validated
    VALIDATED = "validated"  # Confirmed by consensus or world model
    EXECUTING = "executing"  # Agent is following this worldline
    COMPLETED = "completed"  # Successfully followed
    ABANDONED = "abandoned"  # Diverged from reality


@dataclass
class WorldlinePoint:
    """A single point in a worldline trajectory.

    Like a frame in a film strip - one moment in the
    entity's journey through spacetime.
    """

    state: Any
    action: Any
    predicted_reward: float = 0.0
    predicted_next_state: Any = None
    timestamp: float = 0.0
    position: tuple[float, ...] = (0.0, 0.0)
    uncertainty: float = 0.0  # WorldModel prediction uncertainty


@dataclass
class Worldline:
    """A trajectory through spacetime - a planned future path.

    A sequence of WorldlinePoints representing an entity's
    projected journey. Like plotting a course on a map, but
    the map includes time as a dimension.
    """

    worldline_id: str
    entity_id: str
    points: list[WorldlinePoint] = field(default_factory=list)
    status: WorldlineStatus = WorldlineStatus.PROJECTED
    total_reward: float = 0.0
    avg_uncertainty: float = 0.0
    created_at: float = field(default_factory=time.time)
    horizon: int = 0  # Number of steps into the future

    # Validation
    consensus_score: float = 0.0  # How many agents agree this is a good plan
    actual_divergence: float = 0.0  # How far reality diverged from plan

    def __post_init__(self) -> None:
        if self.points and not self.total_reward:
            self.total_reward = sum(p.predicted_reward for p in self.points)
        if self.points and not self.avg_uncertainty:
            uncertainties = [p.uncertainty for p in self.points]
            self.avg_uncertainty = sum(uncertainties) / len(uncertainties) if uncertainties else 0.0
        if self.points and not self.horizon:
            self.horizon = len(self.points)


@dataclass
class PlanningResult:
    """Result of worldline planning - the selected path forward."""

    selected_worldline: Worldline | None
    alternatives: list[Worldline] = field(default_factory=list)
    planning_time: float = 0.0
    confidence: float = 0.0  # [0, 1]
    reason: str = ""


class WorldlinePlanner:
    """Plans trajectories through spacetime using WorldModel predictions.

    Three planning horizons (like biological time perception):
    - Reactive (1-3 steps): Reflexive, high confidence
    - Tactical (4-10 steps): Short-term planning, medium confidence
    - Strategic (11+ steps): Long-term vision, low confidence

    Uses branch-and-evaluate: generates multiple possible worldlines,
    scores them, and selects the best. Like a chess player thinking
    several moves ahead, but across space AND time.
    """

    def __init__(
        self,
        world_model: Any = None,
        temporal_memory: Any = None,
        causal_engine: Any = None,
        event_bus: Any = None,
        reactive_horizon: int = 3,
        tactical_horizon: int = 10,
        strategic_horizon: int = 25,
        num_branches: int = 5,
        uncertainty_discount: float = 0.9,
    ) -> None:
        self._world_model = world_model
        self._temporal = temporal_memory
        self._causal = causal_engine
        self._bus = event_bus

        self._reactive_horizon = reactive_horizon
        self._tactical_horizon = tactical_horizon
        self._strategic_horizon = strategic_horizon
        self._num_branches = num_branches
        self._uncertainty_discount = uncertainty_discount

        # Active worldlines being executed
        self._active_worldlines: dict[str, Worldline] = {}

        # Planning history
        self._planning_history: list[PlanningResult] = []
        self._max_history = 100

        # Accuracy tracking per entity
        self._accuracy: dict[str, list[float]] = defaultdict(list)

        # Pattern-triggered replanning state
        self._replan_pending = False
        self._last_pattern_info: dict[str, Any] = {}

        # Statistics
        self._total_plans = 0
        self._successful_plans = 0

        # Subscribe to temporal pattern events for replanning
        if self._bus is not None:
            self._bus.register_callback(
                "temporal.pattern_detected",
                self._on_pattern_detected,
            )

        logger.info(
            "WorldlinePlanner initialized (horizons: %d/%d/%d, branches: %d)",
            reactive_horizon,
            tactical_horizon,
            strategic_horizon,
            num_branches,
        )

    # =========================================================================
    # Worldline Generation
    # =========================================================================

    def plan(
        self,
        entity_id: str,
        current_state: Any,
        available_actions: list[Any],
        horizon: int | None = None,
        position: tuple[float, ...] = (0.0, 0.0),
    ) -> PlanningResult:
        """Generate and evaluate worldlines for an entity.

        Branches from current state, projects each through the
        WorldModel, scores by cumulative discounted reward, and
        selects the best trajectory.
        """
        start_time = time.time()
        horizon = horizon or self._tactical_horizon

        if not available_actions:
            return PlanningResult(
                selected_worldline=None,
                planning_time=time.time() - start_time,
                reason="no_actions_available",
            )

        # Generate branching worldlines
        worldlines = []
        for branch_idx in range(min(self._num_branches, len(available_actions))):
            # Each branch starts with a different first action
            first_action = available_actions[branch_idx % len(available_actions)]
            worldline = self._project_worldline(
                entity_id=entity_id,
                start_state=current_state,
                first_action=first_action,
                horizon=horizon,
                position=position,
                branch_id=branch_idx,
            )
            worldlines.append(worldline)

        if not worldlines:
            return PlanningResult(
                selected_worldline=None,
                planning_time=time.time() - start_time,
                reason="projection_failed",
            )

        # Score worldlines
        scored = self._score_worldlines(worldlines)

        # Select best
        best = scored[0] if scored else worldlines[0]
        alternatives = scored[1:] if len(scored) > 1 else []

        # Compute confidence based on uncertainty
        confidence = max(0.0, 1.0 - best.avg_uncertainty)

        result = PlanningResult(
            selected_worldline=best,
            alternatives=alternatives,
            planning_time=time.time() - start_time,
            confidence=confidence,
            reason="best_scoring_worldline",
        )

        # Track
        self._total_plans += 1
        self._planning_history.append(result)
        if len(self._planning_history) > self._max_history:
            self._planning_history.pop(0)

        # Publish
        if self._bus and best:
            self._bus.publish(CH_WORLDLINE_PLANNED, {
                "entity_id": entity_id,
                "worldline_id": best.worldline_id,
                "horizon": horizon,
                "total_reward": best.total_reward,
                "confidence": confidence,
                "alternatives": len(alternatives),
            })

        return result

    def _project_worldline(
        self,
        entity_id: str,
        start_state: Any,
        first_action: Any,
        horizon: int,
        position: tuple[float, ...],
        branch_id: int,
    ) -> Worldline:
        """Project a single worldline through the WorldModel."""
        wl_id = f"wl-{entity_id}-{int(time.time())}-b{branch_id}"
        points = []
        current_state = start_state
        current_time = time.time()

        for step in range(horizon):
            action = first_action if step == 0 else self._select_action(current_state, step)

            # Use WorldModel if available
            predicted_reward = 0.0
            next_state = current_state
            uncertainty = 0.5

            if self._world_model:
                prediction = self._world_model.step(current_state, action)
                next_state = prediction.next_state
                predicted_reward = prediction.reward
                uncertainty = prediction.uncertainty

            # Discount reward by step (future rewards worth less)
            discounted_reward = predicted_reward * (self._uncertainty_discount ** step)

            point = WorldlinePoint(
                state=current_state,
                action=action,
                predicted_reward=discounted_reward,
                predicted_next_state=next_state,
                timestamp=current_time + step,
                position=position,
                uncertainty=uncertainty,
            )
            points.append(point)
            current_state = next_state

        return Worldline(
            worldline_id=wl_id,
            entity_id=entity_id,
            points=points,
        )

    def _select_action(self, state: Any, step: int) -> Any:
        """Select an action for intermediate worldline steps.

        Uses temporal pattern prediction if available, otherwise
        returns a default action index.
        """
        if self._temporal:
            # Use temporal patterns to predict likely next action
            predicted_type = self._temporal.predict_next_event_type(str(state))
            if predicted_type is not None:
                return predicted_type.value

        return step % 4  # Cycle through action indices as fallback

    def _score_worldlines(
        self, worldlines: list[Worldline]
    ) -> list[Worldline]:
        """Score and rank worldlines by quality.

        Scoring considers:
        1. Total discounted reward (primary)
        2. Uncertainty penalty (lower uncertainty = better)
        3. Temporal pattern alignment (bonus for matching patterns)
        """
        for wl in worldlines:
            # Base score: total reward
            score = wl.total_reward

            # Uncertainty penalty
            score -= wl.avg_uncertainty * 0.5

            # Store score in total_reward for sorting
            wl.total_reward = score

        worldlines.sort(key=lambda w: w.total_reward, reverse=True)
        return worldlines

    # =========================================================================
    # Worldline Execution Tracking
    # =========================================================================

    def begin_execution(self, worldline: Worldline) -> None:
        """Mark a worldline as actively being executed."""
        worldline.status = WorldlineStatus.EXECUTING
        self._active_worldlines[worldline.worldline_id] = worldline

    def check_divergence(
        self,
        worldline_id: str,
        actual_state: Any,
        step: int,
    ) -> float:
        """Check how much reality diverged from the planned worldline.

        Returns divergence score [0, 1]. High divergence suggests
        the worldline should be abandoned and replanned.
        """
        worldline = self._active_worldlines.get(worldline_id)
        if not worldline or step >= len(worldline.points):
            return 1.0

        predicted = worldline.points[step].predicted_next_state
        if predicted is None:
            return 0.5

        # Compute divergence
        try:
            pred_arr = np.asarray(predicted, dtype=float).flatten()
            actual_arr = np.asarray(actual_state, dtype=float).flatten()
            min_len = min(len(pred_arr), len(actual_arr))
            if min_len == 0:
                return 0.5
            divergence = float(np.mean(np.abs(pred_arr[:min_len] - actual_arr[:min_len])))
            normalized = min(1.0, divergence)
        except (ValueError, TypeError):
            normalized = 0.5

        worldline.actual_divergence = max(worldline.actual_divergence, normalized)
        return normalized

    def complete_worldline(
        self, worldline_id: str, success: bool = True
    ) -> None:
        """Mark a worldline as completed or abandoned."""
        worldline = self._active_worldlines.pop(worldline_id, None)
        if not worldline:
            return

        worldline.status = (
            WorldlineStatus.COMPLETED if success else WorldlineStatus.ABANDONED
        )

        if success:
            self._successful_plans += 1
            # Track accuracy
            accuracy = max(0.0, 1.0 - worldline.actual_divergence)
            self._accuracy[worldline.entity_id].append(accuracy)
            # Keep only recent accuracy history
            if len(self._accuracy[worldline.entity_id]) > 50:
                self._accuracy[worldline.entity_id] = self._accuracy[
                    worldline.entity_id
                ][-50:]

    def abandon_worldline(self, worldline_id: str) -> None:
        """Abandon a worldline that diverged too far from reality."""
        self.complete_worldline(worldline_id, success=False)

    # =========================================================================
    # Multi-Horizon Planning
    # =========================================================================

    def plan_multi_horizon(
        self,
        entity_id: str,
        current_state: Any,
        available_actions: list[Any],
        position: tuple[float, ...] = (0.0, 0.0),
    ) -> dict[str, PlanningResult]:
        """Plan at all three horizons simultaneously.

        Returns reactive, tactical, and strategic plans. The agent
        can use the reactive plan for immediate action while the
        strategic plan shapes long-term behavior.
        """
        results = {}

        results["reactive"] = self.plan(
            entity_id, current_state, available_actions,
            horizon=self._reactive_horizon, position=position,
        )
        results["tactical"] = self.plan(
            entity_id, current_state, available_actions,
            horizon=self._tactical_horizon, position=position,
        )
        results["strategic"] = self.plan(
            entity_id, current_state, available_actions,
            horizon=self._strategic_horizon, position=position,
        )

        return results

    # =========================================================================
    # Temporal Pattern Integration
    # =========================================================================

    def get_temporal_context(self, entity_id: str) -> dict[str, Any]:
        """Get temporal context for decision-making.

        Combines planning history, temporal patterns, and accuracy
        into a context dict that DecisionRouter can use.
        """
        accuracy_history = self._accuracy.get(entity_id, [])
        avg_accuracy = (
            sum(accuracy_history) / len(accuracy_history)
            if accuracy_history
            else 0.5
        )

        active = self._active_worldlines.get(
            next(
                (
                    wid
                    for wid, wl in self._active_worldlines.items()
                    if wl.entity_id == entity_id
                ),
                "",
            )
        )

        return {
            "planning_accuracy": avg_accuracy,
            "total_plans": self._total_plans,
            "active_worldline": active.worldline_id if active else None,
            "active_divergence": active.actual_divergence if active else 0.0,
            "active_progress": (
                f"{active.status.value}" if active else "no_active_plan"
            ),
        }

    # =========================================================================
    # EventBus Callbacks
    # =========================================================================

    def _on_pattern_detected(self, channel: str, message: Any) -> None:
        """Handle temporal.pattern_detected - flag replanning needed."""
        try:
            import json
            data = json.loads(message) if isinstance(message, str) else message
            self._replan_pending = True
            self._last_pattern_info = data
            logger.info(
                "Pattern detected (id=%s, confidence=%.2f), replanning flagged",
                data.get("pattern_id", "?"),
                data.get("confidence", 0.0),
            )
        except Exception:
            logger.debug("Failed to process pattern_detected event")

    @property
    def replan_pending(self) -> bool:
        """Whether a new temporal pattern has been detected requiring replanning."""
        return self._replan_pending

    def acknowledge_replan(self) -> dict[str, Any]:
        """Clear replan flag and return the pattern info that triggered it."""
        info = self._last_pattern_info.copy()
        self._replan_pending = False
        self._last_pattern_info = {}
        return info

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_statistics(self) -> dict[str, Any]:
        """Get planning statistics."""
        return {
            "total_plans": self._total_plans,
            "successful_plans": self._successful_plans,
            "active_worldlines": len(self._active_worldlines),
            "success_rate": (
                self._successful_plans / max(self._total_plans, 1)
            ),
            "horizons": {
                "reactive": self._reactive_horizon,
                "tactical": self._tactical_horizon,
                "strategic": self._strategic_horizon,
            },
            "branches_per_plan": self._num_branches,
            "entity_count": len(self._accuracy),
            "replan_pending": self._replan_pending,
        }
