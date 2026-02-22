"""Goal Manager — Persistent goal stack with impasse detection.

Biological analogy: The prefrontal cortex maintains persistent goals
across time (working memory), monitors progress toward those goals,
and detects impasses requiring strategy shifts. The anterior cingulate
cortex (ACC) detects conflict/impasse and triggers cognitive control
adjustments.

References:
- Miller & Cohen (2001): Integrative theory of prefrontal cortex function
- Botvinick et al. (2001): Conflict monitoring and cognitive control
- Anderson (2007): ACT-R cognitive architecture (goal buffer)
- Laird (2012): SOAR cognitive architecture (impasse detection)
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class GoalStatus(Enum):
    """Goal lifecycle states."""
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    SUSPENDED = "suspended"


@dataclass
class GoalFrame:
    """A single goal on the stack.

    Biological analogy: Each goal is like a persistent activation pattern
    in prefrontal cortex working memory — it stays active until achieved,
    abandoned, or replaced by a more urgent goal.
    """
    goal_id: str
    description: str
    priority: float = 0.5           # 0-1, how urgent
    status: GoalStatus = GoalStatus.ACTIVE
    progress: float = 0.0           # 0-1, how complete
    created_step: int = 0
    last_progress_step: int = 0     # When progress last changed
    parent_goal_id: str | None = None  # If this is a subgoal
    reward_baseline: float = 0.0    # Reward when goal was set
    reward_accumulated: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class GoalManager:
    """Persistent goal stack with impasse detection.

    Manages a stack of goals (most recent = most focused).
    Tracks progress via reward accumulation.
    Detects impasses when no progress for N steps.
    Creates subgoals to break through impasses.
    Provides goal_priority to modulate decision-making.

    Architecture:
    - Goal stack: LIFO structure (push new goals, pop completed)
    - Impasse threshold: 5 steps with zero progress (configurable)
    - Subgoal strategy: When impassed, push exploratory subgoal
    - Goal priority: Weighted by urgency, time pressure, and depth
    """

    def __init__(
        self,
        event_bus=None,
        impasse_threshold: int = 5,
        max_stack_depth: int = 5,
        progress_epsilon: float = 0.01,
    ):
        self._bus = event_bus
        self._impasse_threshold = impasse_threshold
        self._max_stack_depth = max_stack_depth
        self._progress_epsilon = progress_epsilon

        # Goal stack (index 0 = bottom, index -1 = top/current)
        self._stack: list[GoalFrame] = []

        # History for analysis
        self._completed_goals: deque[GoalFrame] = deque(maxlen=100)
        self._failed_goals: deque[GoalFrame] = deque(maxlen=50)

        # Statistics
        self._total_goals_created = 0
        self._total_impasses = 0
        self._total_subgoals = 0

        # Push default exploration goal
        self.push_goal(GoalFrame(
            goal_id="default_explore",
            description="Explore environment and accumulate knowledge",
            priority=0.3,
            status=GoalStatus.ACTIVE,
        ))

    @property
    def current_goal(self) -> GoalFrame | None:
        """The top-of-stack goal (most focused)."""
        for goal in reversed(self._stack):
            if goal.status == GoalStatus.ACTIVE:
                return goal
        return None

    @property
    def goal_priority(self) -> float:
        """Current goal priority for decision modulation."""
        goal = self.current_goal
        if goal is None:
            return 0.3  # Baseline exploration priority
        # Priority increases with time pressure (longer stuck = more urgent)
        time_pressure = 0.0
        if goal.last_progress_step > 0:
            steps_stuck = 0  # Will be computed by caller
            time_pressure = min(0.2, steps_stuck * 0.02)
        return min(1.0, goal.priority + time_pressure)

    @property
    def stack_depth(self) -> int:
        """Number of active goals."""
        return sum(1 for g in self._stack if g.status == GoalStatus.ACTIVE)

    def push_goal(self, goal: GoalFrame) -> bool:
        """Push a new goal onto the stack.

        Returns True if pushed, False if stack is full.
        """
        if self.stack_depth >= self._max_stack_depth:
            logger.warning("GoalManager: stack full (%d), cannot push %s",
                          self._max_stack_depth, goal.goal_id)
            return False

        self._stack.append(goal)
        self._total_goals_created += 1

        if self._bus is not None:
            try:
                self._bus.publish("cognition.goal_update", {
                    "action": "push",
                    "goal_id": goal.goal_id,
                    "description": goal.description,
                    "priority": goal.priority,
                    "stack_depth": self.stack_depth,
                })
            except Exception:
                pass

        return True

    def pop_goal(self, status: GoalStatus = GoalStatus.COMPLETED) -> GoalFrame | None:
        """Pop the top active goal from the stack.

        Returns the popped goal or None if stack is empty.
        """
        for i in range(len(self._stack) - 1, -1, -1):
            if self._stack[i].status == GoalStatus.ACTIVE:
                goal = self._stack.pop(i)
                goal.status = status

                if status == GoalStatus.COMPLETED:
                    self._completed_goals.append(goal)
                elif status == GoalStatus.FAILED:
                    self._failed_goals.append(goal)

                if self._bus is not None:
                    try:
                        self._bus.publish("cognition.goal_update", {
                            "action": "pop",
                            "goal_id": goal.goal_id,
                            "status": status.value,
                            "progress": goal.progress,
                            "stack_depth": self.stack_depth,
                        })
                    except Exception:
                        pass

                return goal
        return None

    def update_progress(
        self,
        reward: float,
        step: int,
        agent_id: str = "",
    ) -> dict[str, Any]:
        """Update current goal progress based on reward.

        Returns dict with:
        - impasse_detected: bool
        - goal_priority: float
        - current_goal: str or None
        """
        goal = self.current_goal
        if goal is None:
            return {
                "impasse_detected": False,
                "goal_priority": 0.3,
                "current_goal": None,
            }

        # Update reward accumulation
        goal.reward_accumulated += reward

        # Check if progress was made (reward above baseline)
        progress_delta = reward - goal.reward_baseline
        if progress_delta > self._progress_epsilon:
            goal.progress = min(1.0, goal.progress + abs(progress_delta) * 0.1)
            goal.last_progress_step = step

        # Check for impasse
        steps_since_progress = step - goal.last_progress_step if goal.last_progress_step > 0 else 0
        impasse_detected = steps_since_progress >= self._impasse_threshold

        if impasse_detected:
            self._total_impasses += 1
            logger.info(
                "GoalManager: Impasse detected for '%s' (%d steps no progress)",
                goal.goal_id, steps_since_progress,
            )

            # Create exploratory subgoal to break through
            self._create_subgoal(goal, step)

            if self._bus is not None:
                try:
                    self._bus.publish("cognition.impasse_detected", {
                        "agent_id": agent_id,
                        "goal_id": goal.goal_id,
                        "steps_stuck": steps_since_progress,
                        "progress": goal.progress,
                        "step": step,
                    })
                except Exception:
                    pass

        # Check for goal completion
        if goal.progress >= 0.95:
            self.pop_goal(GoalStatus.COMPLETED)
            logger.info("GoalManager: Goal '%s' completed (progress=%.2f)",
                       goal.goal_id, goal.progress)

        return {
            "impasse_detected": impasse_detected,
            "goal_priority": self.goal_priority,
            "current_goal": goal.goal_id if goal else None,
            "goal_description": goal.description if goal else "",
            "steps_since_progress": steps_since_progress,
        }

    def _create_subgoal(self, parent: GoalFrame, step: int) -> None:
        """Create an exploratory subgoal when impassed.

        Biological analogy: When the ACC detects conflict/impasse,
        it triggers a shift in strategy — typically toward more
        exploratory behavior to find new paths forward.

        Guard: If the parent is already an explore subgoal, pop it
        instead of nesting deeper. Prevents explore_explore_explore_...
        infinite recursion that fills the goal stack with garbage.
        """
        # Guard: don't nest explore-of-explore — pop the stale subgoal instead
        if parent.parent_goal_id is not None and parent.goal_id.startswith("explore_"):
            self.pop_goal(GoalStatus.FAILED)
            logger.info("GoalManager: Popped stale explore subgoal '%s' (no nesting)",
                       parent.goal_id)
            return

        subgoal = GoalFrame(
            goal_id=f"explore_{parent.goal_id}_{step}",
            description=f"Explore alternatives for '{parent.description}'",
            priority=min(1.0, parent.priority + 0.1),
            status=GoalStatus.ACTIVE,
            created_step=step,
            last_progress_step=step,
            parent_goal_id=parent.goal_id,
            reward_baseline=0.0,
        )

        if self.push_goal(subgoal):
            self._total_subgoals += 1
            logger.info("GoalManager: Subgoal '%s' created for impassed '%s'",
                       subgoal.goal_id, parent.goal_id)

    def get_context_for_decision(self) -> dict[str, Any]:
        """Build context dict for DecisionRouter integration.

        Returns values that _route_with_advisory can merge into context.
        """
        goal = self.current_goal
        if goal is None:
            return {"goal_priority": 0.3, "has_goal": False}

        return {
            "goal_priority": self.goal_priority,
            "has_goal": True,
            "goal_description": goal.description,
            "goal_progress": goal.progress,
            "goal_id": goal.goal_id,
            "is_subgoal": goal.parent_goal_id is not None,
        }

    def get_statistics(self) -> dict[str, Any]:
        """Return goal management statistics."""
        return {
            "total_goals_created": self._total_goals_created,
            "total_impasses": self._total_impasses,
            "total_subgoals": self._total_subgoals,
            "stack_depth": self.stack_depth,
            "completed_goals": len(self._completed_goals),
            "failed_goals": len(self._failed_goals),
            "current_goal": self.current_goal.goal_id if self.current_goal else None,
        }
