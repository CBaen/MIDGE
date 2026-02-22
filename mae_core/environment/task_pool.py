"""TaskPool - Mae's action environment.

Biological analogy: The environment provides problems to solve.
Tasks are like food sources, threats to handle, or territories to explore.
Agents claim tasks, work on them, earn rewards, and share solutions.

This is v1 of Mae's body -- the bridge from decision to consequence.
"""

from __future__ import annotations

import logging
import random
import uuid
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Task types and their biological analogies
TASK_TYPES = ["forage", "defend", "explore", "share", "api_call"]


@dataclass
class Task:
    """A unit of work in the task pool.

    Biological analogy: A stimulus in the environment requiring a response.
    forage = food source, defend = threat, explore = unknown territory,
    share = social opportunity.
    """

    task_id: str
    task_type: str              # "forage", "defend", "explore", "share"
    difficulty: float           # 0.0-1.0
    reward_value: float         # base reward if completed
    deadline: int               # steps remaining before expiry
    state: str = "available"    # "available", "claimed", "completed", "expired"
    claimed_by: Optional[str] = None  # agent unique_id
    progress: float = 0.0      # 0.0-1.0 (1.0 = complete)
    created_step: int = 0


class TaskPool:
    """Mae's action environment -- abstract problem space for agent tasks.

    Biological analogy: The environment provides problems to solve.
    Tasks are like food sources, threats to handle, or territories to explore.
    Agents claim and complete tasks to earn rewards, driving the ACT step
    from stub to functional.

    Each step, the pool generates new tasks and expires old ones.
    Agents interact through claim/work/abandon/broadcast operations.
    """

    def __init__(
        self,
        generation_rate: int = 3,
        difficulty_range: tuple[float, float] = (0.1, 0.9),
    ) -> None:
        self.generation_rate = generation_rate
        self.difficulty_range = difficulty_range

        # Active tasks indexed by task_id
        self._tasks: dict[str, Task] = {}

        # Per-agent statistics
        self._agent_stats: dict[str, dict] = {}

        # Pool-level counters
        self._total_generated: int = 0
        self._total_completed: int = 0
        self._total_expired: int = 0
        self._total_abandoned: int = 0
        self._current_step: int = 0

    def step(self, current_step: int) -> None:
        """Advance the task pool: generate new tasks, expire old ones.

        Biological analogy: The environment changes each moment --
        food appears, threats emerge, opportunities vanish.
        """
        self._current_step = current_step

        # Expire tasks whose deadline has run out
        expired_ids = []
        for task_id, task in self._tasks.items():
            if task.state in ("available", "claimed"):
                task.deadline -= 1
                if task.deadline <= 0:
                    expired_ids.append(task_id)

        for task_id in expired_ids:
            task = self._tasks[task_id]
            # Penalize the agent who claimed but didn't finish
            if task.state == "claimed" and task.claimed_by is not None:
                self._update_agent_stat(task.claimed_by, "penalties", 1)
                self._update_agent_stat(task.claimed_by, "total_reward", -0.1)
            task.state = "expired"
            self._total_expired += 1

        # Remove expired and completed tasks older than 10 steps
        # (keep recent ones for stats/inspection)
        stale_ids = [
            tid for tid, t in self._tasks.items()
            if t.state in ("expired", "completed")
            and (current_step - t.created_step) > 10
        ]
        for tid in stale_ids:
            del self._tasks[tid]

        # Generate new tasks
        for _ in range(self.generation_rate):
            self._generate_task(current_step)

    def _generate_task(self, current_step: int) -> Task:
        """Generate a single random task.

        Task economics:
        - Harder tasks pay more (reward = difficulty * 2.0 + 0.5)
        - Harder tasks have shorter deadlines (urgency pressure)
        - Type distribution is uniform across forage/defend/explore/share
        """
        task_type = random.choice(TASK_TYPES)
        difficulty = random.uniform(*self.difficulty_range)
        reward_value = difficulty * 2.0 + 0.5
        deadline = max(5, int(20 * (1.0 - difficulty)))

        task = Task(
            task_id=str(uuid.uuid4())[:8],
            task_type=task_type,
            difficulty=difficulty,
            reward_value=reward_value,
            deadline=deadline,
            state="available",
            created_step=current_step,
        )
        self._tasks[task.task_id] = task
        self._total_generated += 1
        return task

    def get_available_tasks(
        self, task_type: Optional[str] = None,
    ) -> list[Task]:
        """List unclaimed tasks, optionally filtered by type.

        Biological analogy: Scanning the environment for opportunities.
        """
        tasks = [
            t for t in self._tasks.values()
            if t.state == "available"
        ]
        if task_type is not None:
            tasks = [t for t in tasks if t.task_type == task_type]
        return tasks

    def claim_task(self, task_id: str, agent_id: str) -> Optional[Task]:
        """Agent claims a task. Returns the task if successful, None otherwise.

        Biological analogy: Committing to pursue a food source or threat.
        """
        task = self._tasks.get(task_id)
        if task is None or task.state != "available":
            return None

        task.state = "claimed"
        task.claimed_by = str(agent_id)
        self._ensure_agent_stats(str(agent_id))
        self._update_agent_stat(str(agent_id), "tasks_claimed", 1)
        return task

    def work_on_task(
        self, task_id: str, agent_id: str, effort: float,
    ) -> tuple[float, bool, float]:
        """Advance progress on a claimed task.

        Args:
            task_id: The task to work on
            agent_id: The agent doing the work
            effort: How much effort to apply (0.0-1.0)

        Returns:
            (progress_delta, completed, reward)
            - progress_delta: how much progress was made
            - completed: whether the task is now complete
            - reward: reward earned (0 if not completed, full reward if completed)

        Biological analogy: Motor cortex sends command, muscles execute,
        proprioceptive feedback returns.
        """
        task = self._tasks.get(task_id)
        if task is None or task.state != "claimed" or task.claimed_by != str(agent_id):
            return (0.0, False, 0.0)

        # Progress = effort scaled by inverse difficulty
        # (easy tasks progress faster per unit effort)
        progress_delta = effort * (1.0 - task.difficulty)
        task.progress = min(1.0, task.progress + progress_delta)

        if task.progress >= 1.0:
            # Task completed
            task.state = "completed"
            reward = task.reward_value
            self._total_completed += 1
            self._update_agent_stat(str(agent_id), "tasks_completed", 1)
            self._update_agent_stat(str(agent_id), "total_reward", reward)
            return (progress_delta, True, reward)

        return (progress_delta, False, 0.0)

    def abandon_task(self, task_id: str, agent_id: str) -> float:
        """Agent abandons a claimed task. Returns small negative reward.

        Biological analogy: Giving up on a pursuit -- metabolic cost
        with no payoff.
        """
        task = self._tasks.get(task_id)
        if task is None or task.state != "claimed" or task.claimed_by != str(agent_id):
            return 0.0

        task.state = "available"
        task.claimed_by = None
        # Don't reset progress -- another agent can pick up where
        # this one left off (biological: partially dug burrow)
        self._total_abandoned += 1
        self._update_agent_stat(str(agent_id), "tasks_abandoned", 1)
        penalty = -0.1
        self._update_agent_stat(str(agent_id), "total_reward", penalty)
        return penalty

    def broadcast_solution(self, task_id: str, agent_id: str) -> float:
        """Share a completed task's solution with others.

        Returns a small social reward for contributing to collective
        knowledge.

        Biological analogy: Teaching behavior -- sharing foraging
        locations, alarm calls about threats.
        """
        task = self._tasks.get(task_id)
        if task is None or task.state != "completed" or task.claimed_by != str(agent_id):
            return 0.0

        social_reward = 0.2
        self._update_agent_stat(str(agent_id), "solutions_shared", 1)
        self._update_agent_stat(str(agent_id), "total_reward", social_reward)
        return social_reward

    def get_agent_stats(self, agent_id: str) -> dict:
        """Get performance statistics for a specific agent.

        Biological analogy: Fitness assessment -- how well is this
        organism performing in its niche?
        """
        self._ensure_agent_stats(str(agent_id))
        stats = dict(self._agent_stats[str(agent_id)])
        # Compute success rate
        claimed = stats.get("tasks_claimed", 0)
        completed = stats.get("tasks_completed", 0)
        stats["success_rate"] = completed / max(1, claimed)
        return stats

    def get_agent_current_task(self, agent_id: str) -> Optional[Task]:
        """Get the task currently claimed by an agent, if any."""
        agent_id_str = str(agent_id)
        for task in self._tasks.values():
            if task.state == "claimed" and task.claimed_by == agent_id_str:
                return task
        return None

    def get_statistics(self) -> dict:
        """Pool-level statistics for observability.

        Biological analogy: Ecosystem health metrics -- resource
        availability, competition pressure, turnover rate.
        """
        available = sum(1 for t in self._tasks.values() if t.state == "available")
        claimed = sum(1 for t in self._tasks.values() if t.state == "claimed")
        completed = sum(1 for t in self._tasks.values() if t.state == "completed")

        return {
            "current_step": self._current_step,
            "active_tasks": len(self._tasks),
            "available": available,
            "claimed": claimed,
            "completed_recent": completed,
            "total_generated": self._total_generated,
            "total_completed": self._total_completed,
            "total_expired": self._total_expired,
            "total_abandoned": self._total_abandoned,
            "generation_rate": self.generation_rate,
            "agents_active": len(self._agent_stats),
        }

    def _ensure_agent_stats(self, agent_id: str) -> None:
        """Initialize agent stats if not present."""
        if agent_id not in self._agent_stats:
            self._agent_stats[agent_id] = {
                "tasks_claimed": 0,
                "tasks_completed": 0,
                "tasks_abandoned": 0,
                "solutions_shared": 0,
                "penalties": 0,
                "total_reward": 0.0,
            }

    def _update_agent_stat(self, agent_id: str, key: str, delta: float) -> None:
        """Increment an agent stat by delta."""
        self._ensure_agent_stats(agent_id)
        self._agent_stats[agent_id][key] = (
            self._agent_stats[agent_id].get(key, 0) + delta
        )

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"TaskPool(step={stats['current_step']}, "
            f"available={stats['available']}, "
            f"claimed={stats['claimed']}, "
            f"generated={stats['total_generated']}, "
            f"completed={stats['total_completed']}, "
            f"expired={stats['total_expired']})"
        )
