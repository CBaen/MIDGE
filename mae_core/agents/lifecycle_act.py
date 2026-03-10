"""Act lifecycle mixin for MycelialAgent.

Contains _act, _act_explore, _act_exploit, _act_communicate, _act_rest,
_act_api_call. Extracted from lifecycle_decision.py to stay under 500-line
limit.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ActMixin:
    """Action execution lifecycle methods for MycelialAgent."""

    def _act(self, action: Any) -> float:
        """Execute action in the task pool environment.

        Biological analogy: Motor cortex sends command to muscles.
        The environment provides proprioceptive feedback (reward).

        Graceful degradation: if no TaskPool is injected, falls back
        to the base behavior (store action, return 0.0).
        """
        task_pool = getattr(self, "_task_pool", None)
        if task_pool is None:
            # No environment -- degrade gracefully to base behavior
            self.last_action = action
            return 0.0

        # Reset resting flag at start of each action
        self._resting = False

        if isinstance(action, str):
            action_type = action
        elif isinstance(action, dict):
            action_type = action.get("type", str(action))
        else:
            action_type = getattr(action, "type", str(action))
        self.last_action = action

        # Market-role dispatch: route to role-specific implementation
        # before generic TaskPool. Law 5: same class, config-gated behavior.
        _agent_role = getattr(self, "role", None)
        _is_market_role = False
        if action_type in ("explore", "exploit", "communicate"):
            from mae_core.market.market_actions import MARKET_ROLES, act_market
            if _agent_role in MARKET_ROLES:
                _is_market_role = True
                market_reward = act_market(self, action_type)
                if market_reward is not None:
                    return market_reward

        if action_type == "explore":
            reward = self._act_explore(task_pool)
        elif action_type == "exploit":
            reward = self._act_exploit(task_pool)
        elif action_type == "communicate":
            reward = self._act_communicate(task_pool)
        elif action_type == "rest":
            return self._act_rest(task_pool)
        elif action_type == "api_call":
            return self._act_api_call(task_pool)
        else:
            return 0.0

        # Market-role agents that fell through to TaskPool are capped at 0.3.
        if _is_market_role and reward is not None:
            reward = min(0.3, reward)
        return reward

    def _act_explore(self, pool: Any) -> float:
        """Explore the environment: claim easy tasks, deposit discovery markers.

        Biological analogy: Foraging behavior -- low effort, wide search,
        mapping the territory. Returns small rewards from discovery.
        """
        # Look for explore-type tasks first, fall back to any available
        tasks = pool.get_available_tasks(task_type="explore")
        if not tasks:
            tasks = pool.get_available_tasks()
        if not tasks:
            return 0.0

        # Claim the easiest task (lowest difficulty -- exploration favors easy wins)
        tasks.sort(key=lambda t: t.difficulty)
        target = tasks[0]
        claimed = pool.claim_task(target.task_id, str(self.unique_id))
        if claimed is None:
            return 0.0

        self._current_task_id = claimed.task_id

        # Work with moderate effort (exploration is not all-in)
        _progress, completed, reward = pool.work_on_task(
            claimed.task_id, str(self.unique_id), effort=0.7,
        )

        if completed:
            self._current_task_id = None

        # Deposit exploration stigmergy marker (biological: pheromone trail)
        self.deposit_marker(
            "EXPLORATION",
            intensity=0.5,
            metadata={"task_type": target.task_type, "difficulty": target.difficulty},
        )

        return reward

    def _act_exploit(self, pool: Any) -> float:
        """Exploit known resources: work existing task or claim highest-reward.

        Biological analogy: Focused foraging -- full effort on a known
        food source until depleted. Returns large rewards on completion.
        """
        # Check if we already have a claimed task
        if self._current_task_id is not None:
            task = pool._tasks.get(self._current_task_id)
            if task is not None and task.state == "claimed" and task.claimed_by == str(self.unique_id):
                # Continue working on it with full effort
                _progress, completed, reward = pool.work_on_task(
                    self._current_task_id, str(self.unique_id), effort=1.0,
                )
                if completed:
                    # Deposit success marker (biological: food-found pheromone)
                    self.deposit_success_marker(reward)
                    self._current_task_id = None
                return reward
            else:
                # Task vanished or expired -- clear stale reference
                self._current_task_id = None

        # No current task -- claim the highest-reward available task
        tasks = pool.get_available_tasks()
        if not tasks:
            return 0.0

        # Sort by reward (descending) -- exploitation maximizes payoff
        tasks.sort(key=lambda t: t.reward_value, reverse=True)
        target = tasks[0]
        claimed = pool.claim_task(target.task_id, str(self.unique_id))
        if claimed is None:
            return 0.0

        self._current_task_id = claimed.task_id

        # Work with full effort immediately
        _progress, completed, reward = pool.work_on_task(
            claimed.task_id, str(self.unique_id), effort=1.0,
        )
        if completed:
            self.deposit_success_marker(reward)
            self._current_task_id = None

        return reward

    def _act_communicate(self, pool: Any) -> float:
        """Share solutions or work on sharing tasks.

        Biological analogy: Teaching behavior, alarm calls, cooperative
        signaling. Returns social rewards for collective benefit.
        """
        reward = 0.0

        # If we have a completed task, broadcast its solution
        if self._current_task_id is not None:
            task = pool._tasks.get(self._current_task_id)
            if task is not None and task.state == "completed":
                reward += pool.broadcast_solution(self._current_task_id, str(self.unique_id))
                self._current_task_id = None

                # Emit collaboration signal (biological: cooperative vocalization)
                emit_fn = getattr(self, "emit_signal", None)
                if emit_fn is not None:
                    emit_fn("COLLABORATION_REQUEST", {
                        "agent_id": str(self.unique_id),
                        "type": "solution_broadcast",
                        "step": self.step_count,
                    })

                # Publish insight on EventBus for peer consumption
                bus = getattr(self, "_event_bus", None)
                if bus is not None:
                    bus.publish("agent.shared", {
                        "agent_id": str(self.unique_id),
                        "step": self.step_count,
                        "prediction_error": float(getattr(self, "_prediction_error", 0.0)),
                    })
                return reward

        # No completed task to share -- look for "share" type tasks
        tasks = pool.get_available_tasks(task_type="share")
        if not tasks:
            tasks = pool.get_available_tasks()
        if not tasks:
            return 0.0

        # Claim and work on a share task
        target = tasks[0]
        claimed = pool.claim_task(target.task_id, str(self.unique_id))
        if claimed is None:
            return 0.0

        self._current_task_id = claimed.task_id
        _progress, completed, task_reward = pool.work_on_task(
            claimed.task_id, str(self.unique_id), effort=0.8,
        )
        reward += task_reward

        if completed:
            # Auto-broadcast completed share tasks
            reward += pool.broadcast_solution(claimed.task_id, str(self.unique_id))
            self._current_task_id = None

        return reward

    def _act_rest(self, pool: Any) -> float:
        """Rest and consolidate: abandon current task, enter rest state.

        Biological analogy: Sleep -- metabolic recovery, synaptic
        homeostasis, memory consolidation. Small immediate cost,
        but subsequent _learn() benefits from the rest state.
        """
        reward = 0.0

        # Abandon any claimed-but-incomplete task (metabolic cost of stopping)
        if self._current_task_id is not None:
            task = pool._tasks.get(self._current_task_id)
            if task is not None and task.state == "claimed":
                reward += pool.abandon_task(self._current_task_id, str(self.unique_id))
            self._current_task_id = None

        self._resting = True

        # Small positive reward for consolidation (biological: restorative sleep)
        reward += 0.1
        return reward

    def _act_api_call(self, pool: Any) -> float:
        """Request external API consultation via the gateway.

        Biological analogy: Asking the oracle. The agent injects a task
        into the TaskPool with an external_spec. The ApiGateway picks it
        up on its step hook, calls the provider, and completes the task.
        The agent does NOT claim or process the task — gateway handles that.

        Cadence: one oracle request per 8 steps (Fibonacci).
        """
        config = getattr(self, "agent_config", {})
        if not config.get("api_call_enabled", False):
            return 0.0

        # Cadence gate: one oracle request per 8 agent-steps
        steps = getattr(self, "step_count", 0)
        last_oracle = getattr(self, "_last_oracle_step", 0)
        if steps - last_oracle < 8:
            return 0.0

        try:
            from mae_core.external.external_task import inject_external_task

            provider = config.get("preferred_provider", "")
            if not provider:
                for name in ("groq", "mistral", "deepseek", "claude"):
                    provider = name
                    break

            # Build question from agent context
            role = getattr(self, "role", "STEM")
            pred_error = getattr(self, "_prediction_error", 0.0)
            risk = getattr(self, "risk_score", 0.0)
            perf = list(getattr(self, "performance_history", []))[-5:]
            avg_perf = sum(perf) / len(perf) if perf else 0.0

            # Market-role agents get role-specific prompts with live data
            from mae_core.market.market_awareness import build_market_oracle_prompt, is_market_agent
            if is_market_agent(self):
                base_q = (
                    f"You are advising agent {self.unique_id} (role={role}). "
                    f"Prediction error: {pred_error:.2f}, risk: {risk:.2f}, "
                    f"avg performance (last 5): {avg_perf:.3f}. "
                    f"What strategy should this agent prioritize?"
                )
                payload = build_market_oracle_prompt(self, base_q)
            else:
                payload = {
                    "question": (
                        f"You are advising agent {self.unique_id} (role={role}). "
                        f"Prediction error: {pred_error:.2f}, risk: {risk:.2f}, "
                        f"avg performance (last 5): {avg_perf:.3f}. "
                        f"What strategy should this agent prioritize?"
                    ),
                    "context": {
                        "agent_id": str(self.unique_id),
                        "prediction_error": float(pred_error),
                        "role": role,
                        "risk": float(risk),
                    },
                }

            task_id = inject_external_task(
                task_pool=pool,
                provider=provider,
                payload=payload,
                agent_id=str(self.unique_id),
                priority=3,
            )
            if task_id is not None:
                self._last_oracle_step = steps
                logger.info(
                    "Agent %s: ORACLE REQUEST — task %s → %s (pred_error=%.2f)",
                    self.unique_id, task_id, provider, pred_error,
                )
                return 0.05
        except Exception:
            logger.debug("Agent %s: api_call injection failed", self.unique_id, exc_info=True)

        return 0.0
