"""Decision and action lifecycle mixin for MycelialAgent.

Extracted from mycelial_agent.py to prevent monolith growth.
These methods are called by step() in mycelial_agent.py.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class DecisionActionLifecycleMixin:
    """Decision and action lifecycle methods for MycelialAgent.

    All subsystem access is via getattr(self, ..., None) for graceful
    degradation when subsystems are not injected.
    """

    def _inhibit(self) -> bool:
        """INHIBIT: Go/No-Go gate (basal ganglia model).

        Biological analogy: The basal ganglia compute competing Go and No-Go
        signals. If No-Go wins, the thalamocortical loop is suppressed and
        no motor command is issued. This prevents impulsive or dangerous actions.

        Returns True if action should be suppressed (No-Go wins).
        """
        system = getattr(self, "_inhibition_system", None)
        if system is None:
            return False  # No inhibition system = always Go

        try:
            # Gather inputs from current agent state
            pe = getattr(self, "_prediction_error", 0.0)
            risk = getattr(self, "risk_score", 0.0)

            # Energy from organism body state
            body = getattr(self, "_body_state", None)
            energy = body.get("energy_level", 1.0) if isinstance(body, dict) else 1.0
            valence = body.get("emotional_valence", 0.0) if isinstance(body, dict) else 0.0
            arousal = body.get("emotional_arousal", 0.5) if isinstance(body, dict) else 0.5

            # Goal priority from goal manager
            gm = getattr(self, "_goal_manager", None)
            goal_priority = gm.goal_priority if gm is not None else 0.5

            # Quorum pressure from quorum sensing
            qs = getattr(self, "_quorum_state", None)
            quorum_pressure = 0.0
            if qs is not None:
                quorum_pressure = getattr(qs, "density", 0.0) if hasattr(qs, "density") else 0.0

            result = system.evaluate(
                prediction_error=pe,
                risk_score=risk,
                energy_level=energy,
                emotional_valence=valence,
                emotional_arousal=arousal,
                goal_priority=goal_priority,
                reward_trend=getattr(self, "last_reward", 0.0),
                quorum_pressure=quorum_pressure,
                agent_id=str(self.unique_id),
                step=self.step_count,
            )

            if result.inhibited:
                logger.debug(
                    "Agent %s INHIBITED at step %d: %s",
                    self.unique_id, self.step_count, result.reason,
                )
                # Expose inhibition details for JournalWriter (reset by journal hook each step)
                self._inhibited_this_step = True
                self._last_inhibit_reason = result.reason
                self._inhibit_veto_sources = getattr(result, "veto_sources", [])

            return result.inhibited

        except Exception:
            logger.debug(
                "Agent %s: _inhibit failed, defaulting to Go",
                self.unique_id, exc_info=True,
            )
            return False

    def _decide(self) -> Any:
        """Decide on action: advisory-guided router, memory, world model, or default.

        Decision cascade (biological: thalamus → basal ganglia → prefrontal):
        1. If advisory + router available: consult router first
        2. If router returns NONE: fall through to existing logic
        3. Search memory for similar past states (hippocampus retrieval)
        4. Consult world model (prefrontal cortex simulation)
        5. Default action selection
        """
        state_vec = getattr(self, "_curr_state_vector", None)

        # --- Reflex override (organism body state) ---
        organism = getattr(self, "_organism_state", None)
        if organism is not None:
            try:
                reflex = organism.get_reflex_override()
                if reflex is not None:
                    return reflex
            except Exception:
                logger.debug("reflex check failed", exc_info=True)

        # --- Collision avoidance (predictive field) ---
        collision_risks = getattr(self, "_collision_risks", None)
        if collision_risks:
            try:
                if len(collision_risks) > 0:
                    return "rest"  # Stop to avoid collision
            except Exception:
                pass

        # FIX-3: Stigmergy-informed decision bias (biological: chemotaxis)
        # If we're near danger markers and no collision risk, bias toward caution
        danger_grad = getattr(self, "_danger_gradient", None)
        if danger_grad is not None and isinstance(danger_grad, (list, tuple)) and len(danger_grad) > 0:
            # Strong danger gradient biases toward rest/avoidance
            try:
                danger_strength = float(danger_grad[0]) if isinstance(danger_grad[0], (int, float)) else 0.0
                if danger_strength > 0.5:
                    return "rest"
            except (TypeError, ValueError, IndexError):
                pass

        # Advisory-guided decision routing (organism intelligence → agent action)
        advisory = getattr(self, "_current_advisory", None)
        router = getattr(self, "decision_router", None)
        if router is not None and advisory is not None:
            try:
                routed = self._route_with_advisory(router, advisory, state_vec)
                if routed is not None:
                    return routed
            except Exception:
                logger.debug(
                    "Agent %s: advisory-guided routing failed, falling through",
                    self.unique_id, exc_info=True,
                )

        # --- WorldlinePlanner multi-horizon planning ---
        worldline = getattr(self, "_worldline_planner", None)
        if worldline is not None:
            try:
                available_actions = ["explore", "exploit", "communicate", "rest"]
                result = worldline.plan(
                    entity_id=self.unique_id,
                    current_state=self._curr_state_vector,
                    available_actions=available_actions,
                )
                if result is not None:
                    selected = getattr(result, "selected_worldline", None)
                    if selected is not None:
                        points = getattr(selected, "points", None)
                        if points and len(points) > 0:
                            action = points[0].action if hasattr(points[0], "action") else None
                            if action is not None:
                                return action
            except Exception:
                logger.debug("worldline planning failed", exc_info=True)

        # --- CollectiveDreamPlanner consensus ---
        dream = getattr(self, "_collective_dream", None)
        if dream is not None:
            try:
                raw_result = dream.collective_plan(
                    initial_state=self._curr_state_vector,
                    num_dreamers=3,
                    horizon=5,
                )
                if isinstance(raw_result, tuple) and len(raw_result) >= 2:
                    trajectory, dream_result = raw_result
                else:
                    dream_result = raw_result
                if dream_result is not None:
                    status = dream_result.get("status") if isinstance(dream_result, dict) else getattr(dream_result, "status", None)
                    if status == "approved":
                        trajectory = dream_result.get("trajectory") if isinstance(dream_result, dict) else getattr(dream_result, "trajectory", None)
                        if trajectory and len(trajectory) > 0:
                            # trajectory items are (state, action, reward) tuples
                            return trajectory[0][1]
            except Exception:
                logger.debug("collective dream failed", exc_info=True)

        # Search memory for similar past states (biological: hippocampus retrieval)
        # search_similar_experiences returns a SemanticQuery dataclass with .experiences list
        if state_vec is not None and self.semantic_retriever is not None:
            past = self.search_similar_experiences(state_vec, k=3)
            # Unwrap SemanticQuery: extract the .experiences list if present
            past_experiences = getattr(past, "experiences", past) if past else []
            if past_experiences:
                best = max(past_experiences, key=lambda e: getattr(e, "reward", 0.0))
                if getattr(best, "reward", 0.0) > 0:
                    return getattr(best, "action", self._select_action(self.current_state))

        # --- Memory Bridge: deep recall from Qdrant (biological: long-term memory) ---
        mb = getattr(self, "_memory_bridge", None)
        if mb is not None and getattr(mb, "is_available", False):
            try:
                role = getattr(self, "role", "STEM")
                situation = f"agent step {self.step_count} role {role} reward {self.last_reward:.2f}"
                ancestral = mb.recall_ancestral_patterns(
                    query_text=situation, applicable_role=role, limit=3,
                )
                if ancestral:
                    # Use the top ancestral pattern's action if available
                    top = ancestral[0]
                    action_str = getattr(top, "payload", {}).get("action") if hasattr(top, "payload") else None
                    if action_str is not None:
                        return action_str
            except Exception:
                logger.debug("Agent %s: memory_bridge recall failed", self.unique_id, exc_info=True)

        # --- Imitation Learning: apply learned behavioral policies ---
        il = getattr(self, "_imitation_learner", None)
        if il is not None and hasattr(il, "imitate"):
            try:
                imitated = il.imitate(context={
                    "state": state_vec.tolist() if state_vec is not None else [],
                    "step": self.step_count,
                    "reward": getattr(self, "last_reward", 0.0),
                })
                if imitated is not None:
                    return imitated
            except Exception:
                logger.debug("Agent %s: imitation failed", self.unique_id, exc_info=True)

        # --- Causal Reasoning: infer best action from causal model ---
        ce = getattr(self, "causal_engine", None)
        if ce is not None and hasattr(ce, "infer_causes"):
            try:
                causes = ce.infer_causes(
                    effect="high_reward",
                    observations={"risk": self.risk_score, "step": self.step_count},
                )
                if causes and causes.get("likely_causes"):
                    top_cause = list(causes["likely_causes"].keys())[0]
                    # Map causal variable to action type
                    cause_to_action = {"explore": "explore", "exploit": "exploit", "rest": "rest"}
                    if top_cause in cause_to_action:
                        return cause_to_action[top_cause]
            except Exception:
                logger.debug("Agent %s: causal inference failed", self.unique_id, exc_info=True)

        # --- VDN value-guided action selection (biological: basal ganglia) ---
        # The Q-table learns from intrinsic rewards every 10 steps in _learn().
        # Now the learned values get to influence action selection.
        # Epsilon-greedy: Mae discovers diversity herself through exploration.
        vdn = getattr(self, "_vdn_engine", None)
        if vdn is not None and state_vec is not None and self.step_count > 20:
            try:
                import random as _rng
                actions = ["explore", "exploit", "communicate", "rest", "api_call"]
                q_values = []
                for a in actions:
                    action_int = hash(str(a)) % vdn._action_dim
                    q_val = vdn.compute_local_value(state_vec, action_int)
                    q_values.append(q_val)

                max_q = max(q_values)
                if max_q > 0.0:
                    # Decaying epsilon: 20% → 5% over ~500 steps
                    epsilon = max(0.05, 0.20 - (self.step_count * 0.0003))
                    if _rng.random() < epsilon:
                        return _rng.choice(actions)
                    else:
                        best_idx = q_values.index(max_q)
                        return actions[best_idx]
            except Exception:
                logger.debug("Agent %s: VDN selection failed", self.unique_id, exc_info=True)

        # Consult world model (biological: prefrontal cortex simulation)
        if self.world_model is not None:
            wm_action = self.use_world_model()
            if wm_action is not None:
                return wm_action

        # --- Morphogenesis capability gap signal ---
        morpho = getattr(self, "_morphogenesis", None)
        pred_error = getattr(self, "_prediction_error", 0.0)
        if morpho is not None and pred_error > 0.7:
            try:
                # High prediction error = genuinely stuck, may need specialist help
                if hasattr(morpho, "handle_novel_problem"):
                    bus = getattr(self, "signal_bus", None)
                    if bus is not None:
                        import json
                        bus.publish("morphogenesis.capability_gap", json.dumps({
                            "agent_id": self.unique_id,
                            "prediction_error": float(pred_error),
                            "risk_score": float(self.risk_score),
                        }))
            except Exception:
                logger.debug("morphogenesis signal failed", exc_info=True)

        # --- Oracle pathway: ask external when genuinely stuck ---
        config = getattr(self, "agent_config", {})
        if config.get("api_call_enabled", False) and pred_error > 0.5:
            return "api_call"

        return self._select_action(self.current_state)

    def _route_with_advisory(
        self, router: Any, advisory: Any, state_vec: Any,
    ) -> Any | None:
        """Consult DecisionRouter with advisory context.

        Returns an action if the router produces one, or None to fall
        through to the existing memory/world-model/default cascade.
        """
        from mae_core.cognition.decision_router import DecisionTier

        # Build stimulus from advisory's dominant pattern
        dominant = getattr(advisory, "dominant_pattern", None)
        if dominant is not None:
            stimulus = f"{dominant.domain.value}:{dominant.description}"
        else:
            stimulus = "ambient"

        # Build context dict from advisory fields
        context: dict[str, Any] = {
            "threat_level": getattr(advisory, "threat_level", 0.0),
            "opportunity_level": getattr(advisory, "opportunity_level", 0.0),
            "novelty_level": getattr(advisory, "novelty_level", 0.0),
            "active_trends": getattr(advisory, "active_trends", {}),
            "advisory_confidence": getattr(advisory, "confidence", 0.0),
        }
        if state_vec is not None:
            context["state"] = state_vec

        # Enrich with market context for market-role agents.
        # Market agents use market_stimulus for routing — their decisions
        # should be driven by market state, not generic pattern cortex.
        from mae_core.market.market_awareness import get_market_context_for_router, is_market_agent
        if is_market_agent(self):
            market_ctx = get_market_context_for_router(self)
            context.update(market_ctx)
            # Override stimulus with market-state encoding for reflex matching
            market_stim = market_ctx.get("market_stimulus")
            if market_stim:
                stimulus = market_stim

        # Enrich context with organism body state (interoceptive awareness)
        body = getattr(self, "_body_state", None)
        if body:
            context["emotional_valence"] = body.get("emotional_valence", 0.0)
            context["emotional_arousal"] = body.get("emotional_arousal", 0.0)
            context["pain_load"] = body.get("pain_load", 0.0)
            context["energy_level"] = body.get("energy_level", 1.0)
            context["organism_vitality"] = body.get("metacognition_score", 1.0)

        # Strange Loop closure: self-knowledge informs decision-making
        # (biological: prefrontal cortex integrating proprioceptive self-model)
        self_awareness = getattr(self, "_self_awareness", None)
        if self_awareness is not None:
            context["self_awareness"] = self_awareness

        # Force tier only when advisory confidence > 0.6
        force_tier = None
        if getattr(advisory, "confidence", 0.0) > 0.6:
            tier_map = {
                "reflex": DecisionTier.REFLEX,
                "habit": DecisionTier.HABIT,
                "prefrontal": DecisionTier.PREFRONTAL,
            }
            force_tier = tier_map.get(
                getattr(advisory, "recommended_tier", "habit"),
            )

        # Default behavioral repertoire (biological: motor cortex action palette)
        available_actions = [
            {"type": "explore"},
            {"type": "exploit"},
            {"type": "communicate"},
            {"type": "rest"},
            {"type": "api_call"},
        ]

        decision = router.route_decision(
            stimulus=stimulus,
            context=context,
            available_actions=available_actions,
            force_tier=force_tier,
        )

        # NONE tier = "I have no opinion" — fall through
        if decision.tier_used == DecisionTier.NONE:
            return None

        return decision.action_taken

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
        if action_type in ("explore", "exploit", "communicate"):
            from mae_core.market.market_actions import MARKET_ROLES, act_market
            _agent_role = getattr(self, "role", None)
            if _agent_role in MARKET_ROLES:
                market_reward = act_market(self, action_type)
                if market_reward is not None:
                    return market_reward

        if action_type == "explore":
            return self._act_explore(task_pool)
        elif action_type == "exploit":
            return self._act_exploit(task_pool)
        elif action_type == "communicate":
            return self._act_communicate(task_pool)
        elif action_type == "rest":
            return self._act_rest(task_pool)
        elif action_type == "api_call":
            return self._act_api_call(task_pool)
        else:
            return 0.0

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
