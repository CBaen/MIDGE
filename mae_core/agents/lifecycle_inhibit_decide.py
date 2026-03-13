"""Inhibit and Decide lifecycle mixin for MycelialAgent.

Contains _inhibit, _decide, and _route_with_advisory. Extracted from
lifecycle_decision.py to stay under 500-line limit.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class InhibitDecideMixin:
    """Inhibit and decide lifecycle methods for MycelialAgent."""

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
        # MIDGE: disabled — fictional physiology harms trading daemon.
        # _collision_risks and _danger_gradient are spatial navigation concepts
        # from a grid-based agent model. MIDGE agents are not navigating physical
        # space. If either attribute is set by any system (e.g., stigmergy field
        # gradients), these blocks would freeze all agents before market intelligence
        # runs. Removed both blocks entirely. The dead code below is preserved for
        # mae-core compatibility.
        #
        # collision_risks = getattr(self, "_collision_risks", None)
        # if collision_risks:
        #     if len(collision_risks) > 0:
        #         return "rest"  # Stop to avoid collision
        #
        # danger_grad = getattr(self, "_danger_gradient", None)
        # if danger_grad is not None and isinstance(danger_grad, (list, tuple)) and len(danger_grad) > 0:
        #     danger_strength = float(danger_grad[0]) if isinstance(danger_grad[0], (int, float)) else 0.0
        #     if danger_strength > 0.5:
        #         return "rest"

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
                            return trajectory[0][1]
            except Exception:
                logger.debug("collective dream failed", exc_info=True)

        # Search memory for similar past states (biological: hippocampus retrieval)
        if state_vec is not None and self.semantic_retriever is not None:
            past = self.search_similar_experiences(state_vec, k=3)
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
                    cause_to_action = {"explore": "explore", "exploit": "exploit", "rest": "rest"}
                    if top_cause in cause_to_action:
                        return cause_to_action[top_cause]
            except Exception:
                logger.debug("Agent %s: causal inference failed", self.unique_id, exc_info=True)

        # --- VDN value-guided action selection (biological: basal ganglia) ---
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

        # Enrich with market context for market-role agents
        from mae_core.market.market_awareness import get_market_context_for_router, is_market_agent
        if is_market_agent(self):
            market_ctx = get_market_context_for_router(self)
            context.update(market_ctx)
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
