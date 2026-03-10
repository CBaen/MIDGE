"""
decision_reflexes.py - Reflex, habit, and routing helpers for DecisionRouter.

Extracted from decision_router.py. Contains:
  - register_default_reflexes: built-in survival reflex registration
  - check_reflex: stimulus-to-pattern matching (with adrenaline fuzzy mode)
  - check_habit: learned habit lookup
  - track_for_habit_formation: automatic habit formation from repetition
  - invoke_prefrontal: deliberative reasoning (WorldModel + fallback)
  - force_tier: single-tier forced decision
  - create_decision: build RouterDecision + update statistics
"""
from __future__ import annotations
import logging
import time
from typing import TYPE_CHECKING, Any
import numpy as np
if TYPE_CHECKING:
    from mae_core.cognition.decision_router import (DecisionRouter, DecisionTier, Habit, ReflexPattern, RouterDecision)
logger = logging.getLogger(__name__)

def register_default_reflexes(router):
    from mae_core.cognition.decision_router import ReflexPattern as RP
    defaults = [RP("danger", "danger", {"type": "flee"}, 0.99, 10), RP("threat", "threat", {"type": "alert"}, 0.95, 9), RP("collision", "collision", {"type": "avoid"}, 0.98, 10)]
    for pattern in defaults:
        router._reflex_patterns[pattern.pattern_id] = pattern

def check_reflex(router, stimulus, bias=0.0):
    stimulus_lower = stimulus.lower()
    with router._lock:
        for pattern in sorted(router._reflex_patterns.values(), key=lambda p: p.priority, reverse=True):
            if pattern.stimulus_pattern in stimulus_lower:
                return pattern
        if bias > 0.5:
            stimulus_words = stimulus_lower.split()
            for pattern in sorted(router._reflex_patterns.values(), key=lambda p: p.priority, reverse=True):
                pat = pattern.stimulus_pattern.lower()
                min_prefix = max(5, int(len(pat) * (1.0 - (bias - 0.5) * 2)))
                min_prefix = min(min_prefix, len(pat))
                prefix = pat[:min_prefix]
                for word in stimulus_words:
                    if word.startswith(prefix) or prefix in word:
                        return pattern
    return None

def check_habit(router, stimulus):
    with router._lock:
        habit_id = router._habit_lookup.get(stimulus)
        if habit_id and habit_id in router._habits:
            habit = router._habits[habit_id]
            if habit.strength >= 0.3:
                return habit
    return None

def track_for_habit_formation(router, stimulus, action):
    from mae_core.cognition.decision_router import Habit as H
    with router._lock:
        seq = router._prefrontal_sequences[stimulus]
        seq.append(action)
        if len(seq) >= router._habit_threshold:
            action_strs = [str(a) for a in seq[-router._habit_threshold:]]
            if len(set(action_strs)) == 1:
                habit_id = f"auto-habit-{router._habits_formed}"
                habit = H(habit_id=habit_id, stimulus=stimulus, action=action, strength=0.5)
                router._habits[habit_id] = habit
                router._habit_lookup[stimulus] = habit_id
                router._habits_formed += 1
                router._prefrontal_sequences.pop(stimulus, None)
                logger.info("Habit formed: %s for stimulus '%s'", habit_id, stimulus)

def invoke_prefrontal(router, stimulus, context, available_actions=None):
    import random as _rng
    if router._prefrontal_fn is not None:
        result = router._prefrontal_fn(stimulus, context, available_actions)
        if isinstance(result, tuple):
            action, confidence = result[0], result[1] if len(result) > 1 else 0.7
            return action, confidence, "Custom prefrontal function"
        return result, 0.7, "Custom prefrontal function"
    if router._world_model is not None and available_actions:
        try:
            best_actions = []
            best_reward = float("-inf")
            for action in available_actions:
                pred = router._world_model.step(context.get("state", np.zeros(10, dtype=np.float32)), action, deterministic=True)
                if pred.reward > best_reward:
                    best_reward = pred.reward
                    best_actions = [action]
                elif pred.reward == best_reward:
                    best_actions.append(action)
            best_action = _rng.choice(best_actions)
            return (best_action, 0.75, f"WorldModel simulation (best reward={best_reward:.3f})")
        except Exception:
            logger.debug("WorldModel simulation failed, falling back")
    if available_actions:
        action = _rng.choice(available_actions)
        return action, 0.6, "Default selection (random from available)"
    return {"type": "deliberate", "stimulus": stimulus}, 0.5, "Default deliberation"

def force_tier(router, tier, stimulus, context, available_actions):
    from mae_core.cognition.decision_router import DecisionTier as DT
    if tier == DT.REFLEX:
        reflex = check_reflex(router, stimulus)
        if reflex:
            return DT.REFLEX, reflex.action, reflex.confidence, "Forced reflex"
        return DT.NONE, None, 0.0, "No reflex match (forced)"
    if tier == DT.HABIT:
        habit = check_habit(router, stimulus)
        if habit:
            return DT.HABIT, habit.action, habit.strength, "Forced habit"
        return DT.NONE, None, 0.0, "No habit match (forced)"
    if tier == DT.PREFRONTAL:
        action, conf, reason = invoke_prefrontal(router, stimulus, context, available_actions)
        return DT.PREFRONTAL, action, conf, f"Forced prefrontal: {reason}"
    return DT.NONE, None, 0.0, "Unknown tier"

def create_decision(router, tier, stimulus, context, action, confidence, start_time, tier_times, tiers_checked, reasoning):
    from mae_core.cognition.decision_router import RouterDecision as RD
    response_time = (time.perf_counter() - start_time) * 1000
    decision = RD(decision_id=f"dec-{router._total_decisions}", tier_used=tier, stimulus=stimulus, context=context, action_taken=action, response_time=response_time, tier_times=tier_times, tiers_checked=tiers_checked, confidence=confidence, reasoning=reasoning)
    with router._lock:
        router._total_decisions += 1
        router._tier_usage[tier.value] += 1
        router._decision_history.append(decision)
    if router._bus is not None:
        try:
            router._bus.publish("cognition.decision_routed", {"decision_id": decision.decision_id, "tier": tier.value, "stimulus": stimulus, "confidence": confidence, "response_time_ms": response_time})
        except Exception:
            logger.debug("EventBus publish failed for decision_routed")
    return decision
