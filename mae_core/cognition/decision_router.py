"""Decision Router - Three-tier hierarchical decision system.

Implements biological decision hierarchy:
- Tier 1 (Reflex): <1ms pattern matching, immediate responses
- Tier 2 (Habit): 10-100ms learned routines, automatic behavior
- Tier 3 (Prefrontal): 1s+ deliberative reasoning, novel problems

Stimuli cascade through tiers: if reflex matches, no further processing.
If not, try habits. If no habit, escalate to prefrontal cortex.
Repeated prefrontal decisions form new habits automatically.

Biological analogy: Spinal cord → basal ganglia → prefrontal cortex.
Based on: Kahneman (2011) "Thinking Fast and Slow", dual-process theory.

Sub-modules:
  decision_reflexes.py — reflex/habit check, registration, habit formation
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

from mae_core.cognition.decision_reflexes import (
    register_default_reflexes as _register_default_reflexes_fn,
    check_reflex as _check_reflex_fn,
    check_habit as _check_habit_fn,
    track_for_habit_formation as _track_for_habit_formation_fn,
)

logger = logging.getLogger(__name__)


class DecisionTier(Enum):
    REFLEX = "reflex"  # <1ms immediate
    HABIT = "habit"  # 10-100ms automatic
    PREFRONTAL = "prefrontal"  # 1s+ deliberative
    NONE = "none"


@dataclass
class ReflexPattern:
    """A pattern-action pair for immediate response."""

    pattern_id: str
    stimulus_pattern: str  # regex-like pattern to match
    action: Any
    confidence: float = 0.95
    priority: int = 10  # Higher = checked first


@dataclass
class Habit:
    """A learned automatic response."""

    habit_id: str
    stimulus: str
    action: Any
    strength: float = 0.5  # [0, 1] - strengthens with repetition
    activation_count: int = 0
    last_activated: float = 0.0


@dataclass
class RouterDecision:
    """Complete decision output with metadata."""

    decision_id: str
    tier_used: DecisionTier
    stimulus: str
    context: dict[str, Any] = field(default_factory=dict)
    action_taken: Any = None
    response_time: float = 0.0  # Total milliseconds
    tier_times: dict[str, float] = field(default_factory=dict)
    tiers_checked: list[str] = field(default_factory=list)
    confidence: float = 0.0
    reasoning: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class DecisionRouter:
    """Three-tier cognitive decision system with automatic habit formation.

    Tier cascade: Reflex → Habit → Prefrontal
    Habit formation: After N identical prefrontal decisions, a habit forms.
    Executive override: Force prefrontal reasoning (bypass lower tiers).
    """

    def __init__(
        self,
        enable_reflexes: bool = True,
        enable_habits: bool = True,
        enable_prefrontal: bool = True,
        auto_habit_formation: bool = True,
        habit_formation_threshold: int = 5,
        prefrontal_fn: Any | None = None,
        event_bus: Any | None = None,
        world_model: Any | None = None,
        endocrine: Any | None = None,
    ) -> None:
        self._enable_reflex = enable_reflexes
        self._enable_habit = enable_habits
        self._enable_prefrontal = enable_prefrontal
        self._auto_habit = auto_habit_formation
        self._habit_threshold = habit_formation_threshold
        self._prefrontal_fn = prefrontal_fn
        self._bus = event_bus
        self._world_model = world_model
        self._endocrine = endocrine

        # Tier 1: Reflex patterns
        self._reflex_patterns: dict[str, ReflexPattern] = {}

        # Tier 2: Learned habits
        self._habits: dict[str, Habit] = {}
        self._habit_lookup: dict[str, str] = {}  # stimulus -> habit_id

        # Tier 3: Prefrontal tracking for habit formation
        self._prefrontal_sequences: dict[str, list[Any]] = defaultdict(list)

        # Endocrine reflex bias (set by adrenaline via EndocrineSystem)
        self._reflex_bias: float = 0.0

        # Statistics
        self._tier_usage: dict[str, int] = defaultdict(int)
        self._total_decisions = 0
        self._decision_history: deque[RouterDecision] = deque(maxlen=1000)
        self._habits_formed = 0
        self._reflex_bias_triggers: int = 0  # Decisions won by bias-lowered threshold
        self._lock = threading.RLock()

        # Register default danger reflexes
        self._register_default_reflexes()

    def route_decision(
        self,
        stimulus: str,
        context: dict[str, Any] | None = None,
        available_actions: list[Any] | None = None,
        force_tier: DecisionTier | None = None,
    ) -> RouterDecision:
        """Route a stimulus through the three-tier cascade."""
        context = context or {}
        start_time = time.perf_counter()
        tier_times: dict[str, float] = {}
        tiers_checked: list[str] = []
        reasoning: list[str] = []

        # ── Self-awareness valve ──────────────────────────────────────
        # The Strange Loop puts self-awareness data into context.  If
        # present, low health biases toward survival reflexes.  Critical
        # health forces reflex tier outright — the organism cannot afford
        # deliberation when it is about to die.
        self_awareness = context.get("self_awareness")
        if isinstance(self_awareness, dict):
            self_health = self_awareness.get("health", 1.0)
            if isinstance(self_health, (int, float)):
                if self_health < 0.1:
                    # Critical: force reflex tier (survival override)
                    force_tier = DecisionTier.REFLEX
                    reasoning.append(
                        f"Self-awareness CRITICAL: health={self_health:.2f}, "
                        "forcing reflex tier for survival"
                    )
                    logger.info(
                        "Self-awareness override: health=%.2f, forcing reflex",
                        self_health,
                    )
                elif self_health < 0.3:
                    # Low: bias toward reflexes (heightened vigilance)
                    survival_bias = max(self._reflex_bias, 0.7)
                    self._reflex_bias = survival_bias
                    reasoning.append(
                        f"Self-awareness LOW: health={self_health:.2f}, "
                        f"reflex bias raised to {survival_bias:.2f}"
                    )
                    logger.info(
                        "Self-awareness bias: health=%.2f, reflex_bias=%.2f",
                        self_health,
                        survival_bias,
                    )

        # ── Somatic marker valve ─────────────────────────────────────
        # Emotional valence/arousal from organism body state inform
        # tier selection (Damasio's somatic marker hypothesis).
        # Negative valence + high arousal = danger → bias toward reflex.
        # Positive valence + high arousal = opportunity → bias toward prefrontal.
        emotional_valence = context.get("emotional_valence")
        emotional_arousal = context.get("emotional_arousal")
        if emotional_valence is not None and emotional_arousal is not None:
            try:
                valence = float(emotional_valence)
                arousal = float(emotional_arousal)
                if arousal > 0.6:
                    if valence < -0.3:
                        # Negative emotion + high arousal → fight/flight (reflex bias)
                        somatic_bias = min(0.9, self._reflex_bias + arousal * 0.3)
                        self._reflex_bias = max(self._reflex_bias, somatic_bias)
                        reasoning.append(
                            f"Somatic marker: negative valence={valence:.2f}, "
                            f"arousal={arousal:.2f} → reflex bias {somatic_bias:.2f}"
                        )
                    elif valence > 0.3:
                        # Positive emotion + high arousal → explore (dampen reflex)
                        self._reflex_bias = max(0.0, self._reflex_bias - valence * 0.2)
                        reasoning.append(
                            f"Somatic marker: positive valence={valence:.2f}, "
                            f"arousal={arousal:.2f} → reflex bias dampened to {self._reflex_bias:.2f}"
                        )
            except (TypeError, ValueError):
                pass  # Non-numeric values, skip gracefully

        # Force specific tier if requested
        if force_tier is not None:
            tier, action, confidence, reason = self._force_tier(
                force_tier, stimulus, context, available_actions
            )
            tier_times[tier.value] = (time.perf_counter() - start_time) * 1000
            tiers_checked.append(tier.value)
            reasoning.append(reason)
            return self._create_decision(
                tier, stimulus, context, action, confidence,
                start_time, tier_times, tiers_checked, reasoning,
            )

        # Tier 1: Reflex (immediate pattern matching)
        if self._enable_reflex:
            t1_start = time.perf_counter()
            reflex = self._check_reflex(stimulus)
            tier_times["reflex"] = (time.perf_counter() - t1_start) * 1000
            tiers_checked.append("reflex")

            if reflex is not None:
                reasoning.append(f"Reflex match: pattern '{reflex.pattern_id}'")
                return self._create_decision(
                    DecisionTier.REFLEX, stimulus, context,
                    reflex.action, reflex.confidence,
                    start_time, tier_times, tiers_checked, reasoning,
                )
            reasoning.append("No reflex match")

        # Tier 2: Habit (learned automatic responses)
        if self._enable_habit:
            t2_start = time.perf_counter()
            habit = self._check_habit(stimulus)
            tier_times["habit"] = (time.perf_counter() - t2_start) * 1000
            tiers_checked.append("habit")

            if habit is not None:
                habit.activation_count += 1
                habit.last_activated = time.time()
                # Strengthen with use (max 1.0)
                habit.strength = min(1.0, habit.strength + 0.05)
                reasoning.append(
                    f"Habit match: '{habit.habit_id}' "
                    f"(strength={habit.strength:.2f}, uses={habit.activation_count})"
                )
                return self._create_decision(
                    DecisionTier.HABIT, stimulus, context,
                    habit.action, habit.strength,
                    start_time, tier_times, tiers_checked, reasoning,
                )
            reasoning.append("No habit match")

        # Endocrine bias: high adrenaline biases toward reflex tier
        # Uses _reflex_bias (set by EndocrineSystem via set_reflex_bias)
        # to enable fuzzy matching — the second reflex check is genuinely
        # broader than the first, so stimuli that were "close but not exact"
        # can now trigger a reflex under adrenaline pressure.
        if self._reflex_bias > 0.5:
            try:
                reasoning.append(
                    f"Endocrine override: reflex bias {self._reflex_bias:.2f}, "
                    "re-checking with fuzzy matching"
                )
                reflex = self._check_reflex(stimulus, bias=self._reflex_bias)
                if reflex is not None:
                    with self._lock:
                        self._reflex_bias_triggers += 1
                    return self._create_decision(
                        DecisionTier.REFLEX, stimulus, context,
                        reflex.action, reflex.confidence * self._reflex_bias,
                        start_time, tier_times, tiers_checked, reasoning,
                    )
            except Exception:
                logger.debug("Reflex bias re-check failed, continuing normally")

        # Tier 3: Prefrontal (deliberative reasoning)
        if self._enable_prefrontal:
            t3_start = time.perf_counter()
            action, confidence, prefrontal_reason = self._invoke_prefrontal(
                stimulus, context, available_actions
            )
            tier_times["prefrontal"] = (time.perf_counter() - t3_start) * 1000
            tiers_checked.append("prefrontal")
            reasoning.append(f"Prefrontal: {prefrontal_reason}")

            # Track for habit formation
            if self._auto_habit:
                self._track_for_habit_formation(stimulus, action)

            return self._create_decision(
                DecisionTier.PREFRONTAL, stimulus, context,
                action, confidence,
                start_time, tier_times, tiers_checked, reasoning,
            )

        # No tier handled it
        reasoning.append("All tiers disabled or failed")
        return self._create_decision(
            DecisionTier.NONE, stimulus, context,
            None, 0.0,
            start_time, tier_times, tiers_checked, reasoning,
        )

    def executive_override(
        self,
        stimulus: str,
        context: dict[str, Any] | None = None,
        override_reason: str = "Executive override requested",
    ) -> RouterDecision:
        """Force prefrontal reasoning, bypassing reflexes and habits."""
        return self.route_decision(
            stimulus, context, force_tier=DecisionTier.PREFRONTAL
        )

    def register_reflex(self, pattern: ReflexPattern) -> None:
        """Add a reflex pattern."""
        with self._lock:
            self._reflex_patterns[pattern.pattern_id] = pattern

    def register_habit(self, habit: Habit) -> None:
        """Add a learned habit."""
        with self._lock:
            self._habits[habit.habit_id] = habit
            self._habit_lookup[habit.stimulus] = habit.habit_id

    def set_reflex_bias(self, level: float) -> None:
        """Set reflex bias from endocrine adrenaline level.

        Higher bias makes reflexes more sensitive by enabling fuzzy
        matching when exact/substring matching fails.  Called by
        EndocrineSystem.register_decision_router when adrenaline is
        released.

        Args:
            level: Adrenaline-derived bias, 0.0 (no bias) to 1.0 (max).
        """
        self._reflex_bias = max(0.0, min(1.0, level))

    def get_performance_metrics(self) -> dict[str, Any]:
        """Decision-making performance across all tiers."""
        with self._lock:
            total = max(self._total_decisions, 1)
            return {
                "total_decisions": self._total_decisions,
                "tier_distribution": {
                    tier: count / total
                    for tier, count in self._tier_usage.items()
                },
                "reflex_patterns": len(self._reflex_patterns),
                "active_habits": len(self._habits),
                "habits_formed": self._habits_formed,
                "reflex_bias_triggers": self._reflex_bias_triggers,
                "current_reflex_bias": self._reflex_bias,
                "pending_habit_sequences": len(self._prefrontal_sequences),
            }

    def get_tier_health(self) -> dict[str, dict[str, Any]]:
        """Health status for each tier."""
        return {
            "reflex": {
                "enabled": self._enable_reflex,
                "patterns": len(self._reflex_patterns),
            },
            "habit": {
                "enabled": self._enable_habit,
                "habits": len(self._habits),
                "avg_strength": (
                    float(np.mean([h.strength for h in self._habits.values()]))
                    if self._habits
                    else 0.0
                ),
            },
            "prefrontal": {
                "enabled": self._enable_prefrontal,
                "has_custom_fn": self._prefrontal_fn is not None,
            },
        }

    # --- Internal ---

    def _check_reflex(
        self, stimulus: str, bias: float = 0.0
    ) -> ReflexPattern | None:
        """Check if stimulus matches any reflex pattern."""
        return _check_reflex_fn(self, stimulus, bias)

    def _check_habit(self, stimulus: str) -> Habit | None:
        """Check if stimulus has a learned habit."""
        return _check_habit_fn(self, stimulus)

    def _invoke_prefrontal(
        self,
        stimulus: str,
        context: dict[str, Any],
        available_actions: list[Any] | None = None,
    ) -> tuple[Any, float, str]:
        """Deliberative reasoning - the slow, thoughtful path."""
        import random as _rng  # local import: avoids module-level side-effect, used in two branches below

        if self._prefrontal_fn is not None:
            result = self._prefrontal_fn(stimulus, context, available_actions)
            if isinstance(result, tuple):
                action, confidence = result[0], result[1] if len(result) > 1 else 0.7
                return action, confidence, "Custom prefrontal function"
            return result, 0.7, "Custom prefrontal function"

        # Use WorldModel for simulation-based deliberation if available
        if self._world_model is not None and available_actions:
            try:
                best_actions = []
                best_reward = float("-inf")
                for action in available_actions:
                    pred = self._world_model.step(
                        context.get("state", np.zeros(10, dtype=np.float32)),
                        action,
                        deterministic=True,
                    )
                    if pred.reward > best_reward:
                        best_reward = pred.reward
                        best_actions = [action]
                    elif pred.reward == best_reward:
                        best_actions.append(action)
                best_action = _rng.choice(best_actions)
                return (
                    best_action,
                    0.75,
                    f"WorldModel simulation (best reward={best_reward:.3f})",
                )
            except Exception:
                logger.debug("WorldModel simulation failed, falling back")

        # Default: select from available actions or generate response
        if available_actions:
            action = _rng.choice(available_actions)
            return action, 0.6, "Default selection (random from available)"

        return {"type": "deliberate", "stimulus": stimulus}, 0.5, "Default deliberation"

    def _force_tier(
        self,
        tier: DecisionTier,
        stimulus: str,
        context: dict[str, Any],
        available_actions: list[Any] | None,
    ) -> tuple[DecisionTier, Any, float, str]:
        """Force decision through a specific tier."""
        if tier == DecisionTier.REFLEX:
            reflex = self._check_reflex(stimulus)
            if reflex:
                return DecisionTier.REFLEX, reflex.action, reflex.confidence, "Forced reflex"
            return DecisionTier.NONE, None, 0.0, "No reflex match (forced)"
        if tier == DecisionTier.HABIT:
            habit = self._check_habit(stimulus)
            if habit:
                return DecisionTier.HABIT, habit.action, habit.strength, "Forced habit"
            return DecisionTier.NONE, None, 0.0, "No habit match (forced)"
        if tier == DecisionTier.PREFRONTAL:
            action, conf, reason = self._invoke_prefrontal(stimulus, context, available_actions)
            return DecisionTier.PREFRONTAL, action, conf, f"Forced prefrontal: {reason}"
        return DecisionTier.NONE, None, 0.0, "Unknown tier"

    def _track_for_habit_formation(self, stimulus: str, action: Any) -> None:
        """Track prefrontal decisions for automatic habit formation."""
        _track_for_habit_formation_fn(self, stimulus, action)

    def _create_decision(
        self,
        tier: DecisionTier,
        stimulus: str,
        context: dict[str, Any],
        action: Any,
        confidence: float,
        start_time: float,
        tier_times: dict[str, float],
        tiers_checked: list[str],
        reasoning: list[str],
    ) -> RouterDecision:
        """Build RouterDecision and update statistics."""
        response_time = (time.perf_counter() - start_time) * 1000

        decision = RouterDecision(
            decision_id=f"dec-{self._total_decisions}",
            tier_used=tier,
            stimulus=stimulus,
            context=context,
            action_taken=action,
            response_time=response_time,
            tier_times=tier_times,
            tiers_checked=tiers_checked,
            confidence=confidence,
            reasoning=reasoning,
        )

        with self._lock:
            self._total_decisions += 1
            self._tier_usage[tier.value] += 1
            self._decision_history.append(decision)

        if self._bus is not None:
            try:
                self._bus.publish("cognition.decision_routed", {
                    "decision_id": decision.decision_id,
                    "tier": tier.value,
                    "stimulus": stimulus,
                    "confidence": confidence,
                    "response_time_ms": response_time,
                })
            except Exception:
                logger.debug("EventBus publish failed for decision_routed")

        return decision

    def _register_default_reflexes(self) -> None:
        """Register built-in survival reflexes."""
        _register_default_reflexes_fn(self)

    def __repr__(self) -> str:
        return (
            f"DecisionRouter(decisions={self._total_decisions}, "
            f"reflexes={len(self._reflex_patterns)}, "
            f"habits={len(self._habits)})"
        )
