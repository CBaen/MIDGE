"""
decision_reflexes.py - Reflex and habit management for DecisionRouter.

Extracted from decision_router.py. Contains:
  - _register_default_reflexes: built-in survival reflex registration
  - _check_reflex: stimulus-to-pattern matching (with adrenaline fuzzy mode)
  - _check_habit: learned habit lookup
  - _track_for_habit_formation: automatic habit formation from repetition
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mae_core.cognition.decision_router import DecisionRouter, Habit, ReflexPattern

logger = logging.getLogger(__name__)


def register_default_reflexes(router: DecisionRouter) -> None:
    """Register built-in survival reflexes."""
    from mae_core.cognition.decision_router import ReflexPattern as RP

    defaults = [
        RP("danger", "danger", {"type": "flee"}, 0.99, 10),
        RP("threat", "threat", {"type": "alert"}, 0.95, 9),
        RP("collision", "collision", {"type": "avoid"}, 0.98, 10),
    ]
    for pattern in defaults:
        router._reflex_patterns[pattern.pattern_id] = pattern


def check_reflex(
    router: DecisionRouter,
    stimulus: str,
    bias: float = 0.0,
) -> ReflexPattern | None:
    """Check if stimulus matches any reflex pattern.

    Args:
        router: The DecisionRouter instance.
        stimulus: The input stimulus string.
        bias: Reflex bias (0.0-1.0). When > 0.5, enables fuzzy
            matching: any word in the stimulus that shares a
            common prefix (length proportional to bias) with a
            reflex pattern counts as a match.  This models how
            adrenaline makes the nervous system hypersensitive -
            ambiguous signals are more likely to trigger reflexes.
    """
    stimulus_lower = stimulus.lower()
    with router._lock:
        # Check exact/substring matches first (always)
        for pattern in sorted(
            router._reflex_patterns.values(),
            key=lambda p: p.priority,
            reverse=True,
        ):
            if pattern.stimulus_pattern in stimulus_lower:
                return pattern

        # Fuzzy matching only when bias is elevated
        if bias > 0.5:
            # Minimum prefix length decreases as bias rises:
            #   bias 0.5 → min_prefix = full pattern length (no fuzz)
            #   bias 1.0 → min_prefix = 3 chars
            stimulus_words = stimulus_lower.split()
            for pattern in sorted(
                router._reflex_patterns.values(),
                key=lambda p: p.priority,
                reverse=True,
            ):
                pat = pattern.stimulus_pattern.lower()
                # Scale: at bias=1.0 require only 5 chars; at bias=0.5 require full length
                # Floor of 5 prevents overly broad matching ("dan" -> "danger" AND "dance")
                min_prefix = max(5, int(len(pat) * (1.0 - (bias - 0.5) * 2)))
                min_prefix = min(min_prefix, len(pat))
                prefix = pat[:min_prefix]
                for word in stimulus_words:
                    if word.startswith(prefix) or prefix in word:
                        return pattern
    return None


def check_habit(
    router: DecisionRouter,
    stimulus: str,
) -> Habit | None:
    """Check if stimulus has a learned habit."""
    with router._lock:
        habit_id = router._habit_lookup.get(stimulus)
        if habit_id and habit_id in router._habits:
            habit = router._habits[habit_id]
            # Only fire if strength is above threshold
            if habit.strength >= 0.3:
                return habit
    return None


def track_for_habit_formation(
    router: DecisionRouter,
    stimulus: str,
    action: Any,
) -> None:
    """Track prefrontal decisions for automatic habit formation."""
    from mae_core.cognition.decision_router import Habit as H

    with router._lock:
        seq = router._prefrontal_sequences[stimulus]
        seq.append(action)

        if len(seq) >= router._habit_threshold:
            # Check if actions are consistent
            action_strs = [str(a) for a in seq[-router._habit_threshold:]]
            if len(set(action_strs)) == 1:
                # Consistent action → form habit
                habit_id = f"auto-habit-{router._habits_formed}"
                habit = H(
                    habit_id=habit_id,
                    stimulus=stimulus,
                    action=action,
                    strength=0.5,
                )
                router._habits[habit_id] = habit
                router._habit_lookup[stimulus] = habit_id
                router._habits_formed += 1
                router._prefrontal_sequences.pop(stimulus, None)
                logger.info("Habit formed: %s for stimulus '%s'", habit_id, stimulus)
