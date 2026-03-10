"""Pattern Consolidator competitive selection and synaptic downscaling mixin.

Extracted from pattern_consolidator.py to keep the core class under the 500-line cap.
Import from mae_core.patterns.pattern_consolidator for all public names.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ── Competitive Selection Constants ─────────────────────────────────
# Rule of 3/5: 5 for critical processes (consolidation is critical)
MAX_CONSOLIDATION_BUDGET = 5

# Fitness weight vector (must sum to 1.0)
_W_SALIENCE = 0.4
_W_PERSISTENCE = 0.3
_W_NOVELTY = 0.2
_W_EMOTIONAL = 0.1

# Lateral inhibition factor: how much a winner suppresses same-domain peers.
_LATERAL_INHIBITION_FACTOR = 0.5

# Synaptic downscaling: persistence reduction for competition losers.
_DOWNSCALE_PERSISTENCE = 1

# Maximum streak for persistence normalization
_MAX_STREAK = 20


class _PatternConsolidatorSelectionMixin:
    """Mixin providing competitive selection and synaptic downscaling.

    Mixed into PatternConsolidator. Requires the following attributes:
        _consolidation_counts, _total_competitions, _total_candidates,
        _total_winners, _total_suppressed, _last_avg_fitness, _cortex
    """

    def _competitive_select(
        self,
        candidates: list[dict[str, Any]],
        emotional_weight: float,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Select which candidates win consolidation resources.

        Implements a biologically-grounded competitive selection loop:

        1. Compute fitness for each candidate (salience + persistence +
           novelty + emotional weight).
        2. Pick the highest-fitness candidate as winner.
        3. Apply lateral inhibition: suppress same-domain candidates by
           _LATERAL_INHIBITION_FACTOR (like cortical lateral inhibition
           ensuring diverse neural representations).
        4. Repeat until budget is filled or candidates exhausted.

        When fewer candidates than budget exist, all pass through unchanged
        (backward compatible -- no competition needed).

        Args:
            candidates: All extracted pattern dicts from every source.
            emotional_weight: Current threat/arousal level from advisory
                (0-1). High emotion = better consolidation, matching the
                biological finding that emotionally tagged memories are
                preferentially replayed by sharp-wave ripples.

        Returns:
            Tuple of (winners list, competition metadata dict).
        """
        meta: dict[str, Any] = {
            "candidate_count": len(candidates),
            "winner_count": 0,
            "suppressed_count": 0,
            "avg_fitness": 0.0,
            "competed": False,
            "losers": [],
        }

        if not candidates:
            return [], meta

        # If within budget, no competition needed (backward compatible)
        if len(candidates) <= MAX_CONSOLIDATION_BUDGET:
            meta["winner_count"] = len(candidates)
            # Still compute fitness for statistics even without competition
            scores = [
                self._compute_fitness(c, emotional_weight) for c in candidates
            ]
            meta["avg_fitness"] = sum(scores) / len(scores) if scores else 0.0
            return list(candidates), meta

        # ── Competition required ────────────────────────────────────
        meta["competed"] = True
        self._total_competitions += 1

        # Build working list with fitness scores.
        # Each entry: (candidate_dict, fitness_score, domain_str)
        working = []
        for c in candidates:
            score = self._compute_fitness(c, emotional_weight)
            domain = c.get("domain", "unknown")
            working.append({"candidate": c, "score": score, "domain": domain})

        all_scores = [w["score"] for w in working]
        meta["avg_fitness"] = sum(all_scores) / len(all_scores)
        self._last_avg_fitness = meta["avg_fitness"]

        winners: list[dict[str, Any]] = []
        suppressed_count = 0

        while len(winners) < MAX_CONSOLIDATION_BUDGET and working:
            # Find the highest-scoring candidate
            best_idx = 0
            best_score = working[0]["score"]
            for i, w in enumerate(working[1:], start=1):
                if w["score"] > best_score:
                    best_score = w["score"]
                    best_idx = i

            winner_entry = working.pop(best_idx)
            winners.append(winner_entry["candidate"])
            winner_domain = winner_entry["domain"]

            # Lateral inhibition: suppress same-domain candidates.
            for w in working:
                if w["domain"] == winner_domain:
                    w["score"] *= (1.0 - _LATERAL_INHIBITION_FACTOR)
                    suppressed_count += 1

        # Remaining candidates are losers
        meta["losers"] = [w["candidate"] for w in working]
        meta["winner_count"] = len(winners)
        meta["suppressed_count"] = suppressed_count

        # Update running totals
        self._total_candidates += len(candidates)
        self._total_winners += len(winners)
        self._total_suppressed += suppressed_count

        logger.debug(
            "Competitive selection: %d candidates -> %d winners "
            "(suppressed %d same-domain, avg fitness %.3f)",
            len(candidates), len(winners), suppressed_count,
            meta["avg_fitness"],
        )

        return winners, meta

    def _compute_fitness(
        self,
        candidate: dict[str, Any],
        emotional_weight: float,
    ) -> float:
        """Compute consolidation fitness for a single candidate.

        Fitness determines priority in the competition for limited
        consolidation resources. The formula mirrors biological factors
        that determine which memories survive sleep consolidation:

        fitness = salience * 0.4        (how prominent right now)
               + persistence * 0.3      (how long active -- streak duration)
               + novelty * 0.2          (new > familiar -- inverse of
                                         prior consolidation count)
               + emotional_weight * 0.1 (threat/arousal level -- sharp-wave
                                         ripple replay bias)
        """
        # Salience: use confidence as proxy (already 0-1 normalized)
        salience = float(candidate.get("confidence", 0.0))

        # Persistence: streak length normalized by max streak.
        occurrence = float(candidate.get("occurrence_count", 1))
        persistence = min(1.0, occurrence / _MAX_STREAK)

        # Novelty: inverse of how many times this domain has been consolidated.
        domain = candidate.get("domain", "unknown")
        prior_count = self._consolidation_counts.get(domain, 0)
        novelty = 1.0 / (1.0 + prior_count)

        # Emotional weight: from the most recent advisory's threat_level.
        emotion = min(1.0, max(0.0, emotional_weight))

        fitness = (
            _W_SALIENCE * salience
            + _W_PERSISTENCE * persistence
            + _W_NOVELTY * novelty
            + _W_EMOTIONAL * emotion
        )

        return fitness

    def _get_emotional_weight(self) -> float:
        """Get the current emotional/arousal weight from recent advisories.

        Uses the most recent advisory's threat_level as a proxy for
        emotional arousal. Falls back to 0.0 if no advisory available.
        """
        recent = getattr(self._cortex, "_recent_advisories", [])
        if not recent:
            return 0.0

        # Use the most recent advisory's threat level
        latest = recent[-1] if hasattr(recent, '__getitem__') else None
        if latest is None:
            # Try iterating (deque)
            latest_val = None
            for item in recent:
                latest_val = item
            latest = latest_val

        if latest is None:
            return 0.0

        return float(getattr(latest, "threat_level", 0.0))

    def _apply_synaptic_downscaling(
        self, losers: list[dict[str, Any]]
    ) -> None:
        """Reduce persistence of losing candidates (synaptic downscaling).

        During NREM sleep, synapses undergo global homeostatic downscaling
        (Tononi & Cirelli, synaptic homeostasis hypothesis, 2003). In Mae,
        patterns that lose competition get their domain streak reduced in the
        cortex, giving them another chance next cycle with reduced strength.
        """
        if not losers:
            return

        domain_streak = getattr(self._cortex, "_domain_streak", {})
        if not domain_streak:
            return

        for loser in losers:
            loser_domain = loser.get("domain", "")
            # Find the matching domain key in the streak dict
            for domain_key, streak_val in domain_streak.items():
                domain_str = (
                    domain_key.value
                    if hasattr(domain_key, "value")
                    else str(domain_key)
                )
                if domain_str == loser_domain and streak_val > 0:
                    new_val = max(0, streak_val - _DOWNSCALE_PERSISTENCE)
                    domain_streak[domain_key] = new_val
                    logger.debug(
                        "Synaptic downscaling: %s streak %d -> %d",
                        loser_domain, streak_val, new_val,
                    )
                    break
