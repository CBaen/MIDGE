"""Pattern Consolidator - Distills pattern trends into ancestral memory.

Every 89 steps (Fibonacci), the consolidator examines the PatternCortex's
recent advisories and domain streaks, extracts significant trends, and
stores them as ancestral patterns via the MemoryBridge.

This closes the autopoietic loop: patterns detected by the cortex become
ancestral memory, which feeds back into future cortex advisories via
ancestral recall.

Biological analogy: Sleep consolidation. During deep sleep, the hippocampus
replays recent experiences and the cortex extracts general rules. The
consolidator is Mae's sleep-phase pattern extraction.

Competitive Selection (Law 8, Property 7: competition/selection):
    Consolidation has limited capacity -- like protein synthesis in synaptic
    tag-and-capture (Frey & Morris 1997). When more candidate patterns exist
    than the consolidation budget allows, they compete for storage:

    - Fitness scoring: salience, persistence, novelty, emotional weight
    - Lateral inhibition: winner suppresses same-domain candidates (ensures
      diversity, like lateral inhibition in cortical winner-take-all circuits)
    - Synaptic downscaling: losers get persistence reduced (like homeostatic
      downscaling during NREM sleep)
    - Sharp-wave ripple selection: high-threat/emotion patterns consolidate
      preferentially (matches hippocampal replay bias toward salient events)

    When fewer candidates exist than budget, all pass through unchanged
    (backward compatible).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Fibonacci consolidation interval
CONSOLIDATION_INTERVAL = 89

# Rule of Three: minimum consecutive occurrences to store a trend
TREND_STORE_THRESHOLD = 3

# ── Competitive Selection Constants ─────────────────────────────────
# Rule of 3/5: 5 for critical processes (consolidation is critical)
MAX_CONSOLIDATION_BUDGET = 5

# Fitness weight vector (must sum to 1.0)
# Salience: how prominent the pattern is right now
# Persistence: how long the pattern has been continuously active
# Novelty: inverse familiarity -- new patterns consolidate better
# Emotional weight: threat/arousal level -- emotionally tagged memories
#   consolidate preferentially (sharp-wave ripple selection bias)
_W_SALIENCE = 0.4
_W_PERSISTENCE = 0.3
_W_NOVELTY = 0.2
_W_EMOTIONAL = 0.1

# Lateral inhibition factor: how much a winner suppresses same-domain peers.
# 0.5 = halve their fitness. Mirrors cortical lateral inhibition where active
# neurons suppress neighbors to prevent redundant encoding.
_LATERAL_INHIBITION_FACTOR = 0.5

# Synaptic downscaling: persistence reduction for competition losers.
# During NREM sleep, synapses are globally downscaled (Tononi & Cirelli
# synaptic homeostasis hypothesis). Patterns that fail to consolidate
# lose 1 step of persistence, giving them another chance next cycle
# with reduced strength.
_DOWNSCALE_PERSISTENCE = 1

# Maximum streak for persistence normalization
_MAX_STREAK = 20


class _SignalExperience:
    """Adapter: wraps a PatternSignal as an experience for the distiller.

    The distiller expects objects with .action, .reward, .state attributes.
    We derive them from the signal's domain, confidence, and salience:

    - action: integer hash of the domain (groups signals by domain)
    - reward: confidence value (high confidence = positive reward signal)
    - state: 8-dimensional vector built from signal evidence
    """

    __slots__ = ("action", "reward", "state")

    def __init__(self, signal: Any) -> None:
        import numpy as np

        # Action: group by domain (integer code)
        domain = getattr(signal, "domain", None)
        if hasattr(domain, "value"):
            self.action = hash(domain.value) % 100
        else:
            self.action = hash(str(domain)) % 100

        # Reward: confidence as a proxy for "how good was this signal"
        self.reward = float(getattr(signal, "confidence", 0.0))

        # State: build a feature vector from available signal attributes
        salience = float(getattr(signal, "salience", 0.0))
        confidence = float(getattr(signal, "confidence", 0.0))
        occurrence = float(getattr(signal, "occurrence_count", 1))
        ttl = float(getattr(signal, "ttl_steps", 5))

        self.state = np.array([
            confidence,
            salience,
            min(1.0, occurrence / 10.0),
            min(1.0, ttl / 10.0),
            0.0, 0.0, 0.0, 0.0,  # Padding to 8 dimensions
        ], dtype=np.float64)


class PatternConsolidator:
    """Extracts trend patterns from the cortex and stores as ancestral memory.

    Operates on up to four sources of pattern intelligence:
    1. Domain streaks from the cortex (sustained trends)
    2. Meta-patterns from advisories (patterns about patterns)
    3. Cross-domain insights from advisories (correlated intelligence)
    4. Distilled behavioral/state patterns from PatternDistiller (if wired)

    Competitive selection (Law 8, Property 7):
    - When candidates exceed MAX_CONSOLIDATION_BUDGET, they compete via
      fitness scoring with lateral inhibition for diversity.
    - Biological basis: synaptic tag-and-capture (Frey & Morris 1997),
      sharp-wave ripple replay selection, cortical lateral inhibition.
    - Backward compatible: fewer candidates than budget = no competition.

    Circadian gating:
    - If a CircadianRhythm is provided, consolidation only runs during
      CONSOLIDATION phase (like biological sleep spindles gating memory
      replay to sleep). Forced consolidation bypasses this gate.

    Graceful degradation:
    - If memory_bridge is None: extracts patterns locally but does not store
    - If pattern_cortex is None: does nothing
    - If pattern_distiller is None: skips distillation pass
    - If circadian is None: consolidates unconditionally (legacy behavior)
    """

    def __init__(
        self,
        pattern_cortex: Any = None,
        memory_bridge: Any = None,
        pattern_distiller: Any = None,
        event_bus: Any = None,
        circadian: Any = None,
    ) -> None:
        self._cortex = pattern_cortex
        self._bridge = memory_bridge
        self._distiller = pattern_distiller
        self._bus = event_bus
        self._circadian = circadian
        self._total_consolidations = 0
        self._total_trends_stored = 0
        self._total_meta_stored = 0
        self._total_insights_stored = 0
        self._total_distilled_stored = 0
        self._total_circadian_skips = 0
        # Competition statistics
        self._total_candidates = 0
        self._total_winners = 0
        self._total_suppressed = 0
        self._total_competitions = 0
        self._last_avg_fitness = 0.0
        # Novelty tracking: how many times each domain has been consolidated
        # (patterns from frequently consolidated domains are less novel)
        self._consolidation_counts: dict[str, int] = {}

    def consolidate(self, step: int, force: bool = False) -> dict[str, Any]:
        """Run one consolidation pass.

        Called every CONSOLIDATION_INTERVAL steps from main.py step hook.
        Returns a summary dict of what was consolidated.

        Args:
            step: Current simulation step number.
            force: If True, bypass circadian gating (like forced awakening
                   consolidation in biological systems).
        """
        if self._cortex is None:
            return {"skipped": True, "reason": "no_cortex"}

        # Circadian gate: only consolidate during CONSOLIDATION phase
        # unless forced. Like sleep spindles gating memory replay.
        if not force and self._circadian is not None:
            should_consolidate = getattr(
                self._circadian, "should_consolidate_memory", None
            )
            if callable(should_consolidate) and not should_consolidate():
                self._total_circadian_skips += 1
                return {
                    "skipped": True,
                    "reason": "circadian_gate",
                    "phase": getattr(
                        getattr(self._circadian, "current_phase", None),
                        "value",
                        "unknown",
                    ),
                }

        self._total_consolidations += 1
        results: dict[str, Any] = {
            "step": step,
            "consolidation_number": self._total_consolidations,
            "trends_stored": 0,
            "meta_stored": 0,
            "insights_stored": 0,
            "distilled_stored": 0,
        }

        # ── Gather ALL candidates from every source ─────────────────
        all_candidates: list[dict[str, Any]] = []
        all_candidates.extend(self._extract_trend_patterns())
        all_candidates.extend(self._extract_meta_patterns())
        all_candidates.extend(self._extract_insight_patterns())
        all_candidates.extend(self._extract_distilled_patterns())

        # ── Competitive selection ───────────────────────────────────
        # Synaptic tag-and-capture: all candidates are "tagged" by being
        # extracted above. Now they compete for limited consolidation
        # resources (the budget), like memories competing for protein
        # synthesis during sleep.
        emotional_weight = self._get_emotional_weight()
        winners, competition_meta = self._competitive_select(
            all_candidates, emotional_weight,
        )

        # ── Store winners ───────────────────────────────────────────
        for pattern in winners:
            ptype = pattern.get("pattern_type", "")
            stored = self._store_pattern(pattern)
            if stored:
                if ptype == "trend":
                    results["trends_stored"] += 1
                    self._total_trends_stored += 1
                elif ptype == "meta":
                    results["meta_stored"] += 1
                    self._total_meta_stored += 1
                elif ptype == "insight":
                    results["insights_stored"] += 1
                    self._total_insights_stored += 1
                elif ptype.startswith("distilled"):
                    results["distilled_stored"] += 1
                    self._total_distilled_stored += 1
                # Update novelty tracking: this domain has been consolidated
                domain = pattern.get("domain", "unknown")
                self._consolidation_counts[domain] = (
                    self._consolidation_counts.get(domain, 0) + 1
                )

        # ── Synaptic downscaling for losers ─────────────────────────
        # Patterns that lost competition get their persistence reduced,
        # like homeostatic synaptic downscaling during NREM sleep
        # (Tononi & Cirelli). They get another chance next cycle with
        # weakened strength -- if the underlying signal persists, the
        # streak will rebuild and they may win next time.
        losers = competition_meta.get("losers", [])
        self._apply_synaptic_downscaling(losers)

        # Attach competition metadata to results
        results["competition"] = {
            "candidates": competition_meta.get("candidate_count", 0),
            "winners": competition_meta.get("winner_count", 0),
            "suppressed": competition_meta.get("suppressed_count", 0),
            "avg_fitness": competition_meta.get("avg_fitness", 0.0),
            "budget": MAX_CONSOLIDATION_BUDGET,
            "competed": competition_meta.get("competed", False),
        }

        # Publish consolidation event for observability
        total = (
            results["trends_stored"]
            + results["meta_stored"]
            + results["insights_stored"]
            + results["distilled_stored"]
        )
        if self._bus is not None:
            try:
                self._bus.publish("pattern.consolidation", {
                    "step": step,
                    "consolidation_number": self._total_consolidations,
                    "trends_stored": results["trends_stored"],
                    "meta_stored": results["meta_stored"],
                    "insights_stored": results["insights_stored"],
                    "distilled_stored": results["distilled_stored"],
                    "total_stored": total,
                })
            except Exception:
                logger.debug("Failed to publish consolidation event", exc_info=True)

        if total > 0:
            logger.info(
                "PatternConsolidator: step %d, stored %d patterns "
                "(%d trends, %d meta, %d insights, %d distilled)"
                + (" [competed: %d candidates -> %d winners]"
                   if competition_meta.get("competed") else ""),
                step, total,
                results["trends_stored"],
                results["meta_stored"],
                results["insights_stored"],
                results["distilled_stored"],
                *([competition_meta["candidate_count"],
                   competition_meta["winner_count"]]
                  if competition_meta.get("competed") else []),
            )

        return results

    # ── Competitive Selection ────────────────────────────────────────

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
            # Like cortical lateral inhibition where an active neuron
            # suppresses its neighbors. Ensures consolidation diversity --
            # we don't want all 5 slots filled by the same domain.
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

        Args:
            candidate: Pattern dict with confidence, occurrence_count, domain.
            emotional_weight: System-level threat/arousal (0-1).

        Returns:
            Fitness score (0-1 range, higher = more likely to consolidate).
        """
        # Salience: use confidence as proxy (already 0-1 normalized)
        salience = float(candidate.get("confidence", 0.0))

        # Persistence: streak length normalized by max streak.
        # Long-running patterns have had more "rehearsal" -- like memory
        # traces that are replayed more often during waking sharp-wave
        # ripples and thus tagged for sleep consolidation.
        occurrence = float(candidate.get("occurrence_count", 1))
        persistence = min(1.0, occurrence / _MAX_STREAK)

        # Novelty: inverse of how many times this domain has been
        # previously consolidated. New patterns are more valuable
        # (the brain preferentially consolidates novel information).
        domain = candidate.get("domain", "unknown")
        prior_count = self._consolidation_counts.get(domain, 0)
        novelty = 1.0 / (1.0 + prior_count)

        # Emotional weight: from the most recent advisory's threat_level.
        # Emotionally arousing experiences get preferential consolidation
        # (amygdala modulates hippocampal replay, sharp-wave ripples
        # selectively replay high-salience events).
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
        emotional arousal. In biology, the amygdala's response to threat
        modulates hippocampal consolidation -- high-threat memories get
        preferential replay via sharp-wave ripples.

        Falls back to 0.0 if no advisory available (neutral emotion).
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
        (Tononi & Cirelli, synaptic homeostasis hypothesis, 2003). Strongly
        tagged synapses survive; weakly tagged ones are pruned. In Mae,
        patterns that lose competition get their domain streak reduced in the
        cortex, giving them another chance next cycle with reduced strength.

        If the underlying signal persists (the environment keeps producing it),
        the streak will rebuild and the pattern may win next time. If it
        doesn't persist, the downscaling accelerates its natural decay.
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

    # ── Pattern Extraction ───────────────────────────────────────────

    def _extract_trend_patterns(self) -> list[dict[str, Any]]:
        """Extract trend patterns from cortex domain streaks.

        Only includes domains with TREND_STORE_THRESHOLD+ consecutive
        occurrences (Rule of Three).
        """
        domain_streak = getattr(self._cortex, "_domain_streak", {})
        if not domain_streak:
            return []

        patterns = []
        for domain, streak in domain_streak.items():
            if streak < TREND_STORE_THRESHOLD:
                continue

            domain_str = domain.value if hasattr(domain, "value") else str(domain)
            direction = "emerging" if streak <= 5 else "sustained"

            patterns.append({
                "pattern_type": "trend",
                "domain": domain_str,
                "occurrence_count": streak,
                "confidence": min(1.0, 0.4 + 0.1 * streak),
                "description": (
                    f"{domain_str} domain active for {streak} consecutive "
                    f"steps (direction: {direction})"
                ),
                "contributing_agents": self._get_contributing_agents(),
                "trend_direction": direction,
                "cross_domain_context": self._get_co_occurring_domains(domain),
                "applicable_roles": [],
            })

        return patterns

    def _extract_meta_patterns(self) -> list[dict[str, Any]]:
        """Extract meta-patterns from recent advisories.

        Meta-patterns are the strange loop: the cortex detecting
        patterns in its own output.
        """
        recent = getattr(self._cortex, "_recent_advisories", [])
        if not recent:
            return []

        # Deduplicate by domain (keep highest confidence)
        by_domain: dict[str, dict] = {}
        for advisory in recent:
            for meta_sig in getattr(advisory, "meta_patterns", []):
                evidence = getattr(meta_sig, "evidence", {})
                domain_key = evidence.get("recurring_domain", "meta")
                confidence = getattr(meta_sig, "confidence", 0.5)

                if domain_key not in by_domain or confidence > by_domain[domain_key]["confidence"]:
                    by_domain[domain_key] = {
                        "pattern_type": "meta",
                        "domain": domain_key,
                        "occurrence_count": evidence.get("recurrence_count", 0),
                        "confidence": confidence,
                        "description": getattr(meta_sig, "description", ""),
                        "contributing_agents": self._get_contributing_agents(),
                        "applicable_roles": [],
                    }

        return list(by_domain.values())

    def _extract_insight_patterns(self) -> list[dict[str, Any]]:
        """Extract cross-domain insights from recent advisories."""
        recent = getattr(self._cortex, "_recent_advisories", [])
        if not recent:
            return []

        # Collect unique insights (skip Trend: prefix -- captured above)
        seen: set[str] = set()
        patterns = []
        for advisory in recent:
            for insight in getattr(advisory, "correlated_insights", []):
                if insight in seen or insight.startswith("Trend:"):
                    continue
                seen.add(insight)
                patterns.append({
                    "pattern_type": "insight",
                    "domain": "cross_domain",
                    "occurrence_count": 1,
                    "confidence": 0.5,
                    "description": insight,
                    "contributing_agents": self._get_contributing_agents(),
                    "applicable_roles": [],
                })

        return patterns

    def _extract_distilled_patterns(self) -> list[dict[str, Any]]:
        """Extract behavioral and state patterns via PatternDistiller.

        The distiller analyses raw experiences (action/reward/state tuples).
        We gather signal-level evidence from the cortex window and convert
        it into lightweight experience proxies the distiller can process.

        Graceful degradation: returns [] if distiller is unavailable or if
        the cortex window lacks sufficient data.
        """
        if self._distiller is None:
            return []

        # Gather signal evidence from the cortex window
        window = getattr(self._cortex, "_window", [])
        if not window:
            return []

        # Build experience proxies from pattern signals.
        # Each signal has domain (-> action proxy), confidence (-> reward proxy),
        # and salience evidence that can stand in for state dimensions.
        experiences = []
        for digest in window:
            for sig in getattr(digest, "signals", []):
                experiences.append(_SignalExperience(sig))

        patterns: list[dict[str, Any]] = []
        try:
            # detect_behavioral_patterns: action-reward correlations
            behavioral_fn = getattr(self._distiller, "detect_behavioral_patterns", None)
            if callable(behavioral_fn):
                behavioral = behavioral_fn(experiences)
                for p in behavioral:
                    p["pattern_type"] = "distilled_behavioral"
                    p.setdefault("contributing_agents", self._get_contributing_agents())
                patterns.extend(behavioral)
        except Exception:
            logger.debug(
                "PatternConsolidator: distiller behavioral extraction failed",
                exc_info=True,
            )

        try:
            # detect_state_patterns: recurring state configurations
            state_fn = getattr(self._distiller, "detect_state_patterns", None)
            if callable(state_fn):
                state = state_fn(experiences)
                for p in state:
                    p["pattern_type"] = "distilled_state"
                    p.setdefault("contributing_agents", self._get_contributing_agents())
                patterns.extend(state)
        except Exception:
            logger.debug(
                "PatternConsolidator: distiller state extraction failed",
                exc_info=True,
            )

        return patterns

    # ── Storage ──────────────────────────────────────────────────────

    def _store_pattern(self, pattern: dict[str, Any]) -> bool:
        """Store a pattern as ancestral memory via MemoryBridge.

        Returns True if stored, False if skipped (no bridge available).
        """
        if self._bridge is None:
            return False

        store_fn = getattr(self._bridge, "store_ancestral_pattern", None)
        if store_fn is None:
            return False

        try:
            contributing = pattern.pop("contributing_agents", [])
            result = store_fn(pattern, contributing)
            return result is not None
        except Exception:
            logger.debug("PatternConsolidator: failed to store pattern", exc_info=True)
            return False

    # ── Helpers ──────────────────────────────────────────────────────

    def _get_contributing_agents(self) -> list[str]:
        """Get list of agent IDs that contributed to current patterns."""
        window = getattr(self._cortex, "_window", [])
        agents: set[str] = set()
        for digest in window:
            for sig in getattr(digest, "signals", []):
                source = getattr(sig, "source_system", "")
                if source.startswith("agent:"):
                    agents.add(source)
                elif source.startswith("triad:"):
                    agents.add(source)
        return sorted(agents)[:10]

    def _get_co_occurring_domains(self, target_domain: Any) -> list[str]:
        """Find domains that co-occur with the target in the cortex window."""
        domain_streak = getattr(self._cortex, "_domain_streak", {})
        co_occurring = []
        for domain, streak in domain_streak.items():
            if domain != target_domain and streak >= TREND_STORE_THRESHOLD:
                val = domain.value if hasattr(domain, "value") else str(domain)
                co_occurring.append(val)
        return co_occurring

    # ── Statistics ───────────────────────────────────────────────────

    def get_statistics(self) -> dict[str, Any]:
        return {
            "total_consolidations": self._total_consolidations,
            "total_trends_stored": self._total_trends_stored,
            "total_meta_stored": self._total_meta_stored,
            "total_insights_stored": self._total_insights_stored,
            "total_distilled_stored": self._total_distilled_stored,
            "total_patterns_stored": (
                self._total_trends_stored
                + self._total_meta_stored
                + self._total_insights_stored
                + self._total_distilled_stored
            ),
            "total_circadian_skips": self._total_circadian_skips,
            "has_cortex": self._cortex is not None,
            "has_bridge": self._bridge is not None,
            "has_distiller": self._distiller is not None,
            "has_circadian": self._circadian is not None,
            # Competition statistics
            "competition": {
                "total_competitions": self._total_competitions,
                "total_candidates": self._total_candidates,
                "total_winners": self._total_winners,
                "total_suppressed": self._total_suppressed,
                "last_avg_fitness": self._last_avg_fitness,
                "budget": MAX_CONSOLIDATION_BUDGET,
                "consolidation_counts_by_domain": dict(self._consolidation_counts),
            },
        }

    def __repr__(self) -> str:
        total = (
            self._total_trends_stored
            + self._total_meta_stored
            + self._total_insights_stored
            + self._total_distilled_stored
        )
        return (
            f"PatternConsolidator("
            f"consolidations={self._total_consolidations}, "
            f"stored={total}, "
            f"competitions={self._total_competitions})"
        )
