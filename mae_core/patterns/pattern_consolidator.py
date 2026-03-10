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
    than the consolidation budget allows, they compete for storage.

Implementation is split across sub-modules for the 500-line cap:
  pattern_consolidator_extractors.py -- _PatternConsolidatorExtractorsMixin
                                        (extraction, storage, helpers)
  pattern_consolidator_selection.py  -- _PatternConsolidatorSelectionMixin
                                        (competitive selection, fitness, downscaling)
"""

from __future__ import annotations

import logging
from typing import Any

from mae_core.patterns.pattern_consolidator_extractors import (
    _PatternConsolidatorExtractorsMixin,
    _SignalExperience,
    TREND_STORE_THRESHOLD,
)
from mae_core.patterns.pattern_consolidator_selection import (
    _PatternConsolidatorSelectionMixin,
    MAX_CONSOLIDATION_BUDGET,
    _W_SALIENCE,
    _W_PERSISTENCE,
    _W_NOVELTY,
    _W_EMOTIONAL,
    _LATERAL_INHIBITION_FACTOR,
    _DOWNSCALE_PERSISTENCE,
    _MAX_STREAK,
)

logger = logging.getLogger(__name__)

# Fibonacci consolidation interval
CONSOLIDATION_INTERVAL = 89


class PatternConsolidator(
    _PatternConsolidatorExtractorsMixin,
    _PatternConsolidatorSelectionMixin,
):
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
