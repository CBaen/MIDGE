"""Pattern Cortex - Temporal integration and meta-pattern detection.

The PatternBus is the thalamus: it collects signals each step and produces
a digest. The PatternCortex is the association cortex: it remembers the last
13 steps (Fibonacci window), detects trends, identifies meta-patterns
(the strange loop), and queries ancestral memory.

Biological analogy: The thalamus tells you what is happening NOW. The cortex
tells you what has been happening OVER TIME and what it means. The cortex is
where perception becomes understanding.

Three timescales:
- Immediate: this step's PatternDigest (from PatternBus)
- Short-term: last 13 steps (PatternCortex window)
- Long-term: ancestral patterns in Qdrant (via MemoryBridge)

GWT integration: Before selecting the dominant pattern, all signals pass
through a GlobalWorkspace where domain-level coalitions compete for
broadcast access. Only the winner reaches conscious awareness (Law 8,
Property 7: competition/selection). Winners that lack triadic corroboration
receive reduced confidence (Law 1: No Bare Dyads).
"""

from __future__ import annotations

import logging
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

from mae_core.patterns.pattern_signal import (
    PatternDomain,
    PatternForm,
    PatternSignal,
)
from mae_core.patterns.pattern_bus import PatternDigest
from mae_core.patterns.global_workspace import (
    GlobalWorkspace,
    IgnitionResult,
    UNCORROBORATED_PENALTY,
)

logger = logging.getLogger(__name__)

# Fibonacci window size for short-term pattern memory
WINDOW_SIZE = 13

# Rule of Three: minimum consecutive appearances to qualify as a trend
TREND_THRESHOLD = 3

# How many steps a meta-pattern signal lives
META_TTL = 8

# Consolidation interval (Fibonacci)
CONSOLIDATION_INTERVAL = 89


@dataclass
class PatternAdvisory:
    """The cortex's output -- integrated pattern intelligence for decision-making.

    Like the prefrontal cortex's executive summary: here's what matters,
    here's what's trending, here's what your ancestors knew.
    """

    step: int

    # What demands attention right now
    dominant_pattern: PatternSignal | None = None

    # Threat/opportunity/novelty levels (0-1 aggregates across window)
    threat_level: float = 0.0
    opportunity_level: float = 0.0
    novelty_level: float = 0.0

    # Which decision tier fits the current situation
    recommended_tier: str = "habit"  # "reflex" / "habit" / "prefrontal"

    # Plain-text insights from correlation and trend analysis
    correlated_insights: list[str] = field(default_factory=list)

    # Trends detected (domain -> consecutive count)
    active_trends: dict[str, int] = field(default_factory=dict)

    # Meta-patterns (patterns about patterns)
    meta_patterns: list[PatternSignal] = field(default_factory=list)

    # Ancestral matches from deep memory
    ancestral_matches: list[dict] = field(default_factory=list)

    # Overall confidence in this advisory
    confidence: float = 0.0


class PatternCortex:
    """Temporal integration layer -- the association cortex.

    Receives PatternDigests from the PatternBus, maintains a sliding
    window of 13 steps, detects trends and meta-patterns, queries
    ancestral memory, and produces PatternAdvisory outputs.
    """

    def __init__(
        self,
        memory_bridge: Any = None,
        event_bus: Any = None,
    ) -> None:
        """Initialize the cortex.

        Args:
            memory_bridge: Optional MemoryBridge for ancestral recall.
                           Cortex works fine without it (graceful degradation).
            event_bus: Optional EventBus for GWT broadcast publishing.
                       When a pattern ignites, it gets published to
                       ``"gwt.broadcast"`` on this bus. Without it,
                       ignition still works -- it just does not broadcast.
        """
        self._memory_bridge = memory_bridge
        self._event_bus = event_bus
        self._workspace = GlobalWorkspace()
        self._window: deque[PatternDigest] = deque(maxlen=WINDOW_SIZE)
        self._domain_streak: dict[PatternDomain, int] = {}
        self._recent_advisories: deque[PatternAdvisory] = deque(maxlen=WINDOW_SIZE)
        self._total_advisories = 0
        self._total_trends = 0
        self._total_meta_patterns = 0
        self._total_ancestral_queries = 0
        self._total_ignitions = 0

    @property
    def window_size(self) -> int:
        """Current number of digests in the window."""
        return len(self._window)

    def process_digest(self, digest: PatternDigest) -> PatternAdvisory:
        """Process a new PatternDigest and produce an advisory.

        This is the cortex's main loop. Called once per step after
        the PatternBus produces its digest.
        """
        self._window.append(digest)

        # 1. Update domain streaks (for trend detection)
        self._update_domain_streaks(digest)

        # 2. Detect trends (Rule of Three over time)
        active_trends = self._detect_trends()

        # 3. Detect meta-patterns (the strange loop)
        meta_patterns = self._detect_meta_patterns()

        # 4. Query ancestral memory (if available)
        ancestral_matches = self._recall_ancestral(digest)

        # 5. Compute aggregate levels across window
        threat_level = self._compute_domain_level(PatternDomain.THREAT)
        opportunity_level = self._compute_domain_level(PatternDomain.OPPORTUNITY)
        novelty_level = self._compute_domain_level(PatternDomain.NOVELTY)

        # 6. Generate correlated insights
        insights = self._generate_insights(digest, active_trends)

        # 7. Determine recommended decision tier
        recommended_tier = self._recommend_tier(
            threat_level, novelty_level, digest,
        )

        # 8. GWT competitive ignition (Law 8, Property 7: competition/selection)
        #    Pass all signals through the GlobalWorkspace. Domain-level
        #    coalitions compete; only the winner becomes the dominant pattern.
        #    Falls back to salience-based selection when workspace has no winner.
        ignition = self._workspace.compete(
            digest.signals,
            correlated_groups=digest.correlated_groups,
        )

        dominant = None
        if ignition.winner is not None:
            dominant = ignition.winner
        elif digest.signals:
            # Fallback: no workspace winner, use raw salience (backward compat)
            dominant = max(digest.signals, key=lambda s: s.salience)

        # 9. Compute overall confidence
        confidence = self._compute_confidence(digest, active_trends, ancestral_matches)

        # 9b. Apply triadic verification penalty (Law 1: No Bare Dyads)
        #     If the workspace produced a winner but it lacks corroboration,
        #     reduce confidence to reflect the weaker epistemic status.
        if ignition.winner is not None and not ignition.corroborated:
            confidence *= UNCORROBORATED_PENALTY

        # 9c. Publish GWT broadcast on ignition
        if ignition.ignited and dominant is not None:
            self._total_ignitions += 1
            self._publish_gwt_broadcast(digest.step, dominant, ignition)

        advisory = PatternAdvisory(
            step=digest.step,
            dominant_pattern=dominant,
            threat_level=threat_level,
            opportunity_level=opportunity_level,
            novelty_level=novelty_level,
            recommended_tier=recommended_tier,
            correlated_insights=insights,
            active_trends={d.value: c for d, c in active_trends.items()},
            meta_patterns=meta_patterns,
            ancestral_matches=ancestral_matches,
            confidence=confidence,
        )

        self._recent_advisories.append(advisory)
        self._total_advisories += 1

        return advisory

    # ── Trend Detection ──────────────────────────────────────────────

    def _update_domain_streaks(self, digest: PatternDigest) -> None:
        """Track consecutive steps where each domain appears.

        A domain's streak increments if it has signals this step,
        resets to 0 if absent.
        """
        present_domains = set(digest.by_domain.keys()) if digest.by_domain else set()

        for domain in PatternDomain:
            if domain in present_domains:
                self._domain_streak[domain] = self._domain_streak.get(domain, 0) + 1
            else:
                self._domain_streak[domain] = 0

    def _detect_trends(self) -> dict[PatternDomain, int]:
        """Return domains with streaks >= Rule of Three threshold."""
        trends = {}
        for domain, streak in self._domain_streak.items():
            if streak >= TREND_THRESHOLD:
                trends[domain] = streak

        if trends:
            self._total_trends += len(trends)

        return trends

    # ── Meta-Pattern Detection (Strange Loop) ────────────────────────

    def _detect_meta_patterns(self) -> list[PatternSignal]:
        """Detect patterns about patterns -- the strange loop.

        When the cortex's own output shows a recurring dominant domain
        (same domain dominant in 3+ of the last 5 advisories), it
        generates a META-domain PatternSignal about that recurrence.
        """
        if len(self._recent_advisories) < TREND_THRESHOLD:
            return []

        # Look at recent advisories' dominant patterns
        recent = list(self._recent_advisories)[-5:]
        dominant_domains = []
        for adv in recent:
            if adv.dominant_pattern is not None:
                dominant_domains.append(adv.dominant_pattern.domain)

        if len(dominant_domains) < TREND_THRESHOLD:
            return []

        # Count recurring dominants
        counts = Counter(dominant_domains)
        meta_signals = []

        for domain, count in counts.items():
            if count >= TREND_THRESHOLD:
                meta_sig = PatternSignal(
                    source_system="pattern_cortex",
                    domain=PatternDomain.META,
                    form=PatternForm.CORRELATED,
                    confidence=min(1.0, 0.5 + 0.1 * count),
                    salience=min(1.0, 0.4 + 0.1 * count),
                    description=(
                        f"Meta-pattern: {domain.value} has been dominant "
                        f"in {count}/{len(recent)} recent advisories"
                    ),
                    evidence={
                        "recurring_domain": domain.value,
                        "recurrence_count": count,
                        "window_size": len(recent),
                    },
                    ttl_steps=META_TTL,
                )
                meta_signals.append(meta_sig)
                self._total_meta_patterns += 1

        return meta_signals

    # ── Ancestral Recall ─────────────────────────────────────────────

    def _recall_ancestral(self, digest: PatternDigest) -> list[dict]:
        """Query ancestral memory for patterns matching the current situation.

        Only queries when something noteworthy is happening (signals present
        and salience above threshold). Graceful degradation if MemoryBridge
        is unavailable.
        """
        if self._memory_bridge is None:
            return []

        if digest.signal_count == 0 or digest.aggregate_salience < 0.3:
            return []

        recall_fn = getattr(self._memory_bridge, "recall_ancestral_patterns", None)
        if recall_fn is None:
            return []

        # Build query from dominant signals
        query_parts = []
        if digest.dominant_domain is not None:
            query_parts.append(digest.dominant_domain.value)

        for sig in sorted(digest.signals, key=lambda s: s.salience, reverse=True)[:3]:
            query_parts.append(sig.description)

        query_text = " | ".join(query_parts) if query_parts else "general situation"

        try:
            self._total_ancestral_queries += 1
            results = recall_fn(query_text, limit=3)
            return [
                {
                    "pattern": getattr(r, "payload", getattr(r, "metadata", {})),
                    "score": getattr(r, "score", 0.0),
                }
                for r in results
            ]
        except Exception:
            logger.debug("PatternCortex: ancestral recall failed", exc_info=True)
            return []

    # ── Aggregate Computation ────────────────────────────────────────

    def _compute_domain_level(self, domain: PatternDomain) -> float:
        """Compute aggregate level for a domain across the window.

        Returns a 0-1 value representing how prominent this domain
        has been in recent steps. Uses exponential decay: recent
        steps matter more than older ones.
        """
        if not self._window:
            return 0.0

        total = 0.0
        weight_sum = 0.0

        for i, digest in enumerate(self._window):
            # Exponential decay: more recent = higher weight
            weight = 2.0 ** (i - len(self._window) + 1)  # Most recent = weight 1.0
            weight_sum += weight

            domain_signals = digest.by_domain.get(domain, [])
            if domain_signals:
                domain_salience = sum(s.salience for s in domain_signals)
                total += weight * min(1.0, domain_salience)

        return min(1.0, total / weight_sum) if weight_sum > 0 else 0.0

    def _generate_insights(
        self,
        digest: PatternDigest,
        active_trends: dict[PatternDomain, int],
    ) -> list[str]:
        """Generate plain-text insights from current analysis."""
        insights = []

        # Report correlations in current digest
        for group in digest.correlated_groups:
            sources = [s.source_system for s in group]
            domain = group[0].domain.value if group else "unknown"
            insights.append(
                f"Correlated {domain}: {', '.join(sources)} agree"
            )

        # Report cross-domain correlations
        for group in getattr(digest, "cross_domain_groups", []):
            synthetics = [s for s in group if s.source_system == "cross_domain"]
            if synthetics:
                insights.append(synthetics[0].description)

        # Report active trends
        for domain, streak in active_trends.items():
            insights.append(
                f"Trend: {domain.value} active for {streak} consecutive steps"
            )

        return insights

    def _recommend_tier(
        self,
        threat_level: float,
        novelty_level: float,
        digest: PatternDigest,
    ) -> str:
        """Recommend a decision-making tier based on current patterns.

        - reflex: high threat requires fast response
        - prefrontal: high novelty requires careful thought
        - habit: normal operation, use learned behaviors
        """
        if threat_level > 0.6:
            return "reflex"
        if novelty_level > 0.5 or digest.aggregate_salience > 2.0:
            return "prefrontal"
        return "habit"

    def _compute_confidence(
        self,
        digest: PatternDigest,
        active_trends: dict[PatternDomain, int],
        ancestral_matches: list[dict],
    ) -> float:
        """Compute overall confidence in the advisory.

        Higher when: more signals, trends present, ancestral support.
        """
        confidence = 0.3  # Base

        # More signals = more information
        if digest.signal_count > 0:
            confidence += min(0.2, 0.05 * digest.signal_count)

        # Correlations boost confidence
        if digest.correlated_groups:
            confidence += 0.1 * len(digest.correlated_groups)

        # Trends boost confidence
        if active_trends:
            confidence += 0.1

        # Ancestral support is strong evidence
        if ancestral_matches:
            confidence += 0.15

        return min(1.0, confidence)

    # ── GWT Broadcast ────────────────────────────────────────────────

    def _publish_gwt_broadcast(
        self,
        step: int,
        winner: PatternSignal,
        ignition: IgnitionResult,
    ) -> None:
        """Publish an ignited pattern to the EventBus as a GWT broadcast.

        This is the Transfractal Compromise in action: the broadcast is a
        shortcut that bypasses the fractal hierarchy, reaching all modules
        simultaneously -- like long-range axons connecting distant cortical
        columns.

        Gracefully degrades when no EventBus is available.
        """
        if self._event_bus is None:
            return

        publish_fn = getattr(self._event_bus, "publish", None)
        if publish_fn is None:
            return

        broadcast_payload = {
            "type": "gwt.broadcast",
            "step": step,
            "domain": winner.domain.value,
            "source_system": winner.source_system,
            "salience": winner.salience,
            "confidence": winner.confidence,
            "description": winner.description,
            "signal_id": winner.signal_id,
            "corroborated": ignition.corroborated,
            "candidate_count": ignition.candidate_count,
            "activation_map": ignition.activation_map,
        }

        try:
            publish_fn("gwt.broadcast", broadcast_payload)
            logger.debug(
                "GWT broadcast: %s (step=%d, corroborated=%s)",
                winner.domain.value,
                step,
                ignition.corroborated,
            )
        except Exception:
            logger.debug(
                "PatternCortex: GWT broadcast failed", exc_info=True,
            )

    # ── Statistics ───────────────────────────────────────────────────

    def get_statistics(self) -> dict[str, Any]:
        """Return cortex statistics for monitoring."""
        workspace_stats = self._workspace.get_statistics()
        return {
            "window_size": len(self._window),
            "window_capacity": WINDOW_SIZE,
            "total_advisories": self._total_advisories,
            "total_trends_detected": self._total_trends,
            "total_meta_patterns": self._total_meta_patterns,
            "total_ancestral_queries": self._total_ancestral_queries,
            "total_ignitions": self._total_ignitions,
            "active_streaks": {
                d.value: c
                for d, c in self._domain_streak.items()
                if c > 0
            },
            "has_memory_bridge": self._memory_bridge is not None,
            "has_event_bus": self._event_bus is not None,
            "workspace": workspace_stats,
        }

    def get_recent_advisories(self, n: int = 5) -> list[PatternAdvisory]:
        """Return the last N advisories."""
        return list(self._recent_advisories)[-n:]

    def __repr__(self) -> str:
        return (
            f"PatternCortex(window={len(self._window)}/{WINDOW_SIZE}, "
            f"advisories={self._total_advisories})"
        )
