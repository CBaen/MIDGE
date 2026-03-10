"""Pattern Cortex -- trend detection, meta-pattern detection, ancestral recall mixin.

Extracted from pattern_cortex.py to keep the core class under the 500-line cap.
Import from mae_core.patterns.pattern_cortex for all public names.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from mae_core.patterns.pattern_signal import (
    PatternDomain,
    PatternForm,
    PatternSignal,
)

logger = logging.getLogger(__name__)

# Fibonacci window size for short-term pattern memory
WINDOW_SIZE = 13

# Rule of Three: minimum consecutive appearances to qualify as a trend
TREND_THRESHOLD = 3

# How many steps a meta-pattern signal lives
META_TTL = 8


class _PatternCortexDetectionMixin:
    """Mixin providing trend detection, meta-pattern detection, and ancestral recall.

    Mixed into PatternCortex. Requires the following attributes:
        _domain_streak, _recent_advisories, _total_trends,
        _total_meta_patterns, _total_ancestral_queries,
        _window, _memory_bridge
    """

    # ── Trend Detection ──────────────────────────────────────────────

    def _update_domain_streaks(self, digest: Any) -> None:
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

    def _recall_ancestral(self, digest: Any) -> list[dict]:
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

