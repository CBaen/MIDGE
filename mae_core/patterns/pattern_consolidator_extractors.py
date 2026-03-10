"""Pattern Consolidator extraction methods mixin.

Extracted from pattern_consolidator.py to keep the core class under the 500-line cap.
Import from mae_core.patterns.pattern_consolidator for all public names.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Rule of Three: minimum consecutive occurrences to store a trend
TREND_STORE_THRESHOLD = 3


class _PatternConsolidatorExtractorsMixin:
    """Mixin providing pattern extraction and storage methods.

    Mixed into PatternConsolidator. Requires the following attributes:
        _cortex, _bridge, _distiller
    """

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
