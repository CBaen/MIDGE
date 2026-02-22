"""Triage Classifier - Medical-grade signal urgency classification.

Biological analogy: Emergency triage (START protocol) — incoming signals
classified by biological urgency (IMMEDIATE/URGENT/DELAYED/EXPECTANT).
Integrates pain (nociception), threat state (threat detector), and
arousal (endocrine) to provide a unified urgency signal that boosts
SignalPriorityResolver scores.

Connection points:
- Reads NociceptionSystem: is_in_pain(), get_pain_load()
- Reads ThreatDetector: is_shelled, get_active_threats()
- Reads EndocrineSystem: is_stressed(), get_urgency_level()
- Consumed by SignalPriorityResolver._compute_score() as urgency boost

Law compliance:
- Law 8 (Consciousness): Property 7 — competition/selection (biological
  urgency forces triage between competing signals)
- Law 8 (Consciousness): Property 8 — prediction/error-correction (pain
  and threat state are prediction error signals about body integrity)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class TriageCategory(Enum):
    """Medical triage categories applied to signal urgency."""

    IMMEDIATE = "immediate"  # Red: life-threatening, needs instant action
    URGENT = "urgent"  # Yellow: serious, but can wait one beat
    DELAYED = "delayed"  # Green: stable, handle when capacity allows
    EXPECTANT = "expectant"  # Black: too late or too costly to address now


# Category -> urgency boost added to SignalPriorityResolver score
_CATEGORY_BOOST: dict[TriageCategory, float] = {
    TriageCategory.IMMEDIATE: 0.30,
    TriageCategory.URGENT: 0.15,
    TriageCategory.DELAYED: 0.00,
    TriageCategory.EXPECTANT: -0.10,
}


@dataclass
class TriageResult:
    """Outcome of classifying a signal through biological triage."""

    category: TriageCategory
    urgency_boost: float  # Added to SignalPriorityResolver score
    reason: str


class TriageClassifier:
    """Classifies signal urgency using biological state from three systems.

    Medical triage rules:
    - IMMEDIATE: pain overload (>1.5) OR critical threat + high-priority signal
    - URGENT: in pain + moderate signal, OR under active defense, OR stressed + high signal
    - EXPECTANT: low-priority signal during pain overload (shed load)
    - DELAYED: everything else (stable state, normal processing)

    Args:
        nociception: NociceptionSystem (optional — graceful without).
        threat_detector: ThreatDetector (optional — graceful without).
        endocrine: EndocrineSystem (optional — graceful without).
    """

    def __init__(
        self,
        nociception: Any = None,
        threat_detector: Any = None,
        endocrine: Any = None,
    ) -> None:
        self._nociception = nociception
        self._threat_detector = threat_detector
        self._endocrine = endocrine

        # Counters
        self._total_classified: int = 0
        self._category_counts: dict[str, int] = {c.value: 0 for c in TriageCategory}

    def classify(self, signal_type: str, signal_priority: float) -> TriageResult:
        """Classify a signal's urgency based on current biological state.

        Args:
            signal_type: The signal type name (DANGER, OPPORTUNITY, etc.).
            signal_priority: The signal's raw priority [0, 1].

        Returns:
            TriageResult with category, urgency_boost, and reason.
        """
        # Gather biological state (graceful if systems unavailable)
        pain_load = 0.0
        in_pain = False
        if self._nociception is not None:
            try:
                pain_load = self._nociception.get_pain_load()
                in_pain = self._nociception.is_in_pain()
            except Exception:
                pass

        threat_score = 0.0
        under_threat = False
        if self._threat_detector is not None:
            try:
                under_threat = self._threat_detector.is_shelled
                active = self._threat_detector.get_active_threats()
                if active:
                    threat_score = max(t.get("score", 0.0) for t in active)
            except Exception:
                pass

        is_stressed = False
        urgency_level = 0.0
        if self._endocrine is not None:
            try:
                is_stressed = self._endocrine.is_stressed()
                urgency_level = self._endocrine.get_urgency_level()
            except Exception:
                pass

        # Classification (medical triage rules)
        result = self._apply_rules(
            signal_type, signal_priority,
            pain_load, in_pain,
            threat_score, under_threat,
            is_stressed, urgency_level,
        )

        # Update counters
        self._total_classified += 1
        self._category_counts[result.category.value] += 1

        return result

    def _apply_rules(
        self,
        signal_type: str,
        priority: float,
        pain_load: float,
        in_pain: bool,
        threat_score: float,
        under_threat: bool,
        is_stressed: bool,
        urgency_level: float,
    ) -> TriageResult:
        """Apply triage classification rules in priority order."""

        # IMMEDIATE: pain overload OR critical threat + high-priority signal
        if pain_load > 1.5:
            return TriageResult(
                TriageCategory.IMMEDIATE,
                _CATEGORY_BOOST[TriageCategory.IMMEDIATE],
                "pain_overload",
            )
        if threat_score > 0.8 and priority > 0.7:
            return TriageResult(
                TriageCategory.IMMEDIATE,
                _CATEGORY_BOOST[TriageCategory.IMMEDIATE],
                "critical_threat_high_priority",
            )

        # URGENT: in pain + moderate signal, active defense, or stressed + high signal
        if in_pain and priority > 0.5:
            return TriageResult(
                TriageCategory.URGENT,
                _CATEGORY_BOOST[TriageCategory.URGENT],
                "pain_moderate_priority",
            )
        if under_threat and priority > 0.4:
            return TriageResult(
                TriageCategory.URGENT,
                _CATEGORY_BOOST[TriageCategory.URGENT],
                "active_defense",
            )
        if is_stressed and priority > 0.5:
            return TriageResult(
                TriageCategory.URGENT,
                _CATEGORY_BOOST[TriageCategory.URGENT],
                "stressed_high_priority",
            )

        # EXPECTANT: low-priority during pain overload (shed load)
        if pain_load > 1.0 and priority < 0.3:
            return TriageResult(
                TriageCategory.EXPECTANT,
                _CATEGORY_BOOST[TriageCategory.EXPECTANT],
                "low_priority_during_overload",
            )

        # DELAYED: stable, normal processing
        return TriageResult(
            TriageCategory.DELAYED,
            _CATEGORY_BOOST[TriageCategory.DELAYED],
            "stable",
        )

    def get_statistics(self) -> dict[str, Any]:
        """Return triage classification statistics."""
        return {
            "total_classified": self._total_classified,
            "category_counts": dict(self._category_counts),
            "nociception_available": self._nociception is not None,
            "threat_detector_available": self._threat_detector is not None,
            "endocrine_available": self._endocrine is not None,
        }

    def __repr__(self) -> str:
        return (
            f"TriageClassifier(classified={self._total_classified}, "
            f"noci={'yes' if self._nociception else 'no'}, "
            f"threat={'yes' if self._threat_detector else 'no'}, "
            f"endo={'yes' if self._endocrine else 'no'})"
        )
