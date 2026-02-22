"""Pattern Distiller - Extracts recurring patterns from episodic memory.

Analyzes consolidated episodes to find behavioral, structural, temporal,
and causal patterns that transcend individual experiences. These become
ancestral memory -- the collective wisdom of Mae.

Biological analogy: The slow extraction of semantic knowledge from
episodic memories. How "I burned my hand on the stove three times"
becomes "stoves are hot."

Rule of Three: Patterns require at least 3 occurrences to be real.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class PatternDistiller:
    """Extracts and distills patterns from episodes for ancestral storage.

    Finds:
    - Behavioral patterns: action-reward correlations
    - Temporal patterns: time-of-cycle effects
    - State patterns: recurring state configurations
    """

    def __init__(
        self,
        min_occurrences: int = 3,  # Rule of Three
        confidence_threshold: float = 0.5,
    ) -> None:
        self._min_occurrences = min_occurrences
        self._confidence_threshold = confidence_threshold
        self._total_distilled = 0

    def distill(
        self,
        experiences: list[Any],
        agent_contexts: dict[str, dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Extract all pattern types from a batch of experiences.

        Returns list of pattern dicts ready for ancestral storage.
        """
        if len(experiences) < self._min_occurrences:
            return []

        patterns: list[dict[str, Any]] = []
        patterns.extend(self.detect_behavioral_patterns(experiences))
        patterns.extend(self.detect_state_patterns(experiences))

        self._total_distilled += len(patterns)
        return patterns

    def detect_behavioral_patterns(
        self,
        experiences: list[Any],
    ) -> list[dict[str, Any]]:
        """Find action-reward correlations.

        Example: "Action 0 (explore) consistently yields positive reward"
        """
        # Group experiences by action
        by_action: dict[Any, list[float]] = defaultdict(list)
        for exp in experiences:
            action = exp.action
            by_action[action].append(float(exp.reward))

        patterns = []
        for action, rewards in by_action.items():
            if len(rewards) < self._min_occurrences:
                continue

            mean_reward = float(np.mean(rewards))
            std_reward = float(np.std(rewards))

            # Confidence: how consistent is the reward direction?
            if std_reward > 0:
                # Signal-to-noise ratio as confidence proxy
                confidence = min(1.0, abs(mean_reward) / (std_reward + 1e-6))
            else:
                confidence = 1.0 if abs(mean_reward) > 0.01 else 0.0

            if confidence < self._confidence_threshold:
                continue

            direction = "positive" if mean_reward > 0.01 else "negative" if mean_reward < -0.01 else "neutral"

            patterns.append({
                "pattern_type": "behavioral",
                "domain": f"action-{action}",
                "occurrence_count": len(rewards),
                "confidence": round(confidence, 3),
                "description": (
                    f"Action {action} yields {direction} reward "
                    f"(mean={mean_reward:.3f}, std={std_reward:.3f}) "
                    f"across {len(rewards)} experiences"
                ),
                "mean_reward": round(mean_reward, 4),
                "std_reward": round(std_reward, 4),
                "action": action,
                "applicable_roles": [],  # Will be filled by caller
            })

        return patterns

    def detect_state_patterns(
        self,
        experiences: list[Any],
    ) -> list[dict[str, Any]]:
        """Find recurring state configurations correlated with outcomes.

        Clusters states by their dominant dimension and checks reward correlation.
        """
        if not experiences or len(experiences) < self._min_occurrences:
            return []

        # Find dominant state dimension for each experience
        dominant_dims: dict[int, list[float]] = defaultdict(list)
        for exp in experiences:
            state = np.asarray(exp.state)
            if state.size == 0:
                continue
            # Dominant = dimension with highest value
            dominant_idx = int(np.argmax(state[:8]))  # Only check key dims
            dominant_dims[dominant_idx].append(float(exp.reward))

        patterns = []
        for dim_idx, rewards in dominant_dims.items():
            if len(rewards) < self._min_occurrences:
                continue

            mean_reward = float(np.mean(rewards))
            confidence = min(1.0, abs(mean_reward) / (float(np.std(rewards)) + 1e-6))

            if confidence < self._confidence_threshold:
                continue

            direction = "positive" if mean_reward > 0.01 else "negative" if mean_reward < -0.01 else "neutral"

            patterns.append({
                "pattern_type": "state",
                "domain": f"dim-{dim_idx}-dominant",
                "occurrence_count": len(rewards),
                "confidence": round(confidence, 3),
                "description": (
                    f"When state dimension {dim_idx} is dominant, "
                    f"outcomes are {direction} "
                    f"(mean reward={mean_reward:.3f}) "
                    f"across {len(rewards)} experiences"
                ),
                "mean_reward": round(mean_reward, 4),
                "dominant_dimension": dim_idx,
                "applicable_roles": [],
            })

        return patterns

    def merge_with_existing(
        self,
        new_patterns: list[dict[str, Any]],
        existing_patterns: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge new patterns with existing ones.

        If a pattern with the same domain already exists, update its
        occurrence count and confidence. Otherwise add as new.
        """
        existing_by_key: dict[str, dict[str, Any]] = {}
        for p in existing_patterns:
            key = f"{p.get('pattern_type', '')}:{p.get('domain', '')}"
            existing_by_key[key] = p

        merged = list(existing_patterns)  # Start with existing

        for new_p in new_patterns:
            key = f"{new_p.get('pattern_type', '')}:{new_p.get('domain', '')}"
            if key in existing_by_key:
                # Update existing pattern
                existing = existing_by_key[key]
                existing["occurrence_count"] = (
                    existing.get("occurrence_count", 0)
                    + new_p.get("occurrence_count", 0)
                )
                # Weighted average of confidence
                old_conf = existing.get("confidence", 0.5)
                new_conf = new_p.get("confidence", 0.5)
                old_n = existing.get("occurrence_count", 1) - new_p.get("occurrence_count", 0)
                new_n = new_p.get("occurrence_count", 1)
                if old_n + new_n > 0:
                    existing["confidence"] = round(
                        (old_conf * old_n + new_conf * new_n) / (old_n + new_n), 3
                    )
                # Update description
                existing["description"] = new_p.get("description", existing["description"])
            else:
                merged.append(new_p)

        return merged

    @property
    def total_distilled(self) -> int:
        return self._total_distilled

    def __repr__(self) -> str:
        return (
            f"PatternDistiller(min_occurrences={self._min_occurrences}, "
            f"total_distilled={self._total_distilled})"
        )
