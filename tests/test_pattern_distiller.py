"""Tests for PatternDistiller — ancestral memory extraction.

Verifies behavioral pattern detection, state pattern detection,
merge logic, and Rule of Three enforcement.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from mae_core.memory.pattern_distiller import PatternDistiller


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class MockExperience:
    """Minimal experience-like object for testing."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray = field(default_factory=lambda: np.zeros(8))
    done: bool = False


def _make_exp(action: int, reward: float, dominant_dim: int = 0) -> MockExperience:
    """Create experience with a specific dominant state dimension."""
    state = np.zeros(8, dtype=np.float32)
    state[dominant_dim] = 1.0  # Make this dim dominant
    return MockExperience(state=state, action=action, reward=reward)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_defaults(self):
        pd = PatternDistiller()
        assert pd._min_occurrences == 3
        assert pd._confidence_threshold == 0.5
        assert pd.total_distilled == 0

    def test_custom_params(self):
        pd = PatternDistiller(min_occurrences=5, confidence_threshold=0.8)
        assert pd._min_occurrences == 5
        assert pd._confidence_threshold == 0.8

    def test_repr(self):
        pd = PatternDistiller()
        r = repr(pd)
        assert "PatternDistiller" in r
        assert "min_occurrences=3" in r
        assert "total_distilled=0" in r


# ---------------------------------------------------------------------------
# Behavioral Patterns
# ---------------------------------------------------------------------------


class TestBehavioralPatterns:
    def test_detects_consistent_positive_action(self):
        """Action with consistent positive reward should be detected."""
        pd = PatternDistiller()
        exps = [_make_exp(action=0, reward=0.8) for _ in range(5)]
        patterns = pd.detect_behavioral_patterns(exps)
        assert len(patterns) == 1
        p = patterns[0]
        assert p["pattern_type"] == "behavioral"
        assert p["domain"] == "action-0"
        assert p["occurrence_count"] == 5
        assert p["confidence"] == 1.0  # zero std, positive mean
        assert p["mean_reward"] > 0
        assert "positive" in p["description"]

    def test_detects_consistent_negative_action(self):
        pd = PatternDistiller()
        exps = [_make_exp(action=1, reward=-0.5) for _ in range(4)]
        patterns = pd.detect_behavioral_patterns(exps)
        assert len(patterns) == 1
        assert "negative" in patterns[0]["description"]
        assert patterns[0]["mean_reward"] < 0

    def test_rule_of_three_enforced(self):
        """Fewer than 3 occurrences of an action -> no pattern."""
        pd = PatternDistiller()
        exps = [_make_exp(action=0, reward=0.9), _make_exp(action=0, reward=0.8)]
        patterns = pd.detect_behavioral_patterns(exps)
        assert patterns == []

    def test_multiple_actions_detected(self):
        pd = PatternDistiller()
        exps = (
            [_make_exp(action=0, reward=0.8) for _ in range(4)]
            + [_make_exp(action=1, reward=-0.6) for _ in range(3)]
        )
        patterns = pd.detect_behavioral_patterns(exps)
        assert len(patterns) == 2
        domains = {p["domain"] for p in patterns}
        assert "action-0" in domains
        assert "action-1" in domains

    def test_low_confidence_filtered(self):
        """High variance rewards -> low confidence -> filtered out."""
        pd = PatternDistiller(confidence_threshold=0.8)
        # Rewards that average near zero with high variance
        rewards = [1.0, -1.0, 0.5, -0.5, 0.8, -0.8]
        exps = [_make_exp(action=0, reward=r) for r in rewards]
        patterns = pd.detect_behavioral_patterns(exps)
        assert patterns == []

    def test_neutral_direction(self):
        """Mean reward near zero -> neutral direction."""
        pd = PatternDistiller(confidence_threshold=0.0)  # Accept all
        exps = [_make_exp(action=0, reward=0.001) for _ in range(5)]
        patterns = pd.detect_behavioral_patterns(exps)
        assert len(patterns) == 1
        assert "neutral" in patterns[0]["description"]

    def test_pattern_fields_complete(self):
        """All required fields present in output."""
        pd = PatternDistiller()
        exps = [_make_exp(action=2, reward=0.5) for _ in range(3)]
        patterns = pd.detect_behavioral_patterns(exps)
        assert len(patterns) == 1
        p = patterns[0]
        required = {
            "pattern_type", "domain", "occurrence_count", "confidence",
            "description", "mean_reward", "std_reward", "action",
            "applicable_roles",
        }
        assert required.issubset(set(p.keys()))


# ---------------------------------------------------------------------------
# State Patterns
# ---------------------------------------------------------------------------


class TestStatePatterns:
    def test_detects_dominant_dimension_pattern(self):
        """When dim 0 is dominant and reward is consistently high."""
        pd = PatternDistiller()
        exps = [_make_exp(action=0, reward=0.7, dominant_dim=0) for _ in range(5)]
        patterns = pd.detect_state_patterns(exps)
        assert len(patterns) == 1
        p = patterns[0]
        assert p["pattern_type"] == "state"
        assert p["domain"] == "dim-0-dominant"
        assert p["dominant_dimension"] == 0
        assert "positive" in p["description"]

    def test_multiple_dimensions(self):
        pd = PatternDistiller()
        exps = (
            [_make_exp(action=0, reward=0.9, dominant_dim=2) for _ in range(4)]
            + [_make_exp(action=0, reward=-0.7, dominant_dim=5) for _ in range(3)]
        )
        patterns = pd.detect_state_patterns(exps)
        assert len(patterns) == 2
        dims = {p["dominant_dimension"] for p in patterns}
        assert 2 in dims
        assert 5 in dims

    def test_rule_of_three_for_state(self):
        pd = PatternDistiller()
        exps = [_make_exp(action=0, reward=0.9, dominant_dim=3) for _ in range(2)]
        patterns = pd.detect_state_patterns(exps)
        assert patterns == []

    def test_empty_state_skipped(self):
        pd = PatternDistiller()
        exps = [MockExperience(state=np.array([]), action=0, reward=0.5) for _ in range(5)]
        patterns = pd.detect_state_patterns(exps)
        assert patterns == []

    def test_empty_experiences(self):
        pd = PatternDistiller()
        assert pd.detect_state_patterns([]) == []

    def test_state_pattern_fields_complete(self):
        pd = PatternDistiller()
        exps = [_make_exp(action=0, reward=0.6, dominant_dim=1) for _ in range(3)]
        patterns = pd.detect_state_patterns(exps)
        assert len(patterns) == 1
        required = {
            "pattern_type", "domain", "occurrence_count", "confidence",
            "description", "mean_reward", "dominant_dimension",
            "applicable_roles",
        }
        assert required.issubset(set(patterns[0].keys()))


# ---------------------------------------------------------------------------
# Distill (combined)
# ---------------------------------------------------------------------------


class TestDistill:
    def test_combines_behavioral_and_state(self):
        pd = PatternDistiller()
        exps = [_make_exp(action=0, reward=0.8, dominant_dim=0) for _ in range(5)]
        patterns = pd.distill(exps)
        types = {p["pattern_type"] for p in patterns}
        assert "behavioral" in types
        assert "state" in types

    def test_too_few_experiences(self):
        pd = PatternDistiller(min_occurrences=3)
        exps = [_make_exp(action=0, reward=0.5) for _ in range(2)]
        assert pd.distill(exps) == []

    def test_tracks_total_distilled(self):
        pd = PatternDistiller()
        exps = [_make_exp(action=0, reward=0.8, dominant_dim=0) for _ in range(5)]
        patterns = pd.distill(exps)
        assert pd.total_distilled == len(patterns)
        # Second call accumulates
        pd.distill(exps)
        assert pd.total_distilled == len(patterns) * 2

    def test_agent_contexts_accepted(self):
        """agent_contexts param accepted without error."""
        pd = PatternDistiller()
        exps = [_make_exp(action=0, reward=0.5) for _ in range(3)]
        patterns = pd.distill(exps, agent_contexts={"agent-3": {"role": "EXPLORER"}})
        assert isinstance(patterns, list)


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


class TestMerge:
    def test_merge_new_pattern(self):
        pd = PatternDistiller()
        existing = [{"pattern_type": "behavioral", "domain": "action-0", "occurrence_count": 5, "confidence": 0.9, "description": "old"}]
        new = [{"pattern_type": "behavioral", "domain": "action-1", "occurrence_count": 3, "confidence": 0.7, "description": "new"}]
        merged = pd.merge_with_existing(new, existing)
        assert len(merged) == 2

    def test_merge_updates_existing(self):
        pd = PatternDistiller()
        existing = [{"pattern_type": "behavioral", "domain": "action-0", "occurrence_count": 5, "confidence": 0.9, "description": "old"}]
        new = [{"pattern_type": "behavioral", "domain": "action-0", "occurrence_count": 3, "confidence": 0.7, "description": "updated"}]
        merged = pd.merge_with_existing(new, existing)
        assert len(merged) == 1
        assert merged[0]["occurrence_count"] == 8
        assert merged[0]["description"] == "updated"

    def test_merge_empty_existing(self):
        pd = PatternDistiller()
        new = [{"pattern_type": "state", "domain": "dim-0", "occurrence_count": 4, "confidence": 0.8, "description": "new"}]
        merged = pd.merge_with_existing(new, [])
        assert len(merged) == 1

    def test_merge_empty_new(self):
        pd = PatternDistiller()
        existing = [{"pattern_type": "behavioral", "domain": "action-0", "occurrence_count": 5, "confidence": 0.9, "description": "old"}]
        merged = pd.merge_with_existing([], existing)
        assert len(merged) == 1
        assert merged[0]["description"] == "old"

    def test_merge_different_types_same_domain(self):
        """behavioral:action-0 and state:action-0 are distinct keys."""
        pd = PatternDistiller()
        existing = [{"pattern_type": "behavioral", "domain": "action-0", "occurrence_count": 5, "confidence": 0.9, "description": "beh"}]
        new = [{"pattern_type": "state", "domain": "action-0", "occurrence_count": 3, "confidence": 0.7, "description": "state"}]
        merged = pd.merge_with_existing(new, existing)
        assert len(merged) == 2
