"""Tests for ExperienceNarrator — text narration of numerical experiences."""

import time
from unittest.mock import MagicMock
import numpy as np
import pytest

from mae_core.memory.experience import Experience
from mae_core.memory.experience_narrator import (
    DEFAULT_ACTION_NAMES,
    DEFAULT_STATE_DIMS,
    ExperienceNarrator,
    _characterize_reward,
    _level_word,
    _sign_label,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def narrator():
    return ExperienceNarrator()


@pytest.fixture
def agent_context():
    return {
        "agent_id": "agent-3",
        "agent_role": "EXPLORER",
        "parent_id": "colony",
        "peer_count": 4,
        "circadian_phase": "ACTIVE",
        "step": 1000,
        "organ_id": "sensory-system",
    }


@pytest.fixture
def experience():
    return Experience(
        state=np.array([0.8, 0.2, 0.7, 0.5, 0.3, 0.1, 0.6, 0.4]),
        action=0,
        reward=0.85,
        next_state=np.array([0.7, 0.3, 0.5, 0.6, 0.4, 0.1, 0.6, 0.5]),
        done=False,
        info={},
    )


@pytest.fixture
def negative_experience():
    return Experience(
        state=np.array([0.3, 0.8, 0.2, 0.1, 0.1, 0.7, 0.1, 0.2]),
        action=3,
        reward=-0.5,
        next_state=np.array([0.2, 0.9, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1]),
        done=True,
        info={},
    )


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_characterize_reward_positive(self):
        assert _characterize_reward(0.9) == "excellent"
        assert _characterize_reward(0.6) == "good"
        assert _characterize_reward(0.2) == "mild positive"

    def test_characterize_reward_neutral(self):
        assert _characterize_reward(0.0) == "neutral"

    def test_characterize_reward_negative(self):
        assert _characterize_reward(-0.4) == "poor"
        assert _characterize_reward(-0.8) == "very poor"

    def test_sign_label(self):
        assert _sign_label(0.5) == "positive"
        assert _sign_label(0.0) == "neutral"
        assert _sign_label(-0.5) == "negative"

    def test_level_word(self):
        assert _level_word(0.9) == "very high"
        assert _level_word(0.7) == "high"
        assert _level_word(0.5) == "moderate"
        assert _level_word(0.3) == "low"
        assert _level_word(0.1) == "very low"


# ---------------------------------------------------------------------------
# State Characterization
# ---------------------------------------------------------------------------


class TestCharacterizeState:
    def test_basic_state(self, narrator):
        state = np.array([0.9, 0.1, 0.5, 0.3, 0.7, 0.2, 0.8, 0.6])
        desc = narrator.characterize_state(state)
        assert "energy=very high" in desc
        assert "risk=very low" in desc
        assert "exploration_drive=moderate" in desc

    def test_empty_state(self):
        narrator = ExperienceNarrator(state_dim_names={})
        state = np.array([0.5, 0.5])
        desc = narrator.characterize_state(state)
        assert "2-dim state" in desc

    def test_custom_dim_names(self):
        narrator = ExperienceNarrator(state_dim_names={0: "hunger", 1: "thirst"})
        state = np.array([0.8, 0.1])
        desc = narrator.characterize_state(state)
        assert "hunger=very high" in desc
        assert "thirst=very low" in desc


class TestCharacterizeTransition:
    def test_rising_dimension(self, narrator):
        state = np.array([0.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        next_state = np.array([0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        desc = narrator.characterize_transition(state, next_state)
        assert "energy rising" in desc

    def test_falling_dimension(self, narrator):
        state = np.array([0.5, 0.5, 0.8, 0.5, 0.5, 0.5, 0.5, 0.5])
        next_state = np.array([0.5, 0.5, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5])
        desc = narrator.characterize_transition(state, next_state)
        assert "exploration_drive falling" in desc

    def test_stable(self, narrator):
        state = np.array([0.5] * 8)
        desc = narrator.characterize_transition(state, state)
        assert desc == "state stable"


# ---------------------------------------------------------------------------
# Narration
# ---------------------------------------------------------------------------


class TestNarrate:
    def test_basic_narration(self, narrator, experience, agent_context):
        text = narrator.narrate(experience, agent_context)
        assert "Agent agent-3" in text
        assert "EXPLORER" in text
        assert "explore" in text  # action 0
        assert "0.850" in text  # reward
        assert "step 1000" in text
        assert "4 peers" in text

    def test_negative_experience_narration(self, narrator, negative_experience, agent_context):
        text = narrator.narrate(negative_experience, agent_context)
        assert "rest" in text  # action 3
        assert "Episode complete" in text
        assert "poor" in text  # negative reward

    def test_narration_uniqueness(self, narrator, experience, agent_context):
        ctx1 = {**agent_context, "step": 100}
        ctx2 = {**agent_context, "step": 200}
        t1 = narrator.narrate(experience, ctx1)
        t2 = narrator.narrate(experience, ctx2)
        assert t1 != t2  # Different steps produce different narrations

    def test_narrate_batch(self, narrator, experience, agent_context):
        batch = [experience] * 5
        results = narrator.narrate_batch(batch, agent_context)
        assert len(results) == 5
        assert all(isinstance(r, str) for r in results)
        assert all(len(r) > 50 for r in results)

    def test_minimal_context(self, narrator, experience):
        text = narrator.narrate(experience, {})
        assert "Agent unknown" in text
        assert "GENERALIST" in text


# ---------------------------------------------------------------------------
# Consolidation Summary
# ---------------------------------------------------------------------------


class TestConsolidationSummary:
    def test_basic_summary(self, narrator, agent_context):
        experiences = [
            Experience(
                state=np.random.rand(8),
                action=i % 4,
                reward=float(i) * 0.1,
                next_state=np.random.rand(8),
                done=False,
            )
            for i in range(10)
        ]
        text = narrator.narrate_consolidation_summary(1, agent_context, experiences)
        assert "Consolidation #1" in text
        assert "agent-3" in text
        assert "10 experiences" in text
        assert "mean reward" in text

    def test_empty_consolidation(self, narrator, agent_context):
        text = narrator.narrate_consolidation_summary(0, agent_context, [])
        assert "0 experiences" in text


# ---------------------------------------------------------------------------
# Pattern Narration
# ---------------------------------------------------------------------------


class TestPatternNarration:
    def test_basic_pattern(self, narrator):
        pattern = {
            "pattern_type": "behavioral",
            "domain": "exploration",
            "occurrence_count": 15,
            "confidence": 0.87,
            "description": "Explorers find high rewards in low-threat areas",
        }
        text = narrator.narrate_pattern(pattern, ["agent-0", "agent-3", "agent-4"])
        assert "behavioral" in text
        assert "exploration" in text
        assert "15 times" in text
        assert "0.87" in text
        assert "agent-0" in text

    def test_many_agents_truncated(self, narrator):
        agents = [f"agent-{i}" for i in range(10)]
        text = narrator.narrate_pattern(
            {"pattern_type": "temporal", "domain": "rest", "occurrence_count": 5, "confidence": 0.5, "description": "test"},
            agents,
        )
        assert "5 others" in text


# ---------------------------------------------------------------------------
# Payload Building
# ---------------------------------------------------------------------------


class TestBuildPayload:
    def test_payload_fields(self, narrator, experience, agent_context):
        payload = narrator.build_payload(
            experience,
            agent_context,
            witness_hash="abc123",
            consolidation_id=7,
        )
        assert payload["type"] == "episode"
        assert payload["agent_id"] == "agent-3"
        assert payload["agent_role"] == "EXPLORER"
        assert payload["reward_sign"] == "positive"
        assert payload["action_taken"] == 0
        assert payload["witness_hash"] == "abc123"
        assert payload["consolidation_id"] == 7
        assert payload["primary_store"] == "mae_narrative"
        assert payload["verification_store"] == "pickle"
        assert payload["balance_store"] == "mae_ancestral"

    def test_payload_negative_reward(self, narrator, negative_experience, agent_context):
        payload = narrator.build_payload(negative_experience, agent_context)
        assert payload["reward_sign"] == "negative"
        assert payload["episode_complete"] is True
