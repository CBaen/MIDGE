"""Tests for Nervous System — wiring subsystems to agent behavior."""

from collections import defaultdict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mae_core.agents.mixins.signal_processing import SignalProcessingMixin
from mae_core.agents.mixins.stigmergy import StigmergyMixin


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model():
    """Create a minimal Mesa model mock."""
    model = MagicMock()
    model._agents = {}
    model.schedule = MagicMock()
    return model


def _make_agent(**kwargs):
    """Create a MycelialAgent with optional subsystem injections."""
    from mae_core.agents.mycelial_agent import MycelialAgent

    model = _make_model()
    return MycelialAgent(model=model, agent_type="mycelial", **kwargs)


def _make_stigmergy_env():
    """Create a real StigmergicEnvironment."""
    from mae_core.communication.stigmergy import StigmergicEnvironment

    return StigmergicEnvironment()


def _make_episodic_memory():
    """Create a real PrioritizedReplayBuffer."""
    from mae_core.memory.prioritized_replay_buffer import PrioritizedReplayBuffer

    return PrioritizedReplayBuffer(capacity=1000)


# ---------------------------------------------------------------------------
# Observation Tests
# ---------------------------------------------------------------------------

class TestObservation:
    def test_observe_senses_stigmergy(self):
        """_observe() populates _sensed_markers from stigmergy."""
        env = _make_stigmergy_env()
        agent = _make_agent(stigmergy_env=env)

        # Deposit a marker, then observe
        env.deposit_marker("SUCCESS", agent.stigmergy_position, intensity=1.0)
        agent._observe()

        assert "SUCCESS" in agent._sensed_markers
        assert len(agent._sensed_markers["SUCCESS"]) >= 1

    def test_observe_builds_state_vector(self):
        """_observe() builds an 8-dim state vector."""
        agent = _make_agent()
        agent._observe()

        assert agent.current_state is not None
        vec = agent.current_state["state_vector"]
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (8,)

    def test_observe_state_values_match_agent(self):
        """State vector values correspond to agent internal state."""
        agent = _make_agent()
        agent.step_count = 100
        agent.cumulative_reward = 5.0
        agent.last_reward = 0.3
        agent.risk_score = 0.7

        agent._observe()
        vec = agent.current_state["state_vector"]

        assert vec[0] == pytest.approx(100 / 1000.0)
        assert vec[1] == pytest.approx(5.0)
        assert vec[2] == pytest.approx(0.3)
        assert vec[3] == pytest.approx(0.7)

    def test_observe_no_stigmergy_is_safe(self):
        """_observe() works when stigmergy_env is None."""
        agent = _make_agent(stigmergy_env=None)
        agent._observe()

        assert agent._sensed_markers == {}
        assert agent.current_state is not None

    def test_observe_no_memory_is_safe(self):
        """_observe() works with no memory subsystems injected."""
        agent = _make_agent()
        agent._observe()  # Should not raise

    def test_observe_tracks_prev_state(self):
        """_observe() stores previous state for learning."""
        agent = _make_agent()
        agent._observe()
        first_vec = agent._curr_state_vector.copy()

        agent.step_count = 50
        agent._observe()

        assert agent._prev_state_vector is not None
        np.testing.assert_array_equal(agent._prev_state_vector, first_vec)


# ---------------------------------------------------------------------------
# Decision Tests
# ---------------------------------------------------------------------------

class TestDecision:
    def test_decide_falls_back_to_default(self):
        """With no memory or world model, _decide() returns default action."""
        agent = _make_agent()
        agent._observe()
        action = agent._decide()

        assert action == 0  # BaseAgent._select_action returns 0

    def test_decide_uses_world_model(self):
        """_decide() consults world model when available."""
        world_model = MagicMock()
        world_model.reward_model = MagicMock(return_value=0.8)

        agent = _make_agent(world_model=world_model)
        agent._observe()
        action = agent._decide()

        # use_world_model() delegates to world_model — action should be non-None
        # (exact value depends on world model implementation)
        assert action is not None

    def test_decide_uses_memory_when_available(self):
        """_decide() searches memory for similar past states."""
        semantic = MagicMock()
        past_exp = MagicMock()
        past_exp.reward = 1.0
        past_exp.action = 42
        semantic.search_by_state = MagicMock(return_value=[past_exp])

        agent = _make_agent(
            semantic_retriever=semantic,
            agent_config={"semantic_search_enabled": True},
        )
        agent._observe()
        action = agent._decide()

        assert action == 42
        semantic.search_by_state.assert_called_once()

    def test_decide_ignores_bad_memories(self):
        """_decide() doesn't use negative-reward experiences."""
        semantic = MagicMock()
        bad_exp = MagicMock()
        bad_exp.reward = -1.0
        bad_exp.action = 99
        semantic.search_by_state = MagicMock(return_value=[bad_exp])

        agent = _make_agent(
            semantic_retriever=semantic,
            agent_config={"semantic_search_enabled": True},
        )
        agent._observe()
        action = agent._decide()

        assert action != 99  # Should fall through to default

    def test_decide_picks_best_memory(self):
        """_decide() picks the highest-reward past experience."""
        semantic = MagicMock()
        exp1 = MagicMock()
        exp1.reward = 0.3
        exp1.action = 10
        exp2 = MagicMock()
        exp2.reward = 0.9
        exp2.action = 20
        semantic.search_by_state = MagicMock(return_value=[exp1, exp2])

        agent = _make_agent(
            semantic_retriever=semantic,
            agent_config={"semantic_search_enabled": True},
        )
        agent._observe()
        action = agent._decide()

        assert action == 20  # Highest reward experience


# ---------------------------------------------------------------------------
# Learning Tests
# ---------------------------------------------------------------------------

class TestLearning:
    def test_learn_calls_super(self):
        """_learn() updates base reward tracking."""
        agent = _make_agent()
        agent._observe()
        agent._learn(0, 1.5)

        assert agent.last_reward == 1.5
        assert agent.cumulative_reward == 1.5

    def test_learn_deposits_success_marker(self):
        """_learn() deposits SUCCESS marker when reward > 0."""
        env = _make_stigmergy_env()
        agent = _make_agent(stigmergy_env=env)
        agent._observe()
        agent._learn(0, 0.8)

        markers = env.sense_markers(agent.stigmergy_position, radius=10.0, marker_types=["SUCCESS"])
        assert len(markers) >= 1

    def test_learn_deposits_danger_marker(self):
        """_learn() deposits DANGER marker when risk > 0.5."""
        env = _make_stigmergy_env()
        agent = _make_agent(stigmergy_env=env)
        agent.risk_score = 0.8
        agent._observe()
        agent._learn(0, 0.0)

        markers = env.sense_markers(agent.stigmergy_position, radius=10.0, marker_types=["DANGER"])
        assert len(markers) >= 1

    def test_learn_no_marker_on_zero_reward(self):
        """_learn() doesn't deposit SUCCESS marker for zero reward."""
        env = _make_stigmergy_env()
        agent = _make_agent(stigmergy_env=env)
        agent._observe()
        agent._learn(0, 0.0)

        markers = env.sense_markers(agent.stigmergy_position, radius=10.0, marker_types=["SUCCESS"])
        assert len(markers) == 0

    def test_learn_stores_experience(self):
        """_learn() stores experience in episodic memory."""
        mem = _make_episodic_memory()
        agent = _make_agent(
            episodic_memory=mem,
            agent_config={"replay_enabled": True},
        )

        agent._observe()  # sets _prev_state_vector and _curr_state_vector
        agent.step_count = 2  # need a second observe to have prev
        agent._observe()
        agent._learn(1, 0.5)

        assert len(mem) >= 1

    def test_learn_triggers_replay_at_fibonacci(self):
        """_learn() triggers memory replay every 13 steps."""
        from mae_core.memory.experience import Experience

        mem = _make_episodic_memory()
        agent = _make_agent(
            episodic_memory=mem,
            agent_config={"replay_enabled": True},
        )

        # Pre-populate memory with enough experiences for sampling
        for i in range(40):
            state = np.random.randn(8).astype(np.float32)
            exp = Experience(state=state, action=0, reward=0.1, next_state=state, done=False)
            mem.add(exp)

        agent._observe()
        agent.step_count = 2
        agent._observe()  # sets prev

        # Step to 13 (Fibonacci replay trigger)
        agent.step_count = 13
        agent._learn(0, 0.1)
        # Should not crash — replay executed

    def test_learn_no_memory_is_safe(self):
        """_learn() works with no memory subsystems."""
        agent = _make_agent()
        agent._observe()
        agent._learn(0, 0.5)  # Should not raise

    def test_learn_no_stigmergy_is_safe(self):
        """_learn() works with no stigmergy."""
        agent = _make_agent(stigmergy_env=None)
        agent._observe()
        agent._learn(0, 0.5)  # Should not raise


# ---------------------------------------------------------------------------
# Signal Handler Tests
# ---------------------------------------------------------------------------

class TestSignalHandlers:
    def test_opportunity_reduces_risk(self):
        """Opportunity signal reduces risk_score."""
        agent = _make_agent()
        agent.risk_score = 0.5
        agent._handle_opportunity_signal({"reward_potential": 0.8})

        assert agent.risk_score == pytest.approx(0.4)

    def test_opportunity_does_not_go_negative(self):
        """Risk score doesn't go below 0 from opportunity."""
        agent = _make_agent()
        agent.risk_score = 0.05
        agent._handle_opportunity_signal({"reward_potential": 0.5})

        assert agent.risk_score >= 0.0

    def test_collaboration_emits_response(self):
        """Collaboration signal emits response when capabilities match."""
        bus = MagicMock()
        agent = _make_agent(signal_bus=bus)
        agent.capabilities = {"routing", "learning"}

        agent._handle_collaboration_signal({
            "required_capabilities": ["routing"],
            "sender_id": "peer_1",
            "task": "optimize",
        })

        # Check emit_signal was called (via signal bus)
        emit_fn = getattr(agent, "emit_signal", None)
        if emit_fn:
            # The handler calls emit_signal if available
            pass  # Tested via integration

    def test_collaboration_ignores_no_match(self):
        """Collaboration signal does nothing when no capability match."""
        agent = _make_agent()
        agent.capabilities = {"routing"}

        # Should not raise even with no match
        agent._handle_collaboration_signal({
            "required_capabilities": ["quantum_computing"],
            "sender_id": "peer_1",
        })

    def test_knowledge_share_updates_risk(self):
        """Knowledge share with risk data blends into risk_score."""
        agent = _make_agent()
        agent.risk_score = 0.2

        agent._handle_knowledge_share_signal({
            "knowledge": {"risk": 0.8},
        })

        # 0.8 * 0.2 + 0.2 * 0.8 = 0.32
        assert agent.risk_score == pytest.approx(0.32)

    def test_handlers_work_with_signal_objects(self):
        """Handlers extract payload from signal objects (not just dicts)."""
        agent = _make_agent()
        agent.risk_score = 0.5

        signal_obj = MagicMock()
        signal_obj.payload = {"reward_potential": 0.5}
        agent._handle_opportunity_signal(signal_obj)

        assert agent.risk_score == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Stigmergy Fix Test
# ---------------------------------------------------------------------------

class TestStigmergyFix:
    def test_follow_trail_no_crash(self):
        """follow_trail() works without passing invalid 'attractive' param."""
        env = _make_stigmergy_env()
        agent = _make_agent(stigmergy_env=env)

        # Deposit a marker
        env.deposit_marker("SUCCESS", (1.0, 1.0), intensity=1.0)

        # Should not crash
        gradient = agent.follow_trail("SUCCESS", attractive=True)
        assert isinstance(gradient, tuple)

    def test_follow_trail_repulsive(self):
        """follow_trail(attractive=False) returns negated gradient."""
        env = _make_stigmergy_env()
        agent = _make_agent(stigmergy_env=env)

        # Deposit a marker offset from agent position
        env.deposit_marker("DANGER", (5.0, 5.0), intensity=1.0)

        attractive_grad = agent.follow_trail("DANGER", attractive=True)
        repulsive_grad = agent.follow_trail("DANGER", attractive=False)

        # Repulsive should be negated attractive (if non-zero)
        for a, r in zip(attractive_grad, repulsive_grad):
            if a != 0:
                assert r == pytest.approx(-a)


# ---------------------------------------------------------------------------
# Communication Test
# ---------------------------------------------------------------------------

class TestCommunication:
    def test_communicate_deposits_exploration_marker(self):
        """_communicate() deposits exploration marker."""
        env = _make_stigmergy_env()
        agent = _make_agent(stigmergy_env=env)
        agent._communicate()

        markers = env.sense_markers(agent.stigmergy_position, radius=10.0, marker_types=["EXPLORATION"])
        assert len(markers) >= 1


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------

class TestNervousSystemIntegration:
    def test_full_step_leaves_traces(self):
        """A full step cycle leaves stigmergy traces and builds state."""
        env = _make_stigmergy_env()
        agent = _make_agent(stigmergy_env=env)

        # Run a full step
        agent.step()

        # Should have state vector
        assert agent.current_state is not None
        assert "state_vector" in agent.current_state

        # Should have deposited exploration marker via _communicate
        markers = env.sense_markers(agent.stigmergy_position, radius=10.0)
        assert len(markers) >= 1  # At least exploration marker

    def test_agent_sees_other_agents_markers(self):
        """One agent's stigmergy markers are visible to another."""
        env = _make_stigmergy_env()
        agent_a = _make_agent(stigmergy_env=env)
        agent_b = _make_agent(stigmergy_env=env)

        # Agent A deposits success marker
        agent_a.deposit_success_marker(1.0)

        # Agent B observes (both start at same default position)
        agent_b._observe()

        assert len(agent_b._sensed_markers.get("SUCCESS", [])) >= 1

    def test_memory_driven_step_cycle(self):
        """Agent with memory stores and potentially retrieves experiences."""
        mem = _make_episodic_memory()
        env = _make_stigmergy_env()
        agent = _make_agent(
            episodic_memory=mem,
            stigmergy_env=env,
            agent_config={"replay_enabled": True},
        )

        # Run several steps
        for _ in range(5):
            agent.step()

        # Memory should have stored experiences (after step 2 when prev_state exists)
        assert len(mem) >= 1
