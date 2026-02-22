"""Tests for Codebase Audit Fixes — 15 bugs found by audit.

Round 1 (9 bugs): crash bugs, truthiness bugs, orphan subsystems, stubs, dead code.
Round 2 (6 bugs): signal param mismatches, argument ordering, dead stats code.
"""

from collections import deque
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model():
    model = MagicMock()
    model._agents = {}
    model.schedule = MagicMock()
    return model


def _make_agent(**kwargs):
    from mae_core.agents.mycelial_agent import MycelialAgent
    model = _make_model()
    return MycelialAgent(model=model, agent_type="mycelial", **kwargs)


# ---------------------------------------------------------------------------
# Fix 1: CollectiveConsensusMixin.sense_quorum uses sense_all()
# ---------------------------------------------------------------------------

class TestFix1SenseQuorum:
    def test_sense_quorum_calls_sense_all(self):
        """sense_quorum() calls sense_all(), not sense()."""
        quorum = MagicMock()
        quorum.sense_all = MagicMock(return_value={"DANGER": 0.8, "FOOD": 0.2})

        agent = _make_agent(
            quorum_sensor=quorum,
            agent_config={"quorum_sensing_enabled": True},
        )
        result = agent.sense_quorum()

        quorum.sense_all.assert_called_once()
        assert result is not None
        assert result["concentrations"] == {"DANGER": 0.8, "FOOD": 0.2}

    def test_sense_quorum_thresholds(self):
        """Thresholds are checked correctly from sense_all dict."""
        quorum = MagicMock()
        quorum.sense_all = MagicMock(return_value={"DANGER": 0.8, "FOOD": 0.2})

        agent = _make_agent(
            quorum_sensor=quorum,
            agent_config={"quorum_sensing_enabled": True},
        )
        result = agent.sense_quorum()

        # DANGER (0.8) >= default threshold (0.5) → met
        # FOOD (0.2) < default threshold (0.5) → not met
        assert "DANGER" in result["thresholds_met"]
        assert "FOOD" not in result["thresholds_met"]

    def test_sense_quorum_disabled(self):
        """sense_quorum() returns None when disabled."""
        quorum = MagicMock()
        agent = _make_agent(
            quorum_sensor=quorum,
            agent_config={"quorum_sensing_enabled": False},
        )
        assert agent.sense_quorum() is None

    def test_sense_quorum_no_sensor(self):
        """sense_quorum() returns None when no sensor."""
        agent = _make_agent()
        assert agent.sense_quorum() is None


# ---------------------------------------------------------------------------
# Fix 2: store_policy_for_transfer correct args
# ---------------------------------------------------------------------------

class TestFix2StorePolicy:
    def test_store_policy_passes_correct_args(self):
        """store_policy_for_transfer passes task_id, agent_id, policy_state."""
        kb = MagicMock()
        agent = _make_agent(knowledge_base=kb)

        task = MagicMock()
        task.task_id = "task-42"
        agent.current_task = task

        agent.store_policy_for_transfer({"weights": [1, 2, 3]})

        kb.store_policy.assert_called_once()
        call_kwargs = kb.store_policy.call_args
        # Should use keyword args matching KnowledgeBase.store_policy signature
        assert call_kwargs.kwargs["task_id"] == "task-42"
        assert call_kwargs.kwargs["agent_id"] == str(agent.unique_id)
        assert call_kwargs.kwargs["policy_state"] == {"weights": [1, 2, 3]}

    def test_store_policy_wraps_non_dict(self):
        """Non-dict policy is wrapped in a dict."""
        kb = MagicMock()
        agent = _make_agent(knowledge_base=kb)
        agent.current_task = MagicMock(task_id="t1")

        agent.store_policy_for_transfer("raw_policy_blob")

        call_kwargs = kb.store_policy.call_args
        assert call_kwargs.kwargs["policy_state"] == {"weights": "raw_policy_blob"}


# ---------------------------------------------------------------------------
# Fix 3: store_value_function_for_transfer correct args
# ---------------------------------------------------------------------------

class TestFix3StoreValueFunction:
    def test_store_value_function_passes_correct_args(self):
        """store_value_function_for_transfer passes task_id, agent_id, value_state."""
        kb = MagicMock()
        agent = _make_agent(knowledge_base=kb)
        agent.current_task = MagicMock(task_id="task-99")

        agent.store_value_function_for_transfer({"v_weights": [4, 5]})

        kb.store_value_function.assert_called_once()
        call_kwargs = kb.store_value_function.call_args
        assert call_kwargs.kwargs["task_id"] == "task-99"
        assert call_kwargs.kwargs["agent_id"] == str(agent.unique_id)
        assert call_kwargs.kwargs["value_state"] == {"v_weights": [4, 5]}


# ---------------------------------------------------------------------------
# Fix 4: use_world_model calls world_model.step() not .reward_model()
# ---------------------------------------------------------------------------

class TestFix4WorldModel:
    def test_use_world_model_calls_step(self):
        """use_world_model() calls world_model.step(), not .reward_model()."""
        pred = MagicMock()
        pred.reward = 0.5

        wm = MagicMock()
        wm.step = MagicMock(return_value=pred)

        agent = _make_agent(
            world_model=wm,
            agent_config={"world_model_enabled": True, "num_actions": 3},
        )
        agent._observe()
        action = agent.use_world_model()

        assert wm.step.called
        assert not hasattr(wm, "reward_model") or not wm.reward_model.called
        assert action is not None

    def test_use_world_model_picks_best_action(self):
        """use_world_model() returns action with highest predicted reward."""
        rewards = {0: 0.1, 1: 0.9, 2: 0.3}

        def mock_step(state, action_idx, deterministic=True):
            pred = MagicMock()
            pred.reward = rewards.get(action_idx, 0.0)
            return pred

        wm = MagicMock()
        wm.step = MagicMock(side_effect=mock_step)

        agent = _make_agent(
            world_model=wm,
            agent_config={"world_model_enabled": True, "num_actions": 3},
        )
        agent._observe()
        action = agent.use_world_model()

        assert action == 1  # Highest reward

    def test_use_world_model_disabled_fallback(self):
        """Disabled world model falls back to _select_action."""
        wm = MagicMock()
        agent = _make_agent(
            world_model=wm,
            agent_config={"world_model_enabled": False},
        )
        agent._observe()
        action = agent.use_world_model()

        assert action is not None
        wm.step.assert_not_called()


# ---------------------------------------------------------------------------
# Fix 5: episodic_memory consolidation truthiness
# ---------------------------------------------------------------------------

class TestFix5Truthiness:
    def test_consolidate_checks_empty_memory(self):
        """consolidate_memory() doesn't skip empty buffer due to truthiness."""
        from mae_core.memory.prioritized_replay_buffer import PrioritizedReplayBuffer

        mem = PrioritizedReplayBuffer(capacity=1000)
        consolidator = MagicMock()
        consolidator.min_memory_size = 10
        consolidator.consolidate = MagicMock(return_value=MagicMock(loss_reduction=0.1))

        agent = _make_agent(
            episodic_memory=mem,
            memory_consolidator=consolidator,
            agent_config={"consolidation_enabled": True},
        )

        # Memory is empty (len=0, falsy). With the fix, the min_size
        # check should still fire and return None (not enough data).
        result = agent.consolidate_memory()
        assert result is None
        consolidator.consolidate.assert_not_called()

    def test_consolidate_works_with_enough_data(self):
        """consolidate_memory() works when buffer has enough data."""
        from mae_core.memory.prioritized_replay_buffer import PrioritizedReplayBuffer
        from mae_core.memory.experience import Experience

        mem = PrioritizedReplayBuffer(capacity=1000)
        # Fill with 15 experiences
        for _ in range(15):
            exp = Experience(
                state=np.zeros(8, dtype=np.float32),
                action=0, reward=0.1,
                next_state=np.zeros(8, dtype=np.float32),
                done=False,
            )
            mem.add(exp)

        consolidator = MagicMock()
        consolidator.min_memory_size = 10
        consolidator.consolidate = MagicMock(return_value=MagicMock(loss_reduction=0.1))

        agent = _make_agent(
            episodic_memory=mem,
            memory_consolidator=consolidator,
            agent_config={"consolidation_enabled": True},
        )

        result = agent.consolidate_memory()
        consolidator.consolidate.assert_called_once()


# ---------------------------------------------------------------------------
# Fix 6 & 7: DecisionRouter and CausalEngine injection
# ---------------------------------------------------------------------------

class TestFix67OrphanInjection:
    def test_decision_router_injected(self):
        """MycelialAgent stores decision_router when provided."""
        dr = MagicMock()
        agent = _make_agent(decision_router=dr)
        assert agent.decision_router is dr

    def test_causal_engine_injected(self):
        """MycelialAgent stores causal_engine when provided."""
        ce = MagicMock()
        agent = _make_agent(causal_engine=ce)
        assert agent.causal_engine is ce

    def test_both_default_to_none(self):
        """Without injection, both are None."""
        agent = _make_agent()
        assert agent.decision_router is None
        assert agent.causal_engine is None

    @patch("dotenv.load_dotenv")
    def test_main_injects_both(self, _mock_dotenv):
        """main.py create_mae() injects decision_router and causal_engine."""
        from main import create_mae

        model, systems = create_mae(num_agents=3)
        agents = systems["agents"]
        per_agent = systems["per_agent_systems"]

        for agent in agents:
            assert agent.decision_router is not None, \
                f"Agent {agent.unique_id} missing decision_router"
            assert agent.causal_engine is not None, \
                f"Agent {agent.unique_id} missing causal_engine"

            # Verify they match what was created per-agent
            expected_dr = per_agent[agent.unique_id]["decision_router"]
            expected_ce = per_agent[agent.unique_id]["causal_engine"]
            assert agent.decision_router is expected_dr
            assert agent.causal_engine is expected_ce


# ---------------------------------------------------------------------------
# Fix 8: _learn_from_batch uses reward-based TD errors
# ---------------------------------------------------------------------------

class TestFix8LearnFromBatch:
    def test_learn_from_batch_uses_rewards(self):
        """_learn_from_batch() returns reward-based TD errors, not random."""
        agent = _make_agent()

        batch = []
        for r in [0.5, -0.3, 1.0, 0.0]:
            exp = MagicMock()
            exp.reward = r
            batch.append(exp)

        weights = np.ones(4) / 4.0
        td_errors, loss = agent._learn_from_batch(batch, weights)

        # TD errors should match rewards exactly
        np.testing.assert_array_almost_equal(td_errors, [0.5, -0.3, 1.0, 0.0])
        # Loss is weighted mean of absolute TD errors
        expected_loss = np.mean(np.abs([0.5, -0.3, 1.0, 0.0]) * weights)
        assert loss == pytest.approx(expected_loss)

    def test_learn_from_batch_deterministic(self):
        """Calling twice with same batch gives same result (not random)."""
        agent = _make_agent()

        batch = [MagicMock(reward=0.7), MagicMock(reward=-0.2)]
        weights = np.array([0.6, 0.4])

        td1, loss1 = agent._learn_from_batch(batch, weights)
        td2, loss2 = agent._learn_from_batch(batch, weights)

        np.testing.assert_array_equal(td1, td2)
        assert loss1 == loss2

    def test_learn_from_batch_missing_reward(self):
        """Experiences without reward attribute default to 0.0."""
        agent = _make_agent()

        exp = MagicMock(spec=[])  # No attributes
        batch = [exp]
        weights = np.array([1.0])

        td_errors, loss = agent._learn_from_batch(batch, weights)
        assert td_errors[0] == 0.0


# ---------------------------------------------------------------------------
# Fix 9: store_episode_for_transfer actually stores
# ---------------------------------------------------------------------------

class TestFix9StoreEpisode:
    def test_store_episode_calls_knowledge_base(self):
        """store_episode_for_transfer() calls knowledge_base.store_episode()."""
        kb = MagicMock()
        kb.store_episode = MagicMock(return_value="ep-42")

        agent = _make_agent(knowledge_base=kb)
        agent.current_task = MagicMock(task_id="task-1")
        agent.episode_transitions = deque([
            {"state": np.zeros(4), "action": 0, "reward": 0.5},
            {"state": np.ones(4), "action": 1, "reward": 1.0},
        ])

        episode_id = agent.store_episode_for_transfer(total_reward=1.5, success=True)

        assert episode_id == "ep-42"
        kb.store_episode.assert_called_once()
        call_kwargs = kb.store_episode.call_args.kwargs
        assert call_kwargs["task_id"] == "task-1"
        assert call_kwargs["total_reward"] == 1.5
        assert call_kwargs["success"] is True
        assert len(call_kwargs["transitions"]) == 2

    def test_store_episode_clears_buffer(self):
        """store_episode_for_transfer() clears episode_transitions after storing."""
        kb = MagicMock()
        kb.store_episode = MagicMock(return_value="ep-1")

        agent = _make_agent(knowledge_base=kb)
        agent.current_task = MagicMock(task_id="t")
        agent.episode_transitions = deque([{"x": 1}])

        agent.store_episode_for_transfer(total_reward=1.0, success=True)
        assert len(agent.episode_transitions) == 0

    def test_store_episode_no_clear(self):
        """store_episode_for_transfer(clear_buffer=False) keeps buffer."""
        kb = MagicMock()
        kb.store_episode = MagicMock(return_value="ep-2")

        agent = _make_agent(knowledge_base=kb)
        agent.current_task = MagicMock(task_id="t")
        agent.episode_transitions = deque([{"x": 1}])

        agent.store_episode_for_transfer(total_reward=1.0, success=True, clear_buffer=False)
        assert len(agent.episode_transitions) == 1

    def test_store_episode_does_not_call_clear_on_kb(self):
        """store_episode_for_transfer() never calls knowledge_base.clear()."""
        kb = MagicMock()
        kb.store_episode = MagicMock(return_value="ep-3")

        agent = _make_agent(knowledge_base=kb)
        agent.current_task = MagicMock(task_id="t")
        agent.episode_transitions = deque([{"x": 1}])

        agent.store_episode_for_transfer(total_reward=0.5, success=False)
        kb.clear.assert_not_called()


# ===========================================================================
# Round 2 Fixes (6 bugs from second audit)
# ===========================================================================

# ---------------------------------------------------------------------------
# Fix 10: emit_signal passes sender_id (not source_agent_id)
# ---------------------------------------------------------------------------

class TestFix10EmitSignal:
    def test_emit_signal_passes_sender_id(self):
        """emit_signal() passes sender_id to SignalBus, not source_agent_id."""
        bus = MagicMock()
        bus.emit_signal = MagicMock(return_value=True)

        agent = _make_agent(signal_bus=bus)
        agent.emit_signal("DANGER", {"risk_level": 0.9})

        bus.emit_signal.assert_called_once()
        call_kwargs = bus.emit_signal.call_args.kwargs
        assert "sender_id" in call_kwargs, "Should pass sender_id"
        assert "source_agent_id" not in call_kwargs, "Should NOT pass source_agent_id"
        assert call_kwargs["sender_id"] == str(agent.unique_id)

    def test_emit_signal_includes_payload(self):
        """emit_signal() passes all expected params."""
        bus = MagicMock()
        bus.emit_signal = MagicMock(return_value=True)

        agent = _make_agent(signal_bus=bus)
        agent.emit_signal("OPPORTUNITY", {"reward": 0.5}, priority=0.8, ttl=5.0)

        call_kwargs = bus.emit_signal.call_args.kwargs
        assert call_kwargs["signal_type"] == "OPPORTUNITY"
        assert call_kwargs["payload"] == {"reward": 0.5}
        assert call_kwargs["priority"] == 0.8
        assert call_kwargs["ttl"] == 5.0


# ---------------------------------------------------------------------------
# Fix 11: subscribe_to_signal passes only valid params
# ---------------------------------------------------------------------------

class TestFix11Subscribe:
    def test_subscribe_no_agent_id(self):
        """subscribe_to_signal() does NOT pass agent_id to SignalBus."""
        bus = MagicMock()
        bus.subscribe = MagicMock(return_value=True)

        agent = _make_agent(signal_bus=bus)
        # Reset mock after __init__ (SignalPriorityResolver subscribes 5 times)
        bus.subscribe.reset_mock()

        callback = lambda s: None
        agent.subscribe_to_signal("DANGER", callback)

        bus.subscribe.assert_called_once()
        call_kwargs = bus.subscribe.call_args.kwargs
        assert "agent_id" not in call_kwargs, "Should NOT pass agent_id"
        assert call_kwargs["signal_type"] == "DANGER"
        assert call_kwargs["callback"] is callback

    def test_subscribe_tracks_callback(self):
        """subscribe_to_signal() stores callback for later unsubscribe."""
        bus = MagicMock()
        bus.subscribe = MagicMock(return_value=True)

        agent = _make_agent(signal_bus=bus)
        callback = lambda s: None
        agent.subscribe_to_signal("DANGER", callback)

        assert agent._signal_callbacks.get("DANGER") is callback


# ---------------------------------------------------------------------------
# Fix 12: unsubscribe passes callback not agent_id
# ---------------------------------------------------------------------------

class TestFix12Unsubscribe:
    def test_unsubscribe_passes_callback(self):
        """unsubscribe_from_signal() passes stored callback, not agent_id."""
        bus = MagicMock()
        bus.subscribe = MagicMock(return_value=True)
        bus.unsubscribe = MagicMock(return_value=True)

        agent = _make_agent(signal_bus=bus)
        callback = lambda s: None
        agent.subscribe_to_signal("DANGER", callback)
        agent.unsubscribe_from_signal("DANGER")

        bus.unsubscribe.assert_called_once_with("DANGER", callback)
        assert "DANGER" not in agent._signal_callbacks


# ---------------------------------------------------------------------------
# Fix 13: get_strongest_marker argument order
# ---------------------------------------------------------------------------

class TestFix13StrongestMarker:
    def test_get_strongest_marker_arg_order(self):
        """get_strongest_nearby_marker() passes args in correct order."""
        env = MagicMock()
        marker = MagicMock()
        env.get_strongest_marker = MagicMock(return_value=marker)

        agent = _make_agent(stigmergy_env=env)
        result = agent.get_strongest_nearby_marker("SUCCESS")

        env.get_strongest_marker.assert_called_once()
        call_kwargs = env.get_strongest_marker.call_args
        # First positional arg is position
        assert call_kwargs.args[0] == agent.stigmergy_position
        # marker_type and radius as kwargs
        assert call_kwargs.kwargs["marker_type"] == "SUCCESS"
        assert call_kwargs.kwargs["radius"] == agent.sensing_radius
        assert result is marker


# ---------------------------------------------------------------------------
# Fix 14: evaluate_transfer no extra kwarg
# ---------------------------------------------------------------------------

class TestFix14EvaluateTransfer:
    def test_evaluate_transfer_no_target_task_id(self):
        """evaluate_transfer_performance() doesn't pass target_task_id."""
        engine = MagicMock()
        engine.evaluate_transfer = MagicMock(return_value={"speedup": 1.5})

        agent = _make_agent(transfer_engine=engine)
        agent.current_task = MagicMock(task_id="t1")
        agent.transfer_enabled = True

        result = agent.evaluate_transfer_performance(
            baseline_performance=0.5,
            current_performance=0.8,
            baseline_samples=100,
            current_samples=50,
        )

        call_kwargs = engine.evaluate_transfer.call_args.kwargs
        assert "target_task_id" not in call_kwargs, "Should NOT pass target_task_id"
        assert call_kwargs["baseline_performance"] == 0.5
        assert call_kwargs["transfer_performance"] == 0.8


# ---------------------------------------------------------------------------
# Fix 15: semantic_retriever statistics uses n_indexed
# ---------------------------------------------------------------------------

class TestFix15SemanticStats:
    def test_episodic_stats_uses_n_indexed(self):
        """get_episodic_memory_statistics() uses n_indexed, not get_statistics()."""
        from mae_core.memory.prioritized_replay_buffer import PrioritizedReplayBuffer

        mem = PrioritizedReplayBuffer(capacity=100)
        retriever = MagicMock()
        retriever.n_indexed = 42
        # Explicitly verify get_statistics doesn't exist on real object
        del retriever.get_statistics

        agent = _make_agent(
            episodic_memory=mem,
            semantic_retriever=retriever,
            agent_config={"replay_enabled": True},
        )

        stats = agent.get_episodic_memory_statistics()
        assert stats is not None
        assert stats["semantic_retrieval"] == {"n_indexed": 42}
