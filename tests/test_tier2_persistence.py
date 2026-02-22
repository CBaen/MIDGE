"""Unit tests for Tier 2 Persistence — serialize/restore across all subsystems."""

import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pytest

from mae_core.backbone.event_bus import EventBus
from mae_core.cognition.world_model import WorldModel, WorldModelConfig
from mae_core.learning.frl import FederatedLearningEngine
from mae_core.learning.knowledge_base import KnowledgeBase, TaskDescriptor
from mae_core.learning.maml import MAMLLearner
from mae_core.learning.vdn import ValueDecompositionEngine
from mae_core.memory.coordinator import MemoryCoordinator
from mae_core.memory.episodic_memory import EpisodicMemory
from mae_core.memory.experience import Experience
from mae_core.memory.generative_replay import GenerativeReplayMemory
from mae_core.memory.memory_consolidator import MemoryConsolidator
from mae_core.memory.semantic_retriever import SemanticRetriever


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir(tmp_path):
    """Temporary persist directory for each test."""
    d = tmp_path / "persist"
    d.mkdir()
    return d


@pytest.fixture
def bus():
    return EventBus()


def _make_experience(reward=1.0, state_val=0.5):
    return Experience(
        state=np.array([state_val, 0.1, 0.2]),
        action=1,
        reward=reward,
        next_state=np.array([state_val + 0.1, 0.2, 0.3]),
        done=False,
        info={},
        timestamp=1000.0,
    )


# ---------------------------------------------------------------------------
# TestEpisodicMemoryPersistence
# ---------------------------------------------------------------------------

class TestEpisodicMemoryPersistence:
    def test_empty_round_trip(self, tmp_dir):
        mem = EpisodicMemory(capacity=100)
        meta = mem.serialize(tmp_dir)
        assert meta["size"] == 0

        mem2 = EpisodicMemory(capacity=100)
        mem2.restore(tmp_dir, meta)
        assert len(mem2) == 0

    def test_with_experiences(self, tmp_dir):
        mem = EpisodicMemory(capacity=100)
        for i in range(10):
            mem.store(
                state=np.array([float(i), 0.1]),
                action=i % 3,
                reward=float(i) * 0.1,
                next_state=np.array([float(i) + 0.1, 0.2]),
                done=False,
            )

        meta = mem.serialize(tmp_dir)
        assert meta["size"] == 10
        assert meta["total_stored"] == 10

        mem2 = EpisodicMemory(capacity=100)
        mem2.restore(tmp_dir, meta)
        assert len(mem2) == 10
        assert mem2.total_stored == 10

    def test_priorities_preserved(self, tmp_dir):
        mem = EpisodicMemory(capacity=100)
        mem.store(
            state=np.array([1.0, 2.0]),
            action=0,
            reward=5.0,
            next_state=np.array([1.1, 2.1]),
            done=False,
            priority=10.0,
        )
        original_max = mem.max_priority

        meta = mem.serialize(tmp_dir)
        mem2 = EpisodicMemory(capacity=100)
        mem2.restore(tmp_dir, meta)
        assert mem2.max_priority == original_max

    def test_sampling_works_after_restore(self, tmp_dir):
        mem = EpisodicMemory(capacity=100)
        for i in range(20):
            mem.store(
                state=np.array([float(i)]),
                action=0,
                reward=1.0,
                next_state=np.array([float(i) + 1]),
                done=False,
            )

        meta = mem.serialize(tmp_dir)
        mem2 = EpisodicMemory(capacity=100)
        mem2.restore(tmp_dir, meta)

        batch, indices, weights = mem2.sample(5)
        assert len(batch) == 5
        assert len(indices) == 5
        assert len(weights) == 5


# ---------------------------------------------------------------------------
# TestSemanticRetrieverPersistence
# ---------------------------------------------------------------------------

class TestSemanticRetrieverPersistence:
    def test_empty_round_trip(self, tmp_dir):
        sr = SemanticRetriever(embedding_dim=32)
        meta = sr.serialize(tmp_dir)
        assert meta["n_indexed"] == 0

        sr2 = SemanticRetriever(embedding_dim=32)
        sr2.restore(tmp_dir, meta)
        assert sr2.n_indexed == 0

    def test_with_indexed_data(self, tmp_dir):
        sr = SemanticRetriever(embedding_dim=32)
        for i in range(5):
            sr.add(_make_experience(reward=float(i)))
        assert sr.n_indexed == 5

        meta = sr.serialize(tmp_dir)
        sr2 = SemanticRetriever(embedding_dim=32)
        sr2.restore(tmp_dir, meta)
        assert sr2.n_indexed == 5


# ---------------------------------------------------------------------------
# TestGenerativeReplayPersistence
# ---------------------------------------------------------------------------

class TestGenerativeReplayPersistence:
    def test_buffer_only(self, tmp_dir):
        gr = GenerativeReplayMemory(state_dim=3, action_dim=2)
        for i in range(5):
            gr.store(_make_experience(reward=float(i)))

        meta = gr.serialize(tmp_dir)
        assert meta["buffer_size"] == 5
        assert meta["is_trained"] is False

        gr2 = GenerativeReplayMemory(state_dim=3, action_dim=2)
        gr2.restore(tmp_dir, meta)
        assert gr2.buffer_size == 5
        assert gr2.total_stored == 5

    def test_after_compression(self, tmp_dir):
        gr = GenerativeReplayMemory(state_dim=3, action_dim=2, keep_recent=5)
        for i in range(50):
            gr.store(_make_experience(reward=float(i) * 0.01))

        gr.compress(force=True)
        meta = gr.serialize(tmp_dir)
        assert meta["compression_method"] == "gzip"

        gr2 = GenerativeReplayMemory(state_dim=3, action_dim=2, keep_recent=5)
        gr2.restore(tmp_dir, meta)
        assert gr2.is_compressed
        assert gr2.total_stored == 50


# ---------------------------------------------------------------------------
# TestConsolidatorPersistence
# ---------------------------------------------------------------------------

class TestConsolidatorPersistence:
    def test_round_trip(self, tmp_dir):
        mem = EpisodicMemory(capacity=100)
        mc = MemoryConsolidator(episodic_memory=mem)
        mc._steps_since = 42
        mc._total_consolidations = 3

        meta = mc.serialize(tmp_dir)
        assert meta["steps_since"] == 42
        assert meta["total_consolidations"] == 3

        mc2 = MemoryConsolidator(episodic_memory=mem)
        mc2.restore(tmp_dir, meta)
        assert mc2._steps_since == 42
        assert mc2._total_consolidations == 3

    def test_counters_survive(self, tmp_dir):
        mem = EpisodicMemory(capacity=100)
        mc = MemoryConsolidator(episodic_memory=mem)
        for _ in range(100):
            mc.step()

        meta = mc.serialize(tmp_dir)
        mc2 = MemoryConsolidator(episodic_memory=mem)
        mc2.restore(tmp_dir, meta)
        assert mc2._steps_since == 100


# ---------------------------------------------------------------------------
# TestWorldModelPersistence
# ---------------------------------------------------------------------------

class TestWorldModelPersistence:
    def test_default_round_trip(self, tmp_dir):
        wm = WorldModel()
        meta = wm.serialize(tmp_dir)
        assert meta["training_steps"] == 0

        wm2 = WorldModel()
        wm2.restore(tmp_dir, meta)
        assert wm2._training_steps == 0

    def test_after_training(self, tmp_dir):
        config = WorldModelConfig(use_ensemble=True, num_ensemble=3)
        wm = WorldModel(config=config)

        states = np.random.randn(5, 10).astype(np.float32)
        actions = np.array([0, 1, 2, 0, 1])
        next_states = np.random.randn(5, 10).astype(np.float32)
        rewards = np.random.randn(5).astype(np.float32)
        wm.train_step(states, actions, next_states, rewards)

        meta = wm.serialize(tmp_dir)
        assert meta["training_steps"] == 1
        assert meta["ensemble_size"] == 3

        wm2 = WorldModel(config=config)
        wm2.restore(tmp_dir, meta)
        assert wm2._training_steps == 1
        assert len(wm2._ensemble_weights) == 3


# ---------------------------------------------------------------------------
# TestVDNPersistence
# ---------------------------------------------------------------------------

class TestVDNPersistence:
    def test_q_table_survives(self, tmp_dir, bus):
        vdn = ValueDecompositionEngine(agent_id="a1", event_bus=bus)
        state = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        vdn.compute_local_value(state, 0)
        vdn.update_value_function(state, 0, 1.0, state * 1.1, False)

        meta = vdn.serialize(tmp_dir)
        assert meta["q_table_size"] == 2  # state + next_state entries

        vdn2 = ValueDecompositionEngine(agent_id="a1", event_bus=bus)
        vdn2.restore(tmp_dir, meta)
        assert len(vdn2._q_table) == 2

    def test_credit_history_survives(self, tmp_dir, bus):
        vdn = ValueDecompositionEngine(agent_id="a1", event_bus=bus)
        vdn._credit_history = [0.1, 0.5, 0.9]

        meta = vdn.serialize(tmp_dir)
        assert meta["credit_history_size"] == 3

        vdn2 = ValueDecompositionEngine(agent_id="a1", event_bus=bus)
        vdn2.restore(tmp_dir, meta)
        assert vdn2._credit_history == [0.1, 0.5, 0.9]


# ---------------------------------------------------------------------------
# TestFRLPersistence
# ---------------------------------------------------------------------------

class TestFRLPersistence:
    def test_peer_trust_survives(self, tmp_dir, bus):
        frl = FederatedLearningEngine(agent_id="a1", event_bus=bus)
        frl._peer_trust["a2"] = 0.8
        frl._peer_trust["a3"] = 0.3
        frl._peers = {"a2", "a3"}

        meta = frl.serialize(tmp_dir)
        assert meta["peers"] == 2

        frl2 = FederatedLearningEngine(agent_id="a1", event_bus=bus)
        frl2.restore(tmp_dir, meta)
        assert frl2._peer_trust["a2"] == 0.8
        assert frl2._peer_trust["a3"] == 0.3
        assert frl2._peers == {"a2", "a3"}

    def test_counters_survive(self, tmp_dir, bus):
        frl = FederatedLearningEngine(agent_id="a1", event_bus=bus)
        frl._shared_count = 10
        frl._received_count = 5

        meta = frl.serialize(tmp_dir)
        frl2 = FederatedLearningEngine(agent_id="a1", event_bus=bus)
        frl2.restore(tmp_dir, meta)
        assert frl2._shared_count == 10
        assert frl2._received_count == 5


# ---------------------------------------------------------------------------
# TestMAMLPersistence
# ---------------------------------------------------------------------------

class TestMAMLPersistence:
    def test_meta_parameters_survive(self, tmp_dir):
        kb = KnowledgeBase()
        maml = MAMLLearner(knowledge_base=kb)
        maml._meta_parameters = {"weights": np.array([1.0, 2.0, 3.0])}
        maml._meta_initialized = True

        meta = maml.serialize(tmp_dir)
        assert meta["meta_initialized"] is True

        maml2 = MAMLLearner(knowledge_base=kb)
        maml2.restore(tmp_dir, meta)
        assert maml2._meta_initialized is True
        np.testing.assert_array_equal(
            maml2._meta_parameters["weights"],
            np.array([1.0, 2.0, 3.0]),
        )

    def test_training_history_survives(self, tmp_dir):
        kb = KnowledgeBase()
        maml = MAMLLearner(knowledge_base=kb)
        maml._training_history = [{"iterations": 10, "final_loss": 0.5}]

        meta = maml.serialize(tmp_dir)
        assert meta["training_rounds"] == 1

        maml2 = MAMLLearner(knowledge_base=kb)
        maml2.restore(tmp_dir, meta)
        assert len(maml2._training_history) == 1


# ---------------------------------------------------------------------------
# TestKnowledgeBasePersistence
# ---------------------------------------------------------------------------

class TestKnowledgeBasePersistence:
    def test_tasks_and_episodes(self, tmp_dir):
        kb = KnowledgeBase()
        task = TaskDescriptor(task_id="t1", name="test task", domain="test")
        kb.register_task(task)
        kb.store_episode("t1", "a1", [{"s": 0, "a": 1}], 5.0, True)

        meta = kb.serialize(tmp_dir)
        assert meta["tasks"] == 1
        assert meta["total_episodes"] == 1

        kb2 = KnowledgeBase()
        kb2.restore(tmp_dir, meta)
        assert len(kb2._tasks) == 1
        episodes = kb2.retrieve_successful_episodes("t1")
        assert len(episodes) == 1
        assert episodes[0].total_reward == 5.0

    def test_policies_survive(self, tmp_dir):
        kb = KnowledgeBase()
        kb.store_policy("t1", "a1", {"lr": 0.01}, performance=0.9)

        meta = kb.serialize(tmp_dir)
        assert meta["total_policies"] == 1

        kb2 = KnowledgeBase()
        kb2.restore(tmp_dir, meta)
        best = kb2.retrieve_best_policy("t1")
        assert best is not None
        assert best["performance"] == 0.9


# ---------------------------------------------------------------------------
# TestMemoryCoordinatorPersistence
# ---------------------------------------------------------------------------

class TestMemoryCoordinatorPersistence:
    def test_delegates_to_children(self, tmp_dir, bus):
        mc = MemoryCoordinator(event_bus=bus, agent_id="a1")
        for i in range(10):
            mc.store(
                state=np.array([float(i), 0.1]),
                action=i % 3,
                reward=float(i) * 0.1,
                next_state=np.array([float(i) + 0.1, 0.2]),
                done=False,
            )

        meta = mc.serialize(tmp_dir)
        assert "episodic" in meta
        assert meta["episodic"]["size"] == 10

        mc2 = MemoryCoordinator(event_bus=bus, agent_id="a1")
        mc2.restore(tmp_dir, meta)
        assert len(mc2.episodic) == 10

    def test_partial_restore(self, tmp_dir, bus):
        """Restore succeeds even if some subsystems have no saved state."""
        mc = MemoryCoordinator(event_bus=bus, agent_id="a1")
        # Only save with minimal metadata (no semantic/generative keys)
        meta = {"agent_id": "a1", "episodic": mc.episodic.serialize(tmp_dir)}

        mc2 = MemoryCoordinator(event_bus=bus, agent_id="a1")
        mc2.restore(tmp_dir, meta)  # Should not raise


# ---------------------------------------------------------------------------
# TestGracefulDegradation
# ---------------------------------------------------------------------------

class TestGracefulDegradation:
    def test_missing_file_starts_fresh(self, tmp_dir):
        """Restore from a nonexistent directory produces no error."""
        mem = EpisodicMemory(capacity=100)
        mem.restore(tmp_dir / "nonexistent", {"total_stored": 5})
        assert len(mem) == 0  # started fresh, no crash

    def test_capacity_mismatch_skips_restore(self, tmp_dir):
        """Different capacity between save/load skips restore gracefully."""
        mem = EpisodicMemory(capacity=50)
        for i in range(5):
            mem.store(
                state=np.array([float(i)]),
                action=0, reward=1.0,
                next_state=np.array([float(i) + 1]),
                done=False,
            )
        meta = mem.serialize(tmp_dir)

        mem2 = EpisodicMemory(capacity=200)  # Different capacity
        mem2.restore(tmp_dir, meta)
        assert len(mem2) == 0  # Skipped due to capacity mismatch
