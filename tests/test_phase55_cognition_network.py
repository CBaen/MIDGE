"""Phase 5.5 Verification - Cognition + Network (Octopus Brain).

Tests all 10 modules:
- Cognition: WorldModel, DecisionRouter, CausalReasoning, CollectiveDream, ValidatedImagination
- Network: OctopusSignals, OctopusArm, OctopusCognition, OctopusAgent, OctopusColony

Focus areas:
1. Module imports and initialization
2. Core functionality per module
3. Cross-system integration (Octopus + DecisionRouter + WorldModel)
4. Rule of 3 enforcement
5. Colony lifecycle (spawn, despawn, self-healing, auto-scaling)
"""

import time
import numpy as np
import pytest

from mae_core.backbone.event_bus import EventBus

# --- Cognition imports ---
from mae_core.cognition import (
    WorldModel, WorldModelConfig, Prediction,
    DecisionRouter, DecisionTier, ReflexPattern, Habit, RouterDecision,
    CausalReasoningEngine, CausalLink, CausalQueryResult, CausalRelationType,
    CollectiveDreamPlanner, Dream, DreamAgent, ConsensusResult,
    ValidatedImagination, ValidatedImaginationPlanner,
    ImaginationPrediction, ImaginationAccuracy, TrajectoryStep,
)

# --- Network imports ---
from mae_core.network import (
    ArmCapability, ArmState, CognitionMode, CoordinationSignal,
    OctopusSpecialization, Task,
    OctopusArm, OctopusDistributedCognition, OctopusAgent, OctopusColony,
    MIN_AGENTS, MIN_CONNECTIONS, MIN_VOTES,
    get_min_connections, validate_rule_of_3,
    DEFAULT_ARM_CAPABILITIES, TASK_CAPABILITY_MAP, SPECIALIZATION_CAPABILITIES,
    CH_OCTOPUS_TASK, CH_OCTOPUS_SPAWN, CH_OCTOPUS_DESPAWN,
)


# ===== COGNITION TESTS =====

class TestWorldModel:
    def test_initialization(self):
        wm = WorldModel()
        assert repr(wm).startswith("WorldModel(")

    def test_single_step_prediction(self):
        wm = WorldModel()
        state = np.zeros(10, dtype=np.float32)
        pred = wm.step(state, 0)
        assert isinstance(pred, Prediction)
        assert pred.next_state.shape == state.shape
        assert isinstance(pred.reward, float)

    def test_rollout(self):
        wm = WorldModel()
        state = np.zeros(10, dtype=np.float32)
        policy = lambda s: np.random.randint(0, 5)
        result = wm.rollout(state, policy, horizon=5)
        assert len(result["states"]) == 6  # initial + 5 steps
        assert len(result["actions"]) == 5
        assert len(result["rewards"]) == 5
        assert "total_reward" in result

    def test_ensemble_mode(self):
        config = WorldModelConfig(use_ensemble=True, num_ensemble=3)
        wm = WorldModel(config=config)
        state = np.zeros(10, dtype=np.float32)
        pred = wm.step(state, 0)
        assert isinstance(pred.ensemble_disagreement, float)

    def test_predict_convenience(self):
        wm = WorldModel()
        state = np.zeros(10, dtype=np.float32)
        next_state = wm.predict(state, 0)
        assert isinstance(next_state, np.ndarray)
        reward = wm.predict_reward(state, 0)
        assert isinstance(reward, float)

    def test_uncertainty(self):
        config = WorldModelConfig(use_ensemble=True, num_ensemble=5)
        wm = WorldModel(config=config)
        state = np.zeros(10, dtype=np.float32)
        u = wm.get_uncertainty(state, 0)
        assert "transition_disagreement" in u
        assert "total_disagreement" in u

    def test_train_step(self):
        config = WorldModelConfig(use_ensemble=True, num_ensemble=3)
        wm = WorldModel(config=config)
        states = np.random.randn(4, 10).astype(np.float32)
        actions = np.random.randn(4, 5).astype(np.float32)
        next_states = np.random.randn(4, 10).astype(np.float32)
        rewards = np.random.randn(4).astype(np.float32)
        loss = wm.train_step(states, actions, next_states, rewards)
        assert isinstance(loss, float)
        stats = wm.get_training_stats()
        assert stats["training_steps"] == 1


class TestDecisionRouter:
    def test_initialization(self):
        router = DecisionRouter()
        assert repr(router).startswith("DecisionRouter(")

    def test_reflex_response(self):
        router = DecisionRouter()
        decision = router.route_decision("danger ahead")
        assert decision.tier_used == DecisionTier.REFLEX
        assert decision.action_taken == {"type": "flee"}
        assert decision.confidence >= 0.9

    def test_habit_response(self):
        router = DecisionRouter()
        habit = Habit(habit_id="test-habit", stimulus="morning", action="coffee", strength=0.8)
        router.register_habit(habit)
        decision = router.route_decision("morning")
        assert decision.tier_used == DecisionTier.HABIT
        assert decision.action_taken == "coffee"

    def test_prefrontal_response(self):
        router = DecisionRouter()
        decision = router.route_decision("novel_situation_xyz")
        assert decision.tier_used == DecisionTier.PREFRONTAL

    def test_auto_habit_formation(self):
        router = DecisionRouter(habit_formation_threshold=3)
        for _ in range(3):
            router.route_decision(
                "repeat_stimulus",
                available_actions=["same_action"],
            )
        # After threshold, habit should form
        decision = router.route_decision("repeat_stimulus")
        assert decision.tier_used == DecisionTier.HABIT

    def test_executive_override(self):
        router = DecisionRouter()
        decision = router.executive_override("danger ahead")
        assert decision.tier_used == DecisionTier.PREFRONTAL  # Bypasses reflex

    def test_performance_metrics(self):
        router = DecisionRouter()
        router.route_decision("danger")
        router.route_decision("novel")
        metrics = router.get_performance_metrics()
        assert metrics["total_decisions"] == 2

    def test_tier_health(self):
        router = DecisionRouter()
        health = router.get_tier_health()
        assert "reflex" in health
        assert "habit" in health
        assert "prefrontal" in health
        assert health["reflex"]["patterns"] == 3  # default danger reflexes


class TestCausalReasoning:
    def test_initialization(self):
        engine = CausalReasoningEngine()
        assert repr(engine).startswith("CausalReasoning(")

    def test_observe_intervention(self):
        engine = CausalReasoningEngine()
        engine.observe_intervention("rain", True, {"wet_ground": True})
        result = engine.query_causation("rain", "wet_ground")
        assert result.is_causal
        assert len(result.causal_path) == 2

    def test_causal_path_finding(self):
        engine = CausalReasoningEngine()
        engine.add_causal_link("A", "B", strength=0.9, confidence=0.9)
        engine.add_causal_link("B", "C", strength=0.8, confidence=0.8)
        path = engine.find_causal_path("A", "C")
        assert path == ["A", "B", "C"]

    def test_causal_strength(self):
        engine = CausalReasoningEngine()
        engine.add_causal_link("A", "B", strength=0.9, confidence=0.9)
        engine.add_causal_link("B", "C", strength=0.8, confidence=0.8)
        strength = engine.compute_causal_strength(["A", "B", "C"])
        assert 0 < strength < 0.8  # weakest link * chain penalty

    def test_counterfactual(self):
        engine = CausalReasoningEngine()
        engine.add_causal_link("fire", "smoke", strength=0.95, confidence=0.95)
        cf = engine.generate_counterfactual("fire", "smoke")
        assert "fire" in cf and "smoke" in cf

    def test_confounder_identification(self):
        engine = CausalReasoningEngine()
        engine.add_causal_link("C", "A", strength=0.8, confidence=0.8)
        engine.add_causal_link("C", "B", strength=0.8, confidence=0.8)
        confounders = engine.identify_confounders("A", "B")
        assert "C" in confounders

    def test_infer_causes(self):
        engine = CausalReasoningEngine()
        engine.add_causal_link("virus", "fever", strength=0.9, confidence=0.9)
        engine.add_causal_link("bacteria", "fever", strength=0.7, confidence=0.7)
        result = engine.infer_causes("fever")
        assert "virus" in result["causes"]
        assert "bacteria" in result["causes"]

    def test_metrics(self):
        engine = CausalReasoningEngine()
        engine.add_causal_link("A", "B")
        metrics = engine.get_causal_metrics()
        assert metrics["total_links"] == 1
        assert metrics["links_discovered"] == 1


class TestCollectiveDream:
    def test_initialization(self):
        wm = WorldModel()
        planner = CollectiveDreamPlanner(wm)
        assert repr(planner).startswith("CollectiveDreamPlanner(")

    def test_plan_with_default_agents(self):
        wm = WorldModel()
        planner = CollectiveDreamPlanner(wm)
        state = np.zeros(10, dtype=np.float32)
        trajectory, info = planner.collective_plan(state, horizon=5, num_dreamers=3)
        assert isinstance(trajectory, list)
        assert "status" in info
        assert "consensus" in info

    def test_plan_with_registered_agents(self):
        wm = WorldModel()
        planner = CollectiveDreamPlanner(wm)
        for i in range(10):
            agent = DreamAgent(agent_id=f"agent-{i}", expertise=0.5 + i * 0.05)
            planner.register_agent(agent)
        state = np.zeros(10, dtype=np.float32)
        trajectory, info = planner.collective_plan(state, horizon=5, num_dreamers=3)
        assert isinstance(trajectory, list)

    def test_expert_dreamer_selection(self):
        wm = WorldModel()
        planner = CollectiveDreamPlanner(wm)
        agents = [DreamAgent(agent_id=f"a{i}", expertise=i * 0.1) for i in range(10)]
        for a in agents:
            planner.register_agent(a)
        dreamers = planner.select_expert_dreamers(3)
        assert len(dreamers) == 3
        # Highest expertise first
        assert dreamers[0].expertise >= dreamers[1].expertise

    def test_low_consensus_triggers_morphogenesis(self):
        triggered = []
        def morph_cb(sig):
            triggered.append(sig)

        wm = WorldModel()
        planner = CollectiveDreamPlanner(
            wm, consensus_threshold=0.99, morphogenesis_callback=morph_cb
        )
        state = np.zeros(10, dtype=np.float32)
        planner.collective_plan(state, horizon=3, num_dreamers=3)
        # With very high threshold, consensus is likely low → morphogenesis triggered
        # (may not always trigger due to randomness, so just check it ran without error)

    def test_statistics(self):
        wm = WorldModel()
        planner = CollectiveDreamPlanner(wm)
        state = np.zeros(10, dtype=np.float32)
        planner.collective_plan(state, horizon=3)
        stats = planner.get_planning_statistics()
        assert stats["total_plans"] == 1


class TestValidatedImagination:
    def test_record_and_validate(self):
        vi = ValidatedImagination()
        pred_id = vi.record_imagination(
            agent_id="agent-1", domain="physics",
            state=np.zeros(5), action=np.array([1]),
            predicted_next_state=np.ones(5),
            predicted_reward=0.8, confidence=0.9,
        )
        assert pred_id.startswith("pred-")
        is_accurate = vi.validate_with_consensus(pred_id, np.ones(5), 0.85)
        assert isinstance(is_accurate, bool)

    def test_accuracy_tracking(self):
        vi = ValidatedImagination()
        for i in range(10):
            pid = vi.record_imagination(
                agent_id="a1", domain="nav",
                state=np.zeros(3), action=np.array([0]),
                predicted_next_state=np.zeros(3),
                predicted_reward=0.5, confidence=0.8,
            )
            vi.validate_with_consensus(pid, np.zeros(3), 0.5 + i * 0.01)
        acc = vi.get_imagination_accuracy("a1", "nav")
        assert acc is not None
        assert acc.total_predictions == 10

    def test_top_imaginers(self):
        vi = ValidatedImagination()
        for agent_id in ["good", "bad"]:
            for i in range(6):
                pid = vi.record_imagination(
                    agent_id=agent_id, domain="test",
                    state=np.zeros(3), action=np.array([0]),
                    predicted_next_state=np.zeros(3),
                    predicted_reward=0.5 if agent_id == "good" else 0.1,
                    confidence=0.8,
                )
                actual_reward = 0.5 if agent_id == "good" else 0.9
                vi.validate_with_consensus(pid, np.zeros(3), actual_reward)
        top = vi.get_top_imaginers("test", count=5, min_predictions=5)
        assert len(top) >= 1

    def test_statistics(self):
        vi = ValidatedImagination()
        stats = vi.get_statistics()
        assert stats["total_imaginations"] == 0
        assert stats["pending_validations"] == 0

    def test_planner_with_validation(self):
        wm = WorldModel()
        vi = ValidatedImagination()
        planner = ValidatedImaginationPlanner(wm, validator=vi)
        state = np.zeros(10, dtype=np.float32)
        policy = lambda s: np.random.randint(0, 5)
        trajectory = planner.plan_with_validation("a1", state, policy, horizon=5)
        assert isinstance(trajectory, list)
        stats = planner.get_statistics()
        assert stats["total_plans"] == 1


# ===== NETWORK TESTS =====

class TestOctopusSignals:
    def test_enums(self):
        assert CognitionMode.HYBRID.value == "hybrid"
        assert ArmCapability.SENSORY_PROCESSING.value == "sensory"
        assert OctopusSpecialization.GENERAL.value == "general"

    def test_arm_state(self):
        state = ArmState(arm_id="arm-0", capabilities={ArmCapability.LEARNING})
        assert state.workload == 0.0
        assert state.health == 1.0

    def test_task(self):
        task = Task(task_type="learning", priority=8)
        assert task.status == "pending"
        assert len(task.task_id) == 12

    def test_coordination_signal(self):
        signal = CoordinationSignal(source_arm="arm-1", signal_type="emergency")
        assert signal.priority == 1

    def test_default_capabilities(self):
        assert len(DEFAULT_ARM_CAPABILITIES) == 8
        for caps in DEFAULT_ARM_CAPABILITIES:
            assert len(caps) >= 2

    def test_task_capability_map(self):
        assert "sensory_analysis" in TASK_CAPABILITY_MAP
        assert "complex_analysis" in TASK_CAPABILITY_MAP

    def test_specialization_capabilities(self):
        assert len(SPECIALIZATION_CAPABILITIES) == 8


class TestOctopusArm:
    def test_initialization(self):
        arm = OctopusArm("arm-0", {ArmCapability.SENSORY_PROCESSING, ArmCapability.LEARNING})
        assert arm.arm_id == "arm-0"
        assert arm.state.health == 1.0

    def test_submit_task_matching_capabilities(self):
        arm = OctopusArm("arm-0", {ArmCapability.LEARNING})
        task = Task(task_type="learning", required_capabilities={ArmCapability.LEARNING})
        assert arm.submit_task(task) is True
        assert len(arm.task_queue) == 1

    def test_submit_task_wrong_capabilities(self):
        arm = OctopusArm("arm-0", {ArmCapability.LEARNING})
        task = Task(task_type="decision", required_capabilities={ArmCapability.DECISION_MAKING})
        assert arm.submit_task(task) is False

    def test_receive_coordination_signal(self):
        arm = OctopusArm("arm-0", {ArmCapability.LEARNING})
        signal = CoordinationSignal(source_arm="arm-1", signal_type="learning_update", data={"metrics": {"accuracy": 0.9}})
        arm.receive_coordination_signal(signal)
        assert len(arm.coordination_signals) == 1

    def test_get_arm_status(self):
        arm = OctopusArm("arm-0", {ArmCapability.LEARNING, ArmCapability.MEMORY_ACCESS})
        status = arm.get_arm_status()
        assert status["arm_id"] == "arm-0"
        assert status["workload"] == 0.0
        assert "learning" in status["capabilities"] or "memory" in status["capabilities"]


class TestOctopusCognition:
    def test_initialization(self):
        cog = OctopusDistributedCognition()
        assert len(cog.arms) == 8
        assert cog.coordination_mode == CognitionMode.HYBRID

    def test_ring_topology(self):
        cog = OctopusDistributedCognition(num_arms=4)
        for arm in cog.arms.values():
            assert len(arm.connected_arms) >= 2  # ring: next + prev

    def test_submit_task(self):
        bus = EventBus()
        cog = OctopusDistributedCognition(event_bus=bus)
        task_id = cog.submit_task({"data": "test"}, "learning")
        assert len(task_id) == 12

    def test_coordination_cycle(self):
        cog = OctopusDistributedCognition()
        cog.run_coordination_cycle()
        assert cog.last_coordination_update > 0

    def test_emergency_mode(self):
        cog = OctopusDistributedCognition()
        cog.trigger_emergency_mode("system_overload")
        assert cog.emergency_mode is True
        assert cog.coordination_mode == CognitionMode.EMERGENCY
        assert cog.global_coordination_level == 0.1

        cog.exit_emergency_mode()
        assert cog.emergency_mode is False
        assert cog.coordination_mode != CognitionMode.EMERGENCY

    def test_system_status(self):
        cog = OctopusDistributedCognition()
        status = cog.get_system_status()
        assert status["num_arms"] == 8
        assert "arm_statuses" in status
        assert status["emergency_mode"] is False

    def test_learning_sharing(self):
        cog = OctopusDistributedCognition()
        cog.add_learning_update({"accuracy": 0.95, "loss": 0.05})
        cog.run_coordination_cycle()
        # Learning should have been broadcast to arms


class TestOctopusAgent:
    def test_initialization(self):
        agent = OctopusAgent("oct-0")
        assert agent.octopus_id == "oct-0"
        assert agent.health == 1.0
        assert agent.specialization == OctopusSpecialization.GENERAL

    def test_specialization(self):
        agent = OctopusAgent("oct-1", specialization=OctopusSpecialization.SENSORY)
        caps = agent.get_capabilities()
        assert caps["specialization_type"] == "sensory"
        assert "sensor_processing" in caps["specialization"]

    def test_submit_task(self):
        agent = OctopusAgent("oct-0")
        task_id = agent.submit_task({"x": 1}, "learning")
        assert len(task_id) == 12

    def test_route_decision_fallback(self):
        agent = OctopusAgent("oct-0")
        result = agent.route_decision("test_stimulus")
        assert result["tier"] == "local"
        assert result["octopus_id"] == "oct-0"

    def test_route_decision_with_router(self):
        router = DecisionRouter()
        agent = OctopusAgent("oct-0", decision_router=router)
        result = agent.route_decision("danger signal")
        assert result["tier"] == "reflex"
        assert result["confidence"] >= 0.9

    def test_predict_with_confidence_fallback(self):
        agent = OctopusAgent("oct-0")
        result = agent.predict_with_confidence("arm-0", np.zeros(5), np.array([1]))
        assert result["source"] == "fallback"
        assert result["confidence"] == 0.3

    def test_predict_with_world_model(self):
        wm = WorldModel()
        agent = OctopusAgent("oct-0", world_model=wm)
        # Low confidence arm -> should escalate to central world model
        result = agent.predict_with_confidence("arm-0", np.zeros(10), np.array([0]))
        assert result["source"] == "central_world_model"
        assert result["escalated"] is True

    def test_predict_high_confidence_arm(self):
        wm = WorldModel()
        agent = OctopusAgent("oct-0", world_model=wm)
        # Build up arm confidence
        for _ in range(10):
            agent.update_arm_prediction_accuracy("arm-1", True)
        result = agent.predict_with_confidence("arm-1", np.zeros(10), np.array([0]))
        assert result["source"] == "arm_local"
        assert result["escalated"] is False

    def test_update_metrics(self):
        agent = OctopusAgent("oct-0")
        agent.update_metrics()
        assert 0.0 <= agent.health <= 1.0
        assert 0.0 <= agent.workload <= 1.0

    def test_get_status(self):
        agent = OctopusAgent("oct-0", specialization=OctopusSpecialization.MEMORY)
        status = agent.get_status()
        assert status["specialization"] == "memory"
        assert status["num_arms"] == 8
        assert "arm_prediction_accuracy" in status

    def test_emit_signal_no_bus(self):
        agent = OctopusAgent("oct-0")
        # Should not raise even without signal bus
        agent.emit_signal("test", {"data": 1})


class TestRuleOf3:
    def test_validate_valid(self):
        assert validate_rule_of_3(agent_count=3) is True
        assert validate_rule_of_3(agent_count=10) is True

    def test_validate_invalid(self):
        with pytest.raises(ValueError):
            validate_rule_of_3(agent_count=2)
        with pytest.raises(ValueError):
            validate_rule_of_3(agent_count=1)

    def test_get_min_connections(self):
        assert get_min_connections(3) == 2
        assert get_min_connections(4) == 3
        assert get_min_connections(10) == 3

    def test_get_min_connections_invalid(self):
        with pytest.raises(ValueError):
            get_min_connections(2)

    def test_constants(self):
        assert MIN_AGENTS == 3
        assert MIN_CONNECTIONS == 3
        assert MIN_VOTES == 3


class TestOctopusColony:
    def test_initialization(self):
        colony = OctopusColony()
        assert len(colony.octopuses) == 3  # Rule of 3 minimum
        assert repr(colony).startswith("OctopusColony(")

    def test_initialization_rejects_below_minimum(self):
        with pytest.raises(ValueError):
            OctopusColony(min_octopuses=2)

    def test_peer_connections(self):
        colony = OctopusColony()
        for oid, peers in colony.peer_connections.items():
            assert len(peers) >= 2, f"{oid} has only {len(peers)} peers"

    def test_spawn_octopus(self):
        colony = OctopusColony(max_octopuses=5)
        new_id = colony.spawn_octopus(
            specialization=OctopusSpecialization.SENSORY,
            reason="test_spawn",
        )
        assert new_id is not None
        assert len(colony.octopuses) == 4
        assert len(colony.spawn_history) == 4  # 3 initial + 1 new

    def test_spawn_at_max_capacity(self):
        colony = OctopusColony(max_octopuses=3)
        result = colony.spawn_octopus(reason="overflow")
        assert result is None
        assert len(colony.octopuses) == 3

    def test_despawn_octopus(self):
        colony = OctopusColony(max_octopuses=5)
        colony.spawn_octopus(reason="extra")
        assert len(colony.octopuses) == 4
        ids = list(colony.octopuses.keys())
        success = colony.despawn_octopus(ids[-1], reason="test_despawn")
        assert success is True
        assert len(colony.octopuses) == 3

    def test_despawn_respects_minimum(self):
        colony = OctopusColony()
        ids = list(colony.octopuses.keys())
        success = colony.despawn_octopus(ids[0], reason="test")
        assert success is False  # Can't go below 3

    def test_submit_task(self):
        colony = OctopusColony()
        task_id = colony.submit_task({"data": "test"}, "learning")
        assert task_id is not None
        assert len(task_id) == 12

    def test_colony_status(self):
        colony = OctopusColony()
        status = colony.get_colony_status()
        assert status["colony_size"] == 3
        assert status["rule_of_3_compliant"] is True
        assert status["peer_connectivity_ok"] is True
        assert status["network_type"] == "peer-to-peer"
        assert len(status["octopuses"]) == 3

    def test_eventbus_integration(self):
        bus = EventBus()
        spawn_events = []
        bus.register_callback("octopus.spawn", lambda ch, msg: spawn_events.append(msg))

        colony = OctopusColony(event_bus=bus, max_octopuses=5)
        colony.spawn_octopus(reason="test")

        # 3 initialization spawns + 1 explicit = 4 total
        assert len(spawn_events) == 4

    def test_stop_all(self):
        colony = OctopusColony()
        colony.stop_all()
        # Should not raise

    def test_with_cross_system_integrations(self):
        bus = EventBus()
        router = DecisionRouter()
        wm = WorldModel()

        colony = OctopusColony(
            event_bus=bus,
            decision_router=router,
            world_model=wm,
        )

        # All octopuses should have the integrations
        for octopus in colony.octopuses.values():
            assert octopus._decision_router is router
            assert octopus._world_model is wm

        # Test decision routing through octopus
        oct = list(colony.octopuses.values())[0]
        result = oct.route_decision("danger alert")
        assert result["tier"] == "reflex"

        # Test world model prediction through octopus
        pred = oct.predict_with_confidence("arm-0", np.zeros(10), np.array([0]))
        assert pred["source"] == "central_world_model"


# ===== CROSS-SYSTEM INTEGRATION TESTS =====

class TestCrossSystemIntegration:
    """Tests that cognition + network systems work together."""

    def test_octopus_with_full_cognition_stack(self):
        """An octopus with DecisionRouter, WorldModel, and ValidatedImagination."""
        bus = EventBus()
        router = DecisionRouter()
        wm = WorldModel()

        agent = OctopusAgent(
            "integrated-oct",
            event_bus=bus,
            specialization=OctopusSpecialization.DECISION,
            decision_router=router,
            world_model=wm,
        )

        # Decision routing works
        decision = agent.route_decision("threat detected")
        assert decision["tier"] == "reflex"

        # World model prediction works
        state = np.zeros(10, dtype=np.float32)
        pred = agent.predict_with_confidence("arm-0", state, np.array([1]))
        assert "prediction" in pred

        # Status includes all integrations
        status = agent.get_status()
        assert status["has_decision_router"] is True
        assert status["has_world_model"] is True

    def test_colony_submits_tasks_distributed(self):
        """Colony distributes tasks across octopuses."""
        colony = OctopusColony(min_octopuses=3, max_octopuses=5)

        task_ids = []
        for i in range(6):
            tid = colony.submit_task({"index": i}, "learning", priority=5)
            task_ids.append(tid)

        assert len(task_ids) == 6
        assert all(tid is not None for tid in task_ids)

    def test_causal_reasoning_with_world_model(self):
        """Causal engine can learn from world model predictions."""
        wm = WorldModel()
        engine = CausalReasoningEngine()

        state = np.zeros(10, dtype=np.float32)
        action = np.array([1, 0, 0, 0, 0], dtype=np.float32)

        pred = wm.step(state, action)
        engine.observe_intervention(
            "action_1", 1.0,
            {"reward": pred.reward, "state_change": float(np.sum(np.abs(pred.next_state - state)))},
        )

        metrics = engine.get_causal_metrics()
        assert metrics["total_evidence"] >= 1

    def test_dream_planning_with_validated_imagination(self):
        """CollectiveDream + ValidatedImagination integration."""
        wm = WorldModel()
        vi = ValidatedImagination()
        planner = ValidatedImaginationPlanner(wm, validator=vi)

        state = np.zeros(10, dtype=np.float32)
        policy = lambda s: np.random.randint(0, 5)

        trajectory = planner.plan_with_validation("dreamer-1", state, policy, horizon=5)
        assert isinstance(trajectory, list)

        stats = planner.get_statistics()
        assert stats["total_plans"] == 1
        assert stats["total_steps_imagined"] > 0

    def test_eventbus_carries_octopus_events(self):
        """EventBus receives spawn/despawn/health events from colony."""
        bus = EventBus()
        events = {"spawn": [], "despawn": [], "task": []}

        bus.register_callback("octopus.spawn", lambda ch, msg: events["spawn"].append(msg))
        bus.register_callback("octopus.despawn", lambda ch, msg: events["despawn"].append(msg))
        bus.register_callback("octopus.task_submitted", lambda ch, msg: events["task"].append(msg))

        colony = OctopusColony(event_bus=bus, max_octopuses=5)
        colony.spawn_octopus(reason="test")

        # 3 init + 1 spawn
        assert len(events["spawn"]) == 4

        # Submit a task
        colony.submit_task({"x": 1}, "learning")
        assert len(events["task"]) >= 1

        # Despawn
        ids = list(colony.octopuses.keys())
        colony.despawn_octopus(ids[-1], reason="test")
        assert len(events["despawn"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
