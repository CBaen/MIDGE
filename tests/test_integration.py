"""Integration Tests - Mae as a living organism.

Tests the whole system working together, not individual parts.
Verifies 33-layer bootstrap, agent lifecycle, cross-system communication,
persistence round-trips, and circadian cycle transitions.
"""

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from main import create_mae


@pytest.fixture
def mae_organism(tmp_path):
    """Create a minimal Mae organism for testing."""
    model, systems = create_mae(
        num_agents=3,
        cycle_length=20,
        persist_dir=str(tmp_path / "mae_test"),
    )
    yield model, systems
    model.shutdown()


@pytest.fixture
def persist_dir(tmp_path):
    """Temporary persistence directory for round-trip tests."""
    d = tmp_path / "mae_persist"
    d.mkdir()
    yield str(d)


class TestBootstrap:
    """Verify the 33-layer bootstrap creates a complete organism."""

    def test_model_created(self, mae_organism):
        model, systems = mae_organism
        assert model is not None
        assert model.event_bus is not None
        assert model.state_store is not None

    def test_all_shared_systems_present(self, mae_organism):
        _, systems = mae_organism
        expected_keys = [
            "event_bus", "circadian", "endocrine",
            "enforcer", "watchdog", "auditor",
            "substrate", "physarum", "signal_bus", "gnn_communicator",
            "stigmergy", "quorum_space", "predictive_field",
            "knowledge_base", "transfer_engine", "maml_learner",
            "curiosity", "haven", "imitation",
            "threat_detector", "input_validator", "pearl_defense",
            "shared_world_model", "collective_dream",
            "validated_imagination", "shared_causal_engine",
            "auto_healer", "capability_discovery", "somatic_map",
            "morph_coordinator", "organ_builder",
            "temporal_memory", "worldline_planner",
            "holon_registry",
            "connection_registry",
            "awareness_pulse",
            "fractal_generator",
            "stem_cell_registry",
            "task_pool",
            "organism_action",
            "integration_meter",
            "topology_analyzer",
            "witness_notifier",
            "triadic_verifier",
            "attentional_gate",
            "mitosis_monitor",
        ]
        for key in expected_keys:
            assert key in systems, f"Missing system: {key}"
            assert systems[key] is not None, f"System is None: {key}"

    def test_market_systems_present(self, mae_organism):
        """Market intelligence systems exist in systems dict (Layer 33).

        Market systems tolerate None (graceful degradation) — we only
        check that the key exists, not that it's non-None.
        """
        _, systems = mae_organism
        market_keys = [
            "sec_edgar_client", "price_fetcher", "house_stock_watcher",
            "job_tracker", "usa_spending_client", "sam_gov_client",
            "cluster_detector", "politician_tracker", "filing_time_analyzer",
            "contract_predictor", "thompson_sampler", "convergence_alerter",
            "velocity_detector", "correlation_tracker", "outcome_tracker",
            "session_sweep_detector",
            "signal_archive_reader",
            "lag_correlation_analyzer",
            "thompson_calibrator",
            "kelly_position_sizer",
        ]
        for key in market_keys:
            assert key in systems, f"Missing market system: {key}"

    def test_agents_created(self, mae_organism):
        model, systems = mae_organism
        assert len(systems["agents"]) == 3
        assert len(model.agents) == 3

    def test_per_agent_systems_created(self, mae_organism):
        _, systems = mae_organism
        per_agent = systems["per_agent_systems"]
        assert len(per_agent) == 3
        for uid, agent_sys in per_agent.items():
            assert "memory_coordinator" in agent_sys
            assert "world_model" in agent_sys
            assert "decision_router" in agent_sys
            assert "causal_engine" in agent_sys
            assert "quorum_sensor" in agent_sys

    def test_learning_engines_created(self, mae_organism):
        _, systems = mae_organism
        assert len(systems["frl_engines"]) == 3
        assert len(systems["vdn_engines"]) == 3

    def test_triad_enforcement_active(self, mae_organism):
        _, systems = mae_organism
        report = systems["triad_report"]
        assert report["total_registered"] == 16
        assert report["compliance_rate"] == 1.0


class TestAgentLifecycle:
    """Verify agents step through their lifecycle correctly."""

    def test_agents_step(self, mae_organism):
        model, systems = mae_organism
        model.step()
        for agent in systems["agents"]:
            assert agent.step_count >= 1

    def test_agents_accumulate_steps(self, mae_organism):
        model, systems = mae_organism
        model.run(5)
        for agent in systems["agents"]:
            assert agent.step_count >= 5

    def test_agent_state_serializable(self, mae_organism):
        model, systems = mae_organism
        model.run(3)
        for agent in systems["agents"]:
            state = agent.serialize_state()
            # Must be JSON-serializable
            serialized = json.dumps(state)
            restored = json.loads(serialized)
            assert restored["step_count"] == state["step_count"]
            assert "convergence" in restored
            assert "gamification" in restored

    def test_agent_get_all_statistics(self, mae_organism):
        model, systems = mae_organism
        model.run(3)
        for agent in systems["agents"]:
            stats = agent.get_all_statistics()
            assert "base" in stats
            assert "performance" in stats
            assert "convergence" in stats


class TestCircadianCycle:
    """Verify circadian rhythm drives phase transitions."""

    def test_circadian_starts_active(self, mae_organism):
        _, systems = mae_organism
        circadian = systems["circadian"]
        stats = circadian.get_statistics()
        assert stats["current_phase"] == "ACTIVE"

    def test_circadian_transitions(self, mae_organism):
        model, systems = mae_organism
        circadian = systems["circadian"]
        # Run enough steps to complete at least one full cycle
        # cycle_length=20: ACTIVE=12, CONSOL=5, REST=3
        model.run(25)
        stats = circadian.get_statistics()
        assert stats["cycle_count"] >= 1

    def test_endocrine_responds_to_phases(self, mae_organism):
        model, systems = mae_organism
        endocrine = systems["endocrine"]
        model.run(25)
        stats = endocrine.get_statistics()
        # After a full cycle, endocrine should have processed phase changes
        assert "hormone_levels" in stats or "is_stressed" in stats


class TestCrossSystems:
    """Verify systems communicate across boundaries."""

    def test_eventbus_has_callbacks(self, mae_organism):
        _, systems = mae_organism
        bus = systems["event_bus"]
        stats = bus.get_stats()
        # Should have callback channels from cross-wiring
        assert stats["callback_channels"] > 0

    def test_somatic_map_exists(self, mae_organism):
        _, systems = mae_organism
        somatic = systems["somatic_map"]
        assert somatic is not None
        assert hasattr(somatic, "get_statistics")

    def test_haven_registered_with_model(self, mae_organism):
        model, _ = mae_organism
        assert model.haven_coordinator is not None

    def test_vdn_registered_with_model(self, mae_organism):
        model, _ = mae_organism
        assert model.vdn_engine is not None

    def test_substrate_has_agents(self, mae_organism):
        _, systems = mae_organism
        substrate = systems["substrate"]
        if hasattr(substrate, "get_statistics"):
            stats = substrate.get_statistics()
            assert stats.get("registered_agents", 0) >= 3


class TestPersistence:
    """Verify agent state survives restart."""

    def test_save_and_restore_agents(self, persist_dir):
        # Run 1: Create, run, shutdown
        m1, s1 = create_mae(num_agents=3, cycle_length=20, persist_dir=persist_dir)
        m1.run(10)

        # Capture state before shutdown
        agent_steps_before = {a.unique_id: a.step_count for a in s1["agents"]}
        m1.shutdown()

        # Run 2: Reboot — agents should restore
        m2, s2 = create_mae(num_agents=3, cycle_length=20, persist_dir=persist_dir)

        for agent in s2["agents"]:
            assert agent.step_count >= 10, (
                f"Agent {agent.unique_id} lost state: step_count={agent.step_count}"
            )
        m2.shutdown()

    def test_persistence_accumulates(self, persist_dir):
        # Run 1: 5 steps
        m1, _ = create_mae(num_agents=3, cycle_length=20, persist_dir=persist_dir)
        m1.run(5)
        m1.shutdown()

        # Run 2: 5 more steps
        m2, s2 = create_mae(num_agents=3, cycle_length=20, persist_dir=persist_dir)
        m2.run(5)

        for agent in s2["agents"]:
            assert agent.step_count >= 10, (
                f"Steps didn't accumulate: {agent.step_count}"
            )
        m2.shutdown()

    def test_model_state_persists(self, persist_dir):
        m1, _ = create_mae(num_agents=3, cycle_length=20, persist_dir=persist_dir)
        m1.run(5)
        m1.shutdown()

        # Verify state file exists
        state_path = Path(persist_dir) / "state_store.json"
        assert state_path.exists()
        data = json.loads(state_path.read_text())
        assert "store" in data

    def test_mixin_state_persists(self, persist_dir):
        m1, s1 = create_mae(num_agents=3, cycle_length=20, persist_dir=persist_dir)
        m1.run(5)

        # Get mixin state before shutdown
        agent = s1["agents"][0]
        state_before = agent.serialize_state()
        m1.shutdown()

        # Reboot and check mixin state
        m2, s2 = create_mae(num_agents=3, cycle_length=20, persist_dir=persist_dir)
        agent2 = s2["agents"][0]
        state_after = agent2.serialize_state()

        assert state_after["convergence"]["learning_iterations"] >= state_before["convergence"]["learning_iterations"]
        m2.shutdown()


class TestFullRun:
    """Verify Mae can run a complete simulation without errors."""

    def test_short_run(self, mae_organism):
        model, _ = mae_organism
        model.run(10)
        stats = model.get_system_stats()
        assert stats["current_step"] >= 10
        assert stats["active_agents"] >= 3  # starts with 3, may grow via ReproductiveSystem

    def test_full_circadian_cycle(self, mae_organism):
        model, systems = mae_organism
        # cycle_length=20, run 25 to ensure one full cycle
        model.run(25)
        stats = model.get_system_stats()
        assert stats["current_step"] >= 25

    def test_model_stats_available(self, mae_organism):
        model, _ = mae_organism
        model.run(5)
        stats = model.get_system_stats()
        assert "current_step" in stats
        assert "active_agents" in stats
        assert "backbone" in stats
        assert "engines_enabled" in stats


class TestPhysarumOptimizer:
    """Verify Physarum slime mold topology optimization."""

    def test_physarum_exists(self, mae_organism):
        _, systems = mae_organism
        physarum = systems["physarum"]
        assert physarum is not None
        assert hasattr(physarum, "get_statistics")

    def test_physarum_runs_with_model(self, mae_organism):
        model, systems = mae_organism
        model.run(10)
        stats = systems["physarum"].get_statistics()
        assert stats["optimization_steps"] >= 10
        assert stats["enabled"] is True

    def test_physarum_tracks_conductance(self, mae_organism):
        model, systems = mae_organism
        model.run(5)
        stats = systems["physarum"].get_statistics()
        assert "avg_conductance" in stats
        assert stats["avg_conductance"] >= 0.0

    def test_physarum_config_accessible(self, mae_organism):
        _, systems = mae_organism
        stats = systems["physarum"].get_statistics()
        assert "config" in stats
        assert "mu" in stats["config"]
        assert "gamma" in stats["config"]


class TestPearlDefense:
    """Verify nacre-inspired pearl defense system."""

    def test_pearl_defense_exists(self, mae_organism):
        _, systems = mae_organism
        pearl = systems["pearl_defense"]
        assert pearl is not None
        assert hasattr(pearl, "get_statistics")

    def test_pearl_defense_initial_stats(self, mae_organism):
        _, systems = mae_organism
        stats = systems["pearl_defense"].get_statistics()
        assert stats["total_started"] == 0
        assert stats["total_completed"] == 0
        assert stats["total_dissolved"] == 0
        assert stats["active_pearls"] == 0

    def test_pearl_defense_passes_clean_input(self, mae_organism):
        _, systems = mae_organism
        pearl = systems["pearl_defense"]
        validator = systems["input_validator"]
        # Set high trust so inputs pass through
        validator.set_trust("trusted_source", 0.9)
        report = pearl.validate("trusted_source", "message", {"data": "hello"})
        assert report.result.value == "passed"

    def test_pearl_defense_runs_with_model(self, mae_organism):
        model, systems = mae_organism
        model.run(5)
        # Pearl step hook should have run without errors
        stats = systems["pearl_defense"].get_statistics()
        assert isinstance(stats["active_pearls"], int)


class TestOctopusMemory:
    """Verify octopus memory integration (emergent capability #9)."""

    def test_octopus_agent_accepts_memory(self):
        from mae_core.backbone.event_bus import EventBus
        from mae_core.network.octopus_agent import OctopusAgent
        from mae_core.memory.coordinator import MemoryCoordinator

        bus = EventBus()
        memory = MemoryCoordinator(event_bus=bus, agent_id="octopus-test")
        agent = OctopusAgent(
            octopus_id="test-oct",
            event_bus=bus,
            memory_coordinator=memory,
        )
        status = agent.get_status()
        assert status["has_memory"] is True

    def test_octopus_agent_without_memory(self):
        from mae_core.network.octopus_agent import OctopusAgent

        agent = OctopusAgent(octopus_id="test-oct-no-mem")
        status = agent.get_status()
        assert status["has_memory"] is False

    def test_octopus_records_task_outcome(self):
        from mae_core.backbone.event_bus import EventBus
        from mae_core.network.octopus_agent import OctopusAgent
        from mae_core.memory.coordinator import MemoryCoordinator

        bus = EventBus()
        memory = MemoryCoordinator(event_bus=bus, agent_id="octopus-mem")
        agent = OctopusAgent(
            octopus_id="test-oct-mem",
            event_bus=bus,
            memory_coordinator=memory,
        )
        # Record a task outcome
        agent.record_task_outcome("hunt", success=True, reward=1.0)
        # Memory should have stored it
        assert memory.episodic.total_stored >= 1

    def test_octopus_recalls_similar(self):
        from mae_core.backbone.event_bus import EventBus
        from mae_core.network.octopus_agent import OctopusAgent
        from mae_core.memory.coordinator import MemoryCoordinator

        bus = EventBus()
        memory = MemoryCoordinator(event_bus=bus, agent_id="octopus-recall")
        agent = OctopusAgent(
            octopus_id="test-oct-recall",
            event_bus=bus,
            memory_coordinator=memory,
        )
        # Store some experiences
        agent.record_task_outcome("hunt", success=True, reward=1.0)
        agent.record_task_outcome("hunt", success=False, reward=-0.5)
        # Recall should return stored experiences
        results = agent._recall_similar("hunt", k=5)
        assert len(results) >= 1


class TestHolonProtocol:
    """Holon Protocol integration: fractal self-awareness at every level."""

    def test_holon_registry_exists(self, mae_organism):
        _, systems = mae_organism
        registry = systems["holon_registry"]
        assert registry is not None

    def test_holon_registry_has_mae_root(self, mae_organism):
        _, systems = mae_organism
        registry = systems["holon_registry"]
        entry = registry.get_entry("mae")
        assert entry is not None
        assert entry.holon_type == "organism"
        assert entry.parent_id is None

    def test_holon_registry_has_colony(self, mae_organism):
        _, systems = mae_organism
        registry = systems["holon_registry"]
        entry = registry.get_entry("colony")
        assert entry is not None
        assert entry.holon_type == "colony"
        # After fractal organize, colony is in somatic-system/growth subsystem
        assert entry.parent_id == "growth"

    def test_agents_registered_as_holons(self, mae_organism):
        _, systems = mae_organism
        registry = systems["holon_registry"]
        agents = systems["agents"]
        for agent in agents:
            entry = registry.get_entry(str(agent.unique_id))
            assert entry is not None, f"Agent {agent.unique_id} not in holon registry"
            assert entry.holon_type == "agent"
            # Agents are now children of triads (tissue level), not colony directly
            assert entry.parent_id.startswith("agent_triad_"), (
                f"Agent {agent.unique_id} parent={entry.parent_id}, "
                f"expected agent_triad_*"
            )

    def test_shared_systems_registered_as_holons(self, mae_organism):
        _, systems = mae_organism
        registry = systems["holon_registry"]
        # Spot-check a few key systems — after fractal organize,
        # systems are children of subsystems, not direct children of mae
        for name in ("event_bus", "circadian", "substrate", "somatic_map"):
            entry = registry.get_entry(name)
            assert entry is not None, f"System {name} not in holon registry"
            assert entry.holon_type == "system"
            # Parent is now a subsystem, not mae directly
            assert entry.parent_id != "mae", f"{name} should be in a subsystem, not mae"

    def test_agents_know_their_peers(self, mae_organism):
        _, systems = mae_organism
        agents = systems["agents"]
        agent = agents[0]
        peers = agent.holon_know_peers()
        # With 3 agents, each should see 2 peers
        assert len(peers) == 2

    def test_agents_know_parent(self, mae_organism):
        _, systems = mae_organism
        agents = systems["agents"]
        agent = agents[0]
        parent_info = agent.holon_know_up()
        assert parent_info is not None
        # Agents are now in triads (tissue level)
        assert parent_info["parent_id"].startswith("agent_triad_")

    def test_agent_holon_know_self(self, mae_organism):
        _, systems = mae_organism
        agent = systems["agents"][0]
        self_model = agent.holon_know_self()
        assert self_model["holon_type"] == "agent"
        assert "capabilities" in self_model
        assert "performance" in self_model

    def test_holon_statistics_in_agent_stats(self, mae_organism):
        _, systems = mae_organism
        agent = systems["agents"][0]
        all_stats = agent.get_all_statistics()
        assert "holon" in all_stats
        assert "holon_id" in all_stats["holon"]

    def test_holon_serialization_roundtrip(self, mae_organism):
        _, systems = mae_organism
        agent = systems["agents"][0]
        agent.holon_remember("test_key", "test_value")
        state = agent.serialize_state()
        assert "holon" in state

        # Create a fresh agent and restore
        from mae_core.agents.mycelial_agent import MycelialAgent
        model = systems["model"]
        agent2 = MycelialAgent(model)
        agent2.restore_state(state)
        assert agent2._holon_memory.get("test_key") == "test_value"

    def test_holon_registry_statistics(self, mae_organism):
        _, systems = mae_organism
        registry = systems["holon_registry"]
        stats = registry.get_statistics()
        assert stats["total_holons"] > 30  # 33 systems + 3 agents + mae + colony
        assert "agent" in stats["type_counts"]
        assert "system" in stats["type_counts"]
        assert stats["roots"] == ["mae"]


class TestConnectionTriads:
    """Connection Registry integration: triadic witnessing for all connections."""

    def test_connection_registry_exists(self, mae_organism):
        _, systems = mae_organism
        registry = systems["connection_registry"]
        assert registry is not None

    def test_connections_registered_at_bootstrap(self, mae_organism):
        _, systems = mae_organism
        registry = systems["connection_registry"]
        stats = registry.get_statistics()
        assert stats["total_connections"] > 40

    def test_all_connections_have_witnesses(self, mae_organism):
        _, systems = mae_organism
        registry = systems["connection_registry"]
        bare = registry.get_bare_dyads()
        assert len(bare) == 0, (
            f"Bare dyads: {[b.connection_id for b in bare]}"
        )

    def test_no_bare_dyads(self, mae_organism):
        _, systems = mae_organism
        registry = systems["connection_registry"]
        stats = registry.get_statistics()
        assert stats["bare_dyads"] == 0

    def test_connection_verify_runs(self, mae_organism):
        model, systems = mae_organism
        model.run(30)
        registry = systems["connection_registry"]
        stats = registry.get_statistics()
        assert stats["total_verifications"] >= 1

    def test_connection_statistics(self, mae_organism):
        _, systems = mae_organism
        registry = systems["connection_registry"]
        stats = registry.get_statistics()
        assert "type_counts" in stats
        assert "eventbus_pubsub" in stats["type_counts"]
        assert "direct_reference" in stats["type_counts"]
        assert "witness_load" in stats
        assert stats["systems_connected"] > 10


class TestBidirectionalAwareness:
    """Bidirectional awareness: every system knows its place in the hierarchy."""

    def test_all_shared_systems_have_holon_proxy(self, mae_organism):
        _, systems = mae_organism
        shared_names = [
            "circadian", "endocrine", "substrate", "physarum",
            "signal_bus", "somatic_map", "auto_healer", "knowledge_base",
            "connection_registry", "awareness_pulse",
        ]
        for name in shared_names:
            system = systems[name]
            assert hasattr(system, "_holon"), f"{name} missing _holon proxy"
            assert system._holon is not None, f"{name}._holon is None"
            assert system._holon.holon_id == name

    def test_system_knows_parent(self, mae_organism):
        _, systems = mae_organism
        # After fractal organize, auto_healer's parent is "emergence" subsystem
        parent = systems["auto_healer"]._holon.know_up()
        assert parent is not None
        assert parent["parent_id"] == "emergence"
        assert parent["parent_type"] == "subsystem"

    def test_system_knows_peers(self, mae_organism):
        _, systems = mae_organism
        # After fractal organize, substrate is in "fabric" subsystem
        # with physarum and signal_bus as peers
        peers = systems["substrate"]._holon.know_peers()
        peer_ids = {p["holon_id"] for p in peers}
        assert "physarum" in peer_ids
        assert "signal_bus" in peer_ids
        assert len(peers) == 2  # Exactly 2 peers in a triad

    def test_awareness_pulse_runs(self, mae_organism):
        model, systems = mae_organism
        model.run(30)
        stats = systems["awareness_pulse"].get_statistics()
        assert stats["pulse_count"] >= 1


class TestFractalGenerator:
    """Verify the fractal generator creates explicit recursive structure."""

    def test_fractal_generator_exists(self, mae_organism):
        _, systems = mae_organism
        assert "fractal_generator" in systems
        assert systems["fractal_generator"] is not None

    def test_hierarchy_depth(self, mae_organism):
        _, systems = mae_organism
        registry = systems["holon_registry"]
        stats = registry.get_statistics()
        # mae → organ → subsystem → system = 3, plus colony → agent = 4
        assert stats["max_depth"] >= 4

    def test_organs_exist(self, mae_organism):
        _, systems = mae_organism
        registry = systems["holon_registry"]
        for organ_name in ("nervous-system", "sensory-system", "cognitive-system", "somatic-system"):
            entry = registry.get_entry(organ_name)
            assert entry is not None, f"Organ {organ_name} missing"
            assert entry.holon_type == "organ"
        # nervous-system is the bridge organ — stays as direct mae child
        assert registry.get_entry("nervous-system").parent_id == "mae"
        # Other organs are under organ clusters (after ORGAN_GROUPING)
        mae_subtree = registry.get_subtree("mae")
        for organ_name in ("sensory-system", "cognitive-system", "somatic-system", "metabolic-system"):
            assert organ_name in mae_subtree, f"Organ {organ_name} not in mae subtree"

    def test_subsystems_exist(self, mae_organism):
        _, systems = mae_organism
        registry = systems["holon_registry"]
        from mae_core.backbone.fractal_generator import FRACTAL_GROUPING
        for organ_name, subsystems_spec in FRACTAL_GROUPING.items():
            organ_subtree = registry.get_subtree(organ_name)
            for sub_name in subsystems_spec:
                entry = registry.get_entry(sub_name)
                assert entry is not None, f"Subsystem {sub_name} missing"
                assert entry.holon_type == "subsystem"
                assert sub_name in organ_subtree, (
                    f"Subsystem {sub_name} not in {organ_name} subtree"
                )

    def test_all_systems_reachable_from_mae(self, mae_organism):
        _, systems = mae_organism
        registry = systems["holon_registry"]
        subtree = registry.get_subtree("mae")
        for name in ("event_bus", "substrate", "auto_healer", "colony"):
            assert name in subtree, f"{name} not reachable from mae"

    def test_fractal_connections_have_witnesses(self, mae_organism):
        _, systems = mae_organism
        conn_registry = systems["connection_registry"]
        bare = conn_registry.get_bare_dyads()
        assert len(bare) == 0, f"Bare dyads found: {bare}"

    def test_fractal_generator_has_proxy(self, mae_organism):
        _, systems = mae_organism
        gen = systems["fractal_generator"]
        assert hasattr(gen, "_holon")
        self_info = gen._holon.know_self()
        assert self_info["holon_type"] == "system"


class TestStemCellRefactor:
    """Verify the stem cell registry tracks agent genomes and epigenomes."""

    def test_stem_cell_registry_exists(self, mae_organism):
        _, systems = mae_organism
        assert "stem_cell_registry" in systems
        assert systems["stem_cell_registry"] is not None

    def test_all_agents_registered(self, mae_organism):
        _, systems = mae_organism
        registry = systems["stem_cell_registry"]
        stats = registry.get_statistics()
        assert stats["total_registered"] == 3

    def test_default_role_is_stem(self, mae_organism):
        _, systems = mae_organism
        registry = systems["stem_cell_registry"]
        dist = registry.get_role_distribution()
        assert dist == {"STEM": 3}

    def test_redifferentiate_changes_role(self, mae_organism):
        _, systems = mae_organism
        registry = systems["stem_cell_registry"]
        agents = systems["agents"]
        agent_id = str(agents[0].unique_id)
        registry.redifferentiate(agent_id, "EXPLORER", step=1)
        epi = registry.get_epigenome(agent_id)
        assert epi.role == "EXPLORER"
        assert epi.lineage == ["STEM"]
        dist = registry.get_role_distribution()
        assert dist["EXPLORER"] == 1
        assert dist["STEM"] == 2

    def test_stem_cell_registry_has_proxy(self, mae_organism):
        _, systems = mae_organism
        reg = systems["stem_cell_registry"]
        assert hasattr(reg, "_holon")
        self_info = reg._holon.know_self()
        assert self_info["holon_type"] == "system"


class TestTier2Persistence:
    """Verify Tier 2 persistence: learned state survives restart."""

    def test_memory_survives_restart(self, tmp_path):
        """Store experiences, shutdown, reboot, verify memory is restored."""
        import numpy as np

        # Boot 1: create Mae, store experiences, shutdown
        model1, sys1 = create_mae(num_agents=2, persist_dir=tmp_path / "mae")
        agent_sys = list(sys1["per_agent_systems"].values())[0]
        mc = agent_sys["memory_coordinator"]
        for i in range(10):
            mc.store(
                state=np.array([float(i), 0.1]),
                action=i % 3,
                reward=float(i) * 0.1,
                next_state=np.array([float(i) + 0.1, 0.2]),
                done=False,
            )
        assert len(mc.episodic) == 10
        model1.shutdown()

        # Boot 2: create Mae again, verify memory restored
        model2, sys2 = create_mae(num_agents=2, persist_dir=tmp_path / "mae")
        agent_sys2 = list(sys2["per_agent_systems"].values())[0]
        mc2 = agent_sys2["memory_coordinator"]
        assert len(mc2.episodic) == 10
        model2.shutdown()

    def test_vdn_q_table_survives_restart(self, tmp_path):
        """VDN Q-table persists across restarts."""
        import numpy as np

        model1, sys1 = create_mae(num_agents=2, persist_dir=tmp_path / "mae")
        vdn_engines = sys1["vdn_engines"]
        first_uid = list(vdn_engines.keys())[0]
        vdn = vdn_engines[first_uid]

        state = np.zeros(10)
        vdn.update_value_function(state, 0, 1.0, state * 1.1, False)
        q_before = len(vdn._q_table)
        assert q_before > 0
        model1.shutdown()

        model2, sys2 = create_mae(num_agents=2, persist_dir=tmp_path / "mae")
        vdn2 = sys2["vdn_engines"][first_uid]
        assert len(vdn2._q_table) == q_before
        model2.shutdown()

    def test_shared_knowledge_persists(self, tmp_path):
        """KnowledgeBase tasks and episodes survive restart."""
        from mae_core.learning.knowledge_base import TaskDescriptor

        model1, sys1 = create_mae(num_agents=2, persist_dir=tmp_path / "mae")
        kb = sys1["knowledge_base"]
        kb.register_task(TaskDescriptor(task_id="nav", name="navigate", domain="grid"))
        kb.store_episode("nav", "a0", [{"s": 0}], 10.0, True)
        model1.shutdown()

        model2, sys2 = create_mae(num_agents=2, persist_dir=tmp_path / "mae")
        kb2 = sys2["knowledge_base"]
        eps = kb2.retrieve_successful_episodes("nav")
        assert len(eps) == 1
        assert eps[0].total_reward == 10.0
        model2.shutdown()
