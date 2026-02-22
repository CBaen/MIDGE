"""Unit tests for Agent Mitosis -- autopoietic production loop."""

import json
from collections import deque

import pytest

from mae_core.agents.mitosis import (
    CADENCE,
    CH_AGENT_MITOSIS,
    HEALTH_THRESHOLD,
    MEMORY_SEED_SIZE,
    MIN_HISTORY,
    MUTATION_RATE,
    MitosisMonitor,
)
from mae_core.agents.stem_cell import ROLE_PROFILES, StemCellRegistry
from mae_core.backbone.event_bus import EventBus
from mae_core.backbone.holon_protocol import HolonRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockModel:
    """Minimal Mesa model stand-in for testing."""

    def __init__(self):
        self.agents = []
        self.total_agents_created = 0
        self._next_unique_id = 100
        self.time = 0

    def track_agent_created(self, agent):
        self.total_agents_created += 1


class MockAgent:
    """Minimal agent stand-in for testing mitosis."""

    _id_counter = 0

    def __init__(self, unique_id=None, health_rewards=None):
        if unique_id is not None:
            self.unique_id = unique_id
        else:
            MockAgent._id_counter += 1
            self.unique_id = MockAgent._id_counter
        self.agent_config: dict = {}
        self.reward_history = deque(maxlen=200)
        if health_rewards:
            for r in health_rewards:
                self.reward_history.append(r)
        # Attributes that exist on real agents
        self.sensing_radius = 5.0
        self.exploration_bonus = 0.1
        self.max_learning_iterations = 100
        self.step_count = 0


def _make_healthy_agent(uid: int) -> MockAgent:
    """Create a MockAgent with health well above HEALTH_THRESHOLD."""
    # Health = (mean_reward + 1) / 2. For health=0.85: mean_reward=0.7
    return MockAgent(unique_id=uid, health_rewards=[0.7] * (MIN_HISTORY + 5))


def _make_unhealthy_agent(uid: int) -> MockAgent:
    """Create a MockAgent with health well below HEALTH_THRESHOLD."""
    return MockAgent(unique_id=uid, health_rewards=[-0.8] * (MIN_HISTORY + 5))


def _make_monitor(
    agents: list[MockAgent] | None = None,
    max_agents: int = 10,
    bus: EventBus | None = None,
) -> tuple[MitosisMonitor, StemCellRegistry, HolonRegistry, EventBus, MockModel]:
    """Create a MitosisMonitor with standard test fixtures."""
    bus = bus or EventBus()
    model = MockModel()
    registry = StemCellRegistry(event_bus=bus)
    holon_reg = HolonRegistry()
    holon_reg.register("mae", holon_type="organism")
    holon_reg.register("colony", holon_type="colony", parent_id="mae")

    if agents:
        for agent in agents:
            registry.register(agent)

    monitor = MitosisMonitor(
        model=model,
        stem_cell_registry=registry,
        holon_registry=holon_reg,
        event_bus=bus,
        max_agents=max_agents,
    )
    return monitor, registry, holon_reg, bus, model


# ---------------------------------------------------------------------------
# TestMitosisMonitor
# ---------------------------------------------------------------------------

class TestMitosisMonitor:
    """Tests for the autopoietic production loop."""

    def test_healthy_parent_triggers_mitosis(self, monkeypatch):
        """A healthy agent in a colony with capacity should produce a child."""
        parent = _make_healthy_agent(1)
        monitor, registry, holon_reg, bus, model = _make_monitor(
            agents=[parent], max_agents=5,
        )

        events = []
        bus.register_callback(
            CH_AGENT_MITOSIS,
            lambda ch, msg: events.append(json.loads(msg)),
        )

        # Mock _create_child to avoid importing full MycelialAgent
        child = MockAgent(unique_id=99)
        registry.register(child)

        def fake_create(p, pid, step):
            return child

        monkeypatch.setattr(monitor, "_create_child", fake_create)

        monitor.step(current_step=CADENCE)

        assert len(events) == 1
        assert events[0]["parent_id"] == "1"
        assert events[0]["child_id"] == "99"
        assert monitor._birth_count == 1

    def test_no_mitosis_when_at_capacity(self):
        """Colony at max capacity should not produce new agents."""
        agents = [_make_healthy_agent(i) for i in range(5)]
        monitor, registry, holon_reg, bus, model = _make_monitor(
            agents=agents, max_agents=5,
        )

        monitor.step(current_step=CADENCE)
        assert monitor._birth_count == 0

    def test_no_mitosis_when_unhealthy(self):
        """Unhealthy agents should not trigger mitosis."""
        parent = _make_unhealthy_agent(1)
        monitor, registry, holon_reg, bus, model = _make_monitor(
            agents=[parent], max_agents=5,
        )

        monitor.step(current_step=CADENCE)
        assert monitor._birth_count == 0

    def test_no_mitosis_on_non_cadenced_step(self, monkeypatch):
        """Monitor does nothing on steps not divisible by CADENCE."""
        parent = _make_healthy_agent(1)
        monitor, registry, holon_reg, bus, model = _make_monitor(
            agents=[parent], max_agents=5,
        )

        child = MockAgent(unique_id=99)
        registry.register(child)
        monkeypatch.setattr(monitor, "_create_child", lambda p, pid, s: child)

        # Call on non-cadenced steps -- should be no-op
        non_cadenced = [s for s in range(1, CADENCE * 3) if s % CADENCE != 0]
        for s in non_cadenced:
            monitor.step(current_step=s)
        assert monitor._birth_count == 0

    def test_no_mitosis_on_step_zero(self, monkeypatch):
        """Step 0 is skipped even though 0 % CADENCE == 0."""
        parent = _make_healthy_agent(1)
        monitor, registry, holon_reg, bus, model = _make_monitor(
            agents=[parent], max_agents=5,
        )
        monitor.step(current_step=0)
        assert monitor._birth_count == 0

    def test_locked_agent_not_selected_as_parent(self):
        """Locked agents should not be eligible for mitosis."""
        parent = _make_healthy_agent(1)
        monitor, registry, holon_reg, bus, model = _make_monitor(
            agents=[parent], max_agents=5,
        )
        registry._epigenomes["1"].locked = True

        monitor.step(current_step=CADENCE)
        assert monitor._birth_count == 0

    def test_insufficient_history_skips_agent(self):
        """Agent without enough reward history is not eligible."""
        parent = MockAgent(unique_id=1, health_rewards=[0.7] * (MIN_HISTORY - 1))
        monitor, registry, holon_reg, bus, model = _make_monitor(
            agents=[parent], max_agents=5,
        )

        monitor.step(current_step=CADENCE)
        assert monitor._birth_count == 0

    def test_mutate_epigenome_can_change_role(self, monkeypatch):
        """With mutation, child role differs from parent."""
        parent = _make_healthy_agent(1)
        monitor, _, _, _, _ = _make_monitor(agents=[parent])

        # Force mutation to always happen
        monkeypatch.setattr("mae_core.agents.mitosis.random.random", lambda: 0.0)
        # Pick a deterministic role
        monkeypatch.setattr("mae_core.agents.mitosis.random.choice", lambda x: "HEALER")

        child_role = monitor._mutate_epigenome("EXPLORER")
        assert child_role == "HEALER"

    def test_mutate_epigenome_inherits_role(self, monkeypatch):
        """Without mutation, child inherits parent role."""
        parent = _make_healthy_agent(1)
        monitor, _, _, _, _ = _make_monitor(agents=[parent])

        # Force no mutation
        monkeypatch.setattr("mae_core.agents.mitosis.random.random", lambda: 1.0)

        child_role = monitor._mutate_epigenome("EXPLORER")
        assert child_role == "EXPLORER"

    def test_memory_seeding(self):
        """Parent's recent rewards are seeded into child."""
        parent = MockAgent(unique_id=1, health_rewards=[0.5] * 20)
        child = MockAgent(unique_id=2)

        monitor, _, _, _, _ = _make_monitor()
        monitor._seed_memory(parent, child)

        assert len(child.reward_history) == MEMORY_SEED_SIZE
        assert all(r == 0.5 for r in child.reward_history)

    def test_memory_seeding_insufficient_history(self):
        """Parent with too few rewards doesn't seed child."""
        parent = MockAgent(unique_id=1, health_rewards=[0.5] * 2)
        child = MockAgent(unique_id=2)

        monitor, _, _, _, _ = _make_monitor()
        monitor._seed_memory(parent, child)

        assert len(child.reward_history) == 0

    def test_picks_healthiest_parent(self):
        """Monitor selects the healthiest eligible agent as parent."""
        # Agent 1: health ~0.75, Agent 2: health ~0.85
        agent1 = MockAgent(unique_id=1, health_rewards=[0.5] * (MIN_HISTORY + 5))
        agent2 = MockAgent(unique_id=2, health_rewards=[0.7] * (MIN_HISTORY + 5))
        monitor, registry, _, _, _ = _make_monitor(agents=[agent1, agent2])

        result = monitor._pick_parent()
        assert result is not None
        parent_id, parent = result
        assert parent_id == "2"  # agent2 is healthier

    def test_colony_at_capacity_check(self):
        """Colony at capacity returns True."""
        agents = [_make_healthy_agent(i) for i in range(3)]
        monitor, _, _, _, _ = _make_monitor(agents=agents, max_agents=3)
        assert monitor._colony_at_capacity() is True

    def test_colony_below_capacity_check(self):
        """Colony below capacity returns False."""
        agents = [_make_healthy_agent(i) for i in range(2)]
        monitor, _, _, _, _ = _make_monitor(agents=agents, max_agents=5)
        assert monitor._colony_at_capacity() is False

    def test_emits_event_with_correct_payload(self, monkeypatch):
        """EventBus event has all required fields."""
        parent = _make_healthy_agent(1)
        monitor, registry, holon_reg, bus, model = _make_monitor(
            agents=[parent], max_agents=5,
        )

        child = MockAgent(unique_id=42)
        registry.register(child)

        def fake_create(p, pid, step):
            return child

        monkeypatch.setattr(monitor, "_create_child", fake_create)

        events = []
        bus.register_callback(
            CH_AGENT_MITOSIS,
            lambda ch, msg: events.append(json.loads(msg)),
        )

        monitor.step(current_step=CADENCE)

        assert len(events) == 1
        evt = events[0]
        assert "parent_id" in evt
        assert "child_id" in evt
        assert "child_role" in evt
        assert "parent_health" in evt
        assert "step" in evt
        assert "total_births" in evt
        assert evt["step"] == CADENCE

    def test_statistics(self):
        """get_statistics returns expected shape."""
        agents = [_make_healthy_agent(i) for i in range(3)]
        monitor, _, _, _, _ = _make_monitor(agents=agents, max_agents=10)
        stats = monitor.get_statistics()
        assert "total_births" in stats
        assert "max_agents" in stats
        assert "current_agents" in stats
        assert "at_capacity" in stats
        assert stats["total_births"] == 0
        assert stats["current_agents"] == 3
        assert stats["at_capacity"] is False

    def test_no_duplicate_check_same_step(self, monkeypatch):
        """Calling step() twice with same step should only trigger once."""
        parent = _make_healthy_agent(1)
        monitor, registry, holon_reg, bus, model = _make_monitor(
            agents=[parent], max_agents=5,
        )

        child1 = MockAgent(unique_id=98)
        child2 = MockAgent(unique_id=99)
        children = iter([child1, child2])
        registry.register(child1)
        registry.register(child2)

        def fake_create(p, pid, step):
            return next(children)

        monkeypatch.setattr(monitor, "_create_child", fake_create)

        monitor.step(current_step=CADENCE)
        monitor.step(current_step=CADENCE)  # same step -- should be no-op
        assert monitor._birth_count == 1
