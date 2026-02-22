"""Unit tests for Stem Cell Protocol — genome, epigenome, role profiles, registry."""

import json

import pytest

from collections import deque

from mae_core.agents.stem_cell import (
    CH_STEM_CELL_REDIFFERENTIATED,
    CH_STEM_CELL_REGISTERED,
    DEFAULT_GENOME,
    ROLE_PROFILES,
    AgentEpigenome,
    AgentGenome,
    GeneDescriptor,
    StemCellRegistry,
    redifferentiate,
)
from mae_core.agents.redifferentiation_triggers import (
    CH_AUTO_REDIFFERENTIATED,
    CONSECUTIVE_REQUIRED,
    HEALTH_THRESHOLD,
    RedifferentiationMonitor,
)
from mae_core.backbone.event_bus import EventBus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockAgent:
    """Minimal agent stand-in for testing."""

    def __init__(self, unique_id: int = 1):
        self.unique_id = unique_id
        self.agent_config: dict = {}
        # Some attributes that exist on real agents
        self.sensing_radius = 5.0
        self.exploration_bonus = 0.1
        self.max_learning_iterations = 100


# ---------------------------------------------------------------------------
# TestGeneDescriptor
# ---------------------------------------------------------------------------

class TestGeneDescriptor:
    def test_creation(self):
        gene = GeneDescriptor("test", "mixin", int, 42, 0, 100)
        assert gene.name == "test"
        assert gene.mixin == "mixin"
        assert gene.default == 42

    def test_validation(self):
        gene = GeneDescriptor("x", "m", float, 0.5, 0.0, 1.0)
        assert gene.validate(0.5) is True
        assert gene.validate(1.5) is False
        assert gene.validate(-0.1) is False

    def test_bool_gene_no_range(self):
        gene = GeneDescriptor("flag", "m", bool, True)
        assert gene.validate(True) is True
        assert gene.validate(False) is True


# ---------------------------------------------------------------------------
# TestAgentGenome
# ---------------------------------------------------------------------------

class TestAgentGenome:
    def test_genome_is_frozen(self):
        with pytest.raises(AttributeError):
            DEFAULT_GENOME.genes = ()

    def test_has_22_genes(self):
        assert len(DEFAULT_GENOME.genes) == 22

    def test_default_values_present(self):
        defaults = DEFAULT_GENOME.get_defaults()
        assert "max_learning_iterations" in defaults
        assert defaults["max_learning_iterations"] == 100
        assert defaults["sensing_radius"] == 5.0
        assert defaults["planning_horizon"] == 5

    def test_gene_lookup(self):
        gene = DEFAULT_GENOME.get_gene("exploration_bonus")
        assert gene is not None
        assert gene.mixin == "gamification"
        assert gene.default == 0.1

    def test_gene_lookup_missing(self):
        assert DEFAULT_GENOME.get_gene("nonexistent") is None

    def test_gene_names(self):
        names = DEFAULT_GENOME.get_gene_names()
        assert len(names) == 22
        assert "replay_enabled" in names
        assert "trail_following" in names
        assert "api_call_enabled" in names
        assert "llm_prompt_quality" in names


# ---------------------------------------------------------------------------
# TestAgentEpigenome
# ---------------------------------------------------------------------------

class TestAgentEpigenome:
    def test_default_role_is_stem(self):
        epi = AgentEpigenome()
        assert epi.role == "STEM"
        assert epi.config == {}
        assert epi.lineage == []
        assert epi.locked is False

    def test_config_override(self):
        epi = AgentEpigenome(
            role="EXPLORER",
            config={"sensing_radius": 15.0},
        )
        assert epi.config["sensing_radius"] == 15.0

    def test_lineage_tracking(self):
        epi = AgentEpigenome()
        epi.lineage.append("STEM")
        epi.role = "EXPLORER"
        epi.lineage.append("EXPLORER")
        epi.role = "HEALER"
        assert epi.lineage == ["STEM", "EXPLORER"]


# ---------------------------------------------------------------------------
# TestRoleProfiles
# ---------------------------------------------------------------------------

class TestRoleProfiles:
    def test_all_twelve_roles_defined(self):
        expected = {
            "STEM", "EXPLORER", "LEARNER", "COMMUNICATOR", "HEALER",
            "COORDINATOR", "SPECIALIST", "API_CALLER", "LLM_SPECIALIST",
            "SEC_WATCHER", "CONTRACT_TRACKER", "MARKET_ANALYST",
        }
        assert set(ROLE_PROFILES.keys()) == expected

    def test_stem_uses_defaults(self):
        assert ROLE_PROFILES["STEM"] == {}

    def test_profiles_use_valid_gene_names(self):
        gene_names = DEFAULT_GENOME.get_gene_names()
        for role, config in ROLE_PROFILES.items():
            for knob in config:
                assert knob in gene_names, f"Role {role} uses unknown gene '{knob}'"


# ---------------------------------------------------------------------------
# TestRedifferentiate
# ---------------------------------------------------------------------------

class TestRedifferentiate:
    def test_applies_config_to_agent(self):
        agent = MockAgent()
        epi = redifferentiate(agent, "EXPLORER")
        assert epi.role == "EXPLORER"
        assert agent.agent_config["exploration_bonus"] == 0.4
        assert agent.agent_config["sensing_radius"] == 15.0

    def test_updates_agent_attributes(self):
        agent = MockAgent()
        redifferentiate(agent, "EXPLORER")
        assert agent.sensing_radius == 15.0
        assert agent.exploration_bonus == 0.4

    def test_records_lineage(self):
        agent = MockAgent()
        epi = redifferentiate(agent, "EXPLORER")
        assert epi.lineage == ["STEM"]
        # Without registry, second call creates fresh epigenome (default STEM)
        epi2 = redifferentiate(agent, "HEALER")
        assert epi2.lineage == ["STEM"]

    def test_records_lineage_with_registry(self):
        bus = EventBus()
        registry = StemCellRegistry(event_bus=bus)
        agent = MockAgent()
        registry.register(agent)
        redifferentiate(agent, "EXPLORER", registry=registry)
        redifferentiate(agent, "HEALER", registry=registry)
        lineage = registry.get_lineage("1")
        assert lineage == ["STEM", "EXPLORER"]

    def test_respects_lock(self):
        bus = EventBus()
        registry = StemCellRegistry(event_bus=bus)
        agent = MockAgent()
        registry.register(agent)
        registry._epigenomes["1"].locked = True
        epi = redifferentiate(agent, "EXPLORER", registry=registry)
        assert epi.role == "STEM"  # unchanged

    def test_handles_unknown_role(self):
        agent = MockAgent()
        epi = redifferentiate(agent, "UNKNOWN_ROLE")
        assert epi.role == "UNKNOWN_ROLE"
        # Should still apply genome defaults
        assert agent.agent_config["max_learning_iterations"] == 100

    def test_emits_event(self):
        bus = EventBus()
        registry = StemCellRegistry(event_bus=bus)
        agent = MockAgent()
        registry.register(agent)

        events = []
        bus.register_callback(
            CH_STEM_CELL_REDIFFERENTIATED,
            lambda ch, msg: events.append(json.loads(msg)),
        )
        redifferentiate(agent, "EXPLORER", registry=registry, step=42)
        assert len(events) == 1
        assert events[0]["new_role"] == "EXPLORER"
        assert events[0]["step"] == 42


# ---------------------------------------------------------------------------
# TestStemCellRegistry
# ---------------------------------------------------------------------------

class TestStemCellRegistry:
    def test_register_agent(self):
        registry = StemCellRegistry()
        agent = MockAgent()
        epi = registry.register(agent)
        assert epi.role == "STEM"
        assert registry.get_epigenome("1") is epi

    def test_get_epigenome_missing(self):
        registry = StemCellRegistry()
        assert registry.get_epigenome("999") is None

    def test_role_distribution(self):
        registry = StemCellRegistry()
        for i in range(5):
            registry.register(MockAgent(unique_id=i))
        dist = registry.get_role_distribution()
        assert dist == {"STEM": 5}

    def test_agents_by_role(self):
        registry = StemCellRegistry()
        for i in range(3):
            registry.register(MockAgent(unique_id=i))
        stems = registry.get_agents_by_role("STEM")
        assert len(stems) == 3

    def test_lineage(self):
        registry = StemCellRegistry()
        agent = MockAgent()
        registry.register(agent)
        assert registry.get_lineage("1") == []

    def test_redifferentiate_via_registry(self):
        bus = EventBus()
        registry = StemCellRegistry(event_bus=bus)
        agent = MockAgent()
        registry.register(agent)
        epi = registry.redifferentiate("1", "EXPLORER", step=10)
        assert epi.role == "EXPLORER"
        dist = registry.get_role_distribution()
        assert dist == {"EXPLORER": 1}

    def test_redifferentiate_unknown_agent(self):
        registry = StemCellRegistry()
        result = registry.redifferentiate("999", "EXPLORER")
        assert result is None

    def test_statistics(self):
        bus = EventBus()
        registry = StemCellRegistry(event_bus=bus)
        for i in range(3):
            registry.register(MockAgent(unique_id=i))
        registry.redifferentiate("0", "EXPLORER")
        stats = registry.get_statistics()
        assert stats["total_registered"] == 3
        assert stats["total_redifferentiations"] == 1
        assert stats["genome_size"] == 22
        assert "STEM" in stats["available_roles"]

    def test_emits_registration_event(self):
        bus = EventBus()
        registry = StemCellRegistry(event_bus=bus)
        events = []
        bus.register_callback(
            CH_STEM_CELL_REGISTERED,
            lambda ch, msg: events.append(json.loads(msg)),
        )
        registry.register(MockAgent())
        assert len(events) == 1
        assert events[0]["agent_id"] == "1"


# ---------------------------------------------------------------------------
# TestRedifferentiationMonitor
# ---------------------------------------------------------------------------

class TestRedifferentiationMonitor:
    """Tests for automatic redifferentiation triggers."""

    def _make_agent_with_history(self, uid: int, rewards: list[float]) -> MockAgent:
        """Create a MockAgent with a pre-filled reward_history."""
        agent = MockAgent(unique_id=uid)
        agent.reward_history = deque(rewards, maxlen=200)
        return agent

    def test_performance_trigger_fires_after_sustained_low_health(self):
        """Agent with sustained low rewards gets redifferentiated."""
        bus = EventBus()
        registry = StemCellRegistry(event_bus=bus)
        # Create an agent with 21 very negative rewards (health well below 0.3)
        agent = self._make_agent_with_history(1, [-0.8] * 21)
        registry.register(agent)
        registry.redifferentiate("1", "EXPLORER", step=0)

        monitor = RedifferentiationMonitor(stem_cell_registry=registry, event_bus=bus)

        # Run for CONSECUTIVE_REQUIRED checks (each at a step divisible by 21)
        for i in range(CONSECUTIVE_REQUIRED):
            monitor.step(current_step=(i + 1) * 21)

        epi = registry.get_epigenome("1")
        assert epi.role != "EXPLORER", "Agent should have been redifferentiated away from EXPLORER"

    def test_no_trigger_with_good_health(self):
        """Agent with good rewards is never redifferentiated."""
        registry = StemCellRegistry()
        agent = self._make_agent_with_history(1, [0.9] * 21)
        registry.register(agent)
        registry.redifferentiate("1", "EXPLORER", step=0)

        monitor = RedifferentiationMonitor(stem_cell_registry=registry)
        for i in range(CONSECUTIVE_REQUIRED + 5):
            monitor.step(current_step=(i + 1) * 21)

        epi = registry.get_epigenome("1")
        assert epi.role == "EXPLORER", "Healthy agent should keep its role"

    def test_locked_agent_not_redifferentiated(self):
        """Locked agents are skipped by the performance trigger."""
        registry = StemCellRegistry()
        agent = self._make_agent_with_history(1, [-0.8] * 21)
        registry.register(agent)
        registry._epigenomes["1"].locked = True

        monitor = RedifferentiationMonitor(stem_cell_registry=registry)
        for i in range(CONSECUTIVE_REQUIRED + 5):
            monitor.step(current_step=(i + 1) * 21)

        epi = registry.get_epigenome("1")
        assert epi.role == "STEM", "Locked agent should keep its role"

    def test_role_imbalance_trigger(self):
        """When a role is empty and colony is large enough, fill it."""
        bus = EventBus()
        registry = StemCellRegistry(event_bus=bus)
        # Create enough agents (>= role count), all as EXPLORER (one role oversaturated, others empty)
        num_agents = len(ROLE_PROFILES)
        for i in range(num_agents):
            agent = self._make_agent_with_history(i, [0.1] * 21)
            registry.register(agent)
            registry.redifferentiate(str(i), "EXPLORER", step=0)

        events = []
        bus.register_callback(
            CH_AUTO_REDIFFERENTIATED,
            lambda ch, msg: events.append(json.loads(msg)),
        )

        monitor = RedifferentiationMonitor(stem_cell_registry=registry, event_bus=bus)
        monitor.step(current_step=21)

        # At least one agent should have been moved to an empty role
        assert len(events) >= 1
        assert events[0]["trigger_type"] == "role_imbalance"
        assert events[0]["old_role"] == "EXPLORER"

    def test_emits_event_on_auto_redifferentiation(self):
        """EventBus event is emitted with correct payload."""
        bus = EventBus()
        registry = StemCellRegistry(event_bus=bus)
        agent = self._make_agent_with_history(1, [-0.8] * 21)
        registry.register(agent)

        events = []
        bus.register_callback(
            CH_AUTO_REDIFFERENTIATED,
            lambda ch, msg: events.append(json.loads(msg)),
        )

        monitor = RedifferentiationMonitor(stem_cell_registry=registry, event_bus=bus)
        for i in range(CONSECUTIVE_REQUIRED):
            monitor.step(current_step=(i + 1) * 21)

        assert len(events) >= 1
        evt = events[0]
        assert "agent_id" in evt
        assert "trigger_type" in evt
        assert evt["trigger_type"] == "performance"
        assert "health_at_trigger" in evt

    def test_skips_non_cadenced_steps(self):
        """Monitor does nothing on steps not divisible by 21."""
        registry = StemCellRegistry()
        agent = self._make_agent_with_history(1, [-0.8] * 21)
        registry.register(agent)

        monitor = RedifferentiationMonitor(stem_cell_registry=registry)
        # Call on non-cadenced steps — should be no-op
        non_21 = [s for s in range(1, 200) if s % 21 != 0]
        for s in non_21:
            monitor.step(current_step=s)
        assert monitor._low_health_counts.get("1", 0) == 0

    def test_statistics(self):
        """get_statistics returns expected shape."""
        registry = StemCellRegistry()
        monitor = RedifferentiationMonitor(stem_cell_registry=registry)
        stats = monitor.get_statistics()
        assert "tracked_agents" in stats
        assert "agents_near_trigger" in stats
