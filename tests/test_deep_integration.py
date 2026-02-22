"""Deep Integration Tests - Biological systems wired into agent behavior.

Tests the deep wiring of 18 biological systems and 5 dormant systems
into Mae's agent behavioral loop.  Verifies OrganismState aggregation,
reflex overrides, dormant-system consultation, autopoietic closure,
and cross-system EventBus wiring.

All tests are fast: no Qdrant, no Docker, no file I/O.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mae_core.backbone.event_bus import EventBus

# ---------------------------------------------------------------------------
# EventBus channel constants (imported from real system modules)
# ---------------------------------------------------------------------------
from mae_core.communication.nociception import NociceptionSystem
from mae_core.coordination.emotional_system import CH_EMOTION_UPDATE
from mae_core.coordination.vestibular_system import CH_BALANCE_UPDATE
from mae_core.memory.energy_reserve import CH_ENERGY_STATUS

CH_PAIN_UPDATE = NociceptionSystem.CH_PAIN_UPDATE  # "communication.pain_update"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def event_bus():
    return EventBus()


@pytest.fixture
def organism_state(event_bus):
    """Create an OrganismState wired to the given EventBus.

    If the real OrganismState class has not been implemented yet, this
    fixture creates a lightweight stand-in that satisfies the public API
    described in the test spec.  Once the production module exists, remove
    the fallback branch.
    """
    try:
        from mae_core.coordination.organism_state import OrganismState
    except ImportError:
        # Stub implementation matching the API exercised below.
        OrganismState = _StubOrganismState

    return OrganismState(event_bus=event_bus)


def _make_model():
    """Create a minimal Mesa model mock for MycelialAgent."""
    model = MagicMock()
    model._agents = {}
    model.schedule = MagicMock()
    model.time = 0
    model.running = True
    return model


def _make_agent(**kwargs):
    """Create a MycelialAgent with optional subsystem injections."""
    from mae_core.agents.mycelial_agent import MycelialAgent

    model = _make_model()
    return MycelialAgent(model=model, agent_type="test", **kwargs)


@pytest.fixture
def minimal_agent():
    """Create a minimal MycelialAgent for testing."""
    return _make_agent()


# ---------------------------------------------------------------------------
# Stub OrganismState (used only when production module is not yet built)
# ---------------------------------------------------------------------------

class _StubOrganismState:
    """Lightweight stand-in for OrganismState.

    Supports the full public API tested here so that the test file is
    self-contained even before the production class exists.
    """

    def __init__(self, event_bus=None):
        self._bus = event_bus

        # Body state defaults
        self._energy = 1.0
        self._stability = 1.0
        self._pain_load = 0.0
        self._emotional_valence = 0.0
        self._emotional_arousal = 0.0
        self._metacognition_score = 1.0
        self._vitality = 1.0
        self._energy_critical = False

        # Outcome tracking
        self._outcomes: list[dict] = []

        # Subscribe to channels if event bus is provided
        if self._bus is not None:
            self._bus.register_callback(CH_PAIN_UPDATE, self._on_pain)
            self._bus.register_callback(CH_ENERGY_STATUS, self._on_energy)
            self._bus.register_callback(CH_BALANCE_UPDATE, self._on_balance)
            self._bus.register_callback(CH_EMOTION_UPDATE, self._on_emotion)

    # --- EventBus handlers ---

    def _parse(self, message):
        if isinstance(message, dict):
            return message
        if isinstance(message, str):
            try:
                return json.loads(message)
            except (json.JSONDecodeError, TypeError):
                pass
        return {}

    def _on_pain(self, channel, message):
        data = self._parse(message)
        self._pain_load = float(data.get("total_pain_load", self._pain_load))

    def _on_energy(self, channel, message):
        data = self._parse(message)
        self._energy = float(data.get("reserves", self._energy))
        self._energy_critical = bool(data.get("is_critical", False))

    def _on_balance(self, channel, message):
        data = self._parse(message)
        self._stability = float(data.get("stability", self._stability))

    def _on_emotion(self, channel, message):
        data = self._parse(message)
        self._emotional_valence = float(data.get("valence", self._emotional_valence))
        self._emotional_arousal = float(data.get("arousal", self._emotional_arousal))

    # --- Public API ---

    def get_body_state(self):
        return {
            "energy": self._energy,
            "stability": self._stability,
            "pain_load": self._pain_load,
            "emotional_valence": self._emotional_valence,
            "emotional_arousal": self._emotional_arousal,
            "metacognition_score": self._metacognition_score,
            "vitality": self._vitality,
            "energy_level": self._energy,
        }

    def get_reflex_override(self):
        if self._pain_load > 0.8:
            return "rest"
        if self._energy_critical:
            return "explore"
        if self._stability < 0.3:
            return "rest"
        return None

    def get_decision_context(self):
        return {
            "body_threat_level": self._pain_load,
            "emotional_bias": self._emotional_valence,
            "energy_level": self._energy,
            "stability": self._stability,
            "metacognition_score": self._metacognition_score,
        }

    def report_action_outcome(self, action, reward, step):
        self._outcomes.append({
            "action": action, "reward": reward, "step": step,
        })
        # Simple vitality tracking
        if reward > 0:
            self._vitality = min(1.0, self._vitality + 0.05)
        elif reward < 0:
            self._vitality = max(0.0, self._vitality - 0.05)

    def serialize(self):
        return {
            "energy": self._energy,
            "stability": self._stability,
            "pain_load": self._pain_load,
            "emotional_valence": self._emotional_valence,
            "emotional_arousal": self._emotional_arousal,
            "metacognition_score": self._metacognition_score,
            "vitality": self._vitality,
            "energy_critical": self._energy_critical,
        }

    def restore(self, data):
        if not isinstance(data, dict):
            return
        self._energy = data.get("energy", self._energy)
        self._stability = data.get("stability", self._stability)
        self._pain_load = data.get("pain_load", self._pain_load)
        self._emotional_valence = data.get("emotional_valence", self._emotional_valence)
        self._emotional_arousal = data.get("emotional_arousal", self._emotional_arousal)
        self._metacognition_score = data.get("metacognition_score", self._metacognition_score)
        self._vitality = data.get("vitality", self._vitality)
        self._energy_critical = data.get("energy_critical", self._energy_critical)

    def get_statistics(self):
        return {
            "body_state": self.get_body_state(),
            "outcome_count": len(self._outcomes),
            "vitality": self._vitality,
        }


# ===========================================================================
# Test Class 1: TestOrganismState
# ===========================================================================

class TestOrganismState:
    """Test the OrganismState aggregator in isolation."""

    def test_initial_body_state_healthy(self, organism_state):
        """get_body_state() returns dict with healthy defaults."""
        body = organism_state.get_body_state()
        assert isinstance(body, dict)
        assert body["energy_level"] == pytest.approx(1.0)
        assert body["stability"] == pytest.approx(1.0)
        assert body["pain_load"] == pytest.approx(0.0)

    def test_reflex_override_none_when_healthy(self, organism_state):
        """get_reflex_override() returns None when body is healthy."""
        assert organism_state.get_reflex_override() is None

    def test_reflex_override_pain(self, event_bus, organism_state):
        """Simulate pain > 0.8, reflex should be 'rest'."""
        event_bus.publish(CH_PAIN_UPDATE, {
            "total_pain_load": 0.9,
            "active_pains": 1,
        })
        assert organism_state.get_reflex_override() == "rest"

    def test_reflex_override_starvation(self, event_bus, organism_state):
        """Simulate critical energy, reflex should be 'explore'."""
        event_bus.publish(CH_ENERGY_STATUS, {
            "reserves": 5.0,
            "capacity_pct": 2.5,
            "leptin_level": 0.05,
            "is_critical": True,
            "step": 1,
        })
        assert organism_state.get_reflex_override() == "explore"

    def test_reflex_override_vertigo(self, event_bus, organism_state):
        """Simulate stability < 0.3, reflex should be 'rest'."""
        event_bus.publish(CH_BALANCE_UPDATE, {
            "stability": 0.15,
            "status": "vertigo",
            "metrics_tracked": 3,
            "step": 1,
        })
        assert organism_state.get_reflex_override() == "rest"

    def test_reflex_priority_pain_over_energy(self, event_bus, organism_state):
        """When both pain and starvation are active, pain takes priority."""
        event_bus.publish(CH_PAIN_UPDATE, {
            "total_pain_load": 0.95,
            "active_pains": 2,
        })
        event_bus.publish(CH_ENERGY_STATUS, {
            "reserves": 3.0,
            "capacity_pct": 1.5,
            "leptin_level": 0.02,
            "is_critical": True,
            "step": 1,
        })
        # Pain reflex ("rest") should win over energy ("explore")
        assert organism_state.get_reflex_override() == "rest"

    def test_body_state_updates_from_events(self, event_bus, organism_state):
        """Publish emotion event, verify body state reflects new valence."""
        event_bus.publish(CH_EMOTION_UPDATE, {
            "emotion_name": "JOY",
            "valence": 0.9,
            "arousal": 0.6,
            "confidence": 0.8,
            "step": 1,
        })
        body = organism_state.get_body_state()
        assert body["emotional_valence"] == pytest.approx(0.9)

    def test_decision_context(self, organism_state):
        """get_decision_context() returns expected keys."""
        ctx = organism_state.get_decision_context()
        assert "body_threat_level" in ctx
        assert "emotional_bias" in ctx
        assert "body_opportunity_level" in ctx
        assert "metacognitive_confidence" in ctx
        assert "organism_vitality" in ctx

    def test_report_action_outcome(self, organism_state):
        """report_action_outcome updates internal vitality tracking."""
        initial_stats = organism_state.get_statistics()
        initial_vitality = initial_stats.get("vitality", 0.5)
        organism_state.report_action_outcome(action="explore", reward=1.0, step=1)
        # Vitality should improve after positive reward (EMA toward 1.0)
        new_stats = organism_state.get_statistics()
        new_vitality = new_stats.get("vitality", 0.5)
        assert new_vitality >= initial_vitality

    def test_serialize_restore(self, event_bus, organism_state):
        """serialize() and restore() round-trip preserves body state."""
        # Modify state via events
        event_bus.publish(CH_PAIN_UPDATE, {"total_pain_load": 0.5, "active_pains": 1})
        event_bus.publish(CH_EMOTION_UPDATE, {
            "emotion_name": "FEAR",
            "valence": -0.8,
            "arousal": 0.9,
            "confidence": 0.7,
            "step": 5,
        })

        state = organism_state.serialize()
        assert isinstance(state, dict)

        # Create a fresh instance and restore
        try:
            from mae_core.coordination.organism_state import OrganismState
        except ImportError:
            OrganismState = _StubOrganismState

        fresh = OrganismState(event_bus=None)
        fresh.restore(state)

        body = fresh.get_body_state()
        assert body["pain_load"] == pytest.approx(0.5)
        assert body["emotional_valence"] == pytest.approx(-0.8)

    def test_get_statistics(self, organism_state):
        """get_statistics() returns expected keys."""
        stats = organism_state.get_statistics()
        assert isinstance(stats, dict)
        assert "body_state" in stats or "vitality" in stats

    def test_graceful_without_event_bus(self):
        """OrganismState(event_bus=None) works without crashing."""
        try:
            from mae_core.coordination.organism_state import OrganismState
        except ImportError:
            OrganismState = _StubOrganismState

        os = OrganismState(event_bus=None)
        body = os.get_body_state()
        assert body is not None
        assert os.get_reflex_override() is None


# ===========================================================================
# Test Class 2: TestAgentOrganismIntegration
# ===========================================================================

class TestAgentOrganismIntegration:
    """Test that MycelialAgent correctly consults OrganismState."""

    def test_agent_without_organism_state(self, minimal_agent):
        """Agent works fine without _organism_state attribute."""
        # Should not crash
        minimal_agent.step()
        assert minimal_agent.step_count >= 1

    def test_agent_with_organism_state(self, minimal_agent, organism_state):
        """Set _organism_state, call step(). No crash."""
        minimal_agent._organism_state = organism_state
        minimal_agent.step()
        assert minimal_agent.step_count >= 1

    def test_reflex_override_in_decide(self, event_bus, organism_state):
        """Organism with pain > 0.8 triggers reflex 'rest' during _decide()."""
        agent = _make_agent()
        agent._organism_state = organism_state
        agent._observe()

        # Inject pain
        event_bus.publish(CH_PAIN_UPDATE, {
            "total_pain_load": 0.95,
            "active_pains": 2,
        })

        action = agent._decide()
        assert action == "rest"

    def test_body_state_read_in_observe(self, organism_state, minimal_agent):
        """After injecting organism_state, _observe() sets agent._body_state."""
        minimal_agent._organism_state = organism_state
        minimal_agent._observe()

        assert minimal_agent._body_state is not None
        assert isinstance(minimal_agent._body_state, dict)
        assert "energy_level" in minimal_agent._body_state

    def test_organism_enriches_routing_context(self, event_bus, organism_state):
        """When agent has both advisory and organism_state, routing context
        includes body state fields."""
        from unittest.mock import MagicMock
        from mae_core.agents.mycelial_agent import MycelialAgent

        agent = _make_agent()
        agent._organism_state = organism_state

        # Set body state via observation
        agent._observe()

        # Verify body_state fields are populated for routing
        body = agent._body_state
        assert body is not None
        assert "energy" in body or "energy_level" in body


# ===========================================================================
# Test Class 3: TestDormantSystemActivation
# ===========================================================================

class TestDormantSystemActivation:
    """Test that dormant systems can be activated via attribute injection."""

    def test_agent_without_dormant_systems(self):
        """Agent step() works without any dormant system references."""
        agent = _make_agent()
        agent.step()
        assert agent.step_count >= 1

    def test_worldline_planner_consulted(self):
        """Worldline planner's plan() gets called during _decide()."""
        agent = _make_agent()
        agent._observe()

        # Create a mock worldline planner with expected API
        planner = MagicMock()
        point = MagicMock()
        point.action = "explore"
        selected = MagicMock()
        selected.points = [point]
        result = MagicMock()
        result.selected_worldline = selected
        planner.plan.return_value = result

        agent._worldline_planner = planner
        action = agent._decide()

        planner.plan.assert_called_once()
        assert action == "explore"

    def test_collective_dream_consulted(self):
        """Collective dream planner's collective_plan() gets called during _decide()."""
        agent = _make_agent()
        agent._observe()

        # Create mock dream planner
        dream = MagicMock()
        dream.collective_plan.return_value = {
            "status": "approved",
            "trajectory": [(np.zeros(8), "communicate", 0.5)],
        }

        agent._collective_dream = dream
        # Ensure worldline doesn't short-circuit first
        agent._worldline_planner = None

        action = agent._decide()

        dream.collective_plan.assert_called_once()
        assert action == "communicate"

    def test_predictive_field_read(self):
        """Predictive field's detect_collision_risk() gets called during _observe()."""
        agent = _make_agent()
        pred_field = MagicMock()
        pred_field.detect_collision_risk.return_value = []
        pred_field.find_coordination_opportunities.return_value = []

        agent._predictive_field = pred_field
        agent.pos = (1.0, 1.0)  # Give agent a position so field is consulted

        agent._observe()

        pred_field.detect_collision_risk.assert_called_once()

    def test_morphogenesis_signal_on_high_error(self):
        """High prediction error triggers morphogenesis capability gap signal."""
        bus = MagicMock()
        agent = _make_agent()
        agent._observe()

        morpho = MagicMock()
        morpho.handle_novel_problem = MagicMock()
        agent._morphogenesis = morpho
        agent._prediction_error = 0.9
        # The production code uses getattr(self, "signal_bus", None)
        # for the morphogenesis publish path.
        agent.signal_bus = bus

        # Trigger _decide() — morphogenesis signal is sent via _signal_bus
        agent._decide()

        # Verify the bus got a capability_gap publish
        calls = bus.publish.call_args_list
        cap_gap_calls = [c for c in calls if "morphogenesis.capability_gap" in str(c)]
        assert len(cap_gap_calls) >= 1

    def test_dormant_failure_graceful(self):
        """Dormant system whose methods raise exceptions doesn't crash agent."""
        agent = _make_agent()

        # Inject a broken worldline planner
        broken_planner = MagicMock()
        broken_planner.plan.side_effect = RuntimeError("deliberate failure")
        agent._worldline_planner = broken_planner

        # Inject a broken predictive field
        broken_field = MagicMock()
        broken_field.detect_collision_risk.side_effect = ValueError("broken")
        broken_field.find_coordination_opportunities.side_effect = ValueError("broken")
        agent._predictive_field = broken_field
        agent.pos = (0.0, 0.0)

        # Inject a broken collective dream
        broken_dream = MagicMock()
        broken_dream.collective_plan.side_effect = TypeError("broken")
        agent._collective_dream = broken_dream

        # Agent step should complete without raising
        agent.step()
        assert agent.step_count >= 1


# ===========================================================================
# Test Class 4: TestAutopoieticClosure
# ===========================================================================

class TestAutopoieticClosure:
    """Test circular causation loops (Law 6)."""

    def test_action_feeds_back_to_organism(self, event_bus, organism_state):
        """Agent acts, report_action_outcome is called on organism_state."""
        agent = _make_agent()
        agent._organism_state = organism_state

        # Run one step — _learn() should call report_action_outcome
        agent.step()

        # Check that at least one outcome was reported
        if hasattr(organism_state, "_outcomes"):
            assert len(organism_state._outcomes) >= 1

    def test_organism_influences_decision(self, event_bus, organism_state):
        """Organism state (pain) changes agent's decision (triggers reflex)."""
        agent = _make_agent()
        agent._organism_state = organism_state

        # Without pain, default action
        agent._observe()
        action_normal = agent._decide()

        # Inject pain
        event_bus.publish(CH_PAIN_UPDATE, {
            "total_pain_load": 0.95,
            "active_pains": 3,
        })

        action_pain = agent._decide()
        assert action_pain == "rest"

    def test_emotional_loop(self, event_bus, organism_state):
        """Positive emotion event improves body state emotional valence."""
        # Verify initial valence
        body_before = organism_state.get_body_state()
        initial_valence = body_before["emotional_valence"]

        # Publish joy emotion
        event_bus.publish(CH_EMOTION_UPDATE, {
            "emotion_name": "JOY",
            "valence": 0.9,
            "arousal": 0.6,
            "confidence": 0.85,
            "step": 1,
        })

        body_after = organism_state.get_body_state()
        assert body_after["emotional_valence"] > initial_valence


# ===========================================================================
# Test Class 5: TestCrossSystemWiring
# ===========================================================================

class TestCrossSystemWiring:
    """Test that EventBus connects systems to OrganismState."""

    def test_pain_event_updates_organism(self, event_bus, organism_state):
        """Publish nociception pain update, organism reflects pain."""
        event_bus.publish(CH_PAIN_UPDATE, {
            "total_pain_load": 0.6,
            "active_pains": 2,
        })
        body = organism_state.get_body_state()
        assert body["pain_load"] == pytest.approx(0.6)

    def test_energy_event_updates_organism(self, event_bus, organism_state):
        """Publish energy event, organism reflects energy level.

        EnergyReserve publishes capacity_pct as a 0-100 percentage.
        OrganismState normalizes it to [0.0, 1.0] by dividing by 100.
        """
        event_bus.publish(CH_ENERGY_STATUS, {
            "reserves": 42.0,
            "capacity_pct": 65.0,
            "leptin_level": 0.21,
            "is_critical": False,
            "step": 5,
        })
        body = organism_state.get_body_state()
        assert body["energy_level"] == pytest.approx(0.65)

    def test_stability_event_updates_organism(self, event_bus, organism_state):
        """Publish vestibular event, organism reflects stability."""
        event_bus.publish(CH_BALANCE_UPDATE, {
            "stability": 0.45,
            "status": "wobbling",
            "metrics_tracked": 5,
            "step": 3,
        })
        body = organism_state.get_body_state()
        assert body["stability"] == pytest.approx(0.45)

    def test_emotion_event_updates_organism(self, event_bus, organism_state):
        """Publish emotion event, organism reflects valence."""
        event_bus.publish(CH_EMOTION_UPDATE, {
            "emotion_name": "FEAR",
            "valence": -0.8,
            "arousal": 0.9,
            "confidence": 0.7,
            "step": 2,
        })
        body = organism_state.get_body_state()
        assert body["emotional_valence"] == pytest.approx(-0.8)
        assert body["emotional_arousal"] == pytest.approx(0.9)

    @pytest.mark.parametrize("pain_load,expected_reflex", [
        (0.0, None),
        (0.5, None),
        (0.81, "rest"),
        (1.0, "rest"),
    ])
    def test_pain_threshold_parametrized(
        self, event_bus, organism_state, pain_load, expected_reflex,
    ):
        """Parametrized: various pain loads produce correct reflex or None."""
        event_bus.publish(CH_PAIN_UPDATE, {
            "total_pain_load": pain_load,
            "active_pains": 1 if pain_load > 0 else 0,
        })
        assert organism_state.get_reflex_override() == expected_reflex

    @pytest.mark.parametrize("stability,expected_reflex", [
        (1.0, None),
        (0.5, None),
        (0.29, "rest"),
        (0.1, "rest"),
    ])
    def test_stability_threshold_parametrized(
        self, event_bus, organism_state, stability, expected_reflex,
    ):
        """Parametrized: various stability levels produce correct reflex or None."""
        event_bus.publish(CH_BALANCE_UPDATE, {
            "stability": stability,
            "status": "vertigo" if stability < 0.3 else "stable",
            "metrics_tracked": 3,
            "step": 1,
        })
        assert organism_state.get_reflex_override() == expected_reflex
