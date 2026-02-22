"""Tests for quorum_signal and quorum_space.

Tests signal definitions, decay functions, and chemical concentration environment.
"""

import math
import time
import pytest

from mae_core.communication.quorum_signal import (
    DecayType,
    QuorumSignalType,
    SenseEvent,
    SignalTypes,
    get_signal_type,
    register_custom_signal_type,
    list_signal_types,
)
from mae_core.communication.quorum_space import (
    ConcentrationState,
    QuorumSpace,
)


# ============================================================================
# Tests for quorum_signal
# ============================================================================

class TestDecayTypeEnum:
    """Test DecayType enum values."""

    def test_decay_types_defined(self):
        """All decay types are defined."""
        assert DecayType.EXPONENTIAL.value == "exponential"
        assert DecayType.LINEAR.value == "linear"
        assert DecayType.STEP.value == "step"
        assert DecayType.NONE.value == "none"


class TestQuorumSignalTypeBasics:
    """Test QuorumSignalType definition and properties."""

    def test_basic_creation(self):
        """Create a basic signal type."""
        sig = QuorumSignalType(
            name="TEST",
            description="Test signal",
            decay_type=DecayType.EXPONENTIAL,
            decay_rate=0.1,
            base_strength=1.0,
            max_concentration=1.0,
        )
        assert sig.name == "TEST"
        assert sig.decay_type == DecayType.EXPONENTIAL
        assert sig.decay_rate == 0.1

    def test_minimal_creation(self):
        """Create signal with minimal arguments."""
        sig = QuorumSignalType(name="MINIMAL")
        assert sig.name == "MINIMAL"
        assert sig.decay_type == DecayType.EXPONENTIAL
        assert sig.base_strength == 1.0

    def test_with_metadata_schema(self):
        """Signal with metadata schema."""
        sig = QuorumSignalType(
            name="COMPLEX",
            metadata_schema={"location": "str", "severity": "float"},
        )
        assert "location" in sig.metadata_schema
        assert sig.metadata_schema["severity"] == "float"


class TestExponentialDecay:
    """Test exponential decay computation."""

    def test_no_decay_at_zero_time(self):
        """Exponential decay at t=0 is no decay."""
        sig = QuorumSignalType(
            name="EXP",
            decay_type=DecayType.EXPONENTIAL,
            decay_rate=0.1,
        )
        result = sig.compute_decay(1.0, 0.0)
        assert result == 1.0

    def test_exponential_decay_halves(self):
        """At half-life, exponential decay reaches 0.5."""
        sig = QuorumSignalType(
            name="EXP",
            decay_type=DecayType.EXPONENTIAL,
            decay_rate=0.693,  # ln(2)
        )
        result = sig.compute_decay(1.0, 1.0)  # 1 second = 1 half-life
        assert abs(result - 0.5) < 0.01

    def test_exponential_decay_formula(self):
        """Exponential decay follows e^(-lambda*t)."""
        sig = QuorumSignalType(
            name="EXP",
            decay_type=DecayType.EXPONENTIAL,
            decay_rate=0.5,
        )
        elapsed = 2.0
        result = sig.compute_decay(1.0, elapsed)
        expected = math.exp(-0.5 * 2.0)
        assert abs(result - expected) < 1e-10

    def test_exponential_never_negative(self):
        """Exponential decay never goes negative."""
        sig = QuorumSignalType(
            name="EXP",
            decay_type=DecayType.EXPONENTIAL,
            decay_rate=0.1,
        )
        result = sig.compute_decay(1.0, 1000.0)  # Very long time
        assert result > 0.0


class TestLinearDecay:
    """Test linear decay computation."""

    def test_linear_decay_formula(self):
        """Linear decay is initial - rate * time."""
        sig = QuorumSignalType(
            name="LIN",
            decay_type=DecayType.LINEAR,
            decay_rate=0.2,
        )
        result = sig.compute_decay(1.0, 3.0)
        expected = 1.0 - 0.2 * 3.0
        assert abs(result - expected) < 1e-10

    def test_linear_decay_stops_at_zero(self):
        """Linear decay bottoms out at zero."""
        sig = QuorumSignalType(
            name="LIN",
            decay_type=DecayType.LINEAR,
            decay_rate=0.2,
        )
        result = sig.compute_decay(1.0, 1000.0)  # Far beyond decay point
        assert result == 0.0

    def test_linear_partial_decay(self):
        """Linear decay is partial before reaching zero."""
        sig = QuorumSignalType(
            name="LIN",
            decay_type=DecayType.LINEAR,
            decay_rate=0.1,
        )
        result = sig.compute_decay(1.0, 2.0)
        assert abs(result - 0.8) < 1e-10


class TestStepDecay:
    """Test step decay computation."""

    def test_step_decay_before_half_life(self):
        """Step decay is unchanged before half-life."""
        sig = QuorumSignalType(
            name="STEP",
            decay_type=DecayType.STEP,
            decay_rate=0.5,  # half-life = ln(2)/0.5 ~= 1.386
        )
        result = sig.compute_decay(1.0, 0.5)
        assert result == 1.0

    def test_step_decay_after_half_life(self):
        """Step decay becomes zero after half-life."""
        sig = QuorumSignalType(
            name="STEP",
            decay_type=DecayType.STEP,
            decay_rate=0.693,  # half-life ~= 1.0
        )
        result = sig.compute_decay(1.0, 2.0)  # After half-life
        assert result == 0.0

    def test_step_decay_at_boundary(self):
        """Step decay at exactly half-life."""
        sig = QuorumSignalType(
            name="STEP",
            decay_type=DecayType.STEP,
            decay_rate=1.0,  # half-life = ln(2) ~= 0.693
        )
        half_life = math.log(2) / 1.0
        result = sig.compute_decay(1.0, half_life)
        assert result == 1.0  # At boundary, still intact


class TestNoDecay:
    """Test no-decay scenario."""

    def test_no_decay_type(self):
        """NONE decay type preserves value forever."""
        sig = QuorumSignalType(
            name="NONE",
            decay_type=DecayType.NONE,
            decay_rate=999.0,  # Rate should be ignored
        )
        result = sig.compute_decay(1.0, 1000.0)
        assert result == 1.0


class TestHalfLifeCalculation:
    """Test half-life derivation."""

    def test_exponential_half_life(self):
        """Exponential half-life calculated correctly."""
        decay_rate = 0.5
        sig = QuorumSignalType(
            name="EXP",
            decay_type=DecayType.EXPONENTIAL,
            decay_rate=decay_rate,
        )
        half_life = sig.get_half_life()
        expected = math.log(2) / decay_rate
        assert abs(half_life - expected) < 1e-10

    def test_linear_no_half_life(self):
        """Linear decay has no half-life value."""
        sig = QuorumSignalType(
            name="LIN",
            decay_type=DecayType.LINEAR,
            decay_rate=0.1,
        )
        assert sig.get_half_life() is None

    def test_step_no_half_life(self):
        """Step decay has no half-life value (different model)."""
        sig = QuorumSignalType(
            name="STEP",
            decay_type=DecayType.STEP,
            decay_rate=0.1,
        )
        assert sig.get_half_life() is None

    def test_no_decay_no_half_life(self):
        """No decay type has no half-life."""
        sig = QuorumSignalType(
            name="NONE",
            decay_type=DecayType.NONE,
        )
        assert sig.get_half_life() is None

    def test_zero_rate_half_life_infinite(self):
        """Zero decay rate returns None (no half-life defined)."""
        sig = QuorumSignalType(
            name="EXP",
            decay_type=DecayType.EXPONENTIAL,
            decay_rate=0.0,
        )
        half_life = sig.get_half_life()
        # With zero rate, no valid half-life (division by zero)
        assert half_life is None


class TestSenseEvent:
    """Test SenseEvent dataclass."""

    def test_basic_event(self):
        """Create a basic sense event."""
        event = SenseEvent(
            event_type="sense",
            signal_type="DANGER",
            agent_id="a1",
            concentration=0.5,
        )
        assert event.event_type == "sense"
        assert event.signal_type == "DANGER"
        assert event.concentration == 0.5

    def test_event_timestamp(self):
        """Events have timestamps."""
        event = SenseEvent()
        assert event.timestamp > 0

    def test_event_to_dict(self):
        """Event can be serialized to dict."""
        event = SenseEvent(
            event_type="deposit",
            signal_type="OPPORTUNITY",
            agent_id="a2",
            concentration=0.8,
        )
        d = event.to_dict()
        assert d["event_type"] == "deposit"
        assert d["signal_type"] == "OPPORTUNITY"
        assert d["agent_id"] == "a2"


class TestSignalTypeRegistry:
    """Test built-in signal types."""

    def test_opportunity_signal(self):
        """OPPORTUNITY signal is defined."""
        sig = SignalTypes.OPPORTUNITY
        assert sig.name == "OPPORTUNITY"
        assert "location" in sig.metadata_schema

    def test_danger_signal(self):
        """DANGER signal is defined."""
        sig = SignalTypes.DANGER
        assert sig.name == "DANGER"
        assert "risk_level" in sig.metadata_schema

    def test_consensus_signal(self):
        """CONSENSUS signal is defined."""
        sig = SignalTypes.CONSENSUS
        assert sig.name == "CONSENSUS"

    def test_all_signals_are_exponential(self):
        """Built-in signals use exponential decay."""
        for attr in dir(SignalTypes):
            if isinstance(getattr(SignalTypes, attr), QuorumSignalType):
                sig = getattr(SignalTypes, attr)
                assert sig.decay_type == DecayType.EXPONENTIAL


class TestSignalTypeGlobalRegistry:
    """Test global signal type registry."""

    def test_get_signal_type_found(self):
        """Retrieve registered signal type by name."""
        sig = get_signal_type("DANGER")
        assert sig is not None
        assert sig.name == "DANGER"

    def test_get_signal_type_not_found(self):
        """Non-existent signal type returns None."""
        sig = get_signal_type("NONEXISTENT")
        assert sig is None

    def test_list_signal_types(self):
        """List all registered signal types."""
        types = list_signal_types()
        assert "DANGER" in types
        assert "OPPORTUNITY" in types
        assert "CONSENSUS" in types
        assert len(types) >= 8  # At least the built-in ones

    def test_register_custom_signal(self):
        """Register a custom signal type."""
        custom = QuorumSignalType(
            name="CUSTOM_TEST_SIG",
            description="For testing only",
        )
        register_custom_signal_type(custom)

        retrieved = get_signal_type("CUSTOM_TEST_SIG")
        assert retrieved is not None
        assert retrieved.name == "CUSTOM_TEST_SIG"

    def test_custom_signal_in_list(self):
        """Custom signal appears in list after registration."""
        custom = QuorumSignalType(name="ANOTHER_CUSTOM_TEST")
        register_custom_signal_type(custom)

        types = list_signal_types()
        assert "ANOTHER_CUSTOM_TEST" in types


# ============================================================================
# Tests for quorum_space
# ============================================================================

class TestConcentrationState:
    """Test ConcentrationState dataclass."""

    def test_basic_state(self):
        """Create a concentration state."""
        state = ConcentrationState(
            concentration=0.5,
            contributors={"a1", "a2"},
            deposit_count=2,
        )
        assert state.concentration == 0.5
        assert len(state.contributors) == 2

    def test_default_state(self):
        """Default state is zero."""
        state = ConcentrationState()
        assert state.concentration == 0.0
        assert len(state.contributors) == 0


class TestQuorumSpaceBasics:
    """Test basic QuorumSpace operations."""

    def test_create_quorum_space(self):
        """Create a quorum space."""
        space = QuorumSpace()
        assert space is not None

    def test_register_signal_type(self):
        """Register a signal type in the space."""
        space = QuorumSpace()
        sig = QuorumSignalType(name="CUSTOM", decay_type=DecayType.LINEAR)
        space.register_signal_type(sig)

        # Should be available for deposits
        result = space.deposit_signal("CUSTOM", "a1")
        assert result is True

    def test_auto_register_unknown_signal(self):
        """Unknown signal types are auto-registered."""
        space = QuorumSpace()
        result = space.deposit_signal("UNKNOWN_TYPE", "a1")
        assert result is True

        # Should now be queryable
        conc = space.get_concentration("UNKNOWN_TYPE")
        assert conc > 0


class TestDeposition:
    """Test signal deposition."""

    def test_deposit_increases_concentration(self):
        """Depositing a signal increases concentration."""
        space = QuorumSpace()
        space.deposit_signal("DANGER", "a1")
        conc = space.get_concentration("DANGER")
        assert conc > 0.0

    def test_deposit_with_strength(self):
        """Deposit with custom strength."""
        space = QuorumSpace()
        space.deposit_signal("DANGER", "a1", strength=0.5)
        space.deposit_signal("DANGER", "a2", strength=0.3)

        conc = space.get_concentration("DANGER")
        # May decay slightly, so check lower bound
        assert conc > 0.75

    def test_deposit_capped_at_max(self):
        """Concentration capped at max_concentration."""
        space = QuorumSpace()
        sig_type = QuorumSignalType(
            name="CAPPED",
            max_concentration=1.0,
            decay_type=DecayType.NONE,
        )
        space.register_signal_type(sig_type)

        # Deposit many times
        for _ in range(100):
            space.deposit_signal("CAPPED", "a1", strength=10.0)

        conc = space.get_concentration("CAPPED")
        assert conc <= 1.0

    def test_multiple_contributors_tracked(self):
        """Contributors are tracked separately."""
        space = QuorumSpace()
        space.deposit_signal("TEST", "a1")
        space.deposit_signal("TEST", "a2")
        space.deposit_signal("TEST", "a1")  # a1 again

        contributors = space.get_contributors("TEST")
        assert len(contributors) == 2
        assert "a1" in contributors
        assert "a2" in contributors

    def test_contributor_count(self):
        """Get count of contributors."""
        space = QuorumSpace()
        for i in range(5):
            space.deposit_signal("SIGNAL", f"a{i}")

        count = space.get_contributor_count("SIGNAL")
        assert count == 5


class TestConcentrationRetrieval:
    """Test getting concentration values."""

    def test_get_concentration_zero_if_not_exists(self):
        """Non-existent signal has zero concentration."""
        space = QuorumSpace()
        conc = space.get_concentration("NONEXISTENT")
        assert conc == 0.0

    def test_get_all_concentrations(self):
        """Retrieve all signal concentrations."""
        space = QuorumSpace()
        space.deposit_signal("SIGNAL_A", "a1")
        space.deposit_signal("SIGNAL_B", "a2")

        all_concs = space.get_all_concentrations()
        assert "SIGNAL_A" in all_concs
        assert "SIGNAL_B" in all_concs

    def test_get_state_full_info(self):
        """Get full state for a signal."""
        space = QuorumSpace()
        space.deposit_signal("TEST", "a1", strength=0.5)
        space.deposit_signal("TEST", "a2", strength=0.3)

        state = space.get_state("TEST")
        assert state is not None
        assert state["concentration"] > 0.75  # May decay slightly
        assert state["contributors"] == 2
        assert state["deposit_count"] == 2

    def test_get_state_nonexistent(self):
        """Non-existent signal state is None."""
        space = QuorumSpace()
        state = space.get_state("NONEXISTENT")
        assert state is None


class TestDecayMechanism:
    """Test exponential decay in the space."""

    def test_exponential_decay_over_time(self):
        """Exponential signals decay over time."""
        space = QuorumSpace()
        sig = QuorumSignalType(
            name="DECAYING",
            decay_type=DecayType.EXPONENTIAL,
            decay_rate=1.0,  # Fast decay
        )
        space.register_signal_type(sig)

        space.deposit_signal("DECAYING", "a1", strength=1.0)
        conc_before = space.get_concentration("DECAYING")

        time.sleep(0.5)

        conc_after = space.get_concentration("DECAYING")
        assert conc_after < conc_before

    def test_linear_decay_over_time(self):
        """Linear signals decay at constant rate."""
        space = QuorumSpace()
        sig = QuorumSignalType(
            name="LINEAR_SIG",
            decay_type=DecayType.LINEAR,
            decay_rate=1.0,  # Lose 1.0 per second
        )
        space.register_signal_type(sig)

        space.deposit_signal("LINEAR_SIG", "a1", strength=1.0)
        conc_before = space.get_concentration("LINEAR_SIG")

        time.sleep(0.1)

        conc_after = space.get_concentration("LINEAR_SIG")
        assert conc_after < conc_before

    def test_no_decay_persists(self):
        """NONE decay type preserves concentration."""
        space = QuorumSpace()
        sig = QuorumSignalType(
            name="PERSISTENT",
            decay_type=DecayType.NONE,
        )
        space.register_signal_type(sig)

        space.deposit_signal("PERSISTENT", "a1")
        conc_before = space.get_concentration("PERSISTENT")

        time.sleep(0.5)

        conc_after = space.get_concentration("PERSISTENT")
        assert abs(conc_after - conc_before) < 1e-6

    def test_zero_contributors_when_decayed(self):
        """Contributors cleared when concentration decays to near-zero."""
        space = QuorumSpace()
        sig = QuorumSignalType(
            name="FAST_DECAY",
            decay_type=DecayType.EXPONENTIAL,
            decay_rate=10.0,  # Very fast
        )
        space.register_signal_type(sig)

        space.deposit_signal("FAST_DECAY", "a1")
        contributors_before = space.get_contributor_count("FAST_DECAY")
        assert contributors_before == 1

        time.sleep(0.2)

        conc = space.get_concentration("FAST_DECAY")
        contributors_after = space.get_contributor_count("FAST_DECAY")

        # If concentration is near-zero, contributors should be cleared
        if conc < 0.001:
            assert contributors_after == 0


class TestReset:
    """Test reset operations."""

    def test_reset_single_signal(self):
        """Reset a single signal."""
        space = QuorumSpace()
        space.deposit_signal("TEST", "a1")
        assert space.get_concentration("TEST") > 0

        space.reset_signal("TEST")
        assert space.get_concentration("TEST") == 0.0

    def test_reset_clears_contributors(self):
        """Reset clears contributors by default."""
        space = QuorumSpace()
        space.deposit_signal("TEST", "a1")
        space.deposit_signal("TEST", "a2")

        space.reset_signal("TEST")
        count = space.get_contributor_count("TEST")
        assert count == 0

    def test_reset_preserves_contributors(self):
        """Reset with clear_contributors=False preserves them."""
        space = QuorumSpace()
        space.deposit_signal("TEST", "a1")
        space.deposit_signal("TEST", "a2")

        space.reset_signal("TEST", clear_contributors=False)
        count = space.get_contributor_count("TEST")
        assert count == 2  # Still there

    def test_reset_all(self):
        """Reset all signals."""
        space = QuorumSpace()
        space.deposit_signal("A", "a1")
        space.deposit_signal("B", "a2")
        space.deposit_signal("C", "a3")

        space.reset_all()
        assert space.get_concentration("A") == 0.0
        assert space.get_concentration("B") == 0.0
        assert space.get_concentration("C") == 0.0


class TestEventHistory:
    """Test event history tracking."""

    def test_deposit_recorded(self):
        """Deposits are recorded in history."""
        space = QuorumSpace()
        space.deposit_signal("TEST", "a1", metadata={"key": "value"})

        history = space.get_event_history()
        assert len(history) >= 1
        latest = history[-1]
        assert latest.event_type == "deposit"
        assert latest.signal_type == "TEST"
        assert latest.agent_id == "a1"

    def test_filter_history_by_signal(self):
        """Filter history by signal type."""
        space = QuorumSpace()
        space.deposit_signal("SIGNAL_A", "a1")
        space.deposit_signal("SIGNAL_B", "a2")

        history_a = space.get_event_history(signal_type="SIGNAL_A")
        for event in history_a:
            assert event.signal_type == "SIGNAL_A"

    def test_filter_history_by_agent(self):
        """Filter history by agent ID."""
        space = QuorumSpace()
        space.deposit_signal("TEST", "a1")
        space.deposit_signal("TEST", "a2")

        history_a1 = space.get_event_history(agent_id="a1")
        for event in history_a1:
            assert event.agent_id == "a1"

    def test_history_limit(self):
        """History respects max_history limit."""
        space = QuorumSpace(max_history=5)
        for i in range(10):
            space.deposit_signal("TEST", f"a{i}")

        history = space.get_event_history()
        assert len(history) <= 5


class TestMetrics:
    """Test metrics collection."""

    def test_get_metrics(self):
        """Get space metrics."""
        space = QuorumSpace()
        space.deposit_signal("A", "a1")
        space.deposit_signal("B", "a2")
        space.deposit_signal("A", "a3")

        metrics = space.get_metrics()
        assert metrics["signal_types"] >= 2
        assert metrics["active_signals"] >= 1
        assert metrics["total_deposits"] >= 3

    def test_repr(self):
        """String representation includes info."""
        space = QuorumSpace()
        space.deposit_signal("TEST", "a1")

        repr_str = repr(space)
        assert "QuorumSpace" in repr_str
        assert "signals=" in repr_str
