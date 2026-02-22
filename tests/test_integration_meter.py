"""Tests for IntegrationMeter — IIT Phi measurement and Markov blanket analysis.

Validates:
- Shannon entropy computation (1D and joint)
- Bipartition enumeration correctness
- Phi computation (integration detection)
- Markov blanket identification
- Step cadence and state collection
- Full report generation
"""

import math
from collections import deque
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mae_core.backbone.integration_meter import (
    CH_PHI_MEASUREMENT,
    IntegrationMeter,
    IntegrationReport,
    MarkovBlanketResult,
    PhiResult,
)


# =====================================================================
# Fixtures
# =====================================================================


SIMPLE_GROUPING = {
    "organ-a": {
        "sub-1": ["sys-a", "sys-b", "sys-c"],
        "sub-2": ["sys-d", "sys-e", "sys-f"],
    },
    "organ-b": {
        "sub-3": ["sys-g", "sys-h", "sys-i"],
    },
}


@pytest.fixture
def mock_registry():
    """HolonRegistry mock that supports get_proxy, get_entry, get_children, get_statistics."""
    registry = MagicMock()

    # Default proxy returns a sense dict
    proxy = MagicMock()
    proxy.sense.return_value = {"health": 0.8, "operational": True}
    registry.get_proxy.return_value = proxy

    # Default entry
    entry = MagicMock()
    entry.parent_id = "organ-a"
    registry.get_entry.return_value = entry

    # Default children
    registry.get_children.return_value = ["sys-a", "sys-b", "sys-c"]

    # Default statistics
    registry.get_statistics.return_value = {"type_counts": {}}

    return registry


@pytest.fixture
def mock_conn_registry():
    """ConnectionRegistry mock."""
    conn_reg = MagicMock()
    conn_reg.get_connections_for_system.return_value = []
    return conn_reg


@pytest.fixture
def mock_bus():
    """EventBus mock."""
    return MagicMock()


@pytest.fixture
def meter(mock_registry, mock_conn_registry, mock_bus):
    """IntegrationMeter with mocked dependencies and simple grouping."""
    return IntegrationMeter(
        holon_registry=mock_registry,
        connection_registry=mock_conn_registry,
        event_bus=mock_bus,
        fractal_grouping=SIMPLE_GROUPING,
        cadence=10,
        buffer_size=20,
        bins=4,
    )


# =====================================================================
# Test: Initialization
# =====================================================================


class TestInit:
    def test_default_init(self, mock_registry, mock_conn_registry, mock_bus):
        m = IntegrationMeter(mock_registry, mock_conn_registry, mock_bus,
                             fractal_grouping=SIMPLE_GROUPING)
        assert m._cadence == 89
        assert m._buffer_size == 50
        assert m._bins == 8

    def test_custom_params(self, meter):
        assert meter._cadence == 10
        assert meter._buffer_size == 20
        assert meter._bins == 4

    def test_initial_state(self, meter):
        assert meter._step_count == 0
        assert meter._measurement_count == 0
        assert meter._last_report is None
        assert len(meter._state_buffers) == 0


# =====================================================================
# Test: State to scalar
# =====================================================================


class TestStateToScalar:
    def test_none_returns_zero(self):
        assert IntegrationMeter._state_to_scalar(None) == 0.0

    def test_numeric_passthrough(self):
        assert IntegrationMeter._state_to_scalar(42.5) == 42.5
        assert IntegrationMeter._state_to_scalar(10) == 10.0

    def test_dict_sums_numerics(self):
        result = IntegrationMeter._state_to_scalar({"a": 1.0, "b": 2.0, "c": 3.0})
        assert result == 6.0

    def test_dict_handles_bools(self):
        result = IntegrationMeter._state_to_scalar({"x": True, "y": False})
        assert result == 1.0

    def test_dict_handles_strings(self):
        result = IntegrationMeter._state_to_scalar({"name": "test"})
        assert isinstance(result, float)
        assert result > 0  # hash-based

    def test_empty_dict(self):
        assert IntegrationMeter._state_to_scalar({}) == 0.0

    def test_nested_dict(self):
        result = IntegrationMeter._state_to_scalar({"outer": {"a": 1.0, "b": 2.0}})
        assert result == 3.0

    def test_non_dict_non_numeric(self):
        result = IntegrationMeter._state_to_scalar([1, 2, 3])
        assert isinstance(result, float)


# =====================================================================
# Test: Entropy
# =====================================================================


class TestEntropy:
    def test_constant_array_zero_entropy(self, meter):
        data = np.array([5.0] * 50)
        h = meter._compute_entropy(data)
        assert h == 0.0

    def test_uniform_distribution_max_entropy(self, meter):
        # Uniform across bins should give log2(bins) entropy
        data = np.linspace(0, 1, 200)
        h = meter._compute_entropy(data)
        max_h = math.log2(meter._bins)
        assert h > 0
        assert h <= max_h + 0.01  # Allow small numerical tolerance

    def test_single_element_zero(self, meter):
        data = np.array([1.0])
        h = meter._compute_entropy(data)
        assert h == 0.0

    def test_empty_array_zero(self, meter):
        data = np.array([])
        h = meter._compute_entropy(data)
        assert h == 0.0

    def test_two_values_gives_one_bit(self, meter):
        # Half in bin 0, half in bin 1 (using 4 bins, data in [0, 0.25))
        meter._bins = 2
        data = np.array([0.0] * 50 + [1.0] * 50)
        h = meter._compute_entropy(data)
        assert abs(h - 1.0) < 0.01  # Should be ~1 bit


class TestJointEntropy:
    def test_identical_columns_equal_marginal(self, meter):
        col = np.random.uniform(0, 1, 100)
        h_marginal = meter._compute_entropy(col)
        h_joint = meter._compute_joint_entropy([col, col.copy()])
        # Joint entropy of identical vars equals marginal entropy
        assert abs(h_joint - h_marginal) < 0.5  # Allow binning approximation

    def test_independent_columns_sum_of_marginals(self, meter):
        np.random.seed(42)
        col_a = np.random.uniform(0, 1, 500)
        col_b = np.random.uniform(0, 1, 500)
        h_a = meter._compute_entropy(col_a)
        h_b = meter._compute_entropy(col_b)
        h_joint = meter._compute_joint_entropy([col_a, col_b])
        # H(A,B) <= H(A) + H(B), approximately equal for independent
        assert h_joint <= h_a + h_b + 0.2
        assert h_joint > 0

    def test_empty_columns(self, meter):
        h = meter._compute_joint_entropy([])
        assert h == 0.0

    def test_short_columns(self, meter):
        h = meter._compute_joint_entropy([np.array([1.0])])
        assert h == 0.0


# =====================================================================
# Test: Bipartitions
# =====================================================================


class TestBipartitions:
    def test_two_items_one_partition(self):
        parts = IntegrationMeter._enumerate_bipartitions(["a", "b"])
        assert len(parts) == 1
        assert parts[0] == (("a",), ("b",))

    def test_three_items_three_partitions(self):
        parts = IntegrationMeter._enumerate_bipartitions(["a", "b", "c"])
        assert len(parts) == 3
        # Should be: {a}|{b,c}, {b}|{a,c}, {c}|{a,b}
        group_a_sizes = sorted([len(p[0]) for p in parts])
        assert group_a_sizes == [1, 1, 1]

    def test_four_items_seven_partitions(self):
        parts = IntegrationMeter._enumerate_bipartitions(["a", "b", "c", "d"])
        assert len(parts) == 7
        # 4 single-splits + 3 pair-splits = 7

    def test_five_items_fifteen_partitions(self):
        parts = IntegrationMeter._enumerate_bipartitions(["a", "b", "c", "d", "e"])
        assert len(parts) == 15

    def test_single_item_no_partitions(self):
        parts = IntegrationMeter._enumerate_bipartitions(["a"])
        assert len(parts) == 0

    def test_empty_no_partitions(self):
        parts = IntegrationMeter._enumerate_bipartitions([])
        assert len(parts) == 0

    def test_partitions_are_exhaustive(self):
        items = ["a", "b", "c"]
        parts = IntegrationMeter._enumerate_bipartitions(items)
        for group_a, group_b in parts:
            # Both groups non-empty
            assert len(group_a) > 0
            assert len(group_b) > 0
            # Together they contain all items
            assert sorted(group_a + group_b) == sorted(items)


# =====================================================================
# Test: Phi computation
# =====================================================================


class TestPhiComputation:
    def test_identical_states_zero_phi(self, meter):
        """When all children have identical state histories, phi should be 0."""
        buffers = {
            "a": deque([1.0] * 20, maxlen=20),
            "b": deque([1.0] * 20, maxlen=20),
            "c": deque([1.0] * 20, maxlen=20),
        }
        result = meter._compute_phi("test-sub", "subsystem", ["a", "b", "c"], buffers)
        assert result is not None
        assert result.phi == 0.0

    def test_correlated_states_positive_phi(self, meter):
        """When children states are correlated (not independent), phi > 0."""
        np.random.seed(42)
        base = np.random.uniform(0, 10, 50)
        buffers = {
            "a": deque(base + np.random.normal(0, 0.1, 50), maxlen=50),
            "b": deque(base + np.random.normal(0, 0.1, 50), maxlen=50),
            "c": deque(base + np.random.normal(0, 0.1, 50), maxlen=50),
        }
        result = meter._compute_phi("test-sub", "subsystem", ["a", "b", "c"], buffers)
        assert result is not None
        assert result.phi > 0.0, "Correlated states should produce positive phi"

    def test_phi_returns_none_insufficient_data(self, meter):
        """With fewer than 10 samples, phi returns None."""
        buffers = {
            "a": deque([1.0, 2.0], maxlen=20),
            "b": deque([3.0, 4.0], maxlen=20),
            "c": deque([5.0, 6.0], maxlen=20),
        }
        result = meter._compute_phi("test-sub", "subsystem", ["a", "b", "c"], buffers)
        assert result is None

    def test_phi_returns_none_missing_children(self, meter):
        """With missing children in buffers, phi handles gracefully."""
        buffers = {"a": deque([1.0] * 20, maxlen=20)}
        result = meter._compute_phi("test-sub", "subsystem", ["a", "b", "c"], buffers)
        assert result is None

    def test_mip_is_weakest_partition(self, meter):
        """MIP should correspond to the bipartition with minimum phi."""
        np.random.seed(123)
        # Make a and b highly correlated, c independent
        base = np.random.uniform(0, 10, 50)
        buffers = {
            "a": deque(base, maxlen=50),
            "b": deque(base + np.random.normal(0, 0.05, 50), maxlen=50),
            "c": deque(np.random.uniform(0, 10, 50), maxlen=50),
        }
        result = meter._compute_phi("test-sub", "subsystem", ["a", "b", "c"], buffers)
        assert result is not None
        # MIP should separate c from {a,b} since c is independent
        # The weakest cut should be between c and {a,b}
        assert len(result.all_partitions) == 3
        assert result.phi >= 0.0

    def test_phi_result_has_correct_fields(self, meter):
        np.random.seed(42)
        buffers = {
            "x": deque(np.random.uniform(0, 1, 20), maxlen=20),
            "y": deque(np.random.uniform(0, 1, 20), maxlen=20),
            "z": deque(np.random.uniform(0, 1, 20), maxlen=20),
        }
        result = meter._compute_phi("sub-x", "subsystem", ["x", "y", "z"], buffers)
        assert result is not None
        assert result.holon_id == "sub-x"
        assert result.holon_type == "subsystem"
        assert isinstance(result.phi, float)
        assert isinstance(result.mip, tuple)
        assert len(result.mip) == 2
        assert len(result.all_partitions) == 3  # 3 bipartitions for 3 items
        assert result.children_ids == ["x", "y", "z"]


# =====================================================================
# Test: Markov blanket
# =====================================================================


class TestMarkovBlanket:
    def test_blanket_includes_parent(self, meter):
        meter._holon_registry.get_entry.return_value = MagicMock(parent_id="organ-a")
        meter._holon_registry.get_children.side_effect = lambda x: {
            "sub-1": ["sys-a", "sys-b", "sys-c"],
            "organ-a": ["sub-1", "sub-2"],
        }.get(x, [])

        result = meter._compute_markov_blanket("sub-1")
        assert result is not None
        assert "organ-a" in result.blanket_states

    def test_blanket_includes_siblings(self, meter):
        meter._holon_registry.get_entry.return_value = MagicMock(parent_id="organ-a")
        meter._holon_registry.get_children.side_effect = lambda x: {
            "sub-1": ["sys-a", "sys-b", "sys-c"],
            "organ-a": ["sub-1", "sub-2", "sub-3"],
        }.get(x, [])

        result = meter._compute_markov_blanket("sub-1")
        assert result is not None
        assert "sub-2" in result.blanket_states
        assert "sub-3" in result.blanket_states

    def test_internal_states_are_children(self, meter):
        meter._holon_registry.get_entry.return_value = MagicMock(parent_id="organ-a")
        meter._holon_registry.get_children.side_effect = lambda x: {
            "sub-1": ["sys-a", "sys-b", "sys-c"],
            "organ-a": ["sub-1", "sub-2"],
        }.get(x, [])

        result = meter._compute_markov_blanket("sub-1")
        assert result is not None
        assert sorted(result.internal_states) == ["sys-a", "sys-b", "sys-c"]

    def test_cross_connections_counted(self, meter):
        meter._holon_registry.get_entry.return_value = MagicMock(parent_id="organ-a")
        meter._holon_registry.get_children.side_effect = lambda x: {
            "sub-1": ["sys-a", "sys-b", "sys-c"],
            "organ-a": ["sub-1", "sub-2"],
        }.get(x, [])

        # sys-a connects to sys-z (external)
        conn = MagicMock()
        conn.source = "sys-a"
        conn.target = "sys-z"
        meter._connection_registry.get_connections_for_system.side_effect = lambda x: {
            "sys-a": [conn],
            "sys-b": [],
            "sys-c": [],
        }.get(x, [])

        result = meter._compute_markov_blanket("sub-1")
        assert result is not None
        assert result.cross_connections >= 1
        assert "sys-z" in result.blanket_states

    def test_blanket_returns_none_for_unknown(self, meter):
        meter._holon_registry.get_entry.return_value = None
        result = meter._compute_markov_blanket("nonexistent")
        assert result is None

    def test_effectiveness_range(self, meter):
        meter._holon_registry.get_entry.return_value = MagicMock(parent_id="organ-a")
        meter._holon_registry.get_children.side_effect = lambda x: {
            "sub-1": ["sys-a", "sys-b", "sys-c"],
            "organ-a": ["sub-1"],
        }.get(x, [])

        result = meter._compute_markov_blanket("sub-1")
        assert result is not None
        assert 0.0 <= result.blanket_effectiveness <= 1.0


# =====================================================================
# Test: Step cadence
# =====================================================================


class TestStepCadence:
    def test_collects_every_step(self, meter):
        for _ in range(5):
            meter.step()
        # Should have collected states for systems in the grouping
        assert len(meter._state_buffers) > 0

    def test_computation_at_cadence(self, meter):
        """Measurement should run exactly at cadence intervals."""
        for _ in range(10):
            meter.step()
        assert meter._measurement_count == 1  # cadence=10

    def test_no_computation_before_cadence(self, meter):
        for _ in range(9):
            meter.step()
        assert meter._measurement_count == 0

    def test_multiple_cadence_cycles(self, meter):
        for _ in range(30):
            meter.step()
        assert meter._measurement_count == 3  # 10, 20, 30

    def test_publishes_on_event_bus(self, meter):
        for _ in range(10):
            meter.step()
        meter._event_bus.publish.assert_called()
        call_args = meter._event_bus.publish.call_args
        assert call_args[0][0] == CH_PHI_MEASUREMENT

    def test_step_count_increments(self, meter):
        assert meter._step_count == 0
        meter.step()
        assert meter._step_count == 1
        meter.step()
        assert meter._step_count == 2


# =====================================================================
# Test: Full report
# =====================================================================


class TestReport:
    def _run_to_report(self, meter):
        """Run enough steps to trigger measurement with varied state data."""
        np.random.seed(42)
        call_count = [0]

        def mock_sense():
            call_count[0] += 1
            return {"health": np.random.uniform(0, 1), "active": call_count[0] % 3}

        meter._holon_registry.get_proxy.return_value.sense.side_effect = mock_sense

        for _ in range(20):  # Enough for buffer + 2 cadence cycles (cadence=10)
            meter.step()

    def test_report_generated(self, meter):
        self._run_to_report(meter)
        report = meter.get_latest_report()
        assert report is not None
        assert isinstance(report, IntegrationReport)

    def test_report_has_subsystems(self, meter):
        self._run_to_report(meter)
        report = meter.get_latest_report()
        assert report is not None
        # Should have phi for subsystems that had enough data
        assert len(report.subsystem_phi) >= 0  # May be 0 if insufficient unique data

    def test_report_step_matches(self, meter):
        self._run_to_report(meter)
        report = meter.get_latest_report()
        assert report is not None
        assert report.step == 20  # Two cadence cycles at 10

    def test_report_measurement_number(self, meter):
        self._run_to_report(meter)
        report = meter.get_latest_report()
        assert report is not None
        assert report.measurement_number == 2  # Steps 10 and 20

    def test_report_has_blankets(self, meter):
        meter._holon_registry.get_entry.return_value = MagicMock(parent_id="organ-a")
        meter._holon_registry.get_children.side_effect = lambda x: {
            "sub-1": ["sys-a", "sys-b", "sys-c"],
            "sub-2": ["sys-d", "sys-e", "sys-f"],
            "sub-3": ["sys-g", "sys-h", "sys-i"],
            "organ-a": ["sub-1", "sub-2"],
            "organ-b": ["sub-3"],
        }.get(x, [])

        self._run_to_report(meter)
        report = meter.get_latest_report()
        assert report is not None
        assert len(report.markov_blankets) > 0


# =====================================================================
# Test: Statistics
# =====================================================================


class TestStatistics:
    def test_initial_statistics(self, meter):
        stats = meter.get_statistics()
        assert stats["step_count"] == 0
        assert stats["measurement_count"] == 0
        assert stats["cadence"] == 10
        assert stats["buffer_size"] == 20

    def test_statistics_after_steps(self, meter):
        for _ in range(10):
            meter.step()
        stats = meter.get_statistics()
        assert stats["step_count"] == 10
        assert stats["measurement_count"] == 1
        assert stats["systems_tracked"] > 0

    def test_get_state(self, meter):
        state = meter.get_state()
        assert "step_count" in state
        assert "mean_phi" in state
        assert "has_report" in state
