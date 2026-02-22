"""Tests for TriadicVerifier - Mathematical proof verification for Mae's triadic structure.

Validates:
- Laman rigidity detection for K3 subgraphs
- Peirce irreducibility of witnessed connections
- Hegel synthesis capacity via TriadEnforcer
- Simmel cycle detection for witness relationships
- Step cadence and report generation
- Full verification report
"""

from collections import deque
from unittest.mock import MagicMock, patch

import pytest

from mae_core.backbone.triadic_verifiers import (
    CH_TRIADIC_VERIFICATION,
    TriadicVerifier,
    VerificationReport,
)


# =====================================================================
# Fixtures
# =====================================================================


def _make_triad(source, target, witnesses, conn_type="direct_reference"):
    """Create a mock ConnectionTriad."""
    triad = MagicMock()
    triad.source = source
    triad.target = target
    triad.witnesses = witnesses
    triad.connection_type = MagicMock(value=conn_type)
    return triad


def _make_process(pid, validators, criticality="standard"):
    """Create a mock ProcessTriad with validators."""
    from mae_core.backbone.triad_enforcer import ValidatorType

    process = MagicMock()
    process.process_id = pid
    process.validator_count = len(validators)
    mock_validators = []
    for vid, vtype in validators:
        v = MagicMock()
        v.validator_id = vid
        v.validator_type = vtype
        mock_validators.append(v)
    process.validators = mock_validators
    process.criticality = MagicMock(value=criticality)
    return process


@pytest.fixture
def mock_conn_registry():
    """ConnectionRegistry mock with sample triadic connections."""
    registry = MagicMock()
    # 3 connections forming a triangle: A->B, B->C, A->C all with witnesses
    registry._connections = {
        "A->B": _make_triad("A", "B", ["C", "D"]),
        "B->C": _make_triad("B", "C", ["A", "D"]),
        "A->C": _make_triad("A", "C", ["B", "D"]),
    }
    return registry


@pytest.fixture
def mock_enforcer():
    """TriadEnforcer mock with sample processes."""
    from mae_core.backbone.triad_enforcer import ValidatorType

    enforcer = MagicMock()
    enforcer._processes = {
        "process_1": _make_process("process_1", [
            ("v1", ValidatorType.STRUCTURAL),
            ("v2", ValidatorType.BEHAVIORAL),
            ("v3", ValidatorType.OPERATIONAL),
        ]),
        "process_2": _make_process("process_2", [
            ("v1", ValidatorType.STRUCTURAL),
            ("v2", ValidatorType.CONSENSUS),
            ("v3", ValidatorType.CAUSAL),
        ]),
    }
    return enforcer


@pytest.fixture
def mock_bus():
    """EventBus mock."""
    return MagicMock()


@pytest.fixture
def verifier(mock_conn_registry, mock_enforcer, mock_bus):
    """TriadicVerifier with mocked dependencies."""
    return TriadicVerifier(
        connection_registry=mock_conn_registry,
        triad_enforcer=mock_enforcer,
        event_bus=mock_bus,
        cadence=10,
    )


# =====================================================================
# Test: Initialization
# =====================================================================


class TestInit:
    def test_default_cadence(self, mock_conn_registry, mock_enforcer, mock_bus):
        v = TriadicVerifier(mock_conn_registry, mock_enforcer, mock_bus)
        assert v._cadence == 89

    def test_custom_cadence(self, verifier):
        assert verifier._cadence == 10

    def test_initial_state(self, verifier):
        assert verifier._step_count == 0
        assert verifier._measurement_count == 0
        assert verifier._last_report is None


# =====================================================================
# Test: Laman Rigidity
# =====================================================================


class TestLamanRigidity:
    @patch("mae_core.backbone.fractal_generator.FRACTAL_GROUPING", {
        "organ-a": {"sub-1": ["A", "B", "C"]},
    })
    def test_k3_is_rigid(self, verifier):
        """K3 subgraph (3 nodes, 3 edges) satisfies Laman: 3 = 2(3)-3."""
        # Connections form complete triangle A-B, B-C, A-C
        result = verifier.verify_laman_rigidity()
        assert result["verified_count"] == 1
        assert result["total_count"] == 1
        assert result["compliance_pct"] == 100.0
        assert result["failures"] == []

    @patch("mae_core.backbone.fractal_generator.FRACTAL_GROUPING", {
        "organ-a": {"sub-1": ["A", "B", "C"]},
    })
    def test_incomplete_triangle_fails(self, verifier):
        """A subgraph with fewer than 3 edges is not rigid."""
        verifier._connection_registry._connections = {
            "A->B": _make_triad("A", "B", ["C"]),
            # Only 1 edge, need 3 for Laman
        }
        result = verifier.verify_laman_rigidity()
        assert result["verified_count"] == 0
        assert len(result["failures"]) == 1

    @patch("mae_core.backbone.fractal_generator.FRACTAL_GROUPING", {
        "organ-a": {
            "sub-1": ["A", "B", "C"],
            "sub-2": ["D", "E", "F"],
        },
    })
    def test_multiple_subsystems(self, verifier):
        """Check multiple subsystems independently."""
        verifier._connection_registry._connections = {
            "A->B": _make_triad("A", "B", ["C"]),
            "B->C": _make_triad("B", "C", ["A"]),
            "A->C": _make_triad("A", "C", ["B"]),
            "D->E": _make_triad("D", "E", ["F"]),
            "E->F": _make_triad("E", "F", ["D"]),
            "D->F": _make_triad("D", "F", ["E"]),
        }
        result = verifier.verify_laman_rigidity()
        assert result["verified_count"] == 2
        assert result["total_count"] == 2
        assert result["compliance_pct"] == 100.0

    def test_empty_registry(self, verifier):
        verifier._connection_registry._connections = {}
        result = verifier.verify_laman_rigidity()
        assert result["total_count"] == 0
        assert result["compliance_pct"] == 0.0


# =====================================================================
# Test: Peirce Irreducibility
# =====================================================================


class TestPeirceIrreducibility:
    def test_all_witnessed_connections_irreducible(self, verifier):
        """Every connection with source, target, and witness is irreducible."""
        result = verifier.verify_peirce_irreducibility()
        assert result["verified_count"] == 3
        assert result["total_count"] == 3
        assert result["compliance_pct"] == 100.0

    def test_bare_dyad_not_irreducible(self, verifier):
        """A bare dyad (no witness) is NOT irreducible."""
        verifier._connection_registry._connections["X->Y"] = _make_triad("X", "Y", [])
        result = verifier.verify_peirce_irreducibility()
        assert result["verified_count"] == 3  # The 3 original ones
        assert result["total_count"] == 4
        assert len(result["failures"]) == 1
        assert result["failures"][0]["reason"] == "bare_dyad"

    def test_empty_registry(self, verifier):
        verifier._connection_registry._connections = {}
        result = verifier.verify_peirce_irreducibility()
        assert result["total_count"] == 0


# =====================================================================
# Test: Hegel Synthesis
# =====================================================================


class TestHegelSynthesis:
    def test_compliant_processes_are_synthesis_capable(self, verifier):
        """Processes with 3+ odd complementary validators support synthesis."""
        result = verifier.verify_hegel_synthesis()
        assert result["verified_count"] == 2
        assert result["total_count"] == 2
        assert result["compliance_pct"] == 100.0

    def test_even_validator_count_fails(self, verifier):
        """Even validator count prevents synthesis (deadlock risk)."""
        from mae_core.backbone.triad_enforcer import ValidatorType

        verifier._triad_enforcer._processes["bad"] = _make_process("bad", [
            ("v1", ValidatorType.STRUCTURAL),
            ("v2", ValidatorType.BEHAVIORAL),
        ])
        result = verifier.verify_hegel_synthesis()
        assert result["verified_count"] == 2  # Original 2 still good
        assert result["total_count"] == 3
        assert len(result["failures"]) == 1

    def test_non_complementary_fails(self, verifier):
        """All same validator type is not complementary."""
        from mae_core.backbone.triad_enforcer import ValidatorType

        verifier._triad_enforcer._processes["mono"] = _make_process("mono", [
            ("v1", ValidatorType.STRUCTURAL),
            ("v2", ValidatorType.STRUCTURAL),
            ("v3", ValidatorType.STRUCTURAL),
        ])
        result = verifier.verify_hegel_synthesis()
        assert len(result["failures"]) == 1
        assert "not complementary" in result["failures"][0]["reasons"]

    def test_empty_enforcer(self, verifier):
        verifier._triad_enforcer._processes = {}
        result = verifier.verify_hegel_synthesis()
        assert result["total_count"] == 0


# =====================================================================
# Test: Simmel Witness Cycles
# =====================================================================


class TestSimmelCycles:
    def test_triangle_forms_cycle(self, verifier):
        """A full triangle (A-B, B-C, A-C) with witnesses should form cycles."""
        result = verifier.verify_simmel_cycles()
        # All 3 connections have witnesses reachable within 2 hops
        assert result["verified_count"] == 3
        assert result["compliance_pct"] == 100.0

    def test_bare_dyad_no_cycle(self, verifier):
        """No witness means no cycle."""
        verifier._connection_registry._connections = {
            "X->Y": _make_triad("X", "Y", []),
        }
        result = verifier.verify_simmel_cycles()
        assert result["verified_count"] == 0
        assert len(result["failures"]) == 1

    def test_disconnected_witness_fails(self, verifier):
        """A witness with no connection to source or target fails cycle check."""
        verifier._connection_registry._connections = {
            "A->B": _make_triad("A", "B", ["Z"]),
            # Z has no connections to A or B in the graph
        }
        result = verifier.verify_simmel_cycles()
        assert result["verified_count"] == 0
        assert len(result["failures"]) == 1

    def test_witness_reachable_via_2_hops(self, verifier):
        """Witness reachable within 2 hops counts as a cycle."""
        verifier._connection_registry._connections = {
            "A->B": _make_triad("A", "B", ["C"]),
            "A->M": _make_triad("A", "M", ["B"]),
            "M->C": _make_triad("M", "C", ["A"]),
            "B->N": _make_triad("B", "N", ["A"]),
            "N->C": _make_triad("N", "C", ["B"]),
        }
        # A->B has witness C; A-M-C (2 hops) and B-N-C (2 hops)
        result = verifier.verify_simmel_cycles()
        assert result["verified_count"] >= 1  # At least A->B should pass

    def test_empty_registry(self, verifier):
        verifier._connection_registry._connections = {}
        result = verifier.verify_simmel_cycles()
        assert result["total_count"] == 0


# =====================================================================
# Test: Step Cadence
# =====================================================================


class TestStepCadence:
    def test_no_computation_before_cadence(self, verifier):
        for _ in range(9):
            verifier.step()
        assert verifier._measurement_count == 0

    def test_computation_at_cadence(self, verifier):
        for _ in range(10):
            verifier.step()
        assert verifier._measurement_count == 1

    def test_multiple_cadence_cycles(self, verifier):
        for _ in range(30):
            verifier.step()
        assert verifier._measurement_count == 3

    def test_step_count_increments(self, verifier):
        assert verifier._step_count == 0
        verifier.step()
        assert verifier._step_count == 1

    def test_publishes_on_event_bus(self, verifier):
        for _ in range(10):
            verifier.step()
        verifier._event_bus.publish.assert_called()
        call_args = verifier._event_bus.publish.call_args
        assert call_args[0][0] == CH_TRIADIC_VERIFICATION


# =====================================================================
# Test: Full Report
# =====================================================================


class TestReport:
    def test_report_generated(self, verifier):
        for _ in range(10):
            verifier.step()
        report = verifier.get_latest_report()
        assert report is not None
        assert isinstance(report, VerificationReport)

    def test_report_has_all_verifiers(self, verifier):
        for _ in range(10):
            verifier.step()
        report = verifier.get_latest_report()
        assert "verified_count" in report.laman
        assert "verified_count" in report.peirce
        assert "verified_count" in report.hegel
        assert "verified_count" in report.simmel

    def test_report_step_matches(self, verifier):
        for _ in range(10):
            verifier.step()
        report = verifier.get_latest_report()
        assert report.step == 10

    def test_overall_compliance(self, verifier):
        for _ in range(10):
            verifier.step()
        report = verifier.get_latest_report()
        assert 0.0 <= report.overall_compliance_pct <= 100.0

    def test_report_history(self, verifier):
        for _ in range(20):
            verifier.step()
        assert len(verifier._report_history) == 2


# =====================================================================
# Test: Statistics
# =====================================================================


class TestStatistics:
    def test_initial_statistics(self, verifier):
        stats = verifier.get_statistics()
        assert stats["step_count"] == 0
        assert stats["measurement_count"] == 0
        assert stats["cadence"] == 10

    def test_statistics_after_steps(self, verifier):
        for _ in range(10):
            verifier.step()
        stats = verifier.get_statistics()
        assert stats["step_count"] == 10
        assert stats["measurement_count"] == 1
        assert "last_measurement" in stats

    def test_get_state(self, verifier):
        state = verifier.get_state()
        assert "step_count" in state
        assert "overall_pct" in state
        assert "has_report" in state
