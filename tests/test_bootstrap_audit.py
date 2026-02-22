"""Tests for Layer 32: Triadic Bootstrap Audit (mae_core/bootstrap/audit.py).

Tests cover:
- Formatter functions (pure, unit-testable)
- Round 1 independent domain checks
- Round 2 K3 witnessing
- Verdict logic (healthy / degraded)
- Group 13 connection registration
- EventBus publication
- Graceful degradation when validators are missing
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import pytest

from mae_core.bootstrap.audit import (
    bootstrap_audit,
    _audit_somatic,
    _audit_connections,
    _audit_holons,
    _register_audit_connections,
)


# ---------------------------------------------------------------------------
# Helpers — minimal mock validators
# ---------------------------------------------------------------------------

def _make_somatic(system_count: int = 82, unhealthy: list | None = None) -> MagicMock:
    m = MagicMock()
    m.get_statistics.return_value = {"total_systems": system_count}
    m.get_unhealthy_systems.return_value = unhealthy or []
    return m


def _make_connection_registry(bare_dyads: int = 0, sealed: bool = True, total: int = 214) -> MagicMock:
    m = MagicMock()
    m.get_statistics.return_value = {
        "bare_dyads": bare_dyads,
        "sealed": sealed,
        "total_connections": total,
    }
    return m


def _make_holon_registry(holon_count: int = 92) -> MagicMock:
    m = MagicMock()
    m.get_statistics.return_value = {
        "total_holons": holon_count,
        "type_counts": {"subsystem": 18, "organ": 5},
        "max_depth": 4,
    }
    m.get_all_ids.return_value = list(range(holon_count))
    return m


def _make_ctx(
    somatic_count: int = 82,
    bare_dyads: int = 0,
    sealed: bool = True,
    holon_count: int = 92,
    bus: MagicMock | None = None,
) -> types.SimpleNamespace:
    ctx = types.SimpleNamespace()
    ctx.somatic_map = _make_somatic(somatic_count)
    ctx.connection_registry = _make_connection_registry(bare_dyads, sealed)
    ctx.holon_registry = _make_holon_registry(holon_count)
    ctx.bus = bus or MagicMock()
    return ctx


# ---------------------------------------------------------------------------
# Unit tests: _audit_somatic
# ---------------------------------------------------------------------------

class TestAuditSomatic:
    def test_healthy_above_minimum(self):
        sm = _make_somatic(system_count=82)
        report = _audit_somatic(sm)
        assert report["healthy"] is True
        assert report["system_count"] == 82
        assert report["validator"] == "SomaticMap"
        assert report["issues"] == []

    def test_degraded_below_minimum(self):
        sm = _make_somatic(system_count=10)
        report = _audit_somatic(sm)
        assert report["healthy"] is False
        assert len(report["issues"]) == 1
        assert "10" in report["issues"][0]

    def test_at_minimum_boundary_is_healthy(self):
        sm = _make_somatic(system_count=75)
        report = _audit_somatic(sm)
        assert report["healthy"] is True

    def test_unhealthy_systems_capped_at_ten(self):
        # 15 unhealthy systems → report caps at 10
        unhealthy = [f"sys_{i}" for i in range(15)]
        sm = _make_somatic(system_count=82, unhealthy=unhealthy)
        report = _audit_somatic(sm)
        assert len(report["unhealthy_systems"]) == 10

    def test_exception_returns_degraded(self):
        sm = MagicMock()
        sm.get_statistics.side_effect = RuntimeError("boom")
        report = _audit_somatic(sm)
        assert report["healthy"] is False
        assert "audit error" in report["issues"][0]


# ---------------------------------------------------------------------------
# Unit tests: _audit_connections
# ---------------------------------------------------------------------------

class TestAuditConnections:
    def test_healthy_zero_dyads_sealed(self):
        cr = _make_connection_registry(bare_dyads=0, sealed=True)
        report = _audit_connections(cr)
        assert report["healthy"] is True
        assert report["bare_dyads"] == 0
        assert report["sealed"] is True
        assert report["issues"] == []

    def test_degraded_bare_dyads(self):
        cr = _make_connection_registry(bare_dyads=3, sealed=True)
        report = _audit_connections(cr)
        assert report["healthy"] is False
        assert "3 bare dyads" in report["issues"][0]

    def test_degraded_not_sealed(self):
        cr = _make_connection_registry(bare_dyads=0, sealed=False)
        report = _audit_connections(cr)
        assert report["healthy"] is False
        assert any("not sealed" in i for i in report["issues"])

    def test_degraded_both_issues(self):
        cr = _make_connection_registry(bare_dyads=2, sealed=False)
        report = _audit_connections(cr)
        assert report["healthy"] is False
        assert len(report["issues"]) == 2

    def test_exception_returns_degraded(self):
        cr = MagicMock()
        cr.get_statistics.side_effect = RuntimeError("db gone")
        report = _audit_connections(cr)
        assert report["healthy"] is False
        assert report["bare_dyads"] == -1


# ---------------------------------------------------------------------------
# Unit tests: _audit_holons
# ---------------------------------------------------------------------------

class TestAuditHolons:
    def test_healthy_above_minimum(self):
        hr = _make_holon_registry(holon_count=92)
        report = _audit_holons(hr)
        assert report["healthy"] is True
        assert report["holon_count"] == 92
        assert report["issues"] == []

    def test_degraded_below_minimum(self):
        hr = _make_holon_registry(holon_count=5)
        report = _audit_holons(hr)
        assert report["healthy"] is False
        assert "5" in report["issues"][0]

    def test_at_minimum_boundary_is_healthy(self):
        hr = _make_holon_registry(holon_count=75)
        report = _audit_holons(hr)
        assert report["healthy"] is True

    def test_exception_returns_degraded(self):
        hr = MagicMock()
        hr.get_statistics.side_effect = RuntimeError("registry down")
        report = _audit_holons(hr)
        assert report["healthy"] is False
        assert "audit error" in report["issues"][0]


# ---------------------------------------------------------------------------
# Integration: bootstrap_audit full run
# ---------------------------------------------------------------------------

class TestBootstrapAuditFull:
    def test_healthy_verdict_stored_on_ctx(self):
        ctx = _make_ctx()
        bootstrap_audit(ctx)
        assert ctx.bootstrap_verdict == "bootstrap healthy"

    def test_healthy_audit_report_keys(self):
        ctx = _make_ctx()
        bootstrap_audit(ctx)
        report = ctx.bootstrap_audit_report
        assert "verdict" in report
        assert "somatic" in report
        assert "connections" in report
        assert "holons" in report

    def test_degraded_when_bare_dyads(self):
        ctx = _make_ctx(bare_dyads=1)
        bootstrap_audit(ctx)
        assert ctx.bootstrap_verdict == "bootstrap degraded"

    def test_degraded_when_not_sealed(self):
        ctx = _make_ctx(sealed=False)
        bootstrap_audit(ctx)
        assert ctx.bootstrap_verdict == "bootstrap degraded"

    def test_degraded_when_too_few_systems(self):
        ctx = _make_ctx(somatic_count=5)
        bootstrap_audit(ctx)
        assert ctx.bootstrap_verdict == "bootstrap degraded"

    def test_degraded_when_too_few_holons(self):
        ctx = _make_ctx(holon_count=5)
        bootstrap_audit(ctx)
        assert ctx.bootstrap_verdict == "bootstrap degraded"

    def test_round_two_witnesses_populated(self):
        ctx = _make_ctx()
        bootstrap_audit(ctx)
        report = ctx.bootstrap_audit_report
        # Each validator should have a witnesses dict
        assert "witnesses" in report["somatic"]
        assert "witnesses" in report["connections"]
        assert "witnesses" in report["holons"]

    def test_somatic_witnesses_connection_and_holon(self):
        ctx = _make_ctx()
        bootstrap_audit(ctx)
        witnesses = ctx.bootstrap_audit_report["somatic"]["witnesses"]
        assert "connections_healthy" in witnesses
        assert "holons_healthy" in witnesses

    def test_connection_witnesses_somatic_and_holon(self):
        ctx = _make_ctx()
        bootstrap_audit(ctx)
        witnesses = ctx.bootstrap_audit_report["connections"]["witnesses"]
        assert "somatic_healthy" in witnesses
        assert "holons_healthy" in witnesses

    def test_holon_witnesses_somatic_and_connection(self):
        ctx = _make_ctx()
        bootstrap_audit(ctx)
        witnesses = ctx.bootstrap_audit_report["holons"]["witnesses"]
        assert "somatic_healthy" in witnesses
        assert "connections_healthy" in witnesses

    def test_event_bus_receives_audit_complete(self):
        bus = MagicMock()
        ctx = _make_ctx(bus=bus)
        bootstrap_audit(ctx)
        bus.publish.assert_called_once()
        call_args = bus.publish.call_args
        assert call_args[0][0] == "bootstrap.audit_complete"
        payload = call_args[0][1]
        assert "verdict" in payload
        assert "healthy" in payload

    def test_missing_validators_sets_degraded(self):
        ctx = types.SimpleNamespace()
        # No somatic_map, connection_registry, or holon_registry
        bootstrap_audit(ctx)
        assert ctx.bootstrap_verdict == "bootstrap degraded"

    def test_missing_bus_doesnt_crash(self):
        ctx = _make_ctx()
        ctx.bus = None
        # Should complete without raising
        bootstrap_audit(ctx)
        assert hasattr(ctx, "bootstrap_verdict")

    def test_group_13_connections_registered(self):
        ctx = _make_ctx()
        bootstrap_audit(ctx)
        # connection_registry.register_connection should be called 3 times for K3
        cr = ctx.connection_registry
        assert cr.register_connection.call_count == 3
