"""Bootstrap Layer 32: Triadic Bootstrap Audit.

After all 31 layers complete, three existing validators audit Mae's health
and witness each other's reports. This is the bootstrap's self-check:
the organism confirming its own successful creation before first breath.

Three validators form a K3 (fully connected triad):
  - SomaticMap:          Are all systems registered?
  - ConnectionRegistry:  Are all connections triadic? Is the seal set?
  - HolonRegistry:       Are all holons registered?

Two rounds:
  Round 1 — each validator inspects its own domain independently.
  Round 2 — each validator reads the other two's Round 1 results (witnessing).

One verdict: "bootstrap healthy" or "bootstrap degraded".

Biological analogy: A newborn's first cry. The nervous system (SomaticMap),
the circulatory system (ConnectionRegistry), and the cell-identity system
(HolonRegistry) each confirm they are ready. Together they affirm: Mae is
whole. Or they report what is missing before she takes her first step.
"""

from __future__ import annotations

import logging
from typing import Any
from types import SimpleNamespace

logger = logging.getLogger("midge.bootstrap")

# Minimum counts below which a validator reports degraded.
# Generous floors — partial registration is normal; 0 is a failure.
_MIN_SYSTEMS = 75
_MIN_HOLONS = 75


def bootstrap_audit(ctx: SimpleNamespace) -> None:
    """Run the triadic bootstrap audit (Layer 32).

    Reads health from SomaticMap, ConnectionRegistry, and HolonRegistry.
    Each witnesses the other two's results before the final verdict.
    Stores verdict on ctx.bootstrap_verdict for runtime inspection.
    Publishes bootstrap.audit_complete on EventBus.
    """
    somatic_map = getattr(ctx, "somatic_map", None)
    connection_registry = getattr(ctx, "connection_registry", None)
    holon_registry = getattr(ctx, "holon_registry", None)

    if not all([somatic_map, connection_registry, holon_registry]):
        missing = [
            name for name, obj in [
                ("somatic_map", somatic_map),
                ("connection_registry", connection_registry),
                ("holon_registry", holon_registry),
            ]
            if obj is None
        ]
        logger.warning(
            "Layer 32 - Bootstrap audit skipped: missing validators %s", missing
        )
        ctx.bootstrap_verdict = "bootstrap degraded"
        return

    # === Round 1: Independent domain checks ===
    somatic_report = _audit_somatic(somatic_map)
    connection_report = _audit_connections(connection_registry)
    holon_report = _audit_holons(holon_registry)

    # === Round 2: Each validator witnesses the other two (K3) ===
    # SomaticMap witnesses ConnectionRegistry + HolonRegistry
    somatic_report["witnesses"] = {
        "connections_healthy": connection_report["healthy"],
        "connections_bare_dyads": connection_report.get("bare_dyads", -1),
        "holons_healthy": holon_report["healthy"],
        "holons_count": holon_report.get("holon_count", 0),
    }

    # ConnectionRegistry witnesses SomaticMap + HolonRegistry
    connection_report["witnesses"] = {
        "somatic_healthy": somatic_report["healthy"],
        "somatic_system_count": somatic_report.get("system_count", 0),
        "holons_healthy": holon_report["healthy"],
        "holons_count": holon_report.get("holon_count", 0),
    }

    # HolonRegistry witnesses SomaticMap + ConnectionRegistry
    holon_report["witnesses"] = {
        "somatic_healthy": somatic_report["healthy"],
        "somatic_system_count": somatic_report.get("system_count", 0),
        "connections_healthy": connection_report["healthy"],
        "connections_bare_dyads": connection_report.get("bare_dyads", -1),
    }

    # === Verdict: consensus across all three ===
    all_healthy = (
        somatic_report["healthy"]
        and connection_report["healthy"]
        and holon_report["healthy"]
    )
    verdict = "bootstrap healthy" if all_healthy else "bootstrap degraded"

    # Store on ctx for runtime access
    ctx.bootstrap_verdict = verdict
    ctx.bootstrap_audit_report = {
        "verdict": verdict,
        "somatic": somatic_report,
        "connections": connection_report,
        "holons": holon_report,
    }

    # Log the verdict
    if all_healthy:
        logger.info(
            "Layer 32 - %s | systems=%d  bare_dyads=%d  holons=%d  sealed=%s",
            verdict.upper(),
            somatic_report["system_count"],
            connection_report["bare_dyads"],
            holon_report["holon_count"],
            connection_report["sealed"],
        )
    else:
        issues = []
        if not somatic_report["healthy"]:
            issues.append(f"somatic: {somatic_report['issues']}")
        if not connection_report["healthy"]:
            issues.append(f"connections: {connection_report['issues']}")
        if not holon_report["healthy"]:
            issues.append(f"holons: {holon_report['issues']}")
        logger.warning(
            "Layer 32 - %s | %s",
            verdict.upper(),
            "; ".join(issues),
        )

    # Register the K3 among validators (Group 13 connections)
    _register_audit_connections(ctx)

    # Publish on EventBus
    bus = getattr(ctx, "bus", None)
    if bus is not None:
        try:
            bus.publish("bootstrap.audit_complete", {
                "verdict": verdict,
                "healthy": all_healthy,
                "system_count": somatic_report["system_count"],
                "bare_dyads": connection_report["bare_dyads"],
                "holon_count": holon_report["holon_count"],
                "sealed": connection_report["sealed"],
            })
        except Exception:
            logger.debug("Layer 32 - Could not publish audit result", exc_info=True)


# ---------------------------------------------------------------------------
# Validator functions (Round 1)
# ---------------------------------------------------------------------------

def _audit_somatic(somatic_map: Any) -> dict:
    """SomaticMap validator: are all expected systems registered?"""
    try:
        stats = somatic_map.get_statistics()
        system_count = stats.get("total_systems", 0)
        unhealthy = somatic_map.get_unhealthy_systems(threshold=0.5)

        issues = []
        if system_count < _MIN_SYSTEMS:
            issues.append(
                f"only {system_count}/{_MIN_SYSTEMS}+ systems registered"
            )

        return {
            "validator": "SomaticMap",
            "healthy": system_count >= _MIN_SYSTEMS,
            "system_count": system_count,
            "unhealthy_systems": unhealthy[:10],  # cap log size
            "issues": issues,
        }
    except Exception as exc:
        return {
            "validator": "SomaticMap",
            "healthy": False,
            "system_count": 0,
            "unhealthy_systems": [],
            "issues": [f"audit error: {exc}"],
        }


def _audit_connections(connection_registry: Any) -> dict:
    """ConnectionRegistry validator: zero bare dyads, seal set."""
    try:
        stats = connection_registry.get_statistics()
        bare_dyads = stats.get("bare_dyads", -1)
        sealed = stats.get("sealed", False)
        total_connections = stats.get("total_connections", 0)

        issues = []
        if bare_dyads != 0:
            issues.append(f"{bare_dyads} bare dyads — Law 1 violated")
        if not sealed:
            issues.append("registry not sealed after bootstrap")

        return {
            "validator": "ConnectionRegistry",
            "healthy": bare_dyads == 0 and sealed,
            "bare_dyads": bare_dyads,
            "total_connections": total_connections,
            "sealed": sealed,
            "issues": issues,
        }
    except Exception as exc:
        return {
            "validator": "ConnectionRegistry",
            "healthy": False,
            "bare_dyads": -1,
            "total_connections": 0,
            "sealed": False,
            "issues": [f"audit error: {exc}"],
        }


def _audit_holons(holon_registry: Any) -> dict:
    """HolonRegistry validator: all systems have holon entries."""
    try:
        stats = holon_registry.get_statistics()
        holon_count = stats.get("total_holons", 0)

        issues = []
        if holon_count < _MIN_HOLONS:
            issues.append(
                f"only {holon_count}/{_MIN_HOLONS}+ holons registered"
            )

        return {
            "validator": "HolonRegistry",
            "healthy": holon_count >= _MIN_HOLONS,
            "holon_count": holon_count,
            "type_counts": stats.get("type_counts", {}),
            "max_depth": stats.get("max_depth", 0),
            "issues": issues,
        }
    except Exception as exc:
        return {
            "validator": "HolonRegistry",
            "healthy": False,
            "holon_count": 0,
            "type_counts": {},
            "max_depth": 0,
            "issues": [f"audit error: {exc}"],
        }


# ---------------------------------------------------------------------------
# K3 connections — the three validators witness each other (Group 13)
# ---------------------------------------------------------------------------

def _register_audit_connections(ctx: SimpleNamespace) -> None:
    """Register the K3 triad among the three audit validators.

    SomaticMap ↔ ConnectionRegistry ↔ HolonRegistry (fully connected).
    Each pair uses the third as witness (Law 1 compliant).

    Registered post-seal in ADVISORY mode — will generate advisory logs
    but will not be blocked.
    """
    reg = getattr(ctx, "connection_registry", None)
    if reg is None:
        return

    try:
        from mae_core.backbone.connection_registry import (
            ConnectionType,
            ConnectionCriticality,
        )
        dr = ConnectionType.DIRECT_REFERENCE

        reg.register_connection(
            "somatic_map", "connection_registry", dr,
            witnesses=["holon_registry"],
            criticality=ConnectionCriticality.IMPORTANT,
            description="Audit K3: somatic_map reads connection health",
        )
        reg.register_connection(
            "connection_registry", "holon_registry", dr,
            witnesses=["somatic_map"],
            criticality=ConnectionCriticality.IMPORTANT,
            description="Audit K3: connections registry reads holon count",
        )
        reg.register_connection(
            "holon_registry", "somatic_map", dr,
            witnesses=["connection_registry"],
            criticality=ConnectionCriticality.IMPORTANT,
            description="Audit K3: holon_registry reads system registry",
        )
        logger.info("Group 13 - Bootstrap Audit K3: 3 triadic connections registered")

    except Exception:
        logger.debug("Layer 32 - Could not register audit K3 connections", exc_info=True)
