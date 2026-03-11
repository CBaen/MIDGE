"""Connection Registrations — Group 2: Backbone Self-Monitoring.

Nervous system peers watch each other. This IS the domain where
enforcer/watchdog/auditor are the correct witnesses — they are
each other's peers within the nervous system organ.

Extracted from connection_registrations_bio.py for single-responsibility.
"""

from __future__ import annotations

from typing import Any, Callable

from mae_core.backbone.connection_registry import (
    ConnectionCriticality,
    ConnectionRegistry,
    ConnectionType,
)


def register_backbone_connections(
    registry: ConnectionRegistry,
    systems: dict[str, Any],
    _reg: Callable,
) -> None:
    """Register Group 2: Backbone Self-Monitoring connections.

    Args:
        registry: ConnectionRegistry instance.
        systems: System dict (unused here, kept for uniform signature).
        _reg: Inner registration helper from register_all_connections.
    """
    eb = ConnectionType.EVENTBUS_PUBSUB

    # Connection registry -> auditor (peers: watchdog + enforcer)
    _reg("connection_registry", "auditor", eb,
         channel="connection.registered",
         witnesses=["watchdog", "enforcer"],
         description="New connections -- nervous system peers witness")
    _reg("connection_registry", "auditor", eb,
         channel="connection.verified",
         witnesses=["watchdog", "enforcer"],
         description="Verification results -- nervous system peers")
    _reg("connection_registry", "auditor", eb,
         channel="connection.bare_dyad",
         witnesses=["enforcer", "watchdog"],
         criticality=ConnectionCriticality.IMPORTANT,
         description="Bare dyad detection -- Law 1 violation alert")
    _reg("connection_registry", "auditor", eb,
         channel="connection.health",
         witnesses=["watchdog", "enforcer"],
         description="Connection health -- nervous system peers witness")
    _reg("connection_registry", "auditor", eb,
         channel="connection.blocked",
         witnesses=["enforcer", "watchdog"],
         criticality=ConnectionCriticality.IMPORTANT,
         description="Blocked connection -- enforcement peers witness")
    _reg("connection_registry", "auditor", eb,
         channel="connection.sealed",
         witnesses=["enforcer", "watchdog"],
         criticality=ConnectionCriticality.CRITICAL,
         description="Registry sealed -- enforcement activated")

    # Enforcer -> auditor (peers: watchdog + connection_registry)
    _reg("enforcer", "auditor", eb,
         channel="triad.process_registered",
         witnesses=["watchdog", "connection_registry"],
         description="New process triad -- enforcement peers witness")
    _reg("enforcer", "auditor", eb,
         channel="triad.vote_complete",
         witnesses=["watchdog", "connection_registry"],
         description="Validator vote -- enforcement peers witness")
    _reg("enforcer", "auditor", eb,
         channel="triad.health_report",
         witnesses=["watchdog", "connection_registry"],
         description="Enforcer health -- enforcement peers witness")

    # Watchdog -> auditor (peers: enforcer + connection_registry)
    _reg("watchdog", "auditor", eb,
         channel="watchdog.silent_validator",
         witnesses=["enforcer", "connection_registry"],
         criticality=ConnectionCriticality.IMPORTANT,
         description="Silent validator -- enforcement peers witness")
    _reg("watchdog", "auditor", eb,
         channel="watchdog.health_report",
         witnesses=["enforcer", "connection_registry"],
         description="Watchdog health -- enforcement peers witness")

    # Auditor -> somatic_map (peers: enforcer + watchdog cross-witness)
    _reg("auditor", "somatic_map", eb,
         channel="audit.finding",
         witnesses=["enforcer", "watchdog"],
         criticality=ConnectionCriticality.IMPORTANT,
         description="Audit finding -- enforcement triad witnesses")
    _reg("auditor", "somatic_map", eb,
         channel="audit.health_report",
         witnesses=["enforcer", "watchdog"],
         description="Auditor health -- enforcement triad witnesses")

    # Holon + fractal (structural peers witness)
    _reg("holon_registry", "somatic_map", eb,
         channel="holon.awareness_pulse",
         witnesses=["fractal_generator", "connection_registry"],
         description="Holon pulse -- structural peers witness")
    _reg("holon_registry", "auto_healer", eb,
         channel="holon.anomaly_detected",
         witnesses=["fractal_generator", "somatic_map"],
         criticality=ConnectionCriticality.IMPORTANT,
         description="Holon anomaly -- structural peers witness")
    _reg("fractal_generator", "somatic_map", eb,
         channel="fractal.organized",
         witnesses=["holon_registry", "connection_registry"],
         description="Fractal organization -- structural peers witness")
    _reg("triadic_verifier", "auto_healer", eb,
         channel="triadic.verification",
         witnesses=["connection_registry", "enforcer"],
         criticality=ConnectionCriticality.IMPORTANT,
         description="Low triadic compliance triggers healing assessment")
