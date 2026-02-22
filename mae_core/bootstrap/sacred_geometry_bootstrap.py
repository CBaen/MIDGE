"""Bootstrap K4 Sacred Geometry structures into Mae's network topology.

Optional integration layer for creating K4 tetrahedron working groups
that coordinate cross-module systems with full triadic witnessing.

K4 structures are advisory: they provide coordination topology but
don't enforce or block operations. They generate events and metrics
but allow fallback to simpler paths if K4 is unavailable.

This module should be called AFTER ConnectionRegistry is sealed
(post-Layer 18) to ensure witnesses are properly assigned.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

logger = logging.getLogger("mae.bootstrap")


def bootstrap_k4_tetrahedra(ctx: SimpleNamespace) -> None:
    """Create K4 tetrahedron working groups (optional, post-bootstrap).

    K4 structures group 4 subsystems for enhanced cross-module coordination.
    Each K4 has 6 edges (full connectivity) with triadic witnesses (Law 1).

    Example usage (in main.py or elsewhere):
        bootstrap_k4_tetrahedra(ctx)

    This creates example K4 structures. Production code should create
    K4s based on actual module/subsystem grouping needs.
    """
    from mae_core.backbone.sacred_geometry import K4Generator

    # K4 requires ConnectionRegistry to be sealed (witnesses assigned)
    if not ctx.connection_registry.sealed:
        logger.warning("K4: ConnectionRegistry not sealed, K4 creation deferred")
        return

    generator = K4Generator(
        connection_registry=ctx.connection_registry,
        event_bus=ctx.bus if hasattr(ctx, "bus") else None,
    )

    # Example: Create K4 from major organ systems (if available)
    # These are optional demonstration K4s. Production systems should
    # create K4s based on actual coordination needs.

    # K4 Pattern 1: Sensing -> Cognition -> Decision -> Action
    # (Cross-organ working group for stimulus-response)
    major_organs = []
    if hasattr(ctx, "holon_registry"):
        # Attempt to find 4 major system groups
        try:
            all_ids = ctx.holon_registry.get_all_ids()
            # Filter to systems (not subsystems)
            systems = [
                id_ for id_ in all_ids
                if id_ not in ("mae",)  # Exclude root
            ]
            if len(systems) >= 4:
                major_organs = systems[:4]
        except Exception:
            major_organs = []

    if major_organs and len(major_organs) == 4:
        try:
            k4_stimulus_response = generator.generate_k4(
                name="stimulus-response-tetrahedron",
                nodes=major_organs,
                parent_id="mae",
            )
            report = generator.verify_k4_integrity(k4_stimulus_response)
            if report["overall_valid"]:
                logger.info(
                    "K4: stimulus-response tetrahedron created (nodes=%s, edges=%d, witnesses=%d)",
                    major_organs, len(k4_stimulus_response.edges), k4_stimulus_response.get_witness_count(),
                )
            else:
                logger.warning("K4: stimulus-response tetrahedron validation failed")
        except Exception as e:
            logger.debug("K4: stimulus-response tetrahedron creation failed: %s", e)

    logger.info("K4 Sacred Geometry bootstrap complete")


__all__ = ["bootstrap_k4_tetrahedra"]
