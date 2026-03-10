"""Bootstrap Layers 14-21: Step hooks, EventBus wiring, registrations.

Wires all cross-system connections, endocrine consumers, somatic map,
connection registry, bidirectional awareness, fractal generator,
and stem cell registry.

This file is a thin dispatcher. Implementation lives in:
  - wiring_layers_14_16.py — Layers 14, 15, 15b, 16
  - wiring_layers_17_21.py — Layers 17, 18, 18b, 19, 20, 21
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

logger = logging.getLogger("midge.bootstrap")


def _register_somatic_systems(somatic_map, systems: dict) -> None:
    """Register all systems with SomaticMap for body awareness.

    Uses register_system() if available, falls back to heartbeat().
    """
    for name, system in systems.items():
        try:
            if hasattr(somatic_map, "register_system"):
                somatic_map.register_system(
                    system_id=name,
                    description=type(system).__name__,
                    depends_on=[],
                )
            elif hasattr(somatic_map, "heartbeat"):
                somatic_map.heartbeat(name)
        except Exception:
            logger.debug("Could not register %s with SomaticMap", name)


def bootstrap_wiring(ctx: SimpleNamespace) -> None:
    """Wire all cross-system connections and registrations (Layers 14-21)."""
    from mae_core.bootstrap.wiring_layers_14_16 import _wire_layers_14_16
    from mae_core.bootstrap.wiring_layers_17_21 import _wire_layers_17_21

    _wire_layers_14_16(ctx)
    _wire_layers_17_21(ctx)
