"""Bootstrap Layers 22-25: Deep memory, pattern ecosystem, action environment.

Creates: deep_store, memory_bridge, pattern_distiller, narrator,
pattern_bus, pattern_cortex, pattern_consolidator, attentional_gate,
task_pool, organism_action.

This file is a thin dispatcher. Implementation lives in:
  - patterns_layers_22_25.py — all layers 22 through 25d
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

logger = logging.getLogger("midge.bootstrap")


def _register_somatic_systems(somatic_map, systems: dict) -> None:
    """Register all systems with SomaticMap for body awareness."""
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


def bootstrap_patterns(ctx: SimpleNamespace) -> None:
    """Create deep memory, pattern ecosystem, and action environment (Layers 22-25)."""
    from mae_core.bootstrap.patterns_layers_22_25 import _bootstrap_layers_22_25

    _bootstrap_layers_22_25(ctx)
