"""Bootstrap Layers 26-30: Biological systems, organism state, lifecycle.

Creates: digestive_system, respiratory_system, vestibular_system,
homeostasis, thermoregulation, energy_reserve, circulatory_system,
renal_filter, microbiome, emotional_system, theory_of_mind,
metacognition, nociception, proprioception, lymphatic_system,
senescence, boundary_membrane, reproductive_system, organism_state,
inhibition_system, goal_manager, arousal_regulator.

This file is a thin dispatcher. Implementation lives in:
  - organs_layers_26_27.py — Layers 26-27 (metabolic + social cognition)
  - organs_layers_28_30.py — Layers 28-30 (maintenance, organism state, lifecycle)
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


def bootstrap_organs(ctx: SimpleNamespace) -> None:
    """Create biological systems, organism state, and lifecycle steps (Layers 26-30)."""
    from mae_core.bootstrap.organs_layers_26_27 import _bootstrap_layers_26_27
    from mae_core.bootstrap.organs_layers_28_30 import _bootstrap_layers_28_30

    _bootstrap_layers_26_27(ctx)
    _bootstrap_layers_28_30(ctx)
