"""Triad Wiring — replace permissive defaults with real system methods.

Extracted from triad_registry.py for single-responsibility.

Called after all systems are created (post-Layer 9) to wire actual
validation logic into the TriadEnforcer. Validators registered with
permissive defaults in register_all_triads() get upgraded to real
system method calls.
"""

from __future__ import annotations

import logging
from typing import Any

from mae_core.backbone.triad_enforcer import TriadEnforcer

logger = logging.getLogger(__name__)


# =====================================================================
# Deferred Wiring — Replace permissive defaults with real systems
# =====================================================================

# Map: (validator_id, process_id) → (system_key, method_name)
# Only entries where the validator_id and method name differ from
# the convention need explicit mapping. Convention: validator_id is
# the system key, method is "validate_{process_type}".
_VALIDATOR_METHOD_MAP: dict[str, dict[str, str]] = {
    # validator_id → { process_id → method_name }
    "haven": {
        "self_modification": "validate_modification",
        "decision_making": "validate_decision",
        "healing": "validate_healing",
        "agent_lifecycle": "validate_lifecycle",
        "resource_allocation": "validate_threat",
        "learning_policy": "validate_policy",
        "endocrine_regulation": "validate_endocrine",
        "threat_detection": "validate_threat",
    },
}


def wire_triad_systems(enforcer: TriadEnforcer, systems: dict[str, Any]) -> int:
    """Replace permissive default lambdas with real system methods.

    Called after all systems are created (post-Layer 9) to wire
    actual validation logic into the TriadEnforcer. Validators
    registered with permissive defaults in register_all_triads()
    get upgraded to real system method calls.

    Args:
        enforcer: The TriadEnforcer with registered processes.
        systems: Dict mapping system keys to live system instances.

    Returns:
        Number of validators successfully wired.
    """
    wired = 0

    for process_id, triad in enforcer._processes.items():
        for validator in triad.validators:
            vid = validator.validator_id
            sys = systems.get(vid)
            if sys is None:
                continue

            # Look up method name: explicit map first, then convention
            method_map = _VALIDATOR_METHOD_MAP.get(vid, {})
            method_name = method_map.get(process_id)
            if method_name is None:
                # Convention: process "decision_making" → "validate_decision"
                # Take first word of process_id
                method_name = f"validate_{process_id.split('_')[0]}"

            if hasattr(sys, method_name):
                fn = getattr(sys, method_name)
                validator.validate_fn = lambda ctx, _fn=fn: _fn(ctx)
                wired += 1
                logger.debug(
                    "Wired %s.%s → process %s", vid, method_name, process_id,
                )

    logger.info("wire_triad_systems: %d validators wired to real systems", wired)
    return wired
