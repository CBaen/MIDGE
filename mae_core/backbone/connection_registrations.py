"""Connection Registrations — dispatcher hub.

Biological analogy: The wiring diagram. If ConnectionRegistry is the
lymphatic system (monitoring infrastructure), this module is the
anatomical atlas — the complete map of every nerve, vessel, and
lymph channel in Mae's body.

13 groups organized by organ/function (split into 5 sub-modules):
  connection_registrations_bio.py        — EventBus/DR/CB/SH + Groups 2-3
  connection_registrations_metabolic.py  — Group 1 (metabolic -> OrganismState)
  connection_registrations_agent.py      — Groups 4-5
  connection_registrations_patterns.py   — Groups 6-9
  connection_registrations_advanced.py   — Groups 10-14

Each registration is a witnessed triad (Law 1: no bare dyads).
Witnesses are domain peers, not backbone governance.
"""

from __future__ import annotations

import logging
from typing import Any

from mae_core.backbone.connection_registry import (
    ConnectionRegistry,
    ConnectionType,
)
from mae_core.backbone.connection_registrations_advanced import (  # noqa: F401
    register_advanced_connections,
)
from mae_core.backbone.connection_registrations_agent import (  # noqa: F401
    register_agent_connections,
)
from mae_core.backbone.connection_registrations_bio import (  # noqa: F401
    register_bio_connections,
)
from mae_core.backbone.connection_registrations_patterns import (  # noqa: F401
    register_pattern_connections,
)

logger = logging.getLogger(__name__)


def register_all_connections(
    registry: ConnectionRegistry,
    systems: dict[str, Any],
) -> dict[str, int]:
    """Declare all known system-to-system connections at bootstrap.

    Reads the actual wiring from main.py's create_mae() and registers
    every connection as a witnessed triad.

    Returns summary counts by type.
    """
    counts: dict[str, int] = {
        "eventbus_pubsub": 0,
        "direct_reference": 0,
        "callback_registration": 0,
        "step_hook": 0,
        "total": 0,
    }

    def _reg(src: str, tgt: str, ctype: ConnectionType, **kwargs: Any) -> None:
        registry.register_connection(src, tgt, ctype, **kwargs)
        counts[ctype.value] = counts.get(ctype.value, 0) + 1
        counts["total"] += 1

    register_bio_connections(registry, systems, _reg)
    register_agent_connections(registry, systems, _reg)
    register_pattern_connections(registry, systems, _reg)
    register_advanced_connections(registry, systems, _reg)

    logger.info(
        "ConnectionRegistry: %d connections registered "
        "(EventBus=%d, Direct=%d, Callback=%d, StepHook=%d)",
        counts["total"],
        counts.get("eventbus_pubsub", 0),
        counts.get("direct_reference", 0),
        counts.get("callback_registration", 0),
        counts.get("step_hook", 0),
    )

    return counts
