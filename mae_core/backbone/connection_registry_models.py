"""Connection Registry data models — ConnectionType, ConnectionCriticality, EnforcementMode, ConnectionTriad.

Extracted from connection_registry.py for single-responsibility.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ConnectionType(Enum):
    """How two systems are connected."""

    EVENTBUS_PUBSUB = "eventbus_pubsub"
    DIRECT_REFERENCE = "direct_reference"
    CALLBACK_REGISTRATION = "callback_registration"
    STEP_HOOK = "step_hook"
    MEMORY_DATA_FLOW = "memory_data_flow"
    SUBSTRATE_INTEGRATION = "substrate_integration"


class ConnectionCriticality(Enum):
    """How important a connection is to Mae's operation."""

    STANDARD = "standard"
    IMPORTANT = "important"
    CRITICAL = "critical"


class EnforcementMode(Enum):
    """How strictly the registry enforces triadic witnessing."""

    PERMISSIVE = "permissive"  # Bootstrap phase. No checks. Everything passes.
    ADVISORY = "advisory"  # Log + event, allow everything
    BLOCKING = "blocking"  # Reject bare dyads, disable unhealthy


@dataclass
class ConnectionTriad:
    """A witnessed connection between two systems.

    Every connection in Mae has three+ parties:
    - source: the system that initiates/provides
    - target: the system that receives/consumes
    - witnesses: systems that monitor the connection (minimum 2 for Law 1)

    Law 1 requires: primary pathway (A->B), verification pathway (A->C->B),
    balance pathway (B->C->A). Two witnesses provide redundant oversight
    and eliminate single-witness fragility.
    """

    connection_id: str
    source: str
    target: str
    witnesses: list[str] = field(default_factory=list)
    connection_type: ConnectionType = ConnectionType.DIRECT_REFERENCE
    channel: Optional[str] = None  # EventBus channel name, if applicable
    criticality: ConnectionCriticality = ConnectionCriticality.STANDARD
    description: str = ""
    registered_at: float = field(default_factory=time.time)
    last_verified: float = 0.0
    healthy: bool = True

    @property
    def witness(self) -> Optional[str]:
        """Backward compat: first witness or None."""
        return self.witnesses[0] if self.witnesses else None
