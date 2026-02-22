"""Backbone infrastructure - EventBus, StateStore, VectorStore, Enforcement Triad.

Replaces Redis and ChromaDB with pure Python + FAISS.
Enforcement Triad (TriadEnforcer + TriadWatchdog + TriadAuditor)
enforces Rule of 3 across all Mae processes.
No external servers required.
"""

from mae_core.backbone.fractal_generator import (
    CH_FRACTAL_ORGANIZED,
    CH_FRACTAL_TRIAD_CREATED,
    FRACTAL_GROUPING,
    MODULE_GROUPING,
    ORGAN_GROUPING,
    FractalGenerator,
    FractalLevel,
    FractalReport,
    TriadResult,
)
from mae_core.backbone.connection_registry import (
    CH_CONNECTION_BARE_DYAD,
    CH_CONNECTION_HEALTH,
    CH_CONNECTION_REGISTERED,
    CH_CONNECTION_VERIFIED,
    ConnectionCriticality,
    ConnectionRegistry,
    ConnectionTriad,
    ConnectionType,
)
from mae_core.backbone.event_bus import EventBus
from mae_core.backbone.holon_mixin import ALL_HOLON_CAPABILITIES, HolonMixin
from mae_core.backbone.holon_protocol import (
    CH_AWARENESS_ANOMALY,
    CH_AWARENESS_PULSE,
    AwarenessPulse,
    HolonEntry,
    HolonProxy,
    HolonRegistry,
)
from mae_core.backbone.state_store import StateStore
from mae_core.backbone.triad_auditor import (
    CH_AUDIT_FINDING,
    CH_AUDIT_HEALTH,
    AuditFinding,
    FindingSeverity,
    FindingType,
    TriadAuditor,
)
from mae_core.backbone.triad_enforcer import (
    CH_TRIAD_HEALTH,
    CH_TRIAD_REGISTERED,
    CH_TRIAD_VIOLATION,
    CH_TRIAD_VOTE_COMPLETE,
    ProcessCriticality,
    ProcessTriad,
    TriadEnforcer,
    Validator,
    ValidatorType,
    VoteResult,
)
from mae_core.backbone.triad_watchdog import (
    CH_WATCHDOG_BYPASS,
    CH_WATCHDOG_HEALTH,
    CH_WATCHDOG_SILENT,
    TriadWatchdog,
)

__all__ = [
    "EventBus",
    "StateStore",
    # Fractal Generator (recursive triadic structure)
    "FractalGenerator",
    "FractalLevel",
    "FractalReport",
    "TriadResult",
    "FRACTAL_GROUPING",
    "CH_FRACTAL_TRIAD_CREATED",
    "CH_FRACTAL_ORGANIZED",
    # Connection Registry (triadic witnessing)
    "ConnectionRegistry",
    "ConnectionTriad",
    "ConnectionType",
    "ConnectionCriticality",
    "CH_CONNECTION_REGISTERED",
    "CH_CONNECTION_VERIFIED",
    "CH_CONNECTION_BARE_DYAD",
    "CH_CONNECTION_HEALTH",
    # Holon Protocol (fractal self-awareness)
    "HolonRegistry",
    "HolonMixin",
    "ALL_HOLON_CAPABILITIES",
    "HolonEntry",
    "HolonProxy",
    "AwarenessPulse",
    "CH_AWARENESS_PULSE",
    "CH_AWARENESS_ANOMALY",
    # Triad Enforcer (Rule of 3 - FORMAL)
    "TriadEnforcer",
    "ProcessTriad",
    "Validator",
    "VoteResult",
    "ProcessCriticality",
    "ValidatorType",
    "CH_TRIAD_VIOLATION",
    "CH_TRIAD_REGISTERED",
    "CH_TRIAD_VOTE_COMPLETE",
    "CH_TRIAD_HEALTH",
    # Triad Watchdog (Rule of 3 - OPERATIONAL)
    "TriadWatchdog",
    "CH_WATCHDOG_BYPASS",
    "CH_WATCHDOG_SILENT",
    "CH_WATCHDOG_HEALTH",
    # Triad Auditor (Rule of 3 - BEHAVIORAL)
    "TriadAuditor",
    "AuditFinding",
    "FindingSeverity",
    "FindingType",
    "CH_AUDIT_FINDING",
    "CH_AUDIT_HEALTH",
]
