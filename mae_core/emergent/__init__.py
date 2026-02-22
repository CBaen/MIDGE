"""Emergent capabilities - self-improvement, auto-healing, capability discovery, body awareness.

Auto-healing: Three-phase biological recovery (isolate, assess, restore).
Capability discovery: Detect and validate emergent agent behaviors.
Self-improvement: Track improvement metrics over time.
Somatic Map: Proprioceptive body map - blast radius analysis before self-modification.

These systems give Mae the ability to understand herself, heal herself,
grow capabilities that weren't explicitly designed, and KNOW the impact
of any change before it happens.
"""

from .auto_healer import (
    CH_FAILURE_DETECTED,
    CH_HEALING_COMPLETE,
    CH_HEALING_FAILED,
    CH_HEALING_PHASE,
    CH_HEALING_STARTED,
    AutoHealer,
    FailureReport,
    FailureType,
    HealingAction,
    HealingPhase,
    HealingRecord,
)
from .capability_discovery import (
    CH_CAPABILITY_FOUND,
    CH_CAPABILITY_RETIRED,
    CH_CAPABILITY_VALIDATED,
    CH_IMPROVEMENT_METRIC,
    CapabilityDiscovery,
    CapabilitySignature,
    CapabilityStatus,
    ImprovementMetric,
)
from .lymphatic_system import LymphaticSystem
from .senescence import SenescenceManager
from .proprioception import ProprioceptionSystem
from .microbiome import Microbiome
from .somatic_map import (
    CH_MODIFICATION_APPROVED,
    CH_MODIFICATION_PROPOSED,
    CH_MODIFICATION_REJECTED,
    CH_MODIFICATION_ROLLED_BACK,
    CH_SYSTEM_REGISTERED,
    BlastRadiusReport,
    ModificationRecord,
    ModificationVerdict,
    SomaticMap,
    SystemCriticality,
    SystemNode,
)

__all__ = [
    # Auto-Healing
    "AutoHealer",
    "FailureReport",
    "FailureType",
    "HealingAction",
    "HealingPhase",
    "HealingRecord",
    "CH_FAILURE_DETECTED",
    "CH_HEALING_STARTED",
    "CH_HEALING_PHASE",
    "CH_HEALING_COMPLETE",
    "CH_HEALING_FAILED",
    # Capability Discovery
    "CapabilityDiscovery",
    "CapabilitySignature",
    "CapabilityStatus",
    "ImprovementMetric",
    "CH_CAPABILITY_FOUND",
    "CH_CAPABILITY_VALIDATED",
    "CH_CAPABILITY_RETIRED",
    "CH_IMPROVEMENT_METRIC",
    # Somatic Map (Body Awareness)
    "SomaticMap",
    "SystemNode",
    "SystemCriticality",
    "BlastRadiusReport",
    "ModificationRecord",
    "ModificationVerdict",
    "CH_MODIFICATION_PROPOSED",
    "CH_MODIFICATION_APPROVED",
    "CH_MODIFICATION_REJECTED",
    "CH_MODIFICATION_ROLLED_BACK",
    "CH_SYSTEM_REGISTERED",
    # Lymphatic
    "LymphaticSystem",
    # Senescence
    "SenescenceManager",
    # Proprioception
    "ProprioceptionSystem",
    # Microbiome
    "Microbiome",
]
