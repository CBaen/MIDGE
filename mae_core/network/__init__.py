"""Network topology - octopus distributed cognition colony.

Octopus Brain system:
- OctopusArm: Semi-autonomous processing unit (2/3 of neurons)
- OctopusDistributedCognition: Central brain coordinating 8 arms
- OctopusAgent: Individual octopus with specialization + health
- OctopusColony: P2P multi-octopus network (Rule of 3)

Signal types for inter-component communication.
Rule of 3 constants for structural enforcement.
"""

from .octopus_signals import (
    ArmCapability,
    ArmState,
    CognitionMode,
    CoordinationSignal,
    OctopusSpecialization,
    Task,
    DEFAULT_ARM_CAPABILITIES,
    SPECIALIZATION_CAPABILITIES,
    TASK_CAPABILITY_MAP,
    CH_OCTOPUS_TASK,
    CH_OCTOPUS_COMPLETED,
    CH_OCTOPUS_EMERGENCY,
    CH_OCTOPUS_LEARNING,
    CH_OCTOPUS_HEALTH,
    CH_OCTOPUS_SPAWN,
    CH_OCTOPUS_DESPAWN,
)
from .octopus_arm import OctopusArm
from .octopus_cognition import OctopusDistributedCognition
from .octopus_agent import OctopusAgent
from .octopus_colony import (
    OctopusColony,
    MIN_AGENTS,
    MIN_CONNECTIONS,
    MIN_VOTES,
    MIN_LEARNING_STEPS,
    MIN_MEMORY_RECURRENCES,
    BASELINE_STEP,
    get_min_connections,
    validate_rule_of_3,
)

__all__ = [
    # Enums
    "ArmCapability",
    "CognitionMode",
    "OctopusSpecialization",
    # Data classes
    "ArmState",
    "CoordinationSignal",
    "Task",
    # Classes
    "OctopusArm",
    "OctopusDistributedCognition",
    "OctopusAgent",
    "OctopusColony",
    # Constants
    "DEFAULT_ARM_CAPABILITIES",
    "SPECIALIZATION_CAPABILITIES",
    "TASK_CAPABILITY_MAP",
    "MIN_AGENTS",
    "MIN_CONNECTIONS",
    "MIN_VOTES",
    "MIN_LEARNING_STEPS",
    "MIN_MEMORY_RECURRENCES",
    "BASELINE_STEP",
    # Channels
    "CH_OCTOPUS_TASK",
    "CH_OCTOPUS_COMPLETED",
    "CH_OCTOPUS_EMERGENCY",
    "CH_OCTOPUS_LEARNING",
    "CH_OCTOPUS_HEALTH",
    "CH_OCTOPUS_SPAWN",
    "CH_OCTOPUS_DESPAWN",
    # Functions
    "get_min_connections",
    "validate_rule_of_3",
]
