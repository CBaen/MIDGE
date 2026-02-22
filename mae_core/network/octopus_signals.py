"""Octopus Signals - Shared types for the distributed cognition system.

Defines arm capabilities, cognition modes, coordination signals,
task types, and specializations. Kept separate to prevent circular
imports between arm, cognition, agent, and colony modules.

Biological analogy: Neurotransmitter types and receptor definitions.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class CognitionMode(Enum):
    """How the central brain coordinates its arms."""

    CENTRALIZED = "centralized"  # Central brain controls all arms
    DISTRIBUTED = "distributed"  # Arms work independently
    HYBRID = "hybrid"  # Balanced coordination (default)
    EMERGENCY = "emergency"  # Maximum decentralization, fastest response


class ArmCapability(Enum):
    """What an individual arm can do."""

    SENSORY_PROCESSING = "sensory"
    MEMORY_ACCESS = "memory"
    DECISION_MAKING = "decision"
    COMMUNICATION = "communication"
    LEARNING = "learning"
    ADAPTATION = "adaptation"


class OctopusSpecialization(Enum):
    """Specialization types for octopus agents in a colony."""

    GENERAL = "general"
    SENSORY = "sensory"
    MEMORY = "memory"
    DECISION = "decision"
    LEARNING = "learning"
    COMMUNICATION = "communication"
    ADAPTATION = "adaptation"
    EMERGENCY = "emergency"


# Default capability combinations for 8 arms (biological diversity)
DEFAULT_ARM_CAPABILITIES: list[set[ArmCapability]] = [
    {ArmCapability.SENSORY_PROCESSING, ArmCapability.LEARNING},
    {ArmCapability.MEMORY_ACCESS, ArmCapability.DECISION_MAKING},
    {ArmCapability.COMMUNICATION, ArmCapability.ADAPTATION},
    {ArmCapability.SENSORY_PROCESSING, ArmCapability.DECISION_MAKING},
    {ArmCapability.MEMORY_ACCESS, ArmCapability.LEARNING},
    {ArmCapability.COMMUNICATION, ArmCapability.SENSORY_PROCESSING},
    {ArmCapability.DECISION_MAKING, ArmCapability.ADAPTATION},
    {ArmCapability.LEARNING, ArmCapability.COMMUNICATION},
]

# Task type → required capabilities mapping
TASK_CAPABILITY_MAP: dict[str, set[ArmCapability]] = {
    "sensory_analysis": {ArmCapability.SENSORY_PROCESSING},
    "memory_retrieval": {ArmCapability.MEMORY_ACCESS},
    "decision_making": {ArmCapability.DECISION_MAKING},
    "communication": {ArmCapability.COMMUNICATION},
    "learning": {ArmCapability.LEARNING},
    "adaptation": {ArmCapability.ADAPTATION},
    "complex_analysis": {ArmCapability.SENSORY_PROCESSING, ArmCapability.DECISION_MAKING},
    "intelligent_response": {
        ArmCapability.MEMORY_ACCESS,
        ArmCapability.DECISION_MAKING,
        ArmCapability.LEARNING,
    },
}

# Specialization → enhanced capabilities
SPECIALIZATION_CAPABILITIES: dict[OctopusSpecialization, list[str]] = {
    OctopusSpecialization.GENERAL: ["versatile", "balanced"],
    OctopusSpecialization.SENSORY: ["sensor_processing", "pattern_recognition"],
    OctopusSpecialization.MEMORY: ["memory_retrieval", "experience_indexing"],
    OctopusSpecialization.DECISION: ["decision_making", "strategic_planning"],
    OctopusSpecialization.LEARNING: ["learning", "model_training"],
    OctopusSpecialization.COMMUNICATION: ["communication", "signal_routing"],
    OctopusSpecialization.ADAPTATION: ["adaptation", "self_optimization"],
    OctopusSpecialization.EMERGENCY: ["emergency_response", "crisis_management"],
}


@dataclass
class ArmState:
    """Current state of an individual arm."""

    arm_id: str
    capabilities: set[ArmCapability]
    current_task: str | None = None
    workload: float = 0.0  # [0, 1]
    health: float = 1.0  # [0, 1]
    last_activity: float = 0.0
    coordination_level: float = 0.5  # 0.0=autonomous, 1.0=centralized


@dataclass
class Task:
    """A work unit distributed to arms."""

    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    task_type: str = "general"
    priority: int = 5  # 1-10, higher = more important
    data: dict[str, Any] = field(default_factory=dict)
    required_capabilities: set[ArmCapability] = field(default_factory=set)
    max_processing_time: float = 30.0
    created_time: float = field(default_factory=time.time)
    assigned_arm: str | None = None
    status: str = "pending"  # pending, assigned, processing, completed, failed


@dataclass
class CoordinationSignal:
    """Inter-arm communication message."""

    signal_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    source_arm: str = ""
    signal_type: str = ""  # task_completion, resource_request, emergency, learning_update
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    priority: int = 1  # 1-10


# EventBus channels for octopus system
CH_OCTOPUS_TASK = "octopus.task_submitted"
CH_OCTOPUS_COMPLETED = "octopus.task_completed"
CH_OCTOPUS_EMERGENCY = "octopus.emergency"
CH_OCTOPUS_LEARNING = "octopus.learning_update"
CH_OCTOPUS_HEALTH = "octopus.health_report"
CH_OCTOPUS_SPAWN = "octopus.spawn"
CH_OCTOPUS_DESPAWN = "octopus.despawn"
