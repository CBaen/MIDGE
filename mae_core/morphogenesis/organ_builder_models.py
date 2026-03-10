"""
organ_builder_models.py - Dataclasses and enums for organ morphogenesis.

Extracted from organ_builder.py to break circular dependency between
organ_builder.py and organ_builder_design.py.

Provides: OrganStatus, CoordinationProtocol, OrganTopology,
          ProblemSignature, OrganBlueprint, Organ.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class OrganStatus(Enum):
    """Organ lifecycle status."""

    GROWING = "growing"  # Being constructed
    ACTIVE = "active"  # Operating
    DISSOLVING = "dissolving"  # Being deconstructed
    DISSOLVED = "dissolved"  # No longer active


class CoordinationProtocol(Enum):
    """How agents within an organ coordinate decisions."""

    CONSENSUS = "consensus"  # All agents vote
    HIERARCHICAL = "hierarchical"  # Coordinator decides
    AUCTION = "auction"  # Agents bid for tasks


class OrganTopology(Enum):
    """How agents within an organ are connected."""

    MESH = "mesh"  # All-to-all (small teams)
    STAR = "star"  # Hub-spoke (coordinator-centered)
    RING = "ring"  # Circular chain (ordered pipeline)
    HIERARCHICAL = "hierarchical"  # Two-level coordination


@dataclass
class ProblemSignature:
    """Characterizes a class of problems for morphogenesis.

    Morphogenesis examines these dimensions to decide what kind
    of organ (agent team) to build.
    """

    signature_id: str = ""
    coordination_level: float = 0.5  # 0=independent, 1=tightly coupled
    exploration_level: float = 0.5  # 0=exploit known, 1=explore new
    complexity: float = 0.5  # 0=simple, 1=highly complex
    risk_level: float = 0.3  # 0=safe, 1=dangerous
    temporal_pattern: str = "persistent"  # "episodic" or "persistent"
    sparse_rewards: bool = False
    domain: str = "general"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.signature_id:
            self.signature_id = f"sig_{uuid.uuid4().hex[:8]}"

    def similarity(self, other: ProblemSignature) -> float:
        """How similar is this problem to another? (0-1)."""
        dims = [
            abs(self.coordination_level - other.coordination_level),
            abs(self.exploration_level - other.exploration_level),
            abs(self.complexity - other.complexity),
            abs(self.risk_level - other.risk_level),
        ]
        return 1.0 - (sum(dims) / len(dims))


@dataclass
class OrganBlueprint:
    """Specification for an organ (agent team) to be built.

    Describes composition (agent types and counts), topology
    (how they connect), coordination (how they decide), and
    lifecycle parameters.
    """

    organ_id: str = ""
    name: str = "unnamed_organ"
    purpose: str = ""  # What problem class this organ solves
    composition: dict[str, int] = field(default_factory=dict)  # {agent_type: count}
    topology: OrganTopology = OrganTopology.MESH
    protocol: CoordinationProtocol = CoordinationProtocol.CONSENSUS
    transient: bool = False  # Dissolves after task completion
    max_lifetime: Optional[float] = None  # Seconds (None = permanent)
    max_size: int = 50
    performance_threshold: float = 0.3  # Below this, organ is underperforming
    problem_signature: Optional[ProblemSignature] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.organ_id:
            self.organ_id = f"organ_{uuid.uuid4().hex[:8]}"

    @property
    def total_agents(self) -> int:
        return sum(self.composition.values())

    def validate(self) -> bool:
        """Check blueprint validity."""
        if self.total_agents <= 0:
            return False
        if self.total_agents > self.max_size:
            return False
        return True


class Organ:
    """A living team of agents working together.

    Organs are created by OrganBuilder from blueprints, contain
    agent references, track performance, and can be dissolved.
    """

    def __init__(self, blueprint: OrganBlueprint) -> None:
        self.blueprint = blueprint
        self.organ_id = blueprint.organ_id
        self.status = OrganStatus.GROWING
        self.agents: list[dict[str, Any]] = []
        self.agent_ids: list[int] = []  # Mesa agent unique_ids
        self.coordinator_id: Optional[int] = None

        # Performance tracking
        self.performance_history: list[float] = []
        self.task_count = 0
        self.success_count = 0

        # Lifecycle
        self.creation_time = time.time()
        self.last_active_time = self.creation_time

    def add_agent(self, agent_id: int, agent_type: str, **kwargs: Any) -> None:
        """Register an agent with this organ."""
        self.agents.append(
            {"agent_id": agent_id, "agent_type": agent_type, **kwargs}
        )
        self.agent_ids.append(agent_id)

        # First coordinator-type agent becomes the coordinator
        if self.coordinator_id is None and agent_type == "coordinator":
            self.coordinator_id = agent_id

    def record_task(self, success: bool) -> None:
        """Record a task outcome."""
        self.task_count += 1
        if success:
            self.success_count += 1
        self.last_active_time = time.time()

    def evaluate_performance(self) -> float:
        """Evaluate organ performance (success rate)."""
        if self.task_count == 0:
            return 0.5  # Neutral when no data
        perf = self.success_count / self.task_count
        self.performance_history.append(perf)
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
        return perf

    def should_dissolve(self) -> bool:
        """Check if this organ should be dissolved."""
        # Transient organs dissolve after no activity
        if self.blueprint.transient:
            idle_time = time.time() - self.last_active_time
            if idle_time > 300:  # 5 minutes of no activity
                return True

        # Lifetime exceeded
        if self.blueprint.max_lifetime:
            age = time.time() - self.creation_time
            if age > self.blueprint.max_lifetime:
                return True

        # Persistent underperformance
        if len(self.performance_history) >= 10:
            recent = self.performance_history[-10:]
            avg = sum(recent) / len(recent)
            if avg < self.blueprint.performance_threshold:
                return True

        return False

    def get_statistics(self) -> dict[str, Any]:
        """Get organ statistics."""
        return {
            "organ_id": self.organ_id,
            "name": self.blueprint.name,
            "status": self.status.value,
            "agent_count": len(self.agents),
            "agent_ids": list(self.agent_ids),
            "coordinator_id": self.coordinator_id,
            "task_count": self.task_count,
            "success_count": self.success_count,
            "success_rate": (
                self.success_count / self.task_count if self.task_count > 0 else 0.0
            ),
            "age_seconds": time.time() - self.creation_time,
            "topology": self.blueprint.topology.value,
            "protocol": self.blueprint.protocol.value,
            "transient": self.blueprint.transient,
        }
