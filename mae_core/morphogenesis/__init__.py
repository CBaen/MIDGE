"""Morphogenesis - Mae's growth engine.

Coordinator: Detects novel problems, triggers organ creation/dissolution.
OrganBuilder: Designs blueprints from problem characteristics, spawns teams.
Organ: A living team of agents working together on a problem class.

Integration: CollectiveDream triggers on low consensus, AutoHealing triggers
on failure, Colony triggers on overflow. Endocrine modulates growth rate.
"""

from .coordinator import (
    CH_NOVELTY_DETECTED,
    CH_SPAWN_REQUEST,
    CH_TEAM_CREATED,
    CH_TEAM_DISSOLVED,
    MorphogenesisCoordinator,
    NoveltyDetector,
)
from .reproductive_system import ReproductiveSystem
from .organ_builder import (
    CoordinationProtocol,
    Organ,
    OrganBlueprint,
    OrganBuilder,
    OrganStatus,
    OrganTopology,
    ProblemSignature,
)

__all__ = [
    # Coordinator
    "MorphogenesisCoordinator",
    "NoveltyDetector",
    # Builder
    "OrganBuilder",
    "OrganBlueprint",
    "Organ",
    "ProblemSignature",
    # Enums
    "OrganStatus",
    "OrganTopology",
    "CoordinationProtocol",
    # EventBus channels
    "CH_SPAWN_REQUEST",
    "CH_TEAM_CREATED",
    "CH_TEAM_DISSOLVED",
    "CH_NOVELTY_DETECTED",
    # Reproductive
    "ReproductiveSystem",
]
