"""Learning engines - FRL, VDN, HAVEN, transfer, MAML, imitation, curiosity."""

from .knowledge_base import KnowledgeBase, TaskDescriptor, StoredEpisode

from .frl import (
    FederatedLearningEngine,
    PolicyUpdate,
    PolicyUpdateStrategy,
    AggregationMethod,
)
from .vdn import (
    ValueDecompositionEngine,
    DecompositionMethod,
    CreditAssignmentStrategy,
    CreditResult,
)
from .haven import (
    HavenRiskCoordinator,
    RiskLevel,
    RiskAssessment,
    InterventionType,
    ContagionStatus,
    ContagionReport,
)
from .curiosity import (
    CuriosityDrive,
    CuriositySignal,
    CuriosityType,
)
from .transfer_learning import (
    TransferLearningEngine,
    TransferStrategy,
    TransferResult,
)
from .maml import (
    MAMLLearner,
    MAMLConfig,
    AdaptationResult,
)
from .imitation import (
    ImitationLearning,
    ObservedBehavior,
    DemonstrationTrajectory,
    ImitationPolicy,
    TeacherProfile,
)

__all__ = [
    # Knowledge Base
    "KnowledgeBase",
    "TaskDescriptor",
    "StoredEpisode",
    # Federated RL
    "FederatedLearningEngine",
    "PolicyUpdate",
    "PolicyUpdateStrategy",
    "AggregationMethod",
    # Value Decomposition
    "ValueDecompositionEngine",
    "DecompositionMethod",
    "CreditAssignmentStrategy",
    "CreditResult",
    # HAVEN (Immune System)
    "HavenRiskCoordinator",
    "RiskLevel",
    "RiskAssessment",
    "InterventionType",
    "ContagionStatus",
    "ContagionReport",
    # Curiosity Drive
    "CuriosityDrive",
    "CuriositySignal",
    "CuriosityType",
    # Transfer Learning
    "TransferLearningEngine",
    "TransferStrategy",
    "TransferResult",
    # MAML Meta-Learning
    "MAMLLearner",
    "MAMLConfig",
    "AdaptationResult",
    # Imitation Learning
    "ImitationLearning",
    "ObservedBehavior",
    "DemonstrationTrajectory",
    "ImitationPolicy",
    "TeacherProfile",
]
