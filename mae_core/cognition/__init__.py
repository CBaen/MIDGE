"""Cognitive systems - world model, decision routing, causal reasoning, collective dreaming.

World Model: Foundation imagination engine with ensemble support.
Decision Router: Three-tier cascade (reflex -> habit -> prefrontal).
Causal Reasoning: Correlation vs causation via do-calculus, counterfactuals.
Collective Dream: Swarm-validated planning with expertise-weighted voting.
Validated Imagination: Tracks prediction accuracy, feeds into expertise.
"""

from .world_model import WorldModel, WorldModelConfig, Prediction
from .decision_router import (
    DecisionRouter,
    DecisionTier,
    ReflexPattern,
    Habit,
    RouterDecision,
)
from .causal_reasoning import (
    CausalReasoningEngine,
    CausalLink,
    CausalQueryResult,
    CausalRelationType,
)
from .collective_dream import (
    CollectiveDreamPlanner,
    Dream,
    DreamAgent,
    ConsensusResult,
)
from .validated_imagination import (
    ValidatedImagination,
    ValidatedImaginationPlanner,
    ImaginationPrediction,
    ImaginationAccuracy,
    TrajectoryStep,
)
from .theory_of_mind import TheoryOfMind
from .metacognition import MetacognitionMonitor

__all__ = [
    # World Model
    "WorldModel",
    "WorldModelConfig",
    "Prediction",
    # Decision Router
    "DecisionRouter",
    "DecisionTier",
    "ReflexPattern",
    "Habit",
    "RouterDecision",
    # Causal Reasoning
    "CausalReasoningEngine",
    "CausalLink",
    "CausalQueryResult",
    "CausalRelationType",
    # Collective Dream
    "CollectiveDreamPlanner",
    "Dream",
    "DreamAgent",
    "ConsensusResult",
    # Validated Imagination
    "ValidatedImagination",
    "ValidatedImaginationPlanner",
    "ImaginationPrediction",
    "ImaginationAccuracy",
    "TrajectoryStep",
    # Theory of Mind
    "TheoryOfMind",
    # Metacognition
    "MetacognitionMonitor",
]
