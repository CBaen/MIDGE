"""Communication systems - signal bus, stigmergy, GNN, quorum sensing, predictive fields, spatial consensus."""

from .consensus_metrics import (
    ConfidenceInterval,
    ConfidenceLevel,
    ConsensusConfidenceCalculator,
    is_consensus_reliable,
)
from .gnn_communicator import GNNCommunicator, SendResult
from .gnn_graph import AgentGraph, AgentNode, CommunicationEdge
from .gnn_message import GNNMessage, MessageEncoder, MessageType
from .gnn_propagator import GNNMessagePropagator, RoutingDecision, RoutingOptimizer
from .message_aggregator import AggregatedMessage, MessageAggregator, MessagePriority
from .quorum_sensor import QuorumSensor
from .quorum_signal import (
    DecayType,
    QuorumSignalType,
    SenseEvent,
    SignalTypes,
    get_signal_type,
    list_signal_types,
    register_custom_signal_type,
)
from .quorum_space import ConcentrationState, QuorumSpace
from .signal_bus import Signal, SignalBus
from .signal_priority import (
    PriorityConfig,
    PrioritizedSignal,
    SignalPriorityResolver,
    StepReport,
)
from .stigmergy import StigmergicEnvironment, StigmergicMarker
from .nociception import NociceptionSystem
from .predictive_field import AgentPrediction, PredictiveField
from .triage_classifier import TriageCategory, TriageClassifier, TriageResult
from .spatial_consensus import (
    SpatialConsensusResult,
    SpatialConsensusTracker,
    SpatialVote,
)
from .temporal_decay import (
    DecayConfig,
    DecayProfile,
    TemporalDecayManager,
    calculate_time_decay,
)

__all__ = [
    # Signal Bus
    "Signal",
    "SignalBus",
    # Signal Priority
    "PriorityConfig",
    "PrioritizedSignal",
    "SignalPriorityResolver",
    "StepReport",
    # Stigmergy
    "StigmergicEnvironment",
    "StigmergicMarker",
    # GNN Communication
    "AgentGraph",
    "AgentNode",
    "CommunicationEdge",
    "GNNCommunicator",
    "GNNMessage",
    "GNNMessagePropagator",
    "MessageEncoder",
    "MessageType",
    "RoutingDecision",
    "RoutingOptimizer",
    "SendResult",
    # Quorum Sensing
    "ConcentrationState",
    "ConfidenceInterval",
    "ConfidenceLevel",
    "ConsensusConfidenceCalculator",
    "DecayType",
    "QuorumSensor",
    "QuorumSignalType",
    "QuorumSpace",
    "SenseEvent",
    "SignalTypes",
    "get_signal_type",
    "is_consensus_reliable",
    "list_signal_types",
    "register_custom_signal_type",
    # Message Aggregation
    "AggregatedMessage",
    "MessageAggregator",
    "MessagePriority",
    # Predictive Field
    "PredictiveField",
    "AgentPrediction",
    # Spatial Consensus
    "SpatialConsensusTracker",
    "SpatialConsensusResult",
    "SpatialVote",
    # Temporal Decay
    "DecayConfig",
    "DecayProfile",
    "TemporalDecayManager",
    "calculate_time_decay",
    # Nociception
    "NociceptionSystem",
    # Triage
    "TriageCategory",
    "TriageClassifier",
    "TriageResult",
]
