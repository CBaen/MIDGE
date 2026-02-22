"""Substrate - Mae's body. The mycelial network everything connects through.

Topology: Graph structure (ring, scale-free, small-world, mesh).
Nutrient Flow: Resource distribution via osmotic pressure gradients.
MycelialSubstrate: Orchestrator integrating topology, nutrients, signals.

Every system in Mae connects through substrate. It provides the physical
medium for agent positions, communication paths, resource distribution,
and network health monitoring.
"""

from .mycelial_substrate import (
    CH_AGENT_DEREGISTERED,
    CH_AGENT_REGISTERED,
    CH_HEALTH_REPORT,
    CH_ISOLATION_DETECTED,
    CH_STARVATION_ALERT,
    CH_TOPOLOGY_CHANGED,
    MycelialSubstrate,
)
from .nutrient_flow import FlowConfig, NutrientFlowEngine
from .circulatory_system import CirculatorySystem
from .topology import SubstrateNode, SubstrateTopology, TopologyType

__all__ = [
    # Substrate orchestrator
    "MycelialSubstrate",
    # Topology
    "SubstrateTopology",
    "SubstrateNode",
    "TopologyType",
    # Nutrient flow
    "NutrientFlowEngine",
    "FlowConfig",
    # EventBus channels
    "CH_AGENT_REGISTERED",
    "CH_AGENT_DEREGISTERED",
    "CH_TOPOLOGY_CHANGED",
    "CH_HEALTH_REPORT",
    "CH_STARVATION_ALERT",
    "CH_ISOLATION_DETECTED",
    # Circulatory
    "CirculatorySystem",
]
