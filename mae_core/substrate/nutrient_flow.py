"""Nutrient flow through the substrate - resource distribution.

Biological analogy: Mycelial hyphae transport nutrients via osmotic
pressure gradients. Resources flow from high-concentration regions to
low-concentration regions. Flow rate depends on hyphal conductance
(health of the connection). Starving regions signal for help.

Real biology: Mycorrhizal networks redistribute nutrients between
trees - "mother trees" with excess feed struggling seedlings.
This same principle applies to Mae's agents: resource-rich agents
feed resource-poor neighbors through substrate connections.

Hormonal modulation: Cortisol increases flow urgency (faster
redistribution under stress). Serotonin stabilizes flow
(prevents oscillation in healthy states).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from mae_core.substrate.topology import SubstrateTopology

logger = logging.getLogger(__name__)


@dataclass
class FlowConfig:
    """Configuration for nutrient flow dynamics."""

    base_flow_rate: float = 0.1  # Fraction of gradient transferred per step
    decay_rate: float = 0.01  # Natural resource consumption per step per node
    min_resource: float = 0.0  # Floor for resource level
    max_resource: float = 1.0  # Ceiling for resource level
    starvation_threshold: float = 0.15  # Below this, node is "starving"
    abundance_threshold: float = 0.8  # Above this, node has surplus
    conductance_decay: float = 0.005  # Unused edges lose conductance
    conductance_recovery: float = 0.01  # Active edges gain conductance
    min_conductance: float = 0.1  # Minimum edge health
    max_conductance: float = 1.0  # Maximum edge health


class NutrientFlowEngine:
    """Distributes resources across the substrate topology.

    Resources flow along edges from high to low concentration.
    Flow rate = base_flow_rate * gradient * conductance.
    Conductance strengthens on active edges (Hebbian: use it or lose it).
    """

    def __init__(
        self,
        topology: SubstrateTopology,
        config: Optional[FlowConfig] = None,
    ) -> None:
        self.topology = topology
        self.config = config or FlowConfig()

        # Track flow statistics
        self._total_flow: float = 0.0
        self._flow_steps: int = 0
        self._flow_events: list[tuple[str, str, float]] = []  # recent (from, to, amount)

    def flow_step(self) -> dict[str, float]:
        """Execute one timestep of resource diffusion.

        For each edge, resources flow from high to low. The amount
        transferred is proportional to the gradient and the edge
        conductance. This is a discrete approximation of diffusion.

        Returns:
            Dict of node_id -> net resource change this step.
        """
        nodes = self.topology.get_nodes()
        deltas: dict[str, float] = {nid: 0.0 for nid in nodes}
        self._flow_events.clear()

        for node_id, node in nodes.items():
            neighbors = self.topology.get_neighbors(node_id)
            for neighbor_id in neighbors:
                neighbor = nodes.get(neighbor_id)
                if neighbor is None:
                    continue

                # Gradient: positive means flow FROM node TO neighbor
                gradient = node.resource_level - neighbor.resource_level
                if gradient <= 0:
                    continue  # Only flow downhill

                # Flow amount
                conductance = node.conductance.get(neighbor_id, 1.0)
                flow = self.config.base_flow_rate * gradient * conductance

                # Don't drain below minimum
                flow = min(flow, node.resource_level - self.config.min_resource)
                if flow <= 0:
                    continue

                deltas[node_id] -= flow
                deltas[neighbor_id] += flow
                self._total_flow += flow

                # Strengthen active edge (Hebbian: used connections grow)
                self._strengthen_edge(node_id, neighbor_id)

                if flow > 0.01:
                    self._flow_events.append((node_id, neighbor_id, flow))

        # Apply deltas and natural decay
        for node_id, node in nodes.items():
            node.resource_level += deltas[node_id]
            node.resource_level -= self.config.decay_rate  # Natural consumption
            node.resource_level = max(
                self.config.min_resource,
                min(self.config.max_resource, node.resource_level),
            )

        # Decay unused edges
        self._decay_inactive_edges(nodes)

        self._flow_steps += 1
        return deltas

    def inject_resources(self, node_id: str, amount: float) -> bool:
        """Inject resources into a node (external input).

        Like sunlight reaching a tree or data arriving at an agent.
        """
        node = self.topology.get_node(node_id)
        if node is None:
            return False

        node.resource_level = min(
            self.config.max_resource, node.resource_level + amount
        )
        return True

    def consume_resources(self, node_id: str, amount: float) -> float:
        """Consume resources from a node (agent activity).

        Returns the actual amount consumed (may be less than requested
        if node is low on resources).
        """
        node = self.topology.get_node(node_id)
        if node is None:
            return 0.0

        available = max(0.0, node.resource_level - self.config.min_resource)
        consumed = min(amount, available)
        node.resource_level -= consumed
        return consumed

    def get_resource_level(self, node_id: str) -> float:
        """Get current resource level of a node."""
        node = self.topology.get_node(node_id)
        return node.resource_level if node else 0.0

    def get_starving_nodes(self) -> list[str]:
        """Get nodes below starvation threshold."""
        return [
            nid
            for nid, node in self.topology.get_nodes().items()
            if node.resource_level < self.config.starvation_threshold
        ]

    def get_abundant_nodes(self) -> list[str]:
        """Get nodes above abundance threshold (can share)."""
        return [
            nid
            for nid, node in self.topology.get_nodes().items()
            if node.resource_level > self.config.abundance_threshold
        ]

    def get_resource_gradient(self, node_id: str) -> dict[str, float]:
        """Get resource gradients to each neighbor.

        Positive = neighbor has more (resources flow toward this node).
        Negative = this node has more (resources flow away).
        """
        node = self.topology.get_node(node_id)
        if node is None:
            return {}

        gradients = {}
        for neighbor_id in self.topology.get_neighbors(node_id):
            neighbor = self.topology.get_node(neighbor_id)
            if neighbor:
                gradients[neighbor_id] = neighbor.resource_level - node.resource_level
        return gradients

    def get_total_resources(self) -> float:
        """Total resources across all nodes."""
        return sum(n.resource_level for n in self.topology.get_nodes().values())

    def get_resource_stats(self) -> dict[str, float]:
        """Resource distribution statistics."""
        nodes = self.topology.get_nodes()
        if not nodes:
            return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0, "total": 0.0}

        levels = [n.resource_level for n in nodes.values()]
        mean = sum(levels) / len(levels)
        variance = sum((l - mean) ** 2 for l in levels) / len(levels)

        return {
            "mean": mean,
            "min": min(levels),
            "max": max(levels),
            "std": variance**0.5,
            "total": sum(levels),
            "starving_count": len(self.get_starving_nodes()),
            "abundant_count": len(self.get_abundant_nodes()),
            "total_flow": self._total_flow,
            "flow_steps": self._flow_steps,
        }

    # =========================================================================
    # Hormonal Modulation Interface
    # =========================================================================

    def modulate_flow_rate(self, multiplier: float) -> None:
        """Modulate the base flow rate (hormonal influence).

        Cortisol increases urgency -> faster flow.
        Serotonin stabilizes -> normal flow.
        """
        self.config.base_flow_rate = max(0.01, self.config.base_flow_rate * multiplier)

    def modulate_decay_rate(self, multiplier: float) -> None:
        """Modulate natural decay rate (circadian influence).

        REST phase -> lower decay (conservation).
        ACTIVE phase -> normal decay.
        """
        self.config.decay_rate = max(0.0, self.config.decay_rate * multiplier)

    # =========================================================================
    # Internal
    # =========================================================================

    def _strengthen_edge(self, node_a: str, node_b: str) -> None:
        """Strengthen an active edge (Hebbian learning)."""
        for nid, neighbor_id in [(node_a, node_b), (node_b, node_a)]:
            node = self.topology.get_node(nid)
            if node:
                current = node.conductance.get(neighbor_id, 1.0)
                node.conductance[neighbor_id] = min(
                    self.config.max_conductance,
                    current + self.config.conductance_recovery,
                )

    def _decay_inactive_edges(self, nodes: dict) -> None:
        """Decay edges that weren't used this step (use it or lose it)."""
        active_edges = {(e[0], e[1]) for e in self._flow_events}
        active_edges.update({(e[1], e[0]) for e in self._flow_events})

        for node_id, node in nodes.items():
            for neighbor_id in list(node.conductance.keys()):
                if (node_id, neighbor_id) not in active_edges:
                    node.conductance[neighbor_id] = max(
                        self.config.min_conductance,
                        node.conductance[neighbor_id] - self.config.conductance_decay,
                    )
