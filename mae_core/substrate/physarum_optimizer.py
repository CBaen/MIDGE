"""PhysarumOptimizer - Slime mold adaptive topology optimization.

Biological analogy: Physarum polycephalum (slime mold) solves shortest-path
and network optimization problems by reinforcing high-flow edges and
retracting low-flow ones. The organism forms an efficient transport network
that converges on near-optimal solutions (Tero et al., Science 2010).

Key equation: dD/dt = f(|Q|) - gamma * D
  D = edge conductance (tube thickness)
  Q = flow through edge (from NutrientFlowEngine)
  f(|Q|) = |Q|^mu (flow sensitivity, mu typically 1.0)
  gamma = natural decay rate

This replaces the simple Hebbian strengthening in NutrientFlowEngine
with biologically-accurate Physarum dynamics. Runs as a post-step
optimization pass on the substrate topology.

EventBus channels:
- substrate.topology_optimized: Optimization pass completed
- substrate.edge_pruned: Edge removed due to low conductance
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from mae_core.backbone.event_bus import EventBus
from mae_core.substrate.mycelial_substrate import MycelialSubstrate

logger = logging.getLogger(__name__)

CH_TOPOLOGY_OPTIMIZED = "substrate.topology_optimized"
CH_EDGE_PRUNED = "substrate.edge_pruned"


@dataclass
class PhysarumConfig:
    """Configuration for Physarum optimization dynamics."""

    mu: float = 1.0            # Flow sensitivity exponent: f(Q) = |Q|^mu
    gamma: float = 0.01        # Conductance decay rate per step
    dt: float = 0.1            # Discrete time step for conductance update
    prune_threshold: float = 0.05  # Prune edges below this conductance
    min_edges_per_node: int = 1    # Never prune below this many connections
    max_conductance: float = 2.0   # Cap on conductance growth
    enabled: bool = True           # Can be disabled during REST phase


class PhysarumOptimizer:
    """Slime mold optimizer for substrate topology.

    Runs after each substrate step. Reads flow data from
    NutrientFlowEngine and adjusts edge conductances using
    the Physarum dynamics equation. Prunes starved edges.

    Integration points:
    - MycelialSubstrate: reads topology + nutrient flow
    - NutrientFlowEngine: reads _flow_events for flow magnitudes
    - CircadianRhythm: disabled during REST phase (conservation)
    - SomaticMap: registered for body awareness
    """

    def __init__(
        self,
        substrate: MycelialSubstrate,
        event_bus: Optional[EventBus] = None,
        config: Optional[PhysarumConfig] = None,
    ) -> None:
        self._substrate = substrate
        self._bus = event_bus
        self._config = config or PhysarumConfig()

        # Statistics
        self._optimization_steps: int = 0
        self._total_pruned: int = 0
        self._total_reinforced: int = 0
        self._last_avg_conductance: float = 1.0

        logger.info(
            "PhysarumOptimizer initialized (mu=%.2f, gamma=%.3f, prune=%.3f)",
            self._config.mu,
            self._config.gamma,
            self._config.prune_threshold,
        )

    def step(self) -> None:
        """Execute one Physarum optimization pass.

        1. Read flow events from NutrientFlowEngine
        2. Build flow magnitude map per edge
        3. Update conductances: D += (f(|Q|) - gamma * D) * dt
        4. Prune edges below threshold
        5. Publish optimization event
        """
        if not self._config.enabled:
            return

        flow_engine = self._substrate.nutrient_flow
        topology = self._substrate.topology

        # 1. Build flow magnitude map from recent flow events
        flow_map: dict[tuple[str, str], float] = {}
        for from_id, to_id, amount in flow_engine._flow_events:
            edge_key = (from_id, to_id)
            flow_map[edge_key] = flow_map.get(edge_key, 0.0) + amount
            # Symmetric: flow in either direction strengthens
            reverse_key = (to_id, from_id)
            flow_map[reverse_key] = flow_map.get(reverse_key, 0.0) + amount

        # 2. Update conductances for all edges using Physarum equation
        nodes = topology.get_nodes()
        reinforced = 0
        conductance_sum = 0.0
        conductance_count = 0

        for node_id, node in nodes.items():
            for neighbor_id in list(node.conductance.keys()):
                current_d = node.conductance.get(neighbor_id, 1.0)

                # Flow through this edge
                q = flow_map.get((node_id, neighbor_id), 0.0)

                # Physarum equation: dD/dt = f(|Q|) - gamma * D
                f_q = abs(q) ** self._config.mu
                dd_dt = f_q - self._config.gamma * current_d

                # Update conductance
                new_d = current_d + dd_dt * self._config.dt
                new_d = max(0.0, min(self._config.max_conductance, new_d))

                node.conductance[neighbor_id] = new_d

                if dd_dt > 0:
                    reinforced += 1

                conductance_sum += new_d
                conductance_count += 1

        self._total_reinforced += reinforced

        # 3. Track average conductance
        if conductance_count > 0:
            self._last_avg_conductance = conductance_sum / conductance_count

        # 4. Prune edges below threshold
        pruned = self._prune_weak_edges(topology, nodes)
        self._total_pruned += pruned

        # 5. Publish optimization event
        self._optimization_steps += 1
        if self._bus is not None:
            self._bus.publish(CH_TOPOLOGY_OPTIMIZED, {
                "step": self._optimization_steps,
                "reinforced": reinforced,
                "pruned": pruned,
                "avg_conductance": self._last_avg_conductance,
            })

    def _prune_weak_edges(self, topology, nodes: dict) -> int:
        """Remove edges with conductance below threshold.

        Respects min_edges_per_node to prevent node isolation.
        Like Physarum retracting pseudopods from dead ends.
        """
        pruned = 0
        edges_to_remove: list[tuple[str, str]] = []

        for node_id, node in nodes.items():
            neighbor_count = len(topology.get_neighbors(node_id))

            for neighbor_id, conductance in list(node.conductance.items()):
                if conductance < self._config.prune_threshold:
                    # Don't prune if it would leave the node with too few edges
                    if neighbor_count <= self._config.min_edges_per_node:
                        continue
                    edges_to_remove.append((node_id, neighbor_id))
                    neighbor_count -= 1

        # Remove edges (deduplicate since edges are bidirectional)
        removed: set[tuple[str, str]] = set()
        for node_a, node_b in edges_to_remove:
            edge_key = tuple(sorted([node_a, node_b]))
            if edge_key not in removed:
                topology.remove_edge(node_a, node_b)
                removed.add(edge_key)
                pruned += 1

                if self._bus is not None:
                    self._bus.publish(CH_EDGE_PRUNED, {
                        "from": node_a,
                        "to": node_b,
                        "step": self._optimization_steps,
                    })

        return pruned

    # =========================================================================
    # Circadian Integration
    # =========================================================================

    def set_phase(self, phase: str) -> None:
        """Modulate optimization based on circadian phase.

        ACTIVE: Full optimization (exploration + pruning).
        CONSOLIDATION: Stronger reinforcement, less pruning (solidify good paths).
        REST: Disabled (conservation mode).
        """
        if phase == "REST":
            self._config.enabled = False
        elif phase == "CONSOLIDATION":
            self._config.enabled = True
            # During consolidation, strengthen existing paths more
            self._config.gamma *= 0.5  # Less decay
        else:  # ACTIVE
            self._config.enabled = True

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_statistics(self) -> dict[str, Any]:
        return {
            "optimization_steps": self._optimization_steps,
            "total_pruned": self._total_pruned,
            "total_reinforced": self._total_reinforced,
            "avg_conductance": self._last_avg_conductance,
            "enabled": self._config.enabled,
            "config": {
                "mu": self._config.mu,
                "gamma": self._config.gamma,
                "prune_threshold": self._config.prune_threshold,
            },
        }
