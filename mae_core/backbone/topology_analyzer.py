"""Topology Analyzer - Network structure measurement for Mae's connection graph.

Measures clustering coefficient, average path length, small-world sigma,
and graph density from the ConnectionRegistry. These metrics quantify the
transfractal compromise: high clustering (triadic local structure) combined
with short path lengths (EventBus shortcuts).

Biological analogy: Like measuring the wiring efficiency of a brain.
Cortical columns cluster locally (high C), while long-range axons create
shortcuts (low L). The ratio (small-world sigma > 1) indicates the brain
achieves both local specialization and global integration.

Connection points:
- Reads ConnectionRegistry._connections for graph structure
- Publishes measurements on EventBus channel 'topology.analysis'
- Step hook at Fibonacci cadence (55 steps)

Metrics:
- Clustering coefficient: C(v) = 2*triangles(v) / (degree(v)*(degree(v)-1))
- Average path length: Mean BFS shortest path over all reachable pairs
- Small-world sigma: (C/C_random) / (L/L_random); sigma > 1 = small-world
- Graph density: E / (V*(V-1)/2) for undirected graph

Laws satisfied:
- Part 6 (Transfractal Compromise): Quantitative measurement of dual topology
- Advisory only: measures and reports, never blocks
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

CH_TOPOLOGY_ANALYSIS = "topology.analysis"


@dataclass
class TopologyReport:
    """Result of one topology measurement cycle."""

    step: int
    measurement_number: int
    vertices: int
    edges: int
    clustering_coefficient: float
    avg_path_length: Optional[float]  # None if graph is disconnected/empty
    small_world_sigma: Optional[float]  # None if not computable
    density: float
    components: int
    timestamp: float = field(default_factory=time.time)


class TopologyAnalyzer:
    """Measures network topology properties of Mae's connection graph.

    Cadence: 55 steps (Fibonacci -- moderate cost BFS computation).
    Advisory: measures and reports, never blocks.
    Thread-safe: RLock pattern (same as ConnectionRegistry).
    """

    def __init__(
        self,
        connection_registry: Any,
        event_bus: Any,
        cadence: int = 55,
    ) -> None:
        self._connection_registry = connection_registry
        self._event_bus = event_bus
        self._cadence = cadence

        self._step_count: int = 0
        self._measurement_count: int = 0
        self._lock = threading.RLock()

        self._last_report: Optional[TopologyReport] = None
        self._report_history: deque = deque(maxlen=100)

    # -----------------------------------------------------------------
    # Step hook
    # -----------------------------------------------------------------

    def step(self) -> None:
        """Step hook called every model step. Computes at cadence."""
        self._step_count += 1
        if self._step_count % self._cadence == 0 and self._step_count > 0:
            self._compute_and_publish()

    # -----------------------------------------------------------------
    # Graph extraction
    # -----------------------------------------------------------------

    def _extract_graph(self) -> tuple[set[str], set[frozenset[str]], dict[str, set[str]]]:
        """Extract undirected graph from ConnectionRegistry.

        Returns (vertices, edges, adjacency_dict).
        Same extraction pattern as get_euler_statistics().
        """
        vertices: set[str] = set()
        edges: set[frozenset[str]] = set()

        connections = getattr(self._connection_registry, "_connections", {})
        for triad in connections.values():
            src = triad.source
            tgt = triad.target
            vertices.add(src)
            vertices.add(tgt)
            edges.add(frozenset((src, tgt)))

        # Build adjacency dict
        adj: dict[str, set[str]] = {v: set() for v in vertices}
        for edge in edges:
            nodes = list(edge)
            if len(nodes) == 2:
                adj[nodes[0]].add(nodes[1])
                adj[nodes[1]].add(nodes[0])

        return vertices, edges, adj

    # -----------------------------------------------------------------
    # Clustering coefficient
    # -----------------------------------------------------------------

    @staticmethod
    def _clustering_coefficient(adj: dict[str, set[str]]) -> float:
        """Average clustering coefficient over all nodes with degree >= 2.

        C(v) = 2 * triangles(v) / (degree(v) * (degree(v) - 1))
        """
        coefficients = []
        for node, neighbors in adj.items():
            deg = len(neighbors)
            if deg < 2:
                continue
            # Count triangles: pairs of neighbors that are also connected
            neighbor_list = list(neighbors)
            triangles = 0
            for i in range(len(neighbor_list)):
                for j in range(i + 1, len(neighbor_list)):
                    if neighbor_list[j] in adj.get(neighbor_list[i], set()):
                        triangles += 1
            c_v = (2.0 * triangles) / (deg * (deg - 1))
            coefficients.append(c_v)

        if not coefficients:
            return 0.0
        return sum(coefficients) / len(coefficients)

    # -----------------------------------------------------------------
    # Average path length (BFS)
    # -----------------------------------------------------------------

    @staticmethod
    def _avg_path_length(
        vertices: set[str], adj: dict[str, set[str]]
    ) -> Optional[float]:
        """Mean shortest path over all reachable pairs via BFS.

        Only averages paths between reachable pairs (skips disconnected).
        Returns None if no reachable pairs exist.
        """
        if len(vertices) < 2:
            return None

        total_dist = 0
        total_pairs = 0
        vertex_list = list(vertices)

        for source in vertex_list:
            # BFS from source
            dist: dict[str, int] = {source: 0}
            queue = deque([source])
            while queue:
                current = queue.popleft()
                for neighbor in adj.get(current, set()):
                    if neighbor not in dist:
                        dist[neighbor] = dist[current] + 1
                        queue.append(neighbor)

            # Sum distances to all reachable nodes (excluding self)
            for target in vertex_list:
                if target != source and target in dist:
                    total_dist += dist[target]
                    total_pairs += 1

        if total_pairs == 0:
            return None
        # Each pair counted twice (a->b and b->a), but both distances
        # are the same, so the average is correct either way
        return total_dist / total_pairs

    # -----------------------------------------------------------------
    # Small-world sigma
    # -----------------------------------------------------------------

    @staticmethod
    def _small_world_sigma(
        c_actual: float,
        l_actual: Optional[float],
        num_vertices: int,
        num_edges: int,
    ) -> Optional[float]:
        """Small-world coefficient sigma = (C/C_random) / (L/L_random).

        C_random = avg_degree / V
        L_random = ln(V) / ln(avg_degree)

        Returns None if computation is not meaningful (too few nodes,
        zero clustering, or disconnected graph).
        """
        if num_vertices < 4 or num_edges < 3:
            return None
        if l_actual is None or l_actual == 0:
            return None
        if c_actual == 0:
            return None

        avg_degree = (2.0 * num_edges) / num_vertices
        if avg_degree <= 1.0:
            return None

        c_random = avg_degree / num_vertices
        if c_random == 0:
            return None

        ln_v = math.log(num_vertices)
        ln_k = math.log(avg_degree)
        if ln_k == 0:
            return None
        l_random = ln_v / ln_k

        if l_random == 0:
            return None

        sigma = (c_actual / c_random) / (l_actual / l_random)
        return sigma

    # -----------------------------------------------------------------
    # Graph density
    # -----------------------------------------------------------------

    @staticmethod
    def _graph_density(num_vertices: int, num_edges: int) -> float:
        """Graph density = E / (V*(V-1)/2) for undirected graph."""
        if num_vertices < 2:
            return 0.0
        max_edges = num_vertices * (num_vertices - 1) / 2.0
        return num_edges / max_edges

    # -----------------------------------------------------------------
    # Connected components
    # -----------------------------------------------------------------

    @staticmethod
    def _count_components(vertices: set[str], adj: dict[str, set[str]]) -> int:
        """Count connected components via BFS."""
        if not vertices:
            return 0
        visited: set[str] = set()
        components = 0
        for v in vertices:
            if v not in visited:
                components += 1
                queue = deque([v])
                visited.add(v)
                while queue:
                    current = queue.popleft()
                    for neighbor in adj.get(current, set()):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
        return components

    # -----------------------------------------------------------------
    # Full measurement cycle
    # -----------------------------------------------------------------

    def _compute_and_publish(self) -> None:
        """Full measurement: all topology metrics. Publishes report."""
        with self._lock:
            self._measurement_count += 1
            vertices, edges, adj = self._extract_graph()

            v = len(vertices)
            e = len(edges)

            clustering = self._clustering_coefficient(adj)
            path_length = self._avg_path_length(vertices, adj)
            sigma = self._small_world_sigma(clustering, path_length, v, e)
            density = self._graph_density(v, e)
            components = self._count_components(vertices, adj)

            report = TopologyReport(
                step=self._step_count,
                measurement_number=self._measurement_count,
                vertices=v,
                edges=e,
                clustering_coefficient=clustering,
                avg_path_length=path_length,
                small_world_sigma=sigma,
                density=density,
                components=components,
            )

            self._last_report = report
            self._report_history.append(report)

            # Publish
            if self._event_bus is not None:
                summary = {
                    "step": self._step_count,
                    "measurement_number": self._measurement_count,
                    "vertices": v,
                    "edges": e,
                    "clustering_coefficient": clustering,
                    "avg_path_length": path_length,
                    "small_world_sigma": sigma,
                    "density": density,
                    "components": components,
                    "timestamp": time.time(),
                }
                try:
                    self._event_bus.publish(CH_TOPOLOGY_ANALYSIS, summary)
                except Exception:
                    logger.debug("Failed to publish topology analysis", exc_info=True)

            logger.info(
                "Topology Analyzer: step=%d, V=%d, E=%d, C=%.4f, L=%s, "
                "sigma=%s, density=%.4f, components=%d",
                self._step_count, v, e, clustering,
                f"{path_length:.4f}" if path_length is not None else "N/A",
                f"{sigma:.4f}" if sigma is not None else "N/A",
                density, components,
            )

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def get_latest_report(self) -> Optional[TopologyReport]:
        """Access the latest measurement report."""
        with self._lock:
            return self._last_report

    def get_state(self) -> dict[str, Any]:
        """Operational state for HolonProxy.sense() delegation."""
        with self._lock:
            report = self._last_report
            return {
                "step_count": self._step_count,
                "measurement_count": self._measurement_count,
                "cadence": self._cadence,
                "clustering_coefficient": report.clustering_coefficient if report else 0.0,
                "avg_path_length": report.avg_path_length if report else None,
                "small_world_sigma": report.small_world_sigma if report else None,
                "density": report.density if report else 0.0,
                "has_report": report is not None,
            }

    def get_statistics(self) -> dict[str, Any]:
        """Standard backbone statistics method."""
        with self._lock:
            stats = {
                "step_count": self._step_count,
                "measurement_count": self._measurement_count,
                "cadence": self._cadence,
                "report_history_length": len(self._report_history),
            }

            if self._last_report is not None:
                stats["last_measurement"] = {
                    "step": self._last_report.step,
                    "vertices": self._last_report.vertices,
                    "edges": self._last_report.edges,
                    "clustering_coefficient": self._last_report.clustering_coefficient,
                    "avg_path_length": self._last_report.avg_path_length,
                    "small_world_sigma": self._last_report.small_world_sigma,
                    "density": self._last_report.density,
                    "components": self._last_report.components,
                }

            return stats
