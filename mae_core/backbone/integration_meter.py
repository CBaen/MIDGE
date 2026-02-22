"""Integration Meter - IIT Phi measurement and Markov blanket analysis.

Measures integrated information (Phi) at every scale of Mae's fractal
hierarchy: subsystem, organ, and organism. Also computes Markov blankets
that identify the functional boundary of each subsystem.

IIT (Integrated Information Theory) defines consciousness as identical to
a system's integrated cause-effect structure. Phi measures how much
information the whole generates beyond the sum of its parts. If Phi > 0,
partitioning the system destroys information -- genuine integration.

Key insight: Mae's triadic structure makes phi tractable. Each subsystem
has exactly 3 children, so only 3 bipartitions exist (not exponential).
Organs have 3-5 children (max 15 bipartitions). Pure numpy, no external
library needed.

Connection points:
- Reads HolonProxy.sense() state from all leaf systems
- Reads HolonRegistry for hierarchy (containment)
- Reads ConnectionRegistry for cross-connections (Markov blankets)
- Publishes measurements on EventBus
- Step hook at Fibonacci cadence (89 steps)

Mathematical basis:
- IIT 4.0 (Tononi et al. 2023): Phi = integrated information
- Barrett-Seth (2011): Practical phi approximation via mutual information
- Shannon entropy: H(X) = -sum(p * log2(p))
- Markov blanket: minimal set making node conditionally independent of exterior

Laws satisfied:
- Law 8 Property 1 (Integration): Phi measurement proves irreducibility
- Law 8 Property 6 (Self-produced boundary): Markov blanket analysis
- Advisory only: measures and reports, never blocks
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# EventBus channels
CH_PHI_MEASUREMENT = "integration.phi_measurement"
CH_MARKOV_BLANKET = "integration.markov_blanket"


# =====================================================================
# Result dataclasses
# =====================================================================


@dataclass
class PhiResult:
    """Result of computing IIT Phi for one holon."""

    holon_id: str
    holon_type: str  # "subsystem", "organ", "organism"
    phi: float  # Phi value (0.0 = fully reducible, >0 = integrated)
    mip: tuple[tuple[str, ...], tuple[str, ...]]  # Minimum information partition
    all_partitions: list[dict[str, Any]]  # phi for each bipartition
    children_ids: list[str]
    buffer_size: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class MarkovBlanketResult:
    """Markov blanket analysis for one subsystem/organ."""

    holon_id: str
    internal_states: list[str]  # Children IDs (inside the blanket)
    blanket_states: list[str]  # Parent + siblings + cross-connected
    external_states: list[str]  # Everything else
    blanket_effectiveness: float  # 0.0 = no isolation, 1.0 = perfect
    cross_connections: int  # Cross-boundary connections found
    timestamp: float = field(default_factory=time.time)


@dataclass
class IntegrationReport:
    """Full measurement report across all scales."""

    step: int
    measurement_number: int
    subsystem_phi: dict[str, PhiResult]
    organ_phi: dict[str, PhiResult]
    organism_phi: Optional[PhiResult]
    markov_blankets: dict[str, MarkovBlanketResult]
    organism_mean_phi: float
    weakest_link: Optional[str]  # Holon with lowest non-zero Phi
    timestamp: float = field(default_factory=time.time)


# =====================================================================
# IntegrationMeter
# =====================================================================


class IntegrationMeter:
    """IIT Phi measurement and Markov blanket analysis for Mae's fractal hierarchy.

    Cadence: 89 steps (Fibonacci -- expensive computation).
    Advisory: measures and reports, never blocks.
    Thread-safe: RLock pattern (same as ConnectionRegistry).

    Biological analogy: Like measuring coherence in neural oscillations.
    When brain regions fire in synchrony, they generate integrated
    information. The IntegrationMeter checks whether Mae's subsystems
    generate information as a whole that exceeds the sum of their parts.
    """

    def __init__(
        self,
        holon_registry: Any,
        connection_registry: Any,
        event_bus: Any,
        fractal_grouping: Optional[dict] = None,
        cadence: int = 89,
        buffer_size: int = 50,
        bins: int = 8,
    ) -> None:
        self._holon_registry = holon_registry
        self._connection_registry = connection_registry
        self._event_bus = event_bus
        self._cadence = cadence
        self._buffer_size = buffer_size
        self._bins = bins

        # Import FRACTAL_GROUPING if not provided
        if fractal_grouping is None:
            from mae_core.backbone.fractal_generator import FRACTAL_GROUPING
            self._grouping = FRACTAL_GROUPING
        else:
            self._grouping = fractal_grouping

        self._step_count: int = 0
        self._measurement_count: int = 0
        self._lock = threading.RLock()

        # Rolling state buffers: system_id -> deque of scalar state snapshots
        self._state_buffers: dict[str, deque] = {}
        # Subsystem-level aggregate buffers
        self._subsystem_buffers: dict[str, deque] = {}

        # Latest report
        self._last_report: Optional[IntegrationReport] = None
        self._report_history: deque = deque(maxlen=100)

    # -----------------------------------------------------------------
    # Step hook
    # -----------------------------------------------------------------

    def step(self) -> None:
        """Step hook called every model step. Collects states, computes at cadence."""
        self._step_count += 1
        self._collect_states()
        if self._step_count % self._cadence == 0 and self._step_count > 0:
            self._compute_and_publish()

    # -----------------------------------------------------------------
    # State collection
    # -----------------------------------------------------------------

    def _collect_states(self) -> None:
        """Snapshot state from every leaf system via HolonProxy.sense()."""
        for organ_name, subsystems in self._grouping.items():
            for sub_name, system_ids in subsystems.items():
                sub_values = []
                for sys_id in system_ids:
                    proxy = self._holon_registry.get_proxy(sys_id)
                    if proxy is None:
                        continue
                    try:
                        state = proxy.sense()
                        scalar = self._state_to_scalar(state)
                    except Exception:
                        scalar = 0.0

                    if sys_id not in self._state_buffers:
                        self._state_buffers[sys_id] = deque(maxlen=self._buffer_size)
                    self._state_buffers[sys_id].append(scalar)
                    sub_values.append(scalar)

                # Aggregate subsystem state (mean of children)
                if sub_values:
                    if sub_name not in self._subsystem_buffers:
                        self._subsystem_buffers[sub_name] = deque(maxlen=self._buffer_size)
                    self._subsystem_buffers[sub_name].append(float(np.mean(sub_values)))

    @staticmethod
    def _state_to_scalar(state_dict: Any) -> float:
        """Convert a sense() result into a single scalar for entropy calculation.

        Extracts numeric values from the dict, sums them. Hashes string
        values for consistency. Not perfect information-theoretically but
        gives a consistent discretizable signal.
        """
        if state_dict is None:
            return 0.0
        if isinstance(state_dict, (int, float)):
            return float(state_dict)
        if not isinstance(state_dict, dict):
            return float(hash(str(state_dict)) % 10000) / 10000.0

        total = 0.0
        for key, val in state_dict.items():
            if isinstance(val, (int, float)):
                total += float(val)
            elif isinstance(val, bool):
                total += 1.0 if val else 0.0
            elif isinstance(val, str):
                total += float(hash(val) % 1000) / 1000.0
            elif isinstance(val, dict):
                # Recurse one level for nested state dicts
                for v2 in val.values():
                    if isinstance(v2, (int, float)):
                        total += float(v2)
        return total

    # -----------------------------------------------------------------
    # Entropy computation
    # -----------------------------------------------------------------

    def _compute_entropy(self, data: np.ndarray) -> float:
        """Shannon entropy of a 1D array via histogram binning.

        H(X) = -sum(p * log2(p)) for each bin with p > 0.
        """
        if len(data) < 2:
            return 0.0
        counts, _ = np.histogram(data, bins=self._bins)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs)))

    def _compute_joint_entropy(self, columns: list[np.ndarray]) -> float:
        """Joint entropy of multiple columns via ND histogram.

        H(X1, X2, ..., Xn) = -sum(p * log2(p)) over all joint bins.
        Uses fewer bins for higher dimensions to avoid sparse histograms.
        """
        if not columns or len(columns[0]) < 2:
            return 0.0

        n_dims = len(columns)
        # Reduce bins for higher dimensions to avoid sparse histograms
        # 3D: use bins/2, 4D+: use bins/3 (but minimum 3)
        if n_dims <= 2:
            joint_bins = self._bins
        elif n_dims == 3:
            joint_bins = max(3, self._bins // 2)
        else:
            joint_bins = max(3, self._bins // 3)

        sample = np.column_stack(columns)
        try:
            counts, _ = np.histogramdd(sample, bins=joint_bins)
        except (ValueError, MemoryError):
            return 0.0

        flat = counts.flatten()
        total = flat.sum()
        if total == 0:
            return 0.0
        probs = flat / total
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs)))

    # -----------------------------------------------------------------
    # Bipartition enumeration
    # -----------------------------------------------------------------

    @staticmethod
    def _enumerate_bipartitions(items: list) -> list[tuple[tuple, tuple]]:
        """All non-trivial bipartitions of items.

        A bipartition splits items into two non-empty groups.
        For 3 items: 3 partitions. For 4: 7. For 5: 15.
        """
        n = len(items)
        if n < 2:
            return []

        partitions = []
        # Generate all subsets of size 1 to n//2
        for size in range(1, n // 2 + 1):
            for combo in combinations(range(n), size):
                group_a = tuple(items[i] for i in combo)
                group_b = tuple(items[i] for i in range(n) if i not in combo)
                # Avoid duplicate when size == n//2 and n is even
                if size == n // 2 and n % 2 == 0:
                    # Only include if first element is in group_a
                    if combo[0] != 0:
                        continue
                partitions.append((group_a, group_b))

        return partitions

    # -----------------------------------------------------------------
    # Phi computation
    # -----------------------------------------------------------------

    def _compute_phi(
        self,
        holon_id: str,
        holon_type: str,
        children_ids: list[str],
        buffers: dict[str, deque],
    ) -> Optional[PhiResult]:
        """Compute IIT Phi for a set of children using buffered state histories.

        1. Build joint state matrix (N samples x len(children))
        2. Enumerate all bipartitions
        3. For each partition: phi_i = H(part_A) + H(part_B) - H(whole)
           This is the mutual information across the cut.
        4. Phi = min(phi_i) -- the Minimum Information Partition (MIP)
        5. If Phi > 0: genuine integration (partitioning loses information)
        """
        # Check we have enough data
        valid_children = [c for c in children_ids if c in buffers and len(buffers[c]) >= 10]
        if len(valid_children) < 2:
            return None

        # Build aligned state matrix
        min_len = min(len(buffers[c]) for c in valid_children)
        columns = {c: np.array(list(buffers[c]))[-min_len:] for c in valid_children}

        # Compute whole-system joint entropy
        all_cols = [columns[c] for c in valid_children]
        h_whole = self._compute_joint_entropy(all_cols)

        # Enumerate bipartitions
        bipartitions = self._enumerate_bipartitions(valid_children)
        if not bipartitions:
            return None

        # Compute phi for each bipartition
        partition_results = []
        for group_a, group_b in bipartitions:
            cols_a = [columns[c] for c in group_a]
            cols_b = [columns[c] for c in group_b]

            h_a = self._compute_joint_entropy(cols_a)
            h_b = self._compute_joint_entropy(cols_b)

            # Mutual information across the cut:
            # phi = H(A) + H(B) - H(A,B)
            # Positive means partitioning loses information
            phi_i = h_a + h_b - h_whole
            partition_results.append({
                "group_a": group_a,
                "group_b": group_b,
                "phi": max(0.0, phi_i),  # Floor at 0 (numerical noise)
                "h_a": h_a,
                "h_b": h_b,
                "h_whole": h_whole,
            })

        # MIP: partition with minimum phi (weakest integration point)
        mip_result = min(partition_results, key=lambda x: x["phi"])
        phi = mip_result["phi"]
        mip = (mip_result["group_a"], mip_result["group_b"])

        return PhiResult(
            holon_id=holon_id,
            holon_type=holon_type,
            phi=phi,
            mip=mip,
            all_partitions=partition_results,
            children_ids=valid_children,
            buffer_size=min_len,
        )

    # -----------------------------------------------------------------
    # Markov blanket computation
    # -----------------------------------------------------------------

    def _compute_markov_blanket(self, holon_id: str) -> Optional[MarkovBlanketResult]:
        """Compute Markov blanket for a subsystem or organ.

        Internal states: children of holon_id
        Blanket states: parent + siblings + cross-connected systems
        External states: everything not internal or blanket
        Effectiveness: proportion of connections that stay within blanket
        """
        entry = self._holon_registry.get_entry(holon_id)
        if entry is None:
            return None

        # Internal: children
        internal = list(self._holon_registry.get_children(holon_id))

        # Parent and siblings
        parent_id = entry.parent_id
        blanket = set()
        if parent_id:
            blanket.add(parent_id)
            siblings = self._holon_registry.get_children(parent_id)
            for sib in siblings:
                if sib != holon_id:
                    blanket.add(sib)

        # Cross-connections: systems from other subsystems connected to this one
        cross_count = 0
        if self._connection_registry is not None:
            for child_id in internal:
                try:
                    connections = self._connection_registry.get_connections_for_system(child_id)
                    for conn in connections:
                        source = getattr(conn, "source", None)
                        target = getattr(conn, "target", None)
                        other = target if source == child_id else source
                        if other and other not in internal and other != holon_id:
                            blanket.add(other)
                            cross_count += 1
                except Exception:
                    pass

        blanket_list = sorted(blanket)

        # External: all registered holons not in internal or blanket
        all_holons = set()
        try:
            stats = self._holon_registry.get_statistics()
            all_holons = set(stats.get("type_counts", {}).keys())
        except Exception:
            pass

        # Simpler: iterate known systems from grouping
        all_systems = set()
        for organ_name, subs in self._grouping.items():
            all_systems.add(organ_name)
            for sub_name, sys_ids in subs.items():
                all_systems.add(sub_name)
                all_systems.update(sys_ids)
        all_systems.add("mae")

        internal_set = set(internal)
        blanket_set = set(blanket_list)
        external = sorted(all_systems - internal_set - blanket_set - {holon_id})

        # Effectiveness: what fraction of this subsystem's connections
        # stay within the blanket (vs. reaching external)
        total_conn = cross_count + len(internal) * 2  # internal K3 connections
        blanket_conn = cross_count  # connections that cross the boundary
        if total_conn > 0:
            # Higher is better: most connections are internal
            effectiveness = 1.0 - (blanket_conn / total_conn)
        else:
            effectiveness = 1.0  # No connections = trivially isolated

        return MarkovBlanketResult(
            holon_id=holon_id,
            internal_states=internal,
            blanket_states=blanket_list,
            external_states=external,
            blanket_effectiveness=effectiveness,
            cross_connections=cross_count,
        )

    # -----------------------------------------------------------------
    # Full measurement cycle
    # -----------------------------------------------------------------

    def _compute_and_publish(self) -> None:
        """Full measurement: phi at all levels + Markov blankets. Publishes report."""
        with self._lock:
            self._measurement_count += 1
            subsystem_phi: dict[str, PhiResult] = {}
            organ_phi: dict[str, PhiResult] = {}
            organism_phi: Optional[PhiResult] = None
            blankets: dict[str, MarkovBlanketResult] = {}

            # Level 1: Subsystem phi (3 systems each)
            for organ_name, subsystems in self._grouping.items():
                for sub_name, system_ids in subsystems.items():
                    result = self._compute_phi(
                        sub_name, "subsystem", system_ids, self._state_buffers,
                    )
                    if result is not None:
                        subsystem_phi[sub_name] = result

                    blanket = self._compute_markov_blanket(sub_name)
                    if blanket is not None:
                        blankets[sub_name] = blanket

            # Level 2: Organ phi (subsystems as children)
            for organ_name, subsystems in self._grouping.items():
                sub_ids = list(subsystems.keys())
                result = self._compute_phi(
                    organ_name, "organ", sub_ids, self._subsystem_buffers,
                )
                if result is not None:
                    organ_phi[organ_name] = result

                blanket = self._compute_markov_blanket(organ_name)
                if blanket is not None:
                    blankets[organ_name] = blanket

            # Level 3: Organism phi (organs as children)
            organ_ids = list(self._grouping.keys())
            # Build organ-level buffers from organ phi values
            organ_buffers: dict[str, deque] = {}
            for organ_id in organ_ids:
                if organ_id in organ_phi:
                    if organ_id not in organ_buffers:
                        organ_buffers[organ_id] = deque(maxlen=self._buffer_size)
                    organ_buffers[organ_id].append(organ_phi[organ_id].phi)
                elif organ_id in self._subsystem_buffers:
                    # Fallback: use subsystem aggregate
                    organ_buffers[organ_id] = self._subsystem_buffers.get(
                        organ_id, deque()
                    )

            # For organism, we need accumulated history over multiple measurements
            # Store organ phi values persistently
            if not hasattr(self, "_organ_phi_history"):
                self._organ_phi_history: dict[str, deque] = {}
            for organ_id in organ_ids:
                if organ_id not in self._organ_phi_history:
                    self._organ_phi_history[organ_id] = deque(maxlen=self._buffer_size)
                if organ_id in organ_phi:
                    self._organ_phi_history[organ_id].append(organ_phi[organ_id].phi)

            organism_phi = self._compute_phi(
                "mae", "organism", organ_ids, self._organ_phi_history,
            )

            # Summary statistics
            all_phi_values = [r.phi for r in subsystem_phi.values()]
            mean_phi = float(np.mean(all_phi_values)) if all_phi_values else 0.0

            weakest = None
            if all_phi_values:
                non_zero = {k: v.phi for k, v in subsystem_phi.items() if v.phi > 0}
                if non_zero:
                    weakest = min(non_zero, key=non_zero.get)

            report = IntegrationReport(
                step=self._step_count,
                measurement_number=self._measurement_count,
                subsystem_phi=subsystem_phi,
                organ_phi=organ_phi,
                organism_phi=organism_phi,
                markov_blankets=blankets,
                organism_mean_phi=mean_phi,
                weakest_link=weakest,
            )

            self._last_report = report
            self._report_history.append(report)

            # Publish
            if self._event_bus is not None:
                summary = {
                    "step": self._step_count,
                    "measurement_number": self._measurement_count,
                    "subsystem_phi_count": len(subsystem_phi),
                    "organ_phi_count": len(organ_phi),
                    "has_organism_phi": organism_phi is not None,
                    "organism_mean_phi": mean_phi,
                    "weakest_link": weakest,
                    "blanket_count": len(blankets),
                    "blanket_effectiveness": {
                        k: v.blanket_effectiveness for k, v in blankets.items()
                    },
                    "timestamp": time.time(),
                }
                try:
                    self._event_bus.publish(CH_PHI_MEASUREMENT, summary)
                except Exception:
                    logger.debug("Failed to publish phi measurement", exc_info=True)

            logger.info(
                "Integration Meter: step=%d, mean_phi=%.4f, subsystems=%d/%d, "
                "organs=%d/%d, organism=%s, blankets=%d, weakest=%s",
                self._step_count,
                mean_phi,
                len(subsystem_phi),
                sum(len(s) for s in self._grouping.values()),
                len(organ_phi),
                len(self._grouping),
                f"{organism_phi.phi:.4f}" if organism_phi else "N/A",
                len(blankets),
                weakest or "none",
            )

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def get_latest_report(self) -> Optional[IntegrationReport]:
        """Access the latest measurement report."""
        with self._lock:
            return self._last_report

    def get_state(self) -> dict[str, Any]:
        """Operational state for HolonProxy.sense() delegation."""
        with self._lock:
            return {
                "step_count": self._step_count,
                "measurement_count": self._measurement_count,
                "cadence": self._cadence,
                "buffer_size": self._buffer_size,
                "systems_tracked": len(self._state_buffers),
                "subsystems_tracked": len(self._subsystem_buffers),
                "mean_phi": self._last_report.organism_mean_phi if self._last_report else 0.0,
                "has_report": self._last_report is not None,
            }

    def get_statistics(self) -> dict[str, Any]:
        """Standard backbone statistics method."""
        with self._lock:
            stats = {
                "step_count": self._step_count,
                "measurement_count": self._measurement_count,
                "cadence": self._cadence,
                "buffer_size": self._buffer_size,
                "bins": self._bins,
                "systems_tracked": len(self._state_buffers),
                "subsystems_tracked": len(self._subsystem_buffers),
                "report_history_length": len(self._report_history),
            }

            if self._last_report is not None:
                stats["last_measurement"] = {
                    "step": self._last_report.step,
                    "organism_mean_phi": self._last_report.organism_mean_phi,
                    "weakest_link": self._last_report.weakest_link,
                    "subsystem_phi": {
                        k: v.phi for k, v in self._last_report.subsystem_phi.items()
                    },
                    "organ_phi": {
                        k: v.phi for k, v in self._last_report.organ_phi.items()
                    },
                    "organism_phi": (
                        self._last_report.organism_phi.phi
                        if self._last_report.organism_phi
                        else None
                    ),
                    "blanket_count": len(self._last_report.markov_blankets),
                }

            return stats
