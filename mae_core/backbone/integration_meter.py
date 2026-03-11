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
from typing import Any, Optional

import numpy as np

from mae_core.backbone.integration_meter_blanket import compute_markov_blanket  # noqa: F401
from mae_core.backbone.integration_meter_models import (  # noqa: F401
    IntegrationReport,
    MarkovBlanketResult,
    PhiResult,
)
from mae_core.backbone.integration_meter_phi import compute_phi  # noqa: F401

logger = logging.getLogger(__name__)

# EventBus channels
CH_PHI_MEASUREMENT = "integration.phi_measurement"
CH_MARKOV_BLANKET = "integration.markov_blanket"


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
        # Organ phi history for organism-level measurement
        self._organ_phi_history: dict[str, deque] = {}

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
                    result = compute_phi(
                        sub_name, "subsystem", system_ids,
                        self._state_buffers, bins=self._bins,
                    )
                    if result is not None:
                        subsystem_phi[sub_name] = result

                    blanket = compute_markov_blanket(
                        sub_name,
                        self._holon_registry,
                        self._connection_registry,
                        self._grouping,
                    )
                    if blanket is not None:
                        blankets[sub_name] = blanket

            # Level 2: Organ phi (subsystems as children)
            for organ_name, subsystems in self._grouping.items():
                sub_ids = list(subsystems.keys())
                result = compute_phi(
                    organ_name, "organ", sub_ids,
                    self._subsystem_buffers, bins=self._bins,
                )
                if result is not None:
                    organ_phi[organ_name] = result

                blanket = compute_markov_blanket(
                    organ_name,
                    self._holon_registry,
                    self._connection_registry,
                    self._grouping,
                )
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
            for organ_id in organ_ids:
                if organ_id not in self._organ_phi_history:
                    self._organ_phi_history[organ_id] = deque(maxlen=self._buffer_size)
                if organ_id in organ_phi:
                    self._organ_phi_history[organ_id].append(organ_phi[organ_id].phi)

            organism_phi = compute_phi(
                "mae", "organism", organ_ids,
                self._organ_phi_history, bins=self._bins,
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
