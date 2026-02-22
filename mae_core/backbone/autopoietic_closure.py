"""Autopoietic Closure - Circular production at every scale (Law 6).

Components produce the processes that produce the components.
This closes the autopoietic loop at subsystem, organ, and organism levels.

At the agent level, MitosisMonitor handles closure: agent processes produce
new agents. At higher scales:

- Subsystem closure: processes (systems) are monitored; if they degrade,
  the subsystem can restructure them (reconfig or reinitialize).
- Organ closure: subsystems are monitored; if they degrade, the organ
  can redistribute workload or trigger healing.
- Organism closure: organs are monitored; if they degrade, the organism
  can rebalance resources or trigger system-wide healing.

Cadenced every Fibonacci interval (3, 5, 8, 13, 21, 89 steps).
Advisory only — observes, reports, never blocks.
Law 6: Autopoietic Closure + Law 3: Holon Protocol.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from mae_core.backbone.event_bus import EventBus
from mae_core.backbone.holon_protocol import HolonRegistry

logger = logging.getLogger(__name__)

# EventBus channels
CH_SUBSYSTEM_CLOSURE = "closure.subsystem"
CH_ORGAN_CLOSURE = "closure.organ"
CH_ORGANISM_CLOSURE = "closure.organism"

# Thresholds (advisories — trigger observation and reporting)
HEALTH_THRESHOLD_SUBSYSTEM = 0.5   # Below this: subsystem degradation detected
HEALTH_THRESHOLD_ORGAN = 0.4       # Below this: organ degradation detected
HEALTH_THRESHOLD_ORGANISM = 0.3    # Below this: organism-level intervention advised

# Cadence (steps)
CADENCE_SUBSYSTEM = 5      # Check subsystem health every 5 steps
CADENCE_ORGAN = 8          # Check organ health every 8 steps
CADENCE_ORGANISM = 13      # Check organism health every 13 steps


# =====================================================================
# Data records for closure reports
# =====================================================================


@dataclass
class ClosureReport:
    """Report from a single closure cycle (observation & advice)."""

    scale: str  # "subsystem", "organ", "organism"
    monitored_id: str
    timestamp: float = field(default_factory=time.time)
    healthy_children: int = 0
    degraded_children: int = 0
    children_healed: int = 0
    restructure_advice: str = ""  # Non-empty if action recommended
    metadata: dict[str, Any] = field(default_factory=dict)


# =====================================================================
# Base monitor (shared pattern across scales)
# =====================================================================


class AutopoieticMonitor:
    """Advisory monitor for circular production at any scale.

    Observes: Are components (children) healthy?
    Reports: Which are degraded? What healing/restructuring is advised?
    Acts: Calls heal() on degraded children (advisory, non-blocking).

    The circular closure: components produce processes → processes produce
    components. At each level, the monitor ensures the loop stays active.
    """

    def __init__(
        self,
        holon_registry: HolonRegistry,
        event_bus: Optional[EventBus] = None,
        health_threshold: float = 0.5,
        cadence: int = 5,
    ) -> None:
        self._registry = holon_registry
        self._bus = event_bus
        self._threshold = health_threshold
        self._cadence = cadence
        self._last_check_step: int = 0
        self._cycle_count: int = 0
        self._reports: list[ClosureReport] = []

    def step(self, current_step: int, monitored_id: str, scale: str) -> Optional[ClosureReport]:
        """Run closure check. Called from model step hook or parent monitor.

        Args:
            current_step: Current simulation step
            monitored_id: ID of the holon to monitor (subsystem/organ/organism)
            scale: "subsystem", "organ", or "organism"

        Returns:
            ClosureReport if check ran, None if skipped (cadence).
        """
        if current_step % self._cadence != 0 or current_step == 0:
            return None
        if current_step == self._last_check_step:
            return None

        self._last_check_step = current_step
        return self._monitor_cycle(monitored_id, scale, current_step)

    def _monitor_cycle(
        self, monitored_id: str, scale: str, step: int
    ) -> ClosureReport:
        """Execute one closure monitoring cycle.

        1. Get children
        2. Check health of each child
        3. Identify degraded ones
        4. Call heal() on degraded children (advisory)
        5. Report findings
        """
        report = ClosureReport(scale=scale, monitored_id=monitored_id)

        children = self._registry.get_children(monitored_id)
        if not children:
            self._publish_report(report)
            return report

        for child_id in children:
            proxy = self._registry.get_proxy(child_id)
            health = proxy.get_health()

            if health >= self._threshold:
                report.healthy_children += 1
            else:
                report.degraded_children += 1
                # Advisory: call heal() (never blocks)
                try:
                    proxy.heal()
                    report.children_healed += 1
                except Exception:
                    pass

        # Determine if restructuring is advised
        if report.degraded_children > 0:
            ratio = report.degraded_children / len(children)
            if ratio >= 0.6:
                report.restructure_advice = (
                    f"Major degradation ({ratio:.1%}): consider restructuring"
                )
            elif ratio >= 0.3:
                report.restructure_advice = (
                    f"Moderate degradation ({ratio:.1%}): monitor closely"
                )
            else:
                report.restructure_advice = "Minor degradation: healing in progress"

        report.metadata = {
            "step": step,
            "children_count": len(children),
            "healthy_ratio": (
                report.healthy_children / len(children)
                if children
                else 0.0
            ),
        }

        self._cycle_count += 1
        self._reports.append(report)
        # Keep rolling window of 100 reports
        if len(self._reports) > 100:
            self._reports = self._reports[-100:]

        self._publish_report(report)
        return report

    def _publish_report(self, report: ClosureReport) -> None:
        """Publish closure report on EventBus."""
        if self._bus is None:
            return

        channel = {
            "subsystem": CH_SUBSYSTEM_CLOSURE,
            "organ": CH_ORGAN_CLOSURE,
            "organism": CH_ORGANISM_CLOSURE,
        }.get(report.scale, CH_SUBSYSTEM_CLOSURE)

        try:
            self._bus.publish(channel, {
                "monitored_id": report.monitored_id,
                "healthy": report.healthy_children,
                "degraded": report.degraded_children,
                "healed": report.children_healed,
                "advice": report.restructure_advice,
                "metadata": report.metadata,
                "timestamp": report.timestamp,
            })
        except Exception:
            pass

    def get_statistics(self) -> dict[str, Any]:
        """Closure monitor statistics."""
        total_degraded = sum(r.degraded_children for r in self._reports)
        total_healed = sum(r.children_healed for r in self._reports)
        return {
            "cycles": self._cycle_count,
            "total_degraded_observed": total_degraded,
            "total_healed": total_healed,
            "recent_reports": [
                {
                    "monitored": r.monitored_id,
                    "healthy": r.healthy_children,
                    "degraded": r.degraded_children,
                    "advice": r.restructure_advice,
                }
                for r in self._reports[-10:]
            ],
        }


# =====================================================================
# Subsystem-level closure
# =====================================================================


class SubsystemClosure:
    """Monitor and advise on subsystem-level circular production.

    A subsystem's processes (child systems) maintain the subsystem,
    and the subsystem maintains its processes. This monitor ensures
    the loop stays healthy.

    Checks every CADENCE_SUBSYSTEM steps (5, Fibonacci).
    """

    def __init__(
        self,
        holon_registry: HolonRegistry,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        self._monitor = AutopoieticMonitor(
            holon_registry=holon_registry,
            event_bus=event_bus,
            health_threshold=HEALTH_THRESHOLD_SUBSYSTEM,
            cadence=CADENCE_SUBSYSTEM,
        )

    def step(self, current_step: int, subsystem_id: str) -> Optional[ClosureReport]:
        """Check if a subsystem's child processes are healthy.

        Args:
            current_step: Current simulation step
            subsystem_id: ID of the subsystem to monitor

        Returns:
            ClosureReport if check ran, None if cadenced out.
        """
        return self._monitor.step(current_step, subsystem_id, "subsystem")

    def get_statistics(self) -> dict[str, Any]:
        """Subsystem closure statistics."""
        return self._monitor.get_statistics()


# =====================================================================
# Organ-level closure
# =====================================================================


class OrganClosure:
    """Monitor and advise on organ-level circular production.

    An organ's subsystems maintain the organ, and the organ maintains
    its subsystems. This monitor ensures subsystem health and recommends
    restructuring if needed.

    Checks every CADENCE_ORGAN steps (8, Fibonacci).
    """

    def __init__(
        self,
        holon_registry: HolonRegistry,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        self._monitor = AutopoieticMonitor(
            holon_registry=holon_registry,
            event_bus=event_bus,
            health_threshold=HEALTH_THRESHOLD_ORGAN,
            cadence=CADENCE_ORGAN,
        )

    def step(self, current_step: int, organ_id: str) -> Optional[ClosureReport]:
        """Check if an organ's child subsystems are healthy.

        Args:
            current_step: Current simulation step
            organ_id: ID of the organ to monitor

        Returns:
            ClosureReport if check ran, None if cadenced out.
        """
        return self._monitor.step(current_step, organ_id, "organ")

    def get_statistics(self) -> dict[str, Any]:
        """Organ closure statistics."""
        return self._monitor.get_statistics()


# =====================================================================
# Organism-level closure
# =====================================================================


class OrganismClosure:
    """Monitor and advise on organism-level circular production.

    Mae's organs maintain Mae, and Mae maintains her organs. This monitor
    ensures organ health and recommends system-wide rebalancing if needed.

    Checks every CADENCE_ORGANISM steps (13, Fibonacci).
    """

    def __init__(
        self,
        holon_registry: HolonRegistry,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        self._monitor = AutopoieticMonitor(
            holon_registry=holon_registry,
            event_bus=event_bus,
            health_threshold=HEALTH_THRESHOLD_ORGANISM,
            cadence=CADENCE_ORGANISM,
        )

    def step(self, current_step: int) -> Optional[ClosureReport]:
        """Check if Mae's child organs are healthy.

        Args:
            current_step: Current simulation step

        Returns:
            ClosureReport if check ran, None if cadenced out.
        """
        return self._monitor.step(current_step, "mae", "organism")

    def get_statistics(self) -> dict[str, Any]:
        """Organism closure statistics."""
        return self._monitor.get_statistics()


# =====================================================================
# Coordinator - runs all closure checks
# =====================================================================


class ClosureCoordinator:
    """Orchestrates closure monitoring across all scales.

    Delegates to SubsystemClosure, OrganClosure, OrganismClosure.
    Called from bootstrap or model step hook with step number.
    Each closure monitor internally decides if it's time to check
    (via cadence). Advisory only — never blocks.
    """

    def __init__(
        self,
        holon_registry: HolonRegistry,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        self._registry = holon_registry
        self._bus = event_bus
        self._subsystem_closure = SubsystemClosure(holon_registry, event_bus)
        self._organ_closure = OrganClosure(holon_registry, event_bus)
        self._organism_closure = OrganismClosure(holon_registry, event_bus)
        self._step_count: int = 0

    def step(self, current_step: int) -> list[ClosureReport]:
        """Run closure checks at all scales.

        Each scale checks at its own cadence. Returns all reports
        that actually ran (some may be cadenced out).

        Args:
            current_step: Current simulation step

        Returns:
            List of ClosureReport objects from scales that checked.
        """
        self._step_count += 1
        reports: list[ClosureReport] = []

        # Subsystem-level closure
        subsystem_ids = [
            hid for hid, entry in self._registry._holons.items()
            if entry.holon_type == "subsystem"
        ]
        for sub_id in subsystem_ids:
            report = self._subsystem_closure.step(current_step, sub_id)
            if report is not None:
                reports.append(report)

        # Organ-level closure
        organ_ids = [
            hid for hid, entry in self._registry._holons.items()
            if entry.holon_type == "organ"
        ]
        for organ_id in organ_ids:
            report = self._organ_closure.step(current_step, organ_id)
            if report is not None:
                reports.append(report)

        # Organism-level closure
        report = self._organism_closure.step(current_step)
        if report is not None:
            reports.append(report)

        return reports

    def get_statistics(self) -> dict[str, Any]:
        """Statistics across all closure scales."""
        return {
            "total_steps_coordinated": self._step_count,
            "subsystem_closure": self._subsystem_closure.get_statistics(),
            "organ_closure": self._organ_closure.get_statistics(),
            "organism_closure": self._organism_closure.get_statistics(),
        }
