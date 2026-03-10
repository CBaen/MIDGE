"""Octopus Colony monitoring mixin — health checks, auto-scaling, health reports.

Extracted from octopus_colony.py to respect the 500-line cap.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .octopus_signals import OctopusSpecialization

logger = logging.getLogger(__name__)


class OctopusMonitoringMixin:
    """Monitoring, health-check, and auto-scaling logic for OctopusColony."""

    def _monitoring_loop(self) -> None:
        """Periodic health checks and auto-scaling."""
        while self._running:
            try:
                self._check_health()
                self._check_workload_scaling()
                self._publish_health_report()
                time.sleep(self._monitoring_interval)
            except Exception:
                logger.exception("Colony monitoring error")
                time.sleep(self._monitoring_interval)

    def _check_health(self) -> None:
        """Replace unhealthy octopuses. Spawn first, then despawn."""
        unhealthy: list[tuple[str, OctopusSpecialization]] = []

        for oid, octopus in self.octopuses.items():
            octopus.update_metrics()
            if octopus.health < self.health_threshold:
                unhealthy.append((oid, octopus.specialization))

        for oid, spec in unhealthy:
            logger.warning(
                "Replacing unhealthy %s (health=%.2f)",
                oid, self.octopuses[oid].health,
            )
            new_id = self.spawn_octopus(
                specialization=spec,
                reason=f"replacement_for_{oid}",
            )
            if new_id:
                self.despawn_octopus(oid, reason="health_below_threshold")

    def _check_workload_scaling(self) -> None:
        """Auto-scale based on average workload."""
        if not self.octopuses:
            return

        for octopus in self.octopuses.values():
            octopus.update_metrics()

        avg_workload = (
            sum(o.workload for o in self.octopuses.values()) / len(self.octopuses)
        )

        # Scale up
        if avg_workload > self.spawn_threshold and len(self.octopuses) < self.max_octopuses:
            from .octopus_signals import OctopusSpecialization
            self.spawn_octopus(
                specialization=OctopusSpecialization.GENERAL,
                reason=f"high_load_{avg_workload:.2f}",
            )

        # Scale down (respects Rule of 3 via despawn_octopus)
        elif avg_workload < self.despawn_threshold and len(self.octopuses) > self.min_octopuses:
            least_utilized = min(
                self.octopuses.items(), key=lambda x: x[1].workload
            )
            self.despawn_octopus(
                least_utilized[0],
                reason=f"low_load_{avg_workload:.2f}",
            )

    def _publish_health_report(self) -> None:
        """Publish colony health on EventBus."""
        if self._bus is None:
            return

        from .octopus_signals import CH_OCTOPUS_HEALTH

        avg_health = (
            sum(o.health for o in self.octopuses.values()) / len(self.octopuses)
            if self.octopuses else 0.0
        )
        avg_workload = (
            sum(o.workload for o in self.octopuses.values()) / len(self.octopuses)
            if self.octopuses else 0.0
        )

        from . import octopus_colony as _mod
        self._bus.publish(CH_OCTOPUS_HEALTH, {
            "colony_size": len(self.octopuses),
            "average_health": avg_health,
            "average_workload": avg_workload,
            "rule_of_3_compliant": len(self.octopuses) >= _mod.MIN_AGENTS,
            "timestamp": time.time(),
        })
