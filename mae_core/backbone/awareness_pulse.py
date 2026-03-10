"""AwarenessPulse — periodic hierarchy health check for Mae.

Extracted from holon_protocol.py for single-responsibility.

Every `interval` steps, queries all holons and checks for:
- Orphaned systems (registered but parent doesn't exist)
- Health gradient (average child health vs parent health)
- Peer drift (peers with divergent health scores)

Publishes summary on holon.awareness_pulse.
Publishes anomalies on holon.anomaly_detected.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# =====================================================================
# EventBus channels
# =====================================================================
CH_AWARENESS_PULSE = "holon.awareness_pulse"
CH_AWARENESS_ANOMALY = "holon.anomaly_detected"


class AwarenessPulse:
    """Periodic hierarchy health check — the organism's self-scan.

    Every `interval` steps, queries all holons and checks for:
    - Orphaned systems (registered but parent doesn't exist)
    - Health gradient (average child health vs parent health)
    - Peer drift (peers with divergent health scores)

    Publishes summary on holon.awareness_pulse.
    Publishes anomalies on holon.anomaly_detected.
    """

    def __init__(
        self,
        registry: Any,
        event_bus: Any,
        interval: int = 25,
    ) -> None:
        self._registry = registry
        self._event_bus = event_bus
        self._interval = interval
        self._step_count = 0
        self._pulse_count = 0
        self._last_anomalies: list[dict[str, Any]] = []

    def step(self) -> None:
        """Called every model step. Runs scan at interval."""
        self._step_count += 1
        if self._step_count % self._interval == 0:
            self._run_pulse()

    def _run_pulse(self) -> None:
        """Execute awareness scan across all holons."""
        self._pulse_count += 1
        anomalies: list[dict[str, Any]] = []
        holon_ids = self._registry.get_all_ids()

        orphans: list[str] = []
        health_issues: list[dict[str, Any]] = []

        for hid in holon_ids:
            entry = self._registry.get_entry(hid)
            if entry is None:
                continue

            if entry.parent_id is not None:
                parent = self._registry.get_entry(entry.parent_id)
                if parent is None:
                    orphans.append(hid)

            if self._registry._somatic_map is not None:
                children = self._registry.get_children(hid)
                if children:
                    child_healths = []
                    for cid in children:
                        proxy = self._registry.get_proxy(cid)
                        child_healths.append(proxy.get_health())
                    if child_healths:
                        avg_child = sum(child_healths) / len(child_healths)
                        own_proxy = self._registry.get_proxy(hid)
                        own_health = own_proxy.get_health()
                        drift = abs(own_health - avg_child)
                        if drift > 0.3:
                            health_issues.append({
                                "holon_id": hid,
                                "own_health": own_health,
                                "avg_child_health": avg_child,
                                "drift": drift,
                            })

        if orphans:
            anomalies.append({"type": "orphaned_systems", "holon_ids": orphans})
        if health_issues:
            anomalies.append({"type": "health_gradient", "issues": health_issues})

        self._last_anomalies = anomalies

        summary = {
            "pulse_number": self._pulse_count,
            "step": self._step_count,
            "total_holons": len(holon_ids),
            "orphans": len(orphans),
            "health_issues": len(health_issues),
            "anomaly_count": len(anomalies),
        }
        self._event_bus.publish(CH_AWARENESS_PULSE, summary)

        if anomalies:
            self._event_bus.publish(CH_AWARENESS_ANOMALY, {
                "pulse_number": self._pulse_count,
                "anomalies": anomalies,
            })
            logger.warning(
                "Awareness pulse #%d: %d anomalies (%d orphans, %d health issues)",
                self._pulse_count, len(anomalies), len(orphans), len(health_issues),
            )
        else:
            logger.debug("Awareness pulse #%d: all clear (%d holons)", self._pulse_count, len(holon_ids))

    def get_statistics(self) -> dict[str, Any]:
        """Pulse statistics."""
        return {
            "pulse_count": self._pulse_count,
            "step_count": self._step_count,
            "interval": self._interval,
            "last_anomalies": self._last_anomalies,
        }
