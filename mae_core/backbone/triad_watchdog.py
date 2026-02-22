"""Triad Watchdog - Runtime bypass detection for Rule of 3.

Biological analogy: The reticular activating system (RAS) in the
brainstem. It doesn't make decisions - it monitors arousal and
alertness. If something important happens and the brain doesn't
respond, the RAS triggers an alarm. Similarly, the Watchdog
monitors whether processes are actually USING their triads.

The TriadEnforcer is the conscious enforcement (registration, voting).
The Watchdog is the unconscious enforcement (did you actually ASK?).
The Auditor is the reflective enforcement (are the patterns healthy?).

Three complementary enforcement lenses:
- Enforcer: FORMAL - "Do you have enough validators?"
- Watchdog: OPERATIONAL - "Are you actually calling them?"
- Auditor: BEHAVIORAL - "Are the voting patterns healthy?"

Connection points:
- Subscribes to EventBus for process action events
- Cross-references with TriadEnforcer vote records
- Publishes bypass alerts on EventBus
- SomaticMap registers Watchdog as CRITICAL system
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# EventBus channels
CH_WATCHDOG_BYPASS = "watchdog.bypass_detected"
CH_WATCHDOG_SILENT = "watchdog.silent_validator"
CH_WATCHDOG_HEALTH = "watchdog.health_report"


@dataclass
class BypassAlert:
    """Alert when a process acts without triad validation."""

    process_id: str
    action: str
    timestamp: float = field(default_factory=time.time)
    detail: str = ""


@dataclass
class ProcessActivity:
    """Tracks observed actions vs validated actions for a process."""

    process_id: str
    observed_actions: int = 0  # Actions detected via EventBus
    validated_actions: int = 0  # Actions that went through triad vote
    last_observed: float = 0.0
    last_validated: float = 0.0
    bypass_count: int = 0


class TriadWatchdog:
    """Runtime monitor that detects triad bypass attempts.

    Watches EventBus for process actions and cross-references with
    TriadEnforcer vote records. If a process acts without asking
    its triad, the Watchdog raises an alert.

    Like a security camera: it doesn't stop you, but it records
    that you didn't check in at the gate.
    """

    def __init__(
        self,
        event_bus: Any = None,
        triad_enforcer: Any = None,
        bypass_threshold: int = 3,
    ) -> None:
        self._bus = event_bus
        self._enforcer = triad_enforcer
        self._bypass_threshold = bypass_threshold

        # Track activity per process
        self._activity: dict[str, ProcessActivity] = {}
        self._alerts: list[BypassAlert] = []
        self._max_alerts = 500

        # Channels being monitored for actions
        self._monitored_channels: dict[str, str] = {}  # channel -> process_id

        # Optional ConnectionRegistry for bare dyad checks
        self._connection_registry: Any = None

        self._lock = threading.RLock()

        logger.info("TriadWatchdog initialized (bypass_threshold=%d)", bypass_threshold)

    # =========================================================================
    # Channel Monitoring
    # =========================================================================

    def monitor_channel(self, channel: str, process_id: str) -> None:
        """Register a channel to monitor for a specific process.

        When events arrive on this channel, the Watchdog checks
        whether the process has a corresponding triad vote.
        """
        self._monitored_channels[channel] = process_id
        if self._bus:
            self._bus.register_callback(channel, self._on_action_observed)

        with self._lock:
            if process_id not in self._activity:
                self._activity[process_id] = ProcessActivity(
                    process_id=process_id
                )

    def _on_action_observed(self, channel: str, message: Any) -> None:
        """Called when an action is observed on a monitored channel."""
        process_id = self._monitored_channels.get(channel)
        if not process_id:
            return

        with self._lock:
            activity = self._activity.get(process_id)
            if not activity:
                activity = ProcessActivity(process_id=process_id)
                self._activity[process_id] = activity

            activity.observed_actions += 1
            activity.last_observed = time.time()

    def record_validated_action(self, process_id: str) -> None:
        """Record that a process action went through triad validation.

        Called by TriadEnforcer after a successful vote, or manually
        by systems that integrate with the Watchdog.
        """
        with self._lock:
            activity = self._activity.get(process_id)
            if not activity:
                activity = ProcessActivity(process_id=process_id)
                self._activity[process_id] = activity

            activity.validated_actions += 1
            activity.last_validated = time.time()

    # =========================================================================
    # Bypass Detection
    # =========================================================================

    def check_for_bypasses(self) -> list[BypassAlert]:
        """Check all monitored processes for triad bypasses.

        A bypass is detected when observed_actions exceeds
        validated_actions by more than the threshold.
        """
        new_alerts = []

        with self._lock:
            for pid, activity in self._activity.items():
                gap = activity.observed_actions - activity.validated_actions
                if gap >= self._bypass_threshold:
                    alert = BypassAlert(
                        process_id=pid,
                        action="unvalidated_actions",
                        detail=(
                            f"{activity.observed_actions} observed, "
                            f"{activity.validated_actions} validated, "
                            f"gap={gap}"
                        ),
                    )
                    new_alerts.append(alert)
                    activity.bypass_count += 1
                    self._alerts.append(alert)

                    if self._bus:
                        self._bus.publish(CH_WATCHDOG_BYPASS, {
                            "process_id": pid,
                            "observed": activity.observed_actions,
                            "validated": activity.validated_actions,
                            "gap": gap,
                        })

        # Trim alert history
        if len(self._alerts) > self._max_alerts:
            self._alerts = self._alerts[-self._max_alerts:]

        return new_alerts

    def detect_silent_validators(self) -> list[dict[str, Any]]:
        """Detect validators that haven't been invoked recently.

        A silent validator might be dead, disconnected, or bypassed.
        Like noticing that one of your smoke detectors hasn't beeped
        in years - is it still working?
        """
        silent = []
        if not self._enforcer:
            return silent

        now = time.time()
        for pid, activity in self._activity.items():
            status = self._enforcer.get_process_status(pid)
            if not status:
                continue

            for v in status.get("validators", []):
                if v["invocations"] == 0 and (now - activity.last_observed) < 60:
                    # Process is active but this validator has never been called
                    entry = {
                        "process_id": pid,
                        "validator_id": v["id"],
                        "validator_type": v["type"],
                        "process_actions": activity.observed_actions,
                    }
                    silent.append(entry)

                    if self._bus:
                        self._bus.publish(CH_WATCHDOG_SILENT, entry)

        return silent

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_statistics(self) -> dict[str, Any]:
        """Get watchdog monitoring statistics."""
        with self._lock:
            total_observed = sum(a.observed_actions for a in self._activity.values())
            total_validated = sum(a.validated_actions for a in self._activity.values())
            total_bypasses = sum(a.bypass_count for a in self._activity.values())

            return {
                "monitored_processes": len(self._activity),
                "monitored_channels": len(self._monitored_channels),
                "total_observed_actions": total_observed,
                "total_validated_actions": total_validated,
                "total_bypass_alerts": total_bypasses,
                "validation_rate": (
                    total_validated / max(total_observed, 1)
                ),
                "alert_history_size": len(self._alerts),
            }

    def get_activity(self, process_id: str) -> dict[str, Any] | None:
        """Get activity report for a specific process."""
        with self._lock:
            activity = self._activity.get(process_id)
            if not activity:
                return None
            return {
                "process_id": activity.process_id,
                "observed": activity.observed_actions,
                "validated": activity.validated_actions,
                "bypass_count": activity.bypass_count,
                "validation_rate": (
                    activity.validated_actions
                    / max(activity.observed_actions, 1)
                ),
                "last_observed": activity.last_observed,
                "last_validated": activity.last_validated,
            }

    # =========================================================================
    # Connection Registry Integration
    # =========================================================================

    def set_connection_registry(self, registry: Any) -> None:
        """Wire the ConnectionRegistry for bare dyad detection."""
        self._connection_registry = registry

    def check_bare_dyads(self) -> list[dict[str, Any]]:
        """Check for connections without witnesses (bare dyads).

        Queries the ConnectionRegistry and publishes alerts for
        any connections missing a triadic witness.
        """
        if self._connection_registry is None:
            return []

        bare = self._connection_registry.get_bare_dyads()
        results = []
        for triad in bare:
            entry = {
                "connection_id": triad.connection_id,
                "source": triad.source,
                "target": triad.target,
                "type": triad.connection_type.value,
            }
            results.append(entry)

        if results:
            # Check enforcement mode on registry
            mode = getattr(self._connection_registry, "enforcement_mode", None)
            mode_val = mode.value if mode else "unknown"
            if mode_val == "blocking":
                logger.error(
                    "BLOCKING: %d bare dyads detected by watchdog", len(results),
                )
            if self._bus:
                self._bus.publish(CH_WATCHDOG_HEALTH, {
                    "check": "bare_dyads",
                    "count": len(results),
                    "connections": [r["connection_id"] for r in results],
                    "enforcement": mode_val,
                })

        return results

    def get_alerts(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get recent bypass alerts."""
        with self._lock:
            return [
                {
                    "process_id": a.process_id,
                    "action": a.action,
                    "detail": a.detail,
                    "timestamp": a.timestamp,
                }
                for a in self._alerts[-limit:]
            ]
