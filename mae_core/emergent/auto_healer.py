"""Auto-Healing Architecture - Three-phase biological recovery.

Biological analogy: Salamander limb regeneration + immune wound healing.
When tissue is damaged, the body:
1. ISOLATES - clotting seals the wound (prevents spread)
2. ASSESSES - immune cells inspect damage (root cause analysis)
3. RESTORES - stem cells rebuild tissue (targeted recovery)

Mae's auto-healer follows the same three phases:
1. Isolate failing substrate region (prevent cascade)
2. Use CausalEngine to find root cause (not just symptoms)
3. Restore: restart agents, redistribute load, reconnect

Meta-Healing Triad (Law 6: Autopoietic Closure):
  The healer must be part of the system it heals. AutoHealer monitors
  its own health via three indicators (scan staleness, queue overflow,
  detection blindness) and heals itself when degraded. This forms
  a triadic connection: AutoHealer (system) <-> self_monitor (function)
  <-> SomaticMap (witness), closing the autopoietic loop.

Connection points:
- HAVEN detects risk -> auto-healer receives failure alerts
- CausalEngine provides root cause analysis
- Substrate provides region isolation/restoration
- Morphogenesis can spawn replacement agents
- EventBus publishes healing lifecycle events
- Endocrine triggers cortisol during healing (stress response)

Implementation is split across sub-modules for the 500-line cap:
  auto_healer_models.py  -- HealingPhase, FailureType, FailureReport,
                            HealingAction, HealingRecord
  auto_healer_phases.py  -- _AutoHealerPhasesMixin (pipeline + registration)
  auto_healer_monitor.py -- _AutoHealerMonitorMixin (self-monitoring triad)
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from typing import Any, Callable, Optional

from mae_core.emergent.auto_healer_models import (
    FailureReport,
    FailureType,
    HealingAction,
    HealingPhase,
    HealingRecord,
)
from mae_core.emergent.auto_healer_phases import (
    _AutoHealerPhasesMixin,
    CH_HEALING_COMPLETE,
    CH_HEALING_FAILED,
    CH_HEALING_PHASE,
)
from mae_core.emergent.auto_healer_monitor import (
    _AutoHealerMonitorMixin,
    CH_HEALING_SELF_HEALED,
)

logger = logging.getLogger(__name__)

# EventBus channels (kept here for backward-compatible import)
CH_FAILURE_DETECTED = "healing.failure_detected"
CH_HEALING_STARTED = "healing.started"
# CH_HEALING_PHASE, CH_HEALING_COMPLETE, CH_HEALING_FAILED re-exported below


class AutoHealer(_AutoHealerPhasesMixin, _AutoHealerMonitorMixin):
    """Three-phase biological healing system.

    Phase 1 - ISOLATE: Seal the wound. Prevent cascade failure by
    isolating the affected substrate region. Like blood clotting.

    Phase 2 - ASSESS: Send immune cells to inspect. Use CausalEngine
    to identify root cause, not just symptoms. Like macrophages
    presenting antigens to T-cells.

    Phase 3 - RESTORE: Rebuild tissue. Restart agents, redistribute
    load, reconnect substrate. Like stem cell differentiation at
    the wound site (blastema formation).
    """

    def __init__(
        self,
        event_bus: Any = None,
        substrate: Any = None,
        causal_engine: Any = None,
        haven: Any = None,
        somatic_map: Any = None,
        max_concurrent_healings: int = 3,
        max_history: int = 100,
        auto_isolate_threshold: float = 0.8,
    ) -> None:
        self._bus = event_bus
        self._substrate = substrate
        self._causal = causal_engine
        self._haven = haven
        self._somatic_map = somatic_map

        self._max_concurrent = max_concurrent_healings
        self._auto_isolate = auto_isolate_threshold

        # Active healing records
        self._active_healings: dict[str, HealingRecord] = {}
        self._history: deque[HealingRecord] = deque(maxlen=max_history)

        # Healing callbacks (failure_type -> list of recovery functions)
        self._recovery_actions: dict[FailureType, list[Callable]] = defaultdict(list)

        # Statistics
        self._total_healings = 0
        self._successful_healings = 0
        self._failed_healings = 0
        self._cascade_preventions = 0

        # Per-system healing cooldown: prevents re-healing the same system
        # every scan interval when previous healing had no effect.
        self._healing_cooldowns: dict[str, int] = {}

        # Cortisol-based priority: higher cortisol = more urgency
        self._cortisol_priority: float = 0.0

        self._lock = threading.RLock()

        # Register default recovery actions
        self._register_defaults()

        # Subscribe to HAVEN risk alerts
        if self._bus:
            self._bus.register_callback("haven.risk_alert", self._on_risk_alert)
            self._bus.register_callback("substrate.starvation_alert", self._on_starvation)

        # Proactive scanning state
        self._step_count: int = 0
        self._scan_interval: int = 10  # Full scan every N steps
        self._last_successful_heal: float = time.time()
        self._health_threshold: float = 0.5  # Systems below this are unhealthy

        # Meta-healing triad state (Law 6: Autopoietic Closure)
        # The healer monitors and heals itself — closing the autopoietic loop
        self._connection_registry: Any = None  # Injected after construction
        self._last_scan_step: int = 0  # Step at which last scan ran
        self._total_detections: int = 0  # Total failures detected
        self._last_detection_step: int = 0  # Step at which last detection occurred
        self._self_heal_count: int = 0  # Total self-healing actions taken
        self._self_healing_triad_registered: bool = False
        self._meta_health_check_interval: int = 5  # Check own health every N steps
        self._max_failure_queue_size: int = 50  # Queue overflow threshold
        self._scan_staleness_threshold: int = 10  # Steps without scan = stale
        self._detection_blindness_threshold: int = 20  # Steps without detection = suspicious

        logger.info("AutoHealer initialized (max_concurrent=%d, meta_healing=enabled)", max_concurrent_healings)

    # =========================================================================
    # Proactive Step (periodic body scan)
    # =========================================================================

    def step(self) -> None:
        """Proactive health scan — called each model tick.

        Lightweight: increments a counter every call but only runs
        the full diagnostic every ``_scan_interval`` steps to keep
        per-tick cost near zero.

        Checks:
        1. SomaticMap for systems whose health dropped below threshold.
        2. Internal healing-queue pressure (too many active healings).
        3. Staleness of last successful heal (detects silent failures).
        4. Meta-healing: AutoHealer monitors and heals itself (Law 6).
        """
        self._step_count += 1

        # --- 4. Meta-healing: self-monitor triad (every N steps) ---
        # This runs on a separate cadence from the full scan to ensure
        # the healer can detect its own staleness even if scanning fails.
        if self._step_count % self._meta_health_check_interval == 0:
            self._self_monitor()

        # Only run the full scan periodically
        if self._step_count % self._scan_interval != 0:
            return

        # Track that a scan ran (for scan staleness detection)
        self._last_scan_step = self._step_count

        # --- 1. Query SomaticMap for unhealthy systems ---
        somatic = getattr(self, "_somatic_map", None)
        if somatic is not None:
            get_unhealthy = getattr(somatic, "get_unhealthy_systems", None)
            if get_unhealthy is not None:
                try:
                    sick_systems = get_unhealthy(self._health_threshold)
                    for system_id in sick_systems:
                        # Never heal ourselves — breaks the self-healing loop
                        if system_id == "auto_healer":
                            continue
                        # Per-system cooldown: skip systems healed within last 50 steps
                        last_healed = self._healing_cooldowns.get(system_id, -999)
                        if self._step_count - last_healed < 50:
                            continue
                        # Only file a report if we are not already healing this system
                        failure_id = f"proactive-{system_id}-{self._step_count}"
                        with self._lock:
                            already_healing = any(
                                system_id in r.failure.affected_agents
                                for r in self._active_healings.values()
                            )
                        if not already_healing:
                            failure = FailureReport(
                                failure_id=failure_id,
                                failure_type=FailureType.PERFORMANCE_DEGRADATION,
                                affected_agents=[system_id],
                                severity=0.6,
                                metadata={"source": "proactive_scan"},
                            )
                            self.report_failure(failure)
                except Exception:
                    logger.debug("Proactive somatic scan failed", exc_info=True)

        # --- 2. Check healing-queue pressure ---
        with self._lock:
            active_count = len(self._active_healings)
        if active_count >= self._max_concurrent:
            logger.warning(
                "AutoHealer queue saturated (%d/%d active healings)",
                active_count,
                self._max_concurrent,
            )

        # --- 3. Detect silent failures (no successful heal in a long while) ---
        if self._total_healings > 0:
            since_last = time.time() - self._last_successful_heal
            # If we have had healings but none succeeded recently, log it
            if since_last > 300 and self._failed_healings > self._successful_healings:
                logger.warning(
                    "AutoHealer may be stuck: %d failed vs %d successful, "
                    "last success %.0fs ago",
                    self._failed_healings,
                    self._successful_healings,
                    since_last,
                )

        # Report own health to somatic map
        if somatic is not None:
            heartbeat = getattr(somatic, "heartbeat", None)
            if heartbeat is not None:
                try:
                    own_health = 1.0 if active_count == 0 else max(
                        0.3, 1.0 - (active_count / max(self._max_concurrent, 1))
                    )
                    heartbeat("auto_healer", health=own_health)
                except Exception:
                    pass

    # =========================================================================
    # Failure Detection
    # =========================================================================

    def report_failure(self, failure: FailureReport) -> Optional[HealingRecord]:
        """Report a detected failure and begin healing if possible."""
        with self._lock:
            # Track detection for meta-healing (detection blindness check)
            self._total_detections += 1
            self._last_detection_step = self._step_count

            if len(self._active_healings) >= self._max_concurrent:
                logger.warning(
                    "AutoHealer at capacity (%d/%d), queuing failure %s",
                    len(self._active_healings),
                    self._max_concurrent,
                    failure.failure_id,
                )
                return None

            record = HealingRecord(failure=failure)
            self._active_healings[failure.failure_id] = record
            self._total_healings += 1

            if self._bus:
                self._bus.publish(CH_FAILURE_DETECTED, {
                    "failure_id": failure.failure_id,
                    "failure_type": failure.failure_type.value,
                    "severity": failure.severity,
                    "affected_agents": failure.affected_agents,
                })

            # Begin healing pipeline
            self._execute_healing(record)
            return record

    def _on_risk_alert(self, channel: str, message: Any) -> None:
        """Handle HAVEN risk alerts as potential failures."""
        if isinstance(message, str):
            try:
                import json
                message = json.loads(message)
            except (json.JSONDecodeError, TypeError):
                return
        if isinstance(message, dict):
            risk_score = message.get("risk_score", 0)
            if risk_score >= self._auto_isolate:
                agent_id = message.get("agent_id", "unknown")
                failure = FailureReport(
                    failure_id=f"haven-{agent_id}-{int(time.time())}",
                    failure_type=FailureType.PERFORMANCE_DEGRADATION,
                    affected_agents=[agent_id],
                    severity=risk_score,
                )
                self.report_failure(failure)

    def _on_starvation(self, channel: str, message: Any) -> None:
        """Handle substrate starvation alerts.

        Substrate publishes {"nodes": [list_of_starving_nodes], "step": N}.
        We iterate over the nodes list and file one report per node.
        """
        if isinstance(message, str):
            try:
                import json
                message = json.loads(message)
            except (json.JSONDecodeError, TypeError):
                return
        if isinstance(message, dict):
            nodes = message.get("nodes", [])
            if not nodes:
                return  # No starving nodes — nothing to heal
            for node_id in nodes:
                failure = FailureReport(
                    failure_id=f"starve-{node_id}-{int(time.time())}",
                    failure_type=FailureType.STARVATION,
                    affected_agents=[str(node_id)],
                    severity=0.6,
                )
                self.report_failure(failure)

    # =========================================================================
    # Helpers
    # =========================================================================

    def _publish_phase(self, record: HealingRecord) -> None:
        if self._bus:
            self._bus.publish(CH_HEALING_PHASE, {
                "failure_id": record.failure.failure_id,
                "phase": record.phase.value,
            })
        # Notify somatic map of healing phase changes for body awareness
        if self._somatic_map is not None:
            try:
                self._somatic_map.heartbeat(
                    "auto_healer",
                    health=0.5 if record.phase != HealingPhase.COMPLETE else 1.0,
                )
            except Exception:
                logger.debug("Could not notify somatic_map of phase change")

    # =========================================================================
    # Hormonal Modulation
    # =========================================================================

    def set_cortisol_priority(self, level: float) -> None:
        """Adjust healing urgency based on cortisol level.

        Higher cortisol means the system is stressed, so healing
        becomes more urgent (lower auto-isolate threshold).

        Called by EndocrineSystem via hormone consumer dispatch.
        """
        self._cortisol_priority = max(0.0, min(1.0, level))
        # Under high cortisol, lower the isolation threshold to be more aggressive
        if level > 0.5:
            self._auto_isolate = max(0.4, self._auto_isolate - 0.1 * level)

    def set_hormone_level(self, hormone: str, level: float) -> None:
        """Generic hormone receiver for endocrine integration.

        Allows the endocrine system to modulate healing behaviour
        without requiring a specific named method for every hormone.
        """
        if hormone == "cortisol":
            self.set_cortisol_priority(level)

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_statistics(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_healings": self._total_healings,
                "successful": self._successful_healings,
                "failed": self._failed_healings,
                "active": len(self._active_healings),
                "cascade_preventions": self._cascade_preventions,
                "success_rate": (
                    self._successful_healings / max(self._total_healings, 1)
                ),
                "history_size": len(self._history),
                "meta_healing": {
                    "self_heal_count": self._self_heal_count,
                    "total_detections": self._total_detections,
                    "triad_registered": self._self_healing_triad_registered,
                    "health_threshold": self._health_threshold,
                },
            }

    def get_active_healings(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "failure_id": r.failure.failure_id,
                    "failure_type": r.failure.failure_type.value,
                    "phase": r.phase.value,
                    "actions": len(r.actions_taken),
                }
                for r in self._active_healings.values()
            ]

    def get_healing_history(self, limit: int = 10) -> list[dict[str, Any]]:
        with self._lock:
            recent = list(self._history)[-limit:]
            return [
                {
                    "failure_id": r.failure.failure_id,
                    "failure_type": r.failure.failure_type.value,
                    "root_cause": r.root_cause,
                    "success": r.success,
                    "actions_count": len(r.actions_taken),
                    "duration": (
                        (r.completed_at or time.time()) - r.started_at
                    ),
                }
                for r in recent
            ]
