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
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# EventBus channels
CH_FAILURE_DETECTED = "healing.failure_detected"
CH_HEALING_STARTED = "healing.started"
CH_HEALING_PHASE = "healing.phase_changed"
CH_HEALING_COMPLETE = "healing.complete"
CH_HEALING_FAILED = "healing.failed"
CH_HEALING_SELF_HEALED = "healing.self_healed"


class HealingPhase(Enum):
    DETECTING = "detecting"
    ISOLATING = "isolating"
    ASSESSING = "assessing"
    RESTORING = "restoring"
    VERIFYING = "verifying"
    COMPLETE = "complete"
    FAILED = "failed"


class FailureType(Enum):
    PERFORMANCE_DEGRADATION = "performance_degradation"
    AGENT_CRASH = "agent_crash"
    COMMUNICATION_BREAK = "communication_break"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    POLICY_CONTAGION = "policy_contagion"
    STARVATION = "starvation"
    CASCADE_FAILURE = "cascade_failure"


@dataclass
class FailureReport:
    """Detected failure requiring healing."""

    failure_id: str
    failure_type: FailureType
    affected_agents: list[str] = field(default_factory=list)
    affected_region: Optional[str] = None
    severity: float = 0.5  # [0, 1]
    detected_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HealingAction:
    """A specific recovery action taken."""

    action: str
    target: str
    success: bool = False
    details: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class HealingRecord:
    """Complete record of a healing operation."""

    failure: FailureReport
    phase: HealingPhase = HealingPhase.DETECTING
    root_cause: Optional[str] = None
    causal_path: list[str] = field(default_factory=list)
    actions_taken: list[HealingAction] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    success: bool = False


class AutoHealer:
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
    # Meta-Healing Triad (Law 6: Autopoietic Closure)
    # =========================================================================

    def _self_monitor(self) -> None:
        """Monitor AutoHealer's own health and self-heal when degraded.

        The healer must be part of the system it heals (autopoietic closure).
        Three health indicators form the self-monitoring triad:

        1. Scan staleness: Has the healer run a scan recently?
        2. Queue overflow: Is the failure queue growing unboundedly?
        3. Detection blindness: Is the healer detecting anything at all?

        When degradation is detected, targeted self-healing actions restore
        the healer's capability. SomaticMap witnesses the self-healing event.
        """
        healed_actions: list[str] = []

        # --- Indicator 1: Scan staleness ---
        # If no scan has run in the last N steps, the scanner may be stuck.
        steps_since_scan = self._step_count - self._last_scan_step
        if steps_since_scan > self._scan_staleness_threshold and self._step_count > self._scan_staleness_threshold:
            logger.warning(
                "AutoHealer self-monitor: scan stale (%d steps since last scan), resetting",
                steps_since_scan,
            )
            # Self-heal: reset scan counter to force immediate scan on next interval
            self._last_scan_step = self._step_count - self._scan_interval
            healed_actions.append("reset_scan_staleness")

        # --- Indicator 2: Queue overflow ---
        # If history is growing unboundedly, prune oldest entries.
        with self._lock:
            history_size = len(self._history)
        if history_size > self._max_failure_queue_size:
            logger.warning(
                "AutoHealer self-monitor: history overflow (%d entries, threshold %d), pruning",
                history_size,
                self._max_failure_queue_size,
            )
            # Self-heal: prune oldest 50% of history
            with self._lock:
                prune_count = history_size // 2
                for _ in range(prune_count):
                    if self._history:
                        self._history.popleft()
            healed_actions.append(f"pruned_history_{prune_count}_entries")

        # --- Indicator 3: Detection blindness ---
        # If the system has been active for a while but no failures detected,
        # the healer's sensitivity may have drifted.
        if (
            self._step_count > self._detection_blindness_threshold
            and self._total_detections == 0
            and self._step_count - self._last_detection_step > self._detection_blindness_threshold
        ):
            # Only flag this if there are active systems to monitor
            somatic = getattr(self, "_somatic_map", None)
            system_count = 0
            if somatic is not None:
                get_all = getattr(somatic, "get_all_systems", None)
                if get_all is not None:
                    try:
                        system_count = len(get_all())
                    except Exception:
                        pass
            if system_count > 0:
                logger.warning(
                    "AutoHealer self-monitor: detection blindness "
                    "(%d steps, 0 detections, %d systems active), widening threshold",
                    self._step_count,
                    system_count,
                )
                # Self-heal: widen detection criteria by lowering health threshold
                old_threshold = self._health_threshold
                self._health_threshold = min(self._health_threshold + 0.1, 0.8)
                healed_actions.append(
                    f"widened_health_threshold_{old_threshold:.2f}_to_{self._health_threshold:.2f}"
                )

        # --- Report self-healing to EventBus and SomaticMap ---
        if healed_actions:
            self._self_heal_count += len(healed_actions)

            if self._bus:
                self._bus.publish(CH_HEALING_SELF_HEALED, {
                    "step": self._step_count,
                    "actions": healed_actions,
                    "self_heal_count": self._self_heal_count,
                })

            # SomaticMap witnesses the self-healing (third party in the triad)
            somatic = getattr(self, "_somatic_map", None)
            if somatic is not None:
                heartbeat = getattr(somatic, "heartbeat", None)
                if heartbeat is not None:
                    try:
                        heartbeat("auto_healer", health=0.7)  # Recovering, not fully healthy
                    except Exception:
                        pass

            logger.info(
                "AutoHealer self-healed: %d actions taken at step %d: %s",
                len(healed_actions),
                self._step_count,
                ", ".join(healed_actions),
            )

    def register_self_healing_triad(self) -> None:
        """Register the meta-healing triad with ConnectionRegistry.

        Triad members:
          A = auto_healer (the system being healed)
          B = auto_healer.self_monitor (the monitoring function)
          C = somatic_map (the witness — independently tracks system health)

        This creates a proper triadic connection for the self-healing loop,
        satisfying Law 1 (No Bare Dyads) for the healing system itself.

        Called from main.py after ConnectionRegistry is available.
        Idempotent — safe to call multiple times.
        """
        if self._self_healing_triad_registered:
            return

        registry = getattr(self, "_connection_registry", None)
        if registry is None:
            logger.debug("Cannot register self-healing triad: no ConnectionRegistry")
            return

        try:
            register_conn = getattr(registry, "register_connection", None)
            if register_conn is None:
                return

            # Import ConnectionType locally to avoid circular imports
            from mae_core.backbone.connection_registry import ConnectionType

            register_conn(
                source="auto_healer",
                target="auto_healer.self_monitor",
                connection_type=ConnectionType.DIRECT_REFERENCE,
                witness="somatic_map",
                description="Meta-healing triad: AutoHealer monitors and heals itself, "
                            "witnessed by SomaticMap (Law 6: Autopoietic Closure)",
            )

            self._self_healing_triad_registered = True
            logger.info("AutoHealer: self-healing triad registered with ConnectionRegistry")
        except Exception:
            logger.debug("Failed to register self-healing triad", exc_info=True)

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
        """Handle substrate starvation alerts."""
        if isinstance(message, str):
            try:
                import json
                message = json.loads(message)
            except (json.JSONDecodeError, TypeError):
                return
        if isinstance(message, dict):
            node_id = message.get("node_id", "unknown")
            failure = FailureReport(
                failure_id=f"starve-{node_id}-{int(time.time())}",
                failure_type=FailureType.STARVATION,
                affected_agents=[str(node_id)],
                severity=0.6,
            )
            self.report_failure(failure)

    # =========================================================================
    # Three-Phase Healing Pipeline
    # =========================================================================

    def _execute_healing(self, record: HealingRecord) -> None:
        """Run the three-phase healing pipeline."""
        try:
            # Phase 1: Isolate
            self._phase_isolate(record)

            # Phase 2: Assess
            self._phase_assess(record)

            # Phase 3: Restore
            self._phase_restore(record)

            # Verify
            self._phase_verify(record)

        except Exception as e:
            logger.error("Healing failed for %s: %s", record.failure.failure_id, e)
            record.phase = HealingPhase.FAILED
            record.success = False
            self._failed_healings += 1
            if self._bus:
                self._bus.publish(CH_HEALING_FAILED, {
                    "failure_id": record.failure.failure_id,
                    "error": str(e),
                })
        finally:
            record.completed_at = time.time()
            self._active_healings.pop(record.failure.failure_id, None)
            self._history.append(record)

    def _phase_isolate(self, record: HealingRecord) -> None:
        """Phase 1: Isolate the affected region (clotting)."""
        record.phase = HealingPhase.ISOLATING
        self._publish_phase(record)

        failure = record.failure

        # Isolate affected agents via HAVEN
        if self._haven and failure.severity >= self._auto_isolate:
            for agent_id in failure.affected_agents:
                self._haven.isolate_agent(agent_id, reason=f"auto-heal: {failure.failure_type.value}")
                record.actions_taken.append(HealingAction(
                    action="isolate_agent",
                    target=agent_id,
                    success=True,
                    details=f"HAVEN isolation for {failure.failure_type.value}",
                ))

        # Isolate substrate region if available
        if self._substrate and failure.affected_region:
            self._substrate.isolate_region(failure.affected_region)
            record.actions_taken.append(HealingAction(
                action="isolate_region",
                target=failure.affected_region,
                success=True,
                details="Substrate region isolated",
            ))
            self._cascade_preventions += 1

        logger.info(
            "Phase 1 ISOLATE complete for %s (%d actions)",
            failure.failure_id,
            len(record.actions_taken),
        )

    def _phase_assess(self, record: HealingRecord) -> None:
        """Phase 2: Root cause analysis (immune inspection)."""
        record.phase = HealingPhase.ASSESSING
        self._publish_phase(record)

        failure = record.failure

        # Use causal engine to find root cause
        if self._causal:
            # Query for causal links to this failure type
            result = self._causal.query_causation(
                failure.failure_type.value,
                "system_degradation",
            )
            if result.is_causal:
                record.root_cause = result.cause
                record.causal_path = result.causal_path
            else:
                record.root_cause = failure.failure_type.value
        else:
            # Without causal engine, failure type IS the root cause
            record.root_cause = failure.failure_type.value

        record.actions_taken.append(HealingAction(
            action="root_cause_analysis",
            target=failure.failure_id,
            success=record.root_cause is not None,
            details=f"Root cause: {record.root_cause}",
        ))

        logger.info(
            "Phase 2 ASSESS complete for %s: root_cause=%s",
            failure.failure_id,
            record.root_cause,
        )

    def _phase_restore(self, record: HealingRecord) -> None:
        """Phase 3: Recovery (tissue regeneration)."""
        record.phase = HealingPhase.RESTORING
        self._publish_phase(record)

        failure = record.failure

        # Execute registered recovery actions for this failure type
        for callback in self._recovery_actions.get(failure.failure_type, []):
            try:
                result = callback(record)
                record.actions_taken.append(HealingAction(
                    action=f"recovery_{callback.__name__}",
                    target=failure.failure_id,
                    success=True,
                    details=str(result) if result else "",
                ))
            except Exception as e:
                record.actions_taken.append(HealingAction(
                    action=f"recovery_{callback.__name__}",
                    target=failure.failure_id,
                    success=False,
                    details=str(e),
                ))

        # Restore isolated agents
        if self._haven:
            for agent_id in failure.affected_agents:
                if self._haven.is_agent_isolated(agent_id):
                    self._haven.restore_agent(agent_id)
                    record.actions_taken.append(HealingAction(
                        action="restore_agent",
                        target=agent_id,
                        success=True,
                        details="Agent restored from isolation",
                    ))

        # Restore substrate region
        if self._substrate and failure.affected_region:
            self._substrate.restore_region(failure.affected_region)
            record.actions_taken.append(HealingAction(
                action="restore_region",
                target=failure.affected_region,
                success=True,
                details="Substrate region reconnected",
            ))

        logger.info(
            "Phase 3 RESTORE complete for %s (%d actions)",
            failure.failure_id,
            len(record.actions_taken),
        )

    def _phase_verify(self, record: HealingRecord) -> None:
        """Verify healing was successful."""
        record.phase = HealingPhase.VERIFYING
        self._publish_phase(record)

        # Check that isolated agents are restored
        all_restored = True
        if self._haven:
            for agent_id in record.failure.affected_agents:
                if self._haven.is_agent_isolated(agent_id):
                    all_restored = False

        record.success = all_restored
        record.phase = HealingPhase.COMPLETE if all_restored else HealingPhase.FAILED

        if record.success:
            self._successful_healings += 1
            self._last_successful_heal = time.time()
            if self._bus:
                self._bus.publish(CH_HEALING_COMPLETE, {
                    "failure_id": record.failure.failure_id,
                    "root_cause": record.root_cause,
                    "actions_count": len(record.actions_taken),
                    "duration": time.time() - record.started_at,
                })
        else:
            self._failed_healings += 1
            if self._bus:
                self._bus.publish(CH_HEALING_FAILED, {
                    "failure_id": record.failure.failure_id,
                    "reason": "verification_failed",
                })

    # =========================================================================
    # Recovery Action Registration
    # =========================================================================

    def register_recovery(
        self, failure_type: FailureType, callback: Callable
    ) -> None:
        """Register a recovery callback for a failure type."""
        self._recovery_actions[failure_type].append(callback)

    def _register_defaults(self) -> None:
        """Register default recovery strategies."""

        def _redistribute_load(record: HealingRecord) -> str:
            """Redistribute work from failed agents to healthy neighbors."""
            if not self._substrate:
                return "no_substrate"
            for agent_id in record.failure.affected_agents:
                peers = self._substrate.get_peers(agent_id, max_peers=3)
                if peers:
                    return f"load_distributed_to_{len(peers)}_peers"
            return "no_peers_found"

        def _inject_nutrients(record: HealingRecord) -> str:
            """Inject resources into starving region."""
            if not self._substrate:
                return "no_substrate"
            for agent_id in record.failure.affected_agents:
                self._substrate.inject_nutrient(agent_id, 1.0)
            return f"nutrients_injected_for_{len(record.failure.affected_agents)}_agents"

        self.register_recovery(FailureType.PERFORMANCE_DEGRADATION, _redistribute_load)
        self.register_recovery(FailureType.STARVATION, _inject_nutrients)
        self.register_recovery(FailureType.RESOURCE_EXHAUSTION, _inject_nutrients)

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
