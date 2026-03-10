"""Auto-Healer self-monitoring mixin (Law 6: Autopoietic Closure).

Extracted from auto_healer.py to keep the core class under the 500-line cap.
Import from mae_core.emergent.auto_healer for all public names.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Re-import the channel constant so this module is self-contained
CH_HEALING_SELF_HEALED = "healing.self_healed"


class _AutoHealerMonitorMixin:
    """Mixin providing the Meta-Healing Triad (Law 6: Autopoietic Closure).

    The healer monitors its own health via three indicators:
    1. Scan staleness — has the healer run a scan recently?
    2. Queue overflow — is the failure history growing unboundedly?
    3. Detection blindness — is the healer detecting anything at all?

    Mixed into AutoHealer. Requires the following attributes set by AutoHealer:
        _step_count, _last_scan_step, _scan_interval, _scan_staleness_threshold,
        _history, _max_failure_queue_size, _detection_blindness_threshold,
        _total_detections, _last_detection_step, _health_threshold,
        _self_heal_count, _bus, _lock, _somatic_map
    """

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
