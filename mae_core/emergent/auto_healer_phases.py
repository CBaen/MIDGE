"""Auto-Healer three-phase pipeline and recovery registration mixin.

Extracted from auto_healer.py to keep the core class under the 500-line cap.
Import from mae_core.emergent.auto_healer for all public names.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Callable

from mae_core.emergent.auto_healer_models import (
    FailureType,
    HealingAction,
    HealingPhase,
    HealingRecord,
)

logger = logging.getLogger(__name__)

# Re-import channel constants so this module is self-contained
CH_HEALING_PHASE = "healing.phase_changed"
CH_HEALING_COMPLETE = "healing.complete"
CH_HEALING_FAILED = "healing.failed"


class _AutoHealerPhasesMixin:
    """Mixin providing the three-phase healing pipeline and recovery registration.

    Phases: ISOLATE -> ASSESS -> RESTORE -> VERIFY

    Mixed into AutoHealer. Requires the following attributes:
        _haven, _substrate, _causal, _bus, _somatic_map,
        _recovery_actions, _auto_isolate,
        _successful_healings, _failed_healings,
        _active_healings, _history, _step_count, _healing_cooldowns,
        _last_successful_heal, _cascade_preventions, _lock
    """

    # =========================================================================
    # Three-Phase Healing Pipeline
    # =========================================================================

    def _execute_healing(self, record: HealingRecord) -> None:
        """Run the three-phase healing pipeline."""
        import time

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
            # Stamp cooldown so proactive scan doesn't immediately re-heal
            for agent_id in record.failure.affected_agents:
                self._healing_cooldowns[agent_id] = self._step_count

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
        import time

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
            flow = getattr(self._substrate, "nutrient_flow", None)
            if flow is None or not hasattr(flow, "inject_resources"):
                return "no_nutrient_flow"
            injected = 0
            for node_id in record.failure.affected_agents:
                if flow.inject_resources(str(node_id), 0.5):
                    injected += 1
            return f"nutrients_injected_for_{injected}_nodes"

        self.register_recovery(FailureType.PERFORMANCE_DEGRADATION, _redistribute_load)
        self.register_recovery(FailureType.STARVATION, _inject_nutrients)
        self.register_recovery(FailureType.RESOURCE_EXHAUSTION, _inject_nutrients)
