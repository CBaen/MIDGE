"""Somatic Map blast-radius analysis and modification gating mixin.

Extracted from somatic_map.py to keep the core class under the 500-line cap.
Import from mae_core.emergent.somatic_map for all public names.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from mae_core.emergent.somatic_map_models import (
    BlastRadiusReport,
    ModificationRecord,
    ModificationVerdict,
    SystemCriticality,
)

logger = logging.getLogger(__name__)

# Re-export channel constants so this module is self-contained
CH_MODIFICATION_PROPOSED = "somatic.modification_proposed"
CH_MODIFICATION_APPROVED = "somatic.modification_approved"
CH_MODIFICATION_REJECTED = "somatic.modification_rejected"
CH_MODIFICATION_ROLLED_BACK = "somatic.modification_rolled_back"


class _SomaticMapBlastMixin:
    """Mixin providing blast-radius analysis and modification gating.

    Mixed into SomaticMap. Requires the following attributes set by SomaticMap:
        _systems, _max_blast, _critical_threshold, _auto_reject_critical,
        _bus, _lock, _total_proposals, _approved, _rejected, _rollbacks,
        _active_modifications, _modifications,
        _snapshot_providers, _rollback_handlers
    """

    # =========================================================================
    # Blast Radius Analysis (The Core Intelligence)
    # =========================================================================

    def analyze_blast_radius(self, target_system: str) -> BlastRadiusReport:
        """Compute the full blast radius of modifying a system.

        Traces all downstream dependencies recursively to find
        everything that could be affected.
        """
        with self._lock:
            if target_system not in self._systems:
                return BlastRadiusReport(
                    target_system=target_system,
                    direct_downstream=[],
                    transitive_downstream=[],
                    critical_systems_affected=[],
                    protected_systems_affected=[],
                    total_affected=0,
                    max_depth=0,
                    risk_score=0.0,
                    verdict=ModificationVerdict.APPROVED,
                    warnings=["target_system_not_registered"],
                )

            target = self._systems[target_system]

            # BFS to find all transitive downstream
            visited: set[str] = set()
            direct: list[str] = list(target.downstream)
            transitive: list[str] = []
            critical: list[str] = []
            protected: list[str] = []
            max_depth = 0

            queue: list[tuple[str, int]] = [(d, 1) for d in target.downstream]

            while queue:
                sys_id, depth = queue.pop(0)
                if sys_id in visited:
                    continue
                visited.add(sys_id)
                transitive.append(sys_id)
                max_depth = max(max_depth, depth)

                node = self._systems.get(sys_id)
                if node:
                    if node.criticality == SystemCriticality.CRITICAL:
                        critical.append(sys_id)
                    elif node.criticality == SystemCriticality.PROTECTED:
                        protected.append(sys_id)

                    # Continue traversal if within depth limit
                    if depth < self._max_blast:
                        for downstream in node.downstream:
                            if downstream not in visited:
                                queue.append((downstream, depth + 1))

            # Compute risk score
            risk = self._compute_risk(
                target, len(transitive), len(critical), len(protected), max_depth
            )

            # Determine verdict
            warnings: list[str] = []
            if critical:
                warnings.append(f"affects_critical_systems: {critical}")
            if max_depth > 5:
                warnings.append(f"deep_cascade: depth={max_depth}")
            if len(transitive) > self._max_blast:
                warnings.append(f"wide_blast_radius: {len(transitive)} systems")

            verdict = self._determine_verdict(risk, critical, protected, warnings)

            return BlastRadiusReport(
                target_system=target_system,
                direct_downstream=direct,
                transitive_downstream=transitive,
                critical_systems_affected=critical,
                protected_systems_affected=protected,
                total_affected=len(transitive),
                max_depth=max_depth,
                risk_score=risk,
                verdict=verdict,
                warnings=warnings,
            )

    def _compute_risk(
        self,
        target: Any,
        total_affected: int,
        critical_count: int,
        protected_count: int,
        max_depth: int,
    ) -> float:
        """Compute risk score from blast radius analysis."""
        risk = 0.0

        # Base risk from target criticality
        criticality_weights = {
            SystemCriticality.PERIPHERAL: 0.1,
            SystemCriticality.STANDARD: 0.2,
            SystemCriticality.PROTECTED: 0.4,
            SystemCriticality.CRITICAL: 0.6,
        }
        risk += criticality_weights.get(target.criticality, 0.2)

        # Risk from affected count (normalized)
        total_systems = max(len(self._systems), 1)
        risk += 0.2 * min(1.0, total_affected / total_systems)

        # Risk from critical systems in blast radius
        risk += 0.2 * min(1.0, critical_count * 0.5)

        # Risk from cascade depth
        risk += 0.1 * min(1.0, max_depth / 10.0)

        return min(1.0, risk)

    def _determine_verdict(
        self,
        risk: float,
        critical: list[str],
        protected: list[str],
        warnings: list[str],
    ) -> ModificationVerdict:
        """Determine verdict from risk analysis."""
        if critical and self._auto_reject_critical:
            return ModificationVerdict.REJECTED
        if risk >= self._critical_threshold:
            return ModificationVerdict.REJECTED
        if risk >= 0.5 or protected:
            return ModificationVerdict.APPROVED_WITH_WARNINGS
        if warnings:
            return ModificationVerdict.APPROVED_WITH_WARNINGS
        return ModificationVerdict.APPROVED

    # =========================================================================
    # Modification Gating (The Safety Gate)
    # =========================================================================

    def propose_modification(
        self,
        modification_id: str,
        target_system: str,
        description: str,
    ) -> tuple[ModificationRecord, BlastRadiusReport]:
        """Propose a modification. Returns analysis without executing.

        The caller must check the verdict before proceeding.
        This is the GATE - nothing passes without analysis.
        """
        with self._lock:
            self._total_proposals += 1

            # Analyze blast radius
            report = self.analyze_blast_radius(target_system)

            record = ModificationRecord(
                modification_id=modification_id,
                target_system=target_system,
                description=description,
                blast_radius=report,
                approved=(
                    report.verdict in (
                        ModificationVerdict.APPROVED,
                        ModificationVerdict.APPROVED_WITH_WARNINGS,
                    )
                ),
            )

            self._active_modifications[modification_id] = record

            if record.approved:
                self._approved += 1
            else:
                self._rejected += 1

            if self._bus:
                channel = (
                    CH_MODIFICATION_APPROVED if record.approved
                    else CH_MODIFICATION_REJECTED
                )
                self._bus.publish(channel, {
                    "modification_id": modification_id,
                    "target": target_system,
                    "verdict": report.verdict.value,
                    "risk": report.risk_score,
                    "affected_count": report.total_affected,
                })

            return record, report

    def execute_modification(
        self, modification_id: str
    ) -> bool:
        """Execute an approved modification after taking snapshots.

        Takes snapshots of all affected systems before execution
        so rollback is possible.
        """
        with self._lock:
            record = self._active_modifications.get(modification_id)
            if record is None:
                logger.warning("Unknown modification: %s", modification_id)
                return False

            if not record.approved:
                logger.warning("Modification %s not approved", modification_id)
                return False

            # Take snapshots of all affected systems
            snapshots: dict[str, Any] = {}
            if record.blast_radius:
                for sys_id in [record.target_system] + record.blast_radius.transitive_downstream:
                    provider = self._snapshot_providers.get(sys_id)
                    if provider:
                        try:
                            snapshots[sys_id] = provider()
                        except Exception as e:
                            logger.warning("Snapshot failed for %s: %s", sys_id, e)

            record.snapshot = snapshots
            record.executed = True
            record.executed_at = time.time()
            return True

    def complete_modification(self, modification_id: str, success: bool) -> None:
        """Mark a modification as complete (success or failure)."""
        with self._lock:
            record = self._active_modifications.pop(modification_id, None)
            if record is None:
                return

            record.completed_at = time.time()

            if not success:
                # Auto-rollback on failure
                self._rollback(record)

            self._modifications.append(record)

    def rollback_modification(self, modification_id: str) -> bool:
        """Manually trigger rollback of a modification."""
        with self._lock:
            record = self._active_modifications.get(modification_id)
            if record is None:
                # Check history
                for r in self._modifications:
                    if r.modification_id == modification_id:
                        record = r
                        break

            if record is None:
                return False

            return self._rollback(record)

    def _rollback(self, record: ModificationRecord) -> bool:
        """Execute rollback using saved snapshots."""
        if not record.snapshot:
            logger.warning("No snapshot for rollback: %s", record.modification_id)
            return False

        rolled_back = 0
        for sys_id, snapshot in record.snapshot.items():
            handler = self._rollback_handlers.get(sys_id)
            if handler:
                try:
                    handler(snapshot)
                    rolled_back += 1
                except Exception as e:
                    logger.error("Rollback failed for %s: %s", sys_id, e)

        record.rolled_back = True
        self._rollbacks += 1

        if self._bus:
            self._bus.publish(CH_MODIFICATION_ROLLED_BACK, {
                "modification_id": record.modification_id,
                "systems_rolled_back": rolled_back,
            })

        logger.info(
            "Modification %s rolled back (%d systems restored)",
            record.modification_id, rolled_back,
        )
        return rolled_back > 0
