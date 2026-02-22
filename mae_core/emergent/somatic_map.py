"""Somatic Map - Mae's awareness of her own body.

Biological analogy: The somatosensory cortex maintains a complete map
of the body (the cortical homunculus). Every nerve ending reports to
this map. Every motor command is checked against it. Before surgery,
doctors trace exactly which nerves and blood vessels will be affected.

Mae's Somatic Map is the equivalent:
1. Every system REGISTERS its dependencies (upstream/downstream)
2. Before ANY self-modification, the map computes BLAST RADIUS
3. Modifications are GATED until impact is assessed
4. If modification fails, ROLLBACK is immediate

This is NOT the AutoHealer (reactive). This is NOT the ThreatDetector
(external threats). This is Mae's PROPRIOCEPTION - her knowledge of
her own internal wiring, checked BEFORE every change.

The Somatic Map answers: "If I change X, what breaks?"

Connection points:
- Every system in Mae registers here on initialization
- CapabilityDiscovery checks blast radius before deploying new capabilities
- AutoHealer uses the map to understand cascade paths
- Morphogenesis checks impact before spawning/dissolving organs
- EventBus routes through Somatic Map for modification events

Biological inspirations:
- Somatosensory cortex (body map / homunculus)
- Proprioception (knowing where your limbs are without looking)
- Blood-brain barrier (critical systems have extra protection)
- Immune system MHC markers (self vs non-self recognition)
- Somatic nervous system (sensation before motor response)
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
CH_MODIFICATION_PROPOSED = "somatic.modification_proposed"
CH_MODIFICATION_APPROVED = "somatic.modification_approved"
CH_MODIFICATION_REJECTED = "somatic.modification_rejected"
CH_MODIFICATION_ROLLED_BACK = "somatic.modification_rolled_back"
CH_SYSTEM_REGISTERED = "somatic.system_registered"


class SystemCriticality(Enum):
    """How critical a system is - determines protection level."""

    PERIPHERAL = "peripheral"  # Can be modified freely (like skin cells)
    STANDARD = "standard"  # Normal protection (like muscle)
    PROTECTED = "protected"  # Extra validation required (like organs)
    CRITICAL = "critical"  # Blood-brain barrier level protection (like brainstem)


class ModificationVerdict(Enum):
    """Result of blast radius analysis."""

    APPROVED = "approved"  # Safe to proceed
    APPROVED_WITH_WARNINGS = "approved_with_warnings"  # Proceed with caution
    REJECTED = "rejected"  # Too dangerous
    NEEDS_REVIEW = "needs_review"  # Human/higher authority needed


@dataclass
class SystemNode:
    """A registered system in the somatic map."""

    system_id: str
    description: str
    criticality: SystemCriticality = SystemCriticality.STANDARD
    upstream: set[str] = field(default_factory=set)  # Systems this depends on
    downstream: set[str] = field(default_factory=set)  # Systems that depend on this
    health: float = 1.0  # [0, 1]
    registered_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BlastRadiusReport:
    """Analysis of what a modification would affect."""

    target_system: str
    direct_downstream: list[str]  # Immediately affected
    transitive_downstream: list[str]  # All affected (recursive)
    critical_systems_affected: list[str]  # CRITICAL systems in blast radius
    protected_systems_affected: list[str]  # PROTECTED systems in blast radius
    total_affected: int
    max_depth: int  # How deep the cascade goes
    risk_score: float  # [0, 1] overall risk
    verdict: ModificationVerdict
    warnings: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ModificationRecord:
    """Record of a proposed or executed modification."""

    modification_id: str
    target_system: str
    description: str
    blast_radius: Optional[BlastRadiusReport] = None
    approved: bool = False
    executed: bool = False
    rolled_back: bool = False
    snapshot: Optional[dict[str, Any]] = None  # Pre-modification state
    proposed_at: float = field(default_factory=time.time)
    executed_at: Optional[float] = None
    completed_at: Optional[float] = None


class SomaticMap:
    """Mae's proprioceptive body map - knows every connection.

    Every system registers here. Before any self-modification,
    the map computes blast radius. Modifications are gated.

    Like a surgeon's anatomy knowledge: you don't cut without
    knowing what's connected to what.
    """

    def __init__(
        self,
        event_bus: Any = None,
        max_blast_radius: int = 10,
        critical_system_threshold: float = 0.7,
        auto_reject_critical: bool = True,
    ) -> None:
        self._bus = event_bus
        self._max_blast = max_blast_radius
        self._critical_threshold = critical_system_threshold
        self._auto_reject_critical = auto_reject_critical

        # The body map: system_id -> SystemNode
        self._systems: dict[str, SystemNode] = {}

        # Modification history
        self._modifications: deque[ModificationRecord] = deque(maxlen=500)
        self._active_modifications: dict[str, ModificationRecord] = {}

        # Rollback snapshots: system_id -> snapshot callback
        self._snapshot_providers: dict[str, Callable[[], dict[str, Any]]] = {}
        self._rollback_handlers: dict[str, Callable[[dict[str, Any]], None]] = {}

        # Statistics
        self._total_proposals = 0
        self._approved = 0
        self._rejected = 0
        self._rollbacks = 0

        self._lock = threading.RLock()

        # Subscribe to EventBus channels for body awareness
        if self._bus:
            self._bus.register_callback(
                "healing.failure_detected", self._on_healing_failure
            )
            self._bus.register_callback(
                "morphogenesis.team_created", self._on_team_created
            )
            self._bus.register_callback(
                "defense.activated", self._on_defense_activated
            )

        logger.info("SomaticMap initialized (max_blast=%d)", max_blast_radius)

    # =========================================================================
    # System Registration (Building the Body Map)
    # =========================================================================

    def register_system(
        self,
        system_id: str,
        description: str,
        criticality: SystemCriticality = SystemCriticality.STANDARD,
        depends_on: Optional[list[str]] = None,
        depended_by: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SystemNode:
        """Register a system in the body map.

        Every system should call this on initialization so Mae knows
        her own anatomy.
        """
        with self._lock:
            node = SystemNode(
                system_id=system_id,
                description=description,
                criticality=criticality,
                metadata=metadata or {},
            )

            # Wire upstream dependencies
            if depends_on:
                for dep in depends_on:
                    node.upstream.add(dep)
                    # Also register the reverse connection
                    if dep in self._systems:
                        self._systems[dep].downstream.add(system_id)

            # Wire downstream dependents
            if depended_by:
                for dep in depended_by:
                    node.downstream.add(dep)
                    if dep in self._systems:
                        self._systems[dep].upstream.add(system_id)

            self._systems[system_id] = node

            if self._bus:
                self._bus.publish(CH_SYSTEM_REGISTERED, {
                    "system_id": system_id,
                    "criticality": criticality.value,
                    "upstream_count": len(node.upstream),
                    "downstream_count": len(node.downstream),
                })

            logger.info(
                "System registered: %s (%s, up=%d, down=%d)",
                system_id, criticality.value,
                len(node.upstream), len(node.downstream),
            )
            return node

    def register_all_systems(self, systems_dict: dict[str, Any]) -> None:
        """Convenience method to register many systems at once.

        Takes a dict of {name: system_ref} and registers each one
        as a STANDARD-criticality system.  Called from main.py at
        bootstrap time so that Mae has body awareness of all her parts.

        Args:
            systems_dict: Mapping of system name to system reference.
        """
        for name, system_ref in systems_dict.items():
            if name not in self._systems:
                self.register_system(name, f"{name} subsystem")
            # Store a reference for introspection (e.g. heartbeat proxies)
            node = self._systems.get(name)
            if node is not None:
                node.metadata["system_ref"] = system_ref

    def add_dependency(self, system_id: str, depends_on: str) -> None:
        """Add a dependency: system_id depends on depends_on."""
        with self._lock:
            if system_id in self._systems:
                self._systems[system_id].upstream.add(depends_on)
            if depends_on in self._systems:
                self._systems[depends_on].downstream.add(system_id)

    def register_snapshot_provider(
        self,
        system_id: str,
        snapshot_fn: Callable[[], dict[str, Any]],
        rollback_fn: Callable[[dict[str, Any]], None],
    ) -> None:
        """Register snapshot/rollback handlers for a system.

        snapshot_fn: Returns current state that can be restored.
        rollback_fn: Accepts a snapshot and restores state.
        """
        self._snapshot_providers[system_id] = snapshot_fn
        self._rollback_handlers[system_id] = rollback_fn

    def heartbeat(self, system_id: str, health: float = 1.0) -> None:
        """Report system health (called periodically by each system)."""
        with self._lock:
            if system_id in self._systems:
                self._systems[system_id].last_heartbeat = time.time()
                self._systems[system_id].health = max(0.0, min(1.0, health))

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
        target: SystemNode,
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

    # =========================================================================
    # Queries (Understanding the Body)
    # =========================================================================

    def get_system_info(self, system_id: str) -> Optional[SystemNode]:
        return self._systems.get(system_id)

    def get_all_systems(self) -> list[str]:
        return list(self._systems.keys())

    def get_dependency_chain(
        self, system_id: str, direction: str = "downstream"
    ) -> list[str]:
        """Get the full dependency chain in one direction."""
        visited: set[str] = set()
        result: list[str] = []
        queue = [system_id]

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            if current != system_id:
                result.append(current)

            node = self._systems.get(current)
            if node:
                deps = node.downstream if direction == "downstream" else node.upstream
                for dep in deps:
                    if dep not in visited:
                        queue.append(dep)

        return result

    def get_critical_path(self) -> list[str]:
        """Get all systems on the critical path (CRITICAL + PROTECTED)."""
        return [
            sid for sid, node in self._systems.items()
            if node.criticality in (SystemCriticality.CRITICAL, SystemCriticality.PROTECTED)
        ]

    def get_unhealthy_systems(self, threshold: float = 0.5) -> list[str]:
        """Get systems with health below threshold."""
        return [
            sid for sid, node in self._systems.items()
            if node.health < threshold
        ]

    # =========================================================================
    # Bootstrap Registration
    # =========================================================================

    def register_all_bootstrap_systems(self, systems_dict: dict[str, Any]) -> None:
        """Register all systems from the main.py bootstrap systems dict.

        Takes the systems dict produced by create_mae() and registers
        each known system in the body map with appropriate criticality.

        Args:
            systems_dict: The dict from main.py's create_mae(), e.g.
                {"event_bus": ..., "endocrine": ..., "substrate": ...}
        """
        # Criticality mapping for known system types
        criticality_map: dict[str, SystemCriticality] = {
            "event_bus": SystemCriticality.CRITICAL,
            "model": SystemCriticality.CRITICAL,
            "enforcer": SystemCriticality.CRITICAL,
            "watchdog": SystemCriticality.PROTECTED,
            "auditor": SystemCriticality.PROTECTED,
            "substrate": SystemCriticality.PROTECTED,
            "endocrine": SystemCriticality.STANDARD,
            "circadian": SystemCriticality.STANDARD,
        }

        for name, system in systems_dict.items():
            if system is None or name in ("agents", "triad_report"):
                continue
            criticality = criticality_map.get(name, SystemCriticality.PERIPHERAL)
            self.register_system(
                system_id=name,
                description=f"Bootstrap system: {name}",
                criticality=criticality,
            )

    # =========================================================================
    # EventBus Handlers (Body Awareness)
    # =========================================================================

    def _on_healing_failure(self, channel: str, message: Any) -> None:
        """Update body map when a healing failure is detected."""
        if isinstance(message, str):
            try:
                import json
                message = json.loads(message)
            except (json.JSONDecodeError, TypeError):
                return
        if not isinstance(message, dict):
            return

        failure_type = message.get("failure_type", "unknown")
        severity = message.get("severity", 0.5)
        affected_agents = message.get("affected_agents", [])

        # Reduce health of affected systems
        for agent_id in affected_agents:
            self.heartbeat(str(agent_id), health=max(0.0, 1.0 - severity))

    def _on_team_created(self, channel: str, message: Any) -> None:
        """Add new organ to body map when morphogenesis creates a team."""
        if isinstance(message, str):
            try:
                import json
                message = json.loads(message)
            except (json.JSONDecodeError, TypeError):
                return
        if not isinstance(message, dict):
            return

        organ_id = message.get("organ_id", "")
        if organ_id:
            self.register_system(
                system_id=organ_id,
                description=f"Morphogenesis organ: {message.get('name', organ_id)}",
                criticality=SystemCriticality.PERIPHERAL,
                metadata={"agent_count": message.get("agent_count", 0)},
            )

    def _on_defense_activated(self, channel: str, message: Any) -> None:
        """Track defense state in body awareness."""
        if isinstance(message, str):
            try:
                import json
                message = json.loads(message)
            except (json.JSONDecodeError, TypeError):
                return
        if not isinstance(message, dict):
            return

        # Update any defense-related system's health to reflect active state
        defense_system = message.get("system_id", "defense")
        if defense_system in self._systems:
            # Defense being active doesn't mean unhealthy - just record heartbeat
            self.heartbeat(defense_system, health=1.0)

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_statistics(self) -> dict[str, Any]:
        with self._lock:
            criticality_counts: dict[str, int] = defaultdict(int)
            for node in self._systems.values():
                criticality_counts[node.criticality.value] += 1

            total_deps = sum(
                len(n.upstream) + len(n.downstream) for n in self._systems.values()
            )

            return {
                "total_systems": len(self._systems),
                "criticality_breakdown": dict(criticality_counts),
                "total_dependencies": total_deps // 2,  # Each edge counted twice
                "total_proposals": self._total_proposals,
                "approved": self._approved,
                "rejected": self._rejected,
                "rollbacks": self._rollbacks,
                "active_modifications": len(self._active_modifications),
                "snapshot_providers": len(self._snapshot_providers),
            }

    def get_body_map(self) -> dict[str, dict[str, Any]]:
        """Get the full body map for visualization."""
        with self._lock:
            return {
                sid: {
                    "description": node.description,
                    "criticality": node.criticality.value,
                    "upstream": list(node.upstream),
                    "downstream": list(node.downstream),
                    "health": node.health,
                }
                for sid, node in self._systems.items()
            }
