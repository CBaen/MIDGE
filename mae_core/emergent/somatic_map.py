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

Implementation is split across sub-modules for the 500-line cap:
  somatic_map_models.py -- SystemCriticality, ModificationVerdict,
                           SystemNode, BlastRadiusReport, ModificationRecord
  somatic_map_blast.py  -- _SomaticMapBlastMixin (blast radius + modification gating)
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from mae_core.emergent.somatic_map_models import (
    BlastRadiusReport,
    ModificationRecord,
    ModificationVerdict,
    SystemCriticality,
    SystemNode,
)
from mae_core.emergent.somatic_map_blast import (
    _SomaticMapBlastMixin,
    CH_MODIFICATION_APPROVED,
    CH_MODIFICATION_PROPOSED,
    CH_MODIFICATION_REJECTED,
    CH_MODIFICATION_ROLLED_BACK,
)

logger = logging.getLogger(__name__)

# EventBus channels (kept here for backward-compatible import)
CH_SYSTEM_REGISTERED = "somatic.system_registered"
# CH_MODIFICATION_* re-exported from somatic_map_blast above


class SomaticMap(_SomaticMapBlastMixin):
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
