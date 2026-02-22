"""Octopus Colony - Peer-to-peer multi-octopus network.

NO HIERARCHIES. All octopuses are peers. Coordination emerges
from connections, not from central authority.

Rule of 3:
- Minimum 3 octopuses in any colony
- Each octopus connects to 2-3 peers (2 if only 3 exist, 3+ otherwise)
- Consensus requires minimum 3 votes
- Odd peer counts prevent stalemates

Auto-scaling:
- Spawn when average workload > threshold (default 80%)
- Despawn when average workload < threshold (default 20%)
- Never violate Rule of 3 minimum

Self-healing:
- Replace unhealthy octopuses (health < 30%)
- Spawn replacement BEFORE despawning unhealthy (maintain minimum)
- Re-establish peer connections after topology changes

Biological analogy: Octopus colony as mycelial network.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Final

from ..backbone.event_bus import EventBus
from .octopus_agent import OctopusAgent
from .octopus_signals import (
    OctopusSpecialization,
    CH_OCTOPUS_SPAWN,
    CH_OCTOPUS_DESPAWN,
    CH_OCTOPUS_HEALTH,
)

logger = logging.getLogger(__name__)

# Rule of 3 - Static structural constants
MIN_AGENTS: Final[int] = 3
MIN_CONNECTIONS: Final[int] = 3  # Target; 2 acceptable if only 3 agents
MIN_VOTES: Final[int] = 3

# Rule of 3 - Dynamic learning constants
MIN_LEARNING_STEPS: Final[int] = 4  # Baseline + 3 recurrences
MIN_MEMORY_RECURRENCES: Final[int] = 3
BASELINE_STEP: Final[int] = 0


def get_min_connections(total_agents: int) -> int:
    """Minimum peer connections based on colony size."""
    if total_agents < MIN_AGENTS:
        raise ValueError(f"Need at least {MIN_AGENTS} agents (Rule of 3)")
    if total_agents == 3:
        return 2  # Can only connect to 2 others
    return MIN_CONNECTIONS


def validate_rule_of_3(
    agent_count: int | None = None,
    connections_per_agent: int | None = None,
    vote_count: int | None = None,
) -> bool:
    """Validate Rule of 3 compliance. Raises ValueError on violation."""
    violations: list[str] = []

    if agent_count is not None and agent_count < MIN_AGENTS:
        violations.append(
            f"Agent count {agent_count} < {MIN_AGENTS} (Rule of 3)"
        )
    if connections_per_agent is not None and connections_per_agent < 2:
        violations.append(
            f"Connections {connections_per_agent} < 2 (Rule of 3 minimum)"
        )
    if vote_count is not None and vote_count < MIN_VOTES:
        violations.append(
            f"Vote count {vote_count} < {MIN_VOTES} (Rule of 3)"
        )

    if violations:
        raise ValueError("\n".join(violations))
    return True


class OctopusColony:
    """Peer-to-peer octopus network. NO HIERARCHIES.

    Topology: Ring + cross-connections.
    - Ring: Each octopus connects to next + previous
    - Cross: For n>=3, also connect 2 steps ahead
    - Result: 3-4 peers each (odd preferred for consensus)

    Lifecycle events published on EventBus:
    - octopus.spawn: New octopus joined the colony
    - octopus.despawn: Octopus left the colony
    - octopus.health_report: Periodic colony health snapshot
    """

    def __init__(
        self,
        event_bus: EventBus | None = None,
        min_octopuses: int = MIN_AGENTS,
        max_octopuses: int = 10,
        spawn_threshold: float = 0.8,
        despawn_threshold: float = 0.2,
        health_threshold: float = 0.3,
        monitoring_interval: float = 5.0,
        decision_router: Any | None = None,
        world_model: Any | None = None,
        signal_bus: Any | None = None,
    ) -> None:
        validate_rule_of_3(agent_count=min_octopuses)

        self._bus = event_bus
        self.min_octopuses = min_octopuses
        self.max_octopuses = max_octopuses
        self.spawn_threshold = spawn_threshold
        self.despawn_threshold = despawn_threshold
        self.health_threshold = health_threshold
        self._monitoring_interval = monitoring_interval

        # Cross-system integrations (passed to spawned octopuses)
        self._decision_router = decision_router
        self._world_model = world_model
        self._signal_bus = signal_bus

        # Colony registry (all peers, no hierarchy)
        self.octopuses: dict[str, OctopusAgent] = {}
        self._octopus_counter = 0

        # Peer connections: octopus_id -> set of peer_ids
        self.peer_connections: dict[str, set[str]] = {}

        # History
        self.spawn_history: list[dict[str, Any]] = []
        self.despawn_history: list[dict[str, Any]] = []

        # Monitoring
        self._monitoring_thread: threading.Thread | None = None
        self._running = False

        # Initialize with minimum octopuses
        self._initialize_colony()

    def _initialize_colony(self) -> None:
        """Spawn minimum octopuses and establish peer connections."""
        for _ in range(self.min_octopuses):
            self.spawn_octopus(
                specialization=OctopusSpecialization.GENERAL,
                reason="colony_initialization",
            )
        self._establish_peer_connections()

    # --- Public API ---

    def spawn_octopus(
        self,
        specialization: OctopusSpecialization = OctopusSpecialization.GENERAL,
        reason: str = "manual_spawn",
    ) -> str | None:
        """Spawn a new octopus as a peer. Returns octopus_id or None."""
        if len(self.octopuses) >= self.max_octopuses:
            logger.warning("Cannot spawn - at max capacity (%d)", self.max_octopuses)
            return None

        octopus_id = f"octopus_{self._octopus_counter}"
        self._octopus_counter += 1

        octopus = OctopusAgent(
            octopus_id=octopus_id,
            event_bus=self._bus,
            specialization=specialization,
            decision_router=self._decision_router,
            world_model=self._world_model,
            signal_bus=self._signal_bus,
        )
        octopus.start()

        self.octopuses[octopus_id] = octopus
        self.peer_connections[octopus_id] = set()

        # Reconnect topology with new member
        if len(self.octopuses) > 1:
            self._establish_peer_connections()

        spawn_event = {
            "octopus_id": octopus_id,
            "specialization": specialization.value,
            "reason": reason,
            "timestamp": time.time(),
            "colony_size": len(self.octopuses),
        }
        self.spawn_history.append(spawn_event)

        if self._bus:
            self._bus.publish(CH_OCTOPUS_SPAWN, spawn_event)

        logger.info(
            "Spawned %s (%s) - reason: %s - colony size: %d",
            octopus_id, specialization.value, reason, len(self.octopuses),
        )
        return octopus_id

    def despawn_octopus(self, octopus_id: str, reason: str = "manual_despawn") -> bool:
        """Remove an octopus from the colony. Respects Rule of 3 minimum."""
        if octopus_id not in self.octopuses:
            logger.warning("Cannot despawn %s - not found", octopus_id)
            return False

        if len(self.octopuses) <= self.min_octopuses:
            logger.warning(
                "Cannot despawn %s - at minimum (%d, Rule of 3)",
                octopus_id, self.min_octopuses,
            )
            return False

        octopus = self.octopuses[octopus_id]
        octopus.stop()

        # Remove from peer connections (both directions)
        if octopus_id in self.peer_connections:
            for peer_id in self.peer_connections[octopus_id]:
                if peer_id in self.peer_connections:
                    self.peer_connections[peer_id].discard(octopus_id)
            del self.peer_connections[octopus_id]

        del self.octopuses[octopus_id]

        # Reconnect remaining topology
        self._establish_peer_connections()

        despawn_event = {
            "octopus_id": octopus_id,
            "specialization": octopus.specialization.value,
            "reason": reason,
            "timestamp": time.time(),
            "lifetime": time.time() - octopus.spawn_time,
            "tasks_completed": octopus.tasks_completed,
            "colony_size": len(self.octopuses),
        }
        self.despawn_history.append(despawn_event)

        if self._bus:
            self._bus.publish(CH_OCTOPUS_DESPAWN, despawn_event)

        logger.info(
            "Despawned %s - reason: %s - colony size: %d",
            octopus_id, reason, len(self.octopuses),
        )
        return True

    def submit_task(
        self,
        task_data: dict[str, Any],
        task_type: str,
        priority: int = 5,
    ) -> str | None:
        """Submit task via emergent routing (least-loaded peer). No central router."""
        if not self.octopuses:
            logger.error("No octopuses in colony")
            return None

        for octopus in self.octopuses.values():
            octopus.update_metrics()

        least_loaded = min(self.octopuses.values(), key=lambda o: o.workload)
        task_id = least_loaded.submit_task(task_data, task_type, priority)

        logger.debug(
            "Task %s -> %s (workload=%.2f)",
            task_id, least_loaded.octopus_id, least_loaded.workload,
        )
        return task_id

    def start_monitoring(self) -> None:
        """Start background health and auto-scaling monitor."""
        if self._running:
            return
        self._running = True
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="OctopusColonyMonitor",
        )
        self._monitoring_thread.start()

    def stop_monitoring(self) -> None:
        """Stop the monitoring thread."""
        self._running = False
        if self._monitoring_thread is not None:
            self._monitoring_thread.join(timeout=10.0)
            self._monitoring_thread = None

    def stop_all(self) -> None:
        """Stop monitoring and all octopuses."""
        self.stop_monitoring()
        for octopus in self.octopuses.values():
            octopus.stop()

    def get_colony_status(self) -> dict[str, Any]:
        """Comprehensive colony status report."""
        for octopus in self.octopuses.values():
            octopus.update_metrics()

        octopus_statuses = {}
        for oid, oct in self.octopuses.items():
            octopus_statuses[oid] = {
                "specialization": oct.specialization.value,
                "health": oct.health,
                "workload": oct.workload,
                "tasks_completed": oct.tasks_completed,
                "tasks_failed": oct.tasks_failed,
                "uptime": time.time() - oct.spawn_time,
                "peers": sorted(self.peer_connections.get(oid, set())),
                "peer_count": len(self.peer_connections.get(oid, set())),
            }

        avg_workload = (
            sum(o.workload for o in self.octopuses.values()) / len(self.octopuses)
            if self.octopuses else 0.0
        )
        avg_health = (
            sum(o.health for o in self.octopuses.values()) / len(self.octopuses)
            if self.octopuses else 0.0
        )

        all_have_min_peers = all(
            len(self.peer_connections.get(oid, set())) >= 2
            for oid in self.octopuses
        )

        return {
            "colony_size": len(self.octopuses),
            "min_octopuses": self.min_octopuses,
            "max_octopuses": self.max_octopuses,
            "rule_of_3_compliant": len(self.octopuses) >= MIN_AGENTS,
            "peer_connectivity_ok": all_have_min_peers,
            "average_workload": avg_workload,
            "average_health": avg_health,
            "octopuses": octopus_statuses,
            "total_spawns": len(self.spawn_history),
            "total_despawns": len(self.despawn_history),
            "monitoring_active": self._running,
            "network_type": "peer-to-peer",
        }

    # --- Internal ---

    def _establish_peer_connections(self) -> None:
        """Ring + cross-connections for Rule of 3 compliance.

        Ring: each -> next + previous (bidirectional)
        Cross: for n>=3, also connect 2 steps ahead
        Result: 3-4 peers each (odd count preferred for consensus)
        """
        octopus_ids = list(self.octopuses.keys())
        n = len(octopus_ids)
        if n < 2:
            return

        # Reset connections
        for oid in octopus_ids:
            self.peer_connections[oid] = set()

        for i, oid in enumerate(octopus_ids):
            # Ring: next + previous
            next_id = octopus_ids[(i + 1) % n]
            prev_id = octopus_ids[(i - 1) % n]

            self.peer_connections[oid].add(next_id)
            self.peer_connections[next_id].add(oid)
            self.peer_connections[oid].add(prev_id)
            self.peer_connections[prev_id].add(oid)

            # Cross-connection for Rule of 3
            if n >= 3:
                cross_id = octopus_ids[(i + 2) % n]
                if cross_id != oid:
                    self.peer_connections[oid].add(cross_id)
                    self.peer_connections[cross_id].add(oid)

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

        avg_health = (
            sum(o.health for o in self.octopuses.values()) / len(self.octopuses)
            if self.octopuses else 0.0
        )
        avg_workload = (
            sum(o.workload for o in self.octopuses.values()) / len(self.octopuses)
            if self.octopuses else 0.0
        )

        self._bus.publish(CH_OCTOPUS_HEALTH, {
            "colony_size": len(self.octopuses),
            "average_health": avg_health,
            "average_workload": avg_workload,
            "rule_of_3_compliant": len(self.octopuses) >= MIN_AGENTS,
            "timestamp": time.time(),
        })

    def __repr__(self) -> str:
        return (
            f"OctopusColony(peers={len(self.octopuses)}, "
            f"min={self.min_octopuses}, max={self.max_octopuses})"
        )
