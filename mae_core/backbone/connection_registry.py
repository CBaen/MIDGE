"""Connection Registry - Triadic witnessing for every connection in Mae.

Biological analogy: The lymphatic system. Every blood vessel (connection)
has lymph nodes (witnesses) that monitor what flows through. A bare
blood vessel with no immune surveillance is a vulnerability. The lymphatic
system doesn't block flow - it watches, verifies, and reports.

Mae's fractal blueprint says "no bare dyads" - every connection A-B must
have a witness C. This module registers every system-to-system connection
as a ConnectionTriad and periodically verifies they're alive and witnessed.

The Connection Law:
  - Primary pathway: A -> B (direct signal)
  - Verification pathway: A -> C -> B (witness checks primary)
  - Balance pathway: B -> C -> A (feedback loop)

This creates: non-repudiation, tamper detection, fault isolation,
consensus, systemic memory.

Enforcement modes:
  - PERMISSIVE: Bootstrap phase. No checks. Everything passes.
  - ADVISORY: Log + event on violations. Nothing blocked. (default)
  - BLOCKING: Reject bare dyads at registration. Disable unhealthy
    connections. Query API returns False for violations.

Connection points:
- Created in main.py Layer 18 (after SomaticMap registration)
- Uses SomaticMap topology for intelligent witness assignment
- TriadWatchdog queries for bare dyad detection
- Publishes connection events on EventBus
- seal() called after bootstrap to activate enforcement
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)

# EventBus channels
CH_CONNECTION_REGISTERED = "connection.registered"
CH_CONNECTION_VERIFIED = "connection.verified"
CH_CONNECTION_BARE_DYAD = "connection.bare_dyad"
CH_CONNECTION_HEALTH = "connection.health"
CH_CONNECTION_BLOCKED = "connection.blocked"
CH_CONNECTION_SEALED = "connection.sealed"

# Nervous system witnesses (always available as fallback)
NERVOUS_SYSTEM = ("enforcer", "watchdog", "auditor", "somatic_map")


class ConnectionType(Enum):
    """How two systems are connected."""

    EVENTBUS_PUBSUB = "eventbus_pubsub"
    DIRECT_REFERENCE = "direct_reference"
    CALLBACK_REGISTRATION = "callback_registration"
    STEP_HOOK = "step_hook"
    MEMORY_DATA_FLOW = "memory_data_flow"
    SUBSTRATE_INTEGRATION = "substrate_integration"


class ConnectionCriticality(Enum):
    """How important a connection is to Mae's operation."""

    STANDARD = "standard"
    IMPORTANT = "important"
    CRITICAL = "critical"


class EnforcementMode(Enum):
    """How strictly the registry enforces triadic witnessing."""

    PERMISSIVE = "permissive"  # Bootstrap: no checks at all
    ADVISORY = "advisory"  # Log + event, allow everything
    BLOCKING = "blocking"  # Reject bare dyads, disable unhealthy


@dataclass
class ConnectionTriad:
    """A witnessed connection between two systems.

    Every connection in Mae has three+ parties:
    - source: the system that initiates/provides
    - target: the system that receives/consumes
    - witnesses: systems that monitor the connection (minimum 2 for Law 1)

    Law 1 requires: primary pathway (A->B), verification pathway (A->C->B),
    balance pathway (B->C->A). Two witnesses provide redundant oversight
    and eliminate single-witness fragility.
    """

    connection_id: str
    source: str
    target: str
    witnesses: list[str] = field(default_factory=list)
    connection_type: ConnectionType = ConnectionType.DIRECT_REFERENCE
    channel: Optional[str] = None  # EventBus channel name, if applicable
    criticality: ConnectionCriticality = ConnectionCriticality.STANDARD
    description: str = ""
    registered_at: float = field(default_factory=time.time)
    last_verified: float = 0.0
    healthy: bool = True

    @property
    def witness(self) -> Optional[str]:
        """Backward compat: first witness or None."""
        return self.witnesses[0] if self.witnesses else None


class ConnectionRegistry:
    """Registry of all system-to-system connections with triadic witnessing.

    Every connection between Mae's systems is registered here.
    Each gets an automatically assigned witness based on SomaticMap
    topology. Periodic verification checks that connections and
    witnesses are alive.

    Think of it as a building inspection registry: every room
    (connection) has an assigned inspector (witness), and periodic
    walk-throughs verify everything is in order.
    """

    def __init__(
        self,
        event_bus: Any = None,
        somatic_map: Any = None,
        verify_interval: int = 25,
        enforcement_mode: EnforcementMode = EnforcementMode.ADVISORY,
    ) -> None:
        self._bus = event_bus
        self._somatic_map = somatic_map
        self._verify_interval = verify_interval

        # Enforcement state
        self._post_seal_mode = enforcement_mode
        self._active_mode = EnforcementMode.PERMISSIVE  # Start permissive during bootstrap
        self._sealed = False

        # connection_id -> ConnectionTriad
        self._connections: dict[str, ConnectionTriad] = {}

        # Track which systems participate in connections
        self._system_connections: dict[str, list[str]] = {}  # system -> [connection_ids]

        self._lock = threading.RLock()
        self._step_counter = 0

        # Statistics
        self._total_verifications = 0
        self._total_failures = 0
        self._blocked_count = 0

        # Operational witnessing (set after bootstrap by WitnessNotifier)
        self._witness_notifier: Any = None

        # Channel-based index for O(1) fallback lookup by channel name.
        # Handles source/target name mismatches between EventBus publishers
        # (which extract channel prefix as source) and registrations
        # (which use system names as source).
        self._channel_index: dict[str, str] = {}  # channel -> first conn_id

        logger.info(
            "ConnectionRegistry initialized (verify_interval=%d, post_seal=%s)",
            verify_interval, enforcement_mode.value,
        )

    # =========================================================================
    # Enforcement
    # =========================================================================

    def seal(self) -> None:
        """End bootstrap grace period. Transition to configured enforcement mode.

        Called after register_all_connections() in main.py Layer 18.
        Idempotent — second call is a no-op.
        """
        if self._sealed:
            return
        self._sealed = True
        self._active_mode = self._post_seal_mode
        logger.info(
            "ConnectionRegistry sealed: enforcement=%s (%d connections)",
            self._active_mode.value, len(self._connections),
        )
        if self._bus:
            self._bus.publish(CH_CONNECTION_SEALED, {
                "enforcement_mode": self._active_mode.value,
                "total_connections": len(self._connections),
            })

    def set_enforcement_mode(self, mode: EnforcementMode) -> None:
        """Switch enforcement mode at runtime."""
        old = self._active_mode
        self._active_mode = mode
        if self._sealed:
            self._post_seal_mode = mode
        logger.info(
            "ConnectionRegistry: enforcement %s -> %s",
            old.value, mode.value,
        )

    @property
    def enforcement_mode(self) -> EnforcementMode:
        """Current enforcement mode."""
        return self._active_mode

    @property
    def sealed(self) -> bool:
        """Whether bootstrap grace period is over."""
        return self._sealed

    def is_connection_allowed(
        self,
        source: str,
        target: str,
        channel: str | None = None,
    ) -> tuple[bool, str]:
        """Check if communication from source to target is allowed.

        Returns (allowed, reason). In PERMISSIVE/unsealed mode, always
        returns True. In ADVISORY, always True but logs violations.
        In BLOCKING, returns False for unregistered/unhealthy/bare dyads.
        """
        if not self._sealed or self._active_mode == EnforcementMode.PERMISSIVE:
            return (True, "permissive")

        # Try exact match first (with channel), then without channel
        conn_id = f"{source}->{target}:{channel}" if channel else f"{source}->{target}"
        with self._lock:
            triad = self._connections.get(conn_id)
            if triad is None and channel:
                # Fallback: check without channel
                triad = self._connections.get(f"{source}->{target}")
            if triad is None and channel:
                # Fallback: channel index (handles source/target name mismatches
                # between EventBus publishers and registered system names)
                indexed_id = self._channel_index.get(channel)
                if indexed_id:
                    triad = self._connections.get(indexed_id)

        if triad is None:
            if self._active_mode == EnforcementMode.BLOCKING:
                self._blocked_count += 1
                logger.error(
                    "BLOCKED: unregistered connection %s -> %s", source, target,
                )
                if self._bus:
                    self._bus.publish(CH_CONNECTION_BLOCKED, {
                        "source": source, "target": target,
                        "reason": "unregistered",
                    })
                return (False, "unregistered")
            # ADVISORY
            logger.warning(
                "Advisory: unregistered connection %s -> %s", source, target,
            )
            return (True, "unregistered_advisory")

        if not triad.healthy:
            if self._active_mode == EnforcementMode.BLOCKING:
                self._blocked_count += 1
                logger.error(
                    "BLOCKED: unhealthy connection %s", triad.connection_id,
                )
                if self._bus:
                    self._bus.publish(CH_CONNECTION_BLOCKED, {
                        "connection_id": triad.connection_id,
                        "reason": "unhealthy",
                    })
                return (False, "unhealthy")
            logger.warning(
                "Advisory: unhealthy connection %s", triad.connection_id,
            )
            return (True, "unhealthy_advisory")

        if not triad.witnesses:
            if self._active_mode == EnforcementMode.BLOCKING:
                self._blocked_count += 1
                logger.error(
                    "BLOCKED: bare dyad %s", triad.connection_id,
                )
                if self._bus:
                    self._bus.publish(CH_CONNECTION_BLOCKED, {
                        "connection_id": triad.connection_id,
                        "reason": "bare_dyad",
                    })
                return (False, "bare_dyad")
            logger.warning(
                "Advisory: bare dyad %s", triad.connection_id,
            )
            return (True, "bare_dyad_advisory")

        return (True, "ok")

    # =========================================================================
    # Registration
    # =========================================================================

    def register_connection(
        self,
        source: str,
        target: str,
        connection_type: ConnectionType,
        channel: Optional[str] = None,
        criticality: ConnectionCriticality = ConnectionCriticality.STANDARD,
        description: str = "",
        witness: Optional[str] = None,
        witnesses: Optional[list[str]] = None,
    ) -> ConnectionTriad:
        """Register a connection between two systems.

        Law 1 requires at least 2 witnesses per connection. Pass
        ``witnesses=["w1", "w2"]`` for explicit assignment, or let
        auto-assignment pick 2 via SomaticMap topology heuristics.

        The legacy ``witness`` parameter is merged into the list for
        backward compatibility.
        """
        # Merge witness / witnesses into a single list
        witness_list: list[str] = list(witnesses or [])
        if witness and witness not in witness_list:
            witness_list.insert(0, witness)

        # Build connection ID
        suffix = f":{channel}" if channel else ""
        conn_id = f"{source}->{target}{suffix}"

        with self._lock:
            if conn_id in self._connections:
                return self._connections[conn_id]

            # Auto-fill to 2 witnesses if under-specified
            if len(witness_list) < 2:
                auto = self._assign_witnesses(
                    source, target, exclude=set(witness_list),
                )
                for w in auto:
                    if w not in witness_list:
                        witness_list.append(w)
                    if len(witness_list) >= 2:
                        break

            # Enforcement: reject bare dyads in BLOCKING mode
            if not witness_list and self._active_mode == EnforcementMode.BLOCKING:
                raise ConnectionError(
                    f"Cannot register bare dyad {source}->{target} in BLOCKING mode"
                )
            if not witness_list and self._active_mode == EnforcementMode.ADVISORY:
                logger.warning(
                    "Advisory: registering bare dyad %s->%s (no witnesses available)",
                    source, target,
                )

            triad = ConnectionTriad(
                connection_id=conn_id,
                source=source,
                target=target,
                witnesses=witness_list,
                connection_type=connection_type,
                channel=channel,
                criticality=criticality,
                description=description,
            )

            self._connections[conn_id] = triad

            # Index by channel name for fallback lookup
            if channel and channel not in self._channel_index:
                self._channel_index[channel] = conn_id

            # Track system participation
            for system in (source, target):
                if system not in self._system_connections:
                    self._system_connections[system] = []
                self._system_connections[system].append(conn_id)

            for w in witness_list:
                if w not in self._system_connections:
                    self._system_connections[w] = []

        if self._bus:
            self._bus.publish(CH_CONNECTION_REGISTERED, {
                "connection_id": conn_id,
                "source": source,
                "target": target,
                "witnesses": witness_list,
                "witness": witness_list[0] if witness_list else None,
                "type": connection_type.value,
            })

        return triad

    def _assign_witnesses(
        self,
        source: str,
        target: str,
        count: int = 2,
        exclude: set[str] | None = None,
    ) -> list[str]:
        """Auto-assign witnesses using SomaticMap topology.

        Returns up to ``count`` witnesses (default 2 for Law 1 compliance).

        Heuristic priority:
        1. Shared neighbors in SomaticMap (prefer nervous system)
        2. Round-robin from nervous system tuple for load balance
        """
        excluded = (exclude or set()) | {source, target}
        witnesses: list[str] = []

        if self._somatic_map is None:
            # Fallback: pick from nervous system round-robin
            pair_hash = hash(f"{source}:{target}") % len(NERVOUS_SYSTEM)
            for i in range(len(NERVOUS_SYSTEM)):
                candidate = NERVOUS_SYSTEM[(pair_hash + i) % len(NERVOUS_SYSTEM)]
                if candidate not in excluded and candidate not in witnesses:
                    witnesses.append(candidate)
                    if len(witnesses) >= count:
                        return witnesses
            return witnesses

        # Strategy 1: Find shared neighbors in dependency graph
        try:
            source_node = self._somatic_map.get_system_info(source)
            target_node = self._somatic_map.get_system_info(target)

            if source_node and target_node:
                source_neighbors = source_node.upstream | source_node.downstream
                target_neighbors = target_node.upstream | target_node.downstream
                shared = source_neighbors & target_neighbors - excluded

                if shared:
                    nervous_shared = sorted(shared & set(NERVOUS_SYSTEM))
                    other_shared = sorted(shared - set(NERVOUS_SYSTEM))
                    for c in nervous_shared + other_shared:
                        if c not in witnesses:
                            witnesses.append(c)
                            if len(witnesses) >= count:
                                return witnesses
        except Exception:
            pass

        # Strategy 2: Round-robin from nervous system for remaining slots
        pair_hash = hash(f"{source}:{target}") % len(NERVOUS_SYSTEM)
        for i in range(len(NERVOUS_SYSTEM)):
            candidate = NERVOUS_SYSTEM[(pair_hash + i) % len(NERVOUS_SYSTEM)]
            if candidate not in excluded and candidate not in witnesses:
                witnesses.append(candidate)
                if len(witnesses) >= count:
                    return witnesses

        return witnesses

    def deregister_connection(self, connection_id: str) -> bool:
        """Remove a connection from the registry."""
        with self._lock:
            triad = self._connections.pop(connection_id, None)
            if triad is None:
                return False

            for system in (triad.source, triad.target):
                conns = self._system_connections.get(system, [])
                if connection_id in conns:
                    conns.remove(connection_id)

            # Clean channel index
            if triad.channel and self._channel_index.get(triad.channel) == connection_id:
                del self._channel_index[triad.channel]

            return True

    # =========================================================================
    # Verification
    # =========================================================================

    def verify_all(self) -> dict[str, Any]:
        """Verify all registered connections are healthy.

        Checks that source, target, and witness systems are all
        known to SomaticMap (i.e., alive and registered).
        """
        results = {"total": 0, "healthy": 0, "unhealthy": 0, "bare_dyads": 0}

        with self._lock:
            for conn_id, triad in self._connections.items():
                results["total"] += 1
                healthy = True

                if self._somatic_map:
                    # Check source and target exist in SomaticMap
                    for system_id in (triad.source, triad.target):
                        node = self._somatic_map.get_system_info(system_id)
                        if node is None:
                            healthy = False

                    # Check all witnesses exist
                    if triad.witnesses:
                        for w in triad.witnesses:
                            witness_node = self._somatic_map.get_system_info(w)
                            if witness_node is None:
                                healthy = False
                    else:
                        results["bare_dyads"] += 1

                triad.healthy = healthy
                triad.last_verified = time.time()

                if healthy:
                    results["healthy"] += 1
                else:
                    results["unhealthy"] += 1
                    if self._active_mode == EnforcementMode.BLOCKING and self._bus:
                        self._bus.publish(CH_CONNECTION_BLOCKED, {
                            "connection_id": conn_id,
                            "reason": "unhealthy",
                            "source": triad.source,
                            "target": triad.target,
                        })

            self._total_verifications += 1

        # Topological invariant (advisory — never blocks)
        results["euler"] = self.get_euler_statistics()

        # Operational witnessing stats (if WitnessNotifier is active)
        if self._witness_notifier is not None:
            try:
                results["witnessing"] = self._witness_notifier.get_statistics()
            except Exception:
                pass  # Advisory — never break verification

        if self._bus:
            self._bus.publish(CH_CONNECTION_VERIFIED, results)

        return results

    def step(self) -> None:
        """Step hook for periodic verification."""
        self._step_counter += 1
        if self._step_counter % self._verify_interval == 0:
            results = self.verify_all()
            bare = self.get_bare_dyads()
            if bare:
                if self._active_mode == EnforcementMode.BLOCKING:
                    logger.error(
                        "BLOCKING: %d bare dyads detected", len(bare),
                    )
                else:
                    logger.warning(
                        "ConnectionRegistry: %d bare dyads detected", len(bare),
                    )
                if self._bus:
                    self._bus.publish(CH_CONNECTION_BARE_DYAD, {
                        "count": len(bare),
                        "connections": [b.connection_id for b in bare],
                        "enforcement": self._active_mode.value,
                    })

    # =========================================================================
    # Queries
    # =========================================================================

    def get_connection(self, connection_id: str) -> Optional[ConnectionTriad]:
        """Get a specific connection by ID."""
        return self._connections.get(connection_id)

    def get_connections_for_system(self, system_id: str) -> list[ConnectionTriad]:
        """Get all connections involving a system (as source or target)."""
        conn_ids = self._system_connections.get(system_id, [])
        return [self._connections[cid] for cid in conn_ids if cid in self._connections]

    def get_bare_dyads(self) -> list[ConnectionTriad]:
        """Get connections that have no witnesses assigned."""
        with self._lock:
            return [t for t in self._connections.values() if not t.witnesses]

    def get_unhealthy_connections(self) -> list[ConnectionTriad]:
        """Get connections that failed verification."""
        with self._lock:
            return [t for t in self._connections.values() if not t.healthy]

    def get_coverage_report(self) -> dict[str, Any]:
        """Report what percentage of known systems have triadic connections."""
        with self._lock:
            all_systems = set()
            witnessed_systems = set()

            for triad in self._connections.values():
                all_systems.add(triad.source)
                all_systems.add(triad.target)
                if triad.witnesses:
                    witnessed_systems.add(triad.source)
                    witnessed_systems.add(triad.target)

            total = len(all_systems)
            covered = len(witnessed_systems)

            return {
                "total_systems": total,
                "systems_with_witnesses": covered,
                "coverage_pct": (covered / max(total, 1)) * 100,
                "total_connections": len(self._connections),
                "bare_dyads": len(self.get_bare_dyads()),
            }

    # =========================================================================
    # Topological Invariants (Sacred Geometry — Euler's Formula)
    # =========================================================================

    def get_euler_statistics(self) -> dict[str, Any]:
        """Compute topological invariants of the connection graph.

        Euler's formula for connected planar graphs: V - E + F = 2.
        Mae's graph is neither planar nor simply connected, so we compute
        the generalized Euler characteristic and the "excess edges" above
        a spanning forest. The excess count is the number of non-tree
        shortcuts (EventBus cross-wiring) above the minimal spanning tree.

        Returns dict with vertices, edges, components, excess_edges,
        and euler_characteristic (V - E + C for a graph with C components).
        """
        with self._lock:
            if not self._connections:
                return {
                    "vertices": 0,
                    "edges": 0,
                    "components": 0,
                    "excess_edges": 0,
                    "euler_characteristic": 0,
                }

            # V: unique systems participating as source or target
            vertices: set[str] = set()
            # E: unique undirected edges (unordered pairs)
            edges: set[frozenset[str]] = set()

            for triad in self._connections.values():
                vertices.add(triad.source)
                vertices.add(triad.target)
                edges.add(frozenset((triad.source, triad.target)))

            v = len(vertices)
            e = len(edges)

            # C: connected components via union-find
            parent: dict[str, str] = {node: node for node in vertices}

            def find(x: str) -> str:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            for edge in edges:
                nodes = list(edge)
                if len(nodes) == 2:
                    ra, rb = find(nodes[0]), find(nodes[1])
                    if ra != rb:
                        parent[ra] = rb

            c = len({find(node) for node in vertices})

            # Spanning forest has V - C edges; excess = E - (V - C)
            excess = e - (v - c)

            return {
                "vertices": v,
                "edges": e,
                "components": c,
                "excess_edges": excess,
                "euler_characteristic": v - e + c,
            }

    def check_euler_invariant(self) -> dict[str, Any]:
        """Advisory check: report topological invariants of the connection graph.

        For a spanning forest: E = V - C exactly (0 excess edges).
        For Mae: E > V - C because EventBus shortcuts add non-tree edges.
        The excess_edges count measures how many shortcuts exist above the
        minimal spanning tree — this IS the transfractal compromise in numbers.

        Returns the Euler statistics plus an advisory interpretation.
        """
        stats = self.get_euler_statistics()
        if stats["vertices"] == 0:
            stats["interpretation"] = "empty_graph"
            return stats

        if stats["excess_edges"] == 0:
            stats["interpretation"] = "tree_or_forest"
        else:
            stats["interpretation"] = "shortcuts_present"

        logger.info(
            "Euler invariant: V=%d, E=%d, C=%d, excess=%d, chi=%d (%s)",
            stats["vertices"], stats["edges"], stats["components"],
            stats["excess_edges"], stats["euler_characteristic"],
            stats["interpretation"],
        )
        return stats

    def get_statistics(self) -> dict[str, Any]:
        """Get registry-level statistics."""
        with self._lock:
            type_counts: dict[str, int] = {}
            criticality_counts: dict[str, int] = {}
            witness_counts: dict[str, int] = {}

            for triad in self._connections.values():
                ct = triad.connection_type.value
                type_counts[ct] = type_counts.get(ct, 0) + 1

                cc = triad.criticality.value
                criticality_counts[cc] = criticality_counts.get(cc, 0) + 1

                for w in triad.witnesses:
                    witness_counts[w] = witness_counts.get(w, 0) + 1

            healthy = sum(1 for t in self._connections.values() if t.healthy)
            bare = sum(1 for t in self._connections.values() if not t.witnesses)

            return {
                "total_connections": len(self._connections),
                "healthy": healthy,
                "unhealthy": len(self._connections) - healthy,
                "bare_dyads": bare,
                "type_counts": type_counts,
                "criticality_counts": criticality_counts,
                "witness_load": witness_counts,
                "systems_connected": len(self._system_connections),
                "total_verifications": self._total_verifications,
                "enforcement_mode": self._active_mode.value,
                "sealed": self._sealed,
                "blocked_count": self._blocked_count,
            }
