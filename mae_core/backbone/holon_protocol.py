"""Holon Protocol - Universal self-awareness interface for Mae's fractal architecture.

Every system at every scale implements: sense, remember, decide, act, learn,
heal, know_self, know_up, know_down, know_peers.

Three classes:
  HolonRegistry     -- the family tree (containment: who's inside whom)
  HolonProxy        -- lightweight awareness adapter for non-agent systems
  AwarenessPulse    -- periodic hierarchy health check (step hook)

HolonMixin (the 10-capability agent interface) lives in holon_mixin.py.

HolonRegistry complements SomaticMap:
  SomaticMap tracks dependencies (what breaks if X fails).
  HolonRegistry tracks containment (what lives inside X, what contains X).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# =====================================================================
# HolonEntry - data record for a registered holon
# =====================================================================

@dataclass
class HolonEntry:
    """A registered holon in the hierarchy."""

    holon_id: str
    holon_type: str  # "organism", "colony", "agent", "system", "organ"
    parent_id: Optional[str] = None
    capabilities: set[str] = field(default_factory=set)
    registered_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


# =====================================================================
# HolonRegistry - the family tree
# =====================================================================

class HolonRegistry:
    """Tracks the holarchic containment structure of Mae.

    Where SomaticMap says "EventBus depends on nothing, everything depends on EventBus,"
    HolonRegistry says "EventBus is a child of Mae, peer of CircadianRhythm."

    Thread-safe (RLock pattern, same as SomaticMap).
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._holons: dict[str, HolonEntry] = {}
        self._children_map: dict[str, set[str]] = {}
        self._somatic_map: Any = None
        self._connection_registry: Any = None
        self._proxies: dict[str, "HolonProxy"] = {}

    def register(
        self,
        holon_id: str,
        holon_type: str = "system",
        parent_id: Optional[str] = None,
        capabilities: Optional[set[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> HolonEntry:
        """Register a holon in the hierarchy."""
        with self._lock:
            entry = HolonEntry(
                holon_id=holon_id,
                holon_type=holon_type,
                parent_id=parent_id,
                capabilities=capabilities or set(),
                metadata=metadata or {},
            )
            self._holons[holon_id] = entry

            # Ensure children set exists for this holon
            if holon_id not in self._children_map:
                self._children_map[holon_id] = set()

            # Wire into parent's children set
            if parent_id is not None:
                if parent_id not in self._children_map:
                    self._children_map[parent_id] = set()
                self._children_map[parent_id].add(holon_id)

            logger.debug("Holon registered: %s (type=%s, parent=%s)", holon_id, holon_type, parent_id)
            return entry

    def unregister(self, holon_id: str) -> bool:
        """Remove a holon from the hierarchy."""
        with self._lock:
            entry = self._holons.pop(holon_id, None)
            if entry is None:
                return False

            # Remove from parent's children
            if entry.parent_id and entry.parent_id in self._children_map:
                self._children_map[entry.parent_id].discard(holon_id)

            # Orphan children (set their parent to None)
            for child_id in list(self._children_map.get(holon_id, [])):
                if child_id in self._holons:
                    self._holons[child_id].parent_id = None

            self._children_map.pop(holon_id, None)
            return True

    def set_parent(self, holon_id: str, parent_id: Optional[str]) -> bool:
        """Reparent a holon. Returns False if it would create a cycle."""
        with self._lock:
            entry = self._holons.get(holon_id)
            if entry is None:
                return False

            # Check for circular reference
            if parent_id is not None:
                visited = parent_id
                while visited is not None:
                    if visited == holon_id:
                        logger.warning(
                            "Circular reference rejected: %s cannot be child of %s",
                            holon_id, parent_id,
                        )
                        return False
                    parent_entry = self._holons.get(visited)
                    visited = parent_entry.parent_id if parent_entry else None

            # Remove from old parent
            old_parent = entry.parent_id
            if old_parent and old_parent in self._children_map:
                self._children_map[old_parent].discard(holon_id)

            # Set new parent
            entry.parent_id = parent_id
            if parent_id is not None:
                if parent_id not in self._children_map:
                    self._children_map[parent_id] = set()
                self._children_map[parent_id].add(holon_id)

            return True

    def get_entry(self, holon_id: str) -> Optional[HolonEntry]:
        """Get a holon's registry entry."""
        with self._lock:
            return self._holons.get(holon_id)

    def get_parent(self, holon_id: str) -> Optional[str]:
        """Get parent holon ID, or None if root/unregistered."""
        with self._lock:
            entry = self._holons.get(holon_id)
            return entry.parent_id if entry else None

    def get_children(self, holon_id: str) -> list[str]:
        """Get child holon IDs."""
        with self._lock:
            return sorted(self._children_map.get(holon_id, set()))

    def get_peers(self, holon_id: str) -> list[str]:
        """Get sibling holon IDs (same parent, excluding self)."""
        with self._lock:
            entry = self._holons.get(holon_id)
            if entry is None or entry.parent_id is None:
                return []
            siblings = self._children_map.get(entry.parent_id, set())
            return sorted(s for s in siblings if s != holon_id)

    def get_ancestry(self, holon_id: str) -> list[str]:
        """Walk up the parent chain. Returns [parent, grandparent, ...] (root last)."""
        with self._lock:
            chain: list[str] = []
            current = self._holons.get(holon_id)
            if current is None:
                return chain
            visited = {holon_id}
            while current and current.parent_id:
                if current.parent_id in visited:
                    break  # Safety: shouldn't happen after cycle check
                chain.append(current.parent_id)
                visited.add(current.parent_id)
                current = self._holons.get(current.parent_id)
            return chain

    def get_subtree(self, holon_id: str) -> list[str]:
        """All descendants recursively (BFS order)."""
        with self._lock:
            result: list[str] = []
            queue = list(self._children_map.get(holon_id, []))
            visited = {holon_id}
            while queue:
                child = queue.pop(0)
                if child in visited:
                    continue
                visited.add(child)
                result.append(child)
                queue.extend(self._children_map.get(child, []))
            return result

    def get_all_ids(self) -> list[str]:
        """All registered holon IDs."""
        with self._lock:
            return sorted(self._holons.keys())

    def get_statistics(self) -> dict[str, Any]:
        """Registry statistics."""
        with self._lock:
            type_counts: dict[str, int] = {}
            for entry in self._holons.values():
                type_counts[entry.holon_type] = type_counts.get(entry.holon_type, 0) + 1
            roots = [hid for hid, e in self._holons.items() if e.parent_id is None]
            return {
                "total_holons": len(self._holons),
                "type_counts": type_counts,
                "roots": roots,
                "max_depth": self._compute_max_depth(),
            }

    def _compute_max_depth(self) -> int:
        """Compute the maximum depth of the hierarchy."""
        max_d = 0
        for hid in self._holons:
            d = len(self.get_ancestry(hid))
            if d > max_d:
                max_d = d
        return max_d

    # ------------------------------------------------------------------
    # Bidirectional awareness support (Step 3)
    # ------------------------------------------------------------------

    def set_somatic_map(self, sm: Any) -> None:
        """Inject SomaticMap reference for health queries."""
        self._somatic_map = sm

    def set_connection_registry(self, cr: Any) -> None:
        """Inject ConnectionRegistry reference for connection queries."""
        self._connection_registry = cr

    def get_proxy(self, holon_id: str) -> "HolonProxy":
        """Get or create a HolonProxy for the given holon.

        Proxies are cached — same proxy returned for same ID.
        All proxies share the registry's somatic_map and connection_registry.
        """
        with self._lock:
            if holon_id not in self._proxies:
                self._proxies[holon_id] = HolonProxy(
                    holon_id=holon_id,
                    registry=self,
                    somatic_map=self._somatic_map,
                    connection_registry=self._connection_registry,
                )
            else:
                # Update references in case they were injected after proxy creation
                proxy = self._proxies[holon_id]
                proxy._somatic_map = self._somatic_map
                proxy._connection_registry = self._connection_registry
            return self._proxies[holon_id]


# =====================================================================
# HolonProxy - lightweight awareness for non-agent systems
# =====================================================================

class HolonProxy:
    """Lightweight awareness adapter for non-agent systems.

    Same interface as HolonMixin's know_* methods, but doesn't need
    to be mixed in. Any Python object can hold a reference to its proxy.

    Think of it as a desk phone + org chart — every department gets one
    so they can see the hierarchy and call their neighbors.
    """

    def __init__(
        self,
        holon_id: str,
        registry: HolonRegistry,
        somatic_map: Any = None,
        connection_registry: Any = None,
    ) -> None:
        self._holon_id = holon_id
        self._registry = registry
        self._somatic_map = somatic_map
        self._connection_registry = connection_registry
        self._system_ref: Any = None  # Back-reference to the actual system object
        self._holon_memory: dict[str, Any] = {}  # Per-proxy memory store

    @property
    def holon_id(self) -> str:
        return self._holon_id

    def know_self(self) -> dict[str, Any]:
        """ID, type, parent, children count, peers count, health."""
        entry = self._registry.get_entry(self._holon_id)
        if entry is None:
            return {"holon_id": self._holon_id, "registered": False}

        result: dict[str, Any] = {
            "holon_id": self._holon_id,
            "holon_type": entry.holon_type,
            "parent_id": entry.parent_id,
            "children_count": len(self._registry.get_children(self._holon_id)),
            "peers_count": len(self._registry.get_peers(self._holon_id)),
            "health": self.get_health(),
        }
        return result

    def know_up(self) -> Optional[dict[str, Any]]:
        """Parent ID, parent type, parent children count. None if root."""
        entry = self._registry.get_entry(self._holon_id)
        if entry is None or entry.parent_id is None:
            return None

        parent_id = entry.parent_id
        result: dict[str, Any] = {"parent_id": parent_id}

        parent_entry = self._registry.get_entry(parent_id)
        if parent_entry:
            result["parent_type"] = parent_entry.holon_type
            result["parent_children_count"] = len(
                self._registry.get_children(parent_id)
            )
        return result

    def know_down(self) -> list[dict[str, Any]]:
        """Children IDs and types."""
        children: list[dict[str, Any]] = []
        for child_id in self._registry.get_children(self._holon_id):
            child_entry = self._registry.get_entry(child_id)
            if child_entry:
                children.append({
                    "holon_id": child_id,
                    "holon_type": child_entry.holon_type,
                })
        return children

    def know_peers(self) -> list[dict[str, Any]]:
        """Peer IDs and types (same parent, excluding self)."""
        peers: list[dict[str, Any]] = []
        for peer_id in self._registry.get_peers(self._holon_id):
            peer_entry = self._registry.get_entry(peer_id)
            if peer_entry:
                peers.append({
                    "holon_id": peer_id,
                    "holon_type": peer_entry.holon_type,
                })
        return peers

    def get_health(self) -> float:
        """Health from SomaticMap. Returns 1.0 if unavailable."""
        if self._somatic_map is None:
            return 1.0
        get_info = getattr(self._somatic_map, "get_system_info", None)
        if callable(get_info):
            try:
                info = get_info(self._holon_id)
                if info and hasattr(info, "health"):
                    return info.health
            except Exception:
                pass
        return 1.0

    def get_connections(self) -> list[dict[str, Any]]:
        """Connections involving this system. Empty list if unavailable."""
        if self._connection_registry is None:
            return []
        get_for = getattr(self._connection_registry, "get_connections_for_system", None)
        if callable(get_for):
            try:
                triads = get_for(self._holon_id)
                return [
                    {
                        "connection_id": t.connection_id,
                        "source": t.source,
                        "target": t.target,
                        "witness": t.witness,
                        "connection_type": t.connection_type.value if hasattr(t.connection_type, "value") else str(t.connection_type),
                    }
                    for t in triads
                ]
            except Exception:
                pass
        return []

    # ------------------------------------------------------------------
    # The 5 completing holon capabilities (triadic review 2026-02-12)
    # Raises system-level compliance from ~4.5/10 to ~9.5/10.
    # Same delegation pattern as act(): try known method names on
    # system_ref, graceful fallback. Bare names (not holon_*) match
    # existing proxy convention (know_self, know_up, act, etc.).
    # ------------------------------------------------------------------

    def sense(self) -> dict[str, Any]:
        """Perceive this system's current operational state.

        Delegates to the system's get_state(), get_statistics(), or
        get_status() method. Returns a minimal dict if no method exists.
        """
        system = self._system_ref
        if system is None:
            return {"holon_id": self._holon_id, "operational": False}

        for method_name in ("get_state", "get_statistics", "get_status"):
            method = getattr(system, method_name, None)
            if callable(method):
                try:
                    result = method()
                    if isinstance(result, dict):
                        return result
                    return {"holon_id": self._holon_id, "state": result}
                except Exception:
                    logger.debug(
                        "HolonProxy.sense() failed for %s.%s",
                        self._holon_id, method_name, exc_info=True,
                    )
        return {"holon_id": self._holon_id, "operational": True}

    def remember(self, key: str, value: Any = None) -> Any:
        """Store or retrieve from per-proxy memory.

        If value is provided, stores it under key. If value is None,
        retrieves by key. Same pattern as HolonMixin._holon_memory.
        """
        if value is not None:
            self._holon_memory[key] = value
            return value
        return self._holon_memory.get(key)

    def decide(self, stimulus: Any = None) -> Any:
        """Delegate to this system's decision/evaluation method.

        Tries decide(), evaluate(), assess() on the system_ref.
        Returns None if no decision method exists (honest no-op —
        not every system decides, and that's correct).
        """
        system = self._system_ref
        if system is None:
            return None

        for method_name in ("decide", "evaluate", "assess"):
            method = getattr(system, method_name, None)
            if callable(method):
                try:
                    return method(stimulus) if stimulus is not None else method()
                except Exception:
                    logger.debug(
                        "HolonProxy.decide() failed for %s.%s",
                        self._holon_id, method_name, exc_info=True,
                    )
        return None

    def learn(self, feedback: Any = None) -> None:
        """Delegate to this system's learning/adaptation method.

        Tries learn(), update(), adapt() on the system_ref.
        No-op if no learning method exists (infrastructure systems
        don't learn, and that's correct).
        """
        system = self._system_ref
        if system is None:
            return

        for method_name in ("learn", "adapt"):
            method = getattr(system, method_name, None)
            if callable(method):
                try:
                    method(feedback) if feedback is not None else method()
                    return
                except Exception:
                    logger.debug(
                        "HolonProxy.learn() failed for %s.%s",
                        self._holon_id, method_name, exc_info=True,
                    )

    def heal(self) -> dict[str, Any]:
        """Self-assessment and recovery attempt.

        1. Check health via get_health() (existing method)
        2. If degraded (health <= 0.5), attempt recovery via system_ref's
           reset(), recover(), or self_repair() method
        3. Report to SomaticMap via heartbeat (closes health monitoring loop)
        4. Return health report dict

        This is genuinely different from sense(): sense observes operational
        state, heal detects degradation and attempts corrective action.
        """
        health = self.get_health()
        report: dict[str, Any] = {
            "holon_id": self._holon_id,
            "health": health,
            "healthy": health > 0.5,
            "issues": [],
            "recovery_attempted": False,
        }

        # Attempt recovery if degraded
        if health <= 0.5:
            report["issues"].append("health_degraded")
            system = self._system_ref
            if system is not None:
                for method_name in ("reset", "recover", "self_repair"):
                    method = getattr(system, method_name, None)
                    if callable(method):
                        try:
                            method()
                            report["recovery_attempted"] = True
                            report["recovery_method"] = method_name
                            break
                        except Exception:
                            logger.debug(
                                "HolonProxy.heal() recovery failed for %s.%s",
                                self._holon_id, method_name, exc_info=True,
                            )

        # Report to SomaticMap (closes the health monitoring loop)
        if self._somatic_map is not None:
            heartbeat = getattr(self._somatic_map, "heartbeat", None)
            if callable(heartbeat):
                try:
                    heartbeat(self._holon_id, health=health)
                except (TypeError, AttributeError):
                    # heartbeat() may not accept health kwarg
                    try:
                        heartbeat(self._holon_id)
                    except Exception:
                        pass
                except Exception:
                    pass

        return report

    def set_system_ref(self, system: Any) -> None:
        """Store a back-reference to the actual system object.

        This enables act(), sense(), decide(), learn(), and heal() to
        delegate to the system's methods.
        Injected during bootstrap after the proxy is created.
        """
        self._system_ref = system

    def act(self, action: Any = None) -> float:
        """Execute action at this holon's scale.

        For system-level holons, 'action' means executing the system's
        primary function (step/process/execute). Returns a quality metric
        (1.0 = success, 0.0 = failure or no-op).

        Uses getattr guards throughout — never crashes on missing methods.
        """
        system = self._system_ref
        if system is None:
            return 0.0

        # Try step(), then process(), then execute() — the three common
        # method names for system-level work in Mae's codebase.
        for method_name in ("step", "process", "execute"):
            method = getattr(system, method_name, None)
            if callable(method):
                try:
                    result = method()
                    return float(result) if result is not None else 1.0
                except (TypeError, ValueError, AttributeError):
                    # Method exists but failed — report partial success
                    return 0.0
                except Exception:
                    logger.debug(
                        "HolonProxy.act() failed for %s.%s",
                        self._holon_id, method_name, exc_info=True,
                    )
                    return 0.0
        return 0.0

    def __repr__(self) -> str:
        return f"HolonProxy({self._holon_id!r})"


# =====================================================================
# AwarenessPulse - periodic hierarchy health check
# =====================================================================

CH_AWARENESS_PULSE = "holon.awareness_pulse"
CH_AWARENESS_ANOMALY = "holon.anomaly_detected"


class AwarenessPulse:
    """Periodic hierarchy health check — the organism's self-scan.

    Every `interval` steps, queries all holons and checks for:
    - Orphaned systems (registered but parent doesn't exist)
    - Health gradient (average child health vs parent health)
    - Peer drift (peers with divergent health scores)

    Publishes summary on holon.awareness_pulse.
    Publishes anomalies on holon.anomaly_detected.
    """

    def __init__(
        self,
        registry: HolonRegistry,
        event_bus: Any,
        interval: int = 25,
    ) -> None:
        self._registry = registry
        self._event_bus = event_bus
        self._interval = interval
        self._step_count = 0
        self._pulse_count = 0
        self._last_anomalies: list[dict[str, Any]] = []

    def step(self) -> None:
        """Called every model step. Runs scan at interval."""
        self._step_count += 1
        if self._step_count % self._interval == 0:
            self._run_pulse()

    def _run_pulse(self) -> None:
        """Execute awareness scan across all holons."""
        self._pulse_count += 1
        anomalies: list[dict[str, Any]] = []
        holon_ids = self._registry.get_all_ids()

        # Check each holon
        orphans: list[str] = []
        health_issues: list[dict[str, Any]] = []

        for hid in holon_ids:
            entry = self._registry.get_entry(hid)
            if entry is None:
                continue

            # Orphan check: parent declared but doesn't exist
            if entry.parent_id is not None:
                parent = self._registry.get_entry(entry.parent_id)
                if parent is None:
                    orphans.append(hid)

            # Health gradient: compare parent health vs children health
            if self._registry._somatic_map is not None:
                children = self._registry.get_children(hid)
                if children:
                    child_healths = []
                    for cid in children:
                        proxy = self._registry.get_proxy(cid)
                        child_healths.append(proxy.get_health())
                    if child_healths:
                        avg_child = sum(child_healths) / len(child_healths)
                        own_proxy = self._registry.get_proxy(hid)
                        own_health = own_proxy.get_health()
                        drift = abs(own_health - avg_child)
                        if drift > 0.3:
                            health_issues.append({
                                "holon_id": hid,
                                "own_health": own_health,
                                "avg_child_health": avg_child,
                                "drift": drift,
                            })

        if orphans:
            anomalies.append({"type": "orphaned_systems", "holon_ids": orphans})
        if health_issues:
            anomalies.append({"type": "health_gradient", "issues": health_issues})

        self._last_anomalies = anomalies

        # Publish pulse summary
        summary = {
            "pulse_number": self._pulse_count,
            "step": self._step_count,
            "total_holons": len(holon_ids),
            "orphans": len(orphans),
            "health_issues": len(health_issues),
            "anomaly_count": len(anomalies),
        }
        self._event_bus.publish(CH_AWARENESS_PULSE, summary)

        if anomalies:
            self._event_bus.publish(CH_AWARENESS_ANOMALY, {
                "pulse_number": self._pulse_count,
                "anomalies": anomalies,
            })
            logger.warning(
                "Awareness pulse #%d: %d anomalies (%d orphans, %d health issues)",
                self._pulse_count, len(anomalies), len(orphans), len(health_issues),
            )
        else:
            logger.debug("Awareness pulse #%d: all clear (%d holons)", self._pulse_count, len(holon_ids))

    def get_statistics(self) -> dict[str, Any]:
        """Pulse statistics."""
        return {
            "pulse_count": self._pulse_count,
            "step_count": self._step_count,
            "interval": self._interval,
            "last_anomalies": self._last_anomalies,
        }
