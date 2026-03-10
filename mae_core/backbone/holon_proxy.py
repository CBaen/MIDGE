"""HolonProxy — lightweight awareness adapter for non-agent systems.

Extracted from holon_protocol.py for single-responsibility.

Same interface as HolonMixin's know_* methods, but doesn't need
to be mixed in. Any Python object can hold a reference to its proxy.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


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
        registry: Any,
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
    # ------------------------------------------------------------------

    def sense(self) -> dict[str, Any]:
        """Perceive this system's current operational state."""
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
        """Store or retrieve from per-proxy memory."""
        if value is not None:
            self._holon_memory[key] = value
            return value
        return self._holon_memory.get(key)

    def decide(self, stimulus: Any = None) -> Any:
        """Delegate to this system's decision/evaluation method."""
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
        """Delegate to this system's learning/adaptation method."""
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
        """Self-assessment and recovery attempt."""
        health = self.get_health()
        report: dict[str, Any] = {
            "holon_id": self._holon_id,
            "health": health,
            "healthy": health > 0.5,
            "issues": [],
            "recovery_attempted": False,
        }

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

        if self._somatic_map is not None:
            heartbeat = getattr(self._somatic_map, "heartbeat", None)
            if callable(heartbeat):
                try:
                    heartbeat(self._holon_id, health=health)
                except (TypeError, AttributeError):
                    try:
                        heartbeat(self._holon_id)
                    except Exception:
                        pass
                except Exception:
                    pass

        return report

    def set_system_ref(self, system: Any) -> None:
        """Store a back-reference to the actual system object."""
        self._system_ref = system

    def act(self, action: Any = None) -> float:
        """Execute action at this holon's scale."""
        system = self._system_ref
        if system is None:
            return 0.0

        for method_name in ("step", "process", "execute"):
            method = getattr(system, method_name, None)
            if callable(method):
                try:
                    result = method()
                    return float(result) if result is not None else 1.0
                except (TypeError, ValueError, AttributeError):
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
