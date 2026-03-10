"""HolonRegistry and HolonEntry — the holarchic family tree for Mae.

Extracted from holon_protocol.py for single-responsibility.

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
        self._proxies: dict[str, "HolonProxy"] = {}  # type: ignore[name-defined]

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

    def get_proxy(self, holon_id: str) -> "HolonProxy":  # type: ignore[name-defined]
        """Get or create a HolonProxy for the given holon.

        Proxies are cached — same proxy returned for same ID.
        All proxies share the registry's somatic_map and connection_registry.
        """
        from mae_core.backbone.holon_proxy import HolonProxy
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
