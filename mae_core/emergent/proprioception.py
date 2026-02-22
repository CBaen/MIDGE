"""Proprioception System - Mae's sense of her own body position.

Biological basis: Proprioceptors in muscles, tendons, and joints give
the brain continuous feedback about body position without requiring
vision. Sherrington called it the "sixth sense." Without proprioception,
you cannot walk in the dark, touch your nose with eyes closed, or know
how much force you are applying.

Mae's ProprioceptionSystem is the equivalent:
1. BUILD BODY MAP — traverse fractal structure, populate positions
2. UPDATE POSITIONS — track activity and health per system
3. DETECT TOPOLOGY CHANGES — notice when systems are added/removed
4. RELATIVE AWARENESS — know which systems are neighbors, same organ, etc.

Distinct from SomaticMap: SomaticMap is about blast radius (surgery
planning). Proprioception is about body awareness (where am I, what
is my posture, are my parts moving correctly).

Connection points:
- FractalGenerator provides the hierarchy (organ/subsystem/depth)
- SomaticMap provides dependency graph (neighbors)
- EventBus publishes proprioception updates and topology changes
- SenescenceManager may use health data for wear assessment

Law compliance:
- Law 8, Property 5 (Multi-scale Hierarchy): awareness at every
  fractal level — process, subsystem, organ, organism.
- Law 8, Property 3 (Self-reference): the system knows itself within
  the larger body. Self-locating awareness.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# EventBus channels
CH_PROPRIOCEPTION = "emergent.proprioception_update"
CH_TOPOLOGY_CHANGE = "emergent.topology_changed"


@dataclass
class SystemPosition:
    """A system's position within Mae's fractal body."""

    system_name: str
    depth: int = 0              # Fractal depth (0=organism, 1=organ, 2=subsystem, 3=process)
    organ: str = ""             # Which organ this belongs to
    subsystem: str = ""         # Which subsystem within the organ
    neighbors: list[str] = field(default_factory=list)  # Adjacent systems in dependency graph
    activity: float = 0.0       # Current activity level [0, 1]
    health: float = 1.0         # Current health [0, 1]


class ProprioceptionSystem:
    """Mae's awareness of her own body position and structure.

    Maintains a map of every system's position in the fractal hierarchy
    (which organ, which subsystem, fractal depth) and tracks activity
    and health. Detects topology changes when systems are added or removed.

    Like biological proprioception: you know where your hand is without
    looking. Mae knows where her substrate is in relation to her
    threat detector, what organ they belong to, and whether they are
    neighbors.
    """

    CH_PROPRIOCEPTION = CH_PROPRIOCEPTION
    CH_TOPOLOGY_CHANGE = CH_TOPOLOGY_CHANGE

    def __init__(
        self,
        event_bus: Any = None,
        somatic_map: Any = None,
        fractal_generator: Any = None,
        **kwargs: Any,
    ) -> None:
        self._bus = event_bus
        self._somatic_map = somatic_map
        self._fractal_generator = fractal_generator

        # Position tracking
        self._system_positions: dict[str, SystemPosition] = {}

        # Topology change detection
        self._topology_version: int = 0
        self._previous_topology_hash: str = ""

        logger.info("ProprioceptionSystem initialized")

    # =========================================================================
    # Body Map Construction
    # =========================================================================

    def build_body_map(self) -> None:
        """Build the body map from fractal structure and somatic topology.

        If fractal_generator is available, traverse FRACTAL_GROUPING to
        populate organ, subsystem, and depth for each system. If somatic_map
        is available, populate neighbor lists from the dependency graph.
        """
        # Import here to avoid circular dependency at module level
        from mae_core.backbone.fractal_generator import FRACTAL_GROUPING

        # Build from fractal grouping
        for organ_name, subsystems in FRACTAL_GROUPING.items():
            for sub_name, system_ids in subsystems.items():
                for system_id in system_ids:
                    self._system_positions[system_id] = SystemPosition(
                        system_name=system_id,
                        depth=3,  # Process level (leaf)
                        organ=organ_name,
                        subsystem=sub_name,
                    )

                # Register subsystem itself
                self._system_positions[sub_name] = SystemPosition(
                    system_name=sub_name,
                    depth=2,  # Subsystem level
                    organ=organ_name,
                    subsystem=sub_name,
                )

            # Register organ itself
            self._system_positions[organ_name] = SystemPosition(
                system_name=organ_name,
                depth=1,  # Organ level
                organ=organ_name,
            )

        # Populate neighbors from somatic map
        if self._somatic_map is not None:
            for system_id, pos in self._system_positions.items():
                node = self._somatic_map.get_system_info(system_id)
                if node is not None:
                    all_connections = set()
                    if hasattr(node, "upstream"):
                        all_connections |= node.upstream
                    if hasattr(node, "downstream"):
                        all_connections |= node.downstream
                    pos.neighbors = sorted(all_connections)

        # Compute initial topology hash
        self._previous_topology_hash = self._compute_topology_hash()

        logger.info("Proprioception: body map built (%d systems)",
                     len(self._system_positions))

    # =========================================================================
    # Position Updates
    # =========================================================================

    def update_position(
        self,
        system_name: str,
        activity: float | None = None,
        health: float | None = None,
    ) -> None:
        """Update activity and/or health for a system."""
        pos = self._system_positions.get(system_name)
        if pos is None:
            # Auto-register unknown systems at process depth
            pos = SystemPosition(system_name=system_name, depth=3)
            self._system_positions[system_name] = pos

        if activity is not None:
            pos.activity = max(0.0, min(1.0, activity))
        if health is not None:
            pos.health = max(0.0, min(1.0, health))

    # =========================================================================
    # Topology Change Detection
    # =========================================================================

    def _compute_topology_hash(self) -> str:
        """Hash of current system names — changes when systems are added/removed."""
        names = sorted(self._system_positions.keys())
        raw = "|".join(names)
        return hashlib.md5(raw.encode()).hexdigest()

    def detect_topology_change(self) -> bool:
        """Compare current topology to previous. Returns True if changed."""
        current_hash = self._compute_topology_hash()
        changed = current_hash != self._previous_topology_hash
        if changed:
            self._topology_version += 1
            self._previous_topology_hash = current_hash
        return changed

    # =========================================================================
    # Relative Position Queries
    # =========================================================================

    def get_relative_position(
        self, system_a: str, system_b: str,
    ) -> dict[str, Any]:
        """Determine spatial relationship between two systems.

        Returns dict with: same_organ, same_subsystem, is_neighbor, depth_difference.
        """
        pos_a = self._system_positions.get(system_a)
        pos_b = self._system_positions.get(system_b)

        if pos_a is None or pos_b is None:
            return {
                "same_organ": False,
                "same_subsystem": False,
                "is_neighbor": False,
                "depth_difference": 0,
                "known": False,
            }

        return {
            "same_organ": pos_a.organ == pos_b.organ and pos_a.organ != "",
            "same_subsystem": (pos_a.subsystem == pos_b.subsystem
                               and pos_a.subsystem != ""),
            "is_neighbor": system_b in pos_a.neighbors or system_a in pos_b.neighbors,
            "depth_difference": abs(pos_a.depth - pos_b.depth),
            "known": True,
        }

    # =========================================================================
    # Body Summary
    # =========================================================================

    def get_body_summary(self) -> dict[str, dict[str, Any]]:
        """Organ-level summary: systems, mean health, mean activity per organ."""
        organ_data: dict[str, dict[str, Any]] = {}

        for pos in self._system_positions.values():
            if not pos.organ or pos.depth < 3:
                continue  # Only count leaf-level processes

            if pos.organ not in organ_data:
                organ_data[pos.organ] = {
                    "systems": 0,
                    "total_health": 0.0,
                    "total_activity": 0.0,
                }

            organ_data[pos.organ]["systems"] += 1
            organ_data[pos.organ]["total_health"] += pos.health
            organ_data[pos.organ]["total_activity"] += pos.activity

        # Compute means
        summary = {}
        for organ, data in organ_data.items():
            count = max(data["systems"], 1)
            summary[organ] = {
                "systems": data["systems"],
                "mean_health": data["total_health"] / count,
                "mean_activity": data["total_activity"] / count,
            }

        return summary

    # =========================================================================
    # Step
    # =========================================================================

    def step(self, current_step: int) -> None:
        """Per-step update: refresh positions, detect topology changes, publish.

        If somatic_map is available, refresh health data from it.
        """
        # Refresh from somatic map
        if self._somatic_map is not None:
            for system_id, pos in self._system_positions.items():
                node = self._somatic_map.get_system_info(system_id)
                if node is not None:
                    pos.health = getattr(node, "health", pos.health)

        # Detect topology changes
        topology_changed = self.detect_topology_change()
        if topology_changed and self._bus:
            self._bus.publish(CH_TOPOLOGY_CHANGE, {
                "topology_version": self._topology_version,
                "total_systems": len(self._system_positions),
                "step": current_step,
            })

        # Compute aggregate metrics
        total = len(self._system_positions)
        if total > 0:
            mean_activity = sum(
                p.activity for p in self._system_positions.values()
            ) / total
            mean_health = sum(
                p.health for p in self._system_positions.values()
            ) / total
        else:
            mean_activity = 0.0
            mean_health = 0.0

        # Publish proprioception update
        if self._bus:
            self._bus.publish(CH_PROPRIOCEPTION, {
                "total_systems": total,
                "topology_changed": topology_changed,
                "topology_version": self._topology_version,
                "mean_activity": mean_activity,
                "mean_health": mean_health,
                "step": current_step,
            })

    # =========================================================================
    # Observability
    # =========================================================================

    def get_statistics(self) -> dict[str, Any]:
        total = len(self._system_positions)
        return {
            "total_systems": total,
            "topology_version": self._topology_version,
            "organs": len({p.organ for p in self._system_positions.values() if p.organ}),
            "subsystems": len({p.subsystem for p in self._system_positions.values() if p.subsystem}),
            "mean_health": (
                sum(p.health for p in self._system_positions.values()) / max(total, 1)
            ),
            "mean_activity": (
                sum(p.activity for p in self._system_positions.values()) / max(total, 1)
            ),
        }

    # =========================================================================
    # Tier 2 Persistence
    # =========================================================================

    def serialize(self) -> dict:
        return {
            "topology_version": self._topology_version,
            "positions": {
                name: {
                    "depth": pos.depth,
                    "organ": pos.organ,
                    "subsystem": pos.subsystem,
                    "neighbors": pos.neighbors,
                    "activity": pos.activity,
                    "health": pos.health,
                }
                for name, pos in self._system_positions.items()
            },
        }

    def restore(self, data: dict) -> None:
        self._topology_version = data.get("topology_version", 0)
        for name, pos_data in data.get("positions", {}).items():
            self._system_positions[name] = SystemPosition(
                system_name=name,
                depth=pos_data.get("depth", 0),
                organ=pos_data.get("organ", ""),
                subsystem=pos_data.get("subsystem", ""),
                neighbors=pos_data.get("neighbors", []),
                activity=pos_data.get("activity", 0.0),
                health=pos_data.get("health", 1.0),
            )
        self._previous_topology_hash = self._compute_topology_hash()
