"""Lymphatic System - Mae's garbage collection and recycling.

Biological analogy: The lymphatic system collects interstitial fluid,
filters pathogens, and recycles useful components. Without it, tissue
drowns in its own waste — edema, infection, toxicity.

Mae's LymphaticSystem is the equivalent:
1. COLLECT — sweep stale markers, dead connections, orphan subscriptions
2. SORT — classify waste as recyclable or disposable
3. PROCESS — recycle useful components (publish recycled event),
   dispose of the rest

Connection points:
- StigmergicEnvironment provides stale markers to sweep
- ConnectionRegistry provides dead/unhealthy connections to sweep
- EventBus publishes lymph status, overflow, and recycled events
- AutoHealer may listen for overflow (system drowning in waste)
- Senescence may listen for capacity trends (aging indicator)

Law compliance:
- Law 6 (Autopoietic Closure): removing waste maintains the system
  that produces the waste collectors. Circular causation.
- Law 8, Property 2 (Differentiation): distinguishes recyclable
  from disposable waste — not all waste is equal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# EventBus channels
CH_LYMPH_STATUS = "emergent.lymph_status"
CH_LYMPH_OVERFLOW = "emergent.lymph_overflow"
CH_RECYCLED = "emergent.recycled"

# Valid waste types
WASTE_TYPES = frozenset({
    "stale_marker",
    "dead_connection",
    "expired_memory",
    "orphan_subscription",
})


@dataclass
class WasteItem:
    """A piece of waste collected by the lymphatic system."""

    source: str
    waste_type: str  # one of WASTE_TYPES
    item_data: Any = None
    collected_at: int = 0


@dataclass
class RecycledComponent:
    """A useful component recovered from waste processing."""

    original_type: str
    recycled_value: Any = None
    efficiency: float = 0.0


class LymphaticSystem:
    """Mae's garbage collection and recycling system.

    Periodically sweeps stale markers from stigmergy, dead connections
    from the connection registry, and processes them: recyclable waste
    yields RecycledComponents published on the EventBus; non-recyclable
    waste is disposed.

    Like the biological lymphatic system: if collection falls behind
    production, the system overflows (edema). The overflow event lets
    other systems respond (AutoHealer, SenescenceManager).
    """

    CH_LYMPH_STATUS = CH_LYMPH_STATUS
    CH_LYMPH_OVERFLOW = CH_LYMPH_OVERFLOW
    CH_RECYCLED = CH_RECYCLED

    def __init__(
        self,
        event_bus: Any = None,
        stigmergy: Any = None,
        connection_registry: Any = None,
        **kwargs: Any,
    ) -> None:
        self._bus = event_bus
        self._stigmergy = stigmergy
        self._connection_registry = connection_registry

        # Waste storage
        self._waste_bin: list[WasteItem] = []

        # Processing counters
        self._recycled_count: int = 0
        self._disposed_count: int = 0
        self._total_collected: int = 0

        # Tuning parameters
        self._collection_interval: int = 13  # Fibonacci — collect every 13 steps
        self._max_waste_capacity: int = 100  # Overflow threshold

        # Subscribe to channels that may report waste
        if self._bus:
            self._bus.register_callback(
                "healing.failed", self._on_healing_failed
            )

        logger.info("LymphaticSystem initialized (interval=%d, capacity=%d)",
                     self._collection_interval, self._max_waste_capacity)

    # =========================================================================
    # Collection
    # =========================================================================

    def collect_waste(
        self, source: str, waste_type: str, item_data: Any = None,
        current_step: int = 0,
    ) -> None:
        """Add a waste item to the collection bin.

        External systems can push waste here directly (e.g., memory
        system reporting expired entries).
        """
        if waste_type not in WASTE_TYPES:
            logger.warning("LymphaticSystem: unknown waste type %r from %s",
                           waste_type, source)
        item = WasteItem(
            source=source,
            waste_type=waste_type,
            item_data=item_data,
            collected_at=current_step,
        )
        self._waste_bin.append(item)
        self._total_collected += 1

    def _sweep_stale_markers(self, current_step: int) -> None:
        """Sweep markers older than 50 steps from stigmergy environment.

        The stigmergy environment uses real-time decay, but some markers
        may linger at low intensity. The lymphatic system collects them
        based on step count (simulation time, not wall-clock time).
        """
        if self._stigmergy is None:
            return

        # Access the internal markers dict — the lymphatic system has
        # privileged access to the environment's internals (like immune
        # cells circulating through tissue).
        markers_dict = getattr(self._stigmergy, "_markers", {})
        stale_ids = []

        for marker_id, marker in markers_dict.items():
            # Use metadata timestamp if available, else use step heuristic
            marker_step = getattr(marker, "metadata", {}).get("step", 0)
            marker_age = current_step - marker_step if marker_step > 0 else 51
            if marker_age > 50:
                stale_ids.append(marker_id)
                self.collect_waste(
                    source="stigmergy",
                    waste_type="stale_marker",
                    item_data={"marker_id": marker_id,
                               "marker_type": getattr(marker, "marker_type", "unknown")},
                    current_step=current_step,
                )

        # Remove collected stale markers from stigmergy
        for mid in stale_ids:
            markers_dict.pop(mid, None)

        if stale_ids:
            logger.debug("Lymphatic: swept %d stale markers", len(stale_ids))

    def _sweep_dead_connections(self) -> None:
        """Sweep unhealthy connections from connection registry."""
        if self._connection_registry is None:
            return

        unhealthy = []
        try:
            unhealthy = self._connection_registry.get_unhealthy_connections()
        except Exception:
            return

        for conn in unhealthy:
            self.collect_waste(
                source="connection_registry",
                waste_type="dead_connection",
                item_data={
                    "connection_id": getattr(conn, "connection_id", "unknown"),
                    "source": getattr(conn, "source", ""),
                    "target": getattr(conn, "target", ""),
                },
            )

        if unhealthy:
            logger.debug("Lymphatic: swept %d dead connections", len(unhealthy))

    # =========================================================================
    # Processing
    # =========================================================================

    def _process_waste(self) -> tuple[int, int]:
        """Sort and process collected waste.

        Recyclable types:
        - stale_marker -> can be redeposited as exploration data
        - expired_memory -> patterns can be distilled

        Non-recyclable types:
        - dead_connection -> must be re-established, not recycled
        - orphan_subscription -> just clean up

        Returns:
            (recycled_count, disposed_count) for this batch.
        """
        recyclable_types = {"stale_marker", "expired_memory"}
        recycled = 0
        disposed = 0

        batch = list(self._waste_bin)
        self._waste_bin.clear()

        for item in batch:
            if item.waste_type in recyclable_types:
                component = RecycledComponent(
                    original_type=item.waste_type,
                    recycled_value=item.item_data,
                    efficiency=0.3 if item.waste_type == "stale_marker" else 0.5,
                )
                recycled += 1
                self._recycled_count += 1

                if self._bus:
                    self._bus.publish(CH_RECYCLED, {
                        "original_type": component.original_type,
                        "efficiency": component.efficiency,
                        "source": item.source,
                    })
            else:
                disposed += 1
                self._disposed_count += 1

        return recycled, disposed

    # =========================================================================
    # Step
    # =========================================================================

    def step(self, current_step: int) -> None:
        """Per-step update: sweep and process on schedule.

        Every collection_interval steps:
        1. Sweep stale markers from stigmergy
        2. Sweep dead connections from registry
        3. Process all collected waste

        Always publish status. If waste bin exceeds capacity, publish overflow.
        """
        # Collection runs on schedule (Fibonacci interval)
        if current_step % self._collection_interval == 0 and current_step > 0:
            self._sweep_stale_markers(current_step)
            self._sweep_dead_connections()
            recycled, disposed = self._process_waste()
        else:
            recycled, disposed = 0, 0

        # Overflow detection — continuous, not just on collection steps
        capacity_used = self.get_capacity_used()
        if capacity_used > 1.0 and self._bus:
            self._bus.publish(CH_LYMPH_OVERFLOW, {
                "capacity_used": capacity_used,
                "waste_count": len(self._waste_bin),
                "max_capacity": self._max_waste_capacity,
                "step": current_step,
            })
            logger.warning(
                "Lymphatic OVERFLOW: %.0f%% capacity at step %d",
                capacity_used * 100, current_step,
            )

        # Always publish status
        if self._bus:
            self._bus.publish(CH_LYMPH_STATUS, {
                "collected": self._total_collected,
                "recycled": self._recycled_count,
                "disposed": self._disposed_count,
                "capacity_used": capacity_used,
                "waste_bin_size": len(self._waste_bin),
                "step": current_step,
            })

    # =========================================================================
    # Queries
    # =========================================================================

    def get_capacity_used(self) -> float:
        """Current waste bin usage as fraction of max capacity."""
        return len(self._waste_bin) / max(self._max_waste_capacity, 1)

    # =========================================================================
    # EventBus handlers
    # =========================================================================

    def _on_healing_failed(self, channel: str, message: Any) -> None:
        """Collect failed healing attempts as waste."""
        self.collect_waste(
            source="auto_healer",
            waste_type="orphan_subscription",
            item_data={"channel": channel, "message": str(message)[:200]},
        )

    # =========================================================================
    # Observability
    # =========================================================================

    def get_statistics(self) -> dict[str, Any]:
        return {
            "total_collected": self._total_collected,
            "recycled_count": self._recycled_count,
            "disposed_count": self._disposed_count,
            "waste_bin_size": len(self._waste_bin),
            "capacity_used": self.get_capacity_used(),
            "collection_interval": self._collection_interval,
            "max_waste_capacity": self._max_waste_capacity,
        }

    # =========================================================================
    # Tier 2 Persistence
    # =========================================================================

    def serialize(self) -> dict:
        return {
            "total_collected": self._total_collected,
            "recycled_count": self._recycled_count,
            "disposed_count": self._disposed_count,
            "collection_interval": self._collection_interval,
            "max_waste_capacity": self._max_waste_capacity,
            "waste_bin": [
                {
                    "source": w.source,
                    "waste_type": w.waste_type,
                    "collected_at": w.collected_at,
                }
                for w in self._waste_bin
            ],
        }

    def restore(self, data: dict) -> None:
        self._total_collected = data.get("total_collected", 0)
        self._recycled_count = data.get("recycled_count", 0)
        self._disposed_count = data.get("disposed_count", 0)
        self._collection_interval = data.get("collection_interval", 13)
        self._max_waste_capacity = data.get("max_waste_capacity", 100)
        self._waste_bin = [
            WasteItem(
                source=w["source"],
                waste_type=w["waste_type"],
                collected_at=w.get("collected_at", 0),
            )
            for w in data.get("waste_bin", [])
        ]
