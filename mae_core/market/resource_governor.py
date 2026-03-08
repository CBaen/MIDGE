"""ResourceGovernor — internal budget and API quota governance.

Tracks API call counts per source, enforces per-source and global
budgets, and publishes throttle events via EventBus. Replaces the
need for external governance (Paperclip) with Law 6 autopoietic
self-governance: the organism monitors its own resource consumption.

Usage:
    governor = ResourceGovernor(event_bus=ctx.bus)
    governor.register_source("sec_edgar", hourly_limit=100)
    governor.register_source("finnhub", hourly_limit=500)

    # Before each API call:
    if governor.can_call("sec_edgar"):
        governor.record_call("sec_edgar")
        # ... make the API call
    else:
        # throttled — skip or queue
"""
from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

CH_RESOURCE_THROTTLE = "market.resource.throttle"
CH_RESOURCE_WARNING = "market.resource.budget_warning"


@dataclass
class SourceBudget:
    """Budget configuration for a single API source."""

    name: str
    hourly_limit: int = 1000
    warn_at: float = 0.8  # warn at 80% usage
    calls: deque = field(default_factory=lambda: deque(maxlen=5000))
    throttled_since: float = 0.0
    total_calls: int = 0
    total_throttled: int = 0


class ResourceGovernor:
    """Self-governing API budget system.

    Tracks rolling 1-hour call counts per source. Publishes throttle
    events when budgets are exceeded. Provides statistics for the
    SomaticMap/HolonProxy interface.
    """

    def __init__(
        self,
        event_bus: Any = None,
        global_hourly_limit: int = 5000,
    ) -> None:
        self._bus = event_bus
        self._global_limit = global_hourly_limit
        self._sources: dict[str, SourceBudget] = {}
        self._lock = threading.RLock()
        self._global_calls: deque = deque(maxlen=10000)

    def register_source(
        self,
        source_name: str,
        hourly_limit: int = 1000,
        warn_at: float = 0.8,
    ) -> None:
        """Register a source with its hourly call budget."""
        with self._lock:
            self._sources[source_name] = SourceBudget(
                name=source_name,
                hourly_limit=hourly_limit,
                warn_at=warn_at,
            )

    def can_call(self, source_name: str) -> bool:
        """Check if source has budget remaining. Non-blocking."""
        now = time.time()
        cutoff = now - 3600.0
        with self._lock:
            budget = self._sources.get(source_name)
            if budget is None:
                return True  # unregistered = unlimited

            # Count calls in the last hour
            recent = sum(1 for t in budget.calls if t > cutoff)
            if recent >= budget.hourly_limit:
                if budget.throttled_since == 0.0:
                    budget.throttled_since = now
                    self._publish_throttle(source_name, recent, budget.hourly_limit)
                budget.total_throttled += 1
                return False

            # Check global budget
            global_recent = sum(1 for t in self._global_calls if t > cutoff)
            if global_recent >= self._global_limit:
                self._publish_throttle(
                    "__global__", global_recent, self._global_limit
                )
                return False

            # Warn at threshold
            ratio = recent / budget.hourly_limit if budget.hourly_limit > 0 else 0
            if ratio >= budget.warn_at and self._bus is not None:
                try:
                    self._bus.publish(CH_RESOURCE_WARNING, {
                        "source": source_name,
                        "usage_ratio": ratio,
                        "calls_last_hour": recent,
                        "hourly_limit": budget.hourly_limit,
                    })
                except Exception:
                    pass

            budget.throttled_since = 0.0
            return True

    def record_call(self, source_name: str) -> None:
        """Record an API call for a source."""
        now = time.time()
        with self._lock:
            budget = self._sources.get(source_name)
            if budget is not None:
                budget.calls.append(now)
                budget.total_calls += 1
            self._global_calls.append(now)

    def get_usage(self, source_name: str) -> dict[str, Any]:
        """Get current usage stats for a source."""
        now = time.time()
        cutoff = now - 3600.0
        with self._lock:
            budget = self._sources.get(source_name)
            if budget is None:
                return {"source": source_name, "registered": False}
            recent = sum(1 for t in budget.calls if t > cutoff)
            return {
                "source": source_name,
                "registered": True,
                "calls_last_hour": recent,
                "hourly_limit": budget.hourly_limit,
                "usage_ratio": recent / budget.hourly_limit if budget.hourly_limit > 0 else 0,
                "throttled": budget.throttled_since > 0,
                "total_calls": budget.total_calls,
                "total_throttled": budget.total_throttled,
            }

    def get_statistics(self) -> dict[str, Any]:
        """Full stats for HolonProxy/SomaticMap integration."""
        now = time.time()
        cutoff = now - 3600.0
        with self._lock:
            global_recent = sum(1 for t in self._global_calls if t > cutoff)
            source_stats = {}
            for name, budget in self._sources.items():
                recent = sum(1 for t in budget.calls if t > cutoff)
                source_stats[name] = {
                    "calls_last_hour": recent,
                    "hourly_limit": budget.hourly_limit,
                    "usage_ratio": recent / budget.hourly_limit if budget.hourly_limit > 0 else 0,
                    "throttled": budget.throttled_since > 0,
                }
            return {
                "global_calls_last_hour": global_recent,
                "global_hourly_limit": self._global_limit,
                "sources_registered": len(self._sources),
                "sources_throttled": sum(
                    1 for b in self._sources.values() if b.throttled_since > 0
                ),
                "sources": source_stats,
            }

    def _publish_throttle(
        self, source: str, current: int, limit: int
    ) -> None:
        if self._bus is not None:
            try:
                self._bus.publish(CH_RESOURCE_THROTTLE, {
                    "source": source,
                    "calls_last_hour": current,
                    "hourly_limit": limit,
                    "timestamp": time.time(),
                })
            except Exception:
                pass
        logger.warning(
            "ResourceGovernor: %s throttled (%d/%d calls/hour)",
            source, current, limit,
        )
