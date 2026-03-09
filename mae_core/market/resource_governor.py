"""ResourceGovernor — internal budget and API quota governance.

Tracks API call counts per source, enforces per-source and global
budgets, and publishes throttle events via EventBus. Replaces the
need for external governance (Paperclip) with Law 6 autopoietic
self-governance: the organism monitors its own resource consumption.

Priority tiers (DEB model):
    MAINTENANCE — never throttled (e.g. health checks, heartbeats)
    ACTIVE      — protected; gets a 1.5x budget multiplier under pressure
    EXPLORE     — expendable; throttled first when resources are tight

Usage:
    governor = ResourceGovernor(event_bus=ctx.bus)
    governor.register_source("sec_edgar", hourly_limit=100)
    governor.register_source("finnhub", hourly_limit=500)
    governor.set_source_tier("sec_edgar", SourceTier.ACTIVE)

    # Before each API call:
    if governor.can_call("sec_edgar"):
        governor.record_call("sec_edgar")
        # ... make the API call
    else:
        # throttled — skip or queue

    # Endocrine coupling (called by EndocrineSystem on cortisol change):
    governor.tighten_budgets(0.7)   # 30% reduction on all EXPLORE sources
    governor.relax_budgets(1.3)     # 30% increase on all EXPLORE sources
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

CH_RESOURCE_THROTTLE = "market.resource.throttle"
CH_RESOURCE_WARNING = "market.resource.budget_warning"


class SourceTier(str, Enum):
    """DEB priority tier for an API source.

    MAINTENANCE: Never throttled — always returns True from can_call().
        Use for health checks, heartbeats, organism-internal signals.
    ACTIVE: Protected — gets a 1.5x budget multiplier before the budget
        check. Throttled only when >150% of its nominal hourly limit.
    EXPLORE: Expendable — uses the standard budget check unchanged.
        Throttled first under pressure. tighten_budgets / relax_budgets
        only affect EXPLORE sources.
    """

    MAINTENANCE = "maintenance"
    ACTIVE = "active"
    EXPLORE = "explore"


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
    tier: SourceTier = SourceTier.EXPLORE


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
        tier: SourceTier = SourceTier.EXPLORE,
    ) -> None:
        """Register a source with its hourly call budget and priority tier."""
        with self._lock:
            self._sources[source_name] = SourceBudget(
                name=source_name,
                hourly_limit=hourly_limit,
                warn_at=warn_at,
                tier=tier,
            )

    def set_source_tier(self, source_name: str, tier: SourceTier) -> None:
        """Update the priority tier for a registered source.

        Args:
            source_name: Name of the source to update.
            tier: New SourceTier value.

        If the source is not registered this is a no-op (logs a warning).
        """
        with self._lock:
            budget = self._sources.get(source_name)
            if budget is None:
                logger.warning(
                    "ResourceGovernor.set_source_tier: %s not registered", source_name
                )
                return
            budget.tier = tier
        logger.debug(
            "ResourceGovernor: %s tier set to %s", source_name, tier.value
        )

    def can_call(self, source_name: str) -> bool:
        """Check if source has budget remaining. Non-blocking.

        Tier behaviour:
        - MAINTENANCE: always True (bypasses all budget checks).
        - ACTIVE: effective limit = hourly_limit * 1.5 before checking.
        - EXPLORE: standard budget check unchanged.
        """
        now = time.time()
        cutoff = now - 3600.0
        with self._lock:
            budget = self._sources.get(source_name)
            if budget is None:
                return True  # unregistered = unlimited

            # MAINTENANCE sources are never throttled.
            if budget.tier == SourceTier.MAINTENANCE:
                return True

            # Effective limit depends on tier.
            if budget.tier == SourceTier.ACTIVE:
                effective_limit = int(budget.hourly_limit * 1.5)
            else:
                effective_limit = budget.hourly_limit

            # Count calls in the last hour
            recent = sum(1 for t in budget.calls if t > cutoff)
            if recent >= effective_limit:
                if budget.throttled_since == 0.0:
                    budget.throttled_since = now
                    self._publish_throttle(source_name, recent, effective_limit)
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
            ratio = recent / effective_limit if effective_limit > 0 else 0
            if ratio >= budget.warn_at and self._bus is not None:
                try:
                    self._bus.publish(CH_RESOURCE_WARNING, {
                        "source": source_name,
                        "usage_ratio": ratio,
                        "calls_last_hour": recent,
                        "hourly_limit": budget.hourly_limit,
                        "effective_limit": effective_limit,
                        "tier": budget.tier.value,
                    })
                except Exception:
                    pass

            budget.throttled_since = 0.0
            return True

    def tighten_budgets(self, factor: float) -> None:
        """Reduce all EXPLORE source hourly budgets by factor.

        Called by EndocrineSystem on high-cortisol events.  Only EXPLORE
        sources are affected — ACTIVE and MAINTENANCE sources are untouched.

        Args:
            factor: Multiplier applied to hourly_limit.
                factor=0.7 means 30% reduction (limit *= 0.7).
                Must be in (0, 1] for meaningful tightening.
        """
        if factor <= 0:
            logger.warning("ResourceGovernor.tighten_budgets: factor must be > 0, got %s", factor)
            return
        with self._lock:
            for budget in self._sources.values():
                if budget.tier == SourceTier.EXPLORE:
                    budget.hourly_limit = max(1, int(budget.hourly_limit * factor))
        logger.debug(
            "ResourceGovernor.tighten_budgets: factor=%.2f applied to EXPLORE sources",
            factor,
        )

    def relax_budgets(self, factor: float) -> None:
        """Increase all EXPLORE source hourly budgets by factor.

        Called by EndocrineSystem on low-cortisol events.  Only EXPLORE
        sources are affected — ACTIVE and MAINTENANCE sources are untouched.

        Args:
            factor: Multiplier applied to hourly_limit.
                factor=1.3 means 30% increase (limit *= 1.3).
                Must be >= 1.0 for meaningful relaxation.
        """
        if factor < 1.0:
            logger.warning("ResourceGovernor.relax_budgets: factor < 1.0 (%s) — use tighten_budgets instead", factor)
        with self._lock:
            for budget in self._sources.values():
                if budget.tier == SourceTier.EXPLORE:
                    budget.hourly_limit = int(budget.hourly_limit * factor)
        logger.debug(
            "ResourceGovernor.relax_budgets: factor=%.2f applied to EXPLORE sources",
            factor,
        )

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
