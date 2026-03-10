"""SystemHealthMonitor — aggregate infrastructure health for MIDGE.

Tracks per-subsystem error rates from the market step hooks and
per-operation latency from StepTimer. Classifies overall health into
four tiers (Green / Yellow / Orange / Red) and publishes a change event
on the EventBus whenever the tier transitions.

Health tier semantics
---------------------
Green  — all subsystems healthy (< 5 errors per error_window)
Yellow — 1-2 subsystems degraded (>= 5 errors per error_window)
Orange — 3+ subsystems degraded, or any subsystem failed (>= 20 errors)
Red    — a *core* subsystem has failed (convergence_check, thompson,
          sensing, outcome_evaluation)

Core subsystems trigger Red immediately because their failure prevents
MIDGE from discovering or evaluating any patterns — no other part of
the system can compensate.

Usage::

    monitor = SystemHealthMonitor(event_bus=ctx.bus, step_timer=ctx.step_timer)

    # In a try/except inside a market hook:
    try:
        run_convergence_check()
        monitor.record_success("convergence_check")
    except Exception as exc:
        monitor.record_error("convergence_check", exc)

    # Query state:
    tier   = monitor.evaluate_health()
    stats  = monitor.get_statistics()
    report = monitor.get_latency_report()

Pattern
-------
Follows ResourceGovernor: threading.RLock, EventBus injection as
``event_bus`` kwarg stored as ``self._bus``, ``get_statistics()`` for
HolonProxy, graceful degradation when dependencies are None.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Channel constant — will be added to channels.py by Wiring Builder in Round 2.
# Using the string literal here avoids a circular-import or missing-constant
# error until wiring is complete.
CH_HEALTH_TIER_CHANGE = "market.health.tier_change"

# Subsystems whose failure immediately escalates the tier to Red.
CORE_SUBSYSTEMS: frozenset[str] = frozenset(
    {"convergence_check", "thompson", "sensing", "outcome_evaluation"}
)

# Error-count thresholds
_DEGRADED_THRESHOLD = 5    # >= this many errors → "degraded"
_FAILED_THRESHOLD = 20     # >= this many errors → "failed"


class SystemHealthMonitor:
    """Infrastructure health aggregator.

    Aggregates per-subsystem error counts (recorded by market hooks) and
    per-operation latency (read from StepTimer). Derives an overall health
    tier and publishes a change event whenever the tier transitions.

    Thread-safe: all mutable state is protected by ``_lock`` (RLock so that
    ``record_error`` can call ``evaluate_health`` without deadlocking on a
    re-entrant acquisition).

    Args:
        event_bus: Optional EventBus for publishing tier-change events.
        step_timer: Optional StepTimer whose latency data is surfaced via
            ``get_latency_report()``. May be None — all latency-related
            paths degrade gracefully.
        error_window: Maximum number of error timestamps retained per
            subsystem. Older errors automatically fall off (deque maxlen).
        latency_threshold_ms: Informational threshold in milliseconds.
            Used by ``get_latency_report()`` to flag operations that
            exceed the limit.
    """

    def __init__(
        self,
        event_bus: Any = None,
        step_timer: Any = None,
        error_window: int = 100,
        latency_threshold_ms: float = 5000.0,
    ) -> None:
        self._bus = event_bus
        self._step_timer = step_timer
        self._error_window = error_window
        self._latency_threshold_ms = latency_threshold_ms

        self._lock = threading.RLock()

        # Per-subsystem error timestamps — deque maxlen enforces rolling window.
        self._error_counts: dict[str, deque] = {}

        # Per-subsystem health strings ("healthy" | "degraded" | "failed").
        self._subsystem_health: dict[str, str] = {}

        # Overall tier starts at "green" — we haven't seen any failures yet.
        self._overall_tier: str = "green"

    # =========================================================================
    # Error / Success recording
    # =========================================================================

    def record_error(
        self, subsystem: str, error: Optional[Exception] = None
    ) -> None:
        """Record a single error for *subsystem*.

        Appends the current timestamp to the subsystem's rolling error
        deque, re-evaluates the subsystem's health classification, and
        then recalculates the overall health tier (publishing a change
        event if the tier has shifted).

        Args:
            subsystem: Logical name of the failing subsystem (e.g.
                ``"convergence_check"``, ``"sensing"``).
            error: The exception that was caught, or None. Currently
                stored implicitly through the error count; the exception
                itself is logged at DEBUG level for diagnostics.
        """
        if error is not None:
            logger.debug(
                "SystemHealthMonitor: error in %s — %s", subsystem, error
            )
        with self._lock:
            if subsystem not in self._error_counts:
                self._error_counts[subsystem] = deque(maxlen=self._error_window)
                self._subsystem_health[subsystem] = "healthy"

            self._error_counts[subsystem].append(time.time())
            self._classify_subsystem(subsystem)
            self._evaluate_and_publish()

    def record_success(self, subsystem: str) -> None:
        """Record a successful operation for *subsystem*.

        Resets the subsystem's health classification to "healthy" and
        clears its error deque, then re-evaluates the overall tier.

        Calling this is optional — the monitor functions correctly if
        only ``record_error`` is called — but calling it allows the tier
        to recover after a burst of failures clears.

        Args:
            subsystem: Logical name of the subsystem (same string used
                in ``record_error``).
        """
        with self._lock:
            if subsystem in self._error_counts:
                self._error_counts[subsystem].clear()
            self._subsystem_health[subsystem] = "healthy"
            self._evaluate_and_publish()

    # =========================================================================
    # Health evaluation
    # =========================================================================

    def evaluate_health(self) -> str:
        """Return the current overall health tier (string).

        Tiers (in decreasing order of health):
            ``"green"``  — all subsystems healthy
            ``"yellow"`` — 1-2 subsystems degraded
            ``"orange"`` — 3+ degraded, or any subsystem failed
            ``"red"``    — a core subsystem has failed

        This method re-derives the tier from current error counts each
        time it is called and is safe to call from any thread.

        Returns:
            One of ``"green"``, ``"yellow"``, ``"orange"``, ``"red"``.
        """
        with self._lock:
            return self._compute_tier()

    def is_degraded(self, subsystem: str) -> bool:
        """Return True if *subsystem* is degraded or failed.

        Args:
            subsystem: Logical subsystem name.

        Returns:
            True when the subsystem's health is ``"degraded"`` or
            ``"failed"``; False when healthy or when the subsystem has
            never been seen by this monitor.
        """
        with self._lock:
            status = self._subsystem_health.get(subsystem, "healthy")
            return status in ("degraded", "failed")

    # =========================================================================
    # Reporting
    # =========================================================================

    def get_latency_report(self) -> dict:
        """Return per-operation latency percentiles from StepTimer.

        Delegates entirely to ``StepTimer.get_statistics()`` so the
        monitor never duplicates latency storage.

        Returns:
            Dict mapping operation names to ``{p50_ms, p95_ms, max_ms,
            count, exceeds_threshold}`` dicts.  ``exceeds_threshold`` is
            True when ``max_ms`` exceeds ``latency_threshold_ms``.
            Returns an empty dict when ``step_timer`` is None.
        """
        if self._step_timer is None:
            return {}

        try:
            raw = self._step_timer.get_statistics()
        except Exception:
            logger.debug("SystemHealthMonitor: StepTimer.get_statistics() failed", exc_info=True)
            return {}

        report: dict = {}
        for op, data in raw.items():
            entry = dict(data)
            entry["exceeds_threshold"] = (
                data.get("max_ms", 0.0) > self._latency_threshold_ms
            )
            report[op] = entry
        return report

    def get_statistics(self) -> dict:
        """Full statistics for HolonProxy / SomaticMap integration.

        Returns:
            Dict with keys:
            - ``overall_tier`` (str)
            - ``subsystems`` (dict[str, dict]) — per-subsystem health
              string and error count for the rolling window
            - ``core_subsystems`` (list[str]) — the names in
              CORE_SUBSYSTEMS
            - ``latency_summary`` (dict) — abbreviated latency report
              (operations that exceed the threshold)
            - ``error_window`` (int)
            - ``latency_threshold_ms`` (float)
        """
        with self._lock:
            subsystem_info: dict = {}
            for name, dq in self._error_counts.items():
                subsystem_info[name] = {
                    "health": self._subsystem_health.get(name, "healthy"),
                    "errors_in_window": len(dq),
                }
            # Include subsystems that have only ever been marked "healthy"
            # via record_success but never had an error.
            for name, health in self._subsystem_health.items():
                if name not in subsystem_info:
                    subsystem_info[name] = {
                        "health": health,
                        "errors_in_window": 0,
                    }

            tier = self._compute_tier()
            latency = self.get_latency_report()
            # Abbreviate: only keep operations that exceed the threshold.
            latency_summary = {
                op: data
                for op, data in latency.items()
                if data.get("exceeds_threshold", False)
            }

        return {
            "overall_tier": tier,
            "subsystems": subsystem_info,
            "core_subsystems": sorted(CORE_SUBSYSTEMS),
            "latency_summary": latency_summary,
            "error_window": self._error_window,
            "latency_threshold_ms": self._latency_threshold_ms,
        }

    # =========================================================================
    # Private helpers
    # =========================================================================

    def _classify_subsystem(self, subsystem: str) -> None:
        """Update the health classification for a single subsystem.

        Must be called with ``_lock`` held.
        """
        count = len(self._error_counts.get(subsystem, []))
        if count >= _FAILED_THRESHOLD:
            self._subsystem_health[subsystem] = "failed"
        elif count >= _DEGRADED_THRESHOLD:
            self._subsystem_health[subsystem] = "degraded"
        else:
            self._subsystem_health[subsystem] = "healthy"

    def _compute_tier(self) -> str:
        """Derive the overall tier from current subsystem health.

        Must be called with ``_lock`` held (or called from a context that
        already holds it).

        Tier precedence (highest wins):
            Red    > Orange > Yellow > Green
        """
        degraded_count = 0
        any_failed = False
        core_failed = False

        for name, health in self._subsystem_health.items():
            if health == "failed":
                any_failed = True
                if name in CORE_SUBSYSTEMS:
                    core_failed = True
            elif health == "degraded":
                degraded_count += 1

        if core_failed:
            return "red"
        if any_failed or degraded_count >= 3:
            return "orange"
        if degraded_count >= 1:
            return "yellow"
        return "green"

    def _evaluate_and_publish(self) -> None:
        """Recompute tier and publish a change event if it has shifted.

        Must be called with ``_lock`` held.
        """
        new_tier = self._compute_tier()
        if new_tier == self._overall_tier:
            return

        old_tier = self._overall_tier
        self._overall_tier = new_tier

        degraded_subsystems = [
            name
            for name, health in self._subsystem_health.items()
            if health in ("degraded", "failed")
        ]

        logger.info(
            "SystemHealthMonitor: tier %s → %s (degraded: %s)",
            old_tier,
            new_tier,
            ", ".join(degraded_subsystems) or "none",
        )

        if self._bus is not None:
            try:
                self._bus.publish(
                    CH_HEALTH_TIER_CHANGE,
                    {
                        "old_tier": old_tier,
                        "new_tier": new_tier,
                        "degraded_subsystems": degraded_subsystems,
                        "timestamp": time.time(),
                    },
                )
            except Exception:
                logger.debug(
                    "SystemHealthMonitor: failed to publish tier change", exc_info=True
                )
