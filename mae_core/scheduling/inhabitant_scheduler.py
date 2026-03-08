"""InhabitantScheduler — wall-clock cadence dispatch for bio systems.

Generalizes OctopusColony's _monitoring_loop() pattern into a reusable
scheduler. Each registered system declares its own wall-clock interval;
the scheduler dispatches callbacks on that cadence via a thread pool.

Law 6 compliance: This is organism-internal scheduling. No external cron,
no OS scheduler, no external dependencies. The organism schedules itself.

Design:
- heapq priority queue keyed by (next_run_time, priority, system_name).
  Lower next_run_time fires first. Ties broken by priority then name.
- daemon=True: thread dies with the main process automatically.
- ThreadPoolExecutor: callbacks run in worker threads, not the scheduler
  thread, so a slow callback cannot block other dispatches.
- Thread safety: threading.Lock protects all heap mutations.
  threading.Event signals shutdown.

Constructor signature (for Builder 1 bootstrapping):
    InhabitantScheduler(event_bus=None, max_workers=4)

Usage:
    sched = InhabitantScheduler(event_bus=ctx.bus, max_workers=4)
    sched.register("my_system", my_callback, interval_seconds=30.0)
    sched.start()
    # ... later ...
    sched.stop()
"""

from __future__ import annotations

import heapq
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

CH_INHABITANT_DISPATCHED = "scheduling.inhabitant_dispatched"


@dataclass
class _ScheduleEntry:
    """One registered system in the schedule."""

    system_name: str
    callback: Callable[[], None]
    interval_seconds: float
    priority: int = 0
    run_count: int = 0
    last_run_time: Optional[float] = None

    def __lt__(self, other: "_ScheduleEntry") -> bool:
        # Used by heapq when (time, priority, name) tuples compare equal on time.
        return self.priority < other.priority


class InhabitantScheduler:
    """Daemon scheduler for organism-internal bio-system tasks.

    Each registered system runs on its own wall-clock cadence. Dispatch
    happens in a thread pool so slow callbacks do not starve the schedule.

    Args:
        event_bus: Optional EventBus. If provided, publishes
            "scheduling.inhabitant_dispatched" on each dispatch.
        max_workers: Maximum concurrent callback executions.
    """

    def __init__(
        self,
        event_bus: Any = None,
        max_workers: int = 4,
    ) -> None:
        self._bus = event_bus
        self._max_workers = max_workers

        # Registry: system_name -> ScheduleEntry
        self._entries: dict[str, _ScheduleEntry] = {}

        # Min-heap: (next_run_time, priority, system_name)
        # Lower next_run_time = fire sooner.
        # Higher priority int = lower heap priority (we negate below).
        self._heap: list[tuple[float, int, str]] = []

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._executor: Optional[ThreadPoolExecutor] = None

    # =========================================================================
    # Public API
    # =========================================================================

    def register(
        self,
        system_name: str,
        callback: Callable[[], None],
        interval_seconds: float,
        priority: int = 0,
    ) -> None:
        """Register a system for periodic dispatch.

        Args:
            system_name: Unique identifier for the system.
            callback: Zero-argument callable to invoke on each tick.
            interval_seconds: Wall-clock cadence in seconds.
            priority: Higher value = dispatched first when multiple tasks
                are due at the same time. Default 0.
        """
        entry = _ScheduleEntry(
            system_name=system_name,
            callback=callback,
            interval_seconds=interval_seconds,
            priority=priority,
        )
        with self._lock:
            self._entries[system_name] = entry
            next_run = time.monotonic() + interval_seconds
            # Negate priority so higher priority = lower heap key = fires first.
            heapq.heappush(self._heap, (next_run, -priority, system_name))

        logger.debug(
            "InhabitantScheduler: registered %s (interval=%.1fs, priority=%d)",
            system_name, interval_seconds, priority,
        )

    def unregister(self, system_name: str) -> None:
        """Remove a system from the schedule.

        The next time the heap pops this system's entry it will be
        discarded because it is no longer in _entries. This is the
        standard "lazy deletion" pattern for heapq — safe and O(1).
        """
        with self._lock:
            self._entries.pop(system_name, None)
        logger.debug("InhabitantScheduler: unregistered %s", system_name)

    def reschedule(self, system_name: str, interval_seconds: float) -> None:
        """Change the cadence of a registered system.

        The change takes effect on the next dispatch cycle.
        """
        with self._lock:
            entry = self._entries.get(system_name)
            if entry is None:
                logger.warning(
                    "InhabitantScheduler: reschedule(%s) — not registered",
                    system_name,
                )
                return
            entry.interval_seconds = interval_seconds
            next_run = time.monotonic() + interval_seconds
            heapq.heappush(self._heap, (next_run, -entry.priority, system_name))
        logger.debug(
            "InhabitantScheduler: rescheduled %s to %.1fs",
            system_name, interval_seconds,
        )

    def start(self) -> None:
        """Start the daemon thread and thread pool."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("InhabitantScheduler: already running")
            return

        self._stop_event.clear()
        self._executor = ThreadPoolExecutor(
            max_workers=self._max_workers,
            thread_name_prefix="inhabitant_worker",
        )
        self._thread = threading.Thread(
            target=self._dispatch_loop,
            daemon=True,
            name="InhabitantScheduler",
        )
        self._thread.start()
        logger.info(
            "InhabitantScheduler: started (max_workers=%d)", self._max_workers
        )

    def stop(self) -> None:
        """Signal the daemon thread to stop and wait for it to join."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
        logger.info("InhabitantScheduler: stopped")

    def get_statistics(self) -> dict[str, Any]:
        """Return schedule info, run counts, and last run times.

        Returns a dict with:
            registered_systems: count of registered systems
            systems: per-system dict with interval, run_count, last_run_time
            running: whether the daemon thread is alive
            max_workers: thread pool size
        """
        with self._lock:
            systems_info = {
                name: {
                    "interval_seconds": entry.interval_seconds,
                    "priority": entry.priority,
                    "run_count": entry.run_count,
                    "last_run_time": entry.last_run_time,
                }
                for name, entry in self._entries.items()
            }

        return {
            "registered_systems": len(systems_info),
            "systems": systems_info,
            "running": self._thread is not None and self._thread.is_alive(),
            "max_workers": self._max_workers,
        }

    # =========================================================================
    # Internal
    # =========================================================================

    def _dispatch_loop(self) -> None:
        """Main scheduler loop. Runs in the daemon thread.

        Algorithm:
        1. Peek at the top of the heap (soonest next_run_time).
        2. If now >= next_run_time, pop and dispatch.
        3. If the popped entry is no longer in _entries (lazy-deleted), skip.
        4. Otherwise dispatch in thread pool, update stats, re-push with new time.
        5. If next_run_time is in the future, sleep until then (or 0.1s max
           to remain responsive to stop_event).
        """
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception:
                logger.exception("InhabitantScheduler: error in dispatch loop")
            # Small sleep to yield CPU when nothing is due.
            # _tick() handles the precise sleep itself.

    def _tick(self) -> None:
        """One iteration of the dispatch loop."""
        now = time.monotonic()

        with self._lock:
            if not self._heap:
                # Nothing scheduled — sleep briefly and check again.
                pass
            else:
                next_run, neg_priority, system_name = self._heap[0]
                if now >= next_run:
                    # Pop and dispatch.
                    heapq.heappop(self._heap)
                    entry = self._entries.get(system_name)
                    if entry is None:
                        # Lazy-deleted — skip.
                        return
                    # Capture for closure
                    cb = entry.callback
                    name = system_name
                    interval = entry.interval_seconds
                    priority = entry.priority

                    # Schedule next run before dispatch (avoids drift).
                    new_next = now + interval
                    heapq.heappush(self._heap, (new_next, -priority, name))

                    # Update stats while still holding lock.
                    entry.run_count += 1
                    entry.last_run_time = time.time()
                else:
                    # Not yet due. Sleep until it is (max 0.1s for responsiveness).
                    sleep_duration = min(next_run - now, 0.1)
                    self._stop_event.wait(timeout=sleep_duration)
                    return

        # Dispatch outside the lock so callbacks don't hold it.
        if self._executor is not None:
            self._executor.submit(self._run_callback, name, cb)
        else:
            # Executor not started (shouldn't happen, but be safe).
            try:
                cb()
            except Exception:
                logger.exception(
                    "InhabitantScheduler: callback error for %s", name
                )

        # Publish dispatch event on EventBus (best-effort).
        if self._bus is not None:
            try:
                self._bus.publish(CH_INHABITANT_DISPATCHED, {
                    "system_name": name,
                    "timestamp": time.time(),
                })
            except Exception:
                pass  # Never let EventBus errors break the scheduler.

    def _run_callback(self, system_name: str, callback: Callable[[], None]) -> None:
        """Execute a callback in the thread pool. Catches all exceptions."""
        try:
            callback()
        except Exception:
            logger.exception(
                "InhabitantScheduler: callback error for %s", system_name
            )

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"InhabitantScheduler("
            f"systems={stats['registered_systems']}, "
            f"running={stats['running']}, "
            f"max_workers={self._max_workers})"
        )
