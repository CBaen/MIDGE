"""Market Task Handlers - Injects real execution behavior into OctopusArm.

OctopusArm._execute_current_task() is a stub that marks tasks completed
without doing anything. This module monkey-patches arms in a colony with
a dispatch version that routes market investigation tasks to real handlers.

Injected handlers:
- investigate_partial: Checks convergence for a ticker, fires alert if full
- archaeology_lookup: Matches ticker against active pattern stacks
- situation_check: Lifecycle management for developing situations (eviction)

Thread safety: _developing_situations is shared between EventBus callbacks
(fired from sensing threads) and step hooks. All reads/writes use _situations_lock.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..market.intelligence.convergence_alerter import ConvergenceAlerter
    from ..market.archaeology.pattern_watcher import PatternWatcher
    from ..backbone.event_bus import EventBus
    from .octopus_colony import OctopusColony

logger = logging.getLogger(__name__)

# Channel used for octopus investigation results.
# Builder 1 is adding CH_OCTOPUS_INVESTIGATION as a constant in octopus_signals.py.
# Using the string directly here to avoid import-time dependency on that change.
CH_OCTOPUS_INVESTIGATION = "market.intel.octopus_investigation"

# Maximum check_count before a developing situation is evicted.
MAX_SITUATION_CHECKS = 20
# Maximum age in step-increments before eviction (each check_count += 1 per step).
MAX_SITUATION_AGE_STEPS = 100


def inject_market_handlers(
    colony: "OctopusColony",
    convergence_alerter: "ConvergenceAlerter | None",
    pattern_watcher: "PatternWatcher | None",
    event_bus: "EventBus | None",
) -> int:
    """Inject market task handlers into every arm in the colony.

    Sets colony._developing_situations and colony._situations_lock, then
    monkey-patches _execute_current_task on every OctopusArm found across
    all octopuses in the colony.

    Args:
        colony: The OctopusColony whose arms will be patched.
        convergence_alerter: ConvergenceAlerter instance (may be None).
        pattern_watcher: PatternWatcher instance (may be None).
        event_bus: EventBus for publishing investigation results (may be None).

    Returns:
        Number of arms patched.
    """
    # Attach colony-level shared state for situation tracking.
    colony._developing_situations: dict[str, dict[str, Any]] = {}
    colony._situations_lock = threading.Lock()

    arms_patched = 0

    for octopus in colony.octopuses.values():
        # OctopusAgent wraps OctopusDistributedCognition which holds .arms dict.
        cognition = getattr(octopus, "cognition", None)
        if cognition is None:
            # Fallback: some test stubs may expose arms directly.
            cognition = octopus

        arms_dict = getattr(cognition, "arms", {})
        for arm in arms_dict.values():
            _patch_arm(
                arm,
                colony,
                convergence_alerter,
                pattern_watcher,
                event_bus,
            )
            arms_patched += 1

    logger.info(
        "market_task_handlers: patched %d arms across %d octopuses",
        arms_patched,
        len(colony.octopuses),
    )
    return arms_patched


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _patch_arm(arm: Any, colony: Any, convergence_alerter: Any,
               pattern_watcher: Any, event_bus: Any) -> None:
    """Monkey-patch a single OctopusArm with the dispatch executor."""

    # Build handler registry bound to this arm's context.
    handlers: dict[str, Any] = {
        "investigate_partial": _make_investigate_partial(
            colony, convergence_alerter, event_bus
        ),
        "archaeology_lookup": _make_archaeology_lookup(
            colony, pattern_watcher, event_bus
        ),
        "situation_check": _make_situation_check(colony),
    }
    arm._task_handlers = handlers

    # Capture originals for safe fallback.
    original_execute = arm.__class__._execute_current_task

    def _dispatch_execute(self: Any) -> None:
        """Dispatch to a registered handler or fall back to mark-completed."""
        if self.current_task is None:
            return

        task_type = self.current_task.task_type
        handler = getattr(self, "_task_handlers", {}).get(task_type)

        if handler:
            try:
                handler(self.current_task)
            except Exception:
                logger.exception(
                    "Arm %s: handler for '%s' raised", self.arm_id, task_type
                )
        else:
            # Unknown task type — original behaviour (mark completed silently).
            pass

        # Always mark completed and clear current_task regardless of handler outcome.
        with self._lock:
            self.current_task.status = "completed"
            self.task_history.append(self.current_task)
            self.state.workload = max(0.0, self.state.workload - 0.1)
            self.current_task = None

    # Bind as an instance method on the arm object (not the class).
    import types
    arm._execute_current_task = types.MethodType(_dispatch_execute, arm)


# ---------------------------------------------------------------------------
# Handler factories
# Each factory returns a callable(task) bound to the shared colony state.
# ---------------------------------------------------------------------------


def _make_investigate_partial(
    colony: Any,
    convergence_alerter: Any,
    event_bus: Any,
) -> Any:
    """Return handler for 'investigate_partial' tasks."""

    def _handle(task: Any) -> None:
        ticker: str = task.data.get("ticker", "")
        direction: str = task.data.get("direction", "bullish")

        if not ticker:
            logger.debug("investigate_partial: no ticker in task data")
            return

        # Update developing situation bookkeeping.
        key = f"{direction}:{ticker}"
        with colony._situations_lock:
            if key not in colony._developing_situations:
                colony._developing_situations[key] = {
                    "ticker": ticker,
                    "direction": direction,
                    "domains_seen": task.data.get("domains_seen", []),
                    "missing_domains": task.data.get("missing_domains", []),
                    "first_seen": time.time(),
                    "check_count": 0,
                }
            situation = colony._developing_situations[key]
            situation["check_count"] += 1

        # Ask convergence alerter whether a full alert now fires.
        if convergence_alerter is None:
            return

        check_fn = getattr(convergence_alerter, "check_ticker_convergence_for", None)
        if check_fn is None:
            return

        try:
            alert = check_fn(ticker)
        except Exception:
            logger.exception("investigate_partial: check_ticker_convergence_for raised")
            return

        if alert and event_bus:
            event_bus.publish(CH_OCTOPUS_INVESTIGATION, {
                "source": "investigate_partial",
                "ticker": ticker,
                "direction": direction,
                "alert": alert,
                "check_count": situation["check_count"],
            })
            logger.debug(
                "investigate_partial: full alert fired for %s (%s)", ticker, direction
            )

    return _handle


def _make_archaeology_lookup(
    colony: Any,
    pattern_watcher: Any,
    event_bus: Any,
) -> Any:
    """Return handler for 'archaeology_lookup' tasks."""

    def _handle(task: Any) -> None:
        ticker: str = task.data.get("ticker", "")

        if not ticker:
            logger.debug("archaeology_lookup: no ticker in task data")
            return

        if pattern_watcher is None:
            return

        get_stacks_fn = getattr(pattern_watcher, "get_active_stacks", None)
        if get_stacks_fn is None:
            return

        try:
            all_stacks = get_stacks_fn()
        except Exception:
            logger.exception("archaeology_lookup: get_active_stacks raised")
            return

        if not all_stacks:
            return

        # Filter to stacks that reference this ticker.
        matching = [
            s for s in all_stacks
            if getattr(s, "ticker", None) == ticker
            or (isinstance(s, dict) and s.get("ticker") == ticker)
        ]

        if matching and event_bus:
            event_bus.publish(CH_OCTOPUS_INVESTIGATION, {
                "source": "archaeology_lookup",
                "ticker": ticker,
                "template_matches": [
                    {
                        "template_id": (
                            getattr(m, "template_id", None)
                            or (m.get("template_id") if isinstance(m, dict) else None)
                        ),
                        "stacking_tier": (
                            getattr(m, "stacking_tier", None)
                            or (m.get("stacking_tier") if isinstance(m, dict) else None)
                        ),
                    }
                    for m in matching
                ],
                "match_count": len(matching),
            })
            logger.debug(
                "archaeology_lookup: %d stack match(es) for %s", len(matching), ticker
            )

    return _handle


def _make_situation_check(colony: Any) -> Any:
    """Return handler for 'situation_check' tasks."""

    def _handle(task: Any) -> None:
        ticker: str = task.data.get("ticker", "")
        direction: str = task.data.get("direction", "bullish")

        if not ticker:
            return

        key = f"{direction}:{ticker}"

        with colony._situations_lock:
            situation = colony._developing_situations.get(key)
            if situation is None:
                return

            situation["check_count"] += 1
            check_count = situation["check_count"]
            age_steps = check_count  # Each check corresponds to one step cycle.

            should_evict = (
                check_count > MAX_SITUATION_CHECKS
                or age_steps > MAX_SITUATION_AGE_STEPS
            )

            if should_evict:
                del colony._developing_situations[key]
                logger.debug(
                    "situation_check: evicted %s (check_count=%d)",
                    key,
                    check_count,
                )

    return _handle
