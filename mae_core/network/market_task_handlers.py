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
    pattern_library: Any = None,
    world_model: Any = None,
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
        pattern_library: PatternLibrary for historical template queries (may be None).
        world_model: WorldModel for causal chain queries (may be None).

    Returns:
        Number of arms patched.
    """
    # Attach colony-level shared state for situation tracking.
    # Reuse if already initialized (e.g. by bootstrap before EventBus wiring).
    if not hasattr(colony, "_developing_situations"):
        colony._developing_situations: dict[str, dict[str, Any]] = {}
    if not hasattr(colony, "_situations_lock"):
        colony._situations_lock = threading.Lock()

    # Attach pattern_library and world_model so handler closures can reach them
    # without needing ctx (handlers are arm-level, no direct ctx access).
    colony._pattern_library = pattern_library
    colony._world_model_ref = world_model

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

    # Store references so newly spawned arms can be patched via
    # patch_new_arm().  This closes the post-bootstrap spawn gap.
    colony._handler_refs = {
        "convergence_alerter": convergence_alerter,
        "pattern_watcher": pattern_watcher,
        "event_bus": event_bus,
    }

    logger.info(
        "market_task_handlers: patched %d arms across %d octopuses",
        arms_patched,
        len(colony.octopuses),
    )
    return arms_patched


def patch_new_arm(colony: "OctopusColony", arm: Any) -> bool:
    """Patch a single newly-spawned arm with market handlers.

    Call this from the auto-scaling spawn path so arms added after
    inject_market_handlers() still get the dispatch executor.

    Returns True if the arm was patched, False if handler refs are missing.
    """
    refs = getattr(colony, "_handler_refs", None)
    if refs is None:
        return False
    _patch_arm(
        arm,
        colony,
        refs["convergence_alerter"],
        refs["pattern_watcher"],
        refs["event_bus"],
    )
    return True


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

    # Close over the arm instance directly — no types.MethodType needed.
    # This function is set as arm._execute_current_task and called as arm._execute_current_task().
    _arm_ref = arm

    def _dispatch_execute() -> None:
        """Dispatch to a registered handler or fall back to mark-completed."""
        if _arm_ref.current_task is None:
            return

        task_type = _arm_ref.current_task.task_type
        handler = getattr(_arm_ref, "_task_handlers", {}).get(task_type)

        if handler:
            try:
                handler(_arm_ref.current_task)
            except Exception:
                logger.exception(
                    "Arm %s: handler for '%s' raised", _arm_ref.arm_id, task_type
                )
        # Unknown task type — original behaviour (mark completed silently).

        # Always mark completed and clear current_task regardless of handler outcome.
        with _arm_ref._lock:
            _arm_ref.current_task.status = "completed"
            _arm_ref.task_history.append(_arm_ref.current_task)
            _arm_ref.state.workload = max(0.0, _arm_ref.state.workload - 0.1)
            _arm_ref.current_task = None

    arm._execute_current_task = _dispatch_execute


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
            current_check = situation["check_count"]
            domains_seen = situation.get("domains_seen", [])
            causal_predictions = situation.get("causal_predictions", [])

        # Ask convergence alerter whether a full alert now fires.
        alert = None
        if convergence_alerter is not None:
            check_fn = getattr(convergence_alerter, "check_ticker_convergence_for", None)
            if check_fn is not None:
                try:
                    alert = check_fn(ticker)
                except Exception:
                    logger.exception("investigate_partial: check_ticker_convergence_for raised")

        # --- Pattern library: find historical templates matching domains seen ---
        historical_templates: list[dict] = []
        priority_request_created = False

        # pattern_library lives on the colony (injected by bootstrap) or on ctx.
        # The colony carries _handler_refs but not ctx directly.  We use a
        # module-level weak reference injected by inject_market_handlers (see below).
        pattern_library = getattr(colony, "_pattern_library", None)
        if pattern_library is not None and domains_seen:
            try:
                domains_set = set(domains_seen)
                matches = pattern_library.query_similar(
                    live_sources=domains_set,
                    direction=direction,
                )
                for m in matches:
                    tmpl = m.template
                    total = (tmpl.win_count or 0) + (tmpl.loss_count or 0)
                    win_rate = tmpl.win_count / total if total > 0 else 0.0
                    historical_templates.append({
                        "domain_signature": list(getattr(tmpl, "domains", [])),
                        "win_rate": round(win_rate, 3),
                        "instances": getattr(tmpl, "instance_count", total),
                        "cross_validated": getattr(tmpl, "cross_validated", False),
                    })
                    # Boost missing-domain polling when template is historically reliable
                    if win_rate > 0.6 and getattr(tmpl, "instance_count", total) >= 5:
                        _prio = getattr(colony, "_priority_requests", None)
                        if _prio is None:
                            colony._priority_requests = {}
                            _prio = colony._priority_requests
                        if len(_prio) < 50:
                            _prio[ticker] = {
                                "ticker": ticker,
                                "direction": direction,
                                "domains_seen": domains_seen,
                                "win_rate": win_rate,
                                "timestamp": time.time(),
                            }
                            priority_request_created = True
            except Exception:
                logger.debug("investigate_partial: pattern_library query failed", exc_info=True)

        # --- World model: root causes + ripple effects ---
        world_model = getattr(colony, "_world_model_ref", None)
        if world_model is not None:
            try:
                # Check if ticker maps to downstream effects of causal predictions
                for pred in causal_predictions[:3]:
                    trigger = pred if isinstance(pred, str) else pred.get("trigger", "")
                    if not trigger:
                        continue
                    ripples = world_model.find_ripple_effects(trigger)
                    for ripple in ripples[:5]:
                        if getattr(ripple, "ticker", "") == ticker:
                            logger.debug(
                                "investigate_partial: causal path confirmed %s -> %s (strength=%.2f)",
                                trigger, ticker, getattr(ripple, "strength", 0),
                            )
                # Root causes for this ticker
                root_causes = world_model.find_root_causes(ticker)
                if root_causes:
                    logger.debug(
                        "investigate_partial: %d root cause(s) for %s", len(root_causes), ticker
                    )
            except Exception:
                logger.debug("investigate_partial: world_model query failed", exc_info=True)

        # Publish investigation result if there is anything to report.
        if event_bus and (alert or historical_templates or priority_request_created):
            event_bus.publish(CH_OCTOPUS_INVESTIGATION, {
                "source": "investigate_partial",
                "ticker": ticker,
                "direction": direction,
                "alert": alert,
                "check_count": current_check,
                "historical_templates": historical_templates,
                "priority_request_created": priority_request_created,
            })
            logger.debug(
                "investigate_partial: published for %s (%s) check=%d templates=%d priority=%s",
                ticker, direction, current_check, len(historical_templates), priority_request_created,
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
