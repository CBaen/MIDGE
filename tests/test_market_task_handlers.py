"""Tests for mae_core/network/market_task_handlers.py"""

from __future__ import annotations

import threading
import types
from unittest.mock import MagicMock

from mae_core.network.market_task_handlers import (
    inject_market_handlers,
    _make_situation_check,
    MAX_SITUATION_CHECKS,
    CH_OCTOPUS_INVESTIGATION,
)
from mae_core.network.octopus_signals import ArmCapability, Task


# ---------------------------------------------------------------------------
# Helpers: minimal stubs so tests run without a full bootstrap
# ---------------------------------------------------------------------------


def _make_arm(arm_id: str = "arm-0") -> MagicMock:
    """Minimal OctopusArm stub with the attributes our code touches."""
    arm = MagicMock(spec=[
        "arm_id", "current_task", "task_history", "_lock", "state",
        "_task_handlers", "_execute_current_task",
    ])
    arm.arm_id = arm_id
    arm.current_task = None
    arm.task_history = []
    arm._lock = threading.RLock()
    arm.state = MagicMock()
    arm.state.workload = 0.5
    return arm


def _make_cognition(arms: dict) -> MagicMock:
    cognition = MagicMock()
    cognition.arms = arms
    return cognition


def _make_octopus(arms: dict) -> MagicMock:
    octopus = MagicMock()
    octopus.cognition = _make_cognition(arms)
    return octopus


def _make_colony(num_octopuses: int = 1, arms_per_octopus: int = 2) -> MagicMock:
    colony = MagicMock()
    octopuses = {}
    for oi in range(num_octopuses):
        arms = {f"arm-{oi}-{ai}": _make_arm(f"arm-{oi}-{ai}")
                for ai in range(arms_per_octopus)}
        octopuses[f"octopus_{oi}"] = _make_octopus(arms)
    colony.octopuses = octopuses
    return colony


def _make_task(task_type: str, data: dict = None) -> Task:
    t = Task(task_type=task_type, data=data or {})
    t.status = "pending"
    return t


# ---------------------------------------------------------------------------
# Test 1: inject_market_handlers sets _task_handlers on all arms
# ---------------------------------------------------------------------------


def test_inject_sets_handlers_on_arms():
    """After injection every arm must have a _task_handlers dict."""
    colony = _make_colony(num_octopuses=2, arms_per_octopus=3)

    inject_market_handlers(
        colony,
        convergence_alerter=None,
        pattern_watcher=None,
        event_bus=None,
    )

    for octopus in colony.octopuses.values():
        for arm in octopus.cognition.arms.values():
            assert hasattr(arm, "_task_handlers"), f"{arm.arm_id} missing _task_handlers"
            assert "investigate_partial" in arm._task_handlers
            assert "archaeology_lookup" in arm._task_handlers
            assert "situation_check" in arm._task_handlers


# ---------------------------------------------------------------------------
# Test 2: dispatch calls the right handler for a known task_type
# ---------------------------------------------------------------------------


def test_execute_dispatches_to_handler():
    """When a task_type matches a handler, that handler is called exactly once."""
    colony = _make_colony(num_octopuses=1, arms_per_octopus=1)
    inject_market_handlers(
        colony,
        convergence_alerter=None,
        pattern_watcher=None,
        event_bus=None,
    )

    arm = list(list(colony.octopuses.values())[0].cognition.arms.values())[0]

    # Replace the investigate_partial handler with a spy.
    called = []
    arm._task_handlers["investigate_partial"] = lambda task: called.append(task)

    task = _make_task("investigate_partial", {"ticker": "AAPL", "direction": "bullish"})
    arm.current_task = task

    arm._execute_current_task()

    assert len(called) == 1
    assert called[0].task_type == "investigate_partial"
    # current_task cleared
    assert arm.current_task is None
    assert task.status == "completed"


# ---------------------------------------------------------------------------
# Test 3: unknown task_type does not raise
# ---------------------------------------------------------------------------


def test_unknown_task_type_safe():
    """An unregistered task_type must not raise — just mark completed."""
    colony = _make_colony(num_octopuses=1, arms_per_octopus=1)
    inject_market_handlers(
        colony,
        convergence_alerter=None,
        pattern_watcher=None,
        event_bus=None,
    )

    arm = list(list(colony.octopuses.values())[0].cognition.arms.values())[0]
    task = _make_task("does_not_exist", {"foo": "bar"})
    arm.current_task = task

    # Should not raise.
    arm._execute_current_task()

    assert task.status == "completed"
    assert arm.current_task is None


# ---------------------------------------------------------------------------
# Test 4: developing situation lifecycle — check_count increments
# ---------------------------------------------------------------------------


def test_developing_situation_lifecycle():
    """situation_check handler increments check_count on each call."""
    colony = _make_colony()
    inject_market_handlers(
        colony,
        convergence_alerter=None,
        pattern_watcher=None,
        event_bus=None,
    )

    # Seed a developing situation manually.
    key = "bullish:TSLA"
    import time
    colony._developing_situations[key] = {
        "ticker": "TSLA",
        "direction": "bullish",
        "domains_seen": ["macro"],
        "missing_domains": ["insider"],
        "first_seen": time.time(),
        "check_count": 0,
    }

    arm = list(list(colony.octopuses.values())[0].cognition.arms.values())[0]
    task = _make_task("situation_check", {"ticker": "TSLA", "direction": "bullish"})
    arm.current_task = task
    arm._execute_current_task()

    with colony._situations_lock:
        assert colony._developing_situations[key]["check_count"] == 1

    # Call again.
    task2 = _make_task("situation_check", {"ticker": "TSLA", "direction": "bullish"})
    arm.current_task = task2
    arm._execute_current_task()

    with colony._situations_lock:
        assert colony._developing_situations[key]["check_count"] == 2


# ---------------------------------------------------------------------------
# Test 5: situation evicted after MAX_SITUATION_CHECKS
# ---------------------------------------------------------------------------


def test_situation_evicted_after_max_checks():
    """After check_count > MAX_SITUATION_CHECKS the entry is removed."""
    colony = _make_colony()
    inject_market_handlers(
        colony,
        convergence_alerter=None,
        pattern_watcher=None,
        event_bus=None,
    )

    import time
    key = "bearish:NVDA"
    colony._developing_situations[key] = {
        "ticker": "NVDA",
        "direction": "bearish",
        "domains_seen": ["technical"],
        "missing_domains": [],
        "first_seen": time.time(),
        "check_count": MAX_SITUATION_CHECKS,  # already at limit
    }

    arm = list(list(colony.octopuses.values())[0].cognition.arms.values())[0]
    # One more check pushes count to MAX + 1, triggering eviction.
    task = _make_task("situation_check", {"ticker": "NVDA", "direction": "bearish"})
    arm.current_task = task
    arm._execute_current_task()

    with colony._situations_lock:
        assert key not in colony._developing_situations, (
            f"Expected {key!r} to be evicted after MAX_SITUATION_CHECKS"
        )
