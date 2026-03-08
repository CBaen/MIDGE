# B5 New Systems Build Report — Round 1

**Builder:** B5 (New Systems — InhabitantScheduler + GovernanceLogger)
**Date:** 2026-03-08
**Build:** Round 1 of inhabitant culture activation

---

## Files Created

| File | Lines | Purpose |
|------|-------|---------|
| `mae_core/scheduling/__init__.py` | 9 | Package export for InhabitantScheduler |
| `mae_core/scheduling/inhabitant_scheduler.py` | 248 | Daemon scheduler for bio-system tasks |
| `mae_core/governance/__init__.py` | 9 | Package export for GovernanceLogger |
| `mae_core/governance/governance_logger.py` | 127 | Append-only governance event recorder |
| `tests/test_inhabitant_scheduler.py` | 284 | 24 tests covering all scheduler behaviors |
| `tests/test_governance_logger.py` | 256 | 24 tests covering all logger behaviors |

**Total: 48 tests, all passing.**

---

## InhabitantScheduler — Design Decisions

### Pattern source
Read `mae_core/network/octopus_colony.py` thoroughly before writing anything.
`_monitoring_loop()` is the seed pattern: a daemon thread that runs a `while self._running` loop, calls internal methods, and sleeps. `start_monitoring()` / `stop_monitoring()` wrap `threading.Thread(daemon=True)`.

InhabitantScheduler generalizes this into a reusable scheduler where _any system_ can register its own callback and interval, instead of hard-coding health/scaling logic inside the thread.

### Heap structure
`heapq` with key `(next_run_time, -priority, system_name)`. The negation of priority is intentional: heapq is a min-heap, so smaller key = pops first. A higher `priority` integer is negated to become a smaller key, which means it fires first. This preserves the intuitive API where `priority=10` fires before `priority=0`.

### Lazy deletion for unregister
`unregister()` removes the entry from `_entries` but does not touch the heap. Stale heap entries are silently discarded in `_tick()` when the system_name is no longer in `_entries`. This is O(1) for unregister and avoids a costly O(N) heap rebuild. It is the standard pattern for mutable heapq schedules in Python.

### Tick granularity
The dispatch loop calls `_tick()` continuously. When nothing is due, `_tick()` computes `sleep_duration = min(next_run_time - now, 0.1)` and calls `self._stop_event.wait(timeout=sleep_duration)`. The 0.1s cap keeps the thread responsive to `stop()` without busy-spinning.

### Thread safety
A single `threading.Lock` protects all heap mutations. Stats updates (run_count, last_run_time) happen inside the lock. The EventBus publish and callback dispatch happen *outside* the lock so a slow callback or slow bus cannot hold the lock and starve other dispatches.

### Constructor signature (for Builder 1)
```python
InhabitantScheduler(event_bus=None, max_workers=4)
```
Both parameters are optional. `event_bus=None` disables dispatch notifications. Registers callbacks via `register()` after construction, starts via `start()`.

---

## GovernanceLogger — Design Decisions

### Subscribed channels
Read `resource_governor.py` for the actual channel name strings
(`CH_RESOURCE_THROTTLE = "market.resource.throttle"`, `CH_RESOURCE_WARNING = "market.resource.budget_warning"`).
Read `senescence.py` for `CH_REJUVENATION = "emergent.rejuvenation_needed"` and
`CH_SENESCENT = "emergent.system_senescent"`.
Added `"scheduling.inhabitant_dispatched"` from InhabitantScheduler (this build).

### EventBus subscription mechanism
`event_bus.register_callback(channel, callback)` is the right API (not `subscribe()`). `subscribe()` only adds channels to the listen()-based queue. `register_callback()` wires synchronous delivery directly to the callback. Used this for all five channels.

### Atomic append
Each event opens the file in `"a"` mode, writes one line, and flushes. `open()` in append mode is atomic at the OS level for single-line writes. No explicit lock needed because EventBus delivers callbacks synchronously under its own RLock — callbacks on the same bus are never concurrent.

### Directory creation in constructor
`os.makedirs(parent, exist_ok=True)` runs at `__init__` time, not at first write. This ensures that if the first write fails for a reason other than directory absence, the error message is accurate (not a confusing "directory not found").

### Write failure handling
The entire `open() + write() + flush()` block is wrapped in `try/except OSError`. JSON serialization has its own `try/except (TypeError, ValueError)`. In both cases: log a warning, `return`. `event_count` and `last_event_time` are only updated on successful write.

### Constructor signature (for Builder 1)
```python
GovernanceLogger(event_bus, log_path="data/market/governance_log.jsonl")
```
`event_bus` is required (not optional) — the logger has no purpose without a bus. `log_path` has a sensible default.

---

## Test Design Decisions

### InhabitantScheduler tests
The scheduler uses real threads. Tests use `time.sleep()` with generous multipliers (3–6×) of the declared interval to avoid flakiness on slow CI. Using `threading.Event` callbacks with list.append() is thread-safe (GIL + lock).

The priority ordering test initially used `interval=10.0` with `max_workers=1` — this would need 10 seconds to fire once. Replaced with `interval=0.04` and a statistical invariant: over many cycles, `count(high) >= count(low)` because high-priority is popped first from the heap each cycle. This is both fast (0.4s) and robust.

### GovernanceLogger tests
All tests use `tmp_path` (pytest built-in fixture) so no test touches `data/market/governance_log.jsonl`. Write-failure tests use `patch("builtins.open", side_effect=OSError(...))` rather than constructing invalid paths — more portable across Windows/Linux and doesn't depend on filesystem behavior.

---

## Law Compliance

**Law 6 (Autopoietic Closure):**
- `InhabitantScheduler` has no external dependencies. It schedules organism-internal systems on wall-clock cadences using only Python stdlib (heapq, threading, concurrent.futures).
- `GovernanceLogger` subscribes to organism-internal EventBus channels and writes to a local file. No network, no external service.

**Law 3 (Holon Protocol):**
Both classes implement `get_statistics()` returning a dict. This satisfies the `know_self` capability. Builder 1 can wrap these in HolonProxy for full holon registration.

---

## Interface Summary for Builder 1 (Bootstrap Integration)

```python
# InhabitantScheduler
from mae_core.scheduling import InhabitantScheduler

sched = InhabitantScheduler(event_bus=ctx.bus, max_workers=4)
sched.register("my_bio_system", callback_fn, interval_seconds=30.0, priority=0)
sched.start()
# sched.stop() on shutdown
# sched.get_statistics() for HolonProxy

# GovernanceLogger
from mae_core.governance import GovernanceLogger

gov_logger = GovernanceLogger(
    event_bus=ctx.bus,
    log_path="data/market/governance_log.jsonl",
)
# No start() needed — subscribes at construction time.
# gov_logger.get_statistics() for HolonProxy
```

Both are self-contained and require only `event_bus` (or `None` for InhabitantScheduler) to function. No circular imports — neither imports any market or bootstrap module.
