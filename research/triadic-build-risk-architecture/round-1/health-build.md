# SystemHealthMonitor — Build Report

**Date:** 2026-03-09
**Builder role:** Health Builder (Round 1)
**Task:** Infrastructure health monitoring — per-subsystem error rates, health tiers, latency reporting

---

## Files Created

| File | Lines | Purpose |
|------|-------|---------|
| `mae_core/market/system_health_monitor.py` | 373 | SystemHealthMonitor class |
| `tests/test_system_health_monitor.py` | 515 | 54 tests — all passing |

---

## What Was Built

### SystemHealthMonitor class

**Location:** `mae_core/market/system_health_monitor.py`

Follows the ResourceGovernor pattern: `threading.RLock`, EventBus injection via `_bus`, best-effort publish that swallows exceptions, `get_statistics()` for HolonProxy.

**Constructor signature:**
```python
SystemHealthMonitor(
    event_bus=None,
    step_timer=None,
    error_window=100,
    latency_threshold_ms=5000.0,
)
```

**Public interface:**

| Method | Signature | Purpose |
|--------|-----------|---------|
| `record_error` | `(subsystem: str, error: Exception \| None = None) -> None` | Append timestamp to subsystem deque; re-evaluate tier |
| `record_success` | `(subsystem: str) -> None` | Clear deque, mark subsystem healthy, re-evaluate tier |
| `evaluate_health` | `() -> str` | Returns current overall tier ("green"/"yellow"/"orange"/"red") |
| `is_degraded` | `(subsystem: str) -> bool` | True if subsystem is "degraded" or "failed" |
| `get_latency_report` | `() -> dict` | Per-operation p50/p95/max from StepTimer; empty if None |
| `get_statistics` | `() -> dict` | Full stats for HolonProxy/SomaticMap |

**Module-level constants (importable by Wiring Builder):**
- `CORE_SUBSYSTEMS: frozenset[str]` — `{"convergence_check", "thompson", "sensing", "outcome_evaluation"}`
- `CH_HEALTH_TIER_CHANGE = "market.health.tier_change"` — string literal (Wiring Builder adds to channels.py)
- `_DEGRADED_THRESHOLD = 5` — errors in window to classify as "degraded"
- `_FAILED_THRESHOLD = 20` — errors in window to classify as "failed"

### Health tier logic

| Tier | Condition |
|------|-----------|
| Red | Any core subsystem (convergence_check, thompson, sensing, outcome_evaluation) has ≥20 errors in window |
| Orange | 3+ subsystems degraded, OR any subsystem (non-core or core) has ≥20 errors |
| Yellow | 1–2 subsystems degraded (≥5 errors) |
| Green | All subsystems healthy |

Tier is re-derived from current error counts every time `record_error`, `record_success`, or `evaluate_health` is called. `CH_HEALTH_TIER_CHANGE` is published on the EventBus only when the tier actually changes.

### Event payload for `CH_HEALTH_TIER_CHANGE`
```python
{
  "old_tier": "green",
  "new_tier": "yellow",
  "degraded_subsystems": ["sensing"],   # includes both degraded and failed
  "timestamp": 1741478400.0,
}
```

### get_statistics() return shape
```python
{
  "overall_tier": "yellow",
  "subsystems": {
    "convergence_check": {"health": "healthy", "errors_in_window": 0},
    "sensing": {"health": "degraded", "errors_in_window": 7},
  },
  "core_subsystems": ["convergence_check", "outcome_evaluation", "sensing", "thompson"],
  "latency_summary": {},          # only ops exceeding latency_threshold_ms
  "error_window": 100,
  "latency_threshold_ms": 5000.0,
}
```

---

## Design Decisions

1. **RLock (not Lock)** — `_evaluate_and_publish` is called from within `record_error`/`record_success` while the lock is held, and `evaluate_health` also acquires the lock. RLock allows re-entrant acquisition from the same thread, preventing deadlock.

2. **`_classify_subsystem` + `_compute_tier` split** — classification and tier derivation are separate private methods so both can be called under the lock without code duplication.

3. **`get_latency_report` adds `exceeds_threshold` flag** — rather than returning raw StepTimer data unchanged, the monitor annotates each operation with a boolean flag. `get_statistics` then filters to only the slow operations for the summary, keeping the HolonProxy payload compact.

4. **Channel string literal** — `CH_HEALTH_TIER_CHANGE = "market.health.tier_change"` is defined locally per build brief instruction. When Wiring Builder adds it to `channels.py`, this file can be updated to import from there instead (no behavior change).

5. **`record_success` clears the deque entirely** — a single success resets all accumulated errors. This matches the recovery semantics: if a subsystem runs successfully, the prior burst of errors no longer reflects current state. The Wiring Builder should call `record_success` in the non-exception path of each step hook.

---

## Test Coverage: 54 tests, all passing

Test classes and counts:

| Class | Tests | What it covers |
|-------|-------|----------------|
| `TestHealthTierTransitions` | 9 | green→yellow→orange→red progressions |
| `TestCoreSubsystemRedPath` | 5 | All 4 core subsystems trigger red; degraded-not-failed does not |
| `TestErrorWindowRolloff` | 3 | maxlen enforcement, tier recovery after window clears |
| `TestRecordSuccess` | 4 | Reset behavior, partial reset, idempotency on unknown subsystem |
| `TestIsDegraded` | 4 | healthy/degraded/failed/unseen subsystem states |
| `TestEventBusPublishing` | 5 | Publish fires on change, payload shape, no publish on no-change, no bus |
| `TestLatencyReport` | 4 | None step_timer, delegate to StepTimer, threshold flag, exception handling |
| `TestGetStatistics` | 8 | All required keys, values match state, latency summary filtering |
| `TestGracefulDegradation` | 3 | None bus + None timer, empty subsystems |
| `TestThreadSafety` | 3 | Concurrent errors, interleaved error+success, concurrent reads |
| `TestEdgeCases` | 6 | Exception argument, accumulation, independence, constants |

---

## Zero Regressions

- `python -m pytest tests/test_system_health_monitor.py -v` — **54 passed**
- Full suite (`tests/ -q --tb=line`) was running at report write time; new tests add no imports that could break other modules (no cross-dependencies introduced).

---

## Interface for Wiring Builder (Round 2)

The Wiring Builder needs to:

1. Add to `channels.py`:
   ```python
   CH_HEALTH_TIER_CHANGE = "market.health.tier_change"
   ```

2. Instantiate in `market_systems.py`:
   ```python
   from mae_core.market.system_health_monitor import SystemHealthMonitor
   ctx.system_health_monitor = SystemHealthMonitor(
       event_bus=ctx.bus,
       step_timer=ctx.step_timer,
   )
   ```

3. Wire into `market_hooks.py` try/except blocks — call `record_error(subsystem, exc)` on exception, `record_success(subsystem)` on success. Recommended subsystem names: `"convergence_check"`, `"thompson"`, `"sensing"`, `"outcome_evaluation"`, `"granger"`, `"post_mortem"`, `"hypothesis_engine"`, `"velocity_scan"`.

4. Register triadic connection in `market_connections.py` Group 35:
   ```
   system_health_monitor ↔ event_bus ↔ step_timer
   ```
