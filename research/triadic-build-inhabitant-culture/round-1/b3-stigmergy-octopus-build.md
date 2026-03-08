# B3 Build Report: Stigmergy & Octopus — Round 1

**Builder:** B3 (Stigmergy & Octopus)
**Date:** 2026-03-08
**Files Modified:**
- `mae_core/bootstrap/market_hooks.py`
- `mae_core/network/octopus_colony.py`

**Files Created:**
- `tests/test_cultural_coordination.py`

---

## Task 1: Stigmergy Evaporation Step Hook

### What was built

Added a ~14-line block inside `_register_market_step_hooks()` at the `step % 50` cadence check, immediately before the existing velocity anomaly scan block. The new code:

1. Guards with `hasattr(ctx, 'stigmergy') and ctx.stigmergy is not None`
2. Calls `ctx.stigmergy.sense_markers(position=(0.0, 0.0, 0.0), radius=float("inf"), marker_types=None)` to trigger global decay
3. Logs at DEBUG: `"Stigmergy evaporation triggered (step %d)"`
4. Wraps in try/except with debug logging on failure — hook never raises

Updated the `logger.info` at the bottom of the function to include `stigmergy-evap/50` in the cadence string.

### Key decision: `sense_markers` signature

The build brief specified `marker_type=None` (singular), but the actual `StigmergicEnvironment.sense_markers()` takes `marker_types` (plural, a list or None). Used the correct `marker_types=None` to pass all marker types through. This is a no-op filter — the intent is to trigger `_apply_decay()` globally.

### Key decision: position=(0.0, 0.0, 0.0) vs (0.0, 0.0)

`StigmergicEnvironment` supports variable-dimension positions. Passing a 3-tuple is safe — `_distance()` uses `min(len(a), len(b))` for dimension-matching, and `_grid_key()` just divides each element. The 3-tuple avoids ambiguity if markers were deposited in 3D space.

### Why this matters

`_apply_decay()` removes expired markers but is only called lazily from `sense_markers()`. Without periodic reads, `convergence:{ticker}` markers accumulate without bound. After a ticker signal fades, stale markers continue influencing OctopusColony routing, pulling tasks toward octopuses that were once relevant to a position that no longer exists. The evaporation hook ensures the environment stays current.

---

## Task 2: Gradient-Based Task Routing in OctopusColony

### What was built

**Constructor:** Added optional `stigmergy: Any | None = None` parameter at the end of `__init__()`. Stored as `self._stigmergy`. Fully backward-compatible — all existing callers omitting the argument continue to work unchanged.

**`submit_task()`:** Replaced the single-line `min(octopuses.values(), key=lambda o: o.workload)` with a routing decision tree:

1. Extract `ticker = task_data.get("ticker")` if task_data is a dict.
2. If `self._stigmergy is not None` AND ticker is present:
   - Call `get_strongest_marker(position=(0.0, 0.0), marker_type=f"convergence:{ticker}", radius=float('inf'))`
   - If a marker is found: score each octopus as `workload * 0.7 + dist_penalty * 0.3`
     - Octopus positions are canonical: octopus at list index `i` lives at `(float(i), 0.0)`
     - `dist_penalty = distance_to_marker / max(len(octopuses)-1, 1)` — normalized to [0, 1]
   - If no marker: fall through to workload-only routing
   - All stigmergy access is wrapped in try/except — routing failure is not fatal
3. Fallback: `min(octopuses.values(), key=lambda o: o.workload)` (original behavior)

### Key decision: canonical octopus positions

Octopuses do not carry spatial state. The task brief says to score by "distance to marker position" but octopuses have no inherent position. The canonical layout places octopus at list index `i` at position `(i, 0.0)` — a stable 1D ring. This is:

- Deterministic (same octopus always at same position if colony is stable)
- Compatible with `StigmergicMarker.position` which is a 2-tuple by default
- Reviewable — any reviewer can reason about it without additional state

A more elaborate scheme (assigning spatial coordinates based on specialization or network topology) would require additional state and is premature. The 1D layout captures the gradient concept correctly and can be upgraded later.

### Key decision: 0.7/0.3 workload/distance split

The 70/30 split prioritizes availability over proximity — an overwhelmed octopus near the marker is worse than an idle octopus slightly further away. This matches the spirit of "prefer the octopus closest to that marker's position (by workload * 0.7 + distance_penalty * 0.3 scoring)" from the brief.

### Backward compatibility verification

`OctopusColony` is constructed in `mae_core/bootstrap/market_systems.py`. That call passes named arguments but not `stigmergy=`. The new parameter has `default=None` and is the last parameter, so the existing call signature is unchanged.

---

## Task 3: Tests

**File:** `tests/test_cultural_coordination.py`

**10 tests total, 10 passing.**

### TestStigmergyEvaporationStepHook (4 tests)

Uses a minimal ctx built by calling `_register_market_step_hooks()` directly — no full Mae organism needed. Captures the registered hook via `model.add_step_hook.side_effect`.

| Test | What it verifies |
|------|-----------------|
| `test_stigmergy_evaporation_triggers_decay` | sense_markers called once at step 50, not before |
| `test_stigmergy_evaporation_not_called_without_ctx_attribute` | No attribute → no AttributeError |
| `test_stigmergy_evaporation_not_called_when_none` | stigmergy=None → no call, no error |
| `test_stigmergy_evaporation_fires_again_at_100` | Cadence repeats: call_count=2 after 100 steps |

### TestOctopusColonyGradientRouting (6 tests)

Uses real `OctopusColony` instances with mock `OctopusAgent` submit_task overrides.

| Test | What it verifies |
|------|-----------------|
| `test_constructor_accepts_stigmergy_parameter` | New param stored as `_stigmergy` |
| `test_constructor_without_stigmergy_defaults_none` | Backward compat: `_stigmergy is None` |
| `test_submit_task_without_stigmergy_falls_back_to_workload` | Pure workload when no stigmergy |
| `test_submit_task_no_ticker_ignores_gradient` | Missing ticker key → workload only, get_strongest_marker not called |
| `test_submit_task_with_stigmergy_gradient_routes_to_nearest` | Marker at (2,0) + equal workload → octopus at index 2 chosen |
| `test_submit_task_gradient_falls_back_when_no_marker` | get_strongest_marker returns None → falls back to workload |

---

## Interfaces for Reviewers

**`ctx.stigmergy`**: `StigmergicEnvironment` instance (set by B4/B5 or elsewhere in bootstrap). If absent or None, evaporation is silently skipped.

**`OctopusColony._stigmergy`**: Set by constructor. `None` by default. If set, `submit_task()` uses gradient routing for ticker tasks; otherwise uses original workload routing.

**Pheromone key convention**: `f"convergence:{ticker}"` — e.g. `"convergence:AAPL"`. This key should be deposited by whoever emits convergence alerts with a ticker (likely in market_hooks.py or ConvergenceAlerter). B3 consumes the key; B3 does NOT deposit markers.

---

## What B3 Did NOT Do

- Did not wire `stigmergy` into `OctopusColony` during bootstrap (not assigned). Another builder or the orchestrator needs to pass `stigmergy=ctx.stigmergy` when constructing the colony in `market_systems.py`.
- Did not deposit `convergence:{ticker}` markers anywhere — that is a separate concern (likely belongs in market_hooks.py when a convergence alert fires, or in ConvergenceAlerter itself).
- Did not modify the `sense_markers` call signature — the brief had a minor inconsistency (`marker_type` singular vs actual `marker_types` plural). Used the correct API.

---

## Test Results

```
tests/test_cultural_coordination.py::TestStigmergyEvaporationStepHook::test_stigmergy_evaporation_triggers_decay PASSED
tests/test_cultural_coordination.py::TestStigmergyEvaporationStepHook::test_stigmergy_evaporation_not_called_without_ctx_attribute PASSED
tests/test_cultural_coordination.py::TestStigmergyEvaporationStepHook::test_stigmergy_evaporation_not_called_when_none PASSED
tests/test_cultural_coordination.py::TestStigmergyEvaporationStepHook::test_stigmergy_evaporation_fires_again_at_100 PASSED
tests/test_cultural_coordination.py::TestOctopusColonyGradientRouting::test_constructor_accepts_stigmergy_parameter PASSED
tests/test_cultural_coordination.py::TestOctopusColonyGradientRouting::test_constructor_without_stigmergy_defaults_none PASSED
tests/test_cultural_coordination.py::TestOctopusColonyGradientRouting::test_submit_task_without_stigmergy_falls_back_to_workload PASSED
tests/test_cultural_coordination.py::TestOctopusColonyGradientRouting::test_submit_task_no_ticker_ignores_gradient PASSED
tests/test_cultural_coordination.py::TestOctopusColonyGradientRouting::test_submit_task_with_stigmergy_gradient_routes_to_nearest PASSED
tests/test_cultural_coordination.py::TestOctopusColonyGradientRouting::test_submit_task_gradient_falls_back_when_no_marker PASSED

10 passed in 7.09s
```
