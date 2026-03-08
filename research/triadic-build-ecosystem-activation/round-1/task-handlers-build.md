# Builder 3 Build Report — Task Handlers & Colony Infrastructure

**Date:** 2026-03-08
**Builder:** Builder 3 — Task Handlers & Colony Infrastructure
**Status:** Complete. All 5 tests passing.

---

## Files Created

### `mae_core/network/market_task_handlers.py` (193 lines)

Main export: `inject_market_handlers(colony, convergence_alerter, pattern_watcher, event_bus)`

**What it does:**
1. Sets `colony._developing_situations = {}` and `colony._situations_lock = threading.Lock()` on the colony object.
2. Iterates over every `OctopusAgent` in `colony.octopuses`, then into each agent's `.cognition.arms` dict, and patches each `OctopusArm` with a closure-based `_execute_current_task`.
3. Returns the count of arms patched.

**Arm traversal path discovered from reading the code:**
- `colony.octopuses` → dict of `OctopusAgent`
- `octopus.cognition` → `OctopusDistributedCognition`
- `cognition.arms` → dict of `OctopusArm`

A `getattr(octopus, "cognition", None)` fallback is used so test stubs that expose arms directly (without the intermediate cognition layer) also work.

**Monkey-patch design:**
- Uses a closure over `_arm_ref = arm` rather than `types.MethodType`. This keeps `_execute_current_task` as a plain callable set on the instance, meaning `arm._execute_current_task()` works correctly for both real arms and MagicMock stubs.
- Handler dispatch reads `arm.current_task.task_type`, looks it up in `arm._task_handlers`, calls it if found, then always marks `current_task.status = "completed"` and clears `current_task` under `arm._lock`.
- Exceptions inside a handler are caught and logged — they never prevent task completion.

**Three handlers implemented:**

`investigate_partial`:
- Gets `ticker` + `direction` from `task.data`
- Creates or increments a `_developing_situations` entry under `_situations_lock`
- Calls `convergence_alerter.check_ticker_convergence_for(ticker)` if available
- If the alerter returns a truthy alert, publishes `CH_OCTOPUS_INVESTIGATION` on event_bus

`archaeology_lookup`:
- Gets `ticker` from `task.data`
- Calls `pattern_watcher.get_active_stacks()` if available
- Filters for stacks matching the ticker (supports both object `.ticker` attr and dict `"ticker"` key)
- If matches found, publishes `CH_OCTOPUS_INVESTIGATION` with template match summary

`situation_check`:
- Gets ticker + direction, builds key `"{direction}:{ticker}"`
- Increments `check_count` under `_situations_lock`
- Evicts from `_developing_situations` when `check_count > MAX_SITUATION_CHECKS (20)` or `age_steps > MAX_SITUATION_AGE_STEPS (100)`

**Channel constant:**
Used the string `"market.intel.octopus_investigation"` directly (as instructed — Builder 1 is adding `CH_OCTOPUS_INVESTIGATION` to `octopus_signals.py` in parallel). The constant is also exported from this module as `CH_OCTOPUS_INVESTIGATION` so reviewers and callers can reference it without depending on Builder 1's work being merged first.

---

### `tests/test_market_task_handlers.py` (5 tests, all passing)

| Test | What it verifies |
|------|-----------------|
| `test_inject_sets_handlers_on_arms` | Every arm in a 2-octopus, 3-arms-each colony gets `_task_handlers` with all 3 keys |
| `test_execute_dispatches_to_handler` | Spy on `investigate_partial` slot; verify it's called exactly once and `current_task` is cleared |
| `test_unknown_task_type_safe` | Task with unregistered type completes without raising |
| `test_developing_situation_lifecycle` | Two consecutive `situation_check` calls produce `check_count == 2` |
| `test_situation_evicted_after_max_checks` | Entry at `check_count == MAX_SITUATION_CHECKS` is evicted after one more call |

**Stub design:**
Tests use `MagicMock(spec=[...])` with an explicit spec list so `del arm._task_handlers` (which fails on MagicMock) is never needed, and `arm._execute_current_task = func` assignment is permitted by the spec.

---

## Key Design Decisions

**Closure vs MethodType:** Binding via `types.MethodType` caused AttributeError on MagicMock stubs because Python resolves `self` via the class, not the instance dict. A closure over `_arm_ref` sidesteps this entirely and is simpler.

**Thread safety scope:** `_developing_situations` is written by `investigate_partial` and `situation_check` (called from arm processing threads) and could be read by step hooks (main thread). All dict reads and writes go through `colony._situations_lock`. The eviction `del` also occurs inside the lock.

**Graceful degradation:** Every external call (`convergence_alerter`, `pattern_watcher`, `event_bus`) is guarded with `if X is None` and `getattr(X, "method", None)` checks. If Builder 2's `check_ticker_convergence_for` method isn't present yet, the handler silently returns rather than crashing.

**No monkey-patching the class:** The dispatch function is set on the *instance*, not the class. This means existing arms that have not been patched (if any are created after injection) keep their original stub behaviour. Each call to `inject_market_handlers` re-patches fresh.

---

## Integration Notes for Reviewer

- `inject_market_handlers` should be called from `market_systems.py` (Layer 33 bootstrap) after the colony, convergence alerter, pattern watcher, and event bus are all constructed.
- The `check_ticker_convergence_for(ticker)` method on `ConvergenceAlerter` is being added by Builder 2. Until that lands, the `investigate_partial` handler is a no-op after updating the situation counter.
- `CH_OCTOPUS_INVESTIGATION = "market.intel.octopus_investigation"` is exported from this module. Once Builder 1 adds it to `octopus_signals.py`, the import can be switched there and the local definition removed.
