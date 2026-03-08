# Review 2 — Ecosystem Activation (Wire Octopus + Bridge Pipelines)
**Date:** 2026-03-08
**Reviewer:** Independent (adversarial-first protocol)

---

## (1) Integration Errors

### CRITICAL: `_on_partial_convergence` writes `_developing_situations` BEFORE `inject_market_handlers` guarantees the attribute exists

**File:** `mae_core/bootstrap/market_hooks.py` lines ~389-413

The EventBus callback `_on_partial_convergence` is registered in `_register_market_eventbus()`. It references `colony._developing_situations` and `colony._situations_lock`. These attributes are monkey-patched onto the colony object by `inject_market_handlers()` in `_wire_sensing_hook()`.

**Bootstrap call order in `market.py`:**
```
7. _register_market_eventbus    ← registers _on_partial_convergence callback
8. _register_market_step_hooks
9. _wire_sensing_hook           ← calls inject_market_handlers HERE
```

The callback is registered before `inject_market_handlers` runs. If an EventBus event fires on channel `market.intel.partial_convergence` between step 7 and step 9 (e.g., during `_register_market_step_hooks`), `colony._developing_situations` does not exist yet. The code guards with `hasattr(colony, "_developing_situations")` at line ~392, which prevents a crash — but the developing situation is silently dropped. Any partial convergence that fires during bootstrap is lost with no log. This is a correctness bug, not a crash.

**Fix:** Move the `inject_market_handlers` call to immediately after `_register_market_eventbus`, or move the attribute initialization into `OctopusColony.__init__` so it always exists before any callback can fire.

---

### CRITICAL: `ctx.octopus_colony` uses `world_model=getattr(ctx, "shared_world_model", None)` — attribute does not exist

**File:** `mae_core/bootstrap/market_systems.py` lines 452-458

```python
ctx.octopus_colony = OctopusColony(
    event_bus=getattr(ctx, "bus", None),
    min_octopuses=3,
    max_octopuses=7,
    world_model=getattr(ctx, "shared_world_model", None),   # ← "shared_world_model"
    signal_bus=getattr(ctx, "signal_bus", None),            # ← "signal_bus"
)
```

`ctx` carries `world_model` (instantiated at market_systems.py line ~342), not `shared_world_model`. The `getattr` guard means this silently passes `None`. The colony receives no world model. Whether `OctopusColony.__init__` requires it or treats it as optional needs verification, but this is a likely silent misconfiguration.

Similarly, `signal_bus` does not appear to be a standard ctx attribute in any bootstrap layer. This will also be `None` silently.

---

### MEDIUM: `inject_market_handlers` patches `_execute_current_task` on arms that may not exist yet

**File:** `mae_core/network/market_task_handlers.py` lines 69-77

The function iterates `colony.octopuses.values()` and for each octopus accesses `.cognition.arms`. If `OctopusColony` has been constructed but `start_monitoring()` has not yet spawned its minimum 3 octopuses, `colony.octopuses` may be empty or incomplete. `inject_market_handlers` is called before `start_monitoring()` (market_hooks.py lines 1432-1438):

```python
inject_market_handlers(colony=colony, ...)   # patches existing arms
colony.start_monitoring()                    # spawns octopuses AFTER patching
```

Any octopuses spawned by `start_monitoring()` or later auto-scaling will NOT have the patched `_execute_current_task`. The patch is applied once at boot, not injected into the spawning path. New arms added after `inject_market_handlers` runs get the original unpatched method.

---

## (2) Constraint Violations

### Mae Law 3 Violation: `octopus_colony` is registered as a holon but NOT placed in any K3 triad

**File:** `mae_core/bootstrap/market_registration.py` lines 137-138 and 199-231

`octopus_colony` is added to the flat `market_systems` list for holon registration, and it appears in the `_register_market_fractal` extras list (line ~224 would be expected). However, it is NOT in the extras list in `_register_market_fractal` — checking lines 200-231, the `extras` list ends with `"pattern_library", "pattern_watcher"` and does NOT include `"octopus_colony"`. The colony is registered as a holon but never reparented under `market-intelligence-system`. It is orphaned in the holon hierarchy.

**Evidence:** `_register_market_fractal` extras list (lines 200-225) does not contain `"octopus_colony"`.

---

### Mae Law 1 Partial Violation: Group 34 connections are guarded and may not be registered

**File:** `mae_core/bootstrap/market_connections.py` lines 441-452

Group 34 wraps all three connections in `if getattr(ctx, "octopus_colony", None) is not None`. If the colony fails to construct, no connections are registered for it — which is correct for graceful degradation. However, if the colony IS present, only 3 connections are registered (colony→convergence_alerter, colony→pattern_watcher, colony→event_bus). There is no connection from `convergence_alerter` or `pattern_watcher` back toward `octopus_colony`, meaning the triadic witness structure is asymmetric: the colony has outgoing witnessed connections but no incoming witnessed connections. This creates a directed graph with no reverse pathway, which is consistent with Law 1 only if the existing return path is established through prior groups. Worth verifying that Groups 14-33 cover the reverse direction.

---

### Mae Law 7 Violation: `validate_rule_of_3` is not called at construction time in bootstrap

**File:** `mae_core/bootstrap/market_systems.py` lines 449-461

`OctopusColony` enforces `min_octopuses=3` internally via `validate_rule_of_3`, but the bootstrap does not verify the colony achieved its minimum after construction. If `start_monitoring()` fails silently (wrapped in try/except at market_hooks.py line ~1441), the colony may be running with 0 octopuses. No post-construction check is logged.

---

## (3) Bugs / Logic Errors

### BUG: `_on_partial_convergence` extracts ticker by scanning signals with `for s in signals` but `signals` is a list of raw dicts with no guaranteed `"ticker"` key

**File:** `mae_core/bootstrap/market_hooks.py` lines ~395-404

```python
signals = msg.get("signals", [])
for s in signals:
    if "ticker" in s:
        ticker = s["ticker"]
        break
```

If ConvergenceAlerter emits partial signals where the ticker is stored under a different key (e.g., `"symbol"` — which is the key used in `ConvergenceAlert` dataclass), ticker will always be `None` and the developing situation will never be registered. This needs verification against the actual partial emission payload structure.

---

### BUG: Coordination cycle iterates `colony.octopuses.items()` in a step hook without holding any lock

**File:** `mae_core/bootstrap/market_hooks.py` lines 699-706

```python
for oct_id, oct in colony.octopuses.items():
    oct.cognition.run_coordination_cycle()
```

`colony.octopuses` is a dict that can be mutated by auto-scaling (spawn/despawn) running in a background thread. Iterating it without a lock risks `RuntimeError: dictionary changed size during iteration`. The try/except catches the crash but the coordination cycle is silently skipped — a dropped operation, not a handled one.

---

### BUG: `market_attrs` list in `market.py` includes `"octopus_colony"` for the active count

**File:** `mae_core/bootstrap/market.py` line 107

The log line still reads `"103 connections"` in the format string at line 116:
```python
"Layer 33  - Market Intelligence organ complete: %d systems, %d holons, 103 connections"
```

Group 34 adds 3 new connections, making the total 106 (if colony is present). The hardcoded `103` is stale.

---

## (4) Edge Cases

### If `octopus_colony` constructs but `inject_market_handlers` raises, `start_monitoring()` is never called

**File:** `mae_core/bootstrap/market_hooks.py` lines 1429-1441

```python
try:
    inject_market_handlers(...)
    colony.start_monitoring()   # only reached if inject succeeds
except Exception:
    logger.debug(...)
```

A partial failure in `inject_market_handlers` (e.g., an arm patching error on one of N octopuses) raises and exits the try block, so `start_monitoring()` never runs. The colony is constructed but dormant — silently non-functional. The log at line 1439 only fires on success, so there is no failure log distinguishing "colony failed to construct" from "colony constructed but monitoring never started."

---

### `_developing_situations` has no size cap

**File:** `mae_core/network/market_task_handlers.py`

`colony._developing_situations` is evicted by age (MAX_SITUATION_AGE_STEPS=100) and check count (MAX_SITUATION_CHECKS=20), but only if `situation_check` handler is called. If the colony is running zero or one octopus (e.g., after the bug in (3) above), situation_check never fires and the dict grows unbounded. Under sustained partial convergence emission this is a memory leak.

---

## (5) Regression Risk

### `convergence_alerter.py` — partial emission changes affect all existing tests

**File:** Not directly read; inferred from builder report.

`ConvergenceAlerter` was modified to emit on a new partial convergence channel. Any test that patches `ConvergenceAlerter.analyze()` or asserts on the number of EventBus emissions will be affected. Tests to check:
- `tests/test_convergence_alerter.py` — directly tests convergence logic
- `tests/test_integration.py` — bootstrap test; if convergence alerter behavior changed, signal counts may differ

---

### `patterns.py` translator registration — risk to Pattern Archaeology tests

`bootstrap_patterns` now registers market signal translators. If `PatternBus` or `PatternCortex` receive unexpected market signal types they were not designed to handle, tests in `tests/test_pattern_archaeology.py` and `tests/test_pattern_watcher.py` could see spurious signals or assertion failures.

---

## (6) Security

No new external inputs or trust boundary crossings introduced. OctopusColony receives data only from internal EventBus channels and `ConvergenceAlerter`. No new API surface. No findings.

---

## (7) Test Coverage Gaps

- No test verifies that `inject_market_handlers` is called BEFORE `start_monitoring()`.
- No test verifies that new octopuses spawned after `inject_market_handlers` get the patched `_execute_current_task` (the post-bootstrap spawn path is untested).
- No test exercises the `_on_partial_convergence` callback with a payload that has no `"ticker"` key — the silent no-op path is uncovered.
- No test verifies that `octopus_colony` is reparented under `market-intelligence-system` in the holon fractal.
- No test covers the coordination cycle dict-mutation race condition.
- `tests/test_octopus_bootstrap.py` was listed as new — not read due to context constraints; coverage quality unknown.

---

## (8) What Works

- The bootstrap ordering constraint (`_register_market_step_hooks` before `_wire_sensing_hook`) IS correctly maintained in `market.py`.
- `inject_market_handlers` correctly initializes `_situations_lock` as a `threading.Lock()` and all read/write paths in the handler functions use `with colony._situations_lock`.
- Group 34 connections correctly provide witnesses for all 3 registered connections — no bare dyads within Group 34 itself.
- `min_octopuses=3` is passed at construction, satisfying Law 7 at the construction call site.
- Graceful degradation is consistent: all construction is in try/except, all downstream references use `getattr(ctx, "octopus_colony", None)`.
- `market_attrs` in `market.py` correctly includes `"octopus_colony"` for the active system count.
- Somatic registration includes `octopus_colony` with correct dependencies listed.

---

## Summary — Ranked by Severity

| # | Severity | Finding |
|---|----------|---------|
| 1 | CRITICAL | `shared_world_model` / `signal_bus` attribute name mismatch — colony silently receives None |
| 2 | CRITICAL | EventBus callback registered before `inject_market_handlers` — race window where `_developing_situations` doesn't exist |
| 3 | HIGH | New arms spawned after `inject_market_handlers` are NOT patched — auto-scaling breaks handler injection |
| 4 | HIGH | `inject_market_handlers` failure silently prevents `start_monitoring()` — colony never activates, no error log |
| 5 | HIGH | `octopus_colony` not reparented in fractal hierarchy — Law 3 holon orphan |
| 6 | MEDIUM | Coordination cycle iterates `colony.octopuses` without lock — race with auto-scaling |
| 7 | MEDIUM | `ticker` extraction in `_on_partial_convergence` may use wrong key (`"symbol"` vs `"ticker"`) |
| 8 | MEDIUM | Hardcoded `"103 connections"` in market.py log is stale — should be 106 |
| 9 | LOW | `_developing_situations` unbounded if situation_check handler never fires |
| 10 | LOW | No test for post-bootstrap spawn patching gap |
