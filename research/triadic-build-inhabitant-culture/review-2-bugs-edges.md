# Review 2: Bugs, Edge Cases, Thread Safety
**Reviewer:** Independent (did not build this code)
**Scope:** Round 1 (B1–B6) + Round 2 (B1–B2) — Inhabitant Culture Activation
**Date:** 2026-03-08

---

## Methodology

Read all nine source files in full. Traced all concurrency paths manually. Ran targeted tests. Ran full suite (4,536 tests) in background. Findings are ordered by severity.

---

## 1. Integration Errors

### CRITICAL — `_dispatch_loop` spins at 100% CPU when heap is empty

**File:** `mae_core/scheduling/inhabitant_scheduler.py` **Lines 237–243, 249–252**

When no systems are registered, `_tick()` enters the `if not self._heap: pass` branch and returns immediately. `_dispatch_loop` then loops back, calls `_tick()` again, and repeats — with no sleep. This is a busy-wait loop that consumes 100% of one CPU core.

```python
# _dispatch_loop (line 237)
while not self._stop_event.is_set():
    try:
        self._tick()
    except Exception:
        ...
    # "Small sleep to yield CPU" — but _tick() only sleeps if heap is non-empty
    # and the next entry is in the future. Empty heap = no sleep at all.

# _tick() (line 249-252)
if not self._heap:
    # Nothing scheduled — sleep briefly and check again.
    pass   # <-- FALLS THROUGH, returns, loop spins immediately
```

The comment says "_tick() handles the precise sleep itself" but that is only true when the heap has a future entry. The empty-heap path has zero sleep.

**Severity: HIGH.** The scheduler starts before most systems register. In the window between `start()` and the first `register()`, and any time all systems are unregistered, the thread burns CPU. In production daemon mode with 24/7 operation this wastes a full core continuously.

**Fix:** Replace `pass` with `self._stop_event.wait(timeout=0.1)`.

---

### MEDIUM — `GovernanceLogger` has no lock protecting `_event_count` / `_last_event_time`

**File:** `mae_core/governance/governance_logger.py` **Lines 120–121**

`_on_event` is called synchronously by the EventBus callback system. If two governance events arrive from different threads simultaneously (e.g., ResourceGovernor throttle from the market thread + InhabitantScheduler dispatch from the scheduler thread), both callbacks execute concurrently. The increment and assignment on lines 120–121 are not protected by a lock:

```python
self._event_count += 1        # line 120 — not atomic in Python (read-modify-write)
self._last_event_time = time.time()  # line 121
```

Python's GIL makes most CPython integer increments safe in practice, but this is not guaranteed by the language spec and will be incorrect under alternative implementations or with future GIL removal (PEP 703). The file write itself (lines 111–117) uses a context manager that protects the individual write, but two threads can interleave between the `open()` call and the `write()` + `flush()` such that lines arrive out of order in the file — which is acceptable for append-only JSONL, but the counter desync is a real issue.

**Severity: MEDIUM.** Not a crash risk under CPython today, but correctness is not guaranteed and statistics can under-count if two events arrive in the same microsecond.

**Fix:** Add `threading.Lock()` in `__init__` and acquire it around the increment and timestamp assignment.

---

### LOW — `OctopusColony._check_health` mutates `self.octopuses` while iterating a snapshot, but `self.octopuses` has no lock

**File:** `mae_core/network/octopus_colony.py` **Lines 455–474**

`_check_health` builds the `unhealthy` list by iterating `self.octopuses`, then calls `spawn_octopus` (which modifies `self.octopuses`) and `despawn_octopus` (which also modifies `self.octopuses`) in the loop. These mutations happen on the monitoring thread, while `submit_task()` and other callers access `self.octopuses` from the main thread with no locking.

`spawn_octopus` (line 189) does `self.octopuses[octopus_id] = octopus` and `despawn_octopus` (line 237) does `del self.octopuses[octopus_id]` — both dictionary mutations that are not atomic with respect to concurrent reads like `submit_task`'s `for octopus in self.octopuses.values()` iteration on line 282.

**Severity: LOW.** Under CPython the GIL prevents true simultaneous execution, but dictionary size-change during iteration raises `RuntimeError: dictionary changed size during iteration`. This would surface under high-load real-time operation.

---

## 2. Constraint Violations

### MEDIUM — `tighten_budgets` is semantically broken for high-cortisol values

**File:** `mae_core/coordination/endocrine_system.py` **Lines 517–520**
**File:** `mae_core/market/resource_governor.py` **Lines 193–214**

The docstring on `register_resource_governor` says: "factor = cortisol level (e.g. 0.8 means reduce EXPLORE budgets by 20%)." But the actual call at line 520 passes the raw cortisol level directly to `tighten_budgets(level)`.

`tighten_budgets` interprets `factor` as a multiplier: `hourly_limit = max(1, int(hourly_limit * factor))`. So at cortisol = 0.8, the budget becomes 80% of its current value. That is correct for the first call. But every subsequent cortisol release above 0.6 compounds the reduction permanently — each event multiplies the already-reduced limit by 0.8 again.

At cortisol critical threshold (0.8), ten cortisol-release events in a session would shrink the budget to `1000 * 0.8^10 ≈ 107`. Twenty events: `≈ 11`. There is no mechanism to restore the original limit. `relax_budgets` only fires for cortisol < 0.3, and even then it applies a small increase to whatever the current floor has become — it cannot restore the original baseline because the original baseline is not stored.

**Severity: MEDIUM.** In a long-running daemon session with repeated market stress events, EXPLORE sources could be throttled to near-zero and stay there, effectively disabling exploration permanently until restart.

---

### LOW — `relax_budgets` does not proceed when `factor < 1.0`, but no floor exists on the floor

**File:** `mae_core/market/resource_governor.py` **Lines 216–235**

`relax_budgets` logs a warning when `factor < 1.0` but still executes the multiplication — there is no early return. This means passing `factor=0.5` produces a warning AND silently reduces the budget (same effect as `tighten_budgets`). The warning exists but has no enforcement.

```python
if factor < 1.0:
    logger.warning(...)
# No return — falls through to:
with self._lock:
    ...
    budget.hourly_limit = int(budget.hourly_limit * factor)
```

**Severity: LOW.** The endocrine wiring never produces `factor < 1.0` for `relax_budgets`, so this only fires in misuse scenarios. But for a self-governing system, "warn and proceed wrong" is worse than "warn and refuse."

---

## 3. Bugs and Logic Errors

### MEDIUM — `HomeostasisRegulator.compute_drive_urgency` can return values above 1.0

**File:** `mae_core/coordination/homeostasis.py` **Lines 281–303**

The docstring states: "each value is the urgency float in [0.0, 1.0]." The implementation does `min(1.0, urgency)` which correctly caps the result. However, `_compute_urgency` (lines 214–222) computes:

```python
return abs(setpoint.error) / setpoint.range_width
```

This returns values > 1.0 when the current value is outside the acceptable range by more than the range_width. For `threat_level` (range_width = 0.5, max_acceptable = 0.5), if current_value = 1.0, error = |0.1 - 1.0| = 0.9, urgency = 0.9 / 0.5 = 1.8. The `min(1.0, urgency)` cap on line 302 handles this correctly.

However, `_compute_urgency` is also called directly in `step()` (line 250) and in `get_statistics()` (line 347) without the cap. The uncapped value is stored in `correction_data["urgency"]` (line 258) which is published on the EventBus. Any subscriber receiving `urgency > 1.0` from the EventBus (including `OrganismState._on_homeostasis_correction` on line 291) would set `_homeostasis_deviation` to a value > 1.0.

`OrganismState.get_reflex_override()` uses `_homeostasis_deviation >= 0.7` as the trigger threshold. A value of 1.8 still triggers this correctly, so behavior is not broken. But the documented contract ("urgency in [0.0, 1.0]") is violated in the published event.

**Severity: MEDIUM.** Not a crash, but the published urgency values violate the documented contract. Any future consumer that clamps or normalizes against 1.0 would be confused.

**Fix:** Cap urgency before publishing: `"urgency": round(min(1.0, urgency), 4)` in `step()` line 258.

---

### LOW — `OrganBuilder._on_system_senescent` calls `prune_organs()` without `model` or `substrate`

**File:** `mae_core/morphogenesis/organ_builder.py` **Lines 502–542**

The handler calls `self.prune_organs()` with no arguments (line 531). `prune_organs` delegates to `dissolve_organ(model=None, substrate=None)` (line 471). When `model=None` and `substrate=None`, the dissolution skips agent removal from the Mesa model and substrate deregistration (lines 443–450), leaving dangling agent references in any live Mesa model. Organs marked `DISSOLVED` are removed from `_active_organs` but their Mesa agents continue to exist and will continue to execute `step()` each model tick.

This is partially documented via "metadata-only" organ mode, but in a real running model with actual Mesa agents, this is a resource leak.

**Severity: LOW.** Senescence is currently rare (wear threshold = 1.0). In production long-running runs the accumulation of zombie agents is real but slow.

---

### LOW — `OctopusColony.submit_task` computes `marker.position[1]` with a guard but marker position format is unknown

**File:** `mae_core/network/octopus_colony.py` **Lines 302–307**

```python
(oct_pos[1] - (marker.position[1] if len(marker.position) > 1 else 0.0)) ** 2
```

`marker.position` is accessed by index. The `len(marker.position) > 1` guard assumes `marker.position` is a sequence. If `marker.position` is a 2-tuple, this works. If it's a single float (scalar), `len()` raises `TypeError: object of type 'float' has no len()`. The entire block is wrapped in a try/except that falls back to workload routing, so this does not crash — but every task submission involving stigmergy triggers an exception and falls through silently. The stigmergy feature would be permanently disabled in that case without any visible error (the exception is logged at DEBUG level only, line 321).

**Severity: LOW.** The fallback prevents crashes but the feature silently degrades.

---

## 4. Edge Cases

### EDGE — `InhabitantScheduler._tick`: `name` and `cb` used after lock release but only defined inside the lock branch

**File:** `mae_core/scheduling/inhabitant_scheduler.py` **Lines 262–302**

`name` and `cb` are assigned inside the `if entry is not None:` block within the lock (lines 263–264). They are used after the lock is released at line 283. If the code path somehow reaches line 283 without going through the assignment (e.g., if `_tick` is restructured in the future), `NameError` would occur. Currently, the control flow guarantees assignment before use via the `return` on line 261, but the variables are scoped at function level and the dependency is implicit.

This is a latent fragility, not a current bug.

---

### EDGE — `InhabitantScheduler.stop()` uses `executor.shutdown(wait=False)` — in-flight callbacks continue after stop

**File:** `mae_core/scheduling/inhabitant_scheduler.py` **Line 191**

`stop()` joins the dispatch thread with a 10s timeout, then calls `executor.shutdown(wait=False)`. The dispatch thread stops submitting new callbacks after `_stop_event.is_set()`, so no new work is submitted. However, callbacks already submitted to the executor pool continue running. If a callback takes a long time, it may still be executing after `stop()` returns — the caller has no visibility into this.

The docstring says "wait for it to join" which implies clean shutdown. The current behavior is "stop submitting new work" but not "wait for in-flight work to complete."

**Severity: EDGE CASE.** Callers doing teardown after `stop()` may observe side effects from callbacks still running. Not a crash risk.

---

### EDGE — `EndocrineSystem.release_hormone` calls EventBus `publish` while holding `self._lock` (lines 207–215)

**File:** `mae_core/coordination/endocrine_system.py` **Lines 184–246**

`release_hormone` acquires `self._lock` at line 184 and holds it through the entire method body, including the `event_bus.publish()` call at line 207. EventBus callbacks fire synchronously within `publish()`. If any subscriber calls back into `release_hormone` (e.g., a cascade handler that calls `release_hormone` on another hormone), that reentrant call will also try to acquire `self._lock`. Since `self._lock` is a non-reentrant `threading.Lock()`, this would deadlock.

The current cascade implementation (`_apply_cascades`, line 646) directly modifies `self._levels` within the same lock acquisition, so it does not cause a second `acquire()`. But any external subscriber wired via `event_bus.register_callback("endocrine.hormone_release", ...)` that calls back into `release_hormone` would deadlock.

**Severity: MEDIUM (latent deadlock risk).** The existing code paths do not trigger this today. The `_lock` should be changed to `threading.RLock()` for safety, or the publish should happen after releasing the lock.

---

### EDGE — `HomeostasisRegulator.step` resets `_step_count` incorrectly for step 0

**File:** `mae_core/coordination/homeostasis.py` **Line 233**

```python
self._step_count = current_step if current_step > 0 else self._step_count + 1
```

If `current_step` is passed as `0` explicitly (which happens on the very first step of a simulation), the condition `current_step > 0` is False, so `_step_count` auto-increments instead of being set to 0. This is identical behavior to OrganismState line 490. The step counter starts at 1, never at 0. Minor inconsistency that could cause off-by-one issues in history slicing.

---

## 5. Regression Risk

### MEDIUM — No test covers `relax_budgets` permanent floor erosion

The compound-reduction bug described in Section 2 is not tested. `tests/test_drive_coupling.py` verifies that `tighten_budgets` is called and that `relax_budgets` receives a factor > 1.0 — but does not test multiple sequential cortisol events and verify the budget can be recovered. A soak test running 50 cortisol-release events and asserting the EXPLORE budget remains above some minimum would catch this.

---

### LOW — `test_priority_ordering` uses a probabilistic assertion that can flake on slow CI

**File:** `tests/test_inhabitant_scheduler.py` **Lines 262–267**

The test asserts `count_high >= count_low`. With `max_workers=1` and identical intervals, this is nearly always true. But on a heavily loaded system where thread scheduling delays cause one task to miss its window repeatedly, the count could converge to near-equal values where low occasionally matches high. The test has never been observed to fail, but it is timing-dependent.

---

## 6. Security

Nothing flagged. These are internal organism systems with no external input surfaces. GovernanceLogger appends to a local file; the log path is set at construction time, not from external input.

---

## 7. Test Coverage Gaps

### Missing: `InhabitantScheduler` — register after start

The build brief asked: "What happens if `register()` is called after `start()`?" The tests cover register-before-start only. No test calls `register()` after `start()` is already running and verifies the newly registered system fires.

**Impact:** This path is code-complete (the lock is always held during register), but it is untested.

### Missing: `InhabitantScheduler` — heap grows unboundedly with reschedule

Every call to `reschedule()` pushes a new entry onto the heap (line 157) without removing the old entry (lazy deletion handles it). If `reschedule()` is called hundreds of times on a live system, the heap accumulates stale entries indefinitely. No test probes heap size after repeated rescheduling. No cap or cleanup exists.

### Missing: `GovernanceLogger` — concurrent write test

The race on `_event_count` described in Section 2 is untested. No test publishes events from multiple threads simultaneously.

### Missing: `ResourceGovernor` — `tighten_budgets` does not restore on `relax`

No test verifies that repeated tighten/relax cycles preserve the budget near its original value.

### Missing: `HomeostasisRegulator` — empty setpoints

`compute_drive_urgency()` with `self._setpoints = {}` returns `{}` cleanly. `get_deviation_score()` returns `0.0`. `is_stable()` returns `True` (vacuously). These are correct but untested. The scenario occurs if `setpoint_configs=[]` is passed to the constructor.

---

## 8. What Works

All 47 dedicated tests (24 for InhabitantScheduler, 23 for GovernanceLogger) pass cleanly. The tests exercise the designed-for paths thoroughly and use real threading (not mocked), which gives meaningful coverage of the actual daemon behavior.

**InhabitantScheduler:** Thread safety design is sound. The lock protects all heap mutations. Callback exceptions are isolated in `_run_callback` and never propagate to the dispatch loop. Lazy deletion is correctly implemented. `stop()` signals the event and joins the thread. The only structural defect is the empty-heap spin.

**GovernanceLogger:** The passive observer pattern is correctly implemented. File write failures are isolated and never propagate. JSON serialization failures are caught. The JSONL format is valid. All five governance channels are subscribed. Directory creation at construction time is correct.

**OctopusColony gradient routing:** The fallback to workload routing when stigmergy is unavailable or returns None is correctly implemented (line 888 guard + line 320 try/except). The distance penalty math is correct — Euclidean distance with linear normalization. The `min(1.0, confidence * multiplier)` cap in `_apply_quorum_boost` correctly prevents confidence from exceeding 1.0.

**HomeostasisRegulator:** Setpoint initialization, error computation, correction clamping, and urgency calculation are all correct. The `compute_drive_urgency` public method correctly filters to out-of-range parameters only and applies `min(1.0, urgency)`. Direct endocrine polling in `step()` is cleaner than pure event-drive and avoids missed updates.

**ResourceGovernor priority tiers:** MAINTENANCE sources correctly bypass all budget checks (line 149 returns True unconditionally). The ACTIVE tier multiplier (1.5x, line 154) is applied before the limit check and cannot be affected by `tighten_budgets` or `relax_budgets`, which correctly only target EXPLORE sources.

**OrganBuilder senescence handler:** JSON parsing handles both str and dict message types. An unknown type logs a warning and returns cleanly (lines 525–526). `prune_organs()` failure would propagate as an unhandled exception inside `_on_system_senescent`, but `prune_organs` itself has no known failure modes.

---

## Summary: Issues by Priority

| # | Severity | Location | Description |
|---|----------|----------|-------------|
| 1 | HIGH | `inhabitant_scheduler.py:249-252` | Empty heap spins at 100% CPU — no sleep |
| 2 | MEDIUM (latent) | `endocrine_system.py:184-246` | Lock held across EventBus publish — reentrant caller deadlocks |
| 3 | MEDIUM | `endocrine_system.py:520` + `resource_governor.py` | Compound budget reduction — no recovery to baseline |
| 4 | MEDIUM | `homeostasis.py:258` | Uncapped urgency > 1.0 published on EventBus |
| 5 | MEDIUM | `governance_logger.py:120-121` | `_event_count` / `_last_event_time` unprotected from concurrent writes |
| 6 | LOW | `resource_governor.py:228` | `relax_budgets(factor < 1.0)` warns but still silently reduces budget |
| 7 | LOW | `octopus_colony.py:455-474` | `self.octopuses` modified without lock during monitoring loop |
| 8 | LOW | `organ_builder.py:531` | Senescence prune called without model/substrate — zombie agents in live runs |
| 9 | LOW | `octopus_colony.py:302-307` | `marker.position` type assumption may raise TypeError silently |
| 10 | EDGE | `inhabitant_scheduler.py:191` | `shutdown(wait=False)` — in-flight callbacks outlive `stop()` |
| 11 | EDGE | `inhabitant_scheduler.py:283` | `name`/`cb` implicitly depend on prior assignment path |
| 12 | TEST GAP | `test_inhabitant_scheduler.py` | register-after-start not tested |
| 13 | TEST GAP | `test_inhabitant_scheduler.py` | heap growth from repeated reschedule() not tested |
| 14 | TEST GAP | `test_drive_coupling.py` | tighten/relax compound erosion not tested |

**Mandatory fixes before production:** Issues 1, 3, and 4. Issue 2 is a latent risk that should be addressed before any new EventBus subscriber is wired to the endocrine channel.
