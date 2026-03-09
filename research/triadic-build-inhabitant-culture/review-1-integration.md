# Independent Integration Review — Inhabitant Culture Activation
**Reviewer:** Independent (did not build this code)
**Date:** 2026-03-08
**Scope:** Round 1 (B1-B6) + Round 2 (B1-B2) integration and constraint review

---

## 1. Integration Errors

### CRITICAL: GovernanceLogger crashes when `event_bus=None`

**File:** `mae_core/bootstrap/market_systems.py:490`
**File:** `mae_core/governance/governance_logger.py:77`

The bootstrap constructs GovernanceLogger with `event_bus=getattr(ctx, "bus", None)`. GovernanceLogger's constructor signature marks `event_bus` as a required argument (no default), and at line 77 it unconditionally calls `self._bus.register_callback(...)` for each of the 5 governance channels. If `ctx.bus` is `None`, this raises `AttributeError: 'NoneType' object has no attribute 'register_callback'`.

In production, `ctx.bus` is always set by Layer 1. But in isolated unit tests that construct a bare ctx without a real bus, or if Layer 1 bootstrap fails, `ctx.governance_logger` becomes `None` silently via the `try/except`. This means GovernanceLogger is not observable/testable without a live bus — the test suite's test_integration tests would only confirm the key exists (not that the system is `None`).

**Verified:** `python -c "from mae_core.governance.governance_logger import GovernanceLogger; GovernanceLogger(event_bus=None)"` raises `AttributeError`.

**Fix:** Guard `register_callback` calls: `if self._bus is not None: self._bus.register_callback(...)`. This matches how ResourceGovernor, EventBus publishers, and every other system in this codebase handle an optional bus.

---

### IMPORTANT: `InhabitantScheduler.start()` is never called

**File:** `mae_core/bootstrap/market_systems.py:482-485`

The scheduler is constructed but `.start()` is never called anywhere in the bootstrap. No `start()` call exists in `market_systems.py`, `market_hooks.py`, `market.py`, or anywhere else in `mae_core/bootstrap/`. The daemon thread and thread pool are never launched.

The scheduler is wired into holons, somatic map, and connections — all of which imply it is active. But it does nothing at runtime. Any system registered for dispatch will never fire.

B1's Round 2 report explicitly notes "InhabitantScheduler is constructed but no bio-systems are registered for dispatch." That is the secondary issue. The primary issue is the scheduler isn't even running.

**Fix:** Add `if ctx.inhabitant_scheduler is not None: ctx.inhabitant_scheduler.start()` in the bootstrap, after construction. The companion `.stop()` should be called in the shutdown path.

---

### IMPORTANT: `market.py` module docstring carries stale counts from before this build

**File:** `mae_core/bootstrap/market.py:2-4`

The module docstring reads:
```
Creates 51 market systems, registers holons, wires fractal hierarchy,
registers triadic connections (Group 14-32),
```

After this build the values are 59 systems, Groups 14-37. Line 51 of the same file (the `bootstrap_market` docstring) correctly says "59 systems, 115 connections". Lines 2-4 were not updated.

**Impact:** Future builders using the top-of-file summary as a reference will get wrong counts.

---

### IMPORTANT: B3 stigmergy evaporation uses 2D position; B3 gradient routing also uses 2D

**File:** `mae_core/bootstrap/market_hooks.py` (evaporation hook)
**File:** `mae_core/network/octopus_colony.py:submit_task()` (gradient routing)

B3's evaporation hook calls `sense_markers(position=(0.0, 0.0, 0.0), ...)` (3-tuple). The gradient routing in `submit_task()` uses `get_strongest_marker(position=(0.0, 0.0), ...)` (2-tuple). These two calls differ on the dimension of the position passed to the same `StigmergicEnvironment`.

B3's build report acknowledges this was intentional: `_distance()` uses `min(len(a), len(b))` for dimension-matching. However, if pheromone markers were deposited using a 2D position and the evaporation hook calls with 3D, the distance calculation uses `min(2,3)=2` dimensions — this is safe but the inconsistency is an unresolved design gap. More critically, the evaporation hook was intended to trigger `_apply_decay()` on all markers globally. Passing a non-zero radius (`float("inf")`) + position `(0,0,0)` is correct for that intent, but the dimension mismatch is a latent fragility.

---

### Advisory: `pheromone key format not deposited anywhere`

**File:** `mae_core/network/octopus_colony.py` (consumer)

B3's gradient routing reads `convergence:{ticker}` markers. B3's report explicitly says it does NOT deposit markers — that is "a separate concern." No other builder deposited them either. The gradient routing feature is wired but never triggered in practice because the pheromone trail it depends on is never populated.

This is documented intent ("future work"), not a bug. But it means the stigmergy gradient routing is dead code at runtime.

---

## 2. Constraint Violations

### Law 1 (No Bare Dyads) — PASS

Groups 35, 36, and 37 all have three connections each with valid witnesses. Verified in `market_connections.py:458-497`. The cross-reference between InhabitantScheduler and GovernanceLogger as mutual witnesses is a sound triadic design.

### Law 3 (Holon Protocol) — PARTIAL

ResourceGovernor, InhabitantScheduler, and GovernanceLogger all implement `get_statistics()`, satisfying `know_self`. They are registered with HolonProxy in `market_registration.py`. However:

- **InhabitantScheduler** `get_statistics()` returns `"running": False` if `start()` is never called. Every HolonProxy query for this system will report it as not running — even in production. This compounds the "start() never called" issue above.

### Law 6 (Autopoietic Closure) — PASS

InhabitantScheduler uses only Python stdlib. GovernanceLogger writes to a local file. ResourceGovernor uses only threading and deque. No external dependencies introduced.

### Law 7 (Rule of 3/5) — Not applicable to these specific changes. No TriadEnforcer-governed voting added.

---

## 3. Bugs and Logic Errors

### IMPORTANT: `relax_budgets` warns but does not reject `factor < 1.0`

**File:** `mae_core/market/resource_governor.py:227-232`

`relax_budgets(factor)` warns when `factor < 1.0` but still applies the multiplication. The docstring says this is for "meaningful relaxation" and factors must be `>= 1.0`. If the endocrine closure in `endocrine_system.py` ever produces a `relax_budgets` call with factor < 1.0 (which the current formula `1.0 + (0.3 - level)` cannot — minimum is `1.0` when `level=0.3`), the budget would shrink under a "relax" call.

The endocrine formula is safe (`level < 0.3` guarantees factor > 1.0). But the lack of a hard guard in `relax_budgets` itself means a future caller could accidentally tighten budgets via the relax path. Should either raise or return early.

### Advisory: `_tick()` lazy-delete path does not sleep before returning

**File:** `mae_core/scheduling/inhabitant_scheduler.py:260-262`

When a lazy-deleted entry is popped from the heap, `_tick()` returns immediately without calling `self._stop_event.wait()`. The `_dispatch_loop` has no sleep of its own. If a large number of stale entries pile up (e.g., systems registered and unregistered rapidly), the scheduler thread burns CPU clearing the heap in a tight loop. Each iteration is O(log N) so this is not catastrophic, but it degrades responsiveness of the thread pool.

In the current codebase this is a non-issue because no systems are registered at all (start() never called), but it is a latent defect for when the scheduler is actually used.

### Advisory: `tighten_budgets` is permanently destructive

**File:** `mae_core/market/resource_governor.py:193-214`

`tighten_budgets` permanently modifies `hourly_limit`. Under sustained high cortisol (multiple `tighten_budgets` calls before `relax_budgets`), EXPLORE source limits compound-shrink: 1000 → 800 → 640 → 512 → ... There is no floor (other than `max(1, ...)` which prevents zero). If the organism experiences a cortisol spike lasting many steps, EXPLORE sources could be throttled to near-zero and never recover without an equal number of relax calls.

The tests confirm this behavior is working as designed, but the lack of a nominal floor (e.g., "never below 10% of original") could cause EXPLORE sources to become permanently unusable if cortisol is chronically elevated.

---

## 4. Edge Cases

### GovernanceLogger write path: `event_data` may be a JSON string (already serialized)

**File:** `mae_core/governance/governance_logger.py:88-108`

EventBus delivers callbacks with the already-serialized JSON string (per B5's design note and B6's handler). The `_on_event` method writes `{"event": event_data}` where `event_data` is a string like `'{"source": "sec_edgar", ...}'`. This double-encodes the event: the JSONL record becomes `{"timestamp": "...", "channel": "...", "event": "{\"source\": ...}"}` — a string nested inside JSON.

This is technically valid but makes the log file harder to read and parse. Future consumers of `governance_log.jsonl` that try to access `record["event"]["source"]` will get a `TypeError` (string not subscriptable).

**Verified by reading GovernanceLogger line 98:** `"event": event_data` — no pre-parse step.

---

## 5. Regression Risk

### Market.py `_wire_bio_systems` is called as step 11 in `bootstrap_market()`

**File:** `mae_core/bootstrap/market.py:75`

The bio wiring now includes the Endocrine → ResourceGovernor cortisol coupling block. This is a net-new subscription on `HormoneType.CORTISOL`. ResourceGovernor will now receive every cortisol event the endocrine system fires — including normal baseline fluctuations. If cortisol oscillates between 0.3 and 0.6 frequently (neutral zone), there is no effect. But if cortisol reaches > 0.6 (stress events), `tighten_budgets` fires, compounding the permanent-shrink issue noted above.

The existing endocrine tests do not test interactions with ResourceGovernor. The new tests in `test_drive_coupling.py` cover the method call but not the compound-shrink scenario across many steps.

### OrganBuilder now subscribes to `CH_SENESCENT` unconditionally at Layer 10

**File:** `mae_core/bootstrap/foundation.py:205`
**File:** `mae_core/morphogenesis/organ_builder.py:265-268`

B6 added `event_bus=ctx.bus` to the OrganBuilder constructor in foundation.py. This means OrganBuilder now subscribes to `"emergent.system_senescent"` at bootstrap Layer 10. When a senescent event fires, `_on_system_senescent` calls `self.prune_organs()` which marks organs as DISSOLVED and removes them from `_active_organs`.

If any system fires a senescent event during normal operation (which Senescence.py does via the existing `CH_SENESCENT` channel), all active organs are pruned. This is the designed behavior, but it is a behavior change: previously OrganBuilder had no response to senescent events. Existing tests for OrganBuilder do not exercise the senescence path — only the new `test_senescence_lifecycle.py` does.

---

## 6. Security

No new external-facing APIs, deserialization of untrusted input, or credential handling introduced. All new paths (GovernanceLogger, InhabitantScheduler, ResourceGovernor tiers) are organism-internal. No security concerns raised.

---

## 7. Test Coverage

### Missing: GovernanceLogger with `event_bus=None`

`tests/test_governance_logger.py` does not test the `event_bus=None` constructor path, because B5 correctly declared `event_bus` as required. But the bootstrap passes `None` when the bus is absent. No test covers the `AttributeError` crash scenario or verifies the graceful degradation fallback.

### Missing: InhabitantScheduler `.start()` integration

No test verifies that the scheduler is started during bootstrap (because it isn't). The unit tests in `test_inhabitant_scheduler.py` manually call `sched.start()` and `sched.stop()`. The integration test only checks that `inhabitant_scheduler` is a key in the systems dict — it does not assert `sched._thread is not None` (running).

### Missing: Compound tighten/relax cycle test

`test_resource_governor.py` tests single calls to `tighten_budgets` and `relax_budgets`. No test verifies behavior across multiple cortisol cycles — specifically, that a long cortisol spike followed by recovery restores budgets to near their original values.

### Missing: GovernanceLogger double-encoding test

No test verifies the schema of actual lines written to the JSONL file. Tests mock `open()` rather than checking the written content's `event` field type.

### Adequate: B4 quorum boost coverage

`test_quorum_confidence.py` (11 tests) covers all branches of `_apply_quorum_boost` including None quorum_space, below-threshold, cap, and exception swallowing. Adequate.

### Adequate: B2 homeostasis urgency + reflex priority chain

`test_drive_coupling.py` correctly tests the priority interaction (`test_reflex_override_pain_still_higher_priority`, `test_reflex_override_energy_critical_overrides_homeostasis`). The boundary condition at exactly 0.7 is tested. Adequate.

### Adequate: B6 OrganBuilder senescence subscription

`test_senescence_lifecycle.py` (18 tests) covers construction with/without bus, payload parsing, prune call, and rebuild publication. Adequate.

---

## 8. What Works

The cross-builder integration is largely correct. The main wiring points all match:

- **B1→B3**: `stigmergy=getattr(ctx, "stigmergy", None)` passed to OctopusColony. `OctopusColony.__init__` accepts `stigmergy` as the last optional parameter (`octopus_colony.py:115`). Parameter name matches. **PASS.**

- **B1→B4**: `quorum_space=getattr(ctx, "quorum_space", None)` passed to ConvergenceAlerter. `ConvergenceAlerter.__init__` accepts `quorum_space=None` as last optional parameter (`convergence_alerter.py:234`). Parameter name matches. **PASS.**

- **B1→B6**: `event_bus=ctx.bus` passed to OrganBuilder in `foundation.py:205`. `OrganBuilder.__init__` accepts `event_bus: Optional[Any] = None` (`organ_builder.py:245`). Parameter name matches. **PASS.**

- **B1→B2 (endocrine coupling)**: `endocrine.register_resource_governor(ctx.resource_governor)` called in `bio_market_wiring.py:85`. `EndocrineSystem.register_resource_governor(rg)` exists at `endocrine_system.py:503`. Signature matches. **PASS.**

- **B1→B5 (InhabitantScheduler)**: Constructed as `InhabitantScheduler(event_bus=getattr(ctx, "bus", None))`. B5's constructor signature is `InhabitantScheduler(event_bus=None, max_workers=4)`. Keyword arg matches. **PASS.**

- **B1→B5 (GovernanceLogger)**: Constructed as `GovernanceLogger(event_bus=getattr(ctx, "bus", None))`. Keyword arg matches constructor. **PASS for signature match; FAILS at runtime when bus is None** (see Critical finding above).

- **SourceTier priority logic**: MAINTENANCE always returns True, ACTIVE uses 1.5x effective_limit, EXPLORE uses standard limit, `tighten/relax_budgets` only affects EXPLORE. Verified in `resource_governor.py:148-165`. **PASS.**

- **Endocrine cortisol formula**: `tighten_budgets(level)` when `level > 0.6`, `relax_budgets(1.0 + (0.3 - level))` when `level < 0.3`, no-op in neutral zone. Formula produces valid inputs (relax factor always > 1.0, tighten factor always < 1.0). **PASS.**

- **`compute_drive_urgency()` clamp**: Values correctly clamped to `min(1.0, urgency)` at `homeostasis.py:302`. Delegates to existing `_compute_urgency()`. No duplication. **PASS.**

- **Reflex Priority 6**: Added after Priority 5, before `return None`, at `organism_state.py:569`. Uses `>=` on `_HOMEOSTASIS_URGENCY_THRESHOLD`. Does not break existing priority ordering. **PASS.**

- **Law 1 connections**: Groups 35, 36, 37 each have exactly 3 connections with two witnesses each. No bare dyads. **PASS.**

---

## Summary Table

| Finding | Severity | File | Description |
|---------|----------|------|-------------|
| GovernanceLogger crash on `event_bus=None` | Critical | `governance_logger.py:77` | `register_callback` called on None bus |
| InhabitantScheduler never started | Important | `market_systems.py` (missing) | `.start()` never called in bootstrap |
| `market.py` docstring stale counts | Important | `market.py:3-4` | Says "51 systems, Groups 14-32" — both wrong |
| GovernanceLogger double-encodes event_data | Important | `governance_logger.py:98` | JSONL `event` field is a string, not a dict |
| `relax_budgets` applies factor < 1.0 without guard | Important | `resource_governor.py:227` | Warns but does not reject destructive call |
| Pheromone markers never deposited | Advisory | `octopus_colony.py` | Gradient routing is dead code at runtime |
| Lazy-delete busy-loop in `_tick()` | Advisory | `inhabitant_scheduler.py:260-262` | No sleep on lazy-delete return path |
| Compound tighten budget shrink | Advisory | `resource_governor.py:210` | No floor — chronic cortisol starves EXPLORE sources |
| Stigmergy 3D vs 2D position inconsistency | Advisory | `market_hooks.py`, `octopus_colony.py` | Evaporation uses 3-tuple, routing uses 2-tuple |
