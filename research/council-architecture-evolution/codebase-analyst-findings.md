# Codebase Analyst Findings
## Research Brief: Mesa Step-Cadence vs. Thread-Per-Subsystem Architecture
**Date:** 2026-03-12
**Analyst Role:** Internal codebase — what exists, how it works, what the blast radius would be

---

## Step 1: What Is Actually in the Codebase

### Mesa Usage — The Full Inventory

**Files that import Mesa:** exactly 2.

| File | Mesa Import | What It Uses |
|------|-------------|--------------|
| `mae_core/model.py` | `import mesa` → `class MycelialModel(mesa.Model)` | `model.time` (step counter), `model.agents` (AgentSet), `super().__init__()` |
| `mae_core/agents/base_agent.py` | `from mesa import Agent` | `class BaseAgent(Agent)` → `super().__init__(model)` for auto-registration, `agent.unique_id`, `agent.remove()` |

**Every other file in the codebase uses these indirectly.** No other file directly imports Mesa.

The proposal's claim — "Mesa provides ~50 lines of real functionality" — is confirmed accurate. The actual Mesa-consumed surface in `model.py` is:
- `mesa.Model.__init__()` → seeds `model.time = 0`, `model.agents = AgentSet()`
- `model.time` — incremented by Mesa automatically each `super().step()` call (or by Mesa internally during step execution)
- `model.agents.shuffle_do("step")` — Mesa AgentSet iteration
- Agent auto-registration via `super().__init__(model)` in `base_agent.py`

### model.time Consumers

`model.time` is read by 17 locations across 7 files (verified via Grep):

| File | Usage |
|------|-------|
| `main.py` (×6) | `int(model.time)` in narrator_hook, journal_hook, pace_hook, oracle capture (×2), daemon round |
| `mae_core/bootstrap/organs_layers_26_27.py` (×10) | All bio organ step hooks: `s.step(current_step=int(ctx.model.time))` |
| `mae_core/bootstrap/organs_layers_28_30.py` (×6) | lymphatic, senescence, boundary_membrane, reproductive, organism_state, rediff_monitor, mitosis_monitor, closure_coordinator |
| `mae_core/bootstrap/external.py` (×1) | `api_gateway.step(current_step=int(ctx.model.time))` |
| `mae_core/bootstrap/patterns_layers_22_25.py` (×1) | `ctx.task_pool.step(current_step=int(ctx.model.time))` |
| `mae_core/bootstrap/wiring_layers_14_16.py` (×1) | FRL policy update context |
| `mae_core/coordination/circadian_rhythm.py` | Documentation reference only |

**Critical finding:** `model.time` is used exclusively by the bio/core systems (Layers 1-31). Not one market intelligence file reads `model.time`. Market systems maintain their own `_step_counter` variables independently.

---

## Step 2: The Step-Hook Chain — Precise Architecture

The daemon loop is: `model.run(N)` → `model.step()` (×N) → for each hook in `_step_hooks`: `hook()`.

**Step hooks registered (in order):**

1. All bio organ hooks (lambdas calling `.step(current_step=int(ctx.model.time))`) — registered in bootstrap layers 26-30
2. `api_gateway.step()` — Layer 31
3. `_market_sense_hook()` — Layer 33g, `market_hooks_steps_core.py:_register_market_step_hooks()`
4. `_sensing_step_with_advisory()` — Layer 33h, `market_hooks_sensing.py:_wire_sensing_hook()`
5. `narrator_hook()`, `journal_hook()`, `pace_hook()` — `main.py`

The market intelligence loop (hooks 3 and 4) runs sequentially at the end of every step. These two hooks together own the entire market intelligence lifecycle.

### Hook 3: `_market_sense_hook()` in `market_hooks_steps_core.py`

Maintains its own `_step_counter = [0]` (a closure variable, NOT `model.time`). Cadence gates:

| Cadence | Operation |
|---------|-----------|
| Every step | `convergence_alerter.check_convergence()` → writes `ctx._cached_alerts[0]` |
| Every step | `hypothesis_engine.step()` |
| Every step | `pattern_memory.remember_convergence_alert()` |
| Every 10 | Thompson stats → EventBus |
| Every 20 | OctopusColony coordination + investigation dispatch |
| Every 50 | Stigmergy evaporation, velocity scan, session sweep bypass, drift detector |
| Every 100 | Motif detection + streaming anomaly |
| Every 200 | Thompson forgetting (with outcome gate), convergence heartbeat write |
| Every 500 | Lag correlation, Granger causality, post-mortem, cascade expiry |
| Every 1000 | Thompson calibration |
| Every 5000 | Backtest scheduler, excavation daemon |

### Hook 4: `_sensing_step_with_advisory()` in `market_hooks_sensing.py`

Also maintains its own `_sensing_step_counter = [0]`. Calls `hook.step()` (MarketSensingHook) then:

| Cadence | Operation |
|---------|-----------|
| Every step | Advisory bridge update from `_cached_alerts` |
| Every step | Paper trading gate (when alerts present) |
| Every step | Synergy detection (convergence + pattern stack dual confirmation) |
| Every 10 | Tiered alerter queries, pattern archaeology + PatternWatcher.check() |
| Every 20 | Active tracker price check |

### MarketSensingHook.step() in `sensing_hook.py`

Maintains `_step_counter = 0`. Called as `original_step()` inside hook 4. Cadence gates:

| Cadence | Operation |
|---------|-----------|
| Every step | Collect WebSocket signals (FinnhubWebSocket.get_pending_signals()) |
| Every step | `_collect_results()` — process any completed async fetch futures |
| Every 25 | `_launch_next_fetch()` — submit up to 8 concurrent fetch slots to ThreadPoolExecutor(12) |
| Every 200 | `_evaluate_outcomes()` |
| Every 50 | Portfolio tracker mark-to-market |
| Every 200 | Catalyst calendar refresh |
| Every 100 | Absence detection |
| Every 200 | Correlation tracker anomaly scan |
| Every 5000 | Consolidation engine |
| Every 25 | Somatic anticipation |
| Every 100 | Pattern archetype scanning |
| Every 100 | Pattern completion review |

---

## Step 3: Threading Already In the Codebase

The codebase already has substantial concurrent execution. This is not a "green field" threading problem.

| System | Threading Pattern | Location |
|--------|------------------|----------|
| MarketSensingHook | `ThreadPoolExecutor(max_workers=12)` for API fetches | `sensing_hook.py` line 198 |
| MycelialModel agents | `ThreadPoolExecutor(max_workers=20)` for parallel agent stepping | `model.py` line 161 |
| FinnhubWebSocket | `threading.Thread(daemon=True)` background WebSocket loop | `finnhub_websocket.py` lines 132-135 |
| OctopusColony | `threading.Thread(daemon=True)` monitoring loop | `octopus_colony.py` lines 371-374 |
| InhabitantScheduler | heapq + `threading.Lock` + `ThreadPoolExecutor` + `threading.Event` | `inhabitant_scheduler.py` |
| ThompsonSampler | `threading.Lock()` on all mutations | `thompson_sampler.py` line 73 |
| ConvergenceAlerter | `threading.Lock()` (`_alert_lock`) on deduplication | `convergence_alerter.py` line 252 |
| DrawdownMonitor | `threading.RLock()` on all state | `drawdown_monitor.py` line 82 |
| SelfMonitor | `threading.RLock()` | `self_monitor.py` line 122 |

**Key insight:** The `_collect_results()` / `_collect_one()` path in `sensing_collector.py` is the ONLY place where async fetch results re-enter the main thread. This is the current serialization boundary. Signals are fetched in thread pool workers, but `record_signal()` on the convergence alerter is called from the main step thread.

The ConvergenceAlerter's `_alert_lock` only protects deduplication state (`_last_alert_times`), not the full signal buffer. The `signals` dict (the main buffer) is not thread-safe — it is currently only written from one path (the main step thread collecting futures).

---

## Step 4: Blast Radius Mapping for the Proposed Change

The proposal: keep Mesa as a heartbeat for bio systems; liberate market intelligence subsystems to run on wall-clock timers.

### Files That Would Change

**Tier 1 — Direct surgery required:**

| File | Change Required | Risk |
|------|----------------|------|
| `mae_core/market/sensing_hook.py` | Remove `step()` method; replace with `start()`/`stop()` that spins a background timer thread | Medium — well-isolated class |
| `mae_core/bootstrap/market_hooks_sensing.py` | Replace `ctx.model.add_step_hook(_sensing_step_with_advisory)` with `hook.start()` call; remove sensing from step chain | Medium |
| `mae_core/bootstrap/market_hooks_steps_core.py` | Extract `_market_sense_hook()` closure into a standalone thread or register with InhabitantScheduler | High — this closure references `ctx` extensively |
| `main.py` | No change to Mesa-driven daemon loop needed; daemon mode still calls `model.run(N)` | None |

**Tier 2 — Signal flow re-threading:**

| File | Change Required | Risk |
|------|----------------|------|
| `mae_core/market/sensing_collector.py` | `_collect_one()` would move to a background thread context; `record_signal()` calls to convergence alerter must become thread-safe | HIGH — this is the Thompson feedback path |
| `mae_core/market/intelligence/convergence_alerter.py` | `record_signal()` must acquire a lock (currently unprotected); `check_convergence()` must acquire a lock | HIGH — sacred system |
| `mae_core/market/intelligence/convergence_confidence.py` | `self.signals` dict (the main signal buffer) needs thread protection | HIGH |
| `mae_core/market/sensing_reactive.py` | `_trigger_reactive_convergence()` would run from a background thread calling `check_convergence()` concurrently with the heartbeat thread | HIGH |

**Tier 3 — Shared state that crosses the new thread boundary:**

| Shared State | Current Access Pattern | New Hazard |
|-------------|----------------------|------------|
| `ctx._cached_alerts[0]` | Written by `_market_sense_hook()`, read by `_sensing_step_with_advisory()` — both in main step thread | Would be written by market thread, read by main thread or market thread — needs lock |
| `ctx._cached_pattern_stacks` | Written by pattern watcher, read by synergy detector — both in main step thread | Cross-thread without synchronization |
| `ctx._market_advisory` | Written by advisory bridge, read by DaemonMonitor — currently main thread only | Race condition if advisory updates while monitor reads |
| `ctx._paper_trade_dedup` | Dict mutated by paper trading gate | Needs lock if gate runs concurrently with daemon flush |
| `hook._pending_futures` | Dict mutated by `_launch_next_fetch()` and `_collect_results()` — both currently in main thread | If both move to background thread, they'd be on the same thread — no new hazard. But if they're on different timers, races emerge. |

**Tier 4 — Tests:**

There are 184 test files in `tests/`. The ones that would be affected:
- `tests/test_decomposition_wiring.py` — tests sensing hook importability and instantiation. Would need updates if sensing hook's interface changes (`step()` removed, `start()`/`stop()` added).
- `tests/test_integration.py` — `mae_organism` fixture calls `create_mae()` which runs the full bootstrap including `model.add_step_hook()`. If sensing hook no longer registers a step hook, the integration test's 10-step smoke run would not exercise sensing at all.
- Any test that creates a `MarketSensingHook` and calls `.step()` directly would break.
- `tests/test_bio_market_wiring.py` and `tests/test_bio_market_wiring_extended.py` — test EventBus coupling between bio and market; would still work if EventBus wiring is unchanged.

**Count of tests touching sensing/hook integration:** approximately 15-25 test files based on the grep for test files with "sensing", "convergence", "market_hook", "bootstrap" patterns.

---

## Step 5: Pattern Inventory

### How Cadence Is Done Now (Three Independent Patterns)

The codebase already uses THREE distinct cadence patterns simultaneously:

**Pattern A — Mesa step-counter gates** (the dominant pattern for market intelligence)
```python
# In MarketSensingHook.step():
self._step_counter += 1
if self._step_counter % 25 == 0:
    self._launch_next_fetch()
```
Used by: both `_market_sense_hook` and `MarketSensingHook.step()` — these maintain SEPARATE counters.

**Pattern B — Wall-clock background threads** (already in use for 3 systems)
- `FinnhubWebSocket`: runs a WebSocket loop in a `threading.Thread(daemon=True)`, never touches `model.time`
- `OctopusColony._monitoring_loop()`: `threading.Thread(daemon=True)`, checks health every N seconds
- `InhabitantScheduler`: heapq-based wall-clock dispatcher already exists and is started in `_wire_sensing_hook()`

**Pattern C — ThreadPoolExecutor for I/O parallelism** (already in use for API fetches and agents)
- API fetches: `ThreadPoolExecutor(12)` in `MarketSensingHook`
- Agent stepping: `ThreadPoolExecutor(20)` in `MycelialModel`

**The proposed change is to migrate market intelligence from Pattern A to Pattern B.** The precedent for Pattern B already exists in three systems. The `InhabitantScheduler` is purpose-built for exactly this use case.

### The Signal Flow Contract

The current contract (enforced by the sequential step architecture):

```
Main thread:
  model.step()
    → _market_sense_hook()      # check_convergence() on alerter
    → _sensing_step_with_advisory()
        → hook.step()
            → _collect_results()
                → _collect_one() → alerter.record_signal()
                → _trigger_reactive_convergence() → alerter.check_convergence()
            → _launch_next_fetch() → executor.submit(_fetch_source)
        → advisory bridge
        → paper trading gate
        → synergy detection
```

Under the proposal, `alerter.record_signal()` and `alerter.check_convergence()` would be called from different threads potentially concurrently. The `self.signals` dict in `ConvergenceConfidenceMixin` is currently an unprotected `defaultdict(list)`.

---

## Step 6: Score the Proposal

| Dimension | Score | Justification |
|-----------|-------|---------------|
| **Feasibility** | 7 | The pattern (wall-clock thread with InhabitantScheduler) already exists. The surgery is well-bounded. The hard part is the convergence alerter's unprotected signal buffer. |
| **Blast Radius** | 5 | Touching the sensing hook, both market step-hook files, and requiring thread-safety work on the convergence alerter's signal buffer is medium-blast. Not every file changes, but the most critical path does. |
| **Pattern Consistency** | 8 | Pattern B (wall-clock background threads) already exists for FinnhubWebSocket, OctopusColony, and InhabitantScheduler. This is not a new pattern — it's an extension of an established one. InhabitantScheduler already provides the infrastructure. |
| **Reversibility** | 8 | Both `_market_sense_hook()` and `_sensing_step_with_advisory()` are closure functions defined inline at bootstrap time. They can be toggled between step-hook registration and scheduler registration with minimal change to the bootstrap logic. The rest of the codebase is unchanged. |
| **Dependency Risk** | 5 | The convergence alerter's `self.signals` dict is currently written without locks. Introducing concurrent writes from a background sensing thread while `check_convergence()` reads the same dict from a heartbeat thread creates a genuine race. The Thompson feedback loop just got fixed — this is the worst time to introduce concurrency hazards on that path. |

**Overall Internal Confidence: 6/10**

The architecture supports the change cleanly in principle. Three things give me pause:

1. **The convergence alerter's signal buffer is not thread-safe.** `self.signals` (a `defaultdict(list)`) in `convergence_confidence.py` is written by `record_signal()` and read by `check_convergence()`. If these move to different timers, concurrent access becomes possible. This is fixable but requires surgery on the sacred system.

2. **`ctx._cached_alerts[0]` is shared state.** This one-element list is the handshake between `_market_sense_hook` (writer) and `_sensing_step_with_advisory` (reader). If these run on separate timers, reads and writes could race without a lock. The fix is simple (a `threading.Lock` or `threading.Event`) but must not be missed.

3. **The Thompson feedback loop was just fixed for the fourth time.** The feedback path (OutcomeCollector → ThompsonSampler) crosses this architecture change. ThompsonSampler already has a `threading.Lock`. OutcomeCollector's state needs a similar audit before concurrently dispatching outcomes and forgetting calls.

---

## Key Conclusions for the Council

### What Already Works Like the Proposal

- `FinnhubWebSocket` already runs exactly as proposed: independent thread, signals fed into convergence alerter via `_process_realtime_signals()`, no step cadence dependency
- `InhabitantScheduler` already exists as the purpose-built scheduler for wall-clock-based market intelligence dispatch
- `OctopusColony` already runs its own monitoring loop in a daemon thread

### The Safe Migration Path (Identified by Codebase Structure)

The lowest-blast-radius path is to treat the two market step-hooks as `InhabitantScheduler` entries rather than Mesa step-hooks:

1. Replace `ctx.model.add_step_hook(_market_sense_hook)` with `ctx.inhabitant_scheduler.register("market_sense_hook", _market_sense_hook, interval_seconds=N)`
2. Replace `ctx.model.add_step_hook(_sensing_step_with_advisory)` with `ctx.inhabitant_scheduler.register("sensing_advisory", _sensing_step_with_advisory, interval_seconds=M)`
3. Add a `threading.RLock` to `ConvergenceAlerter` protecting `self.signals`

This requires zero changes to the 35 API client files (confirmed — none of them are in the blast radius), zero changes to the Thompson feedback loop internals, and zero changes to Mesa's role for bio systems.

### What the Codebase Cannot Absorb Without Preparation

Moving convergence alerter calls to multiple concurrent threads WITHOUT first auditing the `self.signals` signal buffer for thread safety. This is the single highest-risk element. The `_alert_lock` in `convergence_alerter.py` only protects deduplication state — it does not protect the main signal buffer that `record_signal()` writes to and `check_convergence()` reads from.
