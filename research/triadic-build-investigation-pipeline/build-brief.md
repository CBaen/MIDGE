# Build Brief: Investigation Pipeline — Wire the Octopus + Fix Convergence Timing

## Date: 2026-03-09
## Project: MIDGE

### Goal

The Octopus investigation system is fully built but completely unwired — developing situations pile up with no consumer. Signal convergence checking has a timing gap (new signals don't get checked until the next step). These two fixes close the biggest "dead wire" in MIDGE.

### Build Tasks

**Round 1 (parallel — no file overlap):**

1. **Investigation Dispatcher + Situation Persistence** — Wire the Octopus to actually investigate developing situations. Add step-cadence dispatcher, enhance investigation handlers with PatternLibrary + WorldModel lookups, wire CH_OCTOPUS_INVESTIGATION subscriber, persist developing situations across restarts.

2. **Signal-Triggered Convergence** — After signals are ingested in `_collect_one()`, immediately call `check_convergence()` so new signals are evaluated instantly instead of waiting for the next step tick. Reactive, not polling.

### Team Size: 2 builders + 2 reviewers

### Builder Assignments

| Builder | Domain | Files Owned |
|---------|--------|-------------|
| Investigation Builder | Octopus + handlers | `mae_core/network/market_task_handlers.py`, `mae_core/bootstrap/market_hooks.py` (investigation sections only: lines 390-425, 950-960, new dispatcher), `tests/test_investigation_pipeline.py` (new) |
| Convergence Builder | Sensing + convergence | `mae_core/market/sensing_hook.py`, `tests/test_signal_triggered.py` (new) |

### Project Constraints
1. Never block the step loop — try/except around all handlers
2. No unbounded growth — cap situations, evict stale entries
3. Zero regressions — `python -m pytest tests/ -v` must pass
4. Follow established patterns
5. Tests required for both features

### Verification Plan
1. `python -m pytest tests/test_investigation_pipeline.py tests/test_signal_triggered.py -v` — new tests pass
2. `python -m pytest tests/ -q --tb=line` — zero regressions
3. `python main.py --agents 3 --steps 30` — smoke test

---

## Technical Context for Builders

### Feature 1 — Investigation Dispatcher + Enhancement (Investigation Builder)

**What exists (confirmed by research):**
- `_developing_situations` dict on colony — fills up from `_on_partial_convergence` but NOBODY reads it
- `investigate_partial` handler (market_task_handlers.py:185-241) — calls `check_ticker_convergence_for(ticker)` but this only fires if convergence threshold already crossed (largely redundant)
- `archaeology_lookup` handler (market_task_handlers.py:244-304) — checks live stacks but NOT historical templates
- `situation_check` handler (market_task_handlers.py:307-341) — evicts old entries but never called
- `CH_OCTOPUS_INVESTIGATION` published by handlers but has ZERO subscribers
- Step-20 coordination (market_hooks.py:951-958) runs arm balancing but NOT investigation dispatch
- Max 200 developing situations, never evicted (eviction only in situation_check which is never called)

**What to build:**

A. **Step-cadence dispatcher** (in market_hooks.py, new section after line 958):
   - Every 20 steps (alongside existing coordination cycle), iterate `colony._developing_situations`
   - For each situation with `check_count < 20` and `age < MAX_SITUATION_AGE_STEPS`:
     - Submit `investigate_partial` task to colony: `colony.submit_task({"ticker": sit["ticker"], "direction": sit["direction"], "domains_seen": sit["domains_seen"], "missing_domains": sit["missing_domains"]}, "investigate_partial")`
     - Submit `situation_check` task every 5th check: `colony.submit_task({...}, "situation_check")`
   - Cap: max 5 task submissions per step (avoid flooding)
   - Increment `check_count` on dispatch

B. **Enhance `investigate_partial`** (market_task_handlers.py):
   - KEEP existing `check_ticker_convergence_for()` call (catches cases where convergence completed between checks)
   - ADD: Query `PatternLibrary.query_similar(domains_seen)` to get historical templates matching the developing domain combo. Include win rate, instance count, cross-validation status in the investigation result.
   - ADD: Check `world_model.find_root_causes(ticker)` and `world_model.find_ripple_effects(trigger)` if causal_predictions are available
   - ADD: If historical template win rate > 60% AND 3+ cross-validated symbols, populate `ctx._priority_requests` to boost missing domain polling (connect to Focused Attention)
   - Publish enriched result to `CH_OCTOPUS_INVESTIGATION`

C. **Wire `CH_OCTOPUS_INVESTIGATION` subscriber** (market_hooks.py):
   - When investigation results arrive with high-confidence historical matches:
     - Log at INFO level
     - If strong historical match (win_rate > 0.6, instances >= 5), create a priority request for missing domains
   - This closes the loop: partial convergence → Octopus investigation → historical lookup → focused attention → signal acquisition

D. **Persist `_developing_situations`** to `data/market/developing_situations.json`:
   - Save on every change (atomic write like active_tracker)
   - Load on bootstrap (in market_systems.py after colony creation)
   - Filter out expired entries on load

**Tests (test_investigation_pipeline.py):**
- Dispatcher submits tasks for developing situations
- Max 5 tasks per step cap respected
- Eviction fires via situation_check
- investigate_partial queries PatternLibrary
- investigate_partial queries WorldModel
- CH_OCTOPUS_INVESTIGATION subscriber receives results
- High win-rate match creates priority request
- Persistence round-trip (save/load developing_situations.json)
- Expired entries filtered on load

### Feature 2 — Signal-Triggered Convergence (Convergence Builder)

**What exists (confirmed by research):**
- Hook execution order: Hook 1 (`_market_sense_hook`) calls `check_convergence()` → Hook 2 (`_sensing_step_with_advisory`) calls `SensingHook.step()` which ingests new signals. New signals don't get checked until NEXT step.
- `check_convergence()` is cheap (pure in-memory, called every step already)
- `_collect_one()` at sensing_hook.py:788-944 feeds signals into alerter via `record_signal()` but never triggers convergence check

**What to build:**
- At the end of `_collect_one()` (after all signals from a source are fed), call `self._convergence_alerter.check_convergence()` inline
- Only check if new signals were actually added (not on empty fetch results)
- Emit alerts through the same path as the step hook: publish to EventBus, cache on ctx
- This means convergence is checked BOTH on signal arrival (reactive) AND on step tick (regular sweep). The step-tick check catches any signals from WebSocket or other non-rotation paths.
- The convergence alerter already deduplicates alerts (same direction+domains in cooldown window), so double-checking is safe.

**Important: The sensing hook holds a reference to `self._convergence_alerter` (line 357). It also has `self._bus` (line 354, injected by bootstrap). The hook can call `check_convergence()` and publish results directly.**

**What NOT to change:**
- Do NOT remove the step-tick convergence check in `_market_sense_hook`. It serves as a safety net.
- Do NOT change the hook execution order.

**Tests (test_signal_triggered.py):**
- Signal ingestion triggers convergence check
- Empty fetch result does NOT trigger convergence check
- Alert deduplication still works (no double-alerts from step + signal trigger)
- Convergence fires immediately after signal that completes domain threshold
- WebSocket signals still get checked via step-tick path
