# Ecosystem Activation — Wire the Octopus, Bridge the Pipelines

## Context

Guiding Light declared MIDGE should be "an entire functioning ecosystem — more planet than organism." The 10-team evolution blueprint research revealed two critical architectural gaps:

1. **Two disconnected intelligence pipelines**: Core organism attention (AttentionalGate → GlobalWorkspace → PatternCortex) processes ZERO market data. Market intelligence (SensingHook → ConvergenceAlerter → PatternWatcher) has NO connection to the attention system. These are two brains in one skull that never talk.

2. **Fully built octopus, never activated**: OctopusColony (5 files, ~1500 lines in `mae_core/network/`) is coded, tested, and ready — but never bootstrapped. It's designed as a distributed cognition system with semi-autonomous arms, auto-scaling, and multi-mode coordination. It should become the bridge between the two pipelines.

3. **Partial convergences discarded**: When 2 domains fire on the same ticker (below the min_domains=3 threshold), the ConvergenceAlerter silently returns None. These developing situations should trigger octopus investigations.

**Goal:** Bootstrap the OctopusColony, bridge market signals into the core attention pipeline, and emit partial convergences as investigation triggers. Zero changes to existing signal flow. The convergence alerter still fires exactly as before — partial convergence is an ADDITIONAL emission.

**Build method:** Triadic-construction skill (4 builders, 2 reviewers, 2 rounds). This is a multi-domain, multi-file change touching bootstrap, intelligence, patterns, and network — exactly what triadic-construction is designed for. Evaluation: triadic-construction is the right skill. Its file-ownership model, independent review, and round-based dependency handling fit this task perfectly.

---

## Changes

### Round 1 (no dependencies — all builders work in parallel)

#### 1. New channel constants — `mae_core/market/channels.py`

Add after line 55 (after CH_PATTERN_COMPLETED):

```python
# Ecosystem Activation channels
CH_PARTIAL_CONVERGENCE = "market.intel.partial_convergence"
CH_OCTOPUS_INVESTIGATION = "market.intel.octopus_investigation"
```

#### 2. New file: `mae_core/market/translators/__init__.py` (empty package marker)

#### 3. New file: `mae_core/market/translators/market_signal_translator.py` (~120 lines)

Two translator classes implementing `PatternTranslator` Protocol (from `mae_core/patterns/translators/base.py`):

**`MarketConvergenceTranslator`** — subscribes to `CH_CONVERGENCE`:
- Bullish convergence → `PatternDomain.OPPORTUNITY`, `PatternForm.CORRELATED`
- Bearish convergence → `PatternDomain.THREAT`, `PatternForm.CORRELATED`
- Salience = `confidence * 0.6 + (domain_count / 12) * 0.4` (capped at 1.0)
- `ttl_steps=10` (convergence alerts are time-sensitive)

**`MarketPartialTranslator`** — subscribes to `CH_PARTIAL_CONVERGENCE`:
- Any partial → `PatternDomain.NOVELTY`, `PatternForm.REACTIVE`
- Salience = `min(0.6, len(domains_seen) * 0.2)` (capped lower — it's partial)
- `ttl_steps=15` (partials stay longer to accumulate attention)

Follow exact pattern from `mae_core/patterns/translators/emergent.py`: `@property source_name`, `@property channels`, `def translate(channel, message)`.

#### 4. New file: `mae_core/network/market_task_handlers.py` (~180 lines)

**`inject_market_handlers(colony, convergence_alerter, pattern_watcher, event_bus)`**:
- Sets `colony._developing_situations = {}` (thread-safe dict)
- Sets `colony._situations_lock = threading.Lock()`
- Monkey-patches `_execute_current_task` on each arm with a dispatch version
- Registers three task handlers:

**`_handle_investigate_partial(task)`**: Re-checks if the partial convergence has grown to full. Calls `convergence_alerter.check_ticker_convergence_for(ticker)`. If full alert fires, publishes `CH_OCTOPUS_INVESTIGATION`.

**`_handle_archaeology_lookup(task)`**: Queries `pattern_watcher.get_active_stacks()` for the ticker. Reports matching templates.

**`_handle_situation_check(task)`**: Lightweight periodic check. Updates DevelopingSituation state, evicts if expired (>100 steps).

**DevelopingSituation** is a plain dict stored in `colony._developing_situations`, keyed by `"{direction}:{ticker}"`. Fields: `ticker, direction, domains_seen, missing_domains, first_seen, check_count`.

#### 5. Modify `mae_core/market/intelligence/convergence_alerter.py` — 3 small additions (~25 lines total)

**At line 1209** (the min_domains gate inside `_check_direction_convergence`):
```python
if len(domains_seen) < self.min_domains:
    # Emit partial convergence for ecosystem investigation
    if directional_signals and self._bus is not None:
        try:
            self._bus.publish("market.intel.partial_convergence", {
                "direction": direction,
                "domains_seen": list(domains_seen),
                "missing_domains": self._compute_missing_domains(domains_seen),
                "signals": [{"source": s.source, "strength": s.strength,
                             "metadata": s.metadata} for s in directional_signals[:5]],
                "min_domains_required": self.min_domains,
            })
        except Exception:
            pass  # Never block convergence check
    return None
```

Note: event bus stored as `self._bus` (line 262 of convergence_alerter.py).

**New helper** `_compute_missing_domains(self, domains_seen)` — returns `sorted(set(self.signals.keys()) - domains_seen)`.

**New method** `check_ticker_convergence_for(self, ticker)` — thin wrapper around existing `check_ticker_convergence()` filtered to a single ticker.

### Round 2 (depends on Round 1 — all builders work in parallel)

#### 6. Modify `mae_core/bootstrap/market_systems.py` — add OctopusColony instantiation

After Pattern Discovery block (line 447), before `_register_trust_and_gateway` (line 449):

```python
# --- OctopusColony (ecosystem bridge) ---
try:
    from mae_core.network.octopus_colony import OctopusColony
    ctx.octopus_colony = OctopusColony(
        event_bus=getattr(ctx, "bus", None),
        min_octopuses=3, max_octopuses=7,
        world_model=getattr(ctx, "shared_world_model", None),
        signal_bus=getattr(ctx, "signal_bus", None),
    )
except Exception:
    logger.debug("Market: octopus_colony failed to construct", exc_info=True)
    ctx.octopus_colony = None
```

#### 7. Modify `mae_core/bootstrap/market_hooks.py` — 3 additions

**A. Coordination cycle** (inside `_market_sense_hook`, every 20 steps):
```python
if step % 20 == 0:
    colony = getattr(ctx, "octopus_colony", None)
    if colony is not None:
        for oct in colony.octopuses.values():
            oct.cognition.run_coordination_cycle()
```

**B. Partial convergence subscription** (inside `_register_market_eventbus`):
Subscribe to `CH_PARTIAL_CONVERGENCE` → register as DevelopingSituation in `colony._developing_situations` (with `_situations_lock` guard).

**C. Handler injection + monitoring start** (inside `_wire_sensing_hook`, after bypass dedup init):
```python
colony = getattr(ctx, "octopus_colony", None)
if colony is not None:
    from mae_core.network.market_task_handlers import inject_market_handlers
    inject_market_handlers(colony, ctx.convergence_alerter, ctx.pattern_watcher, ctx.bus)
    colony.start_monitoring()
```

#### 8. Modify `mae_core/bootstrap/patterns.py` — register market translators

After `TriadicPatternTranslator()` (line 183):
```python
try:
    from mae_core.market.translators.market_signal_translator import (
        MarketConvergenceTranslator, MarketPartialTranslator,
    )
    _translators.append(MarketConvergenceTranslator())
    _translators.append(MarketPartialTranslator())
except ImportError:
    pass  # MIDGE not present — mae-core standalone
```

#### 9. Modify supporting bootstrap files

- **`market.py`**: Add `"octopus_colony"` to `market_attrs` list (line 105)
- **`market_registration.py`**: Add octopus_colony to somatic map and holon registry
- **`market_connections.py`**: Add Group 33 — 3 triadic connections (octopus↔alerter↔watcher, Law 1)
- **`main.py`**: Add `"octopus_colony"` to `_build_systems_dict` market block

#### 10. Tests — 3 new files (~190 lines total)

**`tests/test_market_signal_translator.py`** (~60 lines):
- `test_bullish_convergence_becomes_opportunity()`
- `test_bearish_convergence_becomes_threat()`
- `test_neutral_direction_returns_none()`
- `test_partial_becomes_novelty()`
- `test_salience_scales_with_domain_count()`

**`tests/test_market_task_handlers.py`** (~80 lines):
- `test_inject_sets_handlers_on_arms()`
- `test_execute_dispatches_to_handler()`
- `test_unknown_task_type_safe()`
- `test_developing_situation_lifecycle()`
- `test_situation_evicted_after_max_checks()`

**`tests/test_octopus_bootstrap.py`** (~50 lines):
- `test_colony_on_ctx_after_bootstrap()`
- `test_colony_has_three_octopuses()`
- `test_colony_registered_as_holon()`

---

## Builder Assignments (for triadic-construction)

| Builder | Domain | Files Owned | Round |
|---------|--------|-------------|-------|
| Builder 1 | Channels + Translators | `channels.py`, `market/translators/*`, `tests/test_market_signal_translator.py` | 1 |
| Builder 2 | Convergence Alerter | `convergence_alerter.py` | 1 |
| Builder 3 | Task Handlers | `network/market_task_handlers.py`, `tests/test_market_task_handlers.py` | 1 |
| Builder 4 | Bootstrap Wiring | `market_systems.py`, `market_hooks.py`, `market_registration.py`, `market_connections.py`, `market.py`, `patterns.py`, `main.py`, `tests/test_octopus_bootstrap.py` | 2 |

Reviewers (2, independent — never built):
- Reviewer 1: Focus on translator Protocol compliance, partial emission safety, salience formulas
- Reviewer 2: Focus on Law 1 connections, thread safety, colony lifecycle, regression risk

---

## What This Does NOT Change

- Existing convergence alerter behavior (still fires at min_domains=3, same confidence/strength)
- Signal processing pipeline (adapters, sensing hook, paper trades)
- Existing test assertions — no count changes, no behavior changes
- The 33 biological systems — those remain as-is (ecosystem activation of bio systems is a SEPARATE build)
- Agent behavior — agents still read the same shared advisory

---

## Key Technical Details

- Event bus attribute on ConvergenceAlerter is `self._bus` (line 262)
- PatternTranslator Protocol: `@property source_name`, `@property channels`, `def translate(channel, message)` (from `mae_core/patterns/translators/base.py`)
- Translator registration: `ctx.pattern_bus.register_translator(t)` (patterns.py line 186)
- Colony spawns 3 octopuses on construction (no lazy init)
- `colony.start_monitoring()` starts background health check thread — must be called AFTER handler injection
- `OctopusArm._execute_current_task()` is a stub — the handler injection monkey-patches it with real dispatch
- Thread safety: `colony._situations_lock` protects `_developing_situations` dict (written by EventBus callback, read by step hook)

---

## Verification

```bash
# 1. New tests
python -m pytest tests/test_market_signal_translator.py tests/test_market_task_handlers.py tests/test_octopus_bootstrap.py -v

# 2. Existing convergence/intelligence tests (zero regressions)
python -m pytest tests/test_convergence_alerter.py tests/test_integration.py -v

# 3. Full test suite
python -m pytest tests/ -q

# 4. Smoke test (octopus should appear in bootstrap log)
python main.py --agents 3 --steps 30 2>&1 | grep -i octopus

# 5. Verify pipeline bridge (market signals reach PatternBus)
python main.py --agents 3 --steps 30 2>&1 | grep -i "translator\|pattern.*market"
```

---

## Document Parity

| File | Field | Old → New |
|------|-------|-----------|
| `CLAUDE.md` | Market modules | 114 → 117 (+translators/__init__.py, +market_signal_translator.py, +market_task_handlers.py) |
| `CLAUDE.md` | Systems | 55 market → 56 market (+octopus_colony) |
| `CLAUDE.md` | Connections | 425 → 428 (+3 Group 33) |
| `HANDOFF.md` | Stats, current state | Update counts + add octopus status |
| `mae_core/CONNECTIONS.md` | Connection index | Add Group 33 |
| `tests/test_integration.py` | expected_keys | Add "octopus_colony" |
| `main.py` | systems_dict | Add "octopus_colony" |

---

## What Comes Next (NOT this build)

1. **Biological system activation** — Give the 33 dormant bio systems real market jobs (separate triadic-construction build)
2. **Agent-level situation claiming** — SEC_WATCHER claims insider situations, MARKET_ANALYST claims highest-convergence
3. **DevelopingSituation → full investigation** — Octopus arms query archaeology, prediction markets, targeted re-fetch
4. **Focused attention** — When partial convergence starts, increase polling priority for missing domains
