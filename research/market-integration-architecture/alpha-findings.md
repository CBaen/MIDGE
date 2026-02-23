# Alpha Findings — Bootstrap Integration & Mae Law Compliance
**Witness Alpha | Analytical Lens: Bootstrap Integration & Mae Law Compliance**
**Date: 2026-02-22**

---

## Executive Summary

MIDGE's 16 market intelligence modules are currently standalone Python objects. They import each other but have zero contact with Mae's nervous system: no EventBus channels, no ConnectionRegistry triads, no HolonRegistry entries, no SomaticMap registration, no step hooks, and no stem cell roles. This document provides a precise, code-grounded design for Layer 33 — `mae_core/bootstrap/market.py` — that makes market intelligence a full citizen of the organism.

---

## Part 1: How the Bootstrap Works

### The orchestration chain

`main.py` calls seven bootstrap functions in sequence (lines 53-59):
```
bootstrap_foundation(ctx)    # Layers 1-11
bootstrap_agents(ctx)        # Layers 12-13
bootstrap_wiring(ctx)        # Layers 14-21
bootstrap_patterns(ctx)      # Layers 22-25
bootstrap_organs(ctx)        # Layers 26-30
bootstrap_external(ctx)      # Layer 31
bootstrap_audit(ctx)         # Layer 32
```

Layer 33 (market) must be inserted before `bootstrap_audit()`. The audit at Layer 32 checks `_MIN_SYSTEMS = 75` (audit.py line 35) and `_MIN_HOLONS = 75` (audit.py line 36). Adding 16 market systems will push both counts higher and keep the audit healthy.

### How systems enter the context (`SimpleNamespace ctx`)

Every bootstrap module receives `ctx` by reference and adds attributes to it. Example from foundation.py line 64:
```python
ctx.model = MycelialModel(persist_dir=Path(ctx.persist_dir))
ctx.bus = ctx.model.event_bus
```

Layer 33 will follow this pattern: `ctx.thompson_sampler = ThompsonSampler(...)`, etc.

### The seal boundary (critical timing constraint)

`connection_registry.seal()` is called at wiring.py line 517, end of Layer 18. After seal, enforcement transitions from PERMISSIVE to ADVISORY. Post-seal registrations still work (audit.py's Group 13 connections register post-seal at Layer 32). However: **every system registered post-seal must call `register_connection()` after `seal()` returns.** Layer 33 is post-seal and this is correct — same pattern as external.py Layer 31.

### The `_register_somatic_systems()` helper

Three bootstrap modules define this helper locally. It calls `somatic_map.register_system(system_id, description, depends_on=[])`. Layer 33 must call this for every market system before registering connections, because `ConnectionRegistry.verify_all()` checks SomaticMap for source/target/witness existence (connection_registry.py lines 496-509).

---

## Part 2: Connection Registry Deep Dive

### Law 1 requirements

`ConnectionRegistry.register_connection()` (connection_registry.py lines 305-397) auto-fills witnesses to minimum 2 if fewer are provided (lines 339-347). The `NERVOUS_SYSTEM` fallback tuple (line 54) is:
```python
NERVOUS_SYSTEM = ("enforcer", "watchdog", "auditor", "somatic_map")
```

For market systems, using `witnesses=["auditor", "connection_registry"]` or `witnesses=["threat_detector", "auditor"]` is appropriate — these are the system-domain witnesses used throughout external.py.

### ConnectionType options

From connection_registry.py lines 57-65:
- `EVENTBUS_PUBSUB` — for EventBus channel subscriptions
- `DIRECT_REFERENCE` — for Python object references held as attributes
- `STEP_HOOK` — for periodic step callbacks registered on the model
- `MEMORY_DATA_FLOW` — for data store reads/writes
- `CALLBACK_REGISTRATION` — for EventBus callback subscriptions

### How abstract names work

Wiring.py lines 451-464 registers abstract system names like "decision_router", "frl", "morphogenesis" as SomaticMap entries so connection verification doesn't flag them as unhealthy. Layer 33 must do the same for abstract names used in market triads (e.g., "market_intelligence" as an abstract organ grouping name).

---

## Part 3: HolonRegistry and HolonProxy

### Registration pattern

Wiring.py lines 469-479: every shared system registers as `holon_type="system"` with `parent_id="mae"`. After fractal organization at Layer 20, parent_id gets changed to the appropriate organ.

Layer 33 market systems should initially register with `parent_id="mae"`, then the fractal grouping will reparent them to `market-intelligence-system` (the new organ).

### Proxy injection pattern

Wiring.py lines 579-603: after registration, each system gets `system._holon = ctx.holon_registry.get_proxy(name)` and `proxy.set_system_ref(system)`. This gives the system all 10 holon capabilities via delegation. Layer 33 must inject proxies into all market system instances.

The 10 capabilities are provided entirely through HolonProxy delegation (holon_protocol.py lines 402-580):
1. `sense()` — delegates to `get_state()`, `get_statistics()`, `get_status()`
2. `remember(key, value)` — per-proxy memory store
3. `decide(stimulus)` — delegates to `decide()`, `evaluate()`, `assess()`
4. `act(action)` — delegates to `step()`, `process()`, `execute()`
5. `learn(feedback)` — delegates to `learn()`, `adapt()`
6. `heal()` — checks health, calls `reset()`, `recover()`, `self_repair()`
7. `know_self()` — ID, type, parent, children count
8. `know_up()` — parent info
9. `know_down()` — children list
10. `know_peers()` — sibling list

**Key implication:** To get meaningful `sense()` output, each market system needs a `get_statistics()` or `get_state()` method. Most already have `get_stats()` (ThompsonSampler line 332) but the proxy looks for `get_statistics`. A thin adapter is needed, or we add `get_statistics = get_stats` aliases.

---

## Part 4: Fractal Generator — Where Market Intelligence Fits

### Current FRACTAL_GROUPING (fractal_generator.py lines 87-116)

Five organs: `nervous-system`, `sensory-system`, `cognitive-system`, `somatic-system`, `metabolic-system`.

### ORGAN_GROUPING (fractal_generator.py lines 142-145)

Two clusters under mae:
- `organ-cluster-vital`: [metabolic-system, somatic-system]
- `organ-cluster-cognitive`: [cognitive-system, sensory-system]
- `nervous-system` stays as direct child of `mae` (the bridge organ)

**Mae's current K3 children: `[organ-cluster-vital, organ-cluster-cognitive, nervous-system]`**

### Where to place Market Intelligence

The current structure has `mae` with 3 children (K3). Adding a fourth organ breaks K3 at the organism level. There are two valid approaches:

**Option A: Extend organ-cluster-cognitive to a triad**
Replace `["cognitive-system", "sensory-system"]` with `["cognitive-system", "sensory-system", "market-intelligence-system"]`. This makes `organ-cluster-cognitive` a proper K3.

**Option B: Create a new organ cluster**
Create `organ-cluster-external`: `["market-intelligence-system"]` — but this is a singleton, violating triadic principle. Not recommended.

**Option C: Rethink organism level — 4 children is non-triadic, advisory only**
Since FractalGenerator is advisory (fractal_generator.py line 202-207: warns but proceeds), adding a fourth direct child of `mae` is legally permissive. It generates a non_triadic_groups advisory log entry.

**Recommendation: Option A**. Extend `organ-cluster-cognitive` to `["cognitive-system", "sensory-system", "market-intelligence-system"]`. Market intelligence IS cognitive — it senses the external information environment and reasons about it. This is biologically coherent.

### Market Intelligence organ internal structure

Following Law 4 (fractal self-similarity) and Law 2 (K3 atoms):

```
market-intelligence-system (organ)
├── market-sensing (subsystem)
│   ├── sec_edgar_client       (API data fetcher: SEC Form 4, 8-K)
│   ├── price_fetcher          (API data fetcher: yfinance + AlphaVantage)
│   └── government_data        (API data fetcher: usa_spending + sam_gov)
├── market-edge (subsystem)
│   ├── cluster_detector       (insider cluster detection)
│   ├── politician_tracker     (congressional trade correlation)
│   └── filing_time_analyzer   (behavioral signal from filing timing)
└── market-learning (subsystem)
    ├── thompson_sampler       (Bayesian explore/exploit)
    ├── convergence_alerter    (crown jewel: multi-domain synthesis)
    └── velocity_detector      (rate-of-change anomaly detection)
```

Notes:
- `house_stock_watcher` and `job_tracker` fit in `market-sensing` but that creates a 5-member subsystem. They can join `contract_predictor` in an "extended-edge" subsystem or be grouped as `government_data_extended`.
- `correlation_tracker` and `learning_config` belong in `market-learning` but again exceeds 3. `learning_config` is not an instantiable system (it's a config dict module) — it should NOT be registered as a holon. `correlation_tracker` joins the `market-learning` subsystem as the 4th item (advisory non-triadic warning acceptable).

Revised clean structure:
```
market-intelligence-system (organ)
├── market-sensing (subsystem: K3)
│   ├── sec_edgar_client
│   ├── price_fetcher
│   └── job_tracker         (hiring blitz = market signal)
├── market-edge (subsystem: K3)
│   ├── cluster_detector
│   ├── politician_tracker
│   └── contract_predictor  (pre-announcement winner prediction)
└── market-learning (subsystem: K3)
    ├── thompson_sampler
    ├── convergence_alerter
    └── velocity_detector
```

Remaining modules with no K3 home: `house_stock_watcher`, `filing_time_analyzer`, `usa_spending`, `sam_gov`, `correlation_tracker`. These should be registered individually as system holons under `market-intelligence-system` but outside the three clean subsystems (advisory non-triadic).

---

## Part 5: Stem Cell Role Profiles

### Current roles (stem_cell.py lines 139-192)

Nine profiles: STEM, EXPLORER, LEARNER, COMMUNICATOR, HEALER, COORDINATOR, SPECIALIST, API_CALLER, LLM_SPECIALIST.

The genome (lines 80-112) has 20 genes across 10 mixins. The `api_call_enabled` gene (line 110) gates external API access.

### Three new market-specific role profiles

**SEC_WATCHER** — Monitors SEC EDGAR for insider trades and material events. Needs: API access, episodic memory (trade history), semantic search (find similar past patterns).
```python
"SEC_WATCHER": {
    "api_call_enabled": True,
    "llm_prompt_quality": 0.6,
    "replay_enabled": True,
    "consolidation_enabled": True,
    "semantic_search_enabled": True,
    "sensing_radius": 10.0,
    "exploration_bonus": 0.15,  # Explore new tickers
    "world_model_enabled": True,
    "planning_horizon": 7,       # 7-day market outlook
    "capabilities": frozenset({"market_sense", "insider_track", "sec_watch"}),
},
```

**CONTRACT_TRACKER** — Monitors government contracts, SAM.gov opportunities, hiring trends. Needs: API access, strong world model (contract award prediction), quorum sensing (coordinate with other trackers).
```python
"CONTRACT_TRACKER": {
    "api_call_enabled": True,
    "llm_prompt_quality": 0.6,
    "quorum_sensing_enabled": True,   # Coordinate tracking across symbols
    "world_model_enabled": True,
    "planning_horizon": 14,            # Contracts have longer lead times
    "replay_enabled": True,
    "transfer_enabled": True,          # Transfer contract patterns across sectors
    "capabilities": frozenset({"market_sense", "contract_track", "govt_monitor"}),
},
```

**MARKET_ANALYST** — Synthesizes signals across domains using convergence alerter and Thompson sampler. Needs: high-quality LLM reasoning, full memory capabilities, world model.
```python
"MARKET_ANALYST": {
    "api_call_enabled": True,
    "llm_prompt_quality": 0.85,
    "world_model_enabled": True,
    "planning_horizon": 10,
    "replay_enabled": True,
    "consolidation_enabled": True,
    "semantic_search_enabled": True,
    "generative_memory_enabled": True,
    "transfer_enabled": True,
    "maml_enabled": True,
    "quorum_sensing_enabled": True,
    "capabilities": frozenset({"market_analyze", "convergence_detect", "signal_synthesize"}),
},
```

### Where to add these

`C:\Users\baenb\projects\MIDGE\mae_core\agents\stem_cell.py`, line 192, append to `ROLE_PROFILES` dict.

---

## Part 6: API Client Routing Through BoundaryMembrane

### Current pattern (external.py lines 179-186)

```python
def _trust_provider(ctx, name):
    if ctx.boundary_membrane is not None:
        try:
            if hasattr(ctx.boundary_membrane, "register_source"):
                ctx.boundary_membrane.register_source(name, trust=0.8)
        except Exception:
            logger.debug("Could not register %s with BoundaryMembrane", name)
```

### What market API clients need

The market API clients (sec_edgar/client.py, price_fetcher.py, house_stock_watcher.py, job_tracker.py, usa_spending.py, sam_gov.py) currently use `requests` directly. They bypass BoundaryMembrane entirely.

**Recommended approach for Layer 33:** Register market API sources with BoundaryMembrane trust scoring. Do NOT rewrite the clients to route through ApiGateway immediately (that is a larger refactor and creates circular dependencies). Instead:

1. Register each external data source as a trusted source in BoundaryMembrane with appropriate trust levels (lower than LLM providers since these are unvalidated market data):
   ```python
   market_sources = [
       ("sec_edgar", 0.90),      # SEC = US government, high trust
       ("yfinance", 0.75),       # Free data, moderate trust
       ("alpha_vantage", 0.80),  # Commercial API, higher trust
       ("rapidapi", 0.65),       # Third-party aggregator, moderate trust
       ("usa_spending", 0.85),   # US government, high trust
       ("sam_gov", 0.80),        # US government, high trust
   ]
   for name, trust in market_sources:
       _trust_provider(ctx, name)  # reuse the helper from external.py
   ```

2. The `InputValidator` (ctx.input_validator) can validate market data payloads via `ctx.input_validator.validate()`. Market systems should call this when processing raw API responses.

3. Full ApiGateway routing is a Phase 2 task. The architecture is prepared by registering the sources now.

---

## Part 7: Step Hooks for Periodic Market Sensing

### Current step hook pattern (wiring.py lines 50-55)

```python
ctx.model.add_step_hook(ctx.predictive_field.step if hasattr(ctx.predictive_field, "step") else lambda: None)
```

### Market-specific sensing cadence

Market data has natural refresh rates:
- SEC EDGAR: ~hourly at best (rate-limited to 10 req/s but new filings come hourly)
- Price data: real-time but we don't need tick-by-tick — per-cycle (once per run or every N steps)
- Congressional trades: daily
- Government contracts: daily to weekly

**Recommended hook structure:**

```python
_market_step_counter = [0]

def _market_sense_hook():
    _market_step_counter[0] += 1
    step = _market_step_counter[0]

    # Every step: check convergence_alerter (lightweight, pure in-memory)
    if hasattr(ctx.convergence_alerter, "check_convergence"):
        try:
            alerts = ctx.convergence_alerter.check_convergence()
            if alerts:
                for alert in alerts:
                    ctx.bus.publish("market.convergence_alert", alert.to_dict())
        except Exception:
            logger.debug("Convergence alerter step failed", exc_info=True)

    # Every 10 steps: Thompson sampler update (check for pending outcomes)
    if step % 10 == 0 and hasattr(ctx.thompson_sampler, "get_stats"):
        try:
            stats = ctx.thompson_sampler.get_stats()
            ctx.bus.publish("market.thompson_stats", stats)
        except Exception:
            logger.debug("Thompson sampler stats step failed", exc_info=True)

    # Every 50 steps: Velocity detector anomaly scan
    if step % 50 == 0 and hasattr(ctx.velocity_detector, "detect_velocity_anomalies"):
        try:
            anomalies = ctx.velocity_detector.detect_velocity_anomalies()
            if anomalies:
                ctx.bus.publish("market.velocity_anomaly", {"anomalies": len(anomalies)})
        except Exception:
            logger.debug("Velocity detector step failed", exc_info=True)

ctx.model.add_step_hook(_market_sense_hook)
```

The API client calls (SEC EDGAR, price fetcher etc.) should NOT be in step hooks — they are slow network calls that would block the simulation. They should be triggered by endocrine state (low melatonin = ACTIVE phase = good time to sense) or explicitly from a MARKET_ANALYST agent's `api_call` action.

---

## Part 8: Full Layer 33 Design — `mae_core/bootstrap/market.py`

### Module structure

```
mae_core/bootstrap/market.py
├── bootstrap_market(ctx) — main entry point
├── _instantiate_market_systems(ctx) — create all 13 instantiable market objects
├── _register_market_holons(ctx) — register with HolonRegistry, inject proxies
├── _register_market_somatic(ctx) — register with SomaticMap
├── _register_market_connections(ctx) — all triadic connections (Group 14)
├── _register_market_fractal(ctx) — add market organ to fractal hierarchy
├── _register_market_stem_roles(ctx) — add role profiles to ROLE_PROFILES
├── _register_market_step_hooks(ctx) — periodic sensing hooks
└── _register_market_eventbus(ctx) — EventBus channel wiring
```

### Systems to instantiate

13 systems (not 16 — learning_config is a config dict module, not an instantiable class; house_stock_watcher, usa_spending, sam_gov are stateless client modules, not objects with persistent state):

| ctx attribute | Class | Module path |
|---|---|---|
| `ctx.sec_edgar_client` | `SECEdgarClient` | `mae_core.market.apis.sec_edgar.client` |
| `ctx.price_fetcher` | `PriceFetcher` | `mae_core.market.apis.price_fetcher` |
| `ctx.house_stock_watcher` | `HouseStockWatcher` | `mae_core.market.apis.house_stock_watcher` |
| `ctx.job_tracker` | `JobTracker` | `mae_core.market.apis.job_tracker` |
| `ctx.usa_spending_client` | `USASpendingClient` | `mae_core.market.apis.usa_spending` |
| `ctx.sam_gov_client` | `SAMGovClient` | `mae_core.market.apis.sam_gov` |
| `ctx.cluster_detector` | `ClusterDetector` | `mae_core.market.edge.cluster_detector` |
| `ctx.politician_tracker` | `PoliticianTracker` | `mae_core.market.edge.politician_tracker` |
| `ctx.filing_time_analyzer` | `FilingTimeAnalyzer` | `mae_core.market.edge.filing_time_analyzer` |
| `ctx.contract_predictor` | `ContractPredictor` | `mae_core.market.edge.contract_predictor` |
| `ctx.thompson_sampler` | `ThompsonSampler` | `mae_core.market.intelligence.thompson_sampler` |
| `ctx.convergence_alerter` | `ConvergenceAlerter` | `mae_core.market.intelligence.convergence_alerter` |
| `ctx.velocity_detector` | `VelocityDetector` | `mae_core.market.intelligence.velocity_detector` |
| `ctx.correlation_tracker` | `CorrelationTracker` | `mae_core.market.intelligence.correlation_tracker` |

Note: `learning_config` module exports `LEARNING_CONFIG` dict and `update_config()` function. It is not a class. It should be treated as configuration that `ThompsonSampler` already reads internally (thompson_sampler.py lines 141-143).

### Graceful degradation requirement

Every instantiation should be wrapped in try/except with fallback to None, following the patterns in bootstrap_patterns.py (deep memory can be None if Qdrant unavailable). Market APIs need env vars (`RAPIDAPI_KEY`, `ALPHA_VANTAGE_KEY`, `SAM_GOV_API_KEY`). Missing keys should produce `None` on ctx, not a crash.

---

## Part 9: Complete Triadic Connection List (Group 14)

Every connection A -> B must have witnesses. The standard market witnesses are `["auditor", "connection_registry"]` for monitoring connections and `["threat_detector", "input_validator"]` for defense-sensitive connections (API calls touching external data).

### Market Sensing subsystem K3 (internal triads)

```
# sec_edgar_client <-> price_fetcher, witnessed by job_tracker
sec_edgar_client -> price_fetcher, witness=[job_tracker, auditor]
price_fetcher -> job_tracker, witness=[sec_edgar_client, auditor]
job_tracker -> sec_edgar_client, witness=[price_fetcher, auditor]
```

### Market Edge subsystem K3

```
cluster_detector -> politician_tracker, witness=[contract_predictor, auditor]
politician_tracker -> contract_predictor, witness=[cluster_detector, auditor]
contract_predictor -> cluster_detector, witness=[politician_tracker, auditor]
```

### Market Learning subsystem K3

```
thompson_sampler -> convergence_alerter, witness=[velocity_detector, auditor]
convergence_alerter -> velocity_detector, witness=[thompson_sampler, auditor]
velocity_detector -> thompson_sampler, witness=[convergence_alerter, auditor]
```

### Cross-subsystem connections (primary integration paths)

These are the connections that make market intelligence flow through Mae's organism:

```
# Edge detectors publish to EventBus, witnessed by defense systems
cluster_detector -> event_bus, type=EVENTBUS_PUBSUB, channel="market.cluster_signal",
    witnesses=[threat_detector, auditor]

politician_tracker -> event_bus, type=EVENTBUS_PUBSUB, channel="market.politician_trade",
    witnesses=[threat_detector, auditor]

contract_predictor -> event_bus, type=EVENTBUS_PUBSUB, channel="market.contract_prediction",
    witnesses=[threat_detector, auditor]

filing_time_analyzer -> event_bus, type=EVENTBUS_PUBSUB, channel="market.filing_signal",
    witnesses=[threat_detector, auditor]

# Intelligence layer subscribes to edge signals (CALLBACK_REGISTRATION)
convergence_alerter -> event_bus, type=CALLBACK_REGISTRATION, channel="market.cluster_signal",
    witnesses=[thompson_sampler, auditor]

convergence_alerter -> event_bus, type=CALLBACK_REGISTRATION, channel="market.politician_trade",
    witnesses=[thompson_sampler, auditor]

convergence_alerter -> event_bus, type=CALLBACK_REGISTRATION, channel="market.contract_prediction",
    witnesses=[thompson_sampler, auditor]

# Thompson Sampler learns from convergence alerts (DIRECT_REFERENCE)
convergence_alerter -> thompson_sampler, type=DIRECT_REFERENCE,
    witnesses=[velocity_detector, knowledge_base]
    description="Convergence results update Thompson distributions"

# Velocity Detector feeds convergence alerter (DIRECT_REFERENCE)
velocity_detector -> convergence_alerter, type=DIRECT_REFERENCE,
    witnesses=[thompson_sampler, auditor]
    description="Velocity data enriches convergence signal strength"

# API clients -> BoundaryMembrane (defense compliance)
sec_edgar_client -> boundary_membrane, type=DIRECT_REFERENCE,
    witnesses=[input_validator, threat_detector]
    description="SEC data passes through immune boundary"

price_fetcher -> boundary_membrane, type=DIRECT_REFERENCE,
    witnesses=[input_validator, threat_detector]
    description="Price data passes through immune boundary"

# Market organ -> Mae's learning systems (integration)
convergence_alerter -> knowledge_base, type=DIRECT_REFERENCE,
    witnesses=[thompson_sampler, auditor]
    description="Convergence alerts stored in organism knowledge base"

thompson_sampler -> event_bus, type=EVENTBUS_PUBSUB, channel="market.thompson_stats",
    witnesses=[auditor, connection_registry]
    description="Bayesian signal quality broadcast"

# Step hook registration
convergence_alerter -> model, type=STEP_HOOK,
    witnesses=[auditor, connection_registry]
    description="Periodic convergence check each simulation step"

# Convergence alerts -> EndocrineSystem (market stress response)
convergence_alerter -> event_bus, type=EVENTBUS_PUBSUB, channel="market.convergence_alert",
    witnesses=[endocrine, auditor]
    description="Strong convergence triggers hormonal response (adrenaline/dopamine)"
```

### Total: approximately 23 triadic connections in Group 14

All use 2 witnesses per Law 1. All have source, target, and witnesses registered in SomaticMap before `verify_all()` runs.

---

## Part 10: HolonRegistry — Market Systems as Holons

### Registration order

1. Register `market-intelligence-system` as organ-level holon (parent=`organ-cluster-cognitive`)
2. Register three subsystem triads: `market-sensing`, `market-edge`, `market-learning` (parent=`market-intelligence-system`)
3. Register individual system holons (parent=their subsystem)
4. Register remaining systems without a clean subsystem assignment (parent=`market-intelligence-system`)

### 10 capabilities per market system

All 10 capabilities are provided automatically via HolonProxy after `proxy.set_system_ref(system)`. The proxy delegates:

- `sense()` → `system.get_statistics()` or `system.get_stats()`
- `remember(k,v)` → proxy's internal dict
- `decide()` → `system.decide()` or `system.evaluate()` (most market systems have neither — proxy returns None, which is correct)
- `act()` → `system.step()` or `system.process()` (convergence_alerter could implement `step()`)
- `learn(feedback)` → `system.learn()` or `system.adapt()` — ThompsonSampler has `update()` which is close
- `heal()` → checks health, calls `system.reset()` if degraded
- `know_self()` → from HolonRegistry entry
- `know_up()` → parent subsystem info
- `know_down()` → child systems (for subsystem holons)
- `know_peers()` → sibling systems in same subsystem

**Recommended adapter additions** (non-breaking, additive only):

For `ThompsonSampler` (thompson_sampler.py):
```python
def get_statistics(self) -> dict:
    """Alias for HolonProxy.sense() delegation."""
    return self.get_stats()
```

For `ConvergenceAlerter` (convergence_alerter.py):
```python
def get_statistics(self) -> dict:
    """Statistics for HolonProxy.sense() delegation."""
    return {
        "domain_count": len(self.signals),
        "alert_count": len(self.alerts),
        "recent_alerts": [a.to_dict() for a in self.alerts[-3:]],
    }

def step(self) -> None:
    """Step hook for periodic convergence check."""
    alerts = self.check_convergence()
    return len(alerts)
```

For `VelocityDetector`:
```python
def get_statistics(self) -> dict:
    return {"signal_count": len(self.signals), ...}
```

---

## Part 11: Integration with Mae's Learning Loop

### Thompson Sampler → Mae's FRL/VDN reliability scores

Currently: FRL engines (per agent) maintain their own Q-tables. Thompson Sampler maintains signal reliability in `data/market/thompson_distributions.json`.

**Integration path:** Subscribe to `frl.policy_update` channel (already published in wiring.py line 118-129). When FRL reports high reward from an api_call action that used market data, feed that as a `success=True` update to ThompsonSampler. This creates a learning feedback loop: market signals that cause profitable agent actions increase in Beta distribution alpha.

### Convergence Alerter → EndocrineSystem

Market convergence is a real-world stress/opportunity signal that should modulate Mae's hormonal state. Wire via EventBus:

```python
def _on_market_convergence(channel, serialized):
    msg = json.loads(serialized) if isinstance(serialized, str) else serialized
    strength = msg.get("strength", 0.0)
    direction = msg.get("direction", "neutral")
    if direction == "bullish" and strength > 0.7:
        ctx.endocrine.release_hormone(HormoneType.DOPAMINE, min(0.4, strength * 0.4), "market_opportunity")
    elif direction == "bearish" and strength > 0.7:
        ctx.endocrine.release_hormone(HormoneType.ADRENALINE, min(0.5, strength * 0.5), "market_threat")

ctx.bus.register_callback("market.convergence_alert", _on_market_convergence)
```

### Convergence Alerter → KnowledgeBase

Strong convergence alerts should be stored in Mae's shared knowledge base for multi-step reasoning. The KnowledgeBase has a `store()` method. Wire post-alert-generation.

---

## Part 12: Full FRACTAL_GROUPING Update

The `FRACTAL_GROUPING` dict in `fractal_generator.py` (lines 87-116) needs a new entry. However, since Layer 33 runs after Layer 20 (FractalGenerator.organize() called at wiring.py line 633), the market organ must be added via direct `fractal_generator.generate_triad()` calls rather than modifying FRACTAL_GROUPING. FRACTAL_GROUPING modification would affect ALL future bootstrap calls and should be done as a separate architectural decision.

**For Layer 33, use the generate_triad() API directly:**

```python
# Register subsystem triads
ctx.fractal_generator.generate_triad(
    name="market-sensing",
    holon_type="subsystem",
    children_ids=["sec_edgar_client", "price_fetcher", "job_tracker"],
    parent_id="market-intelligence-system",
)
ctx.fractal_generator.generate_triad(
    name="market-edge",
    holon_type="subsystem",
    children_ids=["cluster_detector", "politician_tracker", "contract_predictor"],
    parent_id="market-intelligence-system",
)
ctx.fractal_generator.generate_triad(
    name="market-learning",
    holon_type="subsystem",
    children_ids=["thompson_sampler", "convergence_alerter", "velocity_detector"],
    parent_id="market-intelligence-system",
)

# Register the organ itself
ctx.fractal_generator.generate_triad(
    name="market-intelligence-system",
    holon_type="organ",
    children_ids=["market-sensing", "market-edge", "market-learning"],
    parent_id="organ-cluster-cognitive",  # reparent from "mae"
)
```

This updates `organ-cluster-cognitive` from 2 to 3 children, making it a proper K3.

---

## Part 13: Changes Required in main.py

### Import addition

```python
from mae_core.bootstrap.market import bootstrap_market
```

### Insertion point (main.py line 58-59, between external and audit)

```python
bootstrap_external(ctx)      # Layer 31: external API gateway
bootstrap_market(ctx)        # Layer 33: market intelligence organ
bootstrap_audit(ctx)         # Layer 32: triadic bootstrap audit (self-check)
```

Note: Layer 33 is numbered 33 because it follows Layer 32's audit design intent, but in execution order it runs before the audit. Layer 32 runs last by design to verify the complete organism. The number reflects "what's new" not "execution position."

### systems_dict update

`_build_systems_dict(ctx)` in main.py (lines 72-192) must add all market systems:

```python
# Market Intelligence (Layer 33)
"sec_edgar_client": ctx.sec_edgar_client,
"price_fetcher": ctx.price_fetcher,
"house_stock_watcher": ctx.house_stock_watcher,
"job_tracker": ctx.job_tracker,
"usa_spending_client": ctx.usa_spending_client,
"sam_gov_client": ctx.sam_gov_client,
"cluster_detector": ctx.cluster_detector,
"politician_tracker": ctx.politician_tracker,
"filing_time_analyzer": ctx.filing_time_analyzer,
"contract_predictor": ctx.contract_predictor,
"thompson_sampler": ctx.thompson_sampler,
"convergence_alerter": ctx.convergence_alerter,
"velocity_detector": ctx.velocity_detector,
"correlation_tracker": ctx.correlation_tracker,
```

Each must tolerate `None` (graceful degradation when env vars absent).

---

## Part 14: Document Parity Impact

Per CLAUDE.md document parity rule, the following counts change when Layer 33 is implemented:

| Metric | Before | After |
|--------|--------|-------|
| Systems | 85 | ~99 (+14 market systems) |
| Market modules wired | 0 | 16 |
| Bootstrap layers | 32 | 33 |
| Fractal organs | 5 | 6 |
| Triadic connections | 313 | ~336 (+23 Group 14) |
| Stem cell roles | 9 | 12 (+3 market roles) |

Files to update: CLAUDE.md, HANDOFF.md, README.md, CONNECTIONS.md, data/MAES-MATHEMATICAL-IDENTITY.md, tests/test_integration.py, main.py.

---

## Part 15: Key Risks and Constraints

### Risk 1: Circular import — market modules import each other

`politician_tracker.py` imports from `mae_core.market.apis.sec_edgar` and `mae_core.market.apis.usa_spending`. `contract_predictor.py` imports from `mae_core.market.apis.job_tracker`, `sam_gov`, and `sec_edgar`. These are already resolved (MEMORY.md confirms all imports fixed). No new circular risk from Layer 33 since Layer 33 only imports market classes, not the other way.

### Risk 2: ThompsonSampler creates files at import time

`thompson_sampler.py` line 109: `DATA_DIR.mkdir(parents=True, exist_ok=True)`. This runs at construction time, not import time. This is fine — Layer 33 constructs ThompsonSampler after the bootstrap has established working directories.

### Risk 3: ClusterDetector makes Qdrant HTTP calls

`cluster_detector.py` line 178: `requests.post(f"{self.qdrant_url}/...")`. This is in `find_clusters()`, not `__init__()`. Construction is safe. Step hooks should not call `find_clusters()` directly — only agent `api_call` actions or explicit CLI invocations should trigger network calls.

### Risk 4: Audit thresholds

`audit.py` `_MIN_SYSTEMS = 75` and `_MIN_HOLONS = 75`. Adding 14 market systems increases both counts. The audit should remain healthy (both thresholds currently met with ~85 systems and ~107 holons).

### Risk 5: organ-cluster-cognitive becomes non-K3 briefly

During fractal organization, `organ-cluster-cognitive` starts with 2 children. After Layer 33 adds `market-intelligence-system`, it has 3. This is the correct final state. The advisory warning at 2 children is acceptable during bootstrap.

---

## Part 16: Recommended Implementation Sequence

1. Add `get_statistics()` adapters to ThompsonSampler, ConvergenceAlerter, VelocityDetector (additive methods, no behavior change)
2. Add three ROLE_PROFILES to stem_cell.py: SEC_WATCHER, CONTRACT_TRACKER, MARKET_ANALYST
3. Create `mae_core/bootstrap/market.py` following the design above
4. Add `bootstrap_market(ctx)` call to main.py between external and audit
5. Update `_build_systems_dict()` in main.py
6. Update document parity in all tracked files
7. Run `python -m pytest tests/ -v` — all 2425 existing tests must still pass
8. Run `python main.py --agents 3 --steps 30` — verify Layer 33 log messages appear

Layer 33 should produce log output like:
```
Layer 33a - Market systems instantiated: 14 systems (graceful degradation: 0 failed)
Layer 33b - Market holons registered: 17 holons (14 systems + 3 subsystems)
Layer 33c - Market fractal: market-intelligence-system organ created (3 subsystems, 9 K3 connections)
Layer 33d - Market connections: 23 triadic connections registered (Group 14)
Layer 33e - Market EventBus: 8 channels wired (3 publish, 5 subscribe)
Layer 33f - Market step hooks: 1 sense hook registered (Fibonacci cadence: 1/10/50 steps)
Layer 33  - Market Intelligence organ complete: 14 systems, 17 holons, 23 connections
```

---

## Summary

Layer 33 (`mae_core/bootstrap/market.py`) integrates 14 instantiable market intelligence systems as full Mae citizens. The design satisfies all 8 Mae mathematical laws:

- **Law 1 (No Bare Dyads):** 23 triadic connections registered with explicit 2-witness pairs
- **Law 2 (Triadic Generator):** Three K3 subsystems (market-sensing, market-edge, market-learning) with 3 members each
- **Law 3 (Holon Protocol):** All 10 capabilities via HolonProxy injection on every market system
- **Law 4 (Fractal Self-Similarity):** Market organ fits into organ-cluster-cognitive, completing its K3; market-intelligence-system has 3 subsystem children
- **Law 5 (Stem Cell):** Three new role profiles (SEC_WATCHER, CONTRACT_TRACKER, MARKET_ANALYST) added to ROLE_PROFILES
- **Law 6 (Autopoietic Closure):** Convergence alerter publishes to EventBus → EndocrineSystem modulates agents → agents take market actions → outcomes update ThompsonSampler → ThompsonSampler influences convergence weights → circular closure
- **Law 7 (Rule of 3/5):** All three subsystems have exactly 3 members; no subsystem has 2 (bare dyad would apply at system level too)
- **Law 8 (Eight Properties):** Market organ integrates external information (differentiation), feeds into Mae's decision cascade (integration), and has self-referential Bayesian learning via ThompsonSampler (self-reference, prediction/error-correction)
