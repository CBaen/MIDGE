# Mae Connection Map — Index

**No system lives alone. Mae is a connected organism.**

Each module owns its own connection docs. This file is the index + cross-cutting summaries.

Last updated: 2026-02-22

---

## Module Connection Maps

| Module | File | Systems Covered |
|--------|------|----------------|
| backbone | [backbone/CONNECTIONS.md](backbone/CONNECTIONS.md) | EventBus, Enforcement Triad, Holon Protocol, Connection Registry, FractalGenerator |
| agents | [agents/CONNECTIONS.md](agents/CONNECTIONS.md) | StemCellRegistry (AgentGenome, AgentEpigenome, redifferentiate) |
| cognition | [cognition/CONNECTIONS.md](cognition/CONNECTIONS.md) | WorldModel, DecisionRouter, CausalReasoning, CollectiveDream, ValidatedImagination |
| network | [network/CONNECTIONS.md](network/CONNECTIONS.md) | OctopusArm, Cognition, Agent, Colony |
| substrate | [substrate/CONNECTIONS.md](substrate/CONNECTIONS.md) | MycelialSubstrate, Topology, NutrientFlow, PhysarumOptimizer |
| learning | [learning/CONNECTIONS.md](learning/CONNECTIONS.md) | FederatedRL |
| memory | [memory/CONNECTIONS.md](memory/CONNECTIONS.md) | MemoryCoordinator, 10 subsystems |
| communication | [communication/CONNECTIONS.md](communication/CONNECTIONS.md) | SignalBus, Stigmergy, Quorum, GNN, PredictiveField, SpatialConsensus |
| morphogenesis | [morphogenesis/CONNECTIONS.md](morphogenesis/CONNECTIONS.md) | MorphogenesisCoordinator, OrganBuilder |
| coordination | [coordination/CONNECTIONS.md](coordination/CONNECTIONS.md) | EndocrineSystem, CircadianRhythm |
| emergent | [emergent/CONNECTIONS.md](emergent/CONNECTIONS.md) | AutoHealer, CapabilityDiscovery, SomaticMap |
| defense | [defense/CONNECTIONS.md](defense/CONNECTIONS.md) | ThreatDetector, InputValidator, PearlDefense |
| planning | [planning/CONNECTIONS.md](planning/CONNECTIONS.md) | TemporalMemory, WorldlinePlanner |

---

## Connection Matrix Summary

**Total EventBus channels wired:** 60+
**Total cross-system connections mapped:** 80+

| Phase | Systems | EventBus Channels | Direct Wiring | Stub/Ready |
|-------|---------|-------------------|---------------|------------|
| 5.1 Foundation | EventBus, StateStore, VectorStore | 6 | 6 | 0 |
| 5.2 Memory | 10 files + coordinator | 5 | 5 (EventBus) | 12 (consumers not wired) |
| 5.3 Communication | 7 channels | 8 | 8 | 2 (substrate position) |
| 5.4 Learning | 8 engines | 12 | 12 | 4 |
| 5.5 Cognition + Octopus | 10 systems | 18 | 18 | 6 |
| 5.6 Growth | Substrate(3), Morphogenesis(2), Coordination(2) | 14 | 14 (EventBus) | 10 (cross-system stubs) |
| 5.7 Self-Improvement | AutoHealer, CapabilityDiscovery, SomaticMap, ThreatDetector, InputValidator | 15 | 15 (EventBus) + 2 (subscriptions) | 6 (cross-system stubs) |
| 5.8 Temporal | TemporalMemory, WorldlinePlanner | 6 | 6 (EventBus) | 4 (cross-system stubs) |
| 5.8+ Enforcement | TriadEnforcer, Watchdog, Auditor, Registry | 10 | 10 (EventBus) | 0 |
| 5.8++ Holon | HolonRegistry, HolonMixin, HolonProxy, AwarenessPulse | 2 | 36 (proxy injection) + 1 (bootstrap) | 0 |
| 5.8+++ Connection | ConnectionRegistry | 4 | 372 (217 core + 47 fractal + 55 bootstrap + 53 market, all witnessed, 0 bare dyads) | 0 |
| **5.9 Integration** | **API, Dashboard, Domain Config** | **0** | **0** | **20+ (all stubs need wiring)** |

---

## Wiring Architecture: How Systems Connect

Mae's wiring follows a consistent pattern across all phases:

**Pattern 1: EventBus pub/sub (WIRED everywhere)**
Every system defines channel constants (`CH_*`) and publishes lifecycle events.
This is fully wired across all 60+ channels. Any system can subscribe to any channel.

**Pattern 2: `Any`-typed optional parameters (BUILT, not wired)**
Systems accept cross-system references as `Optional[Any]` init parameters:
- AutoHealer accepts `substrate`, `causal_engine`, `haven`
- WorldlinePlanner accepts `world_model`, `temporal_memory`, `causal_engine`
- MorphogenesisCoordinator accepts `model`, `substrate`
These are READY for wiring - the calling code exists, it just needs a real object passed in.

**Pattern 3: Direct method calls (PLANNED)**
Some cross-system connections require one system to directly call another's methods:
- DecisionRouter reading `endocrine.get_reflex_bias()`
- OctopusAgent storing task outcomes via `memory.store()`
These require code changes in the consuming system.

**Phase 5.9 integration work — COMPLETE:**
`main.py` `create_mae()` now performs all integration:
1. Creates all system instances (28+ shared, 5 per-agent)
2. Wires `Any`-typed params with real objects (substrate, causal_engine, haven, etc.)
3. Registers circadian phase callbacks to Endocrine
4. Registers all 28 systems with SomaticMap
5. Calls `register_all_triads()` with real system references
6. Runs step loop via model.run() with step hooks for circadian, endocrine, predictive_field, auto_healer, capability_discovery

---

## Rules for Future Development

1. **Every new system must update its module's CONNECTIONS.md** with Provides/Requires tables
2. **Every PLANNED connection must specify which phase will wire it**
3. **No system may be built without connection points** to at least 2 other systems
4. **Substrate is the body** - systems needing topology/position should wire through substrate
5. **EventBus is the nervous system** - every lifecycle event publishes here
6. **Memory is the context** - every system that reasons must have memory access
7. **No orphan systems** - if a system has no consumers, it should not exist
8. **SomaticMap must know about every system** - register on init, heartbeat periodically
9. **Triad enforcement covers every process** - register via triad_registry
10. **`Any`-typed stubs are intentional** - they prevent circular imports while marking ready connections

---

## Market Connections (Group 14 — Layer 33) — additions

The following 3 connections were added with the Session Sweep Detector (total market connections before Group 15: 26):

| Connection | Type | Witness | Notes |
|-----------|------|---------|-------|
| session_sweep_detector ↔ event_bus | eventbus_pubsub | convergence_alerter | Publishes sweep signals to CH_MARKET_SIGNAL |
| convergence_alerter ↔ event_bus (session_sweep channel) | eventbus_pubsub | session_sweep_detector | Receives session sweep signals for multi-domain synthesis |
| session_sweep_detector ↔ price_fetcher | direct_reference | convergence_alerter | Fetches intraday price data to confirm sweep patterns |

## Market Connections (Group 15 — Layer 33) — lag/calibration/sizing

12 connections for 4 new intelligence systems (total market connections: 38):

| Connection | Type | Witness | Notes |
|-----------|------|---------|-------|
| signal_archive_reader ↔ event_bus | eventbus_pubsub | lag_correlation_analyzer | Publishes archive load events |
| lag_correlation_analyzer ↔ event_bus | eventbus_pubsub | signal_archive_reader | Publishes lag findings to CH_LAG_FINDING |
| lag_correlation_analyzer ↔ signal_archive_reader | direct_reference | convergence_alerter | Reads archived signals for cross-correlation |
| thompson_calibrator ↔ event_bus | eventbus_pubsub | thompson_sampler | Publishes calibration stats to CH_THOMPSON_STATS |
| thompson_calibrator ↔ thompson_sampler | direct_reference | convergence_alerter | Seeds missing distribution keys, reads priors |
| thompson_calibrator ↔ convergence_alerter | data_flow | thompson_sampler | Calibration informs confidence adjustments |
| kelly_position_sizer ↔ event_bus | eventbus_pubsub | thompson_sampler | Publishes sizing recommendations to CH_KELLY_SIZING |
| kelly_position_sizer ↔ thompson_sampler | direct_reference | convergence_alerter | Reads p_win from Thompson distribution means |
| kelly_position_sizer ↔ convergence_alerter | data_flow | thompson_sampler | Triggered by per-ticker convergence alerts |
| signal_archive_reader ↔ convergence_alerter | data_flow | lag_correlation_analyzer | Archive data enriches convergence context |
| lag_correlation_analyzer ↔ convergence_alerter | data_flow | thompson_calibrator | Lag findings inform cross-domain synthesis |
| kelly_position_sizer ↔ lag_correlation_analyzer | data_flow | convergence_alerter | Leading indicators inform position timing |

## Market Connections (Group 16 — Layer 33) — TA indicators

3 connections for the TA indicators edge module (total market connections: 48):

| Connection | Type | Witness | Notes |
|-----------|------|---------|-------|
| ta_indicators → price_fetcher | direct_reference | convergence_alerter, auditor | TA indicators compute RSI/MACD/Bollinger from daily OHLCV |
| convergence_alerter → ta_indicators | direct_reference | thompson_sampler, auditor | Convergence subscribes to TA indicator signals |
| ta_indicators → thompson_sampler | data_flow | convergence_alerter, auditor | Thompson learns TA indicator reliability from outcomes |

## Market Connections (Group 17 — Layer 33) — backtest bridge

3 connections for the backtest-to-hypothesis bridge (total market connections: 51):

| Connection | Type | Witness | Notes |
|-----------|------|---------|-------|
| backtest_analyzer → hypothesis_registry | direct_reference | hypothesis_validator, auditor | Registers backtest-derived hypotheses in registry |
| backtest_analyzer → hypothesis_validator | direct_reference | hypothesis_registry, auditor | Backtest hypotheses flow to validator for DSR evaluation |
| hypothesis_engine → backtest_analyzer | direct_reference | hypothesis_generator, auditor | Engine drives backtest analysis on generation cadence |
