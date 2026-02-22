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
| 5.8+++ Connection | ConnectionRegistry | 4 | 336 (211 core + 47 fractal + 55 bootstrap + 23 market, all witnessed, 0 bare dyads) | 0 |
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
