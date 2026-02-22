# Communication — Connection Map

> Part of Mae's connection map. Index: [mae_core/CONNECTIONS.md](../CONNECTIONS.md)

**Status definitions:** WIRED (systems call each other), BUILT (code exists, not wired), STUB (interface exists via `Any`), PLANNED (neither exists yet).

---

## Communication Systems (7 channels)

**Status:** BUILT. All 7 subsystems complete.

| Channel | Speed | Range | Persistence | Use Case |
|---------|-------|-------|-------------|----------|
| Electrical (SignalBus) | Fast (<1ms) | Network-wide | None | Emergency, sync |
| Pheromone (Stigmergy) | Slow (decays) | Local | Trail-based | Exploration, marking |
| Quorum (QuorumSensor) | Medium | Population | Threshold | Consensus |
| GNN (GNNCommunicator) | Learned | Selective | Weights | Intelligent routing |
| Aggregation (MessageAggregator) | Varies | Varies | Dedup buffer | Noise reduction |
| PredictiveField | Spatial | Local | Grid decay | Flock coordination |
| SpatialConsensus | Medium | Regional | Position-aware | Location-aware voting |

---

## Signal Priority Protocol

**Status:** WIRED. Thalamus-like signal triage system.

**Biological analogy:** Thalamic relay — prioritizes sensory input before it reaches the cortex.

**Architecture:**
- **SignalPriorityResolver** (signal_priority.py) — Per-agent queue with priority scoring, coalescing, budget enforcement
- Plugs into BaseAgent.step() — processes queued signals before lifecycle phases
- Maps signals to DecisionTier (Reflex/Habit/Prefrontal) based on priority score

**Connections:**
- signal_priority.py **imports from** signal_bus.py (Signal, SignalBus)
- signal_priority.py **imports from** decision_router.py (DecisionTier)
- mycelial_agent.py **creates** SignalPriorityResolver instance
- base_agent.py **calls** resolver.process() at top of step()

**Priority scoring:** `priority_weight=0.5, urgency_weight=0.3, recency_weight=0.2`
- Default urgency map: DANGER=1.0, COLLABORATION_REQUEST=0.7, CONVERGENCE=0.6, OPPORTUNITY=0.5, KNOWLEDGE_SHARE=0.3

**Budget enforcement:** 10 signals per step, deferred overflow with 3-step max age

**Preemption:** Signals with priority >= 0.9 bypass queue immediately

**Tier mapping:**
- >= 0.8: Reflex (immediate)
- >= 0.5: Habit (routine)
- < 0.5: Prefrontal (deliberative)

**Coalescing:** Same-type signals merged with log(count) boost to priority

---

**Priority protocol (now auto-enforced via SignalPriorityResolver):**
1. Emergency (electrical) always takes priority
2. Consensus (quorum) overrides individual signals
3. Learned routes (GNN) preferred over random
4. Pheromone trails provide ambient context, not direct commands
5. Aggregator deduplicates before delivery
6. PredictiveField provides spatial coordination context
7. SpatialConsensus integrates location into decisions

**Substrate integration designed but not yet wired:**
- PredictiveField notes: "Substrate provides agent positions"
- SpatialConsensus notes: "Substrate provides agent positions"
- Both systems accept positions as parameters (ready for wiring)

---

## Related Modules

- [substrate/CONNECTIONS.md](../substrate/CONNECTIONS.md) — Substrate provides positions for PredictiveField and SpatialConsensus
- [network/CONNECTIONS.md](../network/CONNECTIONS.md) — SignalBus provides electrical signaling to octopuses
- [backbone/CONNECTIONS.md](../backbone/CONNECTIONS.md) — All channels publish on EventBus
