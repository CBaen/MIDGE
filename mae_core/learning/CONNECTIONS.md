# Learning — Connection Map

> Part of Mae's connection map. Index: [mae_core/CONNECTIONS.md](../CONNECTIONS.md)

**Status definitions:** WIRED (systems call each other), BUILT (code exists, not wired), STUB (interface exists via `Any`), PLANNED (neither exists yet).

---

## FederatedRL (frl.py) — Peer Policy Sharing

**Provides:**
- Policy sharing via EventBus (replaces Redis pub/sub)
- Peer trust scoring (performance, round-robin, adaptive strategies)
- Privacy-preserving policy updates (share gradients, not raw data)

**Consumed by:**

| Consumer | How It's Used | Value | Status |
|----------|-------------|-------|--------|
| All agents | Peer learning | WIRED |
| MAML | (future) Meta-weights shared via FRL | PLANNED |

**Requires:**
- EventBus (for policy distribution) | WIRED
- Peer agent IDs (for trust tracking) | WIRED
- Future: HAVEN verification of peer policies | PLANNED
- Future: InputValidator for policy validation | BUILT, not wired

**Growth:**
- MAML meta-learning should share meta-weights through FRL channels
- HAVEN should verify incoming policies before integration (trust verification)
- InputValidator validates policy updates before acceptance
- Endocrine modulation: oxytocin increases peer trust threshold

---

---

## Tier 2 Persistence (serialize/restore)

The following learning subsystems implement `serialize()` and `restore(data)` for persisting learned state across sessions:

| Subsystem | What Is Persisted |
|-----------|-------------------|
| ValueDecompositionEngine (VDN) | Mixing network weights, value decomposition state |
| FederatedLearningEngine (FRL) | Peer trust scores (JSON), policy cache |
| MAMLLearner | Meta-parameters, adaptation state |
| KnowledgeBase | Shared skill repository |

Data flows to/from `data/mae/subsystems/agents/{agent_id}/` (per-agent: VDN, FRL, MAML) and `data/mae/subsystems/shared/` (KnowledgeBase). FRL trust scores use JSON serialization; other subsystems use pickle for heavy objects. Missing files on restore cause the subsystem to start fresh (graceful degradation).

---

## Related Modules

- [defense/CONNECTIONS.md](../defense/CONNECTIONS.md) -- InputValidator validates incoming policy updates
- [backbone/CONNECTIONS.md](../backbone/CONNECTIONS.md) -- FRL publishes `frl.policy_update` on EventBus
- [memory/CONNECTIONS.md](../memory/CONNECTIONS.md) -- FRL uses memory for replay batches
