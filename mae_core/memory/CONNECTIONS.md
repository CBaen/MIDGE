# Memory — Connection Map

> Part of Mae's connection map. Index: [mae_core/CONNECTIONS.md](../CONNECTIONS.md)

**Status definitions:** WIRED (systems call each other), BUILT (code exists, not wired), STUB (interface exists via `Any`), PLANNED (neither exists yet).

---

## Memory Systems (10 files)

**Status:** BUILT. coordinator.py, episodic_memory.py, prioritized_replay_buffer.py, semantic_retriever.py, experience_vae.py, generative_replay.py, memory_consolidator.py, working_memory.py, sum_tree.py, experience.py.

**MemoryCoordinator provides:**
- `store()` -> Store experience across all subsystems (episodic + generative + working + semantic)
- `sample()` -> Prioritized sampling from episodic memory
- `update_priorities()` -> TD-error priority updates
- `search()` -> Cascading search: WorkingMemory(O(1)) -> Episodic(O(log N)) -> Semantic(embedding)
- `consolidate()` -> Offline "sleep" learning phase
- `should_consolidate()` -> Check if consolidation is due
- Novelty detection via semantic distance (emits `memory.novel_experience`)
- Capacity monitoring (emits `memory.capacity_warning` at 90% utilization)

**EventBus channels (all WIRED):**
- `memory.experience_stored` -> Published on every store()
- `memory.consolidation_started` / `memory.consolidation_complete`
- `memory.novel_experience` -> When semantic distance exceeds threshold
- `memory.capacity_warning` -> At 90% capacity

**Imports:** EventBus (WIRED). All internal subsystems wire through coordinator.

**Subsystem details:**

| Subsystem | Purpose |
|-----------|---------|
| EpisodicMemory | Prioritized experience buffer with semantic indexing |
| PrioritizedReplayBuffer | TD-error weighted sampling via SumTree |
| SemanticRetriever | FAISS-backed embedding similarity search |
| ExperienceVAE | Variational autoencoder for experience compression |
| GenerativeReplayMemory | VAE-based synthetic experience generation |
| MemoryConsolidator | Offline learning (replay during "sleep") |
| WorkingMemory | 7+/-2 slot attention buffer with decay |
| SumTree | Efficient priority sampling data structure |
| Experience | Dataclass for state/action/reward/next_state/done |

**Consumed by (what needs memory):**

| Consumer | What It Needs | Status |
|----------|--------------|--------|
| Learning (FRL, VDN, Curiosity) | `sample()` experiences | BUILT, not wired |
| DecisionRouter | Working memory context | BUILT, not wired |
| WorldModel | Training batches | BUILT, not wired |
| CausalEngine | Stored interventions | BUILT, not wired |
| CollectiveDream | Past dream outcomes | BUILT, not wired |
| ValidatedImagination | Prediction-outcome pairs | BUILT, not wired |
| OctopusAgent | Task outcome history | BUILT, not wired |
| Morphogenesis | Capability gap patterns | BUILT, not wired |
| Endocrine | Emotional context | BUILT, not wired |
| TemporalMemory | Temporal event chains | BUILT, not wired |
| Curiosity | Novelty comparison | BUILT, not wired |
| Transfer Learning | Task embeddings | BUILT, not wired |

**The memory system is complete and publishes all lifecycle events on EventBus. The remaining work is wiring consuming systems to call memory's methods directly.**

### Tier 2 Persistence (serialize/restore)

The following memory subsystems implement `serialize()` and `restore(data)` for persisting learned state across sessions:

| Subsystem | What Is Persisted |
|-----------|-------------------|
| EpisodicMemory | Replay buffer experiences, priorities, SumTree state |
| SemanticRetriever | FAISS index, stored documents |
| GenerativeReplayMemory | VAE weights, generated experiences |
| MemoryConsolidator | Consolidation stats, schedule state |
| MemoryCoordinator | Delegates serialize/restore to all child subsystems |

Data flows to/from `data/midge/subsystems/agents/{agent_id}/` (per-agent) and `data/midge/subsystems/shared/` (shared systems). Pickle is used for heavy objects (SumTree, numpy arrays, experience buffers). Metadata is stored in StateStore. Missing files on restore cause the subsystem to start fresh (graceful degradation).

---

## Related Modules

- [coordination/CONNECTIONS.md](../coordination/CONNECTIONS.md) — Consolidation triggers melatonin via Endocrine
- [morphogenesis/CONNECTIONS.md](../morphogenesis/CONNECTIONS.md) — Capacity warnings trigger growth
- [learning/CONNECTIONS.md](../learning/CONNECTIONS.md) — FRL/VDN consume memory samples
