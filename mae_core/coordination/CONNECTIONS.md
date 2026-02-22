# Coordination — Connection Map

> Part of Mae's connection map. Index: [mae_core/CONNECTIONS.md](../CONNECTIONS.md)

**Status definitions:** WIRED (systems call each other), BUILT (code exists, not wired), STUB (interface exists via `Any`), PLANNED (neither exists yet).

---

## EndocrineSystem (endocrine_system.py) — Hormonal Modulation

**Status:** BUILT. 427 lines. 6 hormones.

**Provides:**
- `release_hormone(type, amount, trigger)` / `suppress_hormone()`
- `get_level()` / `get_all_levels()` / `get_global_state()`
- Convenience modulation readouts: `get_exploration_bias()`, `get_trust_level()`, `get_urgency_level()`, `get_reflex_bias()`
- `is_stressed()` / `is_resting()` -> Quick state checks
- `subscribe(hormone, callback)` -> Per-hormone subscription
- `set_circadian_phase(phase)` -> Responds to circadian transitions
- `step()` -> Hormone decay toward baseline + periodic state publish
- Cascade effects: adrenaline triggers cortisol, cortisol suppresses serotonin, etc.

**6 hormones:**

| Hormone | Trigger | Effect | Consuming Systems (designed) |
|---------|---------|--------|------------------------------|
| Dopamine | Reward, novelty | Exploration, creativity | Curiosity, CollectiveDream, Transfer |
| Serotonin | Success, stability | Cooperation, patience | FRL trust, Quorum threshold |
| Cortisol | Stress, failure | Urgency, lower quality | Colony spawn, DecisionRouter reflex |
| Oxytocin | Cooperation | Trust, peer sharing | FRL peer selection, Imitation |
| Adrenaline | Emergency | Speed, minimize deliberation | DecisionRouter reflex, Colony emergency |
| Melatonin | Circadian REST | Consolidation, reduce activity | Memory consolidation, Substrate flow |

**EventBus channels (all WIRED):**
- `endocrine.hormone_release`
- `endocrine.state_update`

**Integration notes:** Consuming systems are designed to read hormone levels but the actual calls from those systems to EndocrineSystem methods are not yet wired. The EndocrineSystem publishes on EventBus and systems can subscribe, but direct method calls (e.g., DecisionRouter reading `get_reflex_bias()`) are not connected.

---

## CircadianRhythm (circadian_rhythm.py) — The Clock

**Status:** BUILT. 270 lines. 3 phases.

**Provides:**
- `step()` -> Advance clock, detect phase transitions
- Phase queries: `is_active()`, `is_consolidating()`, `is_resting()`
- Activity multiplier: `get_activity_multiplier()` (1.0 / 0.5 / 0.1)
- `should_consolidate_memory()` / `should_learn()`
- `on_phase_change(callback)` -> Register phase transition callback
- Phase change publishes on EventBus (WIRED)

**Phase schedule:**

| Phase | Duration (default) | What Happens | Systems Active |
|-------|-------------------|-------------|----------------|
| ACTIVE | 60% of cycle | Normal operation | All systems |
| CONSOLIDATION | 25% of cycle | Offline learning | Memory consolidation, generative replay |
| REST | 15% of cycle | Minimal activity | Health recovery, weight decay |

**EventBus channels (all WIRED):**
- `circadian.phase_change`

**Integration designed (not auto-wired):**
- EndocrineSystem has `set_circadian_phase()` method ready to receive phase changes
- MycelialSubstrate has `set_phase()` method ready to modulate flow/decay
- MemoryCoordinator uses `should_consolidate()` flag from consolidator
- The callback registration to connect these is a Phase 5.9 wiring task

---

## Related Modules

- [substrate/CONNECTIONS.md](../substrate/CONNECTIONS.md) — Circadian phases modulate substrate flow rates
- [memory/CONNECTIONS.md](../memory/CONNECTIONS.md) — Consolidation phase triggers memory sleep cycles
- [morphogenesis/CONNECTIONS.md](../morphogenesis/CONNECTIONS.md) — Endocrine growth hormones modulate spawn rate
