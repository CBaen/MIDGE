# Morphogenesis — Connection Map

> Part of Mae's connection map. Index: [mae_core/CONNECTIONS.md](../CONNECTIONS.md)

**Status definitions:** WIRED (systems call each other), BUILT (code exists, not wired), STUB (interface exists via `Any`), PLANNED (neither exists yet).

---

## MorphogenesisCoordinator (coordinator.py) — Growth Engine

**Status:** BUILT. 359 lines.

**Provides:**
- `handle_novel_problem(signature)` -> Novelty detection + blueprint + spawn
- `force_create_organ()` -> Testing convenience
- `dissolve_organ()` -> Clean organ removal
- `step()` -> Periodic pruning of underperforming organs
- Subscribes to `morphogenesis.spawn_request` on EventBus (WIRED)
- Accepts `substrate` param for agent registration (BUILT, stub via `Any`)
- Accepts `model` param for Mesa agent creation (BUILT, stub via `Any`)
- Hormonal modulation: `set_growth_rate()` for endocrine integration (BUILT, not wired)

## OrganBuilder (organ_builder.py) — Specialized Team Creation

**Provides:**
- `design_organ(signature)` -> Blueprint from problem analysis
- `grow_organ(blueprint)` -> Spawn agents, connect through substrate topology
- `dissolve_organ()` -> Remove agents from model and substrate
- `prune_organs()` -> Auto-dissolve underperforming organs
- 4 organ topologies: Mesh, Star, Ring, Hierarchical
- 3 coordination protocols: Consensus, Hierarchical, Auction
- When substrate is provided: auto-connects organ agents through substrate topology (WIRED internally)

**EventBus channels (all WIRED):**
- `morphogenesis.spawn_request` (subscribed by coordinator)
- `morphogenesis.team_created` / `morphogenesis.team_dissolved`
- `morphogenesis.novelty_detected`

**Receives from (designed connections):**

| Source | Trigger | Status |
|--------|---------|--------|
| CollectiveDream | Low consensus -> spawn_request | BUILT (channel exists, publish not wired) |
| AutoHealer | Failure -> replacement spawn | BUILT, not wired |
| Colony | Capacity overflow -> scaling | BUILT, not wired |
| Memory | Pattern recognition -> capability gaps | BUILT, not wired |
| Endocrine | Growth hormones -> `set_growth_rate()` | BUILT (method exists, not called) |

---

## Related Modules

- [substrate/CONNECTIONS.md](../substrate/CONNECTIONS.md) — OrganBuilder grows agents through substrate topology
- [coordination/CONNECTIONS.md](../coordination/CONNECTIONS.md) — Endocrine growth hormones modulate spawn rate
- [memory/CONNECTIONS.md](../memory/CONNECTIONS.md) — Capacity warnings trigger growth signals
