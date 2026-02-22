# Planning — Connection Map

> Part of Mae's connection map. Index: [mae_core/CONNECTIONS.md](../CONNECTIONS.md)

**Status definitions:** WIRED (systems call each other), BUILT (code exists, not wired), STUB (interface exists via `Any`), PLANNED (neither exists yet).

---

## TemporalMemory (temporal_memory.py) — Spatiotemporal Timeline

**Provides:**
- `record_event(event)` -> Store 4D event with auto-linking
- Temporal neighbor discovery (events within time window)
- Causal link discovery (same entity, close in time -> causal chain)
- Pattern detection (recurring event sequences)
- `trace_causal_chain()` -> Follow dominoes forward or backward
- `find_common_causes()` -> Common ancestors of two events
- `predict_next_event_type()` -> Pattern-based prediction
- Multi-dimensional querying: by time range, entity, position

**EventBus channels (all WIRED):**
- `temporal.event_recorded`
- `temporal.causal_link_discovered`
- `temporal.pattern_detected`

**Cross-system connections:**

| System | Connection | Status |
|--------|-----------|--------|
| EventBus | Publishes temporal events | WIRED (via `Any` param) |
| CausalEngine | `observe_correlation()` when causal links found | BUILT (stub via `Any`) |
| WorldlinePlanner | `temporal_memory` param for pattern integration | BUILT (stub via `Any`) |

---

## WorldlinePlanner (worldline_planner.py) — Trajectory Planning Through Spacetime

**Provides:**
- `plan(entity, state, actions, horizon)` -> Generate and rank worldlines
- `plan_multi_horizon()` -> Reactive (1-3) + Tactical (4-10) + Strategic (11-25) simultaneously
- `begin_execution()` / `check_divergence()` / `complete_worldline()` -> Execution tracking
- `get_temporal_context()` -> Context for DecisionRouter deliberation
- Branch-and-evaluate: N branches, score by discounted reward minus uncertainty

**EventBus channels (all WIRED):**
- `planning.worldline_planned` / `planning.worldline_selected` / `planning.worldline_validated`

**Cross-system connections:**

| System | Connection | Status |
|--------|-----------|--------|
| EventBus | Publishes planning events | WIRED (via `Any` param) |
| WorldModel | `step()` for state projection | BUILT (stub via `Any`) |
| TemporalMemory | `predict_next_event_type()` for action selection | BUILT (stub via `Any`) |
| CausalEngine | `causal_engine` param (not yet used in methods) | BUILT (stub via `Any`) |

---

## Related Modules

- [cognition/CONNECTIONS.md](../cognition/CONNECTIONS.md) — WorldModel and CausalEngine consumed by planners
- [memory/CONNECTIONS.md](../memory/CONNECTIONS.md) — TemporalMemory feeds causal discoveries to memory
