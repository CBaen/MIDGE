# Emergent — Connection Map

> Part of Mae's connection map. Index: [mae_core/CONNECTIONS.md](../CONNECTIONS.md)

**Status definitions:** WIRED (systems call each other), BUILT (code exists, not wired), STUB (interface exists via `Any`), PLANNED (neither exists yet).

---

## AutoHealer (auto_healer.py) — Three-Phase Recovery

**Status:** BUILT. 511 lines.

**Provides:**
- `report_failure(failure)` -> Begin three-phase healing pipeline
- Phase 1 ISOLATE: HAVEN agent isolation + substrate region isolation
- Phase 2 ASSESS: CausalEngine root cause analysis
- Phase 3 RESTORE: Execute recovery actions + reconnect
- Phase 4 VERIFY: Confirm healing success
- Registers default recovery actions (load redistribution, nutrient injection)
- Subscribes to `haven.risk_alert` on EventBus (WIRED)
- Subscribes to `substrate.starvation_alert` on EventBus (WIRED)

**EventBus channels (all WIRED):**
- `healing.failure_detected` / `healing.started` / `healing.phase_changed`
- `healing.complete` / `healing.failed`

**Cross-system connections:**

| System | Connection | Status |
|--------|-----------|--------|
| EventBus | Imports callback registration | WIRED |
| HAVEN | `haven.risk_alert` subscription | WIRED (EventBus callback) |
| Substrate | `substrate.starvation_alert` subscription | WIRED (EventBus callback) |
| CausalEngine | `query_causation()` in Phase 2 | BUILT (stub via `Any`) |
| HAVEN | `isolate_agent()` / `restore_agent()` in Phases 1,3 | BUILT (stub via `Any`) |
| Substrate | `isolate_region()` / `restore_region()` in Phases 1,3 | BUILT (stub via `Any`) |
| Substrate | `get_peers()` for load redistribution | BUILT (stub via `Any`) |
| Morphogenesis | (future) Spawn replacement agents | PLANNED |

---

## CapabilityDiscovery (capability_discovery.py) — Finding Novel Behaviors

**Status:** BUILT. 341 lines.

**Provides:**
- `observe_performance()` -> Detect novel capability from performance anomaly
- `submit_validation()` -> Validate capability over multiple rounds
- `deploy_capability()` / `retire_capability()` -> Lifecycle management
- `track_metric()` -> Self-improvement metric tracking
- `get_improvement_summary()` -> Improvement dashboard

**EventBus channels (all WIRED):**
- `improvement.capability_found` / `improvement.capability_validated`
- `improvement.capability_retired` / `improvement.metric`

---

## SomaticMap (somatic_map.py) — Body Awareness / Proprioception

**Status:** BUILT. 629 lines.

**Provides:**
- `register_system()` -> Every Mae system registers its dependencies
- `analyze_blast_radius()` -> Compute full cascade of modifying any system
- `propose_modification()` -> Gate ALL self-modifications through impact analysis
- `execute_modification()` -> Snapshot before execute for rollback safety
- `complete_modification()` / `rollback_modification()` -> Auto-rollback on failure
- `get_body_map()` -> Full dependency graph for visualization
- `get_critical_path()` -> Systems on the critical path (CRITICAL + PROTECTED)
- `get_unhealthy_systems()` -> Systems below health threshold
- Heartbeat monitoring: each system calls `heartbeat()` periodically

**EventBus channels (all WIRED):**
- `somatic.modification_proposed` / `somatic.modification_approved`
- `somatic.modification_rejected` / `somatic.modification_rolled_back`
- `somatic.system_registered`

**Consumed by (designed):**
- CapabilityDiscovery checks blast radius before deploying capabilities
- AutoHealer uses body map to understand cascade paths
- Morphogenesis checks impact before spawning/dissolving
- TriadRegistry uses SomaticMap as STRUCTURAL validator type

---

## Related Modules

- [defense/CONNECTIONS.md](../defense/CONNECTIONS.md) — ThreatDetector updates SomaticMap on defense activation
- [backbone/CONNECTIONS.md](../backbone/CONNECTIONS.md) — SomaticMap is key validator for TriadRegistry; HolonProxy reads health from SomaticMap
- [substrate/CONNECTIONS.md](../substrate/CONNECTIONS.md) — AutoHealer isolates substrate regions during healing
