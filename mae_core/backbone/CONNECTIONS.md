# Backbone — Connection Map

> Part of Mae's connection map. Index: [mae_core/CONNECTIONS.md](../CONNECTIONS.md)

**Status definitions:** WIRED (systems call each other), BUILT (code exists, not wired), STUB (interface exists via `Any`), PLANNED (neither exists yet).

---

## EventBus (event_bus.py) — The Nervous System

**Biological analogy:** Spinal cord + nerve fibers

**Provides to ALL systems:**
- Channel-based pub/sub messaging (publish/register_callback/listen)
- Stream operations for ordered data flow
- Numpy-safe JSON serialization
- Zero external dependencies (replaces Redis)

**Current subscribers:**
| Channel | Publisher | Subscriber | Status |
|---------|-----------|------------|--------|
| `octopus.task_submitted` | OctopusCognition | Colony monitor | WIRED |
| `octopus.spawn` | OctopusColony | Monitor, logging | WIRED |
| `octopus.despawn` | OctopusColony | Monitor, logging | WIRED |
| `octopus.emergency` | OctopusCognition | All arms | WIRED |
| `octopus.learning_update` | OctopusCognition | All arms | WIRED |
| `octopus.health_report` | OctopusColony | Monitor | WIRED |
| `frl.policy_update` | FederatedRL | Peer agents | WIRED |
| `haven.risk_alert` | HAVEN | AutoHealer, all agents | WIRED |
| `memory.experience_stored` | MemoryCoordinator | Curiosity drive | WIRED |
| `memory.consolidation_started` | MemoryCoordinator | Logging | WIRED |
| `memory.consolidation_complete` | MemoryCoordinator | Learning systems | WIRED |
| `memory.novel_experience` | MemoryCoordinator (via SemanticRetriever) | Curiosity drive | WIRED |
| `memory.capacity_warning` | MemoryCoordinator | Morphogenesis | WIRED |
| `signal.electrical` | SignalBus | Agents, octopuses | WIRED |
| `stigmergy.pheromone` | Stigmergy | Nearby agents | WIRED |
| `quorum.threshold_reached` | QuorumSensor | Consensus system | WIRED |
| `substrate.agent_registered` | MycelialSubstrate | Logging | WIRED |
| `substrate.agent_deregistered` | MycelialSubstrate | Logging | WIRED |
| `substrate.topology_changed` | MycelialSubstrate | Logging | WIRED |
| `substrate.health_report` | MycelialSubstrate | Monitoring | WIRED |
| `substrate.starvation_alert` | MycelialSubstrate | AutoHealer | WIRED |
| `substrate.isolation_detected` | MycelialSubstrate | Logging | WIRED |
| `morphogenesis.spawn_request` | CollectiveDream (external) | MorphogenesisCoordinator | WIRED |
| `morphogenesis.team_created` | MorphogenesisCoordinator | Logging | WIRED |
| `morphogenesis.team_dissolved` | MorphogenesisCoordinator | Logging | WIRED |
| `morphogenesis.novelty_detected` | MorphogenesisCoordinator | Logging | WIRED |
| `endocrine.hormone_release` | EndocrineSystem | All subscribers | WIRED |
| `endocrine.state_update` | EndocrineSystem | All subscribers | WIRED |
| `circadian.phase_change` | CircadianRhythm | Endocrine, Substrate, Memory | WIRED |
| `healing.failure_detected` | AutoHealer | Monitoring | WIRED |
| `healing.started` | AutoHealer | Monitoring | WIRED |
| `healing.phase_changed` | AutoHealer | Monitoring | WIRED |
| `healing.complete` | AutoHealer | Monitoring | WIRED |
| `healing.failed` | AutoHealer | Monitoring | WIRED |
| `improvement.capability_found` | CapabilityDiscovery | Morphogenesis | WIRED |
| `improvement.capability_validated` | CapabilityDiscovery | Logging | WIRED |
| `improvement.capability_retired` | CapabilityDiscovery | Logging | WIRED |
| `improvement.metric` | CapabilityDiscovery | Monitoring | WIRED |
| `defense.threat_detected` | ThreatDetector | AutoHealer | WIRED |
| `defense.activated` | ThreatDetector | Monitoring | WIRED |
| `defense.threat_neutralized` | ThreatDetector | Monitoring | WIRED |
| `defense.validation_failed` | InputValidator | HAVEN | WIRED |
| `defense.trust_updated` | InputValidator | Monitoring | WIRED |
| `somatic.modification_proposed` | SomaticMap | Monitoring | WIRED |
| `somatic.modification_approved` | SomaticMap | Monitoring | WIRED |
| `somatic.modification_rejected` | SomaticMap | Monitoring | WIRED |
| `somatic.modification_rolled_back` | SomaticMap | Monitoring | WIRED |
| `somatic.system_registered` | SomaticMap | Monitoring | WIRED |
| `temporal.event_recorded` | TemporalMemory | Monitoring | WIRED |
| `temporal.causal_link_discovered` | TemporalMemory | CausalEngine | WIRED |
| `temporal.pattern_detected` | TemporalMemory | Monitoring | WIRED |
| `planning.worldline_planned` | WorldlinePlanner | Monitoring | WIRED |
| `planning.worldline_selected` | WorldlinePlanner | Monitoring | WIRED |
| `planning.worldline_validated` | WorldlinePlanner | Monitoring | WIRED |
| `triad.violation` | TriadEnforcer | Watchdog, Auditor | WIRED |
| `triad.registered` | TriadEnforcer | Watchdog | WIRED |
| `triad.vote_complete` | TriadEnforcer | Monitoring | WIRED |
| `triad.health` | TriadEnforcer | Watchdog | WIRED |
| `watchdog.bypass` | TriadWatchdog | Auditor | WIRED |
| `watchdog.silent` | TriadWatchdog | Auditor | WIRED |
| `watchdog.health` | TriadWatchdog | Monitoring | WIRED |
| `audit.finding` | TriadAuditor | Monitoring | WIRED |
| `audit.health` | TriadAuditor | Monitoring | WIRED |
| `connection.registered` | ConnectionRegistry | Monitoring | WIRED |
| `connection.verified` | ConnectionRegistry | Monitoring | WIRED |
| `connection.bare_dyad` | ConnectionRegistry | Monitoring | WIRED |
| `connection.health` | ConnectionRegistry | Monitoring | WIRED |
| `holon.awareness_pulse` | AwarenessPulse | Monitoring | WIRED |
| `holon.anomaly_detected` | AwarenessPulse | Monitoring | WIRED |
| `stem_cell.registered` | StemCellRegistry | Monitoring | WIRED |
| `stem_cell.redifferentiated` | StemCellRegistry | Monitoring | WIRED |

**Growth:** EventBus is the universal connector. Every new system MUST define its event channels in this table. No system communicates without EventBus awareness.

---

## Enforcement Triad (triad_enforcer.py, triad_watchdog.py, triad_auditor.py, triad_registry.py) — Rule of 3

**Status:** BUILT. 4 files.

**TriadEnforcer provides:**
- Process registration with 3 validators (odd-only, no deadlock)
- Formal vote execution: all 3 validators vote, majority wins
- 6 validator types: STRUCTURAL, BEHAVIORAL, OPERATIONAL, PREDICTIVE, CONSENSUS, CAUSAL

**TriadWatchdog provides:**
- Runtime monitoring that validators aren't bypassed
- Silent validator detection
- Health reporting

**TriadAuditor provides:**
- Behavioral auditing of validator decisions
- Finding tracking with severity levels

**TriadRegistry provides:**
- `register_all_triads()` -> Wires all Mae systems with validator triads at startup
- Uses SomaticMap as STRUCTURAL validator, HAVEN as BEHAVIORAL, AutoHealer as OPERATIONAL
- References WorldModel, ValidatedImagination, Quorum, CausalEngine, TemporalMemory, NutrientFlow, Endocrine

**EventBus channels (all WIRED):**
- `triad.violation` / `triad.registered` / `triad.vote_complete` / `triad.health`
- `watchdog.bypass` / `watchdog.silent` / `watchdog.health`
- `audit.finding` / `audit.health`

---

## Holon Protocol (holon_protocol.py) — Fractal Self-Awareness

**Status:** WIRED. 4 classes: HolonRegistry, HolonMixin, HolonProxy, AwarenessPulse.

**HolonRegistry provides:**
- `register/unregister` — Add/remove holons from the containment hierarchy
- `get_parent/get_children/get_peers` — Navigate the hierarchy
- `get_ancestry/get_subtree` — Walk up or down the tree
- `set_parent` — Reparent (with circular reference protection)
- `get_statistics` — Registry-level metrics
- `set_somatic_map(sm)` — Inject SomaticMap for health queries
- `set_connection_registry(cr)` — Inject ConnectionRegistry for connection queries
- `get_proxy(holon_id)` — Get/create cached HolonProxy for a system

**HolonMixin provides (10 universal capabilities on every agent):**
- `holon_sense()` — Perceive local state + neighbors
- `holon_remember(key, value)` — Memory storage/retrieval
- `holon_decide(stimulus)` — Route decision through existing infrastructure
- `holon_act(action)` — Execute in problem domain
- `holon_learn(action, reward)` — Update from outcomes
- `holon_heal()` — Self-assessment health check, reports to SomaticMap
- `holon_know_self()` — Self-model (ID, type, capabilities, health, hierarchy counts)
- `holon_know_up()` — Parent context
- `holon_know_down()` — Children state
- `holon_know_peers()` — Sibling awareness

**HolonProxy provides (same interface for non-agent systems):**
- `know_self()` — ID, type, parent, children count, peers count, health
- `know_up()` — Parent ID, type, children count
- `know_down()` — Children IDs and types
- `know_peers()` — Peer IDs and types
- `get_health()` — From SomaticMap (1.0 if unavailable)
- `get_connections()` — From ConnectionRegistry

**AwarenessPulse provides (step hook, fires every 25 steps):**
- Orphan detection (parent declared but doesn't exist)
- Health gradient check (parent vs children health divergence)
- Publishes summary on EventBus

**Hierarchy at bootstrap (3 agents):**
- mae (organism) -> colony (colony) -> agent-0, agent-1, agent-2 (agents)
- mae (organism) -> 35 shared systems
- Total: 41 holons

**Layer 19 in bootstrap:** Injects `_holon` proxy into all 36 shared systems. Creates AwarenessPulse step hook.

**Relationship to SomaticMap:** Complementary. SomaticMap tracks dependencies (what breaks if X fails). HolonRegistry tracks containment (what lives inside X, who are siblings). HolonProxy reads health from SomaticMap.

**EventBus channels:**
- `holon.awareness_pulse` — Periodic hierarchy health summary
- `holon.anomaly_detected` — Orphans, health gradients, peer drift

**Consumed by:** Every MycelialAgent via HolonMixin. Every shared system via HolonProxy (`_holon` attribute).

---

## Connection Registry (connection_registry.py) — Triadic Witnessing with Enforcement

**Status:** WIRED. Created in Layer 18 of bootstrap. Sealed (activated) at end of Layer 18.

**Biological analogy:** The lymphatic system. Every blood vessel (connection) has lymph nodes (witnesses) that monitor what flows through.

**Enforcement modes:**
- **PERMISSIVE:** Bootstrap phase. No checks. Everything passes. (default during bootstrap)
- **ADVISORY:** Log + event on violations. Nothing blocked. (default after seal)
- **BLOCKING:** Reject bare dyads at registration. Reject unregistered/unhealthy connections on check. Return False on is_connection_allowed().

**ConnectionRegistry provides:**
- `register_connection(source, target, type, ...)` -> Register a witnessed connection triad. In BLOCKING mode, rejects bare dyads.
- `seal()` -> End bootstrap grace period. Transition to configured enforcement mode (default ADVISORY). Idempotent.
- `set_enforcement_mode(mode)` -> Switch enforcement at runtime
- `is_connection_allowed(source, target, channel)` -> Check if communication is permitted. Returns `(bool, reason_str)`. In BLOCKING mode, rejects unregistered/unhealthy/bare dyads.
- `verify_all()` -> Batch verification of all connections (every 25 steps). In BLOCKING mode, publishes connection.blocked for unhealthy triads.
- `get_bare_dyads()` -> Connections missing witnesses
- `get_coverage_report()` -> What % of systems have triadic connections
- `get_statistics()` -> Counts, health, coverage, witness load, enforcement mode, sealed flag

**Auto-witness assignment heuristic:**
1. Find systems in SomaticMap that touch BOTH source and target (shared neighbor)
2. If none, pick from nervous system (enforcer, watchdog, auditor, somatic_map) via hash round-robin
3. Fallback: enforcer (always exists)

**Connection types registered (227+ total at bootstrap, 158 registered via register_all_connections + ~69 fractal K3/auto_healer/bidirectional awareness):**
- 5 registration groups: Metabolic/OrganismState, Backbone Self-Monitoring, Cognition, Agent Lifecycle, Defense/Healing
- 12 CRITICAL, 16 IMPORTANT, 199 STANDARD
- 0 bare dyads, all connections have 2+ witnesses

**Witness load distribution:**
- Balanced: no single witness >15% of assignments
- Domain peer witnesses (not just backbone round-robin)

**EventBus channels (all WIRED):**
- `connection.registered` — New connection registered
- `connection.verified` — Batch verification report (every 25 steps)
- `connection.bare_dyad` — Bare dyads detected (enforcement mode in payload)
- `connection.health` — Health summary
- `connection.blocked` — Connection blocked by enforcement (BLOCKING mode only)
- `connection.sealed` — Registry sealed, enforcement activated

**Integrates with:** TriadWatchdog (check_bare_dyads escalates in BLOCKING mode), SomaticMap (witness assignment), HolonRegistry (connection queries).

**Bootstrap sequence:**
1. Layer 18: ConnectionRegistry created (PERMISSIVE mode)
2. Layer 18: register_all_connections() declares all system connections (158 registered)
3. Layer 18 end: seal() activates enforcement (default ADVISORY)
4. Periodic (every 25 steps): verify_all() checks connection health

---

## Fractal Generator (fractal_generator.py) — Recursive Fractal Organizer

**Status:** WIRED. Created in Layer 20 of bootstrap.

**Biological analogy:** Embryonic morphogen gradients. The pattern that tells cells "you are part of the nervous system" or "you are part of the somatic system" — organizing flat collections into hierarchical organs and subsystems.

**FractalGenerator provides:**
- `organize()` -> Build fractal hierarchy from flat system list (4 organs, 13 subsystems)
- K3 wiring of every triad of children at each level with natural witnesses
- Fractal depth: max 4 (mae -> organ -> subsystem -> system, or mae -> organ -> subsystem -> colony -> agent)

**Fractal hierarchy:**
- **nervous-system** — backbone systems (EventBus, StateStore, VectorStore, Triad system, etc.)
- **sensory-system** — communication and signal systems
- **cognitive-system** — cognition, learning, planning, memory systems
- **somatic-system** — defense, emergent, substrate, morphogenesis, coordination, network systems

**Connections added:** ~69 K3 fractal + auto_healer + bidirectional awareness connections (227+ total in Mae)

**EventBus channels (all WIRED):**
| Channel | Publisher | Subscriber | Status |
|---------|-----------|------------|--------|
| `fractal.triad_created` | FractalGenerator | Monitoring | WIRED |
| `fractal.organized` | FractalGenerator | Monitoring | WIRED |

**Integrates with:** HolonRegistry (registers organs/subsystems as holons), ConnectionRegistry (registers K3 connections with witnesses), EventBus (publishes organization events).

---

## Related Modules

- [emergent/CONNECTIONS.md](../emergent/CONNECTIONS.md) — SomaticMap is a key validator and awareness source
- [coordination/CONNECTIONS.md](../coordination/CONNECTIONS.md) — Endocrine publishes on EventBus channels
- [learning/CONNECTIONS.md](../learning/CONNECTIONS.md) — FRL publishes policy updates via EventBus
