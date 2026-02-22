# Network (Octopus) — Connection Map

> Part of Mae's connection map. Index: [mae_core/CONNECTIONS.md](../CONNECTIONS.md)

**Status definitions:** WIRED (systems call each other), BUILT (code exists, not wired), STUB (interface exists via `Any`), PLANNED (neither exists yet).

---

## OctopusArm (octopus_arm.py) — Autonomous Limb

**Provides:**
- Independent task processing with background thread
- Capability-based task filtering
- Peer coordination signal handling (4 types)
- Health and workload self-monitoring
- Emergency load shedding

**Consumed by:**

| Consumer | How It's Used | Value | Status |
|----------|-------------|-------|--------|
| OctopusDistributedCognition | Arms compose the octopus body | WIRED |
| OctopusAgent | Health metrics aggregate to agent health | WIRED |
| Colony monitoring | Arm-level metrics for colony health | WIRED |

**Requires:**
- ArmCapability set (what this arm can do)
- CoordinationSignals from central brain and peer arms
- Optional: Connected arms set (ring topology)

**Growth:**
- Arms should eventually have mini-WorldModels (arm-local prediction)
- Arms should emit signals on SignalBus (inter-octopus arm communication)
- Arms should have memory (remember recent task patterns)
- Arm capabilities should evolve through learning (specialization deepens)

---

## OctopusDistributedCognition (octopus_cognition.py) — Central Brain

**Provides:**
- Coordination of 8 arms (submit, balance, learn)
- Adaptive mode switching (CENTRALIZED/HYBRID/DISTRIBUTED/EMERGENCY)
- Workload balancing across arms
- Learning propagation to all arms
- Emergency mode protocol

**Consumed by:**

| Consumer | How It's Used | Value | Status |
|----------|-------------|-------|--------|
| OctopusAgent | Core cognition system | WIRED |
| Colony monitoring | System status reports | WIRED |

**Requires:**
- OctopusArm instances (created internally)
- Optional: EventBus for lifecycle events

**Growth:**
- Coordination mode should be influenced by endocrine system
- Emergency mode should trigger morphogenesis (spawn helper octopuses)
- Learning propagation should use CausalReasoning (propagate WHY, not just WHAT)

---

## OctopusAgent (octopus_agent.py) — Individual Octopus

**Provides:**
- Unified interface: submit tasks, route decisions, predict with confidence
- Specialization-based capability boosting
- Health tracking (arm health * 0.7 + success rate * 0.3)
- Arm-level prediction with confidence-based escalation to central WorldModel

**Consumed by:**

| Consumer | How It's Used | Value | Status |
|----------|-------------|-------|--------|
| OctopusColony | Colony manages multiple octopuses | WIRED |
| EventBus | Publishes lifecycle events | WIRED |

**Requires:**
- EventBus (for lifecycle events) | WIRED
- Optional: DecisionRouter (three-tier arm cognition) | WIRED
- Optional: WorldModel (central prediction when arm confidence low) | WIRED
- Optional: SignalBus (electrical signaling) | WIRED
- Future: Substrate (network topology registration) | BUILT, not wired
- Future: Memory systems (task outcome storage) | BUILT, not wired
- Future: Morphogenesis coordinator (colony growth requests) | BUILT, not wired

**Growth:**
- Each octopus should store task outcomes in episodic memory
- High-performing octopuses should be imitated by new spawns (imitation learning)
- Octopus health should influence endocrine hormone levels
- Octopus specialization should deepen through transfer learning

---

## OctopusColony (octopus_colony.py) — The Network

**Provides:**
- P2P multi-octopus network (NO hierarchies)
- Rule of 3 enforcement (min 3 octopuses, 2-3 peers each)
- Auto-scaling (workload-based spawn/despawn)
- Self-healing (replace unhealthy octopuses)
- Emergent task routing (least-loaded, no central router)
- Colony health monitoring

**Consumed by:**

| Consumer | How It's Used | Value | Status |
|----------|-------------|-------|--------|
| MycelialModel | (future) Colony as sub-network in model | PLANNED |
| EventBus | Publishes spawn/despawn/health events | WIRED |

**Requires:**
- EventBus (for lifecycle events) | WIRED
- Optional: DecisionRouter (propagated to octopuses) | WIRED
- Optional: WorldModel (propagated to octopuses) | WIRED
- Optional: SignalBus (propagated to octopuses) | WIRED
- Future: Substrate (manages actual topology) | BUILT, not wired
- Future: Morphogenesis (receives spawn/dissolve commands) | BUILT, not wired
- Future: Endocrine system (hormone-modulated thresholds) | BUILT, not wired

**Growth:**
- Colony spawn threshold should be modulated by cortisol (stress = faster spawning)
- Colony should report to morphogenesis when it detects capability gaps
- Inter-colony communication (multiple colonies for different domains)
- Colony consensus should feed into quorum sensing system

---

## Related Modules

- [cognition/CONNECTIONS.md](../cognition/CONNECTIONS.md) — WorldModel and DecisionRouter consumed by OctopusAgent
- [communication/CONNECTIONS.md](../communication/CONNECTIONS.md) — SignalBus provides electrical signaling to octopuses
- [morphogenesis/CONNECTIONS.md](../morphogenesis/CONNECTIONS.md) — Colony growth triggers morphogenesis
