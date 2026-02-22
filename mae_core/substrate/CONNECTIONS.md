# Substrate — Connection Map

> Part of Mae's connection map. Index: [mae_core/CONNECTIONS.md](../CONNECTIONS.md)

**Status definitions:** WIRED (systems call each other), BUILT (code exists, not wired), STUB (interface exists via `Any`), PLANNED (neither exists yet).

---

## MycelialSubstrate (mycelial_substrate.py) — The Soil

**Biological analogy:** Mycelial network underground. The soil everything grows in.

**Status:** BUILT. 3 files: mycelial_substrate.py (619 lines), topology.py (500 lines), nutrient_flow.py (266 lines).

**Provides:**
- **Agent topology management:** Register/deregister agents at graph positions
- **Nutrient flow:** Resource distribution via osmotic pressure gradients (Hebbian edge strengthening)
- **Signal propagation:** `propagate_signal()` carries signals along edges with attenuation
- **Peer discovery:** `get_peers()` returns network neighbors (Rule of 3 convention)
- **Position tracking:** `get_agent_position()`, `get_all_agent_positions()`
- **Topology graph export:** `get_topology_graph()` for GNN routing
- **Region isolation:** `isolate_region()` / `restore_region()` for AutoHealer containment
- **Dynamic growth:** `grow_node()` / `prune_node()` for Morphogenesis
- **Health monitoring:** Periodic health reports, starvation alerts, isolation detection
- **Circadian integration:** `set_phase()` modulates flow and decay rates
- **4 topology types:** Ring, Scale-Free (Barabasi-Albert), Small-World (Watts-Strogatz), Mesh

**EventBus channels (all WIRED):**
- `substrate.agent_registered` / `substrate.agent_deregistered`
- `substrate.topology_changed`
- `substrate.health_report`
- `substrate.starvation_alert` (subscribed by AutoHealer)
- `substrate.isolation_detected`

**Imports:** EventBus (WIRED)

**Consumed by:**

| Consumer | What It Needs | Connection | Status |
|----------|--------------|------------|--------|
| MorphogenesisCoordinator | `substrate` param, `register_agent()`, `grow_node()` | Optional init param | BUILT (stub via `Any`) |
| OrganBuilder | `substrate` param for agent registration + topology wiring | Optional param in `grow_organ()` | BUILT (stub via `Any`) |
| AutoHealer | `substrate` param, `isolate_region()`, `get_peers()` | Optional init param | BUILT (stub via `Any`) |
| OctopusColony | Agent topology, peer connections | NOT CONNECTED | BUILT, not wired |
| SignalBus | Signal propagation paths | NOT CONNECTED | BUILT, not wired |
| Stigmergy | Spatial environment | NOT CONNECTED | BUILT, not wired |
| GNNCommunicator | `get_topology_graph()` | `substrate` init param | WIRED (main.py injects substrate) |
| SpatialConsensus | Agent positions | NOT CONNECTED | BUILT, not wired |
| PredictiveFields | Field propagation | `substrate` init param | WIRED (main.py injects substrate) |
| CircadianRhythm | `set_phase()` integration | Phase change callback | WIRED (circadian->endocrine in main.py) |
| EndocrineSystem | Hormone distribution via substrate | EventBus hormone_release | WIRED (consumers registered in main.py) |

---

## Related Modules

- [morphogenesis/CONNECTIONS.md](../morphogenesis/CONNECTIONS.md) — OrganBuilder grows agents through substrate topology
- [coordination/CONNECTIONS.md](../coordination/CONNECTIONS.md) — Circadian phases modulate substrate flow rates
- [emergent/CONNECTIONS.md](../emergent/CONNECTIONS.md) — AutoHealer isolates substrate regions during healing
