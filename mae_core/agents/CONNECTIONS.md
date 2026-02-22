# Agents — Connection Map

> Part of Mae's connection map. Index: [mae_core/CONNECTIONS.md](../CONNECTIONS.md)

**Status definitions:** WIRED (systems call each other), BUILT (code exists, not wired), STUB (interface exists via `Any`), PLANNED (neither exists yet).

---

## StemCellRegistry (stem_cell.py) — Agent Genome/Epigenome Tracking

**Status:** WIRED. Created in Layer 21 of bootstrap.

**Biological analogy:** The bone marrow stem cell niche. Tracks every cell's genome (DNA catalog) and epigenome (gene expression state), and can trigger redifferentiation to any of 7 roles.

**StemCellRegistry provides:**
- `register(agent)` -> Register agent with genome/epigenome tracking
- `redifferentiate(agent_id, role)` -> Change agent role by applying profile config
- `get_epigenome(agent_id)` -> Get agent's current epigenome (role, overrides, lineage)
- `get_all_agents()` -> List all tracked agents and their roles

**AgentGenome (frozen singleton):**
- 20 configurable genes across 10 mixins
- Singleton `DEFAULT_GENOME` shared by all agents
- Read-only catalog of what CAN be configured

**AgentEpigenome (per-agent):**
- Current role (one of 7 ROLE_PROFILES)
- Active config overrides
- Role lineage (history of role changes)
- Lock flag (prevent accidental redifferentiation)

**ROLE_PROFILES (7 predefined configurations):**
- STEM, EXPLORER, LEARNER, COMMUNICATOR, HEALER, COORDINATOR, SPECIALIST

**redifferentiate() (standalone function):**
- Applies role profile config to agent_config dict and mixin attributes
- Records role change in epigenome lineage
- Publishes `stem_cell.redifferentiated` on EventBus

**EventBus channels (all WIRED):**
| Channel | Publisher | Subscriber | Status |
|---------|-----------|------------|--------|
| `stem_cell.registered` | StemCellRegistry | Monitoring | WIRED |
| `stem_cell.redifferentiated` | StemCellRegistry | Monitoring | WIRED |

**Connections:**
| Source | Target | Type | Status |
|--------|--------|------|--------|
| StemCellRegistry | EventBus | pub/sub (stem_cell.registered, stem_cell.redifferentiated) | WIRED |

**Integrates with:** EventBus (publishes registration and redifferentiation events), HolonRegistry (registered as holon at Layer 21).

---

## Existing Agent Systems (no new connections)

BaseAgent, MycelialAgent, and the 10 mixins have no direct cross-system EventBus connections of their own. Their connections are mediated through the systems they compose (MemoryCoordinator, SignalBus, Stigmergy, etc.) -- see those modules' CONNECTIONS.md files.

---

## Related Modules

- [backbone/CONNECTIONS.md](../backbone/CONNECTIONS.md) -- EventBus channels, HolonProtocol, ConnectionRegistry
- [emergent/CONNECTIONS.md](../emergent/CONNECTIONS.md) -- SomaticMap tracks agent health
- [morphogenesis/CONNECTIONS.md](../morphogenesis/CONNECTIONS.md) -- OrganBuilder creates agents
