# Build Brief: Inhabitant Culture Activation
## Date: 2026-03-08
## Project: MIDGE
## Source: Expedition synthesis at `research/expedition-inhabitant-culture/synthesis.md`

### Goal

Transform MIDGE's biological systems from passive EventBus subscribers into active inhabitants with intrinsic drives, self-governance, independent schedules, and emergent cultural coordination. The organism becomes an ecosystem where 50+ entities live their own lives.

### What's Already Done (since synthesis was written)

The Bio-Market Activation and Wire the Octopus builds completed significant groundwork:
- 29 bio systems wired to market EventBus channels (bio_market_wiring.py + bio_market_wiring_extended.py)
- OctopusColony bootstrapped with market task handlers
- Partial convergence emission from ConvergenceAlerter
- Market → PatternBus bridge (translators)
- CuriosityDrive endocrine wire fixed (set_exploration_bonus())
- Stigmergy deposits for convergence/predictions (but NO periodic evaporation)
- QuorumSpace deposits for convergence/stacks (but NOT used as confidence multiplier)

### What This Build Does (the remaining gaps)

**Layer 0 — Bootstrap ResourceGovernor:**
ResourceGovernor exists as a class with tests but is NOT in any bootstrap module. Zero references in any bootstrap file. Must be instantiated and registered before endocrine coupling can work.

**Layer 1 — Drive-to-Action Coupling:**
Market signals reach bio systems (done). But bio system outputs don't reach agent decisions. The gap:
- HomeostasisRegulator computes urgency → nobody reads it for decisions
- OrganismState tracks `_homeostasis_deviation` → but `get_reflex_override()` ignores it
- EndocrineSystem cortisol → ResourceGovernor budget tightening (not wired — ResourceGovernor not bootstrapped)

**Layer 2 — Cultural Coordination:**
Stigmergy and QuorumSpace exist and receive deposits. But:
- Stigmergy evaporation is lazy (only on sense_markers() call) — no periodic cleanup
- OctopusColony task selection is workload-only — ignores stigmergy gradients entirely
- Quorum count not used as confidence multiplier on convergence alerts

**Layer 3 — InhabitantScheduler (new file, approved):**
Bio systems are called reactively by EventBus. They need independent wall-clock schedules to become true inhabitants with their own lives.

**Layer 4 — GovernanceLogger (new file, approved) + Priority Tiers + Lifecycle:**
- GovernanceLogger: audit trail for governance events (append-only JSONL)
- ResourceGovernor priority tiers: MAINTENANCE (never throttled) → ACTIVE (protected) → EXPLORE (expendable)
- SenescenceManager → OrganBuilder lifecycle loop: currently disconnected (OrganBuilder has zero EventBus integration)

### Build Tasks

**Round 1 (no dependencies — all builders parallel):**

1. Bootstrap ResourceGovernor into Layer 33 (market_systems.py, market.py, market_registration.py, market_connections.py, main.py)
2. Add `compute_drive_urgency()` to HomeostasisRegulator; wire `_homeostasis_deviation` into `get_reflex_override()` in OrganismState
3. Add periodic stigmergy evaporation (every 50 steps in market_hooks.py); wire OctopusColony gradient-based task routing; wire quorum count as confidence multiplier on convergence alerts

**Round 2 (depends on Round 1 — all builders parallel):**

4. Add priority tiers to ResourceGovernor (MAINTENANCE/ACTIVE/EXPLORE); add `register_resource_governor()` consumer to EndocrineSystem; wire cortisol → budget tightening in bootstrap
5. Create InhabitantScheduler — heapq daemon thread dispatching to ThreadPoolExecutor; bootstrap it
6. Create GovernanceLogger — EventBus subscriber, append-only JSONL; wire SenescenceManager → OrganBuilder lifecycle (add event_bus to OrganBuilder, subscribe to CH_SENESCENT); bootstrap both

### Team Size: 4 builders + 3 reviewers

Why 4 builders: 4 distinct file domains (bootstrap, coordination, communication/network, new systems). Tasks span 15+ files across 6 packages. Dependencies require 2 rounds.

Why 3 reviewers: Major feature set touching agent decision spine, thread safety (daemon scheduler), cross-system governance. 3 independent reviewers = odd count for consensus (Law 7).

### Builder Assignments

| Builder | Domain | Files Owned | Round |
|---------|--------|-------------|-------|
| B1: Foundation | Bootstrap files + resource_governor.py | `market_systems.py`, `market.py`, `market_registration.py`, `market_connections.py`, `main.py`, `resource_governor.py`, `bio_market_wiring.py`, `foundation.py` | 1 + 2 |
| B2: Drive Architect | Coordination files | `homeostasis.py`, `organism_state.py`, `endocrine_system.py`, `tests/test_drive_coupling.py` | 1 + 2 |
| B3: Cultural Coordinator | Hooks + network + alerter | `market_hooks.py`, `octopus_colony.py`, `convergence_alerter.py`, `tests/test_cultural_coordination.py` | 1 only |
| B4: New Systems | Scheduling + governance + morphogenesis | `scheduling/inhabitant_scheduler.py`, `governance/governance_logger.py`, `organ_builder.py`, `tests/test_inhabitant_scheduler.py`, `tests/test_governance_logger.py`, `tests/test_senescence_lifecycle.py` | 2 only |

### Round Structure

**Round 1** (B1, B2, B3 in parallel):
- B1: ResourceGovernor bootstrap (instantiate, register holon, add triadic connections, add to attrs list and systems dict)
- B2: HomeostasisRegulator `compute_drive_urgency()` public method + OrganismState `get_reflex_override()` homeostasis-aware activation
- B3: Stigmergy evaporation step hook + OctopusColony gradient routing + quorum confidence multiplier

Why this ordering: ResourceGovernor must exist in execution path before endocrine coupling (Round 2). Homeostasis drive and cultural coordination are independent of each other and of ResourceGovernor.

**Round 2** (B1, B2, B4 in parallel):
- B1: ResourceGovernor priority tiers + endocrine→RG wiring call in bio_market_wiring.py + bootstrap InhabitantScheduler + GovernanceLogger
- B2: EndocrineSystem `register_resource_governor()` consumer method (cortisol → budget tightening)
- B4: Create InhabitantScheduler (~50 lines) + GovernanceLogger (~40 lines) + OrganBuilder senescence wiring

Why this ordering: B1 and B2 coordinate on the endocrine→RG coupling (B2 creates the consumer method, B1 calls it from bootstrap). B4 creates the new files that B1 bootstraps.

### Project Constraints

From MIDGE CLAUDE.md:
- **Law 1 (No Bare Dyads):** ResourceGovernor needs triadic connections (3 pathways with witness)
- **Law 3 (Holon Protocol):** New systems need HolonProxy registration (10 capabilities)
- **Law 6 (Autopoietic Closure):** InhabitantScheduler must be self-contained (no external cron — organism schedules itself)
- **Law 7 (Rule of 3/5):** Odd validator counts. Minimum 3.
- **Zero regressions:** All 4,536+ existing tests must pass
- **No monoliths:** New files under 500 lines. market_hooks.py is already 1,480 lines — minimal additions only
- **Advisory enforcement:** Drives modulate decisions, never block them. `get_reflex_override()` returns suggestions, not commands
- **Document parity:** System counts, connection counts, holon counts must be updated everywhere

### Verification Plan

```bash
# 1. New tests
python -m pytest tests/test_drive_coupling.py tests/test_cultural_coordination.py tests/test_inhabitant_scheduler.py tests/test_governance_logger.py tests/test_senescence_lifecycle.py -v

# 2. Existing tests (zero regressions)
python -m pytest tests/test_resource_governor.py tests/test_convergence_alerter.py tests/test_integration.py -v

# 3. Full test suite
python -m pytest tests/ -q

# 4. Smoke test
python main.py --agents 3 --steps 30

# 5. Verify drive activation (homeostasis urgency should appear in decision context)
python main.py --agents 3 --steps 30 2>&1 | grep -i "reflex\|urgency\|drive"

# 6. Verify stigmergy evaporation (should log decay)
python main.py --agents 3 --steps 100 2>&1 | grep -i "stigmerg\|evaporat\|decay"
```

### Document Parity

| File | Field | Expected Change |
|------|-------|-----------------|
| `CLAUDE.md` | Systems count | 149 → 152 (+InhabitantScheduler, +GovernanceLogger, +ResourceGovernor bootstrapped) |
| `CLAUDE.md` | Connections | 428 → 434 (+6 new triadic connections) |
| `HANDOFF.md` | Stats, current state | Update all counts + add inhabitant culture status |
| `mae_core/CONNECTIONS.md` | Connection index | Add Group 35 (ResourceGovernor), Group 36 (InhabitantScheduler) |
| `tests/test_integration.py` | expected_keys | Add new system keys |
| `main.py` | systems_dict | Add new systems |
