# Build Brief: Inhabitant Culture Activation
## Date: 2026-03-08
## Project: MIDGE
## Source: Expedition synthesis at `research/expedition-inhabitant-culture/synthesis.md`

### Goal

Transform MIDGE's biological systems from passive EventBus subscribers into active inhabitants with intrinsic drives, self-governance, independent schedules, and emergent cultural coordination. The organism becomes an ecosystem where 50+ entities live their own lives.

### What's Already Done (since synthesis was written)

The Bio-Market Activation and Wire the Octopus builds completed significant groundwork:
- 29 bio systems wired to market EventBus channels (bio_market_wiring.py + bio_market_wiring_extended.py)
- OctopusColony bootstrapped with market task handlers (investigate_partial, archaeology_lookup, situation_check)
- Partial convergence emission from ConvergenceAlerter
- Market → PatternBus bridge (MarketConvergenceTranslator, MarketPartialTranslator)
- CuriosityDrive endocrine wire fixed (set_exploration_bonus())
- Stigmergy deposits for convergence/predictions (but NO periodic evaporation)
- QuorumSpace deposits for convergence/stacks (but NOT used as confidence multiplier)

### What This Build Does (the remaining gaps)

**Layer 0 — Bootstrap ResourceGovernor:**
ResourceGovernor exists as a class (`mae_core/market/resource_governor.py`, 194 lines) with tests (`tests/test_resource_governor.py`) but is NOT in any bootstrap module. Zero references in any bootstrap file. `ctx.resource_governor` does not exist at runtime. Must be instantiated and registered before endocrine coupling can work.

**Layer 1 — Drive-to-Action Coupling:**
Market signals reach bio systems (done). But bio system outputs don't reach agent decisions:
- `HomeostasisRegulator` computes urgency via private `_compute_urgency()` — needs public `compute_drive_urgency()` method
- `OrganismState` tracks `_homeostasis_deviation` (line 161) but `get_reflex_override()` (line 530) ignores it — checks only pain/stability/oxygen/energy/toxin
- `EndocrineSystem` cortisol should tighten ResourceGovernor budget — needs `register_resource_governor()` consumer method

**Layer 2 — Cultural Coordination:**
Stigmergy and QuorumSpace exist and receive deposits, but outputs aren't used:
- Stigmergy evaporation is lazy (`_apply_decay()` only fires on `sense_markers()` call) — no periodic cleanup hook
- `OctopusColony.submit_task()` (line 256) routes by workload only — ignores `stigmergy.get_gradient()` entirely
- Quorum contributor count not used as confidence multiplier on convergence alerts

**Layer 3 — InhabitantScheduler (new file, approved):**
Bio systems are called reactively by EventBus. They need independent wall-clock schedules (heapq daemon thread dispatching to ThreadPoolExecutor) — generalization of OctopusColony's monitoring pattern.

**Layer 4 — GovernanceLogger (new file, approved) + Priority Tiers + Lifecycle:**
- GovernanceLogger: append-only JSONL audit trail for governance events
- ResourceGovernor priority tiers: MAINTENANCE (never throttled) → ACTIVE (protected) → EXPLORE (expendable under pressure)
- SenescenceManager → OrganBuilder lifecycle: OrganBuilder has zero EventBus integration (no event_bus param, no subscriptions). When a system hits wear >= 1.0, SenescenceManager publishes CH_SENESCENT — but nobody rebuilds.

### Build Tasks

**Round 1 (no dependencies — ALL 6 builders in parallel):**

1. **B1 Foundation**: Bootstrap ResourceGovernor into Layer 33 (instantiate, holon, somatic, triadic connections, attrs list, systems dict)
2. **B2 Drive Architect**: Add `compute_drive_urgency()` to HomeostasisRegulator + wire `_homeostasis_deviation` into `get_reflex_override()` in OrganismState
3. **B3 Stigmergy & Octopus**: Periodic stigmergy evaporation step hook + OctopusColony gradient-based task routing
4. **B4 Quorum & Convergence**: Wire quorum contributor count as confidence multiplier on convergence alerts
5. **B5 New Systems**: Create InhabitantScheduler (~50 lines) and GovernanceLogger (~40 lines) as standalone classes with tests
6. **B6 Lifecycle**: Wire OrganBuilder to CH_SENESCENT (add event_bus, subscribe to senescence channel, trigger rebuild/prune on senescent events)

**Round 2 (depends on Round 1 — 2 builders in parallel):**

7. **B1 Foundation**: Add priority tiers to ResourceGovernor + wire endocrine→RG call in bio_market_wiring.py + bootstrap InhabitantScheduler and GovernanceLogger (from B5's files)
8. **B2 Drive Architect**: Add `register_resource_governor()` consumer method to EndocrineSystem (cortisol → budget tightening)

### Team Size: 6 builders + 3 reviewers = 9 agents

Why 6 builders: Maximum parallelism. Each layer gets its own specialist. 6 builders in Round 1 means all independent work completes simultaneously. Only 2 builders needed in Round 2 (ResourceGovernor-dependent work only).

Why 3 reviewers: Major feature set touching agent decision spine (get_reflex_override), thread safety (daemon scheduler), cross-system governance, and a new EventBus subscriber on a lifecycle-critical channel. 3 independent reviewers for consensus (Law 7).

### Builder Assignments

| Builder | Domain | Files Owned | Round |
|---------|--------|-------------|-------|
| B1: Foundation | Bootstrap + resource_governor | `bootstrap/market_systems.py`, `bootstrap/market.py`, `bootstrap/market_registration.py`, `bootstrap/market_connections.py`, `bootstrap/bio_market_wiring.py`, `bootstrap/foundation.py`, `main.py`, `market/resource_governor.py`, `tests/test_integration.py` | 1 + 2 |
| B2: Drive Architect | Coordination | `coordination/homeostasis.py`, `coordination/organism_state.py`, `coordination/endocrine_system.py`, `tests/test_drive_coupling.py` (new) | 1 + 2 |
| B3: Stigmergy & Octopus | Hooks + network | `bootstrap/market_hooks.py`, `network/octopus_colony.py`, `tests/test_cultural_coordination.py` (new) | 1 only |
| B4: Quorum & Convergence | Alerter | `market/intelligence/convergence_alerter.py`, `tests/test_quorum_confidence.py` (new) | 1 only |
| B5: New Systems | Scheduling + governance | `scheduling/__init__.py` (new pkg), `scheduling/inhabitant_scheduler.py` (new), `governance/__init__.py` (new pkg), `governance/governance_logger.py` (new), `tests/test_inhabitant_scheduler.py` (new), `tests/test_governance_logger.py` (new) | 1 only |
| B6: Lifecycle | Morphogenesis | `morphogenesis/organ_builder.py`, `tests/test_senescence_lifecycle.py` (new) | 1 only |

### Round Structure

**Round 1** (B1, B2, B3, B4, B5, B6 — all 6 in parallel):
- B1: ResourceGovernor bootstrap (instantiate in market_systems.py, add to market.py attrs, register holon/somatic in market_registration.py, add triadic connections in market_connections.py, add to systems dict in main.py)
- B2: HomeostasisRegulator `compute_drive_urgency()` public method returning dict of {parameter: urgency} + OrganismState `get_reflex_override()` extended with homeostasis check (Priority 6: high homeostasis deviation → "rest")
- B3: Stigmergy evaporation every 50 steps in market_hooks.py + OctopusColony gradient routing (read `stigmergy.get_gradient()` + `quorum_space.get_concentration()` in task selection)
- B4: Quorum confidence multiplier — when `quorum_space.get_contributor_count(f"{direction}:{ticker}")` >= 3, multiply convergence confidence (follow existing pattern from SignalPriorityResolver dopamine/melatonin modulation)
- B5: Create InhabitantScheduler (heapq priority queue, daemon thread, ThreadPoolExecutor dispatch, register/unregister/reschedule interface) + GovernanceLogger (EventBus subscriber for governance channels, append-only JSONL at `data/market/governance_log.jsonl`)
- B6: Add `event_bus` parameter to OrganBuilder.__init__(), subscribe to `emergent.system_senescent` (CH_SENESCENT), on senescent event → call `prune_organs()` + publish rebuild request

No dependencies — all 6 can work simultaneously.

**Round 2** (B1, B2 — 2 in parallel, after Round 1):
- B1: Add DEB priority tiers enum (MAINTENANCE/ACTIVE/EXPLORE) to ResourceGovernor with `set_source_tier()` + `can_call()` respects tiers. Wire `ctx.endocrine.register_resource_governor(ctx.resource_governor)` in bio_market_wiring.py. Bootstrap InhabitantScheduler + GovernanceLogger from B5's files (instantiate in foundation.py, register in market_registration.py, add to market.py attrs, add to main.py systems dict).
- B2: Add `register_resource_governor(resource_governor)` consumer method to EndocrineSystem — cortisol > 0.6 → `resource_governor.tighten_budgets(factor)`, cortisol < 0.3 → `resource_governor.relax_budgets(factor)`. Follow existing pattern from `register_threat_detector()`.

Why this ordering: B1 Round 2 needs ResourceGovernor to exist (from B1 Round 1). B2 Round 2 needs EndocrineSystem consumer pattern (same file they edited in Round 1) + ResourceGovernor class (exists, just needs the priority tier interface from B1 Round 2). Interface coordination: B2 writes `register_resource_governor(rg)` method, B1 writes the call `ctx.endocrine.register_resource_governor(ctx.resource_governor)` in bootstrap. Both follow the Build Brief spec.

### Project Constraints

From MIDGE CLAUDE.md:
- **Law 1 (No Bare Dyads):** ResourceGovernor, InhabitantScheduler, GovernanceLogger all need triadic connections with witnesses
- **Law 3 (Holon Protocol):** New bootstrapped systems need HolonProxy registration (10 capabilities at system scale)
- **Law 6 (Autopoietic Closure):** InhabitantScheduler must be self-contained — organism schedules itself, no external cron
- **Law 7 (Rule of 3/5):** Odd validator counts minimum 3
- **Zero regressions:** All 4,536+ existing tests must pass
- **No monoliths:** New files under 500 lines. `market_hooks.py` is 1,480 lines — minimal additions only (evaporation step is ~5 lines)
- **Advisory enforcement:** Drives modulate decisions, never block them. `get_reflex_override()` returns suggestions, not commands
- **Document parity:** System counts, connection counts, holon counts must be updated across all tracking files
- **Existing interfaces preserved:** `convergence_alerter.py` fire conditions unchanged (min_domains=3, same Thompson weighting). `OctopusColony` existing `submit_task()` still works for non-gradient tasks. `EndocrineSystem` existing consumers unaffected.

### Key Technical Details for Builders

- **ResourceGovernor** (`mae_core/market/resource_governor.py`): 194 lines. Constructor takes `event_bus`. Has `register_source()`, `can_call()`, `record_call()`, `_publish_throttle()`. Publishes `market.resource.throttle` and `market.resource.budget_warning`.
- **HomeostasisRegulator** (`mae_core/coordination/homeostasis.py`): 390 lines. Private `_compute_urgency()` exists. Emits per-parameter corrections on `coordination.homeostasis_correction`. Subscribes to `endocrine.state_update`.
- **OrganismState** (`mae_core/coordination/organism_state.py`): 791 lines. `get_reflex_override()` at line 530 — returns "rest"/"explore"/None. `_homeostasis_deviation` at line 161, updated from homeostasis corrections at line 287. Called from `lifecycle_decision.py:106`.
- **EndocrineSystem** (`mae_core/coordination/endocrine_system.py`): 645 lines. 9 existing `register_*()` consumer methods. Pattern: register method stores callback, `_on_hormone_update()` calls all registered consumers when hormone state changes.
- **StigmergicEnvironment** (`mae_core/communication/stigmergy.py`): 215 lines. `get_gradient(position, marker_type, radius)` returns normalized direction tuple. `_apply_decay()` is private, called lazily in `sense_markers()`. Bootstrap: `ctx.stigmergy = StigmergicEnvironment()` at foundation.py:131.
- **QuorumSpace** (`mae_core/communication/quorum_space.py`): 211 lines. `get_contributor_count(signal_type)` returns int. `get_concentration(signal_type)` returns float. Bootstrap: `ctx.quorum_space = QuorumSpace()` at foundation.py:132.
- **OctopusColony** (`mae_core/network/octopus_colony.py`): 474 lines. `submit_task()` at line 256 — workload-only routing. Background `_monitoring_loop()` on 5-second interval.
- **OrganBuilder** (`mae_core/morphogenesis/organ_builder.py`): 546 lines. No event_bus parameter. No subscriptions. `prune_organs()` exists. `CH_SENESCENT = "emergent.system_senescent"`.
- **ConvergenceAlerter** (`mae_core/market/intelligence/convergence_alerter.py`): event_bus is `self._bus` (line 262). Confidence formula at `_compute_confidence()`.
- **bio_market_wiring.py** (`mae_core/bootstrap/bio_market_wiring.py`): Contains endocrine consumer registration calls. Pattern: `ctx.endocrine.register_X(ctx.X)`.
- **market_hooks.py** (`mae_core/bootstrap/market_hooks.py`): 1,480 lines. Step hooks at 1/10/20/50/200/500 step intervals. Step-20 block at line 702 (OctopusColony coordination).

### Verification Plan

```bash
# 1. New tests
python -m pytest tests/test_drive_coupling.py tests/test_cultural_coordination.py tests/test_quorum_confidence.py tests/test_inhabitant_scheduler.py tests/test_governance_logger.py tests/test_senescence_lifecycle.py -v

# 2. Existing tests that could regress
python -m pytest tests/test_resource_governor.py tests/test_convergence_alerter.py tests/test_integration.py tests/test_bio_market_wiring.py -v

# 3. Full test suite (zero regressions)
python -m pytest tests/ -q

# 4. Smoke test
python main.py --agents 3 --steps 30

# 5. Verify drive activation (homeostasis urgency in decision context)
python main.py --agents 3 --steps 30 2>&1 | grep -i "reflex\|urgency\|drive"

# 6. Verify stigmergy evaporation
python main.py --agents 3 --steps 100 2>&1 | grep -i "stigmerg\|evaporat\|decay"
```

### Document Parity

| File | Field | Expected Change |
|------|-------|-----------------|
| `CLAUDE.md` | Systems count | 149 → 152 (+InhabitantScheduler, +GovernanceLogger, +ResourceGovernor bootstrapped) |
| `CLAUDE.md` | Connections | 428 → 437 (+9 new triadic connections: 3 per new system) |
| `HANDOFF.md` | Stats, current state | Update all counts + add inhabitant culture status |
| `mae_core/CONNECTIONS.md` | Connection index | Add Groups 35-37 |
| `tests/test_integration.py` | expected_keys | Add "resource_governor", "inhabitant_scheduler", "governance_logger" |
| `main.py` | systems_dict | Add new systems |
