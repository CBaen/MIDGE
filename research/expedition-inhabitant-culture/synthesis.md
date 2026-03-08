# Expedition Synthesis: Inhabitant Culture
## Date: 2026-03-08
## Vetted by: Orchestrator
## Alignment: Checked against Research Brief

---

## High Confidence (all 4 teams + both validators converged)

### 1. The gap is wiring, not capability
Every team and both validators independently diagnosed the same root cause: MIDGE has the biological organs but they are not connected to behavior. The HomeostasisRegulator computes correction signals that almost nobody reads. The EndocrineSystem broadcasts hormones that have only 3 partial consumers. The Stigmergy and QuorumSpace objects are fully built and instantiated but agents don't read their gradients to decide what to investigate next.

**Implication:** The build phase is primarily wiring work — connecting existing systems — with 2 targeted new files.

### 2. Stigmergy and QuorumSpace are further along than anyone knew
Both validators confirmed: `StigmergicEnvironment` and `QuorumSpace` are FULLY IMPLEMENTED in `mae_core/communication/`, instantiated at bootstrap (`foundation.py` lines 131-132), and registered in SomaticMap. `StigmergyMixin` is part of every `MycelialAgent`. Team 2 searched the wrong directory and incorrectly reported them as absent. The actual gaps: (a) no evaporation step runs periodically, (b) agents don't read gradient maps when selecting investigation targets.

### 3. ResourceGovernor is NOT bootstrapped
Both validators caught this: ResourceGovernor exists as a class with tests, but is NOT instantiated in any bootstrap layer. Teams 1, 2, and 3 all built recommendations on top of it without verifying it's in the execution path. **This must be bootstrapped first before any governance wiring.**

### 4. EndocrineSystem already has partial behavioral wiring
Teams overstated the disconnection. Validator 2 found 3 existing behavioral consumers: ThreatDetector (cortisol), AutoHealer (cortisol), and SignalPriorityResolver (dopamine/melatonin modulating budget_per_step). The pattern for hormone → behavior coupling already exists and is tested. What's missing: ResourceGovernor, individual agent `_decide()` priority, and cultural norm weighting are NOT hormone-coupled.

### 5. OrganismState already subscribes to homeostasis corrections
Team 1 said "nothing subscribes" to homeostatic correction signals. Validator 1 found `OrganismState` does subscribe and tracks `_homeostasis_deviation`. The gap is narrower: this deviation doesn't feed into `get_reflex_override()` which is the existing gateway into `_decide()`. A surgical 5-10 line change activates this path.

---

## Battle-Tested Approaches (filtered for alignment)

### Homeostatic Reinforcement Learning (Team 1)
Drive = distance from setpoint. Reward = drive reduction. `HomeostasisRegulator` already computes the error signal. Adding `compute_drive_urgency()` → `OrganismState.get_reflex_override()` → `_decide()` is the shortest path to intrinsic drives. **Evidence:** Keramati & Gutkin (eLife 2014, peer-reviewed, widely cited), extended to continuous space in 2024.

### HPA Cascade as Self-Governance (Team 3)
GlobalWorkspace (Hypothalamus) → EndocrineSystem (Pituitary) → ResourceGovernor (Adrenal). Three EventBus subscriptions close a self-regulating governance loop with negative feedback. **Evidence:** Universal biological pattern, 50+ years of research. Mapped to existing MIDGE systems that need only coupling.

**Validator caveat:** Using GlobalWorkspace as the "Hypothalamus" creates an architectural coupling between the attention system and resource governance. This is a consequential design decision, not a trivial wire.

### DEB Priority Tiers on ResourceGovernor (Team 3)
MAINTENANCE (never throttled) → ACTIVE (protected) → EXPLORE (expendable under pressure). **Evidence:** Dynamic Energy Budget theory, applied to 1000+ species. Converts ResourceGovernor from "all sources equal" to biologically correct priority hierarchy.

### Two-Tier Scheduling Architecture (Team 4)
**Tier 1:** Mesa 3.5 `schedule_recurring` for step-relative cadences (replaces `step % N` branches). **Tier 2:** InhabitantScheduler daemon thread with heapq priority queue for wall-clock inhabitants. OctopusColony's monitoring thread is the proven prototype.

**Validator caveat:** MIDGE's `add_step_hook` is its own invention (not Mesa's). Mesa 3.5 interaction with this system is UNTESTED. Also, `sensing_hook.py` uses an internal `_step_counter`, not `model.steps` — migration is more complex than implied.

### Quorum Sensing as Confidence Amplifier (Team 2)
When 3+ independent quorum deposits land on the same ticker+direction, that IS collective confidence. Use quorum count as a multiplier on Thompson confidence, not a replacement. **Evidence:** 50+ years of biological quorum sensing research, applied to immune system density-threshold behaviors.

---

## Novel Approaches (filtered for feasibility)

### Learning Progress as Emergent Drive (Team 1 — IMGEP)
Drives emerge from capability gaps — templates with widest Clopper-Pearson confidence intervals have highest learning potential. A `LearningProgressMonitor` injects investigation goals toward under-explored patterns. This is MIDGE generating its own curiosity from within, not from prescribed rules.

**Filtered:** This borders on PatternArchaeology redesign (destructive boundary). Recommend implementing as a goal INJECTOR into existing TaskPool, not a modification of archaeology itself.

### History-Driven Role Differentiation (Team 2)
Deploy inhabitants identical, seed them differently, let interaction history differentiate them into specialists. After 50 energy-sector signals, an agent's epigenome shifts toward energy specialization — not by assignment but by experience. This IS Law 5 (Stem Cell) at the culture level.

### Cultural Transmission Through Senescence Bottlenecks (Team 2)
When an OctopusArm despawns, its findings are compressed and written to PatternLibrary. The arm's cultural contribution survives its death. Designed transmission bottlenecks create evolutionary pressure on which patterns are worth preserving.

**Validator caveat:** OrganBuilder does NOT listen to `CH_SENESCENT` — this lifecycle loop is not wired. Presenting it as existing is incorrect.

### Maximum Entropy Resource Allocation (Team 3)
Under budget pressure, reduce all sources proportionally rather than binary throttle. Then hard-throttle only sources furthest from their setpoint. More graceful degradation than current binary can_call() behavior.

---

## Emerging Approaches (filtered for relevance)

### Mesa 3.5 Agent Self-Scheduling
Agents call `self.model.schedule_event(self.act, after=interval)` to re-schedule themselves, becoming dormant between activations. ~50% runtime reduction in Mesa benchmarks. Eliminates wasted activation of inactive agents.

**Status:** Needs prototype verification with MIDGE's hook system before committing.

### Microbiome Strain Competition for Cultural Diversity
Already built: 5 competing strains with population dynamics and Shannon diversity monitoring. Missing: strain fitness is not connected to market outcomes. Connecting it to Thompson sampler results would make the microbial ecology evolve with market conditions.

---

## Synthesized Recommendation

### The Architecture: How 50+ Inhabitants Come to Life

**The architectural fork that no team resolved:** Should bio systems become Mesa agents or remain non-agent EventBus participants?

**Answer:** They remain non-agent systems. Making 29 bio systems into Mesa agents would require massive refactoring with unpredictable test impacts. Instead, bio systems gain independent schedules through InhabitantScheduler (wall-clock daemon) and emit EventBus messages that influence agent behavior. They become inhabitants with their own clocks without becoming Mesa agents.

**The 15 → 50+ path (no new entity types needed):**
- 29 bio systems with InhabitantScheduler schedules = 29 inhabitants
- 12 Mesa agents (existing) = 12 inhabitants
- 3-24 OctopusArms (auto-scaling, existing) = 3-24 inhabitants
- **Total: 44-65 inhabitants** from existing entity types

### Five-Layer Build Order

**Layer 0 — Foundation (prerequisite to everything):**
- Bootstrap ResourceGovernor into Layer 33
- This is the most critical finding: 3 of 4 teams built on a system not in the execution path

**Layer 1 — Drive-to-Action Coupling (highest leverage, all teams agree):**
- Wire HomeostasisRegulator deviation → `OrganismState.get_reflex_override()` → `_decide()` cascade
- Wire EndocrineSystem cortisol → ResourceGovernor budget tightening (extend existing SignalPriorityResolver pattern)
- Wire EndocrineSystem dopamine → agent exploration_bonus modulation (path already exists via `set_exploration_bonus()`)
- These are the "nervous connections" between organs and muscles

**Layer 2 — Cultural Coordination (activate what exists):**
- Add periodic evaporation step for StigmergicEnvironment (every 50 steps)
- Wire OctopusArm task selection to read Stigmergy gradient maps (ticker heat map)
- Wire quorum count as confidence multiplier on convergence alerts
- These make the pheromone trails and collective voting real

**Layer 3 — Scheduling Infrastructure (after Layer 1-2 prove value):**
- Prototype Mesa 3.5 compatibility with MIDGE's add_step_hook (MUST test before committing)
- Build InhabitantScheduler: heapq daemon thread dispatching to ThreadPoolExecutor (~50 lines)
- Migrate bio systems from "EventBus reactive" to "scheduled with own clock"
- These give inhabitants independent lives

**Layer 4 — Governance & Lifecycle:**
- GovernanceLogger: one new file, subscribes to governance channels, appends to JSONL
- Priority tiers on ResourceGovernor (MAINTENANCE/ACTIVE/EXPLORE)
- Wire SenescenceManager → OrganBuilder lifecycle loop (currently disconnected)
- Wire senescence wear clock to CircadianRhythm (don't penalize off-hours dormancy)
- These give the ecosystem self-regulation

### New Files Required (2 — both need Guiding Light approval)
1. `InhabitantScheduler` — generalization of OctopusColony monitoring pattern (~50 lines)
2. `GovernanceLogger` — append-only audit trail for governance events (~40 lines)

### What This Does NOT Change
- ConvergenceAlerter (min_domains=3, Thompson weighting, confidence formula)
- Thompson sampler (Bayesian distributions, forgetting cadence)
- PatternArchaeology (excavation, templates, fingerprints)
- The Mesa step loop (remains the organism's heartbeat)
- Existing test assertions (zero regressions)

---

## Disagreements

### Mesa 3.5 readiness
Team 3 says production-ready. Team 4 says untested with MIDGE's hooks. Both validators side with Team 4. **Resolution:** Prototype-first. Don't assume compatibility.

### Scope of "wiring"
Teams say "mostly wiring, 3 subscriptions." Validators say scope is larger than implied — ResourceGovernor not bootstrapped, OrganBuilder→Senescence not wired, 3 distinct endocrine coupling targets. **Resolution:** The work is wiring in nature but not small. Plan for a triadic-construction build with 4+ builders.

### Modulation-first vs. autonomous-first
Team 1 says start with drive modulation (safer). The Research Brief asks for autonomous wakeup. **Resolution:** Both are needed. Layer 1 (modulation) proves drives affect behavior. Layer 3 (scheduling) gives inhabitants independent clocks. Modulation first is the right build order, not the final state.

---

## Filtered Out

1. **SPADE-BDI framework adoption** (Team 1) — Rejected: introduces new dependency and programming paradigm. The BDI *pattern* (desire competition) is valuable and adopted via drive-weighted advisory.

2. **Ray actors** (Team 4) — Rejected: process-level isolation is overkill for single-machine. 50x memory overhead. External single point of failure.

3. **Pykka actor model** (Team 4) — Rejected: adds framework without adding capability over plain threads + EventBus.

4. **asyncio migration** (Team 4) — Rejected for now: requires migrating all sync clients to async. Right at 200+ inhabitants; overfitted for 50.

5. **Active Inference / FEP as implementation framework** (Team 1) — Filtered: theoretically beautiful but production implementations in continuous-space multi-agent Python are rare (2025). Adopt as conceptual framing, not implementation target.

6. **Team 2's "implement Stigmergy and QuorumSpace"** — Filtered: both already exist and are instantiated. Replaced with "activate evaporation + wire gradient reads."

7. **Team 3's GovernanceLogger as "the one new file"** — Filtered from emphasis: ResourceGovernor bootstrap is the actual prerequisite. GovernanceLogger is important but comes after Layer 0.

---

## Risks

1. **ResourceGovernor bootstrap could break tests** — it's a new system entering the execution path. Needs careful isolation in conftest.py.

2. **Endocrine → ResourceGovernor coupling blast radius** — SomaticMap's `analyze_blast_radius()` should be run before wiring. This creates a new cross-system dependency.

3. **Mesa 3.5 upgrade** — `add_step_hook` is MIDGE's invention, not Mesa's. The interaction with Mesa 3.5's event scheduler is unknown. Prototype before committing.

4. **`_decide()` cascade modification** — The cascade is the agent's decision-making spine. Adding homeostatic urgency before the advisory changes behavior for all agents. Needs comprehensive test coverage for the new path.

5. **Thread safety of InhabitantScheduler** — Bio systems calling EventBus from daemon threads is safe (RLock). But if they directly call Thompson sampler or ConvergenceAlerter, race conditions may emerge. All inter-system communication must go through EventBus.

6. **Cultural convergence to monoculture** — If all inhabitants imitate the currently-successful strategy, diversity collapses. Microbiome's Shannon diversity index is the health metric. CuriosityDrive is the biological dissenter that introduces novelty.
