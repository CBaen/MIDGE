# Team 3 Findings: Self-Governance Within Autopoietic Closure
## Date: 2026-03-08
## Researcher: Team Member 3

---

## Executive Summary

The core question is: how do you achieve Paperclip's governance features (budgets, heartbeats, accountability, org charts) purely from within MIDGE's existing biological systems, with no external platform? This research finds that MIDGE already contains — in nearly complete form — every biological primitive needed to construct a full self-governance layer. The gap is not capability; it is wiring and activation. Six existing systems, properly connected, become a governance layer that is biologically deeper and more correct than Paperclip ever could be.

---

## Framing: What "Self-Governance" Means Biologically

In biology, there is no CEO. The human body governs 37 trillion cells across 80+ years without a centralized control plane. The mechanisms it uses are:

1. **Hormonal signaling** (endocrine): Broadcast state-changes that all systems read and adapt to. Cortisol doesn't order cells to mobilize glucose — it creates a condition where every glucose-processing cell adjusts its own behavior.
2. **Negative feedback loops** (homeostasis): Each regulated parameter has a setpoint and a corrective signal. No overseer; each loop closes itself.
3. **Quorum sensing**: Collective behaviors emerge only when enough independent signals agree. No single cell decides; the population decides.
4. **Resource priority hierarchy** (DEB theory): Maintenance > Growth > Reproduction. This hierarchy is built into cellular metabolism, not enforced by any authority.
5. **Apoptosis / senescence**: Systems that have consumed resources without producing value self-terminate or signal for replacement. The organism-level benefit is automatic.
6. **Immune privilege**: Critical tissues (brain, eye, fetus) carry special markers that redirect immune responses away from them. Protection is declared from within, not granted from above.
7. **Hierarchical cascade** (HPA axis): Hypothalamus → Pituitary → Adrenal. A three-tier relay where each tier amplifies and shapes the signal, then provides negative feedback to suppress the tiers above it.

Every one of these mechanisms has a direct MIDGE equivalent — most already built.

---

## Internal Mapping: Each Paperclip Feature → Existing MIDGE System

| Paperclip Feature | Biological Mechanism | MIDGE System | Current State |
|-------------------|---------------------|--------------|---------------|
| Per-agent budget cap | DEB maintenance priority + resource allocation | `ResourceGovernor` | Built, wired to market APIs |
| Heartbeat scheduling | Circadian rhythm / ultradian pulses | `CircadianRhythm` + Mesa `schedule_recurring` | Step-based; needs wall-clock activation |
| Org chart / hierarchy | HPA axis cascade + holarchic containment | `HolonRegistry` + `HolonProxy` | Built; parent/child/peer awareness complete |
| Task queue (atomic work claiming) | Quorum sensing threshold + synaptic gating | `TriadEnforcer` voting + `EventBus` channels | Voting built; queue claiming not yet wired |
| Budget warnings (80% / 100%) | Hormone cascade thresholds | `ResourceGovernor` `warn_at=0.8` + `EndocrineSystem` cortisol | Warning built; endocrine coupling not wired |
| Append-only audit log | Episodic memory / engram formation | `EventBus` streams + existing JSONL logs | Stream persistence not yet permanent |
| Agent lifecycle (born/active/retired) | Morphogenesis + senescence + apoptosis | `OrganBuilder` + `SenescenceManager` | Both built; lifecycle not integrated |
| Division hierarchy | Organ-system containment | `HolonRegistry` parent/child | Built; needs population with inhabited divisions |
| Hard enforcement (pause at 100%) | Metabolic shutdown / inhibition | `ResourceGovernor` returns `False` + `InhibitionSystem` | `can_call()` blocks; inhibition not coupled |

The gap in every row is the same: the capability exists in isolation, but the cross-system coupling is not wired. Self-governance is not a new system — it is the activation of existing systems as a coordinated layer.

---

## Battle-Tested Approaches

### Approach 1: DEB-Style Resource Priority Hierarchy

- **What:** Dynamic Energy Budget theory (Kooijman 1986, applied to 1000+ species) establishes a universal biological resource allocation rule: Maintenance first, Growth second, Reproduction third. This is not a policy a CEO enforces; it is structurally embedded in cellular metabolism.
- **Evidence:** Applied successfully to over 1000 species in conservation, aquaculture, and ecotoxicology modeling. Wikipedia DEB article (accessed 2026-03-08). Conservation Physiology DEB paper, Oxford Academic (https://academic.oup.com/conphys/article/10/1/coac061/6701566). Springer DEB mathematical model paper (https://link.springer.com/article/10.1007/s002850000049).
- **Source:** Dynamic Energy Budget Theory Wikipedia (https://en.wikipedia.org/wiki/Dynamic_energy_budget_theory), accessed 2026-03-08.
- **Fits our case because:** ResourceGovernor already implements a fixed `hourly_limit` per source (maintenance cap). The missing piece is a three-tier priority queue: (1) critical system heartbeats get budget unconditionally, (2) active sensing gets budget from the remaining pool, (3) exploration gets whatever is left. This is a 10-line extension of `can_call()` — add a `priority` parameter with tiers MAINTENANCE | ACTIVE | EXPLORE.
- **Tradeoffs:** Fixed ratios may not adapt to regime changes. Needs coupling to EndocrineSystem cortisol to tighten ratios during stress.

### Approach 2: Homeostatic Setpoints as Internal Accountability

- **What:** Walter Cannon's homeostasis model (1929): every regulated parameter has a setpoint, a current value, and a corrective signal. The organism doesn't need an auditor; it continuously self-audits. StatPearls homeostasis documentation confirms: "homeostasis is the state of steady internal, physical, and chemical conditions maintained by living systems."
- **Evidence:** MIDGE already has `Homeostasis` in `mae_core/coordination/homeostasis.py` with 7 vital parameters including `processing_load` and `energy_level`. This is exactly the architecture. Source: StatPearls NCBI (https://www.ncbi.nlm.nih.gov/books/NBK559138/), accessed 2026-03-08.
- **Source:** MIDGE codebase `mae_core/coordination/homeostasis.py` (confirmed by direct read, 2026-03-08). StatPearls NCBI (https://www.ncbi.nlm.nih.gov/books/NBK559138/).
- **Fits our case because:** Add `api_call_rate` and `active_source_count` as Homeostasis setpoints. When the call rate drifts above the setpoint, Homeostasis publishes a correction signal on `coordination.homeostasis_correction`. ResourceGovernor subscribes and tightens limits. The system audits itself.
- **Tradeoffs:** PID-style correction can oscillate if gain is too high. Homeostasis setpoints need market-phase awareness (FOMC day vs normal day).

### Approach 3: HPA-Axis Cascade as Org Chart Without Hierarchy

- **What:** The Hypothalamic-Pituitary-Adrenal axis is a three-tier relay: Hypothalamus → Pituitary → Adrenal. Each tier amplifies the signal and provides negative feedback to the tier above. Wikipedia HPA axis (https://en.wikipedia.org/wiki/Hypothalamic%E2%80%93pituitary%E2%80%93adrenal_axis). Cleveland Clinic (https://my.clevelandclinic.org/health/body/hypothalamic-pituitary-adrenal-hpa-axis).
- **Evidence:** The HPA axis governs stress response, resource reallocation, and priority override in all vertebrates. PMC (https://pmc.ncbi.nlm.nih.gov/articles/PMC4867107/), accessed 2026-03-08.
- **Source:** Wikipedia HPA axis, accessed 2026-03-08. PMC HPA regulation paper, accessed 2026-03-08.
- **Fits our case because:** MIDGE's HolonRegistry already encodes the containment hierarchy (organism → organ → system → agent). The HPA cascade maps directly: GlobalWorkspace (Hypothalamus) publishes a priority override → EndocrineSystem (Pituitary) amplifies with cortisol/adrenaline → ResourceGovernor (Adrenal) tightens or expands per-source budgets. Negative feedback: when ResourceGovernor throttles sources, it publishes to a channel that GlobalWorkspace subscribes to, suppressing further escalation. This is an org chart that enforces itself through chemistry, not commands.
- **Tradeoffs:** Three-tier depth means latency in the response chain. For emergencies (Knight Capital kill switch), a direct fast-path bypass is needed alongside the cascade.

### Approach 4: Quorum Voting as Task Claiming

- **What:** Quorum sensing in bacteria and immune cells: a collective behavior activates only when enough independent signals agree. Nature Reviews Immunology (https://www.nature.com/articles/s41577-018-0040-4). Helper T cells transition from mixed to unified decisions only at high cell densities (PLoS Computational Biology, https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008051).
- **Evidence:** The immune quorum sensing research (Frontiers in Immunology, https://www.frontiersin.org/articles/472904) confirms that quorum-based density sensing regulates population size and synchronizes behavior. Applied to 2026-era AI: the same mechanism prevents duplicate work without a task queue coordinator.
- **Source:** Nature Reviews Immunology (https://www.nature.com/articles/s41577-018-0040-4), accessed 2026-03-08.
- **Fits our case because:** MIDGE's TriadEnforcer already implements majority-vote validation. The missing wire: before any system claims a market situation to investigate, it fires a vote on `triad.vote_complete`. If a quorum of TriadEnforcer validators has already recorded a claim on that ticker+direction combination, the new system receives a rejection and stands down. No central task queue needed — the claiming happens through witnessed consensus.
- **Tradeoffs:** TriadEnforcer votes are synchronous and lock-protected. High-frequency quorum checks (per-tick) would create contention. Needs a lightweight fast-path: check a `_claimed_situations: dict[str, str]` first, only go to full quorum vote on conflicts.

---

## Novel Approaches

### Approach 5: Senescence as Inhabitant Lifecycle Governance

- **What:** Biological senescence is not failure — it is programmed resource governance. The Hayflick limit, telomere shortening, and apoptosis are the organism's mechanism for retiring cells that have accumulated too much wear and replacing them with fresh ones. SenescenceManager already exists in MIDGE at `mae_core/emergent/senescence.py`.
- **Why it's interesting:** This is not a feature of any existing agent orchestration platform, including Paperclip. Paperclip can pause an agent. Senescence actively retires one, frees its resource allocation, and signals for a replacement. The difference is that retirement is driven by the agent's own wear accumulation, not an external admin.
- **Evidence:** PMC programmed cell death and aging paper (https://pmc.ncbi.nlm.nih.gov/articles/PMC4480161/), accessed 2026-03-08. The principle: idle systems age faster ("use it or lose it"). Active systems age slower. When wear exceeds threshold, `CH_SENESCENT` is published. MIDGE's SenescenceManager already implements this logic exactly.
- **Source:** PMC PCD and aging, accessed 2026-03-08. MIDGE codebase `mae_core/emergent/senescence.py`, read directly 2026-03-08.
- **Fits our case because:** Inhabitants that stop producing signal updates accumulate wear rapidly. After N steps of inactivity, the SenescenceManager publishes `CH_SENESCENT`. OrganBuilder listens and dissolves the organ. If the dissolved inhabitant filled a structural role, morphogenesis spawns a replacement. This creates a self-renewing inhabitant population without any manual lifecycle management.
- **Risks:** Senescence parameters need tuning. A market inhabitant that is dormant during off-hours should not be senescent — CircadianRhythm needs to pause the wear accumulation clock during REST phase. This coupling is not currently wired.

### Approach 6: Immune Privilege as Critical System Protection

- **What:** In biology, immune-privileged sites (brain, eye, testis, fetus) carry surface markers (MHC, FasL expression, anti-inflammatory cytokines) that redirect immune attack away from them. Critical tissues protect themselves by broadcasting their criticality — they don't wait for the immune system to find out.
- **Why it's interesting:** MIDGE's SomaticMap already has a `SystemCriticality.CRITICAL` level that marks certain systems for extra protection and auto-rejection of blast-radius modifications. But it doesn't yet broadcast this criticality into the resource allocation layer.
- **Evidence:** Conceptual aspects of self/nonself discrimination, PMC (https://pmc.ncbi.nlm.nih.gov/articles/PMC3136900/), accessed 2026-03-08. Computational convergence with AI systems, bioRxiv 2026 (https://www.biorxiv.org/content/10.64898/2026.02.03.703525v1.full).
- **Source:** PMC self/nonself discrimination paper, accessed 2026-03-08.
- **Fits our case because:** When ResourceGovernor approaches global budget limits, it should not proportionally reduce all sources — it should protect CRITICAL-marked systems (EventBus, Mesa step loop, ConvergenceAlerter) and preferentially throttle PERIPHERAL ones. SomaticMap's criticality registry already exists. ResourceGovernor needs to query it during budget pressure. One method call: `somatic_map.get_critical_path()` → exclude those source IDs from throttling.
- **Risks:** Immune privilege in biology creates blind spots — tumors exploit it. A system could mark itself CRITICAL to avoid resource governance. Mitigation: criticality can only be set during bootstrap (the 33-layer bootstrap already defines a sealed order), not at runtime.

### Approach 7: Maximum Entropy Resource Allocation as Fair Distribution

- **What:** A 2020 paper in Journal of Mathematical Biology (https://link.springer.com/article/10.1007/s00285-020-01499-6) shows that cellular resources are allocated among elementary flux modes according to the principle of maximum entropy — the most unbiased distribution consistent with constraints. This means under resource pressure, the cell allocates fractionally to many pathways rather than fully to one, maximizing optionality.
- **Why it's interesting:** Current ResourceGovernor uses hard per-source limits (binary: allowed or throttled). Maximum entropy allocation would reduce all sources proportionally under global pressure, then hard-throttle only those furthest from their setpoint. This is metabolically more correct and produces more graceful degradation.
- **Evidence:** Springer Journal of Mathematical Biology (https://link.springer.com/article/10.1007/s00285-020-01499-6). Also supported by the PMC metabolic objectives paper (https://pmc.ncbi.nlm.nih.gov/articles/PMC11857637/), both accessed 2026-03-08.
- **Source:** Springer J Math Biol, accessed 2026-03-08.
- **Fits our case because:** When `global_calls_last_hour` exceeds 80% of `global_hourly_limit`, ResourceGovernor enters proportional reduction mode: each source's `hourly_limit` is multiplied by `(1 - pressure_factor)`. When pressure clears, limits restore. This prevents any single source from exhausting the budget while others are completely silenced.
- **Risks:** Proportional reduction requires updating `SourceBudget.hourly_limit` dynamically — currently it's set once at registration. Needs a `_adjusted_limit` field that varies while `_base_limit` stays fixed. Adds complexity to `can_call()`.

---

## Emerging Approaches

### Approach 8: Mesa 3.5 `schedule_recurring` as Biological Clock Replacement

- **What:** Mesa 3.5.0 (released 2025, documented at https://mesa.readthedocs.io/latest/tutorials/3_event_scheduling.html) introduced a public event scheduling API with `model.schedule_recurring(func, Schedule(interval=N))` as a first-class mechanism. This unifies the step loop with discrete-event scheduling, allowing different inhabitants to run on different intervals within the same Mesa model.
- **Momentum:** Mesa 3.5.0 release notes on GitHub (https://github.com/mesa/mesa/releases/tag/v3.5.0) confirm this as a headline feature of the 3.5 release. Mesa documentation updated February 2026. This is production-ready.
- **Source:** Mesa 3.5 release notes (https://github.com/mesa/mesa/releases/tag/v3.5.0), Mesa docs (https://mesa.readthedocs.io/latest/tutorials/3_event_scheduling.html), both accessed 2026-03-08.
- **Fits our case because:** Each inhabitant type can have its own `Schedule(interval=N)`. The Market Sensing Inhabitant runs every 25 steps. The Excavation Inhabitant runs every 5000 steps. The Circadian Inhabitant runs every 1 step. All coordinated within Mesa's existing step loop — no external scheduler, no threads, no wall-clock conversion needed. This is Law 6 compliant: the scheduling is internal to the Mesa model.
- **Maturity risk:** Mesa 3.5 is current but its `schedule_recurring` API is new. MIDGE's test suite would need to cover the new scheduling paths. The 33-layer bootstrap would need to use `schedule_recurring` registrations at appropriate layers. Migration is non-trivial but well-documented.

### Approach 9: Governance-as-a-Service Pattern (Internal, Non-Invasive)

- **What:** A 2025 arXiv paper on Governance-as-a-Service (GaaS, https://arxiv.org/html/2508.18765v1) describes a "modular enforcement layer for agentic environments, functioning as a non-invasive runtime proxy that filters actions based on programmable rule specifications." Every enforcement decision is recorded with timestamp, agent identifier, rule ID, and trust state.
- **Momentum:** arXiv 2025, gaining traction in enterprise AI governance discussions. Not yet production-widely-deployed but the pattern itself is well-documented.
- **Source:** arXiv GaaS paper (https://arxiv.org/html/2508.18765v1), accessed 2026-03-08.
- **Fits our case because:** MIDGE's ResourceGovernor + EventBus + SomaticMap is already a GaaS implementation, just not labeled as such. The `_publish_throttle()` method generates audit records. The `register_callback()` mechanism provides non-invasive wiring. The pattern confirms MIDGE's existing architecture is on a well-validated trajectory.
- **Maturity risk:** arXiv means not yet peer-reviewed. But the pattern (non-invasive enforcement proxy + audit records) maps directly to what MIDGE already does. This is validation, not dependency.

---

## Gaps and Unknowns

1. **The endocrine-to-resource coupling is not wired.** EndocrineSystem publishes `CH_HORMONE_RELEASE`. ResourceGovernor does not subscribe. This is the most important missing connection: cortisol should tighten budgets, adrenaline should grant temporary emergency bypass, melatonin should suspend non-critical sensing during REST. Without this coupling, the hormonal system has no behavioral effect on resource governance.

2. **Senescence wear clock has no market-phase awareness.** SenescenceManager accumulates wear per step regardless of CircadianPhase. A sensing inhabitant that sleeps during market-close should not accumulate wear at the same rate as one that is truly idle during open hours. The coupling to CircadianRhythm's `get_activity_multiplier()` is documented but not implemented.

3. **HolonRegistry hierarchy exists but is not populated as an org chart.** The registry can answer "who is parent of X" and "who are peers of X" — but currently all market systems are registered under flat parent IDs from the bootstrap. The three-division structure (Sensing/Analysis/Protection) from the Paperclip proposal would need to be encoded as parent holons with children registered under them. This is a data population problem, not an architecture problem.

4. **There is no persistent append-only governance audit log.** EventBus streams exist in memory (maxlen=10000, deque-based). There is no path from `CH_RESOURCE_THROTTLE`, `CH_TRIAD_VIOLATION`, `CH_SENESCENT`, or `CH_MODIFICATION_PROPOSED` to a durable JSONL file. This is the one gap that does not have an existing system to plug into — a `GovernanceLogger` that subscribes to all governance channels and appends to `data/market/governance_audit.jsonl` would need to be built.

5. **ResourceGovernor has no priority tiers.** `can_call()` treats all sources equally. Under pressure, critical sources (ConvergenceAlerter's price fetches) and peripheral sources (Google Trends) compete equally. This is not biologically correct and could cause critical signals to be throttled while trivial ones proceed.

6. **The blast radius of wiring endocrine → resource is unknown.** EndocrineSystem currently has no subscribers that change operational behavior — it only publishes. Adding ResourceGovernor as a subscriber to cortisol events would create a new cross-system dependency. SomaticMap's `analyze_blast_radius()` should be run on `endocrine_system` before wiring to confirm the risk score.

---

## Synthesis

### The Strongest Approach

The strongest single approach is the **HPA cascade mapping**: GlobalWorkspace (Hypothalamus) → EndocrineSystem (Pituitary) → ResourceGovernor (Adrenal). This is not a new architecture — it is identifying that these three systems are already in the correct relationship and need three EventBus subscriptions to complete the circuit.

- GlobalWorkspace publishes `market.attention.situation_claimed` → EndocrineSystem releases dopamine (exploration signal).
- EndocrineSystem releases cortisol when `market.resource.budget_warning` fires → ResourceGovernor tightens per-source limits.
- ResourceGovernor publishes `market.resource.throttle` → EndocrineSystem reduces cortisol (negative feedback, closes the loop).

Three subscriptions. No new files. Complete self-governance.

### The Combination That Works Best

A four-layer internal governance stack, all from existing systems:

**Layer 1 — Resource Allocation (DEB priority hierarchy):** ResourceGovernor gains three priority tiers (MAINTENANCE / ACTIVE / EXPLORE). Critical system heartbeats are MAINTENANCE and never throttled. Market sensing is ACTIVE. Speculative excavation is EXPLORE. Under budget pressure, EXPLORE is throttled first, then ACTIVE, MAINTENANCE never.

**Layer 2 — Accountability (homeostasis setpoints):** `api_call_rate` and `active_inhabitant_count` become Homeostasis setpoints. Deviation from setpoint triggers a corrective signal on `coordination.homeostasis_correction`. ResourceGovernor subscribes and adjusts limits. The organism audits itself continuously.

**Layer 3 — Lifecycle (senescence + morphogenesis):** Inhabitants accumulate wear via SenescenceManager. Worn inhabitants publish `CH_SENESCENT`. OrganBuilder listens and dissolves. Morphogenesis spawns replacements when structural gaps emerge. Residents are born, work, age, and retire without manual management.

**Layer 4 — Audit trail (GovernanceLogger):** One new file — `mae_core/governance/governance_logger.py`. Subscribes to governance channels (`CH_RESOURCE_THROTTLE`, `CH_TRIAD_VIOLATION`, `CH_SENESCENT`, `CH_MODIFICATION_PROPOSED`, `CH_AWARENESS_ANOMALY`). Appends to `data/market/governance_audit.jsonl`. Every governance event is timestamped and durable. This is the one genuinely new system needed.

### What the Orchestrator Needs to Know

1. **Paperclip's features are all achievable internally, but the GovernanceLogger is the one new file needed.** Everything else is wiring existing systems. The orchestrator should plan for that addition.

2. **The most critical missing wire is endocrine → resource coupling.** Without it, cortisol has no behavioral effect. This should be the first connection made, as it activates the largest number of downstream behaviors.

3. **Mesa 3.5's `schedule_recurring` is the right mechanism for inhabitant heartbeats.** It keeps everything inside the Mesa step loop (Law 6 compliant), avoids threads, and provides per-inhabitant interval control. The orchestrator should recommend upgrading MIDGE to Mesa 3.5 if not already on it, and using `schedule_recurring` for all inhabitant step registration.

4. **The HolonRegistry parent/child hierarchy needs to be populated with the division structure.** The data exists (157 registered holons) but the three-division org chart is not reflected in parent_id assignments. Populating this is a bootstrap change, not a code change — it belongs in `mae_core/bootstrap/market.py` around Layer 33.

5. **Priority-tiered ResourceGovernor is the highest-leverage single change.** It converts resource governance from "all sources are equal" to "critical systems are immune, active sensing is protected, speculative work is expendable." This is biologically correct and directly addresses the Paperclip budget enforcement requirement.

---

## Source List

- Dynamic Energy Budget Theory — Wikipedia (https://en.wikipedia.org/wiki/Dynamic_energy_budget_theory), accessed 2026-03-08
- Conservation Physiology DEB Paper — Oxford Academic (https://academic.oup.com/conphys/article/10/1/coac061/6701566), accessed 2026-03-08
- Dynamic Metabolic Resource Allocation (Maximum Entropy) — Springer (https://link.springer.com/article/10.1007/s00285-020-01499-6), accessed 2026-03-08
- Metabolic Objectives and Trade-Offs — PMC 2025 (https://pmc.ncbi.nlm.nih.gov/articles/PMC11857637/), accessed 2026-03-08
- Physiology, Homeostasis — StatPearls/NCBI (https://www.ncbi.nlm.nih.gov/books/NBK559138/), accessed 2026-03-08
- Neuro-endocrine-immune regulation of metabolic homeostasis — ScienceDirect 2025 (https://www.sciencedirect.com/science/article/abs/pii/S1359610125000887), accessed 2026-03-08
- Quorum Sensing in the Immune System — Nature Reviews Immunology (https://www.nature.com/articles/s41577-018-0040-4), accessed 2026-03-08
- Quorum Sensing by Monocyte-Derived Populations — Frontiers in Immunology (https://www.frontiersin.org/articles/472904), accessed 2026-03-08
- Helper T Cells and Quorum Sensing — PLoS Computational Biology (https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008051), accessed 2026-03-08
- HPA Axis — Wikipedia (https://en.wikipedia.org/wiki/Hypothalamic%E2%80%93pituitary%E2%80%93adrenal_axis), accessed 2026-03-08
- HPA Axis Regulation — PMC (https://pmc.ncbi.nlm.nih.gov/articles/PMC4867107/), accessed 2026-03-08
- HPA Axis Overview — Cleveland Clinic (https://my.clevelandclinic.org/health/body/hypothalamic-pituitary-adrenal-hpa-axis), accessed 2026-03-08
- Programmed Cell Death and Aging — PMC (https://pmc.ncbi.nlm.nih.gov/articles/PMC4480161/), accessed 2026-03-08
- Conceptual Aspects of Self/Nonself Discrimination — PMC (https://pmc.ncbi.nlm.nih.gov/articles/PMC3136900/), accessed 2026-03-08
- Computational Convergence of Adaptive Immunity and AI — bioRxiv 2026 (https://www.biorxiv.org/content/10.64898/2026.02.03.703525v1.full), accessed 2026-03-08
- Mesa 3.5 Event Scheduling API — Mesa Docs (https://mesa.readthedocs.io/latest/tutorials/3_event_scheduling.html), accessed 2026-03-08
- Mesa 3.5 Release Notes — GitHub (https://github.com/mesa/mesa/releases/tag/v3.5.0), accessed 2026-03-08
- Governance-as-a-Service Multi-Agent Framework — arXiv 2025 (https://arxiv.org/html/2508.18765v1), accessed 2026-03-08
- Audit Trails for AI Agents — Adopt AI (https://www.adopt.ai/glossary/audit-trails-for-agents), accessed 2026-03-08
- Pykka Actor Model — Pykka Docs 4.2.0 (https://pykka.readthedocs.io/stable/), accessed 2026-03-08
