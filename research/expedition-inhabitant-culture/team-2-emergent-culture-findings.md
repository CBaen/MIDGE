# Team 2 Findings: Emergent Culture & Coordination
## Date: 2026-03-08
## Researcher: Team Member 2

---

## Preamble: What MIDGE Already Has (Internal Audit)

The research brief mentions systems that do not yet exist as standalone files (stigmergy.py, quorum_space.py, collective_dream.py, global_workspace.py are absent from the codebase). However, their functionality is partially wired through `mae_core/bootstrap/bio_market_wiring.py`, which implements EventBus callbacks that simulate stigmergic trail deposition and quorum signal accumulation. The existing `Stigmergy` and `QuorumSpace` objects are referenced in the bootstrap but their source files are not present in `mae_core/emergent/` — suggesting these are either in a package not yet committed or are stub objects on `ctx`. This is confirmed by `bio_market_wiring.py` gracefully returning 0 when `ctx.stigmergy` or `ctx.quorum_space` is `None`.

**What exists for culture/coordination:**

| System | Location | Current State | Missing |
|--------|----------|---------------|---------|
| Stigmergy wiring | `bio_market_wiring.py:_wire_stigmergy()` | Deposits `convergence.{direction}` and SUCCESS/DANGER markers via `deposit_marker()` on ticker positions | `stigmergy` object itself may not exist; no agent reads the trails |
| QuorumSpace wiring | `bio_market_wiring.py:_wire_quorum()` | Deposits convergence, pattern stack, and dual-confirmation signals via `deposit_signal()` | `quorum_space` object may not exist; quorum count never reaches agents |
| EndocrineSystem | `mae_core/coordination/endocrine_system.py` | Full hormonal cascade: 6 hormones, decay rates, cascade effects, 7 consumer registration methods | Hormones published on EventBus but nothing changes agent behavior — Team 1's finding |
| Microbiome | `mae_core/emergent/microbiome.py` | 5 competing strains with population dynamics, Shannon diversity, dysbiosis detection | Fully self-contained but not wired to market signals; strains compete on processing load only |
| OctopusColony | `mae_core/network/octopus_colony.py` | P2P colony with auto-scaling 3-10, health monitoring, ring+cross topology | Colony exists; specialization is `GENERAL` for all members — no role differentiation |

**The diagnostic for culture specifically:** MIDGE has the biological substrate for collective behavior but not the coordination layer that converts parallel activity into genuine collective intelligence. The EndocrineSystem is the closest thing to a cultural medium — hormones broadcast global state — but agents are not reading it to modulate their behavior. The OctopusColony has peer topology but all octopuses do the same work. There are no cultural markers, no role traditions, no collective memory of "how we do things here."

**What Team 1 and Team 3 cover (to avoid overlap):**
- Team 1: Drive-to-action coupling — how individual agents get intrinsic motivation. NOT covered here.
- Team 3: Self-governance through biological resource allocation — budgets, heartbeats, org charts. NOT covered here.
- Team 2 (this document): How agents develop *shared norms*, *role differentiation*, *collective memory*, and *emergent group intelligence* — the cultural layer that makes 50 agents collectively smarter than any single agent.

---

## Battle-Tested Approaches

### Approach 1: Stigmergic Trail Coordination (Evaporating Gradient Field)

- **What:** Agents deposit intensity markers in a shared environmental field. Markers evaporate over time (exponential decay). Agents follow gradients toward high-intensity zones. No agent is aware of any other agent — coordination emerges entirely through the shared field. From ant colony foraging to drone swarm search-and-rescue.

- **Evidence:** The mechanism is foundational in swarm robotics. Two independent 2021-2025 research groups (Grasso & Innocente 2021, arXiv:2109.10761; Nguyen 2021, arXiv:2105.03546) applied digital pheromones for decentralized multi-agent coordination and achieved self-organized workload distribution. S-MADRL (Aina & Ha, October 2025, arXiv:2510.03592) demonstrated effective coordination of up to 8 agents without explicit communication using virtual pheromones, with agents self-organizing into "asymmetric workload distributions that reduce congestion." UAV coordination (Devaraju et al. 2022) balanced area coverage and network connectivity using stigmergy-based digital pheromone maps. Stigmergic city modeling (Alfeo et al. 2017) identified persistent and temporary hotspots in crowd dynamics from spatiotemporal pheromone deposits — the exact analog to ticker attention in MIDGE.

- **Source:** arXiv:2510.03592 (October 2025); arXiv:2109.10761 (September 2021); arXiv:2105.03546 (May 2021); Xu et al. arXiv:1911.12504 (November 2019)

- **Fits our case because:** MIDGE's `bio_market_wiring.py` already has the deposit logic. The `_wire_stigmergy()` function deposits `convergence.bullish`, `convergence.bearish`, SUCCESS, and DANGER markers with ticker-derived 2D positions. What is missing is: (1) the Stigmergy object itself with an evaporation step, (2) agent gradient-reading that returns "what tickers are currently hot," (3) calling that read in the sensing hook or agent step. The infrastructure is closer to complete than it appears — it needs activation, not invention.

- **Technical parameters (from literature):** Evaporation rate τ ∈ [0.01, 0.10] per step for short-lived signals (trade signals), τ ∈ [0.001, 0.01] for persistent pattern memory. Deposit intensity proportional to signal strength (0.0–1.0). Gradient read = argmax over neighborhood. Critical insight from Xu et al.: pheromone state should be added to agent observation space — agents don't just "go where trails are" but incorporate trail intensity as a feature for decision-making.

- **Tradeoffs:** Evaporation rate is the critical parameter — too fast and trails disappear before other agents arrive, too slow and old signals attract agents to dead opportunities. For MIDGE's market context, signals decay on different timescales: technical signals (hours), fundamental signals (days), institutional signals (weeks). A multi-layer pheromone field with domain-specific decay rates is more correct than a single field.

---

### Approach 2: Quorum Sensing as Collective Confidence Oracle

- **What:** In bacterial quorum sensing, individual cells produce a signaling molecule (autoinducer). When population density crosses a threshold, autoinducer concentration passes a detection threshold, triggering collective behaviors — biofilm formation, virulence, bioluminescence. The key property: the threshold is population-mediated, not externally imposed. N independent agents agreeing IS the confidence. No formula is needed.

- **Evidence:** The quorum sensing mechanism is 50+ years of biology, applied extensively in distributed systems. In multi-agent AI, its analog appears in decentralized consensus: the paper by Vinitsky et al. (arXiv:2106.09012, 2021) showed that agents can learn behavioral norms by observing public sanctioning events — the quorum of sanction events IS the norm signal, with no central authority determining compliance. The Roundtable Policy framework (September 2025) directly applies quorum-based weighted consensus of multiple LLMs for collective reasoning, achieving state-of-the-art scientific reasoning performance.

- **Source:** Vinitsky et al. arXiv:2106.09012 (2021); Roundtable Policy arXiv (September 2025)

- **Fits our case because:** MIDGE's `_wire_quorum()` deposits signals from ConvergenceAlerter, PatternWatcher, and dual-confirmation sources into a shared quorum space. When 3 independent sources fire on the same ticker+direction, that is a biological quorum event. The quorum count can directly replace or augment the current confidence formula: `quorum_confidence = 1 - (1 - c1)(1 - c2)(1 - c3)` becomes the confidence when 3+ sources have voted, without any top-down formula calculation. This aligns with the PAPERCLIP-PROPOSAL.md insight: "Quorum IS the confidence oracle. When N independent agents converge on same ticker+direction through different paths, the quorum count IS the confidence score."

- **Technical parameters:** Biological quorum sensing uses a single molecular threshold. For MIDGE: minimum quorum size = 3 (Law 2, triadic minimum), quorum window = domain-specific (positioning: 14 days, technical: 72 hours), quorum decay = when any constituent signal expires, the vote count decrements. The QuorumSpace should return not just "quorum reached" but a distribution of recent quorum compositions — this is how the organism learns which quorum types are most reliable.

- **Tradeoffs:** Quorum counting does not distinguish quality from quantity — 3 weak signals can reach quorum. This is why the current convergence alerter uses Thompson-weighted geometric mean. The right integration is quorum count as a *multiplier* on Thompson confidence, not a replacement: strong quorum amplifies a good Thompson score, prevents a bad Thompson score from slipping through without corroboration.

---

### Approach 3: Collective Norm Formation via Social Learning (Peer Imitation + Sanctioning)

- **What:** Norms emerge without central design when agents: (1) observe which behaviors produce good outcomes in peers, (2) imitate those behaviors (social learning), and (3) sanction peers for norm violations. The combination of imitation and sanctioning creates a self-reinforcing cultural ratchet. Once a norm tips past ~30% adoption, it becomes self-sustaining through these mechanisms.

- **Evidence:** Three independent 2024-2026 papers converge on this finding:
  1. Gupta et al. (arXiv:2510.14401, October 2025, accepted AAMAS 2026): social learning + norm-based punishment achieves cooperative norm emergence without explicit reward signals across LLM agents. Works even in resource-scarce environments.
  2. Vinitsky et al. (arXiv:2106.09012, 2021): public sanctioning events as the norm signal — agents classify observed behaviors as approved/disapproved and learn norms from collective sanctioning, not central authority.
  3. Takata et al. (arXiv:2411.03252, 2024): personality differentiation and social norms spontaneously emerge in populations of communicating agents, with diversity maintained through hallucination-like variation.

- **Source:** arXiv:2510.14401 (October 2025); arXiv:2106.09012 (2021); arXiv:2411.03252 (2024)

- **Fits our case because:** MIDGE's agents can observe shared outcome signals (prediction wins/losses, deception events, convergence alerts) and the Thompson sampler already tracks which signal sources have been reliable. The mechanism needed is: when a signal source performs well repeatedly, that source's pattern gets promoted as a "cultural norm" — other agents should attend to it preferentially. The EndocrineSystem's oxytocin (trust, cooperation) is the biological substrate: high oxytocin = high peer trust = stronger imitation. This is already built in `endocrine_system.py:get_trust_level()` — but nothing calls it to modulate how much agents weight peer signals.

- **Technical parameters:** Social learning threshold — agent updates its strategy when observed peer success rate > own success rate by a margin δ ∈ [0.05, 0.15]. Sanctioning signal = prediction failure published to EventBus with source attribution. Cultural memory — the last N outcomes per source (N=20 suggested by Thompson literature) constitute the "cultural record" agents consult.

- **Tradeoffs:** Social learning creates conformity pressure that can collapse diversity. If all agents imitate the currently-successful strategy, the population loses exploratory capability. The Microbiome's Shannon diversity index (already built in `microbiome.py`) is the health metric: if diversity drops below 0.5 (dysbiosis), the cultural system needs a dissenters mechanism — agents that explore regardless of peer pressure. The Curiosity Drive (Team 1's domain) is the biological dissenters: it introduces novelty even when exploitation is culturally dominant.

---

### Approach 4: Environmental Memory as Shared Cognitive Scaffold

- **What:** Instead of direct agent-to-agent communication, agents write to and read from a shared environmental state that persists beyond any individual agent's lifetime. This is the "blackboard" architecture applied to cultural coordination. The environment becomes the culture's memory — the medium through which collective learning accumulates and guides future behavior.

- **Evidence:** Co2PO (arXiv:2602.02970, February 2026) demonstrates a shared blackboard architecture for multi-agent coordination where agents broadcast "positional intent and yield signals" without centralized control, achieving higher returns with cost-compliant policies. The system uses risk-triggered communication — agents only write to the blackboard when their hazard predictor fires, preventing information overload. The Generative Agents paper (Park et al. 2023, arXiv:2304.03442) showed that a shared memory stream across 25 agents produced emergent social coordination — a Valentine's Day party organized itself without any agent being instructed to organize it — because agents read shared context and planned coherently with each other through the environmental record alone.

- **Source:** arXiv:2602.02970 (February 2026); Park et al. arXiv:2304.03442 (2023)

- **Fits our case because:** MIDGE's signal archive (`data/midge/signals/YYYY-MM-DD.jsonl`) and pattern library (`data/market/raw/`) are already a form of environmental memory. The gap is that agents don't query this memory as a cultural scaffold — they re-derive from raw signals each session. The PatternLibrary's `query_similar()` is environmental memory consultation. If agents broadcast their current investigation focus to a shared "cultural focus field," other agents would naturally route toward complementary investigations rather than duplicating work. This is stigmergy at the task level rather than the ticker level.

- **Technical parameters:** Blackboard structure: `{ticker, direction, investigation_depth, claiming_agent, timestamp}`. Selective write: only write when beginning an investigation. Read: at sensing step start, query blackboard for active investigations and de-prioritize duplicate work. Expiry: blackboard entries expire after 3× the domain window (e.g., technical signals: 3×72h = 216h). This prevents stale claims from blocking fresh investigations.

- **Tradeoffs:** Shared state creates read-write contention in concurrent systems. MIDGE's EventBus is thread-safe (RLock), but a blackboard requires atomic claim operations — "I claim this ticker" must be atomic or two agents claim simultaneously. The simplest solution: use EventBus stream entries as the blackboard (write via `write_to_stream`, read via `read_from_stream`) — both already thread-safe and built into the existing EventBus.

---

## Novel Approaches

### Novel Approach 1: Evolved Constitutions via Genetic Programming

- **What:** Rather than designing behavioral norms by hand, let the agent population evolve them through genetic selection. LLM-driven genetic programming generates candidate behavioral rules (constitutions), evaluates them via performance metrics, and iterates through multi-island evolution. The best norms survive and spread. The culture writes itself.

- **Why it's interesting:** This is genuinely autopoietic norm formation — the norms emerge from within the system's own selective dynamics, not from an external designer. It satisfies Law 6 (autopoietic closure) in a way that hand-designed rules do not: the culture reproduces itself through the very mechanisms of agent interaction.

- **Evidence:** Kumar et al. (arXiv:2602.00755, February 2026): LLM-driven genetic programming with multi-island evolution discovered that minimizing communication (0.9% vs. 62.2% of steps) outperformed verbose coordination. Achieved 123% improvement (S=0.556 ± 0.008) over human-designed baselines. The counterintuitive finding — less communication is better — would never have been hand-designed.

- **Source:** arXiv:2602.00755 (February 2026, accepted AAAI/ACM AIES 2025)

- **Fits our case because:** MIDGE's hypothesis engine (RSI Layer 2) is already a form of evolutionary norm selection: hypotheses compete, are validated, promoted, hibernated, or retired. Extending this to behavioral norms — which signal-sourcing strategies, which coordination patterns, which caution thresholds — would let the culture evolve from outcome pressure rather than from system design. The Hypothesis Registry's event-sourced lifecycle (probation → active → hibernated → retired) could directly host behavioral norms alongside market hypotheses.

- **Risks:** Genetic programming can converge to unexpected local optima — norms that score well on metrics but are culturally brittle or adversarially exploitable. The counterintuitive "minimize communication" finding is a warning: evolved norms can be uninterpretable. MIDGE needs a norm interpretability layer (a "cultural commentary" system) so Guiding Light can audit what norms have emerged.

---

### Novel Approach 2: Cultural Evolution Through Population Transmission Bottlenecks

- **What:** Culture evolves not just through imitation but through transmission bottlenecks — moments when knowledge must be compressed and selected for transmission. Each bottleneck acts as a filter that amplifies successful patterns and drops noise. In human culture, bottlenecks are deaths, generations, migrations. In MIDGE, they can be designed: agent senescence events, consolidation phase memory compression, OctopusColony arm despawning.

- **Why it's interesting:** The bottleneck is the evolutionary pressure. Without designed bottlenecks, cultural transmission is lossless — agents accumulate all patterns and none get selected. With bottlenecks, the patterns that survive transmission become the cultural foundation. This is how traditions form.

- **Evidence:** Perez et al. (arXiv:2403.08882, March 2024): cultural evolution in LLM populations manipulates network structure, personality, and information aggregation method. Takata et al. (arXiv:2411.03252, 2024): agent personalities and social norms emerge and evolve through communication, with diversity maintained across generations. Dagan et al. (arXiv:2001.03361, 2020): co-evolution of language and agents produces learnable communication systems — the language co-evolves with the agents using it.

- **Source:** arXiv:2403.08882 (March 2024); arXiv:2411.03252 (2024); arXiv:2001.03361 (2020)

- **Fits our case because:** MIDGE's `SenescenceManager` marks agents for retirement. When an OctopusArm despawns, its investigation findings should be compressed and written to the PatternLibrary before the arm terminates — the arm's cultural contribution survives its death. The `CollectiveDreamPlanner` (referenced in bootstrap wiring) is the consolidation mechanism: during REST phase, it aggregates individual learnings into collective patterns. If the dream planner performs active selection (choosing the most predictive patterns to consolidate, not all patterns), it becomes the designed bottleneck.

- **Risks:** Transmission bottlenecks can cause cultural loss if the selection criterion is wrong. If MIDGE consolidates patterns by recent confidence score, it will lose rare patterns that are highly reliable when they do fire. A recency-weighted but rarity-protected selection criterion is needed: patterns that are recent OR rare AND reliable survive the bottleneck.

---

### Novel Approach 3: Spontaneous Role Differentiation via Interaction History

- **What:** Start all agents identical. As they accumulate different interaction histories, they develop different "character" — some have investigated more macro signals, some more technical, some more insider. This history-driven differentiation naturally produces role specialization without explicit role assignment. The culture develops a division of labor from the bottom up.

- **Why it's interesting:** This eliminates the "who designs all the inhabitants" problem from the research brief entirely. Instead of designing 50 inhabitants with distinct roles, deploy 50 identical stem cells with distinct initial seeds (different signal sources, different tickers) and let history differentiate them. The culture of specialists emerges from accumulated different experiences.

- **Evidence:** Takata et al. (arXiv:2411.03252, 2024): agents that begin identical develop distinct personalities through communication, as "emotions shift through communication and communities form." The differentiation is not designed — it emerges from heterogeneous interaction histories. S-MADRL (arXiv:2510.03592, 2025): agents with identical initialization self-organize into "asymmetric workload distributions" — role differentiation emerges from workload dynamics alone. This is Law 5 (Stem Cell Principle) applied at the culture level: one class, differentiated by experience.

- **Source:** arXiv:2411.03252 (2024); arXiv:2510.03592 (2025)

- **Fits our case because:** MIDGE already has Law 5 (Stem Cell Principle) in the genome. The `AgentGenome` supports epigenome-based role differentiation via configuration. The missing piece is a history-tracking layer that records each agent's accumulated interaction history and uses it to update the agent's epigenome over time. When an agent has investigated 50 energy sector signals, its epigenome should shift toward EIA/commodity weighting — it has become an energy specialist not by assignment but by experience.

- **Risks:** History-driven differentiation is slow — it takes many interaction cycles to produce meaningful differentiation. For MIDGE's daemon mode, this means weeks of running before strong role specialization emerges. A seed differentiation at bootstrap (different initial signal source weights per agent) can accelerate convergence to a stable role ecology without violating Law 5.

---

## Emerging Approaches

### Emerging Approach 1: Sparse Communication Topology for Collective Reasoning

- **What:** Multi-agent debate with sparse communication topology (not all-to-all) achieves comparable or superior collective intelligence at dramatically lower cost. Ring topologies, random sparse graphs, and scale-free networks all outperform all-to-all communication for collective reasoning tasks.

- **Momentum:** Yunxuan Li et al. (arXiv, June 2024) demonstrated sparse topology superiority over all-to-all for multi-agent debate. The Synchronization Dynamics paper (arXiv:2508.12314, August 2025) applied Kuramoto oscillator theory to heterogeneous multi-agent AI systems, showing that "increased coupling promotes robust synchronization despite heterogeneous agent capabilities." The OctopusColony already implements ring + cross-connection topology — directly aligned with the emerging consensus.

- **Source:** arXiv (June 2024, sparse communication topology); arXiv:2508.12314 (August 2025)

- **Fits our case because:** MIDGE's OctopusColony ring+cross topology is already aligned with the finding that sparse topologies outperform all-to-all. The Kuramoto synchronization model suggests that MIDGE's heterogeneous specialists (energy agent, insider agent, macro agent) can achieve synchronized collective intelligence through coupling, even without homogeneous capability. The coupling mechanism in MIDGE is the EventBus + hormone broadcasts — these are the synchronization channels.

- **Maturity risk:** The Kuramoto model for AI synchronization is a 2025 paper — it has mathematical grounding (Kuramoto is decades-old physics) but has not yet been validated at MIDGE's scale (50+ agents with 30 distinct signal domains). The synchronization properties of heterogeneous systems with vastly different processing rates (Finnhub WebSocket at milliseconds vs. Congressional filings at weeks) are unknown.

---

### Emerging Approach 2: Microbiome-Style Strain Competition for Cultural Diversity

- **What:** Rather than assigning roles to agents, maintain multiple competing "strains" of behavioral strategy. Strategies that produce better outcomes grow in population; strategies that fail shrink and die. The cultural ecology maintains diversity through Shannon diversity monitoring and active intervention when monoculture develops (dysbiosis).

- **Momentum:** MIDGE already implements this in `microbiome.py` with 5 competing strains (pattern_decomposer, anomaly_detector, signal_amplifier, noise_filter, nutrient_synthesizer) that compete for resources based on fitness. The Shannon diversity index and dysbiosis detection are implemented. This is battle-tested biology applied to software — the mechanism is already in the codebase.

- **Source:** `mae_core/emergent/microbiome.py` (existing codebase); biological quorum sensing literature

- **Fits our case because:** The Microbiome's competitive strain ecology is the right model for cultural diversity management. If the current market regime strongly rewards momentum-following strategies, the "signal_amplifier" strain will dominate. Dysbiosis detection will fire. The system should then promote contrarian strains — "anomaly_detector" — to restore diversity. This is automatic cultural rebalancing without external intervention. The key extension needed: wire market regime signals into the Microbiome's fitness evaluation, so that the strain ecology evolves with market conditions.

- **Maturity risk:** The current Microbiome operates on abstract "input_type" strings — the fitness evolution is not yet connected to real market outcomes. Connecting strain fitness to Thompson sampler outcomes (did this strategy produce profitable convergence alerts?) is the activation step, but the linkage semantics need careful design.

---

### Emerging Approach 3: Collective Intelligence via Diversity-of-Perspectives Ensemble

- **What:** Rather than agents coordinating to converge on one answer, maintain persistent diversity in perspectives and use ensemble combination. The StackingNet approach (arXiv:2602.13792, February 2026) demonstrates that "diversity from a source of inconsistency becomes collaboration" — heterogeneous models that disagree produce better collective outputs than homogeneous models that agree.

- **Momentum:** StackingNet (February 2026) combines heterogeneous black-box foundation models without accessing internal parameters, improving accuracy, robustness, and fairness across language, vision, and reasoning tasks. This is precisely the multi-domain convergence model MIDGE uses — but applied to the analysis layer, not just the signal layer. If MIDGE's octopus arms have genuinely different specializations (not just identical GENERAL arms), their disagreements become a signal — divergent analyses on the same ticker are informative, not noise.

- **Source:** arXiv:2602.13792 (February 2026)

- **Fits our case because:** MIDGE's ConvergenceAlerter requires 3+ domains to agree. But what happens when some domains agree and others explicitly dissent? Currently, dissenting domains are simply absent from the convergence count. A richer model: if Domain A and Domain B agree but Domain C explicitly contradicts, the convergence alert carries a "contested" flag with the dissenting domain noted. Human judgment can then be applied more precisely. This is collective intelligence through structured disagreement, not just structured agreement.

- **Maturity risk:** "Contested" convergence alerts require Guiding Light to assess the disagreement — this increases the human cognitive load that MIDGE is supposed to reduce. The system needs a meta-level rule for when contested signals are worth surfacing vs. when contradiction implies the signal is too noisy to act on.

---

## Gaps and Unknowns

1. **The Stigmergy and QuorumSpace objects may not exist.** `bio_market_wiring.py` references `ctx.stigmergy` and `ctx.quorum_space` and gracefully skips if they're None. The source files are not present in `mae_core/emergent/`. This is the most critical gap for coordination: the wiring is built, the object being wired to is absent.

2. **How does cultural memory survive restarts?** MIDGE's cultural layer — pheromone trails, quorum vote history, agent interaction histories — lives in memory. When the daemon restarts, culture resets to zero. The PatternLibrary is the only durable cultural memory (persists to JSONL). All other cultural mechanisms need a serialization story or they are reset-amnesia prone.

3. **What is the minimum viable culture?** The research brief asks how 50+ agents form a culture that produces better research than any individual. But MIDGE currently has 12 Mesa agents plus 3 OctopusArms = 15 agents with any agency. The gap from 15 to 50+ is the deployment question (Team 4's domain), but the *cultural* question is: how many agents are needed before collective intelligence exceeds individual intelligence? The SwarmBench paper found that "some rudimentary coordination is observed" but quantified the threshold. For MIDGE's domain, this is unknown.

4. **What is the right evaporation timescale for market signals?** Biological pheromone evaporation is calibrated to the foraging timescale. MIDGE has signals on timescales from milliseconds (Finnhub WebSocket tick data) to years (institutional 13F filings). A single evaporation rate cannot serve all domains. This requires a multi-layer pheromone field with domain-specific decay rates, and the interaction between layers (does a fast-evaporating technical signal reinforce a slow-evaporating fundamental signal?) is untested.

5. **Does sparse topology actually work for MIDGE's heterogeneous domain structure?** The sparse communication research was conducted on homogeneous agents performing identical tasks (debate rounds, math problems). MIDGE's agents have heterogeneous specializations and heterogeneous timescales. Whether sparse topology produces collective intelligence when agents are doing fundamentally different things (energy sector watcher vs. insider trade tracker) is unknown.

6. **Can cultural norms be adversarially exploited?** If MIDGE develops cultural norms around which signal sources to trust, and an adversary knows those norms, they can create spoofed signals in trusted sources that bypass the cultural filter. The HAVEN (immune) system is the defense, but the interaction between cultural trust norms and adversarial signal injection is untested.

---

## Synthesis

### What the landscape looks like for emergent culture in multi-agent systems (2026)

The field is split between two approaches that don't yet communicate well:

**Swarm intelligence tradition (biology-derived):** Stigmergy, quorum sensing, pheromone trails. Proven at scale in robotics and physics. Computationally cheap. Produces genuine emergence from simple rules. The limitation: classical swarm agents are simple rule-following machines. LLM-based agents with complex internal states don't map cleanly to classical swarm math. The 2025 LLM-swarm papers (arXiv:2506.14496) found a 300x computational overhead penalty — LLMs cannot execute classical swarm behavior at speed.

**Social learning tradition (social science-derived):** Norm emergence through imitation, sanctioning, and peer learning. Proven in both biological populations and LLM agent populations (arXiv:2510.14401, arXiv:2411.03252). Produces genuine culture — shared norms that persist without central enforcement. The limitation: slow to converge, requires many interaction cycles, vulnerable to monoculture once a norm tips past the adoption threshold.

**The synthesis that fits MIDGE:** MIDGE is neither a classical swarm nor a social learning system — it is a hybrid biological organism where fast, cheap coordination (stigmergy, quorum sensing) handles market signal routing, while slow cultural learning (norm formation, role differentiation, memory consolidation) handles the deep research capacity that produces genuine edge. MIDGE needs both layers operating at different timescales.

### The strongest approach for MIDGE

The single highest-leverage intervention is **activating the QuorumSpace as the collective confidence oracle**. This is supported by:
- The mechanism is already wired (`_wire_quorum()`)
- It directly replaces the weakest part of MIDGE's current architecture (confidence formula that doesn't discriminate winners from losers)
- It is biologically justified and mathematically simple
- It satisfies Law 1 (triadic witnessing: quorum requires 3+ independent sources), Law 2 (triadic stability), and Law 6 (confidence emerges from internal dynamics, not external formula)

The second highest-leverage intervention is **multi-layer pheromone trails with domain-specific decay** wired to the OctopusColony's task claiming mechanism. If each OctopusArm reads the ticker heat map before selecting a new investigation target, investigations naturally cluster on high-signal tickers without any central dispatcher. This produces the "emergent attention allocation" that MIDGE's convergence engine cannot currently achieve alone.

### What combination works best

**Tier 1 (activate existing infrastructure):**
1. Implement Stigmergy and QuorumSpace as proper objects — they are referenced but may not exist.
2. Add evaporation step to Stigmergy (runs every 50 Mesa steps — fast enough to reflect signal decay).
3. Add quorum threshold check to ConvergenceAlerter: require N≥3 independent quorum deposits on a ticker before issuing an alert. This is a higher bar than current min_domains=3, because quorum requires deposits from genuinely independent sources at independent times.
4. Wire OctopusArm task selection to read ticker heat map from Stigmergy before claiming a new investigation.

**Tier 2 (cultural memory layer):**
5. Serialize pheromone state and quorum history to `data/market/` at each heartbeat — culture survives restarts.
6. Write agent interaction summaries to the PatternLibrary at ARM despawn time — the arm's cultural contribution persists.
7. Wire Microbiome strain fitness to Thompson sampler outcome history — the microbial ecology evolves with market conditions.

**Tier 3 (emergent role differentiation):**
8. Add history-tracking to MycelialAgent: accumulate a per-agent domain interaction count. After N interactions with a domain, bump that domain's weight in the agent's epigenome.
9. Add seed differentiation at bootstrap: assign different initial domain weights to different agents. This accelerates natural differentiation without violating Law 5.

### What the orchestrator needs to know

**The critical insight from the 2026 swarm literature:** LLM agents cannot do classical swarm intelligence (too slow). But MIDGE is not running LLM agents — it is running Python agents with rule-based signal processing. Classical stigmergic coordination IS achievable for MIDGE at low computational cost. The 300x overhead finding does not apply. MIDGE can have true stigmergic coordination where LLM-based systems cannot.

**The critical insight from the norm emergence literature:** Norms emerge most reliably when three conditions hold: (1) agents can observe outcomes (MIDGE has Thompson + OutcomeCollector), (2) agents can imitate successful peers (MIDGE does not yet have peer observation), and (3) norm violations are publicly sanctioned (MIDGE has deception detection + nociception). Two of three conditions are met. The missing condition is peer observation — agents need to see what other agents are investigating and whether those investigations succeeded.

**The architectural key to culture:** Culture in MIDGE is not a separate system to build — it is the **activation of existing dormant systems**. The EndocrineSystem broadcasts the cultural mood. The Microbiome tracks cultural diversity. The Stigmergy field is the cultural territory map. The QuorumSpace is the collective confidence voice. The CollectiveDreamPlanner performs cultural consolidation. Every piece exists. The culture emerges when these systems are connected to each other and to agent action — not through new construction, but through wiring.

The answer to "who designs and builds all 50 inhabitants" is: nobody. Deploy them identical, seed them differently, and let the cultural layer differentiate them over time. The organism designs its own inhabitants.
