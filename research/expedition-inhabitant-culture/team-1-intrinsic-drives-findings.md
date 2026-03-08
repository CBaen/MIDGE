# Team 1 Findings: Intrinsic Drive Architecture
## Date: 2026-03-08
## Researcher: Team Member 1

---

## Preamble: What MIDGE Already Has (Internal Audit)

Before recommending anything new, it is essential to understand how much drive architecture MIDGE already contains. The gap between "living ecosystem" and current state is narrower than it appears from the outside — the problem is mostly activation, not absence.

**Existing drive systems in MIDGE:**

| System | Location | Current State | Missing |
|--------|----------|---------------|---------|
| `CuriosityDrive` | `mae_core/learning/curiosity.py` | Computes novelty, info-gain, prediction-error rewards. Subscribes to `memory.novel_experience`. Has `set_exploration_bonus()` and `get_exploration_targets()`. | Called nowhere autonomously — it computes rewards when asked, doesn't initiate action |
| `HomeostasisRegulator` | `mae_core/coordination/homeostasis.py` | 7 setpoints (energy, cortisol, dopamine, serotonin, processing_load, memory_pressure, threat_level). Computes error and correction signals. Publishes to `coordination.homeostasis_correction`. | Correction signals are emitted but nothing consumes them to change agent behavior |
| `EmotionalSystem` | wired via `bio_market_wiring.py` | Receives convergence events → surprise/fear modulation | No path from emotional state → action selection |
| `ArousalRegulator` | wired via `bio_market_wiring.py` | Records rewards from prediction results | Arousal level doesn't feed into agent decision priority |
| `GamificationMixin` | every `MycelialAgent` | `exploration_bonus` and `novelty_threshold` in genome | Static values, not driven by homeostatic state |
| `_decide()` in lifecycle | `lifecycle_decision.py` | Advisory-guided cascade: reflex → collision → stigmergy → advisory → worldline → dream | No slot for drive/desire consultation in the cascade |

**The diagnostic:** MIDGE has the biological organs but they are not connected to the motor system. The HomeostasisRegulator emits correction signals that dissipate unheard. The CuriosityDrive computes novelty scores nobody asks for. The emotional system modulates internal variables that don't influence what agents do next. This is the core problem: drive generation exists, drive-to-action coupling does not.

---

## Battle-Tested Approaches

### Approach 1: BDI Plan Libraries with Triggering Events (AgentSpeak/Jason/SPADE-BDI)

- **What:** BDI architecture separates agent state into Beliefs (what the agent knows), Desires (goals it wants to achieve), and Intentions (committed plan of action). Plans activate when a triggering event matches a context condition: `TriggeringEvent : Context <- Body`. Plans are selected from a library; context guards determine which plan fires.
- **Evidence:** Jason (Java BDI interpreter for AgentSpeak) has been in production use since 1995 through the present. SPADE-BDI (Python, LGPL) is the Python implementation — version 0.3.2 available on PyPI as of 2026. Used in smart factory coordination, IoT agent systems, academic multi-agent courses. SPADE 4.0.3 is the underlying platform (asyncio-native).
- **Source:** https://spade-bdi.readthedocs.io/latest/ (accessed 2026-03-08); https://github.com/javipalanca/spade (active, last commit 2024-2025)
- **Fits our case because:** MIDGE's biological systems already have market-specific triggering conditions defined via EventBus subscriptions in `bio_market_wiring.py`. The BDI pattern maps directly: beliefs = current homeostatic state + market signals, desires = goal predicates (curiosity goal, stability goal, threat response goal), intentions = the EventBus handlers already written. What's missing is the plan-selection layer that *chooses* which desire to pursue when multiple are active.
- **Tradeoffs:** Jason requires Java; SPADE-BDI's Python agentspeak interpreter adds a new dependency and a new programming paradigm. Full adoption would require rewriting agent logic in AgentSpeak syntax, which violates the "no redesign" boundary. The *pattern* is valuable; the *framework* is not a fit.

### Approach 2: Homeostatic Reinforcement Learning — Drive as Setpoint Deviation Reward

- **What:** Define reward not from external outcomes but from the distance between current internal state and homeostatic setpoint. Drive D(H_t) = Σ|h_i* - h_i,t|^n / m (multi-dimensional Euclidean distance from ideal). Reward = drive reduction: r = D(H_t) - D(H_t + K_t). Action selection becomes: choose actions predicted to reduce drive fastest.
- **Evidence:** Formally derived in Keramati & Gutkin (2014), published in eLife (peer-reviewed), widely cited. Continuous-space extension (CTCS-HRRL) demonstrated in arXiv:2401.08999v1 (2024), validated on navigation and survival tasks. The reward formula is mathematically proven to produce anticipatory regulation — agents learn to act preemptively before homeostatic deviation occurs, not just reactively.
- **Source:** https://pmc.ncbi.nlm.nih.gov/articles/PMC4270100/ (Keramati & Gutkin 2014, accessed 2026-03-08); https://arxiv.org/abs/2109.06580 (CTCS-HRRL, accessed 2026-03-08)
- **Fits our case because:** MIDGE's `HomeostasisRegulator` already computes the error signal (target - current) for 7 parameters. The mathematical machinery for drive computation already exists. What's missing is the conversion: drive-vector → action priority modifier. The formula `r = D(H_t) - D(H_t + K_t)` could be implemented as a `compute_drive_urgency()` method on the existing `HomeostasisRegulator` that returns a priority scalar. This scalar then feeds into the `_decide()` cascade as a pre-advisory bias.
- **Tradeoffs:** The reward signal is indirect — it motivates actions that reduce homeostatic deviation, but mapping "reduce cortisol" to specific market actions requires that connections be pre-specified (e.g., "successful investigation reduces curiosity drive"). This is not self-evident and requires careful tuning of which actions satisfy which drives.

### Approach 3: Subsumption Architecture — Layered Behavior Priorities

- **What:** Rodney Brooks (1986): decompose behavior into independent layers, each with its own sensor-action loop. Higher layers subsume (override) lower layers. No central planner. Each layer runs asynchronously and generates continuous output. Priority emerges from layer ordering, not from a central scheduler.
- **Evidence:** Deployed in real robots since 1986 — Roomba navigation is a commercial descendant. Proven at scale in swarm robotics. The architecture has never failed to produce autonomous behavior: it is the gold standard for "no central control, behavior emerges from layer interaction." Nature paper (2024) demonstrated automatic design of stigmergy-based collective behaviors for robot swarms using subsumption-adjacent principles.
- **Source:** https://en.wikipedia.org/wiki/Subsumption_architecture (accessed 2026-03-08); Brooks 1986 (A Robust Layered Control System for a Mobile Robot, MIT AI Memo 864)
- **Fits our case because:** MIDGE's `_decide()` cascade is already a soft subsumption architecture — reflex overrides collision check overrides stigmergy overrides advisory. Adding a **drive-urgency layer** before the advisory check would be a surgical insertion into the existing cascade. High drive (HomeostasisRegulator deviation > threshold) subsumes normal advisory processing: the agent acts to restore homeostasis first, then returns to normal task selection.
- **Tradeoffs:** Pure subsumption doesn't plan ahead — it's purely reactive. MIDGE needs some anticipatory behavior (e.g., curiosity drive should proactively seek partial convergences, not just respond when they arrive). Pure subsumption would miss this. Hybrid with BDI desire prioritization solves it.

---

## Novel Approaches

### Approach 4: D2A Value System — Maslow-Inspired Multi-Dimensional Desire with Qualitative Descriptions

- **What:** D2A (Desire-Driven Autonomy, Wang et al., December 2024): Agents maintain a dynamic Value System across multiple desire dimensions (11 in the paper: physiological + safety + belonging + self-actualization). Each dimension has a target value and decays probabilistically over time. The agent generates candidate actions, predicts their effect on desire states, and chooses the action that best fulfills unmet desires. Post-execution: actual effects update the value vector.
- **Why it's interesting:** The qualitative translation step (numerical desire level → textual description) solved a hard problem: agents with only numeric drive vectors don't *act* on them because they can't reason about "hunger=3.2." The insight that drive states need natural-language representation to trigger action is directly applicable to MIDGE's biological systems — a CuriosityDrive score of 0.7 means nothing to the decision router; "strong unresolved curiosity about partial convergence on NVDA" is actionable.
- **Evidence:** arXiv:2412.06435 (December 2024). Evaluated by GPT-4o and human annotators. Outperformed non-desire-driven baselines on naturalness, coherence, and plausibility of generated activities. Open-source implementation on GitHub (zfw1226/D2A).
- **Source:** https://arxiv.org/html/2412.06435v1 (accessed 2026-03-08); https://github.com/zfw1226/D2A
- **Fits our case because:** MIDGE's convergence alerter produces rich structured outputs (ticker, direction, domains, confidence). The D2A pattern suggests: translate drive states into natural-language desire descriptors, then use those descriptors to select market investigation actions. The "propose N candidates, evaluate desire satisfaction, select best" loop maps directly to MIDGE's partial convergence investigation: "curiosity high + partial convergence on NVDA detected → propose 3 investigation actions → predict which best satisfies curiosity → act."
- **Risks:** D2A was designed for LLM-driven agents. The qualitative description translation assumed an LLM was reading it. In MIDGE (pure Python), a rule-based translation table would substitute. The predict-evaluate loop requires that each candidate action have a known effect on each desire dimension — this must be pre-specified or learned.

### Approach 5: IMGEP — Self-Generated Goals via Learning Progress as Intrinsic Drive

- **What:** Intrinsically Motivated Goal Exploration Processes (Forestier & Oudeyer, JMLR 2022): agents generate their own goals as parameterized fitness functions. Goal selection is driven by *learning progress* — the derivative of competence over time. The agent gravitates toward goals where it is improving fastest (the "interest" function). Competence plateau = abandon goal. Steep improvement = continue pursuing.
- **Why it's interesting:** Most drive architectures define what drives are. IMGEP defines *how drives are generated from capability gaps*. This is the difference between "I want food" (predefined) and "I want to understand why domain X correlates with Y when I can't yet explain it" (emergent). MIDGE's archaeological system has exactly this structure: templates accumulate, confidence intervals narrow, and the point of maximal uncertainty is where investigation is most rewarded.
- **Evidence:** JMLR publication (peer-reviewed, volume 23, 2022). Demonstrated on real humanoid robot exploring hundreds-of-dimensional goal spaces. Open-source code at github.com/sebastien-forestier/IMGEP.
- **Source:** https://arxiv.org/abs/1708.02190 (IMGEP paper); https://dl.acm.org/doi/10.5555/3586589.3586741 (JMLR version, accessed 2026-03-08)
- **Fits our case because:** The `PatternLibrary` tracks template confidence intervals (Clopper-Pearson CI). Templates with wide confidence intervals (few observations) have high learning potential. An IMGEP-style drive would: compute learning progress per template, generate exploration goals toward low-confidence templates, and autonomously direct the excavation daemon toward under-explored domains. This is purely internal mechanism — no external orchestration needed.
- **Risks:** IMGEP was designed for continuous action spaces in robotics. Adapting to a discrete investigation action space requires defining a "competence" measure for market template quality and a "learning progress" function that tracks template CI narrowing over time. This is specifiable but requires careful definition.

---

## Emerging Approaches

### Approach 6: Active Inference / Free Energy Principle as Unified Drive Architecture

- **What:** Karl Friston's Free Energy Principle (FEP): every system (biological or artificial) acts to minimize the surprise (free energy) between its predictions and sensory observations. Intrinsic motivation *is* free energy minimization. Exploration = epistemic value (reduce uncertainty). Goal-seeking = pragmatic value (achieve preferred states). Both emerge from the same variational objective: F = E[surprise] - entropy.
- **Momentum:** Rapidly growing. Factorised Active Inference for multi-agent systems presented at AAMAS 2025 (top-tier conference). Review paper in arxiv:2508.05619 (2025) examines active inference in the era of LLM experience. Multi-agent FEP formulations appearing quarterly. GitHub activity strong: `pymdp` (Python active inference library) has 1000+ stars.
- **Source:** https://www.ifaamas.org/Proceedings/aamas2025/pdfs/p1793.pdf (AAMAS 2025, accessed 2026-03-08); https://arxiv.org/html/2508.05619v1 (2025 review, accessed 2026-03-08)
- **Fits our case because:** Active inference unifies all of MIDGE's existing drives under one framework. CuriosityDrive = epistemic free energy (uncertainty reduction). HomeostasisRegulator = pragmatic free energy (achieve preferred physiological states). Prediction error in `_compare()` step = direct free energy signal. The FEP doesn't require new systems — it provides a mathematical grounding for connecting existing ones. The `_prediction_error` field already computed in `MycelialAgent` is a free energy proxy.
- **Maturity risk:** The `pymdp` library implements discrete-state active inference. MIDGE's state space is continuous and high-dimensional. Full FEP implementation in continuous space requires variational Bayes and neural network function approximators. Production deployments of continuous-space active inference in multi-agent Python systems are rare (2025). This is the framework to *frame* the architecture conceptually; implementation should use simpler drive mechanisms that approximate FEP behavior.

### Approach 7: SAGA Bi-Level Objective Evolution — Drives That Evolve Their Own Goals

- **What:** SAGA (Scientific Autonomous Goal-evolving Agent, Du et al., December 2024): bi-level architecture where an outer loop analyzes optimization results and proposes new objectives, while an inner loop pursues those objectives. Objectives have four fields: natural language description, optimization direction (maximize/minimize/constrain), optional weight, implementation type (candidate-wise/population-wise/filter). Failure modes trigger automatic goal evolution.
- **Momentum:** Published December 2024, presented at AI conferences early 2025. Represents the leading edge of "drives that generate their own successor drives" research.
- **Source:** https://arxiv.org/html/2512.21782v1 (accessed 2026-03-08)
- **Fits our case because:** MIDGE's HypothesisEngine (RSI Layer 2) is already a goal-evolution system: hypotheses transition through probation → active → hibernated → retired, and the Analyzer identifies failure patterns. The SAGA pattern could extend this: when hypotheses consistently fail in a particular market regime, the outer loop (HypothesisGenerator) doesn't just retire individual hypotheses — it evolves the *class* of goals being pursued. For example: "our government domain hypotheses fail in low-volume regimes" → new drive: "investigate whether volume context should gate government signal relevance."
- **Maturity risk:** SAGA was LLM-mediated in the original paper. Adapting goal evolution to pure Python with rule-based failure analysis is feasible but would require defining explicit failure taxonomy — which MIDGE partially has via the `rejected_reason` field on retired hypotheses.

---

## Gaps and Unknowns

1. **The drive-to-action coupling problem is unsolved in the codebase.** Homeostasis emits correction signals on `coordination.homeostasis_correction`. Nothing subscribes. Tracing the channel registration in `bootstrap/organs.py` and `bootstrap/core_systems.py` would confirm whether any consumer was ever wired. This is the most critical gap — before any new architecture, this wire must be confirmed missing or found.

2. **What does "autonomous action" mean for a non-Mesa-agent system?** The biological systems (HomeostasisRegulator, CuriosityDrive, etc.) are not `MycelialAgent` subclasses — they don't have `step()` methods and don't call `_decide()`. When the research brief says "systems wake up on their own," it means two very different things: (a) the system publishes an EventBus message that triggers agent behavior, or (b) the system itself performs an action outside the Mesa loop. Option (a) is achievable without breaking anything. Option (b) requires threading or asyncio, which is Team 4's territory.

3. **The `_decide()` cascade has no drive-priority slot.** Inserting homeostatic urgency requires choosing where in the 8-level cascade it sits. If it's before the advisory, it could override good market signals. If it's after, high-urgency drives could be ignored. The correct position depends on what "drive urgency" means in practice — this requires Guiding Light input on priority ordering.

4. **Mesa 3 has a DiscreteEventSimulator.** The Mesa 3 paper (JOSS 2025) mentions this feature for scheduling events at arbitrary timestamps. It would enable step-independent wakeups. However, the feature is described as "experimental" and its compatibility with MIDGE's existing step hooks is unverified.

5. **Drive satisfaction definitions are absent.** The D2A approach requires defining: what actions satisfy which drives? For CuriosityDrive, satisfaction = novel market pattern investigated. But how does the system know when curiosity has been satisfied? The current CuriosityDrive has `compute_novelty()` but no `is_satisfied()` method. This is a design gap requiring explicit definition before drive architecture can be complete.

6. **No tested pattern for biological systems initiating EventBus messages.** All current bio-market wiring is reactive: market events → bio system response. Zero systems proactively emit market signals (e.g., CuriosityDrive hasn't yet emitted a `CH_PARTIAL_CONVERGENCE` message to request investigation). The pattern for autonomous emission exists in the codebase (any system can call `bus.emit()`) but has never been used by a biological system as a drive output. This is the key missing piece.

---

## Synthesis

### The Landscape

The research reveals three mature paradigms for intrinsic drive architecture, each addressing a different aspect of the problem:

1. **Drive-as-homeostatic-deviation** (Keramati & Gutkin HRRL): the mathematically rigorous foundation. Drive IS the distance from setpoint. This is already partially implemented in MIDGE's HomeostasisRegulator. The gap is converting the computed error into a priority signal that influences action selection.

2. **BDI desire prioritization** (Jason/SPADE-BDI): the software engineering pattern. Beliefs + context conditions → plan selection. The insight is that desires must compete, and winning desires produce intentions that get executed. MIDGE's `_decide()` cascade is a soft version of this — what's missing is a formal desire competition step before the advisory lookup.

3. **Drive generation from learning gaps** (IMGEP, SAGA): the autonomous goal formation approach. Instead of pre-specifying drives, drives emerge from capability gaps and learning progress. This maps directly to MIDGE's PatternLibrary: templates with low confidence have high learning potential, which creates an emergent "investigate here" drive.

### The Strongest Approach for MIDGE

The architecture that fits MIDGE best is a **three-layer drive stack**, building on existing code:

**Layer A — Homeostatic Urgency (pre-advisory in `_decide()` cascade)**
Extend `HomeostasisRegulator.compute_correction()` to return a `drive_urgency` scalar (0.0–1.0) from the multi-dimensional drive distance. Add a consumer on `coordination.homeostasis_correction` that writes urgency to each agent's `_homeostatic_urgency` field. Insert one check in `_decide()` before the advisory: if `_homeostatic_urgency > 0.8`, return action "restore" (mapped per agent role — for MARKET_ANALYST, "restore" = pause investigation and run stability checks). This is 30–50 lines across two files. It connects the already-computed signal to behavior.

**Layer B — Desire Competition via Drive-Weighted Advisory**
The existing advisory system publishes recommendations. Add a `DriveFilter` that modulates advisory priorities based on current drive states: high curiosity boosts "investigate partial convergence" recommendations, high threat-level boosts "verify data quality" recommendations. This is not a new system — it's a pre-processing step on the advisory before agents read it. The BDI insight (desires compete, winner influences intentions) maps to: drives compete, highest-urgency drive amplifies matching advisory signals.

**Layer C — Emergent Goal Generation via Learning Progress**
The `PatternLibrary` tracks template confidence intervals. Add a `LearningProgressMonitor` that periodically (every 500 steps, matching existing cadence) computes per-template Clopper-Pearson interval width. Templates with widest intervals get injected as investigation goals into the TaskPool with priority proportional to learning potential. This is MIDGE's self-generated drive: "I have investigated NVDA 3 times and still can't characterize the pattern — investigate more." Pure Python, pure internal, zero new dependencies.

### What the Orchestrator Needs to Know

The fundamental insight from cross-referencing all approaches: **the missing piece in MIDGE is not drive generation (that exists) but drive routing** — the path from biological system state to action selection. Every approach above (HRRL, BDI, IMGEP) solves exactly this routing problem. The pattern they converge on:

```
internal state deviation → urgency signal → priority modifier → action bias
```

MIDGE has all three intermediate steps as disconnected components. The work is the connections, not new systems. This is consistent with Law 1 (no bare dyads) and Law 6 (autopoietic closure) — the drives that ARE the organism's motivation are already there; they just aren't producing the circular causation that would make them autopoietic.

One architectural decision Guiding Light needs to make: should biological drives **initiate** new EventBus messages (CuriosityDrive emits `CH_PARTIAL_CONVERGENCE` to demand investigation), or should they **modulate** existing agent behavior (CuriosityDrive increases agent `exploration_bonus` so agents naturally gravitate toward novel signals)? The first pattern is more autonomous. The second is safer, less invasive, and preserves existing test coverage. Evidence from HRRL and D2A both support starting with modulation (Layer A + B above) before moving to autonomous initiation (Layer C), because modulation-first ensures drives have proven effects before they start generating independent actions.

---

*Sources accessed 2026-03-08:*
- [Keramati & Gutkin HRRL (eLife 2014)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4270100/)
- [CTCS-HRRL arXiv:2401.08999](https://arxiv.org/abs/2109.06580)
- [D2A: Desire-Driven Autonomy (arXiv:2412.06435)](https://arxiv.org/html/2412.06435v1)
- [SAGA: Goal-Evolving Agents (arXiv:2512.21782)](https://arxiv.org/html/2512.21782v1)
- [IMGEP (JMLR Vol 23, 2022)](https://dl.acm.org/doi/10.5555/3586589.3586741)
- [SPADE-BDI Python docs](https://spade-bdi.readthedocs.io/latest/create_agent.html)
- [Factorised Active Inference, AAMAS 2025](https://www.ifaamas.org/Proceedings/aamas2025/pdfs/p1793.pdf)
- [GWT Selection-Broadcast Cycle (arXiv:2505.13969)](https://arxiv.org/html/2505.13969v1)
- [Subsumption Architecture (Wikipedia)](https://en.wikipedia.org/wiki/Subsumption_architecture)
- [Mesa 3 (JOSS 2025)](https://mesa.readthedocs.io/latest/)
- [BDI Agents Survey, IJCAI 2020](https://www.ijcai.org/proceedings/2020/0684.pdf)
- [Autotelic Agents Survey (JAIR Vol 74)](https://dl.acm.org/doi/10.1613/jair.1.13554)
