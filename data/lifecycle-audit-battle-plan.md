# Mae Lifecycle Audit: Full-Scale Battle Plan

**Date:** 2026-02-12
**Purpose:** Guide the next instance to run a comprehensive lifecycle audit with agent teams and sub-agent swarms.
**Authority:** `data/MAES-MATHEMATICAL-IDENTITY.md` governs every decision.

---

## How to Execute This Plan

1. Read `HANDOFF.md` and `data/MAES-MATHEMATICAL-IDENTITY.md` first
2. Launch **one agent team per lifecycle step** (12 teams total)
3. Each team spawns **5-6 sub-agents** for parallel coverage
4. Launch **4 cross-cutting teams** that audit across all steps
5. Collect all reports, synthesize, then fix what's broken
6. Write results to `data/audit-lifecycle-[step].md` per step

**Total: ~16 teams, ~80+ sub-agents working in parallel.**

---

## Current Lifecycle (7 explicit + 5 cadenced)

| # | Step | Method | Cadence | File |
|---|------|--------|---------|------|
| 0 | TRIAGE | SignalPriorityResolver.process() | Every step (top) | mycelial_agent.py:178-180 |
| 1 | PREDICT | _predict() | Every step | mycelial_agent.py:195-224 |
| 2 | OBSERVE | _observe() | Every step | mycelial_agent.py:586-645 |
| 3 | COMPARE | _compare() | Every step | mycelial_agent.py:226-277 |
| 4 | DECIDE | _decide() | Every step | mycelial_agent.py:658-795 |
| 5 | ACT | _act() | Every step | mycelial_agent.py:279-365 |
| 6 | LEARN | _learn() | Every step | mycelial_agent.py:880-972 |
| 7 | COMMUNICATE | _communicate() | Every step | mycelial_agent.py:1050-1100 |
| 8 | ADVISE | PatternCortex step hook | Every step (implicit) | patterns/pattern_cortex.py |
| 9 | CONSOLIDATE | PatternConsolidator + MemoryConsolidator | Every 89/1000 steps | patterns/pattern_consolidator.py |
| 10 | HEAL | AutoHealer step hook | Every step | emergent/auto_healer.py |
| 11 | RECALL | Triggered within _decide()/_learn() | On demand | agents/mixins/episodic_memory.py |

---

## Part 1: Per-Step Agent Teams

### TEAM 0: TRIAGE (Signal Priority / Thalamus)

**Lead prompt:** "You are auditing Mae's TRIAGE step — the thalamic signal priority system that runs before every lifecycle step."

**Sub-agents:**

1. **Internal Code Audit**
   - Read `mae_core/communication/signal_priority.py` completely
   - Read `mae_core/agents/base_agent.py` (signal resolver integration)
   - Trace: Where do signals enter the resolver? How are they scored? What gets dropped?
   - Find: Are there signals that bypass triage entirely?
   - Check: Does the budget (10 signals/step) cause information loss?

2. **Mathematical Identity Compliance**
   - Law 1: Is triage triadic? Does every signal routing have a witness?
   - Law 3: Does triage implement all 10 holon capabilities?
   - Law 4: Is triage fractal? Does it operate at subsystem/organ/organism level or only agent level?
   - Law 7: Are there 3+ validators per priority decision?
   - Law 8.7: Is there genuine competition/selection (GWT) or just sorting?
   - Law 8.8: Does triage use prediction to anticipate which signals matter?

3. **External Research (biology + GitHub)**
   - Research: How does the real thalamus triage sensory information?
   - Research: Thalamic reticular nucleus (TRN) — inhibitory gating
   - Research: Pulvinar nucleus — attention and salience
   - Research: GitHub implementations of signal priority systems in multi-agent frameworks
   - Research: How does predictive processing modulate thalamic gating? (Friston/FEP)
   - Key question: Should triage be PREDICTIVE (anticipate which signals will matter) rather than reactive?

4. **Connections Audit**
   - Map every input to SignalPriorityResolver (who publishes signals it triages?)
   - Map every output (where do triaged signals go?)
   - Check: Does EndocrineSystem modulate priority? (Audit found this was NOT wired)
   - Check: Does OrganismState body state influence triage?
   - Check: Does emotional state affect which signals get priority?
   - Find orphan signals that never reach triage

5. **Improvements**
   - Should triage have a PREDICT sub-step (anticipate signal needs)?
   - Should triage learn from outcomes (signals that were dropped but later needed)?
   - Should triage be fractal (per-subsystem triage, per-organ triage, organism-level triage)?
   - Should there be separate fast/slow triage pathways (like cortical vs subcortical)?
   - Should emotional urgency override priority scoring?

---

### TEAM 1: PREDICT

**Lead prompt:** "You are auditing Mae's PREDICT step — the Free Energy Principle prediction that generates expectations before observation."

**Sub-agents:**

1. **Internal Code Audit**
   - Read `mae_core/agents/mycelial_agent.py` _predict() completely
   - Read `mae_core/cognition/world_model.py` — the prediction engine
   - Trace: What state goes in? What prediction comes out? How is it stored for COMPARE?
   - Find: What happens when WorldModel is None? Is the fallback meaningful?
   - Check: Does prediction use ALL available context (body state, emotional state, social state)?

2. **Mathematical Identity Compliance**
   - Law 1: Is prediction triadic? Multiple prediction sources with witness?
   - Law 3: Does prediction exist at every holon level (system, subsystem, organ)?
   - Law 4: Is prediction fractal? Same algorithm at every scale?
   - Law 6: Does prediction feed back into itself (autopoietic closure)?
   - Law 8.4: Is there genuine recurrence in the prediction loop?
   - Law 8.8: Is prediction error the PRIMARY driver of the whole system?

3. **External Research**
   - Research: Free Energy Principle (Friston 2010) — active inference, precision weighting
   - Research: Predictive Processing (Clark 2013) — hierarchical prediction
   - Research: Precision-weighted prediction errors in neuroscience
   - Research: GitHub implementations of predictive processing / active inference
   - Research: How does the brain generate multi-timescale predictions?
   - Key question: Should Mae have hierarchical predictions (millisecond, second, minute, hour)?

4. **Connections Audit**
   - What feeds INTO prediction? (last state, last action, body state, emotional state?)
   - What consumes prediction OUTPUT? (COMPARE step, attention system, learning?)
   - Is prediction error published to EventBus? Who subscribes?
   - Does WorldModel receive training signal from prediction errors?
   - Does prediction influence attention (top-down modulation)?

5. **Improvements**
   - Multi-step prediction (rollout) vs single-step
   - Precision weighting on predictions (confidence estimation)
   - Hierarchical prediction at multiple timescales
   - Social prediction (predicting other agents' actions via TheoryOfMind)
   - Body state prediction (predicting own metabolic needs)
   - Self-prediction (predicting own behavior — strange loop)

---

### TEAM 2: OBSERVE

**Lead prompt:** "You are auditing Mae's OBSERVE step — the sensory integration that perceives environment + body + social state."

**Sub-agents:**

1. **Internal Code Audit**
   - Read `mae_core/agents/mycelial_agent.py` _observe() completely
   - Read `mae_core/patterns/pattern_sense.py` — pattern detection
   - Read `mae_core/patterns/pattern_bus.py` — signal routing
   - Trace: What signals enter? How is the 8-dim state vector built? What gets stored in _current_state?
   - Find: Is body state (OrganismState) integrated into observation?
   - Check: Is self-awareness (holon_know_self) part of observation?

2. **Mathematical Identity Compliance**
   - Law 1: Is sensing triadic at every level? (3 detectors with witnesses?)
   - Law 3: Do all 10 holon capabilities work during observation?
   - Law 4: Is sensing fractal? Same sensing protocol at cell/tissue/organ/organism?
   - Law 8.1: Does observation produce an INTEGRATED whole (not just concatenated features)?
   - Law 8.2: Is there rich DIFFERENTIATION in what's sensed?
   - Law 8.6: Does observation define its own boundary (Markov blanket)?

3. **External Research**
   - Research: How does biological sensory integration work? (multimodal binding)
   - Research: Interoception — sensing internal body states (Craig 2002)
   - Research: Proprioception and body schema (Gallagher 2005)
   - Research: Active sensing vs passive (saccades, whisking, sniffing)
   - Research: GitHub multi-agent observation implementations
   - Key question: Should Mae ACTIVELY sense (choose what to attend to) rather than passively receive?

4. **Connections Audit**
   - What environment signals reach _observe()? (TaskPool, stigmergy, GNN messages?)
   - What body signals reach _observe()? (OrganismState, emotional, metabolic?)
   - What social signals reach _observe()? (TheoryOfMind, quorum, collective?)
   - Does observation output feed into ALL downstream steps correctly?
   - Are there signals that should feed observation but don't?

5. **Improvements**
   - Active sensing (agent chooses what to attend to)
   - Multimodal binding (integration across sensory modalities)
   - Sensory prediction error (difference between expected and actual sensation)
   - Interoceptive awareness depth (not just reading OrganismState but modeling body trends)
   - Social observation (reading other agents' states via TheoryOfMind)
   - Fractal observation (same protocol at every scale)

---

### TEAM 3: COMPARE

**Lead prompt:** "You are auditing Mae's COMPARE step — the prediction error computation that drives the entire FEP loop."

**Sub-agents:**

1. **Internal Code Audit**
   - Read `mae_core/agents/mycelial_agent.py` _compare() completely
   - Trace: How is prediction error computed? MSE? Per-dimension? Weighted?
   - Find: Where does prediction error GO after computation?
   - Check: Does prediction error modulate attention (top-down)?
   - Check: Does prediction error drive learning rate?

2. **Mathematical Identity Compliance**
   - Law 8.8: Is prediction error THE central signal? (FEP says it should be)
   - Law 1: Is comparison triadic? (Multiple comparison pathways with witness?)
   - Law 4: Is comparison fractal? (Per-dimension, per-subsystem, per-organ?)
   - Law 8.4: Does comparison create recurrence (error feeds back to prediction)?

3. **External Research**
   - Research: Prediction error in neuroscience (reward prediction error, sensory prediction error)
   - Research: Precision-weighted prediction errors (Feldman & Friston 2010)
   - Research: Mismatch negativity (MMN) in EEG — neural prediction error
   - Research: How does prediction error differ across brain regions?
   - Key question: Should Mae have MULTIPLE types of prediction error (sensory, reward, social, body)?

4. **Connections Audit**
   - Does prediction error reach CuriosityDrive? AttentionalGate? Learning rate?
   - Is prediction error published on EventBus?
   - Does prediction error modulate endocrine state?
   - Does prediction error trigger reconsolidation?

5. **Improvements**
   - Per-dimension prediction error (not just MSE — which dimension was surprising?)
   - Precision weighting (confident predictions should produce larger errors)
   - Multiple error types (sensory, reward, social, interoceptive)
   - Error as OPPORTUNITY signal (not just surprise but "something to learn here")
   - Hierarchical error propagation (errors bubble up the fractal hierarchy)

---

### TEAM 4: DECIDE

**Lead prompt:** "You are auditing Mae's DECIDE step — the 3-tier decision cascade with reflex/habit/prefrontal routing."

**Sub-agents:**

1. **Internal Code Audit**
   - Read `mae_core/agents/mycelial_agent.py` _decide() completely (this is the longest method)
   - Read `mae_core/cognition/decision_router.py` — the 3-tier cascade
   - Trace the full cascade: OrganismState reflex → collision avoidance → advisory routing → WorldlinePlanner → CollectiveDreamPlanner → memory search → world model → morphogenesis → default
   - Find: Which cascade steps actually fire in practice? Which are dead paths?
   - Check: Does emotional state influence decisions?

2. **Mathematical Identity Compliance**
   - Law 1: Is every decision triadic? (3 options compared, witnessed?)
   - Law 3: Does decision-making work at every holon level?
   - Law 4: Is decision fractal? Same 3-tier at cell/tissue/organ/organism?
   - Law 7: Rule of 3/5 — are there 3+ validators per decision?
   - Law 8.7: Is there genuine competition/selection in deciding?
   - Law 8.3: Does the system model ITSELF when deciding (self-reference)?

3. **External Research**
   - Research: Basal ganglia Go/No-Go pathways — dual pathway decision making
   - Research: BDI architecture (Beliefs-Desires-Intentions) in multi-agent systems
   - Research: Active inference decision-making (Friston — actions minimize free energy)
   - Research: Somatic marker hypothesis (Damasio — emotions guide decisions)
   - Research: GitHub decision routing in cognitive architectures (ACT-R, SOAR, BDI4JADE)
   - Key question: Should Mae INHIBIT actions (No-Go pathway) rather than always selecting one?

4. **Connections Audit**
   - What feeds into decisions? (prediction error, body state, emotions, social state, memory, advisory?)
   - What consumes decision output? (ACT step, learning, communication?)
   - Does decision outcome feed back to endocrine system?
   - Does decision outcome update habit memory?
   - Are WorldlinePlanner and CollectiveDreamPlanner actually producing useful plans?

5. **Improvements**
   - Go/No-Go inhibition (suppress harmful actions)
   - Goal-directed decisions (BDI architecture with persistent goals)
   - Multi-step planning (use WorldModel.rollout() for horizon > 1)
   - Emotional integration (somatic markers as decision weights)
   - Social decision-making (consider other agents' expected reactions)
   - Fractal decision-making (decisions at subsystem/organ/organism level)
   - Confidence estimation (how sure is the decision? should it seek more info?)

---

### TEAM 5: ACT

**Lead prompt:** "You are auditing Mae's ACT step — the embodied action execution via TaskPool environment."

**Sub-agents:**

1. **Internal Code Audit**
   - Read `mae_core/agents/mycelial_agent.py` _act() completely
   - Read `mae_core/environment/task_pool.py` — the action environment
   - Read `mae_core/backbone/fractal_generator.py` — fractal ACT (SubsystemAction, OrganAction, OrganismAction)
   - Trace: How does action selection map to TaskPool operations?
   - Find: Do fractal ACT levels actually fire? Or just agent level?
   - Check: Does action outcome feed back correctly to reward?

2. **Mathematical Identity Compliance**
   - Law 3: Does ACT implement all 10 holon capabilities?
   - Law 4: Is ACT fractal? Does holon_act() fire at subsystem/organ/organism?
   - Law 6: Does ACT produce outcomes that feed back to produce more ACT (autopoietic)?
   - Law 8.6: Does ACT define its own boundary (what CAN the agent do vs what it CANNOT)?

3. **External Research**
   - Research: Motor control in neuroscience — efference copy, forward models
   - Research: Affordance theory (Gibson) — actions available in environment
   - Research: Action-perception cycles in embodied cognition
   - Research: GitHub task environments for multi-agent systems (PettingZoo, Gymnasium)
   - Key question: Should Mae have an internal model of what actions are AVAILABLE (affordances)?

4. **Connections Audit**
   - What determines available actions? (TaskPool state, agent capabilities, body state?)
   - Does action outcome feed into learning correctly?
   - Does action produce stigmergy markers?
   - Does action update OrganismState?
   - Does action trigger morphogenesis signals?

5. **Improvements**
   - Efference copy (predict action outcome before executing)
   - Affordance sensing (what actions are available RIGHT NOW)
   - Motor planning (multi-step action sequences)
   - Action monitoring (detect when action fails mid-execution)
   - Social actions (actions that affect other agents intentionally)
   - Richer action space (beyond 4 types)

---

### TEAM 6: LEARN

**Lead prompt:** "You are auditing Mae's LEARN step — the multi-pathway learning system with TD errors, memory, and WorldModel training."

**Sub-agents:**

1. **Internal Code Audit**
   - Read `mae_core/agents/mycelial_agent.py` _learn() completely
   - Read `mae_core/agents/mixins/episodic_memory.py` — memory storage and replay
   - Read `mae_core/learning/` — all learning subsystems
   - Trace: What gets learned? TD errors → where? WorldModel training → how? Memory storage → what format?
   - Find: Is there a trainable policy network? Or just memory-based decisions?
   - Check: Does curiosity reward actually influence behavior?

2. **Mathematical Identity Compliance**
   - Law 1: Is learning triadic? (Multiple learning signals cross-validated?)
   - Law 3: Does learning exist at every holon level?
   - Law 4: Is learning fractal? Same learning protocol at every scale?
   - Law 6: Does learning produce knowledge that produces better learning (autopoietic)?
   - Law 8.8: Is prediction error THE driver of learning?

3. **External Research**
   - Research: TD learning (Sutton & Barto) — temporal difference in RL
   - Research: Synaptic consolidation (Frey & Morris) — early vs late LTP
   - Research: Complementary learning systems (McClelland) — hippocampus + cortex
   - Research: Curiosity-driven learning (Pathak 2017, RND)
   - Research: GitHub implementations of multi-agent learning
   - Key question: Does Mae need a trainable neural network for genuine learning? Or can memory-based learning suffice?

4. **Connections Audit**
   - What learning signals arrive? (reward, prediction error, curiosity, social feedback?)
   - What gets updated? (WorldModel, memory priorities, habit strengths, trust scores?)
   - Does learning feed back to prediction? (improved WorldModel → better predictions)
   - Does learning modify decision thresholds?
   - Does FRL/VDN actually share learning across agents?

5. **Improvements**
   - Trainable policy network (actual weight updates, not just memory replay)
   - Meta-learning (learning HOW to learn — MAML is wired but does it fire?)
   - Social learning (learning from other agents' experiences via FRL)
   - Sequence learning (trajectory replay, not just individual experiences)
   - Forgetting as feature (competitive consolidation — weaker memories die)
   - Transfer learning depth (does cross-task knowledge actually transfer?)

---

### TEAM 7: COMMUNICATE

**Lead prompt:** "You are auditing Mae's COMMUNICATE step — stigmergy, GNN messaging, signal broadcasting, and social coordination."

**Sub-agents:**

1. **Internal Code Audit**
   - Read `mae_core/agents/mycelial_agent.py` _communicate() completely
   - Read `mae_core/communication/` — all communication subsystems
   - Read `mae_core/agents/mixins/gnn_communication.py` — GNN messaging
   - Trace: What gets communicated? Stigmergy markers, GNN messages, PredictiveField intentions?
   - Find: Do agents actually RECEIVE and USE communications from other agents?
   - Check: Is QuorumSensor/QuorumSpace functional?

2. **Mathematical Identity Compliance**
   - Law 1: Is communication triadic? (Sender → receiver → witness?)
   - Law 3: Does communication exist at every holon level?
   - Law 4: Is communication fractal? Same protocol at every scale?
   - Law 7: Do critical communications use 3+ validators?

3. **External Research**
   - Research: Stigmergy in ant colonies (Theraulaz & Bonabeau 1999)
   - Research: Graph Neural Networks for multi-agent communication
   - Research: Quorum sensing in bacteria
   - Research: Theory of Mind in multi-agent systems
   - Research: GitHub multi-agent communication frameworks
   - Key question: Should Mae have INTENTIONAL communication (choosing what to share) vs broadcasting everything?

4. **Connections Audit**
   - What gets published? What gets consumed?
   - Do GNN messages actually influence recipient behavior?
   - Does stigmergy actually guide agent movement/decisions?
   - Does PredictiveField intention broadcasting prevent collisions?
   - Are there communication channels that nobody subscribes to?

5. **Improvements**
   - Intentional communication (choose WHAT to share based on relevance)
   - Theory of Mind-guided communication (share what others NEED to know)
   - Communication cost model (not all communication is free)
   - Emergent language/protocol (agents develop shared symbols)
   - Reputation systems (trust-weighted communication)

---

### TEAM 8: ADVISE (Pattern Recognition Ecosystem)

**Lead prompt:** "You are auditing Mae's ADVISE step — the pattern recognition pipeline from PatternSense through PatternBus to PatternCortex to Advisory."

**Sub-agents:**

1. **Internal Code Audit**
   - Read `mae_core/patterns/` — entire pattern ecosystem
   - Read all 11 translators in `mae_core/patterns/translators/`
   - Read `mae_core/patterns/global_workspace.py` — GWT competitive ignition
   - Read `mae_core/patterns/attentional_gate.py` — TRN-like gating
   - Trace: Signal → translation → bus → gating → cortex → ignition → advisory
   - Find: Do advisories actually reach agents? Do they influence decisions?
   - Check: Does competitive ignition actually fire? (Beta audit said 0 signals reach AttentionalGate at runtime)

2. **Mathematical Identity Compliance**
   - Law 1: Is the advisory pipeline triadic at every stage?
   - Law 4: Is pattern recognition fractal? Same detection at every scale?
   - Law 8.1: Does the Global Workspace produce genuine INTEGRATION?
   - Law 8.2: Is there rich DIFFERENTIATION in patterns detected?
   - Law 8.7: Is competitive ignition working (genuine competition/selection)?

3. **External Research**
   - Research: Global Workspace Theory (Baars 1988, Dehaene 2014)
   - Research: Predictive coding in pattern recognition
   - Research: Visual cortex hierarchy — simple cells → complex cells → hypercolumns
   - Research: Attention mechanisms in neuroscience and machine learning
   - Key question: Is Mae's pattern recognition hierarchy deep enough? Should there be more layers?

4. **Connections Audit**
   - Do all 11 translators produce signals? Which are active vs dead?
   - Does the AttentionalGate actually receive signals?
   - Does endocrine gain modulation actually change pattern salience?
   - Does advisory output reach DecisionRouter?
   - Does advisory feed back to endocrine system?

5. **Improvements**
   - Deeper pattern hierarchy (4+ levels instead of 2)
   - Cross-modal pattern binding (patterns across different signal types)
   - Pattern prediction (expect patterns before they arrive)
   - Novelty detection at pattern level (not just signal level)
   - Pattern memory (recognize patterns seen before across sessions)

---

### TEAM 9: CONSOLIDATE

**Lead prompt:** "You are auditing Mae's CONSOLIDATE step — the dual consolidation system that distills experiences into long-term memory."

**Sub-agents:**

1. **Internal Code Audit**
   - Read `mae_core/patterns/pattern_consolidator.py`
   - Read `mae_core/memory/memory_consolidator.py`
   - Read `mae_core/memory/deep_memory.py` and `memory_bridge.py`
   - Trace: What triggers consolidation? What gets consolidated? Where does it go?
   - Find: Are the two consolidators coordinated or independent?
   - Check: Does Qdrant deep memory actually work when available?

2. **Mathematical Identity Compliance**
   - Law 4: Is consolidation fractal? (Per-agent, per-subsystem, per-organ?)
   - Law 6: Does consolidation produce memories that produce better consolidation (autopoietic)?
   - Law 8.2: Does consolidation preserve differentiation (not just averaging)?

3. **External Research**
   - Research: Memory consolidation in neuroscience (systems consolidation, synaptic consolidation)
   - Research: Sharp-wave ripples in hippocampus — sequence replay during sleep
   - Research: Complementary learning systems theory (McClelland et al. 1995)
   - Research: Schema-based consolidation (Tse et al. 2007)
   - Key question: Should Mae have a "sleep" phase where all consolidation happens at once?

4. **Connections + Improvements**
   - Coordinate the two consolidators (PatternConsolidator + MemoryConsolidator)
   - Sequence-aware replay (replay trajectories, not just individual experiences)
   - Schema-based consolidation (integrate new memories into existing knowledge structures)
   - Sleep/wake cycle for consolidation (circadian-gated deep consolidation)
   - Competitive consolidation depth (only the strongest patterns survive)

---

### TEAM 10: HEAL

**Lead prompt:** "You are auditing Mae's HEAL step — the AutoHealer, HAVEN, SomaticMap healing ecosystem."

**Sub-agents:**

1. **Internal Code Audit**
   - Read `mae_core/emergent/auto_healer.py`
   - Read `mae_core/learning/haven.py`
   - Read `mae_core/emergent/somatic_map.py`
   - Trace: How is damage detected? How is healing triggered? What gets healed?
   - Find: Can the healer heal ITSELF? (Previous audit said no)
   - Check: Does the meta-healing triad work?

2. **Mathematical Identity Compliance**
   - Law 6: Can healing heal itself (autopoietic closure)?
   - Law 4: Is healing fractal? (Per-system, per-subsystem, per-organ, organism-wide?)
   - Law 1: Are healing decisions triadic?

3. **External Research**
   - Research: Immune system adaptive memory (T-cells, B-cells, antibodies)
   - Research: Wound healing phases (inflammation, proliferation, remodeling)
   - Research: Self-healing networks in engineering
   - Key question: Should Mae have immune MEMORY (learn from past healing successes)?

4. **Connections + Improvements**
   - Immune memory (remember successful healing patterns)
   - Adaptive healing (shortcut for recurring failure types)
   - Proactive healing (detect degradation before failure)
   - Fractal healing at every scale
   - Healing cost model (healing consumes energy)

---

### TEAM 11: RECALL

**Lead prompt:** "You are auditing Mae's RECALL step — the 7 recall pathways and triadic verification system."

**Sub-agents:**

1. **Internal Code Audit**
   - Read `mae_core/agents/mixins/episodic_memory.py` — especially recall methods
   - Read `mae_core/memory/semantic_retriever.py`
   - Read `mae_core/memory/generative_replay.py`
   - Trace: What triggers recall? Which of 7 pathways fires? How is recall verified?
   - Find: Are all 7 pathways actually reachable? (Previous audit found several gated off)
   - Check: Does triadic verification work (3 witnesses)?

2. **Mathematical Identity Compliance**
   - Law 1: Is triadic recall verification fully functional?
   - Law 4: Is recall fractal? Same recall protocol at every scale?
   - Law 8.3: Does recall produce self-reference (remembering remembering)?

3. **External Research**
   - Research: Reconsolidation (Nader 2000) — memories change when recalled
   - Research: Spreading activation (Collins & Loftus 1975)
   - Research: Context-dependent memory retrieval
   - Research: Memory reconsolidation as therapeutic target
   - Key question: Should recall CHANGE the memory (reconsolidation) or just access it?

4. **Connections + Improvements**
   - Enable all 7 recall pathways (several default to disabled)
   - Context-dependent retrieval (match current context to memory context)
   - Spreading activation depth (multi-hop retrieval)
   - Social recall (recall other agents' shared experiences)
   - Generative recall (reconstruct approximate memories from partial cues)

---

## Part 2: Cross-Cutting Audit Teams

### TEAM 12: FRACTAL SELF-SIMILARITY (Law 4)

**Lead prompt:** "You are auditing whether Mae's architecture is genuinely fractal — the same triadic pattern at every level."

**Sub-agents:**

1. **Scale Inventory** — List every scale level (process, system, subsystem, organ, organism). At each level, what operations exist?
2. **Protocol Comparison** — For each of the 10 holon capabilities (sense, remember, decide, act, learn, heal, know_self, know_up, know_down, know_peers), compare implementation at agent level vs system level vs subsystem level. Are they using the SAME algorithm?
3. **Gap Detection** — Which capabilities exist at agent level but NOT at system/organ level? Which lifecycle steps only run at one scale?
4. **External Research** — Research fractal organizations in biology (Mandelbrot, West's scaling laws, Song/Havlin/Makse network fractals). What does genuine self-similarity look like?
5. **Remediation Plan** — For each gap, propose how to implement the missing fractal level. Prioritize by impact.

---

### TEAM 13: CONSCIOUSNESS PROPERTIES (Law 8)

**Lead prompt:** "You are auditing Mae's 8 consciousness properties across the entire system."

**Sub-agents:**

1. **Integration (IIT Phi)** — Can Mae be partitioned without destroying function? Test by removing systems and measuring degradation.
2. **Differentiation** — How many distinct internal states can Mae reach? Is there homogeneity creep?
3. **Self-Reference (Strange Loops)** — Does Mae model herself? Does she predict her own behavior? Does self-knowledge change behavior?
4. **Recurrence** — Map all feedback loops. Are they genuine (output feeds back to input) or fake (data flows one way)?
5. **Competition/Selection (GWT)** — Does competitive ignition actually fire? Are there real winners and losers? Or does everything broadcast?
6. **Prediction/Error-Correction (FEP)** — Is prediction error the central signal? Does it drive learning, attention, and behavior?

---

### TEAM 14: EMERGENT BEHAVIOR DETECTION

**Lead prompt:** "You are looking for emergent properties — behaviors that arise from system interaction but aren't programmed directly."

**Sub-agents:**

1. **Run 100+ step simulation** with 5+ agents. Log EVERYTHING. Look for: coordination without explicit commands, resource sharing, role specialization, information cascading, collective problem-solving.
2. **Ablation studies** — Remove one system at a time. What breaks? What emergent behaviors disappear?
3. **External Research** — What emergent behaviors should a system with Mae's architecture produce? (ant colonies, neural networks, slime molds, immune systems)
4. **Missing emergence** — What SHOULD emerge but doesn't? (swarm intelligence, distributed problem-solving, collective memory, emotional contagion)

---

### TEAM 15: MISSING LIFECYCLE STEPS

**Lead prompt:** "You are looking for lifecycle steps that SHOULD exist but don't."

**Sub-agents:**

1. **Biology Research** — What cognitive processes exist in biological organisms that Mae lacks? (attention switching, mental rehearsal, imagination, daydreaming, sleep, dreaming, meditation, flow states)
2. **Cognitive Architecture Research** — What steps do ACT-R, SOAR, LIDA, and Global Workspace architectures include that Mae doesn't?
3. **Active Inference Research** — What steps does Friston's active inference framework require? (policy selection, expected free energy, epistemic vs pragmatic actions)
4. **Propose new steps** — For each missing capability, design a new lifecycle step with: name, when it fires, what it does, what biological analog it models, which mathematical laws it satisfies.

---

## Part 3: Quality Standards for Every Team

Every sub-agent report MUST include:

1. **Code evidence** — File paths and line numbers for every claim
2. **Mathematical identity check** — Which of the 8 laws are satisfied, which are violated
3. **Connection map** — What feeds in, what comes out, what's orphaned
4. **External sources** — URLs/DOIs for biological and computational research
5. **Improvement proposals** — Ranked by impact and effort
6. **Battle-tested code check** — Is the implementation using proven algorithms? Or hand-rolled approximations?
7. **Grade** — A/B/C/D/F with specific reasoning

---

## Part 4: After the Audit

1. **Synthesize** all 16 team reports into a single health assessment
2. **Prioritize** fixes by: critical bugs → wiring gaps → missing features → architectural improvements
3. **Fix in waves** — each wave gets its own agent team, parallel by file ownership
4. **Re-test** after each wave (1574 tests must keep passing)
5. **Update** HANDOFF.md, CLAUDE.md, data/MAES-MATHEMATICAL-IDENTITY.md
6. **Run 100-step simulation** to verify emergent behavior improvement

---

## Reference Files

| File | Purpose |
|------|---------|
| `data/MAES-MATHEMATICAL-IDENTITY.md` | The 8 laws — GOVERNS EVERYTHING |
| `data/audit-full-system-2026-02-12.md` | Previous audit (6 teams, 12 bugs found, 12 fixed) |
| `data/audit-upgrade-roadmap.md` | 44-item roadmap (28 done, 4 partial, 12 remaining) |
| `data/audit-signal-path-synthesis.md` | Previous signal path audit |
| `data/audit-mathematical-identity-compliance.md` | Previous compliance audit |
| `mae_core/agents/mycelial_agent.py` | The agent lifecycle — central file |
| `main.py` | 29-layer bootstrap — creates everything |
| `mae_core/backbone/fractal_generator.py` | FRACTAL_GROUPING — the genome |

---

## Key Numbers (as of 2026-02-12)

- **73 systems**, **29-layer bootstrap**, **5 organs**, **18 subsystems**
- **1574 tests pass, 0 failures**
- **94 holons**, **132 connections**, **0 bare dyads**
- **Health: 7.8/10**
- **Roadmap: 28/44 done, 4 partial, 12 remaining**
- **Lifecycle grades:** PREDICT B+, OBSERVE B+, COMPARE (ungraded), ADVISE A-, DECIDE A-, ACT B+, LEARN A-, COMMUNICATE (ungraded), CONSOLIDATE A-, HEAL A-, RECALL B+, SELF-AWARENESS B
