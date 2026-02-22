# Cognitive Architecture Gap Analysis

**Date:** 2026-02-12
**Purpose:** Identify specific cognitive processes from ACT-R, SOAR, LIDA, Global Workspace Theory, and Pandemonium Architecture that Mae's current architecture lacks.
**Method:** External research via web search, cross-referenced against Mae's current 73 systems, 12-step lifecycle, and 29-layer bootstrap.

---

## Executive Summary

Mae implements several consciousness principles but lacks critical cognitive architecture components for symbol manipulation, goal management, learning from impasses, and hierarchical competitive processing. The gaps cluster in four areas:

1. **Explicit symbolic reasoning** (ACT-R declarative memory, SOAR semantic memory)
2. **Goal stack management and impasse detection** (ACT-R goal buffer, SOAR subgoaling)
3. **Competitive selection mechanisms** (LIDA attention codelets, Pandemonium demon hierarchy)
4. **Structured working memory buffers** (ACT-R imaginal/temporal modules, GWT workspace limits)

---

## ACT-R Components Analysis

### 1. Declarative Memory Retrieval with Activation Spreading

**Source:** ACT-R (Anderson & Lebiere, Carnegie Mellon)

**Function:** Retrieves chunks from long-term declarative memory based on activation spreading from working memory cues. Activation combines base-level activation (frequency/recency), spreading activation from context, and partial matching similarity.

**Formula:** Activation(i) = BaseLevel(i) + Σ_j W_j * S_ji + noise

**Mae's Closest Match:**
- `EpisodicMemory` with spreading activation (added in refinement wave)
- `SemanticRetriever` with Qdrant vector similarity
- `PatternCortex` temporal recall

**What's Missing:**
- **Chunk-based symbolic memory** — ACT-R uses discrete chunks with slot-value pairs, Mae uses embeddings
- **Partial matching** — retrieval with mismatch penalty when no exact match exists
- **Base-level learning equation** — B_i(t) = ln(Σ_j t_j^(-d)) where t_j are practice times, d is decay rate
- **Source activation** — working memory elements send activation weighted by W (attention parameter)
- **Associative fan effect** — activation diluted when chunks have many associations (S_ji = S - ln(fan_j))

**Criticality:** **MEDIUM** — Mae has spreading activation but lacks symbolic chunk retrieval with mathematical activation dynamics. Current vector similarity doesn't implement frequency/recency decay or associative fan effects.

**Citations:**
- [ACT-R Wikipedia](https://en.wikipedia.org/wiki/ACT-R)
- [Integrated Computational Framework for Neurobiology of Memory](https://link.springer.com/article/10.1007/s42113-023-00189-y)

---

### 2. Goal Stack Management

**Source:** ACT-R Goal Module

**Function:** Goal buffer holds current control state (what the model is doing, where it is in the task). Goals can push/pop in a stack structure for nested task execution. Maps to anterior cingulate cortex.

**Mae's Closest Match:**
- `DecisionRouter` tracks decision type (reflex/habit/prefrontal)
- `WorldlinePlanner` maintains action sequences
- `OrganismState` integration hub

**What's Missing:**
- **Explicit goal stack data structure** — no push/pop goal nesting
- **Goal-driven production selection** — ACT-R productions match against goal buffer state
- **Goal completion detection** — automatic pop when goal satisfied
- **Subgoal creation mechanism** — procedural rules can create subgoals

**Criticality:** **HIGH** — Mae lacks hierarchical goal decomposition. WorldlinePlanner has sequences but no nested goal structures. Critical for complex multi-step reasoning.

**Citations:**
- [ACT-R Reference Manual](https://act-r.psy.cmu.edu/actr6/reference-manual.pdf)
- [ACT-R Wikipedia](https://en.wikipedia.org/wiki/ACT-R)

---

### 3. Conflict Resolution with Utility Learning

**Source:** ACT-R Production System

**Function:** When multiple productions match current buffer state, utility equation selects highest-value production: U(p) = PG - C + noise. Utility learned via reward feedback: U_i(n+1) = U_i(n) + α[R_i(n) - U_i(n)].

**Mae's Closest Match:**
- `DecisionRouter` 3-tier routing (reflex/habit/prefrontal)
- `TDLearning` value estimation
- `CuriosityDrive` intrinsic motivation

**What's Missing:**
- **Production-level utilities** — Mae doesn't have explicit production rules with learned utilities
- **Expected gain (PG)** and **cost (C)** decomposition
- **Conflict set computation** — identifying all matching productions before selection
- **Utility learning from outcomes** — production utilities update based on whether they led to reward

**Criticality:** **MEDIUM** — Mae has TD learning but not production-level utility. Would enable learning which decision patterns are valuable.

**Citations:**
- [ACT-R Wikipedia](https://en.wikipedia.org/wiki/ACT-R)
- [ResearchGate ACT-R Paper](https://www.researchgate.net/publication/329493100_ACT-R_A_cognitive_architecture_for_modeling_cognition)

---

### 4. Temporal Module

**Source:** ACT-R Temporal Module

**Function:** Tracks time intervals, estimates durations, generates timed responses. Allows productions to fire at specific times or after delays.

**Mae's Closest Match:**
- `CircadianRhythm` ultradian/circadian cycles
- `current_step` tracking
- `PatternCortex` 13-step temporal window

**What's Missing:**
- **Explicit time estimation** — learning clock intervals (e.g., "wait 500ms")
- **Timed production firing** — rules that trigger after specific delays
- **Interval discrimination** — comparing durations
- **Weber's Law implementation** — temporal precision degrades with longer intervals

**Criticality:** **LOW** — Mae has rhythm systems but lacks explicit interval timing. Relevant for modeling human reaction times and temporal reasoning.

**Citations:**
- [ACT-R Reference Manual](https://act-r.psy.cmu.edu/actr6/reference-manual.pdf)

---

### 5. Imaginal Module (Mental Workspace)

**Source:** ACT-R Imaginal Buffer

**Function:** Scratch pad for holding and manipulating internal representations. Used for problem-solving, planning, mental imagery. Separate from declarative memory retrieval.

**Mae's Closest Match:**
- `WorldModel` simulation
- `PatternCortex` working memory
- `OrganismState` integration hub

**What's Missing:**
- **Structured slot-value manipulation** — ACT-R imaginal can modify specific slots incrementally
- **Imaginal delay parameter** — creating/modifying chunks takes time (200-500ms)
- **Separation from retrieval** — retrieval buffer is read-only, imaginal is read-write workspace
- **Chunk construction** — building new symbolic structures from parts

**Criticality:** **MEDIUM** — Mae simulates futures but lacks structured symbolic manipulation workspace. Would enable explicit mental arithmetic, spatial reasoning.

**Citations:**
- [ACT-R Reference Manual](https://act-r.psy.cmu.edu/actr6/reference-manual.pdf)
- [ACT-R Wikipedia](https://en.wikipedia.org/wiki/ACT-R)

---

### 6. Buffer System Architecture

**Source:** ACT-R Core Architecture

**Function:** All modules accessed only through buffers. Buffer state = cognitive state. Productions match buffer patterns and request module actions.

**Mae's Closest Match:**
- `EventBus` publish-subscribe
- `SignalBus` signal routing
- `OrganismState` integration hub

**What's Missing:**
- **Strict buffer-based access control** — Mae systems directly call each other, ACT-R enforces buffer mediation
- **Buffer capacity limits** — each buffer holds 1 chunk at a time (forces serialization)
- **Production-buffer binding** — productions explicitly match against buffer contents in condition
- **Module requests via buffer modifications** — changing buffer state requests module action

**Criticality:** **LOW** — Architectural difference. Mae uses event bus, ACT-R uses buffers. Both achieve modular communication but different constraints.

**Citations:**
- [ACT-R Wikipedia](https://en.wikipedia.org/wiki/ACT-R)
- [ResearchGate ACT-R Paper](https://www.researchgate.net/publication/329493100_ACT-R_A_cognitive_architecture_for_modeling_cognition)

---

## SOAR Components Analysis

### 7. Elaboration Phase

**Source:** SOAR Cognitive Architecture (Laird, University of Michigan)

**Function:** Before each decision, elaborate working memory by firing all matching rules until quiescence (no more rules fire). Inference loop before action selection.

**Mae's Closest Match:**
- `_observe()` → `_compare()` → `_decide()` pipeline
- `ConsensusModule` quorum sensing
- `ThalamicGate` triage

**What's Missing:**
- **Rule quiescence detection** — firing rules until fixpoint reached
- **Parallel rule matching** — all matching productions fire simultaneously (vs ACT-R's serial conflict resolution)
- **Working memory augmentation** — elaboration adds inferences to working memory before decision
- **O-supported vs I-supported** — distinction between operator-created and inference-created WM elements

**Criticality:** **MEDIUM** — Mae has inference (OBSERVE/COMPARE) but not parallel rule saturation. Would enable richer pre-decision context.

**Citations:**
- [SOAR Wikipedia](https://en.wikipedia.org/wiki/Soar_(cognitive_architecture))
- [Introduction to SOAR](https://arxiv.org/pdf/2205.03854)

---

### 8. Impasse Detection and Subgoaling

**Source:** SOAR Impasse Mechanism

**Function:** When operator selection fails (tie, conflict, no-change), SOAR detects impasse, creates substate with new goal to resolve it. Four impasse types: tie (multiple operators equal), conflict (mutually exclusive operators), no-change (operator doesn't progress), rejection (all operators rejected).

**Mae's Closest Match:**
- `DecisionRouter` escalation to prefrontal when habit fails
- `AdaptationEngine` learning from failure

**What's Missing:**
- **Automatic impasse detection** — SOAR detects operator selection failure and triggers subgoaling automatically
- **Substate creation** — new working memory context for resolving impasse
- **Impasse type classification** — tie/conflict/no-change/rejection
- **Universal subgoaling** — any decision failure becomes a new goal to solve

**Criticality:** **HIGH** — Mae lacks automatic problem decomposition via impasses. When stuck, Mae doesn't automatically create subgoals to resolve the blockage. Critical for robust problem-solving.

**Citations:**
- [SOAR Wikipedia](https://en.wikipedia.org/wiki/Soar_(cognitive_architecture))
- [Introduction to SOAR](https://arxiv.org/pdf/2205.03854)

---

### 9. Chunking (Learning from Impasse Resolution)

**Source:** SOAR Chunking Mechanism

**Function:** When substate resolves impasse and produces result, SOAR compiles that problem-solving trace into a new production rule. Future encounters of same situation directly apply learned rule, avoiding impasse. Converts deliberation into automatic skill.

**Mae's Closest Match:**
- `HabitFormation` reflex→habit promotion
- `MAMLLearner` meta-learning
- `MemoryConsolidator` offline pattern extraction

**What's Missing:**
- **Explanation-based learning** — analyzing WHY subgoal succeeded to generalize rule
- **Dependency analysis** — determining which working memory elements caused result
- **Automatic rule compilation** — generating production from successful trace
- **Overgeneral chunking problem** — managing when learned rules are too broad

**Criticality:** **HIGH** — Mae consolidates patterns but doesn't compile problem-solving traces into executable rules. Chunking is SOAR's core learning mechanism.

**Citations:**
- [SOAR Wikipedia](https://en.wikipedia.org/wiki/Soar_(cognitive_architecture))
- [Introduction to SOAR](https://arxiv.org/pdf/2205.03854)

---

### 10. Episodic and Semantic Memory with Spreading Activation

**Source:** SOAR Long-Term Memories (EPMEM/SMEM)

**Function:**
- **Episodic (EPMEM):** Automatic recording of working memory snapshots, retrieval via cue-based query or temporal "previous/next" navigation
- **Semantic (SMEM):** Symbolic fact storage with spreading activation. Activation spreads from WM cues through associative links, biasing retrieval toward contextually relevant facts.

**Mae's Closest Match:**
- `EpisodicMemory` with reconsolidation and spreading activation
- `SemanticRetriever` with Qdrant
- Automatic episodic recording already implemented

**What's Missing:**
- **Symbolic semantic memory** — SMEM uses predicate logic facts (e.g., ^color red, ^size large), Mae uses embeddings
- **Explicit spreading activation paths** — SMEM activation follows typed links (attribute, superclass, etc.)
- **Temporal episodic queries** — "next/previous episode after X"
- **Episodic/semantic integration** — SMEM facts retrieved into WM can cue EPMEM recall

**Criticality:** **MEDIUM** — Mae has both memory types but SOAR's symbolic structure enables precise reasoning. Spreading activation exists in Mae but on embeddings not symbolic links.

**Citations:**
- [SOAR Wikipedia](https://en.wikipedia.org/wiki/Soar_(cognitive_architecture))
- [Introduction to SOAR](https://arxiv.org/pdf/2205.03854)

---

### 11. Look-Ahead Planning

**Source:** SOAR Mental Simulation

**Function:** Create internal copy of working memory, mentally apply operators, evaluate outcomes before committing. Multi-step lookahead via recursive substates.

**Mae's Closest Match:**
- `WorldModel` predictive simulation
- `WorldlinePlanner` action sequences
- `CollectiveDreamPlanner` offline planning

**What's Missing:**
- **Internal WM copy mechanism** — SOAR duplicates state to simulate without affecting real state
- **Recursive mental simulation** — planning within planning (subgoals during lookahead)
- **Rollback on failure** — simulated actions don't affect real world
- **Explicit lookahead depth control** — parameter for how many steps to simulate

**Criticality:** **LOW** — Mae has predictive simulation. SOAR's recursive mental substates are more structured but functionality overlaps.

**Citations:**
- [Introduction to SOAR](https://arxiv.org/pdf/2205.03854)

---

## LIDA Components Analysis

### 12. Attention Codelets and Competition

**Source:** LIDA Cognitive Architecture (Franklin, Baars)

**Function:** Attention codelets are mini-agents that each select a portion of the situational model and compete for conscious access. Each codelet has activation level. High-activation codelets form coalitions. Coalitions compete; winner broadcasts to global workspace.

**Mae's Closest Match:**
- `GlobalWorkspaceIntegrator` competitive ignition
- `ThalamicGate` sensory triage
- `ConsensusModule` quorum decisions

**What's Missing:**
- **Codelet architecture** — independent mini-agents with their own activation dynamics
- **Coalition formation** — codelets cooperate to strengthen their combined bid
- **Explicit competition function** — activation-based tournament for conscious access
- **Parallel codelet execution** — all codelets run simultaneously, compete asynchronously

**Criticality:** **HIGH** — Mae has GWT broadcast but lacks competitive codelet layer. LIDA's codelets provide parallel hypothesis generation and selection. Current implementation doesn't have independent agents competing for attention.

**Citations:**
- [LIDA Wikipedia](https://en.wikipedia.org/wiki/LIDA_(cognitive_architecture))
- [Franklin & Baars LIDA Tutorial](https://www.researchgate.net/publication/301760660_A_LIDA_cognitive_model_tutorial)

---

### 13. Structure Building Codelets

**Source:** LIDA Workspace Codelets

**Function:** Specialized codelets that construct higher-level representations from perceptual input. Build semantic structures (objects, events, relationships) in workspace. Distinct from attention codelets.

**Mae's Closest Match:**
- `PatternCortex` pattern extraction
- `WorldModel` state representation
- `MetacognitionMonitor` tracking decision quality

**What's Missing:**
- **Explicit structure-building agents** — dedicated codelets that construct symbolic representations
- **Separation from attention** — structure-building happens before attentional competition
- **Incremental structure growth** — codelets add nodes/links to workspace graph
- **Perceptual symbol system** — grounded symbols built from sensory features

**Criticality:** **MEDIUM** — Mae extracts patterns but lacks explicit symbolic structure construction. LIDA builds semantic graphs from perception.

**Citations:**
- [LIDA Wikipedia](https://en.wikipedia.org/wiki/LIDA_(cognitive_architecture))
- [LIDA Systems-Level Architecture](https://digitalcommons.memphis.edu/cgi/viewcontent.cgi?article=1030&context=ccrg_papers)

---

### 14. Conscious Broadcast Mechanism

**Source:** LIDA Implementation of Global Workspace Theory

**Function:** Winning coalition broadcasts its contents to all unconscious processors. Broadcast is time-limited (~200ms). Receiving systems use broadcast to update their processing. Implements Baars' GWT with explicit timing.

**Mae's Closest Match:**
- `GlobalWorkspaceIntegrator` with ignition threshold
- `EventBus` cross-system broadcasting
- `GWT_CAPACITY_LIMIT = 7` working memory chunks

**What's Missing:**
- **Explicit broadcast duration** — LIDA broadcasts last ~200ms (CONSCIOUS_INTERVAL parameter)
- **Broadcast consumption tracking** — which systems received and used the broadcast
- **Serial broadcast constraint** — only one broadcast at a time (prevents multiple ignitions)
- **Codelet spawning from broadcast** — broadcast triggers new codelets in receiving systems

**Criticality:** **LOW** — Mae has GWT broadcast. LIDA's timing is more explicit but core mechanism exists.

**Citations:**
- [LIDA Wikipedia](https://en.wikipedia.org/wiki/LIDA_(cognitive_architecture))
- [Timing of Cognitive Cycle](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0014803)

---

### 15. Behavior Networks

**Source:** LIDA Action Selection

**Function:** Behaviors are pre-compiled action schemes. Behavior network selects among competing behaviors based on context activation. Each behavior has preconditions, activation spreading from context, and action script.

**Mae's Closest Match:**
- `DecisionRouter` reflex/habit/prefrontal tiers
- `HabitFormation` learned reflexes
- `SignalPriorityResolver` triage

**What's Missing:**
- **Behavior activation spreading** — behaviors activated by contextual cues, compete via activation levels
- **Pre-compiled behavior library** — stored repertoire of action schemes
- **Behavior selection net** — network topology where behaviors inhibit/excite each other
- **Parallel behavior evaluation** — all behaviors compute activation simultaneously

**Criticality:** **MEDIUM** — Mae routes decisions by tier but lacks parallel behavior activation network. LIDA's behavior net enables context-sensitive action selection.

**Citations:**
- [LIDA Systems-Level Architecture](https://digitalcommons.memphis.edu/cgi/viewcontent.cgi?article=1030&context=ccrg_papers)

---

### 16. Deliberation Cycle and Metacognition Cycle

**Source:** LIDA Dual Cycle Architecture

**Function:**
- **Deliberation cycle:** ~300ms cycle of understanding → attention → action selection
- **Metacognition cycle:** Slower cycle monitoring deliberation quality, adjusting parameters, learning

**Mae's Closest Match:**
- 12-step lifecycle per agent per step
- `MetacognitionMonitor` tracking decision quality
- `CircadianRhythm` multi-timescale coordination

**What's Missing:**
- **Explicit cycle separation** — LIDA has fast (~300ms) and slow (seconds) cycles with different functions
- **Cycle phase triggers** — each phase completes before next begins (sequential)
- **Metacognitive parameter tuning** — slow cycle adjusts attention thresholds, learning rates based on performance
- **Cross-cycle coordination** — metacognition reads deliberation history, adjusts future cycles

**Criticality:** **MEDIUM** — Mae has lifecycle and metacognition but not explicit dual-cycle timing. LIDA's separation of fast deliberation and slow tuning enables adaptive control.

**Citations:**
- [LIDA Wikipedia](https://en.wikipedia.org/wiki/LIDA_(cognitive_architecture))
- [Timing of Cognitive Cycle](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0014803)

---

### 17. Perceptual Associative Memory

**Source:** LIDA PAM

**Function:** Long-term memory for perceptual entities and relationships. Activated by sensory input via spreading activation. Most-active nodes pass to workspace. Distinct from episodic/semantic memory.

**Mae's Closest Match:**
- `SemanticRetriever` Qdrant lookup
- `PatternCortex` pattern library
- `StigmergyEnvironment` environmental memory

**What's Missing:**
- **Perceptual grounding** — PAM stores perceptual symbols (visual features, spatial relationships), not abstract concepts
- **Activation-based selection** — top-K activated nodes pass to workspace
- **Rapid priming** — recent perceptions increase activation for related nodes
- **Separation from semantic facts** — PAM is perceptual, SMEM is conceptual

**Criticality:** **LOW** — Mae has retrieval systems. PAM's distinction is perceptual grounding, which Mae doesn't emphasize (no visual system).

**Citations:**
- [LIDA Systems-Level Architecture](https://digitalcommons.memphis.edu/cgi/viewcontent.cgi?article=1030&context=ccrg_papers)

---

## Global Workspace Theory Components

### 18. Global Ignition Mechanism

**Source:** Global Neuronal Workspace Theory (Dehaene)

**Function:** Non-linear ignition when cortical representation reaches threshold. Sudden, coherent, exclusive activation of workspace neurons. Feed-forward propagation → all-or-none global broadcast. Inhibits competing representations.

**Mae's Closest Match:**
- `GlobalWorkspaceIntegrator` with activation threshold
- Competitive ignition added in refinement wave
- TRN gating for inhibition

**What's Missing:**
- **Non-linear ignition dynamics** — sharp transition from local to global (sigmoid or step function)
- **All-or-none property** — representation either ignites fully or not at all (no partial broadcast)
- **Active inhibition of losers** — winning coalition suppresses competitors
- **Feed-forward propagation model** — explicit spreading dynamics (not just threshold check)

**Criticality:** **LOW** — Mae has threshold-based ignition. GNW's non-linear dynamics are more detailed but core mechanism exists.

**Citations:**
- [GWT Wikipedia](https://en.wikipedia.org/wiki/Global_workspace_theory)
- [Global Neuronal Workspace Model](https://www.antoniocasella.eu/dnlaw/Dehaene_Changeaux_Naccache_2011.pdf)
- [Conscious Processing and GNW Hypothesis](https://pmc.ncbi.nlm.nih.gov/articles/PMC8770991/)

---

### 19. Competition for Conscious Access

**Source:** Global Workspace Theory (Baars)

**Function:** Multiple unconscious processors compete to broadcast their contents. Competition based on signal strength, relevance, novelty. Only winner reaches global workspace. Losers remain unconscious.

**Mae's Closest Match:**
- `GlobalWorkspaceIntegrator` competitive selection
- `SignalPriorityResolver` triage
- `ThalamicGate` sensory filtering

**What's Missing:**
- **Explicit competition function** — mathematical formulation of how signals compete (activation levels, inhibition)
- **Multiple candidates in parallel** — all processors submit bids simultaneously
- **Graded competition** — near-winners can influence workspace even if they don't broadcast
- **Context-dependent competition** — relevance varies by task/goal state

**Criticality:** **MEDIUM** — Mae has triage but not full parallel competition. GWT's competitive dynamics enable flexible attention.

**Citations:**
- [GWT Wikipedia](https://en.wikipedia.org/wiki/Global_workspace_theory)
- [Frontiers GWT and Prefrontal Cortex](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2021.749868/full)

---

### 20. Workspace Capacity Limits

**Source:** Global Workspace Theory

**Function:** Workspace holds ~7±2 chunks (Miller's law). Limited capacity forces competition, serializes processing. Working memory bottleneck is architectural, not just resource constraint.

**Mae's Closest Match:**
- `GWT_CAPACITY_LIMIT = 7` in GlobalWorkspaceIntegrator
- Competitive selection when capacity exceeded

**What's Missing:**
- **Chunk displacement mechanism** — when capacity full, new chunks must displace old ones
- **Capacity enforcement across all systems** — GWT limit should apply to all working memory, not just one module
- **Recency/primacy effects** — newer chunks more likely to displace middle chunks
- **Chunking to compress capacity** — grouping elements to fit more in workspace

**Criticality:** **LOW** — Mae has capacity limit constant. Implementation could enforce displacement more rigorously but core constraint exists.

**Citations:**
- [GWT Wikipedia](https://en.wikipedia.org/wiki/Global_workspace_theory)
- [Global Neuronal Workspace Model](https://www.antoniocasella.eu/dnlaw/Dehaene_Changeaux_Naccache_2011.pdf)

---

### 21. Broadcasting to Unconscious Processors

**Source:** Global Workspace Theory

**Function:** Workspace broadcast reaches all unconscious processors (perception, memory, action systems). Processors use broadcast to guide their specialized processing. Enables global coordination without direct processor-to-processor communication.

**Mae's Closest Match:**
- `EventBus` system with cross-module subscriptions
- OrganismState integration hub

**What's Missing:**
- **Broadcast-only communication** — GWT enforces that systems communicate ONLY via workspace, Mae allows direct calls
- **Processor autonomy** — each system decides how to use broadcast (Mae's OrganismState explicitly routes signals)
- **Uniform broadcast format** — same content reaches all systems in GWT, Mae has typed channels
- **No central router** — GWT workspace is passive broadcast medium, Mae's OrganismState actively integrates

**Criticality:** **LOW** — Architectural difference. Mae's EventBus achieves coordination, GWT uses pure broadcast. Both work.

**Citations:**
- [GWT Wikipedia](https://en.wikipedia.org/wiki/Global_workspace_theory)

---

## Pandemonium Architecture Components

### 22. Demon Hierarchy (4 Levels)

**Source:** Pandemonium (Selfridge, 1959)

**Function:** Four-layer processing hierarchy:
1. **Image Demon:** Captures raw input
2. **Feature Demons:** Detect simple features (lines, curves, angles) in parallel
3. **Cognitive Demons:** Match feature combinations to learned patterns
4. **Decision Demon:** Selects winning cognitive demon based on loudest "shouting"

**Mae's Closest Match:**
- `PatternSense` multi-scale feature detection
- `PatternCortex` pattern library
- `DecisionRouter` action selection

**What's Missing:**
- **Explicit 4-level hierarchy** — Mae has sensors→patterns→decisions but not Pandemonium's specific architecture
- **Feature demon parallelism** — all feature demons run simultaneously on same input
- **Shouting metaphor** — demons activate proportional to match quality, "shout" to next level
- **Decision by loudest** — simple max-activation selection

**Criticality:** **LOW** — Mae has hierarchical pattern recognition. Pandemonium's specific 4-level structure is a design choice, not essential mechanism.

**Citations:**
- [Pandemonium Wikipedia](https://en.wikipedia.org/wiki/Pandemonium_architecture)
- [Selfridge Original Paper](https://gwern.net/doc/ai/nn/1959-selfridge.pdf)

---

### 23. Parallel Competitive Processing

**Source:** Pandemonium Architecture

**Function:** All demons at each level run in parallel. Compete by activation strength. No serial search — all features checked simultaneously. Winner-take-all selection at each level.

**Mae's Closest Match:**
- Parallel agent execution (Mesa model.step())
- `ConsensusModule` quorum voting
- `PatternCortex` parallel pattern matching

**What's Missing:**
- **True parallel feature detection** — Mae's agents step in sequence (Mesa schedule), Pandemonium is fully parallel
- **Winner-take-all at each level** — strongest feature demon passes to cognitive level, strongest cognitive demon wins
- **No backtracking** — once decision made, no revision (pure feedforward)
- **Continuous competition** — demons always running, always competing (not turn-based)

**Criticality:** **MEDIUM** — Mae has parallelism but Mesa's turn-based stepping isn't continuous parallel competition. Pandemonium's architecture enables faster pattern recognition.

**Citations:**
- [Pandemonium Wikipedia](https://en.wikipedia.org/wiki/Pandemonium_architecture)
- [Pandemonium Model](https://www.careershodh.com/pandemonium-model/)

---

### 24. Symbol Formation from Feature Detection

**Source:** Pandemonium Pattern Recognition

**Function:** Low-level features (lines, curves) combine via cognitive demons to form higher-level symbols (letters, objects). Symbolic output emerges from subsymbolic competition.

**Mae's Closest Match:**
- `PatternCortex` builds meta-patterns from base patterns
- Fractal pattern recognition across 4 scales

**What's Missing:**
- **Explicit symbol grounding** — Pandemonium connects visual features directly to discrete symbols (A, B, C...)
- **Symbol creation as output** — final output is discrete symbol, not activation pattern
- **Feature-symbol binding** — learned associations between feature combinations and symbols
- **Symbol stability** — once demon wins, output is crisp symbol (not probabilistic)

**Criticality:** **LOW** — Mae works with patterns and embeddings. Pandemonium's discrete symbols are useful for character recognition but not essential for Mae's embodied intelligence.

**Citations:**
- [Pandemonium Wikipedia](https://en.wikipedia.org/wiki/Pandemonium_architecture)
- [Pandemonium Model Evidences](https://www.careershodh.com/pandemonium-model/)

---

## Gap Summary by Criticality

### HIGH Criticality (Must Address)

| Component | Source | Impact |
|-----------|--------|--------|
| Goal Stack Management | ACT-R | No hierarchical goal decomposition, can't handle nested tasks |
| Impasse Detection & Subgoaling | SOAR | No automatic problem decomposition when stuck |
| Chunking (Learning from Impasse Resolution) | SOAR | No compilation of problem-solving traces into rules |
| Attention Codelets & Competition | LIDA | No parallel hypothesis generation with competitive selection |

**Why Critical:** Mae can execute plans but can't hierarchically decompose goals, detect when stuck, learn from problem-solving traces, or run parallel competitive hypothesis generation. These are core to flexible intelligence.

---

### MEDIUM Criticality (Should Address)

| Component | Source | Impact |
|-----------|--------|--------|
| Declarative Memory Retrieval with Activation | ACT-R | No chunk-based symbolic memory with mathematical activation dynamics |
| Conflict Resolution with Utility Learning | ACT-R | No production-level utility learning from outcomes |
| Imaginal Module (Mental Workspace) | ACT-R | No structured symbolic manipulation workspace |
| Elaboration Phase | SOAR | No parallel rule saturation before decisions |
| Episodic/Semantic Memory (Symbolic) | SOAR | Embeddings vs predicate logic facts |
| Structure Building Codelets | LIDA | No explicit symbolic structure construction from perception |
| Deliberation/Metacognition Dual Cycles | LIDA | No explicit fast/slow cycle separation |
| Behavior Networks | LIDA | No parallel behavior activation network |
| Competition for Conscious Access | GWT | No full parallel competition for workspace |
| Parallel Competitive Processing | Pandemonium | Turn-based stepping vs continuous parallel competition |

**Why Medium:** Mae has related capabilities (memory, learning, decision routing) but lacks specific mechanisms (symbolic chunks, utility per rule, dual cycles, parallel competition). Would enhance but not critical.

---

### LOW Criticality (Nice to Have)

| Component | Source | Impact |
|-----------|--------|--------|
| Temporal Module | ACT-R | Explicit interval timing vs rhythm systems |
| Buffer System Architecture | ACT-R | Architectural choice (buffers vs event bus) |
| Look-Ahead Planning | SOAR | Mae has WorldModel simulation |
| Conscious Broadcast Mechanism | LIDA | Mae has GWT broadcast, LIDA's timing more explicit |
| Perceptual Associative Memory | LIDA | Perceptual grounding (Mae has no visual system) |
| Global Ignition Mechanism | GWT | Mae has threshold ignition, GNW more detailed |
| Workspace Capacity Limits | GWT | Mae has capacity limit, could enforce displacement better |
| Broadcasting to Unconscious Processors | GWT | Architectural difference (broadcast vs event bus) |
| Demon Hierarchy | Pandemonium | Mae has hierarchical processing, different structure |
| Symbol Formation | Pandemonium | Embeddings vs discrete symbols |

**Why Low:** Mae has functionally similar capabilities. Differences are architectural choices or domain-specific (visual perception).

---

## Recommendations

### Immediate Priorities

1. **Implement Goal Stack** — Add `GoalManager` system with push/pop stack, nested goal execution, completion detection. Critical for complex reasoning.

2. **Add Impasse Detection** — When `DecisionRouter` can't select action (all options below threshold), trigger impasse handler. Create subgoal to resolve blockage.

3. **Implement Chunking** — When subgoal resolves impasse, compile trace into new production rule. Store in `HabitFormation` for future reuse.

4. **Build Attention Codelet Layer** — Between `PatternCortex` and `GlobalWorkspaceIntegrator`, add parallel codelet agents that compete for broadcast.

### Secondary Enhancements

5. **Symbolic Memory Layer** — Add chunk-based declarative memory alongside embeddings. Use slot-value pairs for explicit reasoning.

6. **Utility Learning per Decision Pattern** — Track which decision heuristics lead to reward. Integrate with `DecisionRouter` tier selection.

7. **Dual Cycle Timing** — Separate fast deliberation cycle (~10 steps) from slow metacognitive tuning cycle (~100 steps).

8. **Behavior Activation Network** — Parallel behavior library with spreading activation. Competes with `DecisionRouter` prefrontal path.

### Research Questions

- Can chunking be implemented on top of Mae's episodic memory? (SOAR compiles traces, Mae distills patterns)
- Should goal stack live in `DecisionRouter` or separate `GoalManager`?
- How to integrate symbolic chunks with embedding-based retrieval? (Hybrid memory)
- Can attention codelets be implemented as short-lived agents spawned per-decision?

---

## Sources

### ACT-R
- [ACT-R Wikipedia](https://en.wikipedia.org/wiki/ACT-R)
- [ACT-R About Page](https://act-r.psy.cmu.edu/about/)
- [ACT-R Reference Manual](https://act-r.psy.cmu.edu/actr6/reference-manual.pdf)
- [ACT-R 7 Reference Manual](http://act-r.psy.cmu.edu/actr7/reference-manual.pdf)
- [ResearchGate: ACT-R Cognitive Architecture](https://www.researchgate.net/publication/329493100_ACT-R_A_cognitive_architecture_for_modeling_cognition)
- [Integrated Computational Framework for Neurobiology of Memory](https://link.springer.com/article/10.1007/s42113-023-00189-y)
- [ACT-R pyactr Implementation](https://link.springer.com/chapter/10.1007/978-3-030-31846-8_2)

### SOAR
- [SOAR Wikipedia](https://en.wikipedia.org/wiki/Soar_(cognitive_architecture))
- [Introduction to SOAR Cognitive Architecture](https://arxiv.org/pdf/2205.03854)
- [SOAR Cognitive Architecture (MIT Press)](https://direct.mit.edu/books/monograph/2938/The-Soar-Cognitive-Architecture)
- [CogRec: Cognitive Recommender with SOAR](https://arxiv.org/html/2512.24113)

### LIDA
- [LIDA Wikipedia](https://en.wikipedia.org/wiki/LIDA_(cognitive_architecture))
- [LIDA: A Working Model of Cognition](https://www.researchgate.net/publication/28765131_LIDA_A_working_model_of_cognition)
- [LIDA Cognitive Model Tutorial](https://www.researchgate.net/publication/301760660_A_LIDA_cognitive_model_tutorial)
- [LIDA Systems-Level Architecture](https://digitalcommons.memphis.edu/cgi/viewcontent.cgi?article=1030&context=ccrg_papers)
- [The Mind According to LIDA](https://ccrg.cs.memphis.edu/tutorial/mindAccordingToLIDA/Brief-Account.pdf)
- [Timing of the Cognitive Cycle](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0014803)
- [Baars & Franklin: LIDA Model of Global Workspace Theory](https://bernardbaars.com/wp-content/uploads/2021/04/BaarsFranklin-ArchitecturalGWT-LIDA-NN2007.pdf)

### Global Workspace Theory
- [Global Workspace Theory Wikipedia](https://en.wikipedia.org/wiki/Global_workspace_theory)
- [Global Neuronal Workspace Model (Dehaene)](https://www.antoniocasella.eu/dnlaw/Dehaene_Changeaux_Naccache_2011.pdf)
- [Conscious Processing and GNW Hypothesis](https://pmc.ncbi.nlm.nih.gov/articles/PMC8770991/)
- [Frontiers: GWT and Prefrontal Cortex](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2021.749868/full)
- [Evolutionary Origins of GNW in Vertebrates](https://academic.oup.com/nc/article/2023/1/niad020/7272926)

### Pandemonium Architecture
- [Pandemonium Architecture Wikipedia](https://en.wikipedia.org/wiki/Pandemonium_architecture)
- [Selfridge's Original Paper (1959)](https://gwern.net/doc/ai/nn/1959-selfridge.pdf)
- [Pandemonium Model Evidences](https://www.careershodh.com/pandemonium-model/)
- [Pandemonium Tutorial](https://www.tutorialspoint.com/pandemonium-architecture)
- [Pandemonium's Friendly Demons](https://mindhacks.com/2021/01/29/pandemoniums-friendly-demons/)

---

**Next Steps:**
1. Present this analysis to Guiding Light
2. Prioritize which gaps to address based on Mae's roadmap
3. Design implementations that respect Mae's 8 Mathematical Laws
4. Research how to integrate symbolic reasoning (ACT-R chunks, SOAR rules) with Mae's embedding-based intelligence
