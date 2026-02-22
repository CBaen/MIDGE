# Mae Lifecycle Audit: Missing Steps Report

**Team 15: Missing Lifecycle Steps Auditor**
**Date:** 2026-02-12
**Auditor:** Claude Opus 4.6
**Sub-agents deployed:** 4 (Biology Research, Cognitive Architecture Research, Active Inference Research, Step Design)

---

## Executive Summary

Mae currently has **12 lifecycle steps** organized into 8 per-step and 4 cadenced. After cross-referencing against biological neuroscience, four major cognitive architectures (ACT-R, SOAR, LIDA, GWT), and Friston's Active Inference framework, this audit identifies **13 missing capabilities** that should become new lifecycle steps or sub-steps.

**Current Lifecycle Completeness Grade: C+**

Mae excels at reactive intelligence (sense-predict-compare-decide-act-learn) but has critical gaps in: executive control (inhibition), attention management, goal persistence, conscious broadcasting, and precision-weighted inference. These gaps mean Mae cannot suppress inappropriate actions, cannot maintain multi-step goals, has no formal attention mechanism, and treats all prediction errors equally regardless of reliability.

---

## Part 1: Current Lifecycle Architecture

### Per-Step (every agent tick, in order)
| # | Step | Function | Biological Analog |
|---|------|----------|-------------------|
| 1 | TRIAGE | Signal priority resolution | Thalamic gating |
| 2 | PREDICT | Generate prior expectation from WorldModel | Visual cortex top-down prediction |
| 3 | OBSERVE | Sense environment, stigmergy, interoception, self-awareness | Sensory cortices + insular cortex |
| 4 | COMPARE | Compute prediction error (surprise) | Mismatch negativity, dopamine RPE |
| 5 | DECIDE | 3-tier routing (reflex/habit/prefrontal) + memory + world model | Basal ganglia + prefrontal cortex |
| 6 | ACT | Execute action (explore/exploit/communicate/rest) | Motor cortex + musculoskeletal |
| 7 | LEARN | Store memory, TD-error, curiosity, world model training | Hippocampus + cerebellum |
| 8 | COMMUNICATE | GNN messages, stigmergy, pattern sharing | Social signaling, pheromones |

### Cadenced (periodic)
| # | Step | Cadence | Function |
|---|------|---------|----------|
| 9 | ADVISE | Per model step | Pattern cortex advisory generation |
| 10 | CONSOLIDATE | Every 89 steps | Memory consolidation (sleep analog) |
| 11 | HEAL | On failure detection | Auto-healing (isolate/assess/restore) |
| 12 | RECALL | Every 13 steps | Episodic memory replay |

---

## Part 2: Research Findings

### 2.1 Biology Research Findings

**Tier 1 Critical Gaps (violate Mae's Mathematical Identity):**

1. **Go/No-Go Inhibitory Control** -- Mae's DECIDE step MUST produce an action every tick. No mechanism exists to suppress responses. The right inferior frontal gyrus and basal ganglia indirect pathway mediate biological inhibition. Without this, Mae violates Law 8 Property 7 (competition/selection requires the ability to select "no action").
   - Sources: [PMC11117662](https://pmc.ncbi.nlm.nih.gov/articles/PMC11117662/), [PMC5747365](https://pmc.ncbi.nlm.nih.gov/articles/PMC5747365/)

2. **Default Mode Network / Mind-Wandering** -- Mae is always externally focused. Biological organisms spend ~50% of waking time in stimulus-independent thought (self-reflection, creative recombination, future planning). The medial prefrontal cortex, posterior cingulate, and precuneus form this network. Without DMN, Mae cannot generate novel solutions during idle periods.
   - Sources: [Default mode network](https://en.wikipedia.org/wiki/Default_mode_network)

3. **Distinct Sleep Stages (NREM1-3, REM)** -- Mae's 3 circadian phases (ACTIVE/CONSOLIDATION/REST) are too coarse. Biological sleep has distinct stages that handle different functions: NREM1-2 (synaptic homeostasis), NREM3 (declarative memory consolidation), REM (emotional integration, creative connections, procedural consolidation). Each requires different processing modes.
   - Sources: [PMID:S1074742723000928](https://www.sciencedirect.com/science/article/abs/pii/S1074742723000928), [Science/sciadv.adj1895](https://www.science.org/doi/10.1126/sciadv.adj1895)

**Tier 2 High-Impact Gaps:**

4. **Working Memory Management** -- Mae has no Baddeley-style working memory (central executive + phonological loop + visuospatial sketchpad + episodic buffer). Current memory is either immediate (current step state) or long-term (episodic storage). No active workspace for manipulating multiple information streams simultaneously.

5. **Selective vs. Diffuse Attention** -- Mae processes all signals identically regardless of task demands. No attention mode switching between focused (narrow, high-resolution) and diffuse (broad, associative) processing. Different tasks require fundamentally different processing modes.

6. **Prospective Memory** -- Mae has no mechanism to remember future intentions ("do X when Y happens" or "do X at step Z"). Plans exist but nothing triggers their execution at the appropriate future moment.

**Tier 3 Moderate Gaps:**

7. **Arousal Regulation (Yerkes-Dodson)** -- No optimal performance curve. EndocrineSystem provides hormones but no feedback control targeting optimal arousal for current task complexity.

8. **Habituation/Sensitization** -- Non-associative learning is absent. Mae cannot decrease responses to repeated benign stimuli or increase responses after harmful stimuli.

9. **Social/Observational Learning** -- Mae has collective consensus but no mechanism to learn from observing other agents' experiences without direct communication.

### 2.2 Cognitive Architecture Research Findings

**HIGH Criticality Missing Capabilities:**

| # | Capability | Architecture | What Mae Has Closest | What's Missing |
|---|-----------|-------------|---------------------|----------------|
| 1 | Goal Stack Management | ACT-R | OrganismState tracks current state | No multi-step intentions, no subgoal creation when blocked, no goal-directed action selection |
| 2 | Imaginal Buffer / Mental Workspace | ACT-R | WorldModel state representation | No editable problem workspace for mental rehearsal |
| 3 | Impasse Detection | SOAR | Morphogenesis gap signal at pred_error > 0.7 | No systematic "stuck" recognition (tie, conflict, no-change, rejection impasses) |
| 4 | Chunking from Problem-Solving | SOAR | Habit formation from repetitive outcomes | Cannot learn rules from impasse resolution episodes |
| 5 | Universal Subgoaling | SOAR | No subgoaling mechanism | Cannot break complex problems into manageable subproblems |
| 6 | Go/No-Go Inhibition | Cross-architecture | Always selects an action | Cannot learn that "doing nothing" is optimal |
| 7 | Goal-Directed Action Selection | Cross-architecture | Reward-driven behavior | No persistent intentions or goal-conditioned policies |

**MEDIUM Criticality:**

| # | Capability | Architecture | Gap |
|---|-----------|-------------|-----|
| 8 | Conflict Resolution via Utility | ACT-R | Cannot choose between competing valid options at same tier |
| 9 | Temporal Module / Time Perception | ACT-R | Cannot plan actions for specific future times |
| 10 | Conscious/Unconscious Distinction | LIDA/GWT | Learns from ALL experiences indiscriminately |
| 11 | Metacognitive Control | LIDA | MetacognitionMonitor exists but cannot redirect processing |

**Key Insight:** Mae excels at reactive intelligence but lacks the goal management and hierarchical problem decomposition that enable complex multi-step behavior. The fractal architecture provides structural foundations, but needs explicit goal representation and impasse-driven learning.

### 2.3 Active Inference Research Findings

**HIGH Criticality:**

1. **Expected Free Energy (EFE) for Policy Selection** -- Mae's DECIDE step uses tier-based matching, not predictive minimization of expected future surprise. In active inference, agents select policies by minimizing G = epistemic value + pragmatic value. Mae reacts to current prediction error rather than minimizing EXPECTED future surprise.
   - Sources: [Friston et al. 2017](https://activeinference.github.io/papers/process_theory.pdf), [2024 reframing](https://arxiv.org/pdf/2402.14460)

2. **Precision Optimization (Attention as Inference)** -- Mae computes raw prediction error with no precision weighting. In the brain, attention IS precision optimization: high precision means "trust this prediction error," low precision means "ignore it." Mae treats all prediction errors equally regardless of reliability.
   - Sources: [Feldman & Friston 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC2666703/), [Cerebral hierarchies](https://royalsocietypublishing.org/doi/10.1098/rstb.2014.0169)

3. **Epistemic vs. Pragmatic Action Decomposition** -- Mae's explore/exploit distinction is hardcoded at the policy level. In active inference, the balance between information-gathering (epistemic) and goal-achieving (pragmatic) actions EMERGES from EFE minimization -- it is not a separate design choice.
   - Sources: [Friston et al. 2015](https://www.sciencedirect.com/science/article/pii/S0149763416301336)

**MEDIUM Criticality:**

4. **Hierarchical Predictive Processing** -- Mae has one flat WorldModel. Biological brains have hierarchical generative models where each level predicts the level below. Predictions flow top-down, prediction errors flow bottom-up.
   - Sources: [Clark 2013](https://www.sciencedirect.com/topics/psychology/predictive-processing)

5. **Interoceptive Inference** -- Mae's OrganismState is OBSERVED but not PREDICTED. The brain predicts its own body states; prediction errors about body state drive autonomic regulation and generate emotions (emotions as interoceptive inference).
   - Sources: [Seth & Friston 2016](https://royalsocietypublishing.org/rstb/article/371/1708/20160007), [Barrett & Simmons 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC5062097/)

6. **Allostatic Regulation** -- Mae's HomeostasisRegulator uses fixed setpoints. Allostasis = setpoints that CHANGE based on predicted future context. The body anticipatorily adjusts baselines before demand arrives.

7. **Deep Temporal Models** -- Mae's WorldlinePlanner does multi-horizon planning but doesn't frame it as inference over deep temporal sequences with probabilistic evaluation.

**LOW Criticality:**

8. **Model Selection / Structure Learning** -- Mae has one fixed WorldModel architecture. Bayesian model comparison would let Mae learn WHICH model to use.
9. **Markov Blanket Dynamics** -- Mae's BoundaryMembrane is static, not dynamically inferred.

---

## Part 3: Proposed New Lifecycle Steps

Based on the converged findings from all four research streams, here are 13 proposed new steps, ranked by priority.

### P0: MUST HAVE (5 steps)

#### NEW STEP 1: INHIBIT (Go/No-Go Control)
- **Category:** Per-step
- **Position:** Between COMPARE and DECIDE (new step 5)
- **What It Does:** Evaluates whether any action should be taken this tick. Implements biological response inhibition through three channels: (1) conflict detection between competing signals, (2) prediction error magnitude check (extreme surprise = freeze), (3) executive override from goals/constraints. Can veto the entire DECIDE-ACT sequence, resulting in a "hold" state.
- **Biological Analog:** Basal ganglia indirect pathway (striatum -> GPe -> STN -> GPi), right inferior frontal gyrus, GABA interneurons. The "NoGo" pathway that must be actively overcome for action to proceed.
- **Mathematical Laws:**
  - Law 1: Inhibition signal -> Executive Context -> Action Permission (triadic)
  - Law 7: Minimum 3 inhibition sources (conflict, error threshold, constraint violation)
  - Law 8 Property 7: Competition/selection now includes selecting "no action"
- **Integration:** COMPARE outputs to both INHIBIT and DECIDE. INHIBIT can block DECIDE execution entirely. When inhibited, agent records the suppressed intention for future analysis.
- **Impact on Existing Steps:** DECIDE gains an inhibition-aware wrapper. If INHIBIT returns False, DECIDE/ACT are skipped but LEARN still fires (learning from inhibition is critical).
- **Priority:** P0 -- This is the #1 most-cited gap across all three research domains. Without it, Mae cannot be safe.

#### NEW STEP 2: ATTEND (Precision-Weighted Attention)
- **Category:** Per-step
- **Position:** Between TRIAGE and PREDICT (new step 2)
- **What It Does:** Computes precision weights (inverse variance) for each signal source. Implements attention as precision optimization per active inference theory. Selects between focused (narrow, high-precision) and diffuse (broad, low-precision) attention modes based on task demands and prediction error history. Outputs attention weights that modulate ALL downstream processing.
- **Biological Analog:** Thalamic reticular nucleus (TRN), pulvinar attention control, acetylcholine/norepinephrine neuromodulation. Three attention networks: alerting (arousal), orienting (selecting), executive (conflict resolution).
- **Mathematical Laws:**
  - Law 1: Signal Source -> Precision Estimate -> Attention Weight (triadic)
  - Law 3: Implements "sense" at the meta-level (sensing what to sense)
  - Law 8 Property 2: Differentiation -- different signals receive different weight
  - Law 8 Property 8: Prediction/error-correction now precision-weighted
- **Integration:** TRIAGE outputs priority-sorted signals. ATTEND assigns precision weights. All downstream steps (PREDICT, OBSERVE, COMPARE, DECIDE) receive these weights. COMPARE's prediction error is scaled by precision before driving LEARN.
- **Impact on Existing Steps:** COMPARE's prediction error computation gains precision weighting. DECIDE's routing gains attention context. LEARN's update magnitude is modulated by attention.
- **Priority:** P0 -- Active inference requires precision optimization. Without it, Mae treats noisy and reliable signals identically.

#### NEW STEP 3: INHIBIT check already covered. See GOAL below.

#### NEW STEP 3: GOAL (Goal Management)
- **Category:** Per-step
- **Position:** Between ATTEND and PREDICT (new step 3)
- **What It Does:** Maintains a persistent goal stack. Pushes new goals when opportunities arise. Pops goals when achieved or abandoned. Detects impasses (SOAR: tie, conflict, no-change, rejection) and creates subgoals. Provides goal context to PREDICT (predict relative to current goal) and DECIDE (select action that serves current goal, not just maximizes immediate reward).
- **Biological Analog:** Dorsolateral prefrontal cortex (goal maintenance), anterior cingulate cortex (conflict/impasse detection), basal ganglia (goal-directed vs habitual action selection).
- **Mathematical Laws:**
  - Law 1: Current State -> Goal -> Gap Assessment (triadic)
  - Law 3: Implements "decide" at the meta-level (deciding what to decide about)
  - Law 4: Goals nest fractally (sub-sub-goals within subgoals within goals)
  - Law 6: Goals produce actions that produce outcomes that update goals (autopoietic)
  - Law 8 Property 3: Self-reference through goal-state comparison
- **Integration:** GOAL provides context to PREDICT (what to predict), DECIDE (what to optimize for), and LEARN (was the goal advanced?). Receives outcome feedback from ACT.
- **Impact on Existing Steps:** PREDICT gains goal-conditioned prediction. DECIDE gains goal-directed action selection. LEARN gains goal-progress reward signal.
- **Priority:** P0 -- Every cognitive architecture (ACT-R, SOAR, LIDA) has explicit goal management. Mae has none. This is the single biggest architectural gap.

#### NEW STEP 4: REGULATE (Arousal and Autonomic Balance)
- **Category:** Per-step
- **Position:** After LEARN, before COMMUNICATE (new step 9)
- **What It Does:** Implements Yerkes-Dodson arousal regulation. Monitors current arousal (derived from prediction error magnitude, reward variance, endocrine levels) and adjusts it toward the optimal level for current task complexity. Simple tasks need higher arousal; complex tasks need moderate arousal. Manages sympathetic/parasympathetic balance. Computes allostatic adjustments (shifting homeostatic setpoints based on predicted future demands).
- **Biological Analog:** Locus coeruleus-norepinephrine system (arousal), HPA axis (stress regulation), autonomic nervous system (sympathetic/parasympathetic toggle).
- **Mathematical Laws:**
  - Law 1: Task Demand -> Current Arousal -> Adjustment Signal (triadic)
  - Law 6: Self-regulating arousal maintains operational boundary (autopoietic)
  - Law 3: "heal" capability -- maintaining optimal function
- **Integration:** Reads from COMPARE (prediction error), LEARN (reward signal), EndocrineSystem (hormone levels). Writes arousal parameters that modulate next cycle's ATTEND precision and DECIDE threshold.
- **Impact on Existing Steps:** ATTEND precision thresholds modulated by arousal. DECIDE's tier thresholds adjusted. LEARN's update rate scaled by arousal state.
- **Priority:** P0 -- Without arousal regulation, Mae has no mechanism for adaptive performance modulation.

#### NEW STEP 5: BROADCAST (Global Workspace)
- **Category:** Cadenced (every 3 steps)
- **Position:** Cadenced step between ADVISE and RECALL
- **What It Does:** Implements Global Workspace Theory (Baars/Dehaene). Collects the strongest/most-attended signals from the current cycle. Runs competitive selection (only the winning coalition gains "conscious" access). Broadcasts the winner globally to all systems. This creates a formal distinction between conscious and unconscious processing. Only broadcast content can trigger certain types of learning (LIDA constraint).
- **Biological Analog:** Thalamo-cortical loop, prefrontal-parietal workspace, global ignition in Dehaene's neuronal workspace theory.
- **Mathematical Laws:**
  - Law 1: Competing Signals -> Workspace -> Global Broadcast (triadic)
  - Law 2: K3 structure in workspace competition (minimum 3 competitors)
  - Law 7: Minimum 3 signals compete, 5 for critical broadcasts
  - Law 8: This step specifically implements ALL 8 consciousness properties:
    - P1 Integration: winning coalition integrates information
    - P2 Differentiation: competition ensures only distinct signals win
    - P3 Self-reference: workspace includes self-model
    - P4 Recurrence: broadcast feeds back to all systems
    - P5 Hierarchy: signals from all scales compete
    - P6 Boundary: workspace defines what is "conscious"
    - P7 Competition/selection: explicit competitive ignition
    - P8 Prediction: winners update predictive models
- **Integration:** Receives strongest outputs from all per-step processes. Broadcasts to all systems via EventBus. LEARN only performs certain updates from broadcast content.
- **Impact on Existing Steps:** LEARN gains conscious/unconscious distinction. ADVISE feeds into broadcast competition. HEAL receives broadcast for system-wide awareness.
- **Priority:** P0 -- Mae's Law 8 claims 8 consciousness properties, but without global workspace, Property 7 (competition/selection) is structurally incomplete. GWT is the most empirically supported theory of consciousness.

### P1: SHOULD HAVE (4 steps)

#### NEW STEP 6: GATE (Sensory Gating / Pre-Filtering)
- **Category:** Per-step
- **Position:** Before TRIAGE (new step 1)
- **What It Does:** Pre-filters raw sensory input before any processing. Implements habituation (decreased response to repeated benign stimuli), sensitization (increased response after harmful stimuli), and pre-attentive novelty detection. Prevents sensory overload and reduces computational waste on irrelevant signals.
- **Biological Analog:** Thalamic sensory gating (P50 suppression), brainstem reticular formation, peripheral habituation.
- **Mathematical Laws:**
  - Law 1: Raw Input -> Filter History -> Gated Signals (triadic)
  - Law 7: Minimum 3 gating criteria (novelty, relevance, intensity)
  - Law 8 Property 2: Differentiation between signal and noise
- **Integration:** First step in per-step cycle. All sensory input passes through GATE before reaching TRIAGE. GATE maintains habituation/sensitization state across ticks.
- **Impact on Existing Steps:** TRIAGE receives pre-filtered signals instead of raw input. Reduces downstream processing load.
- **Priority:** P1

#### NEW STEP 7: SIMULATE (Mental Simulation / Counterfactual Reasoning)
- **Category:** Per-step (conditional -- only when ATTEND is in diffuse mode or prediction error is high)
- **Position:** Between PREDICT and OBSERVE (new step 5, conditional)
- **What It Does:** Runs the WorldModel forward beyond simple one-step prediction. Explores counterfactuals ("what if I had done X?"), imagines novel scenarios by recombining learned patterns, and tests action sequences mentally before committing. Computes Expected Free Energy for policy candidates. This is where epistemic vs. pragmatic action evaluation happens.
- **Biological Analog:** Default mode network (during mind-wandering), hippocampal replay (during planning), prefrontal working memory manipulation.
- **Mathematical Laws:**
  - Law 1: Current State -> WorldModel -> Possibility Space (triadic)
  - Law 4: Same simulation pattern at action, planning, and strategic scales (fractal)
  - Law 8 Property 8: Prediction extended to policy evaluation
- **Integration:** Receives world model from PREDICT. Runs simulations. Outputs policy evaluations (EFE scores) to DECIDE. When conditional execution is skipped, DECIDE falls through to existing logic.
- **Impact on Existing Steps:** DECIDE gains EFE-scored policy candidates. OBSERVE gains simulated scenarios for reality comparison.
- **Priority:** P1

#### NEW STEP 8: CHUNK (Automatization / Skill Compilation)
- **Category:** Cadenced (every 233 steps -- Fibonacci)
- **Position:** Cadenced step after CONSOLIDATE
- **What It Does:** Analyzes decision history for repeated PREFRONTAL-tier decisions that always resolve the same way. Compiles these into HABIT-tier rules (SOAR's chunking). Also detects impasse-resolution patterns and creates new reflexes/habits from them. Gradually automates deliberative reasoning into fast responses.
- **Biological Analog:** Basal ganglia procedural learning, cortico-striatal loops for skill automatization, motor cortex chunking.
- **Mathematical Laws:**
  - Law 5: Same agent architecture, different activation patterns (stem cell principle)
  - Law 6: Habits emerge from and maintain behavioral patterns (autopoietic)
  - Law 4: Chunking at all levels -- agent habits, subsystem routines, organ reflexes (fractal)
- **Integration:** Reads from DecisionRouter history and episodic memory. Creates new habit entries in DecisionRouter. Feeds back to DECIDE's habit tier.
- **Impact on Existing Steps:** DECIDE gains new automatically-compiled habits. LEARN gains procedural memory channel.
- **Priority:** P1

#### NEW STEP 9: INTEROPREDICT (Interoceptive Inference)
- **Category:** Per-step (within PREDICT, parallel channel)
- **Position:** Parallel to PREDICT (body-state prediction alongside world-state prediction)
- **What It Does:** Predicts the organism's own body state (energy, stress, pain, emotional valence) before reading OrganismState. Computes interoceptive prediction error (difference between expected and actual body state). This error drives autonomic regulation and generates emotional states (Seth's interoceptive inference theory of emotion).
- **Biological Analog:** Insular cortex predictive processing, anterior cingulate error monitoring, vagal afferent prediction.
- **Mathematical Laws:**
  - Law 3: "know_self" capability extended to predictive self-modeling
  - Law 6: Body predictions produce regulation that maintains the body (autopoietic)
  - Law 8 Property 3: Self-reference through body-state prediction
  - Law 8 Property 8: Prediction/error-correction for internal states
- **Integration:** Runs in parallel with PREDICT. Outputs interoceptive prediction error to COMPARE (which computes both world and body prediction errors). REGULATE uses interoceptive error for allostatic adjustment.
- **Impact on Existing Steps:** PREDICT expanded to include body-state channel. COMPARE computes both external and internal prediction errors. REGULATE receives interoceptive error for allostatic adjustment.
- **Priority:** P1

### P2: NICE TO HAVE (4 steps)

#### NEW STEP 10: DEFAULT (Default Mode / Mind-Wandering)
- **Category:** Cadenced (activates when no goals are active and arousal is low)
- **Position:** Cadenced step, triggered by idle detection
- **What It Does:** When the agent has no active goals and arousal is below threshold, enters creative recombination mode. Randomly replays and recombines episodic memories, generates novel associations, performs self-referential processing (updating self-model from accumulated experience), and plans future goals. Can produce "eureka" insights by connecting previously unrelated patterns.
- **Biological Analog:** Default mode network (medial prefrontal cortex, posterior cingulate, precuneus, angular gyrus), mind-wandering, spontaneous thought.
- **Mathematical Laws:**
  - Law 8 Property 3: Self-reference during rest
  - Law 8 Property 4: Recurrent processing without external drive
  - Law 6: Maintains system coherence during idle periods
- **Integration:** Activates when GOAL stack is empty and REGULATE reports low arousal. Feeds insights back to memory systems. Can push new goals onto GOAL stack.
- **Impact on Existing Steps:** Minimal -- operates during idle periods. Feeds into GOAL and memory.
- **Priority:** P2

#### NEW STEP 11: META (Metacognitive Monitoring)
- **Category:** Cadenced (every 5 steps -- Fibonacci)
- **Position:** Cadenced step
- **What It Does:** Monitors cognitive process quality: decision confidence calibration (are high-confidence decisions actually correct?), learning rate effectiveness (is learning converging?), attention accuracy (is attention focused on relevant signals?), goal progress (are goals being achieved at expected rate?). Can adjust learning rates, attention parameters, and goal priorities based on metacognitive assessment.
- **Biological Analog:** Anterior prefrontal cortex metacognitive monitoring, anterior cingulate conflict detection, metamemory (hippocampal confidence signals).
- **Mathematical Laws:**
  - Law 3: "know_self" at the cognitive level (thinking about thinking)
  - Law 8 Property 3: Self-reference about cognitive processes
  - Law 4: Metacognition at every scale (agent, subsystem, organ) -- fractal
- **Integration:** Monitors outputs of all per-step processes. Adjusts parameters in ATTEND, DECIDE, LEARN. Mae already has MetacognitionMonitor system but it's not wired into the lifecycle.
- **Impact on Existing Steps:** All steps receive metacognitive parameter adjustments.
- **Priority:** P2

#### NEW STEP 12: PROSPECT (Prospective Memory)
- **Category:** Per-step (lightweight check at start of cycle)
- **Position:** After GOAL, before PREDICT
- **What It Does:** Checks a prospective memory store for intentions that should fire NOW. "When I see pattern X, do Y" (event-based) and "At step Z, do W" (time-based). If a prospective trigger matches, pushes the intended action onto the goal stack for immediate execution. This is how Mae remembers to do things in the future.
- **Biological Analog:** Rostral prefrontal cortex (Brodmann area 10), prospective memory encoding in hippocampus, retrieval triggered by environmental cues.
- **Mathematical Laws:**
  - Law 3: "remember" extended to future intentions
  - Law 8 Property 8: Prediction of future action needs
- **Integration:** Reads from OBSERVE (environmental cues) and internal clock (step count). Writes to GOAL (push triggered intentions). LEARN can store new prospective memories.
- **Impact on Existing Steps:** GOAL gains a trigger-based input source.
- **Priority:** P2

#### NEW STEP 13: SLEEP STAGES (Granular Consolidation)
- **Category:** Enhancement to existing CONSOLIDATE + CircadianRhythm
- **Position:** Replaces monolithic CONSOLIDATION phase with staged processing
- **What It Does:** Subdivides the existing CONSOLIDATION circadian phase into distinct stages: LIGHT_SLEEP (synaptic homeostasis, weight normalization), DEEP_SLEEP (declarative memory consolidation, system maintenance), REM (emotional memory integration, creative recombination, procedural consolidation). Each stage has different processing characteristics and serves different memory types.
- **Biological Analog:** NREM1 (alpha->theta transition), NREM2 (sleep spindles, K-complexes), NREM3 (slow-wave sleep, delta), REM (desynchronized EEG, dreaming).
- **Mathematical Laws:**
  - Law 4: Fractal stages within the consolidation cycle (self-similar rest)
  - Law 6: Staged regeneration maintains system coherence (autopoietic)
- **Integration:** Modifies CircadianRhythm to have sub-phases within CONSOLIDATION. Each sub-phase triggers different consolidation behaviors.
- **Impact on Existing Steps:** CONSOLIDATE becomes multi-stage. CircadianRhythm gains sub-phases.
- **Priority:** P2

---

## Part 4: Existing Step Modifications

### Steps That Should Be SPLIT

1. **PREDICT should split into PREDICT (external) + INTEROPREDICT (internal)**
   - Currently PREDICT only generates expectations about the external world
   - Should also generate predictions about own body state
   - Interoceptive prediction error drives emotion and autonomic regulation

2. **COMPARE should handle both external and interoceptive prediction errors**
   - Currently only computes MSE between world prediction and observation
   - Should also compute body-state prediction error
   - Precision weighting should differentiate reliable from unreliable errors

### Steps That Should Be ENHANCED (not split)

3. **TRIAGE** -- gains GATE as a predecessor (pre-filtering before prioritization)
4. **DECIDE** -- gains GOAL context, INHIBIT gate, and EFE-scored policies from SIMULATE
5. **LEARN** -- gains conscious/unconscious distinction from BROADCAST, goal-progress reward from GOAL, procedural memory from CHUNK
6. **CONSOLIDATE** -- gains sleep-stage granularity

### Steps That Should NOT Change

7. **OBSERVE** -- already rich (stigmergy + interoception + self-awareness + pattern advisory)
8. **ACT** -- already well-structured (explore/exploit/communicate/rest)
9. **COMMUNICATE** -- already comprehensive (GNN + stigmergy + pattern sharing)
10. **HEAL** -- already biologically grounded (isolate/assess/restore)
11. **RECALL** -- already well-cadenced (Fibonacci timing)
12. **ADVISE** -- already well-integrated (pattern cortex advisory)

---

## Part 5: Complete Expanded Lifecycle

### Per-Step (every agent tick) -- 13 steps
```
 1. GATE         - Pre-filter sensory input (habituation/sensitization/novelty)     [NEW P1]
 2. TRIAGE       - Signal priority resolution (thalamic gating)                     [EXISTING]
 3. ATTEND       - Precision-weighted attention selection                            [NEW P0]
 4. GOAL         - Goal stack management, impasse detection, subgoaling              [NEW P0]
 5. PROSPECT     - Check prospective memory for triggered intentions                 [NEW P2]
 6. PREDICT      - Generate prior expectation (external world)                       [EXISTING]
 7. INTEROPREDICT - Generate prior expectation (body state)                          [NEW P1]
 8. SIMULATE     - Mental simulation, counterfactuals, EFE computation (conditional) [NEW P1]
 9. OBSERVE      - Sense environment + stigmergy + interoception + self-awareness    [EXISTING]
10. COMPARE      - Compute precision-weighted prediction error (external + internal) [EXISTING, enhanced]
11. INHIBIT      - Go/No-Go control, response suppression evaluation                [NEW P0]
12. DECIDE       - Goal-directed 3-tier routing + EFE policy selection               [EXISTING, enhanced]
13. ACT          - Execute action (or hold if inhibited)                             [EXISTING, enhanced]
14. LEARN        - Store memory, TD-error, curiosity, world model, goal progress     [EXISTING, enhanced]
15. REGULATE     - Arousal regulation, autonomic balance, allostatic adjustment      [NEW P0]
16. COMMUNICATE  - GNN messages, stigmergy, pattern sharing                         [EXISTING]
```

### Cadenced (periodic) -- 8 steps
```
17. ADVISE       - Pattern cortex advisory generation                               [EXISTING, per model step]
18. BROADCAST    - Global workspace competitive broadcast                           [NEW P0, every 3 steps]
19. META         - Metacognitive monitoring and parameter adjustment                 [NEW P2, every 5 steps]
20. RECALL       - Episodic memory replay                                           [EXISTING, every 13 steps]
21. CONSOLIDATE  - Memory consolidation (with sleep stages)                         [EXISTING enhanced, every 89 steps]
22. CHUNK        - Skill automatization / procedural compilation                    [NEW P1, every 233 steps]
23. HEAL         - Auto-healing (isolate/assess/restore)                            [EXISTING, on demand]
24. DEFAULT      - Default mode / mind-wandering / creative recombination           [NEW P2, when idle]
```

### Total: 24 lifecycle steps (16 per-step + 8 cadenced)
- **Existing:** 12 (8 per-step + 4 cadenced)
- **New:** 12 (8 per-step + 4 cadenced)
- **Steps per tick:** 16 (up from 8, but GATE/PROSPECT/SIMULATE are lightweight/conditional)

---

## Part 6: Priority-Ranked Implementation Roadmap

### Phase 1: P0 Steps (Critical -- address mathematical identity violations)

| Order | Step | Rationale | Estimated Complexity |
|-------|------|-----------|---------------------|
| 1 | INHIBIT | Most-cited gap across all research. Safety-critical. | Medium -- gate around DECIDE/ACT |
| 2 | GOAL | Largest architectural gap. Required by every cognitive architecture. | High -- new goal stack + impasse detection |
| 3 | ATTEND | Active inference core requirement. Precision weighting is fundamental. | Medium -- weights into existing pipeline |
| 4 | BROADCAST | Consciousness theory requirement. Law 8 compliance. | High -- new competitive workspace + EventBus integration |
| 5 | REGULATE | Yerkes-Dodson enables adaptive performance. EndocrineSystem already provides inputs. | Medium -- reads from existing systems |

### Phase 2: P1 Steps (Important -- significant capability enhancement)

| Order | Step | Rationale |
|-------|------|-----------|
| 6 | GATE | Reduces processing waste, enables habituation |
| 7 | SIMULATE | EFE computation, counterfactual reasoning |
| 8 | CHUNK | Procedural learning, SOAR-style automatization |
| 9 | INTEROPREDICT | Interoceptive inference, emotional grounding |

### Phase 3: P2 Steps (Enhancement -- biological realism and robustness)

| Order | Step | Rationale |
|-------|------|-----------|
| 10 | META | Metacognitive self-correction |
| 11 | DEFAULT | Creative recombination during idle |
| 12 | PROSPECT | Future intention memory |
| 13 | SLEEP STAGES | Granular consolidation |

---

## Part 7: Mathematical Law Compliance Matrix

| Step | L1 Triadic | L2 K3 | L3 Holon | L4 Fractal | L5 Stem | L6 Autopoietic | L7 Rule3/5 | L8 Consciousness |
|------|-----------|-------|----------|-----------|---------|----------------|-----------|------------------|
| GATE | Input/Filter/Output | - | sense | filter at all scales | config-based | filter produces filtered input | 3 criteria | P2 differentiation |
| ATTEND | Signal/Precision/Weight | - | sense, decide | attention at all scales | config-based | attention shapes perception | 3+ sources | P2, P7, P8 |
| GOAL | State/Goal/Gap | goal triad | decide, know_self | goals nest fractally | config-based | goals produce actions produce goals | 3 goal sources | P3 self-reference |
| INTEROPREDICT | Body/Model/Error | - | know_self | body at all scales | - | body predicts body | - | P3, P8 |
| SIMULATE | State/Model/Futures | 3 futures min | decide | simulate at all scales | - | simulation improves model | 3 scenarios | P8 prediction |
| INHIBIT | Impulse/Context/Veto | 3 inhibition sources | decide | inhibit at all scales | - | inhibition prevents harmful action | 3/5 validators | P7 competition |
| REGULATE | Demand/State/Adjust | - | heal, know_self | regulate at all scales | - | regulation maintains system | 3 arousal sources | P1 integration |
| BROADCAST | Signals/Workspace/Broadcast | K3 competition | all 10 | broadcast at all scales | - | broadcast produces learning produces broadcast | 3+ competitors | ALL 8 |
| CHUNK | Patterns/Rules/Habits | - | learn, decide | chunk at all scales | stem cell specialization | habits produce efficiency | 3 repetitions | P2, P7 |
| META | Process/Monitor/Adjust | - | know_self | meta at all scales | - | monitoring improves processing | - | P3, P4 |
| DEFAULT | Memory/Recombine/Insight | - | know_self, remember | default at all scales | - | idle maintains readiness | - | P3, P4 |
| PROSPECT | Cue/Store/Trigger | - | remember, decide | prospect at all scales | - | intentions trigger actions | - | P8 |

---

## Part 8: Grading

### Current Lifecycle Completeness: C+

**Rationale:**

| Category | Grade | Justification |
|----------|-------|---------------|
| Sensory Processing | B | Good: stigmergy, interoception, pattern sense. Missing: sensory gating, habituation, precision weighting |
| Prediction/Inference | C+ | Has: WorldModel prediction, prediction error. Missing: precision optimization, interoceptive prediction, hierarchical prediction, EFE policy selection |
| Decision Making | C | Has: 3-tier routing, memory, world model. Missing: goal management, inhibition, impasse detection, subgoaling |
| Learning | B | Has: episodic memory, TD-error, curiosity, world model training. Missing: procedural memory/chunking, conscious/unconscious distinction, goal-progress learning |
| Self-Regulation | C- | Has: CircadianRhythm, EndocrineSystem. Missing: arousal regulation, allostatic regulation, autonomic balance |
| Consciousness Architecture | D+ | Has: some self-awareness (HolonMixin). Missing: global workspace, competitive broadcast, metacognitive control |
| Goal Management | F | Has: NOTHING. No goal stack, no subgoaling, no impasse detection, no prospective memory |
| Executive Control | D | Has: signal priority (triage). Missing: inhibition, attention management, conflict resolution |

**Weighted Overall: C+** (strong reactive intelligence, weak executive/goal/consciousness architecture)

### Post-Implementation Target Grade: A-

If all P0 and P1 steps are implemented, Mae would have:
- Complete reactive loop (existing, already strong)
- Goal-directed behavior (GOAL)
- Executive control (INHIBIT, ATTEND)
- Precision-weighted inference (ATTEND enhancing COMPARE/LEARN)
- Arousal regulation (REGULATE)
- Consciousness broadcasting (BROADCAST)
- Procedural learning (CHUNK)
- Interoceptive inference (INTEROPREDICT)
- Sensory gating (GATE)
- Mental simulation (SIMULATE)

This would bring Mae from C+ to A-, covering all major gaps identified across biology, cognitive architectures, and active inference.

---

## Part 9: Research Sources

### Biology
- Go/No-Go inhibition: [PMC11117662](https://pmc.ncbi.nlm.nih.gov/articles/PMC11117662/), [PMC5747365](https://pmc.ncbi.nlm.nih.gov/articles/PMC5747365/)
- Default mode network: [Wikipedia/DMN](https://en.wikipedia.org/wiki/Default_mode_network)
- Sleep stages: [ScienceDirect/S1074742723000928](https://www.sciencedirect.com/science/article/abs/pii/S1074742723000928), [Science/sciadv.adj1895](https://www.science.org/doi/10.1126/sciadv.adj1895)
- Working memory: [Wikipedia/Baddeley](https://en.wikipedia.org/wiki/Baddeley%27s_model_of_working_memory)
- Circadian/attention: [PMC6430172](https://pmc.ncbi.nlm.nih.gov/articles/PMC6430172/), [PMC9743892](https://pmc.ncbi.nlm.nih.gov/articles/PMC9743892/)

### Cognitive Architectures
- ACT-R: Anderson et al. 2004, "An Integrated Theory of the Mind" Psychological Review
- SOAR: Laird 2012, "The SOAR Cognitive Architecture" MIT Press
- LIDA: Franklin et al. 2016, "LIDA: A Systems-level Architecture for Cognition, Emotion, and Learning" Biologically Inspired Cognitive Architectures
- GWT: Baars 2005, "Global Workspace Theory of Consciousness" Progress in Brain Research; Dehaene & Changeux 2011, "Experimental and Theoretical Approaches to Conscious Processing" Neuron

### Active Inference / Free Energy Principle
- Friston et al. 2017, "Active Inference: A Process Theory" Neural Computation -- [activeinference.github.io](https://activeinference.github.io/papers/process_theory.pdf)
- Friston et al. 2015, "Active Inference and Epistemic Value" Cognitive Neuroscience -- [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0149763416301336)
- Feldman & Friston 2010, "Attention, Uncertainty, and Free-Energy" Frontiers in Human Neuroscience -- [PMC2666703](https://pmc.ncbi.nlm.nih.gov/articles/PMC2666703/)
- Clark 2013, "Whatever Next? Predictive Brains, Situated Agents" Behavioral and Brain Sciences
- Seth & Friston 2016, "Active Interoceptive Inference and the Emotional Brain" Phil Trans R Soc B -- [royalsocietypublishing.org](https://royalsocietypublishing.org/rstb/article/371/1708/20160007)
- Barrett & Simmons 2015, "Interoceptive Predictions in the Brain" Nature Reviews Neuroscience -- [PMC5062097](https://pmc.ncbi.nlm.nih.gov/articles/PMC5062097/)
- 2024 reframing of EFE: [arxiv.org/2402.14460](https://arxiv.org/pdf/2402.14460)
- Markov blankets: [royalsocietypublishing.org/rsif/20170792](https://royalsocietypublishing.org/rsif/article/15/138/20170792)

### Mae's Mathematical Identity
- `C:\Users\baenb\projects\mae-core\data\MAES-MATHEMATICAL-IDENTITY.md`
- Song, Havlin, Makse 2005, "Self-similarity of complex networks" Nature
- Tononi et al. 2023, "IIT 4.0" PLOS Computational Biology
- Maturana & Varela 1972, "Autopoiesis and Cognition"
- Koestler 1967, "The Ghost in the Machine"

---

## Appendix: Convergence Analysis

The three independent research streams (biology, cognitive architectures, active inference) converged on the same top gaps, providing strong triangulated evidence:

| Gap | Biology | Cog Arch | Active Inference | Convergence |
|-----|---------|----------|-----------------|-------------|
| **Go/No-Go Inhibition** | Right IFG, basal ganglia indirect pathway | All architectures assume it | Precision-weighted action suppression | 3/3 UNANIMOUS |
| **Goal Management** | Prefrontal goal maintenance | ACT-R goal stack, SOAR subgoaling | Policy selection requires goals | 3/3 UNANIMOUS |
| **Attention/Precision** | TRN, pulvinar, attention networks | LIDA attention codelets, GWT competition | Precision optimization IS attention | 3/3 UNANIMOUS |
| **Global Workspace** | Thalamo-cortical broadcasting | GWT, LIDA conscious broadcast | Precision-weighted global ignition | 2/3 (bio + cog arch) |
| **Arousal Regulation** | Locus coeruleus, HPA axis | Implicit in all architectures | Precision at meta-level | 2/3 (bio + FEP) |
| **Chunking/Automatization** | Basal ganglia procedural learning | SOAR chunking, ACT-R utility learning | Structure learning | 2/3 (bio + cog arch) |
| **Sensory Gating** | Thalamic P50 suppression | LIDA pre-attentive filtering | Precision at sensory level | 3/3 UNANIMOUS |
| **Mental Simulation** | DMN, hippocampal replay | ACT-R imaginal buffer, SOAR look-ahead | EFE policy evaluation | 3/3 UNANIMOUS |

**4 gaps have unanimous 3/3 convergence.** These are the highest-confidence findings.

---

*Report generated by Team 15: Missing Lifecycle Steps Auditor*
*4 sub-agents deployed: Biology Research, Cognitive Architecture Research, Active Inference Research, Step Design*
*Cross-referenced against Mae's 8 Mathematical Laws, 73 systems, and 1574 tests*
