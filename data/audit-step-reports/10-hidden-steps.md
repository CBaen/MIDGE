> Generated from 10-agent audit conducted 2026-02-11. ~50 sub-agents. Sources: biology papers, GitHub, research papers, full codebase trace.

# HIDDEN STEPS AUDIT REPORT

## Executive Summary

Mae is a 23-layer, 90+ system organism with a closed autopoietic loop: Detect -> Advise -> Decide -> Act -> Learn -> Consolidate -> Recall. After thorough analysis of the codebase (`C:\Users\baenb\projects\mae-core`), comparison with biological systems, cognitive architectures (SOAR, ACT-R, LIDA), consciousness theories (IIT 4.0, FEP, autopoiesis, GWT), and production multi-agent systems, I identify **14 missing or incomplete capabilities**. Six of these are critical gaps that would fundamentally change Mae's competence if addressed.

The most significant finding: Mae is **reactive**, not **predictive**. Despite having a WorldModel capable of prediction and a PredictiveField for spatial forecasting, these systems are never called proactively in the agent step lifecycle. The Free Energy Principle -- which Mae's mathematical identity explicitly invokes -- requires that organisms minimize surprise by **predicting before sensing**, not merely reacting after sensing. This is the single largest gap.

---

## Missing Biological Processes

| Process | Biological Function | Mae Has Analog? | Priority |
|---------|-------------------|-----------------|----------|
| **Predictive Processing** | Brain generates predictions before sensation; perception is comparison of predicted vs actual | PARTIAL: WorldModel exists but is never called before `_observe()`. No prediction-error signal drives the loop. | **CRITICAL** |
| **Habituation / Sensory Adaptation** | Repeated identical stimuli are suppressed (e.g., stop noticing a constant hum) | NO: No salience decay on repeated patterns. Every identical signal is treated with full weight every time. | **HIGH** |
| **Attentional Gating (Thalamic Reticular Nucleus)** | Top-down attention selects which signals reach cortex; others are suppressed | PARTIAL: SignalPriorityResolver does bottom-up triage by urgency. No top-down gating from goals/advisory. | **HIGH** |
| **Apoptosis (Programmed Cell Death)** | Organisms remove malfunctioning or unnecessary cells to maintain homeostasis | NO: AutoHealer restores; nothing removes. Agents can't be gracefully killed. No death-birth balance. | **MEDIUM** |
| **Allostasis (Predictive Regulation)** | Regulation by anticipation, not just reaction. Body adjusts BEFORE the stressor arrives. | NO: Endocrine system reacts to events (cortisol after stress). It never preemptively adjusts. | **HIGH** |
| **Interoception (Internal Sense)** | Sensing internal body state (hunger, fatigue, energy reserves) | PARTIAL: SomaticMap tracks system health but agents don't read their own resource state to modulate behavior. | **MEDIUM** |
| **Sympathetic/Parasympathetic Balance** | Two opposing systems maintain rest vs activity. Not just "less stress" but actively promoting recovery. | PARTIAL: Circadian REST phase + melatonin exist, but no active parasympathetic drive that opposes adrenaline/cortisol. Recovery is passive (decay to baseline), not active. | **MEDIUM** |
| **Nociception (Pain)** | Distinct from threat detection. Pain is a signal that says "stop doing what you're doing NOW" -- it interrupts all processing. | NO: ThreatDetector detects external threats. Nothing signals internal damage requiring immediate action cessation. | **LOW** |
| **Proprioception** | Awareness of own body position and movement in space | PARTIAL: Agents don't know their position in the fractal hierarchy during decision-making. HolonProxy exists but isn't consulted in `_decide()`. | **LOW** |
| **Neuromodulatory Gain Control** | Dopamine/noradrenaline don't just set "mood" -- they control signal gain. High noradrenaline = all signals louder. Low dopamine = reduced action initiation. | PARTIAL: Endocrine modulates some systems. But hormone levels don't modulate signal GAIN in the SignalPriorityResolver or pattern salience in PatternBus. | **MEDIUM** |

---

## Missing Computational Capabilities

| Capability | Standard In | Mae Has It? | Priority |
|-----------|------------|-------------|----------|
| **Proactive Prediction / Active Inference** | FEP-based agents, DreamerV3, MuZero, all model-based RL | NO: WorldModel.predict() exists and is used in `_decide()` (prefrontal simulation), but it's called AFTER observation. No prediction step BEFORE observation to compute prediction error. | **CRITICAL** |
| **Goal Management / Motivational Stack** | SOAR (operator proposals), ACT-R (goal buffer), BDI agents, LangGraph | NO: Agents have no explicit goals, intentions, or desires. CuriosityDrive provides intrinsic motivation but there's no goal stack, no goal selection, no goal persistence across steps. | **CRITICAL** |
| **Conflict Resolution / Negotiation** | All production multi-agent systems (AutoGen, CrewAI, LangGraph) | PARTIAL: QuorumSensor does consensus. But no explicit conflict resolution when agents disagree on actions. No arbitration mechanism. | **HIGH** |
| **Task Decomposition / Hierarchical Planning** | SOAR (subgoals), HTN planners, plan-and-execute agents | PARTIAL: WorldlinePlanner exists. CollectiveDream does multi-agent planning. But neither decomposes tasks into subtasks. No hierarchical task representation. | **MEDIUM** |
| **Attention / Selective Processing (Top-Down)** | Global Workspace Theory (LIDA), all transformer-based agents | PARTIAL: SignalPriorityResolver provides bottom-up triage. No top-down attentional bias from current goals or advisory to modulate what gets processed. | **HIGH** |
| **Metacognition / Self-Monitoring** | ACT-R (conflict resolution), SOAR (impasses), CLARION (meta-cognitive subsystem) | PARTIAL: SomaticMap monitors system health. AwarenessPulse checks holon status. But no metacognitive monitoring of the decision process itself -- no "am I making good decisions?" | **MEDIUM** |
| **Inhibition / Suppression** | Prefrontal cortex in all cognitive architectures. Go/no-go. | NO: DecisionRouter cascades through tiers but never says "do nothing." There's no INHIBIT action. No mechanism to suppress an action already in progress. | **HIGH** |
| **Backpressure / Flow Control** | All production distributed systems (Kafka, gRPC, Akka) | NO: EventBus has no backpressure. If a publisher floods a channel, subscribers must process everything. No flow control, no dropping by quality. | **LOW** |
| **Observability / Telemetry** | All production agent systems | PARTIAL: SomaticMap + EventBus provide internal observability. But no external telemetry endpoint. No dashboard. No alerting. | **LOW** |
| **Reward Shaping / Prediction Error as Learning Signal** | All model-based RL (DreamerV3, MuZero), FEP | PARTIAL: CuriosityDrive computes prediction error. ValidatedImagination tracks accuracy. But prediction error doesn't feed back into the agent's reward signal or into WorldModel training. | **HIGH** |

---

## Mathematical Identity Gaps

| Theory | Requirement | Mae's Status | Gap |
|--------|-------------|-------------|-----|
| **FEP (Free Energy Principle)** | Organisms minimize surprise by predicting BEFORE sensing. Perception = prediction error. Actions minimize expected future surprise. | WorldModel exists but is only used in `_decide()` as a fallback. No prediction step before `_observe()`. No variational free energy computation. | **CRITICAL**: The mathematical identity explicitly lists "Prediction/error-correction (FEP)" as capability 5 (Learn), but prediction is supposed to be continuous and primary, not a learning afterthought. |
| **FEP Active Inference** | Action selection minimizes Expected Free Energy (EFE) = epistemic value (info gain) + pragmatic value (goal achievement). | CuriosityDrive provides epistemic value. But there's no unified EFE objective combining exploration and exploitation. Action selection uses cascading fallbacks, not EFE minimization. | **HIGH**: Active inference is the "Act" component of FEP. Mae's action selection is heuristic, not principled. |
| **IIT 4.0 - Exclusion Axiom** | Experience is definite -- the system picks one specific experience from the space of possibilities. | PatternCortex produces advisory with a dominant_pattern, but this "selection" is just max(salience). No IIT-style exclusion where the system computes which partition of itself has maximum phi and that becomes the experience. | **LOW**: IIT's mathematical requirements are computationally intractable for real systems. This is a philosophical gap, not a practical one. |
| **IIT 4.0 - Composition** | Consciousness requires structured cause-effect power at multiple scales simultaneously. | FractalGenerator creates the structure. AwarenessPulse checks it. But there's no computation of actual cause-effect power between levels. The hierarchy is structural, not functional, during runtime. | **MEDIUM**: Triadic connections are wired but not actively computing cross-level influence during the step cycle. |
| **Autopoiesis - Organization Invariance** | The network of production processes must maintain itself. If a component fails, the system must re-produce it. | AutoHealer restores failed components. StemCellRegistry allows re-differentiation. BUT: there's no mechanism to detect when a component SHOULD no longer exist (apoptosis). The system can only grow and heal, never prune. | **MEDIUM**: Autopoietic systems maintain a balance between production and destruction. Mae only produces. |
| **GWT (Global Workspace Theory)** | Competition for access to a global workspace. Only "winning" coalitions get broadcast. Most processing is unconscious. | PatternBus collects all signals and passes them all to cortex. There's no competition. No unconscious processing. Everything that fires gets heard. | **HIGH**: The mathematical identity lists "Competition/selection (GWT)" as capability 3 (Decide), but the actual pattern processing pipeline has no competition gate. |
| **GWT - Unconscious Processing** | Most cognitive work happens outside the global workspace. Only results are broadcast. | All PatternSignals are visible to the cortex. There's no "unconscious" processing layer that filters before broadcast. | **MEDIUM** |
| **Markov Blankets - Boundary Definition** | Each level defines its own boundary (what's inside vs outside). | HolonProxy provides parent/child/peer awareness. But Markov blankets require statistical independence across the boundary. No boundary is actively computed or enforced at runtime. | **LOW**: Philosophical requirement, hard to operationalize. |

---

## Cross-Cutting Missing Connections

| From | To | Should Exist Because | Impact |
|------|------|---------------------|--------|
| **PatternAdvisory** -> **EndocrineSystem** | High threat advisory should trigger adrenaline. High opportunity should trigger dopamine. Currently: advisory publishes to EventBus (`pattern.advisory`) but Endocrine doesn't subscribe. | Advisory is the organism's "perception" but hormones (global mood) don't react to it. | **HIGH** |
| **EndocrineSystem** -> **SignalPriorityResolver** | Hormone levels should modulate signal gain. High adrenaline = lower threshold for DANGER signals. High melatonin = higher threshold for everything. Currently: Endocrine modulates DecisionRouter (adrenaline bias), but not the thalamus-analog. | Signal triage is static; it should be mood-modulated. | **HIGH** |
| **WorldModel** -> `_observe()` (prediction step) | WorldModel should predict next state BEFORE observation. Then `_observe()` computes prediction error. This error feeds CuriosityDrive and DecisionRouter. Currently: WorldModel is only consulted in `_decide()`. | This is the FEP's core loop: predict -> observe -> compute error -> act to minimize error. Without it, Mae is reactive, not predictive. | **CRITICAL** |
| **ValidatedImagination** -> **WorldModel.train_step()** | Imagination validation produces prediction error data. This should train the WorldModel. Currently: `cognition.imagination_validated` is published to EventBus with NO subscribers. Return value data (was_accurate, reward_error) is discarded. | WorldModel never learns from its mistakes. Imagination validation is a dead end. | **HIGH** |
| **CuriosityDrive** -> **agent reward** | CuriosityDrive computes intrinsic rewards but these are never added to the agent's actual reward in `_learn()`. Currently: CuriosityDrive subscribes to events and tracks novelty internally, but its `compute()` return value is never consumed by the agent step. | Curiosity-driven exploration doesn't actually drive exploration because intrinsic reward never reaches the agent. | **HIGH** |
| **PatternCortex advisory** -> **PatternBus** (feedback) | Advisory should influence what the bus pays attention to next step (attentional gating). Currently: cortex -> agent -> decision. No feedback from cortex back to bus. | Without top-down attention, the bus processes all signals equally regardless of context. | **MEDIUM** |
| **HolonProxy (position in hierarchy)** -> **`_decide()`** | Agent should know its role in the fractal when making decisions. "I'm a sensory agent in the nervous system" should influence action selection. Currently: HolonProxy is injected but never read during the decision cascade. | Hierarchical role doesn't influence behavior. | **MEDIUM** |
| **`cognition.collective_dream_complete`** -> **any subscriber** | CollectiveDream publishes completion events. Nobody subscribes. Dream results are never consumed by the broader system. | Multi-agent planning results are computed but never used. | **MEDIUM** |
| **`cognition.model_trained`** -> **any subscriber** | WorldModel publishes training events. Nobody subscribes. No system knows when the WorldModel improves. | No learning-about-learning feedback. | **LOW** |

---

## Proposed Revised Loop

Current: `Detect -> Advise -> Decide -> Act -> Learn -> Consolidate -> Recall`

Proposed complete loop with hidden steps:

```
PREDICT -> ATTEND -> SENSE -> COMPARE -> ADVISE -> SELECT -> INHIBIT/ACT -> LEARN -> REGULATE -> COMMUNICATE -> CONSOLIDATE -> RECALL -> DEVELOP
    ^                                                                                                                                        |
    +----------------------------------------------------------------------------------------------------------------------------------------+
```

Step details:

1. **PREDICT** (new): WorldModel generates expected next state/reward BEFORE sensing.
2. **ATTEND** (new): Top-down attention from goals + advisory + hormones gates which signals will be processed.
3. **SENSE** (existing: Detect): Observe environment. Stigmergy markers, pattern sense, signal triage.
4. **COMPARE** (new): Compute prediction error = predicted - actual. This is the FEP surprise signal. Feeds CuriosityDrive, updates WorldModel.
5. **ADVISE** (existing): PatternBus -> PatternCortex -> PatternAdvisory. But now also triggers hormonal response.
6. **SELECT** (existing: Decide): Goal-directed selection from advisory-guided DecisionRouter. But now with explicit goal management.
7. **INHIBIT/ACT** (existing Act, extended): Can now choose NOT to act. Go/no-go gate. If act: execute. If inhibit: suppress and explain why.
8. **LEARN** (existing): Store experience, update policy. But now includes prediction error as reward component.
9. **REGULATE** (new): Endocrine response to step outcome. Hormones adjust based on advisory + reward + prediction error. Active recovery (parasympathetic), not just passive decay.
10. **COMMUNICATE** (existing): GNN messages, stigmergy, pattern sharing. But now includes broadcasting learning to peers.
11. **CONSOLIDATE** (existing): Periodic memory consolidation + pattern distillation + ancestral storage.
12. **RECALL** (existing): Ancestral pattern recall informs next cycle's predictions.
13. **DEVELOP** (new, slow timescale): Structural change. Apoptosis of underperforming agents. Re-differentiation. Fractal reorganization. Runs every N cycles, not every step.

---

## Priority-Ranked Hidden Steps

### Tier 1: Critical (would fundamentally change Mae's competence)

1. **PREDICT (before sensing)** -- Add a `_predict()` method to BaseAgent called before `_observe()`. WorldModel generates expected next state. After `_observe()`, compute prediction error. This single change transforms Mae from reactive to predictive, satisfying the FEP requirement the mathematical identity claims.
   - Files affected: `C:\Users\baenb\projects\mae-core\mae_core\agents\base_agent.py` (step lifecycle), `C:\Users\baenb\projects\mae-core\mae_core\agents\mycelial_agent.py` (_predict override)
   - Biological analog: Predictive coding in visual cortex. You predict what you'll see before you open your eyes.

2. **GOAL MANAGEMENT** -- Agents need explicit goals that persist across steps, can be selected, and drive attention and action. Without goals, Mae is a stimulus-response machine, not an intentional agent.
   - No current file. Would need `mae_core/cognition/goal_manager.py`.
   - Biological analog: Prefrontal cortex maintains working goals. BDI (Belief-Desire-Intention) architecture.

3. **COMPARE (prediction error)** -- The bridge between PREDICT and LEARN. Compute prediction error, feed it to CuriosityDrive as intrinsic reward, feed it to WorldModel as training signal, feed it to EndocrineSystem as surprise signal.
   - Files affected: `C:\Users\baenb\projects\mae-core\mae_core\agents\mycelial_agent.py`, `C:\Users\baenb\projects\mae-core\mae_core\learning\curiosity.py`, `C:\Users\baenb\projects\mae-core\mae_core\cognition\world_model.py`
   - Biological analog: Dopaminergic prediction error (Schultz 1997). The single most important learning signal in the brain.

### Tier 2: High Impact (would significantly improve Mae's capability)

4. **ATTEND (top-down gating)** -- Advisory + goals + hormones should modulate what signals reach the cortex. PatternBus should have a salience gate that the cortex controls.
   - Files affected: `C:\Users\baenb\projects\mae-core\mae_core\patterns\pattern_bus.py`, `C:\Users\baenb\projects\mae-core\mae_core\communication\signal_priority.py`
   - Biological analog: Thalamic reticular nucleus. Prefrontal cortex says "pay attention to threats" and the thalamus suppresses non-threat signals.

5. **HABITUATION** -- Salience decay on repeated identical patterns. If the same signal fires 10 steps in a row, its salience should decrease. Novel signals should be louder.
   - Files affected: `C:\Users\baenb\projects\mae-core\mae_core\patterns\pattern_cortex.py`, `C:\Users\baenb\projects\mae-core\mae_core\patterns\pattern_bus.py`
   - Biological analog: Every sensory system habituates. You stop hearing the fridge hum.

6. **INHIBIT (go/no-go)** -- DecisionRouter should be able to return "do nothing." Currently all tiers produce an action or fall through to default. There's no explicit suppression.
   - Files affected: `C:\Users\baenb\projects\mae-core\mae_core\cognition\decision_router.py`
   - Biological analog: Prefrontal inhibitory control. The ability to NOT act is as important as acting.

7. **REGULATE (active homeostasis)** -- PatternAdvisory -> EndocrineSystem wiring. Hormone levels modulate SignalPriorityResolver gain. Active parasympathetic recovery, not just passive decay.
   - Files affected: `C:\Users\baenb\projects\mae-core\main.py` (wiring), `C:\Users\baenb\projects\mae-core\mae_core\coordination\endocrine_system.py`
   - Biological analog: HPA axis closes the cortisol loop. Vagus nerve actively drives recovery.

8. **Wire ValidatedImagination -> WorldModel training** -- The `cognition.imagination_validated` EventBus channel is published to but has NO subscribers. Validation data (prediction accuracy) should train the WorldModel.
   - Files affected: `C:\Users\baenb\projects\mae-core\main.py` (Layer 15 wiring)
   - This is not a missing step but a missing connection that wastes existing computation.

9. **Wire CuriosityDrive -> agent reward** -- Intrinsic curiosity reward should be added to the agent's reward in `_learn()`. Currently computed but never consumed.
   - Files affected: `C:\Users\baenb\projects\mae-core\mae_core\agents\mycelial_agent.py` (_learn method)
   - This is not a missing step but a missing connection that defeats the purpose of the CuriosityDrive.

### Tier 3: Medium Impact (would improve biological fidelity)

10. **GWT Competition Gate** -- Not all patterns should reach the cortex. Implement a competition mechanism in PatternBus where signals compete for limited "broadcast slots." Losers are processed unconsciously (local effect only, no advisory influence).
    - Files affected: `C:\Users\baenb\projects\mae-core\mae_core\patterns\pattern_bus.py`
    - Biological analog: Global Workspace Theory's central claim.

11. **DEVELOP (structural change)** -- Apoptosis: remove underperforming agents. Re-differentiate: change agent roles based on need. Currently Mae can only grow and heal, never prune.
    - Files affected: `C:\Users\baenb\projects\mae-core\mae_core\agents\stem_cell.py`, `C:\Users\baenb\projects\mae-core\mae_core\morphogenesis\coordinator.py`
    - Biological analog: Apoptosis balances mitosis. Metamorphosis changes structure.

12. **ALLOSTASIS (predictive regulation)** -- EndocrineSystem should proactively adjust based on PatternCortex trends, not just react to events. If threat trend is rising for 3 steps, preemptively raise adrenaline before the threat arrives.
    - Files affected: `C:\Users\baenb\projects\mae-core\mae_core\coordination\endocrine_system.py`
    - Biological analog: Allostatic load. Your cortisol rises before the exam, not during it.

13. **METACOGNITION** -- A slow-cycle monitor that asks "are my decisions working?" by tracking decision quality over time. Can trigger executive override when decision quality degrades.
    - Files affected: Would need new system or extension to `C:\Users\baenb\projects\mae-core\mae_core\cognition\decision_router.py`
    - Biological analog: Anterior cingulate cortex monitors for errors and conflict.

14. **INTEROCEPTION feed to agents** -- Agents should read their own resource state (memory utilization, computational budget, fatigue proxy) and factor it into decisions.
    - Files affected: `C:\Users\baenb\projects\mae-core\mae_core\agents\mycelial_agent.py` (_observe method)
    - Biological analog: Hunger, thirst, fatigue are internal senses that drive behavior.

---

## Sources

- [Homeostasis - Biology LibreTexts](https://bio.libretexts.org/Bookshelves/Introductory_and_General_Biology/Introductory_Biology_(CK-12)/13:_Human_Biology/13.03:_Homeostasis)
- [Human body systems: Kenhub](https://www.kenhub.com/en/library/anatomy/human-body-systems)
- [An Analysis and Comparison of ACT-R and Soar](https://arxiv.org/abs/2201.09305)
- [Comparing SOAR, ACT-R, CLARION, DUAL - RoboticsBiz](https://roboticsbiz.com/comparing-four-cognitive-architectures-soar-act-r-clarion-and-dual/)
- [40 years of cognitive architectures - Springer](https://link.springer.com/article/10.1007/s10462-018-9646-y)
- [Free Energy Principle - Wikipedia](https://en.wikipedia.org/wiki/Free_energy_principle)
- [Expected Free Energy Minimization](https://www.emergentmind.com/topics/expected-free-energy-efe-minimization)
- [IIT 4.0 - PLOS Computational Biology](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011465)
- [Multi-Agent System Architecture Guide 2026](https://www.clickittech.com/ai/multi-agent-system-architecture/)
- [Designing Effective Multi-Agent Architectures - O'Reilly](https://www.oreilly.com/radar/designing-effective-multi-agent-architectures/)
- [Apoptosis - Programmed Cell Death - NCBI](https://www.ncbi.nlm.nih.gov/books/NBK499821/)
- [From autopoiesis to self-optimization - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S030326472300134X)