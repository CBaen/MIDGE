> Generated from 10-agent audit conducted 2026-02-11. ~50 sub-agents. Sources: biology papers, GitHub, research papers, full codebase trace.

# Mae Audit: Upgrade Roadmap

Every upgrade recommendation from all 10 audit reports, deduplicated, ranked by impact, with file references and biological analogs. This is the implementation guide.

---

## Tier 1: Critical (Fix Broken Fundamentals)

These must be done first. Without them, downstream improvements have no effect.

### 1.1 Implement Action Environment + Override `_act()`
- **Source reports:** ACT, HIDDEN STEPS
- **Files:** `mae_core/agents/base_agent.py` (lines 92-95), `mae_core/agents/mycelial_agent.py`
- **Problem:** `_act()` is a two-line stub returning 0.0. No action changes any environmental state. All reward is zero.
- **Solution:** Build an environment (task pool, grid, resource landscape, or abstract problem space) that changes when agents act. Override `_act()` in MycelialAgent with triadic execution: Plan (efference copy) -> Execute (change environment) -> Verify (compare prediction vs actual).
- **Biological analog:** Building the musculoskeletal system. Mae has a brain but no body.
- **Effort:** HIGH
- **Impact:** TRANSFORMATIVE -- nothing else works without this.

### 1.2 Add PREDICT Step Before `_observe()`
- **Source reports:** SENSE, HIDDEN STEPS, LEARN
- **Files:** `mae_core/agents/base_agent.py` (step lifecycle), `mae_core/agents/mycelial_agent.py`, `mae_core/cognition/world_model.py`
- **Problem:** Mae is reactive, not predictive. The mathematical identity claims FEP compliance but no prediction occurs before sensing.
- **Solution:** Add `_predict()` to BaseAgent step lifecycle, called before `_observe()`. WorldModel generates expected next state/reward. After `_observe()`, compute prediction error.
- **Biological analog:** Predictive coding in visual cortex. You predict what you'll see before you open your eyes.
- **Effort:** MEDIUM
- **Impact:** Transforms Mae from reactive to predictive. Satisfies FEP. Enables COMPARE step.

### 1.3 Fix `_learn_from_batch()` to Actually Update Weights
- **Source reports:** LEARN
- **Files:** `mae_core/agents/mixins/episodic_memory.py` (line 147)
- **Problem:** Computes TD errors but updates no weights. No policy or value network is modified.
- **Solution:** Implement actual parameter updates. At minimum, update a policy/value network from TD error signal.
- **Biological analog:** Synaptic plasticity -- without weight changes, the organism cannot adapt.
- **Effort:** MEDIUM
- **Impact:** Critical -- enables all downstream learning improvements.

### 1.4 Fix MemoryConsolidator Interface Mismatch
- **Source reports:** LEARN
- **Files:** `mae_core/memory/memory_consolidator.py` (lines 101-156)
- **Problem:** Calls `agent.get_learning_rate()` and `agent.set_learning_rate()` which do not exist.
- **Solution:** Add these methods to BaseAgent, or redesign the consolidator interface.
- **Effort:** LOW
- **Impact:** Unblocks the consolidation pathway.

### 1.5 Fix Endocrine-Router Wiring (Two Bugs)
- **Source reports:** DECIDE
- **Files:** `mae_core/coordination/endocrine_system.py` (lines 456-464), `mae_core/cognition/decision_router.py` (lines 194-210)
- **Problem:** (a) `set_reflex_bias()` does not exist on DecisionRouter. (b) Adrenaline re-check uses identical logic as first check.
- **Solution:** (a) Add `set_reflex_bias()` to DecisionRouter. (b) Implement fuzzier matching for adrenaline-driven re-check.
- **Effort:** LOW
- **Impact:** Makes endocrine influence on decisions functional.

### 1.6 Fix Semantic Recall Return Type Bug
- **Source reports:** RECALL
- **Files:** `mae_core/agents/mycelial_agent.py` (lines 326-331)
- **Problem:** `search_similar_experiences()` returns `SemanticQuery`, code treats it as list.
- **Solution:** Access `.experiences` attribute before calling `max()`.
- **Effort:** LOW
- **Impact:** Fixes the semantic memory recall path.

### 1.7 Pass `available_actions` to Advisory Routing
- **Source reports:** DECIDE
- **Files:** `mae_core/agents/mycelial_agent.py` (line 384)
- **Problem:** Always passes `available_actions=None`, preventing WorldModel simulation.
- **Solution:** Compute available action set and pass it through.
- **Effort:** LOW
- **Impact:** Enables prefrontal deliberation with WorldModel.

### 1.8 Fix PatternBus Signal Mutation Bug
- **Source reports:** SENSE
- **Files:** `mae_core/patterns/pattern_bus.py` (lines 214-216)
- **Problem:** `_detect_correlations()` mutates shared PatternSignal objects in-place.
- **Solution:** Create copies before mutation, or create new synthetic signals.
- **Effort:** LOW
- **Impact:** Fixes data integrity violation.

### 1.9 Fix Z-Score Self-Inclusion Bug
- **Source reports:** SENSE
- **Files:** `mae_core/patterns/pattern_sense.py` (lines 201-213)
- **Problem:** Current value included in its own baseline, biasing z-score toward zero.
- **Solution:** Compute mean/std over `rewards[:-1]`, z-score on `rewards[-1]`.
- **Effort:** LOW
- **Impact:** Improves surprise detection sensitivity.

---

## Tier 2: High Impact (Wire Existing Systems Together)

These are high-value, mostly low-effort connections between existing systems.

### 2.1 Wire CuriosityDrive -> Agent Reward
- **Source reports:** HIDDEN STEPS, LEARN
- **Files:** `mae_core/agents/mycelial_agent.py` (_learn method), `mae_core/learning/curiosity.py`
- **Problem:** Intrinsic curiosity reward computed but never added to agent's actual reward.
- **Biological analog:** Dopaminergic novelty signal driving exploration.
- **Effort:** LOW

### 2.2 Wire ValidatedImagination -> WorldModel Training
- **Source reports:** HIDDEN STEPS
- **Files:** `main.py` (Layer 15 wiring), EventBus channel `cognition.imagination_validated`
- **Problem:** Validation data (was_accurate, reward_error) published with zero subscribers.
- **Biological analog:** Cerebellum learning from prediction errors.
- **Effort:** LOW

### 2.3 Wire PatternDistiller into PatternConsolidator
- **Source reports:** CONSOLIDATE
- **Files:** `mae_core/patterns/pattern_consolidator.py` (line 51)
- **Problem:** Distiller injected but never called. Dead code.
- **Biological analog:** Knowledge distillation during sleep.
- **Effort:** LOW

### 2.4 Wire AwarenessPulse Anomalies -> AutoHealer
- **Source reports:** HEAL
- **Files:** `mae_core/backbone/holon_protocol.py` (line 494), `mae_core/emergent/auto_healer.py`
- **Problem:** `holon.anomaly_detected` has no subscribers. Anomalies are logged and forgotten.
- **Biological analog:** Immune surveillance detecting problems.
- **Effort:** LOW

### 2.5 Wire PatternAdvisory -> EndocrineSystem
- **Source reports:** HIDDEN STEPS
- **Files:** `main.py` (wiring), `mae_core/coordination/endocrine_system.py`
- **Problem:** High threat advisory should trigger adrenaline. Advisory publishes but Endocrine does not subscribe.
- **Biological analog:** Amygdala -> HPA axis activation.
- **Effort:** LOW

### 2.6 Wire EndocrineSystem -> SignalPriorityResolver
- **Source reports:** HIDDEN STEPS
- **Files:** `mae_core/communication/signal_priority.py`, `mae_core/coordination/endocrine_system.py`
- **Problem:** Hormone levels should modulate signal gain. Currently static.
- **Biological analog:** Reticular Activating System modulates thalamic gain.
- **Effort:** LOW

### 2.7 Wire Self-Awareness into Agent Step Lifecycle
- **Source reports:** SELF-AWARENESS
- **Files:** `mae_core/agents/mycelial_agent.py` (_observe, _decide)
- **Problem:** holon_know_self/up/down/peers never called during agent behavior.
- **Solution:** Call in `_observe()`, inject results into state/decision context.
- **Biological analog:** Closes the Strange Loop -- self-knowledge feeds behavior.
- **Effort:** MEDIUM

### 2.8 Implement Competitive Ignition (GWT)
- **Source reports:** ADVISE, DECIDE, HIDDEN STEPS
- **Files:** `mae_core/patterns/pattern_cortex.py`
- **Problem:** Every digest produces advisory. No competition, no ignition threshold.
- **Solution:** Add `should_ignite()` check. Below threshold = quiet advisory. Above = full broadcast.
- **Biological analog:** NMDA-mediated recurrent amplification in Global Workspace.
- **Effort:** MEDIUM

### 2.9 Add Top-Down Attentional Gating (TRN Analog)
- **Source reports:** SENSE, ADVISE, HIDDEN STEPS
- **Files:** `mae_core/patterns/pattern_bus.py`, `mae_core/patterns/pattern_cortex.py`, `main.py`
- **Problem:** No cortex -> bus feedback. Bus processes everything equally.
- **Solution:** Add `attention_weights` to PatternBus. Cortex controls what the bus pays attention to.
- **Biological analog:** Thalamic Reticular Nucleus -- cortex tells thalamus what to attend to.
- **Effort:** MEDIUM

### 2.10 Implement Habituation / Sensory Adaptation
- **Source reports:** SENSE, HIDDEN STEPS
- **Files:** `mae_core/patterns/pattern_sense.py` (lines 94-242), `mae_core/patterns/pattern_bus.py`
- **Problem:** No salience decay on repeated signals. A sustained pattern fires identically every step.
- **Solution:** Add `_salience_decay` dict. Each step a signal repeats, salience decays (e.g., *0.85). Reset on change.
- **Biological analog:** Neural adaptation -- you stop hearing the fridge hum.
- **Effort:** LOW

### 2.11 Wire WorldModel Training into `_learn()`
- **Source reports:** LEARN
- **Files:** `mae_core/agents/mycelial_agent.py` (_learn), `mae_core/cognition/world_model.py`
- **Problem:** WorldModel is never trained on observed transitions during step-level learning.
- **Solution:** After each experience, train WorldModel on the (state, action, next_state, reward) transition.
- **Biological analog:** Cerebellum learning internal models from experience.
- **Effort:** MEDIUM

### 2.12 Pass signal_context to `store_experience()`
- **Source reports:** LEARN
- **Files:** `mae_core/agents/mycelial_agent.py` (_learn), `mae_core/agents/mixins/episodic_memory.py`
- **Problem:** Consensus priority path exists but signal_context is never passed.
- **Solution:** Build signal_context from current advisory/pattern state and pass to store_experience().
- **Effort:** LOW

### 2.13 Gate Consolidation by Circadian Phase
- **Source reports:** CONSOLIDATE
- **Files:** `mae_core/patterns/pattern_consolidator.py`
- **Problem:** Consolidator fires every 89 steps regardless of circadian phase.
- **Solution:** Check CircadianRhythm.should_consolidate_memory() before consolidating.
- **Biological analog:** Sleep spindles gate consolidation to sleep phases.
- **Effort:** LOW

### 2.14 Add Endocrine Modulation of PatternBus Gain
- **Source reports:** ADVISE
- **Files:** `mae_core/patterns/pattern_bus.py`, `main.py`
- **Problem:** PatternBus processes signals identically regardless of Mae's stress/arousal state.
- **Solution:** Inject EndocrineSystem reference. High cortisol/adrenaline = lower THREAT threshold. High melatonin = higher thresholds.
- **Biological analog:** Reticular Activating System modulates thalamic gain.
- **Effort:** MEDIUM

### 2.15 Priority-Based Inbox Draining
- **Source reports:** ADVISE
- **Files:** `mae_core/patterns/pattern_bus.py` (lines 127-134)
- **Problem:** MAX_SIGNALS_PER_STEP=50 takes signals in FIFO order, not priority.
- **Solution:** Sort inbox by salience before draining.
- **Effort:** LOW

---

## Tier 3: Architectural (Deepen Compliance)

These require more structural changes.

### 3.1 Implement COMPARE Step (Prediction Error as Primary Signal)
- **Source reports:** HIDDEN STEPS, LEARN
- **Files:** `mae_core/agents/mycelial_agent.py`, `mae_core/learning/curiosity.py`, `mae_core/cognition/world_model.py`
- **Problem:** No prediction error computation bridges PREDICT and LEARN.
- **Solution:** After PREDICT + SENSE, compute error = predicted - actual. Feed to CuriosityDrive (intrinsic reward), WorldModel (training signal), EndocrineSystem (surprise signal).
- **Biological analog:** Dopaminergic prediction error (Schultz 1997).
- **Effort:** MEDIUM

### 3.2 Implement Goal Management
- **Source reports:** HIDDEN STEPS
- **Files:** New: `mae_core/cognition/goal_manager.py`
- **Problem:** Agents have no explicit goals that persist across steps or drive attention.
- **Solution:** BDI (Belief-Desire-Intention) architecture with goal stack, goal selection, goal persistence.
- **Biological analog:** Prefrontal cortex maintains working goals.
- **Effort:** HIGH

### 3.3 Make Sensing Fractal
- **Source reports:** SENSE
- **Files:** New: `mae_core/patterns/tissue_sense.py`, `mae_core/patterns/organ_sense.py`. Existing: `mae_core/patterns/pattern_cortex.py`
- **Problem:** Each sensing scale uses different architecture.
- **Solution:** Define SenseProtocol with 3 detectors (trend, repetition, surprise). Implement at every scale.
- **Effort:** HIGH

### 3.4 Make Decision Routing Fractal
- **Source reports:** DECIDE
- **Problem:** DecisionRouter only at agent level.
- **Solution:** 3-tier routing at tissue (triads collectively decide), organ, and organism levels.
- **Effort:** HIGH

### 3.5 Add INHIBIT / Go-No-Go to DecisionRouter
- **Source reports:** DECIDE, HIDDEN STEPS
- **Files:** `mae_core/cognition/decision_router.py`
- **Problem:** All tiers produce an action or fall through. No explicit "do nothing."
- **Solution:** Add INHIBIT tier that can suppress action based on context.
- **Biological analog:** Prefrontal inhibitory control. Go/no-go pathways.
- **Effort:** LOW

### 3.6 Implement Reconsolidation
- **Source reports:** CONSOLIDATE, RECALL
- **Files:** `mae_core/patterns/pattern_consolidator.py`, `mae_core/memory/memory_bridge.py`
- **Problem:** Ancestral memory is immutable once stored.
- **Solution:** Before storing new pattern, search for similar existing ones. If match found, merge/update via PatternDistiller.merge_with_existing().
- **Biological analog:** Retrieved memories become labile and can be modified.
- **Effort:** MEDIUM

### 3.7 Add Dual-Pathway Habit System (Go/NoGo)
- **Source reports:** DECIDE
- **Files:** `mae_core/cognition/decision_router.py`
- **Problem:** Single habit lookup with exact string match.
- **Solution:** Two competing pathways: Go (promote action) and NoGo (inhibit action). Positive RPE strengthens Go, negative strengthens NoGo.
- **Biological analog:** Direct and indirect pathways in striatum.
- **Effort:** MEDIUM

### 3.8 Meta-Healing Triad
- **Source reports:** HEAL
- **Files:** `mae_core/emergent/auto_healer.py`, `mae_core/learning/haven.py`, `mae_core/emergent/somatic_map.py`
- **Problem:** Healing system cannot heal itself.
- **Solution:** AutoHealer + HAVEN + SomaticMap each monitor the other two. If one stops heartbeating, the other two detect it.
- **Biological analog:** Immune system self-monitoring.
- **Effort:** MEDIUM

### 3.9 Implement Precision-Weighted Integration
- **Source reports:** ADVISE
- **Files:** `mae_core/patterns/pattern_cortex.py` (lines 296-319)
- **Problem:** All signals contribute equally. Confidence field carried but not used.
- **Solution:** Weight each signal's contribution by its confidence during aggregate computation.
- **Biological analog:** Precision-weighting in predictive processing.
- **Effort:** LOW

### 3.10 Add Efference Copy Mechanism
- **Source reports:** SENSE, ACT
- **Files:** `mae_core/patterns/pattern_sense.py`, `mae_core/agents/mycelial_agent.py`
- **Problem:** Sense system has no awareness of what actions are being taken.
- **Solution:** Maintain action -> expected_reward mapping. Compare actual vs expected. Distinguish self-caused from externally-caused changes.
- **Biological analog:** Motor cortex sends copy of command to cerebellum for prediction.
- **Effort:** MEDIUM

### 3.11 Implement DEVELOP Step (Apoptosis + Re-differentiation)
- **Source reports:** HIDDEN STEPS, HEAL
- **Files:** `mae_core/agents/stem_cell.py`, `mae_core/morphogenesis/coordinator.py`
- **Problem:** Mae can only grow and heal, never prune.
- **Solution:** Slow-timescale structural change: remove underperforming agents, re-differentiate roles, fractal reorganization.
- **Biological analog:** Apoptosis balances mitosis. Metamorphosis changes structure.
- **Effort:** HIGH

### 3.12 Generative Self-Model (Strange Loop)
- **Source reports:** SELF-AWARENESS
- **Files:** `mae_core/backbone/holon_protocol.py`
- **Problem:** holon_know_self() returns static dict.
- **Solution:** Predictive self-model that predicts own next state, detects divergence, updates from prediction error, feeds confidence back into decisions.
- **Biological analog:** Lipson-style self-modeling robots.
- **Effort:** HIGH

### 3.13 Triadic Recall Verification
- **Source reports:** RECALL
- **Problem:** All recall is dyadic (caller -> store -> result).
- **Solution:** Every recall passes through Querier -> Store -> Witness triad. Witness validates relevance and staleness.
- **Effort:** MEDIUM

### 3.14 Fuzzy/Semantic Habit Matching
- **Source reports:** DECIDE
- **Files:** `mae_core/cognition/decision_router.py`
- **Problem:** Habit lookup is exact string match only.
- **Solution:** Use embedding-based similarity via SemanticRetriever. Similar stimuli activate similar habits.
- **Biological analog:** Pattern completion in striatum.
- **Effort:** MEDIUM

### 3.15 Multi-Step Prefrontal Rollouts
- **Source reports:** DECIDE
- **Files:** `mae_core/agents/mixins/advanced_features.py`
- **Problem:** Prefrontal tier uses WorldModel.step() (single-step). WorldModel supports rollout().
- **Solution:** Use rollout() with planning horizon from AdvancedFeaturesMixin.
- **Effort:** LOW

### 3.16 Add REGULATE Step (Active Homeostasis)
- **Source reports:** HIDDEN STEPS
- **Files:** `mae_core/coordination/endocrine_system.py`, `main.py`
- **Problem:** Endocrine system reacts. It never preemptively adjusts (allostasis).
- **Solution:** PatternCortex trends drive preemptive hormonal adjustment. Active parasympathetic recovery.
- **Biological analog:** HPA axis closes the cortisol loop. Vagus nerve actively drives recovery.
- **Effort:** MEDIUM

### 3.17 Coordinate Two Consolidation Systems
- **Source reports:** CONSOLIDATE
- **Files:** `mae_core/patterns/pattern_consolidator.py`, `mae_core/memory/memory_consolidator.py`
- **Problem:** PatternConsolidator (89 steps) and MemoryConsolidator (1000 steps) operate independently.
- **Solution:** ConsolidationCoordinator orchestrates both within circadian CONSOLIDATION phase. Triadic: extract, replay, distill.
- **Effort:** MEDIUM

### 3.18 Add Lateral Inhibition to PatternSense
- **Source reports:** SENSE
- **Files:** `mae_core/patterns/pattern_sense.py`
- **Problem:** Three detectors are independent. No contrast enhancement.
- **Solution:** Strong surprise suppresses trend signal. Action repetition boosts surprise sensitivity.
- **Biological analog:** Lateral inhibition sharpens signal boundaries.
- **Effort:** LOW

### 3.19 Immune Memory (Adaptive Healing)
- **Source reports:** HEAL
- **Files:** `mae_core/emergent/auto_healer.py`
- **Problem:** No learning from past healing. Same failure type gets full 3-phase treatment every time.
- **Solution:** Store successful healing records with root causes and recovery actions. On recurrence, skip assessment.
- **Biological analog:** Memory B-cells provide rapid secondary immune response.
- **Effort:** MEDIUM

### 3.20 Sequence-Aware Replay
- **Source reports:** CONSOLIDATE
- **Problem:** Replay samples individual experiences independently.
- **Solution:** Add trajectory buffer. During consolidation, sample contiguous subsequences.
- **Biological analog:** Sharp-wave ripples replay temporal sequences in compressed time.
- **Effort:** MEDIUM

---

## Summary

| Tier | Count | Effort Breakdown |
|------|-------|-----------------|
| Tier 1 (Critical) | 9 | 1 HIGH, 2 MEDIUM, 6 LOW |
| Tier 2 (High Impact) | 15 | 5 MEDIUM, 10 LOW |
| Tier 3 (Architectural) | 20 | 5 HIGH, 10 MEDIUM, 5 LOW |
| **Total** | **44** | |

**Recommended implementation order:** Tier 1 items first (especially 1.1 and 1.2), then Tier 2's LOW-effort wiring tasks (2.1-2.6), then Tier 2's MEDIUM-effort features (2.7-2.14), then Tier 3 as capacity allows.
