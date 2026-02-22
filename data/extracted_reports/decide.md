I now have comprehensive data from all five investigative dimensions. Let me compile the full audit report.

---

# DECIDE Step Audit Report

## Executive Summary

Mae's DECIDE step implements a three-tier decision cascade (Reflex / Habit / Prefrontal) in the `DecisionRouter`, integrated into the agent lifecycle through `MycelialAgent._decide()`. The architecture is fundamentally sound and biologically inspired. The advisory pipeline (PatternBus -> PatternCortex -> PatternAdvisory -> _route_with_advisory) provides organism-level intelligence flowing down to individual agent action selection. However, the audit reveals several gaps: the endocrine-to-router connection is wired but functionally impotent (the adrenaline re-check is identical to the first check), the GWT competition/broadcast mechanism is absent (actions are selected by cascade, not by competitive selection among parallel processes), habit lookup is exact-string-match only (no generalization), and the WorldModel deliberation in the prefrontal tier defaults to "pick the first action" when no world model is present. The system is triadic at the router level (3 tiers) but is NOT fractal at every scale -- there is no decision routing inside organs, tissues, or the organism level.

---

## 1. Data Flow Trace

### Full Path: Environment -> Action

```
Step lifecycle (BaseAgent.step):
  1. _signal_resolver.process()    -- Thalamic triage of queued signals
  2. _observe()                    -- Sense stigmergy, decay working memory,
                                      build state vector, read pattern advisory
  3. _decide()                     -- THE DECISION CASCADE (see below)
  4. _act(action)                  -- Execute, get reward
  5. _learn(action, reward)        -- Store memory, deposit markers, pattern sense
  6. _communicate()                -- GNN messages, pattern sharing
```

### _decide() Decision Cascade

```
_decide():
  |
  +-- [1] Advisory + Router available?
  |     YES -> _route_with_advisory(router, advisory, state_vec)
  |              |
  |              +-- Build stimulus from advisory.dominant_pattern
  |              +-- Build context from advisory fields
  |              +-- If advisory.confidence > 0.6:
  |              |     Force tier based on advisory.recommended_tier
  |              +-- Call router.route_decision(stimulus, context, force_tier)
  |              +-- If decision.tier == NONE: return None (fall through)
  |              +-- Otherwise: return decision.action_taken
  |     NO or FAILED -> fall through
  |
  +-- [2] Semantic memory search (hippocampus retrieval)
  |     state_vec + semantic_retriever -> search_similar_experiences(k=3)
  |     If past experiences with positive reward: return best action
  |     Otherwise: fall through
  |
  +-- [3] World model consultation (prefrontal simulation)
  |     world_model.use_world_model() -> imagined best action
  |     If action produced: return it
  |     Otherwise: fall through
  |
  +-- [4] Default: _select_action(current_state) -> returns 0
```

### DecisionRouter.route_decision() Internal Flow

```
route_decision(stimulus, context, available_actions, force_tier):
  |
  +-- force_tier specified? -> _force_tier() -> done
  |
  +-- Tier 1 (REFLEX): _check_reflex(stimulus)
  |     Sorted by priority (descending)
  |     Match: substring check (pattern.stimulus_pattern in stimulus)
  |     HIT -> return RouterDecision(REFLEX)
  |
  +-- Tier 2 (HABIT): _check_habit(stimulus)
  |     Exact string match (habit_lookup[stimulus])
  |     Strength >= 0.3 required
  |     HIT -> strengthen habit (+=0.05), return RouterDecision(HABIT)
  |
  +-- Endocrine Override: if adrenaline > 0.7
  |     Re-check reflexes (SAME check, no lower threshold)
  |     HIT -> return RouterDecision(REFLEX)
  |
  +-- Tier 3 (PREFRONTAL): _invoke_prefrontal(stimulus, context, actions)
  |     Priority:
  |       1. Custom prefrontal_fn if set
  |       2. WorldModel.step() simulation over available_actions
  |       3. First available action (confidence 0.6)
  |       4. Default dict {"type": "deliberate", "stimulus": ...} (confidence 0.5)
  |     Track for habit formation if auto_habit enabled
  |
  +-- All failed -> return RouterDecision(NONE)
```

### Advisory Pipeline (organism -> agent)

```
Per-step hook (_pattern_step_hook):
  1. PatternBus.process_step() -> PatternDigest
     (11 translators convert EventBus events -> PatternSignals)
  2. PatternCortex.process_digest() -> PatternAdvisory
     - 13-step Fibonacci sliding window
     - Domain streak tracking -> trend detection (Rule of Three)
     - Meta-pattern detection (strange loop)
     - Ancestral recall (MemoryBridge -> Qdrant)
     - Tier recommendation: threat>0.6="reflex", novelty>0.5="prefrontal", else="habit"
  3. Advisory written to _latest_advisory dict
  4. Published to EventBus "pattern.advisory"

Agent reads:
  _observe() reads _pattern_advisory_ref -> self._current_advisory
  _decide() uses advisory in _route_with_advisory()
```

### Endocrine Influence on Decisions

```
EndocrineSystem:
  - Adrenaline > 0.7 -> DecisionRouter re-checks reflexes (lines 195-210)
  - get_reflex_bias() -> adrenaline*0.7 + melatonin*0.3 (helper, not consumed by router)
  - register_decision_router(dr) -> calls dr.set_reflex_bias(level) on adrenaline release
  - BUT: DecisionRouter has NO set_reflex_bias() method -> silently fails via hasattr check
```

---

## 2. Mathematical Identity Compliance

| Requirement | Status | Details |
|---|---|---|
| **Three-tier routing (reflex/habit/deliberation)** | PASS | DecisionTier enum: REFLEX, HABIT, PREFRONTAL + NONE fallback. Cascade order correct. |
| **Competition/selection (GWT)** | PARTIAL | Current: serial cascade (first match wins). GWT requires parallel competition with broadcast. No competing processes bidding for workspace access. Decision is hierarchical, not competitive. |
| **Triadic structure** | PASS | 3 tiers = triadic. The triad is: fast/automatic/deliberate. |
| **Fractal at every scale** | FAIL | DecisionRouter exists only at agent level. No 3-tier routing at: tissue level, organ level, organism level, colony level. The PatternCortex recommends a tier but doesn't implement routing. The FractalGenerator creates structure but doesn't replicate decision logic. |
| **GWT broadcast mechanism** | FAIL | After decision, no broadcast to other subsystems. GWT says the winning coalition should be broadcast globally. EventBus publishes "cognition.decision_routed" but this is passive logging, not GWT-style broadcast that recruits other processors. |

---

## 3. Biological Comparison

| Biological Reality | Mae Implementation | Gap Analysis |
|---|---|---|
| **Reflex arc**: Monosynaptic or polysynaptic, spinal cord, <50ms, hardwired | **ReflexPattern**: Substring matching against registered patterns, fixed actions, hardcoded "danger/threat/collision" defaults | Correct metaphor. Gap: biological reflexes can be modulated by descending cortical inhibition (executive veto). Mae has executive_override() but only for forcing prefrontal, not for suppressing reflexes. |
| **Basal ganglia habit**: Striatum learns stimulus-response via dopaminergic RPE, direct pathway (Go) vs indirect pathway (NoGo), cortico-basal ganglia-thalamic loop | **Habit**: Exact string match on stimulus, strength parameter, strengthens with use, automatic formation after 5 consistent prefrontal decisions | Major gap: No Go/NoGo dual pathway. No competition between actions -- only one habit per stimulus. No dopamine modulation of habit strength. No habit decay from disuse. No generalization (must be exact stimulus string). |
| **Prefrontal deliberation**: Model-based reasoning, working memory integration, prospective simulation, conflict monitoring, inhibitory control | **_invoke_prefrontal**: WorldModel.step() evaluates actions deterministically, returns best predicted reward | Partial. Has simulation-based evaluation. Missing: working memory integration (exists in memory system but not fed to prefrontal), conflict monitoring (no detection of competing options), inhibitory control (no suppression of prepotent responses). |
| **Dopamine RPE**: Prediction error drives learning, phasic vs tonic dopamine, D1 (direct) vs D2 (indirect) receptor types | EndocrineSystem.DOPAMINE: Released on reward/novelty, modulates CuriosityDrive exploration bonus | Shallow. No reward prediction error computation. No differential D1/D2 pathway effects. Dopamine is a scalar level, not a prediction error signal. |
| **Thalamic relay**: Filters and prioritizes all sensory input to cortex, gates information flow | SignalPriorityResolver: Priority scoring, budget enforcement, tier mapping, per-agent thalamus | Good match. Budget enforcement is biological. Tier mapping to decision tiers is creative and appropriate. |
| **Adrenaline (norepinephrine)**: Fight-or-flight, locus coeruleus, increases arousal, shifts to fast reactive processing | EndocrineSystem.ADRENALINE -> DecisionRouter "re-checks reflexes" | Implementation is broken: re-check uses identical matching logic (no lowered threshold). set_reflex_bias() is called but method doesn't exist on DecisionRouter. |
| **Habit formation**: Gradual, requires hundreds-thousands of repetitions, chunking, context-dependent | Auto-habit: 5 identical prefrontal decisions -> habit | Threshold too low. Real habits require extensive repetition. No context-dependency (same stimulus in different contexts should form different habits). No chunking of action sequences. |
| **Cortical-basal ganglia loops**: Multiple parallel loops (motor, oculomotor, prefrontal, limbic) | Single decision pathway | Missing parallel loops. Biology has 4-5 parallel cortico-basal ganglia loops handling different action domains simultaneously. |

---

## 4. External State of Art Comparison

| System/Paper | What It Does | Mae's Gap |
|---|---|---|
| **Gurney, Prescott & Redgrave (2001)** - GPR basal ganglia model | Two-channel model with STN-GPe selection loop implementing salience-based competition between action channels. Actions compete in parallel. | Mae uses serial cascade, not parallel competition. No salience-based competition between alternative actions. |
| **TD2Q model** (direct/indirect pathway RL) | Dual Q-matrices: G-matrix (Go/direct) and N-matrix (NoGo/indirect), updated by temporal difference RPE | Mae has single habit lookup, no dual pathway, no temporal difference learning in habit formation |
| **Options Framework** (Sutton, Precup, Singh 1999) | Temporally extended actions with initiation sets, policies, and termination conditions. Hierarchical composition. | Mae's habits are single-step stimulus-response. No temporal extension, no initiation conditions, no hierarchical composition of sub-policies. |
| **GWT Selection-Broadcast** (Baars 2024) | Parallel processes compete for workspace access. Winner is broadcast globally, recruiting all processors. | Mae's EventBus publishes decision events but doesn't implement competitive workspace access. No broadcast-recruitment cycle. |
| **DreamerV3** (Hafner et al. 2023) | Learned world model for policy optimization via imagined rollouts, uncertainty-aware | Mae's WorldModel supports rollouts and ensemble uncertainty but prefrontal tier uses single-step evaluation only, not multi-step planning |
| **Interacting cortico-BG-thalamic loops** (Cell 2025) | Multiple parallel loops with hierarchical organization, cognitive maps for model-based control at higher levels | Mae has single decision pipeline, not multiple parallel loops |
| **CogLinks** (PLOS Biology 2024) | Dual corticostriatal + frontal-thalamic architecture for handling different uncertainty types | Mae doesn't distinguish epistemic vs aleatory uncertainty in decision routing |

---

## 5. Specific Bugs Found

### BUG 1: Endocrine-Router Wiring is Broken (Silent Failure)

**File**: `C:\Users\baenb\projects\mae-core\mae_core\coordination\endocrine_system.py` (lines 456-464)
**File**: `C:\Users\baenb\projects\mae-core\mae_core\cognition\decision_router.py`

The endocrine system calls `dr.set_reflex_bias(level)` when adrenaline is released, but `DecisionRouter` has no `set_reflex_bias()` method. The registration code uses a `hasattr` check that silently falls through. The endocrine system's `register_decision_router()` method is effectively dead code.

### BUG 2: Adrenaline Override Re-Check is Identical (No Effect)

**File**: `C:\Users\baenb\projects\mae-core\mae_core\cognition\decision_router.py` (lines 194-210)

The comment says "high adrenaline - try reflex again with lower match bar" but the code calls `self._check_reflex(stimulus)` with the exact same logic as the first check. If the first reflex check didn't match, the second one never will either. This entire block (lines 194-210) is dead code in practice.

### BUG 3: _route_with_advisory Passes available_actions=None

**File**: `C:\Users\baenb\projects\mae-core\mae_core\agents\mycelial_agent.py` (line 384)

The advisory routing always passes `available_actions=None` to `router.route_decision()`. This means the prefrontal tier inside the router never gets a list of actions to evaluate with the WorldModel, falling to the default dict `{"type": "deliberate"}` every time. The WorldModel simulation path in `_invoke_prefrontal` requires `available_actions` to be non-None.

---

## 6. Ranked Upgrade Recommendations

### Priority 1: Fix Broken Wiring (Bug Fixes)

1. **Add `set_reflex_bias()` to DecisionRouter** -- Store the endocrine bias value and use it to modulate the reflex matching threshold (e.g., lower the substring match requirement when adrenaline is high). This fixes the dead endocrine-router connection.

2. **Make adrenaline re-check actually different** -- When adrenaline > 0.7, the reflex re-check should use a fuzzier/broader matching strategy (e.g., partial string matching, lower confidence threshold, check fewer characters). Currently it's a copy-paste of the first check.

3. **Pass available_actions to advisory routing** -- Compute the agent's available action set and pass it through `_route_with_advisory` to `route_decision()` so the prefrontal tier can actually evaluate alternatives with the WorldModel.

### Priority 2: Biological Accuracy Improvements

4. **Dual-pathway habit system (Go/NoGo)** -- Replace single habit lookup with two competing pathways: one promoting action (Go/direct) and one inhibiting (NoGo/indirect). This is how the real basal ganglia works and enables action competition rather than first-match-wins.

5. **Dopamine as RPE, not scalar level** -- Compute reward prediction error (actual reward minus predicted reward) and use that delta to modulate habit strengthening. Positive RPE strengthens the Go pathway, negative RPE strengthens the NoGo pathway. This connects the endocrine dopamine to actual learning.

6. **Fuzzy/semantic habit matching** -- Replace exact string matching with embedding-based similarity (the SemanticRetriever already exists). Similar stimuli should activate similar habits. This mirrors biological pattern completion in the striatum.

7. **Multi-step prefrontal rollouts** -- The WorldModel supports `rollout()` but the prefrontal tier uses `step()` (single-step). Use `rollout()` with the planning horizon from `AdvancedFeaturesMixin` for proper deliberation.

### Priority 3: Mathematical Identity Compliance

8. **GWT competitive selection** -- Replace the serial cascade with a competitive workspace. All three tiers evaluate in parallel, produce candidate actions with confidence scores, and compete for "broadcast" access. The winner's action is broadcast to all subsystems via EventBus. This is the biggest structural gap vs. the mathematical identity's GWT requirement.

9. **Fractal decision routing** -- Instantiate decision routing at every scale: tissue-level (agent triads collectively decide), organ-level (organ has its own 3-tier router), organism-level (PatternCortex already recommends tiers; give it action selection capability). Each scale's DECIDE should be triadic.

10. **Executive veto / inhibitory control** -- Add the ability for higher tiers to suppress lower tier responses (like cortex suppressing a reflex). Currently executive_override only forces prefrontal; it should also be able to inhibit specific reflexes or habits.

### Priority 4: State-of-Art Enhancements

11. **Options/temporally-extended habits** -- Habits should be multi-step policies (like Sutton's options), not single stimulus-response pairs. A habit could be "when X, execute this sequence of 5 actions." Include initiation sets and termination conditions.

12. **Uncertainty-aware tier selection** -- Route based on epistemic uncertainty (how uncertain is the model about this situation?) rather than just pattern advisory. High uncertainty -> prefrontal. Low uncertainty -> habit. This matches CogLinks architecture.

13. **Parallel cortico-BG loops** -- Implement multiple parallel decision loops for different action domains (motor actions, communication actions, learning meta-actions), each with their own reflex/habit/prefrontal cascade. Currently everything goes through one pipeline.

---

## Sources

### Neuroscience
- [Basal ganglia components in decision-making dynamics (PLOS Biology 2024)](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3002978)
- [Computational bottleneck of basal ganglia output (eNeuro 2024)](https://www.eneuro.org/content/12/4/ENEURO.0431-23.2024)
- [Interacting cortico-BG-thalamocortical loops (Cell/Trends in Neurosciences 2025)](https://www.cell.com/trends/neurosciences/fulltext/S0166-2236(25)00192-4)
- [Role of basal ganglia in habit formation (Graybiel 2008)](https://people.duke.edu/~hy43/role%20of%20basal.pdf)
- [Computational models of prefrontal cortex (NCBI)](https://www.ncbi.nlm.nih.gov/books/NBK609777/)
- [Habit learning in hierarchical cortex-basal ganglia loops](https://onlinelibrary.wiley.com/doi/pdf/10.1111/ejn.14730)
- [Dopaminergic action prediction errors as value-free teaching signal (Nature 2025)](https://www.nature.com/articles/s41586-025-09008-9)
- [Direct and indirect pathways improve RL performance (PLOS Comp Bio)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011385)
- [Dynamics of striatal action selection and RL (eLife 2024)](https://elifesciences.org/reviewed-preprints/101747)

### GWT and Consciousness
- [Global Workspace Theory (GWT) and Prefrontal Cortex (Frontiers 2021)](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2021.749868/full)
- [GWT Selection-Broadcast Cycle (arxiv 2025)](https://arxiv.org/html/2505.13969v1)
- [Global Workspace Theory - Wikipedia](https://en.wikipedia.org/wiki/Global_workspace_theory)

### Hierarchical RL and Multi-Agent
- [Taxonomy of Hierarchical Multi-Agent Systems (arxiv 2025)](https://arxiv.org/html/2508.12683)
- [Hierarchical MARL for Cyber Network Defense (2024)](https://arxiv.org/abs/2410.17351)
- [Comparing Top 5 AI Agent Architectures 2025 (MarkTechPost)](https://www.marktechpost.com/2025/11/15/comparing-the-top-5-ai-agent-architectures-in-2025-hierarchical-swarm-meta-learning-modular-evolutionary/)

### Computational Models
- [GPR basal ganglia model (Gurney, Prescott, Redgrave 2001)](https://pubmed.ncbi.nlm.nih.gov/11417052/)
- [Neural basis for uncertainty processing in hierarchical decision making (Nature Comm 2025)](https://www.nature.com/articles/s41467-025-63994-y)
- [Dopamine regulates decision thresholds in RL (Nature Comm 2023)](https://www.nature.com/articles/s41467-023-41130-y)
- [Computational model of action specification in basal ganglia (bioRxiv 2025)](https://www.biorxiv.org/content/10.1101/2025.08.12.669938v1.full)

---

## Key Files Examined

| File | Role in DECIDE |
|---|---|
| `C:\Users\baenb\projects\mae-core\mae_core\cognition\decision_router.py` | Three-tier router (core DECIDE engine) |
| `C:\Users\baenb\projects\mae-core\mae_core\agents\mycelial_agent.py` | Agent lifecycle: _decide(), _route_with_advisory(), _observe() |
| `C:\Users\baenb\projects\mae-core\mae_core\agents\base_agent.py` | Step lifecycle, _select_action() default |
| `C:\Users\baenb\projects\mae-core\mae_core\coordination\endocrine_system.py` | Hormone modulation of decision-making |
| `C:\Users\baenb\projects\mae-core\mae_core\agents\mixins\advanced_features.py` | WorldModel integration, use_world_model() |
| `C:\Users\baenb\projects\mae-core\mae_core\cognition\world_model.py` | Imagination engine for prefrontal deliberation |
| `C:\Users\baenb\projects\mae-core\mae_core\patterns\pattern_cortex.py` | Advisory generation, tier recommendation |
| `C:\Users\baenb\projects\mae-core\mae_core\communication\signal_priority.py` | Thalamic signal triage |
| `C:\Users\baenb\projects\mae-core\main.py` | Bootstrap wiring of all DECIDE components |