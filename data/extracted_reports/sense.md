I now have all the data I need to write the comprehensive audit report. Let me synthesize everything into the final output.

---

# SENSE AUDIT REPORT

## Executive Summary

Mae's SENSE/DETECT step is a multi-layered sensory pipeline that translates environmental stimuli into pattern signals, correlates them across agents and domains, integrates them over time, and feeds advisories into decision-making. The architecture is biologically inspired and structurally sound. However, **it is not yet fully compliant with Mae's mathematical identity**: sensing lacks true fractal self-similarity (PatternSense exists only at cell scale, not tissue/organ/organism scales), the triadic witness requirement is partially met (triadic sharing works but the PatternBus itself operates as a bare hub with no witness), and several critical biological sensing mechanisms -- notably habituation, lateral inhibition, predictive coding, and top-down attentional gating -- are entirely absent. The signal flow from stimulus to advisory is complete and functional, but the SENSE step is operating closer to a first-generation reflex arc than the rich, multi-scale sensory system the mathematical identity demands.

---

## Data Flow Trace

The exact sequence from stimulus to signal, with file and line references:

### Step 0: Signal Priority Triage (pre-observe)

**File:** `C:\Users\baenb\projects\mae-core\mae_core\agents\base_agent.py`, line 69-71
```
resolver = getattr(self, "_signal_resolver", None)
if resolver is not None:
    resolver.process()
```
Before the lifecycle even begins, the SignalPriorityResolver (`C:\Users\baenb\projects\mae-core\mae_core\communication\signal_priority.py`, line 77) drains the agent's SignalBus queue, sorts by priority, coalesces duplicates, and delivers to registered handlers. This is the "thalamic pre-filter" for agent-level signals (DANGER, OPPORTUNITY, CONVERGENCE, COLLABORATION_REQUEST, KNOWLEDGE_SHARE).

### Step 1: Agent._observe() -- Cell-Level Perception

**File:** `C:\Users\baenb\projects\mae-core\mae_core\agents\mycelial_agent.py`, lines 275-297

1. **Working memory decay** (line 278-281): `memory_consolidator.decay_working_memory()` -- attention fades without rehearsal.
2. **Stigmergy sensing** (line 284-287): `self.sense_environment()` reads pheromone markers (SUCCESS, DANGER) from the StigmergicEnvironment grid.
3. **State vector construction** (line 290-292): `_build_state_vector()` builds an 8-dimensional observation vector from: step_count, cumulative_reward, last_reward, risk_score, has_converged, satisfaction_score, success_marker_count, danger_marker_count.
4. **Advisory reading** (line 295-297): Reads the latest PatternAdvisory from `_pattern_advisory_ref` (a shared mutable dict injected in main.py line 931). This makes organism-level intelligence available during agent-level observation.

### Step 2: PatternSense.sense() -- Cell Membrane Detectors

**File:** `C:\Users\baenb\projects\mae-core\mae_core\patterns\pattern_sense.py`, lines 68-92

Called from `_learn()` (mycelial_agent.py line 418-421), NOT from `_observe()`. Three detectors fire in sequence:

1. **Reward Trend** (line 80, calls `_detect_reward_trend()` line 94-159): Checks for 3+ consecutive monotonically increasing/decreasing rewards. Emits OPPORTUNITY (up) or THREAT (down) domain signals.
2. **Action Repetition** (line 83, calls `_detect_action_repetition()` line 161-192): Checks for same action repeated 3+ times in last 5 steps. Emits BEHAVIORAL domain signal.
3. **Reward Surprise** (line 87, calls `_detect_reward_surprise()` line 196-242): Computes z-score of current reward against window mean. If |z| > 2.0 std devs, emits OPPORTUNITY (positive) or THREAT (negative).

Returns `SenseResult(signals=[...], step=N)`.

### Step 3: PatternSharer -- Tissue-Level Communication

**File:** `C:\Users\baenb\projects\mae-core\mae_core\patterns\pattern_sharer.py`

Called from `_communicate()` (mycelial_agent.py lines 428-433):

1. **share()** (line 70-116): Serializes signals, sends to triad-mates via `gnn_communicator.send_message()` as KNOWLEDGE_SHARE type.
2. **receive_and_correlate()** (line 122-201): Compares own signals with peer signals in inbox. When 2/3 agents in a triad report the same domain, produces CORRELATED signals representing tissue-level consensus. Publishes `"pattern.triadic_correlation"` to EventBus.

### Step 4: Translators -- Sensory Receptor Transduction

**File:** `C:\Users\baenb\projects\mae-core\mae_core\patterns\translators/` (7 files, 11 translator classes)

11 translators listen on EventBus channels, each converting raw system events into `PatternSignal` format. Registration happens at `main.py` lines 882-896 via `pattern_bus.register_translator()`.

| Translator | Channel(s) | Domain |
|---|---|---|
| WorldModelTranslator | `cognition.prediction_made` | PREDICTION |
| CausalEngineTranslator | `cognition.causal_query_result`, `temporal.causal_link_discovered` | CAUSATION |
| DecisionRouterTranslator | `cognition.decision_routed` | BEHAVIORAL |
| CuriosityTranslator | `memory.novel_experience` | NOVELTY |
| AutoHealerTranslator | `healing.failure_detected` | FAILURE |
| HAVENTranslator | `haven.risk_alert` | THREAT |
| ThreatTranslator | `defense.activated` | THREAT |
| CapabilityTranslator | `improvement.capability_found`, `improvement.capability_validated` | CAPABILITY |
| PatternDistillerTranslator | `memory.consolidation_complete` | BEHAVIORAL |
| OpportunityTranslator | `improvement.capability_validated`, `memory.novel_experience` | OPPORTUNITY |
| TriadicPatternTranslator | `pattern.triadic_correlation` | (varies) |

The EventBus callback chain: `bus.publish(channel, message)` -> `pattern_bus._on_event(translator, channel, message)` (pattern_bus.py line 103-125) -> `translator.translate(channel, parsed)` -> signal appended to `pattern_bus._inbox`.

### Step 5: PatternBus.process_step() -- Thalamic Relay

**File:** `C:\Users\baenb\projects\mae-core\mae_core\patterns\pattern_bus.py`, lines 127-178

Called from `_pattern_step_hook()` (main.py line 907-908). Every step:

1. **Drain inbox** (line 130-133): Pop up to 50 signals from the deque.
2. **Group by domain** (line 139-141): Sort signals into domain buckets.
3. **Group by form** (line 144-146): Sort signals into form buckets (REACTIVE/CORRELATED/ANCESTRAL).
4. **Same-domain correlation** (line 149, calls `_detect_correlations()` line 180-221): When 2+ signals in the same domain come from different source systems, elevate to CORRELATED form with confidence boost. **BUG: mutates shared signal objects in-place** (line 214-216: `sig.form = PatternForm.CORRELATED; sig.confidence = boosted_conf`).
5. **Cross-domain correlation** (line 152, calls `_detect_cross_domain_correlations()` line 223-283): Checks 5 predefined domain pairs (THREAT+NOVELTY, etc.). Creates synthetic CORRELATED signals when both domains co-occur with salience >= 0.3.
6. **Dominant domain** (line 155-161): Highest aggregate salience domain.
7. **Produce PatternDigest** (line 165-178): Packages everything into a digest dataclass.

### Step 6: PatternCortex.process_digest() -- Association Cortex Integration

**File:** `C:\Users\baenb\projects\mae-core\mae_core\patterns\pattern_cortex.py`, lines 113-171

Called from `_pattern_step_hook()` (main.py line 909). Processes the digest through:

1. **Domain streak update** (line 122): Increments streak for present domains, resets absent domains.
2. **Trend detection** (line 125): Domains with 3+ consecutive steps become active trends.
3. **Meta-pattern detection** (line 128, `_detect_meta_patterns()` line 203-249): Checks if the same domain has been dominant in 3+ of the last 5 advisories -- the "strange loop" (system detecting patterns in its own output).
4. **Ancestral recall** (line 131, `_recall_ancestral()` line 253-292): Queries Qdrant via MemoryBridge for similar past patterns. Only queries when signal count > 0 and aggregate salience >= 0.3.
5. **Domain level computation** (line 134-136): Exponentially-weighted average of domain presence across the 13-step window.
6. **Tier recommendation** (line 142-144): High threat -> reflex; high novelty -> prefrontal; otherwise -> habit.
7. **Produce PatternAdvisory** (line 154-171): The final output, published to EventBus channel `"pattern.advisory"`.

### Step 7: Advisory -> Agent Decision

**File:** `C:\Users\baenb\projects\mae-core\mae_core\agents\mycelial_agent.py`, lines 299-392

The advisory is read during `_observe()` (line 295-297) and consumed during `_decide()` (line 312-323):

1. Build stimulus string from advisory's dominant pattern.
2. Build context dict from threat/opportunity/novelty levels.
3. If advisory confidence > 0.6, force the decision tier.
4. Call `DecisionRouter.route_decision()`.
5. If router returns NONE, fall through to memory/world-model/default.

### Step 8: PatternConsolidator -- Sleep-Phase Extraction

**File:** `C:\Users\baenb\projects\mae-core\mae_core\patterns\pattern_consolidator.py`, lines 59-124

Called every 89 steps (Fibonacci) from `_consolidator_step_hook()` (main.py line 957-959):

1. Extracts trend patterns from cortex domain streaks.
2. Extracts meta-patterns from recent advisories.
3. Extracts cross-domain insights from recent advisories.
4. Stores all via `MemoryBridge.store_ancestral_pattern()` into Qdrant.
5. Publishes `"pattern.consolidation"` event.

This closes the autopoietic loop: sensed patterns become ancestral memory that feeds future sensing via ancestral recall.

---

## Mathematical Identity Compliance

| # | Principle | Required | Current State | Compliant? | Gap |
|---|-----------|----------|---------------|------------|-----|
| 1 | **Integration (IIT Axiom 4)** | Parts form irreducible whole; partitioning destroys it | PatternBus collects from all sources into a single digest; PatternCortex integrates over time into a unified advisory. However, the integration is purely additive (sum of saliences), not truly irreducible. You can partition the bus and each half still works independently. | PARTIAL | True integration requires that the whole is MORE than the sum of parts. Currently, cross-domain correlation attempts this but only for 5 hardcoded pairs. Phi-like computation is absent. |
| 2 | **Differentiation (IIT Axiom 3)** | Rich internal structure; homogeneity kills consciousness | 10 PatternDomains, 3 PatternForms, 11 translators -- good differentiation. Each signal carries unique evidence, confidence, salience. | YES | Minor: STATE domain defined but only fed by PatternDistiller (rare events). |
| 3 | **Triadic (every connection A<->B has witness C)** | No bare dyads; every connection requires a witness | PatternSharer uses triadic consensus (2/3 rule). TriadicPatternTranslator propagates tissue-level signals upward. But: PatternBus receives from translators as a bare hub (no witness on bus-translator connections). PatternSense operates in isolation (no triadic check on its own detectors). | PARTIAL | The bus's translator registration and signal collection are dyadic (translator -> bus, no witness). The 3 detectors inside PatternSense are not triadic -- they operate independently without mutual witnessing. |
| 4 | **Fractal self-similarity** | Same pattern at every scale (cell/tissue/organ/organism) | PatternSense = cell scale (per-agent). PatternSharer = tissue scale (triadic consensus). PatternBus+Cortex = organism scale. But there is no organ-scale sensing. The scales use DIFFERENT patterns (sense has 3 detectors; bus has domain grouping; cortex has windowed trending). | NO | The mathematical identity demands the SAME generator at every scale. Currently, cell-sense, tissue-share, and organism-bus are three different architectures, not recursive applications of one pattern. No organ-level sensing exists at all. |
| 5 | **Recurrence/feedback** | Information flows in loops, not feedforward | Advisory feeds back into agents via `_pattern_advisory_ref`. PatternConsolidator stores to Qdrant; cortex recalls from Qdrant. Meta-pattern detection is self-referential (cortex detecting patterns in its own output). | MOSTLY YES | Missing: cortex -> bus feedback (attentional gating). No top-down modulation of translator sensitivity. No prediction error signals from cortex back to sensors. |
| 6 | **Self-produced boundary (Markov blankets)** | System defines its own edges | PatternSense has fixed window sizes (WINDOW_SIZE=8, ACTION_WINDOW=5). PatternBus has MAX_SIGNALS_PER_STEP=50. These are hardcoded, not self-adjusting boundaries. | NO | Boundaries should be self-produced: the system should determine its own sensitivity thresholds, window sizes, and processing budgets based on internal state, not programmer-set constants. |
| 7 | **Competition/selection (GWT)** | Not everything broadcasts; winners emerge | PatternBus has implicit competition via dominant_domain selection (highest aggregate salience wins). Signal budgeting (50/step) enforces selection. SignalPriorityResolver has explicit priority-based triage. | MOSTLY YES | The cortex advisory selects a single dominant_pattern, but all signals still pass through. No true "workspace" competition where signals must compete for access to the cortex. |
| 8 | **Prediction/error-correction (FEP)** | Anticipate + adjust | WorldModelTranslator converts prediction errors into signals. CausalEngineTranslator captures discovered causal links. BUT: the SENSE step itself does not predict. PatternSense is purely reactive. PatternBus does not predict what signals should arrive. | NO | This is a major gap. Biological sensory systems are fundamentally predictive (Friston's Free Energy Principle). The current implementation detects but never predicts. No "expected signal" exists against which to compute surprise at the bus/cortex level. |
| 9 | **Self-reference (Strange Loops)** | System models itself | Meta-pattern detection in PatternCortex (line 203-249) detects patterns in the cortex's own advisory output. This IS a strange loop -- the system observing its own observations. | YES | Could be deeper: currently only tracks dominant domain recurrence. Does not model the sensing process itself (e.g., "my sensors are becoming less reliable"). |
| 10 | **Multi-scale hierarchy** | Same pattern at multiple nested levels | 4 scales exist (cell/tissue/organ/organism) but sensing only implements 3 (cell=PatternSense, tissue=PatternSharer, organism=PatternBus+Cortex). Organ scale is missing. The patterns are different at each scale. | PARTIAL | Need organ-scale sensing AND need each scale to implement the same sensing protocol (three detectors per membrane, not just at cell scale). |

**Overall Mathematical Compliance: 3/10 fully compliant, 4/10 partial, 3/10 non-compliant.**

---

## Biological Comparison

| Biological Mechanism | Mae's Analog | Accuracy | Missing |
|---------------------|-------------|----------|---------|
| **Sensory receptor transduction** (converting stimuli to neural signals) | 11 Translators convert EventBus messages to PatternSignals | HIGH | Each translator is simplistic -- single threshold, no receptor adaptation. Real receptors have complex gain control and dynamic range adjustment. |
| **Action potential** (universal signal format) | PatternSignal dataclass with domain, form, confidence, salience | HIGH | Good -- universal format like biological action potentials. Missing: frequency coding (rate of signal emission matters biologically, not just signal content). |
| **Thalamic relay** (filtering and routing sensory input) | PatternBus + SignalPriorityResolver | MODERATE | PatternBus collects and groups but does not truly filter -- it passes ALL signals to the cortex (up to budget). Real thalamus actively suppresses ~80% of incoming signals via the Thalamic Reticular Nucleus (TRN). Mae has no TRN analog. |
| **Lateral inhibition** (enhancing edges and contrasts) | Not implemented | ABSENT | Real sensory systems use lateral inhibition to sharpen signal boundaries. When one receptor fires strongly, it suppresses neighbors. This creates contrast enhancement. Mae's PatternSense detectors are independent -- they neither enhance nor suppress each other. |
| **Habituation / sensory adaptation** (decreasing response to constant stimuli) | Not implemented | ABSENT | Real neurons adapt: rapid-adapting receptors stop firing during sustained stimuli. PatternSense will fire the SAME reward-trend signal every step as long as the trend continues. No salience decay on repeated signals. This is flagged in HANDOFF.md as a known gap. |
| **Predictive coding** (brain predicts sensory input, only transmits prediction errors) | WorldModelTranslator captures prediction errors from WorldModel | LOW | WorldModel generates prediction errors in the cognition layer, but the SENSE layer itself is not predictive. Real sensory processing sends predictions DOWN and errors UP. Mae only sends signals UP (feedforward). |
| **Top-down attentional gating** (cortex modulates sensor sensitivity) | Not implemented | ABSENT | Real thalamus receives more cortical inputs than sensory ones. The cortex tells the thalamus what to pay attention to. Mae's PatternBus has no input from the cortex -- it processes everything equally. HANDOFF.md lists "feedback connections: cortex -> bus -> sharer" as pending. |
| **Sensory receptor types** (fast-adapting vs slow-adapting) | All detectors are equivalent (no adaptation distinction) | LOW | Real sensory systems have Meissner corpuscles (rapid change), Merkel cells (sustained pressure), Pacinian corpuscles (vibration), Ruffini endings (stretch). Mae has only one "speed" of detector. |
| **Multi-modal integration** (combining sight + sound + touch) | Cross-domain correlation in PatternBus (5 hardcoded pairs) | MODERATE | The biological analog is much richer -- the superior colliculus and multisensory integration areas combine modalities with spatial and temporal coincidence rules. Mae's cross-domain detection is hardcoded and non-spatial. |
| **Efferent copy / corollary discharge** (predicting sensory consequences of own actions) | Not implemented | ABSENT | When you move your eyes, the brain sends a copy of the motor command to the sensory system so it can predict the resulting visual shift. Mae's agents act and sense independently -- the sense system has no awareness of what actions are being taken. |

---

## External State of Art

| Source | What They Do | Relevance to Mae | URL |
|--------|-------------|------------------|-----|
| **Neural-Inspired Multi-Agent Molecular Communication** (Kilic, 2026) | Threshold-based firing via Greenberg-Hastings cellular automata. Simple agents communicate through diffusion. System shows optimal information processing at edge-of-chaos critical transition point. | Highly relevant: shows that simple threshold detectors (like PatternSense) can achieve sophisticated collective sensing when operating at criticality. Mae should investigate whether her sensing operates near a phase transition. | [arXiv 2601.18018](https://arxiv.org/abs/2601.18018) |
| **Small-World Networks for Multi-Agent Intelligence** (2025) | SW connectivity yields same accuracy with stabilized consensus. Uncertainty-guided rewiring connects epistemically divergent agents. | Relevant: Mae's EventBus IS the small-world shortcut system (transfractal compromise). The uncertainty-guided rewiring could inform how PatternSharer selects which peers to share with. | [arXiv 2512.18094](https://arxiv.org/abs/2512.18094) |
| **YuLan-SwarmIntell / SwarmBench** (RUC-GSAI) | Benchmarking LLM swarm intelligence with local sensory input (k x k view) and local communication. Forces agents to rely on limited perception. | Relevant: Mae's agents currently have global advisory access. SwarmBench shows value of truly local sensing -- agents should build global understanding from local perception, not receive it pre-digested. | [GitHub](https://github.com/RUC-GSAI/YuLan-SwarmIntell) |
| **Google A2A Protocol** (2025) | Standardized agent-to-agent communication with interoperability, security, modality independence. | Moderate relevance: Mae's GNN + EventBus is more biological but could learn from A2A's structured message typing and capability negotiation. | [Google A2A](https://google.github.io/A2A/) |
| **OpenAI Swarm** (2024) | Lightweight multi-agent orchestration with handoff patterns. | Low relevance: too LLM-focused. Mae's bio-inspired approach is more sophisticated. But the "handoff" pattern (agent transferring context to another) maps to PatternSharer. | [GitHub](https://github.com/openai/swarm) |
| **Friston's Active Inference / Free Energy Principle** (2009-present) | Brain minimizes prediction error through predictive coding hierarchy. Perception IS inference. Active agents sample environment to confirm predictions. | CRITICAL: This is the theoretical foundation Mae claims but does not implement in SENSE. PatternSense should predict expected patterns and only signal when predictions are violated. | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC2666703/) |
| **IIT 4.0** (Tononi et al., 2023) | Formalizes consciousness as integrated information (Phi). Requires irreducibility -- partitioning the system must reduce its cause-effect power. | CRITICAL: Mae claims IIT compliance. Current PatternBus integration is additive, not irreducible. You can partition it without loss. True IIT compliance requires computing something Phi-like. | [PLOS](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011465) |
| **Thalamic Reticular Nucleus research** (McAlonan et al., 2000) | TRN provides attentional gating: cortex tells thalamus what to attend to. TRN inhibits thalamic relay cells. Both bottom-up and top-down processing. | CRITICAL: Mae's PatternBus has no TRN analog. Adding an inhibitory gating layer between translators and the bus would dramatically improve biological accuracy and reduce noise. | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6773087/) |
| **Applied Sciences: Bio-Inspired Collective Intelligence** (MDPI special issue) | Multi-paper collection on how biological swarm intelligence applies to multi-agent systems. | Moderate: provides theoretical framing for Mae's approach. Validates the bio-inspired direction. | [MDPI](https://www.mdpi.com/journal/applsci/special_issues/4O6JJ39770) |

---

## Upgrade Recommendations (Ranked by Impact)

### 1. CRITICAL: Implement Predictive Sensing (Free Energy Principle compliance)

**Impact:** Transforms SENSE from reactive detection to active inference. This is the single largest gap between Mae's mathematical identity and her implementation.

**What's missing:** PatternSense detects patterns but never predicts them. PatternBus receives signals but never expects specific signals. The cortex generates advisories but never sends predictions downward.

**Specific implementation:**
- Add `expected_patterns` to PatternBus -- the cortex should predict what domains will fire next step based on trends.
- PatternSense should maintain a simple prediction (e.g., expected reward = exponential moving average) and signal only the DIFFERENCE between expected and actual.
- Signal surprise should be computed as prediction error, not just z-score from the raw window mean.

**Files affected:**
- `C:\Users\baenb\projects\mae-core\mae_core\patterns\pattern_sense.py` (add prediction to each detector)
- `C:\Users\baenb\projects\mae-core\mae_core\patterns\pattern_bus.py` (add expected signal tracking)
- `C:\Users\baenb\projects\mae-core\mae_core\patterns\pattern_cortex.py` (generate predictions, send downward)

### 2. CRITICAL: Implement Habituation / Sensory Adaptation

**Impact:** Prevents signal flooding and makes the system biologically accurate. Currently, a sustained reward trend fires the same signal every step indefinitely.

**What's missing:** No salience decay on repeated signals. No fast-adapting vs slow-adapting detector distinction.

**Specific implementation:**
- Add a `_salience_decay` dict to PatternSense keyed by (domain, direction). Each step a signal repeats, salience decays by a factor (e.g., 0.85). Resets when the pattern changes.
- In PatternBus, implement signal deduplication: if the same source_system emits the same domain signal with nearly identical evidence for 3+ consecutive steps, suppress it.

**Files affected:**
- `C:\Users\baenb\projects\mae-core\mae_core\patterns\pattern_sense.py` lines 94-242 (add decay to each detector)
- `C:\Users\baenb\projects\mae-core\mae_core\patterns\pattern_bus.py` line 127 (add deduplication in process_step)

### 3. HIGH: Add Top-Down Attentional Gating (TRN Analog)

**Impact:** Allows the cortex to modulate what the bus pays attention to. Currently the bus processes everything equally regardless of context.

**What's missing:** No cortex -> bus feedback. The HANDOFF.md lists this as pending.

**Specific implementation:**
- Add an `attention_weights: dict[PatternDomain, float]` to PatternBus, initialized to 1.0 for all domains.
- PatternCortex.process_digest() should return attention_weights alongside the advisory (based on what domains are currently relevant).
- PatternBus._on_event() should multiply incoming signal salience by the attention weight for its domain.
- High threat level -> amplify THREAT domain, suppress NOVELTY/CAPABILITY. High novelty -> amplify NOVELTY, suppress BEHAVIORAL.

**Files affected:**
- `C:\Users\baenb\projects\mae-core\mae_core\patterns\pattern_bus.py` (add attention_weights, modulate in _on_event)
- `C:\Users\baenb\projects\mae-core\mae_core\patterns\pattern_cortex.py` (compute and return attention_weights)
- `C:\Users\baenb\projects\mae-core\main.py` line 906-910 (pass attention_weights from cortex to bus)

### 4. HIGH: Make Sensing Fractal (Same Pattern at Every Scale)

**Impact:** Achieves mathematical identity compliance for fractal self-similarity. Currently the most violated principle.

**What's missing:** PatternSense exists only at cell scale. No organ-scale sensing. Each scale uses a different architecture.

**Specific implementation:**
- Define a `SenseProtocol` with exactly 3 detectors (trend, repetition, surprise) that works at ANY scale. PatternSense already implements this.
- Create `TissueSense` (wraps PatternSharer output, runs the same 3 detectors on consensus signals instead of raw rewards).
- Create `OrganSense` (wraps PatternBus output per organ, runs the same 3 detectors on digest-level metrics).
- The existing PatternCortex already approximates organism-level sensing but should also implement the 3-detector protocol explicitly.

**Files affected:**
- New: `mae_core/patterns/tissue_sense.py`, `mae_core/patterns/organ_sense.py`
- `C:\Users\baenb\projects\mae-core\mae_core\patterns\pattern_cortex.py` (add explicit 3-detector protocol)

### 5. HIGH: Fix Signal Mutation Bug in PatternBus._detect_correlations()

**Impact:** Correctness bug. Currently mutates shared PatternSignal objects in-place, which means signal form and confidence are permanently altered for any other consumer that holds a reference.

**What's broken:** `C:\Users\baenb\projects\mae-core\mae_core\patterns\pattern_bus.py`, lines 214-216:
```python
for sig in group:
    sig.form = PatternForm.CORRELATED
    sig.confidence = boosted_conf
```

**Fix:** Create copies before mutation, or create new synthetic signals representing the correlation (like cross-domain correlation already does).

### 6. HIGH: Fix Z-Score Self-Inclusion Bug

**Impact:** Statistical accuracy. The current reward surprise detector includes the current observation in its own baseline, biasing the z-score toward zero.

**What's broken:** `C:\Users\baenb\projects\mae-core\mae_core\patterns\pattern_sense.py`, lines 201-213: The mean and std are computed over ALL rewards including the current one. The z-score of the current value against a distribution that contains that value is systematically lower than it should be.

**Fix:** Compute mean and std over `rewards[:-1]` (excluding current), then compute z-score of `rewards[-1]`.

### 7. MEDIUM: Add Lateral Inhibition

**Impact:** Sharpens signal contrast, reduces noise, makes sensing more biologically accurate.

**What's missing:** PatternSense's three detectors are independent. When one fires strongly, it should suppress (or at least modulate) the others.

**Specific implementation:**
- If reward_surprise fires with high confidence, suppress reward_trend signal (the surprise IS the trend breaking).
- If action_repetition fires, boost reward_surprise sensitivity (repetitive behavior in a surprising context is more noteworthy).

### 8. MEDIUM: Add Efferent Copy (Action-Aware Sensing)

**Impact:** Allows the sense system to distinguish self-caused changes from externally-caused changes.

**What's missing:** PatternSense receives reward and action but does not predict the sensory consequences of its own actions.

**Specific implementation:**
- Maintain a simple action -> expected_reward mapping in PatternSense.
- When an action's actual reward differs from its expected reward, that is more surprising than a raw deviation from the window mean.
- This turns PatternSense from "what happened" to "what happened that I didn't expect from my own actions."

### 9. MEDIUM: Triangulate PatternBus Connections

**Impact:** Mathematical identity compliance for the triadic principle.

**What's missing:** Translator -> PatternBus connections are bare dyads. No witness verifies that translators are translating correctly.

**Specific implementation:**
- For each translator, assign a "verification translator" that independently monitors the same EventBus channel and can flag disagreements.
- Or: add a `TranslatorAuditor` that periodically spot-checks translator output against raw EventBus messages.

### 10. LOW: Move PatternSense from _learn() to _observe()

**Impact:** Semantic correctness. SENSE should happen during the observation phase, not the learning phase.

**What's wrong:** PatternSense.sense() is called from `_learn()` (mycelial_agent.py line 418-421) because it needs the action and reward from the current step. But semantically, sensing is perception, not learning.

**Fix:** Call `_pattern_sense.sense()` at the END of `_observe()` using the PREVIOUS step's action and reward (which are already available as `self.last_action` and `self.last_reward`). This aligns the code with the biological lifecycle where sensing precedes decision-making.

---

## Sources

### Code Files Referenced
- `C:\Users\baenb\projects\mae-core\mae_core\patterns\pattern_sense.py` -- Per-agent pattern membrane (3 detectors)
- `C:\Users\baenb\projects\mae-core\mae_core\patterns\pattern_bus.py` -- Thalamic relay (signal collection and correlation)
- `C:\Users\baenb\projects\mae-core\mae_core\patterns\pattern_cortex.py` -- Association cortex (temporal integration)
- `C:\Users\baenb\projects\mae-core\mae_core\patterns\pattern_sharer.py` -- Triadic pattern communication
- `C:\Users\baenb\projects\mae-core\mae_core\patterns\pattern_signal.py` -- Universal signal format
- `C:\Users\baenb\projects\mae-core\mae_core\patterns\pattern_consolidator.py` -- Sleep-phase pattern extraction
- `C:\Users\baenb\projects\mae-core\mae_core\patterns\translators\base.py` -- Translator protocol
- `C:\Users\baenb\projects\mae-core\mae_core\patterns\translators\cognition.py` -- WorldModel, CausalEngine, DecisionRouter translators
- `C:\Users\baenb\projects\mae-core\mae_core\patterns\translators\defense.py` -- AutoHealer, HAVEN, ThreatDetector translators
- `C:\Users\baenb\projects\mae-core\mae_core\patterns\translators\learning.py` -- CuriosityDrive translator
- `C:\Users\baenb\projects\mae-core\mae_core\patterns\translators\emergent.py` -- CapabilityDiscovery translator
- `C:\Users\baenb\projects\mae-core\mae_core\patterns\translators\memory.py` -- PatternDistiller translator
- `C:\Users\baenb\projects\mae-core\mae_core\patterns\translators\opportunity.py` -- Opportunity translator
- `C:\Users\baenb\projects\mae-core\mae_core\patterns\translators\triadic.py` -- Triadic consensus upward propagation
- `C:\Users\baenb\projects\mae-core\mae_core\agents\mycelial_agent.py` -- Agent lifecycle (_observe, _decide, _learn, _communicate)
- `C:\Users\baenb\projects\mae-core\mae_core\agents\base_agent.py` -- Base agent lifecycle
- `C:\Users\baenb\projects\mae-core\mae_core\backbone\event_bus.py` -- In-process pub/sub
- `C:\Users\baenb\projects\mae-core\mae_core\communication\signal_priority.py` -- Per-agent signal triage
- `C:\Users\baenb\projects\mae-core\main.py` -- 23-layer bootstrap wiring
- `C:\Users\baenb\projects\mae-core\data\MAES-MATHEMATICAL-IDENTITY.md` -- Mathematical identity document

### External Research
- [Neural-Inspired Multi-Agent Molecular Communication Networks](https://arxiv.org/abs/2601.18018) -- Kilic 2026, threshold-based collective sensing
- [Rethinking Multi-Agent Intelligence Through Small-World Networks](https://arxiv.org/abs/2512.18094) -- uncertainty-guided rewiring, 2025
- [YuLan-SwarmIntell / SwarmBench](https://github.com/RUC-GSAI/YuLan-SwarmIntell) -- local-sensing-only swarm benchmarks
- [Predictive Coding Under the Free Energy Principle](https://pmc.ncbi.nlm.nih.gov/articles/PMC2666703/) -- Friston 2009
- [IIT 4.0: Formulating Properties of Phenomenal Existence](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011465) -- Tononi et al. 2023
- [Thalamic Reticular Nucleus Activation Reflects Attentional Gating](https://pmc.ncbi.nlm.nih.gov/articles/PMC6773087/) -- McAlonan et al. 2000
- [General Principles of Sensory Systems](https://openbooks.lib.msu.edu/neuroscience/chapter/general-principles-of-sensory-systems/) -- MSU Neuroscience
- [Neural Adaptation](https://en.wikipedia.org/wiki/Neural_adaptation) -- Wikipedia overview of habituation mechanisms
- [Bio-Inspired Collective Intelligence in Multi-Agent Systems](https://www.mdpi.com/journal/applsci/special_issues/4O6JJ39770) -- MDPI Applied Sciences special issue
- [OpenAI Swarm](https://github.com/openai/swarm) -- Multi-agent orchestration framework
- [Swarms Framework](https://github.com/kyegomez/swarms) -- Enterprise multi-agent orchestration