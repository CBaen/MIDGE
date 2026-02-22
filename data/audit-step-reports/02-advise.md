> Generated from 10-agent audit conducted 2026-02-11. ~50 sub-agents. Sources: biology papers, GitHub, research papers, full codebase trace.

# ADVISE AUDIT REPORT

## Executive Summary

Mae's ADVISE step (PatternBus + PatternCortex + PatternConsolidator) is a well-architected sensory integration pipeline that converts raw EventBus signals into actionable PatternAdvisory objects consumed by agents. The code is structurally sound with clear data flow, graceful degradation at every boundary, and strong autopoietic closure via the consolidation loop. However, the system **lacks true competitive ignition** (Global Workspace Theory), uses **passive broadcast rather than gated access**, performs **no active prediction or error correction**, and has **no inhibitory/suppressive mechanism** analogous to the thalamic reticular nucleus. These four gaps represent the most significant deviations from both Mae's mathematical identity and biological reality. The most impactful upgrade would be introducing competitive ignition with a salience threshold before advisory production.

---

## Data Flow Trace

### Complete Signal Path: EventBus Event --> Agent Decision

**Phase 1: Translation (Sensory Receptors)**

| Step | What Happens | File:Line |
|------|-------------|-----------|
| 1a | EventBus publishes event on a channel (e.g., `haven.risk_alert`) | `mae_core/backbone/event_bus.py:64-94` |
| 1b | PatternBus's `_on_event()` callback fires for registered translator channels | `mae_core/patterns/pattern_bus.py:103-125` |
| 1c | Translator converts raw JSON into a `PatternSignal` (or returns None) | e.g., `mae_core/patterns/translators/defense.py:71-92` |
| 1d | Signal appended to `PatternBus._inbox` (deque, maxlen=200) | `mae_core/patterns/pattern_bus.py:118` |

11 translators are registered, covering: WorldModel, CausalEngine, DecisionRouter, Curiosity, AutoHealer, HAVEN, ThreatDetector, Capability, PatternDistiller, Opportunity, TriadicPattern. Each subscribes to specific EventBus channels.

**Phase 2: Digestion (Thalamic Relay)**

| Step | What Happens | File:Line |
|------|-------------|-----------|
| 2a | `_pattern_step_hook()` fires once per model step | `main.py:906-925` |
| 2b | `PatternBus.process_step()` drains inbox (up to MAX_SIGNALS_PER_STEP=50) | `mae_core/patterns/pattern_bus.py:127-178` |
| 2c | Signals grouped `by_domain` and `by_form` | `pattern_bus.py:139-146` |
| 2d | Same-domain, different-source correlations detected; confidence boosted by `min(1.0, max_conf + 0.1 * log(n))` | `pattern_bus.py:180-221` |
| 2e | Cross-domain correlations detected from 5 predefined high-value pairs; synthetic CORRELATED signal created | `pattern_bus.py:223-283` |
| 2f | Dominant domain identified (highest aggregate salience) | `pattern_bus.py:155-161` |
| 2g | `PatternDigest` dataclass assembled and returned | `pattern_bus.py:165-178` |

**Phase 3: Integration (Association Cortex)**

| Step | What Happens | File:Line |
|------|-------------|-----------|
| 3a | `PatternCortex.process_digest()` receives digest | `mae_core/patterns/pattern_cortex.py:113-171` |
| 3b | Digest appended to 13-step sliding window | `pattern_cortex.py:119` |
| 3c | Domain streak counters updated (present domains increment, absent reset to 0) | `pattern_cortex.py:175-188` |
| 3d | Trends detected (domains with streak >= 3, Rule of Three) | `pattern_cortex.py:189-199` |
| 3e | Meta-patterns detected (same domain dominant in 3+ of last 5 advisories) | `pattern_cortex.py:203-249` |
| 3f | Ancestral recall via MemoryBridge.recall_ancestral_patterns() (if available and salience >= 0.3) | `pattern_cortex.py:253-292` |
| 3g | Aggregate threat/opportunity/novelty levels computed with exponential decay across window | `pattern_cortex.py:296-319` |
| 3h | Correlated insights generated (plain-text) | `pattern_cortex.py:321-349` |
| 3i | Decision tier recommended: "reflex" if threat > 0.6, "prefrontal" if novelty > 0.5 or salience > 2.0, else "habit" | `pattern_cortex.py:351-367` |
| 3j | Dominant pattern selected (highest salience in current digest) | `pattern_cortex.py:147-149` |
| 3k | Confidence computed from base 0.3 + signal count + correlations + trends + ancestral matches, capped at 1.0 | `pattern_cortex.py:369-397` |
| 3l | `PatternAdvisory` dataclass assembled and returned | `pattern_cortex.py:154-171` |

**Phase 4: Delivery (Advisory --> Agent)**

| Step | What Happens | File:Line |
|------|-------------|-----------|
| 4a | Advisory stored in `_latest_advisory["advisory"]` dict (mutable mailbox) | `main.py:910` |
| 4b | Advisory summary published to EventBus on `pattern.advisory` channel | `main.py:913-925` |
| 4c | On agent step, `_observe()` reads `_pattern_advisory_ref.get("advisory")` into `self._current_advisory` | `mae_core/agents/mycelial_agent.py:295-297` |
| 4d | In `_decide()`, advisory is passed to `_route_with_advisory()` if router + advisory both exist | `mycelial_agent.py:312-323` |
| 4e | `_route_with_advisory()` extracts stimulus from dominant_pattern, builds context dict (threat/opportunity/novelty levels, trends, confidence), optionally forces tier if confidence > 0.6 | `mycelial_agent.py:341-392` |
| 4f | `DecisionRouter.route_decision()` cascades: Reflex -> Habit -> Prefrontal | `mae_core/cognition/decision_router.py:128-240` |
| 4g | If router returns `DecisionTier.NONE`, falls through to memory/world-model/default cascade | `mycelial_agent.py:325-339` |

**Phase 5: Consolidation (Autopoietic Closure)**

| Step | What Happens | File:Line |
|------|-------------|-----------|
| 5a | Every 89 steps (Fibonacci), `PatternConsolidator.consolidate()` fires | `main.py:956-959` |
| 5b | Extracts trend patterns, meta-patterns, and cross-domain insights from cortex state | `mae_core/patterns/pattern_consolidator.py:128-218` |
| 5c | Stores extracted patterns as ancestral memory via `MemoryBridge.store_ancestral_pattern()` | `pattern_consolidator.py:222-240` |
| 5d | These ancestral patterns are retrieved by `PatternCortex._recall_ancestral()` in future steps, closing the loop | `pattern_cortex.py:253-292` |

**Parallel Path: Per-Agent Pattern Sense + Triadic Sharing**

| Step | What Happens | File:Line |
|------|-------------|-----------|
| P1 | `PatternSense.sense()` runs during `_learn()`, detecting reward trends, action repetition, reward surprises | `mae_core/patterns/pattern_sense.py:68-92`, called from `mycelial_agent.py:419-421` |
| P2 | `PatternSharer.share()` sends signals to triad-mates via GNN during `_communicate()` | `mae_core/patterns/pattern_sharer.py:70-116`, called from `mycelial_agent.py:429-433` |
| P3 | `PatternSharer.receive_and_correlate()` detects 2/3 agreement (triadic consensus) | `pattern_sharer.py:122-201` |

---

## Mathematical Identity Compliance

| Principle | Required (from MAES-MATHEMATICAL-IDENTITY.md) | Current State | Compliant? | Gap |
|-----------|-----------------------------------------------|---------------|------------|-----|
| **Integration** (IIT) | Parts form irreducible whole; partitioning destroys it | PatternCortex integrates 13 steps with exponential decay; combines signals, trends, meta-patterns, ancestral recall into single PatternAdvisory. Digest groups signals by domain AND form. | Partially | Integration is purely additive (sum/average). No measure of whether the whole exceeds the sum. No Phi-like computation. Removing one signal source does not degrade non-linearly -- it just removes its contribution. |
| **Differentiation** (IIT) | Rich internal structure; homogeneity kills consciousness | 10 PatternDomains, 3 PatternForms, 11 translators from different subsystems. Advisories carry distinct threat/opportunity/novelty axes, trends, meta-patterns, ancestral matches. | Yes | Good differentiation of signal types. |
| **Competition/selection** (GWT) | Not everything broadcasts; winners emerge | **NO.** Every signal that enters the inbox gets processed. Every digest becomes an advisory. There is no ignition threshold, no competitive elimination. Signals are grouped and summed, not competed. The only "selection" is MAX_SIGNALS_PER_STEP=50 budget (FIFO, not priority-based). | **No** | **Critical gap.** GWT requires signals to compete for workspace access via nonlinear ignition. Currently, all signals reach the cortex unconditionally. |
| **Prediction/error-correction** (FEP/Active Inference) | Anticipate + adjust | **NO.** The cortex does not predict what signals SHOULD arrive. It does not compute prediction error. It observes what happened and summarizes. The `recommended_tier` is reactive classification, not predictive inference. | **No** | **Critical gap.** No generative model, no expected surprise, no precision-weighting. The system is purely observational. |
| **Self-reference** (Strange Loops) | System models itself | Meta-pattern detection (PatternCortex detecting patterns in its own advisory output) IS a strange loop. The cortex watching its own dominant-domain recurrence is genuine self-reference. | Yes | Exists but shallow -- only tracks domain recurrence. Does not model its own accuracy, reliability, or bias. |
| **Recurrence/feedback** | Information flows in loops | Consolidation loop: Cortex -> Consolidator -> MemoryBridge -> Cortex (ancestral recall). Advisory -> Agent -> EventBus events -> Translators -> PatternBus -> Cortex. Both loops exist. | Yes | The loops exist but operate at very different timescales (89 steps vs. 1 step). No fast recurrence within a single step. |
| **Multi-scale hierarchy** (Fractal) | Same pattern at nested levels | Three scales exist: PatternSense (agent), PatternSharer (triad), PatternBus+Cortex (organism). Each detects patterns and produces signals. | Yes | Good fractal layering. The triadic consensus mechanism (2/3 agreement) elegantly mirrors the organism-level correlation detection. |
| **Self-produced boundary** (Markov blankets) | System defines its own edges | PatternBus has MAX_SIGNALS_PER_STEP=50 and inbox maxlen=200. PatternCortex has 13-step window. These ARE boundaries, but they are static/hardcoded, not self-produced. | Partially | Boundaries do not adapt. A living system modulates its own sensitivity. |
| **Triadic structure** | Every connection A-B requires witness C | Cross-domain correlation uses pairs (5 predefined), not triads. Same-domain correlation requires 2+ sources (not specifically 3). PatternSharer uses 2/3 triadic consensus. | Partially | The organism-level ADVISE is not explicitly triadic. The PatternBus-PatternCortex-PatternConsolidator trio is a de facto triad, but this is not formalized. |

---

## Biological Comparison

| Biological Mechanism | Mae's Analog | Accuracy | Missing |
|---------------------|-------------|----------|---------|
| **Thalamic relay nuclei** (receive raw sensory signals, route to appropriate cortical area) | PatternBus receives EventBus events via translators, groups by domain, routes to PatternCortex | Good. The translation from heterogeneous events to a common PatternSignal format mirrors the thalamus converting diverse sensory modalities into neural firing patterns. | Real thalamus has **two modes**: relay mode (faithful transmission) and burst mode (salience amplification). PatternBus has only relay mode. No burst/amplification. |
| **Thalamic Reticular Nucleus (TRN)** (GABAergic inhibition, attentional gating, suppresses irrelevant signals) | **No analog.** MAX_SIGNALS_PER_STEP is a static budget, not active inhibition. | **Missing** | This is the biggest biological gap. The TRN selectively suppresses thalamic relay based on top-down cortical feedback and emotional salience. Mae has no suppressive gating -- all signals pass through. |
| **Association cortex** (integrates multiple modalities over time, temporal binding) | PatternCortex with 13-step window, exponential decay, trend detection | Good. The sliding window with exponential decay approximates temporal integration. Domain-level computation across the window mirrors temporal binding. | Real association cortex uses oscillatory coherence (gamma binding, alpha inhibition) for temporal integration. No oscillatory dynamics in Mae. |
| **Reticular Activating System (RAS)** (arousal gating, modulates thalamic relay gain based on arousal state) | EndocrineSystem exists separately but does NOT modulate PatternBus gain. | **Missing** | The endocrine system modulates agents' DecisionRouters but not the pattern ecosystem. PatternBus processes signals identically regardless of Mae's stress/arousal state. |
| **Global Workspace ignition** (nonlinear threshold crossing, only winning coalition broadcasts) | **No analog.** Every digest produces an advisory. No threshold. No competition between signal coalitions. | **Missing** | In GNW, specialized processors compete for workspace access. Only signals that achieve ignition (crossing a nonlinear threshold via recurrent amplification) enter the global workspace. Mae broadcasts ALL advisories. |
| **Corticothalamic feedback** (layer 6 projections back to thalamus, modulating what gets relayed) | Advisory -> Agent -> EventBus -> Translators is a feedback loop, but does not modulate PatternBus sensitivity. | Weak | Real corticothalamic feedback changes thalamic gain in real-time. Mae's feedback is structural (same pathway) but not functional (no modulation). |
| **Pulvinar nucleus** (higher-order thalamus, salience filtering, coordinates cortico-cortical communication) | Cross-domain correlation detection in PatternBus | Partial | The pulvinar actively coordinates which cortical areas communicate. Mae's cross-domain detection is passive observation of co-occurrence, not active coordination. |
| **Hippocampal-cortical dialogue** (consolidation during rest, memory replay) | PatternConsolidator every 89 steps, ancestral recall via MemoryBridge | Good | Good biological analog. The 89-step interval approximates sleep consolidation. Pattern extraction from recent trends mirrors cortical rule extraction. |
| **Predictive coding** (cortex generates predictions, thalamus computes prediction error) | **No analog.** Cortex does not predict. No prediction error signal. | **Missing** | This is fundamental to how the brain actually works. Every cortical area maintains a generative model and computes surprise. Mae's cortex is purely reactive. |
| **Precision weighting** (unreliable signals get attenuated, reliable ones amplified) | Confidence field exists on PatternSignal but is not used for weighting during integration. | **Missing** | In predictive processing, precision (inverse variance) weights prediction errors. High-confidence signals should dominate. In Mae, confidence is carried but all signals contribute equally to aggregate levels. |

---

## External State of Art

| Source | What They Do | Relevance to Mae | URL |
|--------|-------------|------------------|-----|
| Chateau-Laurent et al. (2025) "GNWT-based architectures for reasoning" | Neural GWT implementation: modules as neural nets, central workspace as low-dimensional latent, communication via attention/cross-attention. Outperforms LSTM and Transformer baselines in causal/sequential reasoning. | Direct template for Mae's missing competitive ignition. Central workspace with attention-gated access is exactly what ADVISE needs. | [arxiv.org/2505.13969](https://arxiv.org/html/2505.13969v1) |
| Ye et al. (2025) "GNW for digital twins" | Global workspace controller with ignition, conflict resolution, broadcast, re-entry updates. Emergent conscious-like properties in multi-agent LLM systems. | Shows GWT is implementable in multi-agent architectures. The "conflict resolution + broadcast" pattern is what Mae should adopt. | [emergentmind.com/gnwt](https://www.emergentmind.com/topics/global-neuronal-workspace-theory-gnwt) |
| PMC (2025) "Detailed theory of thalamic and cortical microcircuits for predictive visual inference" | Thalamic relay cells gated by TRN inhibition based on layer 6 feedback. Feed-forward pathway from V1 layer 5 to higher-order thalamus. | Precise biological blueprint for how PatternBus should implement gated relay. TRN = inhibitory gating based on cortex feedback. | [pmc.ncbi.nlm.nih.gov/PMC11800772](https://pmc.ncbi.nlm.nih.gov/articles/PMC11800772/) |
| Halassa & Kastner (2020) "Thalamic bridge from sensory perception to cognition" | Higher-order thalamic nuclei (pulvinar) actively gate cortico-cortical communication. Not just a relay -- a dynamic coordinator. | Validates that PatternBus should be more than a passive relay. Should actively coordinate which signals reach the cortex. | [sciencedirect.com/S0149763420306473](https://www.sciencedirect.com/science/article/pii/S0149763420306473) |
| Dehaene, Changeux, Naccache (2011) "Global Neuronal Workspace Model" | Foundational paper: ignition = nonlinear threshold crossing via recurrent amplification. Two-state model: subliminal (below threshold) vs. conscious (above threshold, globally broadcast). | Core theoretical reference for what Mae's ADVISE should implement. Signals below threshold should NOT produce advisories. | [Dehaene_Changeaux_Naccache_2011.pdf](https://www.antoniocasella.eu/dnlaw/Dehaene_Changeaux_Naccache_2011.pdf) |
| Google A2A Protocol (2025) | Standard inter-agent communication with "Agent Card" discovery, task lifecycle, and multimodal support. | Shows that the multi-agent industry is moving toward standardized advisory protocols. Mae's EventBus channel conventions approach this. | [talan.com/ai-agents](https://www.talan.com/global/en/ai-agents-vs-multi-agent-systems-solo-expertise-orchestrated-collective-intelligence) |
| Friston (2009) "Predictive coding under the free-energy principle" | Thalamus encodes prediction errors, cortex maintains generative model. Perception = inference. Action = minimizing expected free energy. | The theoretical framework Mae claims but does not implement. PatternCortex should maintain a generative model and compute surprise. | [pmc.ncbi.nlm.nih.gov/PMC2666703](https://pmc.ncbi.nlm.nih.gov/articles/PMC2666703/) |
| Distinguishing Autonomous AI Agents from Collaborative Agentic Systems (2025) | Comprehensive framework for modern intelligent architectures. Covers orchestration, coordination, and advisory patterns. | Useful framing for Mae's holarchic agent coordination. | [arxiv.org/2506.01438](https://arxiv.org/html/2506.01438v1) |

---

## Upgrade Recommendations (Ranked by Impact)

### 1. CRITICAL: Implement Competitive Ignition (GWT Compliance)

**What:** Signals should compete for workspace access. Not every digest should produce a full advisory. Introduce an ignition threshold.

**How (conceptually):** Add a `should_ignite()` check in PatternCortex.process_digest(). When aggregate salience exceeds a threshold (or when multiple signals reinforce each other via recurrence), ignition occurs and a full PatternAdvisory is produced. Below threshold, produce a minimal/quiet advisory with `recommended_tier="habit"` and low confidence. This creates the two-state (subliminal vs. conscious) dynamic that GWT requires.

**Biological analog:** Only signals that achieve NMDA-mediated recurrent amplification enter the global workspace. Below-threshold signals are processed unconsciously.

**Mathematical identity:** Directly addresses the Competition/Selection principle (item 7 in the Eight Necessary Properties).

**Impact:** HIGH -- transforms ADVISE from "observe everything" to "only important things reach awareness."

**Files affected:** `mae_core/patterns/pattern_cortex.py`

### 2. HIGH: Add Thalamic Reticular Nucleus (Inhibitory Gating)

**What:** Create a gating mechanism that suppresses irrelevant or redundant signals before they reach the cortex. Uses top-down feedback from previous advisories to modulate what the PatternBus relays.

**How (conceptually):** After PatternBus drains the inbox, apply an inhibition pass: if the cortex's last advisory already covers a domain, suppress low-salience signals in that domain. The cortex sends back a "current focus" signal that biases the PatternBus's processing.

**Biological analog:** TRN inhibits thalamic relay cells based on cortical feedback. Emotionally salient signals from the amygdala can override TRN suppression.

**Mathematical identity:** Creates genuine selection pressure. Not everything gets through.

**Impact:** HIGH -- prevents signal flooding and creates attentional focus.

**Files affected:** `mae_core/patterns/pattern_bus.py`

### 3. HIGH: Implement Prediction Error (Active Inference Compliance)

**What:** PatternCortex should maintain a simple generative model predicting what signals SHOULD arrive next step, then compute prediction error (surprise).

**How (conceptually):** Based on recent domain streaks and trends, the cortex predicts next-step domain distribution. When actual signals deviate significantly from prediction, the prediction error itself becomes a high-salience signal. The advisory's `novelty_level` should reflect prediction error rather than just signal presence.

**Biological analog:** Cortex generates top-down predictions. Thalamus relays prediction errors (actual minus expected). Surprise drives learning and attention.

**Mathematical identity:** Directly addresses Prediction/Error-Correction principle (item 8).

**Impact:** HIGH -- transforms ADVISE from reactive summarization to anticipatory inference.

**Files affected:** `mae_core/patterns/pattern_cortex.py`

### 4. MEDIUM: Precision-Weighted Integration

**What:** When computing aggregate domain levels and advisory confidence, weight each signal by its confidence (precision) rather than treating all signals equally.

**How (conceptually):** In `_compute_domain_level()`, multiply each signal's contribution by its confidence. High-confidence sources (e.g., AutoHealer at 0.9) should dominate over low-confidence sources. This is exactly precision-weighting from predictive processing.

**Biological analog:** Precision (inverse variance) determines the weight of prediction errors at every level of the cortical hierarchy.

**Impact:** MEDIUM -- better signal quality differentiation without major structural changes.

**Files affected:** `mae_core/patterns/pattern_cortex.py:296-319`

### 5. MEDIUM: Endocrine Modulation of PatternBus Gain

**What:** Connect the EndocrineSystem to the PatternBus so stress/arousal levels modulate signal processing sensitivity.

**How (conceptually):** Inject EndocrineSystem reference into PatternBus. When cortisol/adrenaline is high, lower the salience threshold for THREAT signals (sensitize to danger). When melatonin is high, raise thresholds (relax during consolidation phases).

**Biological analog:** Reticular Activating System modulates thalamic gain based on arousal state.

**Impact:** MEDIUM -- connects the hormonal system to sensory processing, which is biologically fundamental.

**Files affected:** `mae_core/patterns/pattern_bus.py`, `main.py`

### 6. MEDIUM: Priority-Based Inbox Draining

**What:** Change PatternBus inbox drain from FIFO to priority-ordered. Currently, MAX_SIGNALS_PER_STEP=50 takes signals in arrival order. Should take highest-salience signals first.

**How (conceptually):** Before draining, sort inbox by salience (or a composite score). This ensures that under budget pressure, the most important signals survive.

**Biological analog:** The thalamus does not process signals in arrival order; high-urgency signals preempt.

**Impact:** MEDIUM -- simple change with meaningful behavioral improvement under load.

**Files affected:** `mae_core/patterns/pattern_bus.py:127-134`

### 7. LOW: Formalize the PatternBus-PatternCortex-PatternConsolidator Triad

**What:** Register this trio as an explicit triad in the ConnectionRegistry with triadic witnessing. Currently they are wired sequentially but not registered as a triadic relationship.

**How (conceptually):** In Layer 23 of main.py, register the three components as a formal triad where each witnesses the other two's connections.

**Mathematical identity:** Satisfies the "No bare dyads" Connection Law.

**Impact:** LOW -- structural completeness rather than functional change.

**Files affected:** `main.py` (Layer 23 section)

### 8. LOW: Oscillatory Dynamics for Temporal Binding

**What:** Add simple oscillatory modulation to the cortex's window processing. Gamma-like fast oscillation for binding within a step, alpha-like slow oscillation for inhibition between steps.

**How (conceptually):** Modulate the exponential decay weights in `_compute_domain_level()` with a sinusoidal component tied to the step counter. This creates rhythmic sensitivity peaks and troughs.

**Biological analog:** Gamma oscillations (30-80 Hz) bind features within a percept. Alpha oscillations (8-12 Hz) implement inhibitory gating.

**Impact:** LOW -- adds biological realism but functional benefit is unclear without the other changes.

**Files affected:** `mae_core/patterns/pattern_cortex.py`

---

## Sources

- [Cortical control of adaptation and sensory relay mode in the thalamus (PNAS)](https://www.pnas.org/doi/10.1073/pnas.1318665111)
- [The pulvinar nucleus and its role in cognitive functions (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S0361923025004848)
- [Attentional Modulation of Thalamic Reticular Neurons (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6674014/)
- [A thalamic bridge from sensory perception to cognition (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S0149763420306473)
- [Inhibition of thalamic relay nuclei scales the aperiodic and alpha band oscillations (MIT Press)](https://direct.mit.edu/imag/article/doi/10.1162/imag_a_00451/127395/)
- [Hypothesis on Selection-Broadcast Cycle Structure: GWT (arxiv)](https://arxiv.org/html/2505.13969v1)
- [GWT and Prefrontal Cortex: Recent Developments (Frontiers)](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2021.749868/full)
- [Conscious Processing and the Global Neuronal Workspace Hypothesis (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8770991/)
- [Global Neuronal Workspace Theory (EmergentMind)](https://www.emergentmind.com/topics/global-neuronal-workspace-theory-gnwt)
- [Adversarial testing of GNW and IIT (Oxford Academic, 2025)](https://academic.oup.com/nc/article/2025/1/niaf037/8280147)
- [A detailed theory of thalamic and cortical microcircuits for predictive visual inference (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11800772/)
- [The Global Neuronal Workspace Model of Conscious Access (Dehaene et al.)](https://www.antoniocasella.eu/dnlaw/Dehaene_Changeaux_Naccache_2011.pdf)
- [Predictive coding under the free-energy principle (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2666703/)
- [Neuroanatomy, Reticular Activating System (NCBI)](https://www.ncbi.nlm.nih.gov/books/NBK549835/)
- [Spatial Organization of Multisensory Responses in Temporal Association Cortex (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6666661/)
- [Distinguishing Autonomous AI Agents from Collaborative Agentic Systems (arxiv)](https://arxiv.org/html/2506.01438v1)
- [Multi-Agent System Architecture Guide for 2026 (ClickIT)](https://www.clickittech.com/ai/multi-agent-system-architecture/)
- [AI Agents vs. Multi-agent systems (Talan)](https://www.talan.com/global/en/ai-agents-vs-multi-agent-systems-solo-expertise-orchestrated-collective-intelligence)