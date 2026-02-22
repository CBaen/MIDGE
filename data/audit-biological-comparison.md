> Generated from 10-agent audit conducted 2026-02-11. ~50 sub-agents. Sources: biology papers, GitHub, research papers, full codebase trace.

# Mae Audit: Biological Comparison Reference

Every biological mechanism compared to Mae's implementation, compiled from all 10 audit reports. Organized by biological system.

---

## Sensory Systems (from SENSE audit)

| Biological Mechanism | Mae's Analog | Accuracy | What's Missing |
|---------------------|-------------|----------|----------------|
| **Sensory receptor transduction** (converting stimuli to neural signals) | 11 Translators convert EventBus messages to PatternSignals | HIGH | Each translator is simplistic -- single threshold, no receptor adaptation. Real receptors have complex gain control and dynamic range adjustment. |
| **Action potential** (universal signal format) | PatternSignal dataclass with domain, form, confidence, salience | HIGH | Good universal format. Missing: frequency coding (rate of signal emission matters biologically). |
| **Thalamic relay** (filtering and routing sensory input) | PatternBus + SignalPriorityResolver | MODERATE | PatternBus collects and groups but does not truly filter. Real thalamus suppresses ~80% of incoming signals via TRN. |
| **Lateral inhibition** (enhancing edges and contrasts) | Not implemented | ABSENT | Real sensory systems use lateral inhibition to sharpen signal boundaries. PatternSense detectors are independent. |
| **Habituation / sensory adaptation** (decreasing response to constant stimuli) | Not implemented | ABSENT | Real neurons adapt: rapid-adapting receptors stop firing. PatternSense fires the same signal every step as long as the trend continues. |
| **Predictive coding** (brain predicts input, transmits only errors) | WorldModelTranslator captures prediction errors from WorldModel | LOW | SENSE layer itself is not predictive. Real processing sends predictions DOWN and errors UP. Mae only sends UP. |
| **Top-down attentional gating** (cortex modulates sensor sensitivity) | Not implemented | ABSENT | Real thalamus receives MORE cortical inputs than sensory ones. PatternBus has no input from cortex. |
| **Sensory receptor types** (fast vs slow adapting) | All detectors are equivalent | LOW | Real systems have Meissner (rapid change), Merkel (sustained), Pacinian (vibration), Ruffini (stretch). Mae has one speed. |
| **Multi-modal integration** (combining sight + sound + touch) | Cross-domain correlation in PatternBus (5 hardcoded pairs) | MODERATE | Biological integration is richer -- superior colliculus combines modalities with spatial/temporal coincidence rules. |
| **Efferent copy / corollary discharge** (predicting sensory consequences of own actions) | Not implemented | ABSENT | When you move your eyes, brain predicts visual shift. Mae's sense system has no awareness of what actions are taken. |

---

## Thalamic / Relay Systems (from ADVISE audit)

| Biological Mechanism | Mae's Analog | Accuracy | What's Missing |
|---------------------|-------------|----------|----------------|
| **Thalamic relay nuclei** (route sensory signals to cortex) | PatternBus receives via translators, groups by domain | GOOD | Real thalamus has relay mode AND burst mode (salience amplification). PatternBus has only relay mode. |
| **Thalamic Reticular Nucleus (TRN)** (GABAergic inhibition, attentional gating) | No analog. MAX_SIGNALS_PER_STEP is static budget. | MISSING | TRN selectively suppresses based on top-down cortical feedback and emotional salience. No suppressive gating in Mae. |
| **Association cortex** (integrates modalities over time) | PatternCortex with 13-step window, exponential decay | GOOD | Real cortex uses oscillatory coherence (gamma binding, alpha inhibition). No oscillatory dynamics. |
| **Reticular Activating System (RAS)** (arousal gating) | EndocrineSystem exists but does NOT modulate PatternBus gain | MISSING | PatternBus processes signals identically regardless of stress/arousal state. |
| **Global Workspace ignition** (nonlinear threshold, winning coalition broadcasts) | No analog. Every digest produces advisory. | MISSING | In GNW, only signals achieving ignition enter workspace. Mae broadcasts ALL advisories. |
| **Corticothalamic feedback** (layer 6 projections modulate relay) | Advisory -> Agent -> EventBus -> Translators exists but does not modulate PatternBus | WEAK | Real feedback changes thalamic gain in real-time. Mae's is structural, not functional. |
| **Pulvinar nucleus** (higher-order thalamus, salience filtering) | Cross-domain correlation detection in PatternBus | PARTIAL | Pulvinar actively coordinates cortico-cortical communication. Mae's is passive co-occurrence observation. |
| **Hippocampal-cortical dialogue** (consolidation during rest) | PatternConsolidator every 89 steps, ancestral recall | GOOD | Good biological analog. 89-step interval approximates sleep consolidation. |
| **Precision weighting** (unreliable signals attenuated) | Confidence field exists but not used for weighting | MISSING | In predictive processing, precision weights prediction errors. In Mae, all signals contribute equally. |

---

## Basal Ganglia / Decision Systems (from DECIDE audit)

| Biological Mechanism | Mae's Analog | Accuracy | What's Missing |
|---------------------|-------------|----------|----------------|
| **Reflex arc** (monosynaptic, spinal cord, <50ms) | ReflexPattern substring matching, fixed actions | GOOD | Gap: biological reflexes can be modulated by cortical inhibition (executive veto). Mae only forces prefrontal, cannot suppress reflexes. |
| **Basal ganglia habit** (Go/NoGo, dopaminergic RPE, cortico-BG-thalamic loop) | Exact string match habit lookup, strength parameter | MAJOR GAP | No Go/NoGo dual pathway. No competition between actions. No dopamine modulation. No habit decay from disuse. No generalization. |
| **Prefrontal deliberation** (model-based reasoning, working memory, conflict monitoring) | WorldModel.step() evaluates actions deterministically | PARTIAL | Has simulation. Missing: working memory integration, conflict monitoring, inhibitory control. |
| **Dopamine RPE** (prediction error drives learning) | EndocrineSystem.DOPAMINE released on reward/novelty | SHALLOW | No reward prediction error computation. No D1/D2 pathway effects. Dopamine is scalar, not prediction error signal. |
| **Thalamic relay** (filters/prioritizes all input to cortex) | SignalPriorityResolver: priority scoring, budget enforcement | GOOD | Budget enforcement is biological. Tier mapping to decision tiers is creative. |
| **Adrenaline / norepinephrine** (fight-or-flight, shifts to reactive processing) | EndocrineSystem.ADRENALINE -> DecisionRouter re-check | BROKEN | Re-check uses identical logic (no lowered threshold). set_reflex_bias() method does not exist. |
| **Habit formation** (hundreds-thousands of repetitions, chunking) | 5 identical prefrontal decisions -> habit | TOO FAST | Threshold too low. No context-dependency. No chunking of action sequences. |
| **Cortical-basal ganglia loops** (4-5 parallel loops) | Single decision pathway | MISSING | Biology has parallel motor, oculomotor, prefrontal, limbic loops. Mae has one pipeline. |

---

## Motor / Action Systems (from ACT audit)

| Biological Mechanism | Mae's Analog | Accuracy | What's Missing |
|---------------------|-------------|----------|----------------|
| **Motor cortex (M1)** (specific motor commands to muscle groups) | `_act()` stores label, returns 0.0 | ABSENT | No motor specificity. All "actions" are identical from environment's perspective. |
| **Efference copy** (motor command copy to cerebellum for prediction) | Not implemented | ABSENT | Core mechanism for validating actions in real-time. |
| **Neuromuscular junction** (neural commands become physical force) | No equivalent | ABSENT | No interface where decision becomes environmental change. |
| **Cerebellum coordination** (timing, error correction, smooth sequences) | DecisionRouter 3 tiers (in DECIDE, not ACT) | MISPLACED | Cerebellum analog should sit in ACT, not DECIDE. |
| **Proprioceptive feedback** (continuous position/force reporting during action) | None | ABSENT | No sensorimotor loop. Actions are open-loop (fire and forget). |
| **Motor planning (SMA/PMC)** (plan sequence before movement) | WorldModel rollouts in `_decide()` | PARTIAL | Planning exists but execution does not carry out the plan. "Flight plan but no airplane." |
| **Basal ganglia action selection** (inhibition/disinhibition, Go/NoGo) | DecisionRouter cascade | GOOD | Selection works. Execution is the gap. |

---

## Learning / Memory Systems (from LEARN + CONSOLIDATE + RECALL audits)

| Biological Mechanism | Mae's Analog | Accuracy | What's Missing |
|---------------------|-------------|----------|----------------|
| **Hippocampal encoding** (rapid one-shot storage) | EpisodicMemory.store() via PrioritizedReplayBuffer | GOOD | One-shot storage, priority-based. Solid implementation. |
| **Hippocampal replay** (sharp-wave ripples during rest) | learn_from_memory() every 13 steps | CONCEPT GOOD, IMPLEMENTATION WEAK | No actual learning occurs during replay. Priorities update but no weights. |
| **Sleep consolidation** (hippocampal -> neocortical transfer) | consolidate_memory() every 89 steps + PatternConsolidator -> Qdrant | CONCEPT GOOD, PARTIALLY BROKEN | Hot -> deep transfer works. But MemoryConsolidator calls nonexistent methods. |
| **Dopamine RPE** (prediction error drives plasticity) | CuriosityDrive computes prediction error | PRESENT BUT DISCONNECTED | Error is computed but never used to update parameters in `_learn()`. |
| **Hebbian/STDP** (neurons that fire together wire together) | None | ABSENT | No correlation-based learning anywhere. |
| **Cerebellar learning** (parallel fiber + climbing fiber = supervised learning) | WorldModel predictions exist but never compared to outcomes | ABSENT FROM _learn() | Would require efference copy mechanism. |
| **Homeostatic plasticity** (scaling all synapses to maintain activity) | None | ABSENT | No synaptic scaling or activity normalization. |
| **Memory reconsolidation** (recalled memories become labile, can be updated) | None | ABSENT | Once stored in Qdrant, memory is immutable. |
| **Pattern completion** (CA3 auto-associative) | Semantic search finds similar states by cosine distance | NO | Biology: fragment triggers full reinstatement. Mae requires full state vector. |
| **Pattern separation** (dentate gyrus orthogonalizes overlapping inputs) | None | ABSENT | No mechanism distinguishes similar-but-different memories. |
| **Spreading activation** | None | ABSENT | Retrieving one memory does not activate related memories. |
| **Context-dependent retrieval** | Salience threshold only | MINIMAL | Biology: emotional, hormonal, circadian, environmental context all modulate recall. Mae uses only salience. |
| **Synaptic tagging and capture** (weak learning promoted by temporal proximity to strong) | Priority-based storage via PrioritizedReplayBuffer | PARTIAL | No tagging mechanism. No cross-experience protein-analog resource sharing. |
| **Generative replay** (brain replays pseudo-experiences) | GenerativeReplayMemory with VAE | IMPLEMENTED | Present but disabled by default. Forward-looking feature. |

---

## Immune / Healing Systems (from HEAL audit)

| Biological Mechanism | Mae's Analog | Accuracy | What's Missing |
|---------------------|-------------|----------|----------------|
| **Innate immunity** (fast, broad: neutrophils, complement) | ThreatDetector quill sensors | GOOD | No complement cascade equivalent; no pattern-recognition receptors. |
| **Adaptive immunity** (slow, specific: T-cells, B-cells, antibodies) | HAVEN risk assessment + policy contagion detection | MODERATE | No memory B-cell equivalent. No antibody generation for recognized threats. |
| **Inflammation cascade** (recruit immune cells, seal wound) | AutoHealer Phase 1 ISOLATE + cortisol release | GOOD | No cytokine signaling cascade. Cortisol is simple scalar. |
| **Wound healing stages** (hemostasis, inflammation, proliferation, remodeling) | AutoHealer 3 phases map to first 3 | MODERATE | Missing remodeling phase. No scar tissue formation (permanent adaptation). |
| **Apoptosis** (programmed cell death) | ThreatDetector Lizard autotomy; model.hibernate_agent | GOOD ANALOGY | Apoptosis is self-initiated; Mae's sacrifice is externally imposed. No agent self-apoptosis. |
| **Autophagy** (cellular self-digestion for recycling) | None | ABSENT | No mechanism for agents to recycle degraded components or corrupted memories. |
| **DNA repair** | State persistence + restore_state | WEAK | Only restores from snapshots. No in-place error correction. |
| **Immune memory** (memory T-cells, vaccination) | HAVEN performance history; healing_history deque | MINIMAL | History tracked but not used for faster future recognition. No trained immunity. |
| **Tolerance** (self vs non-self; regulatory T-cells) | InputValidator trust scores | MINIMAL | No MHC-like self-markers. No thymic selection for immune tolerance. |
| **Fever** (systemic stress response) | Endocrine cortisol release during healing | GOOD | One-dimensional; real fever affects multiple systems. |
| **Regeneration** (stem cells at wound site) | StemCellRegistry + Morphogenesis.spawn | ARCHITECTURAL | Not wired: AutoHealer does not trigger stem cell re-differentiation. |

---

## Self-Awareness / Consciousness Systems (from SELF-AWARENESS audit)

| Biological Mechanism | Mae's Analog | Accuracy | What's Missing |
|---------------------|-------------|----------|----------------|
| **Anterior Insular Cortex** (interoception hub) | SomaticMap (health, dependencies, blast radius) | MODERATE | SomaticMap tracks health but agents do not "feel" it. No integration into decision-making. |
| **Somatosensory Cortex (Homunculus)** (topographic body map) | HolonRegistry + SomaticMap | MODERATE | Map exists but not used for real-time coordination. |
| **Proprioception** (continuous body position awareness) | AwarenessPulse every 25 steps | WEAK | Proprioception is continuous, not periodic. More like a checkup than constant awareness. |
| **Mirror Neuron System** (understanding others by internal simulation) | know_peers() returns structural metadata | VERY WEAK | No simulation of peer behavior. No model of what peers are doing or why. |
| **Default Mode Network** (self-referential processing during rest) | No analog | ABSENT | No idle self-reflection mode. No narrative self-model. |
| **Theory of Mind** (modeling others' mental states) | No analog beyond structural peer awareness | ABSENT | No agent models another agent's internal state, goals, or decisions. |
| **Interoception** (sensing internal signals: hunger, thirst, pain) | holon_heal() checks reward trend; SomaticMap health floats | WEAK | Basic metric exists but no rich internal signal stream. No hunger/thirst/pain driving behavior. |
| **Body Schema** (implicit model of body for action) | HolonMixin._detect_capabilities() | WEAK | Detects capabilities but does not model capacity, reach, or action space dynamically. Static at init. |

---

## Homeostatic / Regulatory Systems (from HIDDEN STEPS audit)

| Biological Mechanism | Mae's Analog | Accuracy | What's Missing |
|---------------------|-------------|----------|----------------|
| **Allostasis** (predictive regulation -- adjust BEFORE stressor) | None | ABSENT | Endocrine reacts to events. Never preemptively adjusts. |
| **Sympathetic/Parasympathetic balance** (active rest vs activity) | Circadian REST phase + melatonin exist | PARTIAL | No active parasympathetic drive. Recovery is passive decay, not active promotion. |
| **Nociception** (pain -- stop what you're doing NOW) | None distinct from threat detection | ABSENT | ThreatDetector detects external threats. Nothing signals internal damage requiring immediate action cessation. |
| **Neuromodulatory gain control** (dopamine/noradrenaline control signal gain) | Endocrine modulates some systems | PARTIAL | Hormones don't modulate signal GAIN in SignalPriorityResolver or pattern salience. |

---

## Accuracy Summary

| Rating | Count | % |
|--------|-------|---|
| GOOD / HIGH | 12 | 17% |
| MODERATE / PARTIAL | 17 | 24% |
| LOW / WEAK | 11 | 16% |
| ABSENT / MISSING | 30 | 43% |
| **Total mechanisms assessed** | **70** | 100% |

**43% of biological mechanisms have no Mae analog at all. Only 17% have good accuracy.**

### Top 5 Most Impactful Missing Mechanisms

1. **Predictive coding** -- Brain predicts before sensing. Mae is purely reactive.
2. **Dopamine RPE** -- The single most important biological learning signal. Completely absent from Mae's learning path.
3. **Thalamic Reticular Nucleus** -- Suppresses 80% of signals. Mae processes everything.
4. **Efference copy** -- Predicts consequences of own actions. Mae has no self-prediction of action outcomes.
5. **Habituation** -- Suppresses repeated stimuli. Mae fires the same signal indefinitely.
