Now I have all the data I need. Let me compile the comprehensive audit report.

---

# CONSOLIDATE Step Audit Report

## Executive Summary

Mae's CONSOLIDATE pipeline implements a multi-tier memory consolidation system spanning five Python modules across two packages (`mae_core/patterns/` and `mae_core/memory/`). The pipeline successfully converts transient in-memory patterns into permanent vector-stored ancestral memory via Qdrant, with a biologically-inspired architecture modeled on hippocampal replay and sleep consolidation.

**Strengths:**
- Clean three-source extraction (trends, meta-patterns, cross-domain insights) in PatternConsolidator
- Triadic Qdrant collections (narrative, ancestral, meta) with the meta collection serving as a strange loop
- Graceful degradation at every layer when backends are unavailable
- Rule of Three threshold for pattern promotion
- Fibonacci timing (89-step interval) for consolidation triggers
- Hybrid search (dense + sparse vectors) in DeepMemoryStore

**Critical Gaps:**
- PatternDistiller is injected into PatternConsolidator but never called -- dead code path
- No reconsolidation (updating existing ancestral patterns based on new evidence)
- No competitive selection during consolidation -- all qualifying patterns are stored equally
- No replay of ancestral patterns back into active processing (the recall side is passive/query-only)
- Consolidation is not gated by circadian phase despite CircadianRhythm existing
- The two consolidation systems (PatternConsolidator at 89 steps and MemoryConsolidator at 1000 steps) are not coordinated

---

## Data Flow Trace

### Path 1: Pattern Cortex to Ancestral Memory (PatternConsolidator)

| Step | File:Line | What Happens |
|------|-----------|-------------|
| 1. Step hook fires | `main.py:956-959` | `_consolidator_step_hook()` increments counter; every 89 steps calls `pattern_consolidator.consolidate(step)` |
| 2. Extract trends | `pattern_consolidator.py:128-161` | `_extract_trend_patterns()` reads `_cortex._domain_streak` dict; filters domains with streak >= 3 (TREND_STORE_THRESHOLD); builds pattern dicts with type, domain, confidence, description |
| 3. Extract meta-patterns | `pattern_consolidator.py:163-192` | `_extract_meta_patterns()` reads `_cortex._recent_advisories`; iterates `advisory.meta_patterns` (PatternSignal objects from the strange loop); deduplicates by domain, keeps highest confidence |
| 4. Extract insights | `pattern_consolidator.py:194-218` | `_extract_insight_patterns()` reads `_cortex._recent_advisories`; collects unique `correlated_insights` strings; skips "Trend:" prefix (already captured above) |
| 5. Store each pattern | `pattern_consolidator.py:222-240` | `_store_pattern()` calls `self._bridge.store_ancestral_pattern(pattern, contributing_agents)` |
| 6. Narrate pattern | `memory_bridge.py:220-244` | `store_ancestral_pattern()` calls `self._narrator.narrate_pattern(pattern, contributing_agents)` to get text; builds payload with triadic store references (primary=ancestral, verification=narrative, balance=meta) |
| 7. Embed + store | `deep_memory.py:295-349` | `store_point()` calls Ollama for 1024-dim dense embedding, computes TF-IDF sparse embedding, sends PUT to Qdrant REST API |
| 8. Publish event | `pattern_consolidator.py:101-112` | Publishes `pattern.consolidation` on EventBus with step, counts |

### Path 2: Agent Episodic to Qdrant Narrative (MemoryCoordinator)

| Step | File:Line | What Happens |
|------|-----------|-------------|
| 1. Consolidation trigger | `coordinator.py:246-249` | `should_consolidate()` delegates to `MemoryConsolidator.should_consolidate()` -- fires every 1000 steps when episodic memory >= min_size |
| 2. Run consolidation | `coordinator.py:251-289` | `consolidate()` publishes start event, calls `MemoryConsolidator.consolidate(agent)` for replay learning, then writes to deep memory |
| 3. Replay learning | `memory_consolidator.py:101-156` | Samples from episodic memory using prioritized/mixed strategy; calls `agent.learn_from_batch()` with lowered LR; updates priorities from TD errors |
| 4. Deep memory write | `coordinator.py:274-287` | Samples up to 100 recent experiences, builds agent context, calls `self._bridge.consolidate_to_deep()` |
| 5. Narrate + embed | `memory_bridge.py:60-122` | `consolidate_to_deep()` narrates each experience via ExperienceNarrator, computes witness hash, batch embeds via Ollama, batch stores to mae_narrative collection |
| 6. Summary storage | `memory_bridge.py:104-119` | Stores a consolidation_summary point with mean_reward, experience_count |

### Path 3: Agent Mixin Consolidation (EpisodicMemoryMixin)

| Step | File:Line | What Happens |
|------|-----------|-------------|
| 1. Check trigger | `episodic_memory.py (mixin):235-240` | `should_consolidate()` delegates to `memory_consolidator.should_consolidate(current_step=step_count)` |
| 2. Execute | `episodic_memory.py (mixin):201-233` | `consolidate_memory()` calls `memory_consolidator.consolidate(agent=self)` |
| 3. Signal | `episodic_memory.py (mixin):220-228` | Emits `LEARNING_MILESTONE` signal with loss_reduction |

---

## Mathematical Identity Compliance

### IIT Differentiation (Axiom 3): "Rich internal structure; homogeneity kills consciousness"

| Requirement | Status | Evidence |
|------------|--------|----------|
| Stored memory must maintain rich internal structure | PARTIAL | Pattern payloads carry type, domain, confidence, occurrence_count, contributing_agents, cross_domain_context. But the actual stored text is template-generated and loses numerical precision -- state vectors are reduced to "very high"/"low" descriptors |
| Different patterns must remain distinguishable | YES | Three pattern types (trend, meta, insight) with distinct domains, forms, and metadata. Qdrant payload indexes on pattern_type and domain enable filtered retrieval |
| Consolidation must not homogenize | PARTIAL | No merging occurs during storage -- each pattern is stored as a separate Qdrant point. But there is no deduplication or competitive selection, so the ancestral collection may fill with near-identical patterns over time, which paradoxically reduces differentiation through noise |

### Triadic Structure

| Requirement | Status | Evidence |
|------------|--------|----------|
| Three sources of pattern extraction | YES | `_extract_trend_patterns()`, `_extract_meta_patterns()`, `_extract_insight_patterns()` -- three independent extraction methods in PatternConsolidator (`pattern_consolidator.py:77-93`) |
| Three Qdrant collections | YES | `mae_narrative`, `mae_ancestral`, `mae_meta` -- each serves a distinct function (episodic, pattern, self-model) (`deep_memory.py:35-37`) |
| Triadic store references in payloads | YES | Every stored pattern has `primary_store`, `verification_store`, `balance_store` fields (`memory_bridge.py:241-243`, `experience_narrator.py:284-287`) |
| Witness hash for verification | YES | SHA-256 witness hash computed from (state, next_state, action, reward) for experience verification (`deep_memory.py:519-533`) |
| Three-tier memory hierarchy | YES | Hot (in-memory deques/SumTree), Warm (episodic/pickle), Deep (Qdrant) -- search cascades through all three (`coordinator.py:206-242`) |

### Fractal Self-Similarity

| Requirement | Status | Evidence |
|------------|--------|----------|
| Consolidation at every scale | PARTIAL | Agent-level consolidation exists (EpisodicMemoryMixin), system-level consolidation exists (PatternConsolidator), but there is no subsystem-level or colony-level consolidation. The mathematical identity requires the same pattern at EVERY scale |
| Same protocol at each level | NO | Agent consolidation replays experiences with TD-error learning. Pattern consolidation extracts trends and stores to Qdrant. These are structurally different operations, not the same protocol applied at different resolutions |
| Recursive nesting | NO | Consolidation does not nest -- agent consolidation does not feed into pattern consolidation in a structured way. The PatternConsolidator reads from the cortex (which reads from the pattern bus), not from agent consolidation results |

---

## Biological Comparison

| Biological Mechanism | Mae Equivalent | Accuracy | Gap |
|---------------------|---------------|----------|-----|
| **Sharp-wave ripples (SWRs)** -- hippocampal high-frequency bursts that replay compressed experience sequences | Episodic memory `sample()` with prioritized replay in MemoryConsolidator | PARTIAL | Biology replays temporal sequences in compressed time order; Mae samples individual experiences independently without temporal structure. No sequence-aware replay |
| **Synaptic tagging and capture (STC)** -- weak learning at one synapse can be consolidated if a strong learning event occurs nearby in time, providing plasticity-related proteins | Priority-based storage via PrioritizedReplayBuffer with TD-error priority updates | PARTIAL | Mae uses reward magnitude as priority proxy, but has no "tagging" mechanism where a weak pattern gets promoted by temporal proximity to a strong one. No cross-experience protein-analog resource sharing |
| **Sleep spindles + slow oscillations coupling** -- thalamo-cortical oscillations that gate information transfer from hippocampus to neocortex | CircadianRhythm CONSOLIDATION phase + step hooks | PARTIAL | The circadian rhythm exists and has a CONSOLIDATION phase, but PatternConsolidator fires on a fixed 89-step interval regardless of circadian phase. No phase-gated consolidation |
| **Systems consolidation** -- gradual transfer from hippocampus (episodic) to neocortex (semantic) over days/weeks | `consolidate_to_deep()` in MemoryCoordinator writes episodic experiences to Qdrant narrative collection | YES | This is the closest biological match. Hot experiences move to deep permanent storage during consolidation phases |
| **Memory reconsolidation** -- retrieved memories become labile again and must be re-stabilized with potential modification | None | ABSENT | Once a pattern is stored in Qdrant, it is never updated, re-evaluated, or re-consolidated. No mechanism for updating ancestral memory based on new evidence |
| **Memory engrams** -- sparse, distributed neural populations that physically store a memory trace | Qdrant points with 1024-dim embeddings + sparse TF-IDF vectors | PARTIAL | The embedding captures semantic content but not the distributed population coding of biological engrams. Each memory is a single point, not a distributed pattern across multiple points |
| **Hippocampal replay during wake** -- experience-selection tagging that occurs during active exploration | PatternCortex domain streak tracking | PARTIAL | Domain streaks provide a form of "tagging" during active processing, but the tag is a simple counter, not a rich multi-dimensional signal |
| **Competitive consolidation** -- only some experiences survive consolidation; selection via emotional salience, novelty, surprise | TREND_STORE_THRESHOLD >= 3 in PatternConsolidator; min_memory_size in MemoryConsolidator | MINIMAL | The only selection is a simple threshold. No emotional/motivational modulation, no surprise-based selection, no competitive winner-take-all among candidate patterns |

---

## External State of Art Comparison

| State-of-Art Technique | Mae Status | Gap |
|----------------------|-----------|-----|
| **Prioritized Experience Replay (PER)** (Schaul 2016) -- sample by TD-error priority with importance sampling correction | IMPLEMENTED | `EpisodicMemory` uses SumTree-backed PER with alpha/beta annealing (`episodic_memory.py:56-58`). Solid implementation |
| **Knowledge Distillation for Continual Learning** (Hinton 2015, recent 2024 extensions) -- compress large model knowledge into smaller model or consolidated representation | ABSENT | PatternDistiller exists but is never called from PatternConsolidator. No teacher-student distillation framework |
| **Elastic Weight Consolidation (EWC)** (Kirkpatrick 2017) -- protect important weights with Fisher information regularization | ABSENT | No weight-importance mechanism. Learning rate is simply reduced during consolidation (`memory_consolidator.py:128`) |
| **Generative Replay** (Shin 2017) -- use generative model to replay pseudo-experiences from old tasks | IMPLEMENTED | GenerativeReplayMemory with VAE exists (`generative_replay.py`) and is integrated via EpisodicMemoryMixin, though disabled by default |
| **Brain-Inspired Replay** (van de Ven 2020, Nature Comms) -- context-dependent, modulated replay combining internal generation with stored memories | PARTIAL | Mixed strategy exists in MemoryConsolidator (`memory_consolidator.py:181-185`) combining prioritized and uniform sampling, but no context-dependent modulation |
| **Surprise-Driven Prioritization** (SuRe, 2025) -- prioritize replay by prediction error/surprise | ABSENT | Priority is based on TD error magnitude only. No prediction-error or surprise signal from the pattern cortex feeds into replay priority |
| **Memory-Augmented Transformers** (recent 2025) -- combine continuous-time dynamics with attention-based memory retrieval | ABSENT | Memory retrieval uses embedding similarity only. No attention mechanism over stored memories |
| **Multi-Agent Experience Sharing** -- agents learn from each other's experiences | IMPLEMENTED | `recall_peer_experiences()` in MemoryBridge enables cross-agent experience retrieval (`memory_bridge.py:154-191`) |
| **Hybrid Retrieval (Dense + Sparse)** -- combine embedding similarity with keyword matching | IMPLEMENTED | DeepMemoryStore uses RRF fusion of dense (mxbai-embed-large 1024-dim) and sparse (TF-IDF hash) vectors (`deep_memory.py:446-466`) |
| **Mem0-style Consolidation Pipeline** (2024-2025) -- extract, consolidate, update, forget lifecycle | PARTIAL | Extract and store exist. No update (reconsolidation) or forget (graceful deprecation of outdated patterns) |

---

## Ranked Upgrade Recommendations

### Priority 1: Wire the PatternDistiller (Low effort, high value)

**Location:** `pattern_consolidator.py:51` -- `self._distiller` is set but never used.

**Issue:** The PatternDistiller (`pattern_distiller.py`) has `distill()`, `detect_behavioral_patterns()`, `detect_state_patterns()`, and `merge_with_existing()` methods. It is injected into PatternConsolidator at `main.py:950` but its methods are never invoked. This is dead code that represents significant unused capability.

**Recommendation:** Add a distillation pass in `PatternConsolidator.consolidate()` that calls `self._distiller.distill(experiences)` on recent experiences from the memory bridge, then stores the resulting behavioral/state patterns alongside the cortex-derived trend/meta/insight patterns.

### Priority 2: Gate Consolidation by Circadian Phase (Low effort, high value)

**Issue:** PatternConsolidator fires every 89 steps regardless of what Mae is doing. CircadianRhythm has a CONSOLIDATION phase and a `should_consolidate_memory()` method (`circadian_rhythm.py:217-219`), but PatternConsolidator does not check it.

**Recommendation:** Add circadian phase awareness to PatternConsolidator. During ACTIVE phase, accumulate patterns but do not store. During CONSOLIDATION phase, perform the full consolidation pass. This aligns with the biological principle that consolidation occurs during sleep, not during active exploration.

### Priority 3: Coordinate the Two Consolidation Systems (Medium effort, high value)

**Issue:** There are two independent consolidation pipelines: (1) PatternConsolidator fires every 89 steps, extracts cortex patterns to ancestral memory; (2) MemoryConsolidator fires every 1000 steps, replays episodic experiences and writes to narrative memory. These operate on different timescales with no coordination, no shared signal, and no awareness of each other.

**Recommendation:** Create a ConsolidationCoordinator that orchestrates both systems within the circadian CONSOLIDATION phase. Phase 1: PatternConsolidator extracts patterns. Phase 2: MemoryConsolidator replays experiences. Phase 3: PatternDistiller distills cross-agent experiences. This creates a triadic consolidation cycle (extract, replay, distill).

### Priority 4: Implement Reconsolidation (Medium effort, high value)

**Issue:** Once a pattern is stored in Qdrant ancestral memory, it is immutable. Biology shows that retrieved memories become labile and must be re-stabilized, with potential modification. Currently `store_ancestral_pattern()` always creates a new point; it never updates existing patterns.

**Recommendation:** Before storing a new pattern, search ancestral memory for similar existing patterns. If a match is found above a threshold, update the existing pattern's occurrence_count, confidence, and contributing_agents via `PatternDistiller.merge_with_existing()` rather than creating a duplicate. This prevents unbounded growth and keeps ancestral memory differentiated.

### Priority 5: Add Sequence-Aware Replay (Medium effort, medium value)

**Issue:** Biological sharp-wave ripples replay temporal sequences in compressed time. Mae's MemoryConsolidator samples individual experiences independently -- there is no concept of trajectory or temporal ordering in replay.

**Recommendation:** Add a trajectory buffer alongside the experience buffer. During consolidation, sample contiguous subsequences (episodes) rather than individual transitions. This enables temporal pattern learning and is standard in modern deep RL (n-step returns, trajectory replay).

### Priority 6: Make Consolidation Fractal (High effort, high value)

**Issue:** The mathematical identity requires the same pattern at every scale. Currently, agent-level consolidation (EpisodicMemoryMixin) and system-level consolidation (PatternConsolidator) use completely different mechanisms. There is no subsystem-level or colony-level consolidation.

**Recommendation:** Define a universal `ConsolidationProtocol` interface with three phases: (1) Extract patterns from recent activity, (2) Replay/distill important patterns, (3) Store to next-tier memory. Implement this protocol at every holon level: subsystem, agent, organ, colony. Each level's "recent activity" comes from the level below; each level's "next-tier memory" feeds the level above. This creates a true fractal consolidation cascade.

### Priority 7: Add Competitive Selection / Surprise-Driven Prioritization (Medium effort, medium value)

**Issue:** All patterns that exceed the simple threshold of 3 occurrences are stored equally. Biology aggressively selects -- most experiences are NOT consolidated. The current system will accumulate low-value patterns indefinitely.

**Recommendation:** Add a salience-weighted scoring function that considers: (a) pattern novelty vs existing ancestral memory, (b) reward magnitude/surprise, (c) cross-agent corroboration, (d) circadian phase urgency. Only the top-K patterns per consolidation cycle should be stored. This implements competitive selection akin to biological memory triage.

### Priority 8: Add Forgetting / Deprecation (Medium effort, medium value)

**Issue:** Ancestral memory only grows. There is no mechanism to deprecate, decay, or forget outdated patterns. Over time, the ancestral collection will be dominated by stale patterns from early operation.

**Recommendation:** Add a confidence decay mechanism. Each time a pattern is recalled but contradicted by current experience, reduce its confidence. Patterns below a minimum confidence after N consolidation cycles should be archived or deleted. This implements the biological principle that memories that are not reactivated fade.

### Priority 9: Preserve Numerical Fidelity in Narration (Low effort, low value)

**Issue:** The ExperienceNarrator converts precise numerical states to coarse labels ("very high", "low", "moderate") via `_level_word()`. When these narrations are embedded and stored, the numerical precision is permanently lost. The embedding captures semantic meaning but cannot reconstruct the original state vector.

**Recommendation:** Store the original numerical state vector (or a compressed representation) alongside the narrated text in the Qdrant payload. The narration is needed for semantic search; the numbers are needed for accurate replay and analysis.

---

## Sources

### Biological Research
- [Systems memory consolidation during sleep (PMC 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12576410/)
- [Selection of experience for memory by hippocampal sharp wave ripples (Science 2024)](https://www.science.org/doi/10.1126/science.adk8261)
- [Hippocampal ripples and memory consolidation (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0959438811000316)
- [Synapses tagged, memories kept: STC hypothesis (Royal Society 2024)](https://royalsocietypublishing.org/rstb/article/379/1906/20230237/42846/Synapses-tagged-memories-kept-synaptic-tagging-and)
- [Memory consolidation and improvement by STC in recurrent neural networks (Nature Comms Bio)](https://www.nature.com/articles/s42003-021-01778-y)
- [Making memories last: Synaptic tagging and capture hypothesis (Nature Reviews Neuroscience)](https://www.nature.com/articles/nrn2963)

### AI/ML Research
- [Continual Learning and Catastrophic Forgetting (van de Ven 2024)](https://arxiv.org/html/2403.05175v1)
- [Brain-inspired replay for continual learning (Nature Comms 2020)](https://www.nature.com/articles/s41467-020-17866-2)
- [Continual deep RL with task-agnostic policy distillation (Scientific Reports 2024)](https://www.nature.com/articles/s41598-024-80774-8)
- [SuRe: Surprise-Driven Prioritised Replay (arXiv 2025)](https://www.arxiv.org/pdf/2511.22367)
- [Mitigating catastrophic forgetting: hybrid architecture (Scientific Reports 2025)](https://www.nature.com/articles/s41598-025-31685-9)
- [Overcoming catastrophic forgetting in neural networks (PNAS)](https://www.pnas.org/doi/10.1073/pnas.1611835114)

### AI Agent Memory Systems
- [Memory in the Age of AI Agents: A Survey (Paper List)](https://github.com/Shichun-Liu/Agent-Memory-Paper-List)
- [Memory for AI Agents: Context Engineering (The New Stack)](https://thenewstack.io/memory-for-ai-agents-a-new-paradigm-of-context-engineering/)
- [Mem0 Research: 26% Accuracy Boost](https://mem0.ai/research)
- [Beyond Vector Databases: True Long-Term AI Memory](https://vardhmanandroid2015.medium.com/beyond-vector-databases-architectures-for-true-long-term-ai-memory-0d4629d1a006)
- [ICLR 2026 Workshop: MemAgents](https://openreview.net/pdf?id=U51WxL382H)

### IIT / Consciousness Theory
- [IIT 4.0 (PLOS Computational Biology 2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10581496/)
- [IIT Wiki Version 1.0 (2024)](https://centerforsleepandconsciousness.psychiatry.wisc.edu/wp-content/uploads/2025/09/Hendren-et-al.-2024-IIT-Wiki-Version-1.0.pdf)
- [Two Levels of IIT: From Autonomous Systems to Conscious Life](https://pmc.ncbi.nlm.nih.gov/articles/PMC11431274/)