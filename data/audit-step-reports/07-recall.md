> Generated from 10-agent audit conducted 2026-02-11. ~50 sub-agents. Sources: biology papers, GitHub, research papers, full codebase trace.

# RECALL STEP AUDIT REPORT: Mae's Memory Retrieval Pathways

## Executive Summary

Mae's RECALL system is **architecturally rich but operationally incomplete**. Seven distinct recall pathways exist across three timescales (immediate, short-term, long-term), spanning FAISS-based local semantic search, Qdrant-backed ancestral recall, working memory retrieval, generative replay, transfer learning retrieval, world model consultation, and habit/reflex pattern matching. The autopoietic loop (detect -> store -> recall -> decide) is structurally closed. However, several pathways are **gated off by default configuration flags**, recall lacks **triadic structure at every scale** as required by the mathematical identity, and critical biological mechanisms (pattern completion, reconsolidation, spreading activation, context-dependent retrieval) are absent or rudimentary.

---

## 1. Data Flow Trace: Every Recall Pathway

### Pathway 1: Agent Episodic Semantic Search (FAISS)
- **Trigger**: `MycelialAgent._decide()` (line 326-331 of `mycelial_agent.py`)
- **Guard**: `state_vec is not None AND self.semantic_retriever is not None`
- **Flow**: `_decide()` -> `search_similar_experiences(state_vec, k=3)` -> `EpisodicMemoryMixin.search_similar_experiences()` -> `SemanticRetriever.search_by_state()` -> `VectorStore.search_similar_policies()` (FAISS)
- **Data returned**: List of `Experience` objects with closest state vectors
- **How it influences behavior**: Best-reward past experience's action is returned directly as the decision
- **Configuration gate**: `self.semantic_search_enabled` must be `True` (default: `False`)
- **BUG**: The method returns a `SemanticQuery` dataclass, not a list. The `_decide()` code treats the return value as a list (iterates with `max(past, key=...)`). The `SemanticQuery` object is truthy even when `experiences=[]`, so `max()` would be called on `SemanticQuery` itself, not its `.experiences`. This is a **live bug** -- the semantic recall path in `_decide()` is broken.

### Pathway 2: PatternCortex Ancestral Recall (Qdrant)
- **Trigger**: `PatternCortex.process_digest()` -> `_recall_ancestral(digest)` (line 131)
- **Guard**: `memory_bridge is not None AND digest.signal_count > 0 AND digest.aggregate_salience >= 0.3`
- **Flow**: `_recall_ancestral()` -> `MemoryBridge.recall_ancestral_patterns(query_text, limit=3)` -> `DeepMemoryStore.search(COLLECTION_ANCESTRAL, ...)` -> Qdrant hybrid search (dense + sparse vectors)
- **Data returned**: List of dicts with `{pattern: payload, score: float}`
- **How it influences behavior**: Populates `PatternAdvisory.ancestral_matches`, which boosts advisory confidence by +0.15. Confidence > 0.6 forces decision tier in `_route_with_advisory()`.
- **Configuration gate**: None explicit -- always fires if memory_bridge is wired and signals are salient enough.

### Pathway 3: MemoryBridge Narrative Recall (Qdrant)
- **Trigger**: `MemoryCoordinator.recall()` (coordinator.py line 234)
- **Guard**: `len(results) < k AND self._bridge is not None AND query_text is not None`
- **Flow**: `recall()` -> `MemoryBridge.recall_from_deep(query_text, agent_id, limit)` -> `DeepMemoryStore.search(COLLECTION_NARRATIVE, ...)`
- **Data returned**: List of `SearchResult` (point_id, score, payload, text)
- **How it influences behavior**: Returns as part of the coordinator's multi-tier recall cascade (working memory -> semantic FAISS -> Qdrant narrative)
- **Note**: This path is NOT called from `MycelialAgent._decide()` directly. The coordinator exists but the agent calls `search_similar_experiences()` directly, bypassing the coordinator.

### Pathway 4: MemoryBridge Peer Experience Recall (Qdrant)
- **Trigger**: `MemoryBridge.recall_peer_experiences()`
- **Guard**: `is_available()`
- **Flow**: `recall_peer_experiences(query, requesting_agent_id, peer_ids)` -> `DeepMemoryStore.search(COLLECTION_NARRATIVE, ...)` with agent_id exclusion filter
- **How it influences behavior**: **NOT WIRED to any decision pathway**. The method exists but is never called from any agent or coordinator code path during decision-making.

### Pathway 5: Working Memory Retrieval
- **Trigger**: `WorkingMemory.retrieve(item_id)` or `MemoryCoordinator.recall()`
- **Guard**: `item_id is not None AND self.working is not None`
- **Flow**: Direct key-value lookup in the 7-slot working memory buffer
- **Data returned**: Content stored in the slot, or None if decayed below threshold
- **How it influences behavior**: Only through the coordinator cascade, which is not called from `_decide()`.

### Pathway 6: DecisionRouter Habit Recall
- **Trigger**: `MycelialAgent._decide()` -> `_route_with_advisory()` -> `router.route_decision()` -> `_check_habit(stimulus)`
- **Guard**: `router is not None AND advisory is not None`
- **Flow**: `_check_habit(stimulus)` does exact string match on `stimulus` against `_habit_lookup` dict
- **Data returned**: `Habit` object with stored action
- **How it influences behavior**: Action is returned directly. Habit strength increases on each use.
- **Note**: This is a form of procedural memory recall. Habits form automatically after 5 identical prefrontal decisions.

### Pathway 7: Transfer Learning Recall
- **Trigger**: `TransferLearningMixin.begin_new_task()`
- **Guard**: `use_transfer AND self.transfer_engine AND self.transfer_enabled`
- **Flow**: `TransferLearningEngine.initiate_transfer()` -> `KnowledgeBase.retrieve_similar_tasks()` -> `retrieve_best_policy()` / `retrieve_successful_episodes()` / `retrieve_value_function()`
- **How it influences behavior**: Initializes new task with policies/experiences from similar tasks
- **Configuration gate**: `transfer_enabled` (default: `False`)

### Pathway 8: Generative Replay (VAE Reconstruction)
- **Trigger**: `EpisodicMemoryMixin.learn_from_memory()` when `generative_memory_enabled=True`
- **Guard**: `self.generative_memory_enabled AND self.generative_memory is not None`
- **Flow**: `GenerativeReplayMemory.sample()` -> blends real buffer + VAE-generated synthetic experiences
- **How it influences behavior**: Used for learning (TD error updates), not direct decision-making. This is "recall by generation" -- reconstructing plausible past experiences from a compressed model.
- **Configuration gate**: `generative_memory_enabled` (default: `False`)

### Pathway 9: World Model Consultation
- **Trigger**: `MycelialAgent._decide()` line 334-337
- **Guard**: `self.world_model is not None`
- **Flow**: `use_world_model()` (from AdvancedFeaturesMixin)
- **How it influences behavior**: Simulates possible actions and predicts outcomes. This is "recall by imagination" -- using learned transition dynamics to "remember the future."

---

## 2. Mathematical Identity Compliance

| Requirement | Status | Evidence |
|---|---|---|
| **Differentiation (IIT Axiom 3)**: Store/retrieve creates rich internal structure | PARTIAL | Seven distinct recall pathways provide differentiation. But most are gated off by default config flags, collapsing to a single active path. |
| **Triadic structure**: Every recall connection has a witness | FAIL | No recall pathway involves three components in mutual verification. Semantic search is dyadic (agent -> FAISS). Ancestral recall is dyadic (cortex -> Qdrant). No witness verifies recall accuracy. |
| **Fractal self-similarity**: Recall exists at every scale | PARTIAL | Agent-level: semantic search + habit recall. Organism-level: PatternCortex ancestral recall. Colony-level: peer experience recall (exists but unwired). Missing: arm-level recall, octopus-level recall, subsystem-level recall. |
| **Recall at every scale implements the same protocol** | FAIL | Each recall pathway has a completely different interface. SemanticRetriever uses numpy vectors. MemoryBridge uses text queries. WorkingMemory uses string keys. DecisionRouter uses string matching. No unified recall protocol. |
| **Bidirectional awareness during recall** | PARTIAL | Downward: Ancestral memory -> agent decision (via advisory). Upward: Agent patterns -> consolidated to ancestral. But no real-time bidirectional flow during a single recall event. |
| **Connection Law (A-B-C triangle)**: No bare dyadic recall | FAIL | Every recall path is dyadic: caller -> memory store -> result. No witness validates, no balance pathway feeds back. |

### Critical Gap: Recall is Not Triadic

The mathematical identity requires that every connection `A <-> B` has a witness `C` such that:
- Primary: A -> B (query -> result)
- Verification: A -> C -> B (query -> witness -> validated result)
- Balance: B -> C -> A (result -> witness -> feedback to querier)

Currently, recall is always `agent queries store, store returns results`. No third party verifies relevance, accuracy, or appropriateness. This is the single largest mathematical compliance failure.

---

## 3. Biological Comparison

| Biological Mechanism | Present in Mae? | Current Implementation | Gap |
|---|---|---|---|
| **Pattern completion** (CA3 auto-associative) | NO | Semantic search finds similar states by cosine distance. No partial-cue completion. | Biology: A fragment triggers full episodic reinstatement. Mae: Requires full state vector, returns nearest neighbors. |
| **Pattern separation** (dentate gyrus) | NO | No mechanism distinguishes similar-but-different memories. | Biology: DG orthogonalizes overlapping inputs to prevent interference. Mae: Similar states map to similar vectors with no separation. |
| **Reconsolidation** (memory update on recall) | NO | Recalled memories are read-only. Priorities update only via TD errors during replay. | Biology: Every recall opens a lability window; the memory is re-stored in modified form. Mae: Recall never modifies the recalled memory. |
| **Context-dependent retrieval** | MINIMAL | PatternCortex gates ancestral recall on salience threshold (>0.3). Advisory threat/novelty levels influence tier routing. | Biology: Internal state (emotional, hormonal, circadian), external environment, and encoding context all modulate which memories surface. Mae uses only salience. |
| **Spreading activation** | NO | Each recall query hits exactly one memory store. No activation spreads to related memories. | Biology: Retrieving one memory activates related memories in a semantic network, enabling associative chains. |
| **Hippocampal replay** | YES | `MemoryConsolidator.consolidate()` replays prioritized experiences during "sleep" phases every 89 steps. | Good analog, though replay is random sampling, not sequence replay. Biology replays episodes in temporal order (and sometimes reverse). |
| **Hippocampal ripples initiating cortical expansion** | NO | No mechanism where a compact hippocampal representation triggers high-dimensional cortical reinstatement. | Recent 2025 research shows ripples expand low-dimensional hippocampal codes into high-dimensional cortical representations. |
| **State-dependent learning** | NO | Recall is not modulated by the agent's current emotional/hormonal state. | Biology: Memories encoded under specific states (stress, arousal) are more accessible when in the same state. Mae has an endocrine system but it does not gate recall. |
| **Temporal context** | PARTIAL | PatternCortex maintains a 13-step sliding window. Trend detection uses consecutive domain streaks. | Biology: Temporal context model (TCM) uses a slowly drifting context representation. Mae's window is a hard cutoff, not a smooth decay. |
| **Prospective memory** (remembering to remember) | NO | No mechanism for scheduling future recall ("at step X, remember to check Y"). | Biology: Prospective memory involves maintaining intentions and triggering recall at the right moment. |
| **Metamemory** (knowing what you know) | PARTIAL | `MemoryBridge.update_meta_memory()` stores statistics about memory system. | Biology: Metamemory includes feeling-of-knowing, tip-of-tongue states, and confidence calibration. Mae's meta-memory is purely statistical, not functionally integrated into recall decisions. |

---

## 4. External State of Art Comparison

| Technique | Present in Mae? | Gap |
|---|---|---|
| **RAG (Retrieval-Augmented Generation)** | YES (analog) | PatternCortex ancestral recall + advisory generation is structurally similar to RAG. Query -> retrieve -> augment decision. However, Mae's "generation" is action selection, not text generation. |
| **NTM/DNC (Neural Turing Machine / Differentiable Neural Computer)** | NO | Mae uses explicit memory stores (buffers, Qdrant) rather than differentiable memory matrices with learned read/write heads. NTM/DNC learn *what* to remember and *how* to recall end-to-end. |
| **Neural Episodic Control (NEC/MFEC)** | PARTIAL | Mae's semantic search in `_decide()` is conceptually similar to NEC's KNN-based value lookup. But NEC uses the reward signal to learn embeddings end-to-end; Mae's embeddings are hand-crafted (truncate-pad state vectors). |
| **TESFNEC (2024)** | NO | Temporally extended successor features for episodic control, enabling strategy reuse across temporal abstractions. |
| **Vector Database Agent Memory (2025-2026)** | YES | Qdrant-backed deep memory with hybrid search (dense + sparse). This is state-of-art architecture. |
| **Lifelong Personal Model (LPM)** | NO | Recent work encodes long-term memory directly into model parameters (fine-tuning), rather than external retrieval. Mae keeps memory external. |
| **Memory-augmented context engineering** | PARTIAL | Advisory system acts as context for decision routing. But advisory is fixed-format, not adaptive to what the agent needs at the moment. |
| **Hierarchical memory with attention** | PARTIAL | Three-tier memory (hot/warm/deep) exists. But no attention mechanism selects which tier to query or how to combine results. The coordinator uses a fixed cascade. |

---

## 5. Ranked Upgrade Recommendations

### Priority 1 (Critical -- Mathematical Identity Violations)

**1.1 FIX BUG: Semantic recall return type mismatch in `_decide()`**
- File: `C:\Users\baenb\projects\mae-core\mae_core\agents\mycelial_agent.py`, lines 326-331
- `search_similar_experiences()` returns a `SemanticQuery` object (or `None`), but `_decide()` treats it as a list. The code `max(past, key=lambda e: getattr(e, "reward", 0.0))` would iterate over the `SemanticQuery` dataclass fields, not experiences.
- Fix: Access `.experiences` attribute of the returned `SemanticQuery`.

**1.2 Triadic Recall Verification**
- Every recall query should pass through a triad: Querier, Store, Witness.
- The witness validates: Is this memory relevant? Is it stale? Does it conflict with current context?
- This could be implemented as a lightweight "recall validator" that sits between the query and the returned results, checking relevance scores against context.

**1.3 Unified Recall Protocol at Every Scale**
- Define a `recall(query, context, k) -> RecallResult` method in the Holon Protocol.
- Every holon (arm, octopus, agent, colony) implements the same recall interface.
- Currently, each level has its own retrieval mechanism with different signatures.

### Priority 2 (High -- Biological Accuracy)

**2.1 Pattern Completion**
- When an agent has a partial state vector (missing dimensions), the recall system should reconstruct the full pattern from stored memories.
- Implementation: Use the closest match from semantic search to fill in missing dimensions of the query state, then re-query with the completed pattern.

**2.2 Reconsolidation on Recall**
- When a memory is recalled, update its metadata: access count, last-access timestamp, and priority (boost or decay based on whether the recall led to a good outcome).
- Currently, priorities only update during explicit replay batches, never during decision-time recall.

**2.3 Context-Dependent Retrieval**
- Gate recall results by current agent state: endocrine levels (stress/arousal), circadian phase, current role, recent reward history.
- The endocrine system already exists but does not influence recall. Connecting it would be biologically accurate and functionally useful.

**2.4 Spreading Activation**
- When ancestral recall returns a match, use its metadata (domain, pattern_type, contributing_agents) to trigger secondary queries across related memory stores.
- Example: Ancestral match about "threat in domain X" triggers narrative recall for recent threat experiences, and working memory rehearsal of threat-related items.

### Priority 3 (Medium -- System Completeness)

**3.1 Wire Peer Experience Recall**
- `MemoryBridge.recall_peer_experiences()` exists but is never called during decision-making. Wire it into the coordinator's recall cascade or the agent's `_decide()`.

**3.2 Enable Default Configuration Flags**
- `semantic_search_enabled`, `replay_enabled`, `consolidation_enabled`, `generative_memory_enabled`, `transfer_enabled` are all `False` by default. Either change defaults or ensure `main.py` bootstrap sets them to `True`.

**3.3 Temporal Sequence Replay**
- Current consolidation replays random batches. Add an option to replay episodes in temporal order (forward or reverse), matching hippocampal replay dynamics.

**3.4 Endocrine-Gated Recall**
- High adrenaline should bias recall toward threat memories. High dopamine should bias toward opportunity/reward memories. Cortisol should suppress memory consolidation (biological accuracy).

### Priority 4 (Low -- Future Architecture)

**4.1 Learned Embeddings for Episodic Search**
- Replace hand-crafted `truncate_pad` encoding in `SemanticRetriever.encode_experience()` with a learned embedding (like NEC). The reward signal should shape what "similar" means.

**4.2 Prospective Memory**
- Allow agents to schedule future recall events: "At step N, query for X." This enables planning and intention maintenance.

**4.3 Metamemory Integration**
- Use the meta-memory store (mae_meta collection) to calibrate recall confidence. If the system knows its narrative memory is sparse, it should rely more on ancestral memory, and vice versa.

---

## 6. Sources

### Neuroscience
- [Hippocampal ripples initiate cortical dimensionality expansion for memory retrieval (2025)](https://www.biorxiv.org/content/10.1101/2025.04.22.649929v2.full)
- [Enduring Role for Hippocampal Pattern Completion (2024)](https://www.jneurosci.org/content/44/18/e1740232024)
- [Holistic Recollection via Pattern Completion Involves CA3 (2019)](https://www.jneurosci.org/content/39/41/8100)
- [Reconsolidation and the Dynamic Nature of Memory](https://pmc.ncbi.nlm.nih.gov/articles/PMC4588064/)
- [Human Memory Reconsolidation Explained Using Temporal Context Model](https://pmc.ncbi.nlm.nih.gov/articles/PMC3432313/)
- [Mechanisms of Memory Updating: State Dependency vs Reconsolidation](https://pmc.ncbi.nlm.nih.gov/articles/PMC8740636/)
- [Machine Memory Intelligence: Inspired by Human Memory Mechanisms (2025)](https://www.engineering.org.cn/engi/EN/10.1016/j.eng.2025.01.012)

### AI / ML State of Art
- [Retrieval-Augmented Generation: Comprehensive Survey (2025)](https://arxiv.org/html/2506.00054v1)
- [Neural Episodic Control (Pritzel et al. 2017)](https://proceedings.mlr.press/v70/pritzel17a.html)
- [Temporally Extended Successor Feature NEC (2024)](https://www.nature.com/articles/s41598-024-65687-w)
- [AI-Native Memory and Context-Aware AI Agents (2025)](https://ajithp.com/2025/06/30/ai-native-memory-persistent-agents-second-me/)
- [How Vector Databases Power Agentic AI Memory](https://www.getmonetizely.com/articles/how-do-vector-databases-power-agentic-ais-memory-and-knowledge-systems)
- [Beyond RAG: Context Engineering and Semantic Layers (2025)](https://towardsdatascience.com/beyond-rag/)

### Mae Internal
- Mathematical Identity: `C:\Users\baenb\projects\mae-core\data\MAES-MATHEMATICAL-IDENTITY.md`
- PatternCortex: `C:\Users\baenb\projects\mae-core\mae_core\patterns\pattern_cortex.py`
- MemoryBridge: `C:\Users\baenb\projects\mae-core\mae_core\memory\memory_bridge.py`
- EpisodicMemory mixin: `C:\Users\baenb\projects\mae-core\mae_core\agents\mixins\episodic_memory.py`
- SemanticRetriever: `C:\Users\baenb\projects\mae-core\mae_core\memory\semantic_retriever.py`
- MycelialAgent._decide(): `C:\Users\baenb\projects\mae-core\mae_core\agents\mycelial_agent.py`
- DeepMemoryStore: `C:\Users\baenb\projects\mae-core\mae_core\memory\deep_memory.py`
- PatternConsolidator: `C:\Users\baenb\projects\mae-core\mae_core\patterns\pattern_consolidator.py`
- DecisionRouter: `C:\Users\baenb\projects\mae-core\mae_core\cognition\decision_router.py`
- TransferLearning mixin: `C:\Users\baenb\projects\mae-core\mae_core\agents\mixins\transfer_learning.py`
- WorkingMemory: `C:\Users\baenb\projects\mae-core\mae_core\memory\working_memory.py`
- GenerativeReplay: `C:\Users\baenb\projects\mae-core\mae_core\memory\generative_replay.py`
- MemoryConsolidator: `C:\Users\baenb\projects\mae-core\mae_core\memory\memory_consolidator.py`
- MemoryCoordinator: `C:\Users\baenb\projects\mae-core\mae_core\memory\coordinator.py`
- KnowledgeBase: `C:\Users\baenb\projects\mae-core\mae_core\learning\knowledge_base.py`