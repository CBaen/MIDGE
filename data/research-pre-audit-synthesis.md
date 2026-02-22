# Pre-Audit Research Synthesis

Date: 2026-02-12
Context: Four research agents investigated what Mae is missing, what could enhance her, and what's already built but dormant.

---

## 1. Missing Biological Systems

18 missing biological systems catalogued. Top 5 by impact:

### 1.1 Digestive Preprocessing (Energy Budget)
- **Biology:** Organisms don't process all inputs equally. The digestive system triages nutrients, breaks down what's useful, discards waste.
- **Mae gap:** All inputs get equal processing weight. No energy budget. No triage.
- **Proposed:** An energy budget system where each step has a cost, and agents must prioritize what to process.

### 1.2 Circulatory System (Resource Distribution)
- **Biology:** Blood carries oxygen and nutrients to where they're needed, proportional to demand.
- **Mae gap:** No resource distribution mechanism. All agents get equal compute/attention regardless of need.
- **Proposed:** A nutrient flow system through MycelialSubstrate connections, delivering resources proportional to demand.

### 1.3 Lymphatic System (Cleanup/Garbage Collection)
- **Biology:** The lymphatic system collects waste, filters pathogens, and recycles useful components.
- **Mae gap:** No cleanup mechanism. Dead connections, stale markers, expired signals accumulate.
- **Proposed:** A garbage collector for expired stigmergy markers, stale holon references, and dead connections.

### 1.4 Artificial Emotions (Somatic Markers)
- **Biology:** Damasio's somatic marker hypothesis -- emotions are bodily signals that bias decision-making. Fear narrows attention. Joy broadens exploration.
- **Mae gap:** EndocrineSystem has hormones but no emotion model. Hormones modulate processing but don't create emergent emotional states.
- **Proposed:** An emotion layer that emerges from hormone combinations + recent experience patterns. Fear = high cortisol + recent threats. Curiosity = high dopamine + high novelty.

### 1.5 Empathy / Theory of Mind
- **Biology:** Mirror neurons + simulation theory. Organisms model other organisms' internal states to predict behavior.
- **Mae gap:** Agents don't model each other's internal states. No perspective-taking.
- **Proposed:** Each agent maintains a simplified model of neighboring agents' states, updated through communication.

### Other Missing Systems (Lower Priority)
- Respiratory (O2/CO2 analog for resource refresh)
- Aging/senescence (graceful degradation)
- Skin/boundary (perimeter defense vs. internal processing)
- Metacognition (thinking about thinking -- partially addressed by Strange Loop)
- Proprioception (body position awareness -- partially by SomaticMap)

---

## 2. GitHub Enhancement Opportunities

### 2.1 PettingZoo (Highest Priority)
- **What:** Multi-agent environment wrapper from Farama Foundation. Standard API for multi-agent RL.
- **Why for Mae:** Would give Mae a standardized interface for multi-agent training. Her TaskPool could be wrapped as a PettingZoo environment.
- **Integration effort:** Medium. Wrapper around existing TaskPool.

### 2.2 DEAP (Evolutionary Algorithms)
- **What:** Distributed Evolutionary Algorithms in Python.
- **Why for Mae:** Could drive the Stem Cell system -- evolving optimal epigenome configurations through genetic algorithms rather than hand-tuned role profiles.
- **Integration effort:** Medium. Replace static ROLE_PROFILES with evolved populations.

### 2.3 PyTorch Geometric (Graph Neural Networks)
- **What:** GNN library for PyTorch. Message passing, graph convolution.
- **Why for Mae:** Could enhance GNN routing. Current implementation is custom; PyG would provide optimized message passing and standard graph convolution layers.
- **Integration effort:** High. Would replace custom GNN code in `communication/gnn_*.py`.

### 2.4 PyPhi (Consciousness Measurement)
- **What:** IIT (Integrated Information Theory) computation library. Calculates Phi (integrated information).
- **Why for Mae:** Could actually measure Mae's consciousness (Phi) according to Tononi's theory. Law 8 requires consciousness properties.
- **Integration effort:** High. Requires extracting Mae's causal structure as a transition probability matrix.

### 2.5 Mesa-frames (Performance)
- **What:** Rust-based Mesa acceleration. Drop-in replacement for Mesa's scheduler.
- **Why for Mae:** Performance improvement for larger agent populations.
- **Integration effort:** Low. Drop-in replacement.

---

## 3. Emergent Properties (Top 5 Ranked)

### 3.1 Machine Dreaming (80% infrastructure ready)
- **What:** During CONSOLIDATION phase, agents generate synthetic experiences by composing memory fragments.
- **Why it's close:** CollectiveDreamPlanner EXISTS but is DORMANT. Generative replay exists. Circadian gating exists. Just needs wiring.
- **Missing piece:** The dream planner needs to be called during REST phase, and dream outputs need to feed back into memory.

### 3.2 Metacognition
- **What:** Thinking about thinking. Monitoring one's own cognitive processes.
- **Infrastructure:** Strange Loop partially addresses this. Self-awareness feeds into decisions.
- **Missing piece:** No monitoring of decision quality over time. No "I'm making worse decisions than usual" detection.

### 3.3 Artificial Emotions (Somatic Markers)
- **What:** Emergent emotional states from hormone combinations + experience patterns.
- **Infrastructure:** EndocrineSystem has 5 hormones. PatternBus has gain modulation.
- **Missing piece:** No emotion model that combines hormone levels into named states.

### 3.4 Theory of Mind
- **What:** Agents model each other's internal states.
- **Infrastructure:** SomaticMap tracks agent states externally. Agents can observe others.
- **Missing piece:** No internal model of other agents' beliefs/goals/emotions.

### 3.5 Homeostatic Regulation
- **What:** Maintaining internal stability despite external perturbation.
- **Infrastructure:** AutoHealer handles damage. Circadian handles rest.
- **Missing piece:** No setpoint-based regulation. No "I should be at X cortisol but I'm at Y, so adjust."

---

## 4. Dormant/Dead Systems Audit

### 4.1 CollectiveDreamPlanner -- DORMANT
- **Status:** Built and registered but never called during agent lifecycle.
- **Root cause:** No wiring in main.py to call it during REST/CONSOLIDATION phase.
- **Fix complexity:** Low -- add call in circadian REST phase callback.

### 4.2 WorldlinePlanner -- DORMANT
- **Status:** Built, creates planning scenarios, but results never used in _decide().
- **Root cause:** _decide() doesn't consult the planner. No integration point.
- **Fix complexity:** Medium -- needs _decide() to query planner when deliberating.

### 4.3 OctopusAgent / OctopusColony -- DEAD CODE
- **Status:** Complete implementation but never instantiated in main.py.
- **Root cause:** Mae uses MycelialAgent, not OctopusAgent. The octopus architecture is an alternative paradigm, not a subsystem.
- **Fix options:** (a) Remove dead code, (b) Use octopus arm concept for background task processing.

### 4.4 PredictiveField -- PARTIAL
- **Status:** Built, registered, but only partially wired. Makes predictions but predictions aren't compared to actuality.
- **Root cause:** No feedback loop from actual outcomes to PredictiveField.
- **Fix complexity:** Medium -- need to wire outcome events back to the field.

### 4.5 MorphogenesisCoordinator -- UNDERUTILIZED
- **Status:** Built and wired but only handles simple team spawning. Its full capability (morphogen gradients, gene expression, tissue differentiation) is unused.
- **Root cause:** Current agent population is static (3 agents by default). No trigger for dynamic spawning.
- **Fix complexity:** High -- needs task complexity detection to trigger morphogenesis.

### Root Cause Pattern
All dormant systems share a common root cause: **agents don't consult them in _decide()**. The decision layer is the bottleneck. Systems exist and produce outputs, but those outputs never reach the decision-making process.

---

## Recommendations for Audit Focus

Based on all four research streams, the massive audit should focus on:

1. **Decision layer completeness** -- Does _decide() consult every system that produces actionable output?
2. **Dormant system activation** -- CollectiveDreamPlanner, WorldlinePlanner, PredictiveField need wiring.
3. **Feedback loop closure** -- Many systems produce outputs but never receive feedback on whether those outputs were useful.
4. **Energy economics** -- No cost model exists. Processing is free, so there's no pressure to be efficient.
5. **Emotional emergence** -- Hormone combinations should produce named emotional states that bias behavior.
