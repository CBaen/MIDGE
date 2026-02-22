I now have a comprehensive picture of the entire LEARN step. Let me compile the full audit report.

---

# LEARN STEP AUDIT REPORT

## 1. INTERNAL CODE AUDIT: Complete Learning Pathway Trace

### 1.1 The _learn() Entry Point

**File:** `C:\Users\baenb\projects\mae-core\mae_core\agents\mycelial_agent.py` (lines 394-421)

The reward signal enters through `BaseAgent.step()` at line 77:
```
reward = self._act(action)
self._learn(action, reward)
```

`MycelialAgent._learn()` calls `super()._learn(action, reward)` which updates:
- `self.last_reward = reward`
- `self.cumulative_reward += reward`
- `self.reward_history.append(reward)`

Then it performs five learning activities:

| Activity | Trigger Condition | Target System |
|---|---|---|
| Episodic storage | `prev_state AND curr_state AND episodic_memory` | `EpisodicMemoryMixin.store_experience()` |
| Stigmergy deposit | `reward > 0` OR `risk_score > 0.5` | `StigmergyMixin.deposit_success_marker()` / `deposit_danger_marker()` |
| Memory replay | `step_count % 13 == 0` | `EpisodicMemoryMixin.learn_from_memory()` |
| Memory consolidation | `step_count % 89 == 0 AND should_consolidate()` | `EpisodicMemoryMixin.consolidate_memory()` |
| Pattern sensing | `_pattern_sense is not None` | `PatternSense.sense()` |

### 1.2 Experience Storage Pathway

**File:** `C:\Users\baenb\projects\mae-core\mae_core\agents\mixins\episodic_memory.py` (lines 48-101)

`store_experience()` creates an `Experience(state, action, reward, next_state, done, info)` and adds it to:
1. **PrioritizedReplayBuffer** (SumTree-backed, O(log N)) -- always
2. **GenerativeReplayMemory** (VAE compression) -- if `generative_memory_enabled`

Consensus-based priority is optionally computed via `quorum_sensor.get_consensus_priority()` if signal_context is provided. However, in the current `_learn()` call, NO signal_context is ever passed, so consensus priority is never actually used during the standard learn path.

### 1.3 Memory Replay Pathway

**File:** `C:\Users\baenb\projects\mae-core\mae_core\agents\mixins\episodic_memory.py` (lines 103-199)

`learn_from_memory()` has two modes:
1. **Standard replay**: Samples from PrioritizedReplayBuffer, calls `_learn_from_batch()`, updates priorities via TD errors
2. **Generative replay**: Samples from GenerativeReplayMemory (mix of real + VAE-synthetic experiences)

**Critical finding:** `_learn_from_batch()` (line 147) uses reward as the TD error signal:
```python
td_errors = np.array([getattr(exp, "reward", 0.0) for exp in batch])
loss = float(np.mean(np.abs(td_errors) * weights))
```
This is a placeholder. There is no value network, no actual TD learning, no gradient update. The "learning" from replay only updates priorities in the SumTree -- it does not update any policy or model parameters.

### 1.4 Memory Consolidation Pathway

**File:** `C:\Users\baenb\projects\mae-core\mae_core\memory\memory_consolidator.py` (lines 101-156)

`MemoryConsolidator.consolidate()` expects the agent to have `get_learning_rate()`, `set_learning_rate()`, and `learn_from_batch()` methods. It lowers the learning rate by `lr_multiplier` (default 0.5), replays `consolidation_steps` (default 100) batches, then restores the LR.

**Critical finding:** `BaseAgent` and `MycelialAgent` do NOT implement `get_learning_rate()` or `set_learning_rate()`. The consolidator will raise `AttributeError` if actually invoked. The consolidation path is effectively dead code.

### 1.5 Pattern Sensing Pathway

**File:** `C:\Users\baenb\projects\mae-core\mae_core\patterns\pattern_sense.py`

`PatternSense.sense()` detects three patterns per agent per step:
1. **Reward trend** (3+ monotonic steps)
2. **Action repetition** (same action 3+ times in 5)
3. **Reward surprise** (z-score > 2.0)

Signals are stored in `_last_sense_result` and shared via `PatternSharer` in `_communicate()`. This is real triadic learning: agent detects, shares to tissue-level, tissue correlates across agents.

### 1.6 Transfer Learning / MAML Pathway

**File:** `C:\Users\baenb\projects\mae-core\mae_core\agents\mixins\transfer_learning.py`
**File:** `C:\Users\baenb\projects\mae-core\mae_core\learning\maml.py`

Transfer learning and MAML are available but are NOT called from `_learn()`. They are task-level operations called manually via `begin_new_task()`, `store_episode_for_transfer()`, etc. They operate at a different timescale (task-level, not step-level).

MAML's `_outer_loop_update()` (line 239) uses random perturbation instead of actual gradients:
```python
noise = np.random.randn(*self._meta_parameters["weights"].shape) * 0.01
self._meta_parameters["weights"] -= self._config.meta_learning_rate * noise * meta_loss
```
This is an evolution-strategy approximation, not true gradient-based meta-learning.

### 1.7 Adjacent Learning Systems (NOT in _learn() but related)

| System | File | Connection to _learn() |
|---|---|---|
| **CuriosityDrive** | `mae_core/learning/curiosity.py` | Not called from _learn(). Standalone intrinsic motivation. |
| **FederatedLearningEngine** | `mae_core/learning/frl.py` | Not called from _learn(). Peer policy sharing via EventBus. |
| **ValueDecompositionEngine** | `mae_core/learning/vdn.py` | Not called from _learn(). Multi-agent credit assignment. |
| **ImitationLearning** | `mae_core/learning/imitation.py` | Not called from _learn(). Social learning from demonstrations. |
| **HavenRiskCoordinator** | `mae_core/learning/haven.py` | Not called from _learn(). Immune-system risk monitoring. |
| **WorldModel** | `mae_core/cognition/world_model.py` | Used in _decide() not _learn(). Not trained during _learn(). |

---

## 2. MATHEMATICAL IDENTITY COMPLIANCE

### 2.1 Free Energy Principle (FEP) Compliance

The mathematical identity states LEARN must implement "Prediction/error-correction" based on the Free Energy Principle.

**VERDICT: PARTIAL COMPLIANCE - Key gaps identified**

| FEP Requirement | Status | Evidence |
|---|---|---|
| Prediction | PRESENT (partial) | WorldModel predicts next states; PatternSense detects reward trends |
| Prediction error computation | PRESENT (weak) | CuriosityDrive computes prediction error; PatternSense detects surprise via z-score |
| Error-driven learning | ABSENT | No prediction error is used to update any model parameters in _learn(). The reward IS the signal, not a prediction error. |
| Free energy minimization | ABSENT | No variational bound is computed. No generative model is updated from prediction errors. |
| Belief updating | ABSENT | No Bayesian state estimation. State vectors are raw observations, not posterior beliefs. |

**The gap:** Mae has the components for FEP but they are not wired together. The WorldModel makes predictions, CuriosityDrive measures prediction errors, but neither feeds back into the agent's learning update in `_learn()`. The actual learning signal is raw reward, not prediction error. True FEP would require: generate prediction -> compare with observation -> compute free energy -> update generative model -> repeat.

### 2.2 Triadic Structure

The mathematical identity requires LEARN to be triadic.

**VERDICT: PARTIALLY TRIADIC**

The LEARN step has identifiable triadic structure:
1. **Store** (hippocampal encoding) -- `store_experience()`
2. **Replay** (memory recall/rehearsal) -- `learn_from_memory()`
3. **Consolidate** (sleep consolidation) -- `consolidate_memory()`

This maps to biological triads: encoding / retrieval / consolidation. However:

- The triad is not explicitly declared or enforced by the TriadEnforcer
- Pattern sensing adds a second triad: Detect / Share / Correlate
- Transfer learning adds a third: Retrieve / Adapt / Apply
- These triads are not nested or self-similar (not fractal)

### 2.3 Fractal at Every Scale

**VERDICT: PARTIAL**

| Scale | LEARN Implementation | Status |
|---|---|---|
| **Cell** (single agent) | `_learn()` stores experience, replays, consolidates | PRESENT |
| **Tissue** (agent group) | PatternSharer correlates across agents | PRESENT (via pattern sharing) |
| **Organ** (functional module) | FRL shares policies; VDN decomposes credit | PRESENT but disconnected from _learn() |
| **Organism** (whole Mae) | KnowledgeBase stores cross-task knowledge | PRESENT but not triggered by _learn() |

The fractal requirement is met structurally but not processually. Each scale has a learning mechanism, but they are not triggered recursively by the same lifecycle. The HolonProtocol provides `holon_learn()` which delegates to `_learn()`, but there is no evidence that parent holons' `holon_learn()` is triggered by children's learning outcomes.

---

## 3. BIOLOGICAL ACCURACY ASSESSMENT

### 3.1 What Biology Does (Reference Model)

Based on neuroscience research:

1. **Hebbian/STDP plasticity**: "Neurons that fire together wire together." Synaptic weights update based on correlated pre/post activity. Three-factor learning rules add a neuromodulator (dopamine) as a gating signal.

2. **Dopamine RPE**: Dopamine neurons encode reward prediction error (RPE): `delta = actual_reward - predicted_reward`. Positive RPE strengthens connections (LTP); negative RPE weakens them (LTD).

3. **Hippocampal encoding**: Rapid one-shot encoding of episodes in hippocampus. Sharp-wave ripples replay experiences during rest.

4. **Sleep consolidation**: Two-stage model -- hippocampal replay during SWS transfers knowledge to neocortex; REM sleep integrates and generalizes.

5. **Cerebellar learning**: Parallel fiber (context) + climbing fiber (error signal) = supervised learning of motor programs. Prediction error from inferior olive drives plasticity.

6. **Homeostatic plasticity**: Scaling of all synapses to maintain overall activity levels. Prevents runaway excitation or quiescence.

### 3.2 How Mae Maps to Biology

| Biological System | Mae Implementation | Accuracy |
|---|---|---|
| **Hippocampal encoding** | `EpisodicMemory.store()` via PrioritizedReplayBuffer | GOOD. One-shot storage, priority-based. |
| **Hippocampal replay** | `learn_from_memory()` every 13 steps | GOOD concept, WEAK implementation (no actual learning occurs) |
| **Sleep consolidation** | `consolidate_memory()` every 89 steps | GOOD concept, BROKEN implementation (missing interface) |
| **Dopamine RPE** | Reward used directly; CuriosityDrive computes prediction error | ABSENT. No RPE signal. Raw reward is used, not (reward - predicted_reward). |
| **Hebbian/STDP** | None | ABSENT. No correlation-based learning. |
| **Cerebellar learning** | WorldModel predictions exist but are not compared to outcomes | ABSENT from _learn() path |
| **Homeostatic plasticity** | None | ABSENT. No synaptic scaling or activity normalization. |
| **Neuromodulation** | GamificationMixin's exploration bonus loosely maps | VERY WEAK. No actual modulation of learning rate by reward signal. |
| **Working memory** | WorkingMemory with 7+-2 slots, activation decay | GOOD biological accuracy, but not integrated with _learn() |
| **Semantic memory** | SemanticRetriever via FAISS | GOOD. Gist-based retrieval. |
| **Mirror neurons / social learning** | ImitationLearning | PRESENT but disconnected from _learn() |

### 3.3 Biological Gaps (Ranked by Impact)

1. **No reward prediction error** -- The single most important biological learning signal is completely absent. Dopamine RPE is the foundation of biological RL.
2. **No synaptic weight updates** -- `_learn_from_batch()` returns TD errors but does not update any weights/parameters. The system remembers but does not learn.
3. **No Hebbian correlation** -- No mechanism links co-occurring states/actions to strengthen their association.
4. **No homeostatic regulation** -- Nothing prevents runaway reward accumulation or learning rate drift.
5. **No neuromodulatory gating** -- Learning rate is not modulated by uncertainty, novelty, or reward magnitude.

---

## 4. STATE-OF-ART COMPARISON

### 4.1 Experience Replay

Mae implements Prioritized Experience Replay (Schaul et al., 2016) correctly with SumTree-backed O(log N) sampling, importance-sampling weights, and priority annealing. This is a solid implementation.

The Generative Replay (Shin et al., 2017) via VAE compression is a forward-looking addition. Recent work on "dreaming" in spiking networks (2024) validates this approach for biologically plausible learning.

**Gap:** No n-step returns, no Hindsight Experience Replay (HER), no distributed replay.

### 4.2 Meta-Learning

MAML is implemented but with random perturbations instead of true gradients. Recent Meta-DDPG-MAML work (2025) shows 50% faster convergence when MAML is properly integrated with policy gradients.

**Gap:** MAML is not called from the step-level learning loop. It is a task-level operation only.

### 4.3 Multi-Agent Learning

Mae has FRL (federated policy sharing), VDN (value decomposition), and consensus mechanisms. This is a rich multi-agent learning suite. Recent Meta-Enhanced Recurrent MARL (M-RMARL) work combines meta-learning with multi-agent RL for dynamic adaptation.

**Gap:** FRL, VDN, and imitation learning are all disconnected from the per-step `_learn()` cycle.

### 4.4 Active Inference

The Free Energy Principle and Active Inference provide a unified framework where action, perception, and learning all minimize the same quantity (variational free energy). Mae's architecture is well-suited for this but does not implement it. The WorldModel + CuriosityDrive + EpisodicMemory trinity could be rewired into an active inference agent.

---

## 5. UPGRADE RECOMMENDATIONS (Ranked)

### Priority 1 (CRITICAL) -- Fix Broken Paths

**1a. Implement actual learning in `_learn_from_batch()`**
Currently returns reward as TD error but updates nothing. Needs a policy/value network that actually gets updated.

**1b. Fix MemoryConsolidator interface mismatch**
`consolidate()` calls `agent.get_learning_rate()` and `agent.set_learning_rate()` which do not exist on BaseAgent or MycelialAgent. Either add these methods or redesign the consolidator interface.

### Priority 2 (HIGH) -- Wire Disconnected Systems

**2a. Integrate CuriosityDrive into `_learn()`**
Compute intrinsic reward and blend with extrinsic reward. The system exists but is never called from the learning path.

**2b. Wire WorldModel training into `_learn()`**
After each experience, train the WorldModel on the observed transition. This closes the prediction/observation loop.

**2c. Pass signal_context to `store_experience()`**
The consensus priority path exists but is never used because `_learn()` does not pass signal_context. This means quorum-influenced prioritization is dead code.

### Priority 3 (MEDIUM) -- Implement FEP Compliance

**3a. Compute Reward Prediction Error**
Replace raw reward with `RPE = reward - WorldModel.predict_reward(state, action)`. This is the biological dopamine signal and the foundation of FEP-compliant learning.

**3b. Add prediction error to experience priority**
Use |RPE| as the priority signal for PER instead of raw |reward|. Higher surprise -> higher replay priority. This is how the hippocampus works.

**3c. Update generative model from prediction errors**
Train the WorldModel on the prediction error signal during `_learn()`, minimizing variational free energy (or its approximation as prediction error).

### Priority 4 (MEDIUM) -- Enforce Triadic / Fractal Structure

**4a. Declare the learning triad explicitly**
Register LEARN's triad (Encode / Replay / Consolidate) with the TriadEnforcer. This makes the structure auditable.

**4b. Wire holon_learn() to propagate up the hierarchy**
When an agent learns, its parent holon should aggregate child learning signals and perform its own learning at a slower timescale.

**4c. Add Fibonacci-aware learning rates**
The Fibonacci replay/consolidation timing (13/89) is biologically inspired. Extend this to learning rate scheduling: faster learning early, slower as knowledge consolidates.

### Priority 5 (LOWER) -- Biological Enrichment

**5a. Add homeostatic plasticity**
Scale learning rates across agents to maintain population-level reward stability. Prevents reward inflation.

**5b. Add Hebbian trace for state-action correlations**
Maintain eligibility traces that strengthen frequently co-occurring state-action pairs independent of reward.

**5c. Add neuromodulatory gating**
Modulate learning rate by novelty (from CuriosityDrive) and confidence (from PatternSense). High novelty -> high learning rate. High confidence -> low learning rate.

**5d. Integrate ImitationLearning as a learning fallback**
When an agent's own learning stalls (detected by PatternSense as reward plateau), switch to observational learning from high-performing peers.

---

## 6. SUMMARY SCORECARD

| Dimension | Score | Notes |
|---|---|---|
| **Code completeness** | 6/10 | Rich infrastructure but key paths are broken or disconnected |
| **FEP compliance** | 3/10 | Components exist but are not wired into prediction/error-correction loop |
| **Triadic structure** | 5/10 | Encode/Replay/Consolidate triad present but not enforced or declared |
| **Fractal recursion** | 4/10 | Learning exists at multiple scales but does not self-similarly recurse |
| **Biological accuracy** | 5/10 | Good hippocampal analogy, missing RPE/Hebbian/homeostatic |
| **State-of-art alignment** | 6/10 | PER, VAE replay, MAML present. Active inference absent. |
| **Functional correctness** | 4/10 | Replay doesn't actually learn; consolidation interface is broken |

**Overall LEARN Step Health: 4.7/10**

The LEARN step has extensive scaffolding and many biologically-inspired subsystems, but the core learning pathway -- the path from reward signal to actual parameter update -- is incomplete. The system stores experiences and replays them, but the replay does not update any weights. The consolidation mechanism calls methods that do not exist. Several powerful systems (CuriosityDrive, WorldModel training, FRL, VDN, ImitationLearning) are available but never invoked from the per-step learning cycle.

The most impactful single fix would be implementing actual parameter updates in `_learn_from_batch()` and computing reward prediction error from the WorldModel. This would simultaneously improve biological accuracy, FEP compliance, and functional correctness.

---

**Sources:**
- [Dopamine Reward Prediction Error Hypothesis (PNAS)](https://www.pnas.org/doi/10.1073/pnas.1014269108)
- [Three-Factor Learning Rules / Neuromodulated STDP (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4717313/)
- [Hippocampal LTP and LTD in Memory (PMC 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11343234/)
- [VTA Dopamine Projections Trigger LTP and Contextual Learning (Nature Comms 2024)](https://www.nature.com/articles/s41467-024-47481-4)
- [Biologically Plausible Model-Based RL via Dreaming (Nature Sci Reports 2024)](https://www.nature.com/articles/s41598-024-65631-y)
- [Meta-Enhanced Hierarchical Multi-Agent RL (ScienceDirect 2025)](https://www.sciencedirect.com/science/article/abs/pii/S1570870525001222)
- [Free Energy Principle (Wikipedia)](https://en.wikipedia.org/wiki/Free_energy_principle)
- [Active Inference and Learning (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5167251/)
- [FEP for Perception and Action: Deep Learning Perspective (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8871280/)
- [Meta-RL Induces Exploration in Language Agents (arXiv 2025)](https://arxiv.org/html/2512.16848v1)