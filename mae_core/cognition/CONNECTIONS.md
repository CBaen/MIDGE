# Cognition — Connection Map

> Part of Mae's connection map. Index: [mae_core/CONNECTIONS.md](../CONNECTIONS.md)

**Status definitions:** WIRED (systems call each other), BUILT (code exists, not wired), STUB (interface exists via `Any`), PLANNED (neither exists yet).

---

## WorldModel (world_model.py) — Imagination

**Provides:**
- `step(state, action)` -> Prediction (next_state, reward, uncertainty)
- `rollout(state, policy, horizon)` -> Full trajectory
- `predict(state, action)` -> next_state (convenience)
- `predict_reward(state, action)` -> float (convenience)
- `get_uncertainty(state, action)` -> ensemble disagreement
- `train_step(states, actions, next_states, rewards)` -> loss

**Consumed by:**

| Consumer | Method Used | Value Provided | Status |
|----------|-----------|----------------|--------|
| CollectiveDreamPlanner | `rollout()` | Expert dreams for consensus voting | WIRED |
| ValidatedImaginationPlanner | `predict()`, `predict_reward()` | Step-by-step validated planning | WIRED |
| OctopusAgent | `predict()` | Central prediction when arm confidence is low | WIRED |
| AdvancedFeaturesMixin | `reward_model()` | Agent-level world model predictions | WIRED |
| WorldlinePlanner | `step()` | State projection for worldline points | BUILT (stub via `Any`) |
| CausalReasoningEngine | (future) | Interventional predictions for causal learning | PLANNED |
| DreamEnvironment | `step()` | Imagined training environment | PLANNED |

**Requires:**
- Training data from agent experiences (Memory -> WorldModel training loop)
- Optional: Custom transition_fn and reward_fn (domain-specific)

**Growth:**
- WorldModel ensemble disagreement feeds Curiosity drive (novel = high uncertainty)
- WorldModel accuracy feeds ValidatedImagination accuracy tracking
- Causal engine validates WorldModel predictions against interventions
- Endocrine modulation: cortisol increases WorldModel uncertainty threshold

---

## DecisionRouter (decision_router.py) — Three-Tier Brain

**Provides:**
- `route_decision(stimulus, context)` -> RouterDecision (tier, action, confidence, timing)
- Automatic habit formation from repeated prefrontal decisions
- `executive_override()` -> Force deliberation

**Consumed by:**

| Consumer | How It's Used | Value | Status |
|----------|-------------|-------|--------|
| OctopusAgent | `route_decision()` | Three-tier arm cognition | WIRED |
| OctopusColony | Propagated to spawned octopuses | Colony-wide decision routing | WIRED |
| MycelialAgent | (future) | Agent-level decision cascading | PLANNED |

**Requires:**
- Reflex patterns (registered at init or by external systems)
- Optional: Custom prefrontal_fn for domain-specific reasoning
- Optional: Learning system feedback for habit refinement

**Growth:**
- Endocrine system modulates tier thresholds (cortisol raises reflex sensitivity)
- CircadianRhythm affects prefrontal capacity (tired = more reflex, less deliberation)
- CausalEngine provides prefrontal with causal reasoning for better deliberation
- WorldlinePlanner `get_temporal_context()` provides planning context for prefrontal

---

## CausalReasoningEngine (causal_reasoning.py) — Understanding Why

**Provides:**
- `observe_correlation()` / `observe_intervention()` -> Causal evidence
- `query_causation(A, B)` -> Is A causing B? Path? Strength?
- `generate_counterfactual()` -> "What if X hadn't happened?"
- `identify_confounders()` -> Hidden common causes
- `infer_causes(effect)` -> Root cause analysis

**Consumed by:**

| Consumer | How It's Used | Value | Status |
|----------|-------------|-------|--------|
| TemporalMemory | `observe_correlation()` in causal link discovery | Feeds temporal causation data | BUILT (stub via `Any`) |
| AutoHealer | `query_causation()` for root cause analysis | Phase 2 ASSESS uses causal path | BUILT (stub via `Any`) |
| WorldlinePlanner | `causal_engine` param for action-outcome reasoning | Causal context for planning | BUILT (stub via `Any`) |
| WorldModel | (future) | Validates model predictions against causal knowledge | PLANNED |
| DecisionRouter | (future) | Prefrontal tier uses causal reasoning | PLANNED |

**Requires:**
- Observation data from agent interactions
- Intervention results from controlled experiments
- Optional: Domain knowledge as prior causal links

**Growth:**
- Memory system stores causal discoveries in semantic memory
- Causal links persist across sessions (long-term knowledge)
- Counterfactual reasoning enables "what-if" planning in CollectiveDream

---

## CollectiveDreamPlanner (collective_dream.py) — Swarm Imagination

**Provides:**
- `collective_plan(state, horizon)` -> Consensus-validated trajectory
- Expert dreamer selection by expertise score
- Expertise-weighted voting (5x for experts, 2.5x for specialists)
- Low consensus triggers morphogenesis callback

**Consumed by:**

| Consumer | How It's Used | Value | Status |
|----------|-------------|-------|--------|
| MycelialAgent | (future) | Agents collectively plan before acting | PLANNED |
| OctopusColony | (future) | Colony-level strategic planning | PLANNED |

**Requires:**
- WorldModel (REQUIRED — imagination engine)
- Registered DreamAgent instances (agents who can dream)
- Optional: morphogenesis_callback for specialist creation

**Growth:**
- ValidatedImagination tracks dream accuracy per dreamer
- Accurate dreamers get higher expertise, more dream influence
- Low consensus automatically triggers morphogenesis (new specialists)
- Endocrine modulation: dopamine boosts dream creativity (exploration)

---

## ValidatedImagination (validated_imagination.py) — Learning From Prediction Error

**Provides:**
- `record_imagination()` -> Record prediction for later validation
- `validate_with_consensus()` -> Compare prediction against reality
- `get_imagination_accuracy()` -> Per-agent per-domain accuracy
- `get_top_imaginers()` -> Rank agents by prediction quality

**Consumed by:**

| Consumer | How It's Used | Value | Status |
|----------|-------------|-------|--------|
| ValidatedImaginationPlanner | Historical accuracy for step validation | WIRED |
| CollectiveDreamPlanner | (future) Expert selection via accuracy | PLANNED |
| GamificationMixin | (future) Accuracy-based achievement tracking | PLANNED |
| TransferLearning | (future) Transfer knowledge from accurate predictors | PLANNED |

**Requires:**
- Agent predictions (from WorldModel)
- Actual outcomes (from environment/consensus)
- Optional: expertise_callback to feed into agent skill system

**Growth:**
- Prediction accuracy feeds directly into agent expertise scores
- Overconfidence detection prevents bad actors from dominating consensus
- Calibration tracking distinguishes "lucky" from "skilled" predictors

---

---

## Tier 2 Persistence (serialize/restore)

WorldModel implements `serialize()` and `restore(data)` for persisting learned state across sessions:

| Subsystem | What Is Persisted |
|-----------|-------------------|
| WorldModel | Ensemble weights, training history |

Data flows to/from `data/midge/subsystems/agents/{agent_id}/` (per-agent WorldModel instances). Pickle is used for ensemble weights and numpy arrays. Missing files on restore cause the WorldModel to start fresh (graceful degradation).

---

## Related Modules

- [network/CONNECTIONS.md](../network/CONNECTIONS.md) -- OctopusAgent consumes WorldModel and DecisionRouter
- [planning/CONNECTIONS.md](../planning/CONNECTIONS.md) -- WorldlinePlanner consumes WorldModel and CausalEngine
- [emergent/CONNECTIONS.md](../emergent/CONNECTIONS.md) -- AutoHealer uses CausalEngine for root cause analysis
