> Generated from 10-agent audit conducted 2026-02-11. ~50 sub-agents. Sources: biology papers, GitHub, research papers, full codebase trace.

# Mae Audit: All Bugs Found

Every bug discovered across all 10 audit reports. Organized by severity, with file:line references and proposed fixes.

---

## Critical Bugs (System Behavior Broken)

### BUG-01: `_act()` is a two-line stub returning constant 0.0
- **Report:** ACT
- **File:** `mae_core/agents/base_agent.py`, lines 92-95
- **What's broken:** `_act()` stores the action as `self.last_action` and returns `0.0`. It is never overridden by `MycelialAgent` or any mixin. No action changes any environmental state. All reward is always zero.
- **Impact:** The causal chain is broken at the most critical junction. Learning, prediction, and everything downstream operates on zero signal.
- **Fix:** Override `_act()` in `MycelialAgent` with a triadic execution cycle (plan/execute/verify) that interacts with an environment and returns actual reward.

### BUG-02: `_learn_from_batch()` does not update any weights
- **Report:** LEARN
- **File:** `mae_core/agents/mixins/episodic_memory.py`, line 147
- **What's broken:** `_learn_from_batch()` computes TD errors as `np.array([getattr(exp, "reward", 0.0) for exp in batch])` and a loss, but does not update any policy, value network, or model parameters. Only SumTree priorities are updated.
- **Impact:** Memory replay is infrastructure without function. The system remembers but does not learn.
- **Fix:** Implement actual parameter updates in `_learn_from_batch()`. At minimum, update a policy/value network based on the TD error signal.

### BUG-03: MemoryConsolidator calls nonexistent agent methods
- **Report:** LEARN
- **File:** `mae_core/memory/memory_consolidator.py`, lines 101-156
- **What's broken:** `consolidate()` calls `agent.get_learning_rate()` and `agent.set_learning_rate()`. Neither `BaseAgent` nor `MycelialAgent` implement these methods. Will raise `AttributeError` if invoked.
- **Impact:** The consolidation pathway is dead code. Memory consolidation never actually executes.
- **Fix:** Add `get_learning_rate()` and `set_learning_rate()` methods to `BaseAgent`, or redesign the consolidator interface.

### BUG-04: Semantic recall return type mismatch in `_decide()`
- **Report:** RECALL
- **File:** `mae_core/agents/mycelial_agent.py`, lines 326-331
- **What's broken:** `search_similar_experiences()` returns a `SemanticQuery` dataclass (or `None`), but `_decide()` treats it as a list and calls `max(past, key=lambda e: getattr(e, "reward", 0.0))`. This iterates over the `SemanticQuery` dataclass fields, not its `.experiences` list.
- **Impact:** The semantic memory recall path in `_decide()` is broken. It would iterate over dataclass fields, not experiences.
- **Fix:** Access `.experiences` attribute of the returned `SemanticQuery` before calling `max()`.

### BUG-05: `_route_with_advisory()` passes `available_actions=None`
- **Report:** DECIDE
- **File:** `mae_core/agents/mycelial_agent.py`, line 384
- **What's broken:** Advisory routing always passes `available_actions=None` to `router.route_decision()`. The prefrontal tier inside the router requires `available_actions` to be non-None to evaluate alternatives with the WorldModel.
- **Impact:** Prefrontal deliberation always falls to the default dict `{"type": "deliberate"}`. WorldModel simulation is never used during advisory-routed decisions.
- **Fix:** Compute the agent's available action set and pass it through `_route_with_advisory()`.

---

## High-Severity Bugs (Feature Silently Broken)

### BUG-06: Endocrine-Router wiring is broken (silent failure)
- **Report:** DECIDE
- **File:** `mae_core/coordination/endocrine_system.py`, lines 456-464
- **File:** `mae_core/cognition/decision_router.py`
- **What's broken:** EndocrineSystem calls `dr.set_reflex_bias(level)` when adrenaline is released. `DecisionRouter` has no `set_reflex_bias()` method. The `hasattr` check silently falls through. `register_decision_router()` is effectively dead code.
- **Impact:** Endocrine system's influence on decision-making is completely non-functional.
- **Fix:** Add `set_reflex_bias()` to DecisionRouter. Store the bias value and use it to modulate reflex matching threshold.

### BUG-07: Adrenaline override re-check is identical to first check
- **Report:** DECIDE
- **File:** `mae_core/cognition/decision_router.py`, lines 194-210
- **What's broken:** Comment says "high adrenaline - try reflex again with lower match bar" but the code calls `self._check_reflex(stimulus)` with the exact same logic. If the first reflex check did not match, the second never will either.
- **Impact:** The entire adrenaline-driven reflex re-check block (lines 194-210) is dead code in practice.
- **Fix:** Implement fuzzier/broader matching for the adrenaline re-check (e.g., partial string matching, lower confidence threshold).

### BUG-08: PatternBus signal mutation bug
- **Report:** SENSE
- **File:** `mae_core/patterns/pattern_bus.py`, lines 214-216
- **What's broken:** `_detect_correlations()` mutates shared `PatternSignal` objects in-place: `sig.form = PatternForm.CORRELATED; sig.confidence = boosted_conf`. Any other consumer holding a reference to these signals sees mutated data.
- **Impact:** Signal form and confidence are permanently altered for downstream consumers. Data integrity violation.
- **Fix:** Create copies before mutation, or create new synthetic signals representing the correlation (as cross-domain correlation already does).

### BUG-09: Z-score self-inclusion in reward surprise detector
- **Report:** SENSE
- **File:** `mae_core/patterns/pattern_sense.py`, lines 201-213
- **What's broken:** Mean and std are computed over ALL rewards including the current one. The z-score of the current value against a distribution that contains that value is systematically biased toward zero.
- **Impact:** Reward surprises are underdetected. The surprise detector is less sensitive than intended.
- **Fix:** Compute mean and std over `rewards[:-1]` (excluding current), then compute z-score of `rewards[-1]`.

### BUG-10: PatternDistiller injected but never called
- **Report:** CONSOLIDATE
- **File:** `mae_core/patterns/pattern_consolidator.py`, line 51
- **What's broken:** `self._distiller` is set at `main.py:950` but its methods (`distill()`, `detect_behavioral_patterns()`, `detect_state_patterns()`, `merge_with_existing()`) are never invoked anywhere.
- **Impact:** Behavioral and state pattern extraction is dead code. Significant unused capability.
- **Fix:** Add a distillation pass in `PatternConsolidator.consolidate()` that calls `self._distiller.distill()`.

### BUG-11: Agent `parent_id` initialization is incorrect
- **Report:** SELF-AWARENESS
- **File:** `main.py`, lines 306-335
- **What's broken:** Agent is created without `parent_id` in the constructor call. `_init_holon()` sets `self._holon_parent_id = None`. The registry entry with `parent_id="colony"` is created AFTER. While `_effective_parent_id()` queries the registry and works at runtime, serialization via `_serialize_holon()` would save `parent_id=None`.
- **Impact:** If an agent is serialized before registry lookup, its parent relationship is lost.
- **Fix:** Pass `parent_id="colony"` to `_init_holon()`, or call `holon_registry.register()` BEFORE `_init_holon()`.

---

## Medium-Severity Bugs (Suboptimal Behavior)

### BUG-12: AutoHealer has no `step()` method
- **Report:** HEAL
- **File:** `mae_core/emergent/auto_healer.py`
- **What's broken:** In `main.py` line 466, `auto_healer.step` is guarded by `hasattr` and falls back to `lambda: None`. AutoHealer is entirely reactive -- it never proactively scans for problems.
- **Impact:** Healing is only triggered by external events. No proactive health monitoring.
- **Fix:** Add a `step()` method that periodically queries SomaticMap for unhealthy systems and checks its own health.

### BUG-13: AwarenessPulse anomalies have no subscribers
- **Report:** HEAL
- **File:** `mae_core/backbone/holon_protocol.py`, line 494
- **What's broken:** AwarenessPulse publishes to `holon.anomaly_detected` and `holon.awareness_pulse`. Nothing subscribes to either channel.
- **Impact:** Orphaned systems and health gradient drift are detected, logged, and forgotten. No corrective action.
- **Fix:** Subscribe AutoHealer to `holon.anomaly_detected`. Subscribe a monitoring hook to `holon.awareness_pulse`.

### BUG-14: Multiple healing EventBus channels have no subscribers
- **Report:** HEAL
- **What's broken:** These channels publish but nothing listens: `healing.started`, `healing.complete`, `healing.failed`, `haven.intervention`, `holon.awareness_pulse`, `holon.anomaly_detected`, `somatic.modification_rolled_back`, `defense.threat_detected`.
- **Impact:** Published healing events are lost. No downstream reactions to healing outcomes.
- **Fix:** Subscribe appropriate handlers to each channel.

### BUG-15: ValidatedImagination EventBus channel has no subscribers
- **Report:** HIDDEN STEPS
- **File:** EventBus channel `cognition.imagination_validated`
- **What's broken:** ValidatedImagination publishes validation results (was_accurate, reward_error) but no system subscribes. Prediction accuracy data is discarded.
- **Impact:** WorldModel never learns from its imagination mistakes. Validation is computational waste.
- **Fix:** Subscribe WorldModel training to this channel.

### BUG-16: CollectiveDream EventBus channel has no subscribers
- **Report:** HIDDEN STEPS
- **File:** EventBus channel `cognition.collective_dream_complete`
- **What's broken:** CollectiveDream publishes completion events. Nobody subscribes. Multi-agent planning results are never consumed.
- **Impact:** Computational waste. Planning happens but results are lost.
- **Fix:** Subscribe the decision system or a planning coordinator.

### BUG-17: CuriosityDrive output never consumed
- **Report:** HIDDEN STEPS
- **What's broken:** CuriosityDrive computes intrinsic curiosity reward but it is never added to the agent's actual reward in `_learn()`.
- **Impact:** Curiosity-driven exploration does not actually drive exploration because intrinsic reward never reaches the agent.
- **Fix:** In `_learn()`, add CuriosityDrive's computed intrinsic reward to the extrinsic reward.

### BUG-18: HEALER stem cell role is cosmetic
- **Report:** HEAL
- **File:** `mae_core/agents/stem_cell.py`, line 157
- **What's broken:** HEALER role profile only adjusts convergence thresholds (satisfaction_threshold=0.95, convergence_threshold=0.001). No actual healing behavior differentiation.
- **Impact:** Agents with HEALER role behave identically to normal agents except for tighter convergence thresholds.
- **Fix:** Give HEALER-role agents actual healing behavior: monitor peer health, trigger AutoHealer reports, assist with recovery coordination.

### BUG-19: Recall configuration flags all default to False
- **Report:** RECALL
- **What's broken:** `semantic_search_enabled`, `replay_enabled`, `consolidation_enabled`, `generative_memory_enabled`, `transfer_enabled` all default to `False`.
- **Impact:** Most recall pathways are gated off by default. Only ancestral recall and habit recall are active.
- **Fix:** Either change defaults or ensure `main.py` bootstrap sets them to `True`.

### BUG-20: Consolidation not gated by circadian phase
- **Report:** CONSOLIDATE
- **What's broken:** PatternConsolidator fires every 89 steps regardless of circadian phase. `CircadianRhythm.should_consolidate_memory()` exists but PatternConsolidator does not check it.
- **Impact:** Consolidation happens during active exploration, not during rest/sleep phases as biology requires.
- **Fix:** Add circadian phase awareness to PatternConsolidator. Consolidate during CONSOLIDATION phase, accumulate during ACTIVE phase.

---

## Summary

| Severity | Count | Examples |
|----------|-------|---------|
| Critical | 5 | `_act()` stub, `_learn_from_batch()` no-op, consolidator interface mismatch, semantic recall type bug, missing available_actions |
| High | 6 | Endocrine-router dead code, signal mutation, z-score bias, distiller dead code, parent_id init bug |
| Medium | 9 | AutoHealer no step(), unsubscribed channels, curiosity reward unused, HEALER role cosmetic, config defaults |
| **Total** | **20** | |
