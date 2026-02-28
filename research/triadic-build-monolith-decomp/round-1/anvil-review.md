## Review of Forge's Round 1 Work

My lens: do the connections and learning paths correctly reference what was built?

---

### Item 1: VDN epsilon-greedy (lifecycle_decision.py)

**Integration: PASS**

The VDN block at lines 251-276 of MIDGE `lifecycle_decision.py` is correctly positioned — after causal reasoning, before world model. It reads `vdn._action_dim`, calls `vdn.compute_local_value(state_vec, action_int)`, and returns before `use_world_model()` is called. The `_vdn_engine` attribute is injected into agents at the end of `bootstrap/agents.py` (line 268: `agent._vdn_engine = ctx.vdn_engines.get(agent.unique_id)`). That wiring was already present before this port. The port connects to the right engine.

**Bug: One real issue — VDN does not handle ties**

Mae-core's VDN block (lines 267-292 of mae-core `lifecycle_decision.py`) is identical to MIDGE's. There is no tie-breaking in the VDN selection: `best_idx = q_values.index(max_q)` always returns the index of the first maximum. This is the same bug that Item 3 fixes in the WorldModel path, but it was NOT fixed here. This means VDN has symmetric tie-breaking on the Q-values (always picks first action alphabetically when Q-values are equal). This is consistent with mae-core — it is not a regression — but it is worth flagging since Item 3 specifically identified this as an exploration symmetry problem.

**Spec Compliance: PASS**

The MIDGE port matches mae-core exactly at lines 251-276. The position, logic, epsilon schedule (`max(0.05, 0.20 - step * 0.0003)`), and `try/except` wrapper are identical.

**Missing port noted in build report (not a defect, a known gap):**

Mae-core `lifecycle_decision.py` line 220 has `self.step_count % 13 == 0` cadence on the memory_bridge block. MIDGE line 203 fires on every eligible step. This is flagged by Forge in Note 1. Without the cadence gate, every eligible step triggers Ollama + Qdrant round-trips. This is a real performance concern in MIDGE's context where the step loop is already managed for latency (parallel senses, background validation). Should be ported as a separate item.

**Missing block not in MIDGE (unlisted in build report):**

Mae-core `_decide()` has a goal-directed action bias block at lines 191-204 that does NOT exist in MIDGE:

```python
# --- Goal-directed action bias (investigation goals → api_call) ---
goal_ctx = getattr(self, "_goal_context", None)
if goal_ctx is not None and goal_ctx.get("has_goal", False):
    goal_id = goal_ctx.get("goal_id", "")
    goal_priority = goal_ctx.get("goal_priority", 0.0)
    config = getattr(self, "agent_config", {})
    if (
        goal_id.startswith("investigate_")
        and goal_priority > 0.4
        and config.get("api_call_enabled", False)
    ):
        return "api_call"
```

This block sits between the CollectiveDreamPlanner block and the semantic memory search. MIDGE is missing it entirely. The `_goal_context` attribute is presumably set by WorldFeed or GoalManager — if MIDGE uses GoalManager (it does, injected at Layer 30), this means investigation goals cannot escalate to `api_call` through the goal pathway in MIDGE. This is a spec divergence that Forge did not flag. Low immediate impact since MIDGE's oracle pathway is mostly disabled, but it is a correctness gap.

---

### Item 2: EventBus injection (bootstrap/agents.py)

**Integration: PASS**

Lines 115-116 of MIDGE `bootstrap/agents.py`:
```python
# FIX: Inject EventBus so _act_api_call() can publish oracle requests
agent._event_bus = ctx.bus
```

This is correct. `ctx.bus` is the shared EventBus created in foundation.py and available at this layer. The agent's `_act_communicate()` at line 560 of MIDGE `lifecycle_decision.py` reads `bus = getattr(self, "_event_bus", None)` and publishes to `"agent.shared"`. Without this injection, that publish silently skips. Fix is correct and the wiring is live.

**Spec Compliance: PASS**

MIDGE `agents.py` is now byte-for-byte identical to mae-core `agents.py` except for the logger name on line 13 (`midge.bootstrap` vs `mae.bootstrap`). Confirmed by direct comparison.

**Note:** The build report correctly identifies that MIDGE's `_act_api_call()` uses `inject_external_task` (task pool pattern) rather than EventBus publish. Mae-core's `_act_api_call()` uses `bus.publish("external.submit_request", ...)`. These are divergent implementations — MIDGE's is more complex but works without `_event_bus` for the oracle path specifically. The EventBus injection only materially affects `_act_communicate()`. The divergence in `_act_api_call()` is a known MIDGE design choice, not introduced here.

---

### Item 3: WorldModel/decision_router tie-breaking (decision_router.py)

**Integration: PASS**

The fix is in `_invoke_prefrontal()` of `DecisionRouter`. Both the WorldModel simulation loop (lines 463-485) and the default fallback (lines 489-492) now use `_rng.choice()` instead of index-0 selection. This is called by any agent whose DecisionRouter reaches prefrontal tier. All agents have a `decision_router` injected at Layer 12 in `bootstrap/agents.py` (line 97: `decision_router=agent_decision_router`). The connection is live.

**Spec Compliance: PASS**

MIDGE `decision_router.py` is now identical to mae-core `decision_router.py`. Confirmed by direct line-by-line comparison. Both files match at lines 463-494.

**No bugs found.**

---

### Item 4: Microbiome feed-before-step (bootstrap/organs.py)

**Integration: PASS**

MIDGE `bootstrap/organs.py` lines 88-116 implement the full feed-before-step pattern correctly:
- `_micro` closure captures `ctx.microbiome` (line 90)
- `_micro_types` list defined (line 91)
- `_feed_microbiome()` closure defined (lines 93-100)
- EventBus callbacks registered for `signal.PREDICTION_ERROR` and `external.response_received` (lines 103-104)
- `_microbiome_step_feed()` step hook added BEFORE the microbiome step hook (line 114)
- Microbiome step hook added AFTER (line 116)

Hook ordering is critical: Python list append means hooks run in registration order. The feed hook at line 114 is registered before the step hook at line 116, guaranteeing feed-then-step execution. This is correct.

**Spec Compliance: PASS**

MIDGE `organs.py` matches mae-core `organs.py` exactly for the microbiome section. Confirmed by comparison at lines 88-116 in both files. The only difference is the logger name (`midge.bootstrap` vs `mae.bootstrap`) and the comment at line 441 (MIDGE says "Layer 27", mae-core says "Layer 19" — documentation only, not a functional difference).

**No bugs found.**

---

### Integration Issues (Forge overall)

None that would cause failures. The missing `_goal_context` block and missing memory_bridge cadence gate are gaps to port separately, not regressions from these changes.

### Bugs / Logic Errors (Forge overall)

1. **VDN tie-breaking not applied**: Item 3 correctly fixes tie-breaking in the WorldModel path of `_invoke_prefrontal()`, but the VDN block in `_decide()` still uses `q_values.index(max_q)` which returns the first maximum. Both MIDGE and mae-core have this gap in the VDN path. Consistency with spec — but the spec has the same bug.

2. **Missing `_goal_context` block**: MIDGE `_decide()` is missing the goal-directed api_call escalation block present in mae-core `_decide()` at lines 191-204. Not introduced by this port, but also not flagged by Forge.

---

## Review of Crucible's Round 1 Work

My lens: do the connections and learning paths correctly reference what was built?

---

### Item 9: Auto-healer starvation nodes fix (auto_healer.py)

**Integration: PASS**

**Fix 1 (`_on_starvation`):**

MIDGE `auto_healer.py` lines 505-528 correctly implement the nodes-list iteration. The fix reads `message.get("nodes", [])`, guards on empty list, and files one `FailureReport` per node. Each report uses `affected_agents=[str(node_id)]`, which is what `_execute_healing()` → finally block reads at lines 564-565: `for agent_id in record.failure.affected_agents: self._healing_cooldowns[agent_id] = self._step_count`. The node IDs now reach the cooldown stamping correctly. The connection between `_on_starvation` and `_execute_healing`'s cooldown stamp is live.

**Fix 2 (Cooldown system):**

`_healing_cooldowns: dict[str, int] = {}` at line 159 of MIDGE `auto_healer.py` is present and correct. The step() scan loop at lines 238-241 checks `if self._step_count - last_healed < 50: continue` before filing proactive reports. The `_execute_healing` finally block at lines 564-565 stamps cooldowns. All three coordinated changes are present.

The self-heal exclusion guard `if system_id == "auto_healer": continue` at line 236 prevents the healer from filing against itself during the somatic scan — necessary because the healer reports its own health to somatic map at line 292. Without this guard, it would try to heal itself through the proactive scan path AND the meta-healing path simultaneously.

**Spec Compliance: PASS**

Both MIDGE and mae-core `auto_healer.py` are identical at all the changed sections (lines 157-160, 232-241, 559-565, 505-528). Confirmed by line-by-line comparison. The MIDGE file matches the mae-core spec exactly.

**Adversarial notes from build report — evaluated:**

- `inject_nutrient` phantom method: confirmed pre-existing. `_register_defaults()` at line 167 calls `self._register_defaults()`, which Crucible correctly identifies as calling `self._substrate.inject_nutrient(agent_id, 1.0)` on an undefined method. This will fail silently in Phase 3 for STARVATION/RESOURCE_EXHAUSTION. Pre-existing debt, not introduced here.

- Race condition in failure_id: valid observation. If two starvation events arrive in the same second with the same node, the second is deduplicated by `_active_healings`. Acceptable.

- Cooldown uses step count not wall time: correct observation. `_step_count` is incremented in `step()` which is called once per model tick. In MIDGE's parallel senses architecture, ticks are not wall-clock seconds. 50-step cooldown could be longer or shorter than expected in real time. Not a bug, but worth knowing.

**No bugs found in the port itself.**

---

### Item 10: Phi forced measurement (growth_tracker.py)

**Integration: PASS**

MIDGE `growth_tracker.py` (`mae_core/backbone/growth_tracker.py`) lines 165-178:

```python
def _read_phi(self, systems: dict) -> float:
    meter = systems.get("integration_meter")
    if meter:
        try:
            if getattr(meter, "_last_report", None) is None:
                if hasattr(meter, "_compute_and_publish"):
                    meter._compute_and_publish()
            stats = meter.get_statistics()
            lm = stats.get("last_measurement") or {}
            return float(lm.get("organism_mean_phi", 0.0) or 0.0)
        except Exception:
            pass
    return 0.0
```

This is correct. `integration_meter` is passed through the `systems` dict in `main.py`. The `_last_report` guard forces a measurement if the cadence hasn't triggered one yet. `organism_mean_phi` is the correct key from `integration_meter.get_statistics()["last_measurement"]`. Both bugs (Bug A: missing guard, Bug B: wrong key `current_phi` → `organism_mean_phi`) are fixed.

**Spec Compliance: PASS**

MIDGE `backbone/growth_tracker.py` matches mae-core `backbone/growth_tracker.py` at `_read_phi()` (lines 169-182 in mae-core, lines 165-178 in MIDGE). The implementations are identical. The only intentional MIDGE divergences are preserved:
- Logger name: `midge.growth` (line 19)
- Default output dir: `data/midge` (line 25)
- Qdrant collection name: `midge_meta` / `midge_narrative` (lines 141, 161)
- File header text: `# MIDGE Growth Tracker` (line 106)
- Deep store: synchronous call (lines 141-160) vs mae-core's daemon thread (lines 159-165)

The synchronous vs async divergence for Qdrant writes is correctly preserved and documented. Crucible's report correctly flags this as a known performance risk (blocking at run-end if Qdrant is slow).

**One observation — `hasattr` guard is redundant but harmless:**

At line 171-172:
```python
if hasattr(meter, "_compute_and_publish"):
    meter._compute_and_publish()
```

This `hasattr` guard is correct — `_compute_and_publish` is a private method that could be renamed in a future IntegrationMeter refactor. If it disappears, the guard silently skips the force-measurement and falls through to `get_statistics()` which would return `None` for `last_measurement`, yielding `0.0`. Graceful degradation. No issue.

**No bugs found.**

---

### Integration Issues (Crucible overall)

None. Both fixes are correctly wired. The `_on_starvation` node IDs now flow through to `_execute_healing`'s cooldown stamp. The `_read_phi` fix now correctly reads `integration_meter.get_statistics()["last_measurement"]["organism_mean_phi"]`.

### Bugs / Logic Errors (Crucible overall)

None introduced by this port.

**Pre-existing issues documented by Crucible (not introduced, worth tracking):**

1. `inject_nutrient` phantom method in `_register_defaults()` — STARVATION/RESOURCE_EXHAUSTION healing silently fails in Phase 3.
2. `_compute_and_publish` is a private method call across module boundaries — low risk, guarded by `hasattr`.

---

## Summary

Both Forge and Crucible's Round 1 work is correct. The ports match the mae-core spec. No regressions introduced.

**Items for Round 1 Fix cycle:**

1. (Forge, low priority) Port memory_bridge Fibonacci-13 cadence gate (`self.step_count % 13 == 0`) from mae-core `lifecycle_decision.py` line 220 into MIDGE. Prevents Ollama + Qdrant round-trips on every eligible step.

2. (Forge, medium priority) Port goal-directed api_call escalation block from mae-core `lifecycle_decision.py` lines 191-204 into MIDGE `_decide()`. MIDGE is missing this block entirely.

3. (Pre-existing, low priority) `inject_nutrient` phantom method in `auto_healer._register_defaults()`. Not introduced here, but should be tracked. The method does not exist on `MycelialSubstrate`.
