## Forge Round 1 Build Report

### Item 1: VDN epsilon-greedy action selection

- **Files changed:** `C:\Users\baenb\projects\MIDGE\mae_core\agents\lifecycle_decision.py`
- **Lines added:** 20 lines inserted between causal reasoning block and WorldModel call (after line 249, before "Consult world model" comment)
- **What was ported:** The VDN block queries the Q-table for all five actions (`explore`, `exploit`, `communicate`, `rest`, `api_call`), finds the max Q-value, and applies decaying epsilon-greedy (20% random at step 0, floors at 5% around step 500 via `max(0.05, 0.20 - step * 0.0003)`). Only fires after step 20 (warmup) and only when max Q > 0. Block is guarded by `try/except` so it degrades silently.
- **Position rationale:** Mae-core places VDN after causal reasoning but before WorldModel. This is correct — VDN is a learned fast path (like basal ganglia), while WorldModel is slow deliberative simulation (prefrontal). The position was preserved exactly.
- **Market-specific code preserved:** Yes. MIDGE's `_decide()` differs from mae-core's in: (1) missing the Fibonacci-13 cadence gate on memory_bridge — this was already absent in MIDGE before this port and was NOT added (out of scope, not part of this item); (2) `_route_with_advisory` has MIDGE-specific market context enrichment block — untouched. The VDN block was inserted into the shared infrastructure section only.

---

### Item 2: EventBus injection for oracle calls

- **Files changed:** `C:\Users\baenb\projects\MIDGE\mae_core\bootstrap\agents.py`
- **Lines added:** 2 lines (comment + assignment) inserted immediately after `agent.curiosity_drive = ctx.curiosity` in the agent creation loop
- **What was ported:** `agent._event_bus = ctx.bus` — injects the shared EventBus into each agent so `_act_api_call()` and `_act_communicate()` can publish oracle requests and collaboration signals via `bus.publish(...)`.
- **Why this was missing:** MIDGE's `_act_api_call()` uses `inject_external_task` (task pool pattern) rather than direct EventBus publish, so it technically works without `_event_bus`. However, `_act_communicate()` in the base lifecycle does use `bus.publish("agent.shared", ...)`. Without the injection, that publish silently no-ops. The fix closes that gap.
- **Market-specific code preserved:** Yes. MIDGE `agents.py` is identical to mae-core except for the logger name (`midge.bootstrap` vs `mae.bootstrap`) and this one missing line. The market wiring (StemCell roles, market refs) lives in `bootstrap/market.py`, not here. No market code was present in this file to preserve.

---

### Item 3: WorldModel/decision_router tie-breaking

- **Files changed:** `C:\Users\baenb\projects\MIDGE\mae_core\cognition\decision_router.py`
- **Lines added/modified:** Two spots in `_invoke_prefrontal()`:
  1. WorldModel simulation loop: replaced `best_action = available_actions[0]` initialization + `best_action = action` assignment with a `best_actions` list that accumulates all tied-best actions, then `_rng.choice(best_actions)` at the end. Added `import random as _rng` before the loop.
  2. Default fallback: replaced `action = available_actions[0]` with `import random as _rng` + `action = _rng.choice(available_actions)`. Updated the return reason string from `"Default selection (first available action)"` to `"Default selection (random from available)"`.
- **Why this matters:** When WorldModel predicts equal rewards for multiple actions (common early in training when weights are near-zero), always picking index 0 breaks exploration symmetry — the same action wins every tie forever. Random tie-breaking distributes exploration uniformly.
- **Market-specific code preserved:** Yes. MIDGE's `decision_router.py` is byte-for-byte identical to mae-core except for these two spots. No market divergences existed in this file.

---

### Item 4: Microbiome feed-before-step

- **Files changed:** `C:\Users\baenb\projects\MIDGE\mae_core\bootstrap\organs.py`
- **Lines added:** 22 lines replacing 1 line (the bare `ctx.model.add_step_hook(lambda s=ctx.microbiome: s.step(...))`)
- **What was ported:** The full feed-then-step pattern from mae-core:
  - `_micro` closure variable capturing `ctx.microbiome`
  - `_micro_types` list: `["pattern", "anomaly", "weak_signal", "noisy", "data"]`
  - `_feed_microbiome(channel, data, input_type)` closure function that calls `_micro.process_input()`
  - Two EventBus callbacks: `signal.PREDICTION_ERROR` → anomaly feed, `external.response_received` → data feed
  - `_microbiome_step_feed(step)` function that feeds all 5 specialization types each step
  - Step hook for `_microbiome_step_feed` (runs BEFORE step)
  - Step hook for `microbiome.step()` (runs AFTER feed)
- **Why order matters:** `Microbiome.step()` reads `_process_counts` to determine if the organism is idle, then resets those counts at the end of step. If the step hook runs first, `_process_counts` is always zero and the organism always appears idle. The feed-before-step pattern ensures counts are populated before the idle check runs.
- **Market-specific code preserved:** Yes. MIDGE's `organs.py` is identical to mae-core except for the logger name (`midge.bootstrap`) and the comment in Layer 29c ("Metacognition created at Layer 19" in mae-core vs "Layer 27" in MIDGE — this is a documentation divergence only, not changed). The microbiome section was the only functional divergence being addressed.

---

### Notes

1. **MIDGE `_decide()` memory_bridge cadence gate absent:** Mae-core's memory_bridge block has a `self.step_count % 13 == 0` cadence guard. MIDGE's version does not. This was NOT ported as it was outside this item's scope. Reviewers should decide whether to port it separately — without it, every eligible step fires an Ollama + Qdrant round-trip, which can be expensive.

2. **VDN `_action_dim` attribute dependency:** The VDN block uses `vdn._action_dim`. If `ValueDecompositionEngine` does not expose this attribute, the block will silently except and fall through. This matches mae-core's behavior. No code change needed, but reviewers should verify the attribute exists on VDN.

3. **Microbiome EventBus callback signatures:** The callbacks use `lambda ch, d: _feed_microbiome(ch, d, "anomaly")`. This assumes the EventBus calls registered callbacks with `(channel, data)` signature. If the EventBus uses a different calling convention, these will silently fail (the try/except in `_feed_microbiome` catches any errors). This matches the mae-core pattern exactly.

4. **Decision router `import random as _rng`:** The `import` is inside the function body (twice: once in the WorldModel block, once in the default fallback). This is a local import per call but Python caches module imports so there is no performance concern. This matches the mae-core pattern.
