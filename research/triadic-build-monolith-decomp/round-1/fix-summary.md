# Round 1 Fix Phase Summary

## Fixes Applied

### 1. `inject_nutrient` phantom method → `nutrient_flow.inject_resources()` (HIGH)
- **File:** `mae_core/emergent/auto_healer.py`
- **Bug:** `_inject_nutrients()` called `self._substrate.inject_nutrient(agent_id, 1.0)` — a method that does not exist on `MycelialSubstrate`. Silently swallowed by try/except, starvation recovery never actually happened.
- **Fix:** Replaced with `nutrient_flow.inject_resources(str(node_id), 0.5)` matching mae-core exactly.
- **Flagged by:** All three reviewers.

### 2. Stale hardcoded `41` in somatic map log (MEDIUM)
- **File:** `mae_core/bootstrap/wiring.py`
- **Bug:** `logger.info("... %d systems ...", 41)` was stale after adding 3 abstract names.
- **Fix:** Replaced `41` with `len(ctx.somatic_map.get_all_systems())` for dynamic count.
- **Flagged by:** Crucible, confirmed by Forge.

### 3. Duplicate `import random as _rng` inside `_invoke_prefrontal` (LOW)
- **File:** `mae_core/cognition/decision_router.py`
- **Bug:** Two duplicate inline imports, one inside a conditional try block (latent NameError if fallback path taken without WorldModel).
- **Fix:** Single import at function top, both inline copies removed.
- **Flagged by:** Crucible.

## Deferred Findings (logged in midge-decisions.md)

### VDN hash(str(action)) instability across restarts (HIGH, inherited)
- `hash()` is randomized per Python process (PYTHONHASHSEED). Q-table action mappings corrupt after restart. Same bug exists in mae-core. Fix upstream first, then port.

### EventBus injection ordering (HIGH → NO FIX NEEDED)
- Investigation confirmed no mixin serializes `_event_bus`. Overwrite vector does not exist. Both MIDGE and mae-core have identical ordering. Latent risk only.

### `agent.shared` registry wrong source system (MEDIUM, inherited)
- Registry names `gnn_communicator` as publisher but actual publisher is `lifecycle_decision._act_communicate()`. Mae-core has same inaccuracy. Not harmful, just confusing for audit tooling.

### `store_ancestral_pattern` blocking call (MEDIUM, pre-existing)
- Mae-core backgrounds both `update_meta_memory` and `store_ancestral_pattern`. Anvil only ported the first. Out of scope for this round — log for future work.

## Test Results
- Awaiting full suite run (3,057 tests expected).
