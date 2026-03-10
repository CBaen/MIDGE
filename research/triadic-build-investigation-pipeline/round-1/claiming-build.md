# Situation Claiming Build Report
**Round:** 1
**Date:** 2026-03-09

## What Was Built

Agent-level situation claiming: stem cell roles now have domain affinity, and the investigation dispatcher routes developing situations to the octopus best suited to investigate them.

## Files Changed

| File | Change |
|------|--------|
| `mae_core/network/market_task_handlers.py` | Added `ROLE_DOMAIN_AFFINITY` dict and `select_preferred_role()` function |
| `mae_core/network/octopus_colony.py` | Added role-affinity routing tier in `submit_task()` (between stigmergy routing and workload fallback) |
| `mae_core/bootstrap/market_hooks.py` | Investigation dispatcher now calls `select_preferred_role()` and injects `preferred_role` into task data |
| `tests/test_situation_claiming.py` | NEW: 27 tests covering all routing paths |

## Design Decisions

**Soft preference, not hard requirement.** If no octopus has the preferred role, routing falls back to least-loaded. The step loop never blocks.

**Priority order in `select_preferred_role()`:**
1. `causal_predictions` non-empty → `HYPOTHESIS_EXPLORER` (follows causal threads)
2. `historical_win_rate > 0.6` → `HYPOTHESIS_VALIDATOR` (validates proven patterns)
3. `insider` or `institutional` in domains → `SEC_WATCHER`
4. `government` or `contracts` in domains → `CONTRACT_TRACKER`
5. 3+ domains seen → `MARKET_ANALYST` (high-complexity synthesis)
6. Highest overlap count by domain set intersection
7. `None` (no preference) if no role clears minimum overlap threshold

**`_genome_role` attribute on OctopusAgent.** The routing check uses `getattr(o, "_genome_role", None)`. This is opt-in — octopuses without the attribute still participate as generic workers. Any bootstrap code that wants to assign a role to a specific octopus simply sets `octopus._genome_role = "SEC_WATCHER"`.

## How Role Assignment Would Work in Practice

The colony currently spawns all octopuses as `GENERAL` with no `_genome_role`. For the claiming to have practical effect, bootstrap or the colony should assign roles on spawn. The infrastructure is now in place — the next step would be:

```python
# In market_systems.py or a future market_bootstrap step:
for idx, oct in enumerate(colony.octopuses.values()):
    role = ["SEC_WATCHER", "CONTRACT_TRACKER", "MARKET_ANALYST"][idx % 3]
    oct._genome_role = role
```

This was left out intentionally — it's a separate concern (how many of each role to spawn) from the routing logic itself.

## Test Results

```
27 passed in 0.49s
```

All 27 new tests pass. Full suite: 970 passed, 1 pre-existing flaky failure in `test_congress_gov_client.py::test_request_fails_without_key` (env var leak from other tests — unrelated to this change, passes in isolation).

## Law 5 Compliance

This is a pure Law 5 implementation: specialization via configuration, not different code. The same `investigate_partial` handler runs on every arm. The routing intelligence lives in the colony's `submit_task()` — the agent genome determines who gets the task, not what they do with it.
