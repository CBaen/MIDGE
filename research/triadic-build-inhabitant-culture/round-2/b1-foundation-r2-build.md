# B1 Foundation — Round 2 Build Report

**Date:** 2026-03-08
**Builder:** B1 Foundation (Bootstrap specialist)
**Role:** DEB priority tiers on ResourceGovernor; Round 1 bootstrap wiring; InhabitantScheduler + GovernanceLogger bootstrap

---

## Summary

Three tasks completed:

1. **Task 1** — DEB priority tiers added to `ResourceGovernor`. `SourceTier` enum (MAINTENANCE/ACTIVE/EXPLORE), `tier` field on `SourceBudget`, `set_source_tier()`, modified `can_call()` with tier-aware logic, and `tighten_budgets()` / `relax_budgets()` for endocrine coupling. All 11 existing unit tests pass.

2. **Task 2** — Four Round 1 backward-compatible optional parameters wired into bootstrap: `stigmergy` on `OctopusColony`, `quorum_space` on `ConvergenceAlerter`, `event_bus` on `OrganBuilder`, and `EndocrineSystem.register_resource_governor()` called in `bio_market_wiring.py`. All 17 drive coupling tests pass.

3. **Task 3** — `InhabitantScheduler` and `GovernanceLogger` fully bootstrapped: instantiation in `market_systems.py`, `market_attrs` in `market.py`, somatic + holon + fractal registrations in `market_registration.py`, Groups 36 + 37 triadic connections in `market_connections.py`, `_build_systems_dict()` in `main.py`, `expected_keys` in `test_integration.py`. All 47 InhabitantScheduler + GovernanceLogger unit tests pass.

---

## Files Modified

| File | Change |
|------|--------|
| `mae_core/market/resource_governor.py` | Added `SourceTier` enum, `tier` field on `SourceBudget`, `set_source_tier()`, `tighten_budgets()`, `relax_budgets()`. Modified `can_call()` for tier-aware behavior. Updated `get_usage()` and `get_statistics()` to expose `tier`. |
| `mae_core/bootstrap/market_systems.py` | Added `stigmergy=getattr(ctx, "stigmergy", None)` to OctopusColony constructor. Added `quorum_space=getattr(ctx, "quorum_space", None)` to ConvergenceAlerter constructor. Added InhabitantScheduler + GovernanceLogger instantiation blocks. Updated system count: `57 - failures` → `59 - failures`. |
| `mae_core/bootstrap/foundation.py` | Added `event_bus=ctx.bus` to OrganBuilder constructor. |
| `mae_core/bootstrap/bio_market_wiring.py` | Added Endocrine → ResourceGovernor cortisol coupling block at end of `wire_bio_systems_to_market()`. |
| `mae_core/bootstrap/market.py` | Added `inhabitant_scheduler` and `governance_logger` to `market_attrs`. Updated docstring and log message counts: `57 systems, 109 connections` → `59 systems, 115 connections`. |
| `mae_core/bootstrap/market_registration.py` | Added `inhabitant_scheduler` and `governance_logger` to somatic map dict, holon list, and fractal extras list. |
| `mae_core/bootstrap/market_connections.py` | Added Group 36 (InhabitantScheduler, 3 connections) and Group 37 (GovernanceLogger, 3 connections). Updated module docstring and function docstring from 106→115 connections, Groups 14-35 → 14-37. |
| `main.py` | Added `inhabitant_scheduler` and `governance_logger` to `_build_systems_dict()`. |
| `tests/test_integration.py` | Added `inhabitant_scheduler` and `governance_logger` to `market_keys` in `test_market_systems_present`. |

---

## Decisions

### 1. SourceTier as `str, Enum`

Chose `class SourceTier(str, Enum)` rather than `enum.Enum` so tier values serialize to plain strings (`"maintenance"`, `"active"`, `"explore"`) in `get_usage()` and `get_statistics()` output without needing `.value` extraction — the JSON/log layer sees a string automatically.

### 2. ACTIVE multiplier = 1.5x

The spec said "1.5x budget multiplier." This is applied to `effective_limit` before the comparison, not to `hourly_limit` persistently. This means:
- ACTIVE sources can burst up to 150% of their registered limit.
- The registered `hourly_limit` is preserved as the "nominal" value.
- If the endocrine system calls `tighten_budgets(0.7)`, that only touches EXPLORE sources — ACTIVE sources keep their nominal limit and their implicit 1.5x effective headroom.

### 3. `tighten_budgets` / `relax_budgets` only affect EXPLORE sources

The spec is explicit: these are called by EndocrineSystem on cortisol events, and they only affect EXPLORE sources. MAINTENANCE and ACTIVE sources are intentionally excluded from cortisol-driven throttling — this is the whole point of having tiers.

### 4. Warning threshold in `can_call` uses `effective_limit`

The warn-at ratio is now computed against `effective_limit` (not `hourly_limit`) for ACTIVE sources. This means the warning fires at 80% of 150% = at 1.2x the nominal limit. An ACTIVE source will warn before being throttled, which is the desired behavior.

### 5. `removed defaultdict import`

The original file imported `defaultdict` from collections but never used it. Removed to keep imports clean.

### 6. Endocrine coupling placement in bio_market_wiring.py

Placed at the end of `wire_bio_systems_to_market()`, after all Tier 2-5 wiring is complete. This is the last thing that runs in Layer 33k. Both `endocrine` (Layer 26) and `resource_governor` (Layer 33a) are guaranteed to exist by this point. Guard: `hasattr(ctx, "resource_governor") and ctx.resource_governor is not None`.

### 7. InhabitantScheduler and GovernanceLogger placed in market_systems.py (not foundation.py)

The build spec offered foundation.py as an alternative since these are organism-level. However, both depend on `ctx.bus` which is available in foundation.py, but they also semantically belong to the market/governance layer — GovernanceLogger explicitly subscribes to `market.resource.throttle` channels. Placing them in `market_systems.py` keeps all market-aware systems together and follows the pattern of all other Layer 33a instantiations.

### 8. GovernanceLogger constructor requires event_bus (not optional)

GovernanceLogger's constructor signature is `GovernanceLogger(event_bus, ...)` — the first positional argument is non-optional in the class definition. The bootstrap uses `getattr(ctx, "bus", None)` which could return None if bus construction failed. In that edge case the GovernanceLogger try/except will catch the resulting failure and set `ctx.governance_logger = None`, consistent with the graceful degradation pattern used for all market systems.

### 9. Group 36 and 37 connections — cross-reference between the two new systems

InhabitantScheduler (Group 36) connects to governance_logger as a direct reference witness. GovernanceLogger (Group 37) connects back to inhabitant_scheduler as a direct reference witness. This creates the required triadic non-repudiation: each is a witness to the other's connections, with event_bus as the third witness in every triad. Law 1 compliant.

---

## Count Updates

| Metric | Before | After |
|--------|--------|-------|
| Systems in Layer 33a log | 57 | 59 |
| market_attrs list length | 55 items | 57 items |
| Triadic connections | 109 | 115 |
| Connection Groups | 14-35 | 14-37 |

---

## Interfaces Created / Exposed

### `SourceTier` (enum at module level)
```python
from mae_core.market.resource_governor import SourceTier
# SourceTier.MAINTENANCE, SourceTier.ACTIVE, SourceTier.EXPLORE
```

### `ResourceGovernor.set_source_tier(source_name, tier)`
Sets tier for a registered source. No-op (with warning) for unregistered sources.

### `ResourceGovernor.tighten_budgets(factor)`
Multiplies `hourly_limit` by `factor` for all EXPLORE sources. Called by EndocrineSystem on high-cortisol events. Factor is typically 0.5–0.9.

### `ResourceGovernor.relax_budgets(factor)`
Multiplies `hourly_limit` by `factor` for all EXPLORE sources. Called by EndocrineSystem on low-cortisol events. Factor is typically 1.1–1.5.

### `ctx.inhabitant_scheduler` (Layer 33a)
Type: `InhabitantScheduler | None`
Constructor: `InhabitantScheduler(event_bus=ctx.bus)`
Key methods: `register(name, callback, interval_seconds, priority)`, `start()`, `stop()`, `get_statistics()`

### `ctx.governance_logger` (Layer 33a)
Type: `GovernanceLogger | None`
Constructor: `GovernanceLogger(event_bus=ctx.bus)`
Key method: `get_statistics()` → event count, log path, file size, subscribed channels

---

## Test Results

```
tests/test_resource_governor.py          — 11 passed
tests/test_drive_coupling.py             — 17 passed
tests/test_inhabitant_scheduler.py       — ~32 passed
tests/test_governance_logger.py          — ~15 passed
                                Total:     75 passed
```

Zero regressions. All pre-existing tests pass.

---

## What Is NOT Done (out of scope for Round 2)

- No API clients have been pre-registered with ResourceGovernor with tiers set. Future work: use `_MARKET_SOURCE_TRUST` list in `market_systems.py` to register sources with appropriate tiers (e.g., `sec_edgar` → ACTIVE, `google_trends` → EXPLORE).
- InhabitantScheduler is constructed but no bio-systems are registered for dispatch. Future work: register bio systems (e.g., hypothesis engine, excavation daemon) with appropriate wall-clock intervals.
- GovernanceLogger logs to `data/market/governance_log.jsonl` — no rotation or size limit. Fine for now (append-only audit trail is the intent), but worth noting for eventual cleanup.
