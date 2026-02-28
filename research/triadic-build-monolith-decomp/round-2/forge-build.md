# Forge Build Report — Round 2: market.py Orchestrator

**Date:** 2026-02-28
**Task:** Rewrite `mae_core/bootstrap/market.py` from 1,722-line monolith to thin orchestrator
**Result:** COMPLETE — 123 lines, import verified OK

---

## What Was Done

Replaced the 1,722-line `market.py` with a 123-line orchestrator that:
1. Imports all functions from the 5 pre-built sub-modules
2. Calls them in the exact original order inside `bootstrap_market(ctx)`
3. Preserves the final active-system count + holon count log block verbatim
4. Adds inline comments explaining why call order is load-bearing

---

## Call Order Preserved (Critical)

The original `bootstrap_market()` called functions in this sequence:

| Step | Function | Module | Dependency constraint |
|------|----------|--------|-----------------------|
| 1 | `_instantiate_market_systems` | market_systems.py | Must be first — all other steps need ctx attributes |
| 2 | `_register_market_somatic` | market_registration.py | Must precede connections |
| 3 | `_register_market_holons` | market_registration.py | Requires objects on ctx |
| 4 | `_register_market_fractal` | market_registration.py | Requires holons registered |
| 5 | `_register_market_connections` | market_connections.py | Requires somatic + holons |
| 6 | `_register_market_stem_roles` | market_registration.py | Independent — order advisory |
| 7 | `_register_market_eventbus` | market_hooks.py | Requires bus + hypothesis_engine |
| 8 | `_register_market_step_hooks` | market_hooks.py | **Writes ctx._cached_alerts** |
| 9 | `_wire_sensing_hook` | market_hooks.py | **Reads ctx._cached_alerts** — must follow step 8 |
| 10 | `_differentiate_market_agents` | market_agents.py | Requires _market_advisory (from step 9) |

The critical `_cached_alerts` handshake between steps 8 and 9 is documented in both the orchestrator docstring and `market_hooks.py`.

---

## Sub-Module Exports Verified

| File | Functions Exported | Lines |
|------|--------------------|-------|
| `market_systems.py` | `_instantiate_market_systems` | 433 |
| `market_registration.py` | `_register_market_somatic`, `_register_market_holons`, `_register_market_fractal`, `_register_market_stem_roles` | 195 |
| `market_connections.py` | `_register_market_connections` | 271 |
| `market_hooks.py` | `_register_market_eventbus`, `_register_market_step_hooks`, `_write_convergence_heartbeat`, `_wire_sensing_hook` | 551 |
| `market_agents.py` | `_differentiate_market_agents`, `_register_market_reflexes` | 243 |
| **market.py (new)** | `bootstrap_market` | **123** |

---

## Verification

```
python -c "from mae_core.bootstrap.market import bootstrap_market; print('OK')"
OK
```

---

## Notes for Reviewers

- The `import json` and `from pathlib import Path` that were in the old market.py header are now gone from the orchestrator — those imports live in `market_hooks.py` where they're actually used.
- The `_write_convergence_heartbeat` function is exported by `market_hooks.py` but not imported into the orchestrator (it's called internally by `_register_market_step_hooks`, not by `bootstrap_market`). This is correct — it's an implementation detail of the hooks module.
- Line count is 123 rather than the target 60-100. The overage is entirely the holon-count block (lines 80-104) which is verbatim from the original and cannot be compressed without changing behavior.
- No sub-module files were modified. The task was write-only to `market.py`.
