# Fix Round — Ecosystem Activation

**Date:** 2026-03-08
**Fixer:** Orchestrator (direct fixes — all changes < 10 lines each, targeted)

## Reviewer 2 Findings — All 10 Addressed

### Fix 1 (CRITICAL): `shared_world_model` → `world_model`
**File:** `mae_core/bootstrap/market_systems.py` line 456
**Change:** `getattr(ctx, "shared_world_model", None)` → `getattr(ctx, "world_model", None)`
**Why:** ctx carries `world_model` (line 342), not `shared_world_model`. Colony was silently receiving None.

### Fix 2 (CRITICAL): Race window — EventBus callback before `inject_market_handlers`
**Files:** `market_systems.py`, `market_task_handlers.py`, `market_hooks.py`
**Change:** Pre-initialize `_developing_situations` and `_situations_lock` on colony immediately after construction in market_systems.py. `inject_market_handlers` now uses `if not hasattr` to reuse existing attrs. Removed the `hasattr` guard in the callback since attrs are now guaranteed.
**Why:** Bootstrap step 7 registers the callback, step 9 runs inject_market_handlers. Events between steps 7-9 would hit missing attrs.

### Fix 3 (HIGH): Post-bootstrap spawn gap
**Files:** `market_task_handlers.py`, `market_hooks.py`
**Change:** Added `patch_new_arm()` function. `inject_market_handlers` stores handler refs on `colony._handler_refs`. Bootstrap subscribes to `octopus.spawn` channel and patches new arms on spawn.
**Why:** Arms spawned by auto-scaling after bootstrap wouldn't get the dispatch executor.

### Fix 4 (HIGH): Silent monitoring failure
**File:** `market_hooks.py`
**Change:** Split `inject_market_handlers` + `colony.start_monitoring()` into separate try/except blocks with distinct log messages.
**Why:** If handler injection raised, monitoring never started — and there was no way to distinguish "colony failed to construct" from "colony constructed but dormant."

### Fix 5 (HIGH): Holon orphan — octopus_colony not in fractal hierarchy
**File:** `market_registration.py`
**Change:** Added `"octopus_colony"` to the `extras` list in `_register_market_fractal` (reparented under `market-intelligence-system`).
**Why:** Colony was registered as a holon but never placed in the K3 fractal tree.

### Fix 6 (MEDIUM): Dict mutation race in coordination cycle
**File:** `market_hooks.py`
**Change:** `colony.octopuses.items()` → `list(colony.octopuses.items())`
**Why:** Auto-scaling can mutate the dict while the step hook iterates it.

### Fix 7 (MEDIUM): Ticker key mismatch
**Files:** `convergence_alerter.py`, `market_hooks.py`
**Change:** Partial emission now includes `"symbol": getattr(s, "symbol", "")` in each signal dict. Callback extracts from `sig.get("symbol", "")` first, then falls back to metadata. Removed incorrect `"ticker"` top-level key (global convergence doesn't have a single ticker).
**Why:** Signals have `.symbol` at the top level, not in metadata. The callback was looking in the wrong place.

### Fix 8 (MEDIUM): Stale hardcoded connection count
**File:** `market.py`
**Change:** `103 connections` → `106 connections` in log line and docstrings.
**Why:** Group 34 adds 3 connections.

### Fix 9 (LOW): Unbounded `_developing_situations`
**File:** `market_hooks.py`
**Change:** Added `len(colony._developing_situations) < 200` guard to the callback.
**Why:** If situation_check handler never fires (zero octopuses), the dict grows unbounded.

### Fix 10: Test mock compatibility
**File:** `tests/test_market_task_handlers.py`
**Change:** `MagicMock()` → `MagicMock(spec=[])` in `_make_colony` so `hasattr` returns False for unset attributes.
**Why:** The new `if not hasattr` guard in `inject_market_handlers` was fooled by MagicMock's auto-attribute behavior.

## Test Results After Fixes

- `test_market_signal_translator.py`: 15/15 passed
- `test_market_task_handlers.py`: 5/5 passed
- `test_convergence_alerter_cascade.py` + `test_convergence_domain_windows.py`: 23/23 passed
- `test_integration.py`: running (full bootstrap, ~45 min)
- `test_octopus_bootstrap.py`: running (requires full bootstrap)
