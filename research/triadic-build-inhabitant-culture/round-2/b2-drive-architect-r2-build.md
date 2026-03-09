# B2 Drive Architect — Round 2 Build Report

**Date:** 2026-03-08
**Builder:** B2 Drive Architect
**Domain:** EndocrineSystem cortisol coupling to ResourceGovernor

---

## Task Completed

### `register_resource_governor()` on EndocrineSystem

**File:** `mae_core/coordination/endocrine_system.py`

Added `register_resource_governor(rg)` as the 10th convenience consumer registration method in the "Convenience Consumer Registration" section, placed immediately after `register_vdn()` — the natural position since it also wires a single hormone (cortisol) to a single consumer.

---

## Pattern Analysis

Read all 9 existing `register_*()` methods before writing a single line. The pattern is completely consistent across all of them:

1. Define a closure `_on_{hormone}(_ht, level)` that calls the target method if it exists (`hasattr` guard), with a `set_hormone_level()` fallback for duck-typed consumers.
2. Call `self.register_consumer(HormoneType.X, "consumer_name", _on_{hormone})`.

The dispatch mechanism lives in `release_hormone()` at lines 217-236: it iterates `self._subscribers[hormone]` (populated by `register_consumer` → `subscribe`) and calls each callback with `(hormone_type, new_level)`. The try/except wrapping is in `release_hormone()` itself — individual callbacks do not need their own try/except. This is why none of the 9 existing closures have try/except.

---

## Implementation Decisions

### Threshold logic in the closure
The task specified two thresholds (0.6 and 0.3) and different semantics for each branch. This logic lives inside the `_on_cortisol` closure rather than in a separate method because:
- All existing closures contain their logic inline
- The closure has direct access to the `rg` reference via closure capture
- Adding a separate method would be inconsistent with the established pattern

### factor formula
- `tighten_budgets(factor)`: factor = cortisol level directly. At cortisol=0.8, factor=0.8, meaning "tighten to 80% of the EXPLORE budget." This is consistent with the task spec.
- `relax_budgets(factor)`: factor = `1.0 + (0.3 - level)`. At cortisol=0.1, factor=1.2, meaning "relax by 20% above baseline." At cortisol=0.29 (just below threshold), factor=1.01 (minimal relaxation). Linear and bounded.

### Neutral zone is a no-op
Cortisol in [0.3, 0.6] produces no call. This is not a bug — it is the intended design. The neutral zone prevents oscillation where every minor cortisol fluctuation triggers budget changes. The `elif` structure ensures the neutral zone is reached by the `else` fallthrough with no action.

### `set_hormone_level` fallback
Included for consistency with all existing register methods. Every existing method has this fallback for systems that implement a generic hormone interface rather than domain-specific methods. ResourceGovernor does not have `set_hormone_level`, but the fallback ensures forward compatibility.

---

## Tests

**File:** `tests/test_drive_coupling.py` (4 new tests, total now 17)

`test_register_resource_governor_high_cortisol` — Releases 0.6 cortisol from baseline 0.2, pushing level above 0.6. Asserts `tighten_budgets` called with a value > 0.6, `relax_budgets` not called.

`test_register_resource_governor_low_cortisol` — Manually sets cortisol to 0.05, then releases 0.05 (resulting level 0.10 < 0.3). Asserts `relax_budgets` called with factor > 1.0, `tighten_budgets` not called.

`test_register_resource_governor_neutral_cortisol` — Releases 0.2 cortisol from baseline 0.2 → level 0.4 (neutral zone). Asserts neither method called.

`test_register_resource_governor_none` — No ResourceGovernor registered. Fires cortisol at 0.8. Asserts no exception raised. Confirms absence of rg reference is safe.

All 17 tests pass in 0.58s. 219 coordination-adjacent tests pass with zero regressions.

---

## Interface Exposed

### `EndocrineSystem.register_resource_governor(rg: Any) -> None`
- Subscribes `rg` to cortisol releases via the standard `register_consumer` path
- Calls `rg.tighten_budgets(cortisol_level)` when cortisol > 0.6
- Calls `rg.relax_budgets(1.0 + (0.3 - cortisol_level))` when cortisol < 0.3
- No-op in neutral zone [0.3, 0.6]
- Falls back to `rg.set_hormone_level("cortisol", level)` if tighten/relax not present
- Safe to call with any duck-typed object; safe to omit entirely

---

## Files Modified

| File | Change |
|------|--------|
| `mae_core/coordination/endocrine_system.py` | Added `register_resource_governor()` method (10th consumer registration) |
| `tests/test_drive_coupling.py` | Added 4 tests in `TestRegisterResourceGovernor` class; added `EndocrineSystem`/`HormoneType`/`MagicMock` imports |

## Files NOT Modified
- `mae_core/market/resource_governor.py` — `tighten_budgets()` and `relax_budgets()` are B1's Round 2 additions; the endocrine coupling is wired against the interface contract only
- All other files
