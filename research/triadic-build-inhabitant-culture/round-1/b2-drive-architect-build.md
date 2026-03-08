# B2 Drive Architect — Round 1 Build Report

**Date:** 2026-03-08
**Builder:** B2 Drive Architect
**Domain:** Coordination files (homeostasis, organism_state)

---

## Tasks Completed

### Task 1: `compute_drive_urgency()` on HomeostasisRegulator

**File:** `mae_core/coordination/homeostasis.py`

Added a public method `compute_drive_urgency()` in the Public Query API section, placed directly before `get_deviation_score()` for logical grouping.

**What it does:** Returns a `dict[str, float]` of `{parameter_name: urgency}` for every parameter currently *outside* its acceptable range. Parameters within range are excluded entirely — an empty dict means perfect homeostasis.

**Design decisions:**

1. **Thin wrapper pattern.** All computation delegates to the existing `_compute_urgency(sp)` private method. No logic was duplicated. This respects the single-source-of-truth principle and avoids drift between the public and private urgency computations.

2. **Only out-of-range parameters included.** The caller (OrganismState) needs to know what is *wrong*, not a full status report of all 7 parameters. Excluding in-range parameters keeps the signal clean and the dict sparse.

3. **Clamp to 1.0.** `_compute_urgency()` can return values above 1.0 when a parameter exceeds its range boundary (e.g., urgency=1.4 for a value far beyond the limit). The public method clamps to `min(1.0, urgency)` to guarantee the contract `[0.0, 1.0]` for callers.

4. **Return type annotation.** Uses `dict[str, float]` with `from __future__ import annotations` already present in the file — consistent with the rest of the module.

---

### Task 2: Priority 6 homeostasis deviation in `get_reflex_override()`

**File:** `mae_core/coordination/organism_state.py`

Two changes:

**Class constant added** immediately after `CH_ORGANISM_ACTION_OUTCOME`:
```python
_HOMEOSTASIS_URGENCY_THRESHOLD: float = 0.7
```
Placed as a class-level annotation inside the class body. This makes it tunable without hunting through logic, and makes the threshold discoverable by reviewers and callers.

**Priority 6 check added** in `get_reflex_override()` after Priority 5 (kidney stress) and before `return None`:
```python
if self._homeostasis_deviation >= self._HOMEOSTASIS_URGENCY_THRESHOLD:
    return "rest"
```

**Why `>=` not `>`:** The threshold represents the exact boundary. At 0.7 the organism is at the limit of acceptable internal deviation — the reflex should fire, not wait for 0.701.

**Why "rest" and not a new action:** The existing `"rest"` action already maps to metabolic consolidation and task release. Introducing a new action would require wiring into `_act()`, `DecisionActionLifecycleMixin`, and all downstream consumers. The biology supports "rest" — when an organism's internal chemistry is destabilized, the correct response is to stop expending energy until homeostasis is restored.

**Why Priority 6 (not higher):** Acute emergencies — pain overload, hypoxia, starvation — are immediate mortal threats that require reflexive override. Homeostasis deviation is a chronic signal indicating internal imbalance that may worsen if action continues. It is serious but not instantly lethal. Placing it after the five acute priorities preserves the emergency response hierarchy while still allowing the organism to self-regulate before normal routing proceeds.

**Advisory enforcement:** `get_reflex_override()` already returns a suggestion string or `None`. The caller in `lifecycle_decision.py` line 106 checks `if reflex is not None: return reflex` — this is already advisory-compatible. No changes needed to the call site.

---

### Task 3: Tests

**File:** `tests/test_drive_coupling.py` (new, 13 tests, all passing)

Tests are organized into two classes:

**`TestComputeDriveUrgency`** (6 tests):
- `test_compute_drive_urgency_returns_dict` — return type
- `test_compute_drive_urgency_empty_when_stable` — all at setpoint → empty dict
- `test_compute_drive_urgency_high_when_deviated` — out-of-range param → high urgency (threat_level pushed to 0.6, above max of 0.5)
- `test_compute_drive_urgency_keys_are_parameter_names` — keys are valid setpoint names
- `test_compute_drive_urgency_values_clamped_to_1` — values never exceed 1.0
- `test_compute_drive_urgency_in_range_params_excluded` — in-range params absent from dict

**`TestReflexOverrideHomeostasisDeviation`** (7 tests):
- `test_reflex_override_homeostasis_deviation` — deviation 0.8 → "rest"
- `test_reflex_override_homeostasis_at_threshold` — exactly at 0.7 → "rest"
- `test_reflex_override_homeostasis_below_threshold` — deviation 0.3 → None
- `test_reflex_override_homeostasis_zero_returns_none` — deviation 0.0 → None
- `test_reflex_override_pain_still_higher_priority` — pain + homeostasis → "rest" (pain wins cascade)
- `test_reflex_override_energy_critical_overrides_homeostasis` — energy critical + homeostasis → "explore" (Priority 4 wins)
- `test_reflex_override_threshold_is_class_constant` — class attribute exists, is float, is in (0, 1)

**One test was corrected during development:** The initial `test_compute_drive_urgency_high_when_deviated` used threat_level=0.5 (the exact max boundary). `Setpoint.in_range` uses `<=`, so 0.5 is *in range* and the param would not appear in the urgency dict. Fixed to use 0.6 (outside the boundary).

---

## Regression Check

- `tests/test_maintenance_systems.py` + `test_metabolic_regulation.py` + `test_metabolic_process.py` + `test_autopoietic_closure.py`: **161 passed**
- `tests/test_nervous_system.py` + `test_deep_integration.py` + `test_holon_protocol.py`: **163 passed**
- `tests/test_drive_coupling.py`: **13 passed**

Zero regressions in all coordination-adjacent test clusters.

---

## Interfaces Exposed

### `HomeostasisRegulator.compute_drive_urgency() -> dict[str, float]`
- Returns only out-of-range parameters
- Values in `[0.0, 1.0]`
- Empty dict = all systems stable
- Suitable for OrganismState to use as drive urgency feed

### `OrganismState._HOMEOSTASIS_URGENCY_THRESHOLD: float = 0.7`
- Class constant, tunable
- Used by `get_reflex_override()` Priority 6

### `OrganismState.get_reflex_override()` Priority 6 behavior
- Fires when `_homeostasis_deviation >= 0.7`
- Returns `"rest"`
- Checked after Priorities 1-5, before returning None
- `_homeostasis_deviation` is populated by `_on_homeostasis_correction()` callback (max urgency per step cycle)

---

## Files Modified

| File | Change |
|------|--------|
| `mae_core/coordination/homeostasis.py` | Added `compute_drive_urgency()` public method |
| `mae_core/coordination/organism_state.py` | Added `_HOMEOSTASIS_URGENCY_THRESHOLD` class constant + Priority 6 check in `get_reflex_override()` |
| `tests/test_drive_coupling.py` | New test file, 13 tests |

## Files NOT Modified (as required)
- `mae_core/coordination/endocrine_system.py` (Round 2)
- All other files
