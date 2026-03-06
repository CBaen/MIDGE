# Calibrator Build Report — Round 1
**Date:** 2026-03-05
**Role:** Calibrator — Thompson Feedback Fix
**Builder:** Claude Sonnet 4.6

---

## Summary

Fixed the Thompson Sampling feedback loop. The Bayesian learning engine was being erased by forgetting twice as fast as outcomes were evaluated. Distributions that reached alpha=1389 decayed to the uninformative prior (Beta(1,1)) over 8 days of daemon operation because: forgetting fired every 100 steps while outcome evaluation fired every 200 steps, and the floor allowed full collapse to an uninformative distribution.

All 4 tasks complete. 16 new tests written, all passing. Zero regressions introduced to any Thompson-related test file (74 tests across 4 files).

---

## Tasks Completed

### Task 1: Fix Forgetting/Learning Cadence Mismatch

**Files changed:**
- `mae_core/market/intelligence/thompson_sampler.py` — floor raised from 1.0 to 2.0
- `mae_core/bootstrap/market_hooks.py` — cadence changed from 100 to 200

**Root cause:** The original floor of 1.0 maps to Beta(1,1) — the fully uninformative prior. A distribution at Beta(1389, 611) with alpha*0.99^N reaches 1.0 in about 200 cycles of forgetting. With forgetting at 100 steps and outcomes at 200 steps, forgetting fires twice per learning cycle — so learned distributions erode before feedback can reinforce them.

**Fix decisions:**
- **Floor raised to 2.0**: Beta(2,2) is still symmetric (mean=0.5) but has lower variance than Beta(1,1), meaning it retains a tiny amount of prior conviction. More importantly, it means a biased distribution (e.g., Beta(30,10) representing a 75% win rate signal) never fully collapses to uninformative — it approaches Beta(2,2) asymptotically. The directional information encoded by the alpha/beta ratio is preserved until the distribution actually reaches the floor.
- **Cadence aligned to 200**: Matches `sensing_hook._outcome_cadence=200` exactly. Now for every learning event (outcome evaluation), there is at most one forgetting event. The dynamics are 1:1 instead of 2:1 erosion.

**Also updated:** The log message in `market_hooks.py` that hardcodes cadence values (changed "forgetting/100" to "forgetting/200") to keep documentation accurate.

### Task 2: Log Forgetting Events

**File changed:** `mae_core/market/intelligence/thompson_sampler.py`

Added a compact JSON summary entry to `thompson_history.jsonl` at the end of each `apply_forgetting()` call. The entry contains:
- `event: "forgetting_applied"`
- `decay_factor`: the factor passed in
- `distributions_affected`: count returned by the method
- `timestamp`: ISO format

**Design decisions:**
- One entry per call, not per distribution. With 50+ distributions, per-distribution logging would bloat the history file rapidly. A summary entry provides sufficient observability — you can see that forgetting fired, what decay was used, and how many distributions were affected.
- Logging happens inside the lock but after `_save_distributions_locked()` completes. This ensures distributions are written to disk before the log entry is appended. The history file write is a separate append operation and is intentionally not part of the locked atomic save (consistent with how `_log_update` works for normal updates).
- Uses the same `self.history_path` attribute already established by the constructor.
- Failure is caught and logged via `logger.debug` — consistent with the project pattern of advisory enforcement (triads observe/report, never block).

### Task 3: Clean Ghost Predictions

**File changed:** `data/market/predictions.jsonl`

Filtered out 10 old-format records that used `prediction_id` as root key instead of `signal_id`. These records had no `source` field and were stuck permanently (the outcome collector cannot match them to signals because it keys on `source`).

**Before:** 11,599 records (or 11,577 at time of pre-check — daemon was writing concurrent records)
**After:** 11,589 records (or 11,567)
**Removed:** 10 ghost records

Ghost record IDs:
- disc_20260208_051518_mock_1, disc_20260208_051518_mock_2
- disc_20260208_075547_mock_1, disc_20260208_075547_mock_2
- disc_20260208_081023_mock_1, disc_20260208_081023_mock_2
- disc_20260208_083524_mock_1, disc_20260208_083524_mock_2
- 0f3394f3-5ab2-402e-ada4-09ce89512a7b (LMT bearish)
- 58c03100-d345-44e4-b525-3a447ba8ddef (BA bearish)

All 10 were Feb 8 2026 records from when MIDGE was still using the old prediction format.

**Implementation:** Python one-shot script run inline (not a runtime change). Atomic write via `.tmp` file + `os.replace()`.

### Task 4: Tests

**File created:** `tests/test_thompson_feedback.py`

16 tests organized into 4 classes:

| Class | Tests | What it verifies |
|-------|-------|-----------------|
| `TestForgettingFloor` | 5 | Floor is 2.0 (not 1.0), even for uninformative prior, after single call, after 500 cycles, with biased distributions, and that Beta(2,2) is informationally superior to Beta(1,1) |
| `TestForgettingLogsToHistory` | 5 | History file is created, entry has all required fields, exactly 1 entry per call (not per distribution), `distributions_affected` matches return value, no entry when no distributions exist |
| `TestExtendedForgettingStaysAboveFloor` | 2 | Regression for the exact case from the build brief: Beta(100,200) over 200+ cycles stays >= 2.0. Extreme case: 10,000 cycles at 0.90 decay still stays at floor |
| `TestCadenceAlignment` | 4 | Simulated hook fires at 200/400 but NOT 100/300. Fires at exact pattern [0,200,400,600]. market_hooks.py does NOT contain "forgetting/100" string. market_hooks.py DOES contain "forgetting/200" string. |

All 16 tests pass.

---

## Verification

```
tests/test_thompson_feedback.py: 16 passed
tests/test_combo_thompson.py: 18 passed
tests/test_contextual_thompson.py: 15 passed
tests/test_thompson_calibrator.py: 11 passed (+ 14 from other files)
Total Thompson-domain tests: 74 passed, 0 failed
```

Full suite: pre-existing test ordering failures exist in `test_composite_hypotheses.py` when run as part of the full suite (shared mutable state between tests in the larger run). These tests pass 23/23 in isolation and were present before my changes — confirmed by stash test on unmodified code.

---

## Files Touched

| File | Change |
|------|--------|
| `mae_core/market/intelligence/thompson_sampler.py` | Floor 1.0→2.0, added forgetting log |
| `mae_core/bootstrap/market_hooks.py` | Cadence 100→200, updated log message |
| `data/market/predictions.jsonl` | Removed 10 ghost records (one-time cleanup) |
| `tests/test_thompson_feedback.py` | NEW — 16 tests |

---

## Out-of-Domain Notes

No changes required to files outside my domain. `sensing_hook.py` was read for context (confirming `_outcome_cadence=200`) but not modified.

The pre-existing test ordering failure in `test_composite_hypotheses.py` (within the full suite run) is not in my domain to fix and was not introduced by my changes.
