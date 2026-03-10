# ADTS Build Report — Regime-Aware Thompson Forgetting

**Date:** 2026-03-09
**Status:** Complete

---

## What Was Built

Adaptive Dynamic Thompson Sampling (ADTS): the Thompson forgetting rate now
adapts to the current market regime instead of using a fixed 0.99 decay.

---

## Files Changed

| File | Change |
|------|--------|
| `mae_core/market/intelligence/thompson_sampler.py` | Added `REGIME_DECAY_RATES` dict + `regime_aware_forget(regime)` method |
| `mae_core/bootstrap/market_hooks.py` | Replaced `apply_forgetting(0.99)` with `regime_aware_forget(regime)` using `ctx.regime_classifier` |
| `tests/test_adts.py` | New: 9 tests covering dict shape, ordering invariants, decay magnitude, floor, and fallback |

---

## Decay Rate Table

| Regime   | Decay | Rationale |
|----------|-------|-----------|
| volatile | 0.90  | Market structure changing fast — recent data dominates |
| bear     | 0.92  | Conditions shifting downward — slightly elevated turnover |
| bull     | 0.95  | Steady uptrend — moderate evidence weight (was hardcoded value) |
| sideways | 0.97  | Range-bound — accumulate evidence, stable signal rankings |
| default  | 0.99  | No price data / unknown regime — conservative fallback, same as before |

---

## Implementation Notes

- `regime_aware_forget()` is a thin wrapper: looks up rate in `REGIME_DECAY_RATES`,
  logs `regime + decay`, delegates to existing `apply_forgetting()`. No new state.
- `market_hooks.py` reads `ctx.regime_classifier` (already bootstrapped as Layer 33).
  If classifier is absent, `regime = "default"` — identical to the old 0.99 behaviour.
- The 2.0 floor in `apply_forgetting()` is preserved; aggressive volatile forgetting
  cannot collapse distributions below Beta(2, 2).
- ~45 lines of new code total.

---

## Test Results

```
tests/test_adts.py  9 passed in 0.23s
Full suite: 970 passed, 1 pre-existing failure (test_congress_gov_client — live API env issue, unrelated)
```

Zero regressions introduced.

---

## Behavioural Impact

In practice the regime classifier caches its result for the calendar day (one
SPY price fetch). The step-200 forgetting cadence calls `regime_aware_forget()`
roughly every 3-4 minutes at pace 2.0.  In a volatile regime (0.90), a
distribution with alpha=10 reaches the 2.0 floor after ~19 forgetting events
(~57 minutes at step 200 cadence).  In sideways (0.97) it takes ~64 events
(~3.2 hours).  This means volatile markets rotate signal trust significantly
faster — which is the intended behaviour.
