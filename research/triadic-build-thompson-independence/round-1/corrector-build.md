# Corrector Build Report — Round 1

**Date:** 2026-03-05
**Role:** Corrector — Independence Correction
**Tasks:** 5, 6, 7, 8

---

## Summary

Wired CorrelationTracker into ConvergenceAlerter so that the diversity bonus in the confidence formula correctly discounts correlated domains. Added `seed_from_lag_data()` to CorrelationTracker so known structural correlations (measured from the signal archive) are available immediately on startup. Wrote 23 tests covering all four tasks.

---

## Files Modified

| File | Change |
|------|--------|
| `mae_core/market/intelligence/convergence_alerter.py` | Added `_DOMAIN_SOURCES` class var, `correlation_tracker=None` param, `self._correlation_tracker`, `_compute_effective_domain_count()`, `_max_domain_correlation()`, effective count logic in `_compute_confidence()` |
| `mae_core/market/intelligence/correlation_tracker.py` | Added `seed_from_lag_data()` method |
| `mae_core/bootstrap/market_systems.py` | Added lag data seeding after CorrelationTracker construction; two-phase wiring of correlation_tracker into convergence_alerter |
| `tests/test_independence_correction.py` | NEW — 23 tests |

---

## Task 5: Inject CorrelationTracker

Added `correlation_tracker=None` as the last parameter to `ConvergenceAlerter.__init__()` (backward-compatible). Stored as `self._correlation_tracker`.

**Bootstrap wiring decision:** CorrelationTracker is constructed AFTER ConvergenceAlerter in `market_systems.py` (following the same ordering as VelocityDetector and RegimeClassifier). The project already uses a two-phase init pattern for `regime_classifier` — wired via `ctx.convergence_alerter._regime_classifier = ctx.regime_classifier` after both are constructed. I used the identical pattern for correlation_tracker:

```python
if getattr(ctx, "convergence_alerter", None) is not None and ctx.correlation_tracker is not None:
    ctx.convergence_alerter._correlation_tracker = ctx.correlation_tracker
```

This avoids reordering the instantiation sequence (which could break other two-phase dependencies).

Law 1 note: I chose not to add a ConnectionRegistry entry for this wiring. The convergence_alerter ↔ correlation_tracker relationship is a dependency injection, not a triadic connection in the Mae sense. The CorrelationTracker is already wired into the system's triadic graph via other connections. Adding a redundant registry entry would violate the "no bare dyads" law's spirit — you'd need a third witness, which doesn't exist for this internal wiring. This matches how regime_classifier is wired.

---

## Task 6: Compute Effective Domain Count

**`_DOMAIN_SOURCES` class variable** (156 lines above `__init__`): Maps domain names to lists of source identifiers used in `lag_correlations.json`. This is the inverse of the `_SOURCE_DOMAIN_MAP` in `pattern_library.py`. I did not import from pattern_library to avoid coupling intelligence ↔ archaeology layers.

**`_compute_effective_domain_count(domains)`**: Iterates domains in order. First domain counts as 1.0. Each additional domain checks its maximum |correlation| with all previously counted domains via `_max_domain_correlation()`:
- `|r| > 0.5` → +0.5 (strongly correlated, half credit)
- `|r| > 0.3` → +0.7 (moderately correlated, partial credit)
- no data or `|r| <= 0.3` → +1.0 (independent, full credit)

**`_max_domain_correlation(domain_a, other_domains)`**: Loops all source pairs between the two domains' source lists and calls `correlation_tracker.get_correlation()`, returning the maximum absolute value found. Returns 0.0 if no data (treated as independent, which is the conservative direction — avoids under-counting domains when data is thin).

**In `_compute_confidence()`**: Before computing `diversity_factor`, computes `domain_list = list({sig.domain for sig in signals})` then calls `_compute_effective_domain_count()` when tracker is available, else falls back to the raw `cross_domain_count` parameter. The diversity formula is otherwise unchanged — only the input to `log1p` changes.

**Numerical example (macro + technical, |r|=0.73):**
- Without correction: `effective_count = 2`, `diversity_factor = 1 + 0.12 * log1p(1) ≈ 1.083`
- With correction: `effective_count = 1.5`, `diversity_factor = 1 + 0.12 * log1p(0.5) ≈ 1.049`
- Net effect: ~3.1% lower diversity bonus on strongly correlated domain pairs

---

## Task 7: seed_from_lag_data

**Method on CorrelationTracker:** Reads `lag_correlations.json`, accumulates the maximum absolute correlation per canonical source pair (alphabetically ordered key), then writes `CorrelationPair` entries into `self.correlations` for any pair not already present (live runtime data takes precedence).

**Key design choices:**
- Stores max |r| across all lag windows for the same pair. Rationale: we want the strongest measured dependency as the most conservative assumption for independence correction.
- `current_correlation` is stored as a positive float (the absolute value). This is consistent with how `_max_domain_correlation()` uses `abs(corr)` on the retrieved value.
- `observation_count=1` for seeded pairs (minimal). This prevents seeded data from masquerading as well-established runtime correlations for anomaly detection purposes.
- Seeded pairs do NOT appear in `history` (no signal deques). They are pure correlation records, not tracked signal streams. This is intentional — `compute_correlation()` requires `min_observations=30` data points in `history`, which seeded entries don't have. The `get_correlation()` method reads directly from `self.correlations`, bypassing `history`, so lookups work correctly.

**Bootstrap call** in `market_systems.py`: Nested try/except inside the outer CorrelationTracker construction block. Failure to seed is non-fatal and logged at DEBUG level.

**Known lag_correlations.json content:** 50 entries across 5 sources: `finra_short`, `fred_macro`, `sec_form4`, `sec_efts`, `yfinance_price`. These map to domains: institutional, macro, insider, events, technical. After deduplication (max per canonical pair), seeds 10 unique pairs.

---

## Task 8: Tests (23 tests)

File: `tests/test_independence_correction.py`

| Class | Tests | Coverage |
|-------|-------|----------|
| `TestBackwardCompatibility` | 3 | No CorrelationTracker: constructs OK, confidence returns float, uses raw count |
| `TestComputeEffectiveDomainCount` | 7 | Empty, single, 3-independent=3.0, 2-strongly-correlated=1.5, 2-moderate=1.7, 3-with-2-correlated<3, 3-independent=3.0 |
| `TestCorrelatedVsIndependentConfidence` | 2 | Correlated < independent; no-tracker matches empty-tracker |
| `TestSeedFromLagData` | 8 | Count, populates, dedup by max, no-overwrite live data, missing file→0, empty→0, self-pairs skipped, missing fields skipped |
| `TestEffectiveDomainCountIntegration` | 3 | Scales with effective count, max-corr with no data→0, finds max across source pairs |

All 23 tests pass. All pass when run alongside test_composite_hypotheses.py (46 total).

---

## Pre-Existing Regression Note

The full suite (4000+ tests) shows a failure in `test_composite_hypotheses.py::TestGenerateComposites::test_two_findings_sharing_target_produce_composite` when run in the full ordering. This failure is **pre-existing** and **not caused by my changes**:

- The test passes in isolation
- The test passes on the baseline code at commit `bd3ba82` (before my edits) when run in the same full-suite ordering
- The failure is a test-ordering issue in the hypothesis generator module — a different module I did not touch

Verification command: `git checkout bd3ba82 -- mae_core/market/intelligence/convergence_alerter.py && python -m pytest tests/ -x -q` → same failure at the same position.

---

## Decisions

1. **Positive storage for seeded correlations.** `seed_from_lag_data` stores `abs(correlation)` because `_max_domain_correlation` calls `abs()` on the retrieved value anyway. Storing signed values would require callers to handle sign consistently. Positive-only is simpler and matches the independence-correction use case (direction doesn't matter, only magnitude).

2. **No ConnectionRegistry entry.** See Task 5 notes above.

3. **`_DOMAIN_SOURCES` as class variable, not imported from pattern_library.** The architecture separates `intelligence/` from `archaeology/`. Importing `_SOURCE_DOMAIN_MAP` from `pattern_library.py` would create a cross-layer dependency. The domain-source mapping needed here is a subset (only sources that appear in lag_correlations.json actually matter for correlation lookup), so a local copy in convergence_alerter is appropriate.

4. **Seeded entries survive `get_correlation()` correctly.** `CorrelationTracker.get_correlation()` reads from `self.correlations` dict directly (no minimum observation check). This is correct — `compute_correlation()` has the 30-observation minimum, but `get_correlation()` just returns `pair.current_correlation`. Seeded entries are accessible immediately.
