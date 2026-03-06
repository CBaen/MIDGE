# Review-1 — Round 1 Independent Review
**Date:** 2026-03-06
**Reviewer:** Claude Sonnet 4.6 (independent — did not build either component)
**Scope:** Calibrator (Tasks 1-4) + Corrector (Tasks 5-8)

---

## Verdict

**Conditional pass.** Both builds are largely correct and safe to ship. Three findings require attention before marking this round closed: one confirmed bug, one inaccuracy in the build report documentation, and one latent technical debt item. None block the core functionality from working correctly. Zero regressions were introduced to any existing tests.

---

## 1. Integration Errors

### Finding I-1: Build report claims r=0.73 for macro+technical — INCORRECT

**Severity: Medium (documentation/reasoning flaw, not a runtime bug)**

The Corrector's build report uses `macro + technical |r|=0.73` as the primary numerical example in both the report body and the docstring for `_compute_effective_domain_count()`. This is factually wrong.

- The `r=0.73` figure originates from the Phase 0 report, which measured it for `finra_short + fred_macro` — a lag-8 correlation.
- `finra_short` maps to the **institutional** domain in `_DOMAIN_SOURCES`, not technical.
- `fred_macro` maps to **macro**.
- The actual max `|r|` for `fred_macro ↔ yfinance_price` (macro ↔ technical path) in `lag_correlations.json` is **0.4823** — moderate, not strong.

Consequence at runtime: when `macro + technical` appear together in a convergence, the seeded data produces a **moderate (0.7 credit)** reduction, not the **strong (0.5 credit)** described in the report and tests. The tests use a manually constructed tracker with `r=0.73` injected directly, so they pass — but they test a scenario that does not match the actual seeded production data.

The **correct** strong-correlation pair at runtime is `institutional ↔ macro` (`finra_short ↔ fred_macro`, max |r|=0.5678). The Corrector's docstring example is misleading.

**Impact on correctness:** None at runtime — the formula is correct, only the example is wrong. But the next engineer reading the code will form incorrect intuitions about which domain pairs are most affected.

**Recommendation:** Update the `_compute_effective_domain_count` docstring and the Corrector report to use the correct pair: `"e.g. institutional+macro r=0.57"`. Update any hardcoded test comments that say `"macro+technical |r|=0.73"`.

---

### Finding I-2: Build report claims 10 unique seeded pairs — INCORRECT (9 actual)

**Severity: Low (documentation error only)**

The Corrector report (Task 7) states: "After deduplication (max per canonical pair), seeds 10 unique pairs."

Actual count verified by running `seed_from_lag_data` against production `lag_correlations.json`: **9 pairs seeded**, not 10.

Canonical pairs:
```
('finra_short', 'fred_macro'), ('finra_short', 'sec_form4'), ('finra_short', 'yfinance_price'),
('fred_macro', 'sec_efts'), ('fred_macro', 'sec_form4'), ('fred_macro', 'yfinance_price'),
('sec_efts', 'sec_form4'), ('sec_efts', 'yfinance_price'), ('sec_form4', 'yfinance_price')
```

This is a documentation error with no runtime consequence. The seed method's logic is correct.

---

### Finding I-3: Cross-builder floor interaction — CLEAN

The critical integration question was: does Thompson floor=2.0 interact with the effective domain count to produce unexpected confidence values?

Verified: Beta(2,2) has `samples = 2+2-2 = 2`, which is below the 5-observation blend threshold in `_get_thompson_weight`. The weight formula returns 1.0 (neutral) for both Beta(1,1) (old floor) and Beta(2,2) (new floor). The floor change produces **identical Thompson weights** when distributions are at the floor state. No unexpected confidence inflation or deflation from the cross-builder interaction.

---

## 2. Constraint Violations

### Finding C-1: No Law 1 violation (confirmed acceptable)

The Corrector chose not to register a ConnectionRegistry entry for the `convergence_alerter ↔ correlation_tracker` dependency injection. The report justifies this by noting the CorrelationTracker is already in the triadic graph via other connections, and adding a bare dyad entry would itself violate Law 1 (which requires a third witness). This reasoning is sound and matches the existing pattern used for `regime_classifier`. No action required.

---

## 3. Bugs and Logic Errors

### Finding B-1: Floor of 2.0 is NOT applied during `update()` — CONFIRMED CORRECT (not a bug)

`update()` does not apply the floor. After `update(success=False)`, a Beta(1,1) distribution becomes Beta(1,2) — below the floor. This is intentional: Bayesian updates must be exact. The floor is only a forgetting-decay guard. The next `apply_forgetting()` call will raise both parameters to the floor.

This means a signal that records its very first loss sits at Beta(1,2) (mean=0.33) until the next forgetting event, at which point both alpha and beta jump to 2.0 (mean=0.5) — erasing the directional signal from that single outcome. This is a known tradeoff of the floor=2.0 design, explicitly documented in the build report. Not a bug; acceptable design decision for thin-data distributions.

### Finding B-2: `get_distribution()` initializes new distributions at Beta(1,1), inconsistent with floor

New distributions are initialized at `{"alpha": 1.0, "beta": 1.0}` in `get_distribution()`. This is below the stated floor of 2.0. The floor is never applied at initialization time — only during forgetting. This means a brand-new signal that gets sampled or weighted before any forgetting event operates on Beta(1,1), not Beta(2,2).

**Assessment:** Minor inconsistency. The practical effect is negligible because:
- New signals have `samples = 0`, which triggers the full neutral-blend (`blend=0.0`) in `_get_thompson_weight`, yielding weight=1.0 regardless of mean.
- The first forgetting event (at most 200 steps later) raises it to Beta(2,2).

**Not a regression** — this was already the behavior before the floor change. But the floor claim "distributions never go below 2.0" is technically imprecise; it should read "forgetting never drives distributions below 2.0."

---

## 4. Edge Cases

### Finding E-1: Non-deterministic effective domain count across daemon restarts

**Severity: Low (latent technical debt)**

In `_compute_confidence()`:
```python
domain_list = list({sig.domain for sig in signals})
```

`{...}` is a set comprehension. Python set iteration order is **undefined by the language spec** (though CPython is consistent within a single process run due to hash seeding). With `PYTHONHASHSEED` randomized (the default), the iteration order of the domain set can differ between daemon restarts.

`_compute_effective_domain_count()` is **order-dependent**: the first domain always gets 1.0 credit, and subsequent domains are evaluated against all previously-counted domains. For a 3-domain convergence where domain A correlates with B but not C, the effective count varies depending on whether A or C is processed first.

**Demonstrated example:**
- Domains {A, B, C}: A-B strong, A-C independent, B-C moderate
- Order A,B,C → effective = 2.2
- Order C,A,B → effective = 2.5

**Current practical impact:** Within a single daemon run, the order is stable (same PYTHONHASHSEED). Across restarts, confidence for the same signal combination could vary by a small amount (< 5% of the diversity bonus). This is not visible in tests because tests run in a single process.

**Fix:** Change `list({sig.domain for sig in signals})` to `sorted({sig.domain for sig in signals})` in `_compute_confidence()`. One-line fix; makes the behavior deterministic and testable.

---

### Finding E-2: Seeded CorrelationPairs have `last_updated=None`

Pairs created by `seed_from_lag_data()` have `last_updated=None` (the dataclass default). Any code path that accesses `pair.last_updated` without a None guard would raise `AttributeError` or fail silently.

Verified: `convergence_alerter.py` does not access `last_updated` anywhere. The `update_correlations()` method sets `last_updated` on pairs it processes, but it only processes signals that have `history` entries — seeded pairs have no history. So seeded pairs can sit with `last_updated=None` indefinitely.

**Verdict:** No current failure path. Low risk. Worth noting for future code that may add `last_updated`-based staleness checks.

---

## 5. Regression Risk

**No regressions introduced.**

Tests run:
- `test_thompson_feedback.py`: 16 passed
- `test_independence_correction.py`: 23 passed
- `test_combo_thompson.py`: 18 passed
- `test_contextual_thompson.py`: 15 passed
- `test_thompson_calibrator.py`: 11 passed (+14 others)
- `test_convergence_alerter_cascade.py`: passes
- `test_convergence_domain_windows.py`: passes
- `test_correlation_tracker_wiring.py`: passes
- `test_lag_correlation_analyzer.py`: passes
- All 62 pre-existing convergence/correlation tests pass.

Total Thompson-domain + new tests: **97 passed, 0 failed**.

Pre-existing failure in `test_composite_hypotheses.py` (shared mutable state, test-ordering issue) is confirmed pre-existing, not caused by either builder. Verified by both builders independently.

---

## 6. Security

No new external inputs, deserialization paths, or injection surfaces introduced.

`seed_from_lag_data()` reads from a local file. Path is constructed from `__file__` (not user input) in market_systems.py. The JSON parsing has appropriate try/except around it. No risk.

---

## 7. Test Coverage

### Calibrator tests

Coverage is solid. Key gaps:

- **No test for the floor inconsistency between initialization and forgetting.** A signal at Beta(1,1) that gets a loss (Beta(1,2)) and then forgetting (Beta(2,2)) loses its directional information. This edge case is not tested. Low severity but worth documenting.
- **No test for thread safety.** Concurrent calls to `update()` and `apply_forgetting()` are not tested. Given the single `threading.Lock`, this is acceptably safe, but a threading integration test would give confidence.
- The cadence tests simulate the hook logic in test code rather than calling the actual hook. If the hook is refactored, the test would still pass while the behavior changes. Acceptable for current purposes.

### Corrector tests

Coverage is solid. Key gaps:

- **No test using the actual `lag_correlations.json` production file** to verify that the real-world seeded data produces the expected domain-pair reductions. The `test_seed_returns_correct_count` test uses a manually constructed fixture. A test against the real file would have caught the `r=0.73 / 10 pairs` documentation errors at build time.
- **No test for the order-sensitivity of `_compute_effective_domain_count`** with three domains where order affects the result. All current tests use domain sets where the order does not change the outcome.
- `TestEffectiveDomainCountIntegration::test_confidence_scales_with_effective_count` uses `>=` not `>`, meaning it passes even when correlated and independent produce identical confidence. The test will not catch a regression where the correlation adjustment stops working.

---

## 8. What Works

The core fixes are correct and address the root causes identified in Phase 0.

**Calibrator:** The 2:1 forgetting/learning cadence mismatch was the right diagnosis and the fix is clean. Changing cadence from 100 to 200 aligns forgetting with outcome evaluation. Floor 2.0 correctly preserves directional memory in distributions with real evidence (e.g., Beta(30,10) decays toward Beta(2,2), not Beta(1,1), so the alpha > beta ratio is preserved until the floor is actually reached). The forgetting history log is correctly placed inside the lock, correctly uses a single summary entry per call, and failure is correctly advisory-only.

**Corrector:** The two-phase wiring pattern is correct and consistent with the existing `regime_classifier` approach. `_compute_effective_domain_count()` correctly implements the greedy sequential algorithm. `seed_from_lag_data()` correctly deduplications by max |r| and correctly preserves live runtime data. `get_correlation()` correctly bypasses the 30-observation minimum for seeded entries. Backward compatibility with `correlation_tracker=None` is correctly implemented and tested.

The cross-builder interaction (floor change + effective domain count) produces no unexpected behavior: Thompson weights at the floor state return 1.0 (neutral) regardless of whether the floor is 1.0 or 2.0.

---

## Summary of Actions Required

| # | Finding | Severity | Action |
|---|---------|----------|--------|
| I-1 | Corrector report + docstring cite wrong source pair for r=0.73 example | Medium | Update docstring and test comments to use `institutional+macro` example |
| I-2 | Corrector report claims 10 seeded pairs, actual is 9 | Low | Fix report (documentation only) |
| E-1 | `list(set())` in `_compute_confidence` is order-non-deterministic | Low | Change to `sorted(...)` — one-line fix |
| E-2 | Seeded `CorrelationPair.last_updated=None` | Low | Acceptable for now; note in HANDOFF |
| B-2 | `get_distribution()` initializes below floor | Low | Clarify docstring only; behavior is acceptable |

No changes to production code are strictly blocking. Finding E-1 (sorted domains) is the one code change worth making before this is considered fully closed — it is a one-line fix that eliminates a latent non-determinism.
