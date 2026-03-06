# Corrector Fixes — Based on Independent Reviews

**Date:** 2026-03-06
**Reviews addressed:** review-1.md (Reviewer 1), review-2.md (Reviewer 2)

## Fixes Applied

### Fix 1: Sort domain list for deterministic effective count
**Flagged by:** Both reviewers independently (high confidence)
**Location:** `convergence_alerter.py:716`
**Change:** `list({sig.domain for sig in signals})` → `sorted({sig.domain for sig in signals})`
**Why:** The greedy algorithm in `_compute_effective_domain_count` is order-dependent. Python set iteration order is non-deterministic across restarts (hash randomization). Sorting ensures the same signal set always produces the same effective count.

### Fix 2: Correct domain pair example in docstring
**Flagged by:** Reviewer 1 (factually verified)
**Location:** `convergence_alerter.py:690, 714`
**Change:** `"macro+technical at r=0.73"` → `"institutional+macro at r=0.57"`
**Why:** The r=0.73 figure from Phase 0 is between `finra_short` and `fred_macro`, which maps to institutional+macro domains (via _DOMAIN_SOURCES), not macro+technical. The formula is correct; the example was misleading.

## Findings Not Fixed (Advisory)

| Finding | Reviewer | Why Not Fixed |
|---------|----------|---------------|
| I/O latency in lock during forgetting log | R2 | Performance concern, not correctness. Single-threaded daemon context. Future optimization. |
| `CorrelationPair.last_updated` annotation | Both | Pre-existing issue, not introduced by this build. Out of scope. |
| Missing test cases (multi-regime, unknown domain, etc.) | Both | Advisory. Tests cover all critical paths. Can be added incrementally. |
