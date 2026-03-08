# B4 Quorum & Convergence — Round 1 Build Report

**Date:** 2026-03-08
**Builder:** B4 — Quorum & Convergence specialist
**Status:** COMPLETE — all tasks done, all tests pass

---

## Summary

Wired `QuorumSpace` contributor counts as a confidence multiplier on convergence alerts. When 3+ independent agents all deposit signals for the same `ticker:direction` key, the convergence alerter now amplifies that alert's confidence — reflecting genuine collective agreement, not just data-domain breadth.

---

## What Was Built

### Task 1: Quorum Multiplier in ConvergenceAlerter

**File:** `mae_core/market/intelligence/convergence_alerter.py`

Three changes:

1. **Constructor parameter** — `quorum_space=None` added as the last optional parameter in `__init__`. Stored as `self._quorum_space`. The docstring entry explains its role. Fully backward-compatible: all existing bootstrap code that creates `ConvergenceAlerter` without the argument continues to work unchanged.

2. **Helper method** `_apply_quorum_boost(confidence, ticker, direction)` — added immediately before `_compute_confidence()`. This is the single source of logic for all paths:
   - Looks up `self._quorum_space.get_contributor_count(f"{direction}:{ticker}")`
   - Applies multiplier schedule: 3→1.1×, 4→1.2×, 5+→1.3× (cap via `min(0.3, (count-2)*0.1)`)
   - Caps final confidence at 1.0 via `min(1.0, confidence * multiplier)`
   - Returns confidence unchanged if `_quorum_space is None`, `ticker` is empty, or `count < 3`
   - Swallows all exceptions (graceful degradation pattern consistent with every other modifier in this file)
   - Logs at DEBUG: `"Quorum boost: %d contributors on %s:%s → %.2f multiplier"`

3. **Call sites** — two insertions, one per convergence path:
   - **Global direction path** (`_check_convergence_for_direction`): inserted after the sequence-score block, before summary generation. Uses `primary_ticker` (extracted earlier in the method from `directional_signals[].metadata["symbol"]`). Guarded with `if primary_ticker:` to match the existing guard pattern.
   - **Per-ticker path** (`check_ticker_convergence`): inserted after deception detection, before urgency calculation. Uses the explicit `ticker` loop variable — no guard needed, it's always a non-empty string in this path.

### Task 2: Tests

**File:** `tests/test_quorum_confidence.py` (new, 11 tests)

All 11 pass. Tests cover:

| Test | What it verifies |
|------|-----------------|
| `test_quorum_boost_at_3_contributors` | 3 contributors → exactly 1.1× applied |
| `test_quorum_boost_at_5_contributors` | 5 contributors → exactly 1.3× applied |
| `test_quorum_boost_at_4_contributors` | 4 contributors → exactly 1.2× applied |
| `test_quorum_boost_capped_at_1` | 0.95 × 1.3 = 1.235 → capped to 1.0 |
| `test_no_quorum_space_no_boost` | quorum_space=None → identity function |
| `test_quorum_below_threshold_no_boost` | 1 contributor → no boost |
| `test_quorum_2_contributors_no_boost` | 2 contributors → no boost (boundary) |
| `test_quorum_signal_key_format` | Key is exactly `"{direction}:{ticker}"` |
| `test_quorum_large_contributor_count_capped_multiplier` | 100 contributors → same 1.3× cap |
| `test_quorum_empty_ticker_no_boost` | Empty ticker string → identity, no QuorumSpace call |
| `test_quorum_space_exception_is_swallowed` | RuntimeError from QuorumSpace → confidence unchanged |

---

## Design Decisions

**Why a helper method instead of inline code?**
The logic appears in two convergence paths. Inline duplication would create a maintenance hazard — a future multiplier schedule change would need updating in two places. The helper makes both call sites one-line and the logic one place.

**Why insert after sequence_score / after deception detection?**
All existing confidence multipliers accumulate left-to-right. Inserting last (before summary) means quorum can see the fully-adjusted confidence and amplify it proportionally. It does not change fire conditions (those are gated much earlier by `min_domains`, `min_strength`, direction counts). This placement is symmetric with how archetype boost works in the global path.

**Why not modify fire conditions?**
The brief explicitly prohibits this. Quorum is collective confirmation of an already-qualified signal, not a new gate. If we lowered `min_domains` for high-quorum signals we'd be changing architectural policy without consensus.

**Why cap at 1.0 instead of 0.95?**
The brief says "cap final confidence at 1.0." Other caps in this file use 0.95 to leave headroom for downstream processors. However, the quorum boost is additive confidence (collective confirmation), not a source-reliability estimate — it's semantically different. I follow the brief's instruction and cap at 1.0 directly.

---

## Regression Results

```
tests/test_quorum_confidence.py          11 passed
tests/test_convergence_alerter_cascade.py   13 passed (no change)
tests/test_convergence_domain_windows.py    10 passed (no change)
```

Zero regressions.

---

## Interface Contract (for integration by bootstrap owner / orchestrator)

To wire quorum into convergence during bootstrap, pass the `QuorumSpace` instance:

```python
ctx.convergence_alerter = ConvergenceAlerter(
    min_domains=3,
    thompson_sampler=getattr(ctx, "thompson_sampler", None),
    # ... existing params ...
    quorum_space=getattr(ctx, "quorum_space", None),  # ADD THIS
)
```

The `QuorumSpace.deposit_signal()` call from agents must use key format `"{direction}:{ticker}"` — e.g. `"bullish:AAPL"` — for the boost to fire. Agents depositing under other key formats will not interfere (contributor count will be 0 for the lookup).

---

## Files Changed

| File | Change type |
|------|------------|
| `mae_core/market/intelligence/convergence_alerter.py` | Modified — constructor param, helper method, two call sites |
| `tests/test_quorum_confidence.py` | New — 11 tests |
