# Builder 2 — Convergence Alerter Modification: Build Report

**Date:** 2026-03-08
**Builder:** Builder 2
**File modified:** `mae_core/market/intelligence/convergence_alerter.py`

---

## What Was Built

### Task 1 — Partial convergence emission at min_domains gate

**Location:** Line ~1209 in `_check_direction_convergence()`

The existing `if len(domains_seen) < self.min_domains: return None` guard was extended. When:
- `directional_signals` is non-empty (at least one directional signal exists), AND
- `self._bus` is not None (EventBus is wired)

...it now publishes `"market.intel.partial_convergence"` with:
- `direction` — bullish/bearish
- `domains_seen` — domains that did fire
- `missing_domains` — domains present in `self.signals` but absent from this partial convergence
- `signals` — up to 5 directional signals (source, strength, metadata)
- `min_domains_required` — the gate threshold

The entire publish is wrapped in `try/except pass` so it can never raise into the convergence check. The `return None` is unchanged — existing behavior is identical.

### Task 2 — `_compute_missing_domains()` helper

Added as a private method just before `check_ticker_convergence`. Computes `set(self.signals.keys()) - domains_seen` and returns it sorted. This is the source for the `missing_domains` field in the partial convergence event.

### Task 3 — `check_ticker_convergence_for()` wrapper

Added immediately after `_compute_missing_domains()`. Calls the existing `check_ticker_convergence(min_domains=self.min_domains)` and filters the result list to alerts where any signal's `metadata["symbol"]` matches the requested ticker. Uses `getattr(a, "signals", [])` defensively.

---

## Verification

- `self._bus` confirmed at constructor line 262: `self._bus = event_bus`
- Channel string used literally: `"market.intel.partial_convergence"` (constant to be wired in Round 2 by Builder 4)
- No changes to any confidence/strength calculation path
- No changes to the `return None` exit — partial emission fires before the return, never instead of it
- No new imports required — all types already present

---

## Lines Changed

| Change | Location |
|--------|----------|
| Partial emission block (12 lines added) | ~line 1209, inside `_check_direction_convergence` |
| `_compute_missing_domains` (4 lines) | Before `check_ticker_convergence` |
| `check_ticker_convergence_for` (10 lines) | Before `check_ticker_convergence` |

Total net addition: ~26 lines. File was ~1430 lines, now ~1456. Well under 500-line monolith threshold for this file.

---

## Constraints Met

- ZERO changes to existing convergence behavior
- Partial emission wrapped in try/except pass — cannot block or slow main path
- Channel string used directly (not via constant — Builder 1 adding that in parallel)
- Existing tests unchanged
