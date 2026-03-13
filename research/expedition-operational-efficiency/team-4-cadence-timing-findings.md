# Team 4 — Cadence Conflicts & Timing Findings

**Date:** 2026-03-12
**Angle:** Are MIDGE's cadence changes and timing creating conflicts that prevent learning?

---

## Critical Findings

### Finding 1: Forgetting is gated on outcomes — but outcomes require days. The gate works correctly but never opens.

**Files:** `mae_core/bootstrap/market_hooks_steps_core.py` lines 173–201

The forgetting block (every 75 steps) has a guard:

```python
if current_evaluated > _last_evaluated_count[0]:
    sampler.regime_aware_forget(regime)
    _last_evaluated_count[0] = current_evaluated
else:
    logger.debug("Skipping Thompson forget — no outcomes graded since last forget")
```

This is the right protection — forgetting without learning would erode priors below the 2.0 floor and destroy information. The gate is working as designed. **But the gate never opens** because outcomes require days (minimum 5 days for sec_form8k, up to 90 days for contract/hiring), while MIDGE runs in steps that complete in seconds. Zero outcomes are ever graded within a session, so `current_evaluated` never exceeds `_last_evaluated_count[0]`, so forgetting never fires, so Thompson never moves. This is correct behavior given zero learning inputs — but it confirms that Thompson learning cannot happen within any single daemon session.

### Finding 2: When forgetting DID fire on a prior (before the gate was added), it collapsed toward 2.0, not toward 1.0.

**File:** `mae_core/market/intelligence/thompson_sampler.py` lines 337–338

```python
params["alpha"] = max(2.0, params["alpha"] * decay_factor)
params["beta"] = max(2.0, params["beta"] * decay_factor)
```

At prior Beta(1,1): `1.0 * 0.99 = 0.99` → floored to `2.0`. This actually UPGRADES the prior to Beta(2,2). Beta(2,2) has mean 0.5 (same as Beta(1,1)) but lower variance (more confident). After repeated forgetting cycles on an unlearned distribution, every distribution converges to Beta(2,2). The mean stays at 0.5 but the distribution tightens — meaning Thompson sampling converges toward always returning ~0.5, reducing exploration. The floor prevents collapse but introduces a slow corruption: all unlearned distributions gradually lose their exploration value. Since the gate now prevents this during zero-outcome periods, this is currently harmless.

### Finding 3: Outcome windows are in real calendar days. Steps complete in seconds. There is no mechanism to bridge this.

**File:** `mae_core/market/intelligence/outcome_collector.py` lines 43–56

```
OUTCOME_WINDOWS = {
    "sec_form4": 45,           # 45 real calendar days
    "insider_cluster": 60,     # 60 real calendar days
    "convergence_combo": 14,   # 14 real calendar days
    "pattern_stack": 14,       # 14 real calendar days
    "sec_form8k": 5,           # 5 real calendar days (shortest)
}
```

At `pace=2.0`, each step takes ~2 real seconds. 1 real day = 43,200 steps. The shortest window (5 days = sec_form8k) requires 216,000 steps. The daemon runs to 500 steps in a smoke test. Even long sessions don't approach this. **Outcome grading for any registered prediction requires MIDGE to be restarted days later**, at which point OutcomeTracker.check_pending_outcomes() evaluates predictions whose window has elapsed. This is architecturally correct but means learning feedback takes days per cycle, not minutes.

### Finding 4: Signals are timestamped with fetch time, not data time. Domain windows miscalculate for after-hours operation.

**Files:** `mae_core/market/fetchers_insider.py` line 114, `mae_core/market/apis/price_fetcher.py` lines 366/423/482, `mae_core/market/intelligence/convergence_buffer.py` line 230

All signal timestamps are set to `datetime.now()` at fetch time. When MIDGE runs at 9:30 PM ET, an insider trade that settled at 2 PM gets stamped 9:30 PM. The convergence buffer uses this timestamp against domain windows (e.g., `insider` domain window = 72 hours default). This is not incorrect — the signal *arrived* at 9:30 PM — but it means an insider trade disclosed at 2 PM and fetched at 9:30 PM appears 7.5 hours newer than it is. For fast-expiring domains, this gives phantom freshness to stale data. It does not affect convergence logic today because domain windows are measured in hours (72h default), so a 7.5h offset is minor.

The more significant issue: all signals fetched in one 25-step fetch cycle get very similar timestamps (within seconds of each other), making cross-domain temporal ordering unreliable. `domain_sequence` in convergence alerts (which tracks which domain fired first to compute `sequence_score`) is computed from these fetch-time timestamps, not data-event timestamps. The ordering within a convergence burst is effectively random because all signals from one fetch batch have near-identical timestamps.

### Finding 5: Fetch cadence (every 10 steps) is too aggressive for rate-limited APIs. Starved domains cannot contribute to convergence.

**File:** `mae_core/market/sensing_hook.py` line 104 — `fetch_cadence: int = 10`

With 35 sources and 12 concurrent workers, the full source rotation completes approximately every 94 steps (35 sources / 12 workers * ~32 steps to cycle through). At pace=2.0, this is ~3 minutes of real time. APIs with strict rate limits (yfinance, Finnhub WebSocket, Google Trends) will return errors on many of these calls.

When a fetch fails, `signals = []` and the source contributes nothing to the convergence buffer. If yfinance (which serves the `technical` domain through price/TA data) consistently fails due to 429s, the `technical` domain goes dark. Since convergence requires min_domains=3, losing `technical` as a reliable domain means convergence requires 3 of the remaining working domains. The domains most likely to be starved are: `technical` (yfinance), `sentiment` (StockTwits/Google Trends), `crypto` (CoinGecko rate limits). If 2 of these 3 are dark simultaneously, convergence becomes structurally impossible for any alert that would have included them.

There is no backoff, retry, or rate-limit awareness in the sensing scheduler. Failed fetches are silently dropped.

---

## Root Causes

1. **Learning requires days; sessions last minutes.** The architecture assumes MIDGE runs continuously for weeks. A daemon that runs for hours registers predictions but cannot grade them. Grading requires restarting days later. This is not a bug — it is the design — but it means Thompson learning is epoch-based (cross-session), not session-based.

2. **All signal timestamps are fetch-time, not event-time.** This is a design choice that makes implementation simple but makes temporal ordering within a convergence burst meaningless. Signals from one fetch cycle all share near-identical timestamps.

3. **No rate-limit awareness or circuit breaker in the fetcher scheduler.** Sources that fail with 429 are retried at the same cadence. High-volume domains (yfinance, StockTwits) are hit on every rotation regardless of prior failure.

4. **Forgetting gate correctly blocks zero-learning forgetting, but log messages are DEBUG-level, making this invisible in production output.** When the gate fires, the operator sees nothing. The "Skipping Thompson forget" message is only visible at DEBUG log level.

---

## Recommended Fixes

### Fix 1: Surface the learning gap in dashboard output (trivial, high value)
The forgetting skip log is at DEBUG level. Promote to INFO with a step count and evaluated count. This makes it immediately visible that learning is paused and why.

**File:** `mae_core/bootstrap/market_hooks_steps_core.py` line 193 — change `logger.debug` to `logger.info` for the skip message.

### Fix 2: Run outcome evaluation at startup, not only during session
OutcomeCollector.evaluate() is called on a step cadence. Predictions registered in a prior session may be ready to grade when MIDGE restarts. Add a startup evaluation pass in `market_hooks.py` or bootstrap, before the step loop begins. This closes the cross-session feedback gap and ensures stale predictions don't accumulate indefinitely.

### Fix 3: Add per-source failure tracking and backoff in the sensing scheduler
When a source fails with an exception (especially HTTP 429), increment a per-source failure counter. Skip that source for N rotation cycles proportional to failure count. This prevents wasted API slots on dead sources and preserves quota for working ones.

Rough design: `_source_backoff: Dict[str, int]` — decrement each step, skip dispatch if > 0. On failure: set to 5 (skip 5 rotations). On 429: set to 20.

### Fix 4: Use event-time timestamps where available from API responses
Several APIs return timestamps in their response data (SEC filing dates, Congressional trade dates, COT report dates). Where available, use the data-event timestamp as the signal timestamp, and store fetch-time separately as `received_at`. `enrich_signal()` already has a `received_at` field concept for SEC filings — this pattern should be generalized.

---

## Gaps / Unknowns

- **Pace=2.0 actual step rate**: 2 real seconds per step is an assumption. If the step loop is CPU-bound by other operations, actual step rate may be slower, making the day-to-step conversion less extreme.
- **Whether OutcomeTracker.check_pending_outcomes() is called at startup**: Not verified. If it is already called at startup (not just on step cadence), Fix 2 may already be implemented.
- **Actual 429 rate from yfinance and Finnhub**: The daemon output log would show this, but was not read. The failure mode is inferred from architecture.
- **Whether domain windows for the convergence alerter (72h default) use wall-clock time correctly**: The buffer save/load logic correctly uses wall-clock expiry. This appears sound.

---

## Synthesis

MIDGE's cadence changes did not create new conflicts — they accelerated an existing structural mismatch. The core tension is that Thompson learning is measured in days (outcome windows) but step loops run in seconds. The forgetting gate (which was added to prevent erosion during zero-learning periods) correctly protects Thompson from degrading, but it also means Thompson is completely static during any finite daemon session. Learning can only happen at session restart, when OutcomeCollector grades predictions whose windows have elapsed.

The after-hours timestamp issue is real but minor in magnitude (7.5h offset against 72h windows). The more significant timing problem is within-batch timestamp noise: all signals in one fetch cycle share near-identical timestamps, making temporal domain ordering in convergence alerts structurally unreliable.

The largest operational risk is API starvation: if yfinance (technical domain) is rate-limited during an after-hours session when markets are closed and prices aren't moving, the technical domain goes silent. Without it, convergence alerts that require 3 domains cannot form unless 3 other domains all fire on the same ticker simultaneously — a much rarer event.

The growth sprint cadences (halved) did not break anything structurally. They did increase the frequency of hitting API rate limits, which suppresses signal flow from the most data-rich sources.

**Priority order for fixes:** Fix 2 (startup evaluation — closes the learning loop immediately) > Fix 1 (surface the gap — costs nothing) > Fix 3 (backoff — prevents API waste) > Fix 4 (event-time timestamps — improves temporal ordering quality).
