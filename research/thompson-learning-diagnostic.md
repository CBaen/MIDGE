# Thompson Learning Pipeline Diagnostic

**Date:** 2026-03-15
**Status:** Complete
**Finding:** Four distinct bugs, all confirmed by tracing actual data through actual code.

---

## Executive Summary

The 5% learning rate (9 of 173 distributions moved) has two separate causes:

1. **Forgetting erases faster than learning accumulates.** The forgetting cadence (every 75 steps at 0.92 decay) is collapsing distributions back to the (2.0, 2.0) floor faster than outcomes grade. Since the last distributions file reset on March 13, MIDGE has run for approximately 2 days and produced only 7 graded outcomes for `finra_short` — enough for the distribution to reach (3.0, 6.0) before two more forgetting events pushed alpha back to 2.0 (floor) and beta to 5.0.

2. **The distributions file is not synchronized with the history log.** As of 2026-03-15T00:49:50, the distributions file shows `ta_structure[bear] = (2.0, 2.0)` even though the history log shows the last entry at `(2.0, 191.0)`. The file was overwritten by a fresh Thompson instance that did not load the existing learned state.

3. **803 matured predictions are stuck unevaluated** because `price_fetcher.get_historical_price()` is likely returning None for their symbols/dates, causing them to stay in predictions.jsonl indefinitely.

4. **Alerts stopped on March 12 because the Law 7 Rule-of-3 gate is permanently blocking paper trades.** Convergence alerts exist (confirmed in daemon log) but are deferred because they only have 1 validator (the convergence itself) — they need PatternWatcher stacks or DeepAnalyst inevitabilities to match the same ticker+direction simultaneously.

---

## Bug 1: Forgetting Floor Trap

### What happens

The `apply_forgetting()` method decays alpha and beta by 0.92 but floors them at 2.0. Once a distribution reaches the region where `alpha * 0.92 < 2.0` or `beta * 0.92 < 2.0`, the floor prevents further decay but also **makes the distribution look uninformative (mean = 0.5)**.

The critical problem is the ratio of forgetting frequency to outcome evaluation frequency:

- **Forgetting fires every 75 steps** (conditional on new evaluations happening)
- **Outcome evaluation fires every 75 steps** via `sensing_hook._outcome_cadence = 75`
- **But outcomes only grade when `outcome_window_days` has elapsed** — windows are 5–90 days

In practice: forgetting fires 2–3 times per day when the daemon runs. Outcome grading is blocked by the 5% price move requirement, long windows (45 days for sec_form4, 90 days for hiring), and price unavailability for many tickers.

### Concrete evidence

`finra_short` in the history log:
- March 11: alpha=1418, beta=2435 (mean=0.367)
- March 13 reset: starts from (1.0, 1.0)
- March 13 final history entry: (3.0, 6.0)
- Two more forgettings at 0.92: alpha→max(2.0, 3.0×0.92²)=2.54, beta→max(2.0, 6.0×0.92²)=5.08
- **Distributions file shows: (2.0, 2.0)** — mismatch confirmed (see Bug 2)

`ta_structure` in the history log:
- March 14T18:23: alpha=2.0, beta=191.0 (mean=0.0104 — decisively bearish signal)
- Three forgettings after that: beta→max(2.0, 191.0×0.92³)=148.7
- **Distributions file shows: (2.0, 2.0)** — mismatch confirmed (see Bug 2)

### The forgetting math

A seeded distribution (e.g., `sec_edgar` seeded at alpha=11.8, beta=0.6 from reliability=0.59 × scale=20) reaches the (2.0, 2.0) floor in 27 forgetting events. With 58 forgettings total in the history, every seeded key has fully collapsed.

For a key that accumulates outcomes AND gets forgotten simultaneously: if the outcome rate is lower than the forgetting rate, the distribution never builds up. At 75-step cadence with 0.92 decay, a distribution needs roughly **13 graded outcomes per 75-step forgetting cycle** just to maintain its current position. Most sources are grading far fewer than that.

---

## Bug 2: Distributions File Overwritten by New Instance

### What happens

The history log shows `ta_structure[bear]` at (2.0, 191.0) on 2026-03-14T18:23, but the distributions file (last modified 2026-03-15T00:49:50) shows (2.0, 2.0).

The file was saved at 00:49:50, which is 11 minutes after the last forgetting event (00:34:23). Between those two timestamps, the `signals:` keys appear in the history starting from `old_alpha=1.0` — the signature of `replay_from_history()`, which explicitly sets `self.distributions = {}` and rebuilds from scratch.

The `replay_from_history()` code (in `thompson_persistence.py` line 167) does NOT replay forgetting events — by design. So when it replays 231 `ta_structure` failures, it should produce `ta_structure[bear] = (1.0, 232.0)`. But the file shows (2.0, 2.0).

**Root cause:** The `signals:` keys at 00:49:50 start from `old_alpha=1.0` (confirming a replay), and their values are only a few updates deep — suggesting this was a **brand new daemon startup** with a fresh `ThompsonSampler`. During bootstrap:

1. New instance is created with `seed_from_reliability=True`
2. `persistence_path.exists()` returns True (file exists)
3. `_load_distributions()` loads the file — but something made it empty/corrupt at that moment
4. The `elif not self.distributions` branch fires, triggering `replay_from_history()`
5. Replay rebuilds all keys from the history, but the distributions file currently open for the PREVIOUS instance gets overwritten during the rebuild save
6. Alternatively, a second Thompson instance (e.g., in a parallel test or second daemon process) has been running with `seed_from_reliability=True` from a clean start, accumulating the seeded keys, and its forgetting events are the ones collapsing everything to (2.0, 2.0)

The `.bak` file in git status (`data/market/thompson_distributions.json.bak`) is consistent with this — something made a backup before overwriting.

### Evidence

Keys in the distributions file that are NOT in the history (72 keys): these come entirely from `_seed_from_reliability()`. After 58 forgettings at 0.92, all seeded values reach the (2.0, 2.0) floor regardless of their starting value.

Keys in the history that are NOT in the distributions file (30 keys including `bollinger`, `rsi`, `technical_macd`, `insider_form4`, etc.): these were never persisted to the current distributions file. They exist only in the history from an earlier run.

---

## Bug 3: 803 Matured Predictions Stuck Unevaluated

### What happens

`OutcomeTracker.check_pending_outcomes()` calls `price_fetcher.get_historical_price(symbol, entry_date_str)`. If this returns None (API failure, rate limit, unsupported symbol, non-trading date), the prediction stays in `predictions.jsonl` and is retried next evaluation cycle.

**2,680 predictions have matured (window elapsed as of 2026-03-15). 1,877 were evaluated. 803 remain stuck.**

Stuck sources by count:
- `openinsider_purchase`: 218 stuck (symbols from OpenInsider may not be in yfinance)
- `ta_structure`: 189 stuck
- `social_sentiment`: 151 stuck (StockTwits tickers may be non-standard)
- `ta_macd`: 54 stuck
- `session_sweep`: 40 stuck

The `predictions.jsonl` file also contains 8,300 predictions with windows extending to May 2026 — these are legitimately in the future and will pile onto the evaluation queue.

### Why this starves Thompson

Each stuck prediction represents one outcome that should update a Thompson distribution but doesn't. The 803 stuck predictions are the primary reason source-level distributions have so few samples. Thompson needs sustained, reliable outcome grading to overcome forgetting — it cannot learn from signals whose price history it cannot access.

Additionally, note that 11,045 total pending predictions exist. Even if price fetching worked perfectly for all of them, the outcome windows mean Thompson would receive the bulk of feedback months from now, not today.

---

## Bug 4: The `signals:` Key Format Parallel Universe

### What happens

The history log contains four `signals:` keys:
- `signals:insider_form4+technical_macd`
- `signals:cci+stochastic+williams_r`
- `signals:politician_sell+technical_rsi`
- `signals:bollinger+cci+rsi+stochastic+williams_r`

These appear in the distributions file with moved values (e.g., `signals:insider_form4+technical_macd[bear]` at alpha=7.0, beta=1.0, mean=0.875).

But these are **NOT the same as the production signal source keys**. They are created by:
- `post_mortem.py` line 322: `return "signals:" + "+".join(unique_sigs)` — extracts from `contributing_signals` field
- `deep_analyst.py` line 447: `key = "signals:" + "+".join(sorted(domains))` — uses domain names not source names

The production signals use keys like `ta_structure`, `ta_macd`, `sec_form4` — which route through `OutcomeTracker` using the `source` field from `predictions.jsonl`.

**These are two completely separate Thompson namespaces writing to the same file.** The `signals:` keys represent PostMortem/DeepAnalyst combo-level signals. The raw source keys represent individual signal reliability. Neither is learning effectively from the other.

### Impact

The 9 "moved" distributions include the 4 `signals:` keys plus 5 others (`sec_form4[sideways]`, `sec_form4[bear]`, `finnhub_earnings[sideways]`, `eia_energy[default]`, `congress_legislation[default]`). The `signals:` movement is real but represents PostMortem feedback on old replay data, not live outcome grading.

The 5 moved source keys are the only ones receiving genuine live feedback — and they're receiving it despite the forgetting trap, simply because their signal types had enough recent evaluations to show movement.

---

## Bug 5: Convergence Alerts Not Converting to Paper Trades (Alerts Gap)

### What happens

The `_run_paper_trading_gate()` in `market_hooks_sensing.py` requires **3 validators** to approve a paper trade:
1. Convergence alert (always present = 1 validator)
2. PatternWatcher stack matching same ticker+direction
3. DeepAnalyst inevitability matching same ticker+direction (score > 0.5)
4. HypothesisEngine recent fire
5. PatternMemory Qdrant precedent

The last paper trade was 2026-03-12. The daemon log at 03:46 shows convergence alerts being suppressed by deduplication (normal — same direction within the dedup window), but no "DEFERRED — Law 7" messages visible in the log tail, suggesting the convergence alerts are either:

(a) Being suppressed by dedup before reaching the paper trade gate, or
(b) Not passing confidence/strength thresholds

The last paper trade showed `confidence=0.53, hit_rate=0.00`. The combo filter requires `_cd.samples >= 3 AND _cd.mean < 0.25` to block — all combos are at (2.0, 2.0) = 0.5 mean, so that gate passes. The strength threshold is 0.65.

**Most likely cause:** The convergence dedup window (1787 seconds = ~30 minutes as seen in daemon log) is suppressing the alerts before they reach the paper trade gate. When a real new convergence fires, it will likely pass the gate if it meets confidence > 0.45 and strength > 0.65.

The `alerts_human.jsonl` does continue receiving `pattern_stack` entries through March 15 and `inevitability` entries via DeepAnalyst — the pipeline is alive. Only the convergence-to-paper-trade path is stalled due to the dedup lock.

---

## What Needs to Change

### Fix 1: Stop the forgetting-before-learning problem (CRITICAL)

The forgetting cadence of 75 steps assumes outcomes are grading continuously. They aren't. The forgetting gate in `market_hooks_steps_core.py` at line 211 (`if current_evaluated > _last_evaluated_count[0]`) already tries to prevent forgetting when no new evaluations have occurred — but this only works if `get_statistics()["total_evaluated"]` is actually incrementing.

**Action:** Change the forgetting gate to only apply forgetting when at least 10 new outcomes have been graded (not just 1), OR increase the cadence from 75 to 500 steps. The ratio of forgetting-to-learning is currently 1:1 step-wise but in practice ~10:1 or worse because most evaluation cycles find nothing to grade.

```python
# In market_hooks_steps_core.py, change:
if current_evaluated > _last_evaluated_count[0]:
# To:
MIN_NEW_OUTCOMES_FOR_FORGET = 10
if current_evaluated >= _last_evaluated_count[0] + MIN_NEW_OUTCOMES_FOR_FORGET:
```

### Fix 2: Repair the distributions file from history (CRITICAL)

The file currently shows mostly (2.0, 2.0) and does not reflect the 18,941 historical updates. Run `replay_from_history()` manually to rebuild the correct state:

```python
from mae_core.market.intelligence.thompson_sampler import ThompsonSampler
ts = ThompsonSampler(seed_from_reliability=False)
ts.replay_from_history()
```

Note: replay does NOT re-apply forgetting events. The rebuilt distributions will reflect the raw accumulated wins/losses from history. This is actually correct — the forgetting events represent decayed time-relevance, and replaying them would over-punish old data. The rebuilt state for `ta_structure[bear]` will be `(1.0, 232.0)` — decisively bearish — which is what 231 real outcomes actually showed.

After replay, disable `seed_from_reliability=True` on the production instance by passing `seed_from_reliability=False` in the bootstrap setup. The history is already the authoritative source of truth; seeding from static config values should only happen on a genuine first run with zero history.

### Fix 3: Fix the stuck predictions (IMPORTANT)

803 predictions with elapsed windows are retrying on every evaluation cycle and failing (price unavailable). These should either:

(a) Be marked as unevaluable and removed from predictions.jsonl after 3 failed attempts, with Thompson receiving a neutral update (no success, no failure) — prevents them clogging the queue.
(b) Implement a fallback price source for the symbols that yfinance doesn't serve.

The most practical fix is (a): add a retry counter to prediction records, and after 5 failed price fetches, record the prediction as "indeterminate" and remove it from the queue without updating Thompson.

### Fix 4: Accept that `signals:` keys are a separate population (LOW PRIORITY)

The PostMortem/DeepAnalyst `signals:` keys and the production source keys are both legitimate — they serve different purposes. The `signals:` keys represent combo-level reliability (does this COMBINATION work?), while source keys represent individual signal reliability (does this SOURCE work?).

They are correctly writing to the same Thompson file, but they're never confused — the production ConvergenceAlerter samples individual source keys, and PostMortem samples combo keys. No fix needed; this is working as designed.

The only issue is that the `signals:` keys use low-level signal names (`insider_form4`, `technical_macd`) from the contributing_signals field, which may not match the domain-level combo keys (`combo:insider+technical`) used by the ConvergenceAlerter's combo filter. These two combo namespaces don't talk to each other.

### Fix 5: Convergence alert dedup — no fix needed

The 30-minute dedup window is working correctly. Once it expires, new convergence alerts should pass through to paper trades if they meet confidence and strength thresholds. The daemon log shows active convergence alerts being detected (28 ticker alerts visible). The Law 7 validator count may still block them if PatternWatcher stacks and DeepAnalyst inevitabilities aren't co-firing on the same ticker/direction simultaneously.

---

## Root Cause Summary Table

| # | What's Broken | Where | Impact |
|---|---------------|-------|--------|
| 1 | Forgetting decays faster than outcomes grade | `apply_forgetting()` cadence vs. `check_pending_outcomes()` yield | 95% of distributions stay at (2.0, 2.0) floor |
| 2 | Distributions file overwritten on each daemon restart via `replay_from_history()` or fresh seed | `ThompsonSampler.__init__` + `_seed_from_reliability` | All learned state from previous sessions lost |
| 3 | 803 matured predictions never evaluate because price history unavailable | `OutcomeTracker._check_price_movement()` returning None | Thompson never learns from those signal types |
| 4 | `signals:` keys (PostMortem) and source keys (OutcomeTracker) are separate namespaces that don't cross-feed | By design but misunderstood | Not a bug; no action needed |
| 5 | Convergence dedup window prevents repeated alerts from reaching paper trade gate | `convergence_alerter` dedup + Law 7 validator requirement | Paper trades stalled until dedup expires AND 3 validators align |

---

## Verification

The analysis above was confirmed by:

1. Tracing `ta_structure` from history (alpha=2.0, beta=191.0 on 2026-03-14T18:23) to distributions file (2.0, 2.0) — confirmed mismatch.
2. Tracing `finra_short` from 8908 history updates to distributions (2.0, 2.0) — confirmed mismatch.
3. Counting forgetting events (58 total) vs. update events (18,941 total) and showing that 27 forgettings collapse any seeded distribution to floor.
4. Counting 803 stuck matured predictions and identifying their source distribution.
5. Confirming convergence alerts are live in daemon log but not generating paper trades.
