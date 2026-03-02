# MIDGE Architecture Audit — Beta Revision (Systems Reliability Engineer)

**Auditor Role:** Witness Beta — Systems Reliability Engineer
**Revision Date:** 2026-03-01
**Based On:** Original findings, Beta cross-review, Lead cross-review, Alpha cross-review

---

## Preface: How I'm Approaching This Revision

I entered this audit as a systems reliability engineer: find what breaks under load, find the concurrency hazards, find the persistence failures. The cross-reviews challenged me on three things: whether my sequencing concern about building the output path before fixing data integrity was too conservative, whether my uncertainty on finra_short was justified, and whether my "profile before you strip" position on organism overhead is operationally sound. I've done additional codebase research before taking positions on each.

The most important new finding in this revision is about finra_short. I raised the base-rate question in my cross-review and neither Lead nor Alpha could answer it. I can now answer it with evidence from the codebase.

---

## Part 1: Point-by-Point Response to Cross-Review Challenges

### Challenge 1: Lead concedes CF-1 (Thompson JSONL lock) is better-supported than their own Issue 2

**Lead's cross-review position:** "Beta's CF-1 analysis is better-supported than my Issue 2 assessment. The dedup race I found is real, but Beta's Thompson thread-safety finding is more pervasive and already has a confirmed production failure."

**My response: STAND FIRM, but now with triadic consensus.**

This is not a challenge — it is agreement, and strong agreement. Lead's concession is honest and well-reasoned. I accept it. The convergence of all three auditors on CF-1 as the most operationally urgent reliability issue is itself meaningful: it is the only finding with both confirmed prior occurrence (per MEMORY.md) AND an unfixed root cause (the lock gap in `_log_update()` at thompson_sampler.py line 272 is demonstrably present in the current code).

My original finding stands without revision. The asymmetry between `_save_distributions()` (locked) and `_log_update()` (unlocked) is the bug pattern that previously required a full Thompson rebuild from 9,462 outcomes. It is unambiguously the highest-priority reliability fix.

---

### Challenge 2: Alpha stands firm on finra_short as anti-signal, Lead concedes it is actively harmful

**Alpha's cross-review position:** "35.8% win rate is not 'slightly below neutral,' it is meaningfully worse than random in a binary 5% directional prediction at the implied threshold."

**Lead's cross-review position:** "Alpha is right that this requires either a direction flip or exclusion, not just a down-weight."

**My response: REVISE — I now have evidence that contradicts both conclusions. The finra_short base rate question I raised is answerable, and the answer changes everything.**

I researched the outcome_collector.py and outcome_tracker.py to understand exactly how success is determined for finra_short predictions. Here is what I found:

**The actual mechanics in outcome_tracker.py (lines 291-302):**

```python
magnitude_ok = abs(pct_change) >= self.min_price_move_pct

if direction == "up":
    direction_ok = pct_change > 0
elif direction == "down":
    direction_ok = pct_change < 0
else:
    # No direction specified — magnitude alone determines success
    direction_ok = True

success = magnitude_ok and direction_ok
```

**The actual finra_short outcomes in data/market/outcomes.jsonl (lines 99-108):**

```json
{"source": "finra_short", "symbol": "AAPL", "direction": "", "price_change_pct": 0.8314, "success": false}
{"source": "finra_short", "symbol": "BA",   "direction": "", "price_change_pct": -7.9853, "success": true}
{"source": "finra_short", "symbol": "AVGO", "direction": "", "price_change_pct": 21.0094, "success": true}
{"source": "finra_short", "symbol": "CRM",  "direction": "", "price_change_pct": -5.2644, "success": true}
```

**The direction field is empty for every finra_short record.** This means finra_short is evaluated as **direction-agnostic**: "success" = the stock moved 5% in EITHER direction within the outcome window. The Thompson update counts ANY 5% magnitude move as a success, regardless of which way the stock went.

Alpha and Lead's entire "anti-signal" analysis rests on the premise that finra_short is predicting direction and getting it wrong 64.2% of the time. That premise is false. finra_short is not predicting direction at all. It is predicting volatility — "this heavily-shorted stock will move significantly."

**What does 35.8% mean when direction is empty?**

The relevant base rate is: "what fraction of large-cap stocks in the watchlist move 5% in either direction within 14 days?" This is the `yfinance_price` distribution, which is the most data-rich non-directional price signal in the system. From thompson_distributions.json:

- `yfinance_price` default: alpha=85.04, beta=285.43 → mean = 0.230 (23%)
- `yfinance_price` sideways: alpha=143.16, beta=500.89 → mean = 0.222 (22%)

finra_short at 35.8% success for a direction-agnostic 5% move compares to a base rate of roughly 22-23% for the same measurement. That is a 13-15 percentage point lift over baseline on 1,265 samples. This is a statistically significant positive signal, not an anti-signal.

**Why Alpha's "anti-signal" conclusion is wrong:**

Alpha compared 35.8% to 50% (coin flip), treating finra_short as a directional predictor. But 50% is the wrong baseline for a non-directional 5% move. If the base rate for a 5% move in either direction is 23%, then finra_short at 35.8% represents a 13-point lift. The Thompson weight of 0.858 is not insufficient neutralization of an anti-signal — it is modest underweighting of a positive signal.

**Why Lead's "marginal edge" framing is also wrong but in the right direction:**

Lead was appropriately uncertain ("marginal edge") but still implicitly compared against the directional 50% baseline. The correct comparison is against the direction-agnostic base rate, which makes finra_short a genuine positive-edge signal.

**The correct action:**

Do NOT flip finra_short. Do NOT exclude finra_short. The handling is correct but the interpretation has been wrong. The fix is to update the domain assignment so finra_short contributes to convergence correctly as a volatility signal (not a directional signal), and to ensure that when convergence_alerter.py combines finra_short with directional sources, it does not treat the direction-agnostic finra_short as confirmation of direction.

There is one genuine problem: the signal_adapters likely convert finra_short into a directional signal when they shouldn't. Let me note this as a specific item to check.

**Revised risk assessment for finra_short:** POSITIVE EDGE SOURCE (incorrectly analyzed as anti-signal by all three auditors). Priority action: verify that the finra_short adapter assigns empty direction when recording predictions. If adapters are injecting a direction, that is the actual bug — not the signal itself.

---

### Challenge 3: Lead disagrees with my bypass-path concern (Divergence C), Alpha partially agrees

**Lead's cross-review position:** "Both changes should be implemented — Lead's bypass for backtested edge AND Alpha's domain independence enforcement."

**Alpha's cross-review position:** "Lead's Option B fix is better-supported for near-term financial utility. My fix (removing noise sources from voting) is better for long-term signal quality."

**My response: PARTIAL REVISION — I soften my bypass-path concern but maintain the sequencing caveat.**

My original position was that a bypass path "will expand" and become three tracks. Alpha's cross-review actually walks this back from their original position: Alpha now says Lead's bypass is better for near-term utility and their own noise-source removal is better for the long term. Both cross-reviews acknowledge both changes should happen.

I am revising my position from "bypass introduces architectural debt that will compound" to "bypass is acceptable if time-bounded." Specifically: the session_sweep_ifvg bypass should be implemented with an explicit constraint that any new source added to the bypass list requires a minimum of 30 days of real-market backtest validation and a formal review. Written as a code comment in the bypass path, not just a convention. This makes the expansion concern auditable.

I still stand firm that Alpha's domain independence analysis is architecturally more rigorous than the bypass approach. The bypass makes session_sweep_ifvg fire now. Domain independence makes the entire convergence system sound. Both are needed; neither alone is sufficient. My sequencing: domain independence audit first (it may reveal that session_sweep_ifvg already satisfies a genuine 3-domain requirement via technical + volatility + fundamental once domains are correctly bucketed), then bypass as a fallback if it doesn't.

---

### Challenge 4: Alpha's organism overhead claim — my "profile before you strip" position challenged

**Alpha's cross-review position:** "My framing is correct but my 60% estimate is rough. Beta found specific hot paths; I identified the general overhead category. Combined they are stronger than either alone."

**Lead's cross-review position:** "The reward misalignment Beta found is more financially dangerous than the compute cost — agents are learning wrong priorities."

**My response: STAND FIRM on profiling-first, REVISE on reward misalignment.**

On the organism tax profiling question: I stand firm. Alpha explicitly concedes the estimate is rough without profiling data. My PA-1 through PA-3 findings give more specific evidence than Alpha's system-count-based estimate. My position was: run the StepTimer marathon session first, identify confirmed hot paths, then remove surgically. No cross-review has presented evidence that changes this. Alpha's acknowledgment ("Beta found specific hot paths") actually supports my position.

On reward misalignment (Alpha Kill Item 7, surfaced in Lead's cross-review): I revise my assessment upward in severity. In my original findings I identified the market_actions.py dispatch system but did not examine the reward ceiling values. Both Lead and Alpha independently identified the ceiling problem: market agents cap at 0.5 reward while TaskPool exploit returns up to 1.0. This is a training signal that teaches the VDN Q-table over thousands of steps that abstract task completion is twice as valuable as market intelligence. This is a direct financial alignment failure, not just an organism overhead concern.

Revised position: the reward ceiling fix (raise market action ceiling to 0.8, cap TaskPool for market-role agents at 0.3) should be Priority 4 in my list, not omitted. It is more immediately financial than most persistence issues because it actively shapes what the learning system values.

---

### Challenge 5: The deduplication race — Alpha says it is worse than I rated it

**Alpha's cross-review position:** "Beta's 'currently safe' assessment is too reassuring given that the corruption already happened in production. Lead's monotonic clock + lock fix is correct regardless of the concurrency path."

**My response: REVISE — Alpha is right.**

My original RC-1 assessment said the deduplication problem was "currently safe, but fragile" because collection is main-thread-only. I was technically precise but operationally wrong. Lead's discovery_log.jsonl evidence (20+ identical CONV-20260227-0001 alerts in one second) is confirmed production corruption. The storm happened. My "currently safe" framing was too narrow — I was thinking about threading paths and missed that the storm can occur in a fast single-threaded step loop when check_convergence() is called multiple times within a single wall-clock second.

Revised assessment for RC-1 / deduplication: **CONFIRMED OCCURRED.** Upgrade from LOW risk to HIGH risk. The fix (threading.Lock() around the check-and-update, using time.monotonic() with persisted state across restarts) is correct. I stand by my failure-mode analysis of monotonic clock drift across restarts — that specific detail needs the persisted timestamp fix alongside the monotonic clock usage.

---

### Challenge 6: My finra_short base rate question — can I answer it?

**My original question from the cross-review:** "What is the base rate of a 5% move in either direction for any given ticker in 45 days? If the base rate is 25%, finra_short at 35.8% is positive edge. Neither Lead nor Alpha computes this base rate."

**Answer: YES. I can answer it now. See Challenge 2 above for full analysis.**

Short version: finra_short is direction-agnostic (confirmed in outcomes.jsonl). The base rate for a direction-agnostic 5% move is approximately 22-23% (confirmed from yfinance_price Thompson distribution). finra_short at 35.8% is a 13-15 percentage point positive lift. It is not an anti-signal. Alpha and Lead's "flip or exclude" recommendation was based on a false premise about direction.

---

## Part 2: What the Cross-Reviews Found That Changes My Original Analysis

### New Evidence That Upgrades My Risk Assessments

**1. Alert storm is confirmed production failure (RC-1 upgraded to HIGH)**

Lead's discovery_log.jsonl evidence of 20+ identical alerts is production evidence, not theoretical risk. My original RC-1 was rated LOW because I was thinking about threading. Alpha correctly identified the single-threaded path. I am upgrading RC-1 to HIGH based on confirmed occurrence.

**2. Reward misalignment is a financial alignment failure (new finding, Priority 4)**

I did not examine the reward ceiling values in my original findings. Both Alpha and Lead identified this. The VDN Q-table is being trained across thousands of steps to value TaskPool exploitation at up to 1.0 while market intelligence caps at 0.5. This is not a performance issue — it is the Q-learner actively learning the wrong objective. It should be fixed before any paper trading is built on top of this agent pool.

**3. Static confidence values in adapters compound Thompson weight errors (from Lead)**

Lead traced that `from_congressional_trade` sets static confidence=0.75, while Thompson shows mean=0.164. The confidence formula uses the static value as input and applies Thompson weight as a multiplier — so 0.75 × 0.664 = 0.498 still contributes misleadingly high confidence. I had looked at the Thompson distributions but had not traced this formula interaction. This is now part of my data integrity picture.

### What Changed My View on finra_short (Most Important Revision)

I raised the base rate question. I researched the answer. The answer reverses the consensus of all three auditors on finra_short. Specifically:

- Alpha called it "anti-signal" based on 35.8% < 50% for directional prediction
- Lead called it "marginal edge"
- I called for base rate research before concluding
- The research shows: finra_short is direction-agnostic, making 50% the wrong baseline; the correct base rate (from yfinance_price distribution) is ~22-23%; finra_short at 35.8% is a genuine positive-edge signal

The correct action is NOT to flip or exclude finra_short. The correct action is to verify the signal adapter assigns empty direction (preserving direction-agnosticism) and to ensure convergence_alerter.py handles direction-agnostic signals correctly when computing joint confidence with directional sources. These are small, targeted fixes — not the architectural change Alpha and Lead recommended.

---

## Part 3: Where I Stand Firm Against All Cross-Reviews

### Stand Firm A: Build the output path AFTER fixing data integrity, not before

All three cross-reviews and all three original findings agree: build the paper trading output path. Lead, Alpha, and the cross-reviews converge on this as Priority 1.

I stand firm that it should not be built before the Thompson lock gap (CF-1) is fixed. My reasoning remains unchanged: building an output path before fixing the confirmed production bug that previously corrupted the Thompson distributions creates a paper trading system that immediately learns from corrupted beliefs. The Thompson distributions are what differentiate MIDGE from a 200-line script. Corrupted distributions produce a paper trading book with systematically wrong confidence weights on every signal.

The sequencing argument does not require much time: CF-1 is one line of code (add `_log_update()` inside `self._lock`). It is done first, then the output path is built. The delay is hours, not weeks.

### Stand Firm B: Organism tax requires profiling data before removal

No cross-review presented profiling data that would change my position. Alpha concedes the 60% estimate is rough. My specific complexity findings (PA-1: CorrelationTracker O(n² × m²), PA-2: prune on every record_signal) give more concrete evidence than Alpha's system-count-based estimate. Until a StepTimer marathon run produces actual latency data, surgical removal risks cutting diagnostic coverage (the renal_filter, nociception system) without knowing what the actual bottleneck is.

### Stand Firm C: The TOCTOU race in hypothesis registry (RC-2) is medium risk, not disposable

Neither Lead nor Alpha found this finding independently, and neither cross-review challenged it directly. I stand firm: background validation thread + agent-triggered validation calling `self._promote()` / `self._retire()` without a lock between the check and mutation is a TOCTOU race. In the current production system with 0 hypothesis promotions, the race rarely fires. As the system matures and promotions begin occurring, this race will materialize. It should be fixed before the hypothesis lifecycle produces live trading signals.

### Stand Firm D: Monotonic clock drift on restarts must be addressed alongside the dedup fix

Both Alpha and Lead support monotonic clock for the deduplication fix. I stand firm on the detail: `time.monotonic()` resets on process restart. If the dedup state is not persisted to disk, every restart resets the dedup timer to 0, giving a 4-hour unprotected window (the default dedup interval). The fix needs `time.monotonic()` for thread safety AND `datetime.now()` persisted to disk for restart safety. Not monotonic alone.

---

## Part 4: Revised Top 5 Priority List for Making MIDGE Financially Effective

These priorities incorporate all three cross-reviews, the confirmed production evidence, and the finra_short base rate research that changes the consensus analysis.

---

### Priority 1: Fix Thompson Thread-Safety and Purge Data Contamination

**What changed from cross-reviews:** Unanimous triadic agreement confirms CF-1 is the most operationally urgent issue. Lead's concession makes this three-way confirmed. The data contamination component (Lead Issue 5, Beta CF-3) was already in my original sequencing — the cross-reviews add evidence it is already affecting production.

**Specific fixes in order:**

1. **CF-1** — Add `_log_update()` call inside `self._lock` in `thompson_sampler.py` line 272. One line. This prevents the next marathon file-lock corruption event.

2. **CF-2** — Replace `Path.write_text()` with atomic `os.replace(tmp, path)` in `learning_config.py`, `hypothesis_engine.py`, `hypothesis_generator.py`, `step_timer.py`. The pattern already exists correctly in `_save_distributions()` — apply it consistently.

3. **CF-3** — Delete "a|b", "a0|b0"..."a6|b6" from `data/market/pair_outcomes.json`. Add a pytest fixture isolation layer to prevent test writes reaching production data paths.

4. **Data purge** — Remove mock-format outcomes from `outcomes.jsonl` (records with `was_correct` field and impossible returns like 49.93%). Remove the 2027-timestamp prediction from `registered_signals.json`. Rebuild Thompson distributions from cleaned outcomes via the existing rebuild procedure.

**Financial mechanism:** Thompson distributions are the Bayesian confidence weight for every signal in the convergence geometric mean. Corrupted distributions produce wrong confidence scores on every alert. The dedup storm (20+ identical alerts confirmed) produces corrupted RSI Layer 2 training data. These are not housekeeping — they are the validity condition for every downstream learning operation.

---

### Priority 2: Build the Paper Trading Output Path

**What changed from cross-reviews:** Lead, Alpha, and I all agree this is required for financial utility. My sequencing position (after CF-1) is maintained — with CF-1 as a one-line fix, the delay before building the output path is hours, not weeks. The cross-review failure mode analyses I wrote (alert storm to trade storm, Kelly with no account size, concurrent writes) remain valid prerequisites for the implementation.

**Specific build:**

1. In `sensing_hook.py:_collect_one()`, after `check_convergence()` returns alerts: if `alert.confidence > 0.75` AND `alert.strength > 0.65`, instantiate `TradeSignal`, write to `data/midge/paper_trades.jsonl`.
2. Wire Kelly fraction from `ctx._latest_kelly`. Establish a paper account value (e.g., $100,000) as configuration, not hardcoded.
3. Wire `OutcomeCollector` to read paper_trades.jsonl exit prices from yfinance and compute dollar P&L alongside the existing directional success/failure.
4. Fix the dedup race (Priority 2a below) before wiring the output path.

**Priority 2a (prerequisite):** Fix the alert deduplication race with `threading.Lock()` + `datetime.now()` (persisted) + `time.monotonic()` for intra-process thread safety. The 20+ duplicate alert storm confirmed in production will produce 20+ simultaneous paper trade entries without this fix.

**Financial mechanism:** This closes the feedback loop. Every other component (Thompson Sampling, DSR-gated hypothesis lifecycle, Kelly sizer) exists to serve a trading outcome that currently does not materialize. Paper trading at zero cost makes MIDGE's learning verifiable against dollar P&L for the first time.

---

### Priority 3: Correct the finra_short Analysis and Fix Direction-Agnostic Signal Handling

**MAJOR REVISION from cross-reviews.** Alpha and Lead recommended flipping or excluding finra_short. I am recommending neither. Here is the evidence:

**From outcome_tracker.py lines 291-302:**
```python
else:
    # No direction specified — magnitude alone determines success
    direction_ok = True
```

**From data/market/outcomes.jsonl (every finra_short record):**
```json
{"source": "finra_short", "direction": "", "success": true/false}
```

finra_short predictions carry empty direction. "Success" means the stock moved 5% in EITHER direction. Alpha and Lead's anti-signal analysis compared 35.8% against a 50% directional baseline — but that is the wrong baseline. The correct baseline is the direction-agnostic 5% move rate.

**The base rate evidence:** `yfinance_price` in `thompson_distributions.json` shows alpha=85.04, beta=285.43 → mean=0.230 for the default regime. yfinance_price signals are the most data-rich market-movement signals evaluated against the same 5% threshold. At 22-23% base rate for a 5% move in 14 days, finra_short at 35.8% represents a 13-15 percentage point lift over baseline on 1,265 samples. This is a real positive-edge signal.

**The correct fixes:**

1. Verify that the finra_short signal adapter correctly passes `direction=""` (empty) all the way from `from_finra_short()` adapter through `OutcomeCollector.register_signals()`. If any intermediate step injects a direction ("down" because high short interest implies bearish), that is the bug — it would make finra_short incorrectly directional.

2. In `convergence_alerter.py`, when computing joint confidence for a convergence alert that includes a direction-agnostic source like finra_short, the directional confidence contribution should be treated as neutral (not reinforcing the directional bias of other sources). The geometric mean formula currently applies finra_short's Thompson weight to the directional confidence value from the adapter — if the adapter is setting a directional confidence (e.g., confidence=0.72, direction="down"), the system is incorrectly treating finra_short as directional evidence.

3. Update the learning_config.py `finra_short` source reliability default to reflect its actual positive-edge status (~0.36 mean at the direction-agnostic baseline), not the 0.75 default that assumes directional accuracy.

**Financial mechanism:** finra_short is MIDGE's most data-rich signal. Incorrectly treating it as an anti-signal or direction predictor corrupts the confidence calculation for every multi-domain alert where it participates. Correctly treating it as a volatility signal — "something significant is going to happen to this stock" — allows it to serve as a legitimate catalyst indicator for convergence alerts that then look to directional sources for direction.

---

### Priority 4: Fix Agent Reward Misalignment and Alert Deduplication

**New priority from cross-reviews.** Alpha and Lead both identified the reward ceiling problem; I had identified the dispatch mechanics but not the reward values. This is now incorporated.

**4A: Reward misalignment (Alpha Kill Item 7, confirmed by Lead):**

For agents in market roles (SEC_WATCHER, CONTRACT_TRACKER, MARKET_ANALYST, HYPOTHESIS_EXPLORER, HYPOTHESIS_VALIDATOR): cap TaskPool exploit reward at 0.3, raise market action ceiling to 0.8. The current calibration (market actions max 0.5, TaskPool exploit up to 1.0) is training the VDN Q-table over thousands of steps to prefer abstract task completion over market intelligence by a factor of 2:1. This is a foundational learning alignment failure.

**4B: Alert deduplication (RC-1, upgraded from LOW to HIGH):**

The deduplication race is confirmed as a production failure (20+ identical alerts in discovery_log.jsonl). Fix with `threading.Lock()` protecting the check-and-update. Use `datetime.now()` for the persisted comparison (survives restarts) alongside thread-local monotonic comparison for intra-process safety. `time.monotonic()` alone is insufficient because it resets on process restart, creating a 4-hour unprotected window after every restart.

**4C: Static confidence disconnect (Lead Issue 6):**

Update `learning_config.py source_reliability` defaults to match actual Thompson distribution means: `congressional = 0.20` (not 0.75, actual Thompson mean = 0.164), `sec_edgar` deprecated in favor of `sec_form4`. The static confidence values in signal adapters (`from_congressional_trade` sets confidence=0.75) combined with wrong config priors inflate confidence in early-session steps before distributions warm-load.

**Financial mechanism:** Reward misalignment shapes what market-role agents learn to optimize over thousands of steps. Deduplication prevents the paper trading book (Priority 2) from receiving 20× the correct number of trade entries on the first convergence alert. Static confidence fixes clean up the one pre-Thompson-warmup window where initial-session alerts can carry misleading confidence values.

---

### Priority 5: Enforce Genuine Domain Independence + Structural Persistence Hardening

**5A: Domain independence (Alpha Section 3, Lead Priority 5, synthesized):**

Add an `independent_domains` super-category constraint to `ConvergenceAlerter`. The 3 required domains must span at least 2 of: `{fundamental, technical, sentiment, macro}` where:
- fundamental = [insider, government, regulatory, contract]
- technical = [technical, volatility]
- sentiment = [sentiment, social, social_sentiment, stocktwits_sentiment, google_trends]
- macro = [positioning, macro]

Three sentiment sources (social_sentiment + stocktwits_sentiment + google_trends) cannot satisfy the 3-domain requirement regardless of domain name count. This is the difference between Law 2 compliance by letter and by spirit.

Separately: implement Lead's Option B parallel direct-output path for session_sweep_ifvg when quality >= 0.65, with an explicit documented constraint that any new source added to the bypass list requires 30-day real-market validation evidence. This makes the bypass auditable.

**5B: Structural persistence hardening (CF-2, PV-2, PV-3):**

- Apply atomic `os.replace(tmp, path)` pattern consistently across all four files that currently use `write_text()` (learning_config.py, hypothesis_engine.py, hypothesis_generator.py, step_timer.py).
- Add hypothesis registry compaction: snapshot current state to a baseline JSON on every N-th startup (or when file exceeds 10MB), then replay only events since the snapshot. This prevents O(n) startup replay from growing indefinitely.
- Add a maximum age prune to `self._registered` in OutcomeCollector: remove signal IDs older than max(OUTCOME_WINDOWS) = 90 days. The set currently grows forever; predictions older than 90 days are guaranteed resolved and their IDs no longer serve deduplication.

**Financial mechanism:** Domain independence makes the convergence alert a genuine multi-source signal rather than a threshold easily satisfied by correlated noise. The structural persistence hardening prevents the class of failures (CF-2, PV-2, PV-3) that would degrade learning state on any session crash — and crashes will occur once paper trading introduces network-dependent operations (yfinance exits) into the session loop.

---

## Summary: What Changed, What Didn't

| Point | Original Position | Revised Position | Evidence That Changed It |
|-------|------------------|------------------|--------------------------|
| CF-1 priority | Critical | Critical — triadic consensus | Lead concedes, Alpha concedes |
| finra_short | "Research base rate before concluding" | POSITIVE EDGE SIGNAL — do not flip or exclude | outcomes.jsonl confirms direction="", yfinance_price provides base rate ~22-23% |
| RC-1 dedup risk | LOW (currently safe) | HIGH (confirmed occurred) | Lead's discovery_log.jsonl evidence of 20+ duplicate alerts |
| Bypass path | Architectural debt concern | Accept with documented constraint | Alpha+Lead cross-review synthesis |
| Organism tax | Profile first | Profile first (unchanged) | Alpha concedes estimate is rough |
| Reward misalignment | Not in original findings | Priority 4A — learning alignment failure | Alpha + Lead cross-review both identify it |
| Static confidence disconnect | Not examined | Priority 4C | Lead Issue 6 |
| Output path sequencing | After CF-1 | After CF-1 (one-line fix, hours not weeks) | Cross-reviews show urgency; CF-1 is trivial |
| Monotonic clock detail | Raised as failure mode | Still a failure mode — monotonic + persisted datetime needed | No cross-review challenged this |

**The most significant contribution of this revision:** The finra_short analysis. All three auditors concluded it was an anti-signal or marginal signal. The evidence in the codebase shows it is a direction-agnostic volatility signal with genuine positive edge versus the correct base rate. This changes Priority 3 from "flip or exclude finra_short" to "correctly interpret finra_short as a volatility signal and verify direction-agnostic handling is preserved end-to-end."
