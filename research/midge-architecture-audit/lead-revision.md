# MIDGE Architecture Audit — Lead Revision
**Auditor Role:** Lead — Signal Pipeline Architect
**Date:** 2026-03-01
**Phase:** 3 — Post-cross-review revision

---

## Preamble: What Changed My Mind and What Did Not

The cross-reviews from Alpha and Beta were valuable. But this document treats them as evidence to weigh, not consensus to echo. On several key points I am revising my position based on code evidence I gathered after reading the reviews. On others I am standing firm, including two points where the cross-review arguments turned out to be factually wrong when checked against the code.

---

## 1. Point-by-Point Analysis of Challenges to My Original Findings

---

### Challenge to Issue 2: Deduplication Failure

**My original finding:** The deduplication race condition at `convergence_alerter.py` lines 424-437 causes the alert storm visible in discovery_log.jsonl. Root cause: `_last_alert_time` is updated after all callers pass the check.

**Beta's challenge:** Beta rated this RC-2 at "Medium likelihood" because "the current concurrency model does not actually create the race via threading — collection is main-thread-only, currently safe but fragile."

**My response: STAND FIRM, but refine the root cause.**

Beta is technically correct that the parallel threading path does not currently exist. But Beta also acknowledged the storm already happened in production (evidence: CONV-20260227-0001 through -0021 logged within one second). Beta's own evidence is at odds with the "currently safe" characterization.

After re-examining the code, I believe the actual storm mechanism is fast-step-loop burst, not concurrent threading. At high step rates, `check_convergence()` is called many times within a single wall-clock second. The dedup check compares `datetime.now()` and finds zero elapsed time — the comparison passes every time because the timestamps are identical. The `_last_alert_time` is updated on the first pass, but all subsequent calls within the same second also pass because `now - _last_alert_time` evaluates to zero seconds, which is less than 4 hours.

Beta's suggested fix — use `datetime.now()` with a `threading.Lock()` — is partially correct. However, Beta's stated reason for preferring `datetime.now()` over `time.monotonic()` (monotonic resets on restart, breaking dedup across sessions) is valid. The correct fix is: `datetime.now()` for the time comparison (so it survives restarts), combined with `threading.Lock()` for the check-and-update atomicity. My original code sketch using `time.monotonic()` was wrong in that specific detail. Beta's correction on the monotonic-vs-wall-clock choice stands.

**Revised conclusion:** The dedup race is confirmed. Beta's monotonic-clock objection is valid and corrects my proposed implementation. The bug is real and production-confirmed. Priority ranking unchanged.

---

### Challenge to my finra_short assessment — "Marginal" vs "Anti-Signal"

**My original finding:** finra_short at 35.8% win rate, 1,263 samples is "marginal edge — barely above random."

**Alpha's challenge:** 35.8% is not "barely above random" for a directional 5% threshold. The correct benchmark is not 50% (coin flip) but the base rate of a 5% directional move in 45 days. If the base rate is 45%, finra_short is meaningfully anti-predictive. Alpha calls for direction flip or exclusion.

**Beta's challenge:** Neither Alpha nor I computed the base rate. Beta calls both Alpha's "anti-signal" and my "marginal" framings premature without this calculation.

**My response: REVISE — Beta's correction is right on the epistemics; Alpha's direction is right on the financial implication.**

The base rate question is decisive and neither I nor Alpha computed it. Beta is correct on that point.

However, I can partially resolve this without the base rate by examining what we already know: the `SUCCESS_THRESHOLD_PCT = 5.0` in `outcome_collector.py` is a one-sided directional threshold. The prediction is "success" only if the price moves 5% in the predicted direction. A random signal predicting UP would succeed whenever the stock happens to move 5% upward — that base rate in a bull market environment (2025-2026) for a diversified watchlist is likely 40-50% at 45-day windows, not 50% for a coin flip. This means 35.8% for finra_short is potentially worse than random in the current environment.

But I cannot confirm this without computing the actual null hypothesis win rate from `outcomes.jsonl`. What I can confirm is that this calculation exists in the data and should be done before either flipping the signal or excluding it.

**Revised position:** Demote finra_short to "observer-only" status in the convergence vote — it continues to update Thompson, continues to be logged, but does not contribute to domain count or the confidence geometric mean until the base rate is computed and either (a) the rate confirms anti-signal behavior, in which case flip or exclude, or (b) the rate shows it is positive against the true null, in which case restore. This is less aggressive than Alpha's outright flip and less passive than my original "marginal" framing. It is also immediately implementable without the base rate calculation.

**What changed:** I accept that I was too soft in calling this "marginal." I am not yet prepared to call it "anti-signal" without the base rate. Observer-only status is the appropriate intermediate response.

---

### Challenge to my Rank 2 recommendation: Session Sweep Bypass (Option B)

**My original finding:** min_domains=3 blocks MIDGE's best-documented edge (session sweep+IFVG, PF 1.84). Recommended Option B: parallel direct-output path for session_sweep_ifvg signals bypassing convergence.

**Alpha's challenge:** Agrees with the bypass as near-term fix but also calls for removing three correlated sentiment sources (google_trends, stocktwits_sentiment, social_sentiment) from domain voting because they count as three independent domain slots while measuring the same underlying retail attention.

**Beta's challenge:** Option B introduces a two-track system with different quality standards. The bypass will expand over time and erode the min_domains invariant. Suggests instead auditing domain assignments so correlated signals compete within a single domain rather than counting independently.

**My response: STAND FIRM on Option B, and CORRECT Alpha's premise with direct code evidence.**

I re-read `signal_adapters/layer6.py` and `signal_adapters/market_data.py` after reading the cross-reviews. Here is what I found:

- `from_stocktwits_sentiment()` emits `domain="sentiment"`
- `from_google_trends()` emits `domain="sentiment"`
- `from_social_sentiment()` in market_data.py emits `domain="sentiment"`

All three sources Alpha claimed were "counting as three independent domain votes" are assigned the SAME domain. The convergence alerter's `_check_direction_convergence()` at line 464-468 takes one signal per domain (the strongest matching signal from each domain bucket). Three sentiment sources firing simultaneously produce one vote in the "sentiment" domain — they do not satisfy min_domains=3 by themselves. Alpha's central concern about three correlated sources gaming the domain count is factually incorrect based on how the code works.

The domain_categories mapping (lines 176-194 of convergence_alerter.py) maps "sentiment" to category "social". The three sentiment sources all land in the same bucket. This domain independence concern is already handled by design.

This is significant: it means the min_domains=3 problem is real but for a different reason than Alpha stated. The problem is not correlated sources gaming the count — it is that the technical domain (where session sweeps fire) is genuinely independent and strong but requires two other genuinely different domains to co-fire within 72 hours before any output is generated. The fix is correct (bypass for documented-edge signals), and the premise (correlated noise satisfying the domain count) is incorrect.

**Beta's bypass-creep concern** is a valid architectural risk. The response is to make the bypass criteria narrow and explicit: only signals with DSR > 0.5 from a completed backtest with >= 100 trades qualify. That standard is currently met by exactly one signal (session_sweep_ifvg). Making the gate explicit prevents bypass creep by requiring the same evidence bar for any future addition.

**Revised conclusion on this point:** Option B remains the correct recommendation. Alpha's supporting argument was factually wrong about domain grouping, but this does not change the conclusion — it only removes one of Alpha's secondary reasons for making a similar change. Beta's concern about bypass creep is addressed by making the qualifying criteria narrow and explicit in the implementation.

---

### Challenge to Issue 5: Outcome Data Contamination

**My original finding:** outcomes.jsonl contains mock data with impossible returns (AAPL 49.93% in one day). This contaminates Thompson distributions.

**Beta's challenge:** Beta found a broader and more actionable structural problem: the `_registered` set in OutcomeCollector grows unbounded, the write_text() call is non-atomic, and a crash mid-write loses the entire registered set. Beta also found test contamination in pair_outcomes.json ("a|b" through "a6|b6") which I had not examined.

**My response: REVISE SCOPE UPWARD — Beta's structural analysis subsumes my symptom-level finding.**

My Issue 5 identified a specific corrupt record. Beta identified the structural reason such records accumulate and cannot be cleaned up without the underlying mechanisms I missed. The correct response sequence is:

1. Fix CF-1 (Thompson `_log_update()` lock gap — already documented in my Priority 2)
2. Replace `write_text()` with atomic `os.replace(tmp, path)` across all four files that use the non-atomic pattern (learning_config.py, hypothesis_engine.py, hypothesis_generator.py, step_timer.py)
3. Add a pruning mechanism to OutcomeCollector's `_registered` set: remove entries where `outcome_due` is more than 90 days in the past
4. Delete "a|b" through "a6|b6" from pair_outcomes.json
5. Delete the 2027-timestamp prediction from registered_signals.json
6. Rebuild Thompson distributions from cleaned outcomes.jsonl (this was already done once via accelerate_learning.py — the procedure exists)

The key addition to my original finding is step 2 (atomic writes) and step 3 (registered set pruning). These prevent the class of failure, not just the specific instance.

---

### Challenge to the "No Output Path" ranking

**Beta's challenge:** Beta agrees the output path must be built but argues it should be built after the deduplication fix and data integrity fixes. Building an output path over a broken dedup mechanism delivers 20 simultaneous paper trade entries on the same signal.

**My response: REVISE THE SEQUENCING — Beta is right about ordering within Priority 1.**

My original ranking had "Build output path" as Rank 1 and "Fix deduplication" as Rank 3. Beta's critique is operationally correct: the specific failure mode Beta describes (alert storm → 20 paper trade entries on first convergence signal) is a real consequence of my recommended ordering.

However, I disagree with Beta's conclusion that this requires a strict sequencing where the output path comes after all reliability fixes. The correct answer is that the output path must include deduplication as part of its own implementation — not as a prerequisite from a separate fix. The paper trading path should gate on `signal_id` deduplication internally (write to paper_trades.jsonl only if signal_id not already present within a 4-hour window). This is one additional gate in the output path implementation, not a separate prior fix.

The revised sequencing: implement the dedup fix (now moved inside the output path gate logic) and the Thompson lock fix (CF-1) concurrently, then build the output path on top of clean infrastructure. The output path remains Priority 1 in terms of financial impact — nothing else matters until a trade can be executed — but its implementation must include dedup.

**What changed:** I acknowledge the sequencing concern. The fix: dedup gate is built into the paper trading path, not treated as a separate prior step.

---

### Challenge: Organism Tax and Reward Misalignment

**My original finding:** I did not examine this at all. I stayed within the market pipeline boundary.

**Alpha's challenge:** Approximately 60% of per-step compute is organism maintenance, and the endocrine pipeline absorbs market signal energy without producing tradeable output.

**Beta's challenge:** The reward ceiling problem is real — market agents capped at 0.5 reward, TaskPool at 1.0 — but Alpha's recommendation to strip 50+ systems without profiling is risky. Beta recommends profiling first.

**My response: ACCEPT BETA'S POSITION — profile before stripping, but acknowledge the reward misalignment is real.**

Alpha's 60% organism tax estimate is unsubstantiated — Alpha acknowledges no StepTimer snapshot exists. Beta's position (profile first, strip confirmed dead weight based on data) is more rigorous.

However, the reward misalignment Beta and Alpha both independently identified is not a profiling question — it is a design choice documented in MEMORY.md ("below TaskPool exploit ceiling, above rest") that was made to prevent market actions from dominating reward space. The problem is that the comparison point was wrong: the ceiling for TaskPool defines what the Q-table considers "maximum value," and market agents are permanently calibrated below that ceiling. This teaches the Q-table the wrong objective function over thousands of steps.

The fix for reward misalignment does not require profiling: for agents assigned market roles (SEC_WATCHER, CONTRACT_TRACKER, MARKET_ANALYST, HYPOTHESIS_EXPLORER, HYPOTHESIS_VALIDATOR), cap TaskPool exploit at 0.3 and raise market action ceiling to 0.8. This is a single-line change per role in market_actions.py. It requires no profiling because the issue is not which systems run, but what the reward signal teaches.

The organism overhead question (Alpha's 60%) is properly deferred to profiling. The reward misalignment fix is not — it is a known design error.

---

### Challenge: Meta-Learner Cold-Start Bias (Beta's Surprise 2)

**My original finding:** I understood the meta-learner as monitoring live hypothesis performance.

**Beta's finding (that I missed entirely):** `_seed_retirement_window_from_registry()` populates the 50-entry retirement window with historical state on cold start. Wire 2 decisions (tighten min_correlation if retirement_rate > 70%) are based on seeded historical data, not live performance. The meta-learner cannot distinguish "15 retirements in this session" from "15 retirements loaded at startup from older registry state."

**My response: ACCEPT this finding. I had not examined this code path.**

This is a genuine cold-start bias in RSI Layer 3. If the meta-learner tightens min_correlation based on seeded retirements from old registry state, it suppresses hypothesis generation in a fresh session that might be performing well. The fix is to not seed the retirement window from historical registry state at cold start — instead, let it fill from live session data only, accepting that the first N steps of Wire 2 operate without enough data to trigger the tightening logic. This is the correct cold-start behavior.

**What this adds to my Priority 4:** Include a fix to `_seed_retirement_window_from_registry()` — either remove the seeding entirely, or add a flag that marks seeded entries as historical so Wire 2 ignores them until enough live-session retirements accumulate.

---

## 2. Findings That Stood Up Unchanged

**Issue 1 (No output path):** Triadic consensus, no substantive challenge to the diagnosis.

**Issue 3 (2027 timestamp prediction):** Alpha confirmed this, Beta found the structural cause (unbounded set growth). My specific finding stands; Beta deepened it.

**Issue 4 (Thompson prior mismatch in learning_config):** `congressional = 0.75` in config vs mean = 0.164 in actual Thompson distribution. Neither Alpha nor Beta challenged this. Confirmed.

**Issue 6 (Static signal confidence disconnect from Thompson reality):** I remain the only auditor who traced the formula interaction: static confidence = 0.75 as geometric mean input, Thompson weight = 0.664, output = 0.75^weighted — still misleadingly high. Neither cross-review addressed this specific formula path. Standing firm on this finding.

**Rotation dilution math (19 sources / 3 slots / 50-step cadence = ~315 steps per source):** Alpha confirmed this as a surprise finding they had not quantified. Beta confirmed it as important. Stands as documented.

**Sell-side directional skew in Form 4 predictions:** I am the only auditor who read predictions.jsonl directional distribution. Every sec_form4 prediction is "down" except one MSFT entry. This is correct behavior given the data, but means bullish convergence from sec_form4 is near-impossible on the current mega-cap watchlist during sustained sell periods. Alpha partially addressed this via watchlist diversification recommendation (Fix B in Priority 4). Stands.

**Two-schema coexistence in predictions.jsonl:** Old format (entry_price=0.0, target_price=0.0) and new OutcomeCollector format coexist. Neither Alpha nor Beta specifically traced this. The old-format records make price-based outcome evaluation meaningless for them. Stands.

---

## 3. Revised Top 5 Priority List for Financial Effectiveness

These priorities incorporate all three cross-reviews and my additional code research. Sequencing is based on dependency chain (what must exist before what), not just estimated impact.

---

### Priority 1: Fix the Bayesian Learning Foundation (CF-1 + Data Purge)

**What and why this is first:** Everything else in MIDGE learns from the Thompson distributions. If those distributions are corrupted, no other improvement matters. CF-1 (Thompson `_log_update()` called without `self._lock`) is confirmed to have caused a full production rebuild once (MEMORY.md). It is still unfixed. The next marathon run will corrupt the Thompson history again.

**Concrete actions:**

1. In `thompson_sampler.py:_log_update()` (line 270), add `with self._lock:` around the file write — same lock that wraps `_save_distributions()`. One line change.
2. Replace `Path.write_text()` with atomic `os.replace(tmp, path)` in: `learning_config.py`, `hypothesis_engine.py`, `hypothesis_generator.py`, `step_timer.py`. Four files, same 3-line pattern already in `_save_distributions()`.
3. Delete "a|b", "a0|b0" through "a6|b6" from `data/market/pair_outcomes.json`.
4. Delete records with impossible returns (AAPL 49.93% 1-day) from `data/market/outcomes.jsonl`.
5. Fix the 2027-timestamp prediction in `data/market/registered_signals.json`.
6. Update `learning_config.py:source_reliability` to match actual Thompson reality: `congressional` from 0.75 to 0.20; `sec_edgar` flagged deprecated in favor of `sec_form4`.
7. Rebuild Thompson distributions from cleaned outcomes.jsonl using the existing `accelerate_learning.py` rebuild procedure.

**Why not data fixes before lock fix:** The lock fix must come first because running any rebuild while the lock gap exists could corrupt the rebuilt distributions before the session ends.

**Financial impact:** Every confidence score MIDGE computes, every convergence gate, every hypothesis promotion decision downstream of Thompson is only as good as the distributions. This is table stakes for everything that follows.

---

### Priority 2: Build the Paper Trading Output Path (with Dedup Gate)

**What:** Convert ConvergenceAlert → TradeSignal → `data/midge/paper_trades.jsonl`. Wire OutcomeCollector to evaluate paper trade exits using real yfinance prices and compute dollar P&L, not just binary directional accuracy.

**Why this is Priority 2:** Without this, MIDGE cannot produce a dollar P&L number. The Thompson distributions, the DSR-gated hypothesis lifecycle, the Kelly sizer — all exist to serve a trading outcome that never materializes. The "research instrument vs. trading instrument" gap cannot be measured until at least one paper trade completes.

**Specific implementation notes incorporating Beta's failure modes:**

- Gate the TradeSignal write on signal_id deduplication — check paper_trades.jsonl for the same signal_id within the past 4 hours before writing. This prevents the alert-storm-to-trade-storm failure mode Beta identified without requiring the dedup fix to be a separate prior step.
- The entry price uses a yfinance snapshot at alert time. Note the 15-minute delay for free tier — paper trade P&L will not reflect real execution prices. This is acceptable for learning feedback purposes; document the delay in the paper trade record so downstream P&L interpretation knows the data quality.
- Account size anchor: configure a `paper_account_value` in learning_config.py (e.g., $100,000 USD). Kelly fraction × account value = notional. Without this anchor, Kelly fraction is a ratio with no denominator.
- Outcome windows for paper trades should match the source type (session_sweep_ifvg: 1-day; sec_form4: 45-day). Aggregate P&L across windows is not directly comparable — track separately by source.
- Paper trade exits: price-based (if stop/target hit) AND time-based (at outcome window expiry). OutcomeCollector already has the time-based resolution logic; add price-based early exit.

**Financial impact:** Closes the only gap between "organism that learns" and "organism that trades." Creates real dollar feedback that the Thompson distributions and hypothesis engine can learn from.

---

### Priority 3: Fix Alert Deduplication — Confirmed Production Failure

**What:** The convergence alerter's dedup guard at lines 424-437 allows multiple alerts within the same wall-clock second because `_last_alert_time` is updated after all callers within the same second pass the check.

**Why this is Priority 3 and not merged into Priority 2:** The dedup fix applies to the discovery log (RSI Layer 2 training data) regardless of whether the paper trading path exists. The 20+ duplicate entries already in discovery_log.jsonl corrupt hypothesis generation priority. This needs fixing independent of the paper trading path.

**Correct fix (incorporating Beta's clock-type correction):**

```python
# In __init__: add
import threading
self._dedup_lock = threading.Lock()

# In check_convergence(), replace lines 424-437:
now = datetime.now()
filtered = []
for alert in alerts:
    direction = alert.direction if hasattr(alert, "direction") else "neutral"
    with self._dedup_lock:
        if (self._last_alert_direction == direction
                and self._last_alert_time is not None
                and (now - self._last_alert_time).total_seconds() / 3600
                < self._min_alert_interval_hours):
            continue
        self._last_alert_direction = direction
        self._last_alert_time = now
    filtered.append(alert)
alerts = filtered
```

Using `datetime.now()` (not `time.monotonic()`) preserves the comparison across process restarts. Beta's correction on this specific point was correct and I am incorporating it.

**Financial impact:** Every corrupted entry in discovery_log.jsonl is a phantom pattern discovery that RSI Layer 2 may attempt to build hypotheses from. Deduplication fix prevents future corruption; existing corrupt entries should be identified and flagged with a `duplicate_storm: true` field so hypothesis_generator.py can skip them.

---

### Priority 4: Unlock Session Sweep Edge + Reward Realignment

**Two independent fixes that both increase MIDGE's ability to use its best-documented edge.**

**Fix A — Session Sweep Bypass:** Implement Option B from my original findings. Create a parallel direct-output path for `session_sweep_ifvg` signals that bypasses `min_domains=3` when signal quality >= 0.65 (Elite tier, from backtest) and the current regime matches the backtest regime window. Add an explicit qualifying gate: this bypass requires a completed backtest with >= 100 trades and DSR > 0.5. Currently only session_sweep_ifvg meets this bar. Make this gate explicit so it cannot expand without meeting the same evidence standard.

Note on domain independence: Alpha's concern that correlated sentiment sources game the domain count is factually incorrect per the code. `social_sentiment`, `stocktwits_sentiment`, and `google_trends` all emit `domain="sentiment"` and are grouped into a single "social" category in domain_categories. They already compete within one domain slot. The min_domains=3 problem is real but for a different reason — the sweep pattern is genuinely isolated in the technical domain and needs two independent domains to co-fire. The bypass addresses this without the domain-grouping fix Alpha called for (which the code already implements).

**Fix B — Reward Realignment:** For agents in market roles (SEC_WATCHER, CONTRACT_TRACKER, MARKET_ANALYST, HYPOTHESIS_EXPLORER, HYPOTHESIS_VALIDATOR), cap TaskPool exploit reward at 0.3 and raise market action ceiling to 0.8. Current calibration (market: max 0.5, TaskPool: up to 1.0) actively teaches the Q-table that abstract task completion matters more than market intelligence. This is a MEMORY.md-documented design choice that solved the wrong problem.

This fix requires no profiling. The organism overhead question (Alpha's 60%) is properly deferred until the StepTimer produces a marathon snapshot. The reward misalignment is a known configuration error, not a profiling question.

**Financial impact:** Fix A makes PF 1.84 session sweep edge actionable without waiting for multi-domain coincidence. Fix B ensures market-role agents are rewarded proportionally to their market intelligence contribution over thousands of training steps.

---

### Priority 5: Slow-Signal Persistence + Cold-Start Meta-Learner Bias Fix

**Two fixes for the learning loop's temporal modeling.**

**Fix A — Per-Domain Convergence Windows:** COT data (weekly, 3-day publication lag) and congressional disclosures (30-45 day reporting lag) both wash out of the global 72-hour convergence window before they can contribute to convergence. Both have `source_reliability` values in learning_config suggesting they should contribute when they fire. The fix: add a `domain_convergence_windows` map keyed by domain:

```python
# In ConvergenceAlerter.__init__:
self.domain_windows = {
    "positioning": timedelta(hours=336),  # COT: 14 days (2 publication cycles)
    "government":  timedelta(hours=168),  # Congressional: 7 days (minimal contribution)
    "contracts":   timedelta(hours=168),  # SAM.gov: 7 days
}
# default: self.convergence_window (72h) for all others
```

In `_prune_old_signals()`, use the domain-specific window. This gives slow signals persistence proportional to their alpha decay rates already defined in `learning_config.py:decay_rates` without changing the fast-signal window for technicals and news.

Note: congressional trades at 16.4% Thompson mean are unlikely to meaningfully contribute to convergence even with extended windows. The per-domain window fix is primarily valuable for COT positioning data (mean=0.279 on 81 sideways observations, the most data-rich positioning source) and future slow signals with better observed reliability.

**Fix B — Meta-Learner Cold-Start Bias:** Beta correctly identified that `_seed_retirement_window_from_registry()` populates the 50-entry retirement window with historical registry state at cold start, causing Wire 2 of the meta-learner to potentially tighten `min_correlation` based on historical retirements rather than live session performance. Fix: mark seeded entries with `seeded_at_startup: True` in the retirement window. Wire 2's retirement rate calculation ignores seeded entries until enough live-session entries exist (threshold: 10 live entries) to compute a reliable rate.

**Financial impact:** Fix A gives COT positioning data a realistic window to participate in convergence. Fix B ensures RSI Layer 3 meta-learning does not suppress hypothesis generation in fresh sessions based on historical artifacts.

---

## Summary Table

| Priority | Change | Key Evidence | Financial Mechanism |
|----------|--------|--------------|---------------------|
| 1 | Fix Thompson lock + purge data contamination | Beta CF-1 (confirmed production failure), Beta CF-3 (pair_outcomes confirmed), Lead Issue 5 | Prevents next distribution corruption; cleans all Bayesian inputs |
| 2 | Build paper trading output path (with dedup gate) | All three agents: TradeSignal never instantiated | Closes feedback loop; creates first real dollar P&L |
| 3 | Fix alert dedup — datetime.now() + Lock | Discovery log storm confirmed (CONV-20260227-0001 through -0021) | Stops RSI Layer 2 corruption from duplicate phantom discoveries |
| 4 | Session sweep bypass + reward realignment | Backtest PF 1.84 blocked by min_domains; Q-table taught wrong objective | Makes best-documented edge actionable; aligns agent learning with financial purpose |
| 5 | Per-domain convergence windows + cold-start meta-learner fix | COT washout from 72h window; Beta cold-start bias finding | Gives slow signals persistence; ensures RSI Layer 3 does not self-suppress on startup |

---

## What the Cross-Reviews Confirmed, Overturned, and Corrected

**Confirmed:**
- No output path (unanimous triadic agreement — the finding with strongest support)
- Thompson thread-safety gap (Beta found CF-1; I now place it at Priority 1)
- Reward misalignment is a real training signal problem, not just efficiency concern
- 0 promoted hypotheses from 30 generated is evidence RSI Layer 2 does not close

**Overturned:**
- My "marginal" characterization of finra_short: Beta's base-rate correction is right. The question is open until computed. Moving finra_short to observer-only status is the correct intermediate step.
- My monotonic clock recommendation for dedup fix: Beta's objection is correct. Use `datetime.now()` + `threading.Lock()`, not `time.monotonic()`.

**I Corrected Peer Arguments:**
- Alpha's claim that google_trends, stocktwits_sentiment, and social_sentiment each count as independent domain votes is factually wrong. All three emit `domain="sentiment"` and are already bucketed as a single domain. The convergence alerter already handles this by design. Alpha's secondary argument for removing them from the vote is therefore unnecessary — they can only contribute one sentiment vote regardless of how many sentiment sources fire simultaneously.

**Standing Firm (with additional evidence):**
- Option B (session sweep bypass) remains correct, despite Beta's bypass-creep concern, because the qualifying criteria can be made narrow and explicit
- Static confidence disconnect from Thompson reality (my Issue 6/Rank 6) — neither peer review traced this formula path; it remains an unaddressed finding
- Sell-side directional skew in Form 4 predictions — only I examined predictions.jsonl directional distribution; finding stands
- Two-schema coexistence in predictions.jsonl — only I identified this; finding stands
