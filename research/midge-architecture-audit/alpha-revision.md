# MIDGE Architecture Audit — Alpha Revision (Adversarial Market Practitioner)

**Auditor Role:** Witness Alpha — Adversarial Market Practitioner
**Revision Date:** 2026-03-01
**Basis:** alpha-findings.md, alpha-cross-review.md, lead-cross-review.md, beta-cross-review.md + additional code verification and empirical base-rate calculation

---

## How to Read This Document

For each challenged point, I either revise my position with new evidence or stand firm with explicit reasoning. Points not challenged remain as stated in alpha-findings.md and alpha-cross-review.md.

---

## Point 1: finra_short as Anti-Signal — REVISED

### Original position (alpha-findings.md, Section 3)
finra_short at 35.8% win rate across 1,263 samples is an anti-signal. Including it in convergence calculations degrades signal quality. Fix: flip its interpretation (contrarian) or exclude it from convergence voting.

### What Beta challenged
Beta (Divergence B in beta-cross-review.md) correctly identified that neither Lead nor I computed the base rate. The outcome collector uses `SUCCESS_THRESHOLD_PCT=5.0` — a prediction is correct if price moves 5% in the predicted direction within the outcome window. A coin flip baseline (50%) is the wrong null hypothesis for this specific test. The null is: what fraction of rolling 45-day windows does a stock move 5% in a given direction?

### What the evidence shows

I ran the empirical calculation against SPY, AAPL, MSFT, GOOGL, AMZN, and QQQ across 2 years of daily data (2,742 rolling 45-day windows):

- Base rate for a 5% bearish move in 45 days: **16.9%**
- Base rate for a 5% bullish move in 45 days: **43.4%**
- finra_short win rate: **35.8%**

finra_short signals high short interest as bearish. Comparing 35.8% to the base rate of a correct bearish 5% move (16.9%) shows finra_short is not an anti-signal — it has **positive edge of approximately 19 percentage points over the null hypothesis**. My coin-flip comparison (35.8% vs 50%) was the wrong baseline. Beta's challenge was correct.

### Revised position

finra_short is NOT an anti-signal and should NOT be flipped or excluded. It is a legitimate directional contributor to convergence alerting. Its Thompson weight of 0.858 (from mean=0.358 mapped to [0.5, 1.5]) is appropriate: it down-weights the source relative to neutral, which is correct given that 16.9% base rate means predicting a bearish 5% move correctly is already harder than a coin flip in bull markets.

**What I retract:** The recommendation to flip or exclude finra_short from convergence voting. It is wrong and would remove MIDGE's most data-rich Thompson distribution.

**What I maintain:** The Thompson weight range [0.5, 1.5] is too compressed to meaningfully distinguish good sources from bad ones — the full range from 0 to 1 maps to only a 1.0-unit spread in weight. A source at 15% win rate (congressional) gets weight 0.65 vs a source at 80% win rate gets weight 1.30. This 2x compression is insufficient differentiation. The weight formula deserves review, but the direction of finra_short as a bearish signal is validated.

**One open question remains:** The base rate calculation used SPY-adjacent large-caps. MIDGE's watchlist overlaps substantially with these, so the calculation is reasonably representative. However, finra_short signals may be applied to individual tickers where the base rate differs from the index basket. This should be evaluated per-ticker if the data volume allows.

---

## Point 2: min_domains=3 and Domain Independence — REVISED

### Original position
The min_domains=3 requirement can be satisfied by combining three retail sentiment sources (social_sentiment, google_trends, stocktwits_sentiment), which would mean three correlated noise sources satisfying Law 2. This is a structural flaw.

### What the code shows

I was empirically wrong about the implementation. Code verification shows:

- `google_trends` → `domain="sentiment"` (layer6.py:191)
- `stocktwits_sentiment` → `domain="sentiment"` (layer6.py:90)
- `social_sentiment` → `domain="sentiment"` (market_data.py:276)

The convergence alerter's `_check_direction_convergence()` collects signals by domain key (`for domain, signals in self.signals.items()`), then accumulates `domains_seen` as a set. Three sources sharing the same domain string "sentiment" add only ONE entry to `domains_seen`. The min_domains=3 check operates on unique domain strings, not signal count.

Three retail sentiment sources cannot satisfy min_domains=3 because they all share the same domain key. The alerter's domain-based deduplication already prevents the correlated noise stacking I was concerned about.

### Revised position

The domain independence architectural concern I raised is partially resolved by the existing implementation. The min_domains=3 requirement correctly groups correlated sources under a single domain name.

**What I retract:** The claim that three retail sentiment sources could satisfy min_domains=3. The code prevents this.

**What I maintain (restated more precisely):** The domain grouping creates a different but real problem. With only one "sentiment" domain slot available, MIDGE's three retail sentiment sources (social_sentiment, google_trends, stocktwits_sentiment) compete for the same slot in the convergence vote. All three running simultaneously only increases sample size for one domain position — they do not add breadth. This is not a flaw in the law but it does mean the 19 sources produce fewer effective domain slots than the raw source count suggests.

**The session sweep bypass from Lead's analysis remains valid.** The session sweep detector produces `domain="technical"` signals. A single-domain signal (technical only) legitimately cannot satisfy min_domains=3. Lead's Option B (parallel direct-output path for BACKTEST_DERIVED signals with quality >= 0.65) is still the right approach for session sweeps. Beta's concern about the bypass expanding is noted, but it can be controlled by restricting it to signals with a minimum DSR-validated backtest. The bypass should be architecture-gated by evidence quality, not domain count.

---

## Point 3: The Organism Tax — STAND FIRM with modification

### Beta's challenge (Divergence D in beta-cross-review.md)
Beta argues that removing non-market step hooks before profiling is operationally risky because: (1) thermoregulation and renal_filter detect runaway resource consumption; (2) the StepTimer snapshot does not yet exist; (3) the actual bottleneck may be convergence_alerter's O(total_signals) prune operation, not the biological hooks.

### My response

Beta's profiling-first principle is operationally correct. I should not have recommended disabling specific systems before profiling data identifies them as the bottleneck. I revise that recommendation.

**What I retract:** The specific list of systems to "audit-log rather than actively step" (reproductive_system, lymphatic_system, vestibular_system, etc.) as an actionable recommendation. Without a StepTimer marathon snapshot, this list is speculation about where the overhead lives.

**What I stand firm on:** The organism tax category is real. Roughly 50 of 80 registered systems fire step hooks with no pathway to any trading decision — this is a structural fact derivable from the systems dict in main.py, independent of where the actual CPU bottleneck is. The appropriate action is not immediate removal but is: (a) run a marathon session to produce the StepTimer snapshot, (b) identify the actual top-CPU-consuming systems from the snapshot, (c) evaluate those specific systems for tradeable output, (d) make targeted changes. The organism tax may or may not be the bottleneck. Profiling is required to determine whether it is material. My framing of "60% overhead ratio" was an estimate; I should have flagged it as an estimate more clearly.

**Beta's circuit breaker point is taken:** If renal_filter or nociception are the systems detecting disk-full (OR-3) and memory growth (PV-3) conditions, removing them eliminates the early warning system for the exact failure modes Beta documented. Any removal decision must first audit which non-market systems serve as operational monitors.

**Revised recommendation:** Run one marathon session with StepTimer active. The snapshot will identify where compute is actually consumed. Only then make system-removal decisions, prioritizing systems confirmed both (a) high CPU from the snapshot AND (b) confirmed to produce no trading signal or operational monitoring value.

---

## Point 4: Circadian Rhythm Fix — STAND FIRM with test-isolation caveat accepted

### Beta's failure mode analysis (beta-cross-review.md, "Alpha Kill Item 1")
Beta raised three valid failure modes for wiring CircadianRhythm to wall-clock time:
1. Test determinism collapse — phase behavior changes based on what time tests run
2. Timezone configuration drift between UTC (Wardenclyffe system) and ET market hours
3. Continuous weekend operation — undefined phase during 60-hour market close

### My response

All three of Beta's failure modes are valid engineering concerns. They are not arguments against the direction of change but are requirements that the implementation must address.

**What I stand firm on:** The underlying observation is correct. The circadian_rhythm.py docstring explicitly states "Unlike real organisms, Mae's cycle is not tied to wall-clock time. Instead, it's driven by simulation steps." The three phases (ACTIVE, CONSOLIDATION, REST) are keyed to step count modulo cycle_length. This means MIDGE's "active" exploration phase fires during overnight hours, weekend maintenance windows, and market holidays with equal probability as during market hours. This is a structural misalignment between organism behavior and market operation.

**What I accept from Beta:** The implementation of a wall-clock fix requires:
- A `test_mode` override that locks CircadianPhase to a configurable value for test runs — solving the test determinism problem
- Explicit `pytz.timezone('America/New_York')` anchoring — solving the Wardenclyffe timezone problem
- A defined "weekend/market-closed" behavior — solving the 60-hour rest concern (recommendation: CONSOLIDATION phase during market-closed periods, enabling memory replay and lag correlation analysis to run without competing with live signal fetching)

These are implementation requirements, not reasons to abandon the direction. The fix without these guardrails causes problems; with them, it produces meaningful alignment between organism dynamics and market time.

---

## Point 5: The Endocrine Signal Dead-End — STAND FIRM

### Challenge received
Neither Lead nor Beta directly challenged this finding. Lead noted it as a gap they missed. Beta did not address it.

### Why I stand firm

The code path is: convergence alert fires → `_collect_one()` in sensing_hook.py → reads from alerter advisory dict → deposits a stigmergy marker with intensity proportional to alert strength → optionally triggers endocrine release via EventBus.

The endocrine release modulates `exploration_bias` in the next N steps for affected agents. No financial output (trade signal, recommendation, position) is produced by this pathway. The dopamine/adrenaline cascade is the mechanism by which market signal strength is converted into abstract agent behavior modification, and that conversion loses the structured information (direction, confidence, contributing domains) that would be needed to produce a tradeable recommendation.

This is not a criticism of the endocrine system for internal organism health — stress from hypothesis failure triggering cortisol is coherent. The problem is specifically that convergence alerts, which are MIDGE's highest-quality output, flow through the endocrine system as their primary response pathway. The financial signal should exit through a structured output path, not through a hormone that widens exploration bandwidth for the next 50 steps.

**This finding has not been challenged with evidence that contradicts it. I stand firm.**

---

## Point 6: Reward Signal Miscalibration — STAND FIRM

### Challenge received
Beta confirmed this finding in their cross-review (Gaps section: "Alpha's reward ceiling analysis. The finding that market-role agents are deliberately capped at 0.5 reward from market actions while non-market TaskPool can return 1.0 surprised me."). Lead also acknowledged it as a gap they missed (Divergence 4).

### Why I stand firm

Beta's confirmation strengthens rather than challenges this point. The VDN Q-table trains on reward signals. An agent that discovers a convergence alert receives max 0.4 reward (from market_actions.py dispatch). The same agent that completes a random TaskPool task at high difficulty receives up to 1.0 reward. Over thousands of training steps, the Q-table learns that abstract task completion is 2-2.5x more valuable than market intelligence discovery. This is the wrong objective function for a system whose purpose is financial pattern detection.

Beta noted this "changes my view of the market_actions.py dispatch: it is not the problem solved I thought it was." Lead stated: "the reward misalignment means market-role agents are learning the wrong objective function. Over thousands of steps, this shapes behavior away from financial utility."

The fix is straightforward: for agents with market roles (SEC_WATCHER, CONTRACT_TRACKER, MARKET_ANALYST, HYPOTHESIS_EXPLORER, HYPOTHESIS_VALIDATOR), cap TaskPool exploit reward at 0.3 and raise market action reward ceiling to 0.8. This does not require architectural changes — it is a configuration adjustment to `market_actions.py` and the TaskPool reward scaling.

---

## Point 7: Beta's Sequencing Argument — ACCEPT

### Beta's position (Divergence A in beta-cross-review.md)
Lead recommends building the output path (paper trading) as Priority 1. Beta argues: build the output path BEFORE fixing the deduplication race means the first convergence alert in a real trading context could emit 20 duplicate paper trade entries on the same signal. Fix plumbing before turning on water.

### My response

Beta's sequencing argument is operationally correct. In my cross-review I ordered:
1. Paper trading output path
2. Thompson lock + data contamination
3. Session sweep bypass + noise source excision
4. Congressional lag compensation
5. Circadian + reward fixes

The correct order should be:
1. Fix CF-1 (Thompson history write without lock) — prevents next distribution corruption
2. Fix data contamination (pair_outcomes.json test entries, outcomes.jsonl mock data, predictions.jsonl schema migration)
3. Fix deduplication race (lead Issue 2 / beta RC-2)
4. Build paper trading output path — now safe because signal quality is clean and dedup is reliable
5. Session sweep direct-output path
6. Reward recalibration + circadian alignment

**I accept this reordering.** The logic is sound: building an output channel on top of corrupted data produces corrupted P&L records, and building output on top of a dedup race produces phantom positions.

---

## Point 8: finra_short's Thompson Weight Range — STILL PROBLEMATIC

### Original position
The [0.5, 1.5] Thompson weight range is too compressed to meaningfully distinguish sources.

### Refinement after revision

Having revised the finra_short direction finding, I can now state the compression problem more precisely:

- congressional (mean=0.164): weight = 0.664
- finra_short (mean=0.358): weight = 0.858
- sweep_bt:CL=F (mean=0.524): weight = 1.024
- hypothetical excellent source (mean=0.80): weight = 1.300

The ratio from worst-performing source (congressional) to best-performing source (CL=F sweep) is 1.024/0.664 = 1.54x. In the geometric mean log computation, this means the best-documented backtest source influences confidence only 54% more than the source with 16% win rate. In any rational weighting scheme, a source with verified 52% win rate from 26 observations should be given substantially more influence over a source at 16% from 33 observations.

The compression is not about direction (I was wrong on that for finra_short) but about differentiation power. The weight formula should be reviewed to allow the posterior mean's extreme values to have more impact. A simple alternative: weight = dist.mean directly (range [0, 1]) for mature distributions, blending to 0.5 for thin data. This gives congressional weight 0.164 vs CL=F sweep weight 0.524 — a 3.2x ratio that more accurately reflects the quality difference.

**This finding stands. The specific fix I recommend (use posterior mean directly rather than 0.5+mean) did not appear in any other agent's analysis.**

---

## Point 9: Congressional Signal — Accept Lead's Watchlist Observation, Maintain Lag Analysis

### Lead's addition
Lead found a systematic sell-side skew in recent sec_form4 and congressional predictions — the watchlist (AAPL, MSFT, GOOGL, AMZN, NVDA, META) skews toward names where insider selling is routine RSU vesting and uninformative. This is a valid observation I missed.

### My position
Congressional lag compensation (my Build Item 5) and Lead's watchlist diversification are independent fixes addressing different root causes. The lag compensation addresses the fundamental STOCK Act timing reality: even with a perfectly diverse watchlist, a trade disclosed 30-45 days after execution is priced in. The 16.4% Thompson win rate is more plausibly explained by timing than by watchlist composition, since the academic literature shows congressional trade edge exists specifically at trade execution, not at disclosure.

Both fixes are warranted. Neither substitutes for the other. I accept Lead's watchlist observation and add it to the recommendation set. I maintain that lag compensation is the more structurally important fix because it would apply regardless of watchlist composition.

---

## Revised Top 5 Priority List

These are ordered by financial impact, incorporating all three phases of review and the empirical corrections above. The sequencing follows Beta's argument that data integrity must precede output construction.

---

### Priority 1: Fix the Thompson Data Foundation (CF-1 + contamination purge)

**What:** Three components, ordered by dependency:

Step 1 — Fix CF-1 (Beta, confirmed unfixed): Add `_log_update()` call inside `self._lock` in thompson_sampler.py line 263. Wrap the `self.distributions[signal_id][regime]` dict mutation (lines 241-244) inside the same lock. This is 3 lines of change. It prevents a recurrence of the confirmed production failure that wiped 9,462 outcomes to Beta(1,1).

Step 2 — Replace non-atomic writes: Audit all `Path.write_text()` calls in `learning_config.py`, `hypothesis_engine.py`, `hypothesis_generator.py`, `hypothesis_validator.py`, and `outcome_collector.py`. Replace with the `os.replace(tmp, path)` pattern already used correctly in `_save_distributions()`. This protects meta-learner gate tuning and hypothesis state from silent corruption on crash.

Step 3 — Purge contamination: Remove "a|b", "a0|b0" through "a6|b6" ghost entries from `data/market/pair_outcomes.json`. Remove mock outcome records from `outcomes.jsonl` (entries with impossible returns like 49.93% 1-day for AAPL). Fix the 2027-timestamp prediction. Rebuild Thompson distributions from the cleaned outcomes.jsonl — as done after the previous corruption, using the existing rebuild procedure.

**Why it is Priority 1:** Every confidence score MIDGE computes is derived from Thompson distributions. If those distributions are partially contaminated by mock data, every convergence alert is calibrated incorrectly. CF-1 will cause another rebuild on the next concurrent marathon run — the prior rebuild did not fix the root cause, only the symptom. Without clean Thompson data, the paper trading output path (Priority 4) will record corrupted P&L.

**Financial mechanism:** Thompson accuracy -> convergence confidence calibration -> whether MIDGE fires alerts at the right threshold -> whether paper trades are entered at the right confidence level.

---

### Priority 2: Fix the Alert Deduplication Race (Lead Issue 2 / Beta RC-2)

**What:** Add `threading.Lock()` around the dedup check-and-update in `convergence_alerter.py:check_convergence()`. Use `datetime.now()` for persisted comparison (not `time.monotonic()` per Beta's failure mode analysis — monotonic resets on restart and cannot be persisted). The fix:

```python
# Add to __init__: self._dedup_lock = threading.Lock()
with self._dedup_lock:
    now = datetime.now()
    if (self._last_alert_time and
        (now - self._last_alert_time).total_seconds() < self._min_alert_interval_hours * 3600):
        return None  # Within dedup window
    self._last_alert_time = now
```

Per Lead's evidence, the alert storm (CONV-20260227-0001 through -0021, 20+ identical alerts in one second) already occurred in production. This is not a theoretical race.

**Why it is Priority 2:** Building the paper trading output path over a broken deduplication mechanism delivers 20 simultaneous paper trade entries on the same signal — as Beta's failure mode analysis demonstrated. Dedup must be reliable before output is enabled.

---

### Priority 3: Fix the Thompson Weight Compression

**What:** Revise the weight formula in `convergence_alerter.py:_get_thompson_weight()`. Current formula: `weight = 0.5 + dist.mean` maps [0,1] → [0.5, 1.5], compressing the quality spread to a 1.54x ratio between worst and best performing sources. Proposed revision: for mature distributions (samples >= 20), use `weight = max(0.1, dist.mean)` directly, preserving zero as a floor. For thin distributions, blend to 0.5 as currently implemented.

The result: congressional (mean=0.164) gets weight 0.164 vs sweep_bt:CL=F (mean=0.524) gets weight 0.524 — a 3.2x quality ratio that reflects the evidence differential. finra_short (mean=0.358, positive edge confirmed) gets weight 0.358 — appropriately weighted between congressional and the sweep backtest sources.

Note: finra_short's direction is maintained as bearish (high short interest = bearish signal). The empirical base-rate calculation confirms it has positive edge at 35.8% vs 16.9% null.

**Why it is Priority 3:** The Thompson learning loop's financial utility depends on its ability to distinguish high-quality sources from low-quality sources. A 1.54x ratio between the most and least reliable sources means the geometric mean is dominated by the confidence values coming from the adapters (which are still static and require their own review), not by the learned reliability. Widening the weight spread allows Thompson's accumulated evidence to take its rightful role in confidence computation.

---

### Priority 4: Build the Paper Trading Output Path

**What:** Following Beta's sequencing requirement, this is now Priority 4, not Priority 1. The components remain the same:

- Convert `ConvergenceAlert → TradeSignal` when confidence > 0.75 AND strength > 0.65 in `sensing_hook.py:_collect_one()`
- Write to `data/midge/paper_trades.jsonl` with: timestamp, ticker, direction, entry_price (from yfinance at signal time), kelly_fraction (from `ctx._latest_kelly`), confidence, contributing_domains, dedup_id (signal_id hash)
- Define a fixed account value (e.g., $100,000 notional) for Kelly fraction → dollar amount conversion — this was missing from Lead's specification and Beta correctly flagged it
- Add ticker-level deduplication in PaperTradingBook to prevent duplicate entries from residual dedup timing windows
- Wire OutcomeCollector to read exit prices from paper_trades.jsonl and compute dollar P&L, not just directional accuracy

Additional requirement from Beta's failure mode analysis: use `os.replace(tmp, path)` for all paper_trades.jsonl writes, not `write_text()`.

**Why it is Priority 4:** This is the change that converts MIDGE from a research instrument to a trading instrument. Without dollar P&L, the Thompson distributions, hypothesis lifecycle, and Kelly sizer are all optimizing toward a metric (directional accuracy at 5% threshold) that is a proxy for financial return, not financial return itself. Once this is live, every accumulated Thompson outcome can be compared against the dollar outcome of the paper trade it informed — creating the feedback loop that justifies the entire RSI Layer 1→2→3 architecture.

---

### Priority 5: Session Sweep Direct-Output Path + Reward Recalibration

**What:** Two complementary changes that both increase the financial signal-to-noise ratio of MIDGE's behavior.

Fix A (session sweep direct output): Implement Lead's Option B for session_sweep_ifvg signals. When a sweep signal scores quality >= 0.65 (Elite tier) AND the current regime matches the backtest regime (sideways or volatile), generate a direct TradeSignal bypassing the min_domains=3 convergence check. Gate the bypass with: (a) minimum backtest sample size 20, (b) DSR > 0.5 (existing validator criterion), (c) regime match. Beta's concern about bypass expansion is addressed by condition (b) — only DSR-validated backtest results can access the bypass path. Add audit logging for bypass-path alerts distinct from convergence-path alerts, so the Law 2 compliance record remains unambiguous.

Fix B (reward recalibration): For agents in market roles (SEC_WATCHER, CONTRACT_TRACKER, MARKET_ANALYST, HYPOTHESIS_EXPLORER, HYPOTHESIS_VALIDATOR), set TaskPool exploit reward ceiling to 0.3 and raise market action reward ceiling to 0.8. This aligns what the VDN Q-table learns to value with what MIDGE's purpose requires. The current calibration (market actions max 0.5, TaskPool exploit max 1.0) is confirmed by both Beta and Lead as teaching the wrong objective.

**Why it is Priority 5:** Session sweeps are MIDGE's only source with verified positive edge from historical backtesting (PF 1.84 at Elite tier). They are currently blocked from producing output because they fire in a single domain (technical). The bypass makes MIDGE's best-documented edge actionable. The reward recalibration ensures that market-role agents are learning to pursue market intelligence rather than abstract task completion. Combined, these two changes move MIDGE's behavior meaningfully toward its stated financial purpose.

---

## What Moved, What Did Not

### Positions Revised

| Point | Direction | Reason |
|-------|-----------|--------|
| finra_short as anti-signal | Fully retracted | Empirical base rate: 35.8% vs 16.9% null = positive edge. Beta's challenge was correct. |
| min_domains=3 domain stacking | Partially retracted | Code groups correlated sources under one domain string. Three sentiment signals count as one domain. The mechanism already prevents the correlated stacking I described. |
| Organism tax removal before profiling | Partially revised | Beta's circuit breaker argument is sound. Profile first; targeted removal based on confirmed data. |
| Priority ordering | Revised | Accept Beta's sequencing: data integrity before output construction. |

### Positions Maintained

| Point | Evidence Basis |
|-------|---------------|
| Endocrine signal as dead-end for financial output | Unchallenged. Code path confirmed: convergence alert → hormone → exploration bias → no trade. |
| Reward miscalibration (max 0.5 vs max 1.0) | Confirmed by Beta as a "surprise." Confirmed by Lead as a gap. No counter-evidence offered. |
| Congressional 30-45 day lag as structural, not watchlist issue | Theoretical edge at disclosure time is well-documented. The 16.4% win rate is more consistent with information being priced in than with watchlist composition. |
| Circadian rhythm step-time mismatch with market time | Confirmed by code. Doc explicitly states it is step-based. Beta's failure modes are implementation requirements, not arguments against the direction. |
| Thompson weight compression problem | Restated more precisely. The 1.54x quality ratio between congressional and CL=F sweep is insufficient differentiation for meaningful source ranking. |
| No execution path is the central financial gap | Unanimous triadic agreement. No challenge received. |

---

## The Finding No Agent Made That Changes My View Most

Running the base rate calculation for Beta's finra_short challenge was the most productive outcome of this phase. It revealed that my coin-flip comparison was the wrong null hypothesis for the specific success criterion MIDGE uses. The directional prediction success threshold (5% move in 45 days) has a base rate well below 50% for bearish predictions in a bull-market-skewed equity universe. This is not a subtle distinction — finra_short went from "anti-signal, remove immediately" to "positive edge, keep and rely on" based on one empirical calculation that any of the three auditors could have run.

The lesson is operational: before characterizing a source as anti-signal based on win rate alone, compute the null hypothesis win rate for the specific success criterion being used. MIDGE's outcome evaluation is not a coin flip. It is a conditional probability that depends on the threshold, the window, and the market environment.

This same principle applies to the Thompson weight compression recommendation: the right fix is not to simply widen the range but to align the weight scale with the actual base-rate-adjusted edge of each source. That calculation requires per-source base rates, which could be computed from the existing outcomes.jsonl data.
