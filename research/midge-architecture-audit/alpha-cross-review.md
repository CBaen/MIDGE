# MIDGE Architecture Audit — Alpha Cross-Review (Adversarial Market Practitioner)

**Reviewing:** Lead findings (Signal Pipeline Architect) + Beta findings (Systems Reliability Engineer)
**Date:** 2026-03-01
**Stance:** Brutally honest. Adversarial lens does not soften during cross-review.

---

## 1. Reasoning Divergence Points

### Divergence A: finra_short — What Does 35.8% Actually Mean?

**My reasoning chain:** finra_short at 35.8% across 1,263 samples is an anti-signal. MIDGE treats it as a directional signal weighted at 0.858 in the geometric mean — insufficiently downweighted to neutralize a source demonstrably worse than random. The fix is to either flip its interpretation (high short interest → contrarian long) or exclude it from convergence voting.

**Lead's reasoning chain:** Lead identified finra_short as "most trained distribution" and called 35.8% win rate at a 5% threshold "marginal edge — barely above random." Lead's language is softer: "marginal edge."

**Where we diverged:** Lead is measuring against 5% threshold returns. My framing is binary directional prediction. If "win" means "price moved the same direction as the signal by 5%", and 35.8% of the time it did — then 64.2% of the time price moved in the OPPOSITE direction by 5%. That is not marginal. That is a pronounced anti-signal that every other system I know of would stop feeding into a multi-source vote.

**Which is better-supported:** Mine. Lead correctly identifies the mechanical fact but doesn't follow it to its logical conclusion: this source should be inverted or removed from convergence voting entirely. A source with this profile at this sample size is no longer an open question. It is an answered one.

---

### Divergence B: The min_domains=3 Problem — Constraint or Feature?

**My reasoning chain:** min_domains=3 is philosophically sound (implements Law 2) but operationally broken when the domains are measuring the same underlying phenomenon (retail sentiment via three different lenses). Three bad independently-pointing signals do not converge to a good signal.

**Lead's reasoning chain:** Lead frames this as an "edge leak" — min_domains=3 blocks MIDGE's best-documented signal (session sweeps) unless unrelated signals coincide. Lead recommends either a fast_track flag for BACKTEST_DERIVED patterns or a parallel direct-output path bypassing the convergence engine.

**Where we agreed on the problem, diverged on the fix:** I called for removing unqualified noise sources (Google Trends, StockTwits) from the convergence vote. Lead called for bypassing min_domains for high-quality patterns. These are complementary, not contradictory, but they have different risk profiles.

**Which is better-supported:** Lead's Option B (parallel direct-output path for session_sweep_ifvg signals bypassing convergence) is better-supported for *near-term financial utility*. It gets the one documented-edge signal firing without touching the convergence framework. My fix (removing noise sources from voting) is better-supported for *long-term signal quality*. Both should be done. Lead's fix first, mine second.

---

### Divergence C: Congressional Trades — Are They Useless or Mis-timed?

**My reasoning chain:** Congressional trades at 16.4% win rate on 53 samples are well below random. The likely cause is the 30-45 day STOCK Act reporting lag — the edge exists at trade execution, not at disclosure. Recommendation: add `reporting_lag_days` field and treat the signal as 30-45 day old information in the freshness calculation.

**Lead's reasoning chain:** Lead also identified the sell-side skew in predictions (every recent sec_form4 prediction was "down") and the cluster_detector threshold of 3+ insiders buying as a stronger filter. Lead did not specifically address the 30-45 day reporting lag as a lag-compensation opportunity.

**Beta's reasoning chain:** Beta did not address congressional trade quality.

**Where I diverged from both:** I identified the lag-compensation mechanism as actionable. Lead identified the watchlist problem (too many mega-cap tech names where insider selling is routine and uninformative). These are both correct and I missed the watchlist angle — Lead caught something I didn't. However, lag compensation is the more fundamental issue: even with a diverse watchlist, a signal 30-45 days stale is priced in regardless of which stocks are watched.

**Which is better-supported:** Both fixes are needed and independent. Lead's watchlist fix is easier to implement and immediately improves signal quality. My lag-compensation fix addresses the deeper structural problem of treating stale information as fresh.

---

### Divergence D: The Organism Tax — How Bad Is It?

**My reasoning chain:** Roughly 60% of per-step compute is organism maintenance with no trading pathway. ~50 of 80 registered systems fire step hooks for zero financial contribution.

**Beta's reasoning chain:** Beta did not quantify organism tax. Beta focused on concurrency and persistence — the systems that exist and how reliably they run.

**Lead's reasoning chain:** Lead focused on latency in the signal pipeline (2-day EDGAR delay + rotation lag), not compute overhead.

**Where we diverged:** Beta is an SRE — they care about whether existing systems run correctly, not whether those systems should exist. Lead is focused on signal latency, not organism overhead. I am the only auditor who examined whether the organism infrastructure earns its compute cost.

**Which is better-supported:** My framing is correct but my 60% estimate is rough — it is based on system count ratios without actual profiling. Beta's finding PA-1 (CorrelationTracker is O(n² × m²)) and PA-2 (prune on every record_signal) gives more concrete evidence of where non-trading overhead actually bites. Beta found the specific hot paths; I identified the general overhead category. Combined they are stronger than either alone.

---

### Divergence E: The Discovery Log Alert Storm — Causes and Severity

**My reasoning chain:** I identified the alert storm in discovery_log.jsonl (20+ identical alerts in one second) and traced it to the circadian mismatch between step time and wall-clock time. The dedup check compares wall-clock seconds which are non-monotonic in fast loops.

**Lead's reasoning chain:** Lead independently confirmed the alert storm with specific evidence (CONV-20260227-0001 through CONV-20260227-0021) and correctly identified the root cause: `_last_alert_time` is updated AFTER multiple callers pass the check. Lead provided a specific lock-based fix with monotonic clock replacement.

**Beta's reasoning chain:** Beta identified the same race condition (RC-1) but characterized it as "currently safe because collection is main-thread-only, but fragile." Beta was less alarmed because the immediate concurrency path is not active.

**Where we diverged:** Beta is technically more precise — the current concurrency model does not actually create the race via threading. Lead's framing is more accurate for the actual observed behavior: the storm happened during a marathon run, which means either the step loop called check_convergence() in a tight burst within one wall-clock second, or the dedup interval is expiring and re-firing within a single second at high step rates. The race is real even without multi-threading if step cadence is fast enough.

**Which is better-supported:** Lead's characterization and fix are better-supported. Beta's "currently safe" assessment is too reassuring given that the corruption already happened in production. Lead's monotonic clock + lock fix is correct regardless of the concurrency path.

---

### Divergence F: Data Contamination — How Serious Is the Mock Data?

**My reasoning chain:** I called out that 15 hypotheses generated, 0 promoted, 3 active suggests the validator is not producing tradeable outputs. I noted Thompson data tells a sobering story but did not dig into the actual data files for contamination.

**Lead's reasoning chain:** Lead found a concrete 2027-timestamp bug in predictions.jsonl (`sec_form4:GOOGL:01/25/2027`), documented the two-schema coexistence in predictions.jsonl, and traced mock data contamination in outcomes.jsonl (49.93% 1-day return for AAPL — physically impossible).

**Beta's reasoning chain:** Beta found test data contamination in pair_outcomes.json ("a|b", "a0|b0" entries) and the critical finding that these ghost pairs are corrupting hypothesis generation priority ordering.

**Where I missed badly:** I missed all three concrete data contamination findings. Lead and Beta both went directly to the data files; I analyzed the data in aggregate from thompson_distributions.json but didn't examine predictions.jsonl, outcomes.jsonl, or pair_outcomes.json at the line level. This is my biggest gap.

**Consequence:** The Thompson distributions I analyzed as representing real learned signal quality are partially contaminated by mock data. My conclusion that "congressional at 16.4% is well below random" may be accurate, but the finra_short and sec_form4 numbers I cited could be partially influenced by mock outcomes with impossible returns. This weakens my signal quality verdict somewhat — I stand behind the direction of the conclusions but the specific numbers need the contamination removed first.

---

## 2. Agreements — Where Independent Analysis Converged

**No execution path** (all three agents): This was independently identified by all three. Alpha called it "MIDGE cannot trade." Lead mapped the exact execution gap in the signal flow. Beta characterized it as "Fatal for financial utility." Agreement is complete and the finding is well-evidenced.

**Thompson data is thin on most sources** (Alpha + Lead): Both identified that 29 of 46 distribution keys have zero real observations. Independent convergence on this point.

**finra_short is an anti-signal** (Alpha + Lead): Both reached this conclusion from different entry points. Alpha from the win rate math. Lead from the "marginal edge" observation (though Lead undersold the severity).

**Session sweeps are MIDGE's most credible edge** (Alpha + Lead): Both identified the sweep + IFVG backtest results as the one source with real quantified edge. Agreement is complete.

**Static confidence values in adapters are not calibrated to Thompson reality** (Alpha + Lead): Both identified the disconnect between `from_congressional_trade` setting confidence=0.75 while Thompson shows mean=0.164. Lead called this the "confidence score disconnect"; I called it "geometric mean makes this worse." Same finding, same fix.

**yfinance rate limiting is unmanaged** (Alpha + Beta): I noted the 15-minute delay for free tier data. Beta documented that the accelerate_learning.py pipeline left ~3,382 predictions unresolved due to rate limits, and that there is no backoff or circuit breaker. Same finding, Beta had more specific evidence.

---

## 3. Gaps

### What Lead Found That I Missed

**The 2027-timestamp bug.** `sec_form4:GOOGL:01/25/2027` in predictions.jsonl is a prediction that will never mature. This is a concrete data integrity problem, not a theoretical one. I did not examine predictions.jsonl at the line level.

**The two-schema coexistence in predictions.jsonl.** Old format (entry_price, target_price fields) and new format (signal_id, outcome_window_days fields) coexist. Old format records have `entry_price=0.0`, making price-based outcome evaluation meaningless. I did not detect this.

**The 4-hour dedup interval and its specific race condition.** I identified the alert storm as a problem; Lead provided the exact mechanism and a correct fix.

**Rotation dilution math.** Lead calculated that with 19 sources and 3 concurrent slots at 50-step cadence, each source gets a slot every ~315 steps. This is concrete and I did not quantify it.

**The discovery log's fake early entries.** Lead identified that early discovery_log.jsonl entries (2026-02-08) are synthetic bootstrap records, and that CONV-20260225-0001's "crypto" domain doesn't correspond to any configured data source — evidence of an earlier MIDGE version. I missed this.

### What Beta Found That I Missed

**The Thompson history JSONL write without a lock.** This is the bug that already caused a full rebuild (per MEMORY.md). It is not fixed. Beta found it; I did not look at the write path.

**Non-atomic write_text() calls vs. the correct os.replace pattern.** Beta identified that ThompsonSampler uses os.replace (correct) but four other files use write_text (incorrect, truncation-vulnerable). I did not audit the write atomicity.

**Test data contamination in pair_outcomes.json.** "a|b", "a0|b0" through "a6|b6" ghost entries. This is production-affecting and I missed it entirely.

**Hypothesis engine TOCTOU race condition.** Background validation thread and main thread can both call promote/retire without a lock between the check and the mutation. I did not analyze the hypothesis lifecycle concurrency.

**The registered signals set growing unbounded.** O(n log n) serialization on every registration batch will eventually create measurable main-thread latency. I did not analyze the OutcomeCollector's write path.

**The CorrelationTracker O(n² × m²) complexity.** I noted the organism tax as a general concern; Beta found the specific hot path.

**run_service.bat infinite crash-restart loop on Python env failures.** This is an operational gap I did not examine.

### What I Found That Lead and Beta Missed

**The reward signal calibration problem.** Market-role agents are capped at 0.5 reward from market actions while TaskPool abstract tasks return up to 1.0. This teaches the VDN Q-table that abstract task completion matters more than market intelligence. Neither Lead nor Beta examined the reward signal structure or its financial implications.

**The circadian rhythm's lack of market hours awareness.** The circadian system runs on step time while markets run on wall-clock time. MIDGE's "morning exploration phase" bears no relationship to market open. Neither Lead nor Beta flagged this as a structural problem.

**The endocrine signal as dead-end telephone game.** "Convergence alert → dopamine → exploration bias → agent explores more" produces zero financial output. The endocrine layer between market signal and market action is pure overhead. Lead documented the output gap; I traced the specific mechanism by which the endocrine layer absorbs signal energy without producing tradeable output.

**The Google Trends + StockTwits domain problem is more than thin data.** I made the specific point that social_sentiment, google_trends, and stocktwits_sentiment all measure the same underlying phenomenon (retail sentiment) and including all three satisfies min_domains=3 while capturing only one information source. Lead and Beta both flagged thin data on these sources, but neither made the domain independence point: three retail sentiment lenses voting together is not three-domain confirmation.

**The 0-promoted-hypothesis problem at convergence_state.json step 1600.** 30 generated hypotheses, 0 promoted. This is live evidence that the hypothesis pipeline produces no trading-relevant output. Beta mentioned it in passing in PA-5; I identified it as evidence that the RSI Layer 2 loop does not close. Lead's signal flow diagram stops at the pheromone marker; the hypothesis pipeline dead-end is a parallel non-closing loop.

**Competitive positioning.** Neither Lead nor Beta compared MIDGE to alternatives (Bloomberg, QuantConnect, a 200-line Python script). I am the only one who asked "is this better than the alternatives?" The answer has important implications for where to focus development effort.

---

## 4. Surprises — Findings That Changed My Thinking

**Beta's CF-1 is not a new bug — it's an unfixed old one.** The Thompson history JSONL write without a lock is the exact bug that caused the documented rebuild. I assumed the rebuild had identified and fixed the root cause. It did not. The rebuild procedure was added; the bug remains. This is more alarming than a new bug because it means the next marathon run is likely to corrupt the Thompson history again, require another rebuild, and the cycle repeats. This is not a theoretical risk — it has a confirmed recurrence probability based on prior occurrence.

**Lead's prediction schema coexistence is a stealth problem.** Two different prediction formats coexisting in predictions.jsonl means the OutcomeCollector is computing Thompson updates with some predictions that have `entry_price=0.0`. I had assumed the outcome data quality was merely thin. It is not just thin — some records are structurally invalid as price predictions. This changes how I think about the Thompson data I analyzed: the contamination may be more severe than the win rate numbers alone suggest.

**Beta's pair_outcomes.json contamination explains the 0-promotion rate.** I assumed 0 promoted hypotheses out of 30 generated was a validator gate problem. Beta's finding suggests an alternative mechanism: test fixture ghost entries ("a|b" with 21 retirements, "a0|b0" etc. with undeserved priority bonuses) are competing for and winning hypothesis generation priority slots. The real source pairs — `sec_form4→finnhub_earnings` — may not be getting enough generation attempts because ghost pairs consume the priority budget. This is a different root cause than I identified, and it is more tractable to fix.

**Lead's rotation dilution math reframes the sensing architecture.** I knew each source was polled periodically. Lead's quantification — each source eligible only every ~315 steps — means that for a fast-moving opportunity (e.g., session sweep that opens and closes in 90 minutes), MIDGE may literally miss the window between the kill-zone start and the source's next rotation slot. This is not just latency — it is an architectural design issue for any time-sensitive signal. The session sweep detector's own kill-zone guard is not the binding constraint; the rotation scheduling is.

---

## 5. Synthesis — Top 5 Most Impactful Changes for MIDGE's Financial Effectiveness

These are ordered by financial impact, not by implementation difficulty.

### Priority 1: Build the Paper Trading Output Path

**What:** Convert ConvergenceAlert → TradeSignal → paper_trades.jsonl. Wire KellyPositionSizer output to TradeSignal generation. Wire OutcomeCollector to read real P&L from paper trades, not just directional accuracy.

**Why it is #1:** Without this, nothing else in MIDGE has financial consequences. The organism detects patterns. It learns which patterns are reliable. It generates hypotheses. All of this work ends in a pheromone marker and a heartbeat file. The entire RSI Layer 1→2→3 architecture cannot produce return data because it has never placed a trade. Paper trading costs nothing and closes the feedback loop between pattern quality and dollar P&L. Every other improvement on this list improves pattern quality; this one is the difference between a research instrument and a trading instrument.

**Specific action:** In `sensing_hook.py:_collect_one()`, after `check_convergence()` returns alerts: if `alert.confidence > 0.75` AND `alert.strength > 0.65`, instantiate TradeSignal, write to `data/midge/paper_trades.jsonl`. Wire Kelly sizing from `ctx._latest_kelly`. PaperTradingBook class to track P&L.

---

### Priority 2: Fix the Thompson History Lock and Purge Data Contamination

**What:** Two independent fixes that both protect the Bayesian learning signal.

Fix A (Beta CF-1): Add the `_log_update()` call inside `self._lock` in thompson_sampler.py. One line.

Fix B (Lead Issue 5 + Beta CF-3): Remove mock outcomes from outcomes.jsonl (those with impossible returns like 49.93%), remove ghost entries from pair_outcomes.json ("a|b" through "a6|b6"), fix the 2027-timestamp prediction in registered_signals.json, rebuild Thompson distributions from cleaned outcomes.

**Why it is #2:** The Bayesian learning loop is MIDGE's core competitive advantage over a simple signal script. If the Thompson distributions are contaminated, every confidence score MIDGE computes is wrong. The discovery log alert storm (20+ duplicate entries) is already corrupting the RSI Layer 2 training data. The pair_outcomes ghost entries may be suppressing legitimate hypothesis generation. Fixing data integrity before adding new capabilities is architectural discipline, not housekeeping.

**Note:** This has immediate urgency because CF-1 (the unfixed lock bug) will corrupt the Thompson history again on the next marathon run.

---

### Priority 3: Unlock Session Sweeps from min_domains=3 AND Excise Synthetic Domain Sources

**What:** Two complementary signal quality fixes.

Fix A (Lead Rank 2): Create a parallel direct-output path for session_sweep_ifvg signals that bypasses the convergence engine when quality >= 0.65 (Elite tier). This makes MIDGE's best-documented edge (PF 1.84) actionable without waiting for coincidental multi-domain agreement.

Fix B (Alpha Kill List #5): Remove google_trends, stocktwits_sentiment, and social_sentiment from the min_domains count. Allow them to be tracked and logged as observer signals. Remove them from convergence voting until they accumulate >= 100 real-outcome samples with > 50% win rate. Replace their domain slots — if a 3-domain requirement must be met, require that the 3 domains be genuinely independent: technical, fundamental, and positioning. social/retail sentiment sources are not independent of each other and do not add orthogonal information.

**Why it is #3:** The session sweep pattern has quantified edge from real backtesting. It is currently blocked from generating any output by a law-compliance requirement designed for sources without documented edge. Separately, the convergence engine is currently satisfiable by combining three retail sentiment sources — which means a bullish convergence alert can be generated without any fundamental or positioning confirmation. Both fixes improve signal quality; the first makes good signals fire, the second stops bad combinations from firing.

---

### Priority 4: Congressional Lag Compensation + Watchlist Diversification

**What:** Two structural improvements to the fundamental signal layer.

Fix A (Alpha Build List #5): Add `reporting_lag_days` field to the congressional trade signal adapter. In the convergence window freshness calculation, subtract the reporting lag. A congressional trade disclosed today was executed 30-45 days ago — treat it as 30-45 day old information, which means it typically falls outside a 72-hour convergence window. This correctly models what the data shows: congressional trades at 16.4% win rate are priced in by disclosure time.

Fix B (Lead Rank 7): Replace 3-4 mega-cap tech watchlist slots with mid-cap names where insider buying is more informative. The current watchlist (AAPL, MSFT, GOOGL, AMZN, NVDA, META + defense primes) skews toward names where routine RSU vesting and estate-planning sales dominate Form 4 filings. Mid-cap industrials, biotech, and energy names produce more informative insider signals.

**Why it is #4:** The fundamental signal layer (congressional + sec_form4) has theoretical backing but weak empirical results. The lag compensation fix correctly models the STOCK Act timing reality. The watchlist fix improves the quality of Form 4 signals entering the system. Combined, they have a reasonable chance of lifting congressional and sec_form4 Thompson means toward or above 50% over the next data accumulation cycle. If they don't, we have confirmed these sources lack edge and can remove them.

---

### Priority 5: Wire Circadian Rhythm and Agent Rewards to Market Reality

**What:** Two organism-level fixes that align MIDGE's internal dynamics with market behavior.

Fix A (Alpha Kill List #1): Wire CircadianRhythm to wall-clock time with phases: pre-market (4-9:30 ET), regular-hours (9:30-16 ET), after-hours (16-20 ET), overnight (20-4 ET). Integrate a MarketCalendar (pandas_market_calendars) to suppress data fetches during market holidays and early closes.

Fix B (Alpha Kill List #7): For agents in market roles (SEC_WATCHER, CONTRACT_TRACKER, MARKET_ANALYST, HYPOTHESIS_EXPLORER, HYPOTHESIS_VALIDATOR), cap the TaskPool exploit reward at 0.3 and raise the market action reward ceiling to 0.8. The current calibration (market actions: max 0.5, TaskPool exploit: up to 1.0) is teaching the VDN Q-table that abstract task completion matters more than market intelligence.

**Why it is #5:** These are organism coherence fixes, not trading infrastructure. MIDGE will function without them. But the circadian mismatch means MIDGE is fetching stale overnight futures data during high-energy circadian phases and potentially running in consolidation mode during market open. The reward miscalibration means market-role agents are learning the wrong objective function. Over thousands of steps, this shapes behavior away from financial utility. These are not urgent — they are important for long-run quality.

---

## Closing Assessment

Lead and I are aligned on the architectural diagnosis: MIDGE is a complete signal research organism with no output path. Lead provided more granular evidence; I provided more adversarial framing. The synthesis is the same.

Beta found the systemic reliability risks that neither Lead nor I examined. CF-1 (unfixed Thompson lock bug) is the most immediate operational risk — it will cause another rebuild. The non-atomic write_text calls are a genuine Windows-specific hazard that neither of us caught.

Where I diverge most sharply from both: they are both sufficiently alarmed about the execution gap but neither named what needs to happen on a specific timeline. MIDGE without an execution path is not a trading system regardless of how sophisticated the signal detection becomes. This is not a roadmap item — it is the question of whether the system is building toward financial utility at all.

The organism architecture is coherent and internally consistent. It is also insulating MIDGE from the market reality it is supposed to engage with. Dopamine as a trading response, steps as a time unit, abstract task rewards competing with market intelligence rewards — these are the places where the metaphor is actively working against the mission.

The good news: all five priority fixes are tractable. None requires architectural redesign. MIDGE's strongest genuine asset — the Thompson Sampling + convergence synthesis with multi-domain confirmation — is sound. It just needs a clean data foundation, an output path, and calibration of its organism dynamics to market time.
