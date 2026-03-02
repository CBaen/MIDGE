# MIDGE Architecture Audit — Beta Cross-Review (Systems Reliability Engineer)

**Auditor Role:** Witness Beta — Systems Reliability Engineer
**Cross-Review Date:** 2026-03-01
**Reviewing:** Lead (Signal Pipeline Architect) + Alpha (Adversarial Market Practitioner) findings

---

## 1. Reasoning Divergence Points

### Divergence A: Is the Output Gap "Fatal" or a Design Phase?

**Lead's position:** The missing trade execution path is "Fatal for financial utility." Lead identifies the `TradeSignal` dataclass as defined but never instantiated, maps the alert propagation chain terminating at a pheromone deposit, and recommends building the output path as Rank 1 priority.

**Alpha's position:** Agrees there is no execution layer, but frames it as MIDGE being "a research instrument, not a trading system" — treating the gap as a category error rather than a missing feature.

**My position:** Both are correct on the facts but diverge on implication. My concern is downstream: **building the output path before fixing the data integrity issues creates a system that trades on corrupted Bayesian beliefs.** Specifically:
- CF-3 (test data in pair_outcomes.json) corrupts hypothesis priority ordering
- The mock outcomes contaminating outcomes.jsonl (Lead's Issue 5) would feed fake P&L into any paper trading book built on top
- The deduplication race condition (Lead's Issue 2, my RC-2) means the first convergence alert fired in a real trading context could emit 20 duplicate trade signals

The sequencing of Lead's recommendations is the real problem. Lead's Rank 3 (fix deduplication) should happen before Rank 1 (build output path). Building an output path over a broken deduplication mechanism delivers 20 simultaneous paper trade entries on the same signal. **My reasoning: fix the plumbing before turning on the water.**

**Which conclusion is better-supported:** My sequencing concern is more operationally grounded. Lead's urgency on the output path is correct in isolation, but Lead does not examine what happens when a corrupted signal storm hits an output layer.

---

### Divergence B: Finra Short Win Rate Interpretation

**Alpha's position:** finra_short at 35.8% win rate across 1,263 samples is "anti-signal" — including it in convergence calculations degrades signal quality. Alpha recommends either flipping its interpretation (contrarian) or excluding it from convergence voting.

**Lead's position:** Documents finra_short as "most trained distribution" and notes "35.8% win rate at 5% threshold is marginal edge — barely above random if the 5% move happens in the wrong direction with similar frequency." Lead frames it as thin edge, not anti-signal.

**My divergence from both:** The win rate alone is not sufficient to reach either conclusion. The critical question is: **what is the base rate of a 5% move in either direction for any given ticker in 45 days?** If the base rate of a 5% move in the correct direction is 40%, then finra_short at 35.8% is mildly anti-predictive. If the base rate is 25%, finra_short at 35.8% is actually positive edge. Neither Lead nor Alpha computes this base rate — they compare 35.8% to 50% (coin flip), but a directional 5% move within 45 days is NOT a coin flip. The outcome_collector's `SUCCESS_THRESHOLD_PCT=5.0` means a prediction is "correct" if the stock moves 5% in the predicted direction within the outcome window — a bar that stocks clear on volatility alone in many environments.

**Which conclusion is better-supported:** Alpha's "anti-signal" call may be overconfident without the base rate calculation. Lead's "marginal edge" is more appropriately uncertain. My recommendation: before excluding or flipping finra_short, compute the null-hypothesis win rate for a 5% directional move in the current volatility environment, then compare. The data already exists in outcomes.jsonl.

---

### Divergence C: The min_domains=3 Requirement

**Lead's position:** min_domains=3 blocks MIDGE's best-documented edge (session sweep, PF=1.84) from generating actionable output. Recommends either a `fast_track` flag or a parallel bypass path for BACKTEST_DERIVED patterns.

**Alpha's position:** min_domains=3 is correct as a quality filter in principle, but the domains it combines include fundamentally correlated signals (social_sentiment, google_trends, stocktwits_sentiment all measuring retail sentiment through different lenses), making the "3 independent domains" assumption false. Alpha frames the problem as domain independence, not the threshold itself.

**My divergence from Lead:** Lead's Option B (parallel bypass for session_sweep_ifvg) introduces a **two-track system with different quality standards.** Once a bypass path exists, it will expand. The next high-conviction signal will want a bypass too. Then we have three tracks. Law 2 (No Bare Dyads) was not chosen arbitrarily — it prevents single-source confirmation bias. My reliability concern: a bypass path is a maintenance surface that erodes the invariant over time.

**My agreement with Alpha:** Alpha's diagnosis is better. The problem is not the threshold — it is that the domains are not actually independent. The fix is not a bypass; it is auditing domain assignments so that correlated signals are grouped into the same domain, not treated as independent votes. If social_sentiment and stocktwits_sentiment both contribute to convergence, they should compete within the "retail_sentiment" domain rather than each counting as a separate domain vote.

**Which conclusion is better-supported:** Alpha's domain-independence analysis is more rigorous. Lead's bypass recommendation has a clean implementation path but introduces architectural debt that will compound.

---

### Divergence D: The Organism Tax

**Alpha's position:** ~50 non-market systems fire step hooks for zero trading benefit. Estimates 60% of per-step compute is organism maintenance. Recommends audit-logging rather than active stepping for reproductive, lymphatic, vestibular, and similar systems.

**My divergence from Alpha:** This conclusion is reached without profiling data. Alpha acknowledges this: "The StepTimer snapshot was not present." My reliability lens: **you do not strip systems before you profile them.** The StepTimer is already wired (I documented it in PA-3 as O(n log n) sorts per call). Run a marathon session, capture the step_timer_snapshot.json, then identify which systems actually consume meaningful CPU. The actual bottleneck may be convergence_alerter's O(total_signals) prune-on-every-record (my PA-2) rather than the biological hooks.

More importantly, the biological systems serve as circuit breakers. If thermoregulation or the renal_filter are detecting pathological states in the organism, disabling them removes the only diagnostic signal for runaway resource consumption — the exact scenario that leads to the disk-full silent failure I documented in OR-3.

**Which conclusion is better-supported:** Alpha's observation that the organism tax is real and unmeasured is correct. Alpha's recommendation to strip systems before profiling is operationally risky. My position: profile first, then surgically remove confirmed dead weight.

---

### Divergence E: Circadian Rhythm Fix

**Alpha's recommendation:** Wire CircadianRhythm to actual UTC wall-clock time, keyed to market sessions (pre-market 4-9:30 ET, regular hours 9:30-16 ET, after-hours 16-20 ET, overnight 20-4 ET).

**My concern Alpha did not raise:** This is a consequential architectural change that affects agent reward calibration, exploration/exploitation balance, and any temporal pattern in Thompson learning. The circadian rhythm's current behavior (step-based) is at least deterministic and reproducible. Wall-clock time introduces a new variable: the system behaves differently depending on what time of day it is initialized. A cold start at 2am produces a different learning trajectory than a cold start at 9am, even on identical market data.

**The reliability failure mode:** Wall-clock circadian rhythms make test runs environment-dependent. Tests that pass at 10am fail at 2am if they depend on circadian phase behavior. Alpha's recommendation is correct directionally but needs test isolation strategy built alongside it.

---

## 2. Agreements

**Three-way agreement: No trade execution path exists.**
Lead, Alpha, and I independently confirmed that ConvergenceAlert propagation terminates at a pheromone marker with no external output. All three found the `TradeSignal` dataclass unused. The alert storm deduplication race is independently dangerous — Lead found it in the discovery log, I found the TOCTOU race in the deduplication guard, Alpha identified the endocrine pipeline as the wrong output channel.

**Two-way agreement (Lead + Beta): Prediction data integrity is broken.**
Lead identified the 2027-timestamp artifact in predictions.jsonl and the impossible AAPL 49.93% return in outcomes.jsonl. I independently found test fixture contamination in pair_outcomes.json (CF-3). Both analyses reach the same conclusion from different files: mock data and test artifacts are mixing with production Bayesian learning data.

**Two-way agreement (Alpha + Beta): Thompson weight mapping is too narrow.**
Alpha notes a 35.8% source gets Thompson weight 0.858 — "not sufficient to neutralize an anti-signal." I independently noted that legacy key names in learning_config (sec_edgar = 0.95) have vestigial Thompson distributions (my analysis of thompson_distributions.json in CF-4 appendix) that mislead warm-starts. Both analyses reach the same root: the [0.5, 1.5] weight range is too compressed to distinguish bad sources from neutral ones.

**Two-way agreement (Alpha + Beta): Observation without profiling data.**
Alpha estimated 60% organism tax without a StepTimer snapshot. I documented StepTimer exists but found no snapshot either. We agree the measurement tool is built but has not produced its primary output yet.

---

## 3. Gaps

### What Lead Found That I Missed

**Signal direction bias in Form 4 data (Lead Issue 6).** Lead examined the predictions.jsonl and found every recent sec_form4 prediction is "down" — attributing this to tech mega-cap executive sell patterns (RSU vesting, planned sales). I did not examine the directional distribution of the prediction file. This is an important signal quality observation: the watchlist selection creates systematic sell-side skew that makes bullish convergence from sec_form4 nearly impossible during tech sell periods. I would have caught this had I examined predictions.jsonl with the same thoroughness Lead applied.

**Latency estimation (Lead Section 2).** Lead traced the latency from SEC EDGAR filing to MIDGE convergence check, estimating 2 days + rotation lag. I did not quantify the end-to-end latency. This matters for assessing whether MIDGE can capture edge before it is priced in — which is precisely the concern Alpha raises about congressional trades at 30-45 day reporting lag.

**Slow-signal domain persistence (Lead Rank 5).** Lead proposed per-domain convergence windows keyed to known alpha decay rates. I identified the 72-hour window as a PV-4 growth problem but did not analyze which signal types are actually losing signal value before they can contribute to convergence. Lead's insight that COT (weekly), congressional (monthly), and SAM.gov (months) all wash out of a 72-hour window is operationally important and I missed it.

### What Alpha Found That I Missed

**Reward signal calibration bias (Alpha Problem 3 / Kill Item 7).** Alpha identified that market-role agents are capped at 0.5 reward while TaskPool exploit can return up to 1.0. This teaches the VDN Q-table that abstract task completion outweighs market intelligence. I analyzed the agent action dispatch system but did not examine the reward magnitude calibration or its effect on what the Q-table learns to value. This is a subtle but real misalignment between organism goals and trading goals.

**Session sweep timing reality (Alpha Build Item 1).** Alpha noted that MIDGE fetches session sweep signals via yfinance 1-minute candles on weekends, when futures markets are closed. The sensing hook runs 24/7 regardless of market hours. I identified the sensing hook's ThreadPoolExecutor scaling and its skip-if-busy pattern, but did not examine whether the underlying data fetches are temporally valid. Alpha's market-hours observation is correct and I should have caught it — the SessionSweepDetector's kill zone guard only prevents signal generation within MIDGE, not the API calls themselves.

**Full pipeline backtest gap (Alpha Build Item 6).** Alpha notes that sweep_backtest.py tests the session sweep detector in isolation, not the full convergence pipeline. My analysis focused on persistence integrity for existing data files. I did not ask the higher-level question: has the convergence pipeline ever been validated end-to-end against historical data? The answer is no, and Alpha is correct that this is the most important missing validation.

### What I Found That Neither Lead Nor Alpha Found

**CF-1: Thompson history JSONL write without lock.** The `_log_update()` in thompson_sampler.py is called from `update()` without holding `self._lock`, while `_save_distributions()` IS locked. This asymmetry is the precise bug pattern that previously caused the marathon file-lock corruption per MEMORY.md. Neither Lead nor Alpha identified this specific lock gap.

**CF-2: Non-atomic write_text() inconsistency.** The codebase has both atomic writes (`os.replace(tmp, path)` in _save_distributions) and non-atomic writes (`Path.write_text()` in learning_config, hypothesis_engine, hypothesis_generator, step_timer). This inconsistency means the developer knew about atomic writes but did not apply the pattern consistently. Neither other agent identified the pattern inconsistency.

**RC-2: TOCTOU race in hypothesis registry.** The background validation thread and agent-triggered validation both call `self._promote()` / `self._retire()` with a check-then-act pattern on hypothesis status, but no lock guards the check-to-act window. This specific race pattern was not in Lead or Alpha findings.

**PV-3: Registered signals set growth.** The `self._registered` set in OutcomeCollector grows unboundedly. The write_text() call serializes the entire sorted set on every batch, O(n log n) per batch, blocking the main thread. This will become a meaningful latency issue over months of operation.

**EG-4: run_service.bat infinite loop on env failures.** The bat file restarts on any exit including ImportError or corrupted venv. No ERRORLEVEL check, no max restart count. A corrupted Python environment produces an infinite 30-second crash loop that fills logs until disk is full — at which point OR-3 (no disk space guard) takes over and silently swallows all writes.

**OR-4: Partial state recovery is undefined.** If config_snapshot.json exists but retirement_window.json is missing, the system boots silently with an empty retirement window. Wire 2 of meta-learning then observes 0% retirement rate and loosens min_correlation, potentially flooding hypothesis generation. There is no documented or tested recovery procedure for mixed partial states.

---

## 4. Surprises

**Lead's alert storm evidence.** The discovery_log.jsonl containing 20+ identical CONV-20260227 entries logged within one second was more concrete than I expected. I had identified the TOCTOU race condition in the deduplication guard theoretically (RC-2), but Lead's evidence shows it already fired in production. This changes my risk rating for RC-2 from "Medium likelihood" to "Confirmed occurred."

**Alpha's reward ceiling analysis.** The finding that market-role agents are deliberately capped at 0.5 reward while non-market TaskPool can return 1.0 surprised me. I had assumed the market_actions.py dispatch was a complete replacement for TaskPool for market agents. Alpha's Kill Item 7 reveals it is only a partial replacement — agents still fall through to TaskPool for some action types, and when they do, the Q-table receives higher reward signals for abstract tasks than for market discoveries. This changes my view of the market_actions.py dispatch: it is not the problem solved I thought it was.

**Alpha's no-real-time-data observation.** I knew yfinance has polling latency, but Alpha articulates a consequence I had not traced: for ICT session sweeps specifically, which depend on 1-minute candle timing for kill zone confirmation, a 15-minute delay makes the entry signals unreliable. The quality score (displacement, FVG/ATR ratio, kill zone tier) is computed against stale candles. An "elite" signal computed 15 minutes after the actual sweep may correspond to a pattern that has already reversed. This matters for the backtest-to-live gap: the sweep_backtest.py uses historical data at the correct timestamps; live operation uses delayed data. The edge may exist in the backtest and not survive the latency.

**Lead's Thompson prior mismatch.** The `learning_config.py` source_reliability for `congressional = 0.75` seeds Beta(1.5, 0.5) as the prior, but the actual Thompson distribution shows mean=0.164 from 33 real samples. This means for any session that starts without a warm thompson_distributions.json, congressional trades are heavily overweighted for the first N steps. I had examined the thompson_distributions.json entries but did not cross-reference them against the learning_config priors. Lead caught a three-way misalignment (config prior vs. distribution reality vs. meta-learner adjustment target) that I missed.

---

## 5. What Breaks — Failure Mode Analysis for Lead and Alpha Recommendations

### Lead Rank 1: Build the Output Path (TradeSignal → Paper Trades)

**Failure mode 1: Alert storm to trade storm.**
The deduplication race condition (Lead Issue 2, confirmed occurred per discovery_log evidence) means the first time a convergence alert fires in the new output path, it may emit 20 simultaneous paper trade entries on the same signal. The PaperTradingBook Alpha specifies (with open positions, realized P&L) would have to deduplicate entries by signal_id, or it accumulates 20 phantom positions with 20× the Kelly-sized notional. The output path must be gated behind the deduplication fix, not built in parallel.

**Failure mode 2: Mock data in the P&L baseline.**
outcomes.jsonl contains the impossible 49.93% AAPL return record (Lead Issue 5). If the paper trading book feeds off the same outcome evaluation pipeline, it will periodically record fictitious P&L that inflates the book's Sharpe ratio and misleads Thompson updates. Clean the data before building the accounting layer.

**Failure mode 3: Kelly fraction with no account size.**
Lead notes Kelly sizing is already computed in `ctx._latest_kelly`. But Kelly fraction requires an account value to translate to a notional size. The KellyPositionSizer presumably computes a fraction (0.03). 3% of what? There is no account value configured anywhere in MIDGE. Without this anchor, the paper trading book either invents a notional or produces undefined behavior when converting fraction to position size. Lead's recommendation does not address this.

**Failure mode 4: Concurrent writes to paper_trades.jsonl.**
If `_collect_one()` in sensing_hook runs on the main thread and writes to paper_trades.jsonl, while the background backtest scheduler also potentially writes results to a file at the same time, and both share the same disk write path, the non-atomic file writes (CF-2) produce corrupted paper trade records on any crash. The paper trading book is only as durable as the file layer it writes to.

---

### Lead Rank 2: Bypass min_domains=3 for Session Sweep (Option B)

**Failure mode 1: The bypass path expands.**
Once a bypass exists for session_sweep_ifvg, every future high-conviction signal will request one. This is not theoretical — Lead's own framing ("explicitly backtested pattern") will apply identically to any future signal that completes a backtest. The maintenance surface grows. After three bypass exceptions, the min_domains=3 requirement is effectively non-operative for any source that bothers to run a backtest.

**Failure mode 2: Backtest regime mismatch.**
The sweep_backtest ran 48 days of historical data in a particular volatility regime. The elite-tier 45.3% win rate (quality >= 0.65) was computed in that specific regime. If MIDGE is currently in "sideways" regime (confirmed in convergence_state.json), and the backtest was conducted during a different regime, the 45.3% figure may not hold. Bypassing min_domains=3 means the bypass fires regardless of current regime — Alpha's circadian/market-time analysis applies here too. The session sweep bypass should at minimum be regime-gated.

**Failure mode 3: Domain accounting break.**
Law 2 tracking is enforced by ConnectionRegistry and TriadEnforcer, not by the convergence alerter domain count. A bypass in convergence_alerter.py does not affect the law-level audit. But if Guiding Light later asks "how many law-2-compliant alerts did we fire?", the answer becomes ambiguous — some alerts fired with 3 domains, others via bypass. The audit trail breaks.

---

### Lead Rank 3: Fix Alert Deduplication (Thread-Safe with Monotonic Clock)

**Failure mode 1: Monotonic clock drift across restarts.**
`time.monotonic()` resets on every process restart. If the deduplication state (`_last_alert_monotonic`) is not persisted to disk and recovered on restart, every restart resets the dedup timer to 0. The first 4 hours after every restart are unprotected — the system fires any convergence alert it encounters without suppression. The existing wall-clock approach at least uses real time that persists across restarts.

**Fix:** Use `datetime.now()` for the persisted comparison and `threading.Lock()` for thread safety around the check-and-update. Do not switch to monotonic unless the timestamp is also persisted.

**Failure mode 2: Lock contention path.**
Adding `self._dedup_lock` to `check_convergence()` means every step that calls `check_convergence()` (currently every step, per Alpha's overhead analysis) acquires and releases a lock. In CPython, uncontested lock acquisition is cheap. However, if HolonProxy, sensing hook, and an agent all call `check_convergence()` within the same step, lock contention is possible. The fix is correct but should use `threading.RLock()` if `check_convergence()` can be called recursively (e.g., from within a convergence callback).

---

### Alpha Kill Item 1: Wire CircadianRhythm to Wall-Clock Time

**Failure mode 1: Test determinism collapse.**
Every test that uses the circadian rhythm must now be time-aware. Tests that pass at 10am EST may fail at 2am EST if they depend on phase behavior. The entire test suite (3,119 tests) must be audited for circadian sensitivity, or a test-mode override must be built into CircadianRhythm that locks it to a fixed phase for test runs. Alpha does not mention this.

**Failure mode 2: Timezone configuration drift.**
"ET" (Eastern Time) changes between EST and EDT. The pandas_market_calendars library handles this correctly, but only if the system timezone is configured correctly. On Wardenclyffe (Windows 11, potentially UTC by default), the wall-clock comparison may use UTC while the market calendar uses ET. This is a classic off-by-one bug for calendar boundaries (holidays, early closes). The fix needs an explicit `pytz.timezone('America/New_York')` anchor, not an assumed local timezone.

**Failure mode 3: Continuous-mode weekend runs.**
MIDGE runs continuously via run_service.bat. On weekends, NYSE and CME are closed. If CircadianRhythm is walled to market hours, what phase does MIDGE enter on Saturday? Options: (a) all agents enter "rest" phase continuously, (b) the rhythm cycles based on next open, (c) the system pauses entirely. Alpha does not specify. Option (a) means all market agents are in rest for 60 hours (Friday close to Sunday open), during which any weekend gap events (geopolitical news, earnings pre-announcements) are not processed. Option (c) breaks the continuous learning assumption.

---

### Alpha Build Item 2: Paper Trading Execution Layer

**Failure mode 1: P&L contamination from stale data.**
The most important reliability concern for a paper trading book built on MIDGE's current architecture: yfinance has a 15-minute delay for free tier. Entry price for a paper trade is the current delayed price. Exit price for outcome evaluation is also delayed. But the delay may differ between entry and exit (different API calls at different times). The P&L calculation is entry vs. exit price from two asynchronous API calls with different delays, producing artificial slippage in the paper P&L that does not reflect real market behavior.

**Failure mode 2: No position deduplication.**
If the same convergence signal fires in two consecutive sensing cycles (e.g., the deduplication race fires twice), the paper book accepts two positions on the same signal unless there is ticker-level deduplication. Lead's paper_trades.jsonl write does not include deduplication logic.

**Failure mode 3: Outcome window mismatch.**
The OutcomeCollector uses OUTCOME_WINDOWS (1d, 5d, 10d, 30d, 45d per source type). A paper trading book needs a specific exit rule — either time-based (exit after N days) or price-based (stop loss / target). If the paper book exits at the OutcomeCollector's resolution window (which varies by source), the P&L is not comparable across sources. A paper trade based on session_sweep_ifvg exits in 1 day; a congressional trade exits in 45 days. Aggregating their P&L into a single book Sharpe ratio is mathematically invalid without normalization.

---

### Alpha Build Item 4: finra_short Anti-Signal Handling (Flip or Exclude)

**Failure mode 1: Contrarian flip introduces regime dependency.**
If finra_short is flipped (high short interest = contrarian long signal), the direction flip must be regime-aware. In a trending bear market, high short interest confirming the trend is NOT a contrarian signal — the shorts are right. In a sideways or recovering market, high short interest can be contrarian. A blanket direction flip without regime conditioning will misfire in bear regimes. The Thompson distribution does not currently capture this dimension.

**Failure mode 2: Exclusion creates blind spot.**
If finra_short is excluded from convergence voting, the convergence alerter loses the signal source with the largest sample size. The remaining convergence votes are from sources with thin data (< 30 samples each for most non-sweep sources). Excluding the most-observed source to chase statistical purity while accepting less-observed sources with apparent positive means (e.g., sec_form4 sideways regime at mean=0.578 from 5 samples) is backwards from a Bayesian perspective.

**The correct path Alpha did not specify:** Investigate whether finra_short's 35.8% win rate at a 5% directional threshold represents negative edge against the base rate (see Divergence B above). If the base rate of a 5% directional move in 45 days is 30%, finra_short at 35.8% is mildly positive. If the base rate is 45%, finra_short is anti-signal. Do not flip or exclude before computing the null.

---

### Alpha Kill Item 4: Disable 50+ Non-Market Step Hooks

**Failure mode 1: Removes diagnostic instrumentation before understanding resource profile.**
The renal_filter, senescence, and nociception systems detect runaway resource consumption. Disabling them before profiling means the first indication of a resource problem is a crash, not a logged warning. These systems are cheap to run per step (single method call checking a threshold) but provide the only early warning for the disk-full (OR-3) and memory growth (PV-3, PV-4) scenarios I documented.

**Failure mode 2: Breaks ConnectionRegistry invariants.**
The 385 triadic connections include connections to and from many of the "non-market" systems. Disabling those systems' step hooks while leaving their registry entries would produce TriadEnforcer warnings on every step — advisory-only, but noisy. Full removal requires also removing their triadic connections and updating all count-tracking documents (Document Parity Rule in CLAUDE.md). Alpha does not scope this dependency chain.

**Failure mode 3: Bootstrap layer ordering dependency.**
Some "non-market" systems are initialized early in bootstrap (Layers 1-10) and provide shared services (EventBus, ConnectionRegistry, HolonRegistry) that market systems depend on. Selectively disabling step hooks for non-market systems requires distinguishing "infrastructure systems" (cannot be disabled) from "purely biological monitoring systems" (can be audit-log-only). Alpha's list includes thermoregulation and nociception (safely disableable) but does not distinguish them from systems like the wiring layer or fractal_generator (which bootstrap Layer 33 depends on).

---

## Summary Assessment

**Lead's highest-risk recommendation:** Rank 1 (output path) without first fixing Rank 3 (deduplication). The failure mode is a trade storm on first convergence alert.

**Alpha's highest-risk recommendation:** Kill Item 4 (disable 50+ step hooks) without profiling first and without auditing infrastructure dependencies. The failure mode is removing diagnostic coverage for the exact resource issues I documented.

**The recommendation neither agent made that is most critical:** Fix CF-1 (Thompson history write without lock). This is the bug that previously required a full Thompson rebuild from 9,462 outcomes. It is still unfixed. It will corrupt the Bayesian learning state again under concurrent operation. All other improvements — paper trading, dynamic gates, meta-learning — learn from corrupted distributions until this lock gap is closed.

**Sequencing recommendation from a reliability lens:**
1. Fix CF-1 (Thompson history lock gap) — prevents next distribution corruption
2. Fix CF-3 + Lead Issue 5 (purge test contamination from pair_outcomes.json and outcomes.jsonl) — cleans the learning signal
3. Fix Lead Issue 2 / RC-2 (deduplication race) — prevents alert storms
4. Build output path (Lead Rank 1) — now safe to add because the signal quality is clean
5. Profile step overhead (StepTimer marathon run) — before any system removal
6. Address Alpha's organism tax based on profiling data — not before
