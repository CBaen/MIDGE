# MIDGE Architecture Audit — Lead Cross-Review
**Auditor Role:** Lead — Signal Pipeline Architect
**Date:** 2026-03-01
**Scope:** Synthesizing Alpha (Adversarial Market Practitioner) and Beta (Systems Reliability Engineer) findings against my own pipeline-lens analysis.

---

## 1. Reasoning Divergence Points

### Divergence 1: What "the most critical problem" is

**Where reasoning diverged:** All three agents agree no trade execution path exists. But our priority rankings diverge sharply on what comes second.

- **My ranking (Lead):** The discovery log deduplication failure (Issue 2) and corrupt outcome data (Issue 5) are the top systemic risks after the output gap — because they corrupt the learning signal itself.
- **Alpha's ranking:** The organism-vs-market-time mismatch and endocrine-pipeline dead-weight are listed before data integrity issues. Alpha treats the architectural impedance mismatch as a systemic crisis.
- **Beta's ranking:** Thompson history JSONL write without a lock (CF-1) is listed as the single most critical issue, citing that it already caused a full rebuild once.

**At what step did reasoning diverge?**

My lens entered at the signal flow level and followed data forward. I saw deduplication failure by reading discovery_log.jsonl directly. Alpha entered at the system-design level and asked "what should a trading system do?" before looking at data. Beta entered at the runtime layer and asked "what will break under load?" before looking at trading logic.

All three paths reach a valid critical issue. Beta's CF-1 (Thompson JSONL write without lock) is better-supported than my Issue 2 (deduplication race) because:
1. Beta cites MEMORY.md as direct evidence the exact failure mode already occurred in production ("Marathon file-lock corruption wiped distributions to Beta(1,1)").
2. I identified the dedup race at `_last_alert_time` update timing but Beta identified the deeper pattern: the lock protects the file write but not the in-memory dict mutation (RC-3, thompson_sampler.py lines 240-244).

**Beta's CF-1 analysis is better-supported than my Issue 2 assessment.** The dedup race I found is real, but Beta's Thompson thread-safety finding is more pervasive and already has a confirmed production failure. I give Beta the better conclusion here.

---

### Divergence 2: The finra_short assessment — anti-signal or noisy signal?

**Where reasoning diverged:**

- **My assessment (Lead):** finra_short has 1,265 observations at 35.8% win rate. I called it "marginal edge — barely above random if the 5% move happens in the wrong direction with similar frequency." I treated it as a weak but non-disqualifying signal.
- **Alpha's assessment:** Alpha explicitly calls finra_short an "anti-signal" and recommends either flipping its interpretation (high short interest → contrarian long) or excluding it from convergence voting entirely. Alpha's reasoning: "35.8% win rate is not 'slightly below neutral,' it is meaningfully worse than random" in a binary 5% directional prediction at the implied threshold.

**At what step did reasoning diverge?**

I stopped at "below random, large sample" and noted marginal utility. Alpha took the next step and computed the financial consequence: a source included at 35.8% win rate with a Thompson weight of 0.858 (weight = 0.5 + 0.358) is actively injecting anti-signal into the geometric mean confidence formula. The formula does not clip negative-edge sources — it just slightly down-weights them.

Alpha's conclusion is more financially precise. At 35.8% directional accuracy, including finra_short in a convergence vote means the system is being pushed toward the wrong direction 64.2% of the time that source fires. The Thompson weight of 0.858 barely compensates. **Alpha is right that this requires either a direction flip or exclusion, not just a down-weight.**

I missed this implication by thinking in terms of "sample confidence" rather than "financial edge direction." This is a genuine gap in my analysis.

---

### Divergence 3: The min_domains=3 requirement — mathematical law vs. operational cost

**Where reasoning diverged:**

- **My assessment (Lead):** min_domains=3 blocks MIDGE's best-documented edge (session sweep+IFVG, PF 1.84) from generating alerts because sweep fires in the `technical` domain only. I proposed Option B: a parallel direct-output path for session_sweep_ifvg that bypasses convergence. I framed this as "Option B preserves Law 2 for the convergence engine while allowing the explicitly backtested pattern to fire independently."
- **Alpha's assessment:** Alpha went further and challenged the premise of the domain independence assumption. Alpha's argument: "When social_sentiment, google_trends, and stocktwits_sentiment all point bullish, they are measuring the same underlying thing (retail sentiment) through three slightly different lenses, not three independent information sources." This means the 3-domain minimum can be satisfied by three correlated noise sources, which is worse than a single high-quality technical signal.

**At what step did reasoning diverge?**

I stopped at "the rule blocks the best signal." Alpha continued to ask "does the rule even provide the statistical benefit it claims?" — and found that the rule's assumption of domain independence is violated by the current source list. Three sentiment sources are not three independent witnesses; they are one witness measured three ways.

Alpha's analysis is architecturally deeper. The min_domains=3 constraint provides genuine value only when the 3 domains are informationally independent. The current source roster (social_sentiment, google_trends, stocktwits_sentiment all in the sentiment domain, or multiple congressional/government sources in the government domain) allows the system to satisfy Law 2's letter while violating its spirit.

**Combined conclusion:** Both paths are correct but at different levels. My fix (bypass for backtested edge) is necessary. Alpha's fix (enforce genuine domain independence, not just domain count) is sufficient to make the rule work as intended. Both changes should be implemented.

---

### Divergence 4: Organism overhead — cost vs. financial value

**Where reasoning diverged:**

- **My assessment (Lead):** I did not audit the organism overhead layer. I focused entirely on the signal pipeline and treated the organism infrastructure as background.
- **Alpha's assessment:** Alpha estimated "roughly 60% of per-step compute is organism maintenance rather than market intelligence" — a claim based on the ~50 non-market systems in the systems dict running step hooks. Alpha provided a list of specific systems to audit-log rather than actively step: reproductive_system, lymphatic_system, vestibular_system, thermoregulation, nociception, proprioception, renal_filter, microbiome, senescence, triage_classifier.
- **Beta's assessment:** Beta calculated the reward signal misalignment: market agents are capped at 0.5 reward from market actions while non-market TaskPool exploit can return up to 1.0, effectively teaching the VDN Q-table that abstract tasks matter more than market discoveries.

**At what step did reasoning diverge?**

I stayed within the market pipeline boundary. Alpha and Beta both looked at what was running AROUND the market pipeline and found that the organism substrate is consuming CPU and reward signal in ways that actively compete with market intelligence. These are distinct concerns:
- Alpha's concern: compute cost (CPU cycles not producing trading value)
- Beta's concern: learning signal misalignment (reward system teaching wrong priorities)

Beta's finding about reward misalignment is the more financially dangerous one. If the VDN Q-table is learning that TaskPool exploitation returns 1.0 reward while discovering a convergence alert returns 0.4, the agents are being conditioned to deprioritize the market intelligence function that MIDGE exists to perform. This is a foundational alignment problem, not just an efficiency concern.

**I missed both the organism tax and the reward misalignment entirely.** This is a significant gap in my pipeline-focused analysis.

---

### Divergence 5: Prediction timestamp error — scope of impact

**Where reasoning diverged:**

- **My assessment (Lead):** I identified the 2027 timestamp prediction (sec_form4:GOOGL:01/25/2027) and noted it contaminated the sec_form4 Thompson distribution (11 observations).
- **Beta's assessment:** Beta found a broader pattern — the registered signals set in `outcome_collector.py` grows unbounded, never prunes resolved predictions, and uses non-atomic `write_text()` for persistence. Beta also identified that if the process crashes mid-write, the entire registered_signals.json is lost, causing all previously-registered signals to be re-registered as fresh predictions with their original timestamps on restart.

**At what step did reasoning diverged?**

I identified one specific data integrity failure. Beta followed the same thread to the structural persistence vulnerability that makes such failures recoverable only by accident. The 2027 timestamp bug I found is a symptom; Beta found the absence of a purge mechanism (PV-3) that would allow it to be cleaned up.

**Beta's structural analysis is more actionable than my specific-bug focus.** Fixing the 2027 timestamp without fixing the registered set growth and non-atomic write means the next anomalous timestamp will cause the same class of problem.

---

## 2. Agreements

Where all three agents independently converged:

**Agreement A: No execution layer — unanimous.**
All three agents identified this as fatal for financial utility. My framing: "no output path." Alpha's: "MIDGE does not trade." Beta's: audit focused on what breaks if you try to build one (execution without atomic persistence). All three noted the `TradeSignal` dataclass in `signal.py` is defined but never instantiated. This is the single finding with perfect triadic consensus.

**Agreement B: Thompson distribution quality — unanimous.**
All three agents read `data/market/thompson_distributions.json` and independently reached the same conclusion: the most data-rich sources (finra_short, yfinance_price) have the most evidence of negative or sub-random predictive value. The sweep_bt keys (CL=F, YM=F:bearish) are the only distributions showing positive edge, and they have thin samples (13-27 observations).

**Agreement C: Congressional trade lag kills the signal.**
My analysis: congressional disclosures wash out of the 72-hour window. Alpha's: the edge exists on day 0 of the trade, not day 30-45 when it becomes public. Beta: the retirement_window seeding behavior may be providing historically-biased data as if it were live. Three different entry points, same conclusion: congressional at 16.4% Thompson mean is not a reliable convergence participant.

**Agreement D: Discovery log contains corrupted data.**
I found the alert storm (CONV-20260227-0001 through -0021) from the deduplication race. Alpha cited it as evidence that "MIDGE thinks it found 20 novel patterns when it found one." Beta found the non-atomic JSONL append pattern (RC-4) and noted the discovery log is one of the files with this vulnerability. The root cause analysis differs but the data quality conclusion is the same.

**Agreement E: Hypothesis promotion rate is effectively zero.**
From `data/midge/convergence_state.json`: 30 generated hypotheses, 0 promoted. My analysis flagged this as the DSR threshold blocking promotions. Alpha noted it as evidence that RSI Layer 2 has "zero observable output in production." Beta noted the hypothesis registry grows unboundedly without compaction while producing no promotions — wasted disk I/O for no learning output. Same observed fact, different concerns about it.

---

## 3. Gaps

### What Alpha found that I missed:

**Alpha's organism-metaphor-vs-market tension analysis** (Section 2 of alpha-findings.md) produced findings I did not look for:

1. **Circadian rhythm runs in step time, not wall-clock time.** I did not examine the CircadianRhythm at all. Alpha found that MIDGE's "morning exploration phase" and "afternoon consolidation phase" bear no relationship to market sessions. This is a concrete misalignment between organism behavior and market timing.

2. **Endocrine pipeline is dead weight for market signals.** The path "convergence alert → dopamine → exploration bias" produces zero financial output. I saw the convergence alert go to pheromone markers (market_actions.py) and called it a terminal gap. Alpha found the intermediate telephone game in the endocrine system that delays and dilutes the signal before it even reaches the terminal gap.

3. **Finra_short as anti-signal requiring direction flip.** As documented in Divergence 2 above, I treated this as weak signal. Alpha's more precise financial analysis identifies it as actively harmful at current sample size and win rate.

4. **Competitive position analysis** (Section 6 of alpha-findings.md): Alpha compared MIDGE against Bloomberg, QuantConnect, and a 200-line Python script. This framing helps prioritize: MIDGE's genuine differentiators are Thompson Sampling (learns reliability), DSR-gated hypothesis lifecycle, and multi-domain convergence. Everything else is either replicable with 200 lines or available free in QuantConnect. I did not perform this competitive analysis.

### What Beta found that I missed:

1. **Thompson JSONL history write without lock (CF-1).** I identified the deduplication race at a different level. Beta found the more fundamental asymmetry: `_save_distributions()` IS locked, `_log_update()` is NOT, and both write to the same logical state. This is the bug that already caused the confirmed production failure (marathon file-lock corruption in MEMORY.md).

2. **Non-atomic write_text() on Windows for critical state files (CF-2).** I did not examine the atomic-write pattern at all. Beta found that `learning_config.py`, `hypothesis_engine.py`, `hypothesis_generator.py`, and `step_timer.py` all use `Path.write_text()` which truncates then writes. `ThompsonSampler._save_distributions()` correctly uses `os.replace(tmp, path)`. The inconsistency means a crash during meta-learner update silently wipes all gate tuning. This is the structural root cause of a class of state corruption I found only as symptoms.

3. **Test data contaminating production pair_outcomes.json (CF-3).** The file at `data/market/pair_outcomes.json` contains "a|b", "a0|b0" through "a6|b6" — test fixture names poisoning the hypothesis generator's priority ordering. I did not examine pair_outcomes.json. Beta found this as the clearest example of a confirmed production contamination.

4. **Registered signals set unbounded growth (PV-3).** The `outcome_collector.py` set grows without pruning. The O(n log n) serialization on every batch registration eventually blocks the main thread. Beta estimated this will grow to hundreds of thousands of entries over months. I identified the 2027 timestamp as a symptom; Beta identified the structural cause.

5. **run_service.bat restart loop on Python environment failures (EG-4).** I did not examine the service wrapper. Beta found that import failures cause an infinite 30-second restart loop that fills logs without recovering.

6. **CorrelationTracker O(n² × m²) complexity (PA-1).** I did not examine correlation_tracker.py. Beta found that the update_correlations() inner loop is O(n² × m²) where n is signal count and m is window size — both growing over time. This is a latency bomb that will materialize once signal diversity increases.

### What I found that neither Alpha nor Beta found:

1. **Static signal confidence disconnect from Thompson reality (my Issue 6, Rank 6).** The adapter functions in `signal_adapters/` set static confidence values (congressional=0.75, insider=0.70). The Thompson distributions show congressional mean=0.164, sec_form4 mean=0.360. The confidence formula takes the static value as input and applies the Thompson weight as a multiplier — meaning congressional enters the geometric mean at 0.75, gets multiplied by 0.664, but 0.75 × 0.664 = 0.498 still contributes a misleadingly high confidence. Neither Alpha nor Beta tracked this specific formula interaction. Alpha noted the Thompson weight mapping is insufficient to neutralize anti-signals, which is adjacent, but neither traced the static-confidence-plus-Thompson-weight double-counting to its formula output.

2. **SEC Form 4 systematic sell-side skew on the watchlist (my Issue 6, Rank 7).** The predictions.jsonl shows every recent sec_form4 prediction has direction "down" except one MSFT entry. This is not a data error — it correctly reflects that tech mega-cap executives in early 2026 were predominantly selling. The watchlist (AAPL, MSFT, GOOGL, AMZN, NVDA, META, LMT, RTX, NOC, GD, BA) skews toward mega-cap tech where insider selling is routine and uninformative. Neither Alpha nor Beta examined the directional bias of the realized prediction pool.

3. **Two predictions.jsonl schema variants coexist.** The file contains old-format records (entry_price=0.0, target_price=0.0, stop_loss=0.0) and new-format OutcomeCollector records. The old-format records make price-based outcome evaluation meaningless but are not flagged as invalid. Beta examined the file for test contamination in pair_outcomes.json; I examined predictions.jsonl schema more closely and found this mixed-format problem.

4. **Source rotation latency for high-urgency signals.** My flow map documented that each of 19 sources is eligible every ~315 steps (19 sources / 3 concurrent slots × 50-step cadence). During an active session sweep window, the sweep source may not be polled for several minutes of wall-clock time. Neither Alpha nor Beta calculated this specific rotation cadence.

---

## 4. Surprises

### Surprise 1: The reward misalignment is more than inefficiency — it is an active training signal working against MIDGE's purpose.

Beta's finding that market agents are capped at 0.5 reward from market actions while TaskPool exploitation returns up to 1.0 surprised me because I had accepted the reward ceiling as a reasonable tuning choice (MEMORY.md notes this was intentional). Reading Beta's analysis made the implication clear: the VDN Q-table is actively learning, over thousands of steps, that abstract task completion is twice as valuable as convergence alert discovery. This is not a design oversight — it is a training signal that will gradually optimize agents away from market intelligence. The "intentional" reasoning in MEMORY.md ("below TaskPool exploit ceiling, above rest") was solving the wrong problem: preventing market actions from dominating the reward space, but without recognizing that the TaskPool ceiling sets the reward reference point that defines what "matters" to the Q-learner.

### Surprise 2: The meta-learner's retirement window seeding mechanism means it cannot distinguish history from live performance.

Beta's CF-4 finding about the retirement_window.json seeding behavior changed my thinking about the meta-learning loop (Bridge 5). I had understood the meta-learner as monitoring live hypothesis performance and adjusting generator thresholds accordingly. Beta found that `_seed_retirement_window_from_registry()` populates the 50-entry window with historical state on cold start. This means the Wire 2 decision — tighten min_correlation if retirement_rate > 70% — is based on a window that may primarily contain historical retirements from previous sessions, not current ones. The meta-learner cannot tell if it is looking at 15 recent retirements (live session, maybe tighten) or 15 historical retirements seeded at startup from older registry state (artifact, should not tighten).

This matters because if the meta-learner tightens min_correlation based on seeded historical data, it suppresses hypothesis generation in a fresh session that might actually be performing well. The RSI Layer 3 self-improvement loop has a cold-start bias that I had not considered.

### Surprise 3: Alpha's competitive analysis reveals MIDGE's genuine differentiators are its weakest components.

Alpha compared MIDGE against a 200-line Python script and concluded the script "would produce the same trade signal quality as MIDGE's convergence alerter for the top-performing signal combination." The parts that MIDGE does that the script cannot — Thompson Sampling (learns reliability), DSR-gated hypothesis lifecycle, RSI Layer 2→3 — are the parts that are currently producing zero output (0 promoted hypotheses, no applied Kelly recommendations). The organism infrastructure (125 systems, 385 connections, 144 holons) protects and houses these differentiators, but the differentiators themselves are not yet functional at the level MIDGE was designed for.

This reframed my understanding of the priority ordering. I had been treating "build the output path" as the first step and "make learning work" as the second. Alpha's competitive analysis suggests they are the same step: until Thompson-weighted convergence actually produces and executes a trade, the learning differentiator cannot be evaluated. The 200-line script is MIDGE's competition until MIDGE executes a trade.

### Surprise 4: The pair_outcomes.json test contamination is confirmed present in production.

Beta's CF-3 confirmed what I suspected from examining outcomes.jsonl test contamination but had not traced to pair_outcomes.json specifically. The fact that "a|b", "a0|b0"..."a6|b6" are sitting in the production file and actively influencing hypothesis generator priority ordering — right now, in the current running system — means RSI Layer 2 is operating with corrupted inputs in production. Not a theoretical risk. Confirmed present.

---

## 5. Synthesis — Top 5 Most Impactful Changes for Financial Effectiveness

These are ranked by expected financial impact, incorporating all three perspectives:

### Priority 1: Build the Paper Trading Output Layer

**Evidence:** All three agents agree. TradeSignal is defined, never instantiated (signal.py lines 82-94). ConvergenceAlert produces structured output that gets stored to an advisory dict and deposited as pheromone markers — neither leaves the process boundary.

**What to build:**
- Convert ConvergenceAlert → TradeSignal when confidence > threshold, using the Kelly fraction already stored in `ctx._latest_kelly`
- Write to `data/midge/paper_trades.jsonl` with timestamp, ticker, direction, entry_price, kelly_fraction, confidence, contributing_domains
- Wire OutcomeCollector to read paper_trades.jsonl exit prices from yfinance and compute dollar P&L (not just directional binary)
- This closes the one loop none of the three layers (convergence, Thompson, hypothesis) can currently close: did the recommendation make money?

**Financial impact:** Without this, MIDGE cannot validate any of its learning. The Thompson distributions, the DSR-gated hypothesis lifecycle, the Kelly sizer — all exist to serve a trading outcome that never materializes. This is table stakes.

---

### Priority 2: Fix the Thompson Thread-Safety and Atomic Persistence Layer

**Evidence:** Beta CF-1 (already caused one confirmed production failure), CF-2 (non-atomic writes on Windows), RC-3 (distributions dict mutated without lock). My Issue 5 (mock outcomes in outcomes.jsonl) is the data-quality companion.

**What to fix:**
- Wrap `_log_update()` in `thompson_sampler.py` with `self._lock` (same lock that protects `_save_distributions()`)
- Wrap the `self.distributions[signal_id][regime]` dict mutation in RC-3 with `self._lock` before the file write
- Replace `write_text()` calls in `learning_config.py`, `hypothesis_engine.py`, `hypothesis_generator.py`, and `step_timer.py` with the atomic `os.replace(tmp, path)` pattern already used in `_save_distributions()`
- Purge mock outcomes from `outcomes.jsonl` (entries with sub-day timeframes and prices like 278.12 = 49.93% 1-day return for AAPL)
- Fix the 2027 timestamp prediction in registered_signals.json

**Financial impact:** The Thompson distributions are the central nervous system of MIDGE's signal quality scoring. Corrupting them or silently losing meta-learned gate tuning degrades every convergence confidence score and every hypothesis promotion decision. One more marathon file-lock corruption event wipes months of learning.

---

### Priority 3: Flip or Exclude finra_short as a Convergence Participant

**Evidence:** Alpha's anti-signal analysis, confirmed by Thompson data: alpha=452.65, beta=812.67, mean=0.358, 1,263 samples. At a Thompson weight of 0.858, finra_short is actively injecting directional anti-signal into the convergence geometric mean 64.2% of the time it fires. This is the most data-rich distribution in the system, and it is pointing the wrong direction.

**What to fix:**
- Option A (preferred): In `convergence_alerter.py:_compute_confidence()`, if `dist.mean < 0.45` AND `dist.samples >= 100`, flip the signal direction before computing the weighted contribution. High short interest → contrarian long is academically supported and financially intuitive.
- Option B: Add finra_short to an `observer_only_sources` list that updates Thompson but does not participate in domain convergence voting until mean > 0.50 with meaningful samples.

**Financial impact:** Every convergence alert that included a finra_short directional vote was slightly biased toward the wrong conclusion. With 1,263 samples, this is a well-characterized problem. Flipping or excluding it immediately improves the confidence calibration of every multi-domain alert.

---

### Priority 4: Clean the Bayesian Learning Inputs — Purge Test Contamination and Fix Slow-Signal Domain Persistence

**Evidence:** Beta CF-3 (pair_outcomes.json test data confirmed present), my Issue 2 (deduplication failure — 20+ identical alerts in discovery_log), my Issue 4 (Thompson prior mismatch — learning_config `congressional=0.75` vs actual Thompson mean=0.164), Beta PV-3 (registered signals set unbounded growth).

**What to fix, in order:**
- Delete "a|b", "a0|b0"..."a6|b6" from `data/market/pair_outcomes.json` and run the test suite against a temporary directory to prevent future test contamination of production data files
- Add a threading.Lock() around the dedup check-and-update in `convergence_alerter.py:check_convergence()` using `time.monotonic()` instead of `datetime.now()`
- Update `learning_config.py:source_reliability` defaults to match actual Thompson distribution means: `congressional` should be `0.20` not `0.75`; `sec_edgar` should be deprecated in favor of `sec_form4`
- Add a per-domain convergence window map so COT (weekly data) and congressional (45-day disclosure lag) get 336-hour and 168-hour windows respectively instead of the global 72-hour window that washes them out

**Financial impact:** These four items all affect what enters the Thompson → convergence → hypothesis pipeline. Corrupted pair outcomes distort hypothesis generation priority. Deduplication failures corrupt the RSI Layer 2 training signal. Config priors 4x above reality inflate confidence in the early steps of each session before distributions load. Slow signals that wash out of a 72-hour window can never contribute to convergence even when they carry genuine information.

---

### Priority 5: Enforce Genuine Domain Independence in the Convergence Alerter

**Evidence:** Alpha's analysis (Section 3, "The min_domains=3 Problem"): social_sentiment, google_trends, and stocktwits_sentiment all reside in the sentiment domain and measure the same underlying retail attention signal through three different lenses. Three sentiment sources satisfying min_domains=3 is not triadic witnessing — it is triadic echo.

**What to build:**
- Add an `independent_domains` constraint to `ConvergenceAlerter`: the 3 required domains must span at least 2 of the following super-categories: `{fundamental: [insider, government, regulatory], technical: [technical, volatility], sentiment: [sentiment, social], macro: [positioning, macro]}`
- This ensures convergence requires cross-category confirmation, not just three signals from different source names that all happen to track retail mood
- For session_sweep_ifvg signals specifically: implement Lead Option B (parallel direct-output path for BACKTEST_DERIVED hypotheses with DSR > threshold), bypassing convergence for the one pattern with verified historical edge

**Financial impact:** This is the fix that makes the 3-domain requirement financially meaningful rather than just formally satisfied. An alert that requires fundamental + technical + sentiment or positioning + fundamental + technical is genuinely harder to manufacture from noise than three sentiment proxies coinciding. This will reduce alert frequency but increase alert precision — the direction every signal-quality metric in the Thompson data is pointing toward.

---

## Summary Table

| Priority | Change | Primary Evidence | Financial Mechanism |
|----------|--------|-----------------|---------------------|
| 1 | Paper trading output layer | All three agents: TradeSignal never instantiated | Closes feedback loop, enables P&L measurement |
| 2 | Thompson thread safety + atomic writes | Beta CF-1 (confirmed production failure), CF-2 | Prevents silent corruption of signal reliability scores |
| 3 | Flip or exclude finra_short | Alpha: anti-signal at 35.8%, 1,263 samples | Removes directional anti-signal from geometric mean |
| 4 | Purge test contamination + fix slow-signal persistence | Beta CF-3 confirmed, Lead Issue 2, 4 | Cleans all three learning inputs (hypothesis, convergence, Thompson) |
| 5 | Enforce genuine domain independence | Alpha Section 3 | Makes min_domains=3 a real filter, not a threshold easily satisfied by correlated noise |
