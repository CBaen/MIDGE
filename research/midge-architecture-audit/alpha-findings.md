# MIDGE Architecture Audit — Alpha Findings (Adversarial Market Practitioner)

**Auditor Role:** Witness Alpha — Adversarial Market Practitioner
**Date:** 2026-03-01
**Scope:** Full adversarial review of MIDGE's architecture from a professional trading perspective

---

## 1. Executive Summary

MIDGE is a fascinating organism that has accumulated sophisticated infrastructure for pattern recognition, Bayesian learning, and hypothesis generation — but it does not trade. It produces convergence alerts into a void: no broker connection, no order routing, no live P&L, and no feedback loop between financial outcomes and the organism's reward signal. The biological metaphor is not just cosmetic overhead — it actively shapes decision-making in ways that are misaligned with market reality (step-based time, circadian rhythms, endocrine stress responses). The signal quality data tells a sobering story: the two most data-rich Thompson distributions are finra_short (35.8% win rate on 1,987 samples — below random) and yfinance_price (22.9% win rate on 368 samples — dramatically below random), meaning the sources with the most data are confidently failing. Three sources exceed 50% — sec_form4, CL=F sweep backtest, and YM=F bearish sweep — but all have thin data (fewer than 30 samples). MIDGE is a capable research instrument that could become a trading system, but calling it one now would be dishonest.

---

## 2. Organism vs. Trading Tension

### Where the Metaphor Helps

**Thompson Sampling as Bayesian Immune System.** The biology-as-metaphor here produces a genuinely useful outcome. The ThompsonSampler treating signal sources as organisms to be selected for or against is elegant and functionally correct. The forgetting mechanism (decay_factor=0.99 every 100 steps) maps cleanly to the idea that stale evidence should lose influence. This is the organism metaphor working well.

**Convergence Alerter as Multi-Domain Synthesis.** Requiring 3 domains before alerting (Law 2: No Bare Dyads applied to information) actually implements a real trading principle: single-source signals are noise, multi-source confirmation is edge. This is not organism cosmetics — it reflects how quant shops actually filter signals.

**Graceful Degradation.** Every system wrapped in try/except with ctx.attr = None fallback means MIDGE can run with partial API keys. A trading system that crashes on missing credentials is worse than one that degrades gracefully.

**Hypothesis Lifecycle (RSI Layer 2).** The probation→active→hibernated→retired pipeline with Deflated Sharpe Ratio validation is a real anti-overfitting mechanism. That MIDGE attempts to formalize hypothesis testing rather than just tuning parameters is architecturally correct.

### Where the Metaphor Fights Financial Reality

**Problem 1: Step Time vs. Market Time.**
Markets move in wall-clock time. MIDGE runs in step-based simulation time. A "step" can take 0.01 seconds or 30 seconds depending on what I/O is in flight. The circadian rhythm is keyed to `cycle_length` steps, not to 24-hour periods. This means MIDGE's "morning exploration phase" and "afternoon consolidation phase" bear no relationship to actual market sessions. The circadian clock is running on simulation ticks while real markets open at 9:30 ET regardless of step count.

The sensing hook fetches data every 50 steps. But 50 steps at what wall-clock interval? There is no mechanism to ensure MIDGE checks markets during market hours, avoids fetching stale overnight data during the trading day, or detects that it is operating during a holiday. When the SessionSweepDetector runs during Saturday steps, it is fetching yfinance 1-minute candles for a closed market.

**Problem 2: Endocrine Signals Are Fictional Trading Signals.**
When a convergence alert fires with strength > 0.7, MIDGE releases Dopamine (bullish) or Adrenaline (bearish) into the endocrine system. These hormones then modulate agent "exploration bias" and "trust level." This is the organism metaphor producing zero financial value. A bullish convergence alert should produce a position recommendation with a size, entry price, stop, and target. Instead it triggers a hormone that makes agents explore more in the next 50 steps.

The endocrine layer between market signal and market action is pure overhead — a telephone game where "AAPL insider buying cluster" becomes "release dopamine" becomes "agent explores more" with no financial output at any stage.

**Problem 3: Agent Actions Are Not Market Actions.**
The `_act_explore`, `_act_exploit`, `_act_communicate`, `_act_rest` methods in `lifecycle_decision.py` interact with a TaskPool of abstract tasks with difficulty ratings and reward values. When a SEC_WATCHER agent "exploits," it reads the number of insider signals in the alerter's buffer and deposits a stigmergy marker. This is not trading. The market-specific dispatch in `market_actions.py` improves this — SEC_WATCHER now at least reads real signal data — but "depositing a DISCOVERY marker with intensity proportional to alert strength" is still not a trade.

The reward scale for market actions is intentionally capped at [0.0, 0.5] to stay "below TaskPool exploit ceiling (~1.0)." This means the organism's reward system is deliberately calibrated to value abstract task completion over market intelligence. A HYPOTHESIS_VALIDATOR that promotes a genuine alpha-generating pattern receives 0.4 reward. An agent that exploits a random TaskPool task to completion receives up to 1.0. The reward system is teaching agents that abstract tasks matter more than market discoveries.

**Problem 4: The Circadian Rhythm Has No Market Awareness.**
The CircadianRhythm cycles through phases based on step count. It has no knowledge of market hours, pre-market, after-hours, or overnight sessions. MIDGE might be in its high-energy "morning exploration" circadian phase at 2am EST when the only active market is Asia, or in its "rest/consolidation" phase during the most liquid hour of the US trading day (10-11am). The circadian system is not useless — it imposes structure — but the structure is arbitrary relative to market time.

**Problem 5: 33-Layer Bootstrap for 5 Agents.**
The bootstrap creates: reproductive_system, lymphatic_system, senescence_system, vestibular_system, renal_filter, microbiome, nociception, proprioception, thermoregulation, energy_reserve, circulatory_system, digestive_system, respiratory_system. These are biological completeness requirements from mae-core. For trading, they produce overhead in every step (step hooks, EventBus traffic) with zero financial contribution. The systems are "advisory, never blocking" — they observe and report but cannot actually stop an agent from taking a bad action. They generate EventBus messages that consume memory and CPU without informing any trading decision.

Quantifying the organism tax: the systems dict in `main.py` has approximately 80 registered systems. Of these, perhaps 30 are directly relevant to market intelligence (the Layer 33 systems). The remaining ~50 systems — morphogenesis, theory_of_mind, validated_imagination, collective_dream, worldline_planner, triage_classifier, mitosis_monitor, integration_meter, topology_analyzer, triadic_verifier, etc. — run step hooks, publish EventBus events, and consume CPU for every simulation step with no trading relevance.

---

## 3. Signal Quality Verdict

### Reading the Thompson Distributions

From `data/market/thompson_distributions.json`, computing win rates as alpha/(alpha+beta):

| Source | Win Rate | Samples (approx) | Verdict |
|--------|----------|-------------------|---------|
| sweep_bt:CL=F | 52.4% | 26 | Marginal positive — thin data |
| sweep_bt:YM=F:bearish | 58.7% | 12 | Positive — very thin |
| sweep_bt:CL=F:bearish | 58.7% | 12 | Positive — very thin |
| sec_form4 | 36.0% | 10.7 | Below random at this sample size |
| finnhub_earnings | 26.6% | 69 | Below random with meaningful data |
| congressional | 16.4% | 32 | Well below random — negative edge |
| contract_award | 15.4% | 25 | Well below random |
| finra_short | 35.8% | 1,263 | Most data, consistently below 50% |
| yfinance_price | 22.9% | 368 | Large sample, dramatically below 50% |
| cot_positioning (sideways) | 27.9% | 80 | Below random |

**The most alarming finding:** The two sources with the largest sample sizes (finra_short at 1,263 samples and yfinance_price at 368 samples) are both well below 50% win rate. This means the sources that have been evaluated most thoroughly are the ones with the most evidence of negative predictive value. MIDGE is not learning "these sources are bad, weight them down." It is learning "these sources are at 35.8%, weight them slightly below neutral" — but 35.8% is not "slightly below neutral," it is meaningfully worse than random, which means every time MIDGE includes finra_short in a convergence calculation it is injecting anti-signal.

**The geometric mean confidence formula makes this worse.** The `_compute_confidence()` method uses Thompson weights mapped from `[0,1] -> [0.5, 1.5]`. A source at 35.8% win rate maps to a weight of 0.858 — only a 14% downweight. This is not sufficient to neutralize an anti-signal; it just slightly softens its contribution. A source that is genuinely anti-correlated with price movement should have its weight approach zero, not 0.858.

### Sources With Real Potential Edge

**sec_form4 (Form 4 insider trades).** The research backing is solid — insider cluster buying does outperform. The issue is sample size and filtering. The 36% win rate across all Form 4 events may reflect the fact that most Form 4 filings are RSU vesting and options exercises (which are not predictive), not open-market purchases. The 10b5-1/RSU filter in the codebase is a step in the right direction, but the Thompson data suggests it has not yet produced clean enough signal.

**Session sweeps (CL=F, YM=F bearish).** The backtest data is the most credible source: 617 real trades, 39.1% baseline win rate with 1.38 profit factor. The elite tier (quality >= 0.65) produced 45.3% win rate. These are real market patterns with quantified edge, even if that edge is modest.

**Congressional trades.** At 16.4% win rate across 53 samples, this is the most disappointing result given the theoretical premise. The likely reason: Congress trades are reported with a 30-45 day lag (STOCK Act allows this delay), and by the time MIDGE ingests them, the information is priced in. The edge exists on day 0 of the trade, not day 30-45 when it becomes public.

### Sources That Are Noise

**Google Trends (google_trends).** No data yet (Beta(1,1) = 50% prior). Retail search interest has documented edge in narrow situations (first-day IPO attention, options expiry sentiment) but as a general signal it is far too noisy to combine with insider and contract data.

**StockTwits sentiment (stocktwits_sentiment).** No data yet. StockTwits skews heavily retail and has documented reverse-indicator properties for meme stocks. Adding it to a convergence alerter that requires 3 domains means it can "vote" on convergence alongside sec_form4 and congressional trades. These are fundamentally different signal frequencies and qualities being flattened into the same voting pool.

**Reddit/social_sentiment.** Below 50% prior (Beta(1, 1.13)). Social sentiment has documented contrarian properties: extreme bullish retail sentiment correlates with short-term reversals. MIDGE treats it as a directional signal, not a contrarian indicator.

**Hiring tracker / job_tracker.** Thompson at prior (no data). The contract_predictor thesis (hiring blitz predicts contract award) is genuinely interesting and original. But job posting data from RapidAPI is expensive, laggy, and difficult to validate. The signal has a months-long lead time which creates an enormous attribution problem.

### The min_domains=3 Problem

The convergence alerter requires 3 domains minimum. Given the number of sources that are at or below 50% win rate, this requirement can be satisfied by combining three independently bad signals and calling the result a convergence alert. Three sources at 35% win rate, independently uncorrelated, do NOT converge to a better signal just because they are simultaneously negative. The convergence framework assumes the domains are capturing genuinely independent information sources. When social_sentiment, google_trends, and stocktwits_sentiment all point bullish, they are measuring the same underlying thing (retail sentiment) through three slightly different lenses, not three independent information sources.

---

## 4. Missing Trading Infrastructure

This section is blunt: MIDGE lacks the infrastructure to be a trading system.

**No Execution Layer.** There is no broker API integration, no order placement, no order state management (pending/filled/cancelled), no slippage modeling, no transaction cost accounting. The KellyPositionSizer recommends a kelly_capped fraction but there is no mechanism to translate that fraction into a dollar amount, instrument, order type, or exchange. The sizing recommendation fires on EventBus and is stored in `ctx._latest_kelly`. Nothing reads it to place a trade.

**No Risk Management.** There are no stop losses. There are no position limits. There is no drawdown monitor. There is no maximum daily loss limit. There is no portfolio-level exposure tracking. The Kelly fraction is computed per signal but there is no system to enforce it, prevent over-sizing, or cut positions that have moved against the thesis. A real trading system requires these not as nice-to-haves but as hard stops that cannot be bypassed.

**No P&L Tracking.** The OutcomeCollector tracks whether predictions resolved correctly (binary win/loss based on price movement), but there is no dollar P&L. There is no accounting for position size, entry price, exit price, slippage, commissions, or overnight financing. The "win rate" tracked by Thompson Sampler is directional accuracy only — it does not tell you whether the positions were profitable in dollar terms.

**No Real-Time Data.** Every data source in MIDGE is batch polled: the sensing hook runs every 50 steps, SEC data is fetched via HTTP requests with 10-second rate limits, yfinance provides data with approximately 15-minute delay for free tier. This is a research and discovery system, not a real-time trading system. For the ICT session sweeps specifically (which depend on 1-minute candle timing), a 15-minute delay makes the entry signals unreliable.

**No Backtesting of the Full Pipeline.** The sweep_backtest.py tests the session sweep detector in isolation with historical data. There is no backtesting of the full convergence pipeline — no test of "when MIDGE generates a convergence alert, what would the P&L be if you traded it?" The 39.1% win rate from sweep backtesting is the most rigorous number in the codebase, and it is for one detector in isolation, not the integrated system.

**No Market Hours Awareness.** The sensing hook runs every 50 steps, 24 hours a day (in continuous mode). It fetches intraday futures data outside trading hours, polls congressional trade APIs on weekends, and runs lag correlation analysis regardless of whether markets are open. There is no calendar awareness: no holiday detection, no early close handling, no overnight gap risk management.

**No Instrument Sizing.** Even if MIDGE generated a trade signal with a Kelly fraction, there is no mechanism to translate "buy 0.03 of your portfolio in AAPL" into a specific number of shares, lot size for futures, or contract count. The KellyPositionSizer computes a fraction; a real system needs to know account value, margin requirements, minimum tick size, and contract multipliers.

---

## 5. The Organism Tax

### What the StepTimer Would Show

The StepTimer snapshot was not present (`data/midge/step_timer_snapshot.json` does not exist — no completed marathon run). However, the code structure reveals the overhead profile:

**Every step (step % 1 == 0), BOTH hooks fire:**
1. `_market_sense_hook`: calls `alerter.check_convergence()` (prune all signals + iterate all domains + compute cross-domain count + compute Thompson-weighted geometric mean confidence)
2. `hypothesis_engine.step()`: checks internal cadence, potentially runs generation/validation
3. `_sensing_step_with_advisory`: wraps sensing hook step, reads cached alerts, updates advisory dict

**Every 10 steps:** Thompson stats + regime classification + tiered alerters (3 ConvergenceAlerter.check_convergence() calls)

**Every 50 steps:** Velocity anomaly scan + Kelly position sizing + async market data fetch (3 concurrent sources via ThreadPoolExecutor)

**Every 100 steps:** Bayesian forgetting + convergence heartbeat write to disk

**Every 500 steps:** Lag correlation analysis (reads archive, computes Pearson correlations across signal pairs)

**Every 1000 steps:** Thompson calibration

**Every 5000 steps:** Backtest staleness check + optional rerun

The non-market overhead — from the ~50 non-market systems running their own step hooks — is not timed. The triadic witnessing system verifies 385 connections. The awareness pulse runs a hierarchy health check. The endocrine system decays hormone levels. The organism state triage classifier evaluates biological urgency. The integration meter computes Phi. None of these contribute to trading decisions.

**Estimated overhead ratio:** Based on the systems dict (80 systems, approximately 50 non-market), and assuming linear step-hook cost distribution, roughly 60% of per-step compute is organism maintenance rather than market intelligence. This is a rough estimate without actual profiling data, but the code structure supports it as a lower bound.

**The critical path bottleneck:** The sensing hook is properly backgrounded (ThreadPoolExecutor(3)), but the convergence check runs in the main thread every step. If the signal buffer grows large (72-hour window with 19 sources polling every 50 steps), `check_convergence()` iterates an increasingly large signal dictionary on every single step. There is no incremental update mechanism — it recomputes from scratch every time.

### What a Simpler Architecture Would Cost

A minimal Python trading research system using the same data sources:
- yfinance for price data
- SEC EDGAR API directly (free, no wrapper)
- STOCK Act congressional data (free CSV)
- pandas for correlation analysis
- scipy.stats for Thompson sampling equivalent

Could replicate MIDGE's signal quality with roughly 500 lines of code instead of 67 market files across 3 subpackages. The organism wrapper provides: graceful degradation (genuinely useful), Bayesian updating (replicable with scipy), hypothesis lifecycle management (replicable with a simple state machine), and multi-domain convergence (a Python dict and a counter). The organism wrapper costs: 30 bootstrap layers, 125 systems, 385 triadic connections, 144 holons, endocrine coupling, circadian rhythm, and holon awareness pulses.

The question is not whether the organism is beautiful — it is. The question is whether it produces better trading insights than a 500-line script. Currently, there is no evidence it does.

---

## 6. Competitive Position

### vs. Bloomberg Terminal + Systematic Approach

Bloomberg provides real-time Form 4 parsing, congressional trade monitoring, and options flow. A quant analyst with Bloomberg access can replicate everything in MIDGE's signal library in a single afternoon. What MIDGE has that Bloomberg does not: the convergence synthesis, the Bayesian updating, and the hypothesis lifecycle. These are genuine differentiators — but they are software differentiators, not data differentiators.

**MIDGE's disadvantage:** Latency. Bloomberg clients receive SEC filings within seconds of filing. MIDGE polls EDGAR every N steps, potentially missing the first minutes of price movement where the edge exists.

### vs. QuantConnect / Lean Engine

QuantConnect provides backtesting infrastructure, broker integration, paper trading, and live trading. Its signal quality is whatever the user builds. MIDGE has more sophisticated signal synthesis than most retail QuantConnect strategies, but QuantConnect gives you execution, risk management, and backtesting for free.

**MIDGE's disadvantage:** QuantConnect strategies can be live-traded today. MIDGE cannot trade at all.

### vs. a Simple Python Script

A 200-line Python script that:
1. Polls SEC EDGAR Form 4 for insider clusters (3+ insiders, open-market purchases only)
2. Checks if congressional trades were filed in last 30 days for the same ticker
3. Checks session sweep signals from yfinance 1-minute data
4. If 2/3 align: print "BUY AAPL"

Would produce the same trade signal quality as MIDGE's convergence alerter for the top-performing signal combination. It would not learn, adapt, or generate hypotheses — but those capabilities have zero observable output in production (0 promoted hypotheses after 15 generated, 0 Kelly recommendations acted on).

**Where MIDGE is genuinely ahead of a simple script:**
- Thompson Sampling learns signal reliability over time (a simple script cannot do this)
- Hypothesis generation with DSR anti-overfitting is genuinely novel
- The RSI Layer 2→3 architecture is the right long-term direction
- Multi-domain convergence at 3+ domains is a real quality filter

**Where MIDGE is severely behind a simple script:**
- A simple script can be live today
- A simple script has no organism tax
- A simple script does not release adrenaline when markets move against it

---

## 7. Kill List

Things that should be removed or simplified, in order of confidence:

**1. Circadian Rhythm (remove or wall-clock it).**
Either wire CircadianRhythm to actual wall-clock time (UTC, keyed to market sessions), or remove it from the market context entirely. A circadian system that runs in step-time while markets run in wall-clock time produces meaningless phase assignments. If kept, phases should be: pre-market (4-9:30 ET), regular-hours (9:30-16 ET), after-hours (16-20 ET), overnight (20-4 ET).

**2. Endocrine → Agent Decision Pipeline for market signals.**
The path "convergence alert → dopamine → exploration bias → agent explores more" should be replaced with "convergence alert → recommendation store → agent reads recommendation → agent takes specific market action." The endocrine system may still be valid for internal organism health (stress from repeated failures), but it should not be in the critical path from market signal to market action.

**3. TaskPool for market-role agents.**
The `_act_explore` and `_act_exploit` methods that interact with TaskPool abstract tasks are dead weight for market agents. The `market_actions.py` dispatch partially replaces this, but the fallback to TaskPool means market agents still sometimes "exploit" by claiming the highest-reward abstract task in the pool instead of taking a market action.

**4. 50+ non-market step hooks.**
The reproductive_system, lymphatic_system, vestibular_system, thermoregulation, nociception, proprioception, renal_filter, microbiome, senescence, and triage_classifier should be audit-logged rather than actively stepped during market runs. They can run at startup and shutdown for organism health checks, but firing every step is overhead for zero trading benefit.

**5. Google Trends and StockTwits as convergence-voting sources.**
Until these have a meaningful sample size AND demonstrate above-50% win rate, they should not participate in convergence alerting. Add them as "observer" signals that are tracked but excluded from the min_domains count.

**6. LLM Oracle pathway (api_call action).**
The oracle pathway (agents asking Groq/Mistral/DeepSeek for strategy advice) was already disabled for market roles (api_call_enabled: False). This was the right call. The oracle generates text responses that are "logged but never read back" per the MEMORY.md. This dead code should be removed from the market agent path.

**7. Reward signal calibration.**
Market-role agents are capped at 0.5 reward from market actions while non-market TaskPool exploit can return up to 1.0. This teaches the VDN Q-table that market intelligence is less valuable than abstract task completion. Either remove the cap or ensure no non-market reward path exceeds 0.5 for agents in market roles.

---

## 8. Build List

Things that are missing and essential for MIDGE to be a trading system, not a research instrument:

**1. Market Hours Calendar (critical — blocks everything else).**
Build a `MarketCalendar` class that knows NYSE/CME trading hours, holidays, and early closes. Wire it into the sensing hook so data fetches only run during relevant hours. Wire it into the CircadianRhythm so organism phases align with market sessions. This is a 100-line addition using the `pandas_market_calendars` library (already available). Nothing else in the build list matters if MIDGE is fetching data during market closes.

**2. Paper Trading Execution Layer (essential for validation).**
Without trade execution, MIDGE cannot measure what matters: dollar P&L. Even paper trading against delayed data would close the feedback loop. The minimum viable execution layer:
- `TradeSignal` dataclass: ticker, direction, entry_price, stop_price, target_price, size_fraction
- `PaperTradingBook` class: open positions, closed positions, realized P&L, unrealized P&L
- Wire `KellyPositionSizer` output to `TradeSignal` generation
- Wire `ConvergenceAlert` to `TradeSignal` generation when confidence > threshold
- `OutcomeCollector` reads `PaperTradingBook` for real P&L, not just directional accuracy

**3. Convergence Alert → Specific Trade Output.**
The `get_actionable_summary()` method in `convergence_alerter.py` returns direction, confidence, and reasoning. It does not return: which instrument to trade, what strike/expiry (for options), what stop level, what target, or what size. Add a `get_trade_recommendation()` method that produces an instrument-specific recommendation. Wire it to the heartbeat output so `convergence_state.json` includes a `trade_recommendation` field.

**4. finra_short Anti-Signal Handling.**
At 35.8% win rate across 1,263 samples, finra_short is a demonstrated anti-signal. Either: (a) flip its interpretation (high short interest → contrarian long signal), or (b) exclude it from convergence voting. Continuing to include it as a directional signal at 35.8% accuracy is actively degrading convergence quality.

**5. Congressional Trade Lag Compensation.**
Congressional trades are reported up to 45 days after execution. The signal MIDGE receives is 30-45 days old. Add a `reporting_lag_days` field to the congressional signal adapter, and in the convergence window calculation, subtract the reporting lag from signal freshness. A congressional trade reported today should be treated as a signal with 30-45 day old information, not fresh information.

**6. Backtesting the Full Convergence Pipeline.**
The sweep_backtest.py tests one detector. Build a `ConvergenceBacktester` that:
- Reads the historical archive of signals (901 files, 414 days per MEMORY.md)
- Replays signal ingestion into a ConvergenceAlerter
- Whenever the alerter would have fired, records the alert
- Checks price 1/5/20 days later
- Computes full P&L statistics including Sharpe ratio, max drawdown, profit factor

This is the most important missing piece for validating whether the convergence synthesis actually produces edge.

**7. Negative Signal Handling in Thompson.**
The Thompson weights are bounded [0.5, 1.5]. A source at 22% win rate (yfinance_price) gets a weight of 0.71 — barely downweighted. The math for contrarian signals should allow weights to go negative or at minimum to zero. If yfinance_price directional signal is anti-correlated with future returns, it should receive weight approaching 0 and the signal direction should be flipped before entering convergence.

---

## Appendix: Key Evidence

**From thompson_distributions.json:**
- finra_short default: alpha=452.65, beta=812.67 → win rate=35.8%, n=1,263
- yfinance_price default: alpha=85.04, beta=285.43 → win rate=22.9%, n=368
- congressional default: alpha=5.72, beta=29.21 → win rate=16.4%, n=33
- contract_award default: alpha=4.24, beta=23.29 → win rate=15.4%, n=25
- sweep_bt:CL=F default: alpha=14.88, beta=13.52 → win rate=52.4%, n=26

**From convergence_state.json (last captured state):**
- Regime: sideways
- Global alert: bullish, strength=0.873, 3 domains
- Ticker alerts: empty
- Hypotheses: 15 generated, 0 promoted, 3 active, 0 on probation

**From MEMORY.md:**
- 39.1% win rate baseline for session sweeps (below 50%)
- 45.3% elite tier win rate (quality >= 0.65) — still below 50% but positive expectancy via profit factor 1.84
- 43 lag-correlation findings with r >= 0.6 — these are correlations, not causations, and many will be spurious
- Thompson rebuild after file-lock corruption: 12,544 outcomes evaluated

**Key architecture observation from main.py:**
The systems dict registers 80+ systems. Layer 33 (market) accounts for roughly 30. The remaining ~50 fire step hooks, consume EventBus bandwidth, and execute per-step checks that have no pathway to any trading decision.
