# Team 5 Findings: Win Rate Optimization
## Date: 2026-03-05
## Researcher: Team Member 5

---

## Executive Context

MIDGE currently has a 19.9% win rate (31/156 graded predictions, z=4.74, p<0.0001 vs 9% random baseline). The critical question: is 19.9% already profitable, and if not, what gets us there? This research covers the math of profitability at low win rates, regime filtering, combo-specific position sizing, confidence calibration, and risk-of-ruin analysis.

**The punchline up front:** 19.9% win rate CAN be profitable — if and only if the average winner is at least 4x the average loser. The combo data (66.7% on best combos vs 8.3% on worst) tells a different story: MIDGE should be filtering by combo quality, not chasing a higher overall win rate. The path to profit is selective deployment, not global accuracy improvement.

---

## Battle-Tested Approaches

### 1. The Break-Even Win Rate Framework

**What:** The minimum win rate for profitability is determined entirely by the reward-to-risk (R:R) ratio using the formula: Break-Even Win Rate = 1 / (1 + R).

**Evidence:** This is a mathematical identity, not an empirical finding. Confirmed across multiple practitioner sources.
- At 1:1 R:R — need 50% win rate to break even
- At 2:1 R:R — need 33.3% to break even
- At 3:1 R:R — need 25% to break even
- At 4:1 R:R — need 20% to break even
- At 5:1 R:R — need 16.7% to break even
- At 9:1 R:R — need 10% to break even

**Source:** [QuantifiedStrategies.com Win Rate](https://www.quantifiedstrategies.com/win-rate-trading/) (accessed 2026-03-05); [P&L Ledger Break-Even Table](https://www.pnlledger.com/break-even-win-rate-by-risk-reward-table/) (accessed 2026-03-05); [Kelly Criterion Wikipedia](https://en.wikipedia.org/wiki/Kelly_criterion) (accessed 2026-03-05)

**Fits our case because:** MIDGE is at 19.9% win rate. If average winners are at least 4x average losers (4:1 R:R), we are already at break-even. The practical question is whether MIDGE's 5% success threshold (already implemented in outcome_collector.py) corresponds to a payoff ratio that clears this bar. The outcome_collector.py already uses 5% minimum move — the question is what MIDGE's actual average win magnitude is versus average loss magnitude. **This needs to be measured in the existing data before anything else.**

**Tradeoffs:** Transaction costs push the real break-even higher. At 4:1 theoretical R:R with 0.1% round-trip costs, you actually need ~21% win rate to net positive. MIDGE trades infrequently enough that costs matter less than for high-frequency systems.

**Source:** [Trading Expectancy Calculation](https://www.heygotrade.com/en/blog/win-rate-in-trading) (accessed 2026-03-05)

---

### 2. Fractional Kelly Position Sizing for Low Win Rate Systems

**What:** Use Kelly Criterion to size positions, but apply a 0.3x–0.5x fraction to manage estimation error and drawdown risk.

**Evidence:** Kelly formula: f* = (bp - q) / b where b = payoff ratio, p = win probability, q = (1-p). At 19.9% win rate with 4:1 payoff: f* = (4 × 0.199 - 0.801) / 4 = (0.796 - 0.801) / 4 ≈ near zero. **This is the core problem: at exactly the break-even point, Kelly says bet nothing.** Full Kelly requires edge above break-even. At 25% win rate with 4:1 payoff: f* = (4 × 0.25 - 0.75) / 4 = 0.25/4 = 6.25% of capital per trade. Half-Kelly would be 3.125%.

Research shows 0.3x Kelly returns 51% of optimal profit with only 1/11th of the variance. Professional money managers use 0.10x–0.15x Kelly.

**Source:** [Kelly Criterion - Quantitative Trading (nickyoder.com)](https://nickyoder.com/kelly-criterion/) (accessed 2026-03-05); [Kelly Criterion QuantConnect Research](https://www.quantconnect.com/research/18312/kelly-criterion-applications-in-trading-systems/) (accessed 2026-03-05)

**Fits our case because:** MIDGE already implements Kelly position sizing in `learning_config.py` with a $50K paper account. But Kelly is being applied to a combined 19.9% win rate, not to combo-specific win rates. The best combos (66.7%, 50%) warrant significantly larger Kelly fractions than the worst (8.3%). Applying uniform Kelly to heterogeneous combos is mathematically wrong and leaves money on the table.

**Tradeoffs:** Kelly sizing requires reliable win rate estimates. The QuantConnect research tested 1.5x Kelly (not fractional) and found only 38.5% of parameter combinations beat benchmark — suggesting even this well-tested formula is fragile with noisy estimates. MIDGE has only 156 graded predictions total, well below the 666 needed for 99% confidence in expectancy calculations.

**Source:** [Profit Factor Standards 2024-2025](https://www.greshamllc.com/media/kycp0t30/systematic-report_0525_v1b.pdf) (accessed 2026-03-05)

---

### 3. Combo-Specific Position Sizing (Highest-Leverage Change)

**What:** Apply different Kelly fractions to different domain combinations based on their observed win rates, rather than one global Kelly fraction.

**Evidence:** MIDGE's own replay data shows massive combo-level variance:
- `events+macro+price+sentiment+technical`: 66.7% win rate (n=3)
- `events+fundamentals+institutional+macro+price`: 50% win rate (n=4)
- `events+macro+price`: 31.2% win rate (n=32) — most reliable volume
- `events+insider+institutional+macro+price`: 8.3% win rate (n=12)

At 31.2% win rate with 4:1 R:R: Kelly f* = (4×0.312 - 0.688)/4 = 0.56/4 = 14% of capital (half-Kelly: 7%). At 8.3% win rate with 4:1 R:R: Kelly f* = (4×0.083 - 0.917)/4 = -0.585/4 = negative (don't trade).

The paper_trade_min_combo_mean gate of 0.25 (from `learning_config.py`) already blocks combos below 25% historical win rate. This is the right instinct, but it's a binary gate rather than a gradient. A gradient (smaller positions for 26% combos, larger for 50%+ combos) would improve expected value.

**Source:** MIDGE codebase review — `learning_config.py` (paper_trade_min_combo_mean = 0.25); MIDGE MEMORY.md (replay results section); [Position Sizing Strategies for Algo-Traders](https://medium.com/@jpolec_72972/position-sizing-strategies-for-algo-traders-a-comprehensive-guide-c9a8fc2443c8) (accessed 2026-03-05)

**Fits our case because:** ComboThompson already tracks per-combination Beta distributions. The mean of that distribution IS the estimated win rate for that combo. Using it directly in Kelly sizing is a natural and architecturally consistent extension. The data is already there; it just needs to flow into the position size calculator.

**Tradeoffs:** Small sample sizes make combo-level win rate estimates noisy. The best combos have n=3 and n=4 — statistically meaningless. The 0.25 gate and a minimum-n requirement (e.g., n ≥ 15) before using combo-specific sizing would prevent overfitting to small samples.

---

### 4. Regime-Based Position Sizing (Hybrid Kelly-VIX)

**What:** Scale Kelly fraction by (1 - VIX percentile rank), reducing exposure during high-volatility regimes and increasing it during calm regimes.

**Evidence:** A 2024 arxiv study (arXiv:2508.16598v1) directly tested Kelly vs VIX-scaled Kelly vs hybrid on options strategies. The hybrid approach achieved 23.1% annualized returns with 18.5% volatility in 2024, while pure Kelly produced 17.2% max. Critically, max drawdowns stayed under 11% even in aggressive configurations. The mechanism: positions expand when VIX rank is low (stable conditions), contract when VIX rank is high (stressed conditions). This is "buy the dip on vol, sell the spike on vol" for position sizing.

**Source:** [Hybrid Kelly-VIX Paper](https://arxiv.org/html/2508.16598v1) (accessed 2026-03-05)

**Fits our case because:** MIDGE already has a RegimeClassifier (`regime_classifier.py`) that classifies into bull/bear/volatile/sideways based on SPY returns and annualized volatility. MIDGE also has a VixClient with term structure data. The regime_deltas in `learning_config.py` already adjust hypothesis gates by regime — the same pattern could adjust Kelly fractions. In "volatile" regimes, reduce all position sizes by 40-60%. In "bull" regimes, increase by 10-20%.

**Tradeoffs:** VIX-based scaling is documented primarily for options strategies (the arxiv paper). Equity/futures applicability is less directly studied. MIDGE's regime classification uses a 20-day window, which may lag fast-moving volatility spikes. The VixClient provides more real-time term structure data — the front month VIX level is a better real-time signal than the 20-day SPY vol estimate.

---

### 5. Trend-Following as the Analogous System (20% Win Rate, Provably Profitable)

**What:** Mature trend-following CTA strategies operate with 20-40% win rates and are provably profitable over decades because their payoff ratios are 3:1 to 10:1.

**Evidence:** Documented by CFM, multiple CTA performance databases, and the Turtle Trading experiment. The Turtles expected ~40% win rates with breakout trading. Trend followers accept many small losses for infrequent large gains. Quantified Strategies confirmed: the 200-day moving average applied to S&P 500 achieved profitability with a **28% win rate** because "the strategy captures large trends, making average winning trades substantially larger than average losing trades."

Sharpe ratios for mature trend followers cluster around 0.8-1.1 — similar to equity indices over long periods. It works, but requires patience through long drawdown periods (18-25% historical drawdowns, 3-6 month losing streaks).

**Source:** [QuantifiedStrategies Win Rate](https://www.quantifiedstrategies.com/win-rate-trading/) (accessed 2026-03-05); [CFM Trend Following Paper](https://www.cfm.com/wp-content/uploads/2022/12/266-2018-The-Convexity-of-trend-following.pdf) (accessed 2026-03-05); [Turtle Trading Rules Analysis](https://tradeciety.com/money-management-turtle-traders) (accessed 2026-03-05)

**Fits our case because:** MIDGE's event-driven, multi-domain pattern approach is structurally similar: low hit rate on any given prediction, but when all domains converge, the move should be large. The key missing piece is measuring MIDGE's actual average win magnitude vs average loss magnitude. If MIDGE's winners are capturing 15-20% moves (the kind of moves that happen after true multi-domain convergence) while losers give back 5-8%, the payoff ratio is ~2.5:1 — not enough. If winners average 25%+ and losers average 5-6%, the payoff ratio is 4-5:1 — already profitable.

**Tradeoffs:** Trend following requires instruments with large potential moves. Pure equity long positions are capped by typical 10-20% move expectations. Futures (as Guiding Light mentioned preferring) have leveraged payoffs, improving the effective R:R without needing the stock to move more.

---

## Novel Approaches

### 1. Combo Win Rate as a Continuous Signal (Not a Gate)

**What:** Instead of a binary gate (block combos below 25% WR), use the ComboThompson mean directly as a continuous multiplier on position size: position_size = base_kelly × combo_thompson_mean / 0.50 (normalized to expected win rate).

**Why it's interesting:** The current gate is discontinuous — a combo at 24% gets blocked entirely while a combo at 26% gets full allocation. In reality, a 26% combo with 4:1 R:R has almost no positive expected value. A continuous multiplier would naturally reduce sizing for marginal combos while scaling up for high-confidence ones.

**Evidence:** Theoretical backing from Kelly theory — position size should scale linearly with edge. Continuous Kelly implementations are standard in algorithmic trading literature. The combo Thompson distributions already exist; this is plumbing.

**Source:** [Kelly Criterion Wikipedia](https://en.wikipedia.org/wiki/Kelly_criterion) (accessed 2026-03-05); MIDGE codebase — ComboThompson distributions already implemented per MEMORY.md

**Fits our case because:** Zero architectural changes needed. The ComboThompson Beta distribution already has a `.mean()`. Apply it as a multiplier to the base Kelly fraction. The result: `events+macro+price` at 31.2% WR gets 0.62× base, while `events+macro+price+sentiment+technical` at 66.7% gets 1.33× base.

**Risks:** Sample sizes (n=3, n=4 for top combos) make mean estimates unreliable. Must implement maturity guard (same as existing Thompson maturity guard of n≥5) before using combo-specific sizing. Could also swing wildly with early bad luck on a high-potential combo.

---

### 2. Pre-Event Suppression Windows (Not Post-Event Filters)

**What:** Suppress MIDGE position sizing (not signal generation) in the 24-48 hours BEFORE known high-uncertainty events (FOMC, CPI, NFP), then resume normal sizing after the event resolves.

**Why it's interesting:** The standard approach is to filter signals around events. The better approach is to keep generating signals but suppress position sizes — preserving the learning data while protecting capital. After FOMC resolves, MIDGE's signals may become more directionally reliable as uncertainty collapses.

**Evidence:** Quantpedia research on FOMC meeting effects found that FOMC days account for 13%+ of cumulative returns over a 40-year period (1960-2000), with average daily return 0.27% on FOMC days vs ~0.05% on other days. Pre-FOMC drift 24h before decisions shows positive bias in high-sentiment environments, negative in low-sentiment. This suggests: the 24h before FOMC is a good time to be flat (not because signals are wrong, but because the uncertainty premium makes sizing riskier). After FOMC, positions can be sized normally.

**Source:** [QuantPedia FOMC Effect Strategy](https://quantpedia.com/strategies/federal-open-market-committee-meeting-effect-in-stocks) (accessed 2026-03-05); [Federal Reserve FOMC Calendar](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm) (accessed 2026-03-05); [Pre-FOMC Announcement Drift Research](https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr512.pdf) (accessed 2026-03-05)

**Fits our case because:** MIDGE already has an EconomicCalendar integration for FOMC/CPI/NFP suppression. But the current implementation appears to suppress signal GENERATION. Switching to suppressing position SIZING would preserve learning data (important for Thompson updates) while still protecting capital during uncertainty windows.

**Risks:** If MIDGE signals are actually better around FOMC (pre-announcement drift), suppressing positions around FOMC would hurt returns. Needs empirical testing with MIDGE's own signal data. FOMC effect may have weakened post-2022 after it became widely documented.

---

### 3. Expectancy Measurement as the Core Missing Metric

**What:** Explicitly measure and track E[win amount] - E[loss amount] per combo type, as the primary profitability indicator — not win rate.

**Why it's interesting:** MIDGE tracks win/loss binary outcomes but (from the replay data review) doesn't appear to track average magnitude of wins vs losses by combo. Without this, Kelly sizing is being applied blindly. The 19.9% win rate is either profitable or not depending entirely on the average win magnitude, which is unknown from the replay_results.json reviewed (it showed only "alerts": [], "phase": "replay" — the file appears to be reset or empty).

**Evidence:** ESMA study of 43,000 trading accounts found that traders with win-loss ratios above 2.0 still experienced negative returns in 34% of cases — the reason being their loss magnitudes exceeded win magnitudes despite more wins. Win rate alone is not predictive of profitability.

The QuantConnect research showed that the Kelly fraction is highly sensitive to the payoff ratio estimate. A trader who overestimates their payoff by 20% will over-size positions and face disproportionately larger drawdowns.

**Source:** [Kelly Criterion Position Sizing Explained](https://blog.traderspost.io/article/kelly-criterion-position-sizing-explained) (accessed 2026-03-05); [Profit Factor vs Win Rate vs Payoff Ratio](https://www.pnlledger.com/profit-factor-vs-win-rate-vs-payoff-ratio/) (accessed 2026-03-05)

**Fits our case because:** The 5% success threshold in `outcome_collector.py` sets a binary win/loss based on 5% minimum move. But MIDGE doesn't track *how much* the winner moved (8%? 15%? 40%?) or *how much* the loser moved (5.1%? 12%?). These numbers are the payoff ratio. Without them, Kelly sizing is a formula being applied to missing inputs.

**Risks:** Requires adding magnitude tracking to the outcome grading pipeline. The ActiveTracker already tracks MFE/MAE — the data exists, it just needs to flow into the outcome records.

---

## Emerging Approaches

### 1. DSR-Gated Combo Activation (Anti-Overfitting for Small Samples)

**What:** Before activating combo-specific position sizing, require the combo's Thompson distribution to pass a Deflated Sharpe Ratio test — penalizing for small sample size, non-normality, and selection bias.

**Momentum:** DSR (Bailey & Lopez de Prado, 2013) has grown from academic paper to practitioner standard. Integrated into MIDGE's own hypothesis validation layer already.

**Source:** [Deflated Sharpe Ratio - Wikipedia](https://en.wikipedia.org/wiki/Deflated_Sharpe_ratio) (accessed 2026-03-05); [DSR Medium Article](https://medium.com/balaena-quant-insights/deflated-sharpe-ratio-dsr-33412c7dd464) (accessed 2026-03-05); [Bailey & Lopez de Prado Original Paper](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf) (accessed 2026-03-05)

**Fits our case because:** MIDGE already uses DSR in `hypothesis_validator.py` for hypothesis promotion. The same logic applies to combo win rates. A combo with 66.7% win rate at n=3 should NOT be promoted to high-Kelly sizing — its DSR would reflect the tiny sample. A combo with 31.2% at n=32 has a much more credible DSR score. DSR naturally penalizes selection bias from trying many combos and reporting the best.

**Maturity risk:** The DSR calculation requires enough data points to estimate skew and kurtosis of the return distribution. At n=3 to n=12 (MIDGE's combo sizes), DSR estimates will themselves be unreliable. Minimum n=20-30 for meaningful DSR on a combo may mean waiting months for sufficient data.

---

### 2. Probability Stacking Math Applied to Domain Independence

**What:** Using the independence assumption for uncorrelated domain signals, theoretically: if P(win | domain_A) = 0.4 and P(win | domain_B) = 0.4 and they are independent, P(win | both) ≈ 0.4 × (1/0.5 base rate boost) = higher than either alone.

**Momentum:** The probability stacking concept is gaining traction in retail quantitative trading communities (Edgeful.com, 2025 article) as an explicit framework for confluence-based trading.

**Source:** [Edgeful Probability Stacking 2025](https://www.edgeful.com/blog/posts/trade-backtesting-2025-best-practices) (accessed 2026-03-05)

**Fits our case because:** MIDGE's convergence engine requires min_domains=3 (Law 2). This is already implementing probability stacking. The question is whether the domain signals in MIDGE are genuinely independent. The correlation_tracker already tracks cross-domain correlations. Signals that are highly correlated (e.g., technical + price might be correlated) should NOT be counted as independent confirmations.

**Maturity risk:** The math of independence breaks down quickly. If two "domains" in MIDGE are both derived from price history (technical signals and price signals), they may share 70%+ of their information. True independence is rare in financial data. This needs empirical validation using MIDGE's own correlation_tracker data.

---

### 3. Regime-Conditional Win Rate Tracking

**What:** Track win rates separately for each RegimeClassifier output (bull/bear/volatile/sideways), then use regime-conditional win rates for Kelly sizing rather than global win rates.

**Momentum:** The arxiv hybrid Kelly-VIX paper (2024) and QuantMonitor regime filtering research both demonstrate measurable performance improvement from regime-conditional strategies. MIDGE already has the infrastructure (RegimeClassifier + regime_deltas in learning_config).

**Source:** [Hybrid Kelly-VIX ArXiv Paper 2024](https://arxiv.org/html/2508.16598v1) (accessed 2026-03-05); [QuantMonitor Regime Filtering](https://quantmonitor.net/how-to-identify-market-regimes-and-filter-strategies-by-trend-and-volatility/) (accessed 2026-03-05)

**Fits our case because:** MIDGE's ThompsonSampler already maintains regime-specific distributions (per the architecture). The hypothesis_gates already include regime_deltas. The natural extension: when sizing a paper trade, look up the regime-specific combo win rate, not the global win rate. A `contracts+macro+price` combo might be 40% in bull markets and 15% in bear markets — the Kelly fraction for the same combo should be dramatically different.

**Maturity risk:** Requires sufficient data per regime to estimate regime-conditional win rates. With 156 total graded predictions split across 4 regimes, most regime+combo cells will have sample sizes too small to be reliable. This is a 6-12 month runway investment before paying off.

---

## Gaps and Unknowns

### Critical Unknown: What Is MIDGE's Actual Payoff Ratio?

The replay_results.json file was essentially empty ({"alerts": [], "phase": "replay"}) — the file appears to have been reset. The key number needed to determine if MIDGE is already profitable is: **average winning magnitude vs average losing magnitude**. Without this, all Kelly calculations are using a parameter we don't have.

The 5% success threshold means a win is "any move > 5%." But a 5.1% win is very different from a 35% win for Kelly purposes. This is the most critical gap in the current system.

**What to do:** Pull a sample of graded outcomes from the outcomes.jsonl file and calculate:
- Average price move for winning predictions (those above 5% threshold)
- Average price move for losing predictions
- Payoff ratio = avg_win / avg_loss

If this ratio is above 4.0, MIDGE is already break-even or better. If it's below 3.0, we need to either improve win rate above 25% or find higher-payoff instruments.

---

### Partially Unknown: Sample Size Adequacy

With 156 graded predictions, 19.9% win rate has a 95% CI of approximately [13.8%, 27.2%] (binomial). The edge is proven (z=4.74), but the true win rate could be anywhere in that range.

At the low end (13.8%), even a 6:1 payoff ratio barely breaks even. At the high end (27.2%), a 3:1 payoff ratio is comfortably profitable.

**What to do:** The target should be 400-500 graded outcomes before making significant architectural decisions about win rate optimization. More signal generation + more grade completion is the primary lever.

---

### Unknown: Correlation Between MIDGE Domains

The convergence engine requires min_domains=3 and requires < 30% domain overlap for independence in the PatternWatcher. But financial domains are not independent. Insider trading signals and earnings events are correlated (insiders buy before earnings). Technical signals and price signals may be near-identical.

If MIDGE is counting 3 correlated domains as "3 independent confirmations," the actual probability improvement from stacking is much less than the theoretical maximum.

**What to do:** Run correlation_tracker analysis on the domain signals in the MIDGE signal archive. If insider + events domains have >50% correlation, they should be treated as 1.5 independent signals, not 2.

---

### Unknown: Instrument Appropriateness

Guiding Light's vision mentions preferring instruments "where payoff math is linear (futures-like)." If MIDGE's current predictions are targeting equities (typical 10-20% maximum potential gain over 14-45 day windows), the payoff ratios may be structurally limited. Futures positions on the same directional move would provide 5-10× leverage, improving the effective R:R without requiring larger stock moves.

**What to do:** Measure the actual price move distributions in MIDGE's outcome data, then model what the same moves would return in micro-futures (NQ, ES, MES) vs equity positions.

---

### Evidence Contradictions

The QuantifiedStrategies article claims the "formula only works if the win ratio is above 50%" for the risk-of-ruin calculation, but this applies only to specific formulations (Gambler's Ruin with equal bet sizes). The general risk of ruin formula for asymmetric payoffs does not require 50%+ win rates. This is a documentation error in that source, not a fundamental constraint.

---

## Synthesis

### The Strongest Approach: Fix What's Broken Before Optimizing

MIDGE's architecture is correct. The instinct to use Kelly, Thompson Sampling, combo gates, and regime classifiers is right. But there are three fundamental inputs that are either missing or wrong:

**1. The payoff ratio is unmeasured.** Every Kelly calculation in the system is using an implicit payoff ratio that nobody has computed. Before optimizing position sizing, measure the average winner vs average loser magnitude from the existing outcomes.jsonl data. This takes an hour of analysis and could determine whether MIDGE is already profitable or structurally unprofitable.

**2. Kelly is being applied uniformly to heterogeneous combos.** The combo data proves that `events+insider+institutional+macro+price` (8.3% WR) and `events+macro+price` (31.2% WR) are radically different bets. They should have radically different position sizes. The ComboThompson already has the win rate estimates — apply them to Kelly sizing.

**3. The 19.9% global win rate is a misleading average.** The high-quality combos (n≥15, WR≥30%) are viable. The low-quality combos (n≥10, WR<20%) are destroying the average. Routing paper trades through a stricter combo quality gate — raising `paper_trade_min_combo_mean` from 0.25 to 0.30, and requiring n≥15 samples before using the estimate — would improve reported win rates by eliminating statistically insignificant combos. This is not "finding new patterns"; it's correctly using the data that already exists.

### What Combination Works Best

**Tier 1 (Implement now, zero architectural change):**
- Measure payoff ratio from existing outcome data
- Raise combo sample size requirement to n≥15 before applying combo-specific sizing
- Continue current paper trading at current gates while collecting more outcome data

**Tier 2 (Implement after Tier 1 validates payoff ratio):**
- Combo-specific Kelly fraction using ComboThompson mean as multiplier
- Suppress position sizing (not signal generation) during 24h pre-FOMC/CPI/NFP windows
- Track regime-conditional win rates for future use

**Tier 3 (6-12 month runway, requires more data):**
- Full regime-conditional Kelly sizing
- DSR-gated combo activation
- Move toward futures instruments for better R:R on the same directional signals

### The Minimum Win Rate Question: Already Answered

19.9% win rate CAN be profitable. The Turtle Trading evidence, the 200-day MA evidence, and the mathematical break-even table all confirm: with 4:1 or better payoff ratio, 20% win rate generates positive expectancy. The question is not "how do we improve win rate from 20% to 50%?" The question is "what is our actual payoff ratio, and is it above 4:1?"

If the payoff ratio is 4:1 or better, the right move is to increase trade frequency (more graded predictions, more capital deployed) while maintaining the current quality filter.

If the payoff ratio is below 3:1, the right move is to pursue higher-payoff instruments (futures, long options on convergent signals) rather than trying to improve the underlying win rate.

### What the Orchestrator Needs to Know

The most valuable single action is a **payoff ratio audit** on the existing outcomes.jsonl. Parse all graded outcomes, separate wins (>5% move) from losses, calculate average magnitude of each. This number determines whether MIDGE is already financially profitable at current win rates. Everything else — regime filtering, combo-specific sizing, confidence calibration — is secondary optimization. Without knowing the payoff ratio, all position sizing work is being done blind.

The second most valuable action is **raising the paper trade combo sample size gate**. Currently, a combo with n=3 samples can influence position sizing if its mean > 0.25. The `events+macro+price+sentiment+technical` combo at 66.7% WR with n=3 is statistically a coin flip. Requiring n≥15 before using combo-specific sizing would eliminate false positives while preserving real signal from mature combos like `events+macro+price` (n=32).

---

*Sources consulted:*
- [Kelly Criterion Wikipedia](https://en.wikipedia.org/wiki/Kelly_criterion)
- [QuantifiedStrategies Win Rate](https://www.quantifiedstrategies.com/win-rate-trading/)
- [QuantifiedStrategies Risk of Ruin](https://www.quantifiedstrategies.com/risk-of-ruin-in-trading/)
- [Kelly Criterion - Quantitative Trading (nickyoder.com)](https://nickyoder.com/kelly-criterion/)
- [QuantConnect Kelly Applications](https://www.quantconnect.com/research/18312/kelly-criterion-applications-in-trading-systems/)
- [P&L Ledger Profit Factor vs Win Rate](https://www.pnlledger.com/profit-factor-vs-win-rate-vs-payoff-ratio/)
- [Hybrid Kelly-VIX ArXiv Paper 2024](https://arxiv.org/html/2508.16598v1)
- [QuantPedia FOMC Meeting Effect](https://quantpedia.com/strategies/federal-open-market-committee-meeting-effect-in-stocks)
- [Pre-FOMC Announcement Drift (NY Fed)](https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr512.pdf)
- [Deflated Sharpe Ratio Original Paper](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)
- [Deflated Sharpe Ratio Medium](https://medium.com/balaena-quant-insights/deflated-sharpe-ratio-dsr-33412c7dd464)
- [CFM Trend Following Convexity](https://www.cfm.com/wp-content/uploads/2022/12/266-2018-The-Convexity-of-trend-following.pdf)
- [Probability Stacking Edgeful 2025](https://www.edgeful.com/blog/posts/trade-backtesting-2025-best-practices)
- [QuantMonitor Regime Filtering](https://quantmonitor.net/how-to-identify-market-regimes-and-filter-strategies-by-trend-and-volatility/)
- [Kelly Criterion TradersPost](https://blog.traderspost.io/article/kelly-criterion-position-sizing-automated-trading)
- [Gresham Systematic Report 2025](https://www.greshamllc.com/media/kycp0t30/systematic-report_0525_v1b.pdf)
- [Algofuturestrader Win Rate Ranges](https://algofuturestrader.com/trading-system-win-rates-ranges-realities-and-refinements/)
- [Event-Driven Trading QuantifiedStrategies](https://www.quantifiedstrategies.com/event-driven-trading-strategies/)
- [Insider Trading Returns (Wharton)](https://rodneywhitecenter.wharton.upenn.edu/wp-content/uploads/2014/04/9919.pdf)
- [SSRN Death of Insider Trading Alpha](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5966834)
- [Machine Learning Multi-Factor Quant Trading 2025](https://arxiv.org/html/2507.07107)
