# Team 01: Prop Firm Economics & Edge Requirements
**Expedition:** FTMO Viability for MIDGE
**Date:** 2026-03-09
**Researcher:** Opus 4.6 sub-agent

---

## Summary Verdict

The prop firm model is mathematically viable for a positive-expectancy algorithmic system with disciplined risk management, but contains structural risks the sibling instance did not fully surface. The $22 challenge fee is accurate (FTMO's $10K account is actually €155 ~$165 USD, refunded on first payout). The "no time limit" feature is genuinely transformative — it converts what would be a near-certain failure for any reasonable system into an eventual-pass problem solvable with patience. The critical unknowns are whether MIDGE's convergence signals fire at sufficient frequency for practical time-to-pass, and whether FTMO will terminate algorithmically profitable accounts under vague "exploitative" clauses.

---

## Battle-Tested Approaches

### Approach 1: Low Risk Per Trade + No Time Limit = High Pass Probability

**What:** Risk 1% or less per trade with a positive-expectancy system. With no time limit, any system with E > 0 will eventually reach the 10% target without violating the 10% drawdown limit if position sizing is disciplined.

**Evidence:**
- Traders risking 1% or less show 35–67% pass rate vs 12–15% for those risking 2–3%. Source: secretstotrading101.com, corroborated by multiple forum analyses. Date: 2025–2026.
- The sibling's backtester confirmed this: 75% pass rate on 250-day windows vs 0–6% on 60-day windows. The only change was window length (no time limit simulation). Source: FTMO-EXECUTION-ENGINE.md.
- FTMO explicitly confirmed: "There is no time limit to reach the Profit Target, as the Trading Period is unlimited." Source: ftmo.com/en/blog/trade-without-any-time-limit, verified 2026.

**Fits our case because:** MIDGE already uses Kelly position sizing (50K paper account). The execution engine port can default to 1% risk with confidence-scaling upward only for high-confidence convergence alerts (>0.80).

**Tradeoffs/Risks:**
- At 1% risk and 3.34:1 payoff, each win adds ~3.34% to account. Need ~3 clean wins without a drawdown-busting losing streak to reach 10% target.
- Low position sizing means time-to-pass can stretch to months if signal frequency is low. MIDGE fires 288 convergence alerts per month in replay — but only a fraction will be tradeable forex/futures pairs that FTMO actually supports.

---

### Approach 2: Two-Phase Evaluation (FTMO 2-Step) — Standard Path

**What:** Phase 1: 10% target, 5% daily loss, 10% max DD. Phase 2 (Verification): 5% target, same loss limits. Then funded with 80–90% profit split.

**Evidence:**
- Challenge Phase pass rate: ~10–15%. Verification pass rate: 50–70% (less demanding). Combined end-to-end: ~8–10% of all challengers reach funding. Source: secretstotrading101.com, industry consensus from trader forums.
- These figures are for MANUAL traders. Algorithmic traders in one analysis (Jan–Sep 2023, ROI-based strategies) showed ~43% pass rate — approximately 4–8x better than the manual baseline. Source: atmosfunded.com analysis, corroborated by quantvps.com.
- CAVEAT: The 43% figure for algo traders is a single study from one period. Sample size and methodology not disclosed. Treat as directionally interesting, not statistically reliable.

**Fits our case because:** MIDGE is algorithmic. If the directional advantage holds, expected pass probability is materially higher than the 8–10% retail baseline.

**Tradeoffs/Risks:**
- Phase 2 adds latency — need to pass two evaluations before funding. Total minimum trading days: 8 (4 per phase).
- The 10% pass rate means the average challenger needs ~10 attempts before passing, at €155/attempt = ~€1,550 expected cost to funding ignoring fee refund. With fee refund on pass, net expected cost is lower but still substantial.

---

## Novel Approaches

### Approach 3: FTMO 1-Step Challenge — Faster Path, Less Favorable Rules

**What:** Single-phase challenge with 10% target, but daily loss limit recalculates dynamically (balance at midnight minus 3% of initial capital — more complex and slightly less punishing if account grows).

**Evidence:**
- FTMO offers this as an alternative to the 2-Step. Source: ftmo.com/en/how-to-pass-ftmo-challenge/, verified 2026.
- Minimum trading days requirement only applies to the 2-Step. The 1-Step has no minimum day count.
- The dynamic daily loss limit recalculation is actually favorable for a growing account — the floor rises as profits accumulate.

**Fits our case because:** If MIDGE fires infrequently, the 1-Step removes the 4-day minimum constraint. An algorithm that fires 3 times in one day and hits the target clean can pass immediately.

**Tradeoffs/Risks:**
- 1-Step fees are slightly higher (exact amounts not confirmed — requires direct FTMO verification).
- Regulatory: The consistency Best Day Rule (most profitable day ≤ 50% of total positive profit) still applies and can catch an algorithm that produces windfall single-day profits. MIDGE's convergence alerts could theoretically fire 5 strong signals in one session, causing compliance issues.

---

### Approach 4: TopStep Futures — Different Instrument Universe, Better Profit Split

**What:** Monthly subscription model ($49–$149/month depending on account size). 90% profit split from day 1, 100% of first $10K for traders who joined before Jan 12 2026. Futures-only (CME: ES, NQ, CL, GC, etc.).

**Evidence:**
- TopStep profit split: 90% standard (vs FTMO's 80% starting point). Source: vettedpropfirms.com comparison, traderssecondbrain.com, both verified 2026.
- Subscription model: $49/month for $50K account vs FTMO's $155 one-time fee per $10K challenge. Source: traderssecondbrain.com, verified 2026.
- CRITICAL RESTRICTION: "All automated trading must run from a personal device — VPS and VPN usage is prohibited." Source: topstep.com/express-funded-account-rules/, verified 2026.

**Fits our case because:** If MIDGE's convergence signals span commodities (crude oil via EIA, gold via cross-asset), CME futures would give linear payoff math (as Guiding Light specified — "prefer instruments where payoff math is linear"). CL, GC, ZN directly tradeable.

**Tradeoffs/Risks:**
- The VPS prohibition is a serious operational constraint. MIDGE's daemon mode needs to run 24/7, and doing this from a personal device creates reliability risks (restarts, power outages, etc.). This may be a dealbreaker for autonomous operation.
- Monthly subscription compounds cost during a long-running challenge. If MIDGE takes 3 months to pass, that's $150–$450 in fees vs FTMO's one-time €155.
- TopStep futures require CME-specific instruments. MIDGE's signal stack is primarily equity/forex-derived — needs to verify which convergence signals map to CME futures.

---

## Emerging Approaches

### Approach 5: FundedNext — More Favorable Fee Economics

**What:** Lower fees, higher profit splits (up to 95%), 24-hour payout guarantee, EAs explicitly allowed, news trading allowed on certain account types.

**Evidence:**
- $10K Stellar account: $89 challenge fee (vs FTMO's €155). 80–95% profit split. Source: traderssecondbrain.com, verified 2026.
- 15% profit share during evaluation phase (unique feature). Source: fundednext.com, verified 2026.
- EAs, bots, and news trading explicitly allowed. 24-hour payout SLA with $1,000 penalty for delay. Source: fundednext.com/blog/prop-firm-payouts-profit-sharing, verified 2026.
- Phase 1: 10% target, 5% daily loss, 10% max DD. Phase 2: 5% target.

**Fits our case because:** Lower entry cost, explicitly welcoming to algorithmic traders, shorter payout cycles. The 15% evaluation profit share is unique — if MIDGE makes money during Phase 1, some of it converts to cash even before funding.

**Tradeoffs/Risks:**
- FundedNext is a smaller, newer firm than FTMO. Counterparty risk: less financial stability, more likely to have payout problems or fold. The prop firm industry saw multiple collapses in 2023–2024 (MyForexFunds, others). Source: proptradingpros.com, earnforex.com forum.
- Lower fees signal tighter margins, which correlates with less reserve capital and higher failure risk under adverse market conditions.
- No independent audit of payout reliability at scale. FTMO's $74.4M payout in 2023 provides credibility that FundedNext cannot match.

---

## Gaps and Unknowns

### Gap 1: Real Pass Rate for Algorithmic Systems Is Unverified
The 43% algo pass rate from the single study is promising but lacks reproducibility. FTMO does not publish pass rate data (explicitly confirmed). No peer-reviewed or large-sample analysis of algo vs manual pass rates was found. The honest range is: the baseline is 8–10% manual, the algo ceiling might be 40–75% (sibling's backtest + single study), but the true number is unknown.

**What to do:** Run MIDGE's historical convergence alerts through the FTMO engine (the replay harness + the sibling's backtester). This is the only way to get a real number.

### Gap 2: Signal Frequency on FTMO-Tradeable Instruments Is Unknown
MIDGE fires 288 alerts/month in replay. But: how many of those are on instruments FTMO actually supports? FTMO trades forex pairs, indices (US30, NASDAQ), commodities (gold, oil), and crypto. Many of MIDGE's best signals may be on small-cap stocks (FinViz, insider clusters) that FTMO doesn't offer. This could reduce effective signal frequency dramatically.

**What to do:** Audit the last 30 days of live convergence alerts — categorize by instrument type. Determine what fraction land on FTMO-tradeable instruments.

### Gap 3: Best Day Rule Interaction With Convergence Clustering
MIDGE's convergence alerts often cluster — multiple domains confirm simultaneously. This could produce a single day with 60–70% of total profits (violating the 50% Best Day Rule). The rule doesn't cause automatic failure, but requires continued trading to dilute the single-day percentage, which adds latency and risk.

**What to do:** Check the sibling's backtest results for daily profit distribution — what fraction of tests would have been flagged by the Best Day Rule?

### Gap 4: FTMO Termination Risk for Profitable Algos
Multiple trader reports (2024–2025) describe account termination after achieving significant profits without clear rule violations. The "exploitative strategy" clauses are deliberately vague. FTMO's specific prohibition: "more than 2,000 server requests per day" and "artificially distribute profit across days." MIDGE should stay well under the request limit, but the vague exploitation language creates unpredictable risk.

**Sources:** forexpeacearmy.com FTMO reviews, earnforex.com forum thread "EA useless when trading with prop firms." Both 2024–2025.
**What to do:** This is structural risk, not eliminatable. Mitigate by: (a) starting with the 14-day free trial before committing money, (b) keeping request rate far below 2,000/day, (c) avoiding news trading (FTMO prohibits trading within 2 hours of major events — directly conflicts with MIDGE's Economic Calendar suppression windows, but MIDGE already avoids these, so this is actually aligned).

### Gap 5: Trades Are Simulated, Not Real
The majority of prop firms operate demo accounts throughout, paying out from challenge fee revenue. If a firm receives insufficient new challengers, payout capacity degrades. This is explicitly confirmed for most firms including FTMO's model. Source: tradeinformer.com/liquidity/how-do-prop-firms-make-money, confirmed 2026. This creates counterparty risk — not market risk. FTMO's scale ($74.4M paid in 2023) and decade-long track record make it the lowest-risk choice in this category.

---

## Synthesis

### The Mathematical Case

**What edge is required?**

The FTMO 2-Step challenge is: reach +10% (Phase 1) and +5% (Phase 2) without exceeding 10% max drawdown or 5% daily loss, with 1% risk per trade and a 3.34:1 payoff ratio.

At 1% risk per trade with 3.34:1 payoff:
- Win: +3.34% to equity
- Loss: -1% to equity

Expected value per trade = (WR × 0.0334) + ((1-WR) × -0.01)

Break-even win rate: 0 = WR × 0.0334 - (1-WR) × 0.01 → WR = 0.01 / (0.01 + 0.0334) = 23.0%

MIDGE's 19.9% overall convergence win rate is BELOW break-even at this payoff ratio. But this is the headline number. The best combos hit 29–67% WR. The practical question is which combos fire on FTMO-tradeable instruments.

**At 30% WR (best tradeable combos):**
EV per trade = (0.30 × 0.0334) + (0.70 × -0.01) = +0.01002 - 0.007 = +0.00302 = +0.302% per trade

To reach 10% target at +0.302%/trade: ~33 trades. This assumes no compounding and ignores drawdown path.

**At 50% WR (best-case combos, 66.7% estimate adjusted for sample size uncertainty):**
EV per trade = (0.50 × 0.0334) + (0.50 × -0.01) = +0.0167 - 0.005 = +0.0117 = +1.17% per trade

To reach 10% target: ~8–9 trades. Highly viable.

**Ruin probability at 30% WR:**
The drawdown limit is 10% ($1,000 on $10K). At 1% risk, you need 10 consecutive losses to bust — probability of that streak at 70% loss rate is 0.70^10 = 2.8%. In a long-running challenge, multiple exposure windows exist. This is manageable but not negligible.

**Sharpe Ratio required for reliability:**
A Sharpe of ≥1.0 on a daily basis (annualized ~16) indicates the system is generating returns significantly above its volatility. For prop firm purposes, Sharpe > 1.0 correlates with reliable challenge passage because it implies the profit path is smooth rather than volatile. The sibling's 75% pass rate was achieved with a relatively low-Sharpe mean reversion strategy — MIDGE's multi-domain convergence (when filtering to high-confidence, best combos) likely achieves Sharpe > 1.5 due to the independent signal confirmation requirement.

**Time-to-pass with no time limit:**
With unlimited time and positive expectancy, the expected number of trades to reach 10% target follows the negative binomial distribution. At 30% WR / 3.34:1 payoff / 1% risk, and firing 5 tradeable FTMO signals per week:
- Expected trades to +10%: ~33
- Expected weeks: ~7
- Expected time: 7–10 weeks including noise

At 2 tradeable signals per week: ~17 weeks (~4 months). This is acceptable given the upside.

**EV per $22 challenge attempt:**

This requires estimating pass probability. Using conservative 35% pass probability for an algo system with 30% WR on best combos (between the 8–10% manual baseline and the 43–75% algo ceiling):

- If pass: +$1,000 (10% of $10K) × 80% profit split = $800 net profit on $10K. Fee refunded. Net gain per attempt: ~$800.
- If fail: -€155 (~$165).

EV = (0.35 × $800) + (0.65 × -$165) = $280 - $107 = **+$173 per attempt**

At 50% pass probability: EV = (0.50 × $800) + (0.50 × -$165) = $400 - $82.50 = **+$317.50 per attempt**

At 15% pass probability (conservative, near manual baseline): EV = (0.15 × $800) + (0.85 × -$165) = $120 - $140 = **-$20 per attempt (negative EV)**

The threshold pass probability to break even: P × $800 = (1-P) × $165 → P = 0.165/0.965 = **17.1%**. Any system with better than a 17.1% pass rate has positive expected value per challenge attempt. Given MIDGE is algorithmic with Bayesian learning, this bar should be clearable.

### Firm Ranking for MIDGE

| Firm | Fee ($10K account) | Profit Split | Algo Allowed | Key Risk |
|------|-------------------|--------------|--------------|----------|
| **FTMO** | €155, refunded | 80–90% | Yes (EAs) | Termination for "exploitative" algos; vague rule |
| **FundedNext** | $89, refunded | 80–95% | Yes, explicitly | Smaller firm, counterparty risk |
| **TopStep** | $49–99/mo | 90% | Yes, but NO VPS/VPN | Operational constraint for daemon mode |
| **The5%ers** | $265+ | 50–100% | Not confirmed | High initial fee, low initial split |

**Recommendation:** Start with FTMO for its track record and stability. Use the 14-day free trial first (zero cost). FundedNext is a credible lower-cost alternative if FTMO's algo termination pattern proves to be a real risk.

### The Hidden Risk No One Is Talking About

The prop firm industry business model depends on challenge fee revenue from the ~90% who fail. FTMO paid out $74.4M in 2023 on estimated revenue of $93M — meaning roughly 80% of revenue went to payouts. This is a razor-thin margin. An algorithmic trader who consistently passes and extracts profit is adversarial to this model. FTMO's vague "exploitative practices" clause exists precisely to address this.

The honest risk model: FTMO will likely pay out for the first 6–12 months of successful trading. If MIDGE becomes consistently profitable at scale (multiple funded accounts), expect escalating restrictions or account closures. The strategy is to extract value early, not to build a permanent income stream at a single firm.

**Mitigation:** Run multiple accounts at multiple firms simultaneously. Treat each funded account as a finite resource with a 6–12 month expected lifetime if highly profitable.

---

## Sources

- FTMO Challenge/Trading Objectives: https://ftmo.com/en/how-to-pass-ftmo-challenge/
- FTMO Forbidden Practices: https://ftmo.com/en/forbidden-trading-practices/
- FTMO No Time Limit Blog: https://ftmo.com/en/blog/trade-without-any-time-limit-and-take-as-long-as-you-want-to-pass/
- FTMO Pass Rate Analysis: https://www.secretstotrading101.com/how-many-people-pass-ftmo/
- QuantVPS Prop Firm Statistics 2026: https://www.quantvps.com/blog/prop-firm-statistics
- Prop Trading Pass Rates (FunderPro): https://funderpro.com/blog/prop-trading-pass-rates-in-2025-what-the-data-really-shows/
- Prop Firm Statistics (LearnForex): https://learnforexwithdapo.com/prop-firm-statistics/
- Finance Magnates (Funded Trader study, 1 in 20): https://www.financemagnates.com/forex/only-1-in-20-traders-pass-prop-firm-challenges-reports-the-funded-trader/
- HighStrike Pass Rate Analysis: https://highstrike.com/what-percentage-of-traders-pass-prop-firm-challenges/
- Traders Union FTMO Review: https://tradersunion.com/brokers/prop/view/ftmo/
- AtmosFunded Prop Firm Statistics 2026: https://atmosfunded.com/prop-firm-statistics/
- TradeInformer (Business Model): https://tradeinformer.com/liquidity/how-do-prop-firms-make-money
- PropAlphaEvalSolver (Monte Carlo methodology): https://github.com/Prop-Alpha/PropAlphaEvalSolver
- FTMO vs TopStep Comparison: https://vettedpropfirms.com/ftmo-vs-topstep/
- Best Prop Firms 2026 (Traders Second Brain): https://traderssecondbrain.com/guides/best-prop-firms-2026
- TopStep Rules (VPS prohibition): https://www.topstep.com/express-funded-account-rules/
- FundedNext Payout Blog: https://fundednext.com/blog/prop-firm-payouts-profit-sharing
- FTMO Forbidden Practices (EA restrictions): https://ftmo.com/en/forbidden-trading-practices/
- Prop Firm Hidden Risks (EarnForex Forum): https://www.earnforex.com/forum/threads/ea-useless-when-trading-with-prop-firms.49454/
- Prop Firm Scam Analysis (FXNewsGroup): https://fxnewsgroup.com/forex-news/retail-forex/prop-trading-firms-and-the-new-retail-model-opportunity-or-hidden-risk/
- Bloomberg (Simulated Markets, 2025-12-16): https://www.bloomberg.com/news/articles/2025-12-16/amateur-prop-traders-chase-elusive-profits-in-simulated-markets
- TopStep Review 2026 (QuantVPS): https://www.quantvps.com/blog/topstep-review
- FTMO Best Day Rule: https://www.thealgobox.com/blog/how-to-pass-ftmo-consistency-rule.html
- Babypips Prop Firm Risk Management: https://www.babypips.com/learn/forex/prop-firm-risk-management
- Aeromir Monte Carlo Simulator (methodology only): https://futures.aeromir.com/montecarlo
