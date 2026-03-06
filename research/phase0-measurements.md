# Phase 0 Measurements — MIDGE Baseline Analysis

**Date:** 2026-03-05
**Purpose:** Establish the two critical numbers that determine MIDGE's next steps — payoff ratio and domain independence.
**Analysis only — no files modified.**

---

## The Central Question

MIDGE makes predictions. Before deciding whether to improve the win rate, improve the payoff, or change the strategy entirely, we need to know two things:

1. **When MIDGE is right, does it win enough to cover when it's wrong?** (Payoff Ratio)
2. **Are the "independent" signals MIDGE combines actually independent?** (Domain Independence)

---

## Task 1: Payoff Ratio

### What "Win" and "Loss" Mean in MIDGE

MIDGE's outcome system uses a specific definition: a prediction **wins** if the stock moves more than 5% in the predicted direction within the observation window. A prediction **loses** if it does not — meaning the stock moved less than 5%, or moved in the wrong direction. The 5% threshold was chosen deliberately (replacing an earlier 2% threshold that was too close to random noise).

This creates an important asymmetry: every win, by definition, represents at least a 5% move. Losses can be anything from "barely missed" to "went the wrong way entirely."

### The Raw Numbers

| Metric | Value |
|--------|-------|
| Total outcomes measured | 12,981 |
| Wins | 4,325 (33.3%) |
| Losses | 8,656 (66.7%) |
| Average win magnitude | **11.42%** |
| Median win magnitude | **8.63%** |
| Average loss magnitude | **3.42%** |
| Median loss magnitude | **2.59%** |

### The Payoff Ratio

**Payoff ratio = 3.34:1** (average) / **3.33:1** (median)

In plain language: when MIDGE is right, the move is 3.3 times larger than the move when it's wrong.

### What Does MIDGE Need to Break Even?

At a 33.3% win rate, MIDGE needs a payoff ratio of **2.0:1** to break even — meaning wins must be twice as large as losses on average. MIDGE is at **3.34:1**, which is 67% above break-even.

**Expectancy per trade: +1.52%** — MIDGE is profitable at the 33.3% win rate.

```
Formula: (33.3% x 11.42%) - (66.7% x 3.42%) = +1.52% per trade
```

### The Replay Discrepancy — Critical Finding

The replay harness (running Feb 2026 data through the convergence engine) showed a **19.9% win rate** — much lower than the 33.3% in the outcome files. This gap requires explanation.

At 19.9% win rate, the break-even payoff needed is **4.03:1**. The current payoff of 3.34:1 falls short. Expectancy at 19.9% WR:

```
(19.9% x 11.42%) - (80.1% x 3.42%) = 2.27% - 2.74% = -0.47% per trade
```

**The replay system is operating at a loss. The outcome file system is not.**

The most likely explanation: the outcome file sources (finra_short, yfinance_price, finnhub_earnings) are per-signal evaluations — each individual data source gets its own outcome recorded. The replay harness evaluates convergence alerts — combinations of signals. These are different populations measuring different things. The outcome file's 33.3% is the per-source accuracy; the replay's 19.9% is the accuracy of combined predictions. Convergence predictions are the ones that matter for actual trading.

### Win Magnitude Distribution

| Move Size | Wins | Losses |
|-----------|------|--------|
| Less than 1% | 0 (0%) | 1,748 (20%) |
| 1 — 2% | 0 (0%) | 1,646 (19%) |
| 2 — 5% | 0 (0%) | 4,138 (48%) |
| 5 — 10% | 2,620 (61%) | 642 (7%) |
| 10%+ | 1,704 (39%) | 481 (6%) |

Wins: concentrated in the 5-10% band (61%) with a substantial tail of 10%+ moves (39%).
Losses: concentrated in the 2-5% band (48%), meaning most losses represent "close but not quite" — the asset moved, just not enough, not the right way.

The 5-6% of losses that are large (10%+) represent genuine wrong-direction calls — roughly 481 cases where the asset moved strongly against the prediction. This is the genuine downside tail.

### Source-Level Win Rates (for context)

| Source | Wins / Total | Win Rate |
|--------|-------------|----------|
| finnhub_analyst | 36 / 90 | 40.0% |
| finra_short | 3,333 / 8,834 | 37.7% |
| sec_form4 | 17 / 48 | 35.4% |
| finnhub_earnings | 223 / 697 | 32.0% |
| cot_positioning | 93 / 360 | 25.8% |
| yfinance_price | 596 / 2,771 | 21.5% |
| congressional | 14 / 93 | 15.1% |
| contract_award | 12 / 81 | 14.8% |

finra_short (short interest data) dominates the outcome file — 8,834 of 12,981 records (68%). The signal MIDGE relies on most for volume is producing a 37.7% win rate at the per-source level, which is well above chance.

---

## Task 2: Domain Independence

### Why This Matters

MIDGE combines signals from different domains under the assumption they represent independent sources of evidence. If domain A and domain B are actually correlated — they both move in response to the same hidden factor — then stacking them doesn't add as much confidence as MIDGE thinks it does. It's the difference between two independent witnesses and two witnesses who heard the same rumor.

### What the Data Shows

The `lag_correlations.json` file contains 47 statistically significant correlation records across 5 sources (finra_short, fred_macro, sec_efts, sec_form4, yfinance_price). These are computed over historical signal sequences with various lag windows (how many days one signal leads the other).

**Domain-level summary (maximum absolute correlation observed per domain pair):**

| Domain A | Domain B | Max |r| | Assessment |
|----------|----------|--------|------------|
| macro (FRED) | technical (FINRA short) | **0.73** | Strongly correlated |
| insider (Form 4) | technical (FINRA short) | **0.58** | Notably correlated |
| technical <-> technical | (within-domain) | 0.47 | Moderate |
| insider | institutional (EFTS) | 0.45 | Moderate |
| institutional | technical | 0.42 | Moderate |
| institutional | macro | 0.42 | Moderate |

### The Strongest Correlation in the Data

**FINRA short interest (technical domain) leads FRED macro data by 7 days: r = -0.73, p < 0.0001, n = 53 pairs.**

This is the strongest relationship in the dataset. It means: when short interest spikes, macro indicators tend to move in the opposite direction about a week later. They are not independent — one predicts the other.

**Four relationships exceed the 0.5 threshold (strong correlation):**

1. `finra_short` (technical) → `fred_macro` (macro), lag 7d: r = -0.73
2. `sec_form4` (insider) → `finra_short` (technical), lag 42d: r = +0.58
3. `finra_short` (technical) → `sec_form4` (insider), lag 5d: r = -0.51
4. `sec_form4` (insider) → `yfinance_price` (technical), lag 22d: r = +0.51

### What This Means for MIDGE

MIDGE's convergence engine requires signals from at least 3 different domains (Law 2: the Bare Dyad rule). The implicit assumption is that domain-level separation equals independence.

That assumption is partially violated. Specifically:

- **macro and technical are not independent** (r = 0.73). A convergence alert combining FRED macro signals and FINRA short interest signals is not getting two independent confirmations — it is getting one confirmation and its own echo.
- **insider and technical have a lag relationship** (r = 0.51-0.58). Insider buying today tends to correlate with short interest changes 5-42 days later. This isn't surprising (insiders often act before short sellers cover), but it means these domains are leading/lagging versions of the same information at certain time horizons.

### Data Availability for Further Analysis

The `CorrelationTracker` in `mae_core/market/intelligence/correlation_tracker.py` is designed to compute live rolling Pearson correlations between signal streams, with minimum 30 observations per pair and Bonferroni correction for multiple comparisons. It stores data in memory during a run, not in a persistent cross-session file.

The lag_correlations.json file only covers 5 of MIDGE's 28 data sources. The domains with zero correlation data include: events (finnhub_earnings), sentiment (StockTwits), positioning (COT), government (congressional), and contracts (contract_award). Whether MIDGE's convergence engine is getting true independence when combining these domains is currently unknown.

**What would be needed for a complete independence audit:**
- Minimum 30 simultaneous observations per source pair (requires the daemon to run for several weeks and accumulate CorrelationTracker state)
- The CorrelationTracker would need its persistence path configured to write state to disk between sessions
- Current state: only 5 of 28 sources have measurable cross-correlations in the data

---

## Task 3: Quick Stats

### Outcomes File

| Metric | Value |
|--------|-------|
| Total records | 13,003 |
| Old-style format (return_pct field) | 22 |
| New-style format (success field, 5% threshold) | 12,981 |
| Date range | 1984-04-30 (historical replay entries) to 2026-02-27 |
| Recent data (2025-2026 only) | ~11,896 records across 14 months |

Monthly distribution of recent outcomes (showing data density):
- 2025-03: 1,797 | 2025-07: 1,533 | 2026-02: 1,297 | 2026-01: 970 | 2025-04: 1,471

### Predictions File

| Metric | Value |
|--------|-------|
| Total predictions recorded | 11,396 |
| Date range | 2026-02-08 only (single batch) |
| Primary contributing signals | congress_trade, contract_award, insider_cluster, options_flow, sec_form4 |

The predictions file is almost entirely from a single date and appears to be a mock/seed batch, not the live prediction stream. Live predictions flow through the convergence alert system and are logged separately.

### Thompson Distributions

| Metric | Value |
|--------|-------|
| Total distributions registered | 83 |
| Distributions with real learned data | **2** |
| Uniform-prior (no learning yet) | 81 |

Only 2 of 83 Thompson distributions have diverged from their 50/50 uniform starting point:

| Distribution | Mean Win Probability | Samples |
|--------------|---------------------|---------|
| `sweep_bt:GC=F:bearish` (Gold futures bearish backtests) | **53.2%** | ~6 |
| `finnhub_earnings:sideways` (Earnings in sideways markets) | **30.6%** | ~2 |

This is the most striking finding in the Thompson data: **MIDGE's Bayesian learning engine has not learned anything yet.** 81 of 83 signal distributions are still at their 50/50 prior — identical to day one. The outcome feedback loop exists in the code but the data has not flowed through it in meaningful volume.

### Signal Archive

| Metric | Value |
|--------|-------|
| Total signal files | 906 JSONL files |
| Total signal records | **230,462** |
| Date range | 1978-09-15 (COT historical) to 2027-01-25 |
| Largest domain | events: 104,863 (45.5%) |
| Second largest | insider: 73,411 (31.8%) |
| Third largest | technical: 16,663 (7.2%) |

The signal archive is large and rich. The domain distribution reflects MIDGE's data source mix: events (earnings, Form 8-K) and insider (Form 4, OpenInsider) dominate. Technical, institutional, and contracts follow.

---

## Summary and What the Numbers Tell Us

### The Payoff Ratio: Mixed News

The raw payoff of 3.34:1 is structurally positive — it exceeds the break-even requirement of 2.0:1 at 33.3% win rate. But the replay harness (which tests the convergence engine specifically) shows a 19.9% win rate, which requires 4.03:1 to break even. MIDGE's payoff of 3.34:1 is **below** what is needed at the replay win rate.

The critical unknown is whether the 19.9% replay win rate is the "true" rate or whether it is artificially low due to data quality issues identified in prior analysis (Congressional trades with stale timestamps, ungraded future-dated predictions). If the real convergence win rate is closer to 25-30%, the payoff is sufficient. If 19.9% is accurate, MIDGE needs either a higher payoff threshold or a better signal filter.

### Domain Independence: Partially Violated

MIDGE's assumption that domain separation equals independence is not fully supported. The macro and technical domains have a documented 0.73 correlation. The insider and technical domains show 0.51-0.58 correlation at various lags. Convergence alerts that stack these specific domain combinations are not getting as many independent confirmations as the engine believes.

The good news: only 5 of 28 sources have been measured. The events, sentiment, positioning, and government domains may be genuinely independent — these are the domains that produced MIDGE's strongest replay performance (events+macro+price+sentiment+technical: 66.7% win rate on small samples). Getting correlation measurements on the unchecked domains should be a near-term priority.

### Thompson Learning: Not Yet Active

The most urgent structural finding is that the Bayesian learning system — the mechanism intended to weight reliable signals higher over time — has not updated meaningfully. 81 of 83 distributions are still at the uniform 50/50 prior. This means MIDGE is not yet using the 230,462 signals in its archive to weight its confidence calculations. The outcome feedback loop is built but the data pipeline connecting archive signals to Thompson updates needs attention before MIDGE can begin genuine self-improvement.

---

*Generated by Phase 0 measurement analysis. Source files: `data/market/outcomes.jsonl` (13,003 records), `data/market/lag_correlations.json` (47 records), `data/market/thompson_distributions.json` (83 distributions), `data/midge/signals/` (906 files, 230,462 records).*
