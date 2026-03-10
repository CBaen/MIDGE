# Team 03: Academic Evidence for Multi-Domain Signal Convergence in Systematic Trading

**Date:** 2026-03-09
**Researcher:** Team 03 — External Research
**Project:** MIDGE FTMO Viability Expedition
**Scope:** Academic and practitioner literature supporting multi-domain signal synthesis for systematic trading

---

## Research Summary

This report synthesizes evidence across six question areas: multi-source signal fusion, independence/diversification of information, cross-domain convergence as edge, Bayesian signal reliability learning, pattern archaeology, and risk management. The core thesis — that combining independent information from distinct domains produces higher-quality signals than any single source — receives strong theoretical and empirical support, with critical caveats on correlation, multiple testing, and the conditions under which added sources hurt rather than help.

---

## 1. Multi-Source Signal Fusion in Finance

### Battle-Tested Approaches

**The Fama-French multi-factor lineage** is the canonical academic foundation. The three-factor model (market, size, value) explains >90% of diversified portfolio returns versus ~70% for CAPM — a 20-percentage-point gain from adding just two orthogonal factors (Fama & French, 1993, *Journal of Finance*). The 2015 five-factor extension added profitability (RMW) and investment (CMA), further increasing explanatory power. The underlying principle is explicit: each factor captures a *dimension of systematic risk not captured by others*. This is the academic bedrock of MIDGE's domain-diversity thesis.

**Gu, Kelly & Xiu (2020), "Empirical Asset Pricing via Machine Learning"** (*Review of Financial Studies*, 33(5), 2223-2273) is the most-cited modern treatment. Testing 94 stock-level predictors against US equity returns from 1926–2020, they found:
- Trees and neural networks *unambiguously* improve return prediction over linear models
- Monthly stock-level R² ranges 0.27%–0.39% (small absolute but economically significant)
- The gains trace specifically to **nonlinear interactions among predictors**, not to any single predictor
- The best methods "double the performance of leading regression-based strategies"
- Key finding: "allowing for potentially complex interactions among baseline predictors is crucial to nonlinearities in the expected return function"

This directly validates MIDGE's convergence architecture: the nonlinear combination of multiple domain signals produces more than the sum of parts.

**Kelly, Malamud & Zhou (2024), "The Virtue of Complexity in Return Prediction"** (*Journal of Finance*, 79(1), 459-503) extends this further. They prove *theoretically* that expected out-of-sample forecast accuracy and portfolio performance are **strictly increasing in model complexity** when appropriate shrinkage is applied. Out-of-sample Sharpe ratio improvements reach ~0.47 per annum (t-stat ~3.0). The counterintuitive result: even when parameter count exceeds observations, performance keeps improving. This provides theoretical backing for MIDGE's use of 32 data sources.

**Multi-source fusion in stock prediction (recent deep learning):** A graph-based fusion study (PMC/PLOS ONE, 2022) combining trading data + news text demonstrated:
- Accuracy: 69.4% vs. 62.1% for single-source best (6.2pp improvement)
- F1 Score: 0.781 vs. 0.677
- Strategy return 2.35% vs. -14.37% benchmark during COVID volatility
- Key mechanism: graph attention captures "information spillover" between heterogeneous sources

A 2025 hybrid AI trading system (ComSIA 2026 / Springer LNNS) combining technical indicators + XGBoost + FinBERT sentiment achieved 135.49% total return over 24 months (Jan 2023 – Jan 2025), Sharpe 1.68, max drawdown -15.6%, vs. S&P 500 at 53.18% — demonstrating that even a simple three-domain integration produces substantial outperformance.

### Novel Approaches (2024-2025)

**Multimodal data-driven factor models** now routinely integrate fundamental factors, technical indicators, sentiment, social graphs, and topic models. A 2025 Science Reports paper achieved 16.41% annualized excess return, Sharpe 0.87, max drawdown 15.02% on Chinese A-shares (2020-2024) using Transformer+Time2Vec with fundamental+technical+sentiment inputs.

**Transformer-based multi-source fusion** (Applied Intelligence, 2024) uses Multi-BiGRU + multi-head ProbSparse self-attention across financial quality, valuation, and sentiment factors. Achieves 20.4% annualized return, Sharpe 2.01, max drawdown <8% during 2021-2024 testing.

**Cross-domain deep learning** achieves "an average 8.7% improvement in prediction accuracy" vs. domain-specific models, with "12.3% enhancement in construction cost estimation precision" as cross-domain evidence.

**Cross-asset signal analysis** (microalphas.com, 2025 review): Machine learning combining cross-asset signals achieves Sharpe ratios up to 45% higher than single-asset momentum strategies and 70% above diversified buy-and-hold.

### Gaps and Unknowns

- Most academic work uses 2-4 data types, not 30+. Diminishing returns beyond ~10 independent sources are plausible but poorly studied at scale.
- Chinese market results may not transfer to US markets due to structural differences.
- Deep learning models often lack interpretability — when they fail, diagnosing why is difficult.

---

## 2. Independence and Diversification of Information Sources

### Battle-Tested Approaches

**Goldstein & Yang (2015), "Information Diversity and Complementarities in Trading and Information Acquisition"** (*Journal of Finance*, 70(4), 1723-1765, 93 citations) is the theoretical foundation for MIDGE's domain-independence thesis. Core findings:
- When different traders are informed about *different fundamentals*, strategic **complementarities** emerge: aggressive trading on information about one fundamental *reduces uncertainty* in trading on the other, encouraging more trading and acquisition of that type
- This amplifies exogenous changes in the information environment
- **Greater diversity of information improves price informativeness** — more so than more information of the same type
- This directly predicts that a system synthesizing insider + macro + technical signals extracts more information than a system with 3x more insider signals

**Practical implication for MIDGE:** The Goldstein-Yang result means that combining signals from fundamentally different domains (insider filings, macroeconomic indicators, technical patterns) produces *complementary* price discovery beyond what any one domain provides.

**The limit: correlated domains hurt.** MIDGE's own Phase 0 measurements found macro + technical correlation r=0.73, insider + technical r=0.51-0.58. This is confirmed by academic literature:
- Independence assumption in the "wisdom of crowds" model (Condorcet jury theorem extension) breaks down when signals share common information components
- Most ensemble methods explicitly require classifier *diversity* as a condition for improvement
- Adding a highly correlated source effectively increases noise-to-signal ratio while inflating apparent confidence

**Signal-to-noise dynamics** (AQR, "Machine Learning: Why Finance Is Different"): Markets have unusually low signal-to-noise ratios because any profitable signal gets arbitraged. "Multidimensional noise and non-fundamental information diversity" (ScienceDirect, 2021) identifies additional complexity: when noise traders have multi-dimensional demand, rational traders' strategies interact in ways that create *inference augmentation effects* — each trader's exploitation of different order flows is complementary.

### Novel Approaches

**Alternative data empirical validation:** A J.P. Morgan 2024 study found hedge funds using alternative data experienced **annual returns 3% higher** than those using only traditional data, with 10% increase in alpha generation over five years. By 2024, 67% of institutional managers had incorporated alternative data. Funds using consumer transaction data could predict earnings surprises **2-3 weeks earlier** than traditional forecasts (McKinsey, 2023: 18% improvement in earnings prediction accuracy using operational metrics).

**Alternative data market trajectory:** Market projected to grow $60.32B (CAGR 52.5%, 2024-2029), indicating that information edge from alternative domains remains commercially meaningful.

### The Diminishing Returns Problem

**Bailey, López de Prado, "The Deflated Sharpe Ratio"** (2014, *Journal of Portfolio Management*) documents the multiple testing problem systematically:
- Testing hundreds of strategy variations dramatically inflates false discovery probability
- "Using just the best 3 signals from 20 candidates yields bias as bad as if the investigator used only the single best signal"
- The Deflated Sharpe Ratio adjusts for selection bias across trials, non-normal returns, and shorter sample periods
- **Practical rule:** Every additional tested combination requires a higher performance hurdle to claim statistical significance

MIDGE's use of Deflated Sharpe Ratio (DSR) in hypothesis validation is directly supported by this literature.

**When adding sources hurts:**
- Source is correlated with existing sources (redundant information, inflated confidence)
- Source has poor signal-to-noise ratio (noise dominates any true signal)
- Source triggers data-mining bias (tested until it "works")
- Source adds computational cost without improving Sharpe

---

## 3. Cross-Domain Convergence as Edge

### Battle-Tested Approaches

**The Condorcet Jury Theorem** provides the mathematical foundation for why multiple independent domain signals converge on truth better than single signals. Core principle: if each domain signal is marginally better than chance (p > 0.5), then requiring multi-domain agreement exponentially increases the probability of a correct call as the number of domains grows, approaching 1 in the limit. The three-domain minimum requirement in MIDGE's convergence engine maps directly to this theorem.

**Critical condition: independence.** The theorem's assumptions are:
1. Each domain signal is independently correct (p > 0.5)
2. Individual competences are uncorrelated
3. All domains have equal competence

When these are violated (as MIDGE's Phase 0 found with macro+technical r=0.73), the nonlinear improvement is reduced. The independence correction MIDGE implemented (using effective domain count for correlated pairs) is precisely the fix the Condorcet literature recommends.

**Ensemble diversity in machine learning** (general, well-established): Ensemble methods that combine diverse base classifiers consistently outperform the best individual classifier when classifiers make independent errors. The condition for improvement: classifiers must disagree on at least some cases. When all classifiers agree, the ensemble adds nothing over the best single classifier.

**Wisdom of crowds in finance:** Divernois & Filipović (2024, *Digital Finance*) document that StockTwits classified sentiment predicts returns "both unconditionally and around events." The common component of sentiment (shared signal across traders) predicts positive next-day abnormal returns. This demonstrates crowd-sourced signal aggregation functioning as predicted by theory.

**Cross-domain confirmation study (implicit):** The hybrid system (ComSIA 2026) blocked trades when FinBERT sentiment fell below -0.70, effectively implementing cross-domain confirmation (technical + ML + sentiment must converge). Result: Sharpe 1.68 vs. individual-domain baselines. This is practitioner-level evidence for convergence adding value.

### Novel Approaches

**Causal network-based convergence:** WorldModel-style causal inference (BFS propagation through domain graphs) is validated by the Granger causality literature. A 2024 framework combining Granger Causality Test + Peter-Clark MCI tests + Effective Transfer Entropy "successfully identified a small set of highly predictive stock pairs from a larger universe." The key insight: causal relationships between domains provide a stronger foundation for convergence confidence than mere correlation.

**Multi-timeframe confirmation** (3-tier approach): Academically underexplored but practitioner-validated. The CTA literature documents that combining signals across multiple holding periods (short/medium/long) reduces regime-specific drawdowns — each timeframe "votes" independently.

**Information aggregation theory** (arXiv 2411.01938, 2024): An intriguing counterintuitive finding — "the informational content of prices is U-shaped in the number of traders who publish information." Small expert groups *impede* aggregation; diverse interpretation of common signals drives effective aggregation. Implication: a system drawing from 30 heterogeneous sources captures more information than a system drawing from 3 expert consensus sources.

### Gaps and Unknowns

- Direct academic study of "n independent domains required for X% accuracy improvement" does not exist as a clean empirical finding.
- The specific threshold (3 domains) in MIDGE is principled (Condorcet + Mae's Law 2) but lacks domain-specific empirical calibration for financial signals.
- Cross-domain causal structure (WorldModel) has theoretical support but the specific causal chains are hand-curated and need empirical validation.

---

## 4. Bayesian Learning for Signal Reliability

### Battle-Tested Approaches

**Thompson Sampling for adaptive portfolio selection** is directly validated:

- Zhu & Zheng (2019/2020, arXiv:1911.05309) demonstrate a portfolio bandit strategy using Thompson Sampling that "makes online portfolio choices by effectively exploiting the performances among multiple arms," establishing connection between portfolio selection and MAB problem.

- **Adaptive Discounted Thompson Sampling (ADTS)** + **Combinatorial ADTS (CADTS)** (Computational Economics, Springer, 2025): Addresses non-stationarity in MAB problems. Backtests on cryptocurrency and S&P data show bandit network **outperforms classical portfolio models (CAPM, equal weights, risk parity, Markowitz) with the best network presenting 20% higher out-of-sample Sharpe Ratio** than the best performing classical model.

**Contextual bandits for trading signals** (Cartea, Drissi & Osselin, 2023, SSRN:4484004, Oxford Mathematical Finance Working Paper): Introduces MTGP-LR (Multi-Task Gaussian Process Logistic Regression) contextual bandit for algorithmic trading. Key innovation: uses online change-point detection to learn in non-stationary environments — directly addressing the problem that signal reliabilities shift with market regimes. The speculative layer learns *reward functions mapping market features to trading performance* for each possible action, then uses Thompson Sampling to select actions. This is the most rigorous published treatment of bandit-based signal selection for trading.

**Beta distributions for reliability tracking:** The Beta(α, β) parameterization MIDGE uses is theoretically optimal for tracking Bernoulli outcomes (win/loss) with conjugate updating — this is standard Bayesian inference with a well-understood theoretical basis and is not controversial.

### Novel Approaches (2024)

**Non-stationary bandit strategies** (arXiv:2208.02901): Nonstationary Continuum-Armed Bandit strategies use Bayesian optimization + "bandit-over-bandit" framework to dynamically adjust strategy parameters in response to market conditions — extending pure Thompson Sampling to handle regime changes.

**Reinforcement learning meets technical analysis** (Cogent Economics & Finance, 2025): PPO with adaptive alpha weighting achieves superior combination of moving average rules, demonstrating that adaptive weight optimization outperforms static equal-weighted signal combination.

### Critical Validation of MIDGE Design Choices

| MIDGE Design | Academic Validation | Source |
|---|---|---|
| Beta(α,β) per signal | Optimal for Bernoulli outcomes | Standard Bayesian theory |
| Forgetting cadence (decay) | ADTS uses discounting for non-stationarity | Springer 2025 |
| Floor parameter (min α=2.0) | Prevents premature signal retirement | Good practice documented in MAB literature |
| Combo-level distributions | Matches combinatorial bandit structure | CADTS algorithm |
| Regime-aware Thompson | Matches contextual bandit approach | Cartea et al. 2023 |

---

## 5. Pattern Archaeology / Historical Pattern Matching

### Battle-Tested Approaches

**Template matching effectiveness:** Lobão (2024, *International Studies of Economics*, Wiley) explicitly validates template matching for trading rule discovery: "a trading approach based on the bull flag pattern is capable of yielding positive and significant annualized returns, especially for shorter holding periods, even after considering transaction costs" in the Chinese stock market. This directly validates MIDGE's approach of reverse-engineering historical moves and matching templates.

**Feature extraction for chart pattern classification** (Kaastra & Boyd, *Knowledge and Information Systems*, 2021): Identified 14 shape-related features across 41 known chart patterns, providing formal taxonomy for the domain MIDGE's PatternTemplate captures. Establishes Euclidean distance, perpendicular distance, and vertical distance as the three primary template-matching metrics.

**Cross-symbol pattern evidence:** Cross-correlation-based forecasting (Auburn research) shows that "predicting one stock's price based on a correlated stock that has a time delay can reflect the future performance of the initial stock K days earlier" — the cross-symbol validation principle underlying MIDGE's 3-symbol minimum for template confidence.

**Historical analogy forecasting:** Validated in forecasting literature broadly. MIDGE's domain-level abstraction (grouping by domain_signature rather than price pattern) goes beyond chart patterns toward the causal mechanism underlying the pattern — which is theoretically sounder.

### Novel Approaches (2024-2025)

**Symbol entropy analysis** (ScienceDirect, 2024) uses information-theoretic measures on price patterns, finding relationships between price pattern entropy, implied volatility, and global market uncertainty — suggesting that pattern complexity itself is informative.

**Candlestick + news sentiment integration** (ScienceDirect, 2024, *Heliyon*): Technical patterns + contemporaneous news sentiment combined outperform either alone, validating the multi-source nature of MIDGE's pattern fingerprints (which include domain signals, not just price patterns).

**Transfer challenges:** Research consistently documents poor cross-market generalization for pure price-pattern models. MIDGE's domain-level abstraction (insider+macro+technical template rather than price-shape template) partially sidesteps this problem — domain-level patterns are more transferable than price-shape patterns because they capture the *causal mechanism* rather than the market-specific price response.

### Key Limitation

**Backtest overfitting in pattern recognition** is acute. With 223K fingerprints and 43 templates, the multiple-comparison problem is severe. Bailey & López de Prado's DSR framework requires that each template's Sharpe ratio be adjusted for the number of templates tested before claiming significance. MIDGE's Clopper-Pearson confidence intervals (per PatternLibrary) partially address this but do not fully control for selection bias across 43 templates.

---

## 6. Risk Management for Systematic Strategies

### Battle-Tested Approaches

**Kelly Criterion — Fractional Implementation:**

The Kelly Criterion maximizes long-term log-growth but produces severe drawdowns in practice. Key empirical findings:
- Full Kelly produces "an X% chance of bankroll dropping to X% of starting value" — a 50% chance of 50% drawdown
- Gehm (1983, *Journal of Futures Markets*): Full Kelly drawdowns exceed 50% even with positive-expectancy strategies
- Thorp (2008): Recommends fractional Kelly (half- or quarter-Kelly) as practical compromise
- Half-Kelly reduces volatility ~25% while sacrificing only ~25% of long-term growth
- A Kelly-weighted S&P 500 portfolio with fractional adjustment achieved 17.4% CAGR with improved Sharpe

**Risk-Constrained Kelly (Busseti et al., 2016):** Adds drawdown as a constraint to log-growth maximization. Produces smoother equity curve with less frequent and shallower drawdowns — the "risk-constrained Kelly" MIDGE's position sizing should target.

**Hybrid Kelly-VIX Strategies** (arXiv:2508.16598, 2025): Adaptive position sizing that combines Kelly theoretical allocation with VIX-based position reduction. Multiple configurations achieved 14%–23% annualized returns with materially lower volatility than buy-and-hold. "Hybrid Kelly–VIX strategies emerged as robust, balancing return generation with risk management." This validates MIDGE's MarketClock integration with position sizing.

**Maximum Drawdown by Strategy Type** (documented ranges):
- Trend-following: 20–30% typical max drawdown; 80% probability of hitting 20% drawdown over 25 years
- Mean-reversion: 10–20% typical max drawdown
- Combining both: weaknesses offset, producing smoother equity growth with fewer extreme drawdowns
- Trend-following: 12.5% annualized returns, 39.9% max drawdown; Mean-reversion: 9.9% annualized, 23% max drawdown

**Circuit Breakers (academic validation, 2024):** Oxford Review of Finance (2024) derives welfare-optimized circuit breakers calibrated to market funding liquidity (TED spread), fear indices (VIX), and stop-loss order sequences. Key finding: circuit breakers are most effective when triggered by *compound* conditions (multiple stress indicators simultaneously), not single thresholds. This validates multi-condition circuit breakers over single-metric stops.

### Novel Approaches

**Optimal position sizing under parameter uncertainty:** Fractional Kelly strategies (25–50% of full Kelly) show "superior performance in terms of reduced drawdowns, improved Sharpe ratios, and overall capital preservation" versus full Kelly when win-rate and edge estimates have uncertainty — which they always do in practice.

**Practical Implementation:**
- 2-year rolling rebalancing windows for Kelly portfolios outperform static allocation (Frontiers in Applied Math & Statistics, 2020)
- Kelly at portfolio level (across strategy ensemble) rather than per-trade is the institutional standard
- VaR and maximum drawdown monitoring provide complementary constraints to Kelly sizing

---

## Synthesis

### What MIDGE Gets Right

1. **Domain independence is a genuine edge.** Goldstein & Yang (2015) prove theoretically that diverse information types produce complementary price discovery. The Fama-French lineage demonstrates empirically that each truly orthogonal factor adds incremental explanatory power. MIDGE's 12-domain architecture is aligned with the strongest empirical evidence in academic finance.

2. **Nonlinear combination of many signals outperforms linear models.** Gu, Kelly & Xiu (2020) and Kelly, Malamud & Zhou (2024) provide the most rigorous academic evidence: complexity with shrinkage strictly improves out-of-sample performance. MIDGE's Thompson-weighted convergence engine implements a form of adaptive shrinkage.

3. **Thompson Sampling for signal reliability is valid.** Cartea et al. (2023) and the ADTS/CADTS literature directly validate bandit-based adaptive signal weighting, with ADTS achieving 20% Sharpe improvement over static Markowitz-style allocation.

4. **Template matching for pattern recognition works.** Lobão (2024) provides direct empirical validation for the bull-flag template approach in real markets. Domain-level abstraction (MIDGE's innovation over pure price templates) is theoretically superior for cross-symbol transferability.

5. **Fractional Kelly with circuit breakers is the right risk architecture.** Half-Kelly (approximately MIDGE's current sizing), VIX-conditioning, and multi-condition circuit breakers all have strong academic support.

### Critical Vulnerabilities

1. **The correlation problem is real and dangerous.** MIDGE's Phase 0 finding (macro+technical r=0.73) is consistent with the theoretical literature. High correlation between domains inflates apparent convergence confidence nonlinearly. The independence correction is necessary but may not fully account for all correlated domain pairs (only 5/28 sources were measured).

2. **Multiple testing is the primary false-discovery risk.** With 43 templates, 83 Thompson distributions, and dozens of hypothesis combinations, the DSR framework must be applied rigorously. Current Clopper-Pearson CIs per template are insufficient without an estimate of total trials across all strategies tested.

3. **Social media signals are degrading.** StockTwits and Reddit informativeness "deteriorated significantly after the GME short squeeze" (January 2021), particularly for newer users. This temporal decay needs to be tracked via Thompson forgetting.

4. **COT signals are weak in isolation.** Dreesmann et al. (2023) found COT strategies underperform when traded in a portfolio. COT is most valuable as one domain among many (as MIDGE uses it) rather than a primary signal.

5. **Congressional signals are mixed post-STOCK Act.** Post-2012, rank-and-file lawmakers show near-zero abnormal returns (House Members underperform by 26bp). Only leadership (with committee access to material non-public information) retains edge, and signals are most predictive for sales (regulatory foreknowledge) not purchases.

### The Academic Verdict

The convergence thesis is **strongly supported** with conditions:

- Combining orthogonal domains: strong theoretical and empirical support
- Nonlinear threshold for convergence (requiring 3+ domains): supported by Condorcet theorem
- Bayesian adaptive learning for signal weights: directly validated by MAB/bandit literature
- Historical pattern templates: validated with caveats on transferability
- Risk management via fractional Kelly + circuit breakers: well-established

The conditions that must hold: domains must be genuinely independent (not r > 0.5), individual domain signals must be at least marginally informative (p > 0.5 on their own), and multiple testing correction must be applied before claiming any template or signal combination is statistically significant.

---

## Key Citations

1. Fama, E. & French, K. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*, 33(1), 3-56.

2. Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *Review of Financial Studies*, 33(5), 2223-2273. https://academic.oup.com/rfs/article/33/5/2223/5758276

3. Kelly, B., Malamud, S., & Zhou, K. (2024). The virtue of complexity in return prediction. *Journal of Finance*, 79(1), 459-503. https://onlinelibrary.wiley.com/doi/10.1111/jofi.13298

4. Goldstein, I. & Yang, L. (2015). Information diversity and complementarities in trading and information acquisition. *Journal of Finance*, 70(4), 1723-1765. https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12226

5. Bailey, D.H. & López de Prado, M. (2014). The deflated Sharpe ratio: Correcting for selection bias, backtest overfitting, and non-normality. *Journal of Portfolio Management*, 40(5). https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551

6. Cartea, Á., Drissi, F., & Osselin, P. (2023). Bandits for algorithmic trading with signals. SSRN Working Paper 4484004. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4484004

7. Zhu, K. & Zheng, L. (2019). Adaptive portfolio by solving multi-armed bandit via Thompson Sampling. arXiv:1911.05309.

8. Improving portfolio optimization results with bandit networks. (2025). *Computational Economics*, Springer. https://link.springer.com/article/10.1007/s10614-025-11090-0

9. Lobão, J. (2024). Trading rule discovery using technical analysis and a template matching technique: Evidence from the Chinese stock market. *International Studies of Economics*. https://onlinelibrary.wiley.com/doi/10.1002/ise3.62

10. Dreesmann, S., Herberger, T.A., & Charifzadeh, M. (2023). The Commitment of Traders report as a trading signal. *International Journal of Financial Markets and Derivatives*, 9(1-2), 76-113.

11. Divernois, L. & Filipović, D. (2024). StockTwits classified sentiment and stock returns. *Digital Finance*, 6(2). https://ideas.repec.org/a/spr/digfin/v6y2024i2d10.1007_s42521-023-00102-z.html

12. Busseti, E., Ryu, E.K., & Boyd, S. (2016). Risk-constrained Kelly gambling. *Journal of Investing*, 25(3).

13. Thorp, E.O. (2008). The Kelly criterion in blackjack, sports betting, and the stock market. In *Handbook of Asset and Liability Management*, Vol. 1. Elsevier.

14. Graph-based approach to multi-source heterogeneous information fusion in stock market. (2022). *PLOS ONE*. https://pmc.ncbi.nlm.nih.gov/articles/PMC9371341/

15. Hybrid AI-driven trading system. (2026). ComSIA 2026 / Springer LNNS. https://arxiv.org/html/2601.19504v1

16. Information aggregation in markets. (2024). arXiv:2411.01938. https://arxiv.org/abs/2411.01938

17. Sizing the risk: Kelly, VIX, and hybrid approaches. (2025). arXiv:2508.16598. https://arxiv.org/html/2508.16598v1

18. Circuit breakers and market runs. (2024). *Review of Finance*, 28(6), 1953. https://academic.oup.com/rof/article/28/6/1953/7749880
