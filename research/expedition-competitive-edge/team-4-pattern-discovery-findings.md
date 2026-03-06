# Team 4 Findings: Pattern Discovery Methods
## Date: 2026-03-05
## Researcher: Team Member 4

---

### Contextual Grounding

MIDGE's current pattern discovery stack consists of:
- **Pearson lag-correlation** via `LagCorrelationAnalyzer` (cross-correlates source pairs at 1-90 day lags, Fisher Z p-value filter, `lag_correlations.json` holds 50+ bivariate findings)
- **Pearson rolling correlation** via `CorrelationTracker` (same-time z-score anomaly detection across source pairs, 30-observation window)
- **Convergence detection** via `ConvergenceAlerter` (Thompson-weighted geometric mean confidence when 3+ independent domains align, combo-level Beta distributions)
- **Template matching** via `PatternWatcher` (domain-level fingerprints abstracted from historical moves, independence check <30% domain overlap, stacking tiers)
- **Hypothesis loop** (RSI Layers 1-3): lag findings → hypothesis generator → deflated Sharpe ratio validator → Thompson feedback

The key gap: all current correlation methods are **bivariate Pearson** (linear, same-unit, noise-susceptible) and detect **concurrent or lagged linear co-movement**. They miss nonlinear dependencies, directed causal flow, conditional dependencies given a third domain, and tail-specific dependencies that only emerge during extreme events. These are precisely the gaps that siloed enterprise systems also fail to close — meaning methods that fill them give a genuine competitive edge.

---

### Battle-Tested Approaches

#### 1. Granger Causality / Sparse VAR (statsmodels)

- **What:** Granger causality tests whether domain A's signal history has statistically significant predictive power for domain B's future signal, above and beyond B's own history, using vector autoregression (VAR) with F-test or chi-squared test.
- **Evidence:** Implemented in `statsmodels.tsa.statespace.VAR` and `grangercausalitytests`, maintained for 15+ years, used in thousands of academic finance papers. Applied to Fama-French factors, macroeconomic indicators, and inter-market relationships at scale. The Toda-Yamamoto procedure (recommended for non-stationary financial series) is widely validated.
- **Source:** statsmodels documentation (accessed 2026-03-05): https://www.statsmodels.org/dev/generated/statsmodels.tsa.vector_ar.var_model.VARResults.test_causality.html; Academic review: https://www.researchgate.net/publication/356351292_Granger_Causality_A_Review_and_Recent_Advances
- **Fits our case because:** MIDGE already has `lag_correlations.json` with 50+ bivariate Pearson lag findings. Granger causality is the direct upgrade — it tests whether `finra_short` today causally predicts `sec_form4` in 5 days **better than past sec_form4 alone**. This replaces bivariate Pearson with a directed, conditional test. Directly feeds the HypothesisGenerator: a Granger-significant pair at lag L is a causal story, not just correlation.
- **Tradeoffs:** Standard Granger is **linear only** — misses nonlinear dependencies (common in financial data). Requires stationarity pre-processing (differencing, ADF test). With 28 sources, the full pairwise matrix is 28×27 = 756 pairs × 90 lags = significant but tractable compute on CPU (minutes per analysis run). Sparse VAR (lasso-regularized) handles the dimensionality problem but requires the `pysr3` or `sklearn` lasso pipeline.

---

#### 2. Rolling / Time-Varying Granger Causality

- **What:** Run Granger causality tests in a rolling window (e.g., 60-day sliding window) to detect when causal relationships activate and deactivate — capturing regime-conditional cross-domain dependencies.
- **Evidence:** Validated in financial literature: geopolitical risk → crude oil markets study (2025, ScienceDirect), with rolling window methods (Forward Expanding, Rolling, Recursive Evolving) now standard. Forward/rolling windows documented in Stata symposium proceedings and multiple academic finance papers. Python implementation via rolling `statsmodels` calls.
- **Source:** Rolling Granger methodological paper: https://www.stata.com/symposiums/economics21/slides/Econ21_Baum.pdf; Applied study: https://www.sciencedirect.com/science/article/abs/pii/S0275531925002508 (accessed 2026-03-05)
- **Fits our case because:** MIDGE's regime_classifier already detects market regimes. Rolling Granger would reveal which cross-domain causal links are regime-conditional: "insider→price causality activates in bull regimes but not volatile regimes." This is exactly the kind of context-conditional pattern stacking MIDGE needs for the 95%+ confidence tier.
- **Tradeoffs:** Computationally heavier than static Granger — 60-day rolling window over 400 days = ~340 windows per pair × 756 pairs. Still feasible on i9/64GB. Statistical power drops with smaller windows. Causality stability check required (spurious non-stationarity produces false causal activation).

---

#### 3. Minimum Spanning Tree (MST) / Information Filtering Network

- **What:** Build a correlation network across all 28+ signal sources, then filter to the MST (the network skeleton that preserves most information with fewest edges). MST reveals which domains are structurally central vs. peripheral — and when a peripheral node suddenly connects to the core, that's a signal.
- **Evidence:** Pioneered by Mantegna (1999) for financial cross-correlations, now standard in quantitative network finance. Implemented in `mlfinlab` (Hudson & Thames) and `scipy.sparse.csgraph.minimum_spanning_tree`. Widely used for portfolio diversification and cross-asset signal filtering. PMFG (Planar Maximally Filtered Graph) retains 3(n-2) edges vs. n-1 for MST, capturing richer structure. Applied to crypto markets 2021-2024 (ScienceDirect, 2025).
- **Source:** mlfinlab MST documentation: https://www.mlfinlab.com/en/latest/networks/pmfg.html; Hudson & Thames article (2020, updated 2023); Crypto application: https://www.sciencedirect.com/science/article/abs/pii/S0378437125001256 (accessed 2026-03-05)
- **Fits our case because:** MIDGE has 28 sources across 11 domains. MST on the 28×28 correlation matrix (computed from `CorrelationTracker` data) would reveal which sources cluster structurally. New edges forming in the MST as a market event unfolds = structural cross-domain coupling = a new convergence signal that the current pairwise approach misses because it doesn't model the network topology.
- **Tradeoffs:** MST is inherently linear (Pearson-based). Known instability: small correlation changes produce large MST topology changes. PMFG is more stable but slightly more complex to implement. Does not give directionality (undirected graph). Best used as a meta-layer on top of existing CorrelationTracker, not a replacement.

---

#### 4. Random Matrix Theory (RMT) Denoising

- **What:** Apply Marchenko-Pastur law to the empirical correlation matrix of all signal sources to separate genuine cross-domain structure (eigenvalues above noise threshold λ+) from sampling noise (eigenvalues within the noise band). Only correlations in the signal-carrying eigenspace are acted on.
- **Evidence:** Pioneered by Laloux et al. (2000) for stock return correlations, extensively validated. The skfolio Python library (2024-2025) implements `DenoiseCovariance` and `DetoneCovariance` estimators. Applied to cryptocurrency clusters 2021-2024. The portfolio optimizer blog (https://portfoliooptimizer.io) provides detailed Marchenko-Pastur denoising walkthroughs.
- **Source:** RMT denoising blog (accessed 2026-03-05): https://portfoliooptimizer.io/blog/correlation-matrices-denoising-results-from-random-matrix-theory/; skfolio library: https://skfolio.org/; Crypto application 2025: https://www.sciencedirect.com/science/article/abs/pii/S0378437125001256
- **Fits our case because:** MIDGE's `CorrelationTracker` computes pairwise Pearson on a 30-observation window — highly susceptible to noise given 28 sources and limited history. RMT denoising applied to the correlation matrix before anomaly detection would dramatically reduce false-positive "unusual correlation" alerts. This directly improves the precision of existing systems without changing their architecture.
- **Tradeoffs:** Requires enough observations (rule of thumb: T >> N, where T is time periods and N is signal sources). With 28 sources and 30-day window, T/N = ~1.07 — at the marginal edge. Needs longer history (90+ days). Does not add directionality. Purely a noise-filtering layer.

---

### Novel Approaches

#### 5. Transfer Entropy (infomeasure / IDTxl)

- **What:** Transfer entropy measures the directed reduction in uncertainty about domain B's future, given domain A's history, beyond what B's own history provides — but without the linearity assumption of Granger causality. Detects nonlinear information flow between domains.
- **Why it's interesting:** Granger causality is a special case of transfer entropy under Gaussianity. Transfer entropy captures nonlinear causal channels: "when VIX is above 25, Congressional trades causally precede unusual hiring blitzes" — a relationship invisible to Pearson or even Granger. It also handles ordinal/ranked signals naturally, which is how MIDGE's signals actually arrive (strength 0-1, not raw prices).
- **Evidence:** `infomeasure` (Büth, Acharya, Zanin, Scientific Reports 2025) processes time series of 10^5 elements in under a minute on standard CPU hardware. Includes KSG (k-nearest neighbor), kernel, and ordinal/symbolic estimators. IDTxl (Journal of Open Source Software 2018, maintained through 2025) provides multivariate TE with parallel CPU engines. RTransferEntropy (R, ScienceDirect 2019) applied specifically to financial assets with directional market influence detection. ordpy Python package implements permutation-based TE with minimal overhead.
- **Source:** infomeasure: https://www.nature.com/articles/s41598-025-14053-5 (Scientific Reports, 2025); IDTxl: https://github.com/pwollstadt/IDTxl (accessed 2026-03-05); ordpy: https://pypi.org/project/ordpy/ (accessed 2026-03-05)
- **Fits our case because:** MIDGE's `LagCorrelationAnalyzer` currently uses Pearson cross-correlation at lags 1-90 days. Replacing or augmenting with symbolic transfer entropy (ordpy) at the same lags would detect which domain pairs have **directed nonlinear information flow** — the causal direction that Pearson cannot distinguish. The ordinal/symbolic approach is particularly well-suited to MIDGE's strength-normalized signals (0-1 values that map naturally to ordinal bins).
- **Risks:** TE estimation requires more data than Pearson correlation for equivalent statistical power. With 28 sources and limited daily observations, false discovery rate must be controlled (Bonferroni or FDR — which MIDGE already applies in the hypothesis validator). Multivariate TE (conditioning on all other domains) is computationally expensive; bivariate TE is tractable.

---

#### 6. PCMCI+ (Tigramite) — Conditional Causal Discovery in Time Series

- **What:** PCMCI+ (Peter-Clark Momentary Conditional Independence, extended) discovers the full causal graph across multiple time series simultaneously by testing whether A at time t causally influences B at time t+lag, **conditioned on all other domains**. This removes spurious correlations caused by common drivers (e.g., both insider buys and congressional trades rising because a sector is hot, not because one causes the other).
- **Why it's interesting:** MIDGE's current cross-domain analysis is pairwise. PCMCI+ answers: "Does insider_cluster at t=0 cause price at t=22 days, after controlling for the fact that macro and sentiment were also elevated?" This eliminates false convergence signals where multiple domains are driven by a shared unobserved factor (like a broad bull market).
- **Evidence:** Tigramite v5.2 on PyPI, CPU-only, requires Python ≥3.10 (MIDGE uses Python 3.14). ParCorr test works for <200 data points; CMIknn for nonlinear cases. Benchmarked as comparable to PCMCI performance; CD-NOTS (2024) consistently outperforms PCMCI but is more recent and less mature. Applied to Fama-French factors and S&P 500 stock returns in financial case study (arxiv.org/abs/2312.17375v2).
- **Source:** Tigramite GitHub: https://github.com/jakobrunge/tigramite (accessed 2026-03-05); CD-NOTS financial application: https://arxiv.org/html/2312.17375v2 (accessed 2026-03-05); PyPI: https://pypi.org/project/tigramite/
- **Fits our case because:** The HypothesisGenerator currently takes pairwise lag findings and adds a causal story. PCMCI+ would generate the causal graph directly — discovering not just "A correlates with B at lag 7" but "A causally influences B at lag 7, conditional on C and D" — producing richer, more defensible hypotheses with fewer spurious positives. This directly upgrades RSI Layer 2's input quality.
- **Risks:** Nonstationarity is a known problem for PCMCI (financial time series violate stationarity). CD-NOTS handles this better but is less mature. Maximum lag parameter must be set carefully — too large = exponential condition set size. With 28 sources, computing the full causal graph at lags 1-30 days is feasible but may take minutes to tens of minutes per run (cadenced, not real-time).

---

#### 7. FP-Growth Frequent Pattern Mining on Domain Co-occurrence

- **What:** Treat each convergence alert or signal firing event as a "transaction" (basket of domains present). Run FP-Growth frequent itemset mining to discover which domain combinations co-occur above a support threshold (e.g., appear together in 3%+ of events), then extract association rules: "insider + macro → contracts fires within 5 days in 72% of cases."
- **Why it's interesting:** MIDGE's current convergence detection requires simultaneous co-occurrence within the same time window. FP-Growth would discover **sequential domain patterns** — which domains tend to appear as precursors to high-confidence convergence events — without requiring a fixed window. It treats the signal archive as a transaction database, something no current MIDGE component does.
- **Evidence:** FP-Growth is the fastest frequent itemset algorithm (2x-3x faster than Apriori on dense datasets, one database scan, divide-and-conquer). Implemented in mlxtend (Python, pip-installable), scikit-learn-compatible. Conference paper 2024 (ACM ICSLT) confirms consistent performance superiority. Applied to co-occurrence detection in fraud detection and financial anomaly contexts.
- **Source:** FP-Growth overview: https://towardsdatascience.com/fp-growth-frequent-pattern-generation-in-data-mining-with-python-implementation-244e561ab1c3 (accessed 2026-03-05); ACM conference 2024: https://dl.acm.org/doi/full/10.1145/3678610.3678618
- **Fits our case because:** The signal archive holds 900+ JSONL files with daily signal observations across 28 sources. Discretizing each day's active signals as a "basket" and running FP-Growth would surface multi-domain patterns at specific temporal orderings that the current pairwise lag analysis misses. The result feeds directly into the HypothesisGenerator as new candidate pairs/clusters with empirical support thresholds.
- **Risks:** Interpretation requires domain knowledge to distinguish causal from spurious co-occurrence. Support threshold tuning is critical — too low produces noise, too high misses rare high-value patterns. Sequential rule mining (temporal ordering) requires PrefixSpan or SPADE, which are less widely packaged than FP-Growth for itemsets.

---

#### 8. Bayesian Online Changepoint Detection (BOCPD) for Cross-Domain Coupling Events

- **What:** BOCPD monitors each pairwise cross-domain correlation in real time, computing the posterior probability that a structural break in the correlation has occurred at the current timestep. When two previously uncorrelated domains suddenly couple, BOCPD fires an alert faster than any retrospective analysis can.
- **Why it's interesting:** MIDGE's CorrelationTracker currently flags anomalies using z-scores against historical mean/std. This requires accumulating enough history before detecting a shift. BOCPD would detect the coupling event online, often at the 2nd or 3rd observation into the new regime — generating a signal before the pattern becomes obvious. "Crypto and congressional trades just started moving together" is a deception/coordination signal MIDGE's deception_state.json is designed to track.
- **Evidence:** `bocpd` package on PyPI; Facebook Kats library includes BOCPD. Applied to order flow dynamics in financial markets (Quantitative Finance, 2024: https://www.tandfonline.com/doi/full/10.1080/14697688.2024.2337300). Bayesian changepoint detection applied to Hong Kong stock market (ACM MLSC 2025). The `changepoint` library (Rust + Python, 2024) supports online detection.
- **Source:** bocpd PyPI: https://pypi.org/project/bocpd/ (accessed 2026-03-05); Financial application 2024: https://www.tandfonline.com/doi/full/10.1080/14697688.2024.2337300; Quantbeckman tutorial with code: https://www.quantbeckman.com/p/with-code-switch-off-bayesian-online
- **Fits our case because:** When CorrelationTracker detects a new anomaly, it doesn't know whether it's a new stable regime or noise. BOCPD would estimate the run-length (time since last structural break), giving the convergence engine a confidence modifier: "this cross-domain coupling is 3 days old with 89% posterior probability of being a true regime change." This directly improves alert quality without replacing existing infrastructure.
- **Risks:** Computationally lightweight but requires careful prior specification (hazard rate — expected frequency of changepoints). If misspecified, either too many false breaks (high hazard) or too slow to detect real ones (low hazard). Financial correlations are known to be non-stationary so some parameter tuning is required.

---

### Emerging Approaches

#### 9. Copula-Based Cross-Domain Tail Dependence

- **What:** Fit a copula model (Clayton for lower-tail dependence, Gumbel for upper-tail, Frank for symmetric) to pairs of domain signals. The copula parameters describe how extreme events in domain A co-occur with extreme events in domain B — independent of the marginal distributions of each signal.
- **Momentum:** Active research 2024-2025. DCC-EGARCH-t-copula paper (PLOS ONE, 2025) demonstrated out-of-sample forecasting superiority. Copula-based clustering via evidence accumulation (arxiv.org, 2025). Python libraries: copulas (PyPI), copulae (PyPI), copent (transfer entropy via copula entropy).
- **Source:** PLOS ONE 2025 paper: https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0333794; copent (copula entropy / transfer entropy): https://pypi.org/project/copent/ (accessed 2026-03-05); Tail dependence overview: https://medium.com/@alexfiliakov/modeling-tail-risks-a-brief-introduction-to-copulas-b2d1b6454add
- **Fits our case because:** MIDGE's convergence engine measures average-case correlation across all signals. Tail copulas would reveal which domain pairs only correlate **during extreme events** — when they matter most for trading decisions. "VIX term structure and insider cluster only correlate in the top 10% of their respective signals" is exactly the kind of stacking insight that distinguishes high-confidence from medium-confidence alerts.
- **Maturity risk:** Fitting copulas to short financial time series (30-90 observations per domain pair) gives unreliable parameter estimates. Needs longer history. DCC-EGARCH-t-copula is parameter-heavy and computationally demanding. Simpler Gaussian copula (used as baseline) is more tractable but may miss the nonlinear tail effects that make this method valuable.

---

#### 10. Symbolic/Permutation Transfer Entropy (ordpy)

- **What:** Convert domain signal time series to ordinal patterns (rank-order of values in a sliding window) then compute transfer entropy between pattern sequences. Dramatically faster than kernel TE, robust to noise, and captures nonlinear dependencies without the distributional assumptions of Granger causality.
- **Momentum:** ordpy published in Chaos journal 2021, maintained through 2025 on GitHub. Applied to multiscale analysis of financial time series (World Scientific 2017; MDPI Entropy 2023). Research confirms ordinal/symbolic methods outperform standard TE on noisy financial series. Computational cost: milliseconds per pair on CPU.
- **Source:** ordpy PyPI: https://pypi.org/project/ordpy/ (accessed 2026-03-05); GitHub: https://github.com/arthurpessa/ordpy; MDPI symbolic encoding paper: https://www.mdpi.com/1099-4300/25/7/1009 (accessed 2026-03-05)
- **Fits our case because:** MIDGE's 28 signal sources produce normalized strength values (0-1) sampled at irregular intervals. Ordinal TE maps these naturally to rank patterns, requires no distributional assumption, and runs in milliseconds — making it feasible to run on every pair at every LagCorrelationAnalyzer cadence (every 500 steps). This is the practical replacement for Pearson lag-correlation that fits MIDGE's existing architecture slot.
- **Maturity risk:** Permutation order parameter (embedding dimension) must be tuned — too low misses nonlinear patterns, too high requires exponentially more data. Most financial applications use order 3-5. At daily resolution with 90-day windows, order 4-5 is appropriate but requires careful validation.

---

#### 11. Cross-Asset Ensemble Robustness Testing (WorldQuant/Numerai paradigm)

- **What:** Test every pattern template and hypothesis against multiple independent instruments before promotion. A pattern that works on NVDA AND MSFT AND SPY is structurally real; one that only works on NVDA is likely overfit. Apply this as a gate in the HypothesisValidator.
- **Momentum:** WorldQuant Brain's paradigm (101 Alphas, genetic programming variants) explicitly rewards signals that survive cross-asset testing. Numerai's Meta Model Contribution (MMC) and the new Alpha scoring (Feb 2025) explicitly reward originality and independence from existing signals. The cross-asset ensemble approach (QuantReo newsletter, accessed 2026-03-05) shows that individual model accuracy fluctuates 0.55-0.75 but ensemble of cross-asset models stabilizes around 0.65. Alpha-GPT (EMNLP 2025) showed top-10 performance among 41,000 WorldQuant Championship teams using LLM-driven multi-asset alpha.
- **Source:** Cross-asset ensemble article: https://www.newsletter.quantreo.com/p/cross-asset-learning-finding-true (accessed 2026-03-05); WorldQuant warm-start GP: https://arxiv.org/html/2412.00896v1 (accessed 2026-03-05); Numerai MMC: https://docs.numer.ai/numerai-tournament/scoring/meta-model-contribution-mmc
- **Fits our case because:** MIDGE's PatternWatcher already requires 3+ symbols for cross-validated template confidence boost. This approach would extend that to the Hypothesis loop: a hypothesis promoted to "active" must survive DSR filtering on 3+ instruments from different sectors. This is the institutional discipline that prevents backtest overfitting — WorldQuant, Numerai, and Two Sigma all use cross-asset replication as their primary overfitting guard.
- **Maturity risk:** Requires sufficient historical data per instrument. Some MIDGE hypotheses (e.g., congressional committee-specific patterns) are inherently single-instrument. The robustness requirement must be configurable per hypothesis type.

---

#### 12. Signal Neutralization (Numerai paradigm) for Independence Enforcement

- **What:** After computing a new signal/hypothesis, regress it against all existing Thompson distributions and pattern templates. Remove the linear component explained by known signals. Only the residual — the genuinely new information — is registered as a new hypothesis.
- **Momentum:** Numerai's entire Signals platform is built on this: "finding original signals Numerai doesn't already have." The Neutralizers Matrix (200-column, released Feb 2025) and Chili Target enable explicit orthogonalization against known factors. Python implementation via linear algebra (lstsq), trivially CPU-bound.
- **Source:** Numerai signals overview: https://docs.numer.ai/numerai-signals/signals-overview; Feature neutralization notebook: https://github.com/numerai/example-scripts/blob/master/feature_neutralization.ipynb (accessed 2026-03-05)
- **Fits our case because:** MIDGE's convergence engine currently has no formal check that a new hypothesis is orthogonal to already-discovered patterns. Two hypotheses that both capture "insider+macro convergence" will double-count in Thompson weighting. Signal neutralization before hypothesis registration would enforce the independence constraint that is the theoretical foundation of MIDGE's stacking confidence model (the multiplied complement: `1 - (1-conf_a)(1-conf_b)` only holds when conf_a and conf_b are from truly independent information sources).
- **Maturity risk:** Orthogonalization against sparse signal archives (short history) is unreliable. With 414+ days of archive, this is now tractable. Requires careful basis choice — neutralizing against too many factors over-clips genuine signal.

---

### Gaps and Unknowns

1. **Nonlinear VAR at scale:** The `Nonlincausality` package (LSTM/GRU-based Granger) would capture nonlinear causal structure, but GPU-free performance at 28 sources × 90 lags is uncharacterized. Testing would be required to determine if it's feasible in cadenced (not real-time) batch mode.

2. **Copula estimation at short history:** All copula-based approaches require substantially more data than Pearson for stable parameter estimates. MIDGE's 414-day archive is borderline adequate for some pairs, insufficient for others. No benchmark was found for copula estimation quality at T=90 observations (one domain pair, 90-day window).

3. **CD-NOTS maturity:** The 2024 paper shows CD-NOTS outperforms PCMCI on nonstationary financial data, but the implementation is research-grade (not on PyPI). Requires extraction from the arxiv supplement or direct contact with the authors. Tigramite (PCMCI) is the production-ready alternative.

4. **FP-Growth sequential patterns:** Standard FP-Growth (mlxtend) handles itemsets (unordered co-occurrence). For temporal ordering ("insider fires, THEN 5 days later macro fires, THEN contracts"), SPADE or PrefixSpan is needed. These are available in Python but less documented for financial applications.

5. **MIDGE signal granularity mismatch:** Many cross-domain correlation methods assume regular time-series sampling. MIDGE's signals arrive irregularly (some sources fire daily, others weekly, some event-driven). All statistical approaches require aligning to a common time grid (daily aggregation, which `LagCorrelationAnalyzer` already does). The aggregation method (mean, max, presence/absence) affects all downstream correlation estimates in ways not systematically characterized.

6. **Domain overlap vs. source overlap:** MIDGE's independence check uses <30% domain overlap. With 11 domains and 28 sources, some domains have many sources (technical has 6+, social has 3+). The causal discovery methods operate at the source level but the stacking independence check operates at the domain level. It is unclear whether PCMCI should run at the source or domain level — unresolved architectural question.

---

### Synthesis

**Strongest overall approach: Transfer Entropy (symbolic/ordinal) as upgrade to LagCorrelationAnalyzer**

The single highest-value change is replacing or augmenting the `LagCorrelationAnalyzer`'s Pearson cross-correlation with **ordinal/permutation transfer entropy** (ordpy library, milliseconds per pair, CPU-only). This detects directed, nonlinear information flow between domain pairs at each lag — precisely what Pearson misses. The output format is identical to what the HypothesisGenerator already consumes (`source_a`, `source_b`, `lag_days`, significance measure). This is an architectural drop-in with higher signal quality. The `infomeasure` library (Scientific Reports 2025) provides a well-documented Python implementation with benchmark timing confirmation of sub-minute performance on 10^5 element series.

**Second priority: PCMCI+ (Tigramite) as conditional causal graph**

Once per day (cadenced, not real-time), run PCMCI+ across all 28 signal sources with max_lag=30 days. This produces the causal graph that tells the HypothesisGenerator which correlations are spurious (driven by common factors) and which are genuine (survive conditioning). This eliminates the single largest false-positive source in the current hypothesis loop: patterns that appear to be cross-domain but are actually driven by the same underlying market factor. PCMCI+ is CPU-only, Python ≥3.10, PyPI-available.

**Third priority: RMT denoising + MST for structural network intelligence**

Apply Marchenko-Pastur denoising to the CorrelationTracker's correlation matrix before computing anomaly z-scores. This eliminates noise-driven false anomalies. Simultaneously, compute the MST of the signal network periodically — when the MST topology changes significantly (new edge forming between previously disconnected domains), emit a structural coupling alert. Neither requires any new data sources; both run on existing CorrelationTracker output. Together they improve precision without touching the architecture.

**Fourth priority: BOCPD for real-time structural break detection**

Wire BOCPD onto each domain pair in CorrelationTracker. When a pair's correlation has a posterior break probability >0.85, emit a structural coupling event to the EventBus. This makes MIDGE the first to detect when two domains that have never correlated before begin moving together — arguably the most valuable early-warning signal MIDGE could generate.

**Fifth priority: FP-Growth on signal archive for sequence pattern mining**

Treat each day's set of active signals as a transaction basket. Run FP-Growth (mlxtend, CPU) on the 414+ day archive to discover which domain combinations co-occur above support threshold (3%), then mine association rules for sequential patterns. This discovers patterns the convergence engine cannot find because it only looks for concurrent multi-domain alignment, not sequential domain cascades.

**What the orchestrator needs to know:**

1. MIDGE's biggest gap is that all current correlation is **bivariate Pearson** (linear, symmetric, undirected). The three highest-value upgrades — ordinal transfer entropy, PCMCI+ conditional causal graphs, and RMT denoising — all address this exact gap, and all three are CPU-only, Python-native, and architecturally compatible with MIDGE's existing cadenced analysis pattern.

2. The **independence enforcement problem** is structurally more important than finding more patterns. MIDGE's stacking confidence formula (`1 - (1-a)(1-b)(1-c)`) is only valid when the factors are genuinely independent. Signal neutralization (Numerai paradigm) applied before hypothesis registration would enforce this mathematically, potentially improving alert quality more than adding new discovery methods.

3. There is a **false convergence problem** worth flagging explicitly: when markets are broadly trending, all domain signals tend to fire simultaneously — not because of cross-domain causal structure but because of a common market factor. PCMCI+ (conditional independence) is the principled solution. Until it is implemented, the convergence engine generates systematically more false positives in bull/bear trending regimes than in sideways regimes.

4. **Computational feasibility on Wardenclyffe (i9/64GB/Win11):** All recommended approaches run CPU-only. Ordinal TE: milliseconds per pair. PCMCI+: minutes per full graph (cadenced). RMT denoising: seconds. MST: seconds. BOCPD: milliseconds per update. FP-Growth on 414 days × 28 sources: seconds. None require Docker, GPU, or cloud. All are pip-installable. This is a one-machine problem that a one-machine system can solve.

---

### Key Libraries Summary

| Library | Method | Install | CPU-only | Fits MIDGE |
|---------|---------|---------|----------|------------|
| statsmodels | Granger causality / rolling Granger | pip | Yes | Direct drop-in for LagCorrelationAnalyzer |
| tigramite | PCMCI+ conditional causal graph | pip | Yes | Cadenced HypothesisGenerator input |
| infomeasure | Transfer entropy (KSG, kernel, ordinal) | pip/conda | Yes | Replaces Pearson in LagCorrelationAnalyzer |
| ordpy | Permutation/ordinal TE | pip | Yes | Fastest TE option, milliseconds/pair |
| copent | Transfer entropy via copula entropy | pip | Yes | Nonlinear TE alternative |
| mlxtend | FP-Growth frequent itemsets | pip | Yes | Signal archive sequential mining |
| skfolio | RMT denoising covariance | pip | Yes | CorrelationTracker noise reduction |
| scipy.sparse.csgraph | Minimum spanning tree | already in scipy | Yes | Domain network topology |
| bocpd | Bayesian online changepoint detection | pip | Yes | Real-time coupling detection |
| copulas / copulae | Copula modeling | pip | Yes | Tail dependence (requires more history) |
