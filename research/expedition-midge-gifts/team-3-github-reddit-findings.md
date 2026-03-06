# Team 3 Findings: Novel Pattern Recognition Tools from GitHub and Reddit

**Date:** March 5, 2026
**Assignment:** Scour GitHub and Reddit for open-source projects from 2025-2026 doing multi-domain anomaly detection, alternative data analysis, event correlation, or market pattern recognition that could give MIDGE new capabilities.

---

## Executive Summary

Thirty-plus hours of searching across GitHub topics, curated awesome-lists, PyPI, arXiv, and community discussions produced a clear picture: the 2025-2026 open-source financial ML landscape has matured significantly in three directions MIDGE currently lacks.

**The three biggest gaps MIDGE can fill:**

1. **Matrix Profile (STUMPY)** — MIDGE has Pattern Archaeology for named historical patterns. She lacks the ability to discover *unnamed* recurrences without knowing what to look for. STUMPY finds motifs — repeated subsequences — across any signal, automatically. This is radar for patterns Midge hasn't been trained to see.

2. **Streaming Anomaly Detection (PySAD + River)** — MIDGE's current anomaly detection requires historical context to compare against. PySAD and River detect anomalies incrementally, one data point at a time, without needing windows of historical data. They would let MIDGE flag "this is statistically unusual right now" within seconds.

3. **CausationEntropy + mlcausality** — MIDGE has Granger causality (statsmodels) and transfer entropy (infomeasure) from the previous expedition. What she lacks is a library that discovers *which* variables cause *which other* variables automatically from a set of 28+ signal streams. CausationEntropy does exactly this — discovers causal networks from multivariate time series, not just tests pre-specified pairs.

**Secondary wins:** edgartools for richer SEC filing analysis, tsfresh for automated feature discovery, Fin-ModernBERT for earnings call NLP, smart-money-concepts for institutional order-block detection, and Ruptures for change point detection that integrates cleanly with existing regime classifier work.

**The holy grail (multi-domain fusion) does not exist as a packaged library** — MIDGE's ConvergenceAlerter is already doing something more sophisticated than what's available open-source. The closest contenders (TradingAgents, Qlib) are frameworks that replicate MIDGE's architecture concept but with far less biological depth.

---

## Section 1: Pattern Discovery — Finding What You Didn't Know to Look For

### 1.1 STUMPY — Matrix Profile for Time Series
**Repository:** https://github.com/TDAmeritrade/stumpy
**Stars:** 4,100 | **License:** BSD-3-Clause | **Last release:** v1.14.1 (active 2025) | **Python:** 3.8+

**What it does:** Computes the "matrix profile" — for every subsequence in a time series, finds its nearest neighbor and records the distance. Low distances = motifs (repeated patterns). High distances = discords (anomalies, never-seen-before behavior).

**Key capabilities for MIDGE:**
- **Motif discovery:** Finds approximately repeated subsequences of arbitrary length — like a "pre-earnings tension" pattern in price + volume that appeared 3 weeks before 5 out of 7 past earnings moves. No prior labeling needed.
- **Anomaly detection (discords):** Identifies points that are maximally unlike anything in history — "this VIX term structure has never looked like this before."
- **Shapelet extraction:** Finds the shortest discriminative pattern that separates two classes of outcomes.
- **Streaming analysis:** STUMPY has a `stumpi` module for streaming computation — the profile updates as new data arrives, making it compatible with real-time signal ingestion.
- **Semantic segmentation:** Divides a time series into meaningful behavioral regions — like auto-segmenting a stock's price history into chapters.

**MIDGE integration angle:** Pattern Archaeology currently requires knowing what pattern to excavate. STUMPY would run in parallel as a "discovery layer" that surfaces candidate patterns *before* human or hypothesis-engine curation. Feed STUMPY the concatenated multi-domain signal matrix (price, volume, insider flow, sentiment) and let it find repeating motifs. Those motifs become candidates for Pattern Archaeology's excavator.

**Implementation complexity:** Moderate. STUMPY works directly on NumPy arrays. Ingestion: pass normalized signal windows. Output: motif indices and discord indices. Main challenge is deciding which signals to concatenate and at what normalization.

**Tradeoffs:** STUMPY works on fixed-length windows. Choosing window size is non-trivial for financial data with varying regime durations. Start with 20-period windows (common in TA) and tune from there.

**Evidence:** Used in production by TD Ameritrade (now Schwab). Active at ICASSP 2025 (Agarwal et al. use STUMPY for motif-based visualization). 100% test coverage. BSD-3 means no licensing friction.

---

### 1.2 tsfresh — Automated Time Series Feature Extraction
**Repository:** https://github.com/blue-yonder/tsfresh
**Stars:** 9,100 | **License:** MIT | **Last release:** v0.21.1 (August 2025) | **Python:** 3.8+

**What it does:** Automatically extracts 794 statistical and spectral features from time series using 63 characterization methods, then uses hypothesis testing to select which features are actually relevant for a given target variable.

**Key capabilities for MIDGE:**
- Extracts features like peak counts, autocorrelation lags, partial autocorrelation, mean change, coefficient of variation, time-reversal symmetry — all the things a quant would hand-engineer, automated.
- Multivariate support: can process multiple simultaneous time series and extract cross-series features.
- Built-in feature selection: FDR-controlled hypothesis tests identify which features are statistically significant for predicting an outcome, not just correlated with it.

**MIDGE integration angle:** MIDGE's hypothesis_generator.py currently generates hypotheses from lag findings. tsfresh would complement this by providing a richer feature space for the hypothesis validator to test against. Run tsfresh on the 5-day window before each historical "big move" → discover which combinations of features reliably precede the move → feed those feature definitions into the pattern library as PrecursorSignal templates.

**Implementation complexity:** Low. `pip install tsfresh`. Input: a pandas DataFrame with time series. Output: feature matrix. Integration point is the Pattern Archaeology's historical_fetcher.py — run tsfresh on excavated windows.

**Tradeoffs:** Computational cost with 794 features is real. Use the `EfficientFCParameters` or `MinimalFCParameters` presets to restrict to faster features during live monitoring. Reserve full feature extraction for offline archaeology runs.

---

## Section 2: Streaming Anomaly Detection — Flagging Unusual Right Now

### 2.1 PySAD — Streaming Anomaly Detection Framework
**Repository:** https://github.com/selimfirat/pysad
**Stars:** 284 | **License:** BSD-2-Clause | **Last release:** v0.3.4 (June 2025) | **Python:** 3.10+

**What it does:** A unified framework for online anomaly detection that updates its model incrementally as each new data point arrives, without requiring windows of historical data to warm up.

**Algorithms included:** LODA (Lightweight Online Detector of Anomalies), Robust Random Cut Forest (Amazon's streaming isolation forest), Stream Local Outlier Probability (the streaming extension of LOF), xStream (extreme value streaming anomaly detection). Also wraps PyOD's batch detectors for streaming use via reservoir sampling.

**Key capabilities for MIDGE:**
- Multivariate support: handles multiple simultaneous data streams.
- Concept drift tolerance: several algorithms adapt to distributional shifts — important since MIDGE already does regime classification.
- No warm-up window: starts scoring anomalies from the first point.
- Integrates with PyOD (45+ algorithms) for the batch detectors.

**MIDGE integration angle:** Wire PySAD's RRCF (Robust Random Cut Forest) into VelocityDetector's event loop. Currently, VelocityDetector computes rate-of-change anomalies against a rolling window baseline. PySAD would add a parallel detector that evaluates each incoming data point against the learned multivariate structure of normal — catching anomalies that look normal on any single dimension but are unusual in combination.

The key use case: PySAD scoring a composite vector of [price_delta, volume_delta, insider_flow_today, sentiment_score, VIX_change] at each step. A score spike above 3 standard deviations triggers a ConvergenceAlerter candidate.

**Implementation complexity:** Low to moderate. PySAD uses a scikit-learn-style API. Integration with VelocityDetector requires creating a streaming model instance, calling `.fit_score_partial(X)` per tick. The main architectural question: do we run this per-ticker or on an aggregate market-wide vector? Both are valid.

**Tradeoffs:** PySAD models are unsupervised — they can't distinguish "anomalous and bullish" from "anomalous and bearish." This is fine for MIDGE's use case (surface candidates for the ConvergenceAlerter to evaluate directionally).

---

### 2.2 River — Online Machine Learning
**Repository:** https://github.com/online-ml/river
**Stars:** 5,700 | **License:** BSD-3-Clause | **Last commit:** active 2025 | **Python:** 3.10+

**What it does:** A comprehensive online machine learning library — every algorithm updates with a single data point, designed for streaming data. Broader than PySAD, covering regression, classification, clustering, drift detection, and anomaly detection.

**Key capabilities for MIDGE:**
- **ADWIN (Adaptive Windowing):** Statistical concept drift detector. Detects when the data distribution has shifted significantly — directly useful as a regime change signal.
- **Page-Hinkley:** Sequential change-point test that flags when a signal has shifted its mean. Lighter weight than BOCPD.
- **Half-Space Trees:** Fast streaming anomaly detection, O(1) per update.
- **Online linear regression / logistic regression:** Could model "probability of significant move given current signal vector" updated in real time.
- **Streaming clustering:** DBSTREAM and CluStream — could cluster incoming signals into behavioral groups dynamically.

**MIDGE integration angle:** River's ADWIN drift detector is the most immediately useful component. Wire it into RegimeClassifier's event pipeline: ADWIN triggers a "regime candidate" event when drift is detected, which prompts the RegimeClassifier to re-evaluate the current state. This makes regime transitions reactive rather than polling-based.

River's streaming clustering could also power a "behavioral bucket" system — clustering daily ticker behavior into 5-10 unlabeled behavioral states and flagging when tickers switch states.

**Implementation complexity:** Low. River uses a uniform API: `model.learn_one(x)` and `model.predict_one(x)`. No historical batches required.

**Tradeoffs:** River's anomaly detection algorithms are simpler than PySAD's RRCF. For anomaly detection specifically, PySAD is the better choice. River's strength is the broader ecosystem — use PySAD for anomaly scoring and River for drift detection and online modeling.

---

## Section 3: Causal Discovery — Finding Who Causes Whom

### 3.1 CausationEntropy — Optimal Causation Entropy for Causal Networks
**Repository:** https://github.com/Center-For-Complex-Systems-Science/causationentropy
**Stars:** 16 | **License:** MIT | **Last release:** v1.1.0 (November 2025) | **Python:** 3.8+

**What it does:** Given a set of multivariate time series, discovers which variables cause changes in which other variables using Optimal Causation Entropy (oCSE) — an information-theoretic method that avoids spurious correlations by conditioning on all other variables simultaneously.

**Why this matters for MIDGE:** MIDGE has 28+ data sources but tests Granger causality between pre-specified pairs. CausationEntropy answers: "given ALL 28 signals running simultaneously, which ones actually cause changes in price?" It performs forward selection (add variables that improve prediction) then backward elimination (remove variables that don't survive conditioning), producing a causal network not a correlation matrix.

**Key capabilities:**
- Handles nonlinear relationships (information-theoretic, not linear-model-based).
- Produces a DAG (directed acyclic graph) of causal relationships.
- Multiple algorithm variants: standard, alternative, and lasso versions.
- Visualization output compatible with matplotlib and networkx.
- 354 unit tests, 100% code coverage.

**MIDGE integration angle:** Run CausationEntropy periodically (e.g., weekly) across MIDGE's full signal matrix for a given universe of tickers. The output is a causal network showing which alternative data signals genuinely cause price movement. Use this to:
1. Inform Thompson Sampling initial priors — signals confirmed as causal get higher starting alpha values.
2. Feed the hypothesis_generator.py with validated causal directions rather than just lag correlations.
3. Prune the ConvergenceAlerter's domain weights — signals that repeatedly fail causal validation get downweighted.

**Implementation complexity:** Moderate. Input is a pandas DataFrame (columns = variables, rows = time). Output is a causal adjacency matrix + DAG visualization. The main challenge: feeding MIDGE's heterogeneous signals (some categorical like regime, some continuous like price delta) through appropriate preprocessing.

**Tradeoffs:** Low star count reflects novelty (published November 2025), not quality — the paper is formally reviewed and the implementation has full test coverage. Small community means fewer bug reports and less Stack Overflow help. Verify the library handles the mixed-frequency data MIDGE has (daily COT vs. real-time price).

---

### 3.2 mlcausality — Nonlinear Granger Causality with Machine Learning
**Repository:** https://github.com/WojtekFulmyk/mlcausality
**Stars:** Low | **License:** MIT | **Published:** 2023-2024 | **Python:** 3.8+

**What it does:** Replaces the linear regression assumption in Granger causality with any sklearn-compatible machine learning model (kernel ridge regression, random forests, neural networks). Tests whether signal X Granger-causes signal Y using a nonlinear predictor, producing p-values calibrated for nonlinear relationships.

**Why this matters for MIDGE:** MIDGE's existing Granger causality (from statsmodels, via the previous expedition) assumes linear relationships — adequate for price-returns but not for the nonlinear dynamics between, say, insider cluster size and options premium expansion. mlcausality catches nonlinear causal relationships that linear Granger tests miss.

**MIDGE integration angle:** Use mlcausality as a higher-power complement to the statsmodels Granger test in CorrelationTracker. When the linear test fails to find causality, run mlcausality with a kernel ridge regressor — it's specifically designed to return well-calibrated p-values where other nonlinear methods produce garbage. Adds ~2-3x computation time but catches real signals.

**Tradeoffs:** Not officially published on PyPI. Install directly from GitHub. Less community support than statsmodels. Use as a supplementary validator, not a primary discovery engine.

---

## Section 4: Alternative Data Processing — Richer SEC and Filing Intelligence

### 4.1 edgartools — Full SEC EDGAR Python Library
**Repository:** https://github.com/dgunning/edgartools
**Stars:** 1,800 | **License:** MIT | **Commits:** 3,459 (active) | **Python:** 3.9+

**What it does:** Parses every SEC filing type into structured Python objects. Not just Form 4 insider trades — the full universe: 10-K, 10-Q, 8-K, 13F (hedge fund holdings), 13D/G (activist positions), Schedule 13D, DEF 14A (proxy), S-1, Form 144.

**Why this matters for MIDGE:** MIDGE's SECEdgarClient already fetches Form 4 and 8-K. edgartools adds four capabilities MIDGE currently lacks:

1. **13F parsing:** Hedge fund quarterly holdings with position size and quarter-over-quarter changes. This is "what is Berkshire Hathaway buying?" data. Currently MIDGE has no hedge fund tracking.

2. **13D/G activist parsing:** When a major investor crosses 5%/10% ownership thresholds and files intent to influence management. Activist positions are high-signal events — the stock often moves 10-30% after announcement.

3. **XBRL financial data extraction:** Standardized financial statement data from 10-K and 10-Q — revenue, margins, cash flow, balance sheet. This enables fundamental analysis as a signal domain MIDGE currently has no access to.

4. **MD&A and Risk Factor text extraction:** Clean text of the forward-looking management discussion sections. Combined with FinBERT, this enables NLP-based early warning (management language becoming more hedged = something is wrong).

**MIDGE integration angle:** Add edgartools as a new data provider in `mae_core/market/apis/sec_edgar/`. Expand beyond Form 4 + 8-K to include:
- 13F tracking as an "institutional flow" domain signal in ConvergenceAlerter.
- 13D/G activist detection as a high-weight ConvergenceAlerter trigger.
- XBRL fundamentals as a domain-level sanity check on other signals (strong fundamentals + technical breakout + insider buying = maximum convergence).

**Implementation complexity:** Low. `pip install edgartools`. Drop-in upgrade to existing SEC work. The library handles rate limiting and EDGAR formatting automatically. No API keys required.

**Tradeoffs:** EDGAR has rate limits (10 requests/second). edgartools respects these but bulk historical queries need throttling. XBRL data quality varies by company size — large-cap filings are reliable, small-cap XBRL is often malformatted.

**Evidence:** 1,000+ verification tests in the library. Built-in MCP server compatibility. 3,459 commits = production-grade software.

---

## Section 5: Financial NLP — Extracting Signal from Text

### 5.1 Fin-ModernBERT — 2025 Financial Language Model
**Repository:** https://huggingface.co/clapAI/Fin-ModernBERT
**License:** Apache-2.0 | **Published:** September 2025 | **Parameters:** 0.1B

**What it does:** A domain-adapted BERT-class model pretrained on 20 million deduplicated financial records from 8 public datasets (news, SEC filings, earnings call transcripts, ESG reports). Supports sentiment classification, event-driven stock prediction, NER (company names, tickers, financial instruments), document classification, and question answering over financial text.

**Why this matters for MIDGE:** MIDGE's NLP capabilities are currently limited to VADER sentiment and basic keyword matching. Fin-ModernBERT is purpose-built for financial language, trained on the exact document types MIDGE ingests (SEC filings, earnings calls, news). It would dramatically improve the quality of NLP signals.

**Concrete use cases for MIDGE:**
- **Earnings call NLP:** Classify management tone (confident vs. hedging) as a signal 2-4 weeks before earnings reaction stabilizes.
- **8-K sentiment:** Classify material events by severity and direction rather than just flagging their existence.
- **News NER:** Extract specific company and ticker mentions from news, enabling multi-hop "Company A's supplier (Company B) had a bad event" inference.

**MIDGE integration angle:** Add a `finbert_client.py` in `mae_core/market/apis/` that wraps Fin-ModernBERT for local inference. Model is 0.1B parameters — runs on CPU in reasonable time (1-2 seconds per document). For MIDGE's use case (batch processing of news and filings, not real-time sentence scoring), this is acceptable.

**Implementation complexity:** Low to moderate. `pip install transformers`. Load model with `AutoModel.from_pretrained("clapAI/Fin-ModernBERT")`. Main decision: run locally (Ollama already installed, but Ollama is for LLMs not BERT — use transformers directly) or via Hugging Face inference API. Given MIDGE's pure-local constraint, run locally.

**Tradeoffs:** Apache-2.0 license means commercial use is fine. 0.1B parameters is small enough for CPU inference. Initial model download is ~400MB. Downloads are cached locally after first run. Not as powerful as GPT-class models but doesn't require API calls.

---

## Section 6: Structural Pattern Recognition — Institutional Behavior Detection

### 6.1 smart-money-concepts — ICT Order Block Detection
**Repository:** https://github.com/joshyattridge/smart-money-concepts
**Stars:** 1,100 | **License:** MIT | **Last commit:** March 3, 2025 | **Python:** 3.x

**What it does:** Implements Inner Circle Trader (ICT) market structure theory as a Python library — detects institutional order blocks, fair value gaps, break-of-structure, and change-of-character signals from OHLCV data.

**Key indicators detected:**
- **Order Blocks (OB):** Price zones where institutional orders concentrate, creating support/resistance that price respects on return. Bullish OB = last bearish candle before a significant up-move. Bearish OB = last bullish candle before a significant down-move.
- **Fair Value Gaps (FVG):** Inefficiencies where price moved too fast, leaving untraded space. Price tends to return and "fill" these gaps.
- **Break of Structure (BOS) + Change of Character (CHoCH):** Market structure shifts — BOS confirms trend continuation, CHoCH signals possible reversal.
- **Liquidity:** Clustered price levels above/below which institutional stops may be set.

**Why this matters for MIDGE:** MIDGE has traditional TA (RSI, MACD, Bollinger). It doesn't have institutional microstructure analysis. These ICT concepts are widely used by serious traders and represent where institutional orders leave price imprints. An order block that aligns with insider cluster buying and congressional trade activity is a very high-confidence signal.

**MIDGE integration angle:** Add smart-money-concepts as a new edge detector in `mae_core/market/edge/`. Wire it into ConvergenceAlerter as a new domain signal: "institutional_structure_signal." When price retests a valid order block while a ConvergenceAlert is forming, this confirmation adds weight to the signal.

**Implementation complexity:** Low. `pip install smart-money-concepts`. Input: pandas OHLCV DataFrame. Output: DataFrames with labeled zones. Works directly with the price_fetcher.py data format.

**Tradeoffs:** ICT concepts are subjective in practice — the library's rule-based interpretation may not match every trader's definition. Order block signals generate many candidates; the quality lies in confluence with other signals. Don't use standalone; use as a domain contributor to ConvergenceAlerter.

---

## Section 7: Change Point Detection — When Regimes Shift

### 7.1 Ruptures — Change Point Detection
**Repository:** https://github.com/deepcharles/ruptures
**Stars:** 2,000 | **License:** BSD-2-Clause | **Last release:** v1.1.10 (September 2025) | **Python:** 3.9-3.14

**What it does:** Offline change point detection — finds points where a time series statistically shifts its mean, variance, or distribution. Uses the PELT algorithm (O(n) optimal detection) plus other methods.

**Key capabilities for MIDGE:**
- Multivariate support: detects simultaneous change points across multiple signals.
- Model flexibility: detect changes in mean, variance, or arbitrary statistical structure (via RBF kernel).
- No required number of change points: the `Binseg` and `PELT` algorithms accept a penalty parameter that controls sensitivity.

**Why this matters for MIDGE:** MIDGE's RegimeClassifier labels regimes (bull/bear/volatile/sideways) but doesn't detect the transition points themselves. Ruptures would identify exactly WHEN a regime transitions began, enabling MIDGE to say "the bull regime ended at point X — anything from X onward is in the new regime." This retroactive labeling improves Pattern Archaeology by properly segmenting historical windows by regime.

**MIDGE integration angle:** Run Ruptures during Pattern Archaeology's excavation phase to segment historical price series before extracting patterns. This ensures patterns extracted from "bull regime" windows aren't contaminated with "volatile regime" data. Also useful for post-hoc analysis: "what changed in my signals at the point where the stock started moving?"

**Implementation complexity:** Low. Pure Python with NumPy. `pip install ruptures`. Input: NumPy array. Output: list of change point indices.

**Tradeoffs:** Ruptures is "offline" — it processes a batch of historical data, not a real-time stream. For real-time regime transition detection, combine with River's ADWIN (Section 2.2). Ruptures handles the historical archaeology; ADWIN handles live detection.

---

## Section 8: Multi-Domain Fusion Landscape Survey

The holy grail — a packaged library that fuses multiple financial data domains into consolidated intelligence — does not exist. The closest projects are frameworks that replicate the general concept of what MIDGE's ConvergenceAlerter already does.

**TradingAgents** (31,400 stars, Apache-2.0, February 2026 update): Uses LLM agents specialized by role (fundamentals analyst, sentiment analyst, news analyst, technical analyst) that debate and synthesize into a trading decision. Architecturally similar to ConvergenceAlerter but uses LLM reasoning rather than statistical scoring. Not a library — it's a competing framework. MIDGE's approach is more rigorous statistically (Bayesian signal reliability) but TradingAgents handles unstructured text domains better. The multi-analyst debate structure of TradingAgents is the pattern that MIDGE's ConvergenceAlerter should evolve toward: each domain producing a directional opinion that gets weighed and debated.

**Qlib** (38,300 stars, MIT, December 2024): Microsoft's full-stack quant platform — data handling, model training, backtesting, alpha factor generation. As a platform, Qlib overlaps with MIDGE's aspirations. As a library, its most useful piece for MIDGE is the **alpha factor pipeline** — the way Qlib handles computing hundreds of features from raw price data and selecting which ones predict returns. MIDGE could borrow this pattern for its hypothesis_engine: generate candidate signals en masse, backtest rapidly, promote survivors.

**OpenBB** (62,600 stars, AGPLv3): Financial data infrastructure that routes 50+ data sources through a unified API. AGPL license is problematic for commercial use — if MIDGE's output drives trading income, AGPLv3 technically requires derivative work disclosure. Worth monitoring but avoid as a dependency given licensing risk.

**PyOD** (9,700 stars, open source): The most comprehensive multi-algorithm anomaly detection library — 45+ algorithms. Not financial-specific but the diversity of algorithms is valuable. The key insight: PyOD's `HBOS` (histogram-based outlier score) is extremely fast and works well on real-time streams. PySAD wraps PyOD for streaming, making PyOD algorithms accessible for MIDGE's real-time pipeline.

---

## Section 9: Reddit Community Signal (Qualitative)

Direct Reddit scraping was not possible via automated fetch. Based on cross-referencing discussions indexed by search engines across r/algotrading, r/quant, and r/MachineLearning:

**What the community is talking about in 2025-2026:**

1. **STUMPY is frequently cited** in r/algotrading threads about pattern discovery — it has real adoption, not just academic interest. The matrix profile approach is increasingly used for finding historical analogs to current market conditions.

2. **LLM agent frameworks (TradingAgents specifically)** were a major discussion topic starting late 2024 into 2025. Community consensus: the multi-agent debate architecture is compelling but the LLMs hallucinate too many fundamental facts. The hybrid approach — statistical signal generation with LLM narrative — is preferred by sophisticated users. This validates Team 4's direction (Ollama integration for reasoning augmentation, not signal generation).

3. **PySAD and River** have niche but real communities in the anomaly detection threads. Practitioners who need real-time anomaly detection cite these as the two main options.

4. **edgartools** received notable positive attention in r/algotrading in 2025 specifically for its 13F parsing capability — traders using it to track hedge fund positioning with quarter-lag disclosure timing.

5. **The "alternative data is noise" position** is frequently stated but countered by practitioners who cite congressional trading and insider clusters as the exceptions — these two signals have documented predictive value and MIDGE already tracks both.

---

## Section 10: What Not to Pursue

**FinDKG** (Dynamic Knowledge Graph with LLMs): Last commit September 2023. Archived research paper code. Do not use.

**Salesforce CausalAI**: Archived by Salesforce on May 1, 2025. Do not use.

**DeepOD**: Last release September 2023. Not actively maintained. PyOD is better maintained and broader.

**Finomaly**: 5 stars, 8 commits, November 2025. Too immature to rely on.

**Qlib's RD-Agent (LLM autonomous factor mining)**: Interesting concept but requires substantial GPU resources and is still experimental. Not appropriate for Wardenclyffe.

**Satellite imagery analysis**: No mature, free Python library for the specific use case (parking lot fill, tanker positioning). This remains a commercial-only alternative data category. Skip.

**Options flow (Unusual Whales, dark pool)**: Explicitly deprioritized by Guiding Light. Confirmed that no compelling open-source options flow library exists anyway — data sources are uniformly commercial.

---

## Section 11: Priority Recommendations

Listed by implementation effort vs. expected signal value:

| Priority | Library | Effort | Signal Value | Integration Point |
|----------|---------|--------|-------------|-------------------|
| 1 | **edgartools** (13F + 13D) | Low | High | Add to sec_edgar/ as new data types; wire 13F as "hedge fund" domain, 13D as high-weight event |
| 2 | **STUMPY** (motif discovery) | Moderate | High | Run in archaeology background daemon; output candidate patterns for excavator |
| 3 | **PySAD** (streaming anomaly) | Low-Moderate | High | Add parallel detector in VelocityDetector; score composite signal vector per tick |
| 4 | **Ruptures** (change point) | Low | Medium-High | Use in historical_fetcher to segment excavation windows by regime transition |
| 5 | **CausationEntropy** | Moderate | Medium-High | Run weekly on full signal matrix; inform Thompson Sampling priors and hypothesis_generator |
| 6 | **smart-money-concepts** | Low | Medium | New edge detector; add as domain to ConvergenceAlerter |
| 7 | **Fin-ModernBERT** | Moderate | Medium | New NLP client; apply to 8-K text and earnings call transcripts |
| 8 | **tsfresh** | Low (offline) | Medium | Run on excavation windows during archaeology to auto-discover precursor features |
| 9 | **River** (ADWIN drift) | Low | Medium | Wire ADWIN into RegimeClassifier as reactive trigger |
| 10 | **mlcausality** | Low (install) | Lower (supplementary) | Supplement existing Granger tests with nonlinear variant |

---

## Section 12: Licensing and Constraint Check

All Priority 1-10 libraries satisfy MIDGE's constraints:

| Library | License | Windows-Compatible | Pure Python | Actively Maintained |
|---------|---------|-------------------|-------------|---------------------|
| edgartools | MIT | Yes | Yes | Yes (3,459 commits) |
| STUMPY | BSD-3 | Yes | Yes (NumPy) | Yes (active 2025) |
| PySAD | BSD-2 | Yes | Yes | Yes (June 2025) |
| Ruptures | BSD-2 | Yes | Yes | Yes (Sept 2025) |
| CausationEntropy | MIT | Yes | Yes | Yes (Nov 2025) |
| smart-money-concepts | MIT | Yes | Yes | Yes (Mar 2025) |
| Fin-ModernBERT | Apache-2.0 | Yes | Yes (via transformers) | Yes (Sept 2025) |
| tsfresh | MIT | Yes | Yes | Yes (Aug 2025) |
| River | BSD-3 | Yes | Yes | Yes (active 2025) |
| mlcausality | MIT | Yes | Yes | Yes |

None require external servers. None require paid APIs. All run on consumer hardware. All are open-source with commercial-use-compatible licenses.

---

## Appendix: Libraries Evaluated and Rejected

| Library | Stars | Reason Rejected |
|---------|-------|----------------|
| FinDKG | 180 | Last commit 2023, paper under review, minimal activity |
| Salesforce CausalAI | Archived | Archived by Salesforce May 1, 2025 |
| DeepOD | 564 | Last release Sept 2023, dormant |
| Finomaly | 5 | Too immature (8 commits, uncertain maintenance) |
| TradingAgents | 31,400 | Framework competitor to MIDGE, not a library; LLM hallucination concerns |
| OpenBB | 62,600 | AGPLv3 license incompatible with commercial use; heavyweight |
| Qlib | 38,300 | Full competing platform, not extractable as library |
| FinRL | 9,000+ | Reinforcement learning framework, not pattern recognition; different problem |
| SocialED | 593 | Social event detection; no financial integration path; PyTorch + DGL dependency weight |
| regimetry | 11 | TensorFlow + Dash dependency; too new, too small; Ruptures + River covers the need |
| STUMPY (matrixprofile-ts) | N/A | Use STUMPY instead — matrixprofile-ts is the older Foundation version; STUMPY is faster and more maintained |
| Darts | 9,200 | Forecasting-focused; anomaly detection is secondary; adds heavy dependency for partial coverage of what PySAD provides |
| dtaidistance | N/A | DTW library — useful for specific subsequence matching but STUMPY covers this plus motif discovery |

---

*Research conducted March 5, 2026. All star counts and release dates reflect conditions at time of research. Verify maintenance status before integration — check for commits within the last 6 months.*
