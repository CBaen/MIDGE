# MIDGE Market Data Pipeline Audit — Lead Auditor Findings
**Lens:** Market Data Pipeline Efficiency
**Date:** 2026-03-14
**Scope:** All 149 systems, 33-layer bootstrap, 14 agent mixins

---

## Audit Summary

MIDGE's pipeline traces: **data ingestion → signal creation → pattern detection → convergence synthesis → inevitability surfacing → action**. This audit evaluates every system against that pipeline. The core pipeline is well-constructed. The critical finding is not broken systems — most work. It is **overhead volume**: 40+ bio-simulation systems execute every step, consuming CPU in exchange for indirect, marginal, or zero market intelligence value. Several were already identified as harmful and neutralized through pinning, but the code still runs.

A second finding: **two disconnected attention pipelines** — the core organism's AttentionalGate/GlobalWorkspace and the market intelligence sensing pipeline have no live connection. Market signals do not feed the organism-level attention system that governs agent behavior.

---

## Group 1: Market Intelligence Systems (Layer 33)

These are MIDGE's differentiated systems. The pipeline runs through them.

---

### MarketDataProvider
- **Category:** ESSENTIAL
- **Location:** `mae_core/market/apis/market_data_provider.py`
- **What it does:** Unified data access layer. Injected into all 24 API clients as `provider=`. Routes requests through rate-limiting and caching.
- **Evidence:** Constructed at bootstrap start; injected into every client constructor. ApiGateway registers it as `"market_data"` provider.
- **Reasoning:** Entry point for all market data. Failure here degrades every client simultaneously.

---

### RawStore
- **Category:** ESSENTIAL
- **Location:** `mae_core/market/raw_store.py`
- **What it does:** SQLite persistence layer (WAL mode, per-domain) for all raw API responses before processing. Injected into all 24 clients.
- **Evidence:** Constructed before any API client; `raw_store` kwarg passed to every client in the bootstrap loop. `SocialTextAnalyzer` and `RawDataAnalyst` read from it.
- **Reasoning:** Enables replay, post-mortem analysis, and the SocialTextAnalyzer. Without it, data is processed and discarded — no archaeology.

---

### SECEdgarClient
- **Category:** ESSENTIAL
- **Location:** `mae_core/market/apis/sec_edgar/client.py`
- **What it does:** Fetches Form 4 insider trades and Form 8-K material events from SEC EDGAR.
- **Evidence:** Constructed in bootstrap loop. Wired into MarketSensingHook's source rotation. Form 8-K feeds Form8KSentimentAnalyzer in sensing infrastructure.
- **Reasoning:** Insider trades = "insider" domain. One of the highest-trust sources (0.90). Core pipeline contributor.

---

### HouseStockWatcherClient / SenateStockWatcherClient
- **Category:** ESSENTIAL
- **Location:** `mae_core/market/apis/house_stock_watcher.py`, `senate_stock_watcher.py`
- **What it does:** Congressional stock trade tracking (STOCK Act disclosures) for both chambers.
- **Evidence:** Both wired into sensing hook. `PoliticianTracker` cross-references committee membership. "government" domain.
- **Reasoning:** Congressional trades are a documented edge source. Both chambers contribute to the "government" domain.

---

### JobTracker
- **Category:** ESSENTIAL
- **Location:** `mae_core/market/apis/job_tracker.py`
- **What it does:** Detects hiring blitz events via RapidAPI, which predict expansion before public announcement.
- **Evidence:** Wired into sensing hook source rotation.
- **Reasoning:** Hiring surges are a leading indicator of corporate activity. Contributes to "fundamental" domain.

---

### USASpendingClient / SAMGovClient
- **Category:** ESSENTIAL
- **Location:** `mae_core/market/apis/usa_spending.py`, `sam_gov.py`
- **What it does:** Government contract awards (historical actuals and opportunities).
- **Evidence:** Both wired into sensing hook. Feed ContractPredictor.
- **Reasoning:** Government contracts = "government" domain. SAM.gov adds forward-looking signal.

---

### PriceFetcher
- **Category:** ESSENTIAL
- **Location:** `mae_core/market/apis/price_fetcher.py`
- **What it does:** yfinance + Alpha Vantage fallback for current prices, historical OHLCV, ATR computation.
- **Evidence:** Used by ConvergenceAlerter, RegimeClassifier, OutcomeTracker, OutcomeCollector, ActiveTracker, SignalTranslator (ATR), KellyPositionSizer, FractalResonanceDetector, PatternArchetypeEngine, MotifDetector, and every paper trade. Central dependency.
- **Reasoning:** Every output that references price depends on this. It is the organism's heartbeat sensor.

---

### FinnhubClient / FinnhubWebSocket
- **Category:** ESSENTIAL
- **Location:** `mae_core/market/apis/finnhub_client.py`, `finnhub_websocket.py`
- **What it does:** REST news/sentiment + WebSocket real-time trade ticks. WebSocket streams live price activity directly into the signal buffer.
- **Evidence:** FinnhubWebSocket.start() is called in `_wire_sensing_hook`. REST client wired into sensing rotation.
- **Reasoning:** Real-time streaming is the only live price feed (beyond polling). Critical for market-hours responsiveness.

---

### FREDClient
- **Category:** ESSENTIAL
- **Location:** `mae_core/market/apis/fred_client.py`
- **What it does:** Yield curve data (DGS2, DGS10, T10Y3M), federal funds rate, and macro series.
- **Evidence:** Wired into sensing hook. "macro" domain. Regime classifier may use macro signals.
- **Reasoning:** Yield curve signals are leading macro indicators. One of the highest-trust sources (0.95).

---

### COTClient
- **Category:** ESSENTIAL
- **Location:** `mae_core/market/apis/cot_client.py`
- **What it does:** CFTC Commitment of Traders positioning data (managed money, commercial, non-commercial).
- **Evidence:** Wired into sensing hook. "positioning" domain.
- **Reasoning:** Extreme positioning is a contrarian signal. Feeds the "positioning" domain.

---

### StockTwitsClient
- **Category:** USEFUL
- **Location:** `mae_core/market/apis/stocktwits_client.py`
- **What it does:** Social sentiment + trending tickers. Raw messages stored in RawStore for SocialTextAnalyzer.
- **Evidence:** Wired into sensing hook. RawStore messages feed SocialTextAnalyzer keyword extraction.
- **Reasoning:** Low trust (0.40) but contributes to "sentiment" domain. The SocialTextAnalyzer adds value beyond raw Thompson score.

---

### VIXClient
- **Category:** ESSENTIAL
- **Location:** `mae_core/market/apis/vix_client.py`
- **What it does:** VIX term structure and fear gauge. VIX spikes = regime signal.
- **Evidence:** Wired into sensing hook. RegimeClassifier uses VIX state for regime classification.
- **Reasoning:** VIX is the market's fear thermometer. Critical for regime detection, which gates Thompson forgetting rates.

---

### TrendsClient
- **Category:** USEFUL
- **Location:** `mae_core/market/apis/trends_client.py`
- **What it does:** Google Trends search interest + related rising queries. Rising queries stored in `discovered_keywords.json` for self-expanding keyword discovery.
- **Evidence:** Wired into sensing hook. `discovered_keywords.json` feeds next fetch cycle. "sentiment" domain.
- **Reasoning:** Search trends detect retail attention before price moves. Self-expanding keyword loop is a unique capability.

---

### EIAClient
- **Category:** ESSENTIAL
- **Location:** `mae_core/market/apis/eia_client.py`
- **What it does:** Energy inventory/production data (crude, gasoline, distillates, natural gas).
- **Evidence:** Wired into sensing hook. Trust 0.95. "energy" domain.
- **Reasoning:** Only real-economy domain (supply vs. demand in physical barrels). Highest trust after FRED. Cross-domain with energy equities and macro.

---

### CongressGovClient
- **Category:** USEFUL
- **Location:** `mae_core/market/apis/congress_gov_client.py`
- **What it does:** Legislative tracking (bills affecting sectors/companies).
- **Evidence:** Wired into sensing hook. "government" domain complement to stock watchers.
- **Reasoning:** Legislation can move sectors before votes. Adds forward-looking dimension to government domain.

---

### USDAClient
- **Category:** USEFUL
- **Location:** `mae_core/market/apis/usda_client.py`
- **What it does:** USDA WASDE agricultural reports. Seasonal crop/commodity signals.
- **Evidence:** Wired in bootstrap. "fundamental" domain for agricultural commodities.
- **Reasoning:** Seasonal and limited applicability unless MIDGE watches agricultural commodities. Useful when applicable.

---

### CoinGeckoClient / CoinCapClient
- **Category:** USEFUL
- **Location:** `mae_core/market/apis/coingecko_client.py`, `coincap_client.py`
- **What it does:** 24/7 cryptocurrency price and market cap data.
- **Evidence:** Wired into sensing hook (Wave 2+3). "crypto" domain.
- **Reasoning:** Crypto markets are 24/7 — they provide signal during equity market hours too. Useful for cross-asset confirmation.

---

### OpenInsiderClient
- **Category:** ESSENTIAL
- **Location:** `mae_core/market/apis/openinsider_client.py`
- **What it does:** Pre-filtered high-value insider trades. `get_cluster_buys()` now actively called.
- **Evidence:** Wired in sensing hook. Trust 0.80. Third independent "insider" domain source.
- **Reasoning:** Independent corroboration of SEC EDGAR insider data. Cluster buys are among the highest-conviction insider signals.

---

### EdgarEnhancedClient
- **Category:** USEFUL
- **Location:** `mae_core/market/apis/edgar_enhanced_client.py`
- **What it does:** 13F institutional filings and 13D/G activist filings.
- **Evidence:** Wired into sensing hook. Trust 0.85. "institutional" domain.
- **Reasoning:** Institutional positioning shifts are lagging but high-trust. 13D activist filing is a catalyst.

---

### FinVizClient
- **Category:** USEFUL
- **Location:** `mae_core/market/apis/finviz_client.py`
- **What it does:** Unusual volume screening, short squeeze candidates, insider trades (third source).
- **Evidence:** FinViz insider trades now wired via sensing_fetchers.py. "insider" domain contribution.
- **Reasoning:** Adds volume-based technical signals and a third insider source.

---

### EconomicCalendarClient
- **Category:** ESSENTIAL
- **Location:** `mae_core/market/apis/economic_calendar_client.py`
- **What it does:** FOMC/CPI/NFP/GDP event schedule. Used to suppress convergence alerts during high-noise macro event windows.
- **Evidence:** Injected directly into ConvergenceAlerter constructor (`economic_calendar=`). Active suppression.
- **Reasoning:** FOMC surprise moves invalidate convergence signals. Active suppression prevents false alerts during macro noise events.

---

### MassiveClient
- **Category:** ESSENTIAL
- **Location:** `mae_core/market/apis/massive_client.py`
- **What it does:** Polygon.io grouped daily OHLCV. Volume/price/gap anomaly detection across the full market.
- **Evidence:** Wired into sensing hook. Trust 0.90. Feeds volume anomaly signals into convergence buffer.
- **Reasoning:** Market-wide volume anomaly scan is the broadest signal source. Enables detecting unusual activity that individual ticker watches would miss.

---

### YahooRSSClient
- **Category:** USEFUL
- **Location:** `mae_core/market/apis/yahoo_rss_client.py`
- **What it does:** Per-ticker headline velocity and sentiment keyword extraction from Yahoo RSS.
- **Evidence:** Wired into sensing hook. "events" domain.
- **Reasoning:** News headline velocity is a fast signal. Low latency vs. EDGAR filings.

---

### BinanceFundingClient
- **Category:** USEFUL
- **Location:** `mae_core/market/apis/binance_funding_client.py`
- **What it does:** Binance futures funding rates. Extreme funding = directional signal for crypto.
- **Evidence:** Wired in Wave 2+3 bootstrap. "crypto" domain.
- **Reasoning:** Funding rate extremes predict mean-reversion moves in crypto. Narrow domain but valid signal.

---

### KalshiMarketClient
- **Category:** INERT
- **Location:** `mae_core/market/apis/kalshi_client.py`
- **What it does:** Kalshi prediction market data client.
- **Evidence:** Constructed in Wave 2+3 bootstrap. Not observed in any sensing hook source rotation or convergence record_signal call. SDK installed (`kalshi-python 2.1.4`) but per memory notes "not yet verified against demo env."
- **Reasoning:** No active signal injection into the pipeline observed. Constructed but not wired to produce signals.

---

### AlpacaClient
- **Category:** ESSENTIAL
- **Location:** `mae_core/market/apis/alpaca_client.py`
- **What it does:** Paper trading execution. Submits bracket orders for qualifying US equity convergence alerts.
- **Evidence:** `_submit_to_alpaca()` called in `market_hooks_trades.py` for qualifying alerts. Filters to US equities only.
- **Reasoning:** Execution bridge. Without this, all analysis produces no trades.

---

### ApeWisdomClient
- **Category:** INERT
- **Location:** `mae_core/market/apis/apewisdom.py`
- **What it does:** Reddit/social sentiment for meme stocks.
- **Evidence:** Constructed in bootstrap. Low trust (0.45). Listed as a ThreatDetector sacrificeable component. Not actively observed generating signals in the step hook cadence.
- **Reasoning:** Sentiment signal for a narrow meme-stock subset. ThreatDetector can sacrifice it under stress, suggesting it is considered expendable. Signal quality low.

---

### FINRAShortInterestClient
- **Category:** USEFUL
- **Location:** `mae_core/market/apis/finra_short_interest.py`
- **What it does:** FINRA short interest data with `speculative_short_ratio` enhancement.
- **Evidence:** Wired into sensing hook. Trust 0.85.
- **Reasoning:** Short squeeze conditions are a strong directional catalyst. High-trust source.

---

### SECEFTSClient (Full Text Search)
- **Category:** USEFUL
- **Location:** `mae_core/market/apis/sec_edgar/efts.py`
- **What it does:** SEC EDGAR full-text search across all filings.
- **Evidence:** Wired into bootstrap. Enables keyword-driven filing discovery beyond Form 4/8-K.
- **Reasoning:** Broadens EDGAR coverage. Can find unusual risk disclosures or announcements.

---

### AphaVantageClient (via PriceFetcher fallback)
- **Category:** USEFUL
- **Location:** `mae_core/market/apis/price_fetcher.py` (integrated)
- **What it does:** Alternative price data when yfinance fails.
- **Evidence:** PriceFetcher initializes with `alpha_vantage_key`. Fallback path only.
- **Reasoning:** Redundancy for price data. Not independently active.

---

### ClusterDetector
- **Category:** ESSENTIAL
- **Location:** `mae_core/market/edge/cluster_detector.py`
- **What it does:** Detects insider buying clusters (3+ insiders in 30-day window = high signal).
- **Evidence:** Constructed with Qdrant URL. Wired into sensing pipeline. Per research: "3+ insiders" threshold is the key cluster rule.
- **Reasoning:** Clustering validation of insider signals. Transforms raw Form 4s into a higher-conviction composite signal.

---

### PoliticianTracker
- **Category:** ESSENTIAL
- **Location:** `mae_core/market/edge/politician_tracker.py`
- **What it does:** Cross-references congressional trades with committee membership and related government contracts. 437 known politicians tracked.
- **Evidence:** Constructed in bootstrap. Wired into sensing pipeline.
- **Reasoning:** Committee membership + trade + contract = highest-conviction government signal. Transforms raw congressional data into edge.

---

### FilingTimeAnalyzer
- **Category:** USEFUL
- **Location:** `mae_core/market/edge/filing_time_analyzer.py`
- **What it does:** Behavioral signals from SEC filing timing (late filings, unusual timing patterns).
- **Evidence:** Constructed with Qdrant URL. VelocityDetector wired to it.
- **Reasoning:** Late filings statistically correlate with bad news. Filing time is an independent behavioral signal.

---

### ContractPredictor
- **Category:** USEFUL
- **Location:** `mae_core/market/edge/contract_predictor.py`
- **What it does:** Predicts government contract winners by correlating hiring blitz + insider trades + bid submissions.
- **Evidence:** Constructed with Qdrant URL. Retained as entity-level complement to ConvergenceAlerter.
- **Reasoning:** Pre-announcement winner prediction. High-value if correct. Requires cross-domain data it can receive.

---

### SessionSweepDetector
- **Category:** USEFUL
- **Location:** `mae_core/market/edge/session_sweep_detector.py`
- **What it does:** ICT/Smart Money Concepts — detects liquidity sweeps at session highs/lows (IFVG patterns).
- **Evidence:** Constructed. `_check_sweep_bypass()` called every 50 steps in the step hook. Bypass path at 0.40 quality gate.
- **Reasoning:** Filters out low-quality sweep signals at 44.4% win rate improvement. Contributes to "technical" domain.

---

### TAIndicators
- **Category:** ESSENTIAL
- **Location:** `mae_core/market/edge/ta_indicators.py`
- **What it does:** RSI, MACD, Bollinger Bands, Market Structure, Candlestick pattern detection (vectorized numpy/pandas).
- **Evidence:** Module reference on `ctx.ta_indicators`. Used by sensing hook and multiple clients. `compute_atr()` used by SignalTranslator for stop-loss/take-profit.
- **Reasoning:** Technical indicators are independent signal sources and essential for position sizing (ATR).

---

### OrderFlowDetector
- **Category:** USEFUL
- **Location:** `mae_core/market/edge/order_flow_detector.py`
- **What it does:** Detects institutional order flow imbalances from price/volume patterns.
- **Evidence:** Constructed (Wave 1 gift). Feeds into convergence buffer. Trust 0.60.
- **Reasoning:** Order flow is a near-real-time institutional activity signal. Moderate trust. Pipeline contributor.

---

### DeceptionDetector
- **Category:** ESSENTIAL
- **Location:** `mae_core/market/edge/deception_detector.py`
- **What it does:** Detects data manipulation signals — pump-and-dump patterns, coordinated social sentiment, wash trading indicators.
- **Evidence:** Constructed (Wave 2 gift). State persisted/restored on daemon restart. Publishes `CH_DECEPTION_DETECTED`. Bio systems (emotional, nociception, haven, inhibition, threat_detector) all subscribe to this channel.
- **Reasoning:** Active defense against bad signals entering the pipeline. When deception fires, caution rises and paper trading confidence is penalized. System health impact: critical.

---

### FractalResonanceDetector
- **Category:** INERT
- **Location:** `mae_core/market/edge/fractal_resonance.py`
- **What it does:** Looks for fractal price patterns (self-similar structures across timeframes).
- **Evidence:** Constructed (Wave 2 gift). Listed as ThreatDetector sacrificeable at priority 0.5. Trust 0.65. No observed active signal injection in step hooks reviewed.
- **Reasoning:** Theoretical edge from fractal price patterns. The fact that ThreatDetector can sacrifice it at medium priority suggests it is considered expendable. Expendable = INERT in practice.

---

### ThompsonSampler
- **Category:** ESSENTIAL
- **Location:** `mae_core/market/intelligence/thompson_sampler.py`
- **What it does:** 83 Beta distributions tracking per-source signal reliability. Bayesian explore/exploit weighting. Regime-aware forgetting.
- **Evidence:** Injected into ConvergenceAlerter (confidence weighting), OutcomeCollector (updates), KellyPositionSizer (win probability), HypothesisEngine. Stats published every 10 steps. Forget called every 75 steps gated on new outcomes.
- **Reasoning:** The Bayesian brain. Every confidence number in the system flows through Thompson distributions.

---

### ConvergenceAlerter
- **Category:** ESSENTIAL
- **Location:** `mae_core/market/intelligence/convergence_alerter.py`
- **What it does:** The crown jewel. Synthesizes signals from 30+ sources across 12 domains. Fires when 3+ independent domains align. Thompson-weighted geometric mean confidence. Domain independence correction.
- **Evidence:** `check_convergence()` called every step. `check_ticker_convergence()` called every step and every 50 steps for Kelly sizing. Receives economic_calendar suppression, world_model ripple effects, lag scoring, pattern_memory embedding.
- **Reasoning:** THE output stage. All pipeline effort flows here. No convergence = no alerts, no trades, no system value.

---

### VelocityDetector
- **Category:** USEFUL
- **Location:** `mae_core/market/intelligence/velocity_detector.py`
- **What it does:** Rate-of-change anomaly detection across all signal sources.
- **Evidence:** Scanned every 50 steps. Publishes `CH_VELOCITY_ANOMALY`. Bio systems (homeostasis, nociception) subscribe. Feeds into correlation tracker.
- **Reasoning:** Velocity anomalies indicate acceleration events (catalysts firing). Contributes to pipeline diagnostics and bio-system arousal.

---

### CorrelationTracker
- **Category:** USEFUL
- **Location:** `mae_core/market/intelligence/correlation_tracker.py`
- **What it does:** Cross-domain signal correlation tracking. Seeded from `lag_correlations.json`. Injected into ConvergenceAlerter for independence correction.
- **Evidence:** Seeded at bootstrap with lag data. Injected into ConvergenceAlerter (`_correlation_tracker`). Deque persistence.
- **Reasoning:** Domain independence correction is how MIDGE avoids over-counting correlated signals (e.g., macro+technical r=0.73). Critical for valid confidence calculation.

---

### GrangerAnalyzer
- **Category:** USEFUL
- **Location:** `mae_core/market/intelligence/granger_analyzer.py`
- **What it does:** Granger causality detection between signal time series. Runs every 500 steps, 15 tests, Bonferroni-corrected. Persists to `granger_causality.json`.
- **Evidence:** Slow cadence run confirmed in market_hooks_steps.py. Results inform lag_correlations which seed CorrelationTracker.
- **Reasoning:** Directional causality (not just correlation) distinguishes leading from coincident signals. Improves lag scoring in ConvergenceAlerter.

---

### LagCorrelationAnalyzer
- **Category:** USEFUL
- **Location:** `mae_core/market/intelligence/lag_correlation_analyzer.py`
- **What it does:** Analyzes historical signal archive for cross-domain lag relationships. Results feed `lag_correlations.json`.
- **Evidence:** Constructed with SignalArchiveReader dependency. Run in slow cadence (500 steps). Results loaded by ConvergenceAlerter for sequence scoring.
- **Reasoning:** Temporal domain ordering (which domain fires first) improves convergence confidence. Essential for time-is-the-moat thesis.

---

### RegimeClassifier
- **Category:** ESSENTIAL
- **Location:** `mae_core/market/intelligence/regime_classifier.py`
- **What it does:** Classifies market regime (bull/bear/volatile/sideways) using VIX and price data.
- **Evidence:** Injected into ThompsonSampler (regime-aware forgetting rates), OutcomeCollector (regime-stratified Thompson updates), HypothesisEngine, PostMortemReviewer. Called every 10 steps for stats.
- **Reasoning:** Regime gates Thompson forgetting. At volatile=0.90, 10% faster erosion. At sideways=0.97, 3% erosion. Wrong regime = miscalibrated Thompson.

---

### OutcomeTracker
- **Category:** ESSENTIAL
- **Location:** `mae_core/market/outcome_tracker.py`
- **What it does:** Monitors open predictions against market outcomes. Grades win/loss. Feeds Thompson Sampler.
- **Evidence:** Constructed with price_fetcher and thompson_sampler. Called in step cadence.
- **Reasoning:** Without outcome tracking, Thompson distributions never learn. The feedback loop that turns observations into signal reliability would be broken.

---

### OutcomeCollector
- **Category:** ESSENTIAL
- **Location:** `mae_core/market/intelligence/outcome_collector.py`
- **What it does:** Registers convergence alerts and pattern stacks for outcome tracking. Closes the Bayesian feedback loop. Registers both individual-source and combo-level distributions.
- **Evidence:** Constructed in sensing infrastructure setup. `ctx.outcome_collector = outcome_collector` explicitly set. Combo key registration in step hook for every alert.
- **Reasoning:** This is the learning closure. Without it, Thompson distributions stay at priors forever. Prior history: 4 bugs caused 81/83 distributions to stay at priors — all fixed.

---

### ThompsonCalibrator
- **Category:** USEFUL
- **Location:** `mae_core/market/intelligence/thompson_calibrator.py`
- **What it does:** Calibrates Thompson distributions for overconfidence/underconfidence. Run every 1000 steps.
- **Evidence:** Constructed with ThompsonSampler dependency. Slow cadence run confirmed.
- **Reasoning:** Prevents systematic bias in Thompson estimates. Without calibration, distributions drift from reality over long runs.

---

### KellyPositionSizer
- **Category:** ESSENTIAL
- **Location:** `mae_core/market/intelligence/kelly_position_sizer.py`
- **What it does:** Kelly Criterion position sizing using Thompson win probabilities.
- **Evidence:** Called every 50 steps on per-ticker convergence alerts. Recommendations published on EventBus.
- **Reasoning:** Position sizing determines return expectancy. Kelly formula at 50K account. Without this, all positions would be equal-weighted.

---

### HypothesisRegistry / HypothesisGenerator / HypothesisValidator / HypothesisEngine
- **Category:** ESSENTIAL (as a unit)
- **Location:** `mae_core/market/intelligence/hypothesis_*.py`
- **What it does:** RSI Layer 2. Lag findings → formal hypotheses with causal stories → adversarial DSR validation → probation → active → hibernated → retired lifecycle.
- **Evidence:** HypothesisEngine.step() called every step. Wired into CircadianRhythm CONSOLIDATION phase for extra cadence. Feeds into paper trading gate (Validator 4: recent hypothesis fire).
- **Reasoning:** The self-improving hypothesis loop. Transforms raw lag correlations into testable market theories. If a hypothesis fires on a ticker, it boosts paper trade approval.

---

### BacktestAnalyzer / BacktestScheduler
- **Category:** USEFUL
- **Location:** `mae_core/market/intelligence/backtest_analyzer.py`, `backtest_scheduler.py`
- **What it does:** Bridge between backtest results and RSI Layer 2. Turns backtest findings into formal hypotheses. Run every 5000 steps.
- **Evidence:** Constructed with HypothesisRegistry. HypothesisEngine receives `backtest_analyzer=`. Very slow cadence.
- **Reasoning:** Useful for importing externally run backtest results into the hypothesis loop. Low frequency means limited impact in short daemon runs.

---

### SignalArchiveReader
- **Category:** USEFUL
- **Location:** `mae_core/market/intelligence/signal_archive_reader.py`
- **What it does:** Reads 901+ days of archived JSONL signal files for lag analysis and archive warmup.
- **Evidence:** Dependency for LagCorrelationAnalyzer and GrangerAnalyzer. Archive warmup uses it (injects last 7 days of signals into convergence buffer at startup).
- **Reasoning:** Archive warmup prevents cold-start problem. Without it, MIDGE starts each daemon run with empty convergence buffer and must rebuild.

---

### PostMortemReviewer
- **Category:** USEFUL
- **Location:** `mae_core/market/intelligence/post_mortem.py`
- **What it does:** Analyzes why predictions succeeded or failed every 500 steps. Combo stats, domain ordering, timing accuracy, regime transitions, MFE/MAE. Pushes sequence-aware Thompson updates.
- **Evidence:** Run in slow cadence at 500 steps. Results in `post_mortem_insights.json`.
- **Reasoning:** Closes the "right thesis, wrong timing" loop. More sophisticated than basic outcome tracking.

---

### WorldModel (Market)
- **Category:** ESSENTIAL
- **Location:** `mae_core/market/intelligence/world_model.py`
- **What it does:** 114 nodes, 102 edges, 38 tickers. Curated causal chains (energy, macro, tech, defense, supply chain). `find_ripple_effects()` BFS. `find_root_causes()` reverse BFS.
- **Evidence:** Injected into ConvergenceAlerter (`_world_model`). Every convergence alert carries `ripple_effects`. Backward cascade discovery uses it. CascadeTracker uses it.
- **Reasoning:** Structural knowledge base. Transforms individual signals into cascade predictions. Key for the "inevitability surfacer" mission.

---

### CascadeTracker
- **Category:** USEFUL
- **Location:** `mae_core/market/intelligence/cascade_tracker.py`
- **What it does:** Tracks active causal chains as dominoes confirm. Energy ratio tracking. Publishes `CH_CASCADE_CONFIRMED`.
- **Evidence:** Wired in market_hooks.py. Confirmation events feed into sequential chain boost (cascade domain synthetic signals).
- **Reasoning:** Validates WorldModel predictions in real-time. When dominoes confirm, confidence in remaining predictions should increase.

---

### PatternLibrary / PatternWatcher / ExcavationDaemon
- **Category:** ESSENTIAL (as Pattern Archaeology unit)
- **Location:** `mae_core/market/archaeology/pattern_library.py`, `pattern_watcher.py`, `excavation_daemon.py`
- **What it does:** Reverse-engineers historical moves into domain-level templates. PatternWatcher detects live stacking. ExcavationDaemon continuously builds fingerprints/templates. 223K fingerprints, 43 templates (39 in memory).
- **Evidence:** PatternWatcher.check() called every 10 steps in sensing hook. Stacks feed paper trading gate (Validator 2). OutcomeCollector receives pattern stacks for Thompson feedback.
- **Reasoning:** Historical precedent matching. The entire Pattern Archaeology pipeline validates convergence alerts with historical analogs.

---

### ActiveTracker
- **Category:** ESSENTIAL
- **Location:** `mae_core/market/archaeology/active_tracker.py`
- **What it does:** Monitors live predictions — price-checks every 20 steps. Status transitions (tracking→confirming→confirmed|failed|expired). MFE/MAE tracking.
- **Evidence:** Constructed. Registered in sensing hook. Force-grades on terminal status. Writes plain-language updates.
- **Reasoning:** Without active tracking, MIDGE has no live feedback on whether predictions play out. Required for real-time outcome grading.

---

### AbsenceMonitor
- **Category:** USEFUL
- **Location:** `mae_core/market/intelligence/absence_monitor.py`
- **What it does:** Detects unexpectedly silent sources (e.g., insider trading normally fires daily but has been silent for 3 days). Bootstraps cadences from signal archive.
- **Evidence:** Constructed with archive_reader. Injected into sensing hook. Archive bootstrap attempted at startup.
- **Reasoning:** Silence is a signal. If SEC EDGAR stops producing Form 4s unexpectedly, that silence should trigger investigation.

---

### OctopusColony
- **Category:** USEFUL
- **Location:** `mae_core/network/octopus_colony.py`
- **What it does:** Multi-attention investigation system. Dispatches `investigate_partial` and `situation_check` tasks every 20 steps. Queries PatternLibrary + WorldModel for partial convergences. Persists developing situations.
- **Evidence:** Dispatches confirmed in market_hooks_steps_core.py every 20 steps. `_developing_situations` persisted to disk. `_wire_octopus_colony()` registers handlers.
- **Reasoning:** Bridges the gap between partial convergences (2 domains) and full convergences (3+ domains). Sustained attention on developing situations.

---

### DeepAnalyst
- **Category:** USEFUL
- **Location:** `mae_core/market/intelligence/deep_analyst.py`
- **What it does:** Synthesizes all historical data into a ranked list of "most likely near-term moves" — six scoring components (Thompson, template, WorldModel, lag, density, outcome). Produces `Inevitability` objects.
- **Evidence:** Constructed with thompson_sampler, pattern_library, world_model. `ctx.inevitabilities` read by paper trading gate (Validator 3). Run cadence not confirmed from bootstrap review — may be on-demand.
- **Reasoning:** The "inevitability surfacer" title made concrete. Provides a ranked synthesis across all data. However, run cadence unclear — if not periodically triggered, inevitabilities list may be stale.

---

### EventEmbedder / PatternMemory
- **Category:** USEFUL
- **Location:** `mae_core/market/intelligence/event_embedder.py`, `pattern_memory.py`
- **What it does:** Converts convergence alerts to Qdrant-stored semantic vectors (mxbai-embed-large/Ollama). `find_precedents()` used in paper trading gate (Validator 5).
- **Evidence:** PatternMemory set on ConvergenceAlerter. Embeds called per new alert in step hook. Paper trading gate uses `_pmem.find_precedents()` if available.
- **Reasoning:** Semantic precedent search enables "have I seen something like this before?" Degrades gracefully when Qdrant/Ollama unavailable.

---

### SituationBoard
- **Category:** USEFUL
- **Location:** `mae_core/market/intelligence/situation_board.py`
- **What it does:** Typed replacement for `ctx._market_advisory` dict. Thread-safe findings workspace with decay. Saved every 75 steps.
- **Evidence:** Constructed. Saved in `_write_convergence_heartbeat`. Loaded from disk on startup.
- **Reasoning:** Structured coordination surface. Enables multi-analyst findings to be queried together. Currently underutilized — analysts must actively publish to it.

---

### SocialTextAnalyzer
- **Category:** USEFUL
- **Location:** `mae_core/market/intelligence/social_text_analyzer.py`
- **What it does:** Reads StockTwits messages from RawStore SQLite, extracts keyword themes (options_flow, short_squeeze, earnings_play, macro_fear), sentiment intensity.
- **Evidence:** Constructed with `raw_store`. Trust 0.40 for source. Wired into sensing pipeline.
- **Reasoning:** Extracts structured market intent from unstructured social text. Better signal-to-noise than raw StockTwits scores.

---

### RawDataAnalyst
- **Category:** USEFUL
- **Location:** `mae_core/market/intelligence/raw_data_analyst.py`
- **What it does:** Cross-domain insight engine — reads raw SQLite stores, looks for patterns across domains.
- **Evidence:** Constructed with raw_store. Wiring to pipeline not confirmed from bootstrap review.
- **Reasoning:** Useful concept but active wiring not confirmed. May be in early integration stage.

---

### PortfolioTracker
- **Category:** USEFUL
- **Location:** `mae_core/market/intelligence/portfolio_tracker.py`
- **What it does:** Tracks paper trade positions and portfolio-level P&L.
- **Evidence:** Constructed (Wave 1 gift). Referenced in paper trading gate.
- **Reasoning:** Position awareness prevents over-concentration. Required once paper trading volume increases.

---

### CatalystCalendar
- **Category:** ESSENTIAL
- **Location:** `mae_core/market/intelligence/catalyst_calendar.py`
- **What it does:** Tracks earnings dates, analyst day events, product launches. Injected into ConvergenceAlerter.
- **Evidence:** Injected into ConvergenceAlerter (`_catalyst_calendar`). Earnings proximity boosts convergence confidence.
- **Reasoning:** Earnings catalysts are the primary activation events for convergence signals. A convergence alert 2 days before earnings has different timing value than 2 months before.

---

### CrossAssetConfirmer
- **Category:** USEFUL
- **Location:** `mae_core/market/intelligence/cross_asset_confirmer.py`
- **What it does:** Confirms equity signals with related cross-asset moves (e.g., oil equity + crude futures correlation).
- **Evidence:** Injected into ConvergenceAlerter (`_cross_asset_confirmer`).
- **Reasoning:** Cross-asset confirmation is an independent validator. An equity move confirmed by futures or bonds is higher conviction.

---

### ConsolidationEngine
- **Category:** USEFUL
- **Location:** `mae_core/market/intelligence/consolidation_engine.py`
- **What it does:** Consolidates Thompson distributions and hypothesis registry during circadian consolidation phase.
- **Evidence:** Constructed (Wave 2 gift) with thompson_sampler, hypothesis_registry, hypothesis_engine.
- **Reasoning:** Maintenance task. Prevents Thompson distribution entropy. Low urgency but important for long-run stability.

---

### PatternArchetypeEngine
- **Category:** INERT
- **Location:** `mae_core/market/intelligence/pattern_archetypes.py`
- **What it does:** Identifies high-level pattern archetypes (e.g., "accumulation before breakout").
- **Evidence:** Constructed (Wave 2 gift). No observed signal injection or step hook call in reviewed bootstrap code.
- **Reasoning:** Concept exists but pipeline integration not confirmed. No step hook, no EventBus subscription, no direct call in step hooks reviewed.

---

### SomaticAnticipation
- **Category:** USEFUL
- **Location:** `mae_core/market/intelligence/somatic_anticipation.py`
- **What it does:** Body-state anticipation system — tracks pre-convergence signal patterns and anticipates when full convergence is likely. State persisted.
- **Evidence:** State persisted/restored on daemon restart. Injected with endocrine system reference.
- **Reasoning:** Anticipatory system — helps MIDGE "feel" a convergence building before it fires. Maps well to the bio-organism metaphor.

---

### PatternCompletionEngine
- **Category:** INERT
- **Location:** `mae_core/market/intelligence/pattern_completion.py`
- **What it does:** Predicts likely completion of partial patterns based on historical templates.
- **Evidence:** Listed as ThreatDetector sacrificeable at priority 0.6. No step hook or active call confirmed in bootstrap review.
- **Reasoning:** Low priority + sacrificeable = inert. Not actively contributing to the pipeline in observed code paths.

---

### MotifDetector
- **Category:** USEFUL
- **Location:** `mae_core/market/intelligence/motif_detector.py`
- **What it does:** STUMPY streaming matrix profile for recurring price patterns (motifs) and anomalies (discords).
- **Evidence:** Called every 100 steps in step hook. Generates signals injected into ConvergenceAlerter as `source="motif_match"` or `"price_discord"`.
- **Reasoning:** Time-series motif detection is independent of fundamental/sentiment signals. Adds a pure-price domain signal.

---

### StreamingAnomalyDetector
- **Category:** USEFUL
- **Location:** `mae_core/market/intelligence/streaming_anomaly.py`
- **What it does:** River-based streaming anomaly detection on price/volume vectors.
- **Evidence:** Called every 100 steps alongside MotifDetector. Generates `"streaming_anomaly"` signals.
- **Reasoning:** Catches unusual statistical deviations in real-time. Complements MotifDetector.

---

### DriftDetector
- **Category:** USEFUL
- **Location:** `mae_core/market/intelligence/drift_detector.py`
- **What it does:** ADWIN concept drift detection on price_returns/volume/VIX/sentiment streams.
- **Evidence:** Called every 50 steps in step hook.
- **Reasoning:** Detects when the statistical properties of signals are changing — regime shifts before RegimeClassifier confirms them.

---

### ResourceGovernor
- **Category:** USEFUL
- **Location:** `mae_core/market/resource_governor.py`
- **What it does:** Self-governing API budget — manages rate limits and allocates API call capacity across sources.
- **Evidence:** Constructed. Cortisol-coupling DISABLED (fictional physiology). Operates independently.
- **Reasoning:** Prevents API ban through rate limit management. Essential for 24/7 daemon operation.

---

### CircuitBreaker
- **Category:** ESSENTIAL
- **Location:** `mae_core/market/intelligence/circuit_breaker.py`
- **What it does:** Per-source failure protection with exponential backoff. Injected into sensing hook.
- **Evidence:** Injected into hook (`hook._circuit_breaker`). Prevents cascading failures when API sources fail.
- **Reasoning:** Without circuit breakers, a single failed API call can retry in a tight loop, consuming resources and distorting signal timing.

---

### DrawdownMonitor
- **Category:** ESSENTIAL
- **Location:** `mae_core/market/intelligence/drawdown_monitor.py`
- **What it does:** Tracks portfolio drawdown, halts trading when circuit breaker threshold exceeded. Persistence on daemon flush.
- **Evidence:** State loaded at bootstrap. `is_trading_halted()` checked in paper trading gate.
- **Reasoning:** Critical risk control. Without this, MIDGE could execute into a loss spiral.

---

### SystemHealthMonitor
- **Category:** USEFUL
- **Location:** `mae_core/market/system_health_monitor.py`
- **What it does:** Tracks error/success rates for convergence_check, thompson, sensing, outcome_evaluation. Publishes health metrics.
- **Evidence:** `_shm.record_error()`/`record_success()` called in 8 try/except blocks in step hooks.
- **Reasoning:** Operational visibility. Required for daemon monitoring. Flags system degradation before failures become invisible.

---

### SelfMonitor
- **Category:** USEFUL
- **Location:** `mae_core/market/intelligence/self_monitor.py`
- **What it does:** Behavioral anomaly detection — flags unusual alert patterns (rapid-fire alerts, same-direction clustering). Suppresses alerting when anomalies detected.
- **Evidence:** `_sm.record_alert()` and `_sm.is_alerting_suppressed()` called in paper trading gate.
- **Reasoning:** Prevents feedback loops where MIDGE's own signals would reinforce themselves. Important safeguard.

---

### StepTimer
- **Category:** USEFUL
- **Location:** `mae_core/market/step_timer.py`
- **What it does:** Performance metabolism monitoring — times convergence_check, thompson_stats, velocity_scan, hypothesis_engine.
- **Evidence:** `_timer.track()` context manager used in step hooks.
- **Reasoning:** Identifies bottlenecks. Operational data for daemon performance tuning.

---

### InhabitantScheduler
- **Category:** USEFUL
- **Location:** `mae_core/scheduling/inhabitant_scheduler.py`
- **What it does:** Wall-clock cadence dispatcher for scheduled market tasks independent of simulation step cadence.
- **Evidence:** `_sched.start()` called in `_wire_sensing_hook`. Daemon thread started.
- **Reasoning:** Enables time-of-day scheduling (open/close market events). Decouples real-time market awareness from step count.

---

### GovernanceLogger
- **Category:** USEFUL
- **Location:** `mae_core/governance/governance_logger.py`
- **What it does:** Append-only audit trail for governance events on EventBus.
- **Evidence:** Constructed. EventBus connected.
- **Reasoning:** Audit trail for system decisions. Required for debugging and compliance. Low overhead.

---

### MarketClock
- **Category:** ESSENTIAL
- **Location:** (constructed in sensing infrastructure setup)
- **What it does:** Market hours awareness — tells MarketSensingHook which sources are active during current market session.
- **Evidence:** Injected into MarketSensingHook. Sensing hook applies `set_circadian_scale()` based on `ctx._circadian_activity`.
- **Reasoning:** Without market clock awareness, MIDGE fetches crypto-only sources during equity hours and vice versa, wasting capacity.

---

### SignalTranslator / FTMOEngine
- **Category:** USEFUL
- **Location:** `mae_core/market/execution/signal_translator.py`, `ftmo_engine.py`
- **What it does:** Translates ConvergenceAlerts into ExecutableSignal(entry/stop/target/size). ATR-based stops. FTMOEngine simulates FTMO challenge constraints for backtesting.
- **Evidence:** `_translate_and_log_executable_signal()` called in paper trading gate for approved trades.
- **Reasoning:** Bridge between signal and execution. Without translation, convergence alerts have no entry/stop/target.

---

### TieredAlerters (tactical/strategic/thematic)
- **Category:** USEFUL
- **Location:** `mae_core/market/intelligence/convergence_alerter.py` (instances)
- **What it does:** Three separate ConvergenceAlerter instances with different time windows (48h, 21d, 90d).
- **Evidence:** Constructed in sensing infrastructure. Queried every 10 steps. Results stored in `_market_advisory`.
- **Reasoning:** Multi-timeframe confluence. A signal appearing on all three timeframes is stronger than a signal on only one.

---

### PlainLanguageFormatter
- **Category:** USEFUL
- **Location:** `mae_core/market/plain_language.py`
- **What it does:** Formats convergence alerts and pattern stacks in zero-jargon 5-section human-readable form. Writes to `alerts_human.jsonl`.
- **Evidence:** Called for pattern stacks in sensing hook. Called for paper trades.
- **Reasoning:** Guiding Light reads these. The entire analysis is useless if it cannot be communicated clearly.

---

## Group 2: Infrastructure Systems (Backbone, Foundation, Support)

---

### EventBus
- **Category:** ESSENTIAL
- **Location:** `mae_core/backbone/event_bus.py`
- **What it does:** Publish/subscribe messaging backbone. Every inter-system communication runs through it.
- **Evidence:** `ctx.bus` is the organism's nervous system. Used by every system.
- **Reasoning:** The single communication medium. Without it, no system can talk to any other.

---

### HolonRegistry
- **Category:** USEFUL
- **Location:** `mae_core/backbone/holon_protocol.py`
- **What it does:** Registry of organism entities at multiple scales (organism, organ, subsystem, agent). Provides fractal self-awareness.
- **Evidence:** Every system registered. Used by IntegrationMeter, TopologyAnalyzer, FractalACT.
- **Reasoning:** Required for Law 3 compliance. Provides the structural map that enables self-knowledge.

---

### ConnectionRegistry
- **Category:** USEFUL
- **Location:** `mae_core/backbone/connection_registry.py`
- **What it does:** Tracks all inter-system connections with triadic witness requirements.
- **Evidence:** 428 connections registered. Used by TriadicVerifier, TopologyAnalyzer.
- **Reasoning:** Law 1 enforcement. Auditing only — does not block connections. Overhead is registration, not execution.

---

### SomaticMap
- **Category:** USEFUL
- **Location:** `mae_core/emergent/somatic_map.py`
- **What it does:** Dependency graph / body awareness registry. All systems register here. AutoHealer reads it for proactive scanning.
- **Evidence:** AutoHealer injected with `_somatic_map`. Used in bootstrap audit.
- **Reasoning:** System health topology. AutoHealer uses it to find recovery targets.

---

### ApiGateway
- **Category:** USEFUL
- **Location:** `mae_core/external/api_gateway.py`
- **What it does:** External API access with BoundaryMembrane + InputValidator immune system. MarketDataProvider registered as "market_data" provider.
- **Evidence:** Stepped every step. MarketDataProvider registered. Providers checked at step cadence.
- **Reasoning:** Validated external data access. BoundaryMembrane prevents trust bypass.

---

### BoundaryMembrane
- **Category:** USEFUL
- **Location:** `mae_core/defense/boundary_membrane.py`
- **What it does:** Trust-gate for external data. Registers source trust levels (0.40-0.95). All market sources registered.
- **Evidence:** All 33 market sources registered with trust scores in bootstrap.
- **Reasoning:** Differentiates trusted (0.95 FRED) from less trusted (0.40 StockTwits) sources at the boundary.

---

### InputValidator
- **Category:** USEFUL
- **Location:** `mae_core/defense/input_validator.py`
- **What it does:** Validates and sanitizes external inputs before processing.
- **Evidence:** Constructed with EventBus. Used by ApiGateway and BoundaryMembrane.
- **Reasoning:** Prevents malformed data from corrupting signal processing. Immune system.

---

### PearlDefense
- **Category:** USEFUL
- **Location:** `mae_core/defense/pearl_defense.py`
- **What it does:** Defense against adversarial inputs — wraps problematic data.
- **Evidence:** Step hook registered at Layer 14.
- **Reasoning:** Secondary input defense layer. Stepped but low overhead.

---

### AutoHealer
- **Category:** USEFUL
- **Location:** `mae_core/emergent/auto_healer.py`
- **What it does:** Detects and recovers from system failures. Reads SomaticMap for proactive scanning. SenescenceManager triggers it.
- **Evidence:** Step hook registered at Layer 14. SenescenceManager wired to it.
- **Reasoning:** Self-healing daemon capability. When a system fails, AutoHealer attempts recovery rather than leaving it broken.

---

### ThreatDetector
- **Category:** USEFUL
- **Location:** `mae_core/defense/threat_detector.py`
- **What it does:** Detects threats and can sacrifice expendable systems. Deception quill registered. Sacrificeable components: finnhub_websocket (0.2), apewisdom (0.3), fractal_resonance (0.5), pattern_completion (0.6).
- **Evidence:** Deception quill registered in bio_market_wiring_a.py. Stepped at Layer 14.
- **Reasoning:** System-level defense under extreme stress. Prioritizes core pipeline over supplementary systems.

---

### HAVEN (HavenRiskCoordinator)
- **Category:** USEFUL
- **Location:** `mae_core/learning/haven.py`
- **What it does:** Immune check on signal sources. Market deception flags accumulate per source; cleared by successful outcomes.
- **Evidence:** `_haven_market_flags` dict accumulated in bio_market_wiring_b.py. Set on ConvergenceAlerter via `set_haven_flags()`.
- **Reasoning:** Source-level trust immune system. Distinguishes deceptive sources (flagged, suspected) from reliable ones.

---

### DeepMemoryStore / MemoryBridge / PatternDistiller
- **Category:** USEFUL
- **Location:** `mae_core/memory/deep_memory.py`, `memory_bridge.py`, `pattern_distiller.py`
- **What it does:** Qdrant-backed permanent memory + bridge for agent episodic memory + pattern extraction.
- **Evidence:** Constructed if Qdrant available. Agents get `_memory_bridge` reference. PatternCortex uses memory_bridge.
- **Reasoning:** Long-term memory for the agent-side intelligence. Distinct from market EventEmbedder/PatternMemory. Agent memories could inform market decisions through the PatternCortex pathway.

---

### IntegrationMeter
- **Category:** INERT (for market pipeline)
- **Location:** `mae_core/backbone/integration_meter.py`
- **What it does:** Computes IIT Phi (consciousness integration measure) and Markov blanket. Runs every 89 steps.
- **Evidence:** Stepped every 89 steps. Publishes `integration.phi_measurement`. Endocrine, ArousalRegulator, GlobalWorkspace respond. No market output.
- **Reasoning:** Phi-driven modulation affects agent behavior (cortisol/arousal) but there is no direct path from Phi measurement to market signal processing or confidence. Phi measures the organism's internal coherence, not market signals. For market pipeline efficiency: overhead with no pipeline output.

---

### TopologyAnalyzer
- **Category:** INERT (for market pipeline)
- **Location:** `mae_core/backbone/topology_analyzer.py`
- **What it does:** Computes clustering, path length, sigma, density of the connection network. Every 55 steps.
- **Evidence:** Stepped every 55 steps. No observable market pipeline output.
- **Reasoning:** Graph metric computation for architectural validation. Zero market intelligence contribution. Purely structural auditing.

---

### TriadicVerifier
- **Category:** INERT (for market pipeline)
- **Location:** `mae_core/backbone/triadic_verifiers.py`
- **What it does:** Verifies Laman/Peirce/Hegel/Simmel proofs on the connection graph. Every 89 steps.
- **Evidence:** Stepped every 89 steps. Mathematical proof checking.
- **Reasoning:** Compliance verification for Mae's mathematical laws. Zero market intelligence contribution.

---

### TriadEnforcer / TriadWatchdog / TriadAuditor
- **Category:** INERT (for market pipeline)
- **Location:** `mae_core/backbone/triad_enforcer.py`, `triad_watchdog.py`, `triad_auditor.py`
- **What it does:** Rule of 3/5 compliance checking and bypass monitoring. Every 50 steps.
- **Evidence:** Audit hook fires every 50 steps. Advisory only — does not block.
- **Reasoning:** Architectural law enforcement. Zero market intelligence contribution.

---

### Bootstrap Audit (Layer 32)
- **Category:** INERT (for market pipeline)
- **Location:** `mae_core/bootstrap/audit.py`
- **What it does:** One-time triadic audit of SomaticMap, ConnectionRegistry, HolonRegistry at bootstrap.
- **Evidence:** Runs once at startup.
- **Reasoning:** Startup health check. Zero ongoing cost.

---

### FractalGenerator
- **Category:** INERT (for market pipeline)
- **Location:** `mae_core/backbone/fractal_generator.py`
- **What it does:** Generates K3 triadic structure for the organism hierarchy.
- **Evidence:** Used at bootstrap to generate_triad() calls. No runtime market role.
- **Reasoning:** Structure generation at bootstrap only. Zero runtime market overhead.

---

## Group 3: Bio-Metaphor Systems (Layers 26-30)

These systems were inherited from mae-core. The bio_market_wiring layers gave most of them legitimate market jobs. The audit below assesses what they actually do for the market pipeline versus what they cost in step overhead.

---

### EmotionalSystem
- **Category:** USEFUL (wired)
- **Location:** `mae_core/coordination/emotional_system.py`
- **What it does:** Convergence → surprise/fear state modulation. Deception → fear spike. Via bio_market_wiring.
- **Evidence:** Subscribed to `CH_CONVERGENCE`, `CH_DECEPTION_DETECTED`, `CH_DUAL_CONFIRMATION`.
- **Reasoning:** State affects agent decision-making (emotional state biases action selection). Indirect market pipeline value. Market input → emotional state → agent behavior → trades. Thin but present causal path.

---

### HomeostasisRegulator
- **Category:** USEFUL (wired)
- **Location:** `mae_core/coordination/homeostasis.py`
- **What it does:** Bearish convergence → elevated threat setpoint. Velocity anomaly → elevated processing load. Steps every step.
- **Evidence:** Subscribed to `CH_CONVERGENCE`, `CH_VELOCITY_ANOMALY`.
- **Reasoning:** Risk-aware homeostasis. Affects organism "health" state that feeds OrganismState.

---

### ArousalRegulator
- **Category:** USEFUL (wired)
- **Location:** `mae_core/coordination/arousal_regulator.py`
- **What it does:** Prediction wins/losses → reward signal. Phi-measurement also feeds it.
- **Evidence:** Subscribed to `CH_PREDICTION_RESULT`, `CH_CONVERGENCE`. Phi hook wired.
- **Reasoning:** Yerkes-Dodson arousal for trading performance. Elevated arousal during high-confidence periods; reduced during losses. Affects agent interaction cadence.

---

### InhibitionSystem
- **Category:** ESSENTIAL (wired — directly affects trading)
- **Location:** `mae_core/coordination/inhibition_system.py`
- **What it does:** Deception events raise `_market_caution` (0.0-1.0). High confidence convergence lowers caution. Caution directly penalizes paper trade confidence in the trading gate (up to 30% penalty).
- **Evidence:** `_market_caution` is read in `_run_paper_trading_gate`. Values >0.3 reduce alert confidence. Values that push confidence below threshold block the trade entirely.
- **Reasoning:** This is the only bio system with a DIRECT, QUANTITATIVE effect on trade approval. It is not merely modulatory — it can and does block trades.

---

### CuriosityDrive
- **Category:** USEFUL (wired)
- **Location:** `mae_core/learning/curiosity.py`
- **What it does:** Partial convergence → exploration bonus. Novel hypothesis → exploration bonus. Low-confidence pattern stack → boost.
- **Evidence:** Subscribed to `CH_PARTIAL_CONVERGENCE`, `CH_HYPOTHESIS_DISCOVERED`, `CH_PATTERN_STACK_DETECTED`.
- **Reasoning:** Exploration vs. exploitation balance. May influence which sources get investigated. Indirect.

---

### NociceptionSystem
- **Category:** USEFUL (wired)
- **Location:** `mae_core/communication/nociception.py`
- **What it does:** Deception → acute pain. Prediction failure → referred pain. Velocity anomaly → chronic pain. Steps every step.
- **Evidence:** Subscribed to `CH_DECEPTION_DETECTED`, `CH_PREDICTION_RESULT`, `CH_VELOCITY_ANOMALY`. Feeds TriageClassifier via constructor.
- **Reasoning:** Pain state feeds TriageClassifier (signal priority routing) and OrganismState (body awareness). Pain from market failures = heightened urgency in signal triage.

---

### MetacognitionMonitor
- **Category:** USEFUL (wired)
- **Location:** `mae_core/cognition/metacognition.py`
- **What it does:** Tracks confidence calibration: predicted vs actual outcome per convergence alert. Drives learning rate multiplier adjustments to agents' VDN/WorldModel.
- **Evidence:** Subscribed to `CH_PREDICTION_RESULT`. `should_adjust_learning_rate()` bridges to agent learning rates.
- **Reasoning:** Meta-learning bridge. Confidence calibration → learning rate → agent model adaptation. Indirect but real pathway.

---

### CircadianRhythm
- **Category:** USEFUL (wired, but throttle DISABLED)
- **Location:** `mae_core/coordination/circadian_rhythm.py`
- **What it does:** Phase-based activity modulation. MIDGE: throttle disabled, `_circadian_activity` pinned to 1.0. CONSOLIDATION phase triggers extra hypothesis engine step. REST phase triggers excavation daemon step.
- **Evidence:** Pinning confirmed in bio_market_wiring_b.py line 107. Phase-change callback still triggers hypothesis and excavation steps.
- **Reasoning:** The throttling is neutralized. The residual value: circadian phase-change triggers hypothesis consolidation and excavation steps — legitimate market jobs. Overhead: minimal (the circuit runs but `_circadian_activity=1.0` prevents harm). The system still STEPS every step (adds to step cost) but the throttle danger is neutralized.

---

### EndocrineSystem
- **Category:** USEFUL (wired, cortisol coupling DISABLED)
- **Location:** `mae_core/coordination/endocrine_system.py`
- **What it does:** Hormone modulation of agent behavior. Cortisol → ResourceGovernor coupling DISABLED. Phi → cortisol still active (phi-driven modulation wired).
- **Evidence:** Cortisol-ResourceGovernor coupling commented out in bio_market_wiring.py. Phi coupling wired at Layer 30b.
- **Reasoning:** The harmful coupling is disabled. Residual function: phase transitions and phi-driven cortisol still modulate agent behavior. Low overhead.

---

### EnergyReserve
- **Category:** HARMFUL (neutralized)
- **Location:** `mae_core/memory/energy_reserve.py`
- **What it does:** Simulates energy storage with leptin signaling. Publishes CH_STARVATION when reserves drop below 10.
- **Evidence:** `_reserves` pinned to 100.0 (was 50.0). Original 50.0 caused immediate starvation → InhibitionSystem NoGo suppression → agent actions blocked. Fix documented in code comment.
- **Reasoning:** The fix (pinning to 100.0) neutralized the harm, but the underlying behavior (draining reserves, potentially re-triggering starvation during long runs) remains in the code. Still runs every step. At 100.0 start and no drain re-enabled, harm is suppressed but not eliminated. Category: neutralized harm. Should be audited for whether drain re-enables.

---

### OrganismState
- **Category:** USEFUL (wired, but partially)
- **Location:** `mae_core/coordination/organism_state.py`
- **What it does:** Aggregates all bio system signals into a unified body state. Provides reflex overrides for emergencies. Agents call `get_body_state()`. DecisionRouter calls `get_reflex_override()`.
- **Evidence:** Steps every step. Agent references injected at Layer 29c. `_energy_critical` flag (from EnergyReserve starvation) was the old harm vector — now neutralized.
- **Reasoning:** Body state awareness affects agent decisions. With harmful bio systems neutralized, OrganismState reflects a more stable picture. Still runs every step.

---

### RespiratorySystem
- **Category:** INERT (wired but effect minimal)
- **Location:** `mae_core/coordination/respiratory_system.py`
- **What it does:** Breathing metaphor. Tier 5 wiring gives it a market job via bio_market_wiring_extended_b.py.
- **Evidence:** Runs every step. Tier 5 = "purpose emerges through market connection." Market job is minimal.
- **Reasoning:** Runs every step for minimal market intelligence value. Overhead without proportionate benefit.

---

### DigestiveSystem
- **Category:** INERT (for market pipeline)
- **Location:** `mae_core/coordination/digestive_system.py`
- **What it does:** Nutrient absorption metaphor. Tier 4 wiring gives it a data processing metaphor job.
- **Evidence:** Runs every step. Tier 4 market job.
- **Reasoning:** Metaphorical only. No market signal contribution confirmed in pipeline trace.

---

### ThermoregulationSystem
- **Category:** INERT (for market pipeline)
- **Location:** `mae_core/coordination/thermoregulation.py`
- **What it does:** Temperature regulation metaphor. Tier 5 wiring.
- **Evidence:** Runs every step.
- **Reasoning:** Tier 5 = aspirational. No confirmed market pipeline contribution.

---

### VestibularSystem
- **Category:** INERT (for market pipeline)
- **Location:** `mae_core/coordination/vestibular_system.py`
- **What it does:** Balance/orientation metaphor. Tier 5 wiring.
- **Evidence:** Runs every step.
- **Reasoning:** No confirmed market pipeline contribution.

---

### CirculatorySystem
- **Category:** INERT (for market pipeline)
- **Location:** `mae_core/substrate/circulatory_system.py`
- **What it does:** Information flow through agent network. Tier 4 wiring.
- **Evidence:** Runs every step.
- **Reasoning:** Meta-system level. No direct market signal path confirmed.

---

### RenalFilter
- **Category:** INERT (for market pipeline)
- **Location:** `mae_core/defense/renal_filter.py`
- **What it does:** Filters waste from system messages. Tier 4 wiring.
- **Evidence:** Runs every step.
- **Reasoning:** Filtering metaphor. No confirmed market pipeline contribution.

---

### Microbiome
- **Category:** INERT (for market pipeline)
- **Location:** `mae_core/emergent/microbiome.py`
- **What it does:** Symbiotic signal processing populations. Step-driven feeding (every step).
- **Evidence:** Elaborate feed mechanism wired. No confirmed market output path.
- **Reasoning:** Interesting metaphor with no confirmed market intelligence output.

---

### LymphaticSystem
- **Category:** INERT (for market pipeline)
- **Location:** `mae_core/emergent/lymphatic_system.py`
- **What it does:** Waste removal and immune drainage. Tier 4 wiring.
- **Evidence:** Runs every step.
- **Reasoning:** No confirmed market pipeline contribution.

---

### SenescenceManager
- **Category:** USEFUL (indirect)
- **Location:** `mae_core/emergent/senescence.py`
- **What it does:** Triggers AutoHealer rejuvenation when systems degrade. Wired to AutoHealer via EventBus.
- **Evidence:** `_senescence_to_healing` wired in Layer 29b. Runs every step.
- **Reasoning:** Keeps long-running systems healthy. Indirect support for pipeline uptime.

---

### BoundaryMembrane (bio variant)
- **Category:** USEFUL (see also ApiGateway BoundaryMembrane)
- **Location:** `mae_core/defense/boundary_membrane.py`
- **What it does:** All market source trust registration happens here.
- **Evidence:** 33 market sources registered with trust scores.
- **Reasoning:** Same system serves dual role: input validation + source trust registry.

---

### ReproductiveSystem
- **Category:** INERT (for market pipeline)
- **Location:** `mae_core/morphogenesis/reproductive_system.py`
- **What it does:** Agent spawning based on population load. Tier 4 wiring.
- **Evidence:** Runs every step (load-sampling hook).
- **Reasoning:** Spawns new agents when load is high. No direct market signal contribution.

---

### TheoryOfMind
- **Category:** INERT (for market pipeline)
- **Location:** `mae_core/cognition/theory_of_mind.py`
- **What it does:** Models other agents' internal states. No market wiring.
- **Evidence:** Runs every step. Persistence wired. No market channel subscriptions observed.
- **Reasoning:** Useful for multi-agent coordination but no market domain application. Pure overhead for a trading daemon.

---

### EmotionalSystem (fully)
Already covered above — USEFUL (wired).

---

### ProprioceptionSystem
- **Category:** USEFUL (light)
- **Location:** `mae_core/emergent/proprioception.py`
- **What it does:** Self-position awareness — fractal structure awareness. Tier 5 market wiring.
- **Evidence:** Runs every step. Tier 5.
- **Reasoning:** Architectural self-awareness with minimal market intelligence value.

---

### ClosureCoordinator
- **Category:** INERT (for market pipeline)
- **Location:** `mae_core/backbone/autopoietic_closure.py`
- **What it does:** Verifies autopoietic closure at subsystem/organ/organism scales. Every 5/8/13 steps.
- **Evidence:** Frequent stepping.
- **Reasoning:** Architectural verification only. Zero market intelligence contribution.

---

### RedifferentiationMonitor
- **Category:** INERT (for market pipeline)
- **Location:** `mae_core/agents/redifferentiation_triggers.py`
- **What it does:** Monitors and triggers stem cell role changes. Every 21 steps.
- **Evidence:** Cadenced run.
- **Reasoning:** Agent management overhead. Relevant if MIDGE actively reassigns market roles. Otherwise minimal market value.

---

### MitosisMonitor
- **Category:** INERT (for market pipeline)
- **Location:** `mae_core/agents/mitosis.py`
- **What it does:** Autopoietic agent production loop. Every 13 steps.
- **Evidence:** Cadenced run. `max_agents = num_agents * 2`.
- **Reasoning:** Spawns agents. Irrelevant to market pipeline unless spawned agents contribute market-specific tasks.

---

### MorphogenesisCoordinator / OrganBuilder
- **Category:** INERT (for market pipeline)
- **Location:** `mae_core/morphogenesis/coordinator.py`, `organ_builder.py`
- **What it does:** Dynamic organ formation. Runs every step.
- **Evidence:** Step hook registered.
- **Reasoning:** Architectural growth system. No confirmed market intelligence output.

---

### CollectiveDreamPlanner
- **Category:** INERT (for market pipeline)
- **Location:** `mae_core/cognition/collective_dream.py`
- **What it does:** Collective agent planning/simulation. Market wiring nudges expertise weights on convergence.
- **Evidence:** Market wiring in bio_market_wiring_b.py adjusts dreamer expertise by 0.02 on convergence. No planning output wired to market actions.
- **Reasoning:** Expertise weight nudge is minimal and unverified in its downstream effect. Planning outputs go nowhere.

---

### WorldlinePlanner
- **Category:** INERT (for market pipeline)
- **Location:** `mae_core/planning/worldline_planner.py`
- **What it does:** Multi-step future planning using WorldModel and TemporalMemory.
- **Evidence:** Agents receive `_worldline_planner` reference. No observed call to worldline planning in market pipeline.
- **Reasoning:** Agent planning that is not market-aware. Worldline plans would need to incorporate market signals to be useful.

---

### TemporalMemory
- **Category:** INERT (for market pipeline)
- **Location:** `mae_core/planning/temporal_memory.py`
- **What it does:** Episodic temporal memory for sequence learning. Used by WorldlinePlanner.
- **Evidence:** Constructed and wired to WorldlinePlanner.
- **Reasoning:** No market signal integration path observed.

---

### ValidatedImagination
- **Category:** INERT (for market pipeline)
- **Location:** `mae_core/cognition/validated_imagination.py`
- **What it does:** Validates imagined futures before acting.
- **Evidence:** Stepped. No market channel subscriptions.
- **Reasoning:** Agent-side planning without market data integration.

---

### PredictiveField
- **Category:** INERT (for market pipeline)
- **Location:** `mae_core/communication/predictive_field.py`
- **What it does:** Agent-level prediction field. Stepped every step.
- **Evidence:** Step hook registered at Layer 14.
- **Reasoning:** Agent coordination field. No market signal path.

---

### PhysarumOptimizer
- **Category:** INERT (for market pipeline)
- **Location:** `mae_core/substrate/physarum_optimizer.py`
- **What it does:** Physarum-inspired substrate network optimization.
- **Evidence:** Stepped every step.
- **Reasoning:** Network topology optimization. Zero market intelligence contribution.

---

### GNNCommunicator
- **Category:** INERT (for market pipeline)
- **Location:** `mae_core/communication/gnn_communicator.py`
- **What it does:** GNN-based intelligent routing for agent messages. RoutingOptimizer runs every 21 steps.
- **Evidence:** GNN edge weights optimized. FRL trust updates.
- **Reasoning:** Agent-to-agent routing. Not connected to market signal processing.

---

### MycelialSubstrate
- **Category:** INERT (for market pipeline)
- **Location:** `mae_core/substrate/mycelial_substrate.py`
- **What it does:** Scale-free network topology for agent connections.
- **Evidence:** Constructed. Used by Physarum and GNN.
- **Reasoning:** Network topology. No market signal function.

---

### SignalBus
- **Category:** INERT (for market pipeline)
- **Location:** `mae_core/communication/signal_bus.py`
- **What it does:** Typed signal bus for agent signaling (DANGER/OPPORTUNITY/CONVERGENCE).
- **Evidence:** Agents register handlers for CONVERGENCE signal type. Market agents receive this.
- **Reasoning:** Has a market hook (CONVERGENCE signal type) but the agent response to CONVERGENCE is generic, not market-specific. Marginal.

---

### StigmergicEnvironment
- **Category:** USEFUL (wired)
- **Location:** `mae_core/communication/stigmergy.py`
- **What it does:** Market ticker trail markers — convergence deposits pheromone per ticker. Prediction outcomes deposit success/danger markers.
- **Evidence:** Market wiring subscribes to `CH_CONVERGENCE` and `CH_PREDICTION_RESULT`. Evaporation every 50 steps.
- **Reasoning:** Agents following trails converge on high-activity tickers. Enables emergent agent attention routing via environmental markers.

---

### QuorumSpace
- **Category:** USEFUL (wired)
- **Location:** `mae_core/communication/quorum_space.py`
- **What it does:** Organism-level vote on convergence signals per ticker. Convergence, pattern stacks, and dual confirmations deposit signals.
- **Evidence:** Subscribed to `CH_CONVERGENCE`, `CH_PATTERN_STACK_DETECTED`, `CH_DUAL_CONFIRMATION`. Injected into ConvergenceAlerter constructor (`quorum_space=`).
- **Reasoning:** Quorum sensing = emergent consensus. Multiple independent confirmations of a ticker create a quorum. Meaningful for multi-system validation.

---

### KnowledgeBase / TransferLearningEngine / MAMLLearner
- **Category:** INERT (for market pipeline)
- **Location:** `mae_core/learning/knowledge_base.py`, etc.
- **What it does:** Cross-task knowledge sharing and MAML meta-learning for agent task performance.
- **Evidence:** All constructed and injected into agents.
- **Reasoning:** Generic agent learning — not market-aware. No market signal integration confirmed.

---

### CapabilityDiscovery
- **Category:** INERT (for market pipeline)
- **Location:** `mae_core/emergent/capability_discovery.py`
- **What it does:** Discovers new agent capabilities through exploration.
- **Evidence:** Stepped at Layer 14.
- **Reasoning:** Generic agent improvement. Not market-specific.

---

### PatternBus / PatternCortex / AttentionalGate / PatternConsolidator
- **Category:** INERT (CRITICAL ARCHITECTURAL GAP — see note)
- **Location:** `mae_core/patterns/pattern_bus.py`, etc.
- **What it does:** AttentionalGate gates patterns by prediction error. PatternBus processes events through 11+ translators (including MarketConvergenceTranslator). PatternCortex produces advisories with threat/opportunity levels. These publish to `pattern.advisory` on EventBus.
- **Evidence:** MarketConvergenceTranslator wired. Pattern advisory published to EventBus. Agents receive `_pattern_advisory_ref`. BUT: the advisory affects agent decision-making (threat/opportunity/novelty levels) in agent step cycles. Market signals DO flow into this pipeline via MarketConvergenceTranslator.
- **Reasoning:** Partially connected. MarketConvergenceTranslator means market convergence events feed this pipeline. The pattern advisory modulates agent behavior. HOWEVER: agent decisions at this level are not market trading decisions — they are generic action selections in the Mae simulation. The advisory does not reach the paper trading gate or convergence alerter. This is the DISCONNECTED PIPELINE noted in the Evolution Blueprint.
- **Net verdict for market pipeline:** INERT for direct market output. The connection exists in one direction (market → pattern bus) but the pattern advisory does not flow back to market decision-making. Agents respond to pattern advisories with generic Mesa actions, not market trades.

---

## Group 4: Agent Mixins (14 on every MycelialAgent)

---

### HolonMixin
- **Category:** USEFUL
- **What it does:** Fractal self-awareness — each agent registers as a holon, can introspect its scale and position.
- **Evidence:** Initialized last on every agent. HolonRegistry registration.
- **Reasoning:** Provides the `know_self/know_up/know_down/know_peers` capability required by Law 3.

---

### SensingLifecycleMixin (`_predict`, `_attend`, `_observe`)
- **Category:** USEFUL
- **What it does:** Sensing phase of agent step — predicts, attends, observes environment. `_observe` reads TaskPool tasks.
- **Evidence:** Called every step.
- **Reasoning:** Agents observe their task environment. If market tasks are in TaskPool, they are picked up here.

---

### DecisionActionLifecycleMixin (`_decide`, `_act`)
- **Category:** USEFUL
- **What it does:** Decision + action. Agents select and execute actions from available options.
- **Evidence:** Called every step.
- **Reasoning:** Agent execution. Market-differentiated agents (SEC_WATCHER, MARKET_ANALYST) have api_call_enabled: False in Oracle shutdown, so they select non-API tasks.

---

### LearningLifecycleMixin (`_learn`, `_manage_goals`)
- **Category:** INERT (for market pipeline)
- **What it does:** Agent learning from action outcomes. Manages goals (detects impasses, celebrates progress).
- **Evidence:** Called every step.
- **Reasoning:** Generic agent learning. VDN/WorldModel learning from task rewards, not market signals.

---

### CommunicationLifecycleMixin (`_communicate`, `_broadcast`, `_regulate`)
- **Category:** USEFUL (partial)
- **What it does:** `_communicate` deposits stigmergy markers. `_broadcast` fires GWT competitive ignition. `_regulate` maintains arousal homeostasis.
- **Evidence:** `_broadcast` calls PatternBus. Market advisory from PatternBus influences `_broadcast` output.
- **Reasoning:** `_communicate` writes stigmergy markers (market-wired). `_broadcast` connects to PatternBus which has MarketConvergenceTranslator. Thin market pathway.

---

### ConvergenceMixin
- **Category:** INERT (for market pipeline)
- **What it does:** Tracks agent convergence to a satisfaction threshold. `has_reached_convergence` flag.
- **Evidence:** Checked in agent step.
- **Reasoning:** Generic agent convergence criterion (not market convergence). Name collision with ConvergenceAlerter — different concept entirely.

---

### GamificationMixin
- **Category:** INERT (for market pipeline)
- **What it does:** Points, levels, achievements, streaks for agent performance.
- **Evidence:** Tracked per agent.
- **Reasoning:** Motivational layer for generic agents. No market signal or output.

---

### SignalProcessingMixin + SignalPriorityResolver
- **Category:** USEFUL
- **What it does:** Triage queued signals by priority before agent processing. Handles DANGER/OPPORTUNITY/CONVERGENCE/COLLABORATION_REQUEST/KNOWLEDGE_SHARE signal types.
- **Evidence:** `resolver.process()` called at start of every agent step. CONVERGENCE signal type wired.
- **Reasoning:** Agent-level signal triage. CONVERGENCE signals from the market could route through here. Fast reflexes for high-priority market signals.

---

### StigmergyMixin
- **Category:** USEFUL (market-wired)
- **What it does:** Reads and deposits stigmergy markers. Market convergence and prediction outcomes deposited as markers.
- **Evidence:** Agent deposits markers via `_communicate`. Market wiring deposits convergence markers at ticker positions.
- **Reasoning:** Agents follow convergence marker trails. Emergent attention direction.

---

### GNNCommunicationMixin
- **Category:** INERT (for market pipeline)
- **What it does:** GNN-based intelligent peer message routing.
- **Evidence:** Used between agents for knowledge sharing.
- **Reasoning:** Agent-to-agent routing. Not market-connected.

---

### TransferLearningMixin (including MAMLLearner)
- **Category:** INERT (for market pipeline)
- **What it does:** Cross-task knowledge transfer and MAML fast adaptation.
- **Evidence:** Per-agent transfer tracking.
- **Reasoning:** Generic agent learning optimization. No market domain application.

---

### EpisodicMemoryMixin
- **Category:** INERT (for market pipeline)
- **What it does:** Per-agent episodic memory for task sequences.
- **Evidence:** Per-agent. Memory bridge optionally connected.
- **Reasoning:** Agent task memory. Not connected to market signal recall.

---

### CollectiveConsensusMixin
- **Category:** USEFUL (light)
- **What it does:** Quorum sensing for agent-level consensus. QuorumSpace integration.
- **Evidence:** QuorumSpace wired to market channels in bio_market_wiring_b.py.
- **Reasoning:** Market signals reach QuorumSpace. Agents contributing to quorum on convergence alerts add organism-level consensus validation.

---

### AdvancedFeaturesMixin
- **Category:** INERT (for market pipeline)
- **What it does:** World model planning, morphogenesis, decision router.
- **Evidence:** World model planning requires `world_model_enabled=True` (False by default per config).
- **Reasoning:** Disabled by default. Even when enabled, it plans generic task actions, not market trades.

---

## Group 5: Bootstrap Layers Summary

| Layers | Systems Created | Market Pipeline Role |
|--------|----------------|---------------------|
| 1: Foundation | MycelialModel, EventBus, HolonRegistry | ESSENTIAL — EventBus is the nervous system |
| 2: Coordination | CircadianRhythm, EndocrineSystem | USEFUL — both wired to market channels |
| 3: Triad Enforcement | TriadEnforcer, Watchdog, Auditor | INERT for market |
| 4: Substrate | MycelialSubstrate, PhysarumOptimizer | INERT for market |
| 5: Communication | SignalBus, GNNCommunicator, Stigmergy, QuorumSpace, PredictiveField | USEFUL (Stigmergy, QuorumSpace wired) |
| 6: Learning | KnowledgeBase, Transfer, MAML, Curiosity, HAVEN, Imitation | USEFUL (Curiosity, HAVEN wired) |
| 7: Defense | ThreatDetector, InputValidator, PearlDefense | USEFUL |
| 8: Cognition | WorldModel (core), CollectiveDream, ValidatedImagination, CausalEngine | INERT for market |
| 9: Emergent | AutoHealer, CapabilityDiscovery, SomaticMap | USEFUL (AutoHealer, SomaticMap) |
| 10: Morphogenesis | MorphogenesisCoordinator, OrganBuilder | INERT for market |
| 11: Planning | TemporalMemory, WorldlinePlanner | INERT for market |
| 12-13: Agents | 12 MycelialAgents + per-agent systems | USEFUL (agents process market tasks) |
| 14: Step hooks | PredictiveField, AutoHealer, Physarum, PearlDefense, GNN | Mixed |
| 15-16: Wiring | EventBus cross-wiring, endocrine consumers, HAVEN validators | Mixed |
| 17-21: Advanced wiring | Connection registry, holon awareness, fractal generator, stem cells | INERT for market |
| 22: Deep Memory | DeepMemoryStore, MemoryBridge, PatternDistiller | USEFUL (if Qdrant available) |
| 23: Pattern Ecosystem | AttentionalGate, PatternBus, PatternCortex, PatternConsolidator | INERT for market output (disconnected pipeline) |
| 24: Action Environment | TaskPool | USEFUL — agents pick up market tasks |
| 25-25d: Fractal ACT + meters | OrganismAction, IntegrationMeter, TopologyAnalyzer, TriadicVerifier | INERT for market |
| 26: Metabolic | DigestiveSystem, Respiratory, Vestibular, Homeostasis, Thermoregulation, EnergyReserve, Circulatory, Renal, Microbiome | HARMFUL (neutralized) or INERT |
| 27: Social Cognition | EmotionalSystem, TheoryOfMind, MetacognitionMonitor, Nociception, Proprioception | USEFUL (most wired) except TheoryOfMind |
| 28: Maintenance | LymphaticSystem, SenescenceManager, BoundaryMembrane, ReproductiveSystem | Mixed |
| 29-30: Lifecycle | OrganismState, RedifferentiationMonitor, MitosisMonitor, ClosureCoordinator, InhibitionSystem, GoalManager, ArousalRegulator | Mixed |
| 31: External API | ApiGateway, LLM providers, data providers | USEFUL |
| 32: Bootstrap Audit | TriadAuditor, Health validators | INERT (startup only) |
| 33: Market Intelligence | All market systems (see Group 1) | ESSENTIAL/USEFUL |

---

## Critical Findings

### Finding 1: Disconnected Attention Pipeline
**Evidence:** PatternCortex produces `pattern.advisory` (threat/opportunity/novelty levels) every step. This advisory reaches agents via `_pattern_advisory_ref` and modulates `_attend()` behavior. Market signals feed PatternBus via MarketConvergenceTranslator. BUT: agents' attentional decisions based on pattern advisory do not connect to paper trading gate, convergence alerter, or any market output. Market data flows IN but advisory does not flow OUT to market decisions.
**Impact:** The organism's cognitive attention system and its market intelligence system are two separate loops. They share EventBus messages but produce no shared output.

### Finding 2: High Bio-System Step Overhead
**Evidence:** Layer 26 alone adds 9 step hooks (digestive, respiratory, vestibular, homeostasis, thermoregulation, energy_reserve, circulatory, renal, microbiome). Layer 27 adds 5 more. Plus complex Lambda closures. Many of these run every step with zero market pipeline contribution.
**Estimate:** ~20-25 bio systems run every step for effectively zero market intelligence output. Each adds CPU overhead, memory reads, and increases the risk of exception propagation.

### Finding 3: Neutralized but Active Harmful Systems
**Evidence:** EnergyReserve pinned to 100.0 (still drains every step). CircadianRhythm throttle disabled (still steps every step). Cortisol-ResourceGovernor coupling commented out (cortisol still released via phi). These systems are NEUTRALIZED but not removed — they still consume step budget.
**Impact:** The harms are suppressed, but the overhead remains.

### Finding 4: KalshiMarketClient and PatternArchetypeEngine are Constructed But Not Wired
**Evidence:** KalshiMarketClient constructed in Wave 2+3 but not in any sensing hook source rotation or convergence record_signal call. PatternArchetypeEngine constructed but no step hook or active call observed in reviewed bootstrap.
**Impact:** These consume memory and initialization time for zero current market intelligence contribution.

### Finding 5: DeepAnalyst Lacks Confirmed Step Cadence
**Evidence:** DeepAnalyst constructed with full scoring capability but its run cadence was not confirmed in bootstrap review. `ctx.inevitabilities` is read by the paper trading gate — but if DeepAnalyst is not periodically run, the inevitabilities list stays stale and Validator 3 in the trading gate never fires.
**Impact:** If not periodically run, the six-component "inevitability surfacer" synthesis is not contributing to trade approvals.

### Finding 6: SituationBoard is Underutilized
**Evidence:** SituationBoard constructed as typed replacement for `_market_advisory`. But market analysts (DeepAnalyst, CascadeTracker, etc.) must actively call `situation_board.publish()` to write to it. No confirmed calls from these systems to the board in reviewed code.
**Impact:** The board exists but may be nearly empty at runtime, with most intelligence still flowing through the untyped `_market_advisory` dict.

---

## Recommended Priority for Other Auditors

For the perspective and dissent auditors: the highest-value areas to examine are:

1. **The disconnected attention pipeline** (PatternCortex output vs. market trading gate) — is this actually disconnected or did I miss a connection?
2. **DeepAnalyst run cadence** — where does it get triggered? Is `ctx.inevitabilities` being populated?
3. **Bio-system overhead cost** — is the step overhead from 20+ inert bio systems measurable and material?
4. **KalshiMarketClient** — is there any wiring I missed?
5. **SituationBoard populate calls** — do any market systems actually publish to it?
