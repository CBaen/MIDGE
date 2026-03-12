# Team 4 Findings: Memory & Learning Completeness
## Date: 2026-03-12
## Researcher: Team Member 4

---

## Battle-Tested Approaches

### JSONL Signal Archive (data/midge/signals/)
The signal archive is the most heavily used memory layer. It is written to by `sensing_hook.py` after every sensing cycle and read by:
- **StartupWarmup** (`startup_warmup.py`) — reads last 7 days of archive on daemon restart, injects signals into convergence buffer. This is the correct pattern: the organism starts from knowledge, not zero.
- **SignalArchiveReader** (`signal_archive_reader.py`) — lazy-loading indexed reader used by LagCorrelationAnalyzer and ThompsonCalibrator. Builds `_by_source` and `_by_symbol` indexes.
- **OutcomeCollector.collect_from_archives()** — retroactively registers archived signals as predictions for Thompson feedback.
- **DeepAnalyst** (`deep_analyst.py`) — reads the archive to build signal density scores as one of six scoring components for Inevitability generation.
- **ArchiveScanner** (`archive_scanner.py`) — startup diagnostic; reads last 30 days to log coverage before warmup runs.

This layer is genuinely alive: data flows in, is read back at startup, and feeds multiple downstream systems.

### Thompson Sampling Learning Loop
The Thompson feedback loop is substantially closed. The chain is:
1. Signal arrives → `OutcomeCollector.register_signals()` registers as prediction
2. `OutcomeCollector.register_convergence_alert()` registers combo-level predictions
3. `OutcomeCollector.register_pattern_stack()` registers archaeology stacks
4. `OutcomeCollector.evaluate()` (called every step) → `OutcomeTracker.check_pending_outcomes()` → price check → Thompson update
5. `PostMortemReviewer` (every 500 steps) reads `outcomes.jsonl`, computes sequence-aware stats, feeds back to Thompson

This chain is fully wired. The 4 compounding bugs identified in March 2026 were fixed. `replay_from_history()` on ThompsonSampler can rebuild from 13,190 historical updates.

### Pattern Template Feedback Loop
Also substantially closed:
1. ExcavationDaemon generates fingerprints and templates stored in `pattern_templates.jsonl` and `pattern_library.jsonl`
2. PatternWatcher matches live signals against templates, fires `CH_PATTERN_STACK_DETECTED`
3. `OutcomeCollector.register_pattern_stack()` registers stacks with template IDs
4. `_on_outcome_graded()` callback fires `PatternLibrary.update_outcome(template_id=, won=)` for each graded template

Templates accumulate win/loss data over time. The loop is real.

### Raw Data Analyst (SQLite → Enriched Signals)
`RawDataAnalyst` (`raw_data_analyst.py`) is the primary SQLite reader during daemon operation. It runs every 100 steps and executes four cross-domain routines that actually read from the raw stores:
1. `_analyze_insider_price_context` — reads `get_insider_trades()` + `get_price_snapshots()`
2. `_analyze_fred_macro_regime` — reads `get_fred_observations()`
3. `_analyze_cross_domain_preconvergence` — reads `get_insider_trades()` + `get_trends_history()` + `get_yahoo_headlines()`
4. `_analyze_funding_rate_squeeze` — reads `get_binance_funding_history()`

Enriched signals from these routines are injected directly into `convergence_alerter.record_signal()`. This is a genuine reader of the SQLite raw store.

---

## Novel Approaches

### PatternMemory (Qdrant semantic layer) — Write works, read is barely wired
The semantic memory layer (`pattern_memory.py` + `event_embedder.py`) has a rich API: `remember_convergence_alert()`, `remember_inevitability()`, `find_precedents()`, `get_pattern_context()`, `search_insider_buys()`, `search_high_confidence_alerts()`. The infrastructure is sophisticated.

**What is actually wired for writes:**
- `market_hooks_steps_core.py` line 107: calls `_pm.remember_convergence_alert(_alert)` when a convergence alert fires
- `market_hooks_steps.py` lines 422-428: calls `_pmem.remember_inevitability(_iv)` for top 5 DeepAnalyst Inevitabilities

**What is NOT wired for reads during daemon operation:**
- `find_precedents()` — defined, not called anywhere in bootstrap hooks
- `get_pattern_context()` — defined, not called anywhere in bootstrap hooks
- `recall_similar()` — not called anywhere in bootstrap hooks
- `search_insider_buys()` / `search_high_confidence_alerts()` — not called anywhere

Qdrant is a **write-heavy, read-starved** system during daemon operation. It accumulates semantic embeddings of convergence alerts and inevitabilities but nothing queries them to enrich live analysis. The OctopusColony investigation pipeline (`investigate_partial`) is the only place that logically should call `find_precedents()` — but the actual investigation code queries `PatternLibrary` and `WorldModel`, not `PatternMemory`.

### DeepAnalyst Inevitabilities JSONL — Write only
`market_hooks_steps.py` writes top 10 inevitabilities to `data/midge/inevitabilities.jsonl` (but only if the file exists — the path is created lazily on first write). This file is never read back by any system. DeepAnalyst has its own in-memory state rebuilt from archives every run; it does not read its own prior output.

---

## Emerging Approaches

### Post-Mortem Reviewer (outcomes.jsonl → Thompson)
`post_mortem.py` reads `data/market/outcomes.jsonl`, computes aggregate statistics (combo win rates, domain ordering sequences, MFE/MAE patterns, regime transition failures), writes `data/market/post_mortem_insights.json`, and feeds sequence-aware Thompson updates. This is genuinely novel: it feeds the *ordering* of domain signals (which domain fires first matters) back into Thompson distributions, creating a `domain1>>domain2` sequence key.

The `post_mortem_insights.json` output file, however, is **never read back** by any other system. It is written as a diagnostic artifact but not consumed.

---

## Gaps and Unknowns

### A. Raw Store SQLite: Write-Only Stores

The raw store has **39 store methods** across 5 mixin files and **16 get methods**. The following store methods have **no corresponding get method and no known reader in the codebase**:

| Store Method | Database | Status |
|---|---|---|
| `store_usda_data()` | fred.db (or new) | Write-only — no get method, no reader |
| `store_edgar_filings()` | sec_edgar.db | Write-only — no get method (separate from `get_insider_trades` which reads form4/openinsider) |
| `store_congress_bills()` | legislation.db | Write-only — no get method |
| `store_coincap_assets()` | crypto.db | Write-only — no get method |
| `store_apewisdom_sentiment()` | finnhub.db | Write-only — no get method |
| `store_finnhub_ticks()` | finnhub.db | Write-only — no get method |
| `store_finnhub_sentiment()` | finnhub.db | Write-only — no get method |
| `store_finnhub_economic()` | finnhub.db | Write-only — no get method (only earnings has a get) |
| `store_finviz_short_squeeze()` | finviz.db | Write-only — no get method |
| `store_finviz_unusual_volume()` | finviz.db | Write-only — no get method |
| `store_polygon_ticker_details()` | massive.db | Write-only — no get method |
| `store_massive_bars()` | massive.db | Write-only — no get method |
| `store_job_postings()` | contracts.db | Write-only — no get method |
| `store_finra_short_volume()` | (gov DB) | Write-only — no get method |
| `store_fred_yields()` | fred.db | Write-only (separate from `store_fred_observations` and `get_fred_observations`) |

**Get methods that exist but are only called by RawDataAnalyst (5 of 16 total get methods):**
- `get_insider_trades()` — called by `_analyze_insider_price_context` and `_analyze_cross_domain_preconvergence`
- `get_price_snapshots()` — called by `_analyze_insider_price_context`
- `get_fred_observations()` — called by `_analyze_fred_macro_regime`
- `get_trends_history()` — called by `_analyze_cross_domain_preconvergence`
- `get_yahoo_headlines()` — called by `_analyze_cross_domain_preconvergence`
- `get_binance_funding_history()` — called by `_analyze_funding_rate_squeeze`

**Get methods that exist but have NO callers found in daemon code:**
- `get_congressional_trades()` (government.db) — has a get method, zero callers outside tests
- `get_cot_history()` — called only by `cot_client.py` itself for own use, not by any analyst
- `get_eia_observations()` — exists, no callers
- `get_vix_history()` — exists, no callers
- `get_coingecko_history()` — exists, no callers
- `get_sam_opportunities()` — exists, no callers
- `get_usaspending_contracts()` — exists, no callers
- `get_kalshi_market_history()` — exists, no callers
- `get_finnhub_earnings()` — exists, no callers in bootstrap
- `get_edgar_filings()` — does NOT exist (edgar 13F stored but no get method)

**Summary:** 14 of 16 existing get methods are unused by the daemon. Only 6 get methods are called (all by RawDataAnalyst). The original audit finding that raw_store is "24 of 25 write-only" still substantially holds.

### B. Qdrant — Write-heavy, read-starved

Qdrant receives writes from two places during daemon operation:
- Convergence alerts → `remember_convergence_alert()` in `market_hooks_steps_core.py`
- Inevitabilities → `remember_inevitability()` in `market_hooks_steps.py`

No part of the daemon runtime calls any read method on `PatternMemory` or `EventEmbedder`. The `find_historical_precedents()`, `get_pattern_context()`, `find_precedents()`, `recall_similar()` methods exist but are never invoked by any hook or step function. Qdrant is currently functioning as an append-only log with no retrieval.

**Note:** Qdrant also requires Ollama (`mxbai-embed-large` model) running on port 11434. If either service is down, `PatternMemory._available = False` and all operations silently no-op. There is no monitoring of whether this is actually working.

### C. JSONL Files — Complete Map

**Files in data/midge/ (non-signals):**

| File | Writer | Reader | Notes |
|---|---|---|---|
| `alerts.jsonl` | Not found in daemon code | Not found | May be legacy — no writer discovered |
| `alerts_human.jsonl` | `plain_language.write_plain_alert()` (called from sensing + steps hooks) | Human only — no code reader | Write-only for machines |
| `paper_trades.jsonl` | `_write_paper_trade()` in market_hooks_trades.py | `PortfolioTracker` reads on refresh; dedup state loaded on startup | Live cycle: write + read |
| `paper_trades_bypass.jsonl` | `_check_sweep_bypass()` in market_hooks_trades.py | Not found in daemon | Write-only |
| `executable_signals.jsonl` | `_translate_and_log_executable_signal()` in market_hooks_trades.py | Not found in daemon | Write-only |
| `hypothesis_activity.jsonl` | `market_actions.log_hypothesis_activity()` | Not found in daemon | Write-only |
| `convergence_state.json` | `_write_convergence_heartbeat()` every 100 steps | Not found in daemon | External monitoring only |
| `active_tracking.json` | `ActiveTracker` persistence | `ActiveTracker` loads on startup | Read/write |
| `inevitabilities.jsonl` | `market_hooks_steps.py` DeepAnalyst block | Not found in daemon | Write-only for machines |
| `replay_results.json` | `replay_history.py` CLI | Not found in daemon | External only |
| `vector_store.pkl` | `semantic_retriever.py` via mae_core model | `semantic_retriever.py` loads on startup | Core mae system, not market |

**Files in data/market/:**

| File | Writer | Reader | Notes |
|---|---|---|---|
| `thompson_distributions.json` | `ThompsonSampler._save_distributions()` | `ThompsonSampler._load_distributions()` on startup | Live read/write |
| `thompson_history.jsonl` | `ThompsonSampler` on each update | `ThompsonSampler.replay_from_history()` for recovery | Write + emergency read |
| `predictions.jsonl` | `OutcomeTracker.record_prediction()` | `OutcomeTracker.check_pending_outcomes()` (loads pending) | Live read/write |
| `outcomes.jsonl` | `OutcomeTracker` when grading | `PostMortemReviewer.review()` every 500 steps | Live read/write |
| `hypotheses.jsonl` | `HypothesisRegistry._append_event()` | `HypothesisRegistry.load()` on startup | Live read/write |
| `pattern_templates.jsonl` | `PatternLibrary.persist_batch()` | `PatternLibrary._load()` on startup | Live read/write |
| `pattern_library.jsonl` | `PatternLibrary` (fingerprints) | ID set loaded on startup; full load only on rebuild | Partial live read |
| `discovery_log.jsonl` | `convergence_detection.py` on novel discovery | `convergence_models.read_recent_discoveries()` | Read by ConvergenceAlerter |
| `config_history.jsonl` | `learning_config.update_config()` | Not read by daemon | Write-only (audit trail) |
| `position_sizing_log.jsonl` | `KellyPositionSizer` on recommendation | Not read by daemon | Write-only (audit trail) |
| `governance_log.jsonl` | `GovernanceLogger` via EventBus | Not read by daemon | Write-only (audit trail) |
| `post_mortem_insights.json` | `PostMortemReviewer` every 500 steps | Not read by any other system | Write-only (human review) |

### D. EventBus Channels — Publishers and Subscribers

**Channels with both publishers AND subscribers (functioning):**
- `CH_CONVERGENCE` — published by `market_hooks_steps_core.py`; subscribed by endocrine system, bio wiring (6+ subscribers), cascade tracker, seq chain boost
- `CH_SIGNAL_INGESTED` — published by `market_hooks_sensing.py`; subscribed by hypothesis engine, causal watch, cascade signal check (3 subscribers)
- `CH_VELOCITY_ANOMALY` — published by steps core; subscribed by bio wiring systems
- `CH_DECEPTION_DETECTED` — published by DeceptionDetector; subscribed by multiple bio wiring systems
- `CH_PREDICTION_RESULT` — published implicitly; subscribed by bio systems
- `CH_CASCADE_CONFIRMED` — published by CascadeTracker; subscribed by forward chain boost
- `CH_HYPOTHESIS_PROMOTED/RETIRED` — published by HypothesisRegistry; subscribed by endocrine coupling
- `CH_HYPOTHESIS_FIRED` — published by HypothesisEngine; subscribed by focused attention boost
- `CH_PATTERN_STACK_DETECTED` — published by PatternWatcher; subscribed by bio wiring
- `CH_PARTIAL_CONVERGENCE` — published by ConvergenceAlerter; subscribed by OctopusColony + bio wiring
- `CH_DUAL_CONFIRMATION` — published by sensing hook; subscribed by bio wiring
- `CH_KELLY_SIZING` — published by steps core; subscribed by `_on_kelly_sizing` (stores on ctx)
- `CH_CAUSAL_WATCH` — published by causal watch handler; subscriber **NOT FOUND** — shouting into void

**Channels with publishers but NO subscribers found:**
- `CH_CAUSAL_WATCH` (`market.intel.causal_watch`) — published when WorldModel maps a signal to downstream effects; no subscriber found in any bootstrap file. The causal predictions are never consumed by any downstream system.
- `market.intel.motif_detected` — published when STUMPY detects a motif; no subscriber found
- `market.intel.streaming_anomaly` — published when drift anomaly detected inline; no subscriber found
- `market.intel.drift_detected` — published by `_run_drift_detector`; no subscriber found
- `market.intel.lag_finding` — published by lag correlation analysis; no subscriber found
- `market.intel.granger_finding` — published by Granger analysis; no subscriber found
- `market.intel.deep_analysis` — published by DeepAnalyst block; no subscriber found

**Channels defined but publish/subscribe status unclear:**
- `CH_CLUSTER_DETECTED`, `CH_POLITICIAN_TRADE`, `CH_FILING_ANOMALY`, `CH_CONTRACT_PREDICTED`, `CH_SESSION_SWEEP`, `CH_LAG_FINDING`, `CH_ACTIONABLE`, `CH_THOMPSON_STATS`, `CH_EXIT_SIGNAL`, `CH_CONSOLIDATION_COMPLETE`, `CH_SOMATIC_ANTICIPATION`, `CH_PATTERN_COMPLETED`, `CH_DRAWDOWN_WARNING`, `CH_TRADING_HALTED`, `CH_TRADING_RESUMED`, `CH_HEALTH_TIER_CHANGE`, `CH_BEHAVIORAL_ANOMALY`, `CH_HYPOTHESIS_DISCOVERED`, `CH_CONTRADICTION_DETECTED`, `CH_ABSENCE_DETECTED`, `CH_BACKTEST_REFRESHED`, `CH_GATE_ADJUSTED`, `CH_META_ADJUSTED`

### E. Neo4j — Not wired at all

Zero references to Neo4j, py2neo, neo4j-driver, or `bolt://` anywhere in the codebase. Neo4j is installed as a Docker container (ports 7474/7687 per MEMORY.md) but is completely unwired. MIDGE uses `WorldModel` (`world_model.py`) — a pure Python in-memory NetworkX graph — as its causal knowledge graph. Neo4j provides no functionality to the current system.

**WorldModel vs. Neo4j gap:** WorldModel has 114 nodes, 102 edges, 38 tickers — a curated static graph. Every restart, it is reconstructed from hardcoded Python. No learned causal relationships are persisted to any graph database. CascadeTracker's `record_outcome()` method updates an in-memory `confirmed/expired` flag on chains but these outcomes are not written back to WorldModel's edge weights.

### F. DuckDB — Not wired

Zero references to DuckDB anywhere in the codebase. `RawDataAnalyst` is the cross-domain analyst that logically should use DuckDB for cross-SQLite analytical queries, but it uses direct SQLite connections via `RawStore._get_conn(domain)` instead. The MEMORY.md note "DuckDB — analytical queries across SQLite (installed, unclear if wired)" is now clear: it is not wired.

### G. Thompson Learning Loop — Functionally closed with caveats

The Thompson learning loop is closed in code. Signal → prediction registration → price check → Thompson update → better confidence weighting. The 4 bugs from March 9 were fixed.

**Active concerns:**
- The `combo:` level distributions (registered by `register_convergence_alert`) feed combo-key updates into Thompson, but the ConvergenceAlerter calculates confidence using *individual source* distributions. The combo feedback flows to a combo key (e.g. `combo:insider+macro+technical`) but nothing reads these combo keys back when calculating confidence for the next convergence. There is a mismatch between what is learned and what is used.
- `PostMortemReviewer` writes sequence-aware keys (`domain1>>domain2`) to Thompson but the ConvergenceAlerter's confidence engine does not read sequence-keyed distributions — it reads per-source distributions. The post-mortem learning may be going nowhere actionable.
- `post_mortem_insights.json` is written but never consumed by any learning system.

### H. Pattern Templates — Feedback loop technically closed but outcome window mismatch

Template outcome feedback is wired:
- `OutcomeCollector.register_pattern_stack()` stores template IDs in prediction metadata
- `_on_outcome_graded()` callback updates `PatternLibrary.update_outcome(template_id, won)`

**Concern:** Pattern stacks register predictions with a dynamic window (median of template `expected_move_window_days`, clamped 3-30 days). However, templates are built from historical excavation, and their `expected_move_window_days` reflects the timing of historical patterns. For live templates with no graded instances, the window defaults to 14 days. Many templates have `wins=0, losses=0` — they are never being graded because the window hasn't expired or the patterns aren't firing.

With only 39 templates and the daemon stopped for extended periods, template learning is progressing extremely slowly.

---

## Missing Cross-Layer Bridges

### Bridge 1: Qdrant semantic memory → Live convergence enrichment (MISSING)
Convergence alerts are written to Qdrant but the live analysis pipeline never asks "have we seen this before?" `get_pattern_context()` and `find_precedents()` exist but have zero callers in daemon hooks. The OctopusColony investigation pipeline queries PatternLibrary (file-based) and WorldModel (in-memory) — not PatternMemory (Qdrant). The semantic memory layer is accumulating data with no consumers.

**Where it should be added:** In the `investigate_partial` OctopusColony handler, before submitting a situation as "new," call `ctx.pattern_memory.find_precedents(ticker, signals)` and include the results in the investigation context.

### Bridge 2: Raw Store → Cross-domain analysis (SEVERELY INCOMPLETE)
`RawDataAnalyst` reads 6 of 16 available get methods. The remaining 10 get methods cover:
- Congressional trades (buy/sell patterns before committee votes)
- COT positioning extremes (futures market positioning as a contrarian indicator)
- EIA inventory surprises (energy storage draws as price predictors)
- VIX term structure (fear gauge signals)
- CoinGecko price history (crypto market cycles)
- SAM.gov contract opportunities (pre-award signals)
- USASpending contracts (post-award confirmation)
- Kalshi market prices (prediction market implied probability)
- Finnhub earnings calendars (catalyst timing)
- Yahoo headlines (already partially used via raw_store, but Finnhub earnings unused)

None of these are read by any analyst during daemon operation. They exist as data accumulation only.

### Bridge 3: Causal watch channel → Subscriber (MISSING)
`CH_CAUSAL_WATCH` is published when a signal maps to WorldModel downstream effects. This is the "inevitability detection" layer — noticing dominos before they fall. But nothing subscribes to this channel. The causal predictions are computed and published but immediately lost. A subscriber should register these as developing situations in OctopusColony or inject them as synthetic signals into the convergence engine.

### Bridge 4: post_mortem_insights.json → Learning adjustment (MISSING)
PostMortemReviewer writes structured findings about why predictions succeed or fail (which combos win, which orderings fail, what MFE/MAE patterns exist). This is computed every 500 steps but never read by any other system. The insights should flow back to:
- ConvergenceAlerter to adjust domain independence assumptions
- LearningConfig to update decay rates for consistently-failing domains
- HypothesisGenerator to bias toward sequences that post-mortem identifies as working

### Bridge 5: Neo4j as persisted causal graph (NEVER WIRED)
WorldModel's edge weights never update from confirmed/expired cascade outcomes. CascadeTracker tracks whether predictions come true but the feedback never writes back to update WorldModel's `strength` values. A bridge from CascadeTracker's outcome data → WorldModel edge weight updates → Neo4j persistence would make the causal graph a genuinely learning structure.

### Bridge 6: USDA/COT/VIX/EIA raw store → Signal generation (MISSING)
Five entire data domains (agriculture via USDA, futures positioning via COT, volatility structure via VIX, energy inventories via EIA, prediction markets via Kalshi) are being stored in SQLite but are not read back for signal generation by any analyst. USDA is a unique real-economy signal with no equivalent in the current pipeline. COT managed money positioning is a known contrarian indicator. These are dead data stores.

---

## Synthesis

MIDGE has a well-designed memory architecture with 7+ distinct layers, but **the ratio of write connections to read connections is approximately 4:1.** Most data is being collected and stored with diligence, but very little is being read back for learning or analysis.

**What is genuinely working (data flows in AND out):**
1. Signal archive → Startup warmup (7-day cold-start recovery)
2. Thompson distributions → Confidence weighting (after the March fixes)
3. Pattern templates → PatternWatcher (live template matching)
4. Paper trades → PortfolioTracker (position awareness)
5. Predictions/outcomes → Thompson updates (feedback loop closed)
6. Discovery log → ConvergenceAlerter (novel pattern seeding)
7. Hypotheses log → HypothesisRegistry (event sourced state)

**What is write-only or severely read-starved:**
1. **Qdrant** — accumulating semantic embeddings, nothing reads them during daemon operation
2. **Raw SQLite** — 14 of 16 get methods never called; 15+ store methods have no corresponding get method
3. **CH_CAUSAL_WATCH** — published but no subscribers
4. **post_mortem_insights.json** — computed every 500 steps, never consumed
5. **config_history.jsonl** — audit trail only
6. **position_sizing_log.jsonl** — audit trail only
7. **alerts_human.jsonl** — human-readable output, no machine reader
8. **inevitabilities.jsonl** — write-only for machines
9. **Neo4j** — completely unwired despite being installed
10. **DuckDB** — completely unwired despite being installed

**The highest-value bridge to build:** Add a reader for `get_congressional_trades()`, `get_cot_history()`, `get_eia_observations()`, and `get_vix_history()` into `RawDataAnalyst`. These four routines alone would expand the analyst from 4 cross-domain patterns to 8+, reading data that is already being collected and persisted every sensing cycle. No new API calls required — the data is already there.

**The fastest Qdrant activation:** Add `ctx.pattern_memory.find_precedents(ticker, signals)` inside the OctopusColony `investigate_partial` handler. This requires one line of code and would make semantic memory retrieval a live function rather than an append-only log.

**The structural gap:** Guiding Light's vision is "layers of memory, layers of processing, and we need to use it." The architecture has the layers. The usage is missing. MIDGE currently remembers with 7 layers and reasons from 2.
