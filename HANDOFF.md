# MIDGE Handoff

## What Happened

### Agent-Based Market Sensing (2026-02-22)

Wired MIDGE's market intelligence through Mae's agent system. Previously, market data only flowed through the standalone `midge_scan.py` script — agents ran with empty convergence buffers. Now the 33-layer bootstrap creates a MarketSensingHook that feeds live data into agents during normal operation.

**What was built:**

1. **MarketSensingHook** — `mae_core/market/sensing_hook.py` (NEW). Async market data fetcher that runs inside Mae's step loop. ThreadPoolExecutor(1) with source rotation (SEC Form 4 → 8-K → congressional → hiring → USASpending → SAM.gov). Fetch cadence: every 50 steps. Outcome evaluation: every 200 steps. Same proven async pattern as ApiGateway.

2. **Bootstrap wiring** — `mae_core/bootstrap/market.py` gained Layer 33h (`_wire_sensing_hook`) and Layer 33i (`_differentiate_market_agents`):
   - **33h:** Instantiates MarketSensingHook with all market systems from ctx, creates 3 tiered ConvergenceAlerters (tactical/strategic/thematic), OutcomeCollector, SignalMemory. Adds `_market_advisory` dict as Channel B for market-role agents. Registers the hook as a model step hook.
   - **33i:** Differentiates last 3 agents into SEC_WATCHER, CONTRACT_TRACKER, MARKET_ANALYST via `redifferentiate()`. Injects `_market_advisory_ref` so agents read convergence alerts in their decision cascade.

3. **Two data channels to agents (both active):**
   - **Channel A (Endocrine):** convergence_alerter → CH_CONVERGENCE → endocrine DOPAMINE/ADRENALINE → organism_state → agent._observe() reads _body_state. Was already fully wired in Layer 33f — just needed data flowing in.
   - **Channel B (Market Advisory):** convergence alerts → `_market_advisory` dict → market-role agents read in _observe()/_decide(). Separate from PatternCortex's `_latest_advisory` (which overwrites every step).

4. **Oracle pathway functional:** MarketDataProvider already implements BaseProvider with `execute()` method. Market-role agents have `api_call_enabled=True` → when prediction error > 0.5, _decide() returns "api_call" → inject_external_task() → ApiGateway routes to MarketDataProvider for targeted follow-up.

**Data flow (what was broken → what works now):**
```
Step Hook (cadenced, async)
  ├─ Every 50 steps: Fetch from rotating API source (background thread)
  ├─ Collect results → Convert to MarketSignals (with all filters)
  ├─ Feed into convergence_alerter + 3 tiered alerters
  ├─ Store to Qdrant + JSONL
  │
  ▼
ConvergenceAlerter.check_convergence()
  ├─ Channel A: CH_CONVERGENCE → endocrine → agents read _body_state  [was wired, now has data]
  └─ Channel B: _market_advisory dict → market agents read in _decide() [NEW]
      └─ Oracle pathway: api_call → ApiGateway → MarketDataProvider
```

### Triadic Prediction Optimization (2026-02-22)

Ran a full triadic analysis (Lead/Alpha/Beta, 5-phase protocol) on "what modifications maximize MIDGE's prediction accuracy." Research files at `research/midge-prediction-optimization/`. The deliverable identified 5 layers of improvements ordered by impact.

**Phases A through E implemented:**

#### Phase A — Data Integrity (Layer 1)
1. **Outcome deduplication** — `outcome_tracker.py` now loads already-evaluated signal_ids from `outcomes.jsonl` before processing, preventing the same prediction from being evaluated 8x and poisoning Thompson distributions.
2. **Contract ticker resolution** — `signal.py:from_government_contract()` now calls `ticker_resolver.resolve()` to map company names to tickers. Contract signals are no longer invisible to the feedback loop.
3. **Domain status display fix** — `midge_scan.py` used wrong field names (`avg_strength` → `strength`, `count` → `signal_count`), causing all domain strengths to display as 0.00.

#### Phase B — Signal Quality (Layer 2)
4. **10b5-1/RSU filter** — `signal.py:from_insider_trade()` detects compensation transactions (codes D, A, F, G, M) and reduces strength by 75%, confidence to 0.40. Eliminates ~60% of false bearish signals from scheduled sales.
5. **Congressional $50K minimum** — `midge_scan.py` skips trades below $50K `amount_high`. Eliminates ~85% of congressional noise (Gilbert Cisneros-style portfolio rebalancing).
6. **Bonferroni correction** — `correlation_tracker.py` adjusts anomaly threshold by `+ log(n_pairs) * 0.5`, reducing false positives from ~1.26/cycle to near-zero at 105 pairs.

#### Phase C — Core Architecture (Layer 3, items 3.1-3.3)
7. **Per-ticker convergence** — `convergence_alerter.py` gained `check_ticker_convergence(min_domains=2)` — groups signals by symbol before running convergence. Transforms "the insider domain is bearish" into "RTX has insider selling + hiring bullish across 3 domains."
8. **VelocityDetector wired** — `midge_scan.py` now instantiates VelocityDetector and calls `record()` for each signal before feeding alerter. Signal velocities are no longer all 0.0.
9. **FilingTimeAnalyzer wired** — `midge_scan.py` applies confidence modifiers for suspicious SEC filing times (Friday dumps: -15%, after-hours: -8%) to Form 4 and 8-K signals.

#### Phase D — Feedback Loop (Layer 3, items 3.4-3.5)
10. **OutcomeCollector** — New file `mae_core/market/intelligence/outcome_collector.py`. Bridges signal archives → OutcomeTracker → Thompson Sampler. Per-type outcome windows (45d insider, 60d cluster, 14d congressional, 5d 8-K, 90d contract/hiring). Success threshold raised from 2% to 5% (old threshold barely exceeded random baseline). Wired into `midge_scan.py` as Phase 5/7. Also supports retroactive processing of JSONL archives via `collect_from_archives()`.
11. **Multi-timeframe convergence** — Three tiered ConvergenceAlerter instances in `midge_scan.py`:
    - Tier 1 Tactical (48h): SEC Form 4, Form 8-K
    - Tier 2 Strategic (21d): Congressional trades, contracts, clusters, correlations
    - Tier 3 Thematic (90d): SAM.gov, hiring, contract predictions

    Cross-tier convergence detection: same ticker in 2+ tiers gets 0.7x confidence (Alpha's independence amendment — signals at different timeframes aren't fully independent).

#### Phase E — Calibration (Layer 4)
12. **Decay rate corrections** — All 9 signal types recalibrated per academic literature:
    - InsiderTrade: 0.05→0.035 (20d half-life, Lakonishok & Lee 2001)
    - ClusterSignal: 0.05→0.025 (28d, Alldredge 2019)
    - CongressionalTrade: 0.03→0.05 (14d from disclosure)
    - Form8KEvent: 0.03→0.25 (3d — market prices binary events fast)
    - ContractPrediction: 0.03→0.018 (39d pre-announcement)
    - GovernmentContract: 0.02→0.07 (10d post-announcement drift)
    - HiringSignal: 0.07→0.015 (46d — hiring leads contract 60-120d)
    - SAM.gov: 0.04→0.008 (87d — competition periods)
    - CorrelationSignal: 0.03→0.04 (17d)
13. **Log-linear strength scale** — `signal.py:from_insider_trade()` replaced `min(1.0, value/1M)` with `min(1.0, log1p(value/100K) / log1p(10))`. $100K→0.42, $500K→0.73, $1M→0.83, $5M→0.95. No more cliff at $1M.

### Bug Fixes + Schema (2026-02-22)

Fixed 3 bugs found in live scan output, defined data schema:

1. **10b5-1 plan sale detection (3-layer fix)** — `models.py` gained `is_plan_sale` and `footnotes` fields. `client.py` now extracts `transactionCode` from XML `<transactionCoding>` element AND scans `<footnotes>` for "10b5-1" references (both XML and HTML paths). `signal.py` checks `trade.is_plan_sale` for non-buy dispositions → strength * 0.25, confidence 0.40. Pichai/Kress/Bosworth scheduled sales now correctly penalized.
2. **NaN ticker guard** — `house_stock_watcher.py` ticker guard now catches `"NaN"`, `"N/A"`, `"NONE"` strings. Cleaned stale `congressional:NaN:2026-01-26` from `registered_signals.json`.
3. **Legacy prediction format** — `outcome_tracker.py` uses `pred.get("timestamp") or pred.get("predicted_at")` with graceful skip. 14 legacy predictions now processable.
4. **Data schema** — `data/midge/SCHEMA.md` documents all 11 data types: MarketSignal (9 source-specific metadata shapes), predictions (current + legacy), outcomes, Thompson distributions, convergence alerts, decay rates, outcome windows, signal flow diagram.

---

## Current State

- **2473 tests pass, 0 failures**
- **108 systems** (92 core + 16 market), **127 holons**, **336 connections** (211 core + 47 fractal + 55 bootstrap + 23 market)
- **25 market files** in `mae_core/market/` (bootstrapped as Layer 33, + memory.py, outcome_collector.py, sensing_hook.py)
- **33-layer bootstrap** runs cleanly (Layers 33a-33i)
- **Agent-based market sensing active** — 3 agents differentiated (SEC_WATCHER, CONTRACT_TRACKER, MARKET_ANALYST)
- **7-phase scan pipeline** still available as standalone (`midge_scan.py`)
- **Git:** Remote at `github.com/CBaen/MIDGE`

### Agent-Based Architecture

```
Bootstrap Layer 33h: MarketSensingHook wired as step hook
  ├─ Async fetch on 50-step cadence (rotating: form4 → 8k → congressional → hiring → usaspending → sam.gov)
  ├─ Signals → convergence_alerter (global) + 3 tiered alerters
  ├─ Signals → Qdrant + JSONL archive
  └─ Outcome evaluation on 200-step cadence → Thompson update

Bootstrap Layer 33i: Agent differentiation
  ├─ agents[-3] → SEC_WATCHER (api_call_enabled, market_sense capability)
  ├─ agents[-2] → CONTRACT_TRACKER (api_call_enabled, market_sense capability)
  └─ agents[-1] → MARKET_ANALYST (api_call_enabled, world_model_enabled)

Data channels to agents:
  Channel A (Endocrine): convergence → CH_CONVERGENCE → DOPAMINE/ADRENALINE → body_state
  Channel B (Advisory): convergence → _market_advisory dict → market agents read in _decide()
  Oracle pathway: high pred_error → api_call → ApiGateway → MarketDataProvider
```

### Scan Pipeline (midge_scan.py — standalone alternative)

```
Phase 1/7: Setup — clients, alerter, 3 tiered alerters, memory, velocity, filing, outcome collector
Phase 2/7: Fetch — 8 data sources (SEC, congressional, jobs, USASpending, SAM.gov, prices)
Phase 3/7: Convert — raw results → MarketSignals (with filters: 10b5-1, $50K min, log-linear)
Phase 4/7: Store + Feed — Qdrant + JSONL, enriched with velocity + filing-time modifiers, fed to global + tiered alerters
Phase 5/7: Outcome tracking — register predictions with per-type windows, evaluate matured outcomes → Thompson update
Phase 6/7: Analyze — global convergence, per-ticker convergence, tiered convergence, cross-tier detection
Phase 7/7: Report — markdown intelligence report with all sections
```

### Market Package Structure

| Subpackage | Files | Purpose |
|------------|-------|---------|
| `apis/sec_edgar/` | 3 (models, client, __init__) | SEC insider trades + material events |
| `apis/` | 7 (price_fetcher, house_stock_watcher, job_tracker, usa_spending, sam_gov, ticker_resolver, market_data_provider) | Market data sources + utilities |
| `edge/` | 4 (cluster_detector, politician_tracker, filing_time_analyzer, contract_predictor) | Pattern recognition |
| `intelligence/` | 7 (thompson_sampler, velocity_detector, correlation_tracker, convergence_alerter, learning_config, regime_classifier, outcome_collector) | Bayesian learning + feedback loop |
| `root` | 5 (signal.py, channels.py, outcome_tracker.py, memory.py, sensing_hook.py) | Integration layer + sensing + Qdrant persistence |

---

## What's Next

### Layer 5: Expansion (after 50+ Thompson outcomes)

These items from the triadic deliverable are deferred until the outcome collector has produced enough calibrated results:

1. **Options flow via Unusual Whales** ($35/mo API, "options" domain, sweep orders >$100K)
2. **Senate Stock Watcher** (mirror house_stock_watcher.py for senatestockwatcher.com)
3. **8-K text sentiment via Ollama** (local model classifies 8-K text)
4. **Thompson-weighted convergence** (weight signal contribution by sampled reliability)
5. **Lag-correlation analysis** (which signals genuinely lead others)
6. **Compressed cluster detector** (time-spread scoring within insider clusters)
7. **Position sizing** (Kelly criterion after 100+ outcomes)

### Alpha's Standing Dissent

The convergence alerter's additive confidence formula (`0.5 + 0.1 * categories + 0.1 * strength`) is structurally wrong — two 70% signals combined additively produce 0.90 but joint probability is 0.49. Should be replaced with multiplicative/Bayesian combination. Deferred to Layer 5 item 4 (Thompson-weighted convergence). See `research/midge-prediction-optimization/deliverable.md` Dissenting Notes.

---

## For the Next Instance

Welcome. MIDGE is Mae differentiated for financial markets. Here is what you need to know:

1. **MIDGE = mae-core + market intelligence.** 108 systems, same 8 laws, 33-layer bootstrap. Market organ is Layer 33 (33a-33i).
2. **Mae-core is upstream.** Changes to Mae's genome should be made in `C:\Users\baenb\projects\mae-core` and pulled here. Market-specific changes stay here.
3. **Agents actively sense the market.** MarketSensingHook (Layer 33h) fetches data on cadence and feeds convergence_alerter. Three agents are differentiated into market roles (Layer 33i). Endocrine coupling and market advisory carry signals to all agents.
4. **The crown jewel is `convergence_alerter.py`** — synthesizes signals across ALL domains (insider + congressional + contract + hiring + velocity) into actionable alerts. Now with per-ticker and multi-timeframe convergence.
5. **Thompson Sampling** uses Bayesian explore/exploit. Learned distributions in `data/market/thompson_distributions.json`. Bayesian forgetting prevents stale evidence.
6. **OutcomeCollector** closes the feedback loop: scan signals → register_signals() → per-type windows → price check → Thompson update. Success threshold: 5%.
7. **All 8 Mathematical Laws are satisfied.** See implementation plan Section 12 for compliance map.
8. **2473 tests must keep passing.** Zero regressions.
9. **Deep memory runs on Qdrant** container (port 6333). Start with `docker compose up -d`.
10. **API keys** needed: RAPIDAPI_KEY (job tracker, congressional trades), ALPHA_VANTAGE_KEY (price fallback), SAM_GOV_API_KEY, MAE_TAVILY_API_KEY. SEC EDGAR, yfinance, and USASpending are free.
11. **`python main.py --agents 5 --steps 500`** runs MIDGE with agents sensing the market. Market-role agents need at least 3 agents total.
12. **`python midge_scan.py`** runs a standalone 7-phase scan (no bootstrap needed). Reports go to `data/midge/scans/`, signal archives to `data/midge/signals/`. Use `--dry-run` to skip Qdrant.
13. **`python test_live_apis.py`** tests all 8 API connections individually.
14. **Triadic research** at `research/midge-prediction-optimization/deliverable.md` — prioritized modification plan with remaining Layer 5 items.
15. **Data schema** at `data/midge/SCHEMA.md` — canonical reference for all signal types, predictions, outcomes, Thompson distributions, and convergence alerts.

---

## Previous Sessions

### Agent-Based Market Sensing (2026-02-22)
Wired market intelligence through Mae's agent system. Created MarketSensingHook (sensing_hook.py) for async data fetching in step loop. Added Layer 33h (hook wiring + tiered alerters + advisory bridge) and 33i (agent differentiation into SEC_WATCHER/CONTRACT_TRACKER/MARKET_ANALYST). Two data channels: endocrine (already wired, now has data) + market advisory dict (new). Oracle pathway functional via MarketDataProvider.

### Bug Fixes + Schema (2026-02-22)
Fixed 10b5-1 plan sale detection (3-layer: models.py, client.py XML+HTML, signal.py), NaN ticker guard (house_stock_watcher.py), legacy prediction format (outcome_tracker.py). Created `data/midge/SCHEMA.md` — canonical reference for all MIDGE data types. Verified with 3 live scans.

### Triadic Prediction Optimization (2026-02-22)
Full triadic analysis + Phase A-E implementation. 13 modifications: outcome dedup, contract ticker, display fix, 10b5-1 filter, congressional $50K, Bonferroni, per-ticker convergence, VelocityDetector, FilingTimeAnalyzer, outcome collector, multi-timeframe convergence, decay rate calibration, log-linear strength. Pipeline expanded from 6 to 7 phases with outcome tracking. Research at `research/midge-prediction-optimization/`.

### Market Integration (2026-02-22 — multi-session)
Built full market intelligence integration (Tiers 0-5). Created Layer 33 bootstrap with 16 systems, 23 triadic connections, fractal K3 hierarchy, endocrine coupling, step hooks, Bayesian forgetting. Phase 2 completed: CorrelationTracker persistence, discovery_log reader, 437 Congress members, TickerResolver, MarketDataProvider, ContractPredictor evaluation, regime-aware Thompson Sampling. All verified: 2473 tests pass, 0 alert storms.

### MIDGE Fork (2026-02-22)
Forked mae-core into MIDGE. Ported 16 market intelligence files. Fixed imports and paths. Verified tests pass. Wrote identity docs.
