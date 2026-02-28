# MIDGE Handoff

## What Happened

### Bridge 4+5: Dynamic Gates + Meta-Learning / RSI Layer 3 (2026-02-27)

MIDGE can now self-tune her discovery process. Previously, hypothesis promotion/retirement thresholds were hardcoded constants. Now they live in `learning_config.py` and adjust based on outcome history.

**Bridge 4 — Dynamic Quality Gates:**
- `learning_config.py` gained `hypothesis_gates` section (5 thresholds + `_bounds` + `_regime_deltas`)
- `hypothesis_validator.py` reads gates from config via `_get_gate(key, regime)` with fallback to hardcoded values
- `hypothesis_engine.py` `_review_gates()` runs every 2000 steps: if false-positive rate > 30%, tightens `promote_win_rate` by +0.01. If zero promotions with candidates, loosens by -0.01. Bounds-clamped, cooldown-protected.

**Bridge 5 — Meta-Learning (RSI Layer 3 — the system improves HOW it discovers):**
- `learning_config.py` gained `generator_thresholds` section (`min_correlation`, `min_pairs` + `_bounds`)
- `hypothesis_generator.py` reads thresholds from config via `_get_gen_threshold()`. Tracks pair quality: which (source_a, source_b) pairs produce promoted vs retired hypotheses. Sorts findings by correlation + 0.1 bonus for known-good pairs.
- `thompson_calibrator.py` gained `get_calibration_feedback()`: returns overconfident/underconfident sources with deltas (capped at +/-0.05)
- `hypothesis_engine.py` `_run_meta_learning()` runs every 3000 steps:
  - Wire 1: Calibration feedback → adjusts `source_reliability` in learning_config (overconfident sources reduced, underconfident raised)
  - Wire 2: Retirement rate → adjusts `generator_thresholds.min_correlation` (>70% retired → tighten +0.02, <20% → loosen -0.01)
  - Wire 3: `_promote()`/`_retire()` call `generator.record_outcome()` for pair quality tracking + retirement window tracking

**New channels:** `CH_GATE_ADJUSTED`, `CH_META_ADJUSTED` in `channels.py`.
**41 new tests** across `test_dynamic_gates.py` (15) and `test_meta_learning.py` (26).

### Market-Focused Step Loop (2026-02-27)

MIDGE's agents now actively hunt patterns instead of doing generic TaskPool busywork. Previously, a SEC_WATCHER "exploring" claimed a generic task — the same thing a STEM agent does. Now it scans insider signal buffers. Same agent class (Law 5), different behavior based on role configuration.

**New file:** `mae_core/market/market_actions.py` (~500 lines) — role-keyed action dispatch. Public function: `act_market(agent, action_type) -> float | None`. Returns reward for market-role agents, or None to fall through to generic TaskPool. 15 action handlers across 5 roles.

**What changed:**
1. **`lifecycle_decision.py`** — 8-line dispatch in `_act()`: market agents route to `act_market()` for explore/exploit/communicate before hitting TaskPool
2. **`market_awareness.py`** — `market_stimulus` key added to `get_market_context_for_router()`: encodes market state as matchable string (`convergence:strong:bullish`, `hypothesis:empty`, `market:ambient`)
3. **`lifecycle_decision.py`** — `_route_with_advisory()` uses `market_stimulus` as DecisionRouter stimulus for market agents, enabling reflex-driven routing
4. **`hypothesis_engine.py`** — two public methods: `request_generation()` (agent-triggered, 100-step cooldown) and `request_validation()` (skip-if-busy, returns promoted/retired/busy/none)
5. **`bootstrap/market.py`** — `_register_market_reflexes()`: 4 reflex patterns registered on per-agent DecisionRouters (convergence:strong → exploit, hypothesis:empty → explore, market:ambient → explore)
6. **`bootstrap/market.py`** — `_write_convergence_heartbeat()`: overwrites `data/midge/convergence_state.json` every 100 steps with regime, convergence, ticker alerts, hypothesis stats, Kelly
7. **`bootstrap/market.py`** — Phase 1 bug fixes: auto-redifferentiation ref loss (refs restored on role switch), tiered alerter queries every 10 steps, per-ticker alert storage, Kelly sizing subscriber
8. **`redifferentiation_triggers.py`** — `set_market_refs()` + `_reattach_market_refs()`: when auto-rediff switches an agent to a market role, all live system connections are re-attached

**Observable output files** (all in `data/midge/`, gitignored):
- `agent_activity.jsonl` — written by `_market_broadcast()` communicate action
- `hypothesis_activity.jsonl` — written by hypothesis generate/validate actions
- `convergence_state.json` — overwritten every 100 steps with current snapshot

**41 new tests** in `tests/test_market_actions.py`.

### Bridge 2: Thompson Routing (2026-02-26)

Connected Bridge 1's granular Thompson keys to actual signal processing. Previously, promoted hypotheses seeded `sweep_bt:CL=F:bearish` keys but the convergence alerter always looked up the generic `session_sweep` key for all sweep signals.

**What changed:**

1. **`_resolve_thompson_key()` cascade** — `convergence_alerter.py` gained a new method that replaces the static `_SOURCE_TO_THOMPSON_KEY` lookup for sweep sources. Runs a most-specific-wins cascade: `sweep_bt:{symbol}:{direction}` → `sweep_bt:{symbol}` → `sweep_bt:{direction}` → generic `session_sweep`. Only selects granular keys with >= 5 observations (maturity gate matching thin-data blend threshold).

2. **TIER_ROUTING bugfix** — `sensing_hook.py` was missing `session_sweep_ifvg` from `TIER_ROUTING`, causing IFVG signals to fall to the default "strategic" tier instead of "tactical" where they belong alongside `session_sweep`.

**13 new tests** in `test_convergence_alerter_cascade.py`.

### Bridge 1: Backtest → Hypothesis Engine (2026-02-26)

MIDGE reads her own backtest results and turns them into formal hypotheses she tracks, validates, and learns from. Previously a human had to interpret "CL=F is good, RTY=F is bad" from the backtest report.

**New file:** `mae_core/market/intelligence/backtest_analyzer.py` — reads `sweep_backtest_results.json`, slices 336 trades into statistical aggregates (6 symbol, 2 direction, 2 session, 12 combos), converts qualifying aggregates (n >= 20) to PROBATION hypotheses with pre-populated stats and ICT causal stories.

**What changed:**
1. **`hypothesis.py`** — new `BACKTEST_DERIVED` source type enum
2. **`hypothesis_validator.py`** — precomputed stats path for BACKTEST_DERIVED (no archive scan, uses pre-populated wins/losses/sharpe, still computes DSR)
3. **`hypothesis_engine.py`** — calls `backtest_analyzer.analyze()` on generation cadence, seeds granular Thompson keys (`sweep_bt:{domain_filter}` with `Beta(wins+1.1, losses+0.9)`) on promotion
4. **`bootstrap/market.py`** — BacktestAnalyzer instantiated before HypothesisEngine, 3 Group 17 triadic connections, holon + somatic + fractal registration
5. **`connection_registrations.py`** — Group 17 (backtest bridge): 3 triadic connections

**39 new tests** across test_backtest_analyzer.py (31), test_hypothesis_validator.py (+5), test_hypothesis_engine.py (+3).

### Hypothesis Generation Loop — RSI Layer 2 (2026-02-25)

MIDGE's recursive self-improvement loop. Discovers patterns, formalizes them as testable hypotheses, validates adversarially, promotes or retires based on evidence. The system now improves itself.

**5 new files:**
1. `mae_core/market/intelligence/hypothesis.py` — Hypothesis dataclass with trigger pattern, lifecycle status, causal story, DSR stats
2. `mae_core/market/intelligence/hypothesis_registry.py` — Event-sourced lifecycle management (probation → active → hibernated → retired). Persists to `data/market/hypotheses.jsonl`
3. `mae_core/market/intelligence/hypothesis_generator.py` — Converts lag-correlation findings into formal hypotheses. 11 causal story templates for known source pairs. Unknown pairs flagged "REQUIRES MANUAL REVIEW" (blocks promotion)
4. `mae_core/market/intelligence/hypothesis_validator.py` — Adversarial validation with Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014). Global trial counter penalizes multiple testing. Promotion bars: obs >= 20, win_rate > 0.52, DSR > 0.5, real causal story
5. `mae_core/market/intelligence/hypothesis_engine.py` — Orchestrator: generation every 500 steps, validation every 1000 steps, regime check every 100 steps. Subscribes to CH_SIGNAL_INGESTED, matches incoming signals against active hypothesis triggers

**Modified files:**
- `mae_core/market/channels.py` — 5 new channel constants (CH_SIGNAL_INGESTED, CH_HYPOTHESIS_DISCOVERED/PROMOTED/RETIRED/FIRED)
- `mae_core/market/sensing_hook.py` — EventBus bridge: publishes CH_SIGNAL_INGESTED per signal in `_collect_results()`
- `mae_core/agents/stem_cell.py` — 2 new role profiles (HYPOTHESIS_EXPLORER: high exploration 0.7; HYPOTHESIS_VALIDATOR: adversarial, satisfaction 0.92)
- `mae_core/bootstrap/market.py` — Full wiring: 4 systems, 10 connections (Group 16), 4 holons, 1 fractal K3 subsystem, endocrine coupling (dopamine on promote, cortisol on unexpected retire), EventBus subscriptions, step hooks, agent differentiation at 12+ agents
- `mae_core/market/market_awareness.py` — HYPOTHESIS_EXPLORER and HYPOTHESIS_VALIDATOR added to _MARKET_ROLES, hypothesis stats in router context

**4 new test files, 40 tests.** Systems: 121 (92+29). Connections: 374 (55 market). Holons: 140.

### Self-Calibrating Decision Engine (2026-02-25)

Four new intelligence systems that turn MIDGE from a pattern detector into a self-calibrating decision engine. Infrastructure is built; results improve as data accumulates.

1. **Signal archive reader** — `mae_core/market/intelligence/signal_archive_reader.py`. Read-only interface over existing date-partitioned signal archives (`data/midge/signals/YYYY-MM-DD.jsonl`). `ArchiveRecord` with `__slots__` for memory efficiency. Provides `load_range()`, `query_source()`, `query_symbol()`, `get_timeseries()` for lag-correlation input.

2. **Lag-correlation analyzer** — `mae_core/market/intelligence/lag_correlation_analyzer.py`. Cross-correlates signal sources across time lags 1-90 days. Answers "does congressional buying today predict a price move in 3 weeks?" Pure-Python Pearson r + Fisher Z p-value (no scipy). Persists findings to `data/market/lag_correlations.json`. Step hook cadence: every 500 steps.

3. **Thompson calibrator** — `mae_core/market/intelligence/thompson_calibrator.py`. Two jobs: (a) **Seed fix** — the 15 signal-source-level keys in `learning_config.py` were never seeded into `thompson_distributions.json`, causing every lookup to fall through to uninformative Beta(1,1). Now seeds them from `source_reliability` values at bootstrap. Idempotent. (b) **Calibration diagnostic** — joins predictions.jsonl to outcomes.jsonl, computes per-source Brier scores, 5 calibration buckets, over/under-confidence detection. Step hook cadence: every 1000 steps.

4. **Kelly position sizer** — `mae_core/market/intelligence/kelly_position_sizer.py`. Kelly criterion: `f = (b*p - q) / b` where p = Thompson distribution mean, b = historical win/loss ratio from outcomes.jsonl. Always half-Kelly (f/2), capped at 5%. Confidence tiers: "low" (< 10 outcomes), "medium" (10-49), "high" (50+). Triggers on per-ticker convergence alerts, not every step.

**Bootstrap wiring:** 4 new systems instantiated in `_instantiate_market_systems()`, 12 new triadic connections (Group 15), 4 holons, 4 somatic entries, 4 fractal extras, 3 step hooks (lag/500, calibration/1000, Kelly on convergence at step%50).

### Analytical Improvements (2026-02-25)

Two edge detector upgrades from the Layer 5 roadmap:

1. **Compressed cluster detector** — `cluster_detector.py` gained time-compression scoring. Computes span between first and last trade in a cluster: `compression_score = max(0, 1 - span/window_days)`. Tight clusters (all trades within days) get up to +10% confidence boost. `ClusterSignal` gained `compression_score` field and `to_plain_language()` labels tight clusters. Previously, 5 insiders buying within 3 days scored the same as 5 insiders spread over 28 days.

2. **8-K text sentiment via Ollama** — New `mae_core/market/edge/form8k_sentiment.py`. Classifies 8-K filing text using local Ollama LLM (llama3.2:3b). Returns direction override + confidence modifier (-0.15 to +0.15). Wired into `sensing_hook.py:_enrich_signal()` for `sec_form8k` signals. Degrades gracefully when Ollama is offline (returns None, falls back to rule-based item codes). Bootstrap constructs `Form8KSentimentAnalyzer` in `_wire_sensing_hook()` and passes to `MarketSensingHook`.

### Bayesian Confidence Engine (2026-02-25)

Resolved Alpha's standing dissent: replaced the structurally wrong additive confidence formula in the convergence alerter with a Thompson-weighted geometric mean. This was the last architectural gap — the Thompson Sampler was completely disconnected from confidence calculations.

**What changed:**

1. **Unified `_compute_confidence()` method** — replaces 3 separate additive formulas in `convergence_alerter.py` (`_check_direction_convergence`, `check_ticker_convergence`, `get_actionable_summary`). Uses Thompson-weighted geometric mean: `exp(sum(w_i * log(c_i)) / sum(w_i))` with a multiplicative diversity bonus `1 + 0.12 * log1p(domains - 1)`.

2. **Thompson Sampler wired into convergence alerter** — `ConvergenceAlerter.__init__()` now accepts optional `thompson_sampler` and `regime_classifier`. The `_get_thompson_weight()` method maps signal sources to Thompson distribution keys and returns a reliability weight in [0.5, 1.5]. Thin data (< 5 observations) blends toward neutral weight 1.0.

3. **Signal source tracking** — `Signal` dataclass gained `source: str = ""` field. `record_signal()` passes `source` through. `sensing_hook.py` now sends `source=sig.source` to the alerter. This lets Thompson look up per-source reliability.

4. **Bootstrap wiring** — `market.py` passes `thompson_sampler` to the main convergence alerter. Two-phase regime_classifier injection (constructed after alerter). Somatic dependency list updated.

5. **15 new Thompson distribution seeds** — `learning_config.py` gains signal-source-level keys matching `MarketSignal.source` values from `signal.py`.

**Behavioral change:**
- Old: 3 domains at avg confidence 0.65 → `0.65 + 0.10 = 0.75`
- New: 3 domains at avg confidence 0.65 → `0.65 * 1.13 = 0.73` (more conservative)
- As Thompson accumulates real outcomes, unreliable sources (social sentiment) get down-weighted, reliable sources (SEC filings) get up-weighted automatically.
- Tiered alerters (tactical/strategic/thematic) retain arithmetic fallback — no Thompson injection.

**Alpha's dissent status:** RESOLVED. The additive formula is gone. The Bayesian combination correctly handles correlated signals without inflating confidence.

### Agent-Based Market Sensing (2026-02-22)

Wired MIDGE's market intelligence through Mae's agent system. Previously, market data only flowed through the standalone `midge_scan.py` script — agents ran with empty convergence buffers. Now the 33-layer bootstrap creates a MarketSensingHook that feeds live data into agents during normal operation.

**What was built:**

1. **MarketSensingHook** — `mae_core/market/sensing_hook.py` (NEW). Async market data fetcher that runs inside Mae's step loop. ThreadPoolExecutor(1) with 12-source rotation (SEC Form 4 → 8-K → congressional → senate → hiring → USASpending → SAM.gov → Reddit sentiment → FINRA short → SEC EFTS → Finnhub → FRED macro). Fetch cadence: every 50 steps. Outcome evaluation: every 200 steps. Same proven async pattern as ApiGateway.

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

- **2886 tests pass, 0 failures**
- **121 systems** (92 core + 29 market), **140 holons**, **374 connections** (217 core + 47 fractal + 55 bootstrap + 55 market)
- **40 market files** in `mae_core/market/` (bootstrapped as Layer 33 + 6 API clients + form8k_sentiment + hypothesis loop + backtest_analyzer + backtest_scheduler + step_timer + market_actions)
- **33-layer bootstrap** runs cleanly (Layers 33a-33i)
- **Agent-based market sensing active** — 3+ agents differentiated (SEC_WATCHER, CONTRACT_TRACKER, MARKET_ANALYST, + HYPOTHESIS_EXPLORER + HYPOTHESIS_VALIDATOR at 12+ agents)
- **Hypothesis generation loop active** — RSI Layer 2: lag findings → hypotheses → adversarial validation → DSR gate → promote/retire
- **7-phase scan pipeline** still available as standalone (`midge_scan.py`)
- **Git:** Remote at `github.com/CBaen/MIDGE`

### Agent-Based Architecture

```
Bootstrap Layer 33h: MarketSensingHook wired as step hook
  ├─ Async fetch on 50-step cadence (12-source rotation: form4 → 8k → congress → senate → hiring → usaspending → sam.gov → reddit → finra → efts → finnhub → fred)
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
Phase 2/7: Fetch — 14 data sources (SEC Form4/8K/EFTS, House/Senate trades, jobs, USASpending, SAM.gov, prices, Reddit, FINRA short, Finnhub news/earnings, FRED macro)
Phase 3/7: Convert — raw results → MarketSignals (with filters: 10b5-1, $50K min, log-linear)
Phase 4/7: Store + Feed — Qdrant + JSONL, enriched with velocity + filing-time modifiers, fed to global + tiered alerters
Phase 5/7: Outcome tracking — register predictions with per-type windows, evaluate matured outcomes → Thompson update
Phase 6/7: Analyze — global convergence, per-ticker convergence, tiered convergence, cross-tier detection
Phase 7/7: Report — markdown intelligence report with all sections
```

### Market Package Structure

| Subpackage | Files | Purpose |
|------------|-------|---------|
| `apis/sec_edgar/` | 4 (models, client, efts, __init__) | SEC insider trades + material events + full-text search |
| `apis/` | 13 (price_fetcher, house_stock_watcher, senate_stock_watcher, job_tracker, usa_spending, sam_gov, apewisdom, finra_short_interest, finnhub_client, fred_client, ticker_resolver, market_data_provider) | 14 data sources + utilities |
| `edge/` | 7 (cluster_detector, politician_tracker, filing_time_analyzer, contract_predictor, form8k_sentiment, session_sweep_detector, ta_indicators) | Pattern recognition + text analysis |
| `intelligence/` | 19 (thompson_sampler, velocity_detector, correlation_tracker, convergence_alerter, learning_config, regime_classifier, outcome_collector, signal_archive_reader, lag_correlation_analyzer, thompson_calibrator, kelly_position_sizer, backtest_analyzer, backtest_scheduler, hypothesis, hypothesis_registry, hypothesis_generator, hypothesis_validator, hypothesis_engine) | Bayesian learning + feedback loop + calibration + sizing + hypothesis lifecycle + meta-learning |
| `root` | 7 (signal.py, channels.py, outcome_tracker.py, memory.py, sensing_hook.py, step_timer.py, market_actions.py) | Integration layer + sensing + Qdrant persistence + performance timing + agent actions |

---

## What's Next

### Layer 5: Complete

All 17 planned items from the Layer 5 roadmap are done. Roadmap history at `C:\Users\baenb\.claude\plans\delegated-leaping-map.md`.

**Data-gated (infrastructure built, awaiting data maturation):**
- Lag-correlation findings: needs 30+ days of signal archives
- Thompson calibration accuracy: needs 50+ outcomes
- Kelly sizing confidence: needs 100+ calibrated outcomes
- Meta-learning adjustments: needs retirement window to fill (50 promote/retire events)

**Back burner:**
- **Options flow via Unusual Whales** ($35/mo API — needs Guiding Light approval on spend)

### Resolved Dissents

**Alpha's additive confidence dissent** — RESOLVED 2026-02-25. Replaced with Thompson-weighted geometric mean. See "Bayesian Confidence Engine" section above.

---

## For the Next Instance

Welcome. MIDGE is Mae differentiated for financial markets. Here is what you need to know:

1. **MIDGE = mae-core + market intelligence.** 121 systems, same 8 laws, 33-layer bootstrap. Market organ is Layer 33 (33a-33i).
2. **Mae-core is upstream.** Changes to Mae's genome should be made in `C:\Users\baenb\projects\mae-core` and pulled here. Market-specific changes stay here.
3. **Agents actively sense the market.** MarketSensingHook (Layer 33h) fetches data on cadence and feeds convergence_alerter. Three agents are differentiated into market roles (Layer 33i). Endocrine coupling and market advisory carry signals to all agents.
4. **The crown jewel is `convergence_alerter.py`** — synthesizes signals across ALL domains (insider + congressional + contract + hiring + velocity) into actionable alerts. Now with per-ticker and multi-timeframe convergence.
5. **Thompson Sampling** uses Bayesian explore/exploit. Learned distributions in `data/market/thompson_distributions.json`. Bayesian forgetting prevents stale evidence.
6. **OutcomeCollector** closes the feedback loop: scan signals → register_signals() → per-type windows → price check → Thompson update. Success threshold: 5%.
7. **All 8 Mathematical Laws are satisfied.** See implementation plan Section 12 for compliance map.
8. **2886 tests must keep passing.** Zero regressions.
9. **Deep memory runs on Qdrant** container (port 6333). Start with `docker compose up -d`.
10. **API keys** needed: RAPIDAPI_KEY (job tracker, congressional trades), ALPHA_VANTAGE_KEY (price fallback), SAM_GOV_API_KEY, MAE_TAVILY_API_KEY, MAE_FINNHUB_API_KEY (news sentiment + earnings), FRED_API_KEY (macro indicators). Free/no-key: SEC EDGAR, yfinance, USASpending, Senate Stock Watcher, ApeWisdom, FINRA short volume, SEC EFTS.
11. **`python main.py --agents 6 --steps 500`** runs MIDGE with agents sensing the market. Requires 6 agents (K3 general + K3 market per Law 2).
12. **`python midge_scan.py`** runs a standalone 7-phase scan (no bootstrap needed). Reports go to `data/midge/scans/`, signal archives to `data/midge/signals/`. Use `--dry-run` to skip Qdrant.
13. **`python test_live_apis.py`** tests all 8 API connections individually.
14. **Triadic research** at `research/midge-prediction-optimization/deliverable.md` — prioritized modification plan with remaining Layer 5 items.
15. **Data schema** at `data/midge/SCHEMA.md` — canonical reference for all signal types, predictions, outcomes, Thompson distributions, and convergence alerts.

---

## Previous Sessions

### Phase 2 API Expansion (2026-02-22)
Added 6 new data sources, expanding MIDGE from 6 to 12 sensing sources. Built: SenateStockWatcherClient (GitHub JSON mirror), ApeWisdomClient (Reddit/WSB mention velocity), FINRAShortInterestClient (daily short volume), SECFullTextSearchClient (EFTS keyword search), FinnhubClient (news sentiment + earnings calendar), FREDClient (yield curve, VIX, credit spread, rates, unemployment). All wired into MarketSensingHook source rotation, bootstrap instantiation, convergence tier routing, BoundaryMembrane trust scores. 7 new signal adapters in signal.py. 2473 tests pass.

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
