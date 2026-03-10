# MIDGE Handoff

## What Happened

### Investigation Pipeline + Signal-Triggered Convergence (2026-03-09)

**Octopus investigation wired — developing situations now get investigated:**
- Step-cadence dispatcher (every 20 steps) iterates `_developing_situations` and submits `investigate_partial` + `situation_check` tasks to OctopusColony. Max 5 tasks per step.
- `investigate_partial` enhanced: now queries PatternLibrary for historical templates matching domain combo (win rates, instance counts, cross-validation). Also checks WorldModel for causal chain evidence. High win-rate matches (>60%, 5+ instances) populate `_priority_requests` for Focused Attention.
- `CH_OCTOPUS_INVESTIGATION` subscriber wired — investigation results logged and fed back into priority polling.
- `_developing_situations` now persists to `data/market/developing_situations.json` (atomic write, loaded on bootstrap).
- Situation eviction now runs (was dead code — `situation_check` handler was never called before).

**Signal-triggered convergence — reactive, not polling:**
- After `_collect_one()` ingests signals, immediately calls `check_convergence()` inline. New signals evaluated instantly instead of waiting for next step tick.
- Deduplication prevents double-alerts (step-tick check remains as safety net).
- Per-ticker convergence also checked reactively.

**Tests:** 57 new (test_investigation_pipeline.py + test_signal_triggered.py). Full suite: 961/962 pass.

**Files changed:**
- `mae_core/network/market_task_handlers.py` — PatternLibrary + WorldModel queries in investigate_partial, pattern_library injection
- `mae_core/bootstrap/market_hooks.py` — investigation dispatcher, CH_OCTOPUS_INVESTIGATION subscriber
- `mae_core/bootstrap/market_systems.py` — developing_situations persistence + pattern_library injection
- `mae_core/market/sensing_hook.py` — reactive convergence check after signal ingestion

### Inevitability Surfacing — Focused Attention + Chain Boost + Backward Discovery (2026-03-09)

**Three connected features that make MIDGE a true inevitability investigator:**

1. **Focused Attention (Priority Polling)** — When partial convergence fires (2 domains, not yet 3), MIDGE boosts Thompson sampling scores for sources that serve the missing domains. `_DOMAIN_TO_SOURCES` reverse map, `_priority_requests` dict on SensingHook, 2x Thompson boost for focused sources. Entries expire after 1 hour, capped at 50.

2. **Sequential Chain Boost** — `_on_cascade_confirmed` subscriber in market_hooks.py. When CascadeTracker confirms a domino, injects synthetic `domain="cascade"` signals into ConvergenceAlerter for each remaining predicted domino. Strength proportional to `confirmed_count / total_links`. "cascade" added to domain_categories as "causal".

3. **Backward Cascade Discovery** — `find_root_causes()` on WorldModel (reverse BFS using `predecessors()`). `RootCause` dataclass mirrors `RippleEffect`. When a signal matches a downstream WorldModel node, MIDGE traces backward to find likely genesis triggers. If evidence exists in signal buffer: registers late-joining cascade. If not: populates `_priority_requests` so Focused Attention goes looking.

4. **Temporal Energy Tracking** — CascadeTracker now measures actual vs. predicted lag for each confirmed link. `energy_ratio = predicted_lag / actual_lag` — ratio > 1.0 means energy is accelerating through the chain (dominoes falling faster than expected). Stored per-link, reported in statistics, carried in CH_CASCADE_CONFIRMED payload.

**Tests:** 55 new (23 priority polling + 32 chain boost/backward/energy). Full suite: 961/962 pass (1 pre-existing congress.gov flake).

**Files changed:**
- `mae_core/market/sensing_hook.py` — `_DOMAIN_TO_SOURCES`, `_priority_requests`, Thompson boost in `_launch_thompson_guided()`
- `mae_core/bootstrap/market_hooks.py` — `_on_cascade_confirmed` subscriber, backward discovery in `_on_signal_causal_watch`, priority queue population
- `mae_core/market/intelligence/convergence_alerter.py` — `"cascade": "causal"` in domain_categories
- `mae_core/market/intelligence/world_model.py` — `RootCause` dataclass, `find_root_causes()` reverse BFS
- `mae_core/market/intelligence/cascade_tracker.py` — energy_ratio tracking per link
- `tests/test_priority_polling.py` — NEW: 23 tests
- `tests/test_chain_boost.py` — NEW: 32 tests

### Inevitability Cascade + CascadeTracker (2026-03-09)

**WorldModel wired into live convergence pipeline:**
- Every convergence alert carries `ripple_effects` (downstream tickers from causal graph)
- Partial convergences carry `causal_predictions` for Octopus investigation
- `CH_CAUSAL_WATCH` emitted proactively on signal ingestion when triggers map to strong downstream effects
- Plain-language alerts get CASCADE section showing top 5 downstream dominoes

**CascadeTracker (NEW):**
- `mae_core/market/intelligence/cascade_tracker.py` — watches active causal chains as dominoes confirm
- Registers cascades from convergence alerts, confirms links when signals match predicted tickers
- Closes WorldModel feedback loop: `record_outcome()` was never called before — now confirmed = True, expired = False
- `CH_CASCADE_CONFIRMED` emitted on confirmation (now consumed by Chain Boost subscriber above)
- 16 + 21 = 37 tests

### Bio-Market Activation — Every System Gets a Market Job (2026-03-09)

**29 biological systems wired to market intelligence channels (14 Tier 2+3 + 15 Tier 4+5):**

Tier 2 (one hop away from EndocrineSystem — now directly market-connected):
- **EmotionalSystem** → bullish convergence boosts curiosity, bearish boosts fear, deception spikes fear
- **HomeostasisRegulator** → bearish convergence raises threat_level setpoint, velocity anomaly raises processing_load
- **ArousalRegulator** → prediction win/loss feeds reward signal (Yerkes-Dodson for trading)
- **CuriosityDrive** → partial convergence + novel hypotheses boost exploration (dead endocrine wire fixed via `set_exploration_bonus()`)
- **NociceptionSystem** → deception = acute pain, prediction failure = referred pain, velocity anomaly = chronic pain

Tier 3 (clear market jobs, now EventBus-wired):
- **MetacognitionMonitor** → prediction confidence vs actual outcomes feeds calibration engine
- **ThreatDetector** → deception quill registered, sacrificeable market components for lizard autotomy
- **QuorumSpace** → convergence/pattern stacks deposit signals per ticker (organism-level vote)
- **CircadianRhythm** → phase changes set `ctx._circadian_activity` multiplier for sensing cadence
- **HAVEN** → deception events flag sources, successful predictions reduce suspicion
- **InhibitionSystem** → deception raises `ctx._market_caution`, high-confidence convergence lowers it
- **MemoryConsolidator** → circadian CONSOLIDATION triggers hypothesis engine, REST triggers excavation
- **CollectiveDreamPlanner** → event_bus fix, convergence nudges expertise weights
- **Stigmergy** → convergence deposits pheromone per ticker, prediction outcomes deposit success/danger markers

Tier 4 (market jobs, less direct — now wired):
- **DigestiveSystem** → convergence = high-nutrition data, partials = lower value. Energy budget gates throughput.
- **CirculatorySystem** → convergence requests attention resources, anomalies request compute. Heart rate rises under load.
- **LymphaticSystem** → failed predictions = expired_memory waste, deception = orphan_subscription cleanup.
- **Microbiome** → convergence decomposed by "complex" strain, anomalies by "detector" strain, weak stacks amplified.
- **RenalFilter** → deception teaches toxin patterns, convergence signals filtered for integrity.
- **SenescenceManager** → market systems registered for wear tracking, activity reported on convergence/prediction events.
- **MorphogenesisCoordinator** → 5+ partial convergences in 10 min → spawn investigation organ. Hypotheses boost growth rate.
- **ReproductiveSystem** → convergence activity sets `ctx._market_activity_pressure` for agent population scaling.
- **PearlDefense** → deception triggers quarantine validation (multi-layer nacre process).

Tier 5 (purpose emerges through connection — now wired):
- **RespiratorySystem** → convergence/anomaly = oxygen consumption. Low O2 = organism should throttle sensing.
- **ThermoregulationSystem** → convergence = heat, anomaly = spike. Overheating = shed tasks, cold = scan more.
- **VestibularSystem** → tracks convergence rate + prediction accuracy. Rapid changes = vertigo/instability alert.
- **ProprioceptionSystem** → updates body map with convergence alerter activity/health and outcome tracker health.
- **EnergyReserve** → convergence spends energy, circadian REST stores, ACTIVE releases. Leptin gates scanning.
- **PredictiveField** → convergence updates spatial field with ticker positions for agent coordination.

**Note:** GenerativeReplayMemory is the sole unbootstrapped system (needs state_dim/action_dim design decision). Deferred.

**ResourceGovernor (NEW)** — internal API budget governance. Per-source hourly limits, global hourly cap, warning at 80%, throttle events on EventBus. Replaces external governance (Paperclip rejected — Law 6 autopoietic closure).

**Files created/modified:**
- `mae_core/bootstrap/bio_market_wiring.py` — 14 Tier 2+3 wiring functions + orchestrator
- `mae_core/bootstrap/bio_market_wiring_extended.py` — NEW: 15 Tier 4+5 wiring functions
- `mae_core/market/resource_governor.py` — NEW: budget governance system
- `mae_core/learning/curiosity.py` — added `set_exploration_bonus()` method
- `mae_core/bootstrap/market.py` — calls `wire_bio_systems_to_market(ctx)` as Layer 33k
- `tests/test_bio_market_wiring.py` — 23 Tier 2+3 tests
- `tests/test_bio_market_wiring_extended.py` — NEW: 38 Tier 4+5 tests
- `tests/test_resource_governor.py` — 11 tests

### Ecosystem Activation — Wire the Octopus (2026-03-09)

**Pipeline bridge built (two disconnected pipelines now connected):**
- **OctopusColony bootstrapped** — 5 pre-existing files (~1500 lines), never wired. Now bootstrapped as part of Layer 33 with 3 octopuses, auto-scaling 3-7. Market task handlers injected (investigate_partial, archaeology_lookup, situation_check).
- **Partial convergence emission** — When ConvergenceAlerter sees 2 domains (below min_domains=3), it now emits `CH_PARTIAL_CONVERGENCE` on EventBus. Previously silently returned None. OctopusColony registers these as DevelopingSituation entries for investigation.
- **Market → Attention bridge** — `MarketConvergenceTranslator` and `MarketPartialTranslator` registered in PatternBus. Full convergence alerts → `PatternDomain.OPPORTUNITY/THREAT`. Partial convergences → `PatternDomain.NOVELTY`. Market signals now reach AttentionalGate → GlobalWorkspace for the first time.
- **Post-spawn arm patching** — Subscribe to `octopus.spawn` channel so auto-scaled arms get market handlers.
- **Group 34 connections** — 3 triadic connections (octopus↔alerter, octopus↔watcher, octopus↔eventbus).

**10-point independent review, all findings fixed:**
- 2 CRITICAL (wrong ctx attribute names, race window), 3 HIGH (spawn gap, silent monitoring, holon orphan), 3 MEDIUM (dict race, ticker key, stale count), 2 LOW (size cap, test mock).

**Files changed (12 modified, 4 new):**
- `mae_core/market/channels.py` — 2 new channel constants
- `mae_core/market/translators/__init__.py` — NEW package marker
- `mae_core/market/translators/market_signal_translator.py` — NEW: pipeline bridge translators
- `mae_core/network/market_task_handlers.py` — NEW: octopus arm dispatch handlers
- `mae_core/market/intelligence/convergence_alerter.py` — partial emission + helper methods
- `mae_core/bootstrap/market_systems.py` — OctopusColony instantiation
- `mae_core/bootstrap/market_hooks.py` — coordination cycle, spawn patching, partial callback
- `mae_core/bootstrap/market_connections.py` — Group 34
- `mae_core/bootstrap/market_registration.py` — holon + fractal registration
- `mae_core/bootstrap/market.py` — system count + octopus_colony attr
- `mae_core/bootstrap/patterns.py` — market translator registration
- `main.py` — systems dict
- `tests/test_market_signal_translator.py` — NEW: 15 tests
- `tests/test_market_task_handlers.py` — NEW: 5 tests
- `tests/test_octopus_bootstrap.py` — NEW: 3 tests

### Comprehensive Audit + Build Session (2026-03-09)

**4 broken systems fixed:**
- **FinnhubWebSocket.start() was never called** — real-time streaming was a dead wire. Now started from market_hooks.py bootstrap.
- **ActiveTracker._force_grade() called nonexistent methods** — fast price grading has never worked. Fixed to write outcomes.jsonl directly + update Thompson.
- **SAM.gov estimated_value never populated** — `get_large_contracts()` always returned empty. Now parses `estimatedValue`, `totalBaseAndAllOptionsValue`, and DoD-style fields.
- **MassiveClient volume_ratio always 0** — callers never supplied `bars_by_ticker`. Fixed computation path.

**Raw store expansion 12→24 clients:**
- All API clients now persist raw data to SQLite before processing
- raw_store.py: 296→1,686 lines, 25 store methods covering all domains
- 58 tests in test_raw_store.py (up from 35)
- SEC EDGAR (full Form 4 + derivative tables), Massive (grouped bars), CoinGecko (ATH/ATL/supply), CoinCap (assets), OpenInsider (+ SEC filing URL), FinViz (insider trades, unusual volume, short float), EdgarEnhanced (13D/13F), FinnhubWS (trade ticks), ApeWisdom (social sentiment), JobTracker (full records with skills/experience), SAM.gov (full opportunities + description text)

**Post-mortem prediction reviewer (NEW):**
- `mae_core/market/intelligence/post_mortem.py` — runs every 500 steps
- Analyzes graded predictions by combo, by domain ORDERING, by timing accuracy, by regime transition
- Detects "right thesis, wrong timing" (MFE >= 5% but final return < 2%)
- Flags domain orderings with < 30% win rate
- Pushes sequence-aware Thompson updates (seq:insider>>macro>>technical keys)
- Writes atomic `data/market/post_mortem_insights.json`

**Temporal domain ordering:**
- `ConvergenceAlert` now has `domain_sequence` (domains sorted by timestamp) and `sequence_score` (0.5-1.3 multiplier)
- Alerts where domain ordering matches known lag relationships get confidence boost
- Reversed ordering gets discounted
- Loads `lag_correlations.json` on startup for immediate scoring
- Lag findings automatically wired: every 500-step lag analysis feeds into alerter

**Cultural discovery (NEW):**
- `mae_core/market/intelligence/social_text_analyzer.py` — keyword frequency, sentiment intensity, theme detection (options_flow, short_squeeze, earnings_play, macro_fear), per-ticker aggregation
- Google Trends keyword auto-expansion via `related_queries` — discovers rising queries, stores in `data/market/discovered_keywords.json`, mixes top N into next fetch cycle
- FinViz `get_insider_trades()` wired into sensing pipeline (was fully built, never called)

**Parallelism upgrades:**
- Concurrent fetches: 3→12 (ThreadPoolExecutor)
- Fetch cadence: 50→25 steps (sources rotate 4x faster)
- Agent thread cap raised in model.py
- ExcavationDaemon moved off main thread

**WorldModel causal chain graph (NEW):**
- `mae_core/market/intelligence/world_model.py` — 114 nodes, 102 edges, 38 tradeable tickers
- Curated real-world causal chains across energy, macro, tech, defense, supply chain, geopolitical domains
- `find_ripple_effects(trigger)` traces downstream effects with strength accumulation and lag propagation
- `map_signal_to_trigger(source, metadata)` bridges MIDGE signals to world model events
- `record_outcome(trigger, ticker, was_correct)` for feedback learning
- Wired into convergence_alerter via bootstrap injection

**TA vectorization (COMPLETE):**
- RSI, MACD, Bollinger rewritten from iterative to numpy/pandas vectorized (10-50x faster)
- `ta_indicators.py` — `np.diff`, `pd.Series.ewm`, `pd.Series.rolling`
- 51/51 tests pass including 19 new vectorization tests

**Yahoo Finance RSS (NEW):**
- `mae_core/market/apis/yahoo_rss_client.py` — per-ticker RSS, 5-min cache, velocity detection
- Bullish/bearish keyword sentiment, `get_headlines()` + `get_accelerating()`
- Full sensing pipeline wiring (SOURCE_ROTATION, TIER_ROUTING, signal adapter)
- 36 tests

**STUMPY motif detector (NEW):**
- `mae_core/market/intelligence/motif_detector.py` — per-symbol streaming matrix profile
- Detects motifs (repeated patterns) and discords (anomalous subsequences)
- 50 symbol cap, 20-bar window (~1 month), LRU eviction
- Wired into bootstrap (market_systems.py + market_hooks.py)

**ADWIN drift detector (NEW):**
- `mae_core/market/intelligence/drift_detector.py` — multi-stream concept drift via ADWIN
- River ADWIN when available, pure-Python Hoeffding-bound fallback
- Tracks price_returns, volume, VIX, sentiment streams
- Wired into bootstrap (market_systems.py + market_hooks.py)

**PySAD streaming anomaly detector (NEW):**
- `mae_core/market/intelligence/streaming_anomaly.py` — RRCF anomaly scoring on composite feature vectors
- 4-element vector: [price_change, volume_ratio, sentiment_score, vix_level]
- Complements VelocityDetector (single-signal) with multi-dimensional anomaly detection
- Wired into bootstrap alongside motif_detector

**File splits (monolith prevention):**
- `post_mortem.py` 569→314 lines (computation split to `post_mortem_analysis.py` 245 lines + `post_mortem_utils.py` 57 lines)
- `market_systems.py` 535→453 lines

**Queue updated:** `midge-queue.md` now tracks 70+ items from 3 expedition syntheses that were researched but never queued.

**Files changed (30+ files):**
- All files from raw store expansion (24 clients wired)
- `mae_core/market/intelligence/world_model.py` — NEW: causal chain graph
- `mae_core/market/intelligence/motif_detector.py` — NEW: STUMPY streaming
- `mae_core/market/intelligence/drift_detector.py` — NEW: ADWIN drift
- `mae_core/market/intelligence/post_mortem.py` — split to 3 files
- `mae_core/market/intelligence/post_mortem_analysis.py` — NEW: extracted computation
- `mae_core/market/intelligence/post_mortem_utils.py` — NEW: shared helpers
- `mae_core/market/intelligence/social_text_analyzer.py` — NEW: theme detection
- `mae_core/market/intelligence/convergence_alerter.py` — domain_sequence, sequence_score, world_model
- `mae_core/market/apis/yahoo_rss_client.py` — NEW: headline velocity
- `mae_core/market/edge/ta_indicators.py` — numpy/pandas vectorized
- `mae_core/bootstrap/market_systems.py` — all new systems wired (453 lines)
- `mae_core/bootstrap/market_hooks.py` — FinnhubWS start(), drift/motif hooks, post-mortem, social analyzer

### Raw Store Expansion: 4→12 Domains (2026-03-08)

**raw_store.py expanded** from 296→849 lines with 8 new domain methods: `store_price_snapshot` (yfinance 80+ fields as JSON), `store_fred_observations` (with vintage metadata), `store_stocktwits_messages` (full message text/sentiment/user), `store_finnhub_sentiment/earnings/economic` (ALL countries, quarter/year preserved), `store_congressional_trades` (house+senate), `store_congress_bills` (sponsors, committees), `store_finra_short_volume`.

**8 clients wired** — price_fetcher, FRED, StockTwits, Finnhub, FINRA, House, Senate, Congress.gov all now persist raw data before processing. Total: 12 of 24 clients wired. Bootstrap passes `raw_store` to all 12.

**35 tests** — up from 14. All pass. Zero regressions on full suite.

### Execution Bridges + Sensing Overdrive + Data Audit (2026-03-07b)

**Alpaca bridge** — `alpaca_client.py` built. Paper trading with bracket orders (TP+SL), position tracking, account info. `alpaca-py 0.43.2`, `kalshi-python 2.1.4`, `httpx+selectolax+trafilatura` all installed. Awaiting `ALPACA_API_KEY` + `ALPACA_SECRET_KEY` in `.env`.

**OpenInsider cluster buys WIRED** — `get_cluster_buys()` was fully implemented but never called. Now fires high-confidence signals when 3+ insiders buy same stock within 30 days.

**FINRA speculative short ratio** — `ShortExemptVolume` was parsed then discarded. Now `speculative_short_ratio` separates market-maker structural shorts from speculative shorts.

### Raw Data Pipeline + Expedition Synthesis + Vision Alignment (2026-03-07)

**Raw Data Pipeline** — MIDGE was throwing away ~95% of API data. Now ALL data persisted to SQLite before processing.

**Expedition complete: Autonomous Self-Funding Trading** — synthesis at `research/expedition-autonomous-trading/synthesis.md`.

### Previous Sessions

See git log for full history. Key milestones: Granger causality (2026-03-07), template persistence fix (2026-03-06), Thompson + independence fix (2026-03-05), prediction-to-action (2026-03-05), pattern archaeology v2 (2026-03-04), EIA energy (2026-03-06).

---

## Stats

- **149 systems** (92 core + 57 market), **4,536+ tests**, **157 holons**, **428 connections**
- **119 market files** (33 API + 12 edge + 36 intelligence + 8 signal_adapters + 10 archaeology + 3 translators + 17 root)
- **12 domains**, **32 sources** in sensing rotation (Yahoo RSS + FinViz insider trades added)
- **33-layer bootstrap**, **14 mixins** on MycelialAgent
- **222,916 fingerprints**, **39 templates** (26 cross-validated across 3+ symbols)
- **24/24 API clients** now persist raw data to SQLite

## Current State

- **Daemon: STOPPED.** Needs restart with: `python main.py --daemon --agents 12 --steps 500 --pace 2.0`
- **Raw Data Store: COMPLETE.** All 24 clients store raw data before processing. SQLite per domain in `data/market/raw/`.
- **FinnhubWebSocket: FIXED.** start() now called at bootstrap. Real-time streaming operational.
- **WorldModel: WIRED + BIDIRECTIONAL.** 114 nodes, 102 edges, 38 tickers. Forward: `find_ripple_effects()`. Backward: `find_root_causes()`. Feedback loop closed via CascadeTracker.
- **CascadeTracker: ACTIVE.** Watches causal chains unfold. Confirms links. Energy ratio tracking (actual vs predicted lag). Feeds WorldModel outcomes.
- **Focused Attention: ACTIVE.** Priority polling boosts Thompson scores for missing domains when partial convergence fires.
- **Chain Boost: ACTIVE.** Cascade confirmations inject synthetic "cascade" domain signals into convergence engine.
- **Backward Discovery: ACTIVE.** Mid-chain signal detection traces backward to find genesis. Populates priority queue or registers late-joining cascades.
- **Post-Mortem: WIRED.** Runs every 500 steps alongside Granger. Writes insights to `data/market/post_mortem_insights.json`.
- **Temporal Ordering: ACTIVE.** domain_sequence + sequence_score on every convergence alert.
- **Social Text: WIRED.** SocialTextAnalyzer reads StockTwits messages from raw_store, emits theme signals.
- **Trends Discovery: ACTIVE.** Related queries feed keyword expansion. Up to 30 discovered keywords.
- **TA Vectorization: COMPLETE.** RSI/MACD/Bollinger numpy-vectorized (10-50x faster).
- **Motif Detector: WIRED.** STUMPY streaming matrix profile, 50 symbols, LRU eviction.
- **Drift Detector: WIRED.** ADWIN concept drift on price_returns/volume/VIX/sentiment streams.
- **Streaming Anomaly: WIRED.** PySAD RRCF on composite feature vectors (price/volume/sentiment/VIX).
- **Yahoo RSS: WIRED.** Per-ticker headline velocity + sentiment. 32 sources in rotation.
- **Parallelism: 12 concurrent + 25-step cadence.** Full source rotation in ~94 steps.
- **Templates: REBUILT.** 39 templates live. PatternWatcher matching.
- **Thompson: FIXED.** Forgetting/learning aligned. Independence correction active.
- **OctopusColony: WIRED.** Bootstrapped in Layer 33 with 3 octopuses. Market handlers injected (investigate_partial, archaeology_lookup, situation_check). Partial convergences emitted from alerter. Auto-scaling arms get patched on spawn.
- **Pipeline Bridge: ACTIVE.** MarketConvergenceTranslator + MarketPartialTranslator registered in PatternBus. Market signals now reach AttentionalGate → GlobalWorkspace via PatternTranslator protocol.
- **Bio-Market Activation: COMPLETE.** 29 of 30 biological systems wired to market EventBus channels (14 Tier 2+3 + 15 Tier 4+5). Only GenerativeReplayMemory remains unwired (needs bootstrap). Every system from DigestiveSystem (data=nutrients) to PredictiveField (spatial coordination) now has a real market job. ResourceGovernor built for API budget governance.
- **Kalshi: SDK INSTALLED.** Needs verification against demo env.
- **Alpaca: CLIENT BUILT.** Awaiting API keys.

## Guiding Light's Ecosystem Vision (2026-03-08)

> "MIDGE needs to be an entire functioning ecosystem. She's more of a planet than a singular biological organism. Everything inside her should be active, not passive. They all have their own lives to live."

This fundamentally reframes the architecture. Components are LIVING ENTITIES with intrinsic drives, not passive systems called by step hooks. The 41 "dead" biological systems (pheromones, quorum sensing, immune, nociception, curiosity, circadian, etc.) should be ACTIVATED with real market intelligence jobs, not shed. Attention should EMERGE from collective activity. Confidence should be emergent consensus (quorum), not formula.

**Evolution blueprint**: `research/evolution-blueprint/MIDGE-EVOLUTION-BLUEPRINT.md` — 10-team synthesis.

**Critical finding (NOW FIXED)**: Two disconnected pipelines bridged. OctopusColony bootstrapped. Market signals now reach core attention via PatternTranslator protocol.

## What's Next

See `midge-queue.md` for comprehensive task list (70+ items, 4 priority tiers).

### Immediate
1. **Activate biological systems** — Pheromones=trail-leaving, Quorum=collective confidence, Immune=deception patrol, Curiosity=intrinsic exploration, Circadian=market cycle awareness. These 33 systems are running but market-disconnected.
2. **Agent-level situation claiming** — SEC_WATCHER claims insider situations, MARKET_ANALYST claims highest-convergence.
3. **DevelopingSituation → full investigation** — Octopus arms query archaeology, prediction markets, targeted re-fetch.

### Priority 1: Ecosystem Deepening (from Evolution Blueprint Phase 2)
- ~~Focused attention: when partial convergence starts, increase polling priority for missing domains~~ **DONE**
- Agent-level situation claiming (SEC_WATCHER claims insider situations)
- ~~DevelopingSituation → full investigation pipeline~~ **DONE**
- ~~Signal-triggered convergence (fire on signal arrival, not step tick)~~ **DONE**

### Priority 2: Shed Weight + Speed Up (Evolution Blueprint Phase 1)
- Wall-clock cadences (replace step-based)
- ~~Signal-triggered convergence (fire on signal arrival, not step tick)~~ **DONE**
- ADTS (regime-aware Thompson forgetting, 50 lines)
- Unload 223K fingerprints from RAM

### Priority 3: Execution + Risk (Evolution Blueprint Phases 4-5)
- Kalshi SDK verification + MarketSelector prototype
- DrawdownMonitor, SystemHealthMonitor, SelfMonitor
- Broker-side bracket orders (survive MIDGE process failure)

## Verification

```bash
python -m pytest tests/ -q              # Should be 4536+
python -m pytest tests/test_raw_store.py -v  # 58 tests
python main.py --agents 3 --steps 30    # Smoke test
```

## Flags

- `raw_store.py` at 1,686 lines (large but single-responsibility — each method is independent)
- `market_hooks.py` may be approaching size cap — verify and split if needed
