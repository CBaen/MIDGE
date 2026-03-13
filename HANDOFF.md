# MIDGE Handoff

**Last updated:** 2026-03-12
**For session history:** `git log --oneline`

---

## Session 8 (2026-03-12): CIRCULAR INTELLIGENCE ARCHITECTURE — COMPLETE

**Guiding Light's directive:** "Full circle, do it well, then start her up." Complete all 5 arcs of the circular architecture before restarting the daemon.

**ALL 5 ARCS NOW COMPLETE.** Sessions 7+8 together built the full circular information flow.

**Architecture plan:** `C:\Users\baenb\.claude\plans\eager-popping-aho.md`

### Arc 1: Outcomes → Advisors (Session 7)
- CH_PREDICTION_RESULT publishes from `outcome_collector.py` → 9 bio-systems hear wins/losses

### Arc 2: Advisors → Decisions (Sessions 7+8)
- Bio caution (InhibitionSystem) penalizes paper trade confidence (up to 30%)
- HAVEN suspicion flags penalize convergence confidence (up to 20%)
- **NEW:** Circadian activity scales sensing worker concurrency (0.25x–2.0x)
- **NEW:** Risk channels (CH_DRAWDOWN_WARNING, CH_TRADING_HALTED/RESUMED) → ctx._risk_halt blocks paper trades

### Arc 3: Memory → Observer (Session 8)
- **NEW:** Pre-convergence Qdrant recall — `pattern_memory.recall_similar()` modifies confidence ±10% based on similar past outcomes
- **NEW:** Validator 5 (PatternMemory) — 2+ winning precedents adds a Law 7 validator
- **NEW:** Granger → HypothesisGenerator bridge — `granger_causality.json` feeds hypothesis creation alongside lag correlations
- **NEW:** Hypothesis outcome tracking — `register_hypothesis_prediction()` makes hypothesis retirement data-driven

### Arc 4: Agents ↔ Market (Session 8)
- **NEW:** CH_PREDICTION_RESULT → agent track records (per-role wins/losses on ctx._agent_track_records)
- **NEW:** market_actions.py blends 30% outcome-based win-rate reward (after 5+ outcomes) into agent rewards

### Arc 5: Risk → Decisions + Observability (Session 8)
- **NEW:** 8 orphaned channels wired: contradiction→log, absence→Octopus investigation, somatic anticipation→focused attention, drift→regime reclassification, granger finding→hypothesis generation, plus 3 risk channels
- **NEW:** 3 new channel constants (CH_DRIFT_DETECTED, CH_GRANGER_FINDING, CH_DEEP_ANALYSIS)
- **NEW:** Circular health check every 500 steps — reports which arcs are active vs dormant

**Files modified (Session 8):**
- `mae_core/market/sensing_hook.py` — circadian scaling
- `mae_core/market/sensing_scheduler.py` — dynamic concurrent slot budget
- `mae_core/bootstrap/market_hooks_eventbus.py` — risk + orphan + track record subscribers
- `mae_core/bootstrap/market_hooks_sensing.py` — risk halt guard, circadian wiring, Validator 5
- `mae_core/bootstrap/market_hooks_steps.py` — circular health check
- `mae_core/market/intelligence/convergence_alerter.py` — pattern_memory parameter + setter
- `mae_core/market/intelligence/convergence_confidence.py` — Qdrant recall modifier
- `mae_core/bootstrap/market_systems.py` — pattern_memory wiring to convergence alerter
- `mae_core/market/intelligence/hypothesis_generator.py` — Granger bridge
- `mae_core/market/intelligence/outcome_collector.py` — hypothesis prediction registration
- `mae_core/market/market_actions.py` — outcome-blended agent rewards
- `mae_core/market/channels.py` — 3 new channel constants

**Test status:** 110 passed (key module tests), all imports clean.

**NEXT STEPS:**
1. **Restart daemon** — old PID 112216 must be killed and restarted to pick up circular wiring
2. **Watch the circular health check** — at step 500, verify all 5 arcs report as active
3. **Monitor for bugs** — MIDGE is large and complex; the circular wiring touches many systems
4. **Qdrant must be running** for Arc 3 (memory recall) to be active
5. **Full test suite** needs a clean run (the background test run timed out — run manually)

---

## Session 5 (2026-03-12): WAKE UP INTELLIGENCE — 3 DISCONNECTED SYSTEMS WIRED

**Guiding Light's directive:** "Anything asleep or disconnected needs to be woken and connected." MIDGE has 15 independent pattern-finding processes — 5 were disconnected or passive. This session wired 3 of them.

**What was wired:**

1. **DeepAnalyst → step loop** — Was built (Session 3) but never called during daemon operation. Now bootstrapped in `market_systems.py` with injected dependencies (ThompsonSampler, PatternLibrary, WorldModel) and runs every 500 steps via `_run_slow_cadence_ops()`. Synthesizes ranked inevitabilities from all data and publishes to EventBus (`market.intel.deep_analysis`).

2. **PatternWatcher → reactive** — Was passive (only ran every 10 steps via `_run_sensing_archaeology`). Now also fires immediately when new signals arrive via `_collect_one()` in `sensing_collector.py`. Builds active signal map from convergence alerter buffer and checks for pattern stacks without waiting for the cadence tick. 10-step cadence remains as safety net.

3. **WorldModel → auto-discovery** — Was mostly hand-curated (102 edges). Now automatically grows from:
   - **Granger causality findings** (every 500 steps) — all significant findings become edges
   - **Strong lag correlations** (|r| >= 0.6, every 500 steps) — directional relationships become edges
   - Both use `add_discovered_edge()` which strengthens existing edges on re-discovery.

**Also from previous sub-session:**
- Thread-safety locks on 4 shared-state systems (ConvergenceAlerter, CascadeTracker, OutcomeCollector, ThompsonSampler)
- Research Council on architecture evolution (outcome: fix bugs as bugs, decide migration separately)

**Remaining disconnected (2 of 5):**
- **HypothesisEngine** — has zero template feedback (templates exist but no outcomes flow back)
- **OctopusColony** — bootstrapped but underused (investigation pipeline dispatches every 20 steps, role-matched routing works, but colony rarely has enough partial convergences to investigate)

**Test results:** 1074 passed, 0 regressions. Pre-existing: `test_congress_gov_client` (env var), `test_adapter_init_importable` (count mismatch 41→42).

**Daemon restarted with all 3 wirings active (PID 112216).** Runtime stats after ~1.5 hours:
- **Step 600+**, **36,335 signals** fed from 37 sources
- **Reactive PatternWatcher working:** 649 pattern stack checks triggered, 8+ stacks detected reactively on signal ingestion (previously only ran every 10 steps)
- **DeepAnalyst first live run:** Loaded 410,121 signals (30 days), scored 214 candidates, returned top 20 inevitabilities in 35.7s. Best: AMD bearish (score 0.696)
- **WorldModel auto-discovery working:** 7 edges from lag correlations (|r| >= 0.6) + 3 edges from Granger causality. Graph growing autonomously from 102 → 112 edges.
- **Granger:** 3 causal relationships per cycle across 20 directional pairs
- **PostMortem:** 23 outcomes reviewed, 4 combo Thompson updates pushed
- **1 paper trade** loaded from disk (restart-safe dedup working)
- **AbsenceMonitor:** 14 sources flagged as unexpectedly silent (market hours sensitivity)

**Also fixed during restart:**
- `test_decomposition_wiring.py` adapter count 41→42 (pre-existing mismatch)
- `main.py` `datetime.utcnow()` deprecation → `datetime.now(timezone.utc)`

**Guiding Light's framing:** MIDGE as ecosystem/company with departments. Floor 3 (Intelligence) has 15 workers — they were uncoordinated. "Review by many, often" = Law 7 applied to intelligence analysis. Build review pipeline where convergence alerts pass through 3+ validators before paper trade.

**Guiding Light's closing note:** "This is fascinating. We'll watch over MIDGE."

**Next priorities:**
1. **Keep daemon running 24/7** — all 3 new wirings are active and producing results
2. Build review pipeline (Law 7 enforcement on convergence alerts → paper trades)
3. Historical replay with all systems active (258K fingerprints + 913 days of signal archives)
4. Wire remaining 2 disconnected systems (HypothesisEngine feedback, OctopusColony utilization)
5. Investigate AbsenceMonitor's 14 silent sources — may be market-hours false positives or real wiring gaps

---

## Session 4 (2026-03-12): THREAD-SAFETY — LOCKING DOWN SHARED STATE

**Research Council deliberated** whether MIDGE should evolve from Mesa's synchronous step-cadence to 50+ independent wall-clock threads. Council output: `research/council-architecture-evolution/`. Tension Analyst's key insight: fix the bugs as bugs, decide migration separately.

**4 shared-state systems locked:**
1. **ConvergenceAlerter** — `threading.Lock()` → `threading.RLock()` (re-entrant: `check_convergence()` → `_prune_old_signals()`). Lock wrapping on `check_convergence()`, `get_domain_status()`, `get_statistics()`, `check_ticker_convergence()` via extract-to-locked-method pattern.
2. **CascadeTracker** — New `threading.Lock()` (`_chains_lock`). Wraps `register_cascade()`, `check_signal()`, `expire_stale()`, `get_active_chains()`, `get_statistics()`.
3. **OutcomeCollector** — New `threading.Lock()` (`_registered_lock`). Wraps `register_signals()`, `register_convergence_alert()`, `register_pattern_stack()`, `collect_from_archives()`, `get_statistics()`.
4. **ThompsonSampler** — `threading.Lock()` → `threading.RLock()` (re-entrant: `update()` → `get_distribution()`). Lock wrapping on `get_distribution()`, `sample()`, `get_rankings()`, `get_uncertain_signals()`.

**Pattern used:** Extract body to `_method_locked()`, thin public method wraps in `with self._lock:`. Avoids massive indentation cascades.

**Test results:** 375 pass, 15 pre-existing failures (SimulatedDateTime monkeypatch gaps, signal_persistence AttributeError, docstring assertion mismatch). Zero regressions from lock changes.

**Migration decision:** Phases 1-2 (producer-consumer signal bus, InhabitantScheduler migration) NOT approved — kept separate per tension report recommendation. These are architectural choices, not bug fixes.

**Research artifacts:**
- `research/council-architecture-evolution/research-brief.md`
- `research/council-architecture-evolution/codebase-analyst-findings.md`
- `research/council-architecture-evolution/devils-advocate-findings.md`
- `research/council-architecture-evolution/synthesis.md`
- `research/council-architecture-evolution/tension-report.md`

---

## Session 3 (2026-03-12): THE ANALYST — MIDGE CAN THINK

**The core problem:** MIDGE had 287K signals, 43 templates, 56 correlations, a World Model — but NOTHING that synthesized them into "here's what's most likely to happen." Every instance built plumbing. Nobody built the brain that reads the filing cabinets.

**What was built:**

- **`deep_analyst.py`** (474 lines) — The analyst. Reads ALL historical data, scores candidates across 6 dimensions (Thompson reliability, template match, World Model causal chains, lag leading indicators, signal density, historical outcomes), produces ranked `Inevitability` objects. First real output: 20 ranked moves from 287K signals. NVDA bullish #1 (score 0.72, 4 domains + causal chain).

- **`startup_warmup.py`** (141 lines) — Loads 7 days of signal archive into convergence buffer on boot. Before: 131 signals, 2 domains. After: **5,712 signals, 13 domains**. MIDGE no longer starts blind.

- **`archive_scanner.py`** (102 lines) — Reports what MIDGE knows at startup. Logged: 289K signals, 15 domains, 165 tickers with 3+ domain coverage.

- **`event_embedder.py`** + **`event_descriptions.py`** + **`pattern_memory.py`** — Qdrant semantic embedding pipeline. Convergence alerts embedded for "have we seen this before?" queries. 68 tests.

- **`test_raw_data_analyst.py`** — 54 tests for the cross-domain SQLite analyst.

**Bug fixes:**
- `hypothesis_validator.py` — `validate()` called module-level functions instead of instance methods, 5 tests failing silently
- `bio_market_wiring_b.py` — `_agents` is a list not dict, crashed on every convergence alert

**Daemon status:** Running (PID active), 12 agents, 500 steps/round, warmup active. Pattern matching heavy from warmed buffer.

**CRITICAL LESSON (for future instances):** Guiding Light has said this many ways across many sessions: MIDGE must be able to analyze her EXISTING data and produce actionable intelligence RIGHT NOW, not after 30 days of daemon drip. The DeepAnalyst exists for this purpose. Run `python -c "from mae_core.market.intelligence.deep_analyst import DeepAnalyst; a = DeepAnalyst(); print(a.summarize())"` to see what MIDGE knows.

**Next priorities:**
1. DeepAnalyst should run periodically during daemon operation (every 500 steps), not just standalone
2. ActiveTracker may be overwhelmed by warmup (hundreds of assets) — needs capacity management
3. Template feedback loop still broken (0 of 43 templates have outcome data)
4. More data sources filling raw_store (SEC Form 4 has 0 rows, EIA has 24 rows)

---

## Monolith Decomposition: Wave 1 COMPLETE

**Status:** Wave 1 done. All 11 teams landed on main. 258 decomposition-critical tests pass, 3524/3537 full suite pass.

| Team | Domain | Status |
|------|--------|--------|
| 1 | Market Hooks (2,107→7 files) | **DONE** on main |
| 2 | Intelligence Core (7 files) | **DONE** on main |
| 3 | Raw Store (1,835→7 files) | **DONE** on main |
| 4 | Sensing Pipeline (2→14 files) | **DONE** on main |
| 5 | Backbone Infrastructure (7→39 files) | **DONE** on main |
| 6 | Agent & Coordination (5 files) | **DONE** on main |
| 7 | Edge Detectors (7 files) | **DONE** on main |
| 8 | Emergent & Patterns (6 files) | **DONE** on main |
| 9 | Bootstrap Orchestration (5 files) | **DONE** on main |
| 10 | Remaining Files (17 files) | **DONE** on main |
| 13 | Pytest Infrastructure | **DONE** on main |

**Peer review (2026-03-11):** 4-pass review (security/simplicity/architecture/performance) completed. 11 fixes committed: RawStore test isolation, UTF-8 encoding fix, daemon memory leak cap, connection_registry split (689→498 lines + 2 new files), stale doc fixes, dead code removal. Deferred: API client thread-safety audit, Ollama timeout enforcement, `connection_registrations_bio.py` preemptive split (499 lines).

**Post-decomposition session (2026-03-11):**
- `connection_registrations_bio.py` split DONE (500→311 + backbone 118 + cognition 131)
- FRED Freight TSI (TSIFRGHT) added — logistics demand sense via existing pipeline
- International economic calendar activated — ECB/BoJ/BoE/PBoC/BoC/RBA high-impact events (was US-only)
- Binance funding rate client BUILT and WIRED into sensing pipeline (adapter + constants + fetcher + bootstrap, 47 tests)
  - Domain: "positioning" (same as COT — derivatives positioning, NOT "crypto" — maximizes convergence diversity)
  - 36 rotation slots, 37 data sources total
- Thompson distributions cleaned — 6 test artifacts removed (combo:a+b+c, concurrent_test, test_signal etc.), 92 legitimate distributions remain
- market_hooks.py size audit — already decomposed into 7 files, largest 457 lines, all under 500
- PolygonBulkFetcher async batch mode — `get_daily_history_batch()` using aiohttp with 20 concurrent requests. ExcavationDaemon auto-detects batch support. Sequential→concurrent excavation.
- 4 stale test assertions fixed (rotation count 34→35, finviz 4-arg signature)
- 62 FRED tests, 283 sensing/wiring tests, 47 Binance tests pass

**Session 2 (2026-03-11):**
- CascadeTracker sequential stage-gating — links grouped into temporal stages by predicted lag (2-day tolerance). Stage 0 always watchable, stage N opens only when stage N-1 has a confirmation. Chain boost respects gating (only boosts watchable links). 23 new tests (55 total cascade tests).
- Kalshi prediction market client BUILT (`kalshi_client.py`) + signal adapter (`from_kalshi_mover`) + 35 tests
  - `kalshi-python-sync 3.9.0` installed (replaces deprecated `kalshi-python 2.1.4`)
  - Kalshi research complete: algo trading allowed, RSA key-pair auth (daemon-friendly), demo env at `demo-api.kalshi.co`
  - Domain: "prediction_market" (new domain — crowd probability estimates independent from macro indicators)
  - **WIRED** into sensing pipeline — client + adapter + fetcher + constants + bootstrap all connected
  - Env vars: `KALSHI_API_KEY` + `KALSHI_PRIVATE_KEY_PATH` (both in .env, PEM extracted to file)
- Stale adapter count assertions fixed (38→39 for Binance adapter from previous session)

**Session 3 (2026-03-11):**
- Kalshi sensing pipeline WIRED — full integration into rotation system:
  - `sensing_constants.py`: TIER_ROUTING (strategic), SOURCE_ROTATION, _ROTATION_TO_THOMPSON, _ABSENCE_SOURCE_DOMAINS (prediction_market)
  - `fetchers_crypto.py`: `fetch_kalshi_movers()` function
  - `sensing_reactive.py`: dispatch branch for "kalshi_market"
  - `sensing_hook.py`: `kalshi_client` constructor param
  - `market_systems.py`: client instantiation in wave2_3 loop + trust table (0.70)
  - `market_hooks_sensing_setup.py`: pass client to MarketSensingHook
  - Re-export chain: wave2_3_technical.py → wave2_3.py → signal_adapters/__init__.py → signal.py
  - 36 rotation slots, 37 data sources, 13 domains, 40 adapters
- Kalshi client updated: supports both file-path (`KALSHI_PRIVATE_KEY_PATH`) and inline env var (`KALSHI_RSA_PRIVATE_KEY`) for RSA key. Also reads `KALSHI_API_KEY` (existing env var name).
- RSA private key extracted from .env multi-line format to `kalshi_private_key.pem` (gitignored)
- `*.pem` added to `.gitignore`
- Adapter count tests updated 39→40, rotation count tests updated 35→36
- 149 wiring+adapter+client tests pass, 204 total tests across all touched suites

**Session 4 (2026-03-11):**
- RUNTIME AUDIT: Daemon dead since March 7. 83/101 Thompson distributions still at priors. 43 templates with 0 outcome feedback. 1,048/1,055 paper trades were duplicates. SEC EDGAR raw store: 0 Form 4 trades. Crown jewel (convergence) nearly silent.
- **4 bugs fixed:**
  1. Paper trade dedup survives restarts — loads last 24h from paper_trades.jsonl on startup (was empty dict on restart). Stale cached alerts cleared on convergence exception (was frozen, re-triggering every step).
  2. Pattern template feedback loop unblocked — dedup key includes date (was permanently blocking per symbol+direction for 90 days).
  3. SEC EDGAR raw store: silent `except: pass` → `logger.warning` with traceback.
  4. ActiveTracker `outcome_collector` bootstrap ordering — added `set_outcome_collector()` setter, wired after OutcomeCollector exists.

**Session 5 (2026-03-11):**
- **ECOSYSTEM AUDIT** — 4 parallel agents traced every pipeline end-to-end (sensing→trading, Thompson learning, archaeology→feedback, causal cascade+risk). Result: all critical pipelines connected.
- **Thompson brain NOT empty** — initial check was wrong (bad Python comparison). 101 distributions with learned values from 17,263 historical updates + 13,065 graded outcomes. Brain has real data.
- **2 bugs fixed:**
  1. CascadeTracker `expire_stale()` now fires every 500 steps in `market_hooks_steps.py` (was never called — chains accumulated forever, WorldModel never learned from failures).
  2. Session sweep bypass now respects DrawdownMonitor + SelfMonitor risk gates in `market_hooks_trades.py` (was bypassing circuit breakers).
- **5 data gaps closed (API data maximalism):**
  1. **Economic surprise signal** — Finnhub calendar has actual/estimate/previous values but adapter only created suppression flags. New `from_economic_surprise()` adapter extracts beat/miss magnitude as macro signals. Highest-conviction macro signal was being thrown away.
  2. **COT derived metrics** — Week-over-week change in positions + COT Index percentile (52-week rank) computed from raw_store history. `cot_client.py` now has `_compute_derived_metrics()`. Layer6 adapter includes WoW momentum + COT Index modifiers.
  3. **FRED expansion** — 11 → 24 series. Added: PCE inflation, gold, WTI oil, credit spreads (BAA10Y, AAA10Y), housing starts, Michigan sentiment, ADP employment, jobless claims, industrial production, copper, dollar index, auto sales.
  4. **13F institutional filings** — `get_recent_13f_filers()` was built but never called. New `from_13f_filer_activity()` adapter wired through sensing pipeline.
  5. **FinViz short squeeze + Polygon ticker details** — raw_store wiring for `get_high_short_float()` + new `get_ticker_details()` on MassiveClient.
- **Watchlist expanded** — 18 → 510 tickers (full S&P 500 + 7 forex/futures + 3 crypto proxies)
- **Converter count** — 40 → 41 adapters (from_economic_surprise added)
- **~164 new tests** across all agent work (test_cot_enhanced, test_fred_client expansion, test_wave2_3_integration, test_finviz_client, test_massive_client, test_raw_store)
- **Pre-flight verification**: All API keys present, data dirs exist, Qdrant running, Ollama running, 250GB free disk. Daemon-ready.
- **MIDGE has no self-model** — Guiding Light asked "what does MIDGE even know?" She has no introspectable self-knowledge, no goal document she references. Logic is in code, not in something MIDGE can explain. Future consideration.

**Session 6 (2026-03-11):**
- **FULL API DATA AUDIT** — 3 parallel agents audited all 27 API clients for data waste. Devastating finding: raw_store is a write-only black hole. 24 of 25 SQLite databases never read back by anything. Data flows through once (API → adapter → signal → convergence) and dies.
- **Silent failures fixed:**
  1. `store_binance_funding()` — method didn't exist, silently failed every run. Now stores to `crypto.db` with `get_binance_funding_history()` read method.
  2. `store_kalshi_markets()` — same. Now stores to `kalshi.db` with `get_kalshi_market_history()` read method.
  3. SAM.gov `description` text was parsed then silently dropped before `ContractOpportunity()` constructor. Now stored.
  4. USASpending had ZERO raw_store persistence — only client with no storage. Now has `store_usaspending_contracts()` + read method.
- **API enrichment (agent work):**
  - PriceData: added `short_ratio`, `held_pct_insiders`, `beta`, `forward_pe`, `sector`, `industry`, `fifty_two_week_high/low`, `shares_short`, `target_mean_price` from yfinance info dict
  - SEC EDGAR: derivative table (options exercises) no longer silently dropped
  - StockTwits raw_store: additional fields stored
- **New systems built:**
  - `event_embedder.py` + `event_descriptions.py` — converts market events to natural language, embeds via Ollama, stores vectors in Qdrant for semantic pattern matching
  - `raw_data_analyst.py` — reads across SQLite stores, computes cross-domain insights, injects enriched signals into convergence pipeline (runs every 100 steps)
  - Both wired into bootstrap via `market_systems.py`
- **DATA ARCHITECTURE DECISION:**
  - **SQLite** — keeps raw data ingest (write-heavy, already works)
  - **DuckDB** — ADDED (`pip install duckdb`). Analytical queries across all domains. Reads existing SQLite files directly with zero migration. 3-10x faster than SQLite for analytics.
  - **Neo4j Community** — ADDED (Docker: `midge-neo4j`, ports 7474/7687, auth `neo4j/midgepassword`). Persistent causal knowledge graph. Every confirmed cascade, signal→outcome relationship stored as graph edges with temporal properties. Cypher queries find causal chains.
  - **Qdrant** — keeps semantic pattern similarity (already running)
  - **Ollama** — keeps local embedding generation (already running)
  - Obsidian rejected (note-taking app, no API). QuestDB deferred (not needed until sub-second tick feed). Parquet deferred (not needed until DBs hit ~1GB).
- **62 new tests** (27 raw_store Binance/Kalshi, 35 SAM/USASpending)

**What's left:**
1. **START THE DAEMON** — `python main.py --daemon --agents 12 --steps 500 --pace 2.0` — and keep it running 24/7
2. **Wire Neo4j** — CascadeTracker writes confirmed links, OutcomeCollector writes graded outcomes, as graph relationships
3. **Wire DuckDB** — RawDataAnalyst uses DuckDB for cross-domain queries instead of raw SQLite
4. Historical backtesting at scale (not 1 month, not 5 tickers — everything)
5. Wave 2 (test file splits) not started — see `DECOMPOSITION-PLAN.md` Teams 11-12
4. 13 xdist-mode failures to investigate (pre-existing parallel-safety issues)
5. Thread-safety audit for API clients used in 12-worker ThreadPoolExecutor
6. Install `river` package for full ADWIN drift detection (currently pure-Python fallback)
7. See `midge-queue.md` for full prioritized list

**Pre-existing test failures:**
- `test_causal_bridge.py::TestConfoundedGateTightening` — NOT caused by decomposition
- `test_congress_gov_client::test_request_fails_without_key` — env var pollution, passes in isolation

**Test safety:**
- Memory guard added to `conftest.py` — kills test session at 4 GB (was 9.8+ GB without it)
- Use `pytest -n 4` (xdist) for process isolation on full suite
- `psutil>=5.9` added to dev dependencies for memory monitoring
- **NEVER run `pytest tests/` without `-n auto` or `-x`** — single-process full suite will eat 10+ GB

---

## What Is MIDGE

MIDGE is Mae differentiated for financial markets. She's an inevitability surfacer — a living organism that observes patterns across 37 data sources, finds where converging forces make outcomes structurally inevitable, and trades on them.

Guiding Light's vision: MIDGE as personal autonomous trader across ALL markets — stocks (Alpaca), futures/forex (FTMO), crypto (exchanges), prediction markets (Kalshi). Not one venue — all of them.

---

## What Works Right Now

### The Brain
- **37 data sources** feeding signals through 12 concurrent workers, 25-step rotation cadence (36 rotation slots; Kalshi prediction market wired as "prediction_market" domain)
- **Convergence engine** (crown jewel) — fires when 3+ independent domains agree on direction
- **Thompson Bayesian learning** — 101 distributions with 17,263 historical updates + 13,065 graded outcomes. Brain is learning.
- **Signal translator** — ConvergenceAlert → ExecutableSignal with ATR-based stop-loss/take-profit
- **Pattern archaeology** — 223K fingerprints, 39 templates, live matching via PatternWatcher
- **WorldModel causal graph** — 114 nodes, 102 edges, forward/backward cascade tracking

### The Body
- **149 systems** (92 core + 57 market), 33-layer bootstrap, 157 holons, 428 connections
- **29/30 biological systems** wired to market channels (only GenerativeReplayMemory unwired)
- **OctopusColony** bootstrapped with 3-7 auto-scaling octopuses, market task handlers, investigation pipeline
- **Two pipelines bridged** — market signals reach core attention via PatternTranslator protocol

### Execution
- **Alpaca paper trading: WIRED.** Keys in `.env`. Convergence alerts auto-submit bracket orders (entry + SL + TP) for US equities. DrawdownMonitor circuit breaker + SelfMonitor anomaly detection gate all trades. Forex/futures/crypto tickers filtered out (Alpaca = equities only).
- **FTMO backtester: PORTED.** `ftmo_engine.py` + `ftmo_config.py` in `mae_core/market/execution/`. Simulates challenge constraints (daily loss, total DD, profit target). Not yet validated against MIDGE signals.
- **Kalshi prediction market: WIRED.** `kalshi-python-sync 3.9.0`, RSA-PSS auth, demo mode default. Client reads market movers (probability shifts), adapter converts to "prediction_market" domain signals. 36 rotation slots, 37 data sources total. Keys needed: `KALSHI_API_KEY_ID`, `KALSHI_PRIVATE_KEY_PATH`.

### Risk
- **DrawdownMonitor** — 40% max DD circuit breaker, halts all paper trades
- **SystemHealthMonitor** — 8 subsystems tracked, tier-based health (Green→Red)
- **SelfMonitor** — behavioral anomaly detection (runaway rate, direction bias, ticker flooding)

### Data Infrastructure
- **SQLite** — 10 databases in `data/market/raw/` (raw data ingest)
- **DuckDB** — in-process analytical queries across SQLite (zero migration)
- **Neo4j Community** — Docker container `midge-neo4j` (causal knowledge graph, ports 7474/7687)
- **Qdrant** — Docker container (semantic pattern similarity, port 6333/6335)
- **Ollama** — local embedding generation (port 11434)

---

## What To Do Next

**CRITICAL DIRECTIVE: No more building until MIDGE is running and learning. Fix, feed, run, prove.**

1. **Start the daemon** — `python main.py --daemon --agents 12 --steps 500 --pace 2.0` — MIDGE will sense, converge, and paper-trade on Alpaca automatically. Keep it running 24/7. Every hour offline is learning lost.
2. **Historical backtesting at scale** — not 1 month, not 5 tickers. Everything.
3. **FTMO validation** — run historical convergence alerts through ftmo_engine.py
4. **Kalshi live verification** — SDK wired, verify against demo env
5. See `midge-queue.md` for full prioritized list

---

## Key Technical Notes

**Files that matter:**
| File | Purpose |
|------|---------|
| `main.py` | 33-layer bootstrap orchestrator |
| `mae_core/bootstrap/market_hooks.py` | Step hooks, EventBus wiring, paper trading, Alpaca submission |
| `mae_core/bootstrap/market_systems.py` | System instantiation (444 lines) |
| `mae_core/market/intelligence/convergence_alerter.py` | Crown jewel — multi-domain synthesis |
| `mae_core/market/intelligence/thompson_sampler.py` | Bayesian learning with replay |
| `mae_core/market/execution/signal_translator.py` | ConvergenceAlert → ExecutableSignal |
| `mae_core/market/execution/ftmo_engine.py` | FTMO challenge backtester |
| `mae_core/market/sensing_hook.py` | MarketSensingHook — data fetching orchestrator |
| `data/midge/watchlist.json` | Tickers + keywords MIDGE watches |

**Backbone sub-modules** (split during decomposition):
- `fractal_act.py` → re-export hub: `fractal_act_subsystem.py`, `fractal_act_organ.py`, `fractal_act_organism.py`
- `holon_protocol.py` → re-export hub: `holon_registry.py`, `holon_proxy.py`, `holon_mixin.py`, `awareness_pulse.py`
- `connection_registry.py` → 498 lines + `connection_registry_topology.py` (Euler), `connection_registry_verification.py` (verify/step mixin)
- `connection_registrations.py` → dispatcher: 5 sub-modules (`_bio`, `_metabolic`, `_agent`, `_patterns`, `_advanced`)
- `integration_meter.py` → `integration_meter_phi.py`, `integration_meter_blanket.py`, `integration_meter_models.py`
- `triad_enforcer.py` → `triad_enforcer_models.py`
- `triad_registry.py` → `triad_wiring.py`

**Sensing sub-modules** (split during decomposition):
- `sensing_hook.py` → thin orchestrator: `sensing_constants.py`, `sensing_fetchers.py`, `sensing_lifecycle.py`, `sensing_scheduler.py`, `sensing_collector.py`, `sensing_reactive.py`, `sensing_step_ops.py`
- `sensing_fetchers.py` → re-export hub: `fetchers_insider.py`, `fetchers_government.py`, `fetchers_market_data.py`, `fetchers_technical.py`, `fetchers_social.py`, `fetchers_crypto.py`

**Bootstrap sub-modules** (market_systems.py delegates to):
- `market_infrastructure.py` — OctopusColony, risk monitors, pattern discovery, scheduling
- `market_intelligence.py` — hypothesis engine, archaeology
- `market_gifts.py` — ten gifts (portfolio, order flow, etc.)
- `market_hooks.py` — EventBus channels, step hooks
- `market_registration.py` — holon + fractal registration
- `market_connections.py` — triadic connections
- `market_agents.py` — agent differentiation

**Paper trade pipeline:**
1. Convergence alert fires (3+ domains agree)
2. DrawdownMonitor checks — blocked if halted
3. SelfMonitor checks — blocked if behavioral anomaly
4. `_write_paper_trade()` — logs to `data/midge/paper_trades.jsonl`
5. `_translate_and_log_executable_signal()` — ATR-based SL/TP → `data/midge/executable_signals.jsonl`
6. `_submit_to_alpaca()` — bracket order to Alpaca (US equities only)

**500-line cap enforced** on all files. `connection_registry.py` split to 498 lines + 2 sub-modules. `connection_registrations_bio.py` split to 311 lines + backbone (118) + cognition (131).

**Pre-existing flaky test:** `test_congress_gov_client::test_request_fails_without_key` — passes in isolation, fails in full suite due to env var pollution from another test. Not a real bug.

---

## Guiding Light's Vision

> "MIDGE needs to be an entire functioning ecosystem. She's more of a planet than a singular biological organism. Everything inside her should be active, not passive."

> "The goal is for MIDGE to become my personal trader using inevitabilities, temporal knowledge, and aggregate factors on when to buy/sell/hold — stocks, crypto, futures, ANYTHING that MIDGE can make money off of."

> "$1,000 gate: Deploy capital only when MIDGE demonstrates pattern stacks with 80%+ historical accuracy — inevitability, not prediction."

---

## Research

| Expedition | Location | Key Finding |
|------------|----------|-------------|
| FTMO Viability | `research/expedition-ftmo-viability/` | "Right destination, wrong next step" — fix Thompson first, expand senses, then FTMO |
| Autonomous Trading | `research/expedition-autonomous-trading/` | Kalshi as first venue, Alpaca for equities |
| Competitive Edge | `research/expedition-competitive-edge/` | Cross-domain convergence is MIDGE's structural moat |
| Evolution Blueprint | `research/evolution-blueprint/` | 10-team architectural roadmap |
| Phase 0 Measurements | `research/phase0-measurements.md` | 3.34:1 payoff ratio, 19.9% convergence WR |

---

## Verification

```bash
python -m pytest tests/ -n 4 -q               # Full suite with xdist (3524 pass)
python -m pytest tests/test_decomposition_wiring.py -v  # 61 pass, 2 xfail
python main.py --agents 3 --steps 30           # Smoke test
```

## Stats

- **149 systems** (92 core + 57 market), **4,700+ tests**, **157 holons**, **428 connections**
- **123 market files** (34 API + 12 edge + 36 intelligence + 8 signal_adapters + 10 archaeology + 6 execution + 17 root)
- **37 sources**, **13 domains**, **41 adapters**, **12 concurrent fetches**, **25-step cadence**
- **510 tickers** (S&P 500 + forex/futures/crypto proxies)
- **33-layer bootstrap**, **14 mixins** on MycelialAgent
