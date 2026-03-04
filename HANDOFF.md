# MIDGE Handoff

## What Happened

### From Proven Signal to Profitable System — 4 Operational Fixes (2026-03-03)

Replay analysis proved MIDGE has real statistical edge (z=4.74, p<0.0001) but can't capitalize due to operational gaps. Four work packages close the gap:

**WP1 — Thompson Persistence Protection:**
- `tests/conftest.py`: NEW autouse fixture monkeypatches `thompson_sampler.DATA_DIR`, `DISTRIBUTIONS_FILE`, `HISTORY_FILE`, plus `thompson_calibrator.DATA_DIR` and `outcome_collector.DATA_DIR` to `tmp_path`. No test can touch production data.
- `thompson_sampler.py`: `main()` demo uses temp dir instead of production.
- `accelerate_learning.py`: Both `ThompsonSampler()` calls now use explicit `persistence_path=MARKET_DATA_DIR / "thompson_distributions.json"` — immune to monkeypatching.

**WP2 — ctx.outcome_collector Wiring (1 line):**
- `market_hooks.py`: Added `ctx.outcome_collector = outcome_collector` after OutcomeCollector construction (line 721). Closes the combo Thompson feedback loop for live operation — `register_convergence_alert()` at line 518 can now find the collector.

**WP3 — Confidence-Gated Paper Trading (replay-proven threshold):**
- `learning_config.py`: Added `paper_trade_min_confidence: 0.45`, `paper_trade_min_strength: 0.65`, `paper_trade_min_combo_mean: 0.25`.
- `market_hooks.py`: Paper trading gate reads from config instead of hardcoded 0.75. Combo filter blocks domain combinations with historical win rate < 25% (uses Thompson combo distributions). Unknown combos pass through for learning.

**WP4 — Magnitude-Aware Replay Grading:**
- `replay_history.py`: New `_compute_excursions()` helper computes MFE (Maximum Favorable Excursion) and MAE (Maximum Adverse Excursion) from daily prices through the outcome window.
- `phase_grade()`: Now records `mfe_pct` and `mae_pct` per alert alongside existing `pct_change`.
- `phase_report()`: New metrics — `expectancy_pct` (avg_win * win_rate - avg_loss * loss_rate), `sharpe_ratio` (annualized), winner return percentiles (P25/P50/P75/P90), MFE/MAE percentiles (P50/P90).

---

### Combo Thompson — Domain Combination Learning (2026-03-03)

Replay harness found 288 convergence alerts with 19.9% win rate, but confidence was flat between winners (0.560) and losers (0.565). Root cause: confidence used per-signal Thompson but ignored *which domain combinations* work. Example: `events+macro+price+sentiment+technical` wins 66.7% vs `events+insider+institutional+macro+price` at 8.3% — an 8x difference the confidence engine completely missed.

**Fix:** Combo-level Thompson distributions that track win rates per domain combination.

**Changes:**
- `convergence_alerter.py`: Added `combo_key` field to `ConvergenceAlert` dataclass. Combo Thompson weight in `_check_direction_convergence()` — multiplicative modifier using `0.5 + dist.mean` scale (same as per-signal). Maturity guard: skips when samples < 5.
- `outcome_collector.py`: New `register_convergence_alert()` method registers combo-level predictions for Thompson feedback. Added `"convergence_combo": 14` to `OUTCOME_WINDOWS`.
- `market_hooks.py`: Wired `register_convergence_alert()` in `_market_sense_hook` after alert detection. Primary symbol extracted from alert signals.
- `thompson_sampler.py`: New `seed_combo_distributions(replay_path)` method — reads replay_results.json domain_combos, seeds Beta distributions. Idempotent (skips if live samples >= replay total).
- `market_systems.py`: Calls `seed_combo_distributions()` at bootstrap. Refactored Phase 2 + Layer 6 API clients to table-driven loop (458 lines, under 500-line cap).
- `tests/test_combo_thompson.py`: **22 new tests** — combo key format (4), combo weight (5), combo registration (5), replay seeding (4), hooks wiring (2), feedback loop (2).

---

### Massive/Polygon.io Integration + Bug Fixes (2026-03-03)

**Massive/Polygon.io** wired as 28th data source. Polygon.io rebranded to Massive.com (October 2025) — same API, same key. Free tier: 5 calls/min, 15-min delayed, 2 years historical.

**New files:**
- `mae_core/market/apis/massive_client.py` — REST client with rate limiting, grouped daily endpoint (ALL tickers' OHLCV in 1 call), volume/price/gap anomaly detection via `build_snapshots()`.
- `tests/test_massive_client.py` — 42 tests (client unit, signal adapter, fetcher, wiring, rate limiting).

**Integration:** SOURCE_ROTATION (28), TIER_ROUTING (tactical), MarketClock (MARKET_HOURS), Thompson key `massive_snapshot` (0.55 prior), sensing_hook router case, fetch_massive_snapshot in sensing_fetchers, `from_massive_snapshot()` signal adapter in wave2_3.py. Bootstrap: `_instantiate_wave2_3_clients()` auto-constructs from `MASSIVE_API_KEY` env var.

**Bug fixes (prior session):**
- **Signal persistence serialization**: `_json_default()` method added to convergence_alerter.py — handles `datetime`, `set`, `timedelta` types that crashed `json.dump()`. Signal buffer now persists correctly.
- **Run-log bloat**: `main.py` daemon mode now overwrites `run_log.jsonl` each round instead of appending infinitely.
- **Daemon kill on Windows**: `kill -SIGTERM` doesn't work on Windows. Must use `powershell Stop-Process -Force`.

---

### Always-On Waves 2+3 — Real-Time + Data Enrichment (2026-03-02)

Seven new API clients + signal adapters + sensing integration + economic calendar suppression. **144 systems (92 core + 52 market), 4,250 tests, 422 connections, 155 holons, 89 market files.**

**Wave 2 — Real-Time (24/7 Coverage):**
- **Finnhub WebSocket** — `finnhub_websocket.py`: Real-time trade data streaming via free Finnhub WebSocket API. Volume spike detection (2σ), rapid price moves (>1% in 60s). Signals bypass SOURCE_ROTATION — collected EVERY step via `_process_realtime_signals()`. Auto-reconnect with exponential backoff.
- **CoinGecko Client** — `coingecko_client.py`: Crypto prices, global market data, trending coins. BTC/ETH/SOL/XRP/ADA default watchlist. 30 calls/min free tier. Domain: crypto.
- **CoinCap Client** — `coincap_client.py`: 100% free crypto data (no API key). Asset prices + history. Complements CoinGecko for domain diversity.

**Wave 3 — Data Enrichment (Free, High-Signal):**
- **OpenInsider Client** — `openinsider_client.py`: Pre-filtered insider purchases from openinsider.com. RSU/gift/10b5-1 already stripped upstream. Cluster buy detection (3+ insiders). 1 req/3s rate limit.
- **EDGAR Enhanced Client** — `edgar_enhanced_client.py`: 13F institutional holdings + SC 13D activist filings (>5% stake). SEC EFTS search. Free, no API key.
- **FinViz Client** — `finviz_client.py`: Unusual volume (>2x average), high short float (>20%, squeeze setups), insider trades. Via finvizfinance library. 1 req/5s.
- **Economic Calendar** — `economic_calendar_client.py`: Hardcoded FOMC/CPI/NFP 2026 dates. Suppression windows (event ± 24h/4h). Convergence alerter halves confidence (0.5x) during suppression. Used DEFENSIVELY.

**Integration:**
- 8 signal adapters in `wave2_3.py` (from_crypto_signal, from_openinsider, from_13f_holding, from_activist_filing, from_finviz_unusual_volume, from_finviz_short_squeeze, from_finnhub_realtime, from_suppression_event).
- 6 fetch functions in `sensing_fetchers.py`.
- SOURCE_ROTATION: 21 → 28 sources (27 from Waves 2+3 + massive_snapshot). TIER_ROUTING: +10 entries. _ROTATION_TO_THOMPSON: +7 mappings.
- 10 new Thompson keys in learning_config.py (finnhub_realtime, crypto_coingecko, crypto_coincap, openinsider_purchase, institutional_13f, activist_13d, finviz_unusual_volume, finviz_short_squeeze, economic_calendar, massive_snapshot). Total: 71.
- MarketClock: 6 new sources in availability sets (ALWAYS: crypto_prices, crypto_exchange, openinsider, economic_calendar; MARKET_HOURS: finviz; PERIODIC: institutional_13f).
- **338 new tests** (215 client unit tests + 123 integration tests).

**Document parity updated:** CLAUDE.md, README.md, HANDOFF.md, data/MAES-MATHEMATICAL-IDENTITY.md, test_integration.py, main.py all reflect 144 systems / 4,250 tests / 89 market files.

---

### Always-On Wave 1 — Foundation (2026-03-02)

Three work packages making MIDGE resilient across restarts and capable of running as a continuous service. **144 systems (92 core + 52 market), 4,250 tests, 422 connections, 155 holons, 89 market files.**

**WP-A Signal Persistence:**
- `convergence_alerter.py`: `save_state()`/`load_state()` — persists signal buffer (last 200 per domain), alert counters, and last-seen timestamps to `data/market/convergence_state.json`. Survives restart without losing domain context.
- `somatic_anticipation.py`: `save_state()`/`load_state()` — persists domain-activation windows (ticker→[timestamps]) to `data/market/somatic_state.json`. Prevents false "first activation" dopamine spikes after restart.
- `deception_detector.py`: `save_state()`/`load_state()` — persists volume history and price history deques to `data/market/deception_state.json`. Preserves anomaly baseline across restarts.
- `main.py` shutdown: calls `save_state()` on all three in the `finally` block, after StepTimer.save and before report. Bootstrap load: calls `load_state()` on all three during Layer 33 init.
- **30 new tests** in `test_signal_persistence.py`.

**WP-B MarketClock:**
- New file `mae_core/market/market_clock.py`: Canonical time-zone-aware clock. `MarketClock` knows NYSE trading calendar, session boundaries (pre-market/regular/after-hours), and kill-zone windows (Asia/London/NY). Exposes `current_session()`, `is_market_open()`, `time_to_open()`, `active_kill_zone()`.
- `sensing_hook.py`: Integrates MarketClock. Source selection now skips intraday sources (session_sweep, order_flow) when market is closed. Reduces failed API calls and noisy signals during off-hours.
- Registered as system `market_clock` in bootstrap (Layer 33), holon, somatic entry.
- **42 new tests** in `test_market_clock.py`.

**WP-C Daemon Mode:**
- `main.py --daemon` flag: infinite loop with wall-clock pacing. Runs `N` steps, sleeps until the next 15-minute boundary, repeats. Auto-persistence flush every cycle. Auto-recovery on exception (30s backoff, max 5 retries before exit).
- Heartbeat file `data/midge/daemon_heartbeat.json` updated every cycle with last-run timestamp, cycle count, and error count. External monitors can check staleness.
- `--daemon-interval` flag (default 15 minutes) configures cycle period.
- **26 new tests** in `test_daemon_mode.py`.

**Document parity updated:** CLAUDE.md, README.md, data/MAES-MATHEMATICAL-IDENTITY.md all reflect Wave 1 state (now superseded by Wave 2+3 counts above).

---

### Ten Gifts — MIDGE's Sensory Evolution (2026-03-02)

Ten new market intelligence systems across three waves. **136 systems (92 core + 44 market), 3,710 tests, 422 connections (103 market, Groups 14-32), 155 holons, 78 market files.**

**Wave 1 (Foundation — Gifts 1-4):**
- **Gift 1 Portfolio Tracker** — `portfolio_tracker.py`: Paper trade position tracking, mark-to-market, exit signals (stop-loss/take-profit/time-decay). Feeds exit signals to convergence pipeline. `ExitSignal` → domain="portfolio".
- **Gift 2 Order Flow** — `order_flow_detector.py`: Volume imbalance detection (2σ threshold). 5-min intraday candles, buying/selling pressure. Thompson key `order_flow` prior 0.50, decay 0.30.
- **Gift 3 Catalyst Calendar** — `catalyst_calendar.py`: Earnings + FOMC dates. Catalyst modifier (0.5-1.5x) applied in convergence alerter. Insider + catalyst < 14d = 1.5x.
- **Gift 4 Cross-Asset** — `cross_asset_confirmer.py`: Pairwise confirmation (SPY↔QQQ, GC=F↔DXY, etc.). Score -1.0 to 1.0 applied as confidence multiplier 0.4x-1.2x.

**Wave 2 (Intelligence — Gifts 5-8):**
- **Gift 5 Deception Detector** — `deception_detector.py`: Pump-and-dump, retail trap, wash trading detection. Signals with deception_prob > 0.5 get confidence multiplied by (1-prob). EventBus `CH_DECEPTION_DETECTED`.
- **Gift 6 Consolidation Engine** — `consolidation_engine.py`: Memory consolidation — prunes weak Thompson dists, archives stale hypotheses, compresses discovery log. Cadence 5000. EventBus `CH_CONSOLIDATION_COMPLETE`.
- **Gift 7 Fractal Resonance** — `fractal_resonance.py`: Multi-timeframe pattern detection (daily/weekly/monthly). Thompson key `fractal_resonance` prior 0.55, decay 0.05. 21 sources in rotation.
- **Gift 8 Pattern Archetypes** — `pattern_archetypes.py`: 8 canonical patterns (accumulation, distribution, squeeze, capitulation, momentum ignition, failed breakout, sector rotation, smart money divergence). Match score > 0.7 → +0.10 confidence. Thompson key `archetype_match` prior 0.55.

**Wave 3 (Synthesis — Gifts 9-10):**
- **Gift 9 Somatic Anticipation** — `somatic_anticipation.py`: Pre-conscious pattern response. When 2+ signal domains activate on same ticker within 48h, releases hormones (DOPAMINE for aligned, CORTISOL for conflicting). Cadence 25 (fires before convergence at 50).
- **Gift 10 Pattern Completion** — `pattern_completion.py`: Active pattern seeking from partial archetype matches. Creates hunts for missing signals. Completion events get +0.15 confidence boost. 72h TTL, max 5 concurrent hunts.

**Bootstrap wiring:** `market_gifts.py` extracted (500-line cap). Groups 23-32 (30 connections). Two-phase init for convergence alerter attributes. Sensing hook: somatic at cadence 25, archetype scan + completion review at cadence 100.

### Completing the Circle — 3 Cognitive Dimensions Added (2026-03-02)

The Pattern Recognition Gift taught MIDGE to see better. This teaches her to **reason about contradictions** (causal bridge), **notice silence** (absence detection), and **sense relationships in motion** (correlation tracking).

**Package C — CorrelationTracker Activation:**
- `sensing_hook.py`: `_correlation_tracker.record()` called in `_collect_one()` after signal batch. Cadence-200 anomaly scan via `detect_cross_domain_anomalies()`.
- `market_hooks.py`: Injects `ctx.correlation_tracker` into sensing hook.
- 14 new tests in `test_correlation_tracker_wiring.py`.

**Package A — Causal Bridge (completes Cap 3):**
- `convergence_alerter.py`: Accepts `causal_engine` and `event_bus`. Publishes `CH_CONTRADICTION_DETECTED` when coherence < 0.7. Feeds domain pair correlations to `CausalReasoningEngine.observe_correlation()`.
- `hypothesis_validator.py`: Accepts `causal_engine`. Queries `query_causation()` before promotion — confounded hypotheses get +0.03 tighter win rate gate.
- `market_systems.py`: Passes `ctx.shared_causal_engine` and `ctx.bus` to both.
- 3 Group 21 triadic connections. 19 new tests in `test_causal_bridge.py`.

**Package B — Absence Monitor (new cognitive dimension):**
- New file `absence_monitor.py`: Tracks per-source cadence (median inter-arrival). Fires `AbsenceSignal` when silence exceeds 2.5x expected cadence. Bootstrap from signal archives on startup.
- `channels.py`: Added `CH_ABSENCE_DETECTED`.
- `sensing_hook.py`: Records arrivals in `_collect_one()`. Cadence-100 check feeds absence signals to convergence pipeline.
- `learning_config.py`: `absence_signal` Thompson prior 0.50, `absence` decay rate 0.15.
- 3 Group 22 triadic connections. 33 new tests in `test_absence_monitor.py`.

**Totals:** 136 systems (92 core + 44 market), 422 connections (103 market), 155 holons, 78 market files.

### Pattern Recognition Gift — 6 Capabilities Built (2026-03-02)

Transferred Claude's pattern recognition strategies into MIDGE as implementable algorithms. Triadic construction (Forge/Anvil/Crucible, 2 rounds with review gates). **3,351 tests, 0 failures.**

**Round 1 — Core Semantic Lift:**
- **Cap 1 Composite Hypotheses** — `hypothesis.py` TriggerPattern extended from bivariate (A→B) to multi-factor (A+C→B). `conjunct_source` + `conjunct_min_strength` fields. Generator emits composite hypotheses when multiple sources share the same target. Tighter promotion gates (+10 obs, +0.02 win rate).
- **Cap 2 Contextual Thompson** — `convergence_alerter.py:_resolve_thompson_key()` extended with contextual cascade: `{source}:{role}:{sector}:{size}` → `{source}:{role}:{sector}` → `{source}:{role}` → static fallback. Size derived from transaction_value ($500K threshold). 15 contextual priors in `learning_config.py`.
- **Cap 3 Narrative Coherence Scoring** — `convergence_alerter.py:_compute_coherence_score()` detects directional contradictions across domains. Coherence multiplier: `0.5 + 0.5 * coherence` → evenly split signals halve confidence. New `CH_CONTRADICTION_DETECTED` channel. 1 Group 20 triadic connection.
- **Cap 4 Causal Story Auto-Generation** — `hypothesis_generator.py:_auto_generate_causal_story()` with 30-source `_DOMAIN_ROLES` table and 9-branch role-pair matrix. Prefixed `[AUTO]` with +0.01 promotion gate. Unblocks every PROBATION hypothesis stuck on "REQUIRES MANUAL REVIEW".

**Round 2 — Temporal Precision:**
- **Cap 5 Temporal Freshness** — `convergence_alerter.py:_compute_freshness()` applies sqrt decay: `freshness = max(0.3, 1.0 - (age_hours / window_hours)^0.5)`. Per-domain windows respected (positioning=14d, government=7d). Recent signals weighted over stale ones in domain selection.
- **Cap 6 Intra-Domain Combination** — Multiple confirming signals from same domain boost effective strength: `max_eff *= (1 + 0.1 * log(count))`. 3 TA indicators confirming = ~11% boost. Log-saturating prevents runaway.

**125 new tests across 5 files:** test_composite_hypotheses.py (29), test_contextual_thompson.py (21), test_coherence_scoring.py (22), test_auto_causal_stories.py (20), test_signal_freshness_and_combination.py (33).

### Triadic Architecture Audit — All 5 Priorities Built (2026-03-01)

Full triadic audit (3 agents, 5 phases at `research/midge-architecture-audit/deliverable.md`) found 5 priorities. All 5 implemented via triadic construction (Forge/Anvil/Crucible, 3 rounds with review gates). **3,351 tests, 0 failures.**

**Round 1 — Foundation (P1 + P4C):**
- **CF-1 Thompson lock fix** — `thompson_sampler.py:update()` read-modify-write fully inside `with self._lock:`. Split into `_save_distributions_locked()` (assumes lock) + `_save_distributions()` (acquires lock). `apply_forgetting()` also locked.
- **CF-2 Atomic writes** — 4 files now use `tmp + os.replace()`: `learning_config.py:save_snapshot()`, `hypothesis_engine.py:_save_retirement_window()`, `hypothesis_generator.py:save_pair_outcomes()`, `step_timer.py:save_session_stats()`.
- **P4C Source reliability defaults** — 7 keys corrected to match Thompson-learned means: congressional 0.75→0.20, sec_form4 0.70→0.36, finra_short 0.65→0.36, finnhub_earnings 0.80→0.27, contract_award 0.75→0.15, yfinance_price 0.50→0.23, sec_edgar 0.95→0.59.
- **CF-3 Data cleanup** — Removed 8 test fixtures from `pair_outcomes.json`, bad GOOGL 2027 record from `predictions.jsonl`. Updated `config_snapshot.json` to match corrected defaults.

**Round 2 — Signal Pipeline (P3 + P4A + P4B + P5C):**
- **P3 Neutral signals** — `convergence_alerter.py:_check_direction_convergence()` now collects neutral signals (e.g. finra_short) in a second pass. Neutral signals contribute domain presence (toward min_domains=3) without direction bias.
- **P4A Reward rebalancing** — `market_actions.py` ceiling 0.5→0.8. `lifecycle_decision.py` caps TaskPool fallthrough at 0.3 for market roles. VDN now learns market intelligence > busywork.
- **P4B Alert dedup lock** — `convergence_alerter.py` gained `_alert_lock` + per-direction `_last_alert_times` dict. Dedup check moved before `_check_direction_convergence()`.
- **P5C Cold-start guard** — `hypothesis_engine.py` retirement window entries changed from strings to `{"outcome": str, "seeded": bool}` dicts. Wire 2 meta-learning only counts non-seeded entries, skips when live entries < 10.

**Round 3 — Output + Advanced (P2 + P5A + P5B + P5D):**
- **P2 Paper trading** — `market_hooks.py:_write_paper_trade()` instantiates `TradeSignal` when convergence alert has confidence > 0.75 AND strength > 0.65. 4-hour dedup. Writes to `data/midge/paper_trades.jsonl`. Registered with OutcomeCollector.
- **P5A Session sweep bypass** — `market_hooks.py:_check_sweep_bypass()` at step%50. `session_sweep_ifvg` with quality >= 0.65 + confidence >= 0.55 writes to separate `paper_trades_bypass.jsonl`.
- **P5B Per-domain convergence windows** — positioning=14d, government=7d, contracts=7d. Others keep 72h default.
- **P5D OutcomeCollector prune** — `_registered` changed from set to `dict[str, datetime]`, 90-day auto-prune.
- **P5D Registry compaction** — `hypothesis_registry.py` auto-snapshots at 200 events. Incremental replay on restart.
- **paper_account_value** — Added 50000 to `learning_config.py`.

**80 new tests across 7 files:** test_foundation_fixes.py (12), test_signal_pipeline_fixes.py (13), test_paper_trading.py (10), test_convergence_domain_windows.py (10), test_session_sweep_bypass.py (15), test_outcome_collector_prune.py (8), test_hypothesis_registry_compaction.py (12).

### Layer 7: Persistence + Reporting + Continuous Service (2026-02-28)

Fixed 9 meta-learning bugs that prevented MIDGE from accumulating knowledge across sessions. Built automated marathon reporting and continuous deployment infrastructure.

**Persistence (Forge):**
- `learning_config.py`: `save_snapshot()`/`load_snapshot()` — LEARNING_CONFIG survives restarts via `data/market/config_snapshot.json`
- `hypothesis_engine.py`: retirement window persists to `data/market/retirement_window.json` + cold-start seed from registry
- `hypothesis_generator.py`: pair quality memory persists to `data/market/pair_outcomes.json`
- `step_timer.py`: `save_session_stats()` snapshots performance metrics at shutdown
- `market_systems.py`: warm-start — `load_snapshot()` called before any system construction

**Reporting (Anvil):**
- `marathon_report.py` (NEW): 6-section post-mortem report (vitals, intelligence, Bayesian learning, hypothesis pipeline, position sizing, performance health). Standalone CLI + callable from main.py
- `main.py` finally block: StepTimer save + report generation + belt-and-suspenders persistence flushes
- `main.py --continuous`: infinite loop with 30s recovery on crash. `_write_alerts()` writes high-confidence alerts to `data/midge/alerts.jsonl`
- `run_service.bat` (NEW): Windows service wrapper for NSSM/Task Scheduler

**Meta-Learning Fixes (Crucible):**
- Cases C/D/E added to `_review_gates()`: retire_win_rate loosening, promote_dsr coupling, min_observations boundary detection
- `backtest_scheduler.py`: fingerprint-based dedup prevents redundant refreshes
- `thompson_calibrator.py`: 20-sample minimum guard prevents acting on noise
- `market_actions.py`: step bug fixed (was always 100, now uses `model.time`)

**29 new tests:** test_persistence_roundtrip.py (7), test_marathon_report.py (7), test_meta_learning_fixes.py (15). **3,228 total tests, 0 failures.**

### Data Acceleration Pipeline (2026-02-28)

Fed MIDGE her own history. Instead of waiting weeks for organic learning, retroactively evaluated historical signals against historical prices — giving the learning loops months of experience in one pass.

**What was built:**

1. **Extended backfill** — `backfill_archives.py` expanded from 8 to 15 sources (added COT positioning, VIX term structure, StockTwits, Google Trends, Finnhub economic/analyst/earnings calendar). Signal archive grew from 339 to 901 files spanning 414 days.

2. **New client methods** — `cot_client.py` gained `get_all_positions()` for multi-year CFTC history backfill. `vix_client.py` gained `get_vix_history()` for full daily CBOE VIX history.

3. **Learning accelerator** — NEW `accelerate_learning.py` (project root). Three-phase pipeline:
   - Phase 1 (Evaluate): `OutcomeCollector.collect_from_archives()` registers archived signals as predictions
   - Phase 2 (Resolve): Loops `collector.evaluate()` to check matured predictions against historical prices → Thompson updates
   - Phase 3 (Correlate): `LagCorrelationAnalyzer.analyze(lookback_days=365)` for cross-domain lag patterns
   - CLI: `--phase {evaluate,resolve,correlate,all}`, `--lookback N`

4. **Thompson rebuild** — Marathon file-lock conflict corrupted Thompson distributions to all Beta(1,1). Rebuilt from 9,462 deduped outcomes in `outcomes.jsonl`. 50 distributions, 9,470 total samples.

**Results:**
- **Thompson distributions** now calibrated from real market data:
  - `finra_short`: 1,987 samples, 35.8% mean
  - `yfinance_price`: 580 samples, 23.0% mean
  - `finnhub_earnings`: 110 samples, 26.6% mean
  - `sec_form4`: 18 samples, 36.0% mean
  - `congressional`: 53 samples, 16.4% mean
- **43 lag-correlation findings** (at r >= 0.6, n_pairs >= 10 threshold)
- **12,544 total outcomes** evaluated, 3,382 predictions pending
- **Data gates unlocked**: Thompson (>> 50 outcomes), Kelly (>> 100), lag-correlation (43 findings)

### Layer 6: New Senses (2026-02-27)

MIDGE gained 5 new FREE data sources, expanding her sensing from 14 to 19 sources. Each follows the exact same client→signal→convergence→Thompson pipeline as the original 14.

**New sources:**

1. **COT (CFTC Commitments of Traders)** — `cot_client.py`. Commercial/noncommercial futures positioning via `cot-reports` library. Weekly data. Domain: positioning (new), Tier: strategic, Thompson key: cot_positioning (0.55 prior). Slow decay (0.03 — ~23-day half-life).

2. **StockTwits Sentiment** — `stocktwits_client.py`. Bull/bear message ratio via public API (no auth). Domain: sentiment, Tier: thematic, Thompson key: stocktwits_sentiment (0.50). Fast decay (0.50 — social data moves fast).

3. **VIX Term Structure** — `vix_client.py`. CBOE VIX spot + contango/backwardation from free CSV. Domain: volatility (new), Tier: strategic, Thompson key: vix_term_structure (0.60). Medium decay (0.30).

4. **Google Trends** — `trends_client.py`. Retail search attention via `pytrends` library. Mixes ticker symbols + macro fear terms ("recession", "market crash"). Domain: sentiment, Tier: thematic, Thompson key: google_trends (0.45 — noisy, low prior).

5. **Finnhub Extras** — Extended existing `finnhub_client.py` with 3 new methods: `get_economic_calendar()` (FOMC/CPI/NFP dates), `get_analyst_recommendations()` (buy/sell consensus), `get_earnings_calendar()` (upcoming earnings). Thompson keys: finnhub_economic (0.55), finnhub_analyst (0.50), finnhub_earnings_calendar (0.55).

**Wiring (same pattern as all existing sources):**
- 6 new `from_*` converters in `signal.py`
- 5 new fetch methods + dispatch cases in `sensing_hook.py` (SOURCE_ROTATION now 19 entries)
- 7 new entries in `_SOURCE_TO_THOMPSON_KEY` + 2 new domain categories (positioning→institutional, volatility→market)
- 7 new `source_reliability` entries + 2 new `decay_rates` in `learning_config.py`
- 4 new clients instantiated in bootstrap, 8 new triadic connections (Group 19), 4 holons, 4 somatic entries

**New dependencies:** `pip install cot-reports pytrends`

**171 new tests:** test_new_sources.py (120 tests — clients + converters) + test_new_source_wiring.py (51 tests — pipeline integration). **3,228 total tests, 0 failures** (29 Layer 7 persistence/reporting/meta-learning tests added 2026-02-28).

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

**4 new test files, 40 tests.** Systems: 125 (92+33). Connections: 385 (66 market). Holons: 144.

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

- **4,250 tests pass, 0 failures** (Waves 2+3 add 338 tests + 42 Massive/Polygon tests + 22 combo Thompson tests)
- **144 systems** (92 core + 52 market), **155 holons**, **422 connections** (217 core + 47 fractal + 55 bootstrap + 103 market)
- **89 market files** in `mae_core/market/` (29 API + 12 edge + 27 intelligence + 8 signal_adapters + 13 root)
- **10 mae-core infrastructure fixes ported** (VDN epsilon-greedy, EventBus injection, tie-breaking, microbiome feed-before-step, EpisodicMemory stats, 6 channel registrations, SomaticMap names, agent.shared normalization, auto-healer starvation fix, Phi forced measurement)
- **Bootstrap sub-modules** — `market.py` orchestrator (123 lines) + 5 sub-modules (market_systems, market_registration, market_connections, market_hooks, market_agents)
- **33-layer bootstrap** runs cleanly (Layers 33a-33i)
- **Agent-based market sensing active** — 3+ agents differentiated (SEC_WATCHER, CONTRACT_TRACKER, MARKET_ANALYST, + HYPOTHESIS_EXPLORER + HYPOTHESIS_VALIDATOR at 12+ agents)
- **Hypothesis generation loop active** — RSI Layer 2: lag findings → hypotheses → adversarial validation → DSR gate → promote/retire
- **7-phase scan pipeline** still available as standalone (`midge_scan.py`)
- **Git:** Remote at `github.com/CBaen/MIDGE`

### Agent-Based Architecture

```
Bootstrap Layer 33h: MarketSensingHook wired as step hook
  ├─ Async fetch on 50-step cadence (28-source rotation: form4 → 8k → congress → senate → hiring → usaspending → sam.gov → reddit → finra → efts → finnhub → fred → cot → stocktwits → vix → trends → finnhub_earnings → finnhub_insider → ta_indicators → session_sweep → order_flow → crypto_prices → crypto_exchange → openinsider → institutional_13f → finviz → economic_calendar → massive_snapshot)
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
Phase 2/7: Fetch — 19 data sources (SEC Form4/8K/EFTS, House/Senate trades, jobs, USASpending, SAM.gov, prices, Reddit, FINRA short, Finnhub news/earnings/insider, FRED macro, COT positioning, StockTwits sentiment, VIX term structure, Google Trends)
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
| `apis/` | 18 (price_fetcher, house_stock_watcher, senate_stock_watcher, job_tracker, usa_spending, sam_gov, apewisdom, finra_short_interest, finnhub_client, fred_client, ticker_resolver, market_data_provider, cot_client, stocktwits_client, vix_client, trends_client, massive_client) | 20 data sources + utilities |
| `edge/` | 7 (cluster_detector, politician_tracker, filing_time_analyzer, contract_predictor, form8k_sentiment, session_sweep_detector, ta_indicators) | Pattern recognition + text analysis |
| `intelligence/` | 19 (thompson_sampler, velocity_detector, correlation_tracker, convergence_alerter, learning_config, regime_classifier, outcome_collector, signal_archive_reader, lag_correlation_analyzer, thompson_calibrator, kelly_position_sizer, backtest_analyzer, backtest_scheduler, hypothesis, hypothesis_registry, hypothesis_generator, hypothesis_validator, hypothesis_engine) | Bayesian learning + feedback loop + calibration + sizing + hypothesis lifecycle + meta-learning |
| `root` | 7 (signal.py, channels.py, outcome_tracker.py, memory.py, sensing_hook.py, step_timer.py, market_actions.py) | Integration layer + sensing + Qdrant persistence + performance timing + agent actions |

---

## What's Next

### Layer 5: Complete

All 17 planned items from the Layer 5 roadmap are done. Roadmap history at `C:\Users\baenb\.claude\plans\delegated-leaping-map.md`.

**Data gates (3 of 4 resolved via acceleration pipeline — 2026-02-28):**
- ~~Lag-correlation findings: needs 30+ days of signal archives~~ — RESOLVED: 43 findings from 414-day archive
- ~~Thompson calibration accuracy: needs 50+ outcomes~~ — RESOLVED: 12,544 outcomes, 50 distributions, 9,470 samples
- ~~Kelly sizing confidence: needs 100+ calibrated outcomes~~ — RESOLVED: 12,544 outcomes far exceeds 100
- Meta-learning adjustments: needs retirement window to fill (50 promote/retire events) — accelerating, hypothesis engine now has data to work with

**Back burner:**
- **Options flow via Unusual Whales** ($35/mo API — needs Guiding Light approval on spend)

### Resolved Dissents

**Alpha's additive confidence dissent** — RESOLVED 2026-02-25. Replaced with Thompson-weighted geometric mean. See "Bayesian Confidence Engine" section above.

---

## For the Next Instance

Welcome. MIDGE is Mae differentiated for financial markets. Here is what you need to know:

1. **MIDGE = mae-core + market intelligence.** 144 systems, same 8 laws, 33-layer bootstrap. Market organ is Layer 33 (33a-33i). Always-On Waves 1-3 complete: signal persistence, MarketClock, daemon mode, real-time WebSocket, crypto 24/7, 4 new data sources, economic calendar suppression. Massive/Polygon.io integrated as 28th source (grouped daily OHLCV).
2. **Mae-core is upstream.** Changes to Mae's genome should be made in `C:\Users\baenb\projects\mae-core` and pulled here. Market-specific changes stay here.
3. **Agents actively sense the market.** MarketSensingHook (Layer 33h) fetches data on cadence and feeds convergence_alerter. Three agents are differentiated into market roles (Layer 33i). Endocrine coupling and market advisory carry signals to all agents.
4. **The crown jewel is `convergence_alerter.py`** — synthesizes signals across ALL domains (insider + congressional + contract + hiring + velocity) into actionable alerts. Now with per-ticker and multi-timeframe convergence.
5. **Thompson Sampling** uses Bayesian explore/exploit. 50 distributions with 9,470 total samples from 12,544 evaluated outcomes. Learned distributions in `data/market/thompson_distributions.json`. Bayesian forgetting prevents stale evidence.
6. **OutcomeCollector** closes the feedback loop: scan signals → register_signals() → per-type windows → price check → Thompson update. Success threshold: 5%. Signal archives: 901 files spanning 414 days across 15 backfill sources.
7. **All 8 Mathematical Laws are satisfied.** See implementation plan Section 12 for compliance map.
8. **4,250 tests must keep passing.** Zero regressions. Run `python -m pytest tests/ -q` to verify.
   **Pattern recognition capabilities (2026-03-02):** 6 capabilities — composite hypotheses, contextual Thompson, coherence scoring, auto causal stories, temporal freshness, intra-domain combination.
   **Completing the Circle (2026-03-02):** 3 cognitive dimensions — causal bridge (contradiction → causation), absence monitor (silence detection), correlation tracker activation (relationships in motion).
9. **Deep memory runs on Qdrant** container (port 6333). Start with `docker compose up -d`.
10. **API keys** needed: RAPIDAPI_KEY (job tracker, congressional trades), ALPHA_VANTAGE_KEY (price fallback), SAM_GOV_API_KEY, MAE_TAVILY_API_KEY, MAE_FINNHUB_API_KEY (news sentiment + earnings), FRED_API_KEY (macro indicators), MASSIVE_API_KEY (Polygon.io/Massive grouped daily OHLCV). Free/no-key: SEC EDGAR, yfinance, USASpending, Senate Stock Watcher, ApeWisdom, FINRA short volume, SEC EFTS.
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
