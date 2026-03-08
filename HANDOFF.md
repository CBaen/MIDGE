# MIDGE Handoff

## What Happened

### Raw Store Expansion: 4→12 Domains (2026-03-08)

**raw_store.py expanded** from 296→849 lines with 8 new domain methods: `store_price_snapshot` (yfinance 80+ fields as JSON), `store_fred_observations` (with vintage metadata), `store_stocktwits_messages` (full message text/sentiment/user), `store_finnhub_sentiment/earnings/economic` (ALL countries, quarter/year preserved), `store_congressional_trades` (house+senate), `store_congress_bills` (sponsors, committees), `store_finra_short_volume`.

**8 clients wired** — price_fetcher, FRED, StockTwits, Finnhub, FINRA, House, Senate, Congress.gov all now persist raw data before processing. Total: 12 of 24 clients wired. Bootstrap passes `raw_store` to all 12.

**35 tests** — up from 14. All pass. Zero regressions on full suite.

**12 clients still unwired**: SEC EDGAR, Massive, CoinGecko, CoinCap, OpenInsider, FinViz, EdgarEnhanced, EconCalendar, FinnhubWS, ApeWisdom, JobTracker, SAM.gov.

### Execution Bridges + Sensing Overdrive + Data Audit (2026-03-07b)

**Alpaca bridge** — `alpaca_client.py` built. Paper trading with bracket orders (TP+SL), position tracking, account info. `alpaca-py 0.43.2`, `kalshi-python 2.1.4`, `httpx+selectolax+trafilatura` all installed. Awaiting `ALPACA_API_KEY` + `ALPACA_SECRET_KEY` in `.env`.

**Sensing workers 3→12** — `sensing_hook.py` ThreadPoolExecutor `max_workers` raised. 4x throughput for 30 sources.

**OpenInsider cluster buys WIRED** — `get_cluster_buys()` was fully implemented but never called. Now fires high-confidence signals when 3+ insiders buy same stock within 30 days.

**FINRA speculative short ratio** — `ShortExemptVolume` was parsed then discarded. Now `speculative_short_ratio` separates market-maker structural shorts from speculative shorts.

**Full API data waste audit** — 20 of 24 API clients have no raw_store. Multiple built methods never called (FinViz insider trades, 13F filer list, OpenInsider clusters). See gap list below.

**Daemon data preserved** — 7 learned-state files committed (correlations, outcomes, predictions, dedup registry, Thompson, hypotheses, discovery log). 6 runtime buffers discarded.

### Raw Data Pipeline + Expedition Synthesis + Vision Alignment (2026-03-07)

**Raw Data Pipeline** — MIDGE was throwing away ~95% of API data. VIX downloads 8,000+ daily rows, uses 1. COT downloads 400+ contracts, uses 10. Now ALL data is persisted to SQLite (one DB per domain) in `data/market/raw/` before processing. WAL mode for daemon concurrency. Constructor injection (`raw_store=None`) on VIX/COT/EIA/Trends clients. 14 tests.

**Expedition complete: Autonomous Self-Funding Trading** — 5 research teams + 3 validators. Full synthesis at `research/expedition-autonomous-trading/synthesis.md`. Key findings:
- Kalshi (prediction markets) recommended as first live venue — CFTC-regulated, Python SDK, macro domain overlap
- Alpaca as equities fallback (where proven edge lives)
- MIDGE's cross-domain convergence is a genuine structural moat — no competitor does 12+ domain stacking
- Signal profile is structurally invisible to SEC/FINRA/ARTEMIS surveillance
- Critical unknown: whether equity-measured edge transfers to binary event contracts

**Vision alignment (Guiding Light directive):** MIDGE is an "inevitability surfacer" — global pattern observer across ALL domains. Not "one domain at a time" — observe everywhere, execute across Kalshi + Alpaca + crypto simultaneously. Self-funding loop: earnings → more data → deeper patterns → more earnings. $1,000 gate: deploy only when pattern stacks show 80%+ historical accuracy. Temporal ordering matters (which domain fires first, gestation periods).

**Files changed:**
- `mae_core/market/raw_store.py` — NEW: RawStore class, 4 domain upsert methods, SQLite WAL mode
- `mae_core/market/apis/vix_client.py` — raw_store constructor + full OHLC storage
- `mae_core/market/apis/cot_client.py` — raw_store constructor + all-contract storage
- `mae_core/market/apis/eia_client.py` — raw_store constructor + all-observation storage
- `mae_core/market/apis/trends_client.py` — raw_store constructor + all-hourly storage
- `mae_core/bootstrap/market_systems.py` — Shared RawStore creation + injection into 4 clients
- `tests/test_raw_store.py` — NEW: 14 tests
- `research/expedition-autonomous-trading/synthesis.md` — Expedition Phase 3+4 complete

### Granger Causality + Test Isolation + Bug Fixes (2026-03-07)

**Granger causality analyzer** — highest-value analytical upgrade from expedition roadmap. Tests whether signal A's past *improves prediction* of B beyond B's own autocorrelation (controls for autocorrelation, unlike lag-correlation). Uses statsmodels `grangercausalitytests` with first-differencing for stationarity, Bonferroni correction, atomic persistence.

**Thompson key collision fix** — `economic_calendar` was sharing `finnhub_economic` Thompson key. Two unrelated signal types (suppression windows vs Finnhub news) were corrupting one Beta distribution. Now separate keys.

**Test isolation fix** — `create_mae()` mutates `LEARNING_CONFIG` in place (meta-learning pushes `min_correlation` from 0.6 to 0.85). This polluted 7 tests in `test_composite_hypotheses.py` and `test_hypothesis_generator.py` that ran afterward. Fix: conftest autouse fixture now deep-copies and restores `LEARNING_CONFIG` after each test.

**3 silent failure fixes** — `except Exception: pass` blocks in sensing_hook.py (tier alerter, EventBus publish) and market_hooks.py (regime classifier) now log at DEBUG level.

**Named constants** — Independence correction thresholds `STRONG_CORRELATION_THRESHOLD = 0.5` and `MODERATE_CORRELATION_THRESHOLD = 0.3` replace bare magic numbers in convergence_alerter.py.

**Monolith fix** — `market_systems.py` was 523 lines (over 500 cap). Extracted `_register_trust_and_gateway()` + data-driven edge detector loop → 493 lines.

**Files changed:**
- `mae_core/market/intelligence/granger_analyzer.py` — NEW: GrangerAnalyzer, GrangerFinding, causal strength lookup
- `mae_core/market/sensing_hook.py` — Thompson key collision fix, silent failure logging
- `mae_core/bootstrap/market_hooks.py` — Granger step hook (every 500 steps), silent failure logging
- `mae_core/bootstrap/market_systems.py` — GrangerAnalyzer instantiation, trust refactor (523→493 lines)
- `mae_core/bootstrap/market_registration.py` — granger_analyzer holon entry
- `mae_core/market/intelligence/convergence_alerter.py` — Named correlation constants
- `main.py` — granger_analyzer in systems dict, system count 52→55
- `tests/conftest.py` — LEARNING_CONFIG isolation (deep-copy + restore)
- `tests/test_granger_analyzer.py` — NEW: 15 tests
- `tests/test_wave2_3_integration.py` — Rotation count 28→30
- `tests/test_decomposition_wiring.py` — Converter count 35→37
- `tests/test_integration.py` — granger_analyzer in expected keys

### Template Persistence Fix + Performance Optimization (2026-03-06)

**Critical bug found and fixed:** 222K fingerprints existed but 0 templates — PatternWatcher had nothing to match against. Three compounding bugs:

1. **Crash-unsafe write:** Templates used `open("w")` (truncate-on-open). Any crash or restart with empty `self._templates` zeroed the file. Fingerprints survived because they use `open("a")` (append).
2. **Per-fingerprint persistence:** `_persist_templates()` was called 222K times per excavation run (once per fingerprint via `_update_template()`). Catastrophic I/O.
3. **O(N²) rebuild:** `rebuild_templates()` did linear scan over all templates per fingerprint. `add_instance()` iterated ALL instances to compute avg.

**Fixes:**
- Atomic write: `.tmp` then rename, with Windows fallback (other process may hold file lock)
- Empty guard: Never overwrite non-empty file with empty data
- Batch persistence: `store_batch()` persists templates ONCE at the end
- O(1) key index: `_template_key_index` dict replaces linear scan in `_update_template()` and `rebuild_templates()`
- Incremental avg: `_move_pct_sum` field — O(1) per add instead of O(N) instance scan
- Instance cap: Templates keep last 200 instances (fingerprints are the archive)
- 11 new tests in `test_template_persistence.py`

**Result:** 222,916 fingerprints → 39 templates (26 cross-validated) rebuilt in 6.8s. PatternWatcher now operational.

**Files changed:**
- `mae_core/market/archaeology/pattern_library.py` — Atomic write, batch persist, key index
- `mae_core/market/archaeology/fingerprint.py` — Incremental avg, instance cap, `_move_pct_sum` field
- `tests/test_template_persistence.py` — NEW: 11 tests

### Thompson + Independence Fix (2026-03-05)

**Thompson forgetting bug:** Forgetting (0.99x every 100 steps) outran learning (outcomes every 200 steps). 81/83 distributions decayed to uniform. Fix: cadence 100→200, floor 1.0→2.0.

**Independence correction:** CorrelationTracker was NOT connected to ConvergenceAlerter. Diversity bonus used raw domain count. Fix: inject CorrelationTracker, compute effective domain count (|r|>0.5 → half credit), seed from lag_correlations.json.

**Files changed:**
- `mae_core/market/intelligence/thompson_sampler.py` — Floor 2.0, forgetting summary log
- `mae_core/bootstrap/market_hooks.py` — Cadence 200 steps
- `mae_core/market/intelligence/convergence_alerter.py` — Effective domain count, correlation injection
- `mae_core/market/intelligence/correlation_tracker.py` — `seed_from_lag_data()`
- `mae_core/bootstrap/market_systems.py` — Correlation seeding + two-phase wiring
- `tests/test_thompson_feedback.py` — NEW: 16 tests
- `tests/test_independence_correction.py` — NEW: 23 tests

### Prediction-to-Action (2026-03-05)

Three features per Guiding Light's directive:
- **Dynamic Outcome Windows**: `lag_profile_raw` accumulator, `expected_move_window_days` property
- **Plain-Language Alerts**: `plain_language.py` — zero-jargon 5-section formatter
- **Active Tracking**: `active_tracker.py` — TrackedAsset registry with status transitions

### Pattern Archaeologist v2 (2026-03-04)

Symbol-agnostic template engine. Full excavation completed 3,237 symbols via Polygon.io.

### EIA Energy Data Integration (2026-03-06)

**First real-economy domain.** All 11 prior MIDGE domains were financial-market data. EIA adds physical-world supply/demand signals that cross-reference with insider trades, congressional activity, and technical patterns.

**What it does:**
- Fetches weekly petroleum inventory (crude, gasoline, distillate), natural gas storage, and monthly crude production from EIA API v2
- Inventory BUILD = bearish (supply > demand), DRAW = bullish — inverse logic
- Strength calibrated against typical weekly change ranges
- Affected tickers mapped: XLE, XOP, USO, UNG, VLO, MPC, EQT, etc.
- 6-hour cache (weekly data updates on specific days)
- Strategic tier in convergence engine, 7-day domain window
- Full intelligence layer: Thompson key, source_reliability (0.70), energy decay rate (0.05), domain correlation tracking

**API bugs fixed (live-tested against EIA):**
- Added `data[]=value` param (EIA v2 requires explicit column selection — without it, returns metadata only)
- Gasoline/distillate facets: `SAX` → `SAE` (Ending Stocks, not Excluding SPR)
- Natgas facets: `SAX` → `SWO` (Working Gas total)
- Crude production: added `series: MCRFPUS1` facet (prevents multi-series collision)

**Live data (2026-02-27 report):** All 5 series returning — crude stocks BUILD +3,475K bbl (bearish 0.69), gasoline BUILD +801K (bearish 0.40), distillate BUILD +429K (bearish 0.21), natgas BUILD +65 Bcf (bearish 0.81), crude production +9,655K bbl/mo (bearish 0.97).

**Files changed:**
- `mae_core/market/apis/eia_client.py` — NEW: EIAClient, EnergyIndicator, 5 series + API bug fixes
- `mae_core/market/signal_adapters/market_data.py` — `from_energy_indicator()` adapter
- `mae_core/market/sensing_fetchers.py` — `fetch_eia()` function
- `mae_core/market/sensing_hook.py` — SOURCE_ROTATION (29), TIER_ROUTING, _ROTATION_TO_THOMPSON, _ABSENCE_SOURCE_DOMAINS, __init__, _fetch_source
- `mae_core/bootstrap/market_systems.py` — EIAClient instantiation + trust registration (0.95)
- `mae_core/bootstrap/market_hooks.py` — Pass eia_client to MarketSensingHook
- `mae_core/market/archaeology/pattern_library.py` — `"eia_energy": "energy"` in _SOURCE_DOMAIN_MAP
- `mae_core/market/intelligence/convergence_alerter.py` — Energy domain window + category + _SOURCE_TO_THOMPSON_KEY + _DOMAIN_SOURCES
- `mae_core/market/intelligence/learning_config.py` — source_reliability + decay_rates
- `mae_core/market/plain_language.py` — Energy domain + source translations
- `mae_core/market/signal_adapters/__init__.py` — Re-export from_energy_indicator
- `mae_core/market/signal.py` — Re-export from_energy_indicator
- `tests/test_eia_client.py` — NEW: 38 tests (34 original + 4 intelligence layer)
- `tests/test_new_source_wiring.py` — Updated rotation count 28→29
- `tests/test_integration.py` — Added eia_client to market_keys

**Requires:** `EIA_API_KEY` env var (free: https://www.eia.gov/opendata/register.php) — registered and set in `.env`

### Proven Signal → Profitable System (2026-03-03)

Four work packages closing the operational gap (Thompson isolation, combo feedback, confidence gating, MFE/MAE).

---

## Stats

- **147 systems** (92 core + 55 market), **4,536 tests**, **157 holons**, **425 connections**
- **105 market files** (32 API + 12 edge + 28 intelligence + 8 signal_adapters + 10 archaeology + 15 root)
- **12 domains**, **30 sources** in sensing rotation
- **33-layer bootstrap**, **14 mixins** on MycelialAgent
- **222,916 fingerprints**, **39 templates** (26 cross-validated across 3+ symbols)

## Current State

- **Daemon: STOPPED.** Old daemon (PID 184380, since March 3) and excavation (PID 262480, since March 4) were killed — running pre-fix code.
- **Templates: REBUILT.** 39 templates live in `pattern_templates.jsonl`. PatternWatcher can now match live signals.
- **Thompson: FIXED.** Forgetting/learning cadence aligned. Independence correction active.
- **EIA: LIVE.** All 5 energy series returning real data. Intelligence layer fully wired (Thompson, correlation, decay).
- **Raw Data Store: LIVE.** SQLite per domain in `data/market/raw/`. VIX/COT/EIA/Trends all storing full API data before processing.
- **Congress.gov: WIRED.** Legislative signal client integrated. 11 policy areas mapped to sector ETFs. `CONGRESS_GOV_API_KEY` configured in `.env`.
- **Kalshi: SDK INSTALLED.** `kalshi-python 2.1.4`. `KALSHI_API_KEY` in `.env`. Needs verification against demo env.
- **Alpaca: CLIENT BUILT.** `alpaca-py 0.43.2`. Paper trading bridge ready. Awaiting `ALPACA_API_KEY` + `ALPACA_SECRET_KEY` in `.env`.
- **Sensing: 12 WORKERS.** Up from 3. 4x concurrent signal collection.
- **Web scraping: INSTALLED.** `httpx + selectolax + trafilatura` ready for autonomous data discovery.
- **OpenInsider clusters: WIRED.** 3+ insiders buying = high-confidence signal (was built but never called).
- **FINRA: ENHANCED.** Speculative short ratio now computed (was parsed then discarded).
- **Expedition: COMPLETE.** Full synthesis at `research/expedition-autonomous-trading/synthesis.md`.
- **Granger: WIRED.** Causal analysis runs every 500 steps.
- **Needs restart:** `python main.py --daemon --agents 12 --steps 500 --pace 2.0`

## What's Next

**USE `/triadic-construction` for the next build session — multiple independent features below.**

### Priority 1: Get MIDGE Running Again
1. **Restart daemon on fixed code** — picks up ALL fixes: Thompson, independence, templates, active tracking, EIA, Congress.gov, raw data store
2. **Re-run excavation** — companion process with fixed template code + new EIA + Congress.gov domains

### Priority 2: Data Overdrive — Squeeze Every API Call
3. **Raw store expansion (PARTIAL — 12/24 done)** — 12 clients still unwired: SEC EDGAR (derivative transactions skipped), Massive, CoinGecko, CoinCap, OpenInsider, FinViz, EdgarEnhanced, EconCalendar, FinnhubWS, ApeWisdom, JobTracker, SAM.gov
4. **Wire unused built methods** — FinViz `get_insider_trades()` (built, never called), EDGAR `get_recent_13f_filers()` (built, never called), COT managed money positions (available, not extracted)
5. **Temporal sequence matching** — upgrade pattern templates from "which domains converge" to "in what order, with what gaps." THE differentiator.
6. **Web scraping infrastructure** — httpx/selectolax/trafilatura installed, needs crawler agent that follows links and extracts market-relevant content

### Priority 3: Execution Bridges
7. **Alpaca paper trading** — client built, needs API keys from Guiding Light then wire to convergence alerts
8. **Kalshi SDK verification** — installed, needs auth test against demo env
9. **MarketSelector** — map top 20 historical convergence alerts to Kalshi contracts

### Priority 4: More Domains + Analytics
10. **New real-economy domains** — USDA agriculture, BDI logistics (via FRED), AIS maritime
11. **Transfer entropy** (infomeasure) — nonlinear causal detection beyond Granger
12. **RMT denoising** (skfolio) — clean correlation matrices
13. **FP-Growth** (mlxtend) — sequential pattern mining on signal archive

### Backlog
- Options flow via Unusual Whales ($35/mo — after self-funding validates)
- Coinbase AgentKit for crypto self-funding loop (Stage 2, after Kalshi validates)
- PCMCI+ (Tigramite) for multivariate causal discovery

## Verification

```bash
python -m pytest tests/ -q              # 4536 pass, 0 failures, 2 xfail
python main.py --agents 3 --steps 30    # Smoke test
python -c "from mae_core.market.archaeology.pattern_library import PatternLibrary; lib = PatternLibrary(); print(f'{lib.size} fingerprints, {lib.template_count} templates')"
```
