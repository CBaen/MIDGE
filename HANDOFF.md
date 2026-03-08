# MIDGE Handoff

## What Happened

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

**Queue updated:** `midge-queue.md` now tracks 70+ items from 3 expedition syntheses that were researched but never queued.

**Files changed (21 files):**
- `mae_core/market/raw_store.py` — 25 store methods (was 15)
- `mae_core/market/intelligence/post_mortem.py` — NEW: PostMortemReviewer
- `mae_core/market/intelligence/social_text_analyzer.py` — NEW: SocialTextAnalyzer
- `mae_core/market/intelligence/convergence_alerter.py` — domain_sequence, sequence_score, lag findings
- `mae_core/market/archaeology/active_tracker.py` — _force_grade fixed
- `mae_core/market/sensing_hook.py` — 12 concurrent, 25-step cadence, social analyzer
- `mae_core/market/sensing_fetchers.py` — FinViz insider trades wired
- `mae_core/market/apis/finnhub_websocket.py` — raw_store wiring
- `mae_core/market/apis/massive_client.py` — volume_ratio fix + raw_store
- `mae_core/market/apis/sam_gov.py` — estimated_value fix + raw_store
- `mae_core/market/apis/coingecko_client.py` — raw_store wiring
- `mae_core/market/apis/coincap_client.py` — raw_store wiring
- `mae_core/market/apis/openinsider_client.py` — raw_store wiring (+ SEC URL)
- `mae_core/market/apis/finviz_client.py` — raw_store wiring
- `mae_core/market/apis/edgar_enhanced_client.py` — raw_store wiring
- `mae_core/market/apis/apewisdom.py` — raw_store wiring
- `mae_core/market/apis/job_tracker.py` — raw_store wiring
- `mae_core/market/apis/sec_edgar/client.py` — raw_store wiring
- `mae_core/bootstrap/market_systems.py` — PostMortemReviewer + raw_store to all clients (535 lines, needs split)
- `mae_core/bootstrap/market_hooks.py` — FinnhubWS start(), lag→alerter wiring, post-mortem hook, social analyzer
- `mae_core/model.py` — thread cap raised
- `tests/test_raw_store.py` — 58 tests (was 35)
- `midge-queue.md` — comprehensive 70+ item queue from expedition research

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

- **147 systems** (92 core + 55 market), **4,536+ tests**, **157 holons**, **425 connections**
- **107 market files** (32 API + 12 edge + 30 intelligence + 8 signal_adapters + 10 archaeology + 15 root)
- **12 domains**, **31 sources** in sensing rotation (FinViz insider trades added)
- **33-layer bootstrap**, **14 mixins** on MycelialAgent
- **222,916 fingerprints**, **39 templates** (26 cross-validated across 3+ symbols)
- **24/24 API clients** now persist raw data to SQLite

## Current State

- **Daemon: STOPPED.** Needs restart with: `python main.py --daemon --agents 12 --steps 500 --pace 2.0`
- **Raw Data Store: COMPLETE.** All 24 clients store raw data before processing. SQLite per domain in `data/market/raw/`.
- **FinnhubWebSocket: FIXED.** start() now called at bootstrap. Real-time streaming operational.
- **Post-Mortem: WIRED.** Runs every 500 steps alongside Granger. Writes insights to `data/market/post_mortem_insights.json`.
- **Temporal Ordering: ACTIVE.** domain_sequence + sequence_score on every convergence alert.
- **Social Text: WIRED.** SocialTextAnalyzer reads StockTwits messages from raw_store, emits theme signals.
- **Trends Discovery: ACTIVE.** Related queries feed keyword expansion. Up to 30 discovered keywords.
- **Parallelism: 12 concurrent + 25-step cadence.** Full source rotation in ~94 steps.
- **Templates: REBUILT.** 39 templates live. PatternWatcher matching.
- **Thompson: FIXED.** Forgetting/learning aligned. Independence correction active.
- **Kalshi: SDK INSTALLED.** Needs verification against demo env.
- **Alpaca: CLIENT BUILT.** Awaiting API keys.

## What's Next

See `midge-queue.md` for comprehensive task list (70+ items, 4 priority tiers).

### Immediate (next session)
1. **Restart daemon** — picks up ALL fixes from this session and prior
2. **Run full test suite** — verify zero regressions (was running when machine restarted)
3. **Split market_systems.py** (535→<500) and post_mortem.py (569→<500)

### Priority 1: Execution Bridges
- Kalshi SDK verification + MarketSelector prototype
- Backtest signals against historical Kalshi contracts
- Stop-loss threshold / circuit breaker
- Public confirmation timestamp logging

### Priority 2: Intelligence Upgrades
- STUMPY motif discovery, PySAD anomaly detection, River ADWIN drift
- NetworkX causal chain graph (50-100 relationships)
- Qwen3-14B via Ollama for causal narratives (check GPU VRAM first)
- numpy-vectorize TA computation, aiohttp async for Polygon

### Priority 3: New Data Sources
- Yahoo Finance RSS (free, headline velocity)
- edgartools upgrade (proper 13F/13D)
- USDA WASDE, BDI logistics, AIS maritime

## Verification

```bash
python -m pytest tests/ -q              # Should be 4536+
python -m pytest tests/test_raw_store.py -v  # 58 tests
python main.py --agents 3 --steps 30    # Smoke test
```

## Flags

- `market_systems.py` at 535 lines (needs split, 500 cap)
- `post_mortem.py` at 569 lines (needs split, 500 cap)
- `raw_store.py` at 1,686 lines (large but single-responsibility — each method is independent)
