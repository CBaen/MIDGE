# MIDGE Queue

**Purpose:** Active tasks only. Git history preserves completed work.
**Last updated:** 2026-03-09

---

## Priority 1: Execution Bridges

### Alpaca Equities — WIRED
- [x] Get ALPACA_SECRET_KEY from Guiding Light — DONE (in .env)
- [x] Signal translator built — ConvergenceAlert → ExecutableSignal (36 tests)
- [x] DrawdownMonitor circuit breaker — halts paper trades at 40% DD
- [x] Wire Alpaca client to convergence alerts — `_submit_to_alpaca()` in market_hooks.py
- [ ] Post-catalyst timing strategy for entry (legal protection)
- [ ] Build public confirmation timestamp logging (compliance trail)

### Kalshi Integration
- [ ] Verify kalshi-python SDK against current API + demo env
- [ ] Review Kalshi ToS for algo trading restrictions
- [ ] Prototype MarketSelector — map top 20 alerts to Kalshi contracts
- [ ] Backtest MIDGE signals against historical Kalshi contracts
- [ ] Kalshi REST API client with RSA key-pair auth (daemon-friendly)
- [ ] Kalshi as signal source — new "prediction_market" domain (dual use)
- [ ] Shadow mode stage — $1-10 real trades before scaling

### FTMO (after forex/commodity signal validation)
- [ ] FTMO free trial signup as measurement instrument (14 days free data)
- [ ] Verify US access status for Guiding Light's state (OANDA closing March 31, 2026)
- [ ] MetaAPI bridge for MT5 execution

---

## Priority 2: Intelligence Upgrades

### Pattern Discovery
- [ ] CausationEntropy weekly batch (experimental, monitored)
- [ ] smart-money-concepts as new edge detector domain

### Causal Reasoning
- [ ] Sequential chain stage-gating — require link N to confirm before watching link N+1 (CascadeTracker has chain tracking but no ordering enforcement)

### LLM Integration
- [ ] Run nvidia-smi to determine GPU VRAM
- [ ] Qwen3-14B via Ollama for causal narratives
- [ ] OllamaProvider + OllamaReasoningSubscriber (~300 lines)
- [ ] WHY section in plain_language.py alerts
- [ ] Three-stage bull/bear/synthesis prompt

### Processing Performance
- [ ] aiohttp async for PolygonBulkFetcher (85s→3-10s)
- [ ] Wall-clock cadences (replace step-based)

---

## Priority 3: New Data Sources

- [ ] edgartools as SEC upgrade (proper 13F/13D parsing, 1800 stars)
- [x] BDI logistics via FRED proxy — DONE as FRED Freight TSI (TSIFRGHT), piggybacks fred_yields
- [x] Wire Binance funding rate client into sensing pipeline — DONE (adapter+constants+fetcher+bootstrap, 47 tests, "positioning" domain)
- [ ] AIS maritime (AISHub — verify free tier terms)
- [ ] Web scraping crawler agent (httpx/selectolax/trafilatura installed, unused)
- [x] Central bank events (ECB/BoJ/BoE/PBoC/BoC/RBA) — DONE, Finnhub international high-impact events activated
- [ ] Crypto order book depth + on-chain metrics

---

## Priority 4: Advanced Analytics

- [ ] Transfer entropy (infomeasure) — nonlinear causal detection
- [ ] RMT denoising (skfolio) — clean correlation matrices
- [ ] FP-Growth (mlxtend) — sequential pattern mining on signal archive
- [ ] PCMCI+ (Tigramite) — multivariate causal discovery

---

## Backlog (gated on revenue or approvals)

- Options flow via Unusual Whales ($35/mo — after self-funding validates)
- Coinbase AgentKit for crypto self-funding loop (Stage 2, after Kalshi validates)
- PRAW Reddit integration (API access now requires manual review application)

---

## Housekeeping

- [x] Write tests for USDA client — 44 tests in test_usda_client.py + 7 in test_raw_store.py
- [x] Write tests for FRED client — 59 tests in test_fred_client.py
- [x] Wire USDA + FRED yields into sensing pipeline (rotation, dispatcher, bootstrap)
- [x] Clean Thompson test artifacts from distributions file — DONE (6 artifacts removed, 92 legitimate distributions remain)
- [x] connection_registrations_bio.py split — DONE (500→311 + backbone 118 + cognition 131)
- [ ] market_hooks.py size audit — may need decomposition (approaching 500-line cap)
- [ ] Broker-side bracket orders (survive MIDGE process failure)

---

## Completed (cleared from queue — git history preserves)

- Fix 4 broken systems (FinnhubWS, ActiveTracker, SAM.gov, MassiveClient)
- Raw store expansion 12→24 clients (all wired, 58 tests)
- Parallelism upgrade (12 concurrent, 25-step cadence)
- Post-mortem + temporal domain ordering (domain_sequence, sequence_score)
- Cultural discovery (Trends keyword expansion, social analyzer, FinViz insider trades, watchlist→Trends)
- Pattern discovery (STUMPY motif, ADWIN drift, PySAD RRCF, NetworkX causal graph, TA vectorization, Yahoo RSS)
- File splits (post_mortem 569→314, market_systems 584→444)
- Thompson feedback loop fix (4 bugs, replay_from_history, 24 tests)
- Signal translator (ConvergenceAlert → ExecutableSignal, 36 tests)
- Watchlist expansion (forex/futures/commodity tickers)
- USDA WASDE client built + tests (44+7) + sensing pipeline wired
- FRED client tests (59) + yield curve sensing pipeline wired (DGS2, DGS10, T10Y3M, DTWEXBGS)
- Bio-market activation (29/30 systems wired, ResourceGovernor)
- SystemHealthMonitor error wiring (8 blocks)
- DrawdownMonitor daemon persistence flush
- Focused attention / priority polling
- Investigation pipeline + signal-triggered convergence
- Inevitability cascade + CascadeTracker + chain boost + backward discovery
- Agent-level situation claiming
- ADTS regime-aware Thompson forgetting
- Fingerprint RAM offload (223K → ID set only)
- OctopusColony bootstrapped + pipeline bridge
- Per-domain temporal ordering in convergence alerts
- market_infrastructure.py extraction (market_systems.py 584→444)

---

**When adding a task:**
1. Add under appropriate priority section
2. Git commit preserves history — no separate history file needed
