# MIDGE Queue

**Purpose:** Active tasks only. Git history preserves completed work.
**Last updated:** 2026-03-09

---

## Active — Current Session

### Fix Broken Systems (4 agents working)
- [ ] FinnhubWebSocket.start() never called — real-time streaming is dead wire
- [ ] ActiveTracker._force_grade() calls nonexistent methods — fast grading broken
- [ ] SAM.gov estimated_value never populated — get_large_contracts() broken
- [ ] MassiveClient volume_ratio always 0 — callers don't supply bars_by_ticker

### Raw Store Expansion (12→24 clients)
- [ ] SEC EDGAR (Form 4 derivatives, full 8-K text)
- [ ] Massive (grouped daily bars)
- [ ] CoinGecko (ATH/ATL, supply, sparklines)
- [ ] CoinCap (asset data)
- [ ] OpenInsider (+ SEC filing URL from cell[0])
- [ ] FinViz (insider trades, unusual volume, short float)
- [ ] EdgarEnhanced (13D/13F metadata)
- [ ] FinnhubWS (trade tick persistence)
- [ ] ApeWisdom (social sentiment)
- [ ] JobTracker (full job records: skills, experience, location)
- [ ] SAM.gov (full opportunity data + description text)

### Parallelism Upgrade
- [ ] Concurrent fetches 3→8
- [ ] Fetch cadence 50→25 steps
- [ ] Agent thread cap 8→20
- [ ] ExcavationDaemon off main thread

### Post-Mortem & Temporal
- [ ] Domain sequence tracking on ConvergenceAlert
- [ ] Post-mortem prediction reviewer (WHY did it fail?)
- [ ] Wire lag findings into convergence alerter
- [ ] Sequence scoring (reward correct domain ordering)

### Cultural Discovery
- [ ] Google Trends keyword auto-expansion (related_queries feedback loop)
- [ ] StockTwits message text analyzer (keyword themes, intensity)
- [ ] Wire FinViz get_insider_trades() into sensing pipeline
- [ ] Watchlist keywords → Google Trends integration

---

## Priority 1: Execution Bridges (from expedition-autonomous-trading)

### Pre-Deployment Validations (NO capital until these complete)
- [ ] Backtest MIDGE signals against historical Kalshi contracts
- [ ] Verify kalshi-python SDK against current API + demo env
- [ ] Prototype MarketSelector — map top 20 alerts to Kalshi contracts
- [ ] Define stop-loss threshold / circuit breaker (60% floor suggested)
- [ ] Review Kalshi ToS for algo trading restrictions
- [ ] Build public confirmation timestamp logging (compliance trail)

### Kalshi Integration
- [ ] Kalshi REST API client with RSA key-pair auth (daemon-friendly)
- [ ] Kalshi as signal source — new "prediction_market" domain (dual use)
- [ ] Shadow mode stage — $1-10 real trades before scaling

### Alpaca Equities
- [ ] Get ALPACA_SECRET_KEY from Guiding Light
- [ ] Wire Alpaca client to convergence alerts
- [ ] Post-catalyst timing strategy for entry (legal protection)

---

## Priority 2: Intelligence Upgrades (from expedition-midge-gifts)

### Pattern Discovery
- [ ] STUMPY motif discovery — per-symbol stumpi streaming mode
- [ ] PySAD RRCF streaming anomaly detection → VelocityDetector
- [ ] River ADWIN reactive regime drift → RegimeClassifier
- [ ] CausationEntropy weekly batch (experimental, monitored)
- [ ] smart-money-concepts as new edge detector domain

### Causal Reasoning
- [ ] NetworkX causal chain graph (50-100 curated relationships)
- [ ] Sequential chain detection (A→B→C stage-gating)
- [ ] Per-domain temporal ordering in pattern templates

### LLM Integration
- [ ] Run nvidia-smi to determine GPU VRAM
- [ ] Qwen3-14B via Ollama for causal narratives
- [ ] OllamaProvider + OllamaReasoningSubscriber (~300 lines)
- [ ] WHY section in plain_language.py alerts
- [ ] Three-stage bull/bear/synthesis prompt

### Processing Performance
- [ ] numpy-vectorize Bollinger/RSI computation (10-50x faster TA)
- [ ] aiohttp async for PolygonBulkFetcher (85s→3-10s)

---

## Priority 3: New Data Sources

- [ ] Yahoo Finance RSS — per-ticker headline velocity (free, feedparser)
- [ ] edgartools as SEC upgrade (proper 13F/13D parsing, 1800 stars)
- [ ] USDA WASDE agriculture (free, monthly, orthogonal domain)
- [ ] BDI logistics via FRED proxy (free, 5-30 day lead times)
- [ ] AIS maritime (AISHub — verify free tier terms)
- [ ] Web scraping crawler agent (httpx/selectolax/trafilatura installed, unused)

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

**When adding a task:**
1. Add under appropriate priority section
2. Git commit preserves history — no separate history file needed
