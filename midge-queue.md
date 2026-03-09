# MIDGE Queue

**Purpose:** Active tasks only. Git history preserves completed work.
**Last updated:** 2026-03-09

---

## Completed This Session

### Fix Broken Systems
- [x] FinnhubWebSocket.start() never called — FIXED
- [x] ActiveTracker._force_grade() calls nonexistent methods — FIXED
- [x] SAM.gov estimated_value never populated — FIXED
- [x] MassiveClient volume_ratio always 0 — FIXED

### Raw Store Expansion (12→24 clients) — ALL COMPLETE
### Parallelism Upgrade — ALL COMPLETE (12 concurrent, 25-step cadence, daemon off main thread)
### Post-Mortem & Temporal — ALL COMPLETE (domain_sequence, sequence_score, post-mortem reviewer, lag wiring)
### Cultural Discovery — 3/4 COMPLETE (Trends expansion, social analyzer, FinViz insider trades wired)
- [ ] Watchlist keywords → Google Trends integration

### Pattern Discovery (partial)
- [x] STUMPY motif discovery — per-symbol stumpi streaming mode
- [x] River ADWIN reactive regime drift → DriftDetector
- [x] NetworkX causal chain graph (114 nodes, 102 edges, 38 tickers)
- [x] numpy-vectorize Bollinger/RSI computation (10-50x faster TA)
- [x] Yahoo Finance RSS — per-ticker headline velocity

### File Splits — COMPLETE
- [x] post_mortem.py split (569→314 + 245 + 57)
- [x] market_systems.py trimmed (535→453)

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
- [x] PySAD RRCF streaming anomaly detection → StreamingAnomalyDetector (DONE)
- [ ] CausationEntropy weekly batch (experimental, monitored)
- [ ] smart-money-concepts as new edge detector domain

### Causal Reasoning
- [ ] Sequential chain detection (A→B→C stage-gating)
- [ ] Per-domain temporal ordering in pattern templates
- [x] Wire world_model ripple effects into convergence alert enrichment (DONE — inevitability cascade)
- [x] Partial convergence enriched with causal_predictions for Octopus investigation
- [x] Proactive causal watch emitted on signal ingestion (CH_CAUSAL_WATCH)
- [x] Plain-language CASCADE section in human-readable alerts

### LLM Integration
- [ ] Run nvidia-smi to determine GPU VRAM
- [ ] Qwen3-14B via Ollama for causal narratives
- [ ] OllamaProvider + OllamaReasoningSubscriber (~300 lines)
- [ ] WHY section in plain_language.py alerts
- [ ] Three-stage bull/bear/synthesis prompt

### Processing Performance
- [ ] aiohttp async for PolygonBulkFetcher (85s→3-10s)

### Cultural Discovery (remaining)
- [ ] Watchlist keywords → Google Trends integration

---

## Priority 3: New Data Sources

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
