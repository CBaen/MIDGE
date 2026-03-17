# Research Council Brief: Crypto Trading Intelligence for MIDGE

## Date: 2026-03-17
## Project: MIDGE (Market Intelligence for Day-trading and Genuine Edge)

### The Question

What additional capabilities, data sources, techniques, and structural changes does MIDGE need to become an effective autonomous crypto swing/day trader? The crypto market runs 24/7 — MIDGE's current architecture was built for equities (9:30-4:00 ET). How should her sensing, convergence, and execution adapt?

Guiding Light's words: "I need MIDGE to make me money every single day. I'm interested in cryptocurrency for swing trading and day trading. She needs more sources specifically for crypto."

### Expected Outcome

MIDGE autonomously identifies crypto swing/day trade opportunities, places paper trades on Alpaca, and generates profit. No human intervention needed. She tells Guiding Light what she did, not asks what to do.

### Current State (Session 13 — today)

**What MIDGE has for crypto right now (all wired today):**

| Source | Domain | Tickers | Notes |
|--------|--------|---------|-------|
| CoinGecko prices | crypto | BTC, ETH, SOL, XRP, ADA | 24h/7d change, volume, market cap |
| CoinCap prices | crypto | Top 10 by market cap | Secondary price source |
| Crypto Fear & Greed | sentiment | BTC, ETH, SOL, XRP, ADA | Fanned out today (was BTC only) |
| BTC Dominance | crypto_structure | BTC, ETH, SOL, XRP, ADA | Fanned out today (was BTC only) |
| Kraken Futures | derivatives | BTC, ETH, SOL, LINK, LTC, etc. | NEW today — funding rates, OI |
| mempool.space | on_chain | BTC | NEW today — fee pressure, congestion |
| CoinDesk + Cointelegraph RSS | news | Per-headline ticker matching | NEW today — headline velocity |
| DefiLlama | defi | Per-chain tickers | Wired today — TVL, stablecoin flows |
| Binance Derivatives | positioning | Geo-blocked (HTTP 451 from US) | DEAD — replaced by Kraken |
| TA Indicators | technical | All watchlist tickers | RSI, MACD, Bollinger, structure |
| Pattern Archaeology | pattern matching | 57 crypto templates | From education sprint |

**Execution path:** Convergence alert → signal translator (ATR-based SL/TP) → Alpaca paper trade. Crypto symbol conversion fixed today (BTC → BTC-USD for yfinance, BTC/USD for Alpaca).

**Convergence thresholds:** min_domains=3, min_confidence=0.45, min_strength=0.65. Same for crypto and equities.

**Sensing cadence:** Every 25 steps. At 2s/step pace = ~50 seconds between fetch cycles. At ~24s/step actual (LLM overhead) = ~10 minutes between cycles.

**Architecture:** Mesa 3.4 agent-based model. 33-layer bootstrap. 3 agents, 500 steps/round. Step hooks for sensing, convergence, Thompson learning. Signal buffer persists across restarts. Thompson Sampler with 83+ distributions learns which sources are reliable.

### Project Fingerprint

```
Runtime: Python 3.14, Mesa 3.4 (agent-based framework)
Key dependencies: yfinance, httpx, alpaca-py 0.43.2, stumpy, river, neo4j, qdrant-client, groq
Architecture: 33-layer bootstrap, EventBus pub/sub, fractal triadic structure
State management: In-memory signal buffer + JSON persistence + SQLite raw store + Neo4j knowledge graph
Database/Storage: SQLite (raw data), DuckDB (analytics), Neo4j (causal graph), Qdrant (semantic search)
Known constraints:
  - Mae's 8 Mathematical Laws (triadic structure, no bare dyads, fractal self-similarity)
  - Mesa step loop — all sensing happens in step hooks, not async
  - Binance API geo-blocked from US (HTTP 451) — cannot use Binance.com futures
  - Alpaca paper trading for crypto — bracket orders NOT supported
  - CoinGecko free tier: ~30 req/min rate limit
  - LLM calls (Mistral/DeepSeek) add ~20s per step overhead
Prior failed approaches:
  - Binance derivatives → 451 geo-blocked
  - Absence signals counted as real evidence → false convergence (fixed today)
  - Global convergence alerts used for trading → mixed tickers (fixed today)
Active boundaries:
  - Paper trading ONLY until profitability proven
  - No new paid API subscriptions without ROI justification
  - All changes must respect Mae's 8 Laws
```

### Constraints
- Free APIs only (Guiding Light's budget is $0 for data)
- US-accessible (no geo-blocked endpoints)
- Must work within Mesa step-loop architecture
- No breaking the existing equity pipeline (crypto additions are additive)
- Alpaca is the execution venue for now (CCXT for Binance/Bitget later)

### Destructive Boundaries
- Do NOT remove equity trading capability
- Do NOT break the Thompson learning loop
- Do NOT modify Mae's core mathematical laws
- Do NOT suggest paid data sources without clear ROI path

### Failed Approaches
- Binance.com derivatives API — geo-blocked in US
- Global convergence for trading — mixes signals from different tickers
- Absence signals as evidence — inflates domain count falsely
- Session sweep TA signals — 0% win rate across 500+ outcomes

### Codebase Files for Analysis

| File | What It Does |
|------|-------------|
| `mae_core/market/intelligence/convergence_alerter.py` | Crown jewel — multi-domain synthesis |
| `mae_core/market/intelligence/convergence_ticker.py` | Per-ticker convergence (what generates trades) |
| `mae_core/market/sensing_hook.py` | Async sensing orchestrator |
| `mae_core/market/sensing_constants.py` | Source rotation, domain mappings |
| `mae_core/market/sensing_reactive.py` | Dispatch table for all sources |
| `mae_core/bootstrap/market_hooks_trades.py` | Paper trade gate + Alpaca submission |
| `mae_core/market/intelligence/thompson_sampler.py` | Bayesian learning engine |
| `mae_core/market/apis/alpaca_client.py` | Paper trading execution |
| `mae_core/market/intelligence/learning_config.py` | Self-modifiable thresholds |
| `mae_core/market/intelligence/regime_classifier.py` | Market regime detection |
| `mae_core/market/archaeology/pattern_watcher.py` | Live pattern matching |
| `MIDGE-MODES.md` | Operating modes (SLEEP to SPRINT) |

### External Research Angles
1. **Crypto-specific trading strategies** that work with convergence (multi-signal confirmation) — what successful crypto quant traders use
2. **24/7 market adaptation** — how professional crypto trading systems handle continuous markets (session-based vs continuous, sleep cycles, weekend patterns)
3. **Free crypto data sources** beyond what we already have — on-chain analytics, order book depth, whale tracking, social sentiment platforms
4. **Crypto-specific risk management** — volatility-adjusted position sizing, correlation between crypto assets, drawdown management in 24/7 markets
5. **Exchange mechanics** that create mechanical edge — funding rate arbitrage, basis trading, liquidation cascade prediction
