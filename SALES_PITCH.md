# MIDGE — Sales Pitch & Feature Highlights

---

## One-Liner

**MIDGE is a self-improving market intelligence organism that discovers when insider buying, congressional trades, government contracts, and 24 other data sources all point at the same ticker — and learns from every outcome.**

---

## Elevator Pitch (30 seconds)

Most trading systems watch price and fire rules. MIDGE watches 27 independent data sources — SEC insider filings, congressional stock trades, government contract awards, hiring surges, crypto flows, sentiment, technical patterns — and only alerts when three or more of these disconnected worlds converge on the same ticker and direction.

Then she learns. Every prediction is tracked against real market outcomes. A Bayesian engine (Thompson Sampling) continuously updates which signals actually predict price movement — no manual tuning, no backtesting-only. She generates her own trading hypotheses, tests them against historical data, promotes winners, retires losers, and adjusts how she discovers new patterns. She's not a dashboard. She's an organism that gets smarter with every market day.

---

## Full Pitch (2 minutes)

### The Problem

Financial markets generate enormous amounts of data across disconnected domains. An insider buying cluster at a defense contractor happens the same week Congress members buy the stock, the company wins a $200M contract, and they're hiring aggressively. Each signal alone is noise. Together, they're a pattern that precedes price movement.

No human can watch all these domains simultaneously. Most algorithmic systems can't either — they're built on price data and technical indicators, missing the information advantage that comes from cross-domain convergence.

### The Solution

MIDGE (Market Intelligence Driven by Generative Exploration) is a biologically-modeled market intelligence system. She continuously ingests data from 27 sources across 7 domains:

- **Insider activity** — SEC Form 4 trades, insider buying clusters, OpenInsider pre-filtered purchases
- **Political** — Congressional stock trades (STOCK Act), committee-contract correlations
- **Government contracts** — USASpending.gov awards, SAM.gov opportunities, pre-announcement prediction
- **Institutional** — 13F holdings, 13D activist filings (>5% stake), short interest
- **Sentiment & flow** — StockTwits social sentiment, Google Trends, Finnhub real-time trades, VIX term structure
- **Technical** — RSI, MACD, Bollinger Bands, Market Structure, ICT session sweeps, Fair Value Gaps
- **Macro** — CFTC positioning (Commitment of Traders), FRED economic indicators, crypto as risk proxy

Her crown jewel is the **Convergence Engine** — it only fires an alert when three or more independent domains align on the same ticker. This isn't a scoring system that adds up points. It's a principled requirement grounded in the same mathematics as Byzantine fault tolerance: you need three independent witnesses to establish trust.

### What Makes Her Different

**She learns from reality.** Every convergence alert becomes a tracked prediction with a specific outcome window (5 days for binary events like earnings, 45 days for insider trades, 90 days for hiring signals). When the window closes, she checks whether the price moved. Success updates her Bayesian beliefs upward. Failure updates them downward. Over 12,500 evaluated outcomes, she's learned which sources actually predict movement — and it's not the ones most people watch.

**She improves how she improves.** This is the real edge. Layer 1: she adjusts signal weights from outcomes (Thompson Sampling — done). Layer 2: she generates her own hypotheses from lag correlations, validates them against Deflated Sharpe Ratio to prevent overfitting, and promotes or retires them autonomously (done). Layer 3: she adjusts the parameters of her own learning — gate thresholds, correlation filters, source weights — based on whether her hypotheses actually hold up (done). Three layers of recursive self-improvement, each one making the layer below it work better.

**She never sleeps.** Daemon mode runs 24/7. Crypto sources (CoinGecko, CoinCap) cover off-hours. Finnhub WebSocket streams real-time equity data during market hours. Signal buffers persist across restarts. She picks up exactly where she left off.

**She's built on math, not vibes.** Eight mathematical laws govern every design decision, proven independently by six fields (graph rigidity, semiotics, consensus theory, dialectical philosophy, network sociology, consciousness theory). The architecture isn't a heuristic pile — it's a proof-backed structure where every component serves a mathematically justified purpose.

### By the Numbers

| Metric | Value |
|--------|-------|
| Data sources | 27 live feeds across 7 domains |
| Signal reliability distributions | 70+ learned Bayesian distributions |
| Evaluated outcomes | 12,500+ real market predictions scored |
| Automated tests | 4,250 passing (zero failures) |
| System components | 144 interconnected systems |
| Self-aware subsystems | 155 holons (each implements 10 capabilities) |
| Verified connections | 422 triadic (no unwitnessed links) |

---

## Feature Highlights

### Convergence Engine
The core value proposition. Synthesizes signals from all domains into actionable alerts. Minimum 3-domain convergence required (mathematically justified). Thompson-weighted confidence scoring. Per-ticker deduplication. Multi-timeframe analysis (tactical 24h, strategic 7d, thematic 30d).

### Thompson Sampling (Bayesian Learning)
Every signal source maintains a Beta(alpha, beta) distribution representing learned reliability. The system explores uncertain sources and exploits proven ones — automatically, without tuning. Bayesian forgetting prevents stale data from dominating. Combo-level distributions track which domain *combinations* predict movement (not just individual sources).

### Hypothesis Engine (Self-Improving Discovery)
Discovers lag correlations in historical data. Generates formal hypotheses with causal stories. Validates using Deflated Sharpe Ratio (anti-overfitting). Promotes winners, retires losers. Feeds outcomes back to improve future discovery. The system that discovers patterns also discovers how to discover better patterns.

### ICT Smart Money Concepts
Session sweep detection across Asia/London/NY kill zones. Inverse Fair Value Gap (IFVG) identification. Quality-gated composite scoring. Backtested: quality gate removes 38% of trades while maintaining equal profit. Uses free 1-minute yfinance data on ES=F/NQ=F — zero API cost.

### Always-On Operation
24/7 daemon mode with wall-clock pacing. MarketClock routes the right sources to the right sessions. Crypto provides continuous signal flow when equities close. Signal persistence survives restarts. Heartbeat monitoring for external alerting.

### Deception Detection
Identifies pump-and-dump patterns, retail traps, and wash trading. Confidence automatically reduced when manipulation is detected. Protects against acting on artificially inflated signals.

### Economic Calendar Suppression
Knows FOMC, CPI, NFP dates. Automatically halves confidence during event windows where macro announcements override individual signals. Defensive, not predictive — prevents bad trades, not makes good ones.

---

## Who Is MIDGE For?

- **Quantitative traders** who want cross-domain signal convergence beyond price/volume
- **Research analysts** who need automated monitoring of SEC filings, congressional trades, and contract awards
- **Systematic fund managers** looking for a self-improving signal generation platform
- **Independent traders** who want institutional-grade multi-source intelligence without institutional-grade costs (most data sources are free)

---

## What MIDGE Is Not

- Not a black-box model — every signal, weight, and decision is traceable
- Not a high-frequency trading system — she thinks in days and weeks, not milliseconds
- Not dependent on expensive data feeds — 20 of 27 sources are free/no API key
- Not a static strategy — she learns, adapts, and improves autonomously
