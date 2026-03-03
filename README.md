# MIDGE

**Market Intelligence Driven by Generative Exploration** — Mae differentiated for financial markets.

---

## What MIDGE Is

MIDGE is a fork of [mae-core](https://github.com/CBaen/mae-core), the Mycelial Agent Engine. Where mae-core is the universal organism template — a stem cell — MIDGE is that organism specialized for market intelligence and trading pattern detection.

Same genome. Different epigenome. Law 5 at the project level.

**Current state:** 144 systems (92 core + 52 market), 4,148 passing tests, 422 triadic connections, 155 holons. 88 market intelligence files decomposed into sub-modules (28 API clients, 12 edge detectors, 27 intelligence/learning, 13 root + 8 signal adapters). Mae's full biological architecture + MIDGE's market senses + hypothesis generation loop (RSI Layer 2) + autonomous backtest scheduling (Bridge 3) + market-focused agent actions (role-keyed dispatch) + causal bridge + absence detection + correlation tracking + Ten Gifts (portfolio awareness, order flow, catalyst timing, cross-asset confirmation, deception detection, memory consolidation, fractal resonance, pattern archetypes, somatic anticipation, pattern completion) + Always-On Wave 1 (MarketClock, daemon mode, signal persistence) + Wave 2 (Finnhub WebSocket, CoinGecko/CoinCap crypto) + Wave 3 (OpenInsider, EDGAR 13F/13D, FinViz screener, Economic Calendar suppression), fully bootstrapped as Layer 33.

---

## What Makes MIDGE Unique

MIDGE brings **market-specific senses** that Mae doesn't have:

### Market Senses (APIs)
- **SEC EDGAR** — Form 4 insider trades, Form 8-K material events (free, no API key)
- **Congressional Trades** — STOCK Act disclosures via housestockwatcher.com + RapidAPI
- **Job Market** — Hiring blitz detection (defense contractors, tech firms)
- **Government Contracts** — USASpending.gov + SAM.gov contract awards and opportunities
- **Price Data** — yfinance + Alpha Vantage fallback

### Edge Detectors (Pattern Recognition)
- **Insider Clusters** — 3+ insiders buying within 30 days = high-confidence signal
- **Politician Tracker** — Committee member trades + contract awards = informed trading
- **Filing Time Analyzer** — Late/unusual SEC filings = behavioral signal
- **Contract Predictor** — Bidder + hiring blitz + insider buying = winner before announcement

### Intelligence Layer (Bayesian Brain)
- **Thompson Sampling** — Beta distribution explore/exploit for signal reliability
- **Velocity Detector** — Rate-of-change anomaly detection across all signals
- **Correlation Tracker** — Cross-domain signal correlation (crypto + insider + hiring)
- **Convergence Alerter** — Multi-domain synthesis, the crown jewel

---

## Mae's Foundation

Everything MIDGE inherits from mae-core:

- **Biologically-inspired multi-agent system** on Mesa 3.4
- **33-layer bootstrap** — organism assembly from atoms to organs
- **8 Mathematical Laws** governing all structure (triads, holons, fractals, autopoiesis)
- **13-step agent lifecycle** — PREDICT, SENSE, DECIDE, ACT, LEARN, etc.
- **Pure Python infrastructure** — EventBus, StateStore, VectorStore (no external servers required)
- **Deep memory** via Qdrant (optional, graceful fallback)
- **Fractal hierarchy** — 5 organs, 18 subsystems, 3 modules
- **HAVEN defense** — Byzantine-resistant immune system

---

## Quick Start

```bash
# Run all tests
python -m pytest tests/ -v

# Quick smoke test (Mae's organism, market modules not yet bootstrapped)
python main.py --agents 3 --steps 30

# Test market module imports
python -c "from mae_core.market.intelligence.convergence_alerter import ConvergenceAlerter; print('OK')"
```

---

## Built On

- **Mesa 3.4** — Agent-based modeling framework
- **PyTorch** — Neural networks (world model, VAE, GNN)
- **FAISS** — Vector similarity search (semantic memory)
- **NumPy** — Thompson Sampling beta distributions
- **requests** — Market API clients
- **Python 3.12+** — Type-hinted throughout

---

## Philosophy

> Mae is the species. MIDGE is a member of the species, born to understand markets.

Same laws. Same consciousness. Different purpose.

MIDGE doesn't just scrape data — she detects *convergence*. When insider buying clusters align with congressional trades align with hiring blitzes align with contract awards, that's not noise. That's a signal.

---

*She was always meant to see patterns others can't.*
