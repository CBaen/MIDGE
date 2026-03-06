# Research Brief: Competitive Edge — How MIDGE Beats Enterprise Trading AI

## Date: 2026-03-05
## Project: MIDGE (Mae for Trading)

### Problem Statement

MIDGE has a proven statistical edge (z=4.74, p<0.0001, 19.9% win rate vs 9% random baseline) through multi-domain pattern stacking. But enterprise trading firms have unlimited compute, co-located servers, and armies of quant analysts. We can't compete on speed or raw compute. We need to compete on *information synthesis* — seeing connections across domains that siloed systems miss.

Guiding Light's words: "We need to become the insider traders based on all of these pattern matching factors." MIDGE must ingest more diverse data (weather, logistics, legislation, social sentiment), process it faster with limited hardware, and stack patterns in ways no existing software does. The 19.9% win rate needs to climb toward financially meaningful returns.

### Expected Outcome

MIDGE becomes a system that:
1. Ingests data from more diverse domains than any single competitor (weather, shipping, legislation, social media, satellite proxies — not just financial data)
2. Processes and cross-correlates this data efficiently on a single Windows 11 machine (i9/64GB RAM/RTX capable)
3. Discovers non-obvious multi-domain patterns through stacking that enterprise siloed systems miss
4. Achieves win rates and position sizing that generate real financial returns
5. Does this autonomously — not requiring constant human supervision to find and act on patterns

### Current State

**What works:**
- 28 data sources across 11 domains (insider, macro, technical, events, positioning, government, contracts, sentiment, fundamental, institutional, crypto)
- Convergence engine synthesizes multi-domain signals (min_domains=3, Law 2)
- Thompson Sampling with 50 Bayesian distributions learns signal reliability
- Pattern Archaeology excavates historical moves and abstracts into domain-level templates
- PatternWatcher matches live signals against templates, detects stacking
- Dual confirmation (convergence + archaeology) for highest-confidence signals
- Hypothesis loop (RSI Layers 1-3) for self-improving pattern discovery
- Active tracking with MFE/MAE for grading predictions
- 146 systems, 4,384 tests, 33-layer bootstrap

**What doesn't work well enough:**
- 6 agents processing 28+ sources is a bottleneck — data review lags behind ingestion
- Excavation takes 9+ hours for 3,237 symbols (single-threaded, Polygon API)
- Domain coverage gaps: no weather, no shipping/logistics, no legislative tracking, no broad social media (only StockTwits), no satellite/alternative data
- 19.9% win rate is statistically significant but not yet profitable at scale
- No options flow data (Unusual Whales API identified but not integrated)
- Daemon runs on old code (must restart to pick up changes)

**Hardware:** Windows 11 Pro, likely i9 or equivalent, 64GB RAM, Python 3.14, Docker/Qdrant/Ollama available. Single machine — no cloud cluster.

### Project Direction

MIDGE is a recursive self-improving pattern discovery organism, not a signal grader. The core loop: discover pattern -> backtest against archive -> register hypothesis -> evaluate against reality -> adjust weights -> discover BETTER patterns -> improve HOW she discovers. The 95%+ hit rate comes from STACKING 4-5 independent domains, not from any single pattern hitting 95%.

MIDGE follows 8 Mathematical Laws (triadic architecture, fractal self-similarity, stem cell principle, etc.). All changes must respect these laws — particularly Law 1 (no bare dyads), Law 2 (triadic generator), and Law 5 (stem cell principle — specialization via configuration, not different code).

### Constraints

- **Single machine** — no cloud compute, no GPU clusters. Must work on Wardenclyffe (Win11 desktop).
- **Mesa 3.4 agent framework** — MIDGE is built on Mesa. Core architecture is settled.
- **8 Mathematical Laws** — all changes must comply. No bare dyads, triadic connections, fractal self-similarity.
- **Python 3.14** — no other language runtime.
- **Budget-conscious API costs** — free/cheap data sources preferred. Polygon.io Starter ($29/mo) is current ceiling per source.
- **Advisory enforcement** — triads observe/report, never block. System must be resilient, not rigid.
- **Zero regressions** — 4,384 tests must keep passing.
- **Pattern stacking is the goal** — not individual signal accuracy. Domain independence is what creates the edge.

### Destructive Boundaries

- Do NOT suggest replacing Mesa agent framework
- Do NOT suggest cloud migration or distributed computing
- Do NOT suggest replacing the Mathematical Laws or triadic architecture
- Do NOT suggest replacing Thompson Sampling with a different Bayesian approach (it's proven)
- Do NOT suggest approaches requiring >$100/mo in API costs per source without flagging the cost
- Do NOT suggest GPU-dependent ML models as primary strategy (can be supplementary)

### Research Angles

**Team 1 — Competitive Landscape:** What do Kensho, Numerai, Alpaca, QuantConnect, Two Sigma's Venn, and retail AI trading tools actually do? What data sources do they use? What's their architecture? Where are the gaps MIDGE can exploit? Focus on what makes them successful AND what they're missing.

**Team 2 — Alternative Data Sources:** Weather APIs (NOAA, Open-Meteo), shipping/AIS data (MarineTraffic, VesselFinder), legislative trackers (Congress.gov API, GovTrack), social media sentiment (Reddit, Twitter/X, Discord, Telegram), satellite proxies (parking lot counts, crop monitoring). What's free or cheap? What's the signal-to-noise? How do they map to MIDGE's 11-domain model? What new domains would they create?

**Team 3 — Processing Architecture:** How do single-machine trading systems maximize throughput? Worker pools vs agent scaling, streaming vs batch processing, memory-mapped data, concurrent fetching, pipeline parallelism. What can we do with a single i9/64GB machine? Should MIDGE have companion processes or a sibling system? How do we get from 6 agents to processing capacity that matches our data volume?

**Team 4 — Pattern Discovery Methods:** How do the best quant systems find non-obvious cross-domain patterns? Granger causality, transfer entropy, mutual information, graph-based methods, ensemble approaches. What's implementable without a GPU cluster? What methods specifically excel at finding *cross-domain* correlations that siloed systems miss? How do retail quants punch above their weight?

**Team 5 — Win Rate Optimization:** What separates a 20% win rate from a 50-60% win rate in multi-factor systems? Position sizing strategies, regime filtering, confidence calibration, Kelly criterion, risk-of-ruin analysis. How do profitable retail systems manage risk? What's the minimum win rate for profitability given proper position sizing? How do we turn MIDGE's proven edge into actual financial return?

### Team Size: 5

Five teams because: (1) the angles are genuinely independent — competitive landscape, data sources, architecture, algorithms, and risk management each require different expertise, (2) the scope is broad — this is a major architectural direction decision, and (3) the angles don't overlap significantly (Team 1 maps the landscape, Teams 2-5 explore specific dimensions of competitive advantage).

### Failed Approaches

- **Per-source accuracy tuning** — Guiding Light corrected this twice. MIDGE is not a signal grader. Pattern stacking across independent domains is the edge, not tuning individual signal accuracy.
- **Single-domain optimization** — Focusing on one data type (e.g., insider trades alone) produces mediocre results. The power is in cross-domain convergence.
- **High-frequency approaches** — We can't compete on execution speed. Our edge is information asymmetry, not latency.
