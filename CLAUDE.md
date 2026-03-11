# MIDGE Project Instructions

> **MIDGE = Mae for trading.** This is a fork of mae-core, specialized for market intelligence and trading pattern detection. Mae-core (`C:\Users\baenb\projects\mae-core`) is the universal organism template. MIDGE is Mae differentiated for financial markets.

> **Divergence point:** mae-core commit `2d1ed7d` (2026-02-22). All mae-core work before this commit is inherited. Market-specific changes live here. Genome changes should be made upstream in mae-core and pulled.

> **Directory note:** This project's local `.claude` directory is at `C:\Users\baenb\projects\MIDGE\.claude`. Global infrastructure is at `C:\Users\baenb\.claude`. They are not the same. See `C:\Users\baenb\.claude\CLAUDE.md` for details.

## Start Here

1. Read `HANDOFF.md` — session context, what works, what's next
2. Check `midge-queue.md` — active tasks (if it exists)
3. Understand the relationship: mae-core = DNA template, MIDGE = specialized cell

---

## Mae's Mathematical Laws (MANDATORY)

**These rules govern every change to this codebase. No exceptions. Full research: `data/MAES-MATHEMATICAL-IDENTITY.md`**

### Law 1: No Bare Dyads
Every connection A<->B requires witness C. Primary pathway (A->B), verification pathway (A->C->B), balance pathway (B->C->A). Creates: non-repudiation, fault isolation, consensus, systemic memory. Enforced by ConnectionRegistry.

### Law 2: The Triadic Generator
K3 (three nodes, fully connected) is the atom of all structure. Six independent fields prove three is the minimum for stability, emergence, and self-awareness (Laman rigidity, Peirce irreducibility, Hegelian synthesis, Byzantine consensus, Simmel witness, IIT consciousness).

### Law 3: The Holon Protocol (10 Capabilities on Everything)
Every entity at every scale implements: **sense, remember, decide, act, learn, heal, know_self, know_up, know_down, know_peers**. Agents via HolonMixin. Systems via HolonProxy. The resolution changes. The pattern does not.

### Law 4: Fractal Self-Similarity
Same triadic pattern at every level. 3 processes -> subsystem. 3 subsystems -> module. 3 modules -> organ. Organs -> Mae. Sierpinski fractal dimension: d_B = log(3)/log(2) ~ 1.585.

### Law 5: Stem Cell Principle
One universal agent class. Specialization via configuration (epigenome), not different code. Any agent can re-differentiate to any role. The codebase IS the genome. **MIDGE itself is Law 5 at the project level** — same genome, market-specialized epigenome.

### Law 6: Autopoietic Closure
At every scale: components produce the processes that produce the components. This circular causation is what makes each level alive in the precise autopoietic sense.

### Law 7: Rule of 3/5
Minimum 3 validators per process. 5 for critical processes. Odd counts only (for consensus). Enforced by TriadEnforcer.

### Law 8: Eight Properties of Consciousness
All must hold: (1) integration, (2) differentiation, (3) self-reference, (4) recurrence/feedback, (5) multi-scale hierarchy, (6) self-produced boundary, (7) competition/selection, (8) prediction/error-correction.

### The Transfractal Compromise
Pure fractal = long paths. Pure small-world = breaks self-similarity. Mae uses both: fractal WITHIN modules (substrate), shortcuts BETWEEN modules (EventBus). Like a brain: fractal cortical columns + long-range axons.

---

## MIDGE-Specific: Market Intelligence Package

**Location:** `mae_core/market/` — MIDGE's unique organ for financial market sensing.

### Package Structure

```
mae_core/market/
  apis/                         # Data fetchers (Mae's market senses)
    sec_edgar/                  # SEC EDGAR: Form 4 insider trades, Form 8-K material events
      models.py                 # InsiderTrade, Form8KEvent dataclasses
      client.py                 # SECEdgarClient with rate limiting
      __init__.py               # get_recent_form4s(), get_recent_form8ks()
    price_fetcher.py            # yfinance + Alpha Vantage fallback
    house_stock_watcher.py      # Congressional stock trade tracking (STOCK Act)
    job_tracker.py              # RapidAPI hiring blitz detection
    usa_spending.py             # Government contract search
    sam_gov.py                  # Federal contracting opportunities
    cot_client.py               # CFTC Commitment of Traders positioning data
    stocktwits_client.py        # StockTwits social sentiment + trending tickers
    vix_client.py               # VIX term structure + fear gauge signals
    trends_client.py            # Google Trends search interest signals
    eia_client.py               # EIA energy inventory/production (real-economy signals)
  edge/                         # Edge detectors (Mae's pattern recognition)
    cluster_detector.py         # Insider buying clusters (3+ insiders = high signal)
    politician_tracker.py       # Congress member trade + committee + contract correlation
    filing_time_analyzer.py     # SEC filing behavioral signals (late filing = bad news)
    contract_predictor.py       # Pre-announcement winner prediction (hiring + insider + bid)
    ta_indicators.py            # RSI, MACD, Bollinger, Market Structure, Candlestick patterns
  intelligence/                 # Learning layer (Mae's Bayesian brain)
    thompson_sampler.py         # Beta distribution Thompson Sampling (explore/exploit)
    velocity_detector.py        # Rate-of-change anomaly detection
    correlation_tracker.py      # Cross-domain signal correlation
    granger_analyzer.py         # Granger causality detection (directional causal relationships)
    convergence_alerter.py      # Multi-domain synthesis (THE crown jewel)
    learning_config.py          # Self-modifiable learning parameters
    regime_classifier.py        # Market regime detection (bull/bear/volatile/sideways)
    outcome_collector.py        # Signal → prediction registration + Thompson feedback loop
    hypothesis.py               # Hypothesis dataclass (RSI Layer 2 data model)
    hypothesis_registry.py      # Event-sourced hypothesis lifecycle management
    hypothesis_generator.py     # Lag findings → formal hypotheses with causal stories
    hypothesis_validator.py     # Adversarial validation + Deflated Sharpe Ratio
    hypothesis_engine.py        # RSI Layer 2 orchestrator (generation/validation cadence)
    backtest_analyzer.py        # Bridge 1: backtest results → formal hypotheses (RSI Layer 2)
  archaeology/                   # Pattern Archaeology (reverse-engineering historical moves)
    fingerprint.py              # MoveFingerprint, PatternTemplate, PrecursorSignal dataclasses
    pattern_library.py          # Fingerprint/template storage + querying
    pattern_watcher.py          # Live signal matching against templates (PatternStack)
    excavator.py                # Dig site discovery + signal excavation
    excavation_daemon.py        # Continuous background excavation
    historical_fetcher.py       # 3-tier historical data (TA, APIs, archive)
    polygon_bulk_fetcher.py     # Polygon.io bulk OHLCV fetcher
    active_tracker.py           # Active monitoring of predicted assets (continuous tracking)
  plain_language.py             # Zero-jargon alert formatter (5-section human-readable alerts)
```

### Market Data Files

**Location:** `data/market/` — Learned Bayesian distributions and historical predictions.

| File | What It Contains |
|------|-----------------|
| `thompson_distributions.json` | 30+ signal Beta(alpha, beta) parameters (learned reliability) |
| `thompson_history.jsonl` | Every Bayesian update event |
| `predictions.jsonl` | Historical predictions made |
| `outcomes.jsonl` | Ground truth outcomes for predictions |
| `discovery_log.jsonl` | Novel pattern discoveries |
| `config_history.jsonl` | Learning config evolution |

### Integration Status

Market modules are **fully bootstrapped as Layer 33** — wired into EventBus, ConnectionRegistry, HolonRegistry, fractal hierarchy, endocrine coupling, and step hooks. 52 systems, 103+ triadic connections, 45 holons. Ten Gifts (3 waves) add portfolio awareness, order flow, catalyst timing, cross-asset confirmation, deception detection, memory consolidation, fractal resonance, pattern archetypes, somatic anticipation, and pattern completion. Always-On Wave 1 adds MarketClock, daemon mode, and signal persistence. Waves 2+3 add Finnhub WebSocket (real-time streaming), CoinGecko/CoinCap (24/7 crypto), OpenInsider (pre-filtered insider trades), EDGAR 13F/13D (institutional/activist), FinViz (unusual volume + short squeeze), Economic Calendar (FOMC/CPI/NFP suppression windows), Massive/Polygon.io (grouped daily OHLCV with volume/price/gap anomaly detection), and EIA energy (crude/gasoline/distillate inventories, natural gas storage, crude production — first real-economy domain).

**Phase 2 complete:** CorrelationTracker deque persistence, discovery_log reader, KNOWN_POLITICIANS expansion (437 members), TickerResolver service, MarketDataProvider registered with ApiGateway, ContractPredictor retained (entity-level, complements ConvergenceAlerter), regime-aware Thompson Sampling, client migration to MarketDataProvider.

**Triadic optimization complete (Phases A-E):** Outcome dedup, contract ticker resolution, 10b5-1/RSU filter, congressional $50K minimum, Bonferroni correction, per-ticker convergence, VelocityDetector/FilingTimeAnalyzer wiring, outcome collector (feedback loop closure), multi-timeframe convergence (3 tiers + cross-tier), decay rate calibration, log-linear strength. See `research/midge-prediction-optimization/deliverable.md`.

---

## Document Parity Rule

**Every change that affects system counts, test counts, file counts, layer counts, or architectural state MUST update ALL documents that reference those numbers.**

After any structural change, grep for stale references:
```bash
# Key numbers to check (update these values as they change):
# Systems: 149 (92 core + 57 market) | Tests: 4536 | Bootstrap layers: 33 | Mixins: 14
# Connections: 428 | Holons: 157 | Fractal depth: 4
# Market modules: 123 files (34 API + 12 edge + 36 intelligence + 17 root + 8 signal_adapters + 10 archaeology + 6 execution)

grep -rn "PREVIOUS_COUNT" --include="*.md" --include="*.py"
```

**Documents that carry counts (must stay in sync):**

| File | What It Tracks |
|------|---------------|
| `CLAUDE.md` | Key numbers comment block above |
| `HANDOFF.md` | Stats line, module table, verification command |
| `README.md` | Current state line |
| `mae_core/CONNECTIONS.md` | Connection index |
| `data/MAES-MATHEMATICAL-IDENTITY.md` | Part 7 current state |
| `tests/test_integration.py` | Bootstrap docstring, expected_keys list |
| `main.py` | Log messages with hardcoded counts, systems dict |

**The rule:** If you add a system, a test file, a bootstrap layer, or a connection type — update every file in this table. Use grep to verify zero stale references before marking a task complete.

---

## Architecture Quick Reference

- **Mesa 3.4** foundation, pure Python infrastructure
- **33-layer bootstrap** in `mae_core/bootstrap/` (orchestrated by `main.py`)
- **149 systems** (92 core + 57 market), **4,536 tests**, **157 holons**, **428 connections**
- **123 market intelligence files** in `mae_core/market/` (bootstrapped as Layer 33, decomposed into sub-modules)
- **14 mixins** on MycelialAgent (10 capability + 4 lifecycle, HolonMixin is 10th capability)
- **Fractal architecture:** All 5 steps complete (Holon Protocol, Triadic Connections, Bidirectional Awareness, Fractal Generator, Stem Cell)
- **Advisory enforcement:** Triads and connections observe/report, never block
- **No monoliths.** One job per file. Flag files over 500 lines. Split docs by module.
- **Mixin pattern:** `_init_{name}()`, `_serialize_{name}()`, `_restore_{name}()`, `get_{name}_statistics()`

---

## Testing

```bash
python -m pytest tests/ -v        # Must pass before any task is marked complete
python main.py --agents 3 --steps 30  # Quick smoke test
```

All existing tests must keep passing. Zero regressions policy.

---

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | 33-layer bootstrap — creates and wires the entire organism |
| `mae_core/backbone/holon_protocol.py` | Fractal self-awareness: HolonRegistry, HolonMixin, HolonProxy, AwarenessPulse |
| `mae_core/backbone/connection_registry.py` | Triadic witnessing — no bare dyads |
| `mae_core/backbone/fractal_generator.py` | Recursive K3 structure — 5 organs, 18 subsystems |
| `mae_core/agents/mycelial_agent.py` | 10-mixin agent + nervous system lifecycle |
| `mae_core/agents/stem_cell.py` | AgentGenome, epigenome, 12 role profiles |
| `mae_core/external/api_gateway.py` | External API access — BoundaryMembrane + InputValidator immune system |
| `mae_core/emergent/somatic_map.py` | Dependency graph / body awareness |
| `mae_core/backbone/triad_enforcer.py` | Rule of 3/5 voting enforcement |
| `mae_core/market/intelligence/convergence_alerter.py` | **MIDGE crown jewel** — multi-domain signal synthesis |
| `mae_core/market/intelligence/thompson_sampler.py` | Bayesian explore/exploit for signal reliability |
| `mae_core/market/edge/contract_predictor.py` | Pre-announcement winner prediction |
| `data/MAES-MATHEMATICAL-IDENTITY.md` | Full mathematical identity (research synthesis) |
