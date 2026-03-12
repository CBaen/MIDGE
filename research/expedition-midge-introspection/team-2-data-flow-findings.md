# Team 2 Findings: Data Flow Topology
## Date: 2026-03-12
## Researcher: Team Member 2

---

## Battle-Tested Approaches

### Signal Ingestion → Convergence Pipeline (Path A) — FULLY WIRED

The primary data path is complete and battle-tested:

```
API Client (ThreadPoolExecutor, 12 concurrent)
  → _fetch_source() [sensing_reactive.py]
    → enrich_signal() [velocity, filing-time, Ollama 8-K sentiment]
  → _collect_one() [sensing_collector.py]
    → convergence_alerter.record_signal()   [global alerter]
    → tier_alerter.record_signal()           [tactical/strategic/thematic]
    → deception_detector.record_signal()     [Gift 5]
    → somatic_anticipation.record_signal()   [Gift 9]
    → pattern_completion_engine.check_completions()  [Gift 10]
    → absence_monitor.record_arrival()
    → correlation_tracker.record()
    → EventBus.publish(CH_SIGNAL_INGESTED)
    → store_signals() → Qdrant + JSONL archive
    → outcome_collector.register_signals()
    → _trigger_reactive_convergence()        [immediate check, no waiting]
    → PatternWatcher.check()                 [reactive, same step]
```

Every signal touches 10+ systems before the step ends. The reactive convergence check (firing immediately on ingestion rather than waiting for the step tick) is a key optimization.

### Convergence → Paper Trade Pipeline (Path B) — FULLY WIRED

```
ConvergenceAlerter.check_convergence()
  → ConvergenceAlert [with ripple_effects, domain_sequence, sequence_score]
  → EventBus.publish(CH_CONVERGENCE)
  → _sensing_step_with_advisory() [market_hooks_sensing.py]
    → _run_paper_trading_gate()
      → Law 7 validation (3 validators: convergence + pattern_stack + inevitability/hypothesis)
      → combo Thompson filter (combo WR < 25% → blocked)
      → drawdown circuit breaker check
      → behavioral anomaly check
      → _write_paper_trade() → data/midge/paper_trades.jsonl
      → _translate_and_log_executable_signal() → data/midge/executable_signals.jsonl
        → _submit_to_alpaca() [US equity only, position check, bracket order]
```

### Thompson Learning Path (Path C) — WIRED BUT SLOW

```
Paper trade → OutcomeCollector.register_signals()
  → OutcomeTracker.record_prediction() [with ticker + window]
  → [N days pass, evaluated every 200 steps]
  → OutcomeCollector.evaluate()
  → OutcomeTracker.check_pending_outcomes()
  → price_fetcher.get_current_price()
  → Thompson.update(source, success=True/False)  [writes thompson_distributions.json]
  → _on_outcome_graded() → PatternLibrary.update_outcome(template_id, won=)

PostMortemReviewer [every 500 steps]
  → read outcomes.jsonl
  → compute combo_stats + sequence_stats
  → Thompson.update(combo_key + seq_key) [sequence-aware updates]
  → write post_mortem_insights.json
```

Thompson is updated from three places: OutcomeTracker (individual signals), PostMortemReviewer (combo/sequence patterns), and ThompsonCalibrator (every 1000 steps, archive-based calibration).

### JSONL Signal Archive (Path I-partial) — STRONG WRITE, WEAK READ

Write side: Every signal goes to `data/midge/signals/YYYY-MM-DD.jsonl` via `store_signals()`.

Read side:
- `SignalArchiveReader` is the sole reader; used by `LagCorrelationAnalyzer` and `DeepAnalyst`
- `LagCorrelationAnalyzer` reads to build time series for cross-correlation (every 500 steps)
- `DeepAnalyst` reads to score inevitability candidates (every 500 steps)
- `OutcomeCollector.collect_from_archives()` can retroactively register old signals (CLI tool, not run on cadence)

---

## Novel Approaches

### RawStore → RawDataAnalyst → Convergence Injection (Path H, NOVEL)

The raw store was built for 24 clients but was, until recently, a write-only black hole. `RawDataAnalyst` is the first systematic reader:

```
SQLite raw stores [data/market/raw/*.db]
  → RawDataAnalyst.analyze() [every 100 steps]
    → _analyze_insider_price_context()   [insider + 52wk position amplifier]
    → _analyze_fred_macro_regime()        [yield inversion + CPI acceleration]
    → _analyze_cross_domain_preconvergence() [insider + trends + headlines → pre-alert]
    → _analyze_funding_rate_squeeze()     [Binance consecutive negative funding]
  → Synthetic MarketSignal objects
  → convergence_alerter.record_signal()  [injected back into main pipeline]
```

This is the only path that reads raw SQLite data back into the live pipeline. Four routines cover ~10% of what's stored.

### Cascade Path (Path F) — COMPLETE FEEDBACK LOOP

```
WorldModel [static causal graph, 114 nodes, 102 edges]
  → ConvergenceAlert.ripple_effects [attached during check_convergence()]
  → EventBus.publish(CH_CONVERGENCE) → _on_convergence_register_cascade()
  → CascadeTracker.register_cascade()
  → [subsequent signals arrive on predicted tickers]
  → CH_SIGNAL_INGESTED → _on_signal_cascade_check()
  → CascadeTracker.check_signal()
  → WorldModel.record_outcome(was_correct=True)  [strengthens causal edge]
  → EventBus.publish(CH_CASCADE_CONFIRMED)
  → _on_cascade_confirmed() → alerter.record_signal(domain="cascade")  [forward boost]
  → [chain ages past 30 days]
  → expire_stale() → WorldModel.record_outcome(was_correct=False)  [weakens edges]
```

This is one of MIDGE's most complete feedback loops: WorldModel edges are strengthened on confirmation and weakened on failure. The cascade boost re-enters the convergence pipeline with synthetic signals for remaining predicted dominoes.

### Qdrant Write Path (Path I) — WRITE ONLY IN PRACTICE

```
ConvergenceAlert fires
  → PatternMemory.remember_convergence_alert()  [market_hooks_steps_core.py]
  → EventEmbedder.embed_convergence_alert()
  → Ollama mxbai-embed-large (1024-dim vector)
  → Qdrant upsert [collection: midge_events]

DeepAnalyst.analyze() [every 500 steps]
  → PatternMemory.remember_inevitability() for top 5
  → Qdrant upsert [collection: midge_events]
```

**Qdrant reads** are available via `PatternMemory.recall_similar()`, `find_precedents()`, `get_pattern_context()`, and `search_by_ticker()`. However, searching the code, these read methods are ONLY called from:
1. `OctopusAgent._recall_similar()` — for task context during investigation
2. Test files

No production pipeline (convergence, paper trading, law 7 validation, DeepAnalyst, hypothesis engine) actively queries Qdrant before making decisions. Qdrant is effectively a write-only archive in production.

---

## Emerging Approaches

### Hypothesis Path (Path E) — RECEIVES INPUT, OUTPUT REACHES FOCUSED ATTENTION

```
LagCorrelationAnalyzer [every 500 steps]
  → lag_correlations.json
  → convergence_alerter.set_lag_findings() [temporal domain ordering]
  → WorldModel.add_discovered_edge() [auto-discovery of causal relationships]

HypothesisGenerator [every 500 steps, from lag findings]
  → Hypothesis objects → HypothesisRegistry
  → HypothesisValidator [every 1000 steps, adversarial DSR testing]
  → promote/retire lifecycle
  → CH_HYPOTHESIS_PROMOTED / CH_HYPOTHESIS_RETIRED → endocrine coupling
  → CH_HYPOTHESIS_FIRED → _on_hypothesis_fired()
    → sensing_hook._priority_requests boost [focused attention]
    → ctx._recent_hypothesis_fires[ticker:direction] = now

Paper trade gate [Validator 4]
  → reads ctx._recent_hypothesis_fires (2-hour window)
```

Hypothesis output feeds focused attention (priority fetch scheduling) and provides one validator vote in the Law 7 gate. However, hypothesis outcomes — whether a promoted hypothesis's predicted move materialized — are not tracked back to the HypothesisRegistry. Active hypotheses are never told "you were right" or "you were wrong" via the outcome system.

### Post-Mortem Path (Path G) — FEEDS THOMPSON, NOTHING READS INSIGHTS LIVE

```
outcomes.jsonl → PostMortemReviewer.review() [every 500 steps]
  → compute combo_stats, sequence_stats, timing_accuracy, regime_failures, mfe_mae_patterns
  → _feed_thompson_updates() [pushes combo and seq keys to Thompson]
  → write post_mortem_insights.json [atomic write]

DeepAnalyst._load_combo_stats() [every 500 steps, same cadence]
  → reads post_mortem_insights.json
  → _combo_boost() multiplier (0.8–1.25) applied to inevitability scoring
```

Post-mortem insights reach DeepAnalyst but ONLY via file I/O at the same cadence as their creation. The insights are not fed into:
- ConvergenceAlerter confidence weighting
- ThompsonCalibrator domain ordering
- PatternWatcher template selection
- HypothesisEngine generation priorities

The timing accuracy data (early/on_time/late/missed buckets) and regime failure rates are computed but consumed by nothing except the log and the file on disk.

### Bio-System Paths (Path J) — RECEIVE MARKET DATA, PRODUCE NO MARKET OUTPUT

Bio systems receive market event data through EventBus subscriptions:
- `CH_CONVERGENCE` → EndocrineSystem (dopamine for bullish, adrenaline for bearish, cortisol for retired hypotheses)
- `CH_HYPOTHESIS_PROMOTED` → EndocrineSystem dopamine
- `market.intel.drift_detected` → published when DriftDetector fires
- `market.intel.kelly_sizing` → stored on `ctx._latest_kelly`

The endocrine system receives convergence events and releases hormones. But hormones affect **agent behavior** (exploration rate, energy, aggression), not the market intelligence pipeline. There is no path from endocrine state back into: Thompson weighting, convergence thresholds, signal prioritization, or paper trade sizing. The organism "feels" market pressure but this feeling does not modulate its intelligence decisions.

DriftDetector detects regime changes in price/VIX/sentiment/volume streams. When drift is detected, it publishes `market.intel.drift_detected`. Nothing subscribes to this channel to act on it. The RegimeClassifier is not woken up. Thompson's decay rate is not adjusted.

---

## Gaps and Unknowns

### Dead End 1: Raw Store → 95% Unread
**Value if completed: HIGH**

The raw store captures full API responses across 24 clients. RawDataAnalyst reads 4 cross-domain patterns from it. The remaining stored data is never read back:
- StockTwits raw messages (social_text_analyzer reads from raw_store for discovered keywords — this IS read, but only for keyword harvesting, not sentiment analysis)
- Finnhub news raw text (beyond the processed headline fields)
- EDGAR 13F/13D filing details
- Congressional trade metadata (committee membership, amounts, party affiliation)
- SAM.gov contract opportunity details
- COT managed money positions (only net positioning is used)
- Price snapshots with full yfinance field set (80+ fields, only ~5 used)
- EIA historical inventory draws/builds (only current week used)
- FRED observations older than what the macro regime detector uses

None of these have a reader that extracts patterns or feeds the convergence engine.

### Dead End 2: Qdrant → Not Queried Before Decisions
**Value if completed: HIGH**

PatternMemory stores convergence alerts and inevitabilities in Qdrant with semantic embeddings. The read API (`recall_similar`, `find_precedents`, `get_pattern_context`) is complete and tested. But no decision-making system queries it before acting:
- The paper trade gate does not check "have we seen this exact setup fail before?"
- The convergence alerter does not adjust confidence based on historical similar alerts
- The hypothesis engine does not seed from Qdrant when generating hypotheses
- The DeepAnalyst reads JSONL archives but not Qdrant — missing the semantic dimension

Qdrant has ~4 months of embedded market intelligence. It is a searched-but-never-consulted oracle.

### Dead End 3: Post-Mortem Timing Data → Not Used for Outcome Window Calibration
**Value if completed: MEDIUM**

PostMortemReviewer computes timing accuracy buckets (early/on_time/late/missed). If "late" rate is high for a particular combo, the outcome window for that combo should be extended. Currently, outcome windows in `OUTCOME_WINDOWS` are static per signal type. The post-mortem's timing findings are written to a file and read by nothing that adjusts windows.

### Dead End 4: Hypothesis Outcome Tracking → Not Closed
**Value if completed: HIGH**

Hypotheses have a full lifecycle (probation → active → hibernated → retired) but there is no mechanism to tell a hypothesis "your predicted relationship occurred" or "it didn't occur." The Thompson feedback loop tracks signal-level outcomes and combo-level outcomes. There is no `hypothesis_id` outcome window — we cannot determine if a specific causal hypothesis was validated by subsequent price data. HypothesisEngine `_hypotheses_promoted` counter increments but there is no corresponding `_hypotheses_validated_by_market` counter.

### Dead End 5: Regime-Aware Thompson — Not Applied to Combo Keys
**Value if completed: MEDIUM**

`ThompsonSampler.regime_aware_forget()` applies regime-specific decay rates (volatile=0.90, bull=0.95). But combo keys (e.g. `combo:events+macro+price`) and sequence keys (e.g. `seq:insider>>macro>>technical`) are stored in the same distributions dict. When forgetting runs, all keys decay uniformly. High-frequency combo updates during a bull regime will be over-decayed in a volatile regime. The regime-aware decay applies to individual signal sources but not to the combo/sequence layer where the strongest learning signal lives.

### Dead End 6: WorldModel Edge Discovery → Not Fed Back to Hypothesis Generator
**Value if completed: MEDIUM**

LagCorrelationAnalyzer and GrangerAnalyzer run every 500 steps and auto-discover edges in the WorldModel. These discovered edges represent empirically validated causal leads. But the HypothesisGenerator reads from `lag_correlations.json` directly — not from the WorldModel. If GrangerAnalyzer discovers that `eia_energy` Granger-causes `XOM` with a 3-day lag, this adds a WorldModel edge but does NOT automatically generate a hypothesis that "eia_energy bullish → XOM bullish in 3 days." The two discovery systems (lag/Granger → WorldModel and lag → hypotheses) are partially disconnected.

### Dead End 7: Alpaca Position Data → Never Read Back into MIDGE
**Value if completed: MEDIUM**

When `_submit_to_alpaca()` places a paper trade, it checks existing positions to avoid duplication. But Alpaca positions are never read back to inform the convergence engine or Thompson. If AAPL is up 8% in an Alpaca paper position, that outcome data is not registered with OutcomeTracker. The position P&L from Alpaca is invisible to MIDGE's learning engine — we can place paper trades but we never learn from whether those specific Alpaca trades made money.

### Dead End 8: DriftDetector Output → Nothing Subscribes
**Value if completed: MEDIUM**

DriftDetector publishes `market.intel.drift_detected` when it detects concept drift in price/VIX/sentiment. No system subscribes. The intended reaction (re-run RegimeClassifier, trigger faster Thompson forgetting) is not wired. The drift detection is a sensor with no actuator.

### Dead End 9: Tiered Alerters (Tactical/Strategic/Thematic) → Advisory Only
**Value if completed: LOW-MEDIUM**

Three tier alerters run in parallel with the global convergence alerter. They fire into `ctx._market_advisory[tier_name]` which is read by the convergence heartbeat writer (written to `convergence_state.json`). But tier alerts never reach paper trading, Qdrant embedding, or outcome tracking. The tactical alerter may fire on shorter-timeframe signals that could support faster trade entries, but they are not wired into the Law 7 gate or execution pipeline.

### Dead End 10: DeepAnalyst Inevitabilities → Not Seen by Convergence Alerter
**Value if completed: HIGH**

`DeepAnalyst.analyze()` runs every 500 steps and produces a ranked list of inevitabilities. These are:
- Stored to `ctx.inevitabilities` (used by Law 7 paper trade gate — Validator 3)
- Stored to `data/midge/inevitabilities.jsonl`
- Embedded in Qdrant
- Registered with OutcomeCollector

They are NOT:
- Fed into ConvergenceAlerter as synthetic signals
- Used to adjust Thompson weights for sources contributing to the top inevitabilities
- Used to boost focused attention priority requests
- Compared against incoming signals to detect when an inevitability is "materializing"

DeepAnalyst produces the best synthesis in the system, but its output is consumed only at trade execution time, not upstream in the sensing or convergence layers.

---

## Synthesis

### The Architecture Bias: Strong Write, Weak Read-Back

MIDGE has an excellent ingestion pipeline and multiple storage layers, but a fundamental asymmetry: data flows much more easily into storage than out. The evidence:

- **JSONL signal archive** (900+ files, 414 days): read by 2 systems (LagCorrelationAnalyzer, DeepAnalyst)
- **SQLite raw store** (24 domains, full API responses): read by 1 system (RawDataAnalyst, 4 routines)
- **Qdrant semantic store** (convergence alerts, inevitabilities, templates): read by 1 system in production (OctopusAgent investigation)
- **post_mortem_insights.json**: read by 1 system (DeepAnalyst _load_combo_stats)
- **Thompson distributions**: the most-read store — 5+ systems query it

### The Two-Pipeline Disconnection (from Evolution Blueprint)

The Evolution Blueprint noted two disconnected pipelines. This is confirmed:

**Pipeline 1 (Core organism attention):** AttentionalGate → GlobalWorkspace → PatternCortex
**Pipeline 2 (Market intelligence):** SensingHook → ConvergenceAlerter → PatternWatcher

These remain disconnected at the **decision-making** level. Pipeline 2 influences Pipeline 1 through endocrine coupling (convergence → dopamine → agent exploration). But Pipeline 1's attention machinery (what the organism collectively "notices") does not feed back into Pipeline 2's signal prioritization.

### The Three Most Valuable Completions

**1. Qdrant Read-Back into Paper Trade Gate (HIGH)**
Before approving a paper trade, query Qdrant: "Show me the last 5 times this exact domain combination fired on this ticker. What was the outcome?" This is 10 lines of code using `PatternMemory.get_pattern_context()` already built. The information exists; it is just not consulted.

**2. Hypothesis Outcome Tracking (HIGH)**
Register each promoted hypothesis as a prediction with OutcomeCollector. When a hypothesis fires (`CH_HYPOTHESIS_FIRED`), record the expected move direction and window. When the outcome resolves, feed back to HypothesisRegistry. This closes the only major feedback loop that is currently open. Without this, MIDGE cannot learn which causal relationships it discovers are actually predictive.

**3. Alpaca Position Outcome Registration (MEDIUM)**
Every 200 steps (same cadence as outcome_collector.evaluate()), query Alpaca for closed positions and register their P&L as outcomes with OutcomeCollector. This makes live paper trading results feed the same Thompson distributions that drive future trade selection.

### Missing Temporal Data Path

Across all traced paths, there is no system that tracks **when** data reaches the pipeline versus **when** the event occurred. For example:
- Congressional trades are disclosed ~30-45 days after the trade date
- EDGAR 13F filings are disclosed 45 days after quarter end
- The signal's `timestamp` field is populated from the trade date, not the disclosure date

This means the convergence engine sees "congressional trade" signals dated months ago alongside "FRED macro" signals from today. The `domain_sequence` field sorts by timestamp — so old insider trades may appear to "lead" macro signals that are actually contemporaneous. The temporal ordering of signals across the pipeline is based on event dates, not information-arrival dates, which inflates apparent leading-indicator relationships.

### The One-Way Street: Execution → MIDGE

All execution paths are one-way:
- `paper_trades.jsonl` is written, never read to adjust thresholds dynamically
- `executable_signals.jsonl` is written, never read by any system
- `paper_trades_bypass.jsonl` is written, never read back
- Alpaca orders are placed, outcomes are not retrieved

MIDGE outputs trades but does not learn from their specific results at the execution level. The Thompson loop operates on signal-level predictions (registered via `register_signals()`), not execution-level outcomes (registered via actual position closes). There is a gap between "what we predicted" and "what we traded and what happened to the trade."
