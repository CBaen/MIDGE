# Team 5 Findings: Emergent Capability Analysis — What New Powers Arise from Connection?
## Date: 2026-03-12
## Researcher: Team Member 5

---

## Preamble: Research Method

I read every system file in the brief before drawing conclusions. Findings are grounded in exact code behavior, not speculation. I looked for four things in each combination:

1. What each system **can produce** that it currently discards or keeps local
2. What each system **is missing** that another system already has
3. The **interface gap** — what single bridge would close the loop
4. Whether the emergent capability is **qualitatively new** (not just additive)

Systems inspected: `regime_classifier.py`, `drift_detector.py`, `motif_detector.py`, `granger_analyzer.py`, `world_model.py`, `cascade_tracker.py`, `post_mortem.py`, `social_text_analyzer.py`, `somatic_anticipation.py`, `absence_monitor.py`, `pattern_archetypes.py`, `hypothesis_engine.py`, `hypothesis_validator.py`, `deep_analyst.py`, `pattern_memory.py`, `self_monitor.py`, `drawdown_monitor.py`, `velocity_detector.py`, `consolidation_engine.py`, `pattern_watcher.py`, `signal_translator.py`, `market_hooks_steps.py`, `bio_market_wiring.py`.

---

## Synthesis: The Key Discovery

Most of MIDGE's intelligence lives in separate "bubble" systems. Each system produces an output, but that output rarely becomes another system's input in a way that creates **qualitatively new behavior**. The connections that exist are mostly one-directional data feeds. What is almost entirely absent is **closed feedback loops that generate new knowledge**.

The five emergent capabilities below are not marginal improvements. Each one creates a capability that is genuinely absent from MIDGE today.

---

## Capability A: Predictive Regime Forecasting

**The Emergent Thing:** Instead of knowing "what regime are we in now," MIDGE gains the ability to ask "what regime are we entering, and have we seen this transition before?"

### Systems Involved

| System | What It Does Alone |
|---|---|
| `RegimeClassifier` | Labels the current day "bull/bear/volatile/sideways" based on 20-day SPY stats. Caches result daily. Purely descriptive — reactive, not predictive. |
| `DriftDetector` | Detects statistically significant distribution shifts in real-time streams (price_returns, VIX, sentiment, volume) using ADWIN. Currently feeds price_returns drift events to the bus (`market.intel.drift_detected`) but nothing subscribes to that channel with useful action. |
| `MotifDetector` | Per-symbol STUMPY matrix profile. Detects when the current price subsequence matches a historical shape (motif) or is unlike anything seen before (discord). Currently produces `MotifSignal` objects that enter the signal pipeline but have no connection to regime. |

### The Emergent Capability

When these three connect, you get a **Regime Transition Detector**: "The drift detector is seeing a significant shift in price_returns distribution. The motif detector is finding matches to a subsequence that historically preceded volatile regime entry. The regime classifier shows we are currently in bull. Together: we are likely entering a volatile regime within 5-10 days."

None of these systems alone can say this. RegimeClassifier looks backward (20 days). DriftDetector sees a distribution shift but doesn't know what it means for regime. MotifDetector sees a familiar price shape but doesn't connect it to regime transitions.

### How It Works — Information Flow

1. DriftDetector fires a drift event on `market.intel.drift_detected` with `stream="price_returns"`, `old_mean=+0.002`, `new_mean=-0.001`
2. A new bridge component subscribes to this channel and consults MotifDetector: "do we have any symbols showing discord signals right now?" (Discord = anomalous price action unlike anything in history — exactly what regime transitions look like)
3. If MotifDetector shows 3+ symbols with discord signals in the same window as a DriftDetector price_returns shift, the bridge queries PatternMemory: "have we seen this drift + discord combination before?" (Qdrant semantic search over stored motif events)
4. If PatternMemory returns precedents (e.g., Feb 2020, Aug 2022), bridge publishes `market.intel.regime_transition_warning` with estimated type and timeline
5. This channel is consumed by:
   - ThompsonSampler: increases decay rate for current-regime distributions pre-emptively
   - ConvergenceAlerter: adds "regime transition" as a signal domain with negative confidence modifier on regime-sensitive combos
   - SomaticAnticipation: releases CORTISOL when transition warning fires, biasing MIDGE toward caution

### Value to MIDGE

Regime transitions are when the most money is made and lost. MIDGE currently finds out a regime changed AFTER it happened (20-day lookback). Advance warning of 5-10 days would allow preemptive Thompson adjustments, suppression of regime-sensitive signals in transitional periods, and pursuit of transition-specific pattern templates (volatility squeeze → expansion, etc.).

### Effort to Build

**Moderate.** Three pieces:
1. Bridge component (~80 lines) that subscribes to `market.intel.drift_detected` and queries MotifDetector (both already wired in ctx)
2. One new EventBus channel `market.intel.regime_transition_warning`
3. Three subscribers to that channel (ThompsonSampler already has `regime_aware_forget()`, just needs calling; ConvergenceAlerter already accepts additional signals; SomaticAnticipation already accepts signals)

No new algorithms. Just wiring.

### Example Scenario

Date: March 5, 2026. SPY has been in "bull" for 3 weeks. DriftDetector fires a price_returns drift (old_mean=+0.0015, new_mean=-0.0008). MotifDetector simultaneously shows AAPL, MSFT, and QQQ with discord signals (anomalous subsequences not matching any historical pattern). PatternMemory recalls similar conditions from January 2022 (pre-bear transition) and September 2021 (pre-volatile transition). MIDGE publishes a regime transition warning. ThompsonSampler pre-emptively decays bull-regime distributions. Three days later SPY enters "volatile" — MIDGE's confidence engine was already skeptical of bull-biased signals before the data confirmed the shift.

---

## Capability B: Self-Improving Causal Intelligence

**The Emergent Thing:** MIDGE currently discovers causal relationships (Granger), curates them in a graph (WorldModel), and tracks cascades (CascadeTracker). But the learning loop from cascade outcomes back to hypothesis generation is broken. This combination closes that loop — creating a system that **discovers which causal relationships are real by watching them play out, then generates hypotheses specifically about causal sequences**.

### Systems Involved

| System | What It Does Alone |
|---|---|
| `GrangerAnalyzer` | Every 500 steps, finds statistically significant directional causation between signal sources (p<0.05, Bonferroni corrected). Currently adds discovered edges to WorldModel (`add_discovered_edge(evidence="granger")`). Does NOT distinguish between "Granger found this" and "this actually played out in live markets." |
| `WorldModel` | Curated + discovered causal graph (114 nodes, 102 edges). Has `record_outcome(trigger, ticker, was_correct)` which adjusts edge strength. CascadeTracker calls this when cascade links confirm or expire. But WorldModel cannot currently tell HypothesisGenerator "here are the causal sequences with the highest confirmed hit rates." |
| `CascadeTracker` | Tracks active causal chains as dominoes confirm. Knows `energy_ratio` (actual lag vs predicted). Records outcomes to WorldModel. Currently just a tracker — does NOT feed any learning system beyond WorldModel. |
| `PostMortemReviewer` | Analyzes WHY predictions succeed or fail. Looks at domain orderings (sequence stats). Does NOT yet look at causal chain confirmation rates. |
| `HypothesisEngine` | Generates hypotheses from lag findings. Does NOT receive input from WorldModel about which causal chains have high confirmation rates. |

### The Emergent Capability

**Causal Sequence Hypothesis Generation**: MIDGE can now ask "which 2-hop causal chains in WorldModel have the highest live confirmation rate from CascadeTracker?" and automatically generate hypotheses specifically testing those chains. This is fundamentally different from current hypothesis generation, which looks at signal archives for statistical lag patterns without causal understanding.

Example: WorldModel has edge `crude_price_spike → airline_costs_up → DAL bearish` (3-hop chain). CascadeTracker has confirmed the full chain 7 times in 90 days. PostMortemReviewer finds this sequence appears in domain ordering stats with 71% win rate. HypothesisEngine generates: `Hypothesis: EIA crude bearish (cause) → DAL short (effect), expected lag 5 days`. This hypothesis now enters the standard probation → validate → promote lifecycle, but it was GENERATED from causal evidence rather than just statistical correlation.

### How It Works — Information Flow

1. CascadeTracker keeps `get_active_chains()` and tracks confirmed/expired counts per trigger
2. New bridge: `causal_performance_reporter.py` (~100 lines). Every 1000 steps, iterates WorldModel edges where `evidence == "granger"` or `"lag_correlation"`, cross-references CascadeTracker history to get `hit_count / (hit_count + miss_count)` per chain, produces a ranked `CausalChainPerformance` list
3. PostMortemReviewer gains one new analysis: `compute_causal_ordering_stats(causal_chain_performance)` — compares domain orderings in outcomes.jsonl against known WorldModel chains (does sequence `insider → macro → technical` match any WorldModel 3-hop path?)
4. HypothesisGenerator gains one new input: `top_causal_chains` list. When generating, it considers Granger pairs AND confirmed WorldModel chains as trigger candidates. Chains with >60% confirmation rate AND >5 live confirmations become high-priority hypothesis seeds
5. HypothesisValidator gains awareness: when validating a hypothesis that maps to a known WorldModel causal chain, it queries the chain's live confirmation rate and uses it as a prior (reduces min_observations requirement by up to 5 for high-evidence chains)

### Value to MIDGE

Current hypothesis generation is purely statistical. It finds that "when X happens, Y often follows" but doesn't understand WHY. Causal chains that survive live market confirmation are structurally different from spurious correlations — they represent actual economic mechanisms. A hypothesis derived from a 7x-confirmed causal chain should be promoted faster and with greater confidence than one derived from statistical coincidence.

### Effort to Build

**Moderate.** The hardest part is already done: CascadeTracker has the confirmation data, WorldModel has `record_outcome()`, HypothesisGenerator accepts `lag_findings` as input (already used by backtest_analyzer bridge). The bridge component is the missing piece.

### Example Scenario

March 2026: EIA reports a crude oil inventory draw (bearish for crude supply). WorldModel maps `eia_crude_draw → crude_price_spike → airline_cost_increase → DAL/UAL bearish`. CascadeTracker has confirmed 4 of the last 6 such chains reaching DAL within 7 days. PostMortemReviewer finds that `energy → macro → price` ordering has 68% win rate in sequence stats. HypothesisEngine generates a formal DAL short hypothesis. Within 90 days, the hypothesis accumulates 12 live observations via the outcome collector. DSR clears. Hypothesis promoted. MIDGE now has a standing structural hypothesis about energy-to-airline causal chains, not just an ad hoc convergence pattern.

---

## Capability C: Narrative-Pattern Fusion (Social-Technical Intelligence)

**The Emergent Thing:** Market moves need both a structural pattern AND a narrative reason to sustain. This combination detects when social narrative (what people are talking about) aligns with technical pattern (what the chart is doing) — a powerful filter that eliminates high-confidence false signals.

### Systems Involved

| System | What It Does Alone |
|---|---|
| `SocialTextAnalyzer` | Reads StockTwits messages from SQLite, detects when one theme (earnings_play, short_squeeze, macro_fear, breakout, capitulation) dominates with high intensity. Outputs `SocialTextSignal` with dominant_theme, direction, and strength. Currently feeds into signal pipeline as source `"social_text"` domain `"sentiment"`. |
| `PatternArchetypeEngine` | Matches current signal domains against 8 canonical structural archetypes (accumulation, distribution, squeeze, capitulation, momentum_ignition, failed_breakout, sector_rotation, smart_money_divergence). Returns `ArchetypeMatch` with match_score. Tracks `PartialMatch` for Gift 10 completion hunting. Currently operates on domain presence — does NOT use social narrative to confirm or disqualify archetype matches. |
| `HypothesisEngine` | Has an `archaeological_analyzer` slot. Tracks active hypotheses and monitors incoming signals for trigger matches. Does NOT receive structural archetype matches as an input for hypothesis promotion. |

### The Emergent Capability

**Narrative-Pattern Alignment Detection**: A structural pattern plus a matching narrative is far more reliable than either alone. An "accumulation" archetype (insider buying + range-bound price) is noise without understanding that the market narrative is quiet (no public chatter about the stock). A "capitulation" archetype is most reliable when it co-occurs with "capitulation" social text theme (crowd panic confirming the structural signal).

The emergent capability: **Archetype-Narrative Congruence Score** — a modifier that boosts archetype confidence when the social narrative aligns with what the structural pattern predicts, and discounts it when they diverge.

| Archetype | Aligned Narrative | Divergent Narrative |
|---|---|---|
| accumulation | No dominant social theme (quiet = stealth) | breakout/short_squeeze (too loud = distribution likely) |
| capitulation | capitulation theme (crowd panicking) | breakout/earnings_play (crowd still hopeful = not true bottom) |
| short_squeeze | short_squeeze theme (crowd sees it) | macro_fear (narrative wrong = squeeze unlikely) |
| momentum_ignition | breakout theme (narrative building) | capitulation (narrative contradicts structure) |

### How It Works — Information Flow

1. PatternArchetypeEngine produces an `ArchetypeMatch` for a ticker
2. SocialTextAnalyzer.analyze_ticker(ticker) is called to get the current dominant narrative theme (fast — SQLite read, no API)
3. A lookup table maps `(archetype_id, dominant_theme) → congruence_modifier` where modifier is in [0.5, 1.4]. Quiet (no signal) = neutral (1.0). Aligned = boost. Divergent = penalty.
4. `ArchetypeMatch.match_score` is multiplied by the congruence modifier to produce `adjusted_score`
5. When `adjusted_score >= 0.85` (high alignment), publish `market.intel.archetype_confirmed` with full context. This channel is consumed by:
   - ConvergenceAlerter: treats archetype_confirmed as a synthetic "structural" domain signal
   - HypothesisEngine: treats archetype + narrative alignment as a fast-track trigger for hypothesis generation (skip standard lag-finding phase)
6. When `adjusted_score <= 0.40` (strong divergence), publish `market.intel.archetype_rejected` — acts as a deception flag

### Value to MIDGE

Current ConvergenceAlerter combines signals but doesn't understand whether the *narrative context* supports the structural pattern. The "smart_money_divergence" archetype (insider buying while price falls) is meaningless if the whole market is discussing that company as a fraud. This filter would have caught numerous false positives in MIDGE's Feb 2026 replay (19.9% win rate) where technically valid signal combinations appeared during narrative-adverse environments.

### Effort to Build

**Trivial.** SocialTextAnalyzer already reads from SQLite synchronously. PatternArchetypeEngine already produces match objects. The congruence table is a 10-line static dict. The integration is ~50 lines of bridge code called from the existing archetype-scanning step in the sensing hook.

### Example Scenario

NVDA shows an "accumulation" archetype match (score 0.78). SocialTextAnalyzer returns no dominant theme (low message count, quiet). Congruence modifier: 1.25 (quiet + accumulation = typical smart money behavior). Adjusted score: 0.975. ConvergenceAlerter receives a "structural" domain signal with strength 0.97. Pattern stack fires at tier 3 confidence.

Three days later, NVDA shows the same accumulation archetype (score 0.81). But SocialTextAnalyzer shows "short_squeeze" theme dominating (40% of messages). Congruence modifier: 0.55 (squeeze chatter during "accumulation" = likely distribution, not accumulation). Adjusted score: 0.446. Alert is suppressed. In practice, squeeze chatter during what looks like accumulation often means retail is piling in while smart money distributes — the exact opposite of what the structural signal naively suggests.

---

## Capability D: Somatic Deception Intelligence

**The Emergent Thing:** MIDGE can distinguish between "I'm feeling anticipation and the signals are real" versus "I'm feeling anticipation but the signals are manipulated/absent in suspicious ways." This is market intuition with a built-in fraud detector.

### Systems Involved

| System | What It Does Alone |
|---|---|
| `SomaticAnticipation` | Tracks per-ticker domain accumulation. When 2+ domains activate within 48h, fires pre-convergence hormone response (DOPAMINE for agreement, CORTISOL for mixed). Records anticipation strength and domain names. Does NOT distinguish between signals arriving naturally vs. signals arriving with suspicious timing. |
| `AbsenceMonitor` | Detects when normally-active signal sources go unexpectedly silent. Knows that congressional members who trade weekly going quiet before a vote is a strong signal. Currently fires `CH_ABSENCE_DETECTED` events. **Critical gap:** AbsenceMonitor does NOT communicate with SomaticAnticipation. |
| `DeceptionDetector` (from 10 Gifts — `deception_state.json` exists in data/market/) | Detects coordinated manipulation signals. Based on `deception_state.json`, it is operational. Currently operates independently of somatic anticipation. |
| `SelfMonitor` | Watches MIDGE's own alerting behavior for runaway_rate, confidence_clustering, direction_bias, ticker_flooding. Currently only looks at MIDGE's outputs, not at the INPUT signal patterns. |

### The Emergent Capability

**Contextual Somatic Intelligence**: SomaticAnticipation's hormone response changes based on signal quality context. When anticipation fires AND AbsenceMonitor shows expected sources (insider, SEC) are silent for that ticker, the somatic response becomes CORTISOL-weighted (caution) instead of DOPAMINE-weighted (curiosity). When anticipation fires AND absence is normal AND DeceptionDetector shows no manipulation flags, the response is strongly DOPAMINE (high confidence this is real).

This creates three distinct somatic states instead of one:

1. **Authentic convergence**: Signals arriving naturally + no suspicious absences → DOPAMINE (act)
2. **Incomplete convergence**: Signals partial + key sources unexpectedly silent → CORTISOL + Focused Attention boost for missing sources
3. **Suspicious convergence**: Signals arriving but known manipulated or coordinated → CORTISOL spike + alert to DeepAnalyst that this ticker needs extra scrutiny

### How It Works — Information Flow

1. SomaticAnticipation fires its standard check every 25 steps. Before releasing hormones, it now calls two new checks:
2. `AbsenceMonitor.check_absences()` filtered to sources relevant to the ticker's signal domains (if "insider" domain is present, check `sec_form4` cadence for that ticker category)
3. A simple query to DeceptionDetector: "is this ticker currently flagged?"
4. Based on results, SomaticAnticipation's `_compute_somatic_response()` method applies a **quality modifier** to the base hormone cocktail:
   - `absence_penalty`: if key sources are silent, reduces DOPAMINE, adds CORTISOL
   - `deception_flag`: if ticker is flagged, replaces DOPAMINE with CORTISOL + publishes `market.intel.suspicious_convergence`
5. `market.intel.suspicious_convergence` channel is consumed by:
   - ConvergenceAlerter: adds a skepticism modifier to confidence for that ticker
   - HypothesisEngine: flags any hypotheses triggered by this ticker for extra validation rounds
   - DeepAnalyst: when scoring this ticker, applies a 0.7 multiplier to template score

### Value to MIDGE

Coordinated pump-and-dump schemes deliberately CREATE false multi-domain signal convergence. They generate social buzz (sentiment), sometimes plant fake insider filings (insider domain), and simultaneously suppress the sources that would give away their activity (Form 4 filings go quiet for the real insiders). SomaticAnticipation currently can't tell the difference between authentic and manufactured convergence. This combination can.

### Effort to Build

**Trivial.** AbsenceMonitor and SomaticAnticipation are both on `ctx`. They already run on similar cadences. The modification is 30-40 lines inside `SomaticAnticipation.check_anticipation()`: call `AbsenceMonitor.check_absences()`, filter to relevant sources, adjust the hormone cocktail. DeceptionDetector already has a `deception_state.json` and is wired to EventBus — just need to add the query.

### Example Scenario

GME-like scenario: A low-float stock suddenly accumulates signals across 4 domains (sentiment: short_squeeze, technical: breakout pattern, insider: small purchases, fundamental: aggressive options positioning). SomaticAnticipation fires with strong DOPAMINE at anticipation_strength 0.78. But AbsenceMonitor finds `sec_form4` cadence is broken — there have been ZERO Form 4 filings from this company's real insiders in 10 days despite the stock being active. That's suspicious: real insider buying would generate Form 4 filings within 2 business days. SomaticAnticipation now emits CORTISOL + suspicious_convergence. ConvergenceAlerter applies skepticism modifier. Even if confidence reaches 0.80, the `suspicious_convergence` flag is appended to the alert. MIDGE still surfaces the signal but flags it explicitly as potentially manufactured.

---

## Capability E: Regime-Aware Execution Intelligence

**The Emergent Thing:** Position sizing that adapts to regime, momentum, and recent MIDGE-specific performance — not just ATR. This transforms `SignalTranslator` from a static formula engine into a context-aware trading system.

### Systems Involved

| System | What It Does Alone |
|---|---|
| `SignalTranslator` | Converts ConvergenceAlert → ExecutableSignal. Uses fixed formula: SL = 1.5×ATR, TP = 3.0×ATR, position_size = account_risk_pct / risk_fraction (capped at 10%). No regime awareness. No learning from recent MIDGE performance. account_risk_pct is a constant (0.02 = 2%). |
| `RegimeClassifier` | Knows whether we're in bull/bear/volatile/sideways. Not currently consulted by SignalTranslator. |
| `DrawdownMonitor` | Tracks real equity curve, current drawdown, whether trading is halted. Already publishes `CH_DRAWDOWN_WARNING` and `CH_TRADING_HALTED`. Not currently consulted by SignalTranslator for position sizing adjustments — only checked as binary halt. |
| `PostMortemReviewer` | Knows which domain combos have what win rates, and what the current regime failure patterns are. Writes to `post_mortem_insights.json`. Not currently consumed by SignalTranslator. |
| `VelocityDetector` | Tracks rate of change in signals — are signals accelerating or decelerating? Acceleration is a proxy for momentum confirmation. Not currently connected to execution. |

### The Emergent Capability

**Dynamic Risk Scaling**: Rather than always risking 2% per trade, MIDGE scales position size based on four context factors that are all already computed:

```
base_risk    = 0.02 (2% account risk)
regime_mult  = {bull: 1.0, sideways: 0.8, bear: 0.7, volatile: 0.5}
drawdown_adj = 1.0 - (current_drawdown / max_drawdown) × 0.5  # scales 1.0 → 0.5 as DD grows
combo_conf   = from post_mortem_insights, win rate for this specific domain combo
velocity_mul = 1.15 if signal velocity is positive (accelerating), 0.85 if decelerating

dynamic_risk = base_risk × regime_mult × drawdown_adj × combo_conf × velocity_mul
dynamic_risk = clamp(dynamic_risk, 0.005, 0.04)  # floor 0.5%, ceiling 4%
```

No individual system produces this. Together they create a position sizer that is **self-aware of its own recent performance** (drawdown_adj), **regime-aware** (regime_mult), **historically calibrated** to specific signal combinations (combo_conf), and **momentum-sensitive** (velocity_mul).

### How It Works — Information Flow

1. `SignalTranslator.translate_alert()` currently accepts `account_risk_pct` as a parameter (already the right interface)
2. A new function `compute_dynamic_risk_pct(ctx, alert_dict)` is called by market_hooks.py BEFORE calling translate_alert():
   - Reads `ctx.regime_classifier.classify()` → regime_mult
   - Reads `ctx.drawdown_monitor.get_current_drawdown()` → drawdown_adj
   - Reads `post_mortem_insights.json` combo_stats for this alert's domain combo → combo_conf (cached in memory, refreshed every 500 steps)
   - Reads `ctx.velocity_detector.detect_velocity_anomalies()` → velocity_mul
3. `dynamic_risk` is passed as `account_risk_pct` to `translate_alert()`
4. `ExecutableSignal` gains two new fields: `regime_used` and `dynamic_risk_factors` (for audit trail)
5. `DrawdownMonitor.record_trade_result()` is already called when paper trades close — no change needed

### Value to MIDGE

The Feb 2026 replay showed 19.9% overall win rate but **winners were +11.4%, losers were -3.4%** (3.34:1 payoff ratio). The current fixed 2% risk means MIDGE makes 2% × 3.34 on winners and loses 2% on losers. If MIDGE dynamically sizes: 3% in bull + high combo confidence + accelerating signals, and 0.8% in volatile + unfamiliar combo + decelerating signals, the expectancy improvement is significant even without any change to signal quality.

In volatile regimes, current MIDGE applies full position size to alerts that have lower historical reliability (volatile periods have higher noise). The regime_mult fix alone would reduce volatility regime losses by ~30% at the current 19.9% convergence win rate.

### Effort to Build

**Trivial.** All inputs are already on `ctx` and already computed on the relevant cadences. The `translate_alert()` function already accepts `account_risk_pct`. The new function is ~40 lines. The post_mortem combo lookup requires caching the insights dict (already loaded by DeepAnalyst). Zero new systems. Zero new tests categories. Just a smarter caller.

### Example Scenario

Alert fires: NVDA bullish, domains [insider, macro, technical, institutional], confidence 0.81.

Current behavior: account_risk_pct = 0.02 → position_size_pct = 0.02 / (ATR_fraction) = ~5% of account.

With regime-aware execution:
- Regime: "volatile" → regime_mult = 0.5
- Current drawdown: 8% of 40% max → drawdown_adj = 1.0 - (0.08/0.40) × 0.5 = 0.90
- This combo (insider+macro+technical+institutional) from post_mortem: 29.4% WR, above average → combo_conf = 1.10
- Velocity: insider signals accelerating this week → velocity_mul = 1.15
- dynamic_risk = 0.02 × 0.5 × 0.90 × 1.10 × 1.15 = 0.0114 → 1.14%
- Position size: 0.0114 / ATR_fraction = ~2.85% of account (vs 5% fixed)

In a volatile regime, MIDGE is being correctly cautious. If this were a bull regime with no drawdown: dynamic_risk = 0.02 × 1.0 × 1.0 × 1.10 × 1.15 = 0.025 → 2.5% risk → ~6.25% position. The system rewards certainty and punishes uncertainty, exactly as it should.

---

## Novel Approaches

### Capability F: Memory-Guided Excavation Priority

Currently ExcavationDaemon excavates symbols in a fixed rotation without knowing which areas of history might be most valuable to dig. PatternMemory (Qdrant) stores all historical patterns. DeepAnalyst runs every 500 steps and identifies tickers with multi-domain signal activity.

**The combination:** Before ExcavationDaemon picks its next 10 symbols to excavate, it queries PatternMemory: "find pattern templates that have high win rates but low sample counts." These are templates that work but don't have enough instances to be statistically robust. ExcavationDaemon then prioritizes excavating symbols that are semantically similar to those under-sampled template types.

**The emergent thing:** Directed archaeology. Instead of random excavation, MIDGE digs where its own pattern knowledge is thinnest. The library becomes statistically stronger in exactly the areas where more data is needed.

**Effort:** Moderate — ExcavationDaemon needs a priority_queue concept, PatternLibrary needs a `get_undersampled_templates()` method (trivial — filter templates where wins+losses < 10 but win_rate > 0.55).

### Capability G: Quorum-Based Confidence Emergence

QuorumSpace (bio system, Tier 3 wired) already subscribes to market signals. MIDGE currently computes confidence via a formula (Thompson-weighted geometric mean + diversity bonus). Formulas produce the same confidence regardless of how many independent analytical subsystems "agree."

**The combination:** QuorumSpace receives signals from five analytical subsystems — ConvergenceAlerter, PatternWatcher, PatternArchetypeEngine, SomaticAnticipation, and DeepAnalyst. When 4+ of these subsystems independently reach the same directional conclusion about a ticker, QuorumSpace emits a quorum event. This quorum event becomes a +0.15 confidence boost to the ConvergenceAlerter result, applied ONLY when the quorum holds — never by formula.

**The emergent thing:** Confidence that emerges from genuine multi-system consensus rather than a formula pretending to represent consensus. This is exactly what Guiding Light described: "Confidence should be emergent consensus (quorum), not formula."

**Effort:** Moderate — QuorumSpace already subscribes to bus events. Needs result publishing from PatternArchetypeEngine and SomaticAnticipation to bus channels it monitors.

---

## Emerging Approaches

### Capability H: Temporal Lag Self-Calibration

Granger Analyzer currently runs every 500 steps and publishes lag findings. WorldModel uses lag_days for cascade timing. CascadeTracker records `energy_ratio` (actual lag / predicted lag) per confirmed link.

**The gap:** When CascadeTracker finds energy_ratio consistently 0.6 for a specific chain (actual lag = 60% of predicted), WorldModel's lag_days for that chain should be updated — but currently it isn't. WorldModel only updates edge strength, not lag timing.

**The emergent capability:** WorldModel's lag predictions continuously self-calibrate from real confirmation data. A chain predicted at 7 days that consistently confirms at 4 days will update to 4 days. This improves cascade prediction timing, which feeds back to more accurate `expected_window_days` in convergence alerts and eventually `expected_move_window_days` in signal_translator.

**Effort:** Trivial — CascadeTracker already has the energy_ratio data. WorldModel.record_outcome() needs a `actual_lag` parameter, and a 3-line update to adjust `lag_days` by EMA of actual lags.

### Capability I: Hypothesis-Driven Sensing

HypothesisEngine promotes hypotheses with specific trigger patterns (source_a firing → source_b follows after lag X days). Currently, after promotion, hypotheses just sit in the registry waiting for the trigger to fire naturally.

**The gap:** When a promoted hypothesis has trigger source_a, MIDGE should ACTIVELY look for source_a signals rather than waiting for the rotation to hit it. This is like a doctor who knows a patient is at risk for a specific condition doing targeted testing — not just waiting for symptoms.

**The emergent capability:** Promoted hypotheses populate `_priority_requests` in the sensing hook's Focused Attention system. Sources matching the hypothesis trigger source get 2x polling frequency. This closes a loop: hypothesis discovery → active evidence seeking → faster trigger detection → more outcomes to validate/retire.

**Effort:** Trivial — `_priority_requests` dict already exists in sensing_hook. HypothesisEngine just needs to add entries when hypotheses are promoted: `ctx._market_sensing_hook._priority_requests[trigger_source] = time.time() + 3600` (1-hour boost).

---

## Gaps and Unknowns

1. **DeceptionDetector state**: `data/market/deception_state.json` exists and is modified (in git status), confirming it is running. But the code file was not in the brief. The exact interface for "is this ticker flagged?" is unknown. If DeceptionDetector exposes a per-ticker query method, Capability D is trivial. If it only operates at a global level, the approach needs adjustment.

2. **PatternMemory availability**: PatternMemory gracefully degrades when Qdrant/Ollama are offline. Capabilities A (regime transition) and F (memory-guided excavation) partially depend on Qdrant being available. If Qdrant is down, these capabilities degrade to their component parts. This is acceptable — the core logic still runs, just without semantic search enhancement.

3. **SocialTextAnalyzer database coverage**: The SQLite StockTwits DB must have messages for the ticker in the last 24 hours for the social signal to fire. For less-followed tickers, this will produce no signal (not a false signal — just silence). Capability C's narrative-pattern fusion treats silence as neutral, not as negative evidence, which is correct.

4. **WorldModel ticker node set**: WorldModel's `_is_ticker()` method uses a heuristic (all-caps, 1-5 chars, no underscores). Forex pairs like `EURUSD=X` and futures like `CL=F` will not be recognized as ticker nodes. CascadeTracker Capability B would miss these instruments. This is a known limitation of the WorldModel design.

5. **VelocityDetector integration with execution**: VelocityDetector.detect_velocity_anomalies() returns a list of anomaly dicts. The mapping from "which velocity anomaly" to "velocity_mul for this specific signal" needs a simple heuristic (if any source in alert.domains shows velocity anomaly = positive → 1.15, negative → 0.85). Edge cases where multiple velocity anomalies conflict need a tie-breaking rule.

---

## Synthesis

Five capabilities emerged from this analysis, ranked by value/effort:

| Priority | Capability | Value | Effort | New Code Needed |
|---|---|---|---|---|
| 1 | **E: Regime-Aware Execution** | High — directly impacts P&L every trade | Trivial | ~40 lines |
| 2 | **C: Narrative-Pattern Fusion** | High — false positive filter | Trivial | ~50 lines |
| 3 | **D: Somatic Deception Intelligence** | High — detects manufactured signals | Trivial | ~40 lines |
| 4 | **A: Predictive Regime Forecasting** | Very High — advance warning of regime shifts | Moderate | ~200 lines |
| 5 | **B: Causal Sequence Hypotheses** | Very High — deeper learning loop | Moderate | ~300 lines |
| Bonus | **H: Lag Self-Calibration** | Medium — improves existing prediction timing | Trivial | ~20 lines |
| Bonus | **I: Hypothesis-Driven Sensing** | Medium — faster trigger detection | Trivial | ~10 lines |

**The unifying insight**: MIDGE has built an impressive collection of intelligence organs that are largely talking past each other. Each system produces outputs in isolation. The three trivial capabilities (C, D, E) require no new systems, no new data sources, and no new algorithms — only wiring existing outputs into existing inputs. They could be built in a single session and immediately improve MIDGE's trading behavior.

The two moderate capabilities (A, B) require small bridge components but unlock fundamentally new classes of reasoning: predictive regime awareness and causally-grounded hypothesis generation. These represent the most architecturally meaningful additions MIDGE could make — they turn MIDGE's separate intelligence organs into an interconnected learning organism.

Guiding Light's ecosystem/planet vision is most powerfully expressed by Capability B: a system that discovers causal relationships, watches them play out in live markets, grades its own predictions, and generates targeted hypotheses from what it learns — all automatically, all continuously, all without intervention.
