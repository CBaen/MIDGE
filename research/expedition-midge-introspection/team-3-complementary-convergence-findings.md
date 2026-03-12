# Team 3 Findings: Complementary Convergence — MIDGE's Internal Inevitabilities
## Date: 2026-03-12
## Researcher: Team Member 3

---

## Prefatory Note

This expedition applied MIDGE's own core insight — convergence from independent sources — to her internal architecture. The question: which systems are already producing outputs that other systems desperately need but are currently not receiving?

The findings below are ordered from highest to lowest structural impact. Each represents a wiring gap where the connection is structurally obvious once both sides are examined.

---

## Battle-Tested Approaches

### Pair 1: GrangerAnalyzer → HypothesisGenerator (directional causality → hypothesis seeds)

**System A produces:** `GrangerFinding` objects — statistically validated directional causal relationships between signal sources (e.g., "congressional trades Granger-cause price movement at 14-day lag, p=0.003"). Persists to `granger_causality.json`. Runs every 500 steps.

**System B needs:** `HypothesisGenerator` currently seeds from `LagFinding` objects (via `lag_correlations.json`). It builds hypotheses of the form "if source A fires then source B will follow." But lag correlations detect co-movement, which can be spurious. Granger causality CONTROLS for autocorrelation and tests whether A genuinely adds predictive power beyond what B already predicts from itself.

**The connection:** `HypothesisGenerator._generate_from_lag_findings()` accepts `List[LagFinding]`. A trivial adapter converts `GrangerFinding` to `LagFinding`-shaped input (same fields: source_a, source_b, lag_days, correlation). The resulting hypotheses would be stronger: "granger-confirmed: congressional trades predictive of LMT at lag 14d (p=0.003)" rather than mere co-movement.

**Impact: HIGH** — Creates new capability. Hypotheses derived from Granger causality have an explicit causal claim, not just correlation. The hypothesis validation pipeline (adversarial testing, DSR) would then tell MIDGE which causal claims survive real-world grading. This is the research standard for leading-indicator validation.

**Effort: Trivial** — One adapter function mapping `GrangerFinding` fields to `LagFinding`-shaped dict, injected into the generation cadence call inside `hypothesis_engine._run_generation()`.

**Independence test:** Safe. Granger results feed hypothesis GENERATION (a new source of seeds), not the convergence scoring. No domain independence concern.

---

### Pair 2: DriftDetector → RegimeClassifier (distribution shift detection → regime invalidation)

**System A produces:** `DriftEvent` objects when statistical distribution shifts are detected in named streams (price_returns, vix, sentiment, signal_volume). ADWIN detects these faster than any fixed-window indicator. A drift event says: "the distribution of this stream fundamentally changed at observation N."

**System B needs:** `RegimeClassifier` caches its result for an entire calendar day. It uses a fixed 20-day SPY lookback with hardcoded thresholds. It has no mechanism to be told "regime may have just changed mid-day."

**The connection:** When `DriftDetector` fires on `price_returns` or `vix`, this is a real-time signal that the regime assumption may be stale. A one-line callback in `_run_drift_detector()` (which already exists in `market_hooks_steps.py`) can call `rc._cache_date = None` to invalidate the regime cache, forcing re-classification on the next call. Currently `_run_drift_detector` publishes to `market.intel.drift_detected` but nothing subscribes to that event to invalidate the regime cache.

**Impact: HIGH** — Prevents regime-stale Thompson weights. When a crash starts, the drift detector will catch it within minutes via the price_returns stream. RegimeClassifier currently won't reclassify until the next calendar day. Thompson's regime-aware decay rates (volatile=0.90, bull=0.95) will be wrong for the entire day.

**Effort: Trivial** — Subscribe to `market.intel.drift_detected` in `market_hooks_eventbus.py`. When stream is `price_returns` or `vix` and magnitude exceeds a threshold, call `ctx.regime_classifier._cache_date = None`.

**Independence test:** Safe. DriftDetector feeds a cache-invalidation signal, not a score or weight. The regime classifier still runs its own price logic — it just runs sooner.

---

### Pair 3: VelocityDetector → ConvergenceAlerter (acceleration detection → dynamic signal weighting)

**System A produces:** `VelocityState` objects per signal source, including `velocity_zscore` (how unusual the rate of change is), `is_accelerating` flag, and `current_acceleration`. Velocity anomalies are the most actionable output: "insider buying is accelerating 2.8σ above its historical rate."

**System B needs:** `ConvergenceAlerter` computes signal strength from raw strength values at time of ingestion. It applies Thompson reliability weights but has no mechanism to boost a signal that is currently accelerating. A signal at strength 0.6 that is accelerating rapidly toward convergence is structurally more significant than a stable 0.6.

**The connection:** `VelocityDetector.detect_velocity_anomalies()` returns signals with anomalous velocity. These can be fed as synthetic "velocity" domain signals into `convergence_alerter.record_signal()` with boosted strength (`velocity_zscore * base_strength`). Alternatively, the strength multiplication can happen at the sensing hook before ingestion. The VelocityDetector is already wired (bootstrap confirmed it exists on ctx), but its output is never consumed by the convergence engine.

**Impact: HIGH** — VelocityDetector was designed for leading indicator detection but currently its output has no downstream consumer. This closes the loop. An accelerating insider buying cluster is a materially different signal than a static cluster of the same count.

**Effort: Moderate** — Requires a cadenced step in `_run_slow_cadence_ops` (or the sensing hook) that reads `velocity_detector.detect_velocity_anomalies()` and injects boosted signals. The boost factor needs calibration.

**Independence test:** Concern exists. If velocity signals are injected as a new domain, they would artificially add domain count. They should be injected as a MODIFIER to existing signals, not as a new independent domain. Implementation must not inflate domain count used for convergence threshold.

---

### Pair 4: SocialTextAnalyzer → PatternArchetypeEngine (text theme detection → archetype completion signal)

**System A produces:** `SocialTextSignal` objects with `dominant_theme` (one of: options_flow, short_squeeze, earnings_play, macro_fear, breakout, capitulation), `theme_score`, and `direction`.

**System B needs:** `PatternArchetypeEngine` has 8 built-in archetypes with `required_signals` and `optional_signals` lists. These use domain names like "sentiment", "insider", "technical". The "short_squeeze" archetype requires `["technical"]` and has optional `["positioning", "institutional"]`. But the single most diagnostic text signal for a short squeeze — social text SCREAMING about it — is never presented to the archetype engine.

**The connection:** `SocialTextAnalyzer.analyze_all()` returns signals with `dominant_theme`. Map themes to archetype IDs: `short_squeeze` → "squeeze" archetype, `breakout` → "momentum_ignition" or "accumulation", `capitulation` → "capitulation" archetype. Pass the theme as an additional domain signal to `PatternArchetypeEngine.scan_for_archetypes()` for the same ticker.

**Impact: MEDIUM** — Creates new capability: text-corroborated archetype matches. The "squeeze" archetype will match more accurately when social text is also screaming "squeeze." Currently archetypes only see technical/insider/institutional domain signals.

**Effort: Moderate** — Requires a domain mapping table (theme → archetype domain name) and a call to archetype engine after social text analysis in the sensing hook cadence.

**Independence test:** Concern. "Short squeeze" text and technical signals are correlated (people write about what price is doing). However, since archetypes are not used for convergence DOMAIN COUNTING (they're a separate template-matching system), this doesn't violate domain independence for convergence purposes.

---

### Pair 5: AbsenceMonitor → SomaticAnticipation (silence detection → pre-convergence body response)

**System A produces:** `AbsenceSignal` objects when a normally-active source goes silent. This is an inverted signal: "congressional member who trades weekly has been silent for 21 days before a major vote."

**System B needs:** `SomaticAnticipation` fires based on PRESENCE of signals crossing a domain threshold. It has no mechanism to respond to ABSENCE signals. But absence from informed actors is itself a directional signal (defaults to bearish in `AbsenceSignal`).

**The connection:** `SomaticAnticipation.record_signal()` accepts ticker, domain, direction, strength. An absence signal can be mapped to a ticker (if the absent source is ticker-specific) or to a general "informed actor silence" signal. The somatic system would then accumulate this absence signal alongside positive signals, potentially triggering anticipation even when fewer positive domains have fired — because the absence is itself information.

**Impact: MEDIUM** — Creates new capability: absence-triggered anticipation. The current system is blind to silence as a signal type. The dog-that-didn't-bark pattern (a normally-active congressional trader going silent before a committee vote) would now register as anticipation.

**Effort: Moderate** — Requires mapping absence sources to tickers where possible. Congressional absences can map to defense/healthcare tickers based on committee membership. Non-ticker absences (COT data late, FRED stale) map to macro tickers.

**Independence test:** Safe. Absence signals represent a genuinely different phenomenon from positive signals. They would count as a new domain ("absence") rather than inflating an existing domain's count.

---

## Novel Approaches

### Pair 6: PostMortemReviewer → DeepAnalyst (failure analysis → scoring recalibration)

**System A produces:** `post_mortem_insights.json` containing `combo_stats` (unordered domain combination win rates), `sequence_stats` (ordered domain firing sequence win rates), `mfe_mae_patterns` (right thesis / wrong timing detection), `flagged_orderings` (domain orderings that consistently fail).

**System B needs:** `DeepAnalyst` computes `_combo_boost()` using `post_mortem_insights.json` combo_stats to adjust its scoring via a 0.8-1.25 multiplier. This connection EXISTS but is incomplete in one critical way: `DeepAnalyst._combo_boost()` uses only `combo_key` (unordered) and ignores `sequence_stats` (ordered). A combo that fires in the wrong order has a different win rate than the same combo in the correct order.

**The connection:** Expose `sequence_stats` from `PostMortemReviewer` and add a `_sequence_boost()` method to `DeepAnalyst` that checks the domain firing order of the candidate (already available in `Inevitability.signals` timestamps) against the sequence stats. A combo of [insider, macro, technical] that historically wins 40% when insider fires first but 62% when macro fires first would score differently.

**Impact: HIGH** — Turns DeepAnalyst from a combo-aware scorer into a sequence-aware scorer. This is the next layer of precision beyond domain presence. The temporal ordering insight from Phase 0 measurements is directly applicable here.

**Effort: Moderate** — DeepAnalyst already reads `post_mortem_insights.json`. Extend `_load_combo_stats()` to also load `sequence_stats`. Add `_sequence_boost()` that infers firing order from signal timestamps, looks up the ordered key, and applies the win-rate-based multiplier.

**Independence test:** Safe. Post-mortem is retrospective analysis that modifies scoring weights, not domain membership.

---

### Pair 7: CorrelationTracker.detect_cross_domain_anomalies() → ConvergenceAlerter (unusual correlation → deception signal)

**System A produces:** `detect_cross_domain_anomalies()` returns pairs where normally-uncorrelated domains are suddenly correlating. This is described in the CorrelationTracker documentation as "the most valuable for leading indicator detection: normally uncorrelated domains suddenly correlating." The system exists and runs but its anomaly output is consumed by nothing.

**System B needs:** The `DeceptionDetector` ("Ten Gifts" Gift 5) was built to detect coordinated/manufactured signals. Cross-domain correlation anomalies are the statistical fingerprint of coordinated information: when crypto whale movements suddenly correlate with congressional trades (as cited in `correlation_tracker.py`'s own docstring), someone may be trading on information not yet public.

**The connection:** `CorrelationTracker.detect_cross_domain_anomalies()` → emit `CH_DECEPTION_DETECTED` (or a new channel) → `DeceptionDetector` escalates these as high-priority signals. Alternatively, inject correlation anomaly pairs directly into `convergence_alerter` as a new "deception" domain signal.

**Impact: HIGH** — The CorrelationTracker is running and accumulating pair state but its most valuable output (`detect_cross_domain_anomalies()`) is never called in the step loop. This is a complete waste of a powerful detection capability. Anomalous cross-domain correlation is a genuine structural signal — it indicates someone knows something.

**Effort: Trivial** — Add a cadenced call to `detect_cross_domain_anomalies()` in `_run_slow_cadence_ops`. When anomalies are found, emit them via EventBus. Wire a subscriber that converts them to MarketSignals in the "deception" domain.

**Independence test:** Potential concern. Cross-domain correlation anomalies are derived from the same underlying signals that feed convergence. However, the DETECTION of anomalous correlation is a meta-signal — it is observing the relationship between signals, not the signals themselves. Should be treated as an independent "meta-domain" to avoid double-counting.

---

### Pair 8: MotifDetector → PatternLibrary (price motifs → template validation)

**System A produces:** `MotifSignal` objects of type "motif" (current price action repeats a known historical shape) or "discord" (price action unlike anything seen before). Each motif carries `mp_index` pointing to where in the price history the match was found.

**System B needs:** `PatternLibrary` stores `PatternTemplate` objects accumulated from archaeology (excavation). Templates are matched based on domain signal overlap. But templates currently have no price-shape component — a template saying "insider+macro+technical fired bullishly" is matched purely by domain presence, not by whether the current price action LOOKS LIKE the price action when those domains last converged.

**The connection:** When `MotifDetector` fires a "motif" signal on a ticker, cross-reference against the PatternLibrary's active templates for that ticker. If a high-confidence motif fires AND a pattern template matches, the combination is a dual-confirmation signal: the domain convergence is happening AND the price shape matches past instances. Feed this as a combined score boost into PatternWatcher.

**Impact: MEDIUM** — Creates new capability: price-shape-corroborated template matches. Currently templates are domain-only. Adding price-shape confirmation would reduce false matches where domains converge but the price action doesn't fit the historical fingerprint.

**Effort: Substantial** — Requires PatternTemplate to store price shape fingerprints (or a separate price-shape → template_id index). The MotifDetector tracks symbol streams but doesn't know about templates. Needs a coordination layer.

**Independence test:** Safe. Price shape detection (technical) and domain convergence use different input streams.

---

### Pair 9: DrawdownMonitor → ThompsonSampler (equity loss attribution → source reliability downgrade)

**System A produces:** `record_trade_result()` with realized P&L per ticker and direction. The DrawdownMonitor tracks which trades won and lost.

**System B needs:** `ThompsonSampler` receives success/failure updates from `OutcomeCollector`. But the outcome collector grades success based on PRICE MOVEMENT, not on whether MIDGE actually profited. A trade might be graded "correct" (price moved in the predicted direction) but the execution parameters (stop loss hit before the move, timing off) lost money. DrawdownMonitor knows which trades actually lost capital.

**The connection:** When `DrawdownMonitor.record_trade_result()` records a loss that contradicts an "outcome_correct=True" grade, this is an execution-vs-signal quality gap. While not directly wiring them together (that would confuse signal quality with execution quality), the DrawdownMonitor should emit a signal when a "correct" prediction still loses money (MFE > realized, suggesting timing issues). This feeds back into the post-mortem "right thesis, wrong timing" detection.

**Impact: MEDIUM** — Closes the gap between prediction accuracy (outcome_collector) and actual profitability (drawdown_monitor). Currently these run independently and neither informs the other.

**Effort: Moderate** — Requires DrawdownMonitor to optionally receive the prediction_id associated with each trade. When a trade closes at a loss, check if the corresponding prediction was graded "correct" and emit a "correct-but-unprofitable" event.

**Independence test:** Safe. This is feedback enrichment, not domain signal input.

---

### Pair 10: HypothesisEngine.get_statistics() → SomaticAnticipation (hypothesis firing → pre-conscious ticker attention)

**System A produces:** `HypothesisEngine` publishes `CH_HYPOTHESIS_FIRED` when an active hypothesis's trigger conditions are matched. The event contains `trigger_symbol`, `trigger_source`, and `expected_direction`. Currently this event is only consumed to boost focused attention via `_priority_requests` in the sensing hook.

**System B needs:** `SomaticAnticipation` accumulates signals per ticker to detect pre-convergence patterns. A hypothesis firing is itself a strong pre-convergence signal: MIDGE's own learned hypotheses say "when source X fires for ticker Y, watch for direction Z."

**The connection:** Subscribe `SomaticAnticipation.record_signal()` to `CH_HYPOTHESIS_FIRED`. Convert the hypothesis firing into a somatic signal: domain="hypothesis", direction=expected_direction, strength=hypothesis.confidence_score. This means a hypothesis trigger will immediately create anticipation state for the ticker, boosting endocrine response before any formal convergence fires.

**Impact: MEDIUM** — Closes the loop between MIDGE's learned hypotheses and her pre-conscious attention system. Currently these are parallel tracks. A hypothesis that has been promoted and validated through adversarial testing is MIDGE's most confident prior — it should have the strongest possible effect on her anticipatory state.

**Effort: Trivial** — One new EventBus subscription in `market_hooks_eventbus.py`. Map hypothesis_fired payload to somatic signal fields.

**Independence test:** Concern: hypothesis triggers are derived from the same signal sources that feed convergence. Counting hypothesis firing as a somatic domain could create feedback loops (hypothesis fires → anticipation fires → more sensing → convergence fires → hypothesis fires again). Mitigate by using a separate "hypothesis" domain that is NOT included in convergence domain counting.

---

## Emerging Approaches

### Pair 11: RegimeClassifier + DriftDetector → MotifDetector (regime state → stream selection)

**The gap:** `MotifDetector` tracks a single price stream per symbol. It does not know what market regime it's operating in. A "motif" detected during a volatile regime may not be meaningful — the same price shape during a bull regime has different predictive content.

**The connection:** Pass `regime_classifier.classify()` as metadata when updating MotifDetector streams. Store regime labels alongside motif detections. Over time, `MotifSignal` can carry `regime_at_detection`, allowing PatternLibrary to match motifs only when the current regime matches the historical detection regime.

**Impact: MEDIUM** — Makes motif detection regime-aware. Currently motifs from bull regimes and bear regimes are treated identically.

**Effort: Moderate** — Modify `MotifDetector.update()` to accept an optional `regime` parameter. Store in `MotifSignal.metadata`.

**Independence test:** Safe. Regime is metadata, not a new signal source.

---

### Pair 12: LagCorrelationAnalyzer → AbsenceMonitor (known leading indicators → absence urgency scaling)

**The gap:** `AbsenceMonitor` treats all sources equally when scoring silence urgency. A source that is known to be a leading indicator (from `lag_correlations.json`) should be treated as MORE urgent when silent — its silence deprives MIDGE of advance warning.

**The connection:** At bootstrap, load `lag_correlations.json` into AbsenceMonitor. Sources that appear as `source_a` (leading indicator) in high-correlation findings get a silence urgency multiplier. "Congressional trades going silent before a known lag window (14 days) is more urgent than FRED going silent" because we know congressional trades lead price movement.

**Impact: LOW** — Improves existing capability (absence prioritization) rather than creating new capability.

**Effort: Trivial** — Add `load_lag_priorities(lag_file_path)` method to AbsenceMonitor analogous to CorrelationTracker's existing `seed_from_lag_data()`.

**Independence test:** Safe. This modifies urgency scoring, not signal emission.

---

### Pair 13: ClusterDetector + PoliticianTracker → ContractPredictor (insider clustering context → bid confidence)

**The gap:** `ContractPredictor._check_insider_buying()` makes a separate `get_recent_form4s()` call when evaluating a potential contract winner. `ClusterDetector` already has Qdrant queries built for detecting multi-insider buying within 30 days for the SAME tickers. `PoliticianTracker` has committee-oversight correlation. These three systems separately re-derive what they could share.

**The connection:** `ContractPredictor` should call `ClusterDetector.find_clusters(ticker)` instead of (or in addition to) its raw Form 4 check. A cluster signal is a stronger version of the insider buying check: it confirms 3+ insiders, not just 1. PoliticianTracker's `find_correlations()` could feed ContractPredictor's known-winner list.

**Impact: LOW** — Improves signal quality for an existing capability. ContractPredictor gains higher-quality insider confirmation.

**Effort: Moderate** — Requires dependency injection (ContractPredictor currently constructs its own EDGAR calls). Medium refactor.

**Independence test:** Safe. All three are in the "insider/government" domain — they share domain, but ContractPredictor uses them for entity-level prediction, not domain-level convergence.

---

## Gaps and Unknowns

**1. VelocityDetector current wiring:** The bootstrap confirms VelocityDetector is instantiated but no step hook calls `detect_velocity_anomalies()`. Its output is entirely unconsumed. This is the single largest dead weight in the intelligence layer.

**2. CorrelationTracker anomaly output:** `detect_cross_domain_anomalies()` and `find_leading_pairs()` are never called in any step hook. The tracker accumulates state but its highest-value methods are dead code paths.

**3. SocialTextAnalyzer → PatternArchetypeEngine:** The theme detection runs and produces signals, but PatternArchetypeEngine never receives theme signals — it only receives domain names from the general signal pipeline. Theme-level archetyping is effectively disabled.

**4. `CH_HYPOTHESIS_FIRED` scope:** The hypothesis fired event correctly boosts `_priority_requests` in the sensing hook, but SomaticAnticipation is not subscribed. The somatic system cannot respond to MIDGE's own confirmed hypothesis triggers.

**5. DeepAnalyst sequence awareness:** `_combo_boost()` exists but uses only unordered combos. `sequence_stats` in `post_mortem_insights.json` is written but never read by DeepAnalyst. Domain ordering is known to be predictively significant (from Phase 0 measurements, r=0.73 for macro+technical order-dependence) but is not used by the primary scoring system.

**6. DriftDetector → RegimeClassifier invalidation:** `_run_drift_detector` publishes `market.intel.drift_detected` but no subscriber invalidates the regime cache. The regime can be stale for up to 24 hours after a distribution shift.

---

## Synthesis

The pattern across all 13 pairs is consistent: MIDGE has excellent data-gathering and analysis components but the **output-to-input connections between analysis systems are incomplete**. Systems compute, persist, and then fall silent. Their findings are written to disk files and JSON but the next analytical layer does not pick them up.

The five highest-leverage connections, in order:

**1. GrangerAnalyzer → HypothesisGenerator** (trivial effort, high impact, creates statistically valid causal hypotheses vs. correlation-only hypotheses)

**2. DriftDetector → RegimeClassifier invalidation** (trivial effort, high impact, prevents regime-stale Thompson weights during market transitions)

**3. VelocityDetector → ConvergenceAlerter boost** (moderate effort, high impact, closes the single largest dead wire in the intelligence layer)

**4. CorrelationTracker cross-domain anomalies → DeceptionDetector/ConvergenceAlerter** (trivial effort, high impact, activates currently dead `detect_cross_domain_anomalies()` output)

**5. PostMortemReviewer sequence_stats → DeepAnalyst** (moderate effort, high impact, makes the primary scoring engine sequence-aware — the known single biggest predictor gap from Phase 0 research)

The key architectural insight from this investigation: **MIDGE's analysis layer is substantially more sophisticated than her integration layer**. Each analytical component is well-built but operates as an island. The "internal inevitabilities" are not gaps in capability — they are gaps in wiring. The good news: most connections are trivial to moderate in effort. The capability is already built; it simply needs routing.

The most concerning finding is structural rather than tactical: MIDGE's three velocity/correlation detection systems (VelocityDetector, CorrelationTracker, MotifDetector) were all built with explicit "leading indicator detection" purposes in their docstrings, yet none of their leading-indicator output methods are called in any step hook. These three systems are consuming bootstrap initialization cost and memory but delivering zero value to the convergence pipeline. This is the highest-priority class of fix — not because the connections are complex, but because the systems were designed for exactly this purpose and the final wiring step was simply never completed.
