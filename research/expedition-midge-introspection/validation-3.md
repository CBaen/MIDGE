# Validation 3 — Priority Ranking and Feasibility Verification

**Date:** 2026-03-12
**Validator role:** Stress-test claims, rank by actual trading impact, verify effort estimates.
**Protocol:** Divergence-first. Find what doesn't hold up before noting what does.

---

## Critical Contradiction Resolved First

### Team 1 vs Team 3: "VelocityDetector is ALIVE" vs "single largest dead wire"

Both claims are true simultaneously — and that is the problem.

**What the code actually shows:**

1. VelocityDetector runs. `sensing_lifecycle.py` line 34: `velocity_detector.record(sig.signal_id, sig.strength, sig.timestamp)` — called on every ingested signal. `market_hooks_steps_core.py` line 152-154: `detect_velocity_anomalies()` called every 50 steps. Running = confirmed.

2. The output goes nowhere. `CH_VELOCITY_ANOMALY = "market.intel.velocity_anomaly"` is defined in `channels.py`. `bio_market_wiring_a.py` registers a subscriber on it. But a grep across all production code finds zero `bus.publish(*velocity_anomaly*)` calls. The channel is subscribed to but never published to. The anomaly list returned by `detect_velocity_anomalies()` at step 152-154 is computed and then silently discarded — no publish, no injection into convergence, no logging of anomalies.

**Verdict:** Team 1 was measuring liveness of the *input* side (record). Team 3 was measuring liveness of the *output* side (publish). Neither was wrong. The wire runs from the wall to the middle of the room and stops.

---

## The CH_PREDICTION_RESULT Situation (Team 1 Claim)

Team 1 claims "5 lines fixes CH_PREDICTION_RESULT, waking 9 bio-systems."

**What the code shows:**

- `CH_PREDICTION_RESULT` has 9 registered subscribers across bio_market_wiring files — confirmed.
- In **production code**, there is zero `bus.publish(CH_PREDICTION_RESULT, ...)`. The only publish calls are in test files (`test_bio_market_wiring.py`, `test_bio_market_wiring_extended.py`).
- The channel is entirely subscriber-only in production. Something must call `bus.publish(CH_PREDICTION_RESULT, {"won": True/False, ...})` when `OutcomeTracker` grades a prediction. That publish call does not exist.

**Effort re-estimate:** "5 lines" is accurate for the publish call itself. But the payload must match what all 9 subscribers expect (`won`, `ticker`, `confidence`, etc.). Each subscriber was written against test-side publish calls. Risk of silent payload mismatch across 9 consumers. Call it 15-25 lines with proper payload construction and a field validation pass across all 9 consumers. Not 5.

---

## Top 10 Rankings by Trading Impact

### #1 — CH_PREDICTION_RESULT publish fix (Team 1)
**Category:** Learning speed
**What it does:** Closes the outcome→bio feedback loop. 9 bio-systems (curiosity, nociception, immune, pheromone, quorum sensing, etc.) are waiting to learn from prediction outcomes and currently receive nothing.
**Improves:** Learning speed (bio-systems calibrate to what wins)
**Effort:** 15-25 lines (not 5 — see above)
**Risk:** Payload mismatch across 9 consumers. Each must receive consistent fields. Low regression risk since these systems are currently inert.
**Mae's Laws:** Requires witness — the publish call should be triadic: OutcomeTracker → bus → bio-system, with outcome_collector as witness. Currently dyadic (OutcomeTracker → outcome_collector alone). Fix satisfies Law 1.
**Verdict: High value, verified gap. Do it.**

---

### #2 — VelocityDetector output publication (Team 3)
**Category:** Inevitability detection
**What it does:** `detect_velocity_anomalies()` already runs every 50 steps and returns a ranked list of accelerating signals. Zero lines currently publish the result. Adding `bus.publish(CH_VELOCITY_ANOMALY, anomaly_data)` is ~8 lines.
**Improves:** Detection — velocity anomalies are leading indicators that currently exist as computed data in RAM and get garbage-collected.
**Effort:** 8 lines to publish. The subscriber in `bio_market_wiring_a.py` already exists and was written to receive this.
**Risk:** The injection-into-convergence path (Team 3's secondary ask) is more dangerous. Adding velocity as a *domain* would inflate the domain count in ConvergenceAlerter, potentially lowering the convergence threshold. Do NOT add velocity as a domain. Publish to CH_VELOCITY_ANOMALY only — let bio-systems respond. Separate concern.
**Mae's Laws:** Publishing to existing subscriber is not a new dyad. Clean.
**Verdict: Very high value, minimal risk. Do the publish. Skip convergence domain injection.**

---

### #3 — Alpaca position outcome registration (Team 2)
**Category:** Learning speed + trading performance
**What it does:** Alpaca paper trades execute (confirmed wired), but when positions close, that outcome never flows back into Thompson. MIDGE executes but does not learn from executions.
**Improves:** Thompson calibration specifically for signals that crossed the paper-trade threshold. This is the highest-quality feedback data available — it represents signals where MIDGE had enough confidence to act.
**Effort:** Medium. Requires polling Alpaca for position status, detecting close, computing P&L, calling `outcome_collector.register_convergence_alert()` with result. ~50-80 lines in alpaca_client.py or market_hooks.
**Risk:** Alpaca API calls for position status add latency. Must be async or off-thread.
**Mae's Laws:** Alpaca → OutcomeCollector → Thompson forms a valid triad with PriceFetcher as third arm. Law 1 satisfied.
**Verdict: High value. Requires real implementation effort, not trivial.**

---

### #4 — GrangerAnalyzer → HypothesisGenerator (Team 3)
**Category:** Inevitability detection
**What it does:** GrangerAnalyzer runs every 500 steps and finds directional causal relationships (source A Granger-causes source B at lag N). HypothesisGenerator creates hypotheses from patterns. Currently these two systems are unconnected — Granger findings sit in `granger_causality.json` but never seed hypotheses.
**Improves:** Detection quality — causal hypotheses are structurally stronger than correlation-based ones. "Congressional trades Granger-cause contract awards at 14-day lag" is a testable, directional inevitability claim.
**Effort:** Medium-low. HypothesisGenerator has an API. Granger results have a structured format. Bridge: read top Granger findings after each run, call `hypothesis_generator.generate_from_granger(findings)`. ~40-60 lines in market_hooks_steps.
**Risk:** Hypothesis proliferation — Granger finds many relationships. Need minimum F-statistic threshold to filter (already present in granger_analyzer). Low regression risk.
**Mae's Laws:** Granger → HypothesisGenerator → HypothesisRegistry is a valid triad. Clean.
**Verdict: High value, achievable. Do it.**

---

### #5 — Hypothesis outcome tracking closure (Team 2)
**Category:** Learning speed
**What it does:** Hypotheses transition through probation → active → hibernated → retired, but the outcome_collector does not register convergence alerts that were hypothesis-driven as hypothesis-specific predictions. Thompson learns combo-level, but hypothesis-level learning is broken.
**Improves:** The hypothesis loop's feedback cycle. Without this, hypotheses retire on fixed-schedule rules, not on actual predictive performance.
**Effort:** Low-medium. `register_inevitability()` exists on OutcomeCollector (lines 283-321). The gap is calling it at hypothesis promotion time.
**Risk:** Low. OutcomeCollector is already wired. This is a call-site addition.
**Verdict: Medium-high value. Straightforward.**

---

### #6 — CorrelationTracker.detect_cross_domain_anomalies() activation (Team 3)
**Category:** Inevitability detection
**What it does:** `detect_cross_domain_anomalies()` exists (correlation_tracker.py lines 248-269), is implemented, returns cross-domain anomalies. It is never called in any bootstrap or hook. `update_correlations()` is also never called on a cadence.
**Improves:** Detection — normally-uncorrelated domains suddenly correlating is one of the strongest informed-trading signals.
**Effort:** Low. Wire a call to `detect_cross_domain_anomalies()` in `_run_slow_cadence_ops` every 500 steps alongside GrangerAnalyzer. Then publish to a channel (or log). ~15-20 lines.
**Risk:** The tracker needs signals fed to it via `record()`. Verify that signal ingestion actually calls `correlation_tracker.record()`. If not, the output will be empty. Do not assume it is being fed.
**Caveat:** The `seed_from_lag_data()` method pre-populates known structural correlations. This is good — it means the tracker starts with prior knowledge, not cold. But it still needs live signal feeding to compute anomaly departures from those priors.
**Verdict: Medium-high value. Verify feed side before wiring output side.**

---

### #7 — Regime-Aware Execution / Dynamic Position Sizing (Team 5)
**Category:** Trading performance
**What it does:** Signal Translator (signal_translator.py) uses fixed 1.5×ATR stop and 3.0×ATR target regardless of regime. In volatile regimes, ATR expands — stops become dangerously wide. In sideways regimes, targets are never hit.
**Improves:** Execution quality — right-sized for the current environment.
**Effort:** Medium. RegimeClassifier is bootstrapped. Signal Translator is a dataclass-based function (lines 1-80 reviewed). Adding regime parameter to `translate()` and adjusting ATR multiples by regime is ~30 lines. The ADTS regime-aware Thompson (already built at ~45 lines per MEMORY) is the model.
**Risk:** Requires RegimeClassifier to be passed into SignalTranslator at call site in market_hooks. Search for call sites first.
**Mae's Laws:** Compliant — regime-aware sizing is configuration, not a new structural connection.
**Verdict: Medium-high value for live trading readiness. Straightforward.**

---

### #8 — DriftDetector → RegimeClassifier invalidation (Team 3)
**Category:** Inevitability detection
**What it does:** ADWIN DriftDetector detects regime shifts in data streams. RegimeClassifier uses rolling window classification. When drift is detected, RegimeClassifier should be reset/re-bootstrapped. Currently these two systems do not communicate.
**Improves:** Detection accuracy — regime mis-classification during transition periods is a known source of false convergence alerts.
**Effort:** Low. DriftDetector publishes events (or has a query API). RegimeClassifier has a classify() method. The bridge is: on drift event, call `regime_classifier.reset()` or flag current regime as uncertain. ~15 lines.
**Risk:** Regime thrashing if DriftDetector fires too often. Add a cooldown.
**Verdict: Medium value. Worth doing but lower urgency than the learning-loop fixes.**

---

### #9 — Qdrant find_precedents activation (Team 4)
**Category:** Inevitability detection (pattern memory)
**What it does:** OctopusColony has an investigation pipeline. Team 4 claims Qdrant `find_precedents` is never called in live investigations.
**Improves:** Pattern memory depth — MIDGE can compare current developing situations to all historical precedents stored in Qdrant, not just the PatternLibrary JSONL templates.
**Effort:** Medium-high. Requires Qdrant client to be accessible inside OctopusColony investigation path, then embedding the situation description and querying. The Qdrant infrastructure exists globally but must be wired at this call site.
**Risk:** Qdrant query latency must be async (investigations run off main thread — confirmed). Qdrant may be empty until more situations are indexed.
**Caveat:** Team 4 did not verify whether Qdrant has any MIDGE-domain content. If empty, this produces zero value until an indexing pipeline is also built. Verify content before wiring.
**Verdict: Medium value, higher effort, prerequisite gap (Qdrant content). Lower priority than items 1-8.**

---

### #10 — Narrative-Pattern Fusion (Team 5)
**Category:** Detection signal quality
**What it does:** SocialTextAnalyzer extracts keyword themes from StockTwits. PatternWatcher detects historical pattern matches. When a current narrative (e.g., "short_squeeze") is congruent with a historical pattern template, boost that pattern's stack confidence.
**Improves:** Pattern stack quality — filters false positives where the pattern fires but the narrative context is contradictory.
**Effort:** Medium. SocialTextAnalyzer is built (confirmed in MEMORY). PatternWatcher is built. Bridge requires: for each PatternActivation in PatternStack, query social themes for the ticker, score congruence, apply multiplier. ~50-70 lines.
**Risk:** Social data is noisy and sparse. This should be a soft boost (0.9-1.1 multiplier), not a hard gate, or it will suppress valid signals when StockTwits has no data.
**Verdict: Medium value. Lower urgency — improve signal quality after the learning loops are closed.**

---

## Findings That Did NOT Hold Up

### Team 4: "PostMortem sequence_stats → DeepAnalyst"
Not validated. DeepAnalyst is referenced in MEMORY but its code path was not traced. Cannot confirm this connection gap is real or the effort estimate. Defer.

### Team 5: "Somatic Deception Intelligence" and "Predictive Regime Forecasting"
These are architecturally sound ideas but are new capabilities, not wiring existing built code. They belong on the roadmap queue, not the "fix dead wires" list. Both would require significant new code. Not ranked in top 10 for this expedition.

### Team 5: "Causal Sequence Hypotheses"
Partially overlaps with #4 (GrangerAnalyzer → HypothesisGenerator). The causal story component is already handled by `hypothesis_generator.py`'s auto-causal story generation. Not a separate item.

### Team 2: "Qdrant read-back into paper trade gate"
This is a valid future enhancement but requires the Qdrant indexing pipeline first. Cannot rank this without knowing Qdrant content. Same prerequisite gap as #9.

---

## Contradictions Between Teams

| Claim | Teams | Resolution |
|-------|-------|------------|
| VelocityDetector liveness | T1 says alive, T3 says dead wire | Both correct. Input side runs, output side never published. |
| CH_PREDICTION_RESULT effort | T1 says "5 lines" | Code shows 15-25 lines needed (9 subscriber payload formats) |
| CorrelationTracker fed/unfed | T3 assumes fed | Verify: signal ingestion must call ct.record(). Not confirmed. |

---

## Mae's Laws Compliance Check

| Finding | Law risk |
|---------|----------|
| VelocityDetector as convergence domain | **FAIL Law 7** — adds 1 domain to convergence count without 3 validators. Do not inject as domain. Publish to channel only. |
| CH_PREDICTION_RESULT publish | Clean — closes existing triad. |
| GrangerAnalyzer → HypothesisGenerator | Clean — valid triad. |
| Alpaca outcome registration | Clean — valid triad. |
| All others | No Law violations identified. |

---

## Execution Order (What to Do First)

Given finite context and the directive to prove MIDGE works before adding features:

1. **VelocityDetector publish** (~8 lines) — highest value per line of code written. Anomalies already computed, just not emitted.
2. **CH_PREDICTION_RESULT publish** (~20 lines) — closes the bio-learning loop that 9 systems are waiting on.
3. **GrangerAnalyzer → HypothesisGenerator** (~50 lines) — promotes causal findings to testable hypotheses.
4. **Hypothesis outcome tracking closure** (~20 lines) — makes hypothesis retirement data-driven.
5. **CorrelationTracker feed verification + output wiring** — verify feed side first, then wire output.
6. **Alpaca position outcome registration** — most complex of the critical items, but highest-quality feedback signal.
7. **Regime-aware execution** — improves live trading readiness.

Items 8-10 and Team 5 emergent capabilities: deferred until the learning loops in items 1-6 have run for at least one daemon cycle and produced measurable Thompson updates.

---

## Summary Judgment

The 5 teams correctly identified that MIDGE has a large class of "running but not emitting" systems. The core pattern: systems compute output, output goes into a local variable, variable is garbage-collected. The fix in every case is a `bus.publish()` call plus verifying the subscriber payload contract.

The teams overestimated effort on Team 1's CH_PREDICTION_RESULT (5 lines is too low) and underestimated the prerequisite gap for Qdrant work. The VelocityDetector contradiction is real and important: Team 3's "single largest dead wire" framing is the more useful frame for prioritization purposes.

No proposed fix violates Mae's Laws *if* VelocityDetector is published to its channel rather than injected as a convergence domain.
