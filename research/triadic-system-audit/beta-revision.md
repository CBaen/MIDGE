# Beta Revision: REVISE

**Auditor:** Witness Beta (Devil's Advocate)
**Date:** 2026-03-14
**Phase 3 revision based on:** beta-findings.md, beta-cross-review.md, lead-cross-review.md, alpha-cross-review.md
**Additional research conducted:** Traced `_haven_market_flags` full consumer chain in `convergence_confidence.py` lines 419-429. Verified `resource_governor.can_call()` / `record_call()` call sites across all market Python files.

---

## What Changed and Why

Four positions changed. Two held. All changes are driven by peer evidence plus additional research I conducted in Phase 3 to resolve the remaining disputes.

---

### Change 1: HAVEN — REMOVABLE → USEFUL (significant reversal)

**My Phase 1 position:** HAVEN flags accumulate in `ctx._haven_market_flags` but nothing reads them in the signal processing pipeline.

**What I missed:** In my Phase 2 cross-review, I noted the dispute between Lead (USEFUL) and Alpha (REMOVABLE) and said I "concede uncertainty here." I conducted additional research in Phase 3 to resolve this.

**What the code shows:** `convergence_confidence.py` lines 419-429 read `self._haven_flags` — which is the same dict passed via `set_haven_flags()` in `bootstrap/market.py` line 155-156. The consumer is:

```
if self._haven_flags:
    max_suspicion = max(score for src, score in self._haven_flags.items())
    if max_suspicion > 0.5:
        confidence = max(0.05, confidence * (1.0 - max_suspicion * 0.2))
```

This is a real, quantitative confidence penalty — up to 20% downweight when any flagged source has suspicion > 0.5. The flag accumulation chain is fully closed: deception event → `_on_deception()` in `bio_market_wiring_b.py` adds severity to `ctx._haven_market_flags[source]` → same dict is `self._haven_flags` on ConvergenceAlerter (shared reference set at bootstrap) → confidence calculation reads and acts on it.

**Alpha and I were both wrong** on the same point: we both confirmed the flags are SET but concluded they are never READ. We missed `convergence_confidence.py` because neither of us traced into the confidence calculation internals. Lead was right. Beta (me) and Alpha were wrong.

**Revised verdict:** USEFUL. HAVEN provides genuine source-level trust modulation with a direct, quantitative effect on convergence confidence. It is analogous to InhibitionSystem's `_market_caution` — a bio-system that found a real market application. Both should remain.

**Caveat that stands:** The flags accumulate unboundedly per Alpha's finding. A successful prediction clears 0.2 per source (bio_market_wiring_b.py lines 149-152), but deception events can add unlimited severity. A source that fires CH_DECEPTION_DETECTED repeatedly could accumulate infinite suspicion that is never fully cleared. The accumulation cap is a real correctness gap even though the read path is live.

---

### Change 2: HAVEN flags accumulation cap — new finding

**New finding (Phase 3 only):** `ctx._haven_market_flags[source]` has no upper bound. Severity accumulates on every deception event. The clearing mechanism (`-= 0.2` per winning prediction, only if that source is in the prediction's `sources` list) is asymmetric — one deception event could add 0.5+ suspicion, requiring 2-3 winning predictions on that specific source to clear. A source that rarely appears in prediction outcomes (e.g., `cot_client`) could accumulate suspicion to 5.0+ with no clearance path.

**Recommended fix:** Cap suspicion at 1.0 (`min(1.0, flags.get(source, 0.0) + severity)`) in `_on_deception()`. This is a 1-line correctness fix.

---

### Change 3: ResourceGovernor — INERT (confirmed, not changed)

**My Phase 1 position:** INERT. Cortisol-coupling is disabled. No market code calls its rate-limiting methods.

**Lead's position:** USEFUL. "Operates independently."

**Alpha's revised position (Phase 2):** INERT currently / USEFUL desired state. "Beta's empirical analysis beats Lead's aspirational one."

**Phase 3 verification:** I searched all market Python files for `resource_governor.can_call`, `resource_governor.record_call`, and `resource_governor.register_source`. Zero matches. The only file containing `can_call` and `record_call` is `resource_governor.py` itself (in docstring examples) and the unrelated `circuit_breaker.py`. The `sensing_scheduler.py` calls `_cb.can_call(source)` — that `_cb` is the CircuitBreaker, not the ResourceGovernor.

**Conclusion:** INERT is correct. ResourceGovernor is constructed, registered in connections, and sits on `ctx.resource_governor` — but its rate-limiting API (`can_call` / `record_call` / `register_source`) is never called by any sensing or intelligence code. Rate limiting in the live system is handled entirely by CircuitBreaker. Lead's USEFUL classification is aspirational: it describes what ResourceGovernor should do, not what it does. Alpha and I align: INERT now, wiring it up is an action item, not a reason to mark it currently useful.

**I hold my Phase 1 position on ResourceGovernor.**

---

### Change 4: Organism architecture verdict — HARMFUL → OVERHEAD (precision correction)

**My Phase 1 position:** The organism is a net liability to the market mission.

**What changed:** In my Phase 2 cross-review, I already noted this needed refinement — "I conflated 'overhead' (unnecessary but not harmful) with 'harmful' (actively causes failures)." I was writing toward this change but hadn't committed to it.

**Full revision:** Lead's cross-review made the strongest version of this argument: "Beta's adversarial lens treated the organism as a category error — the organism solves a different problem than MIDGE actually faces." This is a more precise framing than mine. The 14 `# MIDGE: disabled — fictional physiology harms trading daemon` comments document genuine harms that were found and neutralized. The remaining inert systems are overhead, not ongoing harm.

**Revised verdict:** The organism is OVERHEAD at current scale, not HARMFUL. Harmful systems have been actively disabled. The currently-running inert systems cost CPU cycles but do not cause incorrect trading decisions. The distinction matters for prioritization: harmful systems require urgent removal; overhead systems are cleanup when convenient (or when performance constraints demand it).

**What I maintain from Phase 1:** The directional claim stands — the organism as implemented does not serve the market mission, and the minimum viable MIDGE is approximately 20 files, not 649. This is an architectural observation, not a removal directive.

---

### Change 5: OctopusColony output gap — STAND FIRM (with refinement)

**My Phase 1 position:** Investigation results go to logs, not to confidence adjustments. A 70% win-rate template found during investigation does not boost the related convergence alert's confidence.

**Lead's response (Phase 2):** Lead agreed this is "the most operationally significant gap in the investigation pipeline" and called it "the highest-priority fix in the entire audit."

**Alpha's response (Phase 2):** Alpha confirmed "the investigation pipeline produces log output, not pipeline-boosted confidence."

**No peer challenged this finding.** All three auditors now agree the output gap is real. I maintain the position.

**Refinement:** The investigation DOES have one real downstream effect — `_priority_requests` for Focused Attention (Sensing Hook priority polling). This is real and produces measurable behavior change (which sources get polled more aggressively). But the investigation finding itself — the template with historical win rate — does not feed back to confidence. The useful 5% of the pipeline works; the valuable 95% (what the investigation actually LEARNED) terminates at a log line.

---

### Change 6: Three-position table — InhibitionSystem, EmotionalSystem, OrganismState

These were covered in Phase 2 cross-review and no new evidence arose in Phase 3. Confirming the Phase 2 revisions as final:

| System | Phase 1 | Phase 3 Final | Basis |
|--------|---------|---------------|-------|
| InhibitionSystem | Dead code | USEFUL (market_caution pathway is live) | Lead's Phase 1 finding, confirmed Phase 2 |
| EmotionalSystem | INERT | INERT (dead end at reflex lookup) | Beta traced full chain; Lead and Alpha confirmed Phase 2 |
| OrganismState | HARMFUL | OVERHEAD (outputs uniform but don't cause wrong trades) | Alpha + Lead + Beta Phase 2 convergence |
| HAVEN | REMOVABLE | USEFUL (live confidence penalty) | Phase 3 code verification |
| ResourceGovernor | INERT | INERT (confirmed by Phase 3 search) | Phase 3 zero-match search |

---

## Final Revised Positions

### Core pipeline (no change from Phase 1)

**ESSENTIAL — unchanged:** ConvergenceAlerter, ThompsonSampler, OutcomeCollector, AlpacaClient, SensingHook, DrawdownMonitor, CircuitBreaker, PlainLanguageFormatter, RegimeClassifier.

**ESSENTIAL — unchanged:** PatternArchaeology pipeline (PatternWatcher, PatternLibrary, Excavator, ExcavationDaemon), WorldModel (market), GrangerAnalyzer, CascadeTracker, HypothesisEngine.

### Systems where my position changed

**HAVEN: REMOVABLE → USEFUL**
Evidence: `convergence_confidence.py` lines 419-429 confirm the flags are read and applied as a quantitative confidence penalty. The consumer exists and is active. Fix the accumulation cap.

**Organism architecture: HARMFUL → OVERHEAD**
Evidence: Peer review + code analysis confirms harms were already neutralized. Running cost is real; active harm is historical.

### Systems where I hold Phase 1 position

**ResourceGovernor: INERT (confirmed)**
No callers found for `can_call()` or `record_call()`. CircuitBreaker is the active rate-limiting mechanism. ResourceGovernor is constructed overhead.

**OctopusColony output gap: STAND FIRM (all auditors now agree)**
Investigation findings go to logs, not to confidence adjustments. The missing wire is the highest-value unbuilt connection in the system.

**Agent architecture: OVERHEAD / near-zero market value (confirmed)**
All three auditors agree. Agents produce no market alerts. Market intelligence runs in step hooks, independent of agent count or step.

**EnergyReserve: HARMFUL OVERHEAD (confirmed)**
CH_ENERGY_STATUS publishes unconditionally every call. No market value. All three auditors agree.

**CH_CONVERGENCE subscriber density: highest-priority cleanup (confirmed)**
~12 inert bio callbacks fire on every convergence alert. All three auditors agree this is the single highest-value mechanical cleanup.

---

## Net Effect of Revision

Phase 1 was directionally correct. Three findings changed:

1. HAVEN is USEFUL, not REMOVABLE — I missed the consumer in `convergence_confidence.py`. This is the same error Alpha made. The lesson: do not conclude "no consumer" without searching confidence calculation internals, not just wiring files.

2. ResourceGovernor is confirmed INERT by Phase 3 search. Lead's USEFUL categorization was aspirational.

3. "Harmful organism" is more precisely "overhead organism" — harms were already addressed. The urgency is different.

The headline finding survives revision: MIDGE's market intelligence pipeline is capable and well-architected. The organism is mostly overhead for that pipeline. The two highest-priority actions remain:
- **Remove inert bio callbacks from CH_CONVERGENCE** (performance — immediate)
- **Wire OctopusColony investigation findings to convergence confidence** (value completion — highest ROI)

---

*End of Beta Phase 3 revision.*
