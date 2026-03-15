# Alpha Revision: REVISE

**Date:** 2026-03-14
**Basis:** Additional code research informed by Lead and Beta cross-reviews. Three specific positions revised with new evidence. One position corrected from Phase 1. One unresolved disagreement between Lead and Beta resolved.

---

## What Changed and Why

### Revision 1: HAVEN — I was wrong. Lead was right.

**My original position (Phase 1):** REMOVABLE. `ctx._haven_market_flags` accumulates on deception events but flags are never read in the signal processing pipeline.

**My cross-review position:** I maintained uncertainty — "this warrants verification" — but leaned toward REMOVABLE.

**What the code shows:**

After tracing the full chain:

1. `bio_market_wiring_b.py`: HAVEN callback fires on `CH_DECEPTION_DETECTED`, writes suspicion scores to `ctx._haven_market_flags`. On `CH_PREDICTION_RESULT` with `won=True`, reduces scores for contributing sources.

2. `bootstrap/market.py` line 154-156: `_alerter.set_haven_flags(ctx._haven_market_flags)` — passes the live dict (by reference) to ConvergenceAlerter at bootstrap time.

3. `convergence_alerter.py` line 264-266: `set_haven_flags()` stores it as `self._haven_flags`.

4. `convergence_confidence.py` lines 419-429: `if self._haven_flags:` — reads the dict, computes `max_suspicion`, and applies `confidence * (1.0 - max_suspicion * 0.2)` when suspicion > 0.5.

The flags ARE read. They DO affect confidence scoring. The chain is complete: deception detected → source flagged → confidence on alerts from that source reduced by up to 20% → successful prediction clears the flag.

**Revised classification:** USEFUL. This is a source-level immune system with a live feedback loop. Lead was correct; Alpha and Beta were both wrong to classify it REMOVABLE.

**Why I was wrong:** I checked whether flags were "read in signal processing" by searching for the flag variable in hook files. The consumer lives in `convergence_confidence.py` — a mixin file — not in the hook files. Same methodology failure that caused me and Lead to miss QuorumSpace's consumer in Phase 1. The lesson from that failure did not protect me from repeating it on HAVEN.

---

### Revision 2: ResourceGovernor — Beta is right, Lead is wrong.

**My original position (Phase 1):** Did not review.

**My cross-review position:** Beta's empirical analysis (INERT) is more defensible than Lead's aspirational one (USEFUL).

**What the code confirms:**

1. `bio_market_wiring.py` lines 110-124: The cortisol → ResourceGovernor coupling is explicitly commented out with: `logger.debug("Layer 33k - Cortisol → ResourceGovernor coupling DISABLED (fictional physiology)")`.

2. Searching the entire `mae_core/` codebase for `resource_governor.can_call` or `resource_governor.record_call`: zero matches. No market code calls either method.

3. `sensing_scheduler.py` lines 126, 152: The live rate-limiting mechanism is `circuit_breaker.can_call()` — not ResourceGovernor.

ResourceGovernor has a well-designed API (`can_call`, `record_call`, `tighten_budgets`, `relax_budgets`), is constructed and registered in bootstrap, and has triadic connections defined — but no code in the live sensing pipeline calls any of its methods. The circuit breaker is doing the rate-limiting job.

**Revised classification:** INERT currently. Desired state is USEFUL (the design is correct). Action item: wire `resource_governor.can_call()` and `resource_governor.record_call()` into the 31 sensing fetchers as the single governed rate-limiting mechanism. This would replace or layer with the circuit breaker.

**Why Lead was wrong:** Lead described ResourceGovernor as "operates independently" and "essential for 24/7 daemon operation" without tracing which code calls its methods. The design intent is there; the wiring is not.

---

### Revision 3: OrganismState — I move to Beta's position.

**My original position (Phase 1):** USEFUL (partial). Thin causal paths exist.

**My cross-review position:** Closer to Beta's HARMFUL — the full chain terminates before producing differentiated output.

**What the cross-reviews established and I now accept:**

Beta traced the complete chain: convergence event → endocrine hormones → emotional valence on OrganismState → DecisionRouter somatic marker valve → reflex tier lookup → no reflex patterns registered → falls through to prefrontal unchanged. The path exists architecturally but produces no differentiated behavior. Every agent receives `body_threat_level ≈ 0.0` and `body_opportunity_level ≈ 1.0` uniformly regardless of market conditions.

I had stopped tracing at "emotional state biases DecisionRouter" without verifying whether the bias produces different output. Beta's adversarial methodology — follow the chain until it either changes a market output or terminates — was more rigorous.

**Revised classification:** HARMFUL (overhead). Not because it produces bad data, but because it consumes 18 EventBus channel subscriptions and adds serialization overhead on every convergence publish for a system whose outputs are uniform. Beta is right: a causal path that produces uniform output is functionally equivalent to no path.

**One partial disagreement I retain:** Beta's framing of this as "harmful" vs. Lead's "overhead" — I align with Lead here. OrganismState is overhead, not harmful. Its outputs being uniform means it wastes CPU but does not corrupt market signals. The distinction matters for prioritization: overhead = clean up when convenient; harmful = fix now.

---

### Revision 4: QuorumSpace — My REMOVABLE classification from Phase 1 stands corrected (already noted in cross-review, confirming here).

Both Beta and Lead found `convergence_confidence.py` lines 200-204: `quorum_space.get_contributor_count(signal_key)` provides a multi-source consensus bonus. My Phase 1 REMOVABLE classification was based on an incomplete trace. **Correct classification: MARGINALLY USEFUL.** This was already acknowledged in my cross-review; I am confirming it here for completeness.

---

### Revision 5: The unresolved Lead vs. Beta disagreement on HAVEN — Now resolved.

In my cross-review I wrote "I concede uncertainty here; I do not concede that Lead is clearly right" about HAVEN. The code trace above resolves this: Lead was right. Beta's methodology of checking wiring files missed the consumer in `convergence_confidence.py`. Both Beta and I made the same error.

---

## Positions I Examined and Am Holding

### I hold: OctopusColony investigation-to-confidence gap is the highest-priority missing wire.

All three auditors noted this. Beta's specific finding — that `_on_octopus_investigation` logs results and populates `_priority_requests` but does NOT route discovered high-win-rate templates into confidence adjustments — stands uncontested. I verified no additional path was found by Lead or Beta in their cross-reviews. This finding stands.

The gap: a 70%+ win-rate historical template found during Octopus investigation does not boost the live convergence alert's confidence. The `_priority_requests` mechanism is real (it boosts sensing for missing domains), but the template finding itself — the intelligence output of the investigation — does not feed back to alter confidence scoring.

**This is the most actionable single fix in the codebase.** Low wiring effort, closes a genuine loop, makes investigation ROI measurable.

### I hold: The 3x convergence buffer scan per step is a real redundancy.

Neither Lead nor Beta contested this finding. `check_convergence()` runs every step, `check_ticker_convergence(min_domains=3)` runs every step, and `check_ticker_convergence(min_domains=2)` runs every 50 steps for Kelly sizing. The Kelly sizing scan is redundant — the min_domains=3 scan already found everything the min_domains=2 scan would find. Action: cadence the Kelly scan to step%50 only and consolidate with the per-ticker convergence pass.

### I hold: _run_synergy_detection() cadence mismatch.

Synergy detection runs every step but reads data refreshed every 10 steps. The per-step frequency produces 9 redundant reads per actual data update. Action: gate it to match the archaeology cadence (every 10 steps).

### I hold: ReproductiveSystem's consume_market_pressure() dead accumulator.

`ctx._consume_market_pressure` is a lambda stored on context but never invoked in any step hook. Pressure accumulates toward 1.0 without being consumed. Neither Lead nor Beta contested this finding. It is a correctness issue, low severity, clean fix.

### I hold: The MemoryConsolidator duplicate hypothesis_engine.step() invocation.

CircadianRhythm's CONSOLIDATION phase triggers `hypothesis_engine.step()` via MemoryConsolidator. But `hypothesis_engine.step()` is already called every step in the main market hook. On CONSOLIDATION phase ticks, the engine runs twice. The engine's internal modulo gates prevent double-work on the same step, but the double invocation is still wasteful. This was caught by me and missed by both Lead and Beta.

### I hold: InhibitionSystem — Lead's ESSENTIAL classification is correct, but I refine the framing.

All three auditors now agree (after cross-reviews): the organism-level `evaluate()` returns Go unconditionally (dead code), but `_market_caution` is a live trade gate. The ESSENTIAL classification applies to the market-caution pathway, not the biological inhibit pathway. The distinction matters for any future work: the biological evaluate() can be simplified without affecting trading; the `_market_caution` float must be preserved and its deception-event triggering path kept intact.

---

## Updated Position Summary

| System | Phase 1 | Cross-Review | Revision | Reason |
|--------|---------|--------------|----------|--------|
| HAVEN flags | REMOVABLE | Uncertain | **USEFUL** | `convergence_confidence.py` lines 419-429 confirmed as active consumer — flags DO reduce confidence on suspicious sources |
| ResourceGovernor | Not reviewed | INERT (aligned with Beta) | **INERT currently / USEFUL desired** | Zero calls to `can_call()`/`record_call()` in live sensing code; cortisol coupling explicitly commented out |
| OrganismState | USEFUL (partial) | Closer to Beta's HARMFUL | **OVERHEAD (align with Lead's framing)** | Full chain terminates at empty reflex library; outputs uniform; Beta's trace was correct |
| QuorumSpace | REMOVABLE | MARGINALLY USEFUL (corrected) | **MARGINALLY USEFUL** | `convergence_confidence.py` has active consumer confirmed in cross-review |
| InhibitionSystem | USEFUL (low) | ESSENTIAL (market-caution path) | **ESSENTIAL (market-caution path only)** | Lead's argument stands; biological evaluate() is dead code, market-caution pathway is live trade gate |

---

## Final Statement on the Central Finding

The meta-conclusion is unchanged from my Phase 1 and is reinforced by all three peer reviews: MIDGE's market intelligence pipeline is sound and the organism scaffolding is predominantly overhead for that pipeline. The three lenses — pipeline mapping (Lead), adversarial stress-testing (Beta), and cost/value quantification (Alpha) — converged independently on this.

The actionable path has two distinct buckets:

**Bucket 1 — Overhead reduction (mechanical cleanup):**
- Remove inert bio system callbacks from `CH_CONVERGENCE` (~12 callbacks, no market value)
- Remove or stub `EnergyReserve.step()` constant publish
- Gate `_run_synergy_detection()` to every 10 steps
- Remove `SacredGeometry.bootstrap_k4_tetrahedra()` (403 dead lines)
- Consolidate the 3x convergence buffer scan

**Bucket 2 — Missing wire completion (highest-value):**
- Wire OctopusColony investigation findings → confidence adjustment in ConvergenceAlerter (closes investigation ROI loop)
- Wire `resource_governor.can_call()`/`record_call()` into sensing fetchers (activates the designed rate governance)
- Register market-specific reflex patterns on OrganismState's DecisionRouter (activates the EndocrineSystem → behavior pipeline that Beta showed is 10 lines from working)
- Fix `_consume_market_pressure` accumulation

Bucket 2 is higher priority than Bucket 1. Overhead reduction improves performance. Missing wire completion improves intelligence output — which is what determines whether MIDGE finds inevitabilities.
