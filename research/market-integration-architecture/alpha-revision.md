# Alpha Revision — Bootstrap Integration & Law Compliance
**Witness Alpha | Phase 3 Revision**
**Date: 2026-02-22**
**Based on: lead-cross-review.md, alpha-cross-review.md, beta-cross-review.md**

---

## Revision Summary

**Decision: REVISE on five specific issues. STAND FIRM on three others.**

The cross-reviews were genuinely productive. Beta's adversarial lens found execution-layer failures I missed by focusing on registration-layer design. Lead identified the architectural sequencing dependency I treated as an implementation detail. Both changes require material updates to my original findings.

The revisions below are precise — each one names what changed, what did not, and why.

---

## REVISION 1: Channel Names — Accept Lead's Namespace, Reject Mine

**What changed:** My Part 9 proposed EventBus channel names:
```
market.cluster_signal
market.politician_trade
market.contract_prediction
market.filing_signal
market.convergence_alert
market.thompson_stats
market.velocity_anomaly
```

Lead's architecture proposed:
```
market.edge.cluster_detected
market.edge.correlation_found
market.edge.filing_anomaly
market.edge.contract_predicted
market.intel.velocity_anomaly
market.intel.convergence
market.intel.actionable
```

**Why I revise:** Beta identified this as COMPLETE PIPELINE FAILURE (cross-review 5.1). Both namespaces are internally consistent, but they are incompatible. If my bootstrap publishes to `market.cluster_signal` while ConvergenceAlerter subscribes to `market.edge.cluster_detected`, zero signals flow and the system logs success. That is the worst kind of failure — silent and confident.

Lead's namespace has two advantages that tip the decision:

1. The `market.edge.*` / `market.intel.*` hierarchy matches the physical package structure (`edge/` and `intelligence/`). A developer reading the code can predict channel names from the file they're in.
2. `market.intel.actionable` is a channel my design lacked entirely — it names the final synthesis output explicitly, which matters for downstream consumer discoverability.

**Revised channel list (authoritative for bootstrap wiring):**

| Publisher | Channel | Subscribers |
|-----------|---------|-------------|
| ClusterDetector | `market.edge.cluster_detected` | ConvergenceAlerter, step hook |
| PoliticianTracker | `market.edge.politician_trade` | ConvergenceAlerter |
| FilingTimeAnalyzer | `market.edge.filing_anomaly` | ConvergenceAlerter |
| ContractPredictor | `market.edge.contract_predicted` | ConvergenceAlerter |
| VelocityDetector | `market.intel.velocity_anomaly` | ConvergenceAlerter, step hook |
| ConvergenceAlerter | `market.intel.convergence` | EndocrineSystem, KnowledgeBase, step hook |
| ConvergenceAlerter | `market.intel.actionable` | Agent decision cascade |
| ThompsonSampler | `market.intel.thompson_stats` | Audit/monitoring only |

**What did NOT change:** The subscribe/publish architecture itself — ConvergenceAlerter subscribes to edge channels and publishes to intel channels. The topology is identical. Only the string names change.

**Impact on Part 9:** Every channel string in the connection list must be updated to the revised names above before the connection registration code is written.

---

## REVISION 2: Signal Normalizer Is a Pre-Condition, Not an Implementation Detail

**What changed:** My original findings treated the signal normalization problem as "adapters to be resolved in implementation" when writing the `_register_market_eventbus` section. I named no specific file and no specific sequence dependency.

**Why I revise:** Lead and Beta both independently reached the same structural conclusion: the `MarketSignal` dataclass is the data contract that all publishers and subscribers agree on. Without it existing as code first, the EventBus callbacks have no agreed type to pass. My bootstrap design would register the callbacks correctly but the callbacks themselves would receive `ClusterSignal`, `ContractPrediction`, `FilingTimeSignal` objects with incompatible shapes when `ConvergenceAlerter.record_signal()` expects six specific parameters.

This is a sequencing failure, not a wiring failure. The skeleton I designed is correct; the tissue that carries signals through it does not exist yet.

**Revised Layer 33 implementation sequence (replaces Part 16):**

**Step 0 — Pre-conditions (before writing any bootstrap code):**
- Fix `trade.is_purchase` AttributeError (contract_predictor.py:232, politician_tracker.py:276)
- Replace all `http://localhost:6333` hardcodes with configurable Qdrant URL parameter
- Replace all `print()` with `logging.getLogger(__name__)`
- Set `DEFAULT_PRIOR_SCALE = 2` AND `min_variance = 0.001` in thompson_sampler.py (must change together — Beta's finding, confirmed in my cross-review)
- Normalize VelocityDetector output to per-day units (divide `dt` by 86400 before velocity calculation)
- Add `try/except` around HTTP client construction in PoliticianTracker and ContractPredictor `__init__`

**Step 1 — Signal contract:**
- Create `mae_core/market/signal.py` with `MarketSignal` dataclass (Lead's design from their Part 6)
- Create adapter functions: one per source type, translating raw dataclasses to `MarketSignal`
- The congressional trade adapter must use `transaction_date`, NOT `disclosure_date`, for the `timestamp` field (Lead's 45-day finding, confirmed by my cross-review surprise 3)

**Step 2 — Interface adapters (additive, non-breaking):**
- Add `get_statistics()` methods to ThompsonSampler, ConvergenceAlerter, VelocityDetector
- Note: `ConvergenceAlerter.step()` should return None, not `len(alerts)` — the model.py step hook dispatcher at line 119-121 ignores return values, so this is not a runtime error, but returning a count from a method typed `-> None` is an interface lie. Corrected:
```python
def step(self) -> None:
    """Step hook for periodic convergence check (called by HolonProxy.act())."""
    self.check_convergence()
```
- The bootstrap step hook handles alert publishing separately — `step()` here is only for the HolonProxy `act()` delegation, not the main operational loop.

**Step 3 — Stem cell roles:**
- Add SEC_WATCHER, CONTRACT_TRACKER, MARKET_ANALYST to stem_cell.py ROLE_PROFILES (unchanged from Part 5)

**Step 4 — Bootstrap module (market.py):**
- Create `mae_core/bootstrap/market.py` following the original Part 8 structure
- Instantiate `ConvergenceAlerter` with explicit `min_domains=3` (not the default 2) — this is where Mae's Law 2 is enforced as a constructor constraint, not left to the module default

**Step 5 — Step hook with deduplication (replaces Part 7):**

See Revision 3 below for the full corrected hook design.

**Step 6 — main.py integration:**
- Add `bootstrap_market(ctx)` between `bootstrap_external` and `bootstrap_audit`
- Update `_build_systems_dict()`
- Update `test_integration.py` atomically with the bootstrap change — Beta's finding (cross-review 5.8) is correct that the test suite fails on first run if this update is deferred

**Step 7 — Document parity:**
- Update all 7 tracked files simultaneously (counts from Part 14 remain accurate: +14 systems, +17 holons, +23 connections, bootstrap layers 32→33, organs 5→6, stem cell roles 9→12)

**What did NOT change:** The Layer 33 module structure (8 private functions), the triadic connection list (23 connections in Group 14 — though channel names update per Revision 1), the fractal placement (Option A still correct), the holon registration order, and the document parity impact table.

---

## REVISION 3: Step Hook Must Deduplicate — Design Upgraded

**What changed:** My original Part 7 step hook called `ctx.convergence_alerter.check_convergence()` every step with no deduplication guard.

**Why I revise:** Beta's finding (cross-review 5.2, original findings Section 3) is correct and the consequence is severe: if a convergence condition holds for N steps, N identical alerts are published to `market.intel.convergence`, triggering N endocrine hormone releases. The EndocrineSystem accumulates dopamine or adrenaline proportional to run length, not to market events. Behavioral state becomes dominated by simulation cadence rather than information content.

Beta's adversarial framing identified this as the combination failure — neither my step hook design nor Lead's ConvergenceAlerter design is wrong in isolation, but wiring them together without deduplication creates the amplifier. The fix belongs in the bootstrap layer (where the polling cadence is defined), not in ConvergenceAlerter itself (though Beta's recommendation to add deduplication there too is also correct as defense-in-depth).

**Revised step hook (replaces Part 7 hook design):**

```python
_market_step_counter = [0]
_last_convergence_state = [None]  # {"direction": str, "strength": float}

def _market_sense_hook():
    _market_step_counter[0] += 1
    step = _market_step_counter[0]

    # Every step: check convergence (lightweight, pure in-memory)
    if hasattr(ctx, "convergence_alerter") and ctx.convergence_alerter is not None:
        try:
            alerts = ctx.convergence_alerter.check_convergence()
            for alert in alerts:
                last = _last_convergence_state[0]
                is_new_direction = (last is None or last["direction"] != alert.direction)
                is_material_change = (last is not None
                                      and abs(last["strength"] - alert.strength) > 0.1)
                if is_new_direction or is_material_change:
                    ctx.bus.publish("market.intel.convergence", alert.to_dict())
                    _last_convergence_state[0] = {
                        "direction": alert.direction,
                        "strength": alert.strength,
                    }
        except Exception:
            logger.debug("Convergence alerter step failed", exc_info=True)

    # Every 10 steps: Thompson sampler stats
    if step % 10 == 0:
        if hasattr(ctx, "thompson_sampler") and ctx.thompson_sampler is not None:
            try:
                stats = ctx.thompson_sampler.get_stats()
                ctx.bus.publish("market.intel.thompson_stats", stats)
            except Exception:
                logger.debug("Thompson sampler stats step failed", exc_info=True)

    # Every 50 steps: Velocity detector anomaly scan
    if step % 50 == 0:
        if hasattr(ctx, "velocity_detector") and ctx.velocity_detector is not None:
            try:
                anomalies = ctx.velocity_detector.detect_velocity_anomalies()
                if anomalies:
                    ctx.bus.publish("market.intel.velocity_anomaly",
                                    {"anomalies": len(anomalies)})
            except Exception:
                logger.debug("Velocity detector step failed", exc_info=True)

ctx.model.add_step_hook(_market_sense_hook)
```

**Key changes from original:**
1. Deduplication state `_last_convergence_state` suppresses repeat publications of unchanged conditions
2. Publish channel updated to `market.intel.convergence` (Revision 1)
3. All `ctx.*` access guarded with `hasattr` + None check for graceful degradation (see Revision 4)
4. Thompson stats channel updated to `market.intel.thompson_stats`
5. Velocity anomaly channel updated to `market.intel.velocity_anomaly`

**What did NOT change:** The Fibonacci-style cadence structure (1/10/50 steps). The separation of API calls from step hooks (network calls remain agent-triggered, not hook-triggered). The closure-captured `_market_step_counter` pattern.

---

## REVISION 4: Graceful Degradation Must Be Real, Not Deferred Failure

**What changed:** My original Part 8 described graceful degradation as try/except around instantiation, producing None on ctx if construction fails. I framed this as adequate.

**Why I revise:** Beta (cross-review section 1.3) identified the failure mode I missed: several constructors succeed even when their dependencies are unavailable. Specifically, `ClusterDetector.__init__()` initializes Qdrant connection parameters but makes no HTTP call. Construction succeeds. The first call to `find_clusters()` in the step hook fails silently — and my original step hook had no None check at the point of use. The log line "graceful degradation: 0 failed" would appear even when Qdrant is down, because the failure is deferred to operational use, not bootstrap time.

This means my stated graceful degradation guarantee was false. The bootstrap would report health for a dead system.

**Two fixes required:**

First, at construction time, Qdrant-dependent systems should receive the Qdrant URL from ctx rather than using a hardcoded default:
```python
qdrant_url = getattr(ctx, "qdrant_url", "http://localhost:6333")
ctx.cluster_detector = ClusterDetector(qdrant_url=qdrant_url)
ctx.filing_time_analyzer = FilingTimeAnalyzer(qdrant_url=qdrant_url)
```

This is possible because the Foundation layer already establishes the Qdrant connection and ctx carries the URL — the fix is one parameter per constructor, not a full refactor.

Second, at operational use time, the step hook must guard all ctx attribute accesses with None checks (already incorporated into the revised hook in Revision 3 above).

**What this does NOT fix:** The deferred failure for Qdrant-dependent `find_clusters()` calls at first use. If Qdrant is genuinely unavailable, the first `find_clusters()` call in a step hook or agent action will throw. The correct behavior here is: the try/except in the step hook catches it, logs at DEBUG, and continues — the market system is silently inactive for that step. This is adequate for advisory-mode operation. What is NOT adequate is claiming "graceful degradation: 0 failed" at bootstrap time when Qdrant failures will only surface later.

**Corrected log message:**
```
Layer 33a - Market systems instantiated: 14 systems
            (construction-time failures: 0 | operational dependencies: Qdrant, RAPIDAPI_KEY, ALPHA_VANTAGE_KEY, SAM_GOV_API_KEY — failures deferred to first use)
```

This accurately distinguishes construction-time health from operational-time health.

---

## REVISION 5: Alpha's Registered Connections Are Advisory-Compliant, Not Law-Compliant — Acknowledge the Gap

**What changed:** My Part 9 connection list claimed all 23 connections as "Law 1 compliant." Beta's cross-review (section 1.2) correctly identified that three modules — ClusterDetector, FilingTimeAnalyzer, ContractPredictor — make direct HTTP calls to Qdrant that bypass every registered connection path. The registration says the data flows through BoundaryMembrane and the connection is witnessed. The actual data flows through `requests.post("http://localhost:6333/...")` with no witnesses.

**Why I revise:** I stated "advisory enforcement — triads and connections observe/report, never block" as a fact about the architecture, and used it to justify post-seal registration. That is accurate. But Beta's point is sharper: a registered connection that no actual data flows through is not an advisory-compliant connection — it is a phantom connection. The organism believes it is witnessing data that it is not seeing.

**The distinction matters:**
- Advisory mode: the connection is real, enforcement is soft (warns instead of blocks)
- Phantom connection: the connection is registered but the actual data takes a different, unregistered path

My original design produces phantom connections for the three Qdrant-calling modules until the hardcoded URLs are replaced with `ctx.qdrant_url` (Revision 4). After that replacement, the data actually flows through the bootstrap-configured path, and the registered connections become real (advisory-compliant).

**Revised claim for Part 9:**
The 23 connections are Law 1 structurally compliant (each has source, target, and minimum 2 witnesses). They become operationally honest only after the hardcoded Qdrant URLs are replaced in the pre-condition step. The implementation sequence in Revision 2 (Step 0) must complete before the connections in Part 9 are registered, or the registration produces phantom compliance.

**What did NOT change:** The connection topology itself (23 connections, 3 K3 subsystem triads, cross-subsystem paths). The witness assignments. The post-seal registration pattern. Only the compliance claim is narrowed: structural compliance now, operational compliance after pre-conditions.

---

## STAND FIRM 1: Fractal Placement — Option A Is Correct

Lead endorsed Option A (extend `organ-cluster-cognitive` to K3 by adding `market-intelligence-system`). Beta acknowledged the biological coherence argument holds.

I stand on this for the reason my original findings stated and Lead explicitly confirmed: adding `market-intelligence-system` as the third child of `organ-cluster-cognitive` does not merely extend the hierarchy — it repairs a pre-existing bare dyad violation at the cluster level. `cognitive-system` and `sensory-system` are currently two children with no witness. This is Law 1 non-compliant at the organ level.

The fractal placement recommendation is not a design choice; it is a structural correction. Option A is correct.

My cross-review noted one alternative I had not fully explored: placing `market-intelligence-system` as a child of `sensory-system` rather than a peer to it. I considered this and reject it. `sensory-system` handles internal sensory signals (pain, proprioception, body state). Market intelligence handles external information environment modeling. Placing external world-modeling inside internal sensing would violate the biological coherence I used to justify Option A in the first place. Market intelligence senses AND reasons — it belongs at the cluster level alongside cognitive-system and sensory-system, not subordinate to either.

No revision. Option A stands.

---

## STAND FIRM 2: Seal Timing and Post-Seal Registration Pattern

Beta did not address this. Lead did not address this. My finding stands without challenge.

The seal boundary (wiring.py line 517) is the point after which enforcement transitions from PERMISSIVE to ADVISORY. Layer 33 runs post-seal — same as external.py Layer 31, same as audit.py Layer 32. The market connections register in ADVISORY mode, which is correct and consistent with every other post-seal registration in the codebase.

The specific consequence I identified remains accurate: Layer 33 must NOT attempt to call `connection_registry.seal()` (it is already sealed) and must NOT call `register_connection()` on systems that do not yet exist in SomaticMap. The somatic registration in `_register_market_somatic()` must complete before `_register_market_connections()` runs — this ordering is a hard constraint, not a preference.

No revision. The seal timing analysis stands.

---

## STAND FIRM 3: OutcomeTracker Gap — Acknowledged but Deferred Is Correct

Beta's cross-review (5.3) identifies that OutcomeTracker does not exist and the feedback loop never closes without it. This is a correct finding. My original design (Part 11 — the FRL-to-Thompson feedback path) acknowledged the gap implicitly by routing FRL rewards into Thompson updates as a partial substitute.

Beta frames this as "LEARNING NEVER HAPPENS." I want to be precise: **Bayesian calibration from real market outcomes never happens** is accurate. The Thompson Sampler does update — it updates from FRL reward signals when agents take market-influenced actions. This is not the same as measuring whether a specific predicted price direction was correct, but it is not zero learning either.

The OutcomeTracker is a Phase 2 component because:
1. It requires a scheduled task that runs N days after each prediction (outside the simulation step loop)
2. It requires PriceFetcher integration for ground truth
3. Lead correctly identified that `price_fetcher_for_outcomes()` (price_fetcher.py line 263) already has the right shape for this function — it accepts a symbol and prediction date and returns price change over a forward window

The OutcomeTracker's implementation path is clear (use `price_fetcher_for_outcomes()`, store results in `outcomes.jsonl`, call `thompson_sampler.update()`). Its deferral is deliberate: wiring a scheduled asynchronous task into a synchronous simulation bootstrap is a separate architectural problem. Deferring it to Phase 2 is correct — but Beta is right that it must be named as a gap in Layer 33's capabilities, not quietly assumed away.

**Addition to Part 8 log output:**
```
Layer 33  - Market Intelligence organ complete: 14 systems, 17 holons, 23 connections
            NOTE: OutcomeTracker (Bayesian feedback from real outcomes) deferred to Phase 2.
            Thompson Sampler updates from FRL rewards as interim substitute.
```

This is not a revision of my standing design — it is a documentation addition that names the gap explicitly rather than leaving it implicit.

---

## What Remains Unchanged

The following elements of my original findings required no revision based on the cross-reviews:

- The full Layer 33 module structure (`bootstrap_market(ctx)` and its 8 sub-functions)
- The 13 instantiable systems and their ctx attribute names (Part 8 table)
- All three K3 subsystem groupings (market-sensing, market-edge, market-learning)
- The three new stem cell role profiles (SEC_WATCHER, CONTRACT_TRACKER, MARKET_ANALYST)
- The EndocrineSystem coupling via `market.intel.convergence` EventBus callbacks
- The `generate_triad()` API approach for fractal registration at Layer 33 runtime
- The document parity impact table (counts remain accurate, channel strings update per Revision 1)
- The risk of non-triadic advisory warnings for modules outside the three clean K3s
- The `get_statistics = get_stats` alias pattern for HolonProxy delegation

---

## Synthesized Priority Order (Revised)

This replaces the priority order in my cross-review Section 5.

### Pre-conditions (must precede any Layer 33 code):

1. Fix `trade.is_purchase` and `trade.shares_traded` AttributeErrors
2. Replace all hardcoded `http://localhost:6333` with configurable Qdrant URL parameter
3. Replace all `print()` with `logging.getLogger(__name__)` across 16 market files
4. Set `DEFAULT_PRIOR_SCALE = 2` AND `min_variance = 0.001` together
5. Fix VelocityDetector per-second → per-day unit normalization
6. Add `try/except` around HTTP client construction in PoliticianTracker and ContractPredictor
7. Audit and decide on `thompson_distributions.json`: either commit the current file as canonical starting state, or reset to seeding from `learning_config.py` with reduced prior scale (Beta's recommendation; I endorse it)

### Layer 33 implementation (after pre-conditions):

8. Create `mae_core/market/signal.py` with `MarketSignal` dataclass + adapter functions
9. Adapter for congressional trades must use `transaction_date` for `timestamp` field
10. Add `get_statistics()` methods; fix `ConvergenceAlerter.step()` to return None
11. Add three stem cell roles to stem_cell.py
12. Create `mae_core/bootstrap/market.py`
13. Instantiate `ConvergenceAlerter` with explicit `min_domains=3`
14. Use revised channel names (Revision 1) throughout connection list
15. Use deduplicated step hook (Revision 3)
16. Log construction-time vs. operational-time degradation separately (Revision 4)
17. Update `test_integration.py` atomically with the bootstrap addition
18. Add `bootstrap_market(ctx)` to main.py between external and audit
19. Update `_build_systems_dict()` in main.py
20. Update all 7 document parity files

### Phase 2 (deferred):

21. Implement OutcomeTracker using `price_fetcher_for_outcomes()` as foundation
22. Full ApiGateway routing for six market-specific API clients
23. Bayesian forgetting / multiplicative decay in ThompsonSampler
24. Evaluate ContractPredictor decomposition vs. retention as domain-specific sub-convergence unit
25. Regime-aware Thompson distributions (separate Beta per market regime)

---

## Final Assessment

The triadic review found five issues that required material changes to my original design:

| Issue | Action | Reasoning |
|-------|--------|-----------|
| Channel name mismatch with Lead | Revise — adopt Lead's namespace | Silent pipeline failure if not resolved before code is written |
| Signal normalizer sequencing | Revise — promote to Step 1 in sequence | Bootstrap callbacks have no agreed type without it |
| Step hook deduplication | Revise — add `_last_convergence_state` guard | Hormone amplifier from loop repetition without it |
| Graceful degradation claim | Revise — distinguish construction vs. operational health | "0 failed" log is false when Qdrant is down |
| Phantom connection claim | Revise — narrow compliance claim to structural | Operational honesty requires pre-condition fixes first |

Three issues required no revision:

| Issue | Action | Reasoning |
|-------|--------|-----------|
| Fractal Option A placement | Stand firm | Repairs pre-existing bare dyad; endorsed by Lead |
| Seal timing analysis | Stand firm | Unchallenged; consistent with codebase patterns |
| OutcomeTracker deferral | Stand firm with documentation addition | Phase 2 is correct; gap now named explicitly |

The combined three-investigation synthesis produces an implementable plan. No single investigator had the complete picture. The integration failures Beta found in the combination (5.1 channel mismatch, 5.2 hormone amplifier) are the most valuable outputs of the triadic structure — neither would have been visible from inside any single lens.
