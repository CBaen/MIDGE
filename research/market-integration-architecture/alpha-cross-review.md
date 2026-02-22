# Alpha Cross-Review: Bootstrap Integration & Law Compliance
**Witness Alpha | Phase 2 Cross-Review**
**Date: 2026-02-22**

---

## Preamble

This review compares my findings (Bootstrap Integration & Law Compliance) against Lead's findings (Signal Architecture) and Beta's findings (Adversarial/Tuning). The review is organized by divergence first, then agreement, then gaps, then surprises — the order that produces the most useful synthesis.

---

## Section 1: Reasoning Divergence Points

These are places where the three investigators, working independently, reached different conclusions or placed different emphasis. These are the places the triadic structure is designed to surface.

---

### Divergence 1: The Signal Normalizer — Where Does It Live?

**Lead's conclusion (Part 6):** Designs a full `MarketSignal` dataclass with 18 fields as the normalization layer. Treats it as a new component that bridges all data shapes into the `ConvergenceAlerter.record_signal()` interface. Places it conceptually between "API fetchers" and "EventBus: market.signal.raw" in the data flow diagram.

**Alpha's conclusion (Part 8, `_register_market_eventbus`):** Treats the signal normalization problem as an adapter layer inside the bootstrap wiring — each edge detector and API client gets a thin adapter when it publishes to the EventBus. The `MarketSignal` abstraction is not named or designed in my findings.

**Beta's conclusion (Section 3.5):** Identifies the gap explicitly: "there is no common output format that the ConvergenceAlerter can directly consume." Names it "adapter pattern" but does not design it.

**Divergence diagnosis:** Lead went deepest here — designed the full dataclass while I and Beta both registered the gap without filling it. My reasoning stopped at "each edge detector needs an adapter when it publishes to EventBus" and treated the adapter as a detail to be resolved in implementation. Lead's reasoning correctly identified this as a *first-class architectural component* that deserves its own design. The divergence happened at the step where I moved from "what is needed" to "what to build" — I defaulted to bootstrap wiring mechanics rather than asking "what is the shape of the information unit?"

**Synthesis:** Lead's `MarketSignal` dataclass design is the correct resolution. My bootstrap design needs to include instantiation of a `SignalNormalizer` class (or equivalent adapter registry) as a 14th market system on ctx. The `_register_market_eventbus` section of `bootstrap/market.py` is where this wiring happens, but the dataclass itself is a new file: `mae_core/market/signal.py`.

---

### Divergence 2: The Severity of "Standalone" Code

**Alpha's framing:** The market modules are "standalone Python objects" that need to be "wired in." I treated their current direct imports of each other as resolved (MEMORY.md confirms all imports fixed) and focused on what Layer 33 must add. My risk framing was primarily about bootstrap mechanics.

**Beta's framing:** The standalone nature is not just an architectural gap — it is a live operational hazard. Beta found: hardcoded `http://localhost:6333` Qdrant URLs across three files, `print()` for all logging, zero thread safety, and HTTP clients constructed at `__init__` time (meaning network calls fire during bootstrap). These are not "things to wire later" — they are blockers.

**Divergence diagnosis:** My lens (bootstrap integration) led me to focus on what Layer 33 adds. Beta's lens (adversarial) led them to focus on what the existing code does wrong. We were both looking at the same codebase but asking different questions. I noted Risk 3 (ClusterDetector makes Qdrant HTTP calls) but only in the context of step hooks — I did not flag it as a broader concern about all three modules making hardcoded localhost calls. My Risk section identifies it but underweights it.

**Synthesis:** Before Layer 33 is written, three pre-requisites exist that I did not classify as pre-requisites: (1) Replace all `print()` with structured logging. (2) Make Qdrant URLs configurable (via ctx or env var) so they can be provided by bootstrap rather than hardcoded. (3) Add `try/except` around `__init__` HTTP client construction in `PoliticianTracker` and `ContractPredictor`. These are now implementation prerequisites for Layer 33, not post-integration cleanup.

---

### Divergence 3: ThompsonSampler Prior Scale — Severity Assessment

**Alpha's treatment:** I noted that `ThompsonSampler._seed_from_reliability()` converts config values into Beta distributions and that the proxy delegation for `learn()` doesn't quite fit (ThompsonSampler has `update()` not `learn()`). I treated the seeding logic as reasonable starting infrastructure.

**Beta's finding (Section 1.1):** `DEFAULT_PRIOR_SCALE = 10` is dangerous. Beta's reasoning chain: 10 means the prior is equivalent to 10 real observations. For `sec_edgar` at 0.95, this produces Beta(9.5, 0.5) with variance 0.004 — so tight that the system will never explore this signal. More critically, `get_uncertain_signals()` uses `min_variance=0.01`, which means seeded signals at variance 0.004 will never appear in the exploration queue, even though they have zero real validation.

**Divergence diagnosis:** I missed the exploration-blocking consequence entirely. My reasoning about ThompsonSampler focused on the proxy interface mismatch (`update()` vs `learn()`). Beta's reasoning followed the data through to the `get_uncertain_signals()` method and found the compounding effect: tight prior + wrong variance threshold = permanently locked-in overconfidence with no exploration path. This is a more serious finding than my interface mismatch observation.

**Assessment of Beta's recommendation:** `DEFAULT_PRIOR_SCALE = 2` is correct. Additionally, `min_variance` in `get_uncertain_signals()` must be lowered from 0.01 to 0.001 simultaneously — changing prior scale alone without changing the exploration threshold still produces the lock-in. These two changes must be made together.

---

### Divergence 4: Fractal Placement of Market Intelligence

**Alpha's conclusion (Part 4):** Option A — extend `organ-cluster-cognitive` to a triad by adding `market-intelligence-system` as a third child alongside `cognitive-system` and `sensory-system`. Justification: "Market intelligence IS cognitive."

**Lead's treatment:** Does not address fractal placement. Lead's findings focus entirely on data flow and signal architecture. No mention of fractal hierarchy.

**Beta's treatment:** Does not address fractal placement. Beta's findings focus entirely on failure modes and calibration.

**Assessment:** This is a genuine gap in the other two investigators' findings — neither addresses Law 4 (Fractal Self-Similarity) or the fractal placement problem at all. My recommendation stands, but I want to note one consideration I did not fully examine: whether `market-intelligence-system` is better placed under `sensory-system` (as another organ of external sensing) rather than as a peer to it. The distinction matters because `sensory-system` currently handles internal sensory signals while market intelligence handles external information. Making market intelligence a *child* of sensory-system would preserve the K3 at organ level (mae → 3 clusters) without adding a new cluster. This is an alternative I flagged but did not fully explore.

---

### Divergence 5: The min_domains=2 Problem — Different Entry Point, Same Conclusion

**Alpha's treatment:** Not mentioned in my findings. I did not audit the ConvergenceAlerter's default parameters.

**Beta's finding (Section 1.3):** `min_domains=2` means two signals from the same company in two correlated domains (e.g., insider + government contract) would trigger a convergence alert. Beta explicitly connects this to Mae's Law 2 (Triadic Generator): min_domains should be 3, matching the minimum stability requirement.

**Divergence diagnosis:** Beta made a Law compliance finding that falls squarely within my stated lens (Law compliance) and I missed it entirely. My Law compliance checks focused on the bootstrap infrastructure — triadic connections, K3 fractal groupings, witness counts. I did not audit the internal parameters of the intelligence layer for Law compliance. Beta's adversarial lens led them to check whether the convergence logic itself upholds Law 2, which I should have done under my lens.

**Synthesis:** This is a genuine miss on my part. The correction belongs in the bootstrap layer: when `ConvergenceAlerter` is instantiated in `_instantiate_market_systems()`, pass `min_domains=3` as an explicit constructor argument. Do not rely on the default. This is a bootstrap responsibility because it is where Mae's laws are enforced as constraints.

---

### Divergence 6: VelocityDetector Units — Discovery vs. Architecture

**Alpha's treatment:** I identified that the `velocity` parameter in `ConvergenceAlerter.record_signal()` is always 0.0 in current practice (Part 3, VelocityDetector gap). I noted that the normalizer must bridge VelocityDetector to ConvergenceAlerter. I treated this as a wiring gap.

**Beta's finding (Section 1.4):** The units are per-second. A change from 2 to 8 insider buys in one day yields velocity = 6/86400 = 0.0000694 per second. The ConvergenceAlerter's urgency thresholds are calibrated around velocity > 0.1. At daily frequency, urgency will *always* read "days." The fix is to normalize velocity to per-day: divide `dt` by 86400.

**Divergence diagnosis:** We both found the VelocityDetector-to-ConvergenceAlerter gap, but Beta went one step further and found that even if you wire them together correctly, the unit mismatch means the urgency classification is permanently broken for slow signals. I found the wiring problem. Beta found the calibration problem that would persist after correct wiring.

**Synthesis:** The step hook design in my Part 7 is correct in structure. But the velocity values fed to `ConvergenceAlerter.record_signal()` must be in per-day units, not per-second. The normalizer adapter must convert: `velocity_per_day = velocity_detector_output.current_velocity * 86400`.

---

## Section 2: Agreements — Independent Convergence

These findings were reached independently by multiple investigators, giving them higher confidence.

---

### Agreement 1: trade.is_purchase is a crash bug

All three investigators found it independently. Lead found it in Part 10.1 (contract_predictor.py line 231 and politician_tracker.py line 276). Beta found it in Section 2.1 with the fix recommendation. My findings didn't include this specific bug because my lens focused on bootstrap mechanics rather than code-level bugs, but the convergence across Lead and Beta on this point makes it Priority 1.

**Confidence: Very High.** Fix before any live execution.

---

### Agreement 2: Ticker resolution is a real gap

Lead named it in Part 10.2: `GovernmentContract` and `ContractOpportunity` have no ticker field, and `recipient_name` (e.g., "LOCKHEED MARTIN CORPORATION") does not map cleanly to a ticker. Beta confirmed it obliquely in Section 2.3 (`_symbol_to_company()` has 11 hardcoded mappings). My findings did not address ticker resolution directly, but the Lead and Beta convergence is sufficient.

**Recommendation:** The `TickerResolver` service Lead proposes is the correct architecture. It belongs in `mae_core/market/apis/ticker_resolver.py`, and the bootstrap should instantiate it as `ctx.ticker_resolver` (a 15th system on ctx).

---

### Agreement 3: The signal normalizer / adapter layer is the central integration problem

Lead designed the full `MarketSignal` dataclass. Beta named it "adapter pattern" as an integration requirement. My findings reached it from the EventBus channel wiring angle (each publisher needs an adapter). All three paths lead to the same conclusion: one common signal format is needed before any wiring makes sense.

**Confidence: Very High.**

---

### Agreement 4: Qdrant dependency is a hard integration constraint

Beta named it explicitly (Section 3.1): all Qdrant URLs are hardcoded localhost. My findings noted Risk 3 about ClusterDetector. Lead's data flow diagram implicitly assumes Qdrant is available when ClusterDetector and FilingTimeAnalyzer query it.

**Synthesis:** The bootstrap should pass `qdrant_url` as a constructor parameter to any market system that queries Qdrant, sourcing it from `ctx` where the Qdrant connection is already configured (Foundation layer establishes Qdrant connection). The hardcoded `http://localhost:6333` must be replaced with a configurable parameter in three files.

---

### Agreement 5: learning_config.py decay_rates are dead config

Beta found it explicitly (Section 1.2). My findings noted that `learning_config` is a config dict module, not an instantiable class, and should not be registered as a holon — implicitly acknowledging its limited role. Neither of us found the full circuit that would make decay rates live: they need to be consumed by `ThompsonSampler._apply_bayesian_forgetting()` (a method that does not yet exist).

**Confidence: Very High.** Dead config that claims to be "self-modifiable learning parameters" is actively misleading.

---

## Section 3: Gaps — What the Other Investigators Missed

These are findings from my investigation that do not appear in Lead's or Beta's work.

---

### Gap 1: Layer 33 Must Run Before the Audit (Ordering Constraint)

Neither Lead nor Beta address bootstrap execution order. My findings establish that Layer 33 must be inserted between `bootstrap_external` (Layer 31) and `bootstrap_audit` (Layer 32 / final step). The audit checks `_MIN_SYSTEMS = 75` and `_MIN_HOLONS = 75`. If Layer 33 runs after the audit, the market systems are added to ctx but the audit never sees them — and more importantly, the audit's connection verification would run against an incomplete organism.

This is not just a detail. If someone places the `bootstrap_market(ctx)` call in the wrong position in main.py, the system boots without error but market connections are never verified.

---

### Gap 2: The Seal Boundary and Post-Seal Registration

Both Lead and Beta treat ConnectionRegistry as a simple "register connections" task. My findings establish the timing constraint: `seal()` is called at wiring.py line 517 (end of Layer 18). Post-seal registration is still valid — external.py Layer 31 does it — but Layer 33 must follow the same pattern as external.py, not the pattern of earlier bootstrap layers.

The specific consequence: Layer 33 must NOT call `connection_registry.register_connection()` before seal, because it does not exist yet during foundation layers. And it must NOT try to unseal-and-reseal, because seal is one-way.

---

### Gap 3: organ-cluster-cognitive K3 Completion

My findings (Part 12) identify that adding `market-intelligence-system` to `organ-cluster-cognitive` completes it from a 2-node dyad to a proper K3. Neither Lead nor Beta address the fractal hierarchy at all. This is architecturally significant: without this, `organ-cluster-cognitive` remains a bare dyad (violating Law 1 at the organ level), and the organism's cognitive cluster is structurally deficient.

The fix is explicit in my Part 12: use `ctx.fractal_generator.generate_triad()` at Layer 33 runtime rather than modifying `FRACTAL_GROUPING` in the source file.

---

### Gap 4: The HolonProxy get_statistics Name Mismatch

My findings (Part 10) identify that HolonProxy's `sense()` delegation looks for `get_statistics()` but all market systems implement `get_stats()`. This is a silent failure: the proxy calls `getattr(system, "get_statistics", None)` (holon_protocol.py proxy delegation pattern), gets `None`, and `sense()` returns an empty dict. The system appears healthy to the HolonRegistry but returns no meaningful sense data.

This mismatch will not cause any errors — it just silently produces empty sense output for every market system. Neither Lead nor Beta examined the proxy delegation chain in detail enough to find this.

The fix is one line per affected market system: `get_statistics = get_stats`. It is additive and non-breaking.

---

### Gap 5: Three New Stem Cell Roles

Neither Lead nor Beta address the stem cell / agent layer. My findings design three new role profiles (SEC_WATCHER, CONTRACT_TRACKER, MARKET_ANALYST) with specific genome configurations. These are necessary for Law 5 compliance — without them, no agent can specialize for market work within the organism's own rule structure. Agents would need to use SPECIALIST or API_CALLER roles, which lack market-specific capabilities.

---

### Gap 6: EndocrineSystem Coupling via Convergence Alerts

My findings (Part 11) wire the ConvergenceAlerter to the EndocrineSystem: strong bullish convergence releases DOPAMINE, strong bearish convergence releases ADRENALINE. Neither Lead nor Beta address how market signals should modulate the organism's internal state. This is the Law 6 (Autopoietic Closure) requirement: the market organ must produce effects that feed back through the organism, not just pass data out.

Without this coupling, market intelligence is an observer, not a participant. The organism would detect opportunities but its agents would not receive any hormonal signal nudging them to act on those detections.

---

### Gap 7: Document Parity Tracking

My findings (Part 14) enumerate all document parity impacts: systems count changes from 85 to ~99, holons count changes, bootstrap layers change from 32 to 33, fractal organs change from 5 to 6, triadic connections change from 313 to ~336, stem cell roles change from 9 to 12. Seven files must be updated in sync.

Neither Lead nor Beta address documentation parity at all. For this codebase, document parity is a hard requirement in CLAUDE.md — skipping it produces stale references that confuse future instances.

---

## Section 4: Surprises — Findings That Changed My Thinking

---

### Surprise 1: The ConvergenceAlerter is Already a Domain-Specific Convergence Detector

Lead's finding in Part 2.4: "ContractPredictor is the most horizontally integrated — it already IS a convergence detector for the defense sector specifically. It is a domain-specific version of what ConvergenceAlerter does generally."

This reframes the relationship between the edge detectors and the intelligence layer. I had been thinking of ContractPredictor as a data producer (produces predictions) and ConvergenceAlerter as the consumer (synthesizes predictions). Lead's observation shows that ContractPredictor is structurally *isomorphic* to ConvergenceAlerter — it combines hiring data, insider data, and contract opportunity data to produce a probabilistic prediction, which is exactly what ConvergenceAlerter does at a higher level.

This has an architectural implication I did not consider: should ContractPredictor be refactored to publish its component signals to the EventBus and let ConvergenceAlerter handle the synthesis? Or should it remain a specialized sub-convergence unit that publishes a pre-synthesized prediction for ConvergenceAlerter to treat as a single strong signal?

For the bootstrap design, I had placed ContractPredictor in the `market-edge` subsystem as an edge detector. Lead's observation suggests it might belong in `market-learning` (or a fourth tier), since it already does learning-style convergence. I am not changing my recommendation (it fits better in edge given its current implementation), but this is worth flagging for Phase 3.

---

### Surprise 2: The thompson_distributions.json Has Diverged from the Seeding Logic

Beta's finding (Section 7): The JSON file has 22 entries, but `learning_config.py`'s `source_reliability` only seeds 12 keys. The 10 extras were added manually and include contradictory duplicates (`technical_macd` at mean 0.833 and `rsi` at mean 0.167 — which appear to measure related signals with radically different assigned reliabilities). These extras have no path back to the config system and will not be reproduced on a fresh deployment.

I had treated the ThompsonSampler's seeding as "reasonable infrastructure." Beta's audit of the actual JSON file reveals that the current state of `thompson_distributions.json` cannot be regenerated from the codebase — the JSON has become the source of truth and the code has diverged from it.

The bootstrap implication: when Layer 33 instantiates `ThompsonSampler()`, it will read the existing `thompson_distributions.json`. If deployed fresh without this file, the seeding will produce a different distribution set. The file should be committed to the repository as a known-good starting state, not generated at runtime from the (now-incomplete) `learning_config.py` seeds.

---

### Surprise 3: The 45-Day Congressional Disclosure Delay as a Decay Timing Bug

Lead's finding (Part 10.5): The `CongressionalTrade.disclosure_date` is NOT the trade date. `transaction_date` is. Using `disclosure_date` for decay calculations makes a 45-day-old trade appear fresh when it is disclosed.

I had read the HouseStockWatcher code and knew about the 45-day delay conceptually, but I did not trace it to the decay calculation. The signal normalizer's `timestamp` field (which `VelocityDetector.record()` and `CorrelationTracker.record()` use for time-series computation) must use `transaction_date`, not `disclosure_date`. This is not just a cosmetic issue — it would cause the VelocityDetector to see a cluster of 45-day-old trades arrive simultaneously as a velocity spike, which would be classified as a sudden surge of congressional activity when in fact it is delayed reporting of activity that already happened.

This is a concrete correction to the signal normalizer design: the `timestamp` field in `MarketSignal` must be explicitly documented as "the time the underlying event occurred, NOT the time MIDGE received it" — and the congressional trade adapter must map `transaction_date` (not `disclosure_date`) to this field.

---

### Surprise 4: Alert Storm is a Bootstrap Concern, Not Just an Intelligence Layer Concern

Beta's finding (Section 1.3): `check_convergence()` generates a new alert every time it is called if conditions are met, with no deduplication. If conditions persist for 2 hours with 120 polling calls, 120 identical alerts are generated.

I had designed a step hook that calls `ctx.convergence_alerter.check_convergence()` on every simulation step. If the bootstrap runs with multiple agents running many steps (e.g., `--agents 10 --steps 1000`), this is 1000 `check_convergence()` calls per run. If a convergence condition is active for 500 of those steps, 500 alerts are published to `market.convergence_alert`, 500 EventBus callbacks fire, and 500 hormonal releases trigger on the EndocrineSystem.

The fix belongs in the step hook design, not in the ConvergenceAlerter itself (though Beta's recommendation to add deduplication there is also correct). The step hook should track `_last_alert_direction` and `_last_alert_time` and only publish to EventBus if the alert represents a new condition or a material change in strength:

```python
_last_convergence_alert = [None]

def _market_sense_hook():
    alerts = ctx.convergence_alerter.check_convergence()
    for alert in alerts:
        last = _last_convergence_alert[0]
        if (last is None
                or last["direction"] != alert.direction
                or abs(last["strength"] - alert.strength) > 0.1):
            ctx.bus.publish("market.convergence_alert", alert.to_dict())
            _last_convergence_alert[0] = {"direction": alert.direction, "strength": alert.strength}
```

This deduplication lives in the bootstrap layer, which is appropriate because the bootstrap layer is where the polling cadence is defined.

---

## Section 5: Revised Priority Order for Implementation

Synthesizing all three investigations:

### Pre-implementation fixes (must precede Layer 33):

1. Fix `trade.is_purchase` AttributeError in `contract_predictor.py:232` and `trade.shares_traded` in `politician_tracker.py:276`.
2. Change all hardcoded `http://localhost:6333` to configurable parameters in `cluster_detector.py`, `filing_time_analyzer.py`, `contract_predictor.py`.
3. Replace all `print()` with `logging.getLogger(__name__)` across all 16 market module files.
4. Add `try/except` around `__init__` HTTP client construction in `PoliticianTracker` and `ContractPredictor`.
5. Set `DEFAULT_PRIOR_SCALE = 2` and `min_variance = 0.001` in `thompson_sampler.py` (must change together).
6. Normalize velocity units from per-second to per-day in `velocity_detector.py`.
7. Fix timezone handling in `filing_time_analyzer.py`.

### Layer 33 implementation sequence (from my Part 16, revised):

1. Create `mae_core/market/signal.py` with `MarketSignal` dataclass (Lead's design, Part 6).
2. Add `get_statistics()` adapter methods to `ThompsonSampler`, `ConvergenceAlerter`, `VelocityDetector`.
3. Add three ROLE_PROFILES to `stem_cell.py`.
4. Create `mae_core/bootstrap/market.py` with `bootstrap_market(ctx)`.
5. Instantiate `ConvergenceAlerter` with explicit `min_domains=3` (not the default 2).
6. Add deduplication state to the convergence step hook.
7. Add `bootstrap_market(ctx)` to `main.py` between external and audit.
8. Update `_build_systems_dict()` in `main.py`.
9. Update all document parity files.
10. Run full test suite, verify zero regressions.

---

## Summary Assessment

The three-way investigation was productive. Lead's signal architecture findings are the most immediately actionable and fill the largest gap in my work (the `MarketSignal` dataclass and normalization layer). Beta's adversarial findings expose multiple pre-integration requirements that would produce silent failures if my bootstrap design were applied to the unmodified code. My findings provide the structural container (Layer 33 design, fractal placement, triadic connections, document parity) that neither other investigator addressed.

The primary synthesis conclusion: the integration problem has two phases that must be kept separate. Phase A is making the existing market modules safe to instantiate (pre-implementation fixes, primarily Beta's findings). Phase B is wiring them into Mae's organism (Layer 33 design, primarily my findings, with Lead's signal architecture as the shared data contract). Conflating the two phases risks building bootstrap infrastructure that routes broken data through a working organism.
