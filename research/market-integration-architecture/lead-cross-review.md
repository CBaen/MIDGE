# Lead Cross-Review: Signal Architecture Lens
**Phase 2 — Cross-Review of Alpha and Beta Findings**
**Date: 2026-02-22**

---

## Framing

My investigation mapped the signal flow: what each API produces, what each consumer expects, and what normalization layer is missing in between. Alpha investigated how to wire the modules into Mae's bootstrap machinery. Beta stress-tested the modules for failure modes, miscalibration, and integration hazards.

The review below starts where our reasoning diverged — because divergence is where the triadic structure does its work.

---

## Part 1: Reasoning Divergence Points

### 1.1 The Signal Normalizer: Who Builds It and When

**Lead's position:** A `MarketSignal` dataclass and signal normalizer layer are the mandatory first deliverable. Every adapter, channel, and edge detector integration depends on this common format existing first. The architecture is: sources emit raw types, a normalizer collapses them to `MarketSignal`, the intelligence layer consumes only `MarketSignal`.

**Alpha's position:** Alpha acknowledges the lack of a common signal format (Section 3.5 of Beta's findings, which Alpha's structure implicitly assumes away) but treats the bootstrap wiring as the primary deliverable. Alpha's Layer 33 design wires the systems together and registers their connections, but does not specify who translates `ClusterSignal` to `ConvergenceAlerter.record_signal()`. Alpha assumes adapters will exist but does not specify them as a discrete prior dependency.

**Where the reasoning diverged:** Alpha's implementation sequence (Part 16) starts with `get_statistics()` adapters and role profiles, then creates `market.py`. My sequence starts with `MarketSignal` dataclass. Alpha's approach would produce a correctly wired bootstrap that crashes the first time a real signal flows through it, because `check_convergence()` would receive a `ClusterSignal` object where it expects six typed parameters. The normalizer is not optional infrastructure — it is the precondition for any live signal flow. Alpha's sequence is correct for making Mae's organism aware of market systems; it is incomplete for making market data actually move.

**Resolution:** The correct sequence merges both. Alpha's wiring is the skeleton; my normalizer layer is the nervous tissue that carries signals through it. Neither is wrong — they are different levels of the same build. But the implementation order matters: normalizer adapters must exist before the EventBus callbacks are registered, or the callbacks will have nothing to call.

---

### 1.2 Fractal Placement of Market Intelligence

**Lead's position:** I did not address fractal placement directly. My investigation focused on data shapes and channel naming, not on where market systems sit in the holon hierarchy.

**Alpha's position:** Alpha recommends Option A — extend `organ-cluster-cognitive` from 2 to 3 children by adding `market-intelligence-system`. Alpha's reasoning is biologically coherent: market intelligence senses and reasons about the external information environment, which is a cognitive function.

**Beta's position:** Beta did not address fractal structure.

**Surprise from Alpha's finding:** Alpha caught something I missed entirely — that `organ-cluster-cognitive` currently has only 2 children (cognitive-system, sensory-system), which is a bare dyad. Adding `market-intelligence-system` as the third child FIXES a pre-existing Law 1 violation in the organism's own fractal structure. This is not just a correct placement decision — it is a repair. I would not have found this from the signal architecture lens alone.

**My assessment after review:** Alpha's Option A recommendation is correct. The biological coherence argument holds, and the structural benefit of completing the dyad into a triad is an additional argument Alpha stated clearly. I endorse it without qualification.

---

### 1.3 API Clients: Register with BoundaryMembrane vs. Route Through ApiGateway

**Lead's position:** I described a two-phase approach: (1) ApiGateway `external.response_received` adapters for the four registered providers (MarketAux, Finnhub, Alpha Vantage, Tavily), (2) direct API clients for the market-specific sources (SEC EDGAR, HouseStockWatcher, etc.). My reasoning was that the four providers are already in the organism's immune system and should flow through it, while the market-specific clients can be phased into BoundaryMembrane later.

**Alpha's position:** Alpha recommends registering trust scores for market sources with BoundaryMembrane immediately in Layer 33, but explicitly deferring full ApiGateway routing as "Phase 2." Alpha proposes `_trust_provider(ctx, name)` calls for each source.

**Beta's position:** Beta identified that all Qdrant calls use hardcoded `localhost:6333` and are architectural violations. Beta also identified thread-safety problems with direct file writes in ThompsonSampler.

**Where the reasoning diverged:** My approach to the four ApiGateway providers was more specific (describe the `external.response_received` adapter pattern) and my approach to the six market-specific API clients was less specific (left as "phase 2 task"). Alpha's approach inverted this: very specific about trust registration for all six market sources, silent on how the four ApiGateway providers integrate. Beta's findings make both of our approaches look insufficient — the hardcoded localhost:6333 pattern in ClusterDetector, FilingTimeAnalyzer, and ContractPredictor is a deeper violation that neither Alpha nor I fully resolved.

**Resolution:** A correct integration plan needs three categories, not two:
1. Four ApiGateway providers: adapter on `external.response_received` → `MarketSignal` → `market.signal.raw`
2. Six market-specific API clients: trust-register with BoundaryMembrane (Alpha's approach) as Phase 1, full ApiGateway routing as Phase 2
3. Three Qdrant-calling edge detectors: replace hardcoded localhost with `ctx.qdrant_url` from the bootstrap context, which already carries this value. This is a one-line fix per module, not a full refactor, and it must happen in Phase 1.

---

### 1.4 The Velocity-to-Urgency Chain

**Lead's position:** I identified that `velocity` in `ConvergenceAlerter.record_signal()` is always 0.0 in current practice because VelocityDetector is not wired to ConvergenceAlerter. I flagged this as a gap that would cause all alerts to classify as `urgency="days"`.

**Beta's position:** Beta identified the underlying cause that I missed: velocity is computed in units of per-second. For daily-frequency signals, this produces values on the order of 0.0001, which will never cross the `avg_velocity > 0.1` threshold used in ConvergenceAlerter's urgency classification. The wiring gap I found is real, but even if it were fixed by wiring, the velocity values would still be too small by three orders of magnitude to trigger anything other than "days."

**Where the reasoning diverged:** I found the symptom (velocity always 0.0 → urgency always "days"). Beta found the root cause (velocity computed in wrong time units → urgency always "days" even after wiring). My finding and Beta's finding are additive. The fix requires both: (1) wire VelocityDetector to ConvergenceAlerter (my finding), and (2) normalize velocity to per-day units before the wire (Beta's finding). Missing either fix leaves urgency classification broken.

---

### 1.5 The ConvergenceAlerter's Minimum Domain Threshold

**Lead's position:** I identified that ConvergenceAlerter requires signals from at least 2 different *categories* (not just 2 domains) to issue a non-neutral recommendation. With only government/institutional data sources active, all signals fall into "institutional" and convergence always returns "neutral." I framed this as a risk requiring at least one price/technical source active.

**Beta's position:** Beta identified the same structural problem from a different angle: `min_domains=2` is too low because two signals from the same company — one under "insider" domain and one under "government" domain — can trigger a convergence alert. These are not independent signals. Beta recommends raising `min_domains=3` to align with the Law of Triadic Generator.

**Where the reasoning diverged:** I focused on the category-level collapse (institutional signals all look the same). Beta focused on the entity-level correlation (two domains, same company). These are two different failure modes of the same parameter. My failure mode is about source diversity. Beta's failure mode is about entity independence. Both are real. The correct fix is Beta's recommendation (min_domains=3) combined with my observation (at least one technical/price source must be active). Neither alone is sufficient.

---

## Part 2: Agreements — Independent Convergence

The following conclusions were reached independently by two or more investigators:

**The `trade.is_purchase` bug:** Both Lead (Part 10.1, referencing `contract_predictor.py:231` and `politician_tracker.py:276`) and Beta (Section 2.1 and the priority list) identified this as a guaranteed AttributeError crash on first real data. Both traceable to the same root: `InsiderTrade` uses `transaction_type == "A"` not an `is_purchase` property. The fix is unambiguous.

**No common signal format:** Lead identified the normalizer gap as the central architectural problem (Part 6). Beta independently reached the same conclusion from the adversarial direction (Section 3.5: "No Standard Signal Format"). Alpha implicitly assumed the gap would be filled by adapter code but did not specify it, which means Alpha's implementation plan is incomplete without this finding.

**Ticker resolution is missing for government contract sources:** Lead identified this in Part 10.2. Beta confirmed it in Section 2.3 (`_symbol_to_company()` has 11 hardcoded mappings). Both investigations agree this is a required component before government contract data can feed into market-symbol-keyed signals.

**Qdrant is hardcoded localhost and this is wrong:** Lead noted ClusterDetector queries Qdrant directly (Part 2.1). Beta identified this as an architectural violation affecting three modules (Section 3.1). Both agree the fix is to use the bootstrap context's Qdrant URL rather than hardcoded localhost, not a full ApiGateway refactor.

**The feedback loop architecture is sound in principle:** Lead described the price-as-ground-truth feedback loop in detail (Part 8). Alpha confirmed the loop's place in the organism via the EndocrineSystem wiring (Part 11). Neither found a fundamental flaw in the loop design — the problems are in calibration (Beta) and wiring order (Lead), not in the loop concept.

---

## Part 3: Gaps — What Each Investigation Missed

### What Alpha missed

**The normalizer sequence dependency.** Alpha's implementation sequence (Part 16) is the correct engineering sequence for making Mae aware of market systems as holons. It is not the correct sequence for making market data flow. The gap: Alpha does not specify adapters that translate `ClusterSignal`, `ContractPrediction`, `CorrelationSignal`, and `FilingTimeSignal` into `ConvergenceAlerter.record_signal()` calls. These adapters must exist before EventBus callbacks are registered, or the callbacks will crash on first event.

**The `get_statistics()` naming conflict.** Alpha notes that HolonProxy looks for `get_statistics()` but market systems have `get_stats()` (e.g., ThompsonSampler line 332). Alpha recommends adding `get_statistics()` aliases. This is correct but Alpha's proposed implementations for ConvergenceAlerter and VelocityDetector contain bugs: ConvergenceAlerter's proposed `step()` method returns `len(alerts)` (an int) instead of returning None. In Mae's step hook pattern, return values from step hooks are ignored, so this is not a runtime error — but it signals incomplete thinking about what `step()` should do.

**The 45-day STOCK Act timing issue.** Alpha does not address the disclosure delay in CongressionalTrade. The correct signal timestamp is `transaction_date`, not `disclosure_date`. This matters for decay calculation in `MarketSignal.timestamp`. A congressional trade that is 44 days old at time of disclosure will already have decayed significantly (decay_rate=0.03/day = approximately 26% of original strength) and should be treated accordingly.

### What Beta missed

**The `price_fetcher_for_outcomes()` function's architectural role.** Beta audited the PriceFetcher's API calling patterns but did not identify that `price_fetcher_for_outcomes()` (line 263 in `price_fetcher.py`) is already architecturally shaped as the Bayesian feedback function. It accepts a symbol and a prediction date and returns price change over a forward window. This function is the implementation of the feedback loop — not a function that needs to be built. Beta's cold-start analysis would have been better served by noting this exists and is wired incorrectly (not yet called from an OutcomeTracker), rather than treating the feedback loop as entirely absent.

**The `ContractPredictor` is already a convergence detector.** Beta's adversarial investigation focused on what breaks. It did not note that `ContractPredictor` is architecturally redundant with `ConvergenceAlerter` for the defense-sector case. ContractPredictor combines SAM.gov + JobTracker + Form4 data and produces a confidence-weighted prediction — which is exactly what ConvergenceAlerter would do if defense sector signals were flowing through it. This creates a design question: should ContractPredictor be retired in favor of routing its inputs through ConvergenceAlerter, or does its domain-specific confidence formula justify keeping it as a pre-filter? Beta did not surface this.

**The SAM.gov + JobTracker leading indicator chain.** Beta identified that the JobTracker's `jobs_30d` metric is computed from 7-day API data (Section 2.6), which is a real bug. Beta did not identify the signal chain this breaks: SAM.gov solicitation → JobTracker hiring blitz → GovernmentContract award is the primary leading indicator sequence for pre-announcement detection. If JobTracker spikes are systematically false positives, this chain produces noise at its most sensitive detection point. The bug Beta found has a much larger architectural consequence than Beta described.

**The `discovery_log.jsonl` file is a pattern library with no reader.** Beta audited the intelligence modules thoroughly but did not examine `data/market/discovery_log.jsonl`. This file exists per the MEMORY.md and CLAUDE.md documentation, and ConvergenceAlerter is supposed to write novel pattern discoveries to it. No code in any module reads this file to inform future signal weighting or pattern matching. It is a write-only log with no downstream consumer — not a memory, just a diary.

### What Lead (my own investigation) missed

**The `prior_scale=10` problem.** Beta identified that seeding `sec_edgar` at `Beta(9.5, 0.5)` is equivalent to claiming 10 real historical observations before any data is collected. I described the Thompson Sampler's seeding behavior (Part 3.2) without flagging this as dangerous. I noted the seeding values but treated them as reasonable starting points. Beta's adversarial lens caught that a `prior_scale=10` produces a distribution with variance 0.004 — narrow enough that the sampler will almost never explore sec_edgar as a signal type, despite having zero real validation. This is a calibration error I should have caught and did not.

**The `min_variance` threshold interaction.** Related to the above: I described `get_uncertain_signals()` in ThompsonSampler without noting that its default `min_variance=0.01` threshold excludes all seeded high-reliability signals from exploration. The system will never know if its confident signals are actually reliable, because it never tests them against the exploration arm. This is the exploration/exploitation balance breaking in exactly the wrong direction for a new system.

**The `thompson_distributions.json` split-brain problem.** The JSON file has 22 entries, but `learning_config.py`'s `source_reliability` dict has 12 keys. The 10 extras (including `rsi`, `bollinger`, `insider_cluster`, `options_flow`, `congress_trade`, `contract_award`) were manually added and will not survive a fresh boot. Beta found this. I audited the ThompsonSampler's seeding logic but did not audit the JSON file against it.

**The alert deduplication problem.** My EventBus channel architecture (Part 7) described `market.intel.convergence` as the primary convergence output channel. I did not address that `check_convergence()` generates a new alert every time it is called if conditions persist. When wired to a step hook that calls `check_convergence()` every step, a persistent convergence condition will publish a new alert on `market.intel.convergence` every step. Downstream systems (agents, the decision cascade) will receive hundreds of alerts about the same condition. Beta identified this. I did not.

---

## Part 4: Surprises — Findings That Changed My Thinking

### Surprise 1: The organ-cluster-cognitive is already a bare dyad

Alpha's fractal analysis revealed that adding market-intelligence-system to organ-cluster-cognitive does not just extend the hierarchy — it repairs a pre-existing Law 1 violation. The current organism has a bare dyad at the cluster level (cognitive-system and sensory-system with no witness). I built my entire signal architecture assuming I was adding something new to a healthy structure. Alpha found the structure I was adding to was itself incomplete. This changes the framing: wiring market intelligence is not an extension, it is a completion.

### Surprise 2: The Thompson Sampler's learned state is already corrupted

Beta's audit of `thompson_distributions.json` reveals that the file has diverged from its seeding logic. Entries like `rsi` (Beta(1,1)), `bollinger` (Beta(1,5)), and `congress_trade` (Beta(1,1)) exist in the JSON but have no corresponding key in `learning_config.py`'s `source_reliability`. Meanwhile `technical_macd` (Beta(5,1)) and `technical_rsi` (Beta(6,1)) coexist with `rsi` (Beta(1,5)) — two representations of the same signal type with contradictory beliefs. The JSON was manually edited after initial seeding and the edits were not reflected back into the seeding code. A fresh deployment will not reproduce the current state. The Bayesian state that exists is not reproducible from the codebase alone. This is not what I expected to find — I assumed the distributions were the clean output of the seeding logic.

### Surprise 3: The `jobs_30d` bug breaks the leading indicator chain

Beta found that JobTracker's `jobs_30d` metric is actually computed from 7-day API data divided by 30. I knew the leading indicator chain (SAM solicitation → hiring spike → contract award) was the primary value proposition of the contract prediction system, and I documented it in Part 1.6 of my findings. I did not investigate whether the hiring spike detection was correctly implemented. Beta's finding means the most sensitive signal in the leading indicator chain — the hiring blitz that precedes a contract announcement — fires as a false positive almost universally. The chain I described as MIDGE's primary edge is currently generating noise at its most important node.

### Surprise 4: ContractPredictor and ConvergenceAlerter may be redundant

After reading Beta's thorough failure analysis of ContractPredictor alongside my own description of ConvergenceAlerter's architecture, a question I did not ask during Phase 1 becomes pressing: does ContractPredictor belong as a standalone edge detector, or should it be decomposed into its constituent signals (SAM.gov + hiring + insider) and routed through the general convergence architecture? ContractPredictor's confidence formula is domain-specific, which is its strength and its fragility (the `is_active_bidder=True` assumption Beta flagged). ConvergenceAlerter's general formula would be less precise but more calibrated. This is a design decision I did not identify as open during my investigation, but the combination of Lead + Alpha + Beta findings makes it visible.

---

## Part 5: Synthesized Priority Order

Combining all three investigations into a unified priority order for implementation:

**Must fix before any code is written (pre-conditions):**
1. Audit and rebuild `thompson_distributions.json` from scratch. The current file has manually added entries that cannot be reproduced. Reset to seeding from `learning_config.py` only, with `DEFAULT_PRIOR_SCALE = 2` (Beta's recommendation).
2. Fix `InsiderTrade`: add `is_purchase` property and `shares_traded` alias. Both are guaranteed crashes on first real data. (Lead + Beta)
3. Fix `jobs_30d` baseline to use `daily_avg = signal.jobs_7d / 7` not `signal.jobs_30d / 30`. (Beta)

**Must fix before bootstrap wiring (architectural prerequisites):**
4. Write `MarketSignal` dataclass and adapter functions for each source type. (Lead) This is the prerequisite for all EventBus wiring.
5. Replace all hardcoded `http://localhost:6333` Qdrant URLs with `ctx.qdrant_url` parameter. (Beta, confirmed by Lead)
6. Fix `VelocityDetector` to compute velocity in per-day units, not per-second. (Beta)
7. Change `ConvergenceAlerter.min_domains` default from 2 to 3. (Beta, confirmed by Lead's category-collapse finding)
8. Fix `hash()` Qdrant point IDs to use `uuid.UUID(cluster_id).int`. (Beta)

**Bootstrap wiring (Layer 33):**
9. Implement Alpha's `market.py` bootstrap following the design in Alpha's Part 8 — instantiation, holon registration, somatic registration, connections, fractal placement, step hooks.
10. Add `get_statistics()` adapters to ThompsonSampler, ConvergenceAlerter, VelocityDetector as Alpha specifies.
11. Add three stem cell roles (SEC_WATCHER, CONTRACT_TRACKER, MARKET_ANALYST) as Alpha specifies.
12. Wire VelocityDetector output to ConvergenceAlerter `velocity` parameter via normalizer (not direct call). (Lead)

**Must fix before live operation (calibration and safety):**
13. Add alert deduplication to ConvergenceAlerter — suppress re-alert within minimum interval. (Beta)
14. Cap `self.alerts` list at fixed size (deque with maxlen=1000). (Beta)
15. Add timezone handling to FilingTimeAnalyzer (convert to Eastern Time). (Beta)
16. Increase `CorrelationTracker.min_observations` from 10 to 30. (Beta)
17. Implement Bayesian forgetting in ThompsonSampler via multiplicative decay. (Beta)
18. Replace `print()` logging with structured logging throughout. (Beta)
19. Add thread locking to ThompsonSampler file writes. (Beta)
20. Move `learning_config.py`'s history log path to `DATA_DIR` for consistency. (Beta)
21. Use real contact email in SEC_USER_AGENT. (Beta)

**Deferred (Phase 2):**
22. Full ApiGateway routing for the six market-specific API clients. (Alpha + Lead)
23. Regime-aware Thompson Sampling (separate distributions per market regime). (Lead)
24. CorrelationTracker deque persistence across restarts. (Beta)
25. Evaluate ContractPredictor vs. ConvergenceAlerter decomposition. (This review)

---

## Summary

The three investigations are genuinely complementary — each lens found things the others missed. The most important synthesis: Alpha's bootstrap design is the correct skeleton, but it needs Lead's normalizer layer before signals can flow and Beta's fixes before the signals that flow are honest. None of the three investigations alone produces an implementable plan. Together they do.

The single most consequential finding across all three is Beta's discovery of the Thompson Sampler prior scale problem combined with the split-brain JSON state. The system's Bayesian memory is currently both overconfident and non-reproducible. This must be addressed before any wiring is done — otherwise, we are wiring a learning system that is already learning from noise and cannot be reset to a known state.
