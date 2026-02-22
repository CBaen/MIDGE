# Lead Revision: Signal Architecture — Phase 3
**Lens: Signal Architecture**
**Role: Lead (Phase 3 Revision)**
**Date: 2026-02-22**
**Status: REVISE — five positions updated, two held firm, one new position added**

---

## Preamble

After reading all three cross-reviews in full, I conducted additional verification against the source code for the two highest-severity disputes: the VelocityDetector unit mismatch and the ThompsonSampler split-brain state. Both are confirmed. The cross-review process surfaced five issues that change my original positions and produced one emergent issue (the EventBus channel name conflict) that neither my original findings nor the cross-reviews fully resolved. That conflict is resolved here.

This document is organized as: (1) positions I am revising and why, (2) positions I am holding firm and why, (3) the new synthesis position on the EventBus naming conflict.

---

## Section 1: Revised Positions

### Revision 1: Velocity Units — Beta is Right, My Finding Was Incomplete

**Original position (Part 10.3):** I identified that the `velocity` parameter in `ConvergenceAlerter.record_signal()` is always 0.0 in current practice because VelocityDetector is not wired. I framed this as a wiring gap.

**What the cross-review added:** Beta found the underlying cause: VelocityDetector computes velocity in per-second units (`dt = (timestamp - state.last_timestamp).total_seconds()`, velocity_detector.py line 132). The `__main__` demo at line 397 even prints the label `"/sec"` explicitly. ConvergenceAlerter's urgency thresholds (lines 260-265) check `avg_velocity > 0.1` and `avg_velocity > 0.05`. A change from 2 to 8 insider buys in one day produces velocity = 6/86400 ≈ 0.0000694 per second — three orders of magnitude below the 0.05 threshold.

**Revised position:** My finding was the symptom (wiring gap). Beta's finding is the root cause (unit mismatch). Even after correct wiring, urgency classification would be permanently broken without the unit fix. The correct fix has two parts that must both happen:

1. Change the velocity computation in `VelocityDetector.record()` from `dt = (timestamp - state.last_timestamp).total_seconds()` to `dt = (timestamp - state.last_timestamp).total_seconds() / 86400` — making velocity per-day.
2. Wire VelocityDetector output to ConvergenceAlerter via the normalizer (my original finding).

Neither fix alone is sufficient. The state of my original findings that says "the normalizer must bridge VelocityDetector → ConvergenceAlerter" is correct but incomplete. It must now read: "the normalizer must bridge VelocityDetector → ConvergenceAlerter, and VelocityDetector must be corrected to compute velocity in per-day units before that wiring is useful."

**What changed:** The velocity fix is now a pre-wiring requirement, not a wiring detail.

---

### Revision 2: Thompson Distributions Reset Strategy — Beta is Right, My Omission Was Consequential

**Original position (Part 3.2):** I described the Thompson Sampler's seeding behavior and noted the initial seed values without flagging the prior scale or the JSON state as problematic. I treated the distributions as reasonable starting infrastructure.

**What the cross-review added:** Beta identified that `DEFAULT_PRIOR_SCALE = 10` produces distributions too narrow for exploration (variance 0.004 for sec_edgar at Beta(9.5, 0.5)), and that `get_uncertain_signals()` uses `min_variance=0.01`, which excludes these seeded signals from the exploration queue permanently. Additionally, Beta audited the actual `thompson_distributions.json` file and found 23 entries against 12 source_reliability keys in learning_config.py. I verified this directly: the JSON contains `rsi` at Beta(1.0, 5.0) (mean=0.167) and `technical_rsi` at Beta(6.0, 1.0) (mean=0.857) — contradictory beliefs about the same signal type with the JSON as the diverged source of truth.

**Revised position:** The thompson_distributions.json file must be reset before any wiring is done. The reset strategy requires two simultaneous changes:

1. Set `DEFAULT_PRIOR_SCALE = 2` in thompson_sampler.py, producing loose priors that actually allow exploration. At scale=2, sec_edgar seeds at Beta(1.9, 0.1) — mean 0.95 but variance 0.045, which is above the `min_variance=0.01` threshold and will appear in the exploration queue.
2. Lower `min_variance` in `get_uncertain_signals()` from 0.01 to 0.001 simultaneously. Alpha's cross-review confirmed these must change together — changing prior scale alone without changing the exploration threshold still produces lock-in for any signals seeded at moderate scale values.
3. Delete `thompson_distributions.json` and regenerate from the corrected seeding logic. The file cannot be the source of truth because it cannot be reproduced from the codebase. Remove the 11 manually added entries by making the seeding code the authoritative source.

**The reset strategy question:** Beta asked whether the file needs a reset strategy for live deployments (so accumulated real learning is not thrown away on restarts). The answer is: the current file has zero real learning — it was manually edited, not trained. The "accumulated" beliefs in the file are fiction. Deleting it is not losing learning; it is removing corrupted initialization data. For future deployments where real learning has accumulated, the file should be committed to git at checkpoint milestones and treated as a versioned artifact. The current file should not be committed in its present state.

**What changed:** The Thompson Sampler prior scale and distribution file are now classified as pre-implementation requirements, not calibration details.

---

### Revision 3: The OutcomeTracker — It Must Be Built, Not Assumed

**Original position (Part 8):** I described the feedback loop design and cited `price_fetcher_for_outcomes()` (price_fetcher.py line 263) as the function that closes the Bayesian loop. I assumed an OutcomeTracker component would exist to call it on schedule.

**What the cross-review added:** Beta (Section 5.3) made this failure mode explicit: the OutcomeTracker does not exist in any of the 16 market modules, in Alpha's Layer 33 design, or in any bootstrap plan. Without it, `predictions.jsonl` fills with predictions that are never evaluated, and `ThompsonSampler.update()` is never called from real outcomes. The system's defining capability — Bayesian signal reliability learning — is architecturally specified but never instantiated.

I noted in my cross-review (Part 3, "What Beta missed") that `price_fetcher_for_outcomes()` "is the implementation of the feedback loop — not a function that needs to be built." I was wrong to frame it that way. That function is a building block, not the feedback loop itself. The loop still needs a scheduler, a prediction store reader, and the outcome comparison logic to call it on the right schedule with the right parameters.

**Revised position:** OutcomeTracker must be designed and included in Layer 33 as a 15th market system (alongside Alpha's 14). Its responsibilities:

1. Read `predictions.jsonl` on a scheduled cadence (not every step — every N steps to approximate daily checks, or using a simulation-day counter)
2. For each prediction where `outcome_window_days` has elapsed since the signal timestamp, call `PriceFetcher.get_historical_price(outcome_symbol, signal_date + window_days)`
3. Compute whether price moved in the predicted direction by a minimum threshold (2% is a reasonable starting value)
4. Call `ThompsonSampler.update(signal_id, success=bool, regime="default")`
5. Write to `outcomes.jsonl`
6. Publish on `market.outcome.prediction_result` EventBus channel

This is approximately 60-80 lines of Python that must exist before the system can learn. It is not a Phase 2 task — it is required for the core promise of MIDGE. The Alpha bootstrap plan must be updated to include this component.

**What changed:** OutcomeTracker is now a required Layer 33 component, not an assumed future piece.

---

### Revision 4: min_domains Must Be 3 — Beta's Law Compliance Finding Stands

**Original position (Part 10.6, framed as risk):** I identified that with only government/institutional data sources active, all signals fall into the "institutional" category and ConvergenceAlerter always returns neutral. I framed this as a data source diversity problem.

**What the cross-review added:** Beta identified the same structural problem from a different angle — `min_domains=2` allows two correlated signals from the same company in two domains (e.g., one insider trade and one contract award for the same ticker) to trigger a convergence alert. These are not independent signals. Beta explicitly connected this to Law 2 (Triadic Generator): the minimum for genuine independence is 3.

Alpha's cross-review confirmed this finding falls squarely under their Law compliance lens and they missed it in their original investigation — which strengthens it further. The correct fix, as Alpha specified, is to instantiate ConvergenceAlerter with `min_domains=3` as an explicit constructor argument in the bootstrap layer.

**My assessment:** Both failure modes (category collapse and entity correlation) are real and distinct. Raising min_domains to 3 addresses Beta's entity-correlation failure mode. My data-source-diversity concern is a separate issue that min_domains alone cannot solve — you still need at least one technical/price source active to span the behavioral + market categories. These are additive requirements, not alternative fixes.

**Revised position:** The statement in my original findings (Part 10.6) that "MIDGE needs at least one price/technical signal source active" is correct and stands. But it is now accompanied by a second requirement: min_domains must be 3, enforced at bootstrap instantiation, not left at the default of 2.

**What changed:** min_domains=3 is now a bootstrap enforcement requirement, not a runtime calibration option.

---

### Revision 5: The Feedback Loop Framing in My Cross-Review Was Overconfident

**Original cross-review position (Part 3, "What Beta missed"):** I wrote that `price_fetcher_for_outcomes()` "is the implementation of the feedback loop — not a function that needs to be built." I used this to argue Beta should have noted the function exists rather than treating the feedback loop as absent.

**Why this was wrong:** Having a building block function is not the same as having the feedback loop. Beta's finding (5.3) is correct: the OutcomeTracker is missing, and without it, the loop is architecturally specified but not operational. My defense of the status quo was incorrect — I was noting a half-finished foundation and calling it a structure. The loop requires: a scheduler, a store reader, the comparison logic, and the update call. `price_fetcher_for_outcomes()` provides the price fetch step only. Beta's severity classification ("LEARNING NEVER HAPPENS") is accurate.

**What changed:** I am retracting my cross-review criticism of Beta on this point. Revision 3 above is the correction.

---

## Section 2: Positions Held Firm

### Firm 1: Channel Naming — Resolution Below (Neither Alpha Nor I Was Fully Right)

The channel naming conflict between my proposal and Alpha's is real and rated "COMPLETE INTEGRATION FAILURE" by Beta. This is not a position I hold firm — it requires resolution. See Section 3.

### Firm 2: MarketSignal Dataclass Design Is Correct

All three cross-reviews confirm the `MarketSignal` dataclass (my Part 6) as the correct resolution to the normalization gap. Alpha's cross-review endorsed it explicitly: "Lead's `MarketSignal` dataclass design is the correct resolution... a new file: `mae_core/market/signal.py`." Beta confirmed the gap independently from the adversarial direction. No cross-review challenged the 18-field schema.

One field requires modification based on Revision 2: `decay_rate` in the dataclass is populated from `learning_config.decay_rates`, but as Beta found and Alpha confirmed, those decay rates are currently dead config — nothing reads them. The field should remain in the dataclass (it represents a valid design intent), but its docstring must note that the downstream consumer (`ThompsonSampler._apply_bayesian_forgetting()`) does not yet exist. Populating the field is correct; acting on it requires the forgetting mechanism to be built first.

One field requires modification based on Beta's Failure Mode 5.7: `raw_payload: dict` should be replaced with `raw_id: str` — a reference to the original record's identifier, not the full serialized payload. The raw payload stored in Qdrant would bloat the index and defeat the purpose of normalization. Edge detectors that need raw fields (ClusterDetector's Qdrant payload schema) should receive those fields from the Qdrant store via the raw_id lookup, not from the MarketSignal in transit.

**Revised field (not a position change, a field correction):**
```python
# Raw payload reference (for audit — NOT the full payload, just the key)
raw_id: str     # Identifier to look up original record in source system
raw_type: str   # "InsiderTrade", "Form8KEvent", "CongressionalTrade", etc.
```

The `raw_payload: dict` field is removed. The `raw_type` field stays.

### Firm 3: ContractPredictor Redundancy Should Be Evaluated, Not Decided Now

My cross-review identified (Surprise 4) that ContractPredictor is architecturally isomorphic to ConvergenceAlerter for the defense sector specifically, and that wiring both creates systematic confidence inflation for defense signals (Beta's Failure Mode 5.4).

I stand by the framing I used in the cross-review: this is a design decision that should be evaluated, not a finding that one approach is clearly wrong. The two positions are:

- **Option A (decompose):** Retire ContractPredictor. Route SAM.gov + hiring + insider signals through ConvergenceAlerter directly. Less precision, better calibration, no double-counting.
- **Option B (pre-filter):** Keep ContractPredictor as a domain-specific synthesizer. Route its `ContractPrediction` output to ConvergenceAlerter as a `domain="contracts"` signal. Address double-counting by making ContractPredictor NOT publish the raw signals it consumed — only its synthesized output.

Option B is the simpler implementation path and avoids redesigning ContractPredictor. The double-counting failure mode Beta identified is real but mitigated if the rule is: ContractPredictor consumes signals without publishing them to the raw channel; it only publishes its synthesized ContractPrediction. ConvergenceAlerter then sees the prediction but not the component signals again.

This is a judgment call that belongs in Phase 4 synthesis. I am not deciding it here, but I am providing the mitigation that makes Option B viable without full confidence inflation.

### Firm 4: 45-Day STOCK Act Timing Requires transaction_date for MarketSignal.timestamp

My original finding (Part 10.5) established that `CongressionalTrade.transaction_date` is the correct timestamp for signal age and decay, not `disclosure_date`. Alpha's cross-review confirmed this with additional analysis: using disclosure_date would cause VelocityDetector to see a cluster of 45-day-old trades arrive simultaneously as a velocity spike, misclassifying delayed reporting as a sudden surge.

No cross-review disputed this. The congressional trade adapter must map `transaction_date` (not `disclosure_date`) to `MarketSignal.timestamp`. This is held firm.

### Firm 5: The Source-to-Domain Mapping Table (Part 6.2) Is Correct

Alpha and Beta both accepted the source-to-domain mapping without challenge. Beta's adversarial review did not find a case where my domain assignments were wrong. The one mapping worth clarifying after cross-review: `HiringSignal` → `domain="institutional"` — I originally mapped this to `domain="institutional"` in the table. The ConvergenceAlerter domain_categories map puts "government" and "contracts" under "institutional." There is no "hiring" domain. My mapping stands: hiring signals map to "institutional" and will be grouped with government/contract signals in cross-domain category counting. This is intentional — they are all institutional sources.

---

## Section 3: New Synthesis Position — EventBus Channel Name Resolution

Beta identified (Section 5.1) that my channel naming proposal and Alpha's are incompatible. This is correctly rated a complete integration failure if left unresolved. My original channels used three-part naming (`market.edge.cluster_detected`, `market.intel.convergence`). Alpha's used two-part naming (`market.cluster_signal`, `market.convergence_alert`).

**The constraint that resolves this:**

From my original findings (Part 4.2): `event_bus.py` lines 98-119 show that when ConnectionRegistry is sealed, the triadic compliance check extracts the registered system name as `channel.split(".")[0]`. This means the first segment of the channel name must match a registered system name. The registered market system name in Alpha's bootstrap is `"market"` (or more specifically, the subsystem names like `"market-sensing"`, `"market-edge"`, `"market-learning"`).

If the channel prefix is `"market"`, ConnectionRegistry will look for a system registered as `"market"`. If subsystem names are the registered entities (e.g., `"market-edge"`), then the channel prefix must be `"market-edge"`. Three-part names like `"market.edge.cluster_detected"` extract only `"market"` as the prefix — which is the wrong system name unless "market" itself is registered as the root system.

**Resolution: two-part naming with descriptive suffixes.**

The correct convention, following how `external.py` registers channels (`cognition.decision_routed`, `external.response_received`, `pattern.advisory`), is:

```
{registered-system-name}.{event_noun}
```

Where `{registered-system-name}` is the exact string registered with ConnectionRegistry. Alpha's bootstrap plan registers `"market_sensing"`, `"market_edge"`, `"market_learning"` as system names (using underscores, not hyphens, to match Python identifier convention used elsewhere).

**Canonical channel list (resolves the conflict):**

```
# Published by market_sensing subsystem
market_sensing.signal_received       # Raw normalized MarketSignal arrives
market_sensing.signal_scored         # ThompsonSampler weighting applied

# Published by market_edge subsystem
market_edge.cluster_detected         # ClusterDetector fires ClusterSignal
market_edge.correlation_found        # PoliticianTracker fires CorrelationSignal
market_edge.filing_anomaly           # FilingTimeAnalyzer fires timing signal
market_edge.contract_predicted       # ContractPredictor fires ContractPrediction

# Published by market_learning subsystem
market_learning.velocity_anomaly     # VelocityDetector flags anomalous velocity
market_learning.correlation_anomaly  # CorrelationTracker flags cross-domain anomaly
market_learning.convergence_alert    # ConvergenceAlerter fires ConvergenceAlert
market_learning.trade_signal         # Final TradeSignal with Thompson weighting

# Published by market_sensing (feedback)
market_sensing.outcome_observed      # Price checked, outcome computed
market_sensing.prediction_result     # ThompsonSampler.update() called, result logged

# Discovery stream
market_learning.discovery_anomaly    # Novel correlation from CorrelationTracker
market_learning.regime_shift         # Velocity divergence across domains
```

**Why this resolves both sides:**

- My original proposal's three-part names (`market.edge.cluster_detected`) had the right semantic grouping but the wrong prefix structure for ConnectionRegistry compliance. The two-part names above preserve the semantic grouping via the subsystem prefix.
- Alpha's two-part names (`market.cluster_signal`, `market.convergence_alert`) had the right length but used "market" as the prefix, which would require "market" itself to be registered as a triadic system rather than the subsystems. The canonical names above use the subsystem names that Alpha's bootstrap actually registers.

**The one Alpha channel name that must be retained:** Alpha used `market.convergence_alert` as the channel published to the EndocrineSystem (Alpha Part 11). Under the canonical naming, this becomes `market_learning.convergence_alert`. Alpha's endocrine coupling must be updated to subscribe to this channel name. This is a one-line change in the bootstrap.

---

## Section 4: The Corrected Priority Order

This supersedes the priority order in my cross-review. Changes from that version are noted.

**Pre-implementation fixes (must precede any code):**

1. **Fix `trade.is_purchase` and `trade.shares_traded`** — guaranteed crashes. (Lead + Beta, unchanged)
2. **Fix `jobs_30d` baseline computation** — use `jobs_7d / 7` not `jobs_30d / 30`. (Beta, unchanged)
3. **Reset ThompsonSampler:** set `DEFAULT_PRIOR_SCALE = 2`, set `min_variance = 0.001` in `get_uncertain_signals()`, delete and regenerate `thompson_distributions.json`. **(REVISED — these must happen together, and the file must be regenerated, not patched.)**
4. **Fix VelocityDetector to compute velocity in per-day units** — change `total_seconds()` to `total_seconds() / 86400`. **(REVISED — this is now a pre-wiring requirement, not a wiring detail.)**
5. **Replace hardcoded `http://localhost:6333` Qdrant URLs** with configurable parameter in three files. (Beta + Lead, unchanged)
6. **Fix hash() Qdrant point IDs** to use `uuid.UUID(cluster_id).int`. (Beta, unchanged)
7. **Replace all `print()` with structured logging** across 16 market module files. (Alpha, unchanged)

**Architectural prerequisites (before bootstrap wiring):**

8. **Write `mae_core/market/signal.py`** with `MarketSignal` dataclass — use revised field list (remove `raw_payload`, keep `raw_id` + `raw_type`). (Lead, field correction added)
9. **Write source adapters** for each data type mapping to MarketSignal. One adapter per source type.
10. **Set ConvergenceAlerter instantiation to `min_domains=3`** — enforced at bootstrap, not left as default. **(REVISED — this is now a bootstrap enforcement requirement.)**
11. **Wire VelocityDetector output** to ConvergenceAlerter via normalizer using per-day velocity values.

**Bootstrap wiring (Layer 33):**

12. **Instantiate OutcomeTracker** as a 15th market system. **(NEW — required for learning to occur.)**
13. Implement Alpha's `market.py` bootstrap with the corrected subsystem naming for channels.
14. Add `get_statistics()` adapters to market systems for HolonProxy delegation.
15. Add three stem cell roles (SEC_WATCHER, CONTRACT_TRACKER, MARKET_ANALYST).
16. Wire ConvergenceAlerter to EndocrineSystem on `market_learning.convergence_alert` channel.
17. Add deduplication state to convergence step hook. (Alpha's cross-review design is correct)

**Before live operation (calibration and safety):**

18. Add alert deduplication to ConvergenceAlerter — suppress re-alert within minimum interval. (Beta)
19. Cap `self.alerts` to deque with maxlen=1000. (Beta)
20. Add timezone handling to FilingTimeAnalyzer. (Beta)
21. Increase CorrelationTracker.min_observations from 10 to 30. (Beta)
22. Implement Bayesian forgetting via multiplicative decay in ThompsonSampler. (Beta)
23. Add thread locking to ThompsonSampler file writes. (Beta)
24. Move learning_config history log path to DATA_DIR. (Beta)
25. Use real contact email in SEC_USER_AGENT. (Beta)
26. Evaluate ContractPredictor decomposition (Option A vs. B with mitigation). (This review, Phase 4)

**Deferred (Phase 2):**

27. Full ApiGateway routing for six market-specific API clients.
28. Regime-aware Thompson Sampling.
29. CorrelationTracker deque persistence across restarts.
30. TickerResolver service for company-name → ticker mapping.

---

## Section 5: What Did Not Change From Original Findings

The following findings from my Phase 1 investigation were confirmed by cross-review without requiring revision:

- The complete data flow architecture diagram (Part 11) — structurally correct, channel names now updated per Section 3 above
- Source-to-domain mapping table (Part 6.2) — confirmed correct
- Strength normalization rationale (Part 6.3) — confirmed correct
- The `HiringSignal.spike_ratio` normalization (`min(1.0, ratio / 5.0)`) — correct formula; Beta's `jobs_30d` bug affects the input to spike_ratio computation, not the normalization formula itself
- Congressional trade timing requirement (use `transaction_date`, not `disclosure_date`) — confirmed by Alpha cross-review
- Price fetcher as ground truth for feedback loop — confirmed; gap was the missing OutcomeTracker, not the function itself
- ApiGateway registered providers (MarketAux, Finnhub, Alpha Vantage, Tavily) and how they join the signal mesh via `external.response_received` — confirmed, no challenge received
- The `discovery_log.jsonl` is write-only with no reader — confirmed by my cross-review; remains a future work item
- The InsiderTrade attribute bugs (`is_purchase`, `shares_traded`) — confirmed as guaranteed crashes by all three investigators

---

## Summary of Revisions

| Issue | Original Position | Revised Position | Driver |
|-------|------------------|------------------|--------|
| Velocity units | Wiring gap | Unit mismatch + wiring gap; fix VelocityDetector to per-day first | Beta (confirmed in source) |
| Thompson prior scale | Reasonable starting infrastructure | Dangerous overconfidence; reset to DEFAULT_PRIOR_SCALE=2 + min_variance=0.001 + regenerate JSON | Beta (confirmed in source) |
| OutcomeTracker | Assumed to exist | Must be built as Layer 33 component #15 | Beta (5.3) |
| min_domains | Data source diversity risk | Bootstrap enforcement: instantiate with min_domains=3 explicitly | Beta + Alpha cross-review |
| raw_payload field | Include in MarketSignal | Replace with raw_id; full payload bloats Qdrant | Beta (5.7) |
| EventBus channel names | market.edge.* / market.intel.* | market_edge.* / market_learning.* / market_sensing.* | This review (resolves Lead-Alpha conflict) |
| Feedback loop completeness | `price_fetcher_for_outcomes()` nearly sufficient | That function is one building block; OutcomeTracker is the missing structure | Retraction of cross-review overclaim |

The triadic investigation process produced a more complete and implementable architecture than any single-lens investigation could have. The most consequential single finding remains Beta's: the ThompsonSampler's core promise (Bayesian reliability learning) is currently blocked at both the prior level (overconfident, unexplorable seeding) and the loop level (no OutcomeTracker to call update()). The system as currently designed would accumulate confidence in signals it has never validated. Both blockers are pre-implementation requirements that must be resolved before any wiring work begins.
