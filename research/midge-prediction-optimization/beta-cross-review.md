# Beta Cross-Review of Lead and Alpha Findings

**Reviewer:** Witness Beta (Market Microstructure, Timing, Execution)
**Reviewing:** Lead (Signal Processing / Alpha Generation) + Alpha (Risk & Calibration)

---

## 1. Reasoning Divergence Points

### 1.1 What's the Right First Step — Build or Fix?

**Lead's reasoning chain:** Signal pipeline has the right architecture but is missing critical data sources and algorithms. Priorities: (1) per-ticker convergence, (2) 10b5-1 filtering, (3) options flow data, (4) outcome collector.

**Alpha's reasoning chain:** The system's outputs are unreliable because confidence values are fabricated, outcomes are duplicated, and independence assumptions are violated. Priorities: (1) fix outcome duplication, (2) filter transaction codes, (3) establish null model, (4) fix contract symbol.

**My reasoning chain:** The system conflates signals operating on different timescales, doesn't account for economics, and can't tell leading from lagging indicators. Priorities: (1) economic noise filter, (2) 10b5-1 detection, (3) multi-timeframe windows, (4) decay rate corrections.

**Where the reasoning diverges:** Lead is solving for CAPABILITY ("what can the system detect?"). Alpha is solving for INTEGRITY ("can the system be trusted?"). I'm solving for PROFITABILITY ("can the system make money?").

These are three valid framings of the same problem. My belief: profitability is the ultimate test. You can have perfect data integrity (Alpha) and expansive capability (Lead), but if signals operate on different timeframes and get mixed together without position sizing, the system won't generate profit. Conversely, Alpha is right that profitability built on fabricated confidence is illusory — you'll size positions wrong.

**My revised position:** Alpha's integrity fixes should come FIRST because they're prerequisites for meaningful position sizing. My economic filters and multi-timeframe work are the second layer. Lead's capability expansion is the third layer, built on a foundation that can absorb new data sources without degradation.

### 1.2 Options Flow as "Highest Priority Missing Signal"

Lead argues options flow (Unusual Whales at $35/month) is the highest-priority missing data source, estimating 55-65% directional accuracy.

**Where I diverge:** I didn't mention options flow because my analysis focused on optimizing what MIDGE already has. But after reading Lead's argument, I have concerns:

1. **The 55-65% accuracy claim** applies to FILTERED options flow (sweep orders > $100K premium, OI change > 500%). The filtering is where the edge lives. Unusual Whales provides raw data; MIDGE would need a sophisticated filter layer to extract the signal from the noise. That filter layer doesn't exist yet and would need its own calibration.

2. **Latency matters enormously for options flow.** By the time MIDGE's manual scan picks up an options sweep, the options market has already moved. Options flow alpha is measured in minutes to hours, not days. MIDGE's scan cadence (manual, daily-ish) is too slow for this signal type. Options flow belongs in Tier 1 (every 2 hours) at minimum, ideally real-time.

3. **The convergence argument is compelling.** Lead's point that options flow + insider buying on the same ticker within 72 hours creates near-actionable convergence is well-reasoned. This combination specifically bypasses my latency concern because the insider trade is the confirming signal, and insider trades are actionable on a day-scale.

**My conclusion:** Options flow should be added, but AFTER the multi-timeframe architecture is in place. Without Tier 1 (tactical, 48h window), options flow signals would be mixed into the same 7-day convergence window as SAM.gov opportunities, destroying their time-sensitive information content.

### 1.3 Confidence Score Fabrication — How Bad Is It?

Alpha's central finding is that every confidence score in MIDGE is an assertion, not a calibration. Alpha provides a complete inventory of hardcoded values (0.70 insider, 0.65 congressional, 0.75 contract award, etc.) and argues none map to empirical probabilities.

**Where I partially diverge:** Alpha is technically correct that these are not calibrated probabilities. But Alpha's framing suggests the system is DECEPTIVE — "confident-sounding output that has no empirical anchor." I think the framing is too strong. The hardcoded values are reasonable priors based on practitioner experience. They're wrong at the third decimal place, but they're not random. 0.70 for insider trades is in the right ballpark based on Lakonishok & Lee (which found ~55-60% directional accuracy for insider purchases, higher for clusters). The values are optimistic but not fabricated.

**Where Alpha is undeniably right:** The problem isn't the individual values — it's that the system COMPOUNDS them. A 0.70 base confidence boosted by 0.20 for cross-domain convergence reaches 0.90, which is presented as "90% confidence." But 0.70 × 0.70 (two independent 70% signals) yields 0.49 probability of BOTH being correct, not 0.90. The additive formula is the fabrication, not the individual estimates.

### 1.4 Outcome Duplication — Severity Assessment

Alpha found the same prediction_id appearing 8x in outcomes.jsonl. Alpha calls this "the highest priority because it corrupts the entire learning layer."

**My assessment after reading Alpha's evidence:** This IS a real bug and it IS corrupting the Thompson distributions. But the practical impact is small right now because:
1. The distributions are near their seeded priors anyway (alpha+beta barely above 2.0)
2. Only 4 unique predictions exist in the entire outcomes history
3. The duplicated AAPL outcome ($185.5 to $278.12, 49.9% return) is so extreme it might be a test case, not a real prediction

**Where Alpha is right about long-term risk:** If this duplication continues undetected through 50+ scans, the Thompson distributions will become increasingly wrong. The fix is simple (deduplication by prediction_id) and prevents compounding damage. Worth doing early even if current impact is small.

### 1.5 Regulatory Risk — Should It Change Priorities?

Alpha raises SEC MNPI concerns about job posting data and congressional pattern matching. Neither Lead nor I addressed this.

**My response:** Alpha is correct to flag this, but I think the risk is overstated for MIDGE's use case. The specific concern — trading on patterns that "exploit non-public context" — applies to professional fund managers under Advisers Act 204A. If MIDGE is used by a retail individual trader (which appears to be the case), the compliance burden is dramatically lower. The SEC's enforcement history on retail alternative data usage is essentially zero.

**However:** Alpha's point about prospective committee-to-award prediction (Lead's recommendation 4.2) deserves careful thought. Building a system that SPECIFICALLY identifies when a committee member trades before a contract their committee oversees — and then trades on that prediction — is uncomfortably close to "trading on information obtained through a government position" regardless of whether the data is technically public. The fact that all inputs are public doesn't immunize the pattern itself from scrutiny.

**My revised view:** Lead's prospective committee-to-award detector (4.2) should carry a compliance flag in any trade signal it generates. The user should know they're acting on a pattern the SEC has prosecuted others for.

---

## 2. Agreements — Where Independent Work Converged

### Strong Convergence (3 of 3)

1. **10b5-1 plan contamination.** All three found this through different evidence. My contribution: Form 4 XML contains `planName` field for detection.

2. **Thompson Sampler operationally inert.** All three verified independently.

3. **Congressional small-trade noise.** All three identified the Cisneros pattern.

4. **Outcome feedback loop is the enabling fix.** Universal agreement on necessity, different priority rankings.

### Agreement with Nuance (2 of 3 + partial third)

5. **Per-ticker convergence needed.** Lead and I identified specific ticker examples (Lead: RTX, me: GD). Alpha framed it as independence violation rather than convergence gap. Same conclusion, different lens.

6. **Velocity detector disconnected.** Lead and I found this. Alpha analyzed velocity thresholds but missed the wiring gap.

---

## 3. Gaps — What Each Analyst Missed

### Lead Missed:

- **Multi-timeframe architecture.** Lead proposes lag-correlation analysis (measuring temporal lead/lag relationships) but not separate convergence windows by signal type. These are different solutions to the same problem. Lag-correlation discovers relationships; multi-timeframe windows operationalize them. Both are needed, but windows are the foundation.

- **Transaction cost analysis.** Lead's Tier 1 recommendations (congressional filter, 10b5-1 filter, per-ticker convergence, VelocityDetector) are all about signal quality. None address whether the resulting signals are ECONOMICALLY ACTIONABLE. A perfectly filtered, per-ticker convergence alert with 0.65 confidence is worthless if the expected alpha doesn't exceed transaction costs for that stock's liquidity profile.

- **Leading vs. lagging classification.** Lead treats all signals as potentially alpha-generating without distinguishing which are predictive vs. confirmatory. The contract awards section (1.4 in Lead's findings, USASpending) describes post-announcement data as "expected edge" when it's actually lagging confirmation — the market already knows about the award.

- **The domain status table bug.** I found `avg_strength` vs `strength` field name mismatch causing all-zeros display. Lead didn't catch this despite analyzing the scan output.

### Alpha Missed:

- **Decay rate analysis.** Alpha provides no position on decay rates, optimal holding periods, or information half-lives. This is surprising given Alpha's focus on calibration — the decay rate IS a calibration parameter, and every one of them is wrong by 30-100% according to my analysis.

- **Execution strategy.** Alpha's analysis ends at "the confidence number is wrong." It doesn't address how to ACT on signals even if confidence were correctly calibrated. No position sizing, no execution timing, no TWAP/VWAP considerations.

- **Scan frequency optimization.** Alpha doesn't address whether MIDGE's manual scan cadence is appropriate. The latency between information availability and MIDGE's detection is a critical variable for alpha capture.

- **Filing time analyzer disconnection.** I found that filing time behavioral modifiers don't flow into the scan pipeline. Alpha didn't examine the scan pipeline's data flow at all — Alpha's analysis is bottom-up (examining individual components) rather than end-to-end (tracing data through the full pipeline).

### Unique Value Each Analyst Provided:

**Lead uniquely provided:** Data source expansion roadmap, ensemble signal weighting via Thompson-weighted convergence, cross-company insider network concept, 8-K sentiment analysis via Ollama, temporal lead-lag analysis.

**Alpha uniquely provided:** Complete confidence calibration inventory, outcome duplication discovery, contract symbol="" feedback loop break, Bonferroni correction for CorrelationTracker, base rate neglect in ContractPredictor, regulatory/MNPI risk analysis, adversarial attack surface analysis.

**I uniquely provided:** Full decay rate calibration table with academic sources, multi-timeframe architecture (3 tiers), position sizing via fractional Kelly, transaction cost thresholds, leading vs. lagging indicator classification, execution strategy (urgency→method mapping), scan frequency tiering, domain status table display bug.

---

## 4. Surprises — Findings That Changed My Thinking

### From Lead:

**The CorrelationTracker 1-hour alignment window.** I didn't examine this. Lead points out that financial signals operating on day-to-week timescales produce near-zero Pearson correlations when aligned to a 1-hour window. This means the CorrelationTracker is structurally incapable of discovering the lead-lag relationships I'm advocating for. My multi-timeframe architecture would put signals in the right windows, but the CorrelationTracker can't find relationships BETWEEN windows. Lead's lag-correlation proposal and my multi-timeframe architecture are BOTH needed and complementary.

**The Unusual Whales ghost signal.** thompson_distributions.json tracks unusual_whales with Beta(1.0, 0.20). But no client exists. The Thompson Sampler is learning reliability for a data source that can't produce observations. This is harmless (the prior just decays toward 0.5 over time) but it's a concrete example of the gap between design and implementation.

### From Alpha:

**Base rate neglect in ContractPredictor.** Alpha found that every defense contractor starts with 0.20 confidence just for existing. The is_active_bidder flag defaults to True. This means Lockheed Martin gets 0.20 "prediction confidence" for winning any defense contract, which is actually lower than its actual base rate (~15-25% of defense contracts go to top-5 contractors). The confidence formula isn't just fabricated — it's contradictory. 0.20 is supposed to represent "we think this company will win" but it's actually BELOW the prior for large defense contractors.

**Independence violation in convergence.** Alpha's point that "insider" and "congress" domains respond to the same underlying information is important for my multi-timeframe architecture. If I put insider signals and congressional signals in different tiers (Tactical vs. Strategic), but they're responding to the same event at different delays, cross-tier convergence might create the same false independence that single-window convergence creates. I need to account for this in the multi-timeframe design — cross-tier convergence should have a LOWER boost than within-tier convergence from genuinely independent domains.

**The contract symbol="" problem.** This surprised me. I discussed contract signal timing at length — pre-announcement vs post-announcement, optimal holding periods, decay rates — without noticing that contract signals can't even participate in the feedback loop because they have no ticker. My entire decay rate analysis for contract signals is moot until they can be evaluated against price outcomes.

---

*End of Beta Cross-Review.*
