# Alpha Cross-Review of Lead and Beta Findings

**Reviewer:** Witness Alpha (Risk, Calibration, Adversarial Analysis)
**Reviewing:** Lead (Signal Processing / Alpha Generation) + Beta (Timing & Execution)

---

## 1. Reasoning Divergence Points

### 1.1 The "Build More" vs "Fix What's Broken" Tension

**Lead's primary recommendation axis:** Add new data sources (options flow, dark pool, Senate, patent data) and new edge detectors (compressed clusters, cross-company networks, 8-K sentiment via Ollama).

**My primary recommendation axis:** Fix the data integrity bugs, calibrate what exists, and don't add more until what you have produces reliable output.

**Where the reasoning chains diverge:** Lead's chain is: "MIDGE is architecturally sound but data-starved. Add more inputs, and the convergence engine will produce stronger signals." My chain is: "MIDGE's convergence engine has structural validity problems (independence violations, fabricated confidence, outcome duplication). Adding more inputs to a miscalibrated engine produces more confident wrong answers, not better predictions."

**Specific objection to Lead's #6 (Unusual Whales options flow):** Lead recommends spending $35/month on options flow data and projects "55-65% directional accuracy." This claim comes from practitioner consensus, not peer-reviewed research. The survivorship bias in practitioner reporting is severe — funds that succeed with options flow data publish their results; funds that fail close quietly. The actual edge from publicly available options flow data (which is what Unusual Whales provides — NOT proprietary order flow) is likely in the 51-54% range after accounting for data costs and transaction friction. Adding a data source with marginal edge into a convergence engine with fabricated confidence scores will produce higher-confidence alerts that are not meaningfully more accurate.

**Where Lead may be right despite my objection:** Lead's insight about options-insider convergence ("an options flow signal confirmed by insider buying on the same ticker within 72 hours is near-actionable triadic convergence") is theoretically compelling. If the independence between options flow and insider buying is genuine (it often is — options buyers and corporate insiders are different populations), this combination could produce legitimately higher accuracy. I'd accept this data source IF the convergence engine is first fixed to do per-ticker analysis AND the Thompson Sampler has calibrated distributions.

### 1.2 Decay Rate Specificity

**Beta provides a full calibration table with specific numbers.** Lead provides general observations ("the forgetting rate is too slow"). I didn't address decay rates at all.

**Where Beta's reasoning is more rigorous:** Beta cites specific academic papers (Lakonishok & Lee 2001) and derives decay rates from their findings. The two-component decay model (fast market reaction + slow fundamental drift) is a genuinely better model than the single-rate exponential used everywhere in the codebase. Lead's analysis is directionally correct but vague.

**Where I'm skeptical of Beta's precision:** Beta quotes "0.035/day" for insider trades and "0.025/day" for cluster signals as if these are calibrated constants. They are derived from a single study (Lakonishok & Lee) from 2001 on data from the 1990s. Market microstructure has changed dramatically since then — high-frequency trading, information velocity via social media, and the commoditization of Form 4 data all compress the alpha window. Beta's numbers are reasonable starting points but should not be treated as ground truth. They are priors for the Thompson Sampler, not fixed constants.

### 1.3 Position Sizing — Is Kelly Criterion Appropriate Here?

**Beta proposes full Kelly criterion framework** with fractional Kelly (25-50%) and hard caps.

**My concern:** Kelly criterion requires calibrated probability estimates (p) and calibrated expected return estimates (b). MIDGE has neither. Using fabricated confidence values as Kelly inputs produces position sizes that are mathematically well-formed but practically meaningless. Kelly at p=0.82 (from a convergence formula that produces 0.5-0.9 regardless of signal quality) and b=0.6 (from an assumed 4.8% expected gain with no backtest) is not risk management — it's applying an elegant formula to garbage inputs.

**Where Beta is right about the need:** The insight that MIDGE needs to answer "how much?" is correct. Without position sizing, the system cannot be compared to any benchmark. But the implementation should be simple rules-based (5% base allocation, scaled by domain count, hard caps) until confidence calibration is real — not Kelly, which gives false mathematical precision.

### 1.4 Priority 1 Disagreement — The Most Important Fix

**Lead's #1:** Per-ticker convergence.
**Beta's #1:** Economic noise elimination ($50K congressional filter).
**My #1:** Outcome duplication bug.

**Why I stand by my priority:** The outcome duplication bug is not just a data quality issue — it's a poison in the learning system. The Thompson Sampler has been fed 8 copies of the same AAPL outcome, inflating sec_edgar's reliability from its prior of 0.63 toward a number that represents fabricated confidence. Every subsequent scan that consults Thompson rankings will weight SEC EDGAR signals higher than they deserve. This compounds over time. The congressional filter and per-ticker convergence are important, but they produce better outputs from a SINGLE scan. The duplication bug corrupts the entire history of learned reliability.

**However:** I acknowledge that fixing the duplication bug and having zero real observations afterward means the Thompson Sampler returns to pure priors. The practical short-term impact is small. Lead's per-ticker convergence provides immediate value on the next scan. So if the question is "what produces the most value on the next scan," Lead wins. If the question is "what prevents the most damage over the next 50 scans," I win.

### 1.5 Regulatory Dimension — Neither Lead Nor Beta Addressed This

Lead's analysis is entirely technical-optimistic: "add more data, build more detectors, improve the algorithms." Beta's analysis is entirely execution-focused: "fix timing, add position sizing, tier the scan frequency."

Neither addresses whether MIDGE's most profitable signal patterns (committee-member-to-contract-award correlation, job-posting-to-contract-timing) create regulatory exposure. My Section 6 on MNPI risk and SEC enforcement theories is not an abstract concern — the SEC has brought enforcement actions against alternative data users, and the STOCK Act's weak civil penalties do not protect against Rule 10b-5 criminal theories.

**This is not a technical gap — it's a risk management gap.** If MIDGE's most profitable edge (congressional committee correlation) is also its most legally risky edge, that changes the priority ordering of what to build. Lead's recommendation #4.2 (prospective committee-to-award prediction) is the most alpha-rich AND the most legally dangerous feature. Neither Lead nor Beta flags this tension.

---

## 2. Agreements — Where Independent Work Converged

### Strong Convergence (3 of 3)

1. **10b5-1 plan contamination is critical.** All three found this independently. Lead via code review of signal.py, me via JSONL data analysis showing Pichai/Kress patterns, Beta via scan report analysis of systematic bearish bias. Three paths, same conclusion.

2. **Thompson Sampler is inert.** All three verified zero real Bayesian updates. Different evidence: Lead checked thompson_distributions.json values, I checked the samples property computation, Beta analyzed the implications for position sizing.

3. **Congressional noise from small trades dominates the domain.** Universal agreement. Different thresholds proposed but same diagnosis.

4. **The outcome feedback loop is the enabling fix for long-term learning.** All three agree MIDGE cannot improve without it.

### Moderate Convergence (2 of 3)

5. **Per-ticker convergence is architecturally missing.** Lead and Beta both provide detailed analysis. I focused on the independence violation in domain-level convergence, which is the same problem from a different angle.

6. **Velocity detector is disconnected.** Lead and Beta identified this. I analyzed velocity thresholds but didn't check the data flow.

---

## 3. Gaps — What Each Analyst Missed

### Lead Missed:

- **Outcome duplication poisoning.** Lead's feedback loop section (5.1-5.4) correctly identifies that Thompson distributions are at priors but doesn't examine outcomes.jsonl to discover the duplication bug. Lead assumes the problem is "no outcomes" when the actual problem is "corrupted outcomes."
- **Contract symbol="" breaking the feedback loop.** Lead recommends building an outcome collector without noting that contract_award signals can't participate because they have no ticker symbol.
- **Base rate neglect in ContractPredictor.** Lead doesn't examine whether the ContractPredictor's confidence formula accounts for the prior probability of any given company winning. I found it does not — every DEFENSE_CONTRACTORS member starts with 0.20 confidence automatically.
- **RSU vesting contamination.** Lead correctly identifies 10b5-1 plan sales as noise but doesn't distinguish between plan SALES and RSU VESTING (transaction code "D"), which is the larger volume contamination source in the current data.

### Beta Missed:

- **Correlation tracker multiple comparisons problem.** With 105 pairwise comparisons and no correction, expected false positive rate is 1.26 per cycle. Beta didn't examine the CorrelationTracker at all.
- **Outcome duplication bug.** Beta analyzed outcomes as expected returns but didn't check whether the outcomes data itself is valid.
- **Adversarial attack surface.** Beta's execution strategy assumes signal integrity. I identified that congressional trade data can be manufactured by anyone who knows MIDGE's politician tracker list. A sophisticated adversary could construct fake convergence patterns.
- **The contract signal symbol="" problem.** Same as Lead — Beta discusses contract timing extensively without noting the ticker resolution failure.
- **Legal/regulatory risk.** Beta's position sizing and Kelly criterion framework implicitly assumes the user can legally act on all signals. Neither Lead nor Beta addresses whether acting on certain signal patterns creates exposure.

### Unique Value Each Analyst Provided:

**Lead uniquely provided:** Data source gap analysis (options flow, dark pool, Senate), cross-company insider network concept, 8-K sentiment via Ollama, ensemble signal weighting via Thompson-weighted convergence.

**Beta uniquely provided:** Full decay rate calibration table with academic basis, multi-timeframe architecture (3 tiers), position sizing framework, transaction cost analysis, leading vs. lagging indicator classification, execution strategy (TWAP/VWAP).

**I uniquely provided:** Confidence calibration audit (complete inventory of hardcoded values), outcome duplication bug discovery, contract symbol="" feedback loop break, Bonferroni correction for CorrelationTracker, regulatory/MNPI risk analysis, adversarial scenario analysis, base rate neglect in ContractPredictor.

---

## 4. Surprises — Findings That Changed My Thinking

### From Lead:

**The Unusual Whales "ghost signal."** I didn't realize that unusual_whales is configured in thompson_distributions.json AND learning_config.py but has NO actual client implementation. This is a concrete example of the gap between MIDGE's designed architecture and its implemented state. It also means the Thompson Sampler is tracking a signal source that literally cannot produce data — wasted parameter space.

**The CorrelationTracker 1-hour alignment window problem.** Lead pointed out that financial signals operate on day-to-week timescales, making a 1-hour alignment window produce near-zero correlations. I didn't examine the alignment window. This means the CorrelationTracker is not just suffering from multiple comparisons — it's also fundamentally misconfigured to find the kinds of relationships that actually exist in this data.

### From Beta:

**The domain status table bug.** Beta found that the scan report displays all domain strengths as 0.00 because it accesses `status.get("avg_strength", 0)` when the actual field name is `"strength"`. This is a simple display bug, but it means every scan report produced so far has had a broken domain status table. The first scan report was celebrated as a success — but part of its output was silently wrong. This confirms my broader thesis: MIDGE produces confident-looking output that masks underlying data problems.

**Filing time analyzer disconnection from scan pipeline.** Beta found that the filing time behavioral modifiers (Friday dump pattern) are not flowing through to the scan. This means MIDGE's scan doesn't benefit from one of its more academically grounded signal modifiers. Combined with the velocity detector disconnection that Lead and Beta both found, we now have TWO edge detectors that are implemented but not wired into the live scan path.

**Leading vs. lagging classification.** Beta's framework for categorizing each signal as leading or lagging is genuinely useful for prioritization. I analyzed signal validity but didn't organize findings into "what can you still trade on?" vs "what's already priced in?" categories. Beta's classification makes my calibration concerns more actionable — calibrate the leading indicators first because those are the ones with remaining alpha.

---

*End of Alpha Cross-Review.*
