# Lead Cross-Review of Alpha and Beta Findings

**Reviewer:** Lead Analyst (Signal Processing / Alpha Generation)
**Reviewing:** Alpha (Risk & Calibration) + Beta (Timing & Execution)

---

## 1. Reasoning Divergence Points

### 1.1 Priority Ordering — Where We Disagree Most

**My #1:** Per-ticker convergence (algorithm fix that unlocks the architecture's latent power).
**Alpha's #1:** Outcome duplication bug (data integrity fix that prevents corrupted learning).
**Beta's #1:** Economic noise elimination ($50K congressional filter).

The divergence is about WHAT KIND of problem is most urgent. Alpha sees a data integrity crisis — the learning system is being fed fabricated data. Beta sees a signal-to-noise crisis — the inputs are garbage. I see an architectural gap — the right signals can't converge because the system groups them wrong.

**Where my reasoning diverges from Alpha:** Alpha is correct that the outcome duplication bug is real and serious. But fixing it produces a clean learning system with zero real observations to learn from. The Thompson Sampler with clean data but 0 real outcomes is still operationally inert. Per-ticker convergence produces immediately actionable alerts from existing data, even before the learning loop closes. My priority is: generate useful output NOW, then improve the learning loop.

**Where my reasoning diverges from Beta:** Beta's $50K filter is important but mechanical — a 1-line code change. Per-ticker convergence is architectural — it changes what the system CAN see. I'd implement Beta's filter in the same commit but rank the convergence work higher because it's the bottleneck on MIDGE becoming genuinely useful.

### 1.2 Congressional Signal Threshold — $15K vs $50K

I proposed $15,000 minimum. Beta proposes $50,000. Alpha didn't specify a number.

**Where Beta's reasoning is stronger:** Beta's argument that transaction costs kill sub-$50K signals is persuasive. At 0.10-0.30% round-trip costs on mid-cap defense stocks, a signal needs to predict at least a 0.3% move to break even. A $8K trade is noise. But so is a $20K trade. Beta's $50K threshold is better justified by the economics of actually acting on the signal. I'd revise my recommendation to match Beta's $50K.

### 1.3 Confidence as Assertion vs. Calibrated Probability

Alpha's central thesis — that MIDGE's confidence scores are assertions with no empirical basis — is a stronger and more fundamental critique than what I produced. I noted that confidence is "a static function of domain count" and proposed a calibration multiplier. Alpha went deeper: the 0.70 base for insider trades, the 0.65 for congressional — NONE of these map to anything real. The 3-decimal-place precision is actively misleading.

**Where my reasoning stopped short:** I proposed Thompson-weighted averaging as the solution (use learned reliability to weight signals). But Alpha correctly points out that the Thompson distributions themselves have zero real observations. You can't fix fabricated confidence by weighting it with fabricated reliability. The entire stack needs empirical grounding before any confidence number means anything.

**My revised position:** Alpha is right that the confidence calibration problem is deeper than I initially assessed. But the practical path forward is still: (1) close the outcome loop, (2) accumulate 50+ outcomes, (3) then calibrate. We can't wait for calibration before shipping useful output.

### 1.4 Multi-Timeframe Architecture

Beta proposes three independent convergence tiers (Tactical 48h, Strategic 21d, Thematic 90d). I proposed temporal lead-lag analysis but within a single convergence framework.

**Where Beta's framing is better:** My lead-lag analysis focuses on discovering which signals precede others. Beta's multi-timeframe architecture focuses on the simpler, more actionable question: "don't mix fast signals with slow signals in the same window." These are complementary, not competing — but Beta's is the right first step. My lead-lag analysis is Phase 2 refinement.

### 1.5 Position Sizing

I completely overlooked position sizing. Beta's Kelly criterion framework fills a gap I didn't even identify. This is a genuine blind spot in my analysis — I focused entirely on signal quality and missed the "so what do you actually DO with this signal?" question.

---

## 2. Agreements — Where Independent Work Converged

Strong convergence on these findings (all three analysts agree):

1. **10b5-1 plan contamination** is the single worst signal quality problem. All three identified it independently with different evidence paths. Lead: code review of signal.py. Alpha: JSONL data showing Pichai/Kress sales. Beta: scan report analysis showing systematic bearish bias.

2. **Congressional $8K trades are noise.** All three identified Gilbert Cisneros's portfolio rebalancing as the canonical example. Different threshold numbers but same diagnosis.

3. **Thompson Sampler is operationally inert.** All three independently verified that distributions are at their seeded priors with zero real Bayesian updates.

4. **Per-ticker/symbol convergence is missing.** Lead and Beta both identified the GD example as the canonical missed convergence. Alpha didn't specifically call for per-ticker convergence but identified that domain-level independence assumptions are violated — which is the same problem viewed from the statistical side.

5. **The outcome feedback loop is the highest-leverage long-term improvement.** All three agree the system cannot learn without it.

---

## 3. Gaps — What Each Analyst Missed

### Alpha Missed:
- **Data source gaps.** Alpha's analysis is entirely about calibration and data quality of existing signals. No mention of options flow (Unusual Whales), dark pool prints, Senate stock watcher, or any new data sources. Alpha's lens is "what you have is broken" — correct but incomplete without "what you're missing."
- **Velocity detector disconnection.** Alpha didn't identify that VelocityDetector is not wired into the scan pipeline. Alpha analyzed the velocity thresholds but not the fact that velocities are all 0.0.
- **Filing time analyzer disconnection.** Beta caught this; Alpha didn't. The filing time behavioral modifiers (Friday dump pattern) are not flowing into the scan pipeline.

### Beta Missed:
- **Outcome duplication bug.** Alpha found that the same prediction_id appears 8x in outcomes.jsonl, poisoning the Thompson distributions. Beta didn't examine the outcomes data at all.
- **Contract symbol="" problem.** Alpha found that contract signals have no ticker symbol, silently breaking the feedback loop for that entire signal class. Beta discussed contract signals extensively but didn't check the actual data field.
- **Multiple comparisons in CorrelationTracker.** Alpha's Bonferroni correction point is statistically important and Beta didn't address correlation analysis.
- **Regulatory risk.** Alpha raised SEC MNPI concerns about job posting data and congressional pattern matching. Neither Beta nor I addressed legal/compliance dimensions at all.

### What I Missed:
- **Domain status table bug.** Beta found that the report table accesses `status.get("avg_strength", 0)` when the actual field is `"strength"` — all strengths display as 0.00. I didn't catch this display bug.
- **Two-component decay model.** Beta's insight that insider trade alpha has a fast component (market reaction, 25% of alpha, decays at 0.15/day) and slow component (fundamental, 75% of alpha, decays at 0.02/day) is more nuanced than my single-rate recommendation. My decay analysis was superficial by comparison.
- **SAM.gov decay rate is dramatically wrong.** Beta calculated that SAM.gov opportunity signals should decay at 0.008/day (87-day half-life) vs current 0.04/day. I didn't examine SAM.gov decay at all. Beta is clearly correct — competitions last months.

---

## 4. Surprises — Findings That Changed My Thinking

### From Alpha:
**The independence assumption violation in convergence.** Alpha pointed out that "insider" and "congress" domains respond to the same underlying information (advance knowledge of corporate events), meaning their convergence is not independent confirmation but correlated evidence. I proposed Thompson-weighted convergence without questioning whether the domains ARE independent. Alpha is right that the confidence boost formula assumes independence that doesn't hold. This is a deeper architectural issue than I recognized.

**Contract signals have symbol="".** This means the entire contract signal class is invisible to the feedback loop. I recommended building an outcome collector without realizing that 20% of the signal types can't even participate. Ticker resolution for contract signals is a prerequisite, not an afterthought.

### From Beta:
**GD as the canonical missed convergence.** Beta found that General Dynamics has BOTH bullish and bearish insider signals PLUS hiring confirmation on the same ticker — exactly the kind of nuanced, ticker-level analysis that the convergence engine should surface but can't. This is better evidence for my per-ticker convergence argument than the RTX example I used.

**Position sizing makes the output actionable.** Without Kelly criterion or any sizing guidance, MIDGE's alerts are analytical curiosities, not trading signals. This reframing changes the priority ordering — the system needs to answer "how much?" not just "which direction?"

---

*End of Lead Cross-Review.*
