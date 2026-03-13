# Tension Report: Multi-Analyst Architecture Council
## Date: 2026-03-13
## Author: Tension Analyst (Claude Sonnet 4.6 — council observer, not participant)

---

## Preface: What This Document Is

The council produced a synthesis. The synthesis looks clean. This document exists because clean synthesis can mask the most important findings. Tension between what an agent scored and what they decided is where the real signal lives.

I am not analyzing whether the architecture is correct. I am analyzing whether the council reasoned correctly. Those are different questions.

---

## Part 1: Individual Score-Decision Alignment

### Codebase Analyst (CA)

**The central tension: reversibility score contradicts urgency framing**

The CA scored Reversibility at 10/10 — "genuinely near-zero rollback complexity." But their entire recommendation escalation in Phase 2 (revising Overall Risk from 8/10 to 7/10, endorsing the DA's sequencing) is a timing argument. If something is 10/10 reversible, urgency arguments carry less weight. You cannot simultaneously argue "this is so reversible it doesn't matter when we do it" and "we should defer it for operational reasons." These positions are in tension.

The CA's Phase 1 Feasibility score was 9/10. Their Phase 2 revised assessment after reading the DA was: "The architecture will produce value — eventually. The risk is opportunity cost." This is a meaningful intellectual revision — but they did not revise the Feasibility score downward. They left 9/10 standing while endorsing the DA's 6/10 framing as "more accurate when the operational context is considered." This produced a split verdict that the synthesis quietly resolved by accepting the 7/10 Overall Risk number — splitting the difference — rather than forcing a decision about which feasibility framing is correct.

**The blast radius score (8/10) is probably too optimistic**

The CA scored Blast Radius at 8/10, citing "additive only" changes. But the CA's own findings note: market_hooks_steps.py at 577 lines (over cap), market_systems.py at 512 lines (over cap), and the 200-step block at 100 lines already needing a helper extraction. Every one of these overages is a latent refactoring risk that becomes active the moment new code is added. An 8/10 blast radius score should reflect the refactoring chain required before the additive work can happen. The CA flags the risk in the Concerns section but does not incorporate it into the score.

**Score-decision alignment: MODERATE.** The CA's scores were consistently optimistic. Their Phase 2 decisions were more cautious. The gap is real but not egregious — it reflects genuine learning from DA's live-data findings.

---

### External Researcher (ER)

**The most significant score-decision misalignment in the council**

The ER scored Blackboard / SituationBoard Integration Effort at 6/10 — mid-range difficulty. Their synthesis then proposed that the SituationBoard sit UPSTREAM of ConvergenceAlerter, with analysts feeding into it and ConvergenceAlerter reading SituationReports instead of raw signals. The CA immediately identified this as violating the explicit brief constraint ("Do NOT modify the convergence engine"). The DA scored the same design at Risk 7/10, not 4/10.

The ER's 6/10 integration effort and 4/10 risk for the SituationBoard are defensible scores IF the board sits downstream. They are not defensible for the upstream architecture the ER proposed. The ER appears to have scored one design and then recommended a different, riskier one. This is the sharpest score-decision divergence in the entire council.

**The Temporal Analyst (Novel B) score revision is an admission the original score was wrong, not a genuine revision**

The ER scored Temporal Analyst Relevance at 9/10. After reading the DA's data, they revised this to 6/10 for near-term relevance while maintaining 9/10 long-term. This is a reasonable position. But notice what the revision does not do: it does not revise the Integration Effort score (7/10) or acknowledge that a system producing vacuous near-term output is a harder integration than a system producing valuable near-term output. The near-term/long-term split feels like an attempt to preserve the original score while incorporating the DA's critique. A more honest revision would have said: "my integration effort score implicitly assumed data maturity I cannot verify — revising down."

**The ER's most valuable finding never received a score**

The ER identified that the sensing hook has a single-consumer design — it feeds one convergence alerter, and the tiered alerters were starved because of this architectural property. This is a structural root cause finding, not just a "tiered alerters got no signals" observation. It means multi-analyst architectures face a fan-out problem that has not been solved. This insight appears in the ER's Challenge document under Surprise 1 and in their synthesis recommendations. It never received a score. The most consequential finding the ER produced was never quantified — which meant the synthesis could absorb it as a step-5 concern rather than a blocking architectural constraint.

**Score-decision alignment: WEAK.** The ER's decisions drifted significantly from their scores when challenged. The upstream architecture proposal was not in the scoring table at all. The scoring table and the architecture proposal appear to have been written independently.

---

### Devil's Advocate (DA)

**The DA's scores are internally consistent but strategically mismatch their final position**

The DA scored:
- Failure Probability: 5/10 ("likely to be built correctly but unlikely to change outcomes")
- Failure Severity: 8/10 ("failure here is not catastrophic — it is additive waste")
- Overall Risk: 6/10

These scores, taken together, describe something that will probably fail in its value delivery, but the failure is cheap and reversible. That scoring profile is actually an argument FOR building it, not against. If failure probability is 5/10, reversibility is 9/10, and failure severity is 8/10 (trivial), then the expected cost of trying is very low. The DA's conclusion — "Build the situation board. Do not build three analysts yet" — is more conservative than the DA's own scores support.

The DA appears to have experienced the classic advocate problem: they found compelling counter-evidence and concluded "don't build this" when their scores said "building this has low expected cost." The DA would have been more internally consistent at either: (a) lower scores (4/10 Failure Severity = "meaningful waste") to justify the conservative conclusion, or (b) accepting that 5/10 failure probability + 9/10 reversibility = "worth trying with safeguards."

**The orphan argument is the DA's strongest point but was not quantified**

The DA's most powerful argument is: "ctx.inevitabilities has never triggered a single trade. Three analysts reading it will produce three layers of analysis on top of data that changes nothing." This is exactly right, and it changed the CA's thinking. But notice how the DA scored it: it appears under "Failure Mode 1 (Probability: HIGH)" — a qualitative high, not a number. If the DA had given this a score (say: orphan problem means 85% of the proposal's value is unreachable without a prior wire), the synthesis would have been forced to address it as a quantitative gate. Instead, "HIGH" was absorbed into a 5/10 Failure Probability, which is median — which is ambiguous.

**The DA's score of Assumption Fragility (4/10) underweights their own evidence quality**

The DA found that three key assumptions were contradicted by live data. 4/10 on a 10-point scale where 10 = "all verified" means "most assumptions are wrong." The DA's evidence for this is exceptional: they pulled live daemon logs, JSON files with actual counts (23 total outcomes, 4 combo stats, 2 Granger findings), real step-time measurements (0.14 steps/sec), real signal distributions (38.6% sec_form4, not 72% TA). This is the best empirical work in the council. A 4/10 is actually a strong score for the DA's purposes — "this proposal has fragile foundations" — but the scale inverts the intuition. Readers scan 4/10 and think "low confidence." What it means here is "most assumptions don't hold." The DA should have flagged this scale ambiguity.

**Score-decision alignment: MODERATE.** The DA's scores support a more permissive conclusion than they stated. The internal consistency problem is real.

---

## Part 2: Cross-Agent Shared Dimension Comparison

### Overall Risk (CA: 7/10, ER: 7/10 implied, DA: 6/10)

**Spread: 1. Apparent agreement. Actual: disagreement about what "risk" means.**

All three agents rated this within one point of each other. The synthesis interpreted this as "broadly agree this is medium-risk." But the agents were measuring different things.

CA's 7/10 (revised from 8/10): "Additive architecture. The only risk is hitting the 500-line cap on two files." Risk = implementation risk.

DA's 6/10: "Safe to build, but likely to deliver less value than estimated and may delay fixing actual operational failures." Risk = opportunity cost risk.

ER's 7/10 (implied from score table context): Based on external literature suggesting architectures of this type are well-proven. Risk = pattern adoption risk.

Three agents scored the same number while measuring implementation risk, opportunity cost, and pattern adoption risk. The synthesis averaged them and called it "agreement." It is not agreement — it is three different assessments that happened to produce similar numbers. When the synthesis recommends Phase C gates, it is operationalizing the DA's risk framing (opportunity cost). When it recommends the CA's bootstrap pattern, it is operationalizing the CA's risk framing. The synthesis silently selected which risk framing to apply in each phase without naming the selection.

### Reversibility (CA: 10/10, ER: 8/10, DA: 9/10)

**Spread: 2. Genuine mild disagreement. Not examined by synthesis.**

The CA scored 10/10: "Each analyst is a try/except block. If any fails, ctx.analyst_X = None." This is reversibility at the code level.

The ER scored 8/10: slightly lower, but no specific justification given for the deduction.

The DA scored 9/10: essentially agrees with the CA but leaves room for "the situation board involves modifying bootstrap files which could have unexpected interactions."

The synthesis does not note this spread at all. It cites 9.0 average and moves on. The interesting question the spread raises is: does the DA's 9/10 (rather than 10/10) reflect a tacit acknowledgment that the bootstrap file modifications carry slightly more risk than the CA claimed? The CA is explicit that bootstrap extraction to market_analysts.py is additive-only. The DA is slightly less confident without explaining why. This 1-point divergence from an agent whose entire job was to find fault is actually a quiet endorsement of the CA's reversibility assessment.

### Evidence Confidence (CA: 9/10, ER: 8/10, DA: 8/10)

**Spread: 1. Agreement. But note: the DA's evidence is a different KIND of evidence.**

The CA's 9/10 is based on "every claim references a specific file and line number." This is structural evidence — what the code says.

The ER's 8/10 is based on 12 external sources and 8 search chains. This is precedent evidence — what published systems do.

The DA's 8/10 is based on live daemon logs, actual JSON file contents, real step-time measurements. This is empirical evidence — what is actually happening right now.

Three agents, three epistemologies, same scores. The score uniformity obscures something important: the DA's evidence is the most operationally relevant. "Post-mortem has 4 combo_stats" is more decision-relevant than "the pattern has a direct precedent in our codebase" and more current than "FinCon used a similar architecture." The synthesis treats all three evidence bases as equivalent. They are not.

---

## Part 3: Score-Decision Tension Map

### Where scores and decisions diverge most sharply

**Tension 1: CA scored Feasibility 9/10, then recommended deferring Phase C**

The CA's 9/10 Feasibility means "this is highly likely to be built correctly and produce the expected outcome." Their Phase 2 revised recommendation is "build three analyst classes once post-mortem has 50+ combo_stats and Granger has 10+ findings." These two positions are in tension unless you accept the CA's implicit argument that 9/10 feasibility describes the mechanics of construction, not the value of running the result. The CA explicitly distinguishes "build feasibility" vs "outcome feasibility" when challenging the DA's 5/10. But by doing so, they expose that their own 9/10 was also measuring only build feasibility — which is the less interesting number.

**Tension 2: ER scored Temporal Analyst Relevance 9/10 and recommended building it**

After revising to 6/10 near-term relevance, the ER still recommended building the Temporal Analyst — just sequenced after other steps. The revised score (6/10 near-term) says "this will produce limited value for the foreseeable future." The decision says "build it in Phase C." For a system that will produce limited value for months, building it is a harder sell than the score acknowledges. The ER appears to have updated their score in response to the DA but kept their decision constant — which means the score revision was cosmetic, not functional.

**Tension 3: DA scored Failure Probability 5/10 and recommended "build the situation board only"**

5/10 failure probability means "coin flip on whether this produces value." This is not strong evidence for a conservative "don't build" recommendation. If you score something at 50-50 and it is 9/10 reversible, the expected value calculation favors building it unless the opportunity cost is explicitly priced in. The DA prices in opportunity cost verbally ("MEMORY.md directive: stop building, start running") but does not incorporate it into any score. The opportunity cost argument is their strongest argument and it lives outside the scoring framework.

**Tension 4: ER scored Blackboard / SituationBoard Risk at 4/10, then proposed upstream architecture**

This is the sharpest tension in the council. A 4/10 risk score is relatively low risk. The architecture the ER proposed — ConvergenceAlerter reading SituationReports instead of raw signals — is high risk by the brief's own stated constraints. The CA and DA both flagged this explicitly. The ER's score and architecture were not describing the same thing.

---

## Part 4: Observable Weighting Analysis

### Which dimensions actually drove the decisions

The scoring framework offered: Feasibility, Blast Radius, Pattern Consistency, Dependency Risk, Reversibility, Evidence Confidence, Overall Risk. These are seven dimensions. Which ones actually drove the final recommendations?

**What drove the CA's recommendation:**
The CA's Phase 1 conclusion was "build it." Their Phase 2 revision was "build it later." The pivot was not driven by any scored dimension — it was driven by two unscored observations: (1) post-mortem has 4 combo_stats, and (2) the MEMORY.md "stop building" directive. Neither of these appears as a named dimension in the CA's scorecard. The revision was driven by evidence from outside the scoring framework.

**What drove the ER's recommendation:**
External literature quality. The ER cited production systems at scale (FinCon NeurIPS 2024, Google Research benchmarks). Their recommendation tracks most closely to Maturity and Community Health scores, not Risk or Integration Effort. In their score table, the highest-scoring approaches on Maturity (9/10: Blackboard, Manager-Analyst Hub) became the top recommendations. But these maturity scores are for external systems — they describe how proven the pattern is elsewhere, not how likely it is to work in MIDGE. The ER implicitly equated external pattern maturity with local implementation success probability. These are different things.

**What drove the DA's recommendation:**
The DA's decision was driven entirely by three unscored factors: (1) ctx.inevitabilities has never triggered a trade, (2) the MEMORY.md "stop building" directive, and (3) live data sparsity. None of these is a named scoring dimension. The scoring framework — Failure Probability, Failure Severity, Assumption Fragility, Hidden Complexity — was essentially decorative in the DA's case. They gathered the live data, formed a conclusion, and the scores were assigned afterward in a way consistent with that conclusion. This is not a criticism of the DA's reasoning — their live-data approach was the most operationally grounded work in the council. It is an observation that the scoring framework did not capture what drove their decision.

**The synthesis was driven by:**
Majority agreement on the SituationBoard + DA's sequencing logic + CA's architecture blueprint. The actual scores played almost no role in the synthesis. The synthesis document does not cite a single score to justify a recommendation. It cites "the council converged" and "unanimously agreed" — which is qualitative convergence, not score-based synthesis.

**Conclusion:** The scoring framework was a legitimacy mechanism, not a decision mechanism. All three agents made their substantive decisions based on unscored factors (live data, opportunity cost, external maturity, law alignment) and then populated scorecard dimensions. The scores added structure and communication clarity. They did not drive any of the actual recommendations.

---

## Part 5: Confidence Calibration

### Who is over-confident

**The CA is over-confident about Blast Radius**

8/10 Blast Radius with two files already over the 500-line cap and three new files to create is generous. The CA's own Concern #2 notes: "market_hooks_steps.py is 577 lines (already over cap). Any addition needs an extraction." An extraction that is a prerequisite for the actual work is itself blast radius. The CA describes this risk accurately in the text and gives it insufficient weight in the number.

**The ER is over-confident about Integration Effort for the Temporal Analyst**

7/10 Integration Effort for a system whose three primary inputs (lag_correlations: 69 entries dominated by two sources, Granger: 2 findings, cascade energy: 0 confirmed chains) cannot support its purpose. The ER's defense is "the Temporal Analyst can read other analysts' SituationReports instead" — but this redesign appeared in the Challenge phase, not in the original findings. The 7/10 score was for the original design. The defense was for a different design. A system that requires redesign to function is not 7/10 integration effort.

**The ER is over-confident about ConvergenceAlerter upstream architecture**

No score was given for the proposal that ConvergenceAlerter should consume SituationReports. Had it been scored, the Risk would have been materially higher than the 4/10 the ER gave to the downstream SituationBoard. The ER proposed a higher-risk architecture without acknowledging the risk increase.

### Who is under-confident

**The DA is under-confident about their own bottom line**

The DA scored Failure Probability at 5/10 (coin flip) and Reversibility at 9/10, which together suggest the expected cost of trying is very low. But the DA concluded "do not build three analysts yet" — a more conservative stance than those scores support. The DA's actual concern (opportunity cost, operational debt) deserved its own scored dimension. Without it, the DA's scores imply they are less confident in their "don't build" recommendation than they actually are. Their verbal arguments are more forceful than their scores.

**The DA is under-confident about the tiered alerter lesson's generalizability**

The DA uses the tiered alerter failure as their primary warning analogy. But they score it as a medium concern (embedded in 5/10 Failure Probability). The correct reading of the tiered alerter failure is: MIDGE's sensing hook does not fan out. This is a root cause, not a parallel risk. If the sensing hook's single-consumer design is the root cause, then three new analysts — which do not require the sensing hook to fan out — are not structurally parallel to the tiered alerters. The DA identifies this risk correctly in principle but does not follow it to its full structural implication: the real risk is not "analysts produce empty output" but "the fan-out architecture has not been decided."

### Whose confidence is best calibrated

**The CA** — their scores and reasoning are internally consistent, they updated correctly when presented with new data (revising Overall Risk from 8/10 to 7/10 and endorsing sequencing), and they were honest about the limits of their Phase 1 analysis ("I did not check post_mortem_insights.json"). The CA's Phase 2 document is the most epistemically honest of the three challenge documents.

---

## Part 6: What the Tensions Reveal — The Meta-Insight

### Finding 1: The council had three different proposals, not one

Reading all nine documents carefully, the three agents were not all evaluating the same architecture:

- CA's proposal: Three analysts that READ ctx.inevitabilities downstream of DeepAnalyst, write to a SituationBoard, zero modification to protected systems.
- ER's proposal: Three analysts upstream of ConvergenceAlerter, Domain Analyst reads raw signals, ConvergenceAlerter reads SituationReports.
- DA's non-proposal: Don't build three analysts. Build a SituationBoard only. Wire inevitabilities to paper trading gate.

The synthesis selected the CA's architecture, filtered out the ER's upstream approach, and adopted the DA's sequencing. But because the synthesis described this as "the council converged," it obscured that the three agents were scoring different things. The low score spread (Overall Risk: 6-7, Reversibility: 8-10) is not agreement — it is three different architectures that all happen to carry similar risk profiles. The CA's architecture is additive and low-risk. The ER's architecture is a pipeline restructure and higher-risk. The DA's minimal approach is the lowest-risk. They averaged to 6.7 not because they agreed but because high-risk and low-risk averaged to medium-risk.

### Finding 2: The most important question was never scored

All three agents agreed: ctx.inevitabilities has never triggered a trade. This is the pivotal fact of the entire analysis. DeepAnalyst runs. It produces Inevitability objects. Those objects inform zero decisions.

No agent gave this a score. The DA called it "Failure Mode 1 (Probability: HIGH)." The CA called it "Opportunity 2." The ER called it "the critical gap." Agreement on diagnosis, no agreement on severity framing. The synthesis resolved this by making Phase A (wiring inevitabilities) a prerequisite — which is the correct call. But the lack of a score for the orphan problem means there is no quantitative record of how severe this was judged to be. When future agents review this council, they will see a synthesis recommendation to "wire inevitabilities first" without understanding that all three agents considered this to be the single most important finding.

### Finding 3: The DA's most important contribution was methodological, not architectural

The DA's live-data findings (daemon logs, JSON file contents, signal distributions, step-time measurements) changed the council's conclusions more than any architectural argument. The CA revised their Overall Risk score downward and endorsed sequencing explicitly because of the DA's empirical findings, not because of the DA's architectural counter-proposals. The ER revised the Temporal Analyst's near-term relevance score from 9/10 to 6/10 because of the DA's data, not because of theoretical counter-arguments.

The DA's alternative proposals (enhance DeepAnalyst with horizon parameter, wire inevitabilities to decision gate) are reasonable but not novel. What was novel was the empirical investigation: checking actual JSON files, running the daemon, measuring real outputs. This methodological contribution — "look at what actually exists before proposing what should exist" — is the most transferable lesson from this council. It should be standard practice for future councils.

### Finding 4: The council systematically underweighted the delivery problem

The DA raised this directly: "Who reads data/midge/situation.json? How does it reach Guiding Light?" The CA did not address it. The ER did not address it. The synthesis listed it as Risk #1 ("if situation.json is written but nobody reads it, we've built a second orphan") and then mitigated it with "Phase A wires inevitabilities to paper trading gate first."

But the mitigation does not solve the delivery problem — it just proves the consumption path for a different output (paper trading gate, not situation.json). A situation board that writes to a JSON file that nobody reads is architecturally identical to the tiered alerter failure. The synthesis acknowledged this risk and then proceeded to recommend building the situation board anyway, on the grounds that it is "valuable independent of whether analysts are built." This reasoning is circular: the situation board is valuable if someone reads it, and Phase A proves someone reads inevitabilities, but situation.json is not inevitabilities. The delivery problem for the situation board remains open.

The council spent considerable effort on whether to build three analysts. It spent almost no effort on how Guiding Light actually receives and acts on the output. This is the hole at the center of the entire proposal.

### Finding 5: Score convergence may have reduced intellectual diversity

The shared dimension scores (Overall Risk: 7-7-6, Reversibility: 10-8-9, Evidence Confidence: 9-8-8) produced a synthesis that described "low spread" and "broad agreement." Low spread is good in scientific experiments. In adversarial intellectual review, low spread can mean the agents are not actually disagreeing enough.

The DA should have scored Overall Risk lower — their opportunity cost argument, if incorporated as a dimension, would have produced a 4/10 or 3/10. The ER should have scored Temporal Analyst integration effort lower — their live-data update produced a verbal revision but not a score revision. The CA should have scored Blast Radius lower — their own Concerns section describes a refactoring prerequisite that the 8/10 does not capture.

If each agent had incorporated their full evidence into their scores, the spread would have been wider, the synthesis would have faced harder tradeoffs, and the final recommendation might have been more precise about which risks are speculative and which are confirmed. The apparent agreement masked three genuinely different risk assessments that got compressed into similar numbers.

---

## Part 7: Summary Table

| Tension | Agent | Score | Decision | Magnitude |
|---------|-------|-------|----------|-----------|
| Feasibility measures build, not outcome | CA | 9/10 | Defer Phase C | HIGH — changes the proposal |
| Upstream vs downstream architecture | ER | 4/10 risk (downstream) | Proposed upstream | HIGH — score describes different system |
| Failure probability supports trying, not deferring | DA | 5/10 | "Don't build analysts" | MODERATE — scores allow more than position states |
| Temporal Analyst score unchanged despite revision | ER | 9→6 relevance | Still recommends building | MODERATE — decision unchanged despite score change |
| Blast radius understates refactoring prerequisite | CA | 8/10 | Recommends extraction | LOW — acknowledged in text, not in score |
| Opportunity cost is decisive but unscored | DA | Not in framework | Drives entire conclusion | HIGH — most important factor lives outside scores |
| Delivery problem acknowledged, not mitigated | All | Not scored | Proceed anyway | MODERATE — risk named but not resolved |

---

## Closing Observation

The council reached a defensible synthesis. The three-phase approach (wire existing, build board, defer analysts) is reasonable and internally consistent. But the council reached it through mechanisms that did not include the scoring framework they created. The scores described what each agent found. The decisions were made on what each agent felt. The synthesis selected from the decisions and called it convergence.

This is not a criticism. It is a description of how structured adversarial review actually works. The value was in the friction: the DA's empirical findings changed the CA's conclusions. The ER's literature surfaced the fan-out problem. The CA's code trace corrected the tiered alerter diagnosis. That friction is more valuable than the scorecard.

The scorecard is the record. The arguments are the knowledge. Future agents reading this council should read all nine documents, not the synthesis alone — and should read this tension report before forming an opinion about the synthesis's confidence level.
