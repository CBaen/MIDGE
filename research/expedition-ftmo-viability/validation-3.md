# Validation Report 3
## Date: 2026-03-09
## Validator: Validator 3 (Devil's Advocate)

---

### Evidence Challenges

**1. Team 01 — The $22 challenge fee claim is wrong.**
Team 01 opens with "The $22 challenge fee is accurate" but then corrects itself within the same sentence: "FTMO's $10K account is actually €155 ~$165 USD, refunded on first payout." Web search confirms: €155 is the correct figure for the $10K account. The original memory entry ("$22 challenge fee") propagated from the sibling instance's handoff note and was never challenged before this expedition. It is wrong by a factor of 7.5x. This single error meaningfully changes the EV math: at €155 (~$165) instead of $22 per attempt, the break-even pass probability rises from a trivially-clearable bar to a meaningful one. Team 01 recalculates correctly but the framing of "low cost of entry" that permeates all four teams is based on a price point that doesn't exist.

**2. Team 01 — The 43% algo pass rate is a single-source, unverified figure.**
Team 01 acknowledges this caveat but still uses it to anchor the EV calculations ("directionally interesting"). Web search found no independent corroboration of this figure. The only verified industry-scale study (FPFX Tech, 300,000+ accounts) found 14% challenge pass rate overall with no algo/manual breakdown. The 43% figure comes from a single study by atmosfunded.com — a firm that profits from attracting algo traders. This is a marketing claim masquerading as research. The actual algo pass rate is unknown, not 43%.

**3. Team 01 — The "Hidden Risk" framing is backwards.**
Team 01 warns that "a consistently profitable algo is adversarial to FTMO's model" and recommends treating each funded account as finite. This is sound, but the framing buries the real risk: FTMO's business model depends on the 90% who fail paying fees. An algo system that consistently passes is not just at risk of termination — it is structurally exploiting the wrong end of a business model designed to extract money from losers. This isn't a mitigation question, it's a fundamental misalignment of incentives worth naming directly.

**4. Team 02 — The 30-day inactivity termination rule is absent from the analysis.**
Team 02's instrument coverage analysis concludes that FTMO-relevant signals may fire at very low frequency (possibly 0–2/day). Web search confirmed a critical rule Team 02 missed entirely: FTMO deactivates accounts with no trades recorded in 30 days. If MIDGE's FTMO-relevant signal rate is as low as Team 02 projects, the account could be terminated for inactivity before ever reaching the profit target. This creates an operational floor for signal frequency that doesn't exist in the current analysis.

**5. Team 02 — The replay_results.json is empty, which invalidates the core premise.**
Team 02 flags this: `{"alerts": [], "phase": "replay"}`. The research brief states MIDGE has "19.9% overall convergence win rate" and "best combos at 66.7%." These figures come from the Feb 2026 replay. But the replay results file is empty — meaning whatever data produced those percentages either lived in a different file, was computed in memory and never persisted, or was lost. The entire quantitative case for MIDGE-on-FTMO rests on this data, and it cannot be reproduced from the current codebase state. No team raised this as an expedition-blocking gap.

**6. Team 03 — The academic evidence is directionally relevant but not specific to FTMO constraints.**
The literature Team 03 cites (Gu/Kelly/Xiu, Fama-French, Condorcet) validates multi-domain convergence as a valid alpha source. None of it was conducted under prop firm constraints: fixed-account, drawdown-limited, single-path-to-target. A system can have positive expected value across thousands of trades and still fail an FTMO challenge with 80%+ probability due to drawdown path variance. The academic evidence validates the signal engine in isolation. It does not validate the specific deployment context.

**7. Team 04 — The "no open-source competitors" finding creates false security.**
Team 04 concludes that no functional public repos exist, implying MIDGE is building something novel. The actual inference is the inverse: if experienced algorithmic traders are NOT publishing FTMO integration code, either (a) this is a lucrative proprietary moat they protect, or (b) they've tried and the results weren't worth publishing. The absence of public implementations is not evidence of a gap MIDGE can fill — it's a potential warning signal.

---

### Contradictions Between Teams

**Contradiction 1: Signal frequency on FTMO instruments.**
- Team 01 treats 288 alerts/month as adequate raw material for FTMO challenges.
- Team 02 explicitly concludes that 97%+ of signals are US equities (FTMO does not trade equities) and estimates effective FTMO signal frequency at 0–2/day.
These cannot both be true simultaneously. Team 01 uses the raw figure without adjustment; Team 02 audits the actual instrument breakdown. Team 02's analysis is more rigorous and its conclusion is more alarming. The 288/month figure is a red herring for FTMO purposes until instrument coverage is verified.

**Contradiction 2: Confidence-based position sizing.**
- Team 01 proposes confidence-scaling risk (e.g., confidence >0.8 → 2.5% risk).
- Team 02 explicitly states: "Filtering to conf >= 0.7 does not reliably select better signals. The confidence engine has not yet been calibrated." Winners averaged 0.560 confidence, losers 0.565.
Using confidence to scale position size when confidence has near-zero predictive power doesn't just fail to add value — it adds variance without edge. Both teams discuss confidence thresholds as though they mean something; only Team 02 flags that they currently don't.

**Contradiction 3: Time-to-pass assessment.**
- Team 01 estimates 7–10 weeks at 30% WR / 5 tradeable signals per week.
- Team 02 shows that at the overall 19.9% WR (the only validated rate), the system has negative expectancy and will hit the drawdown limit before the profit target with near certainty.
Team 01's 30% WR scenario is the aspirational case. Team 02's 19.9% is the evidenced case. The teams are analyzing different scenarios without explicitly reconciling this.

---

### Alignment Issues

**1. The research framing assumes FTMO is the destination. The actual question is whether FTMO is the right deployment vehicle at all.**
All four teams treat "how do we pass FTMO" as the question and work backwards. No team asks: given MIDGE's current signal profile (primarily US equities, 19.9% overall WR, uncalibrated confidence engine), is there a better first deployment vehicle? Alpaca paper trading against US equities — where 97% of MIDGE's signals already fire — would accumulate real outcome data faster, calibrate Thompson distributions faster, and do so without the €155 per-attempt fee and the FTMO instrument mismatch. The self-funding loop Guiding Light wants may be faster through a different path.

**2. The $1,000 deployment gate exists and is not currently clearable.**
The research brief states: "Deploy capital only when MIDGE demonstrates pattern stacks with 80%+ historical accuracy." Team 02 confirms that high-confidence pattern stacks (>=0.7) are ungraded. The data to clear this gate does not exist. All four teams discuss implementation paths without acknowledging that Guiding Light's own stated precondition for deployment cannot be met today. This is a structural misalignment between the expedition's recommendations and MIDGE's own declared readiness standard.

**3. Teams discuss "fastest path to FTMO" as equivalent to "best path." These are not the same.**
Speed is not a virtue here. The research brief does not ask for the fastest path — it asks for enough confidence to decide "proceed, defer, or abandon." A fast path to a failed challenge at €155/attempt is worse than a slower path that first validates signal quality on actual FTMO instruments.

---

### Missing Angles

**1. The MT4/MT5 execution gap is identified but not sized.**
Team 04 notes that FTMO requires MT4/MT5 execution and that Alpaca is a US equities broker that cannot be used for FTMO trades. The solution proposed (MetaAPI) is mentioned in 1 paragraph. No one analyzed the operational complexity, cost, latency, or reliability of this bridge. This is not a small gap — it requires a new broker account, a new API integration, and a new order management layer before a single live FTMO trade can be placed.

**2. No one checked whether FTMO's forex/commodity instruments are accessible from MetaTrader on Windows with the account structure required.**
MIDGE runs as a daemon on Wardenclyffe (Windows 11). MT5 has a native Windows Python package but it is Windows-only and requires MT5 installed locally. MetaAPI adds cloud infrastructure. Neither path was validated against Wardenclyffe's actual environment.

**3. Regime dependency of the 75% pass rate claim.**
The sibling's 75% pass rate came from Bollinger Band mean reversion on a specific 250-day window. That window corresponds to a specific market regime (late 2024–early 2025). In trending or high-volatility regimes, mean reversion strategies can fail catastrophically. No team asked: what was the market regime during the sibling's backtest, and would the same approach survive a trending or crisis regime?

**4. The prop firm industry's structural instability.**
Multiple prop firms collapsed in 2023–2024 (MyForexFunds, others). Team 01 mentions this briefly for FundedNext but doesn't apply the same lens to FTMO. FTMO survived a 2023 Czech regulatory inquiry and restructured (rebranding FTMO "for Business"). Web search suggests FTMO is more stable than competitors, but no team investigated the current regulatory environment for prop firms in the EU or whether FTMO's structure has changed since the Czech inquiry.

**5. Tax treatment of prop firm payouts.**
Prop firm payouts are typically classified as performance fees or independent contractor income in most jurisdictions, not capital gains. This can materially change after-tax returns. Not mentioned by any team.

---

### Convergence Points

These points received independent agreement across multiple teams — treat as high confidence:

1. **The no-time-limit rule is genuinely transformative** (Teams 01, 02, 04). All three teams that touched this finding agreed independently: removing the time constraint is the most important FTMO structural advantage for a low-frequency, positive-EV system. This is also confirmed by the sibling's backtest (75% on 250-day vs near-zero on 60-day windows).

2. **The instrument coverage gap is the primary blocker** (Teams 01, 02, 04). Three teams independently identified that MIDGE's signals are predominantly US equities and that this must be resolved before FTMO is viable.

3. **The signal translator gap is well-scoped and buildable** (Teams 02, 04). Both teams estimated 100–150 lines of Python. The scope is clear and not in dispute.

4. **FTMO's "exploitative practices" clause creates unpredictable termination risk** (Teams 01, 04). Both flagged this independently, citing forum reports from 2024–2025. Web search confirmed real termination cases exist, though FTMO attributes them to rule violations rather than arbitrary terminations.

5. **The confidence engine is currently non-discriminating** (Teams 01, 02). Both teams touched the phase0 finding that winners and losers averaged nearly identical confidence (0.560 vs 0.565). The implications are most fully developed in Team 02 but Team 01's position sizing proposal contradicts this finding without acknowledging it.

---

### Strongest Case AGAINST Proceeding

The case against proceeding to FTMO now rests on a sequence of compounding gaps rather than any single fatal flaw.

MIDGE's proven statistical edge (z=4.74, p<0.0001) was measured on US equity convergence signals. FTMO does not trade US equities. Team 02 confirmed that 97%+ of MIDGE's live signals fire on instruments FTMO cannot execute. This means the proven edge is not proven on FTMO instruments — it is proven on a different market that MIDGE is not trying to trade at FTMO. The entire quantitative case must be rebuilt from scratch for FTMO-relevant instruments (forex pairs, indices, commodities), and that data does not currently exist.

Compounding this: the confidence engine does not discriminate winners from losers (0.560 vs 0.565 mean confidence across outcomes). Any FTMO strategy that relies on filtering to high-confidence signals — which all four teams propose — is filtering on noise. The Thompson distributions that would calibrate this are 81 of 83 still at uniform 50/50 priors, meaning years of collected data have produced almost no learning signal. The feedback loop that was supposed to close this gap is structurally present but has not functionally closed.

The result: MIDGE approaching FTMO today would be deploying an instrument-mismatched, uncalibrated system against a €155-per-attempt evaluation with a real (not "vague") 30-day inactivity termination rule. At the overall 19.9% WR on US equities (which is below the 23% break-even for FTMO's payout structure), the system has negative expectancy even on its home turf. On FTMO's instruments — where it has no validated edge at all — the outcome distribution is unknown.

The free trial mitigates fee risk but not the more fundamental problem: time and development attention spent on FTMO instrument adaptation, MT4/MT5 bridge construction, and signal translator development would not advance the core capabilities that produce edge. It would be integration work layered on top of an unvalidated signal stack for a new instrument class. The self-funding loop Guiding Light wants would be more efficiently reached by deploying on US equities through Alpaca (where the edge has been measured, the instruments match, and the infrastructure already exists) to accumulate real Thompson calibration data first.

---

### Strongest Case FOR Proceeding

The case for proceeding is structural, not statistical.

MIDGE's 19.9% overall win rate obscures a crucial segmentation: the best domain combinations (events+macro+price, contracts+events+insider+institutional+macro+price) achieve 29–31% win rates at meaningful sample sizes. These combos exist in the historical data. The 3.34:1 payoff ratio means that 30% WR is profitable — not marginally, but meaningfully (+0.302%/trade, or +1.04%/trade on underlying asset basis). Any system that can consistently land in the 30%+ bucket has positive expectancy under FTMO's constraints.

The infrastructure is closer than it appears. Eight sources already produce FTMO-relevant signals: COT positioning (directly tracks EUR, GBP, JPY, Gold, Crude futures), EIA energy (moves CL=F), Economic Calendar (moves every major forex pair), VIX structure (moves ES=F, NQ=F), FRED macro (moves forex), Session Sweep (instrument-agnostic ICT patterns), TA indicators (instrument-agnostic), and fractal resonance (instrument-agnostic). Adding EURUSD=X, GC=F, CL=F, and NQ=F to the watchlist activates all of these on FTMO instruments today, with no new code — just configuration.

The no-time-limit rule is the structural key that changes the math from probabilistic to near-certain. With positive expectancy and no time limit, the question is not "will this system pass" but "how long will it take." A system that fires 2 FTMO-relevant signals per week at 30% WR needs approximately 7–10 weeks to reach the 10% target. The €155 entry fee amortizes to ~$16/week during that window. This is not a high-stakes gamble — it is a low-cost, long-duration validation exercise.

The free 14-day trial changes the risk profile further: the first attempt costs nothing. If MIDGE runs for 14 days on FTMO instruments and shows signal patterns consistent with 30%+ WR on the right combos, that is the data needed to clear the $1,000 deployment gate. The free trial is not just a cost mitigation — it is a measurement opportunity.

The deepest case: MIDGE needs live data on new instruments to calibrate Thompson distributions for those instruments. The FTMO free trial is the lowest-cost way to acquire that data on forex and commodity instruments. Whether or not MIDGE passes the challenge during the free trial is secondary — the 14 days of live signal-to-outcome data on forex instruments is the asset, and it is free.

---

### Overall Assessment

The four teams produced competent, largely honest research, but with an asymmetric blind spot: Teams 01, 03, and 04 are broadly optimistic and supportive of proceeding, while Team 02's internal audit is the most rigorous document in the set and contains the most damaging findings — findings that the other teams cite selectively or not at all. The most important finding (confidence engine is non-discriminating; 97% of signals are on non-FTMO instruments; replay_results.json is empty) comes from Team 02, and none of the other teams adequately reconcile their recommendations with it.

The survivorship bias concern about pass rate statistics is legitimate. The 43% algo pass rate from atmosfunded.com is a marketing claim from an interested party; it should not be used in EV calculations. The 8–10% manual baseline is better-evidenced. The correct honest range for MIDGE's FTMO pass probability, given zero validated data on FTMO instruments, is: unknown, bracketed between 8% (manual retail baseline) and the theoretical ceiling from the system's mathematical properties.

What should be prioritized, in order: (1) Add FTMO instruments to the watchlist immediately — this is a configuration change, not a development task, and it begins accumulating the data that everything else depends on. (2) Audit 30 days of live convergence alerts to measure what fraction land on FTMO-relevant instruments — this single number determines whether FTMO is viable at all without further instrument work. (3) Register for the FTMO free trial — 14 days of free live signal data on forex/commodity instruments is the cheapest possible evidence. (4) Only after those three steps: build signal_translator.py and the execution package. Building the execution infrastructure before knowing whether FTMO-relevant signals exist at meaningful frequency is building a bridge to an island that might not be there.
