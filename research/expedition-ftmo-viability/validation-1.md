# Validation 1: FTMO Viability Expedition
**Date:** 2026-03-09
**Validator:** Sonnet 4.6 sub-agent
**Teams reviewed:** Team 01 (Prop Firm Economics), Team 02 (Capability Audit), Team 03 (Academic Evidence), Team 04 (Competitor Landscape)

---

## Validation Summary

The teams produced solid work overall, but this expedition has a hidden critical failure buried in the findings that none of the four teams surfaced prominently: **Guiding Light is based in the US, and FTMO's US channel (via OANDA partnership) closed its OANDA Prop Trader program effective March 31, 2026 — which is 22 days from today.** This is not a minor gap. It changes the entire premise.

Everything else in this validation is secondary to that finding. I will cover it first.

---

## Section 1: Evidence Challenges

### Challenge 1: The US Availability Crisis — Zero Teams Mentioned It

**Severity: Critical**

FTMO banned US traders in early 2024 due to MetaQuotes restricting prop firm licenses. In mid-2025, FTMO relaunched US access via an OANDA partnership at `ftmo.oanda.com`. However, FTMO has now announced that the OANDA Prop Trader program will **formally conclude on March 31, 2026** — 22 days from today. (Source: ftmo.com/en/press-release/oanda-prop-trader-to-conclude-as-ftmo-strengthens-its-modern-prop-focus/, verified 2026-03-09.)

Not one of the four teams mentions this. Team 01 lists FTMO as the primary recommendation. Team 04 discusses FTMO platform requirements in detail. Neither checked whether FTMO is currently accessible to US-based participants.

**What this means in practice:**
- The standard `ftmo.com` global challenges remain restricted to non-US users.
- `ftmo.oanda.com` (the US pathway) is shutting down March 31, 2026.
- Remaining US-accessible states are: all except Arkansas, Delaware, Louisiana, Montana, South Carolina.
- Even if Guiding Light is in an eligible state, any challenge started now has an unclear continuation path post-March 31.
- There is no confirmed replacement US pathway from FTMO as of today.

**Action required before proceeding:** Verify Guiding Light's state of residence. If in a blocked state, FTMO is entirely off the table. If in an eligible state, the March 31 cutoff means starting a challenge now may produce a mid-challenge orphan account.

**Implication for Team 01's recommendation:** The "start with FTMO for stability" recommendation may be unactionable for this team (the user). FundedNext or TopStep may need to be primary, not fallback.

---

### Challenge 2: The 75% Backtest Pass Rate — Assumptions Are Unstated

**Severity: High**

This claim appears in the Research Brief and is cited by Team 01 as evidence for the "no time limit = eventually pass" argument. The source is `FTMO-EXECUTION-ENGINE.md` — the sibling instance's internal document.

No team questioned the methodology. I traced it:

- The sibling backtested "Bollinger Band mean reversion on AUD/USD at 2% risk" across "250-day windows."
- This is a **single strategy, single instrument, single window length, single risk percentage.**
- The 75% pass rate reflects path randomness given those exact parameters — it is not a generalized result.
- At 2% risk per trade, the pass rate vs. a 250-day window is measuring: "does a medium-variance mean reversion strategy hit +10% before -10% with unlimited time, at aggressive sizing?"
- The answer is yes, 75% of the time — because 2% risk with a ~50%+ WR mean reversion strategy and no time limit gives favorable odds.

**What it does NOT tell us:**
- Whether MIDGE's convergence signals (19.9% WR overall, equity-focused) produce a comparable pass rate.
- Whether the 75% holds on FTMO instruments (forex/indices) rather than AUD/USD.
- Whether 2% risk sizing is safe for MIDGE's signal profile (Team 02 correctly notes the confidence engine doesn't discriminate, meaning the true win rate is the 19.9% aggregate, not the filtered subset, at execution time).

External verification: No independent source confirms a 75% prop firm pass rate for Bollinger Band strategies specifically. The search results show Connors & Alvarez (2006-2012 ETF backtests) citing >75% win rates for BB strategies on ETFs — but win rate per trade is not the same as challenge pass rate. These are different metrics that teams conflated.

**The 75% figure is real but narrowly scoped.** Applying it to MIDGE requires demonstrating MIDGE achieves comparable signal quality on FTMO instruments — which Team 02 explicitly says has not been tested.

---

### Challenge 3: The Infinite Horizon Argument Has a Specific Inactivity Constraint

**Severity: Medium**

Team 01 and Team 04 both rely on the "no time limit = positive expectancy system will eventually pass" argument. This is logically correct IF the system keeps firing trades. External verification found a constraint none of the teams mention:

**FTMO's inactivity rule: accounts terminated if no trade is placed within any 30-day period.** (Source: FTMO FAQ, ftmo.oanda.com, verified 2026-03-09.)

This directly constrains the "infinite patience" approach. If MIDGE's FTMO-tradeable signal frequency drops to fewer than one per 30 days, the account will be terminated by inactivity before positive expectancy can play out.

Team 02 estimates FTMO-tradeable signal frequency at "1-2 per week" at best — that would be fine. But this assumes the watchlist expansion (Step 1 in Team 02's fastest-path recommendations) is already done. Without adding NQ=F, ES=F, forex pairs, and CL=F to the watchlist, the FTMO-tradeable signal frequency is currently near zero.

**This is not a dealbreaker, but the infinite-horizon argument requires a maintained minimum trading cadence, which requires the watchlist fix first.**

---

### Challenge 4: The 43% Algo Pass Rate — Single Unverifiable Source

**Severity: Medium**

Team 01 cites "algorithmic traders showed ~43% pass rate in one analysis (Jan-Sep 2023, ROI-based strategies). Source: atmosfunded.com." Team 01 correctly flags this as "a single study from one period, sample size and methodology not disclosed." However, the teams continue using it as a reference point in their calculations (including Team 01's own EV calculation that uses "conservative 35% pass probability").

External verification: The search found no independent corroboration of the 43% algo pass rate figure. No peer-reviewed study, no large-sample replication. The `atmosfunded.com` source is itself a prop firm competitor — financially motivated to make algo trading appear viable.

**The 35% pass probability used in Team 01's EV calculation is not grounded in verified data.** The actual algo pass rate is unknown. The honest range remains: 8-10% (manual baseline) to 75% (sibling's narrow backtest). MIDGE's actual rate on FTMO instruments is zero — it has never been tested on those instruments.

---

### Challenge 5: Team 03 Academic Evidence — Strong But Partially Misapplied

**Severity: Low-Medium**

Team 03's academic synthesis is the strongest report. However, one application is questionable:

The citation of Gu, Kelly & Xiu (2020) establishes that nonlinear combination of signals outperforms linear models. This is cited as validating MIDGE's convergence architecture. The validation is directionally correct, but Gu et al. study US equity returns using machine learning on stock characteristics. MIDGE is being evaluated for forex/index trading via prop firm challenges. The academic evidence for multi-source fusion is primarily equity-market focused. The transfer to forex prop firm trading is an extrapolation, not a direct confirmation.

Team 03 acknowledges this indirectly ("Chinese market results may not transfer to US markets") but does not flag the equity-to-forex transfer assumption, which is at least as significant.

---

## Section 2: Contradictions Between Teams

### Contradiction 1: Signal Quality Assessment

**Team 01** (Synthesis): "At 30% WR (best tradeable combos): +0.302% per trade... viable."
**Team 02** (Section 7): "The confidence engine currently does not discriminate between winners and losers (winners 0.560, losers 0.565). The 19.9% overall rate is what MIDGE will experience in practice — not the 30%+ combo rate."

These are not reconcilable with the same system. Team 01's economics scenario requires knowing which signals belong to 30%+ WR combos at execution time. Team 02 explicitly confirms that MIDGE cannot do this — confidence scores are near-identical for winners and losers, meaning there is no reliable real-time filter to select the 30%+ combo subset.

**Verdict: Team 02 is correct.** The 30%+ WR scenario requires a combo-level filter that does not yet exist. Team 01's positive EV math is a conditional scenario ("IF MIDGE could isolate 30%+ combos"), not a description of current capability.

### Contradiction 2: Instrument Relevance Count

**Team 02** (Section 6, summary): "9 out of 32 sources are directly FTMO-relevant."
**Team 02** (Section 6, body): Lists 10 sources under "Directly FTMO-useful."

Minor internal inconsistency within Team 02: the body counts `eia_energy` separately from the summary count of 9. This doesn't change the conclusion but indicates the count was assembled ad hoc rather than audited.

### Contradiction 3: MetaAPI Assessment

**Team 04** (Finding 6): "MetaAPI is the fastest viable Python→FTMO execution bridge. Adds one more API dependency."
**Research Brief**: "Constraints: $1,000 gate before deploying capital."

No team priced the MetaAPI integration cost. External verification found MetaAPI is a paid cloud service (free tier: 100K requests/month). Given MIDGE's daemon mode running 24/7 with market hooks, request volume needs to be estimated. If the free tier is exceeded, MetaAPI costs could accumulate during challenge attempts — this is an uncosted dependency in the implementation path. No team quantified it.

---

## Section 3: Alignment Drift

### Drift 1: The Research Brief Asks for a Go/Defer/Abandon Decision — The Teams Give a Build Plan

The expected outcome in the Research Brief is: "A clear, evidence-based assessment of whether the FTMO path is viable... Guiding Light needs enough confidence to decide: proceed, defer, or abandon."

All four teams converge on "proceed, with caveats." None explicitly considers abandonment. This is alignment drift: the teams were invested in the premise (FTMO is viable) before the research began, and shaped their findings accordingly.

A neutral validator should have asked: is there any scenario where abandoning FTMO in favor of Kalshi or Alpaca paper trading (both already installed) is better? No team investigated this. The brief explicitly mentions Kalshi ("Kalshi SDK installed"). Kalshi is a US-regulated prediction market — no geographic restriction, no prop firm termination risk, direct access to event-outcome contracts. This may be a better path for a US-based user than FTMO, but it went unexamined.

### Drift 2: The $1,000 Gate Is Mentioned But Not Applied

The Research Brief states the $1,000 gate: "deploy capital only when MIDGE demonstrates pattern stacks with 80%+ historical accuracy."

Team 02 correctly notes: "The $1,000 deployment gate from Guiding Light's directive requires 80%+ historical accuracy on pattern stacks. Current high-confidence pattern stacks (>= 0.7) are ungraded. This gate is not yet clearable — the data to clear it does not exist."

However, Team 01's EV calculations, Team 04's implementation path, and all team "fastest path" recommendations proceed as if this gate doesn't apply. The gate wasn't repealed — it was quietly bypassed by framing the free trial as "zero cost, no gate needed." This is true for the free trial. It is not true for the challenge fee (€155), which teams recommend spending as the natural "next step" after the trial.

---

## Section 4: Missing Angles

### Missing: Kalshi as a Better-Fit Alternative

Kalshi (prediction markets) is mentioned in the MEMORY.md as already installed, but no team researched it. Kalshi:
- Is US-regulated, US-accessible without state restrictions
- Trades event-outcome contracts (will NFP exceed X? will Fed raise rates?) — directly mapping to MIDGE's macro + economic calendar signals
- Has no drawdown limits, no profit targets, no "exploitative practices" clause
- Allows simple yes/no binary positions with defined risk
- Signals like FOMC, CPI, NFP — already in MIDGE's economic calendar — translate directly to Kalshi contracts

The research brief explicitly identified Kalshi as an installed capability. No team researched whether Kalshi might serve the "prove monetary value" goal better than a forex prop firm challenge for a US-based operator.

### Missing: The Watchlist Fix Is a Prerequisite That Requires No Research

Team 02 correctly identifies "expand watchlist to NQ=F, ES=F, GC=F, CL=F, EURUSD=X" as Step 1. This is 1-2 hours of work. None of the teams recommended doing this immediately, before any other analysis. Given that 97%+ of current signals are on US equities (FTMO doesn't trade), ALL FTMO-relevant analysis is hypothetical until this fix is in place. The experiment cannot even begin until the watchlist is correct.

### Missing: MIDGE's Actual Performance on Futures Tickers

Team 02 notes that ES=F and NQ=F appear in live alerts. No team went back to the signal archive to count how many ES=F / NQ=F convergence alerts exist and what their convergence quality looks like. This data exists (`data/midge/signals/`) and could be extracted in under an hour. The entire "what is MIDGE's FTMO-relevant win rate?" question was left as theoretical when a partial answer was obtainable from existing data.

### Missing: The Best Day Rule Quantification

Team 01 (Gap 3) identifies the Best Day Rule as a risk but says "what to do: check the sibling's backtest results for daily profit distribution." This check was never done. Team 04 mentions the risk in Finding 3 but also doesn't quantify it. The rule can fail a challenge even with positive overall performance. This needed a concrete check, not a to-do item.

---

## Section 5: Agreements (High-Confidence Findings)

The following findings are corroborated by multiple independent teams and external verification:

**A. The "no time limit" feature is real and confirmed.** External search verified FTMO's official blog and 2026 guides confirm unlimited time challenge. The inactivity constraint (30 days) is real but manageable if signal frequency is adequate. Confidence: High.

**B. The signal translation gap is a blocker.** Teams 02 and 04 independently identified the same gap: ConvergenceAlert has no entry price, stop loss, or take profit. Both teams estimated the fix at 100-150 lines. Teams arrived at identical ATR-based SL/TP methodology independently. Confidence: High.

**C. MIDGE's instrument coverage is the primary bottleneck.** Teams 01, 02, and 04 all independently identify that MIDGE's signals are equity-focused and FTMO trades forex/indices/commodities. The watchlist expansion is a prerequisite, not an optional improvement. Confidence: High.

**D. The 19.9% overall WR is a losing system at FTMO's break-even.** Teams 01 and 02 independently computed the break-even win rate at 23.0% and concluded 19.9% is below it. The math is correct and consistent. Confidence: High.

**E. Academic backing for convergence is solid but conditional.** Team 03's synthesis is the most thoroughly sourced section across all four reports. The Goldstein-Yang (2015) independence finding, the Condorcet Jury theorem application, and the ADTS/Thompson Sampling validation are all well-supported. The key condition — that combined domains must be genuinely independent — is where MIDGE falls short (macro+technical r=0.73). Confidence in the academic backing: High. Confidence that MIDGE currently satisfies the conditions: Medium.

**F. Alpaca is not a proxy for FTMO.** Teams 02 and 04 both explicitly flag this. External search confirmed FTMO uses MT4/MT5 on forex/commodity instruments, not Alpaca's US equity universe. Using Alpaca paper trading to "validate" FTMO viability is a category error. Confidence: High.

---

## Section 6: Surprises

### Surprise 1: The OANDA Program Is Closing in 22 Days

This was not in any team's findings and was found via external search. This is the most material finding of this validation. If Guiding Light is a US-based operator (as the system context suggests), the recommended primary path (FTMO) may be entering a structural disruption at the moment this research is concluding.

### Surprise 2: FTMO US has "No News Trading Restriction" — Which Is Different From Global FTMO

External search found that `ftmo.oanda.com` (US version) explicitly offers "no restrictions on news trading" — unlike global FTMO which prohibits trading within 2 hours of major news events. Teams 02 and 04 both highlighted MIDGE's economic calendar suppression as an "alignment" with FTMO's news restriction. For the US channel, this analysis was backwards: the US version explicitly allows news trading, which would have been an advantage for MIDGE's macro signal cluster. With the US channel closing, this advantage evaporates.

### Surprise 3: MIDGE's Confidence Engine Failing at its Core Job Is a More Serious Problem Than the Teams Convey

Team 02 identifies this clearly (Section 2): "Winners 0.560, Losers 0.565 — confidence engine currently has near-zero predictive power." The teams treat this as a known issue that the Thompson learning will eventually fix. But the problem is that 81 of 83 Thompson distributions are at the uniform 50/50 prior — meaning the learning has not yet begun, despite months of signal collection. The confidence failure is not a "calibration lag" — it's evidence that the feedback loop from outcomes to Thompson updates is not functioning adequately. This is the root cause that the team 02 noted (paper trades show `hit_rate: 0.0` for all 8 signals) but did not flag as requiring investigation before FTMO integration proceeds.

If the confidence engine cannot discriminate winners from losers, building a confidence-scaled position sizer (as Team 04 proposes) does not add edge — it adds noise.

---

## Validator Verdict: What This Means for the Go/Defer/Abandon Decision

**The research establishes a coherent path but cannot yet justify committing challenge fees.** The specific reasons:

1. **FTMO US access is uncertain** as of March 31, 2026. Verify geographic eligibility before any other step.

2. **The $1,000 gate has not been cleared.** The data to clear it (graded high-confidence pattern stacks) does not exist. The free trial is the right next step precisely because it can generate that data without committing capital.

3. **Three prerequisites must be satisfied before any challenge fee is spent:**
   - Watchlist expanded (NQ=F, ES=F, GC=F, CL=F, EURUSD=X) — hours of work
   - Signal translator built — 1-2 days of work
   - Historical FTMO-instrument signals audited from the signal archive — 1-2 hours of analysis

4. **Kalshi was not evaluated** as an alternative first path. It may be a better initial "prove monetary value" vehicle for a US-based operator with macro-domain signal strength.

5. **The academic evidence is strong** but applies to a system whose independence conditions are satisfied. MIDGE's current macro+technical correlation problem partially undermines the convergence confidence math. The independence correction helps but is calibrated on only 5 of 28 sources.

**Recommended decision: Defer the challenge fee. Proceed with the free trial and prerequisites.** The expedition has produced sufficient evidence to justify the free trial and the prerequisite work. It has not produced sufficient evidence to justify committing €155+ per challenge attempt.

---

## Summary of Action Items for the Next Instance

Ordered by dependency:

1. **Verify Guiding Light's US state** — is FTMO accessible at all? Check `ftmo.oanda.com` status post-March 31, 2026. If blocked, pivot to FundedNext or Kalshi.

2. **Expand watchlist** — add NQ=F, ES=F, GC=F, CL=F, EURUSD=X, GBPUSD=X to MIDGE's active watchlist. This is the single prerequisite that unlocks all downstream testing.

3. **Audit existing signal archive for FTMO tickers** — scan `data/midge/signals/YYYY-MM-DD.jsonl` for ES=F and NQ=F convergence alerts. Count them, check confidence and combo composition. This produces the first real FTMO-relevant win rate estimate.

4. **Build `signal_translator.py`** — 100-150 lines. Unblocks all integration testing.

5. **Run historical replay through FTMO engine on FTMO instruments** — connects the sibling's backtester to actual MIDGE signals on actual FTMO-style instruments. First real pass-rate estimate.

6. **Register for free trial** (human action: Guiding Light) — only after items 2-4 are complete so the trial generates useful data.

7. **Investigate Thompson feedback loop failure** — why are 81/83 distributions still at uniform prior despite months of signals? This is not a calibration lag, it is a broken feedback loop. Must be diagnosed before any FTMO integration is worth building.

---

## Sources Consulted

- `research/expedition-ftmo-viability/team-01-prop-firm-economics.md`
- `research/expedition-ftmo-viability/team-02-midge-capability-audit.md`
- `research/expedition-ftmo-viability/team-03-academic-evidence.md`
- `research/expedition-ftmo-viability/team-04-competitor-implementation.md`
- `research/expedition-ftmo-viability/research-brief.md`
- External: [FTMO No Time Limit Blog](https://ftmo.com/en/blog/trade-without-any-time-limit-and-take-as-long-as-you-want-to-pass/) — confirmed, unlimited time verified
- External: [FTMO US / OANDA Program Closure](https://ftmo.com/en/press-release/oanda-prop-trader-to-conclude-as-ftmo-strengthens-its-modern-prop-focus/) — closes March 31, 2026
- External: [FTMO Restricted Countries + US Status](https://thepayoutreport.com/ftmo-restricted-countries-list-verified-guide-ftmo-us-update/) — state-level restrictions confirmed
- External: [FTMO Inactivity Rule](https://ftmo.com/au/faq/step-1-ftmo-challenge/) — 30-day inactivity = account termination
- External: [FTMO Forbidden Practices (EA Policy)](https://ftmo.com/en/forbidden-trading-practices/) — EAs allowed with standard risk management
- External: [FTMO pass rate ~8%, 92% fail](https://tradelikemaster.com/blog/how-to-pass-ftmo-challenge-2026-complete-guide) — confirmed 2026
- External: [MetaAPI Python SDK](https://github.com/metaapi/metaapi-python-sdk) — free tier confirmed, pricing page required for cost
- External: [FTMO OANDA US relaunch](https://www.financemagnates.com/forex/ftmo-returns-to-us-new-domain-signifies-oanda-integration/) — US channel confirmed then confirmed closing
