# Validator Report: FTMO Viability Expedition
**Date:** 2026-03-09
**Validator:** Claude Sonnet 4.6
**Protocol:** Divergence-First — challenges before agreements

---

## Divergence-First Protocol

*Order: (1) Evidence Challenges → (2) Contradictions → (3) Alignment Drift → (4) Missing Angles → (5) Agreements → (6) Surprises*

---

## 1. Evidence Challenges

### 1A. The 43% Algo Pass Rate Is a Single-Source Claim With No Methodology

Team 1 cites a "Jan–Sep 2023, ROI-based strategies" analysis from atmosfunded.com showing 43% pass rate for algorithmic traders. Team 1 correctly flags this as "directionally interesting, not statistically reliable." But the flag is buried — the number gets used in the EV calculation at 35% pass probability without adequate discounting.

**The problem:** A single study from one prop firm website about ROI-based strategies in a 9-month period during a specific market regime (2023 was a bull year) is not a reliable prior for MIDGE's strategy in 2026. The methodology (sample size, how "algorithmic" was defined, survivorship bias) is completely undisclosed. This number should not be used to compute EV without a much wider confidence interval around it.

**What the math actually looks like:** Team 1 computes the break-even pass probability at 17.1%. The EV calculation is only positive if the actual pass probability exceeds this threshold. Team 1 presents a midpoint estimate of 35% without justification beyond "between the 8–10% manual baseline and the 43–75% algo ceiling." The ceiling of 75% comes from the sibling's backtest on a Bollinger Band mean reversion strategy — not MIDGE. Conflating the sibling's 75% with MIDGE's expected performance is the core error. If the honest range for MIDGE specifically is 10–30%, the EV per attempt is -$20 to +$127, and may well be negative.

**Verdict:** The EV per challenge attempt calculation is presented with false precision. The uncertainty range should be presented as a band, not a point estimate.

---

### 1B. The Break-Even Win Rate Has Two Incompatible Formulations

Team 1 computes break-even WR using 1% risk and 3.34:1 payoff as: WR = 0.01 / (0.01 + 0.0334) = **23.0%**.

Team 2 computes break-even WR using 3.34:1 payoff ratio defined as avg win 11.4% vs avg loss 3.4%: WR = 0.034 / (0.114 + 0.034) = **23.0%**.

The numbers match, but the underlying logic differs. Team 1 frames position sizing as 1% account risk with 3.34x payoff. Team 2 frames it as literal return percentages from the historical replay. These two framings are not the same system:

- The historical replay measured stock returns (percentage price moves). A 11.4% avg win means the underlying stock moved 11.4%.
- The FTMO constraint is on account equity, not underlying price. A 1% account risk with a 3.34:1 payoff adds 3.34% to the account — not 11.4%.
- The only way these are equivalent is if MIDGE always sizes to achieve exactly 1% account risk, which assumes perfectly calibrated ATR-based stops.

This distinction matters for the drawdown modeling. Team 2's Monte Carlo uses percent-of-account numbers as if they directly map from the replay's price returns. They do not unless position sizing is explicitly calibrated to target 1% risk per trade. That calibration is currently absent (the paper account is sized to $50K, FTMO account is $10K — a 5x mismatch Team 2 correctly flags).

---

### 1C. Academic Citations Verified — Three Pass, One Needs Caution

Citation verification via web search:

| Citation | Status | Notes |
|---|---|---|
| Gu, Kelly & Xiu (2020), *Review of Financial Studies* | **Verified real** | Confirmed at academic.oup.com/rfs. Findings accurately described. |
| Goldstein & Yang (2015), *Journal of Finance* | **Verified real** | Confirmed at onlinelibrary.wiley.com. Core claim on information diversity accurately described. |
| Kelly, Malamud & Zhou (2024), *Journal of Finance* | **Verified real** | Confirmed at onlinelibrary.wiley.com/doi/10.1111/jofi.13298. Findings accurately described. |
| Cartea, Drissi & Osselin (2023), SSRN 4484004 | **Verified real** | Confirmed at papers.ssrn.com. The MTGP-LR description is accurate. |
| Lobão (2024), *International Studies of Economics* | **Verified real** | Confirmed at onlinelibrary.wiley.com/doi/10.1002/ise3.62. Note: the paper is about the SSE Composite Index, not individual Chinese stocks as Team 3's text implies. This is a meaningful distinction — template matching on a single index is methodologically different from applying templates cross-symbol. |
| ADTS/CADTS 2025 Springer, "20% higher Sharpe than classical models" | **Unverified** | The Springer 2025 citation is described without author names or DOI. Cannot independently confirm. The claim of "20% higher Sharpe" is specific enough to be checkable but no paper identifiers were provided to do so. This is a weak citation. |
| ComSIA 2026 hybrid AI system, "135.49% total return" | **Plausible but novel** | Team 3 cites an arxiv paper (2601.19504v1) published in 2026. These results (135.49% return, Sharpe 1.68 over 24 months) are extraordinary. While the paper may be real, extraordinary results from recent conference preprints have higher-than-usual false discovery rates. Should not be treated as confirmatory evidence. |

**Overall assessment of Team 3 academic citations:** The foundational papers (Gu/Kelly/Xiu, Goldstein/Yang, Kelly/Malamud/Zhou, Cartea et al.) are real and accurately described. The near-term conference papers (ComSIA 2026, Springer 2025 unnamed ADTS paper) are less reliable. The core thesis — that multi-domain convergence has academic backing — holds on the foundational sources alone. The newer citations are useful color but should not carry significant weight in decision-making.

---

### 1D. The Confidence-Doesn't-Discriminate Finding Is Buried But Lethal

Team 2 identifies the most important single finding in this entire expedition: **winners average 0.560 confidence, losers average 0.565 confidence.** This means the confidence engine — the primary filter mechanism for every FTMO strategy proposed — currently has near-zero predictive power.

This finding appears in Team 2's section 2 but does not propagate forward into any team's recommendations with sufficient force. Team 4's implementation path still recommends confidence-scaled position sizing (confidence >= 0.80 → 2.0% risk). If confidence scores don't discriminate winners from losers, this scaling system is filtering on noise. Applying it will not improve pass probability — it will introduce inconsistency without providing any edge benefit.

This is not a minor technical gap. It invalidates the core assumption behind every confidence-filtered approach every team proposes. Until Thompson distributions are calibrated by actual live outcomes, confidence thresholds are cosmetic.

**Root cause (from Team 2 / MEMORY.md):** 81 of 83 Thompson distributions are at uniform 50/50 priors. The 230,462 archived signals have not been used to calibrate the distributions. This is a known issue — the fix (run outcome grading against the signal archive) has been identified but not executed.

---

### 1E. The Replay Results File Is Empty

Team 2 documents that `data/midge/replay_results.json` contains `{"alerts": [], "phase": "replay"}`. This means the historical backtest data that several passages reference — "MIDGE's historical convergence alerts" — does not currently exist in the system in a usable form. Team 4's entire Phase 1 recommendation ("Run MIDGE's historical convergence alerts through the combined system") requires data that is currently absent or corrupted.

Every validation claim based on the Feb 2026 replay comes from `research/phase0-measurements.md` (a research report), not from live queryable data. The replay harness exists as code but the results file is empty. Before any historical validation can be run, the replay must be re-executed and results captured.

---

## 2. Contradictions

### 2A. EV Calculation vs. Current Capability State

Team 1 presents a positive EV per challenge attempt (+$173 at 35% pass probability). Team 2 shows the current system is a **losing system at its measured win rate** (EV = -0.45%/trade at 19.9% WR). These are not reconcilable without a key assumption: that MIDGE's win rate in live operation will be materially higher than its measured 19.9%.

Team 1 assumes it will be (by projecting 30%+ WR from best-combo subsets). Team 2 warns that the combo-level filter does not exist in the execution path — the WR on any individual alert is the 19.9% base rate, not the 30%+ rate of the best historical combinations. The real-time combo filter to select only high-WR combos does not currently exist.

**Resolution:** Both teams are right about their own numbers. The contradiction is that Team 1's EV calculation assumes a capability (live combo-level filtering on FTMO instruments) that Team 2 confirms does not exist. The EV is positive only after building the combo filter and verifying it works on FTMO instruments — which has not been done.

---

### 2B. "Proven Edge" vs. "Below Break-Even"

The research brief states: "MIDGE has proven statistical edge (z=4.74, p<0.0001)." Team 2 states: "Negative expectancy at 19.9% WR. Losing system at current observed convergence win rate."

These are not contradictory if properly understood — but the framing could mislead. The z=4.74 statistical test proves that 19.9% WR is significantly above the 9% random baseline. It does NOT prove the system is profitable for FTMO. A system can be statistically better than random and still have negative expected value for prop firm trading. The break-even for FTMO at 3.34:1 payoff is 23.0% — above MIDGE's measured 19.9%.

**Risk of the framing:** "Proven edge" in the brief and memory creates a psychological anchor that MIDGE is already profitable. Teams should have pushed back harder on this. The edge is proven relative to random. It is NOT proven relative to the FTMO break-even threshold.

---

### 2C. Signal Frequency Contradiction

Team 1: "MIDGE fires 288 convergence alerts per month in replay — at 1–2% of signals on FTMO instruments, frequency drops to 0–2/day."

Team 2: "All 247 current convergence alerts over 2 days are on a single ticker: TUSK."

The Feb 2026 replay produced 288 signals/month distributed across many tickers. The current live run produced 247 signals all targeting one ticker. These are radically different distributions. Team 2 correctly flags this but neither team explains why TUSK is dominating current signals or whether this reflects a real anomaly (TUSK was having something happen) or a system malfunction (signal deduplication not working). If the signal buffer is stuck on TUSK, effective signal frequency for all other tickers is near zero, and the FTMO coverage question becomes moot.

---

## 3. Alignment Drift

### 3A. The Brief Asked About MIDGE as Multi-Venue Trader — Teams Answered FTMO-Only

The assignment instructions explicitly note: "The user has since EXPANDED the vision: MIDGE should be a personal trader across ALL markets — stocks, crypto, futures, forex, prediction markets. FTMO is one revenue stream, not the only one."

None of the four teams address multi-venue parallel execution meaningfully. Team 1 mentions FundedNext and TopStep as alternatives to FTMO. Team 4 mentions Alpaca and briefly touches on Kalshi. But no team addresses the expanded vision of MIDGE simultaneously running:
- Alpaca for US equities (17%+ of signals are already equity-appropriate)
- Kalshi for prediction markets (SDK installed, unverified)
- Crypto exchanges via CoinGecko/CoinCap signals (already wired)
- FTMO for forex/indices/commodities

The research brief asks about "autonomous income" — that framing is broader than just FTMO. The multi-venue synthesis is the most direct path to positive expectancy because different venues accept different instrument classes, and MIDGE's signal mix (97% US equities) actually maps well to Alpaca, not FTMO.

**This is the largest alignment gap in the expedition.** All four teams were given FTMO as the primary frame and largely stayed inside it, even when the evidence pointed toward Alpaca being a better first deployment target for MIDGE's current signal mix.

---

### 3B. The $1,000 Gate Is Mentioned But Not Enforced

The research brief states: "$1,000 gate: deploy capital only when MIDGE demonstrates pattern stacks with 80%+ historical accuracy." No team assesses whether this gate is clearable with current data. Team 2 notes: "Current high-confidence pattern stacks (>= 0.7) are ungraded. This gate is not yet clearable — the data to clear it does not exist."

But team recommendations still proceed toward FTMO challenge attempts (Phase 3 = free trial, Phase 4 = first real challenge). The $1,000 gate is not about challenge fee cost — it's about deployment confidence. The sequence teams recommend violates the gate: they recommend attempting the challenge before demonstrating 80%+ accuracy on pattern stacks.

---

### 3C. Team 3 Answers "Is Convergence Architecture Valid?" Not "Can MIDGE Pass FTMO?"

Team 3's scope drift is understandable — they were asked about academic evidence for multi-domain convergence, and they found it. But their synthesis is framed as validating MIDGE's architecture rather than addressing FTMO viability specifically. The practical question is: does academic evidence suggest MIDGE's *current calibration state* (81/83 Thompson distributions at uniform prior, 19.9% WR) can be improved to the 23%+ threshold through the mechanisms proposed?

Team 3 validates that the mechanisms (Thompson sampling, convergence, pattern templates) work *in principle*. It does not address whether MIDGE's specific current state of calibration is sufficient.

---

## 4. Missing Angles

### 4A. Kalshi (Prediction Markets) Is Completely Unaddressed

The SDK is installed. Kalshi is a regulated prediction market where MIDGE's signals could have natural alignment — congressional trading data, legislative pipeline, economic calendar events all map to Kalshi contracts. None of four teams investigated whether Kalshi's contract structure is compatible with MIDGE's signal types or what the liquidity/spread economics look like for an automated system.

This is the highest-upside missing angle given MIDGE's existing data sources.

---

### 4B. What Happens When Confidence Is Fixed

Team 2 identifies that Thompson distributions need calibration via live outcomes. No team estimates how long this calibration process takes or what the WR trajectory looks like during the calibration period. If MIDGE needs 500+ outcome grades to meaningfully calibrate 83 distributions, and it currently generates ~10 grades per day, that's 50+ days of outcome collection before the confidence engine becomes useful. This period constitutes a known deployment window during which MIDGE will perform at or below its current 19.9% rate.

No team recommends a plan for managing this calibration gap — a short-term approach (use combo-level filter with hardcoded WR thresholds based on historical replay data, bypass confidence filtering during calibration period) would bridge this gap.

---

### 4C. The Logging Bug in Paper Trades Is Ignored

Team 2 identifies that `paper_trades.jsonl` has 1,055 records but only 7 unique signal IDs — the same signal is written repeatedly. This is a data integrity issue that means MIDGE's live paper trading history is corrupted. No team recommends fixing this before using the paper trading data for any validation purpose. If the replay harness and historical validation are attempted with corrupted paper trading data, the results will be unreliable.

---

### 4D. MT4/MT5 Bridge Is Mentioned but Uncosted

Team 4 recommends MetaAPI as the Python-to-MT5 bridge. This introduces a new API dependency with its own pricing (free tier: 100K requests/month), latency characteristics, and failure modes. No team investigates whether MetaAPI's free tier is sufficient for MIDGE's trading cadence, or what happens if MetaAPI is down during a trading window. For a system meant to operate autonomously, this single point of failure between MIDGE and FTMO execution deserves more than a single-paragraph mention.

---

### 4E. The "Best Day Rule" Interaction With Convergence Clustering Is Undeveloped

Team 1 and Team 4 both mention the Best Day Rule (most profitable day cannot exceed 50% of total profits under FTMO 1-Step) and both flag it as a risk for MIDGE. But neither team explores the actual probability of violation given MIDGE's clustering behavior or proposes a mitigation beyond "prefer 2-Step." If MIDGE fires 5 high-confidence convergence alerts in one session and all win, that day likely exceeds 50% of any slow-accumulation profit target. A simple alert-spreading mechanism (max N trades per day, spread execution across days for multi-alert sessions) was not designed.

---

## 5. Agreements (Where Teams Converge)

### 5A. Instrument Coverage Is the Critical Bottleneck

All four teams identify that MIDGE's signal mix (~97% US equities) does not naturally map to FTMO's instrument universe (forex, indices, commodities). Team 2 finds 9 of 32 sources are directly FTMO-applicable. Team 4 confirms the Alpaca/FTMO mismatch. Team 1 identifies it as Gap 2. This is the single most important near-term structural issue.

**Validator agrees:** Expanding the watchlist to NQ=F, ES=F, GC=F, CL=F, and major forex pairs (EURUSD=X via yfinance) is correctly identified as the first step. This is low-effort and unlocks TA indicators, session sweeps, and pattern archaeology on FTMO instruments immediately.

### 5B. The Execution Layer Does Not Exist and Must Be Built

Teams 2 and 4 agree precisely on what must be built: `signal_translator.py` (~150 lines) as the critical path item. Team 4 provides a detailed 6-component architecture. Team 2 confirms the three components needed. The convergence on this point is solid.

### 5C. Free Trial Before Real Money

Teams 1, 2, and 4 all recommend the 14-day free FTMO trial before any capital commitment. This is appropriate given the calibration gaps identified. The validator agrees this is the right sequencing.

### 5D. The Architecture Is Correct, the Calibration Is Not

Teams 1, 2, 3, and 4 all separately conclude that MIDGE's fundamental architecture (multi-domain convergence, Thompson weighting, pattern archaeology) is sound. The problem is not design — it is that the system has not yet accumulated enough graded outcomes to calibrate its confidence engine. Academic evidence supports the design. The gap is operational maturity, not conceptual validity.

---

## 6. Surprises

### 6A. Alpaca Is Not a FTMO Proxy

Team 4 surfaces an important finding that should change planning: Alpaca paper trading cannot validate FTMO challenge performance. These are different brokers, different instruments, different price feeds. The Alpaca client already built in MIDGE is a validation tool for US equity signals, not for forex/index trading. This creates a situation where the most readily available validation pathway (Alpaca paper trading) proves nothing about MIDGE's FTMO capability. The team correctly calls this out, but it means the fastest validation path is actually longer than it appeared — requires MetaAPI bridge to MT5 first.

### 6B. MIDGE's News Suppression and FTMO's News Prohibition Are Aligned

This is a genuine positive surprise from Team 2 and confirmed by Team 4: MIDGE already suppresses trading during FOMC/CPI/NFP windows via the Economic Calendar. This is exactly what FTMO requires (no trading within 2 hours of major macroeconomic events). This is a free compliance win — MIDGE already satisfies one of FTMO's most commonly violated rules.

### 6C. The Combo Filter Gap Is More Serious Than Presented

Teams acknowledge that best combos achieve 29–67% WR but the "best combo" numbers have severe sample size problems. The 66.7% WR combo has n=3 (three graded outcomes). By standard statistical inference, a 66.7% WR from n=3 has a 95% Clopper-Pearson interval of approximately [9.4%, 99.2%]. This is statistically indistinguishable from zero edge. The "events+macro+price" combo at 31.2% (n=32) is meaningful data. The small-sample combos are noise, not signal.

This matters because Team 1's EV calculation uses "50% WR (best-case combos)" as a scenario. That scenario has no statistical basis.

---

## Synthesis: Reconciling the Core Tension

**Team 1 says:** EV per challenge attempt is positive at +$173 assuming 35% pass probability.

**Team 2 says:** Current system is a losing system (EV = -0.45%/trade at 19.9% WR).

**Reconciliation:** Both are correct. The EV per challenge attempt is positive *if and only if* MIDGE can reliably execute its 30%+ WR combo subset on FTMO-tradeable instruments. Currently:
- The combo filter does not exist in the execution path
- The 30%+ WR combos are identified post-hoc from replay data, not filtered in real time
- The watchlist does not include FTMO instruments
- The confidence engine does not discriminate winners from losers
- The replay results file is empty

The EV is theoretically positive. It is practically negative until those five conditions are resolved. The correct summary is: **FTMO is viable as a destination, not as the next immediate step.**

---

## Recommendation to Decision-Maker (Guiding Light)

**What the research actually says:**

1. The architecture is right. The academic backing is real. The convergence thesis is validated.

2. The current system is losing money at its measured win rate. The profitable path requires the combo filter, FTMO instrument coverage, and a calibrated confidence engine — none of which currently exist in the execution path.

3. The fastest path to real signal validation is **Alpaca paper trading on US equities first** — not FTMO. MIDGE's signal mix is 97% US equities. Alpaca accepts US equities. The combo filter can be tested on Alpaca immediately. This builds outcome history that calibrates Thompson distributions. Once the confidence engine is calibrated and WR can be verified at 30%+, FTMO becomes viable.

4. The $1,000 gate has not been cleared and cannot be cleared without outcome grading data that does not currently exist. The free trial is appropriate exploration; it is not appropriate to attempt the paid challenge until outcome grading confirms 30%+ WR on the filtered combo subset.

5. Kalshi is the unexamined opportunity. Congressional trades, legislative pipeline, and economic calendar signals map naturally to prediction market contracts. This should be scoped before the MT4/MT5 bridge is built.

**Priority order for next instance:**
1. Fix the paper trade logging bug (data integrity prerequisite)
2. Expand watchlist to FTMO and Alpaca-relevant instruments
3. Run outcome grading to calibrate Thompson distributions
4. Build combo-level real-time filter in execution path
5. Validate WR >= 30% on filtered subset via Alpaca paper trading
6. THEN pursue FTMO free trial

---

*Validation complete. All agreements noted after challenges, per Divergence-First Protocol.*
