# Validation Report: Expedition — Autonomous Trading
## Date: 2026-03-07
## Validator: Independent Review (Divergence-First Protocol)

---

## Overview

Five teams submitted findings covering AI trading infrastructure, market selection, gap analysis, stealth/legal, and cross-domain signal moat. The research is generally well-sourced and directionally sound. However, several critical claims do not hold under scrutiny, one central recommendation rests on a data distortion that is buried in a footnote, and the unanimous Kalshi endorsement obscures a structural mismatch that the teams glossed over. These issues are identified first.

---

## 1. Evidence Challenges

### 1.1 The Polystrat performance claim is marketing-scale, not validated at trading-relevant scale

**Team 1** cites Polystrat as producing "many individual trades with returns exceeding 300%" and calls it "production evidence." **Verification found no independent confirmation of the 4,200 trades or the 300%+ return figure.** The Olas blog post is the primary source. The predecessor agent (Omenstrat) shows 55-65% win rates depending on model configuration — which is very different from 300%+ returns. Polystrat was launched February 10, 2026 — 25 days before this research was written. No independent audit exists. Citing a 25-day-old agent's unverified marketing claims as evidence of "production evidence" for the safe/safety design pattern misstates the evidentiary weight. The *architecture* (Safe smart account, hardcoded wallet functions) may still be sound, but it should not rest on Polystrat's numbers.

### 1.2 The $313 → $438K Polymarket bot is pure latency arbitrage, not pattern-based trading

**Teams 1, 2, and 3** each cite this case. **Team 2 correctly qualifies it** ("This is pure arbitrage infrastructure play, not pattern-based trading") but Team 1 and Team 3 use it as evidence that MIDGE's convergence approach can generate similar returns. They cannot be compared. The bot exploited sub-second price dislocations on Bitcoin markets where YES+NO summed to less than $1. MIDGE's edge is multi-day, multi-domain convergence. Citing a latency arbitrage result as evidence for MIDGE's path is a category error that appears in multiple team findings.

### 1.3 The Kalshi sports volume fact is buried, and it materially weakens the core recommendation

**Team 2** states "sports contracts (75% of Kalshi volume) are not MIDGE's domain." This is understated. **Verified fact: 89% of Kalshi's 2025 fee revenue came from sports. Sports was 90%+ of volume for the final four months of the year.** FOMC and CPI contracts are Kalshi's most active *macro* contracts, but they are a small fraction of total platform activity. Team 2's framing — that Kalshi's macro/energy/legislative contracts represent "~25% of volume" — is optimistic against the actual 10-11% non-sports share. This does not invalidate the Kalshi recommendation, but it means the addressable opportunity set is considerably smaller and potentially less liquid outside NFL/NBA season than the teams collectively imply.

### 1.4 The BDI "26 trading day lead" claim is from a single non-peer-reviewed source

**Team 5** cites McClellan Financial (a newsletter) as evidence that "BDI leads stock market moves by approximately 26 trading days." **Verification found no peer-reviewed confirmation of this specific 26-day figure.** The 2024 academic paper Team 5 cites (Tandfonline) actually finds that NASDAQ 100 is a *leading* indicator for BDI at longer horizons — the causal direction goes the other way for equity prediction. The BDI's 2022 failure (mentioned by Team 5 itself) is documented. The 26-day lead claim is too specific to rely on without better evidence.

### 1.5 OKX OnchainOS had zero production track record at time of research

**Team 1** includes OKX OnchainOS in its "Novel Approaches" section. This product launched March 3, 2026 — four days before research was written. Team 1 does note "Brand new (3 days old at time of research)" but still uses its claimed $300M daily volume and 99.9% uptime as evidence. Those figures are from OKX's own marketing, not from independent measurement of the AI agent layer. This entry should not appear in findings without a much heavier skepticism flag.

### 1.6 The Kalshi authentication token expiry is correctly reported but Team 1's framing is wrong

Team 1 calls the 30-minute token expiration a "tradeoff" of Kalshi as if it is unique friction. **Verification confirmed**: Kalshi has *two* authentication paths — (1) session tokens that expire every 30 minutes, and (2) RSA key-pair authentication (RSA-PSS signed requests) that does not expire. Team 3 correctly identifies "RSA key pair auth" as Kalshi's method. Team 1's framing creates a false friction that misrepresents how a daemon would actually authenticate.

---

## 2. Contradictions

### 2.1 Team 4 says MIDGE's stealth is natural and built-in; Team 3 says zero stealth measures are implemented — both are correct, and they reveal different parts of the problem

**Team 4** argues MIDGE naturally looks like an informed human investor and does not need special stealth measures because surveillance targets HFT patterns. **Team 3** says "zero stealth measures are currently implemented. Order timing, sizing, and frequency patterns all leave fingerprints." These are not in conflict — Team 4 addresses regulatory surveillance, Team 3 addresses broker-level pattern detection and competitive front-running. But the teams do not acknowledge each other. The practical implication: MIDGE is likely safe from SEC/FINRA surveillance (Team 4 is right about that), but a broker noticing mechanical order submission patterns could flag or investigate the account. Both claims are independently true and should be synthesized.

### 2.2 Team 2 and Team 4 describe Polymarket US access differently

Team 2 says Polymarket US access is "invite-only and KYC-gated as of early 2026" with active state enforcement in Nevada and Tennessee. Team 4 says Polymarket "received CFTC Amended Order of Designation in late 2025, enabling U.S. retail access via registered intermediaries." **Both are technically accurate but frame very different risk pictures.** The regulated US return is real, but constrained. The state enforcement risk is real. Neither team synthesizes this: US Polymarket access exists but is legally precarious and practically limited. This contradiction matters because Team 2 recommends Polymarket market-making as a parallel revenue track — which requires either US legal access (limited) or accepting the VPN risk (Team 2 explicitly flags this as a CFTC violation risk).

### 2.3 Teams 1 and 5 both recommend Kalshi, but for different reasons that partially conflict

Team 1 recommends Kalshi because MIDGE's EIA energy and economic_calendar sources "map almost one-to-one" to Kalshi contracts. Team 5 treats Kalshi as a *new input signal domain* (prediction market prices as signals feeding MIDGE's convergence engine). These are different use cases — execution venue vs. data source — and they are never reconciled. If Kalshi is an *execution venue* for MIDGE's macro signals, then prediction market prices are what MIDGE trades against. If Kalshi is a *signal domain*, then MIDGE reads prices to improve its other signals. The teams converge on "Kalshi" without specifying which role it plays, or whether it can/should play both simultaneously.

---

## 3. Alignment Drift

### 3.1 The Brief's primary ask — "how do existing autonomous AI traders work" — is only partially answered

The Research Brief asks for a "clear roadmap showing (1) how existing autonomous AI traders work." Teams 1 and 3 describe infrastructure. Teams 2 and 4 describe markets and legality. But no team provides a concrete description of a working autonomous trader's *decision loop architecture* — how signal-in becomes order-out in a production system with self-calibration. The Polystrat architecture comes closest, but as noted, it is 25 days old with unverified results. The question "how do they actually work in production" is answered mostly with descriptions of infrastructure components rather than a documented loop. This is a meaningful gap against the Brief's expected outcome.

### 3.2 Team 3's "gap is surgical" claim assumes components work correctly in production, which has not been verified

**Team 3** concludes "The execution gap is surgical, not structural. MIDGE is missing only: BrokerClient, RiskGateway, FillTracker, and MarketSelector. Everything else is production-ready." This is optimistic. The Brief asks "where MIDGE stands vs. what's needed." MIDGE's paper trading has a 19.9% win rate — but this was measured in replay against archived signals, with simulated fills. Whether any of MIDGE's 147 systems behave identically under live execution conditions (real fill latency, Thompson updates with real outcomes, active tracking with real price feeds) is unknown. "Production-ready" for paper trading is not the same as "production-ready for execution." The gap may be surgical in code, but it requires a validation stage before it is declared surgical in behavior.

### 3.3 The "self-calibrating" constraint from the Brief is not addressed in relation to live execution

The Research Brief explicitly requires MIDGE to be "self-calibrating." Teams describe Thompson Sampling as MIDGE's existing self-calibration mechanism. But Thompson Sampling currently learns from paper trade outcomes — not live execution outcomes. The transition from paper to live requires the Thompson feedback loop to receive real fills, real prices, and real P&L. None of the teams describe *how* the self-calibration mechanism connects to live execution. This is the single most important constraint in the Brief after "autonomous," and it receives no dedicated treatment.

### 3.4 "Master one domain first" constraint is respected in synthesis but not in expansion recommendations

The Brief constraint is explicit: "master one domain first." Team 5's Phase A recommendation adds three new domains simultaneously (maritime/AIS, USDA, BDI). Phase B adds a fourth (GDELT). These are framed as expansions for the cross-domain moat, not for the "first domain to master." This is an alignment drift from the Brief's constraint. Team 5 is answering the moat question (which it was asked) but the signal expansion advice runs ahead of the "master one domain first" requirement. The Brief's sequencing constraint should gate new domain additions.

---

## 4. Missing Angles

### 4.1 No team addresses the critical unvalidated assumption: MIDGE's equity-measured win rate may not transfer to event contracts

MIDGE's 19.9% win rate (z=4.74, p<0.0001) was measured on *equity price movements* over replay history. All teams recommend deploying MIDGE on *event contracts* (Kalshi FOMC, CPI, NFP) as the first live market. No team tests or models whether equity-derived signals actually predict binary event contract outcomes at comparable accuracy. This is the central empirical unknown of the entire expedition, and it is identified as a gap by Team 2 ("Whether MIDGE's signals translate to event contract edge has not been tested") but not addressed. **The risk: MIDGE's proven edge is in equities; Kalshi is binary event resolution. These are different prediction problems.** A macro signal that correctly predicts directional equity movement may not correctly predict whether CPI beats or misses consensus by a margin sufficient to resolve a binary contract.

### 4.2 The signal-to-Kalshi-contract mapping problem is identified but not solved

Teams 2 and 3 both flag this: no mechanism exists to map a MIDGE convergence alert (ticker-based, directional) to a specific open Kalshi contract. This is described as "1-2 sessions" of work by Team 2 and as a "MarketSelector" component by Team 3. But the difficulty is understated. Kalshi contracts are event-based with specific resolution dates, wordings, and probability ranges. MIDGE's alerts are ticker-based with 3-30 day outcome windows. The mapping requires semantic matching (what event does this convergence predict?), timing alignment (is there an open Kalshi contract with matching resolution date?), and probability translation (MIDGE's confidence to Kalshi contract price). This is not a trivial engineering task.

### 4.3 No team analyzes the compounding effect of MIDGE's known Phase 0 domain correlation problem on Kalshi

Phase 0 (in MIDGE's existing research) found macro+technical at r=0.73. MIDGE has an independence correction that discounts correlated domain pairs. But the Kalshi recommendation rests heavily on the macro domain. If MIDGE's macro convergence signal is already discounted for correlation with technical, the effective domain count for a "macro+energy+technical" Kalshi play is lower than it appears, and the confidence calculation is more conservative. No team models what MIDGE's actual effective domain count looks like for the specific domain combinations most likely to produce Kalshi-tradeable signals.

### 4.4 Private key security for autonomous operation is acknowledged but not solved

Team 3 flags this: "the private key must be accessible to the daemon process. Standard .env file storage is a security risk. Hardware security module (HSM) or secrets manager integration has not been researched." Team 4 does not address this at all in the stealth/security analysis despite it being a critical operational security concern for the self-funding loop. An unsecured private key is an existential risk to the capital in the self-funding loop. The gap is named but no path forward is offered.

---

## 5. Agreements (High-Confidence Zone)

The following findings are supported by multiple independent teams and verified sources:

**5.1 Kalshi is the correct first live market for MIDGE.** All five teams converge on this. The CFTC regulation, USD settlement, Python SDK, and domain alignment with MIDGE's existing macro/energy/government sources are all confirmed. The Federal Reserve study confirming Kalshi's CPI prediction superiority over Bloomberg consensus is verified (Fed Working Paper 2026-010, multiple independent news sources). This convergence is the highest-confidence finding of the expedition.

**5.2 MIDGE's primary legal protection is clean public-source hygiene.** Teams 2 and 4 independently arrive at this. Verified by CFTC enforcement advisory (Feb 2026) and Morrison Foerster analysis: trading on public information is protected, but the CFTC is actively pursuing enforcement against misappropriation of nonpublic information. MIDGE's all-public source corpus is the correct legal moat. This requires documentation as a compliance asset, not just a design assumption.

**5.3 The execution gap is narrower than building from scratch.** Teams 1, 3, and 5 independently confirm MIDGE has TradeSignal, KellyPositionSizer, PortfolioTracker, ActiveTracker, and OutcomeCollector already wired. Codebase verification confirms: `_write_paper_trade()` exists, confidence gates exist, Kelly cap exists. The missing components are real (BrokerClient, RiskGateway, FillTracker, MarketSelector) but they are integrations, not foundational builds.

**5.4 MIDGE's low-frequency, multi-domain approach is structurally invisible to regulatory surveillance.** Teams 2 and 4 independently confirm SEC MIDAS, FINRA's 175+ algorithms, and ARTEMIS all target HFT patterns, pre-announcement timing, and manipulation. MIDGE's signal cadence (multi-day accumulation) and trade frequency (1-3/week) fall outside every known surveillance threshold.

**5.5 The BDI, USDA WASDE, and GDELT are free data sources with legitimate (if not bulletproof) predictive evidence.** Team 5's prioritization of free domains before paid ones is correct sequencing. The GDELT accuracy caveat (55% field accuracy) is correctly flagged, and the recommendation to use a rolling 7-day tone index rather than raw events is sound mitigation.

---

## 6. Surprises

### 6.1 Kalshi is primarily a sports betting platform by revenue — and this changes the liquidity picture for MIDGE

This is the expedition's most counterintuitive finding. The teams consistently frame Kalshi as a macroeconomic prediction market. It is, by revenue, primarily an NFL/NBA platform. 89% of 2025 fee revenue came from sports. The macro contracts (FOMC, CPI, NFP) are real, growing, and institutionally validated — but they represent roughly 10% of the platform. For MIDGE's purposes, this means: the macro domain opportunity is real but smaller than the teams' framing suggests, and liquidity outside of major economic events may be thin. This does not change the recommendation but significantly changes the capital deployment math.

### 6.2 The CFTC is actively investigating prediction market trading for insider trading violations as of 2026 — not a future risk

Team 4's legal analysis frames insider trading risk as low because MIDGE uses public sources. That is correct. But the teams collectively understate the *active enforcement environment* in 2026. The CFTC issued an explicit enforcement advisory on prediction market misconduct in February 2026. The U.S. Attorney for the Southern District of New York explicitly stated on February 5, 2026, that fraud prosecutions relating to prediction market trading are expected. The Iran war bet controversy (CNN, March 7, 2026 — same day as this research) shows the scrutiny is current and politically visible. MIDGE's clean public-source corpus is the correct defense. But deploying to Kalshi now means operating during peak regulatory attention on this exact market. This is not a disqualifier, but it requires the compliance documentation (public confirmation timestamps, audit trail) to be built before deployment, not after.

### 6.3 The domain count in Team 5's table does not match the codebase

Team 5 lists 13 domains (insider, events, macro, technical, sentiment, government, contracts, fundamentals, positioning, crypto, institutional, energy). Codebase verification of `_SOURCE_DOMAIN_MAP` shows **14 distinct domains**: the 13 listed plus `volatility` (mapped from `vix_term_structure`). This is a minor discrepancy but indicates Team 5 did not fully audit the domain map against the actual code. It does not change the substance of the recommendations but means Team 5's "add AIS to go from 12 to 13 domains" framing is slightly off — it would go from 14 to 15.

---

## Summary Assessment

**What holds up strongly:**
- Kalshi as first live market (all teams, verified)
- Public-source legal hygiene as the compliance moat (verified)
- MIDGE's execution gap being integration work, not foundational work (verified in codebase)
- AIS, BDI, USDA WASDE as free domain additions with independent predictive evidence (Team 5)
- MIDGE's structural invisibility to regulatory surveillance (Teams 2 and 4)

**What requires additional validation before acting:**
- Whether MIDGE's equity-domain win rate transfers to event contract prediction (the central empirical unknown)
- The signal-to-Kalshi-contract mapping mechanism (underestimated in complexity)
- Private key security architecture for the self-funding loop
- The actual macro/energy liquidity profile of Kalshi outside of major event windows

**What should be discarded or heavily caveated:**
- Polystrat performance numbers (unverified, 25 days old)
- The $313→$438K comparison to MIDGE's path (different category of strategy)
- OKX OnchainOS as a credible infrastructure option (4 days old at time of research)
- BDI's specific 26-day lead claim (single newsletter source, contradicted by academic direction)
- The "gap is surgical" framing without acknowledging the paper-to-live behavioral validation stage

**Highest-priority unasked question:**
Before any capital is deployed to Kalshi, run MIDGE's historical convergence alerts against Kalshi's historical contract prices and measure the prediction accuracy specifically for event resolution — not equity price direction. That single backtesting exercise would confirm or refute the foundational assumption all five teams share.
