# Expedition Validation Report: Autonomous Trading
## Date: 2026-03-07
## Validator: Independent Cross-Validation

---

## Methodology Note

This report follows divergence-first protocol. Evidence challenges appear before agreements. Every specific challenge references the source claim directly.

---

## 1. Evidence Challenges

### Claim: "$313 to $438K" Polymarket bot

**Teams citing it:** Team 2 (primary), Team 3 (citing "$313 → $414,000")

**Challenge: The two teams cannot agree on the number.**
- Team 2 states: "$313 accumulated ~$437,600 in ~1 month"
- Team 3 states: "$313 → $414,000 arbitrage bot case"
- Same wallet, same event, ~$24,000 discrepancy. Neither team explains the gap or references the same primary source. This is a basic fact-check failure.

**Challenge: The mechanism is mischaracterized in both reports.**
Both teams describe it as "latency arbitrage on 15-minute crypto markets" (Team 2) or "exploiting YES+NO contracts summed to less than $1" (Team 3). The finbold.com and DEV Community sources confirm it was a YES+NO price arbitrage bot — not pattern-based trading. The DEV Community article explicitly states the strategy was "when YES + NO prices briefly sum to less than $1.00, bots simultaneously buy both sides and lock in guaranteed profit." This is pure deterministic arbitrage, not information-edge trading. It has zero relevance to MIDGE's strategy.

**Challenge: The figure may be unrealized.**
The finbold source could not be verified for the body content — the article returned only metadata. Neither team cites a verifiable primary source (e.g., on-chain transaction data, Dune Analytics query) to confirm the $437,600/$414,000 figure represents settled, withdrawn profit rather than open prediction market positions. Polymarket wallets show unrealized contract value — a wallet holding 437,000 YES contracts priced at $1 looks like $437K but resolves only if the prediction resolves YES.

**Verdict on this claim:** The dollar figure is likely real (on-chain data is auditable), but the mechanism is pure price arbitrage with no predictive component. Both teams correctly caveat this ("pure arbitrage, not pattern-based") but then cite it as proof that "the market is real and accessible" — which is not in dispute. The claim does not support the broader argument it is being used to make. Teams should remove it from the "comparable" section entirely.

---

### Claim: "4% monthly return" assumption underpinning self-funding math

**Team citing it:** Team 2, in the capital threshold table and path-to-self-funding calculations.

**Challenge: The 4% figure is not derived from MIDGE's actual edge — it is a circular placeholder.**
Team 2's own analysis shows MIDGE's current convergence win rate is 19.9% with 3.34:1 payoff. The Team 2 EV calculation yields +0.90% per trade at the 29% WR scenario. The path from "+0.90% per trade" to "4% monthly return" requires assumptions about trade frequency that are never stated. At 10 trades/month (the stated assumption), 0.90% per trade = +9% raw gross return per month before fees, slippage, and Kelly sizing. At half-Kelly (~3.9% of account per trade), the actual dollar return per trade is far smaller. The 4% figure appears to be chosen to make the math work, not derived from MIDGE's measured edge.

**Challenge: The 4% assumption is also inconsistent with the research finding that only 0.51% of Polymarket users achieved profits exceeding $1,000 in 2025** (per the market making guide in search results). Sustained 4% monthly returns — ~48% annualized, compounding — would put MIDGE in the top fraction of a percent of prediction market participants. This is possible given the z=4.74 edge, but it requires prediction markets to be MIDGE's actual deployment venue, which hasn't been validated.

**Challenge: The 19.9% win rate was measured on equity price movements, not event contract binary outcomes.** Team 2 acknowledges this as a gap (point 2 in Gaps section). The entire capital requirement table assumes Kalshi event contracts perform like equity positions, but the signal-to-contract mapping has not been tested. The 4% monthly return assumption therefore rests on an untested extrapolation of equity performance to a structurally different instrument.

**Verdict:** The self-funding threshold math ($5K minimum viable, $10K full self-funding) is plausible directionally but not defensible as stated. Revise to explicitly label it as a hypothetical scenario with unvalidated assumptions, not a projection.

---

### Claim: "The gap is surgical, not structural"

**Team citing it:** Team 3, synthesis section.

**What the claim rests on:** Team 3 identifies four specific missing components — BrokerClient, RiskGateway, FillTracker, MarketSelector — and maps every other requirement to existing MIDGE files.

**Verification against actual codebase:**

Team 3 cites these existing files as covering the execution gap:
- `mae_core/market/signal.py` — TradeSignal dataclass
- `mae_core/bootstrap/market_hooks.py` — `_write_paper_trade()`
- `mae_core/market/intelligence/kelly_position_sizer.py` — KellyPositionSizer
- `mae_core/market/intelligence/portfolio_tracker.py` — PortfolioTracker
- `mae_core/market/archaeology/active_tracker.py` — ActiveTracker

These files are confirmed to exist (git status and project CLAUDE.md confirm these paths and components). The HANDOFF.md in project memory confirms `_write_paper_trade()` is wired at confidence >= 0.45.

**Challenge: MarketSelector is not "surgical" — it is structurally novel.**
Team 3 correctly identifies MarketSelector as "the novel piece with no existing analog in MIDGE." But then minimizes it. MarketSelector must: (1) listen for a convergence alert with a ticker + domain combo, (2) query Kalshi's live market catalog, (3) identify a matching event contract, (4) assess whether the contract's resolution timing aligns with MIDGE's signal window, (5) return a tradeable contract ID. This is not a thin wrapper — it requires a domain-to-contract ontology that does not exist anywhere in MIDGE. Team 3's own Gap #2 states: "There is no current mechanism in MIDGE to map a convergence alert to a specific Polymarket/Kalshi market." This directly contradicts the "surgical gap" claim in the synthesis.

**Challenge: The RiskGateway is also non-trivial.**
Team 3 lists 5 enforcement rules for RiskGateway including a "correlation limit (no two positions in the same domain cluster)." MIDGE's existing domain correlation data (from CorrelationTracker and GrangerAnalyzer) is not in a form that can be queried synchronously at trade execution time. This requires integration work that Team 3 does not scope.

**Verdict:** The "surgical gap" framing is defensible for the broker connection itself (BrokerClient + FillTracker). It is not defensible for MarketSelector or full RiskGateway. The gap is surgical for equities execution and structural for prediction market execution. The report should say this explicitly.

---

### Claim: Mosaic Theory "affirmed by both the Supreme Court and the SEC"

**Team citing it:** Team 4, Battle-Tested Approach #1, stated as established protection.

**Challenge: The Supreme Court has never directly affirmed the Mosaic Theory as an insider trading defense.**
Team 4 cites "Supreme Court" endorsement. The actual basis for Mosaic Theory in Supreme Court precedent is *Dirks v. SEC* (1983), which held that analysts can trade on information derived from piecing together public data. *Dirks* recognized the concept implicitly but did not establish "Mosaic Theory" as a named legal defense. Multiple independent legal sources confirm this ambiguity.

**Challenge: The theory has a losing record as an actual trial defense.**
The Tucker Ellis legal analysis (confirmed via WebFetch) finds that defendants who invoked Mosaic Theory in court — Rajaratnam, Gupta, Whitman — were all convicted. The theory's failure in practice is not because MIDGE has MNPI (it doesn't), but because courts scrutinize the entire fact pattern. Team 4 correctly notes Rajaratnam "had actual MNPI alongside the mosaic" — but this is exactly the risk MIDGE faces if any single data vendor in its 30-source stack is later found to have been selling MNPI-adjacent data. One contaminated source poisons the entire mosaic defense.

**Challenge: "Affirmed by the SEC" overstates the legal protection.**
The SEC's Regulation FD guidance acknowledges the theory exists, but Reg FD governs *company disclosure*, not *trader defenses*. The SEC can simultaneously acknowledge the theory in Reg FD guidance and prosecute traders who assert it as a defense. Team 4 conflates these two distinct uses.

**What Team 4 gets right:** The operational conclusion — trade only on timestamped public sources, maintain an audit trail — is correct and well-reasoned. The legal grounding for that recommendation is sound. The issue is that Team 4 presents the Mosaic Theory as a stronger legal shield than it actually is in practice, which could cause Guiding Light to underestimate enforcement risk.

**Verdict:** Mosaic Theory is a real concept with real SEC recognition, but has never cleanly won at trial as a standalone defense. The protection MIDGE has is primarily practical (MIDGE's signal profile looks nothing like what surveillance systems target), not legal. Team 4's synthesis gets this right but the Battle-Tested section overstates the legal defense.

---

### Claim: AISHub free tier is suitable for daemon mode

**Team citing it:** Team 5, as the top-priority free domain addition.

**Challenge: AISHub's documented rate limit is 1 request per minute maximum.**
Direct fetch of AISHub's API documentation confirms: "Don't access the webservice more frequently than once per minute! The web service will return nothing if executed more frequently." MIDGE's daemon fires sensing hooks every 50 steps. Depending on pace, this could mean multiple API calls per minute — which would silently return nothing rather than returning an error, making debugging difficult.

**Challenge: "Free" membership status for continuous daemon operation is unverified.**
AISHub's API page states only "AISHub members are allowed to access AISHub webservice" without differentiating free vs. paid tiers or addressing daemon-mode continuous operation. The terms of service for automated continuous access are not documented on the API page. It is unknown whether running a 24/7 daemon against the free tier violates AISHub's terms and risks account termination.

**Challenge: The 1-request/minute limit makes real-time vessel tracking impractical for MIDGE's daemon.**
MIDGE's ExcavationDaemon runs every 5,000 steps; the MarketSensingHook runs every 50 steps. A 1-request/minute AISHub polling cycle would give 1,440 data points per day. For static route-level aggregation (e.g., "tanker count near Suez Canal this week") this may be sufficient. For "real-time" vessel tracking as described in Team 5, it is insufficient. The lead time claim (20-30 days for crude signals) means freshness is less critical — but this needs to be stated explicitly in the implementation design rather than assumed away.

**What holds up:** The strategic argument for AIS as a domain is sound. The 26-trading-day BDI lead time and the OECD/IMF academic backing are real. The specific recommendation of AISHub as a "free, zero-cost" daemon-compatible solution requires qualification.

**Verdict:** AISHub is a viable starting point for backtesting and low-frequency polling (daily/hourly), not real-time daemon integration. The "free tier suitable for daemon mode" claim is unverified and likely incorrect as stated.

---

## 2. Contradictions Between Teams

### Polymarket dollar figure
Team 2 says $437,600. Team 3 says $414,000. No team cross-checked the other. This is the clearest internal contradiction and the easiest to verify (it is on-chain data). Neither number matters for MIDGE's strategy, but the discrepancy signals that teams did not read each other's work before submitting.

### "Kalshi fees approach zero for high-confidence signals" vs. actual Kalshi fee formula
Team 2 asserts: "Contracts near certainty (price near $0 or $1) have near-zero fees — MIDGE's high-confidence convergence alerts (0.75+ confidence) would target high-certainty contracts, meaning fees approach zero."

Team 3 cites the Kalshi fee formula: `0.07 × contracts × price × (1 − price)`, capped at $1.75 per 100 contracts.

These are consistent — the fee IS near zero near certainty. But MIDGE's convergence confidence score (0.75) is MIDGE's internal confidence, not the contract's implied probability. A contract priced at 50% probability that MIDGE believes is 75% likely would be at the fee maximum (price=0.50, fee peak). Team 2 conflates MIDGE's internal confidence with contract price. A high-confidence MIDGE signal on a mispriced contract is most valuable precisely when the contract price is far from MIDGE's estimate — which is also when fees are highest.

### Stealth: "No stealth measures implemented" vs. "naturally stealthy by design"
Team 3 (Gap #4): "The research brief requires MIDGE to be 'undetectable as algorithmic.' This is achievable (see Stealth section below) but zero stealth measures are currently implemented."
Team 4 (Synthesis): "The surveillance threat to MIDGE is almost entirely misaligned with what regulators actually watch for... MIDGE fits zero of these [surveillance] patterns."

Both are partially correct, but they're answering different questions. Team 4 addresses regulatory surveillance (MIDAS, ARTEMIS, FINRA). Team 3 addresses broker-level pattern detection and competitive front-running. These are distinct threat models. Neither team explicitly scopes the distinction. Team 4's dismissal of stealth concerns is too confident given Team 3's valid concern about broker-level detection.

---

## 3. Alignment Drift

### Drift from Brief: The equity execution gap is understated

The Research Brief states the expected output includes "where MIDGE stands vs. what's needed." MIDGE's z=4.74 edge was measured on equity price movements. All teams converge on Kalshi as the first market. But none of the teams quantify the cost of abandoning the equity market (where the edge was actually measured) for prediction markets (where the edge is assumed to transfer).

The alignment gap: the teams have collectively pivoted MIDGE's first deployment target away from the market where its edge is empirically proven and toward a market where the edge is untested. This is a strategic choice that should be explicitly surfaced to Guiding Light as a decision point, not buried in gap sections.

### Drift from Brief: "self-calibrating" requirement is not addressed

The Research Brief explicitly calls for MIDGE to be "self-calibrating." Team 3 mentions the Thompson Sampler updating on contract resolution. No team addresses whether Kalshi's binary YES/NO resolution event can cleanly update MIDGE's Thompson distributions, which were calibrated on equity price percentage-move outcomes. The outcome window model (3-30 days, graded on % move) does not map to a binary contract resolution. This is an architectural compatibility question that none of the teams answered.

### Drift from Brief: "master one domain first" constraint

The Research Brief says "master one domain first." Team 5 recommends adding 3 new domains immediately (AIS, WASDE, BDI) as Phase A, plus GDELT as Phase B. This contradicts the constraint in the brief. Team 5's moat-expansion recommendations are strategically sound but temporally misaligned — they should be sequenced after the first domain is proven, not recommended concurrently.

---

## 4. Missing Angles

### Nobody verified whether Kalshi actually has markets for MIDGE's signal types right now

Every team recommends Kalshi. No team pulled the current Kalshi market catalog and counted how many open markets align with MIDGE's 12 domains. This is testable in 10 minutes via the Kalshi API. The entire recommendation rests on the assumption that sufficient matching markets exist at any given time. The brief required a "clear roadmap" — a roadmap without confirmed market availability at the destination is incomplete.

### Nobody analyzed MIDGE's signal frequency against prediction market contract availability windows

MIDGE fires convergence alerts at irregular intervals. Kalshi contracts have fixed resolution dates (e.g., March 19, 2026 FOMC). If MIDGE fires a macro convergence alert 25 days before the FOMC date, the contract may already be illiquid or fully priced. If MIDGE fires 2 days before, the contract price has already moved toward the resolution. The optimal entry window is likely 5-15 days before resolution. No team analyzed whether MIDGE's signal timing aligns with this window.

### The "public confirmation timestamp" recommendation has an implementation cost that nobody scoped

Team 4's single most actionable recommendation — add a verifiable public-record timestamp to every trade — requires MIDGE to (a) record the source URL and publication timestamp of every signal that contributed to each convergence alert, and (b) enforce that the trade gate does not fire until all contributing sources are confirmed public. This is a non-trivial audit logging change to the convergence alerter's output schema. No team scoped this implementation cost.

### No team cross-checked the "independent teams converge on Kalshi" conclusion

All 5 teams recommend Kalshi. The brief explicitly asks whether this is genuine independent convergence or shared training bias. The answer: the convergence is partly genuine (CFTC regulation, domain fit, Python SDK) and partly structural bias. All teams were generated from the same base model, trained on the same corpus, given the same research context (MIDGE's 30 sources already overlap with Kalshi's macro domains). The research brief pointed at Kalshi as a candidate in its own framing ("MIDGE's EIA energy source and economic_calendar source already generate signals in exactly the markets Kalshi offers"). When the brief contains the answer, "independent convergence" is not evidence — it is tautology.

This does NOT mean Kalshi is the wrong answer. It means the convergence cannot be used as confidence-amplifying evidence. Kalshi's recommendation must stand on its specific merits, which are real, without the "all teams agree" multiplier.

---

## 5. Agreements (High-Confidence Zone)

The following conclusions were reached independently across multiple teams and are supported by verifiable external evidence:

**Kalshi is the legally cleanest US deployment target for MIDGE's macro signals.** Teams 1, 2, 3, and 4 all arrive here independently with distinct reasoning (infrastructure, capital efficiency, legal clarity, surveillance profile). The CFTC DCM designation is a verifiable fact. The Python SDK is on PyPI. The domain overlap with MIDGE's existing macro sources is real and documented.

**MIDGE's current execution gap is primarily in the output layer, not the signal layer.** Teams 3's appendix mapping existing MIDGE components to execution requirements is well-documented and verifiable against the codebase. The signal generation, position sizing, position tracking, and Thompson feedback loop are genuinely complete. The missing pieces are integration-layer, not architectural.

**MIDGE's natural trading profile (slow, multi-day, multi-domain) is not targeted by existing market surveillance systems.** Team 4's analysis of SEC MIDAS, FINRA surveillance, and ARTEMIS is well-sourced. The specific pattern signatures those systems target (HFT, spoofing, pre-announcement timing) are structurally absent from MIDGE's approach. This is a genuine and defensible finding.

**MIDGE's cross-domain architecture is not replicated by known competitors.** Team 5's survey of alternative data usage among hedge funds, combined with the competitive edge expedition (cited in project memory), consistently finds that 2-3 domain stacking is the industry norm. MIDGE's 12+ domain architecture with independence correction and Bayesian feedback is genuinely differentiated. The moat claim is directionally correct.

**The self-funding loop closes at the $10K account level under conservative assumptions.** Team 2's math is internally consistent given its stated assumptions. The specific number is less important than the structural finding: the payoff ratio (3.34:1) and expected trade frequency make self-funding compute costs plausible at amounts Guiding Light could seed without extraordinary risk.

---

## 6. Surprises

**The "surgical gap" is surgical for equities and structural for prediction markets.** Before reading Team 3's appendix closely, the assumption was that prediction market execution was the easier path. The reverse is true: for US equities via Alpaca, the bridge from `_write_paper_trade()` to a live order is genuinely ~50 lines of Python. For Kalshi, the MarketSelector component — which must dynamically map MIDGE's convergence alerts to specific open contracts — has no analog in the codebase and requires building a domain-to-contract ontology. The easy path is equities. The strategically motivated path (regulatory cleanliness, no PDT rule) is Kalshi. These are in tension.

**The Polymarket bot story is actively misleading for MIDGE's planning.** Both teams cited it as evidence of prediction market viability, but it is actually evidence that a specific deterministic arbitrage opportunity existed (YES+NO < $1). That opportunity is exactly the kind of pricing error that gets closed once identified. Its relevance to MIDGE's pattern-stacking approach is near zero, yet it dominated the "evidence that autonomous trading works" narrative in both reports.

**AISHub's rate limit is more constraining than the maritime signal's strategic case is strong.** The IMF and OECD academic backing for AIS as a trading signal is genuine. But the implementation path — AISHub free tier, 1 request/minute, unverified daemon terms — is fragile. The mismatch between the strategic case (strong) and the implementation path (weak) is the clearest example of a finding that needs more rigorous engineering feasibility assessment before it moves to a task.

---

## Summary Scorecard

| Team | Strongest Finding | Most Significant Gap |
|------|------------------|----------------------|
| Team 1 | Kalshi Stage 1 / AgentKit Stage 2 deployment sequence is the most actionable synthesis | OKX OnchainOS and ElizaOS have zero production track record; their inclusion inflates optionality without grounding |
| Team 2 | Capital threshold math is internally consistent; regulatory analysis of Kalshi is well-sourced | 4% monthly return assumption is a circular placeholder; Polymarket bot citation is misleading |
| Team 3 | Codebase mapping appendix is the most useful deliverable of all 5 reports; deployment pipeline (Paper→Shadow→Live) is well-structured | "Surgical gap" overstates ease of MarketSelector; stealth section references "BJF Trading Group white paper (2026)" with no URL — unverifiable source |
| Team 4 | Mosaic Theory analysis and surveillance threat model are the best-researched sections in the expedition | Overstates Supreme Court endorsement of Mosaic Theory; underestimates broker-level (non-regulatory) detection risk |
| Team 5 | Independence correction as competitive moat is the most original structural insight | AISHub daemon suitability is unverified; "master one domain first" constraint ignored; new domain recommendations are premature |

---

## Recommended Actions Before Moving to Build

1. **Verify Kalshi market availability:** Pull the current Kalshi market catalog via API and count how many open markets map to MIDGE's 12 domains. This is a 30-minute task. Do not begin MarketSelector architecture until this is known.

2. **Resolve the equity-vs-prediction-market first choice explicitly:** MIDGE's proven edge is in equities. The Alpaca execution bridge is genuinely simpler. The PDT rule is the only barrier. Guiding Light should decide whether to (a) deploy to equities with a cash account to avoid PDT, (b) seed $25K to satisfy PDT, or (c) accept the unvalidated transfer of edge to Kalshi. This is a strategic decision that teams left unresolved.

3. **Scope MarketSelector properly:** Before committing to Kalshi as the first live deployment, scope MarketSelector as a standalone research task. A prototype that can take MIDGE's top 20 convergence alert tickers and find matching Kalshi contracts would validate the approach before any live trading infrastructure is built.

4. **Fix the 4% assumption:** Team 2's self-funding calculations should be re-run using MIDGE's actual measured EV per trade, actual alert frequency from the daemon logs, and actual Kalshi fee structure. The result may be more or less optimistic than 4% monthly — but it should be derived, not assumed.

5. **Verify AISHub terms before implementing:** Before building an AIS fetcher, confirm with AISHub that a continuously running daemon on the free tier is permitted, and test the 1-request/minute constraint against MIDGE's step cadence to confirm the polling architecture is viable.
