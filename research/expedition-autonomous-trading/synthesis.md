# Expedition Synthesis: Autonomous Self-Funding Trading Agent
## Date: 2026-03-07
## Vetted by: Orchestrator
## Alignment: Checked against Research Brief
## Sources: 5 research teams + 3 independent validators

---

## Synthesized Recommendation

**MIDGE should deploy to Kalshi macroeconomic event contracts as her first live market, starting with $1,000 seed capital, after completing three pre-deployment validations.**

The recommendation rests on five independent pillars, each verified by multiple teams and stress-tested by validators:

1. **Domain fit** — MIDGE already has 6 macro-domain sources (FRED, EIA energy, economic calendar, COT positioning, VIX term structure, Granger causality) that directly predict the events Kalshi lists as contracts (FOMC rate decisions, CPI outcomes, NFP numbers).

2. **Legal clarity** — CFTC-regulated, USD-settled, no crypto complexity. All MIDGE signal sources are public data. No PDT rule applies.

3. **Execution simplicity** — Python SDK exists (`kalshi-python`), RSA key-pair authentication supports daemon operation, REST API is documented.

4. **Stealth alignment** — Prediction markets welcome AI participants. MIDGE's low-frequency, multi-domain approach falls outside every known surveillance pattern. No special stealth measures needed at this scale.

5. **Self-funding math** — At $5,000-10,000 capital with validated edge, MIDGE can cover her own compute costs ($100-500/month). Kelly criterion scales position sizes automatically as the account grows.

**Critical caveat the teams collectively understated:** MIDGE's proven edge (z=4.74, p<0.0001) was measured on equity price movements, not binary event contract outcomes. Whether the edge transfers to prediction markets is the single most important unknown. This must be validated before capital is deployed.

---

## Pre-Deployment Validations (Must Complete Before Capital)

Three validators independently converged on these as non-negotiable:

### 1. Backtest MIDGE signals against historical Kalshi contracts
Run MIDGE's historical convergence alerts against Kalshi's historical contract prices. Measure whether macro/energy convergence alerts would have predicted event outcomes. This is the validation of the foundational assumption all five teams share. Without it, the financial model is speculation.

### 2. Verify the Kalshi Python SDK
Install `kalshi-python`, authenticate against the demo environment, confirm it covers the current API version and supports RSA key-pair auth. Three teams recommended it; none verified it. An outdated SDK changes the timeline from "1-2 sessions" to potentially weeks.

### 3. Prototype the MarketSelector
Take MIDGE's top 20 historical convergence alerts and manually find matching Kalshi contracts. If fewer than 30% of alerts have a matching open contract at any given time, the Kalshi-first path needs rethinking. This scoping exercise validates whether sufficient matching markets exist.

**Additionally, before live capital:**

4. **Define a stop-loss threshold** — e.g., halt all trading if account drops below 60% of initial seed. No team designed this; Validator 3 flagged it as a capital protection gap.

5. **Review Kalshi Terms of Service** for restrictions on algorithmic trading accounts. No team checked whether Kalshi flags, restricts, or requires disclosure from automated accounts.

6. **Build a public confirmation timestamp** into every trade — log the source URL and publication timestamp of every contributing signal. This is the compliance asset that makes MIDGE's legal defense concrete, not theoretical.

---

## High Confidence (teams converged + validators confirmed)

### Kalshi is the correct first live market
All 5 teams and 3 validators independently arrive here. The CFTC regulation, USD settlement, Python SDK, and domain alignment with MIDGE's existing macro/energy/government sources are all confirmed. The Federal Reserve published a study (Fed Working Paper 2026-010) finding Kalshi's CPI predictions statistically outperform Bloomberg consensus — using the same data MIDGE already ingests.

**Validator qualification (important):** Validator 2 correctly notes that the "all teams agree" multiplier is partly structural bias — all teams share the same training corpus and the research brief itself pointed toward Kalshi. The recommendation must stand on its specific merits (which are real), not on the unanimity alone. Additionally, Kalshi is primarily a sports platform by revenue (89% in 2025). The macro domain MIDGE would trade represents roughly 10% of platform activity. This doesn't invalidate the recommendation but means liquidity outside major events may be thin.

### The execution gap is in the output layer, not the signal layer
Team 3's codebase mapping confirmed: MIDGE already has TradeSignal, KellyPositionSizer, PortfolioTracker, ActiveTracker, OutcomeCollector, confidence gating, combo filtering, plain-language formatting, and daemon mode — all production-ready. The missing pieces are integration-layer: BrokerClient, RiskGateway, FillTracker, MarketSelector.

**Validator qualification:** The gap is "surgical" for equities (Alpaca bridge is genuinely ~50 lines of Python). It is "structural" for prediction markets — MarketSelector requires a domain-to-contract ontology that has no analog in the codebase. Both Validators 2 and 3 challenged the "surgical gap" framing specifically for the Kalshi path.

### MIDGE's slow cadence is structural stealth
Teams 2, 3, 4 and Validators 1, 2, 3 all agree: MIDGE's multi-day signal accumulation, low trade frequency (1-3/week), and variable Kelly sizing naturally produce a trading pattern indistinguishable from an informed fundamental investor. SEC MIDAS, FINRA's 175+ algorithms, and ARTEMIS all target HFT patterns, spoofing, and pre-announcement timing — none of which MIDGE produces.

### Public-source hygiene is the legal moat
MIDGE's all-public source corpus is the correct legal defense. Trading on aggregated public information is protected under Mosaic Theory principles. The operational conclusion — trade only on timestamped public sources, maintain an audit trail — is sound and well-reasoned by Teams 2 and 4.

**Validator qualification:** The Mosaic Theory is a real legal concept with SEC recognition, but has never cleanly won at trial as a standalone defense. The protection MIDGE has is primarily practical (her signal profile looks nothing like what surveillance targets), not legal certainty. Don't treat it as a bulletproof shield.

### Cross-domain architecture is a genuine structural moat
78% of hedge funds use alternative data, but nearly all stack just 2-3 sources. Nobody systematically Granger-tests 12+ domains, runs adversarial hypothesis validation, and stacks 5-6 independent signals before acting. MIDGE's independence correction (discounting correlated domain pairs, discovered in Phase 0) is unique. The moat is the architecture, not any single data source.

---

## Battle-Tested Approaches (proven, filtered for alignment)

### 1. Kalshi REST API + Python SDK as first execution venue
CFTC-regulated, USD-settled, Python SDK available, domain overlap with MIDGE's existing macro/energy sources. RSA key-pair authentication supports daemon operation without 30-minute token refresh. Fee formula (`0.07 x contracts x price x (1-price)`, capped at $1.75/100 contracts) favors contracts near certainty. Linear payoff ($1 at resolution).

### 2. Alpaca as equities execution bridge
Commission-free, paper-identical-to-live code (one env var change), Python SDK (`alpaca-py`). The simplest path to live execution. If the equity-to-prediction-market transfer question is answered negatively, Alpaca becomes the primary venue. Cash accounts avoid PDT rule. This is the fallback.

### 3. Personal retail account — no registration required
A retail trader automating their own personal account requires no SEC registration, no algorithm disclosure, and no compliance filings. The SEC/FINRA requirements apply to broker-dealers and investment advisers managing others' money. MIDGE trading Guiding Light's own capital is legally clean.

### 4. Post-catalyst timing for maximum legal protection
Rather than trading before events (which triggers ARTEMIS), time entries to coincide with or immediately follow the last public confirmation. Log the timestamp. This creates an audit trail demonstrating every trade was triggered by a public information event. Gives up some edge (market may already be moving) but provides the strongest possible defense against any inquiry.

---

## Novel Approaches (strong theoretical backing, filtered for feasibility)

### 1. Kalshi as both execution venue AND signal source
Teams 1 and 5 identified this independently but framed it differently. Kalshi prices are crowd-aggregated probability estimates that update in real time. A 15% swing in "Fed raises rates" contract price = macro signal before the FOMC meeting. MIDGE can use prediction market prices as an input signal (new "prediction_market" domain) while also executing trades there. Dual-purpose: signal source + execution venue.

### 2. Shadow mode as a formal deployment stage
Not just paper trading — submit REAL orders at minimum position sizes ($1-10 per trade) against Kalshi's live API. Measures real fill behavior, fee impact, and timing alignment. Calibrates the paper-to-live transition. Team 3 proposed this; Validator 2 endorsed it.

### 3. Coinbase AgentKit for eventual self-funding loop
Stage 2, after Kalshi validates. MIDGE's crypto signals (CoinGecko + CoinCap) drive trades on Base L2 via AgentKit. Profits accrue in USDC. x402 protocol pays for API costs. Safe smart account enforces spending limits. This closes the "turn it on and walk away" loop — but only after Stage 1 proves the edge is real.

---

## Domain Expansion (sequenced per "master one domain first" constraint)

The research brief explicitly requires mastering one domain first. Team 5 proposed adding 3 domains immediately, which violates this constraint. The correct sequencing:

**Phase 1 (NOW):** Master Kalshi macro contracts. Validate win rate, build MarketSelector, establish feedback loop.

**Phase 2 (After Phase 1 validated):** Add free domains one at a time, testing each for independence before the next:
- AIS maritime (AISHub — verify free tier terms first, 1 req/min limit)
- USDA WASDE (free, monthly, orthogonal to all existing domains)
- BDI freight rates (free, 5-30 day lead times)

**Phase 3 (Revenue-funded):** Options flow / dark pool (~$50/mo via Unusual Whales API). The combination of options_flow + Form 4 insider + congressional + technical is the most defensible "smart money alignment" stack.

---

## Disagreements (both positions presented with evidence)

### Equity-first vs. prediction-market-first
**For equities:** MIDGE's proven edge was measured on equity price movements. Alpaca bridge is ~50 lines of code. No MarketSelector needed. Cash account avoids PDT. The signal-to-execution path is direct.

**For Kalshi:** No PDT rule. Lower surveillance. CFTC regulation provides legal clarity. Macro/energy domain signals map to event contracts. Fee structure favors high-confidence alerts. Minimum capital is $100 vs $25,000 for margin equities.

**The tension:** The easy path (equities) is where the edge is proven. The strategically motivated path (Kalshi) requires an unvalidated assumption that the edge transfers. Validators 2 and 3 both flagged this as a decision point Guiding Light should make explicitly.

### Regulatory risk: negligible vs. requiring active compliance
**Team 4:** "The surveillance threat to MIDGE is almost entirely misaligned with what regulators actually watch for."

**Team 2:** "The CFTC and DOJ are actively investigating suspicious prediction market trades. MIDGE must be able to demonstrate that every signal source is public, timestamped, and legally obtained."

**Validator verdict:** Team 2's caution is better supported by current enforcement context. Team 4's structural analysis is correct for MIDGE's signal profile, but deploying to Kalshi during peak regulatory attention on prediction markets requires the compliance documentation to be built before deployment, not after.

---

## Filtered Out (removed and why)

| Finding | Team | Reason Filtered |
|---------|------|----------------|
| Polystrat 300%+ returns | Team 1 | Unverified, 25 days old, marketing numbers from the agent's own creators |
| $313 to $438K Polymarket bot | Teams 1, 2, 3 | Pure price arbitrage (YES+NO < $1). Different category than MIDGE's strategy. Zero relevance to pattern-stacking edge. |
| OKX OnchainOS | Team 1 | 4 days old at time of research. Zero production track record. Marketing stats, not independent measurement. |
| BDI 26-day lead (specific number) | Team 5 | Single newsletter source (McClellan Financial). Academic paper finds opposite causal direction at longer horizons. BDI as a domain signal is sound; the specific 26-day figure is unreliable. |
| ElizaOS / Cloudflare Agents | Team 1 | TypeScript-only. MIDGE is Python. Reference architecture only, no integration path. |
| Polymarket market-making as parallel revenue | Team 2 | US access legally restricted. Requires USDC + Polygon infrastructure. Introduces complexity before primary domain is mastered. Violates "master one domain first" constraint. |
| 4% monthly return assumption | Team 2 | Circular placeholder, not derived from MIDGE's measured edge. The self-funding math is directionally correct but the specific number should be replaced with MIDGE's actual measured EV once Kalshi backtesting is complete. |
| Mosaic Theory as "strong legal shield" | Team 4 | Overstated. Never won at trial as standalone defense. MIDGE's real protection is practical (signal profile doesn't match surveillance targets), not legal certainty. Downgraded from "strong" to "directionally supportive." |

---

## Risks

### Critical (must address before deployment)
1. **Win rate transfer** — MIDGE's edge was measured on equities, not event contracts. The entire financial model depends on this untested extrapolation.
2. **MarketSelector complexity** — Mapping convergence alerts to specific Kalshi contracts is a non-trivial subsystem, not a thin wrapper. Must be scoped before committing.
3. **Platform-level surveillance** — Kalshi's own internal surveillance is unresearched. As a CFTC-regulated exchange, Kalshi must maintain trade surveillance. Accounts have been flagged in past DOJ/CFTC cases (Iran strike, Maduro).

### Important (address during deployment)
4. **Regulatory environment shifting** — Active DOJ/CFTC scrutiny of prediction markets as of March 2026. Public confirmation timestamps and audit trail must be built before deployment.
5. **Congressional trade MNPI gray area** — STOCK Act disclosures have 30-45 day lag. Trading on recently-disclosed congressional data is technically public, but the trade itself occurred when it wasn't. Legally untested for algorithmic synthesis.
6. **Private key security** — For eventual crypto/self-funding loop, the private key must be accessible to the daemon. HSM or secrets manager integration has not been designed.

### Operational (solvable but must be designed)
7. **Token refresh for daemon** — Kalshi tokens expire every 30 minutes. RSA key-pair auth avoids this, but the auth architecture must be explicitly designed for 24/7 operation.
8. **Capital floor** — No stop-loss threshold defined for the deployment experiment. Guiding Light's seed capital needs protection via a circuit breaker.
9. **Self-calibrating withdrawal loop** — The brief requires MIDGE to be self-calibrating. The mechanism for routing profits to compute costs is conceptually simple but architecturally undesigned.

---

## The Roadmap

### Stage 0: Validate (no capital required)
- Backtest MIDGE signals against historical Kalshi contracts
- Verify `kalshi-python` SDK against current API
- Prototype MarketSelector with top 20 alerts
- Define stop-loss threshold and success criteria
- Review Kalshi ToS for algo restrictions
- Build public confirmation timestamp logging

### Stage 1: Shadow ($100 seed)
- Deploy MIDGE with Kalshi API integration at $1-10 per trade
- Measure real fill behavior, fees, timing alignment
- Run for 30-90 trades before scaling
- Success criterion: win rate on event contracts within 5% of backtested rate

### Stage 2: Live Micro ($1,000 seed)
- Scale to meaningful position sizes (half-Kelly on $1,000)
- 10-15 trades/month expected at current alert frequency
- Thompson Sampler updates deterministically on contract resolution
- Success criterion: positive EV after fees over 30+ trades

### Stage 3: Self-Funding ($5,000-10,000)
- At 3-5% monthly return, covers compute costs
- Kelly criterion scales position sizes automatically
- Account compounds without human intervention
- Guiding Light turns MIDGE on and walks away

### Stage 4: Domain Expansion (after Stage 3 validated)
- Add equities via Alpaca (where the proven edge lives)
- Add crypto via Coinbase AgentKit (24/7, self-funding loop)
- Add new signal domains (AIS, WASDE, BDI) one at a time
- Each new domain validated for independence before the next

---

## Decision Points for Guiding Light

1. **Equity-first or Kalshi-first?** The easy path is equities (proven edge, simple bridge). The strategic path is Kalshi (lower capital, lower surveillance, but unvalidated edge transfer). This expedition recommends Kalshi-first with the pre-deployment validations above, but the choice is yours.

2. **Seed capital amount?** $1,000 for Stage 2 minimum. $5,000 for faster path to self-funding. $10,000 for full compute coverage within months. Recommendation: start at $1,000, scale after validation.

3. **When to start?** After completing Stage 0 validations. The Kalshi account you already opened is exactly right. The pre-deployment work (backtesting, SDK verification, MarketSelector prototype) can begin immediately.
