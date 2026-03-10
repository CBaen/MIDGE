# Expedition Synthesis: FTMO Prop Trading Viability
## Date: 2026-03-09
## Vetted by: Orchestrator
## Alignment: Checked against Research Brief + Guiding Light's expanded multi-venue vision

---

## The One-Sentence Answer

**FTMO is the right destination but the wrong next step.** MIDGE's convergence architecture is academically validated and has proven statistical edge — but that edge was measured on US equities, and FTMO doesn't trade US equities. Start earning where the signals already land (Alpaca), expand MIDGE's senses to cover new markets, then use FTMO as one execution venue among many.

---

## High Confidence (teams converged, validators agreed)

### 1. The convergence architecture works
Four independent academic lines confirm multi-domain signal fusion outperforms single-source approaches (Goldstein & Yang 2015, Gu/Kelly/Xiu 2020, Kelly/Malamud/Zhou 2024). Thompson Sampling for adaptive signal weighting has direct academic validation (Cartea/Drissi/Osselin 2023, Oxford). MIDGE's approach is not just plausible — it's the academically recommended architecture for this problem.

### 2. The payoff ratio makes profitability achievable
At 3.34:1 (avg win 11.4%, avg loss 3.4%), break-even win rate is 23%. The best domain combinations already exceed this: events+macro+price at 31.2% (n=32), contracts+events+insider+institutional+macro+price at 29.4% (n=17). These combos are profitable, not marginally but meaningfully.

### 3. FTMO's "no time limit" is genuinely transformative
All teams and validators agree: removing the time constraint converts a probabilistic gamble into a patience exercise. With positive expectancy and no deadline, the question shifts from "will it pass" to "how long will it take." The sibling's backtester proved this: 75% pass rate on 250-day windows vs near-zero on 60-day.

### 4. The instrument mismatch is the primary blocker
97%+ of MIDGE's convergence signals fire on US equities. FTMO trades forex, indices, and commodities. The proven edge (z=4.74) has NOT been validated on FTMO instruments. This is the gap everything else depends on.

### 5. The signal translator is well-scoped (~150 lines)
Teams 2 and 4 independently estimated the same scope. ConvergenceAlert already carries direction, confidence, and domain breakdown. Translation to (signal, stop_loss, take_profit) via ATR is straightforward.

### 6. Confidence engine currently operates on noise
Winners averaged 0.560 confidence, losers 0.565. Statistically indistinguishable. Every filtering strategy proposed by every team relies on confidence thresholds that don't discriminate. 81 of 83 Thompson distributions remain at uniform priors — the Bayesian learning loop has not functionally closed.

---

## Battle-Tested Approaches

### Start with Alpaca paper trading on US equities
- MIDGE's edge is measured on US equities. Alpaca client already exists.
- Paper trading accumulates real outcome data to calibrate Thompson distributions.
- Zero cost, zero risk, immediate signal coverage.
- Closes the feedback loop that makes everything else work.

### Fractional Kelly position sizing under drawdown constraints
- Academic consensus (Team 3): 0.25-0.50 Kelly fraction optimal under prop firm drawdown limits.
- Already aligned with MIDGE's DrawdownMonitor architecture.
- VIX-conditioned circuit breakers have strong empirical support.

### Multi-venue execution portfolio
- Alpaca for US equities (where signals already land)
- FTMO for forex/indices/commodities (after instrument expansion)
- Kalshi for prediction markets (macro/event signals are natural fit)
- Crypto exchanges for 24/7 coverage (CoinGecko/CoinCap already wired)

---

## Novel Approaches

### FTMO free trial as measurement instrument (not trading venue)
Validator 3's insight: "Whether or not MIDGE passes the challenge during the free trial is secondary — the 14 days of live signal-to-outcome data on forex instruments is the asset, and it is free." The free trial is a data collection opportunity disguised as a trading challenge.

### Watchlist expansion as sensing expansion
Adding forex/commodity tickers to MIDGE's watchlist isn't just an FTMO prerequisite — it's an organism-level capability expansion. COT, EIA, Economic Calendar, VIX, FRED, Session Sweep, TA, and fractal resonance already produce signals relevant to these instruments. The sources exist. The tickers don't. Configuration change, not code.

---

## Disagreements

### EV per challenge attempt
- **Team 1 says positive** (assumes 35%+ pass rate from combo filtering)
- **Team 2 says negative** (overall 19.9% WR is below 23% break-even)
- **Resolution:** Both are correct for different scenarios. The positive EV requires isolating 30%+ WR combos at execution time — a capability MIDGE does not currently have because the confidence engine doesn't discriminate. The combo filter is the bridge between "losing system" and "winning system."

### How urgent is FTMO specifically?
- **Validator 1:** FTMO US access via OANDA closes March 31, 2026 (22 days). May be moot for US users.
- **Validators 2+3:** FTMO is one venue. The expanded vision (Guiding Light: "ANYTHING that Midge can make money off of") makes venue-specific urgency less relevant.
- **Resolution:** Verify US access status. If closing, deprioritize FTMO-specific work. The sensing expansion benefits all venues.

---

## Filtered Out

| Finding | Team | Why Filtered |
|---------|------|-------------|
| $22 challenge fee | Sibling handoff | Actually €155 (~$165). 7.5x higher. Propagated uncorrected through all teams. |
| 43% algo pass rate | Team 1 | Single source (atmosfunded.com), marketing claim from interested party. No independent verification. |
| "Springer 2025 ADTS" paper | Team 3 | Unverifiable citation. Cannot be treated as load-bearing evidence. |
| ComSIA 2026 (135.49% return) | Team 3 | Extraordinary claim, unverifiable. |

---

## Risks

### Structural risks
- **Confidence engine failure:** The most dangerous finding. If Thompson distributions never calibrate, MIDGE cannot distinguish her best signals from her worst. Every venue, not just FTMO, depends on this being fixed.
- **Prop firm business model misalignment:** FTMO profits from the 90% who fail. Consistently profitable algos are adversarial to this model. "Exploitative practices" termination clause has been used against profitable traders (confirmed by forums, 2024-2025). Treat funded accounts as finite resources, not permanent income.
- **replay_results.json is empty:** The core quantitative case (19.9% WR, 3.34:1 payoff) cannot be reproduced from current data files. Either the data lived elsewhere or was lost. This needs investigation.

### Operational risks
- **30-day inactivity rule:** FTMO terminates accounts with no trades in 30 days. If FTMO-relevant signal frequency is too low, accounts die before profit target.
- **MT4/MT5 execution bridge:** FTMO requires MetaTrader execution. MetaAPI wraps this in REST/WebSocket but adds infrastructure complexity and latency.
- **Tax treatment:** Prop firm payouts are performance fees / contractor income, not capital gains. Material impact on after-tax returns. Not investigated.
- **FTMO US access uncertainty:** OANDA Prop Trader program ending March 31, 2026. Guiding Light's state of residence determines eligibility.

---

## Synthesized Recommendation

### The path forward is expansion, not integration

Guiding Light's vision — "ANYTHING that Midge can make money off of" — reframes the FTMO question. FTMO is one revenue stream in a portfolio of execution venues. The highest-ROI work is expanding MIDGE's sensing capabilities so she covers more markets, which creates more convergence possibilities across ALL venues simultaneously.

### Ordered action list

**Phase A: Zero-cost, zero-code (this week)**
1. Add FTMO instruments to watchlist: `EURUSD=X`, `GC=F`, `CL=F`, `NQ=F`, `ES=F`, `GBPUSD=X`, `USDJPY=X`
2. Verify FTMO US access status for Guiding Light's state
3. Sign up for FTMO free trial (14 days of measurement data at no cost)

**Phase B: Fix the learning loop (critical for ALL venues)**
4. Investigate why 81/83 Thompson distributions are at uniform priors
5. Fix the Thompson feedback loop so MIDGE actually learns from outcomes
6. Build the combo filter — select signals only from historically profitable domain combinations

**Phase C: Start earning where signals already land**
7. Get Alpaca API keys from Guiding Light
8. Run Alpaca paper trading on US equities to accumulate calibration data
9. After 30 days of paper trading with positive results → Alpaca live trading (small position sizes)

**Phase D: Expand MIDGE's senses (new domains)**
10. Forex-native sources: central bank speeches, bond yield curves, DXY, cross-currency correlation
11. Commodity sources: USDA agriculture, BDI logistics, metals inventory
12. Crypto expansion: order book depth, on-chain metrics, funding rates
13. Prediction markets: Kalshi as both signal source AND execution venue

**Phase E: FTMO execution (after validation)**
14. Build signal_translator.py (~150 lines)
15. Build MetaAPI bridge for MT5 execution
16. Run FTMO challenge with validated, calibrated signals on forex/commodity instruments

---

## Bottom Line

The architecture is right. The math works for the best combos. The academic evidence is strong. But the plumbing needs fixing before any money flows — the learning loop must close, the confidence engine must learn to discriminate, and MIDGE must expand her senses to cover the markets she wants to trade. FTMO is one destination in a much larger journey. The fastest path to MIDGE's first dollar is through Alpaca, where her signals already land.
