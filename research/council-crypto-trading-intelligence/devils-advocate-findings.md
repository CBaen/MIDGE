# Devil's Advocate Findings: Crypto Trading Intelligence for MIDGE

**Date:** 2026-03-17
**Role:** Devil's Advocate
**Verdict:** The proposal contains 4 structural failures that will result in provable losses, 2 latent risks that degrade over time, and 1 assumption so fundamental that if it's wrong, the entire architecture is wrong.

---

## Scoring Summary

| Risk Dimension | Score (1–10, 10 = worst) |
|---|---|
| Failure Probability | 8/10 |
| Failure Severity | 6/10 (paper trading limits blast radius — for now) |
| Assumption Fragility | 9/10 |
| Hidden Complexity | 8/10 |
| Overall Risk | 8/10 |
| Reversibility | 7/10 (some signals will need to be deleted, not fixed) |
| Evidence Confidence | 7/10 (strong on structural issues, moderate on market dynamics) |

---

## The One Assumption That Breaks Everything

**The convergence engine assumes domain independence. Crypto demolishes this assumption.**

MIDGE's confidence formula uses domain diversity as a multiplier. Three independent domains converging = high confidence. The mathematical foundation is correct — IF the domains are actually independent.

In equity markets, this independence holds: SEC filings, congressional trades, and EIA energy reports come from different actors with different information, different timelines, and different incentives. They genuinely can converge independently.

In crypto, every domain MIDGE has — Fear & Greed, BTC Dominance, CoinGecko prices, CoinCap prices, Kraken funding rates, DefiLlama TVL, on-chain mempool — is downstream of the same single underlying variable: **BTC price direction**.

Research confirms: altcoins show high "followness" to BTC (ScienceDirect, 2025). The average correlation between major crypto assets remains >0.70 in trending markets. When BTC sells off, funding rates go negative, Fear & Greed collapses, TVL drops, and mempool congestion falls — not because four independent things happened, but because one thing happened and everything else reacted.

**What this produces in MIDGE:** A 3-domain crypto convergence alert that appears to have independent confirmation but is actually the same signal measured four times. The convergence confidence score is inflated by a factor of roughly 3-4x what it should be. MIDGE will be systematically overconfident on crypto.

This is not fixable by tuning thresholds. It requires domain re-categorization: crypto's 7 "domains" should count as at most 2 independent dimensions — price/momentum and derivatives/funding.

---

## Structural Failure 1: The Execution Path Has No Stop-Loss for Crypto

**Code evidence:** `market_hooks_trades.py`, `_submit_to_alpaca()`, lines 469–470:
```python
_tp = None if is_crypto else round(signal.take_profit, 2)
_sl = None if is_crypto else round(signal.stop_loss, 2)
```

Alpaca does not support bracket orders for crypto. MIDGE knows this and explicitly sets both stop-loss AND take-profit to `None` for all crypto positions. The ATR-based SL/TP calculated by the signal translator is computed correctly and then **silently discarded**.

The result: every crypto paper trade is a naked market order with no exit plan. MIDGE will enter but has no automated mechanism to exit. Position management is entirely absent. In a 24/7 market that moves 10-20% in hours, this is not a minor gap — it is the difference between a managed trade and an unmanaged exposure.

**Paper trading makes this invisible now.** When MIDGE moves to live trading, this will cause actual losses because the discipline that "paper trading builds" is being built without stop-losses. MIDGE is learning to trade with no risk management, which means the learned behavior is wrong.

The fix requires either: (a) separate GTC limit orders for SL/TP after entry, submitted as independent orders and tracked in a position registry, or (b) avoiding Alpaca for crypto execution entirely.

---

## Structural Failure 2: The 10-Minute Sensing Cadence Is Fatal for Day Trading

The research brief states: "At ~24s/step actual (LLM overhead) = ~10 minutes between cycles."

Crypto day trading requires acting on signals within minutes of formation. By the time MIDGE's convergence check fires at T+10m, the following has already happened:

- A BTC flash move of 3-5% has already attracted stop-loss cascades
- Funding rates have already re-priced on Kraken
- The Fear & Greed Index won't update for another 50 minutes (updates every 60 minutes)
- CoinGecko free-tier data is already 1-5 minutes stale before MIDGE even reads it

The compounding effect: MIDGE reads data that is already 1-5 minutes old, waits another 10 minutes before checking convergence, then submits a market order into a market that moved 8-12 minutes ago.

**Industry consensus is clear:** Retail bots running from home setups are "hundreds of times slower" than institutional systems and "often miss profitable windows entirely" (ForTraders.com, 2025). 52% of automated accounts fail within 3 months. The latency problem is cited as the primary cause.

MIDGE's current cadence makes her appropriate for swing trading (3-7 day holds) but not day trading. The research brief targets both. She can only deliver one.

---

## Structural Failure 3: Crypto Has No Short Side on Alpaca

Alpaca documentation is explicit: **cryptocurrencies are not shortable**. MIDGE can only go long on crypto through Alpaca.

This cuts her effective opportunity set in half. In a bear market, MIDGE will identify bearish convergences on BTC, ETH, SOL — and be unable to execute them. Convergence detection will fire bearish signals that have nowhere to go.

Worse: the Thompson learning loop will register these bearish signals as "unactionable" (no trade placed) rather than "wrong" (trade placed, lost money). The feedback loop cannot learn from bearish crypto signals because they produce no outcome to grade. This is a corrupted learning loop — not just a missing feature.

The paper at `mae_core/bootstrap/market_hooks_trades.py` line 465:
```python
side = "buy" if signal.direction == "long" else "sell"
```
This will submit a sell-side market order on a non-shortable asset. What happens? Alpaca will reject it. The `except Exception` at line 494 swallows it silently. MIDGE will log nothing, learn nothing, and continue generating bearish crypto signals indefinitely.

---

## Structural Failure 4: Pattern Archaeology — 57 Templates Is Not Enough

The research brief states: "57 crypto templates from archaeological mining."

The Pattern Archaeology system requires cross-symbol validation (3+ symbols) and uses Clopper-Pearson confidence intervals. For a template to be statistically meaningful with a Clopper-Pearson CI narrow enough to inform trading decisions (±10%), you need approximately 60-100 instances per template.

57 templates × likely 3-10 instances each = roughly 170-570 total observations, spread across:
- Multiple crypto assets (BTC, ETH, SOL, XRP, ADA)
- Multiple timeframes
- A market history dominated by a single macro cycle (2020-2024 bull run + 2022 bear + 2024 recovery)

**The critical problem:** Crypto has lived through one complete cycle of the current interest-rate-driven macro regime. Every pattern extracted from that single cycle is potentially regime-specific, not universal. A pattern that fired during 2021 ZIRP conditions may be meaningless in 2026 conditions. The archaeology cannot distinguish regime-specific patterns from universal ones with only one cycle of data.

Research literature is unambiguous: "Overfitting causes models to memorize historical noise rather than learning genuine patterns, fitting perfectly to the past but having zero predictive power for the future." With 57 templates from one cycle, the Deflated Sharpe Ratio (which MIDGE applies to hypotheses) should be applied here too — and it would reject most of these templates as insufficiently validated.

---

## Latent Risk 1: Thompson Cold Start on New Crypto Sources

Five new crypto sources were added today: Kraken futures, mempool.space, CoinDesk/Cointelegraph RSS, DefiLlama, and per-ticker Fear & Greed. All five start at the Beta(1,1) uninformative prior — a 50% success probability assumption.

The `learning_config.py` priors for these sources:
- `crypto_coingecko`: 0.50 (neutral)
- `crypto_coincap`: 0.50 (neutral)

The new sources are not even seeded — they will start at Beta(1,1), meaning MIDGE's confidence formula will treat Kraken funding rates with the same weight as sources that have been validated over months of outcomes.

**The timeline problem:** OUTCOME_WINDOWS for crypto signals don't exist in `outcome_collector.py` — there is no `"crypto_kraken"`, `"mempool"`, `"defi_llama"` entry. These will fall through to the default 14-day window. At 1-5 trades per week, meaningful Thompson learning (>20 samples per source) requires 4-20 weeks of live trading before the learning loop has anything useful to say.

MIDGE will trade on unvalidated crypto signals for the first 1-5 months. This is not a problem unique to MIDGE — it is a fundamental constraint of the Bayesian cold-start problem with low observation rates. But the proposal presents this as a working system from day one.

---

## Latent Risk 2: The 24/7 Exposure Problem Is Real and Structural

MIDGE runs on a personal laptop. The research brief acknowledges this ("Known constraints"). The ecosystem modes in `MIDGE-MODES.md` range from SLEEP (0% CPU) to SPRINT (90% CPU). SLEEP mode means all positions are abandoned with no monitoring.

If MIDGE holds a crypto position (no stop-loss as established above) and the laptop goes to sleep or loses power:
1. No stop-loss exists on the order (structural failure 1)
2. No watchdog process monitors open positions independently
3. No circuit breaker closes positions before shutdown
4. No notification fires to Guiding Light

Industry consensus: VPS is universally cited as the minimum infrastructure for 24/7 crypto trading bots. "Home computers face power outages, ISP downtime, or computer crashes." This is not optional for a position-holding system — it is table stakes.

The current architecture treats the laptop as a server. It is not a server.

---

## The Deception Problem: Whales Are Training MIDGE Against Herself

This is the risk that gets missed because it sounds speculative but has empirical support.

Crypto markets are dominated by actors (whales, market makers, coordinated funds) who deliberately create false signals to shake out retail participants. Documented tactics include:
- Spoofing order books to create false demand
- Manufacturing sentiment divergences (letting retail panic while accumulating)
- Coordinating on-chain activity to create misleading signals

MIDGE's convergence engine reads: social sentiment (StockTwits), Fear & Greed index (partially sentiment-derived), on-chain mempool activity. All three of these are channels through which sophisticated actors can and do plant false signals.

If a whale wants to accumulate BTC cheaply, they can manufacture a Fear & Greed signal of "extreme fear" while simultaneously showing bullish on-chain accumulation — creating a false bullish convergence that MIDGE will fire on, while the whale sells into the retail buying.

MIDGE has a `DeceptionDetector` — but it is calibrated on equity market manipulation patterns. Crypto manipulation has different signatures, different speeds (minutes not days), and different mechanisms. The deception detector is likely miscalibrated for crypto.

---

## What This Means for the Proposal

**The proposal as written will not produce profitable crypto trading.** It will produce paper trades in a simulation environment that teaches habits (no stop-loss, no bearish execution, overconfident convergence) that are actively dangerous when transferred to live capital.

**What actually needs to happen before crypto trading can be taken seriously:**

1. Domain de-duplication: Map the 7 crypto "domains" to their 2 genuine independent dimensions. Adjust minimum domain count for crypto alerts to reflect this (min_domains=2 for crypto is equivalent to min_domains=4 for equities).

2. Exit management: Build a crypto position registry with independent GTC limit orders for stop and target, tracked across restarts.

3. Swing trading only, not day trading: Acknowledge the 10-minute cadence is a swing trading cadence. Stop describing this as day trading capability.

4. One-sided only for now: Bearish crypto signals should be logged and tracked but not attempted on Alpaca. Either acquire shorting capability (futures via a different venue) or restrict to long-only and document this constraint.

5. Template validation gate: Apply a minimum of 15 instances per template before a crypto template feeds live convergence. The 57 templates should be audited — most likely have fewer than 10 instances.

6. VPS or persistent process: Open crypto positions require a process that survives laptop sleep. At minimum, a watchdog script that closes all crypto positions if connectivity is lost for >30 minutes.

**The good news:** Every one of these failures is in paper trading mode. Nothing has been lost. The architecture is not wrong — it is equity-first and needs explicit crypto-specific adaptations it doesn't yet have.

---

## Sources Consulted

- [Are Crypto Trading Bots Worth It in 2025?](https://coincub.com/are-crypto-trading-bots-worth-it-2025/)
- [Why Most Trading Bots Lose Money](https://www.fortraders.com/blog/trading-bots-lose-money)
- [Alpaca Crypto Spot Trading Fees](https://docs.alpaca.markets/docs/crypto-fees)
- [Alpaca Crypto Spot Trading](https://docs.alpaca.markets/docs/crypto-trading)
- [Can I margin or short with cryptocurrency? - Alpaca](https://alpaca.markets/support/can-i-margin-or-short-with-cryptocurrency)
- [CoinAPI vs CoinGecko Comparison](https://www.coinapi.io/blog/coinapi-vs-coingecko-crypto-api-comparison)
- [Crypto Whale Market Manipulation](https://www.ccn.com/education/crypto/crypto-market-manipulation-whales-wash-trading-fake-pumps-explained/)
- [VPS for Crypto Trading Bots: 24/7 Uptime](https://www.bluehost.com/blog/vps-for-crypto-trading-bots/)
- [Whale Activity as Leading Indicator - 2025](https://www.ainvest.com/news/whale-activity-leading-indicator-crypto-markets-insights-2025-chain-data-2512/)
- [Bitcoin vs Altcoin Price Correlation](https://finst.com/en/learn/articles/price-correlation-between-bitcoin-and-altcoins)
- [BTC/Nasdaq Correlation 0.52](https://www.coindesk.com/markets/2026/02/17/crypto-slides-as-tech-stocks-and-gold-retreat-bitcoin-nasdaq-correlation-turns-positive)
- [Thompson Sampling Cold-Start Problem](https://arxiv.org/html/2602.00943)
- [Crypto Fear & Greed as Contrarian Indicator](https://www.ainvest.com/news/decoding-crypto-market-sentiment-fear-greed-index-contrarian-compass-2509/)
- [Curve Fitting in Trading](https://www.quantifiedstrategies.com/curve-fitting-trading/)
