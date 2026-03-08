# Team 3 Findings: MIDGE Gap Analysis & Execution Bridge
## Date: 2026-03-07
## Researcher: Team Member 3

---

### Preface: What MIDGE Has vs. What Execution Needs

Before cataloguing approaches, it is worth mapping MIDGE's existing paper trading infrastructure against the execution gap precisely. This frames every recommendation that follows.

**What MIDGE already has (execution-relevant):**

- `TradeSignal` dataclass (`mae_core/market/signal.py`) — the actionable output: asset, direction, confidence, kelly_fraction, contributing_signals, asset_class
- `_write_paper_trade()` in `mae_core/bootstrap/market_hooks.py` — fires when `confidence >= 0.45`, `strength >= 0.65`, `combo_mean >= 0.25`; writes to `data/midge/paper_trades.jsonl`
- `KellyPositionSizer` (`mae_core/market/intelligence/kelly_position_sizer.py`) — half-Kelly, capped at 5% of bankroll; already computing `kelly_capped` per signal source
- `PortfolioTracker` (`mae_core/market/intelligence/portfolio_tracker.py`) — reads `paper_trades.jsonl`, maintains live P&L, emits exit signals on stop-loss/take-profit
- `ActiveTracker` (`mae_core/market/archaeology/active_tracker.py`) — tracks 20 assets, status transitions tracking→confirming→confirmed/failed/expired, force-grades outcomes
- `OutcomeCollector` + `OutcomeTracker` — full Thompson feedback loop already wired

**The exact gap:** `_write_paper_trade()` currently does: convergence alert fires → TradeSignal created → written to `paper_trades.jsonl`. Nothing in that path calls a broker API. The bridge is a single insertion point between the TradeSignal serialization and JSONL append.

---

### Battle-Tested Approaches

#### 1. Alpaca Markets — Equities/ETFs/Crypto (Paper-Identical to Live)

- **What:** Commission-free US equities, ETFs, and crypto broker with a Python SDK (`alpaca-py`) where paper and live trading are identical code — only the API endpoint URL and key pair differ.
- **Evidence:** Alpaca serves tens of thousands of retail algo traders. `alpaca-py` is the current official SDK (replaced the older `alpaca-trade-api-python`). The paper endpoint has been production-stable since 2019.
- **Source:** https://docs.alpaca.markets/docs/paper-trading (accessed 2026-03-07), https://alpaca.markets/sdks/python/getting_started.html (accessed 2026-03-07)
- **Fits our case because:** MIDGE already generates `TradeSignal` with `asset`, `direction`, `kelly_capped`, and `asset_class`. Alpaca accepts exactly these fields. Switching paper to live is one environment variable: `APCA_API_BASE_URL=https://paper-api.alpaca.markets` vs. the live default. No code changes required.
- **Tradeoffs:**
  - Paper trading does NOT simulate market impact, slippage, or order queue position — paper win rates will be optimistic vs. live
  - Partial fills occur randomly 10% of the time in paper, not based on real liquidity
  - Requires US account; crypto support exists but spot-only (no futures/options via base API)
  - Account requires funding; paper account is free with any funded live account

**Execution pattern (Python):**
```python
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

client = TradingClient(api_key, secret_key, paper=True)  # paper=False for live
order = MarketOrderRequest(
    symbol=trade_signal.asset,
    qty=shares,
    side=OrderSide.BUY if trade_signal.direction == "buy" else OrderSide.SELL,
    time_in_force=TimeInForce.DAY,
)
client.submit_order(order)
```

#### 2. Interactive Brokers (IBKR) — Global Markets, All Asset Classes

- **What:** The most comprehensive broker API available — stocks, ETFs, options, futures, forex, bonds, across 150 markets globally. Python via TWS API (native) or `ib_async` library.
- **Evidence:** IBKR serves institutional and retail traders. The TWS API has been production-stable for 15+ years. As of 2025, v10.42 is current with a new Synchronous Wrapper. `ib_async` (the spiritual successor to `ib_insync`) is actively maintained at github.com/ib-api-reloaded/ib_async.
- **Source:** https://www.interactivebrokers.com/campus/ibkr-quant-news/getting-started-with-the-interactive-brokers-python-api/ (accessed 2026-03-07), https://blog.pickmytrade.io/ib-api-python-2026-automated-trading-setup-ibkr-integration/ (accessed 2026-03-07)
- **Fits our case because:** MIDGE's `asset_class` field already distinguishes "stock", "futures", "crypto". IBKR supports all three. MIDGE's multi-timeframe edge (insider + government + macro) often points at instruments where futures give better payoff math (linear) than stocks. IBKR is the only retail broker with genuine futures access.
- **Tradeoffs:**
  - Requires TWS (desktop app) or IB Gateway to be running — adds infrastructure dependency
  - Minimum $500 funded account for API access; paper requires a funded live account
  - API requires IBKR Pro account type (not Lite)
  - Significantly more setup complexity than Alpaca
  - Token/session management adds operational overhead

#### 3. CCXT — Unified Crypto Exchange Layer (100+ Exchanges)

- **What:** Open-source Python library providing a single unified API across 100+ crypto exchanges including Coinbase Advanced Trade, Binance, Kraken, Bybit, and others. As of 2026, over 65% of solo trading bot developers use CCXT.
- **Evidence:** Active GitHub repo (ccxt/ccxt), 30k+ stars. `ccxt.pro` adds WebSocket support. Used in production by thousands of retail and institutional traders.
- **Source:** https://github.com/ccxt/ccxt (accessed 2026-03-07), https://www.bitget.com/academy/how-to-get-started-with-ccxt-in-python-for-cryptocurrency-trading-in-2026-complete-beginners-guide (accessed 2026-03-07)
- **Fits our case because:** MIDGE already tracks crypto signals via CoinGecko and CoinCap. When crypto signals (domain="crypto") converge with macro/insider signals, CCXT is the execution layer. Crypto markets are 24/7, matching MIDGE's daemon mode. CCXT handles exchange-specific auth complexity behind a unified interface.
- **Tradeoffs:**
  - Crypto markets have higher volatility — same Kelly fraction = larger expected drawdown
  - Exchange API key security is the operator's responsibility (no custodian)
  - Rate limits vary per exchange — must be respected to avoid bans
  - No prediction market access through CCXT

#### 4. Polymarket CLOB API — Prediction Market Execution

- **What:** Central Limit Order Book for event contracts on Polygon blockchain. Python SDK: `py-clob-client`. Positions denominated in USDC. A demonstrated $313 → $414,000 arbitrage bot case confirms the market is real and accessible.
- **Evidence:** Official Polymarket repo at github.com/Polymarket/py-clob-client. Polymarket Agents framework (github.com/Polymarket/agents) exists specifically for autonomous AI agents. In January 2026, Polymarket acquired Dome (a unified prediction market API startup), signaling active developer ecosystem investment.
- **Source:** https://agentbets.ai/guides/prediction-market-api-reference/ (accessed 2026-03-07), https://github.com/Polymarket/agents (accessed 2026-03-07), https://blockeden.xyz/blog/2026/01/25/prediction-markets-polymarket-kalshi-ai-agents/ (accessed 2026-03-07)
- **Fits our case because:** MIDGE's cross-domain convergence is purpose-built for prediction markets. When MIDGE detects "Congressional trades + government contract + macro convergence" on a defense company, this almost certainly predicts a specific outcome (contract award, earnings beat) that corresponds to a Polymarket event contract. AI agents now contribute over 30% of prediction market volume — the infrastructure is proven.
- **Tradeoffs:**
  - Requires Polygon wallet + USDC funding — adds crypto infrastructure overhead
  - Market liquidity is shallow ($5k-$15k per side typical) — large positions move price
  - Event resolution is binary — MIDGE's continuous confidence doesn't map perfectly to YES/NO
  - HMAC signing required for all order operations
  - Markets are user-created; finding the right market for a given MIDGE signal requires a discovery layer

#### 5. Kalshi REST API — CFTC-Regulated US Prediction Markets

- **What:** CFTC-regulated US exchange for event contracts (macroeconomic, political, weather). REST API + WebSocket + FIX 4.4. Python SDK: `kalshi-python`. Prices in cents (1-99 per contract).
- **Evidence:** Kalshi is CFTC-regulated, making it the only legally unambiguous prediction market for US-based autonomous systems. FIX 4.4 support means institutional-grade connectivity. AI agents now drive significant volume; one autonomous trading system using Grok-4 integration reported in PANews (2026).
- **Source:** https://docs.kalshi.com/welcome (accessed 2026-03-07), https://blockeden.xyz/blog/2026/01/25/prediction-markets-polymarket-kalshi-ai-agents/ (accessed 2026-03-07), https://zuplo.com/learning-center/kalshi-api (accessed 2026-03-07)
- **Fits our case because:** Kalshi covers FOMC decisions, CPI numbers, Fed funds rate — exactly the macro domain signals MIDGE tracks (FRED macro + economic calendar + finnhub economic). When MIDGE's macro domain fires with conviction, the corresponding Kalshi contract is the cleanest execution vehicle. Linear payoff, known resolution date, no slippage from MIDGE's position size.
- **Tradeoffs:**
  - RSA key-pair auth (more complex than API key)
  - Limited market catalog compared to equity universe — MIDGE signal must match an open Kalshi market
  - Tiered API access; high-volume trading may require tier upgrade
  - Demo environment available but live requires funded account

---

### Novel Approaches

#### 1. Prediction Market Arbitrage as MIDGE's First Domain

- **What:** Use MIDGE's cross-domain signals to identify mispricings between Polymarket/Kalshi event contracts and their implied probabilities from options markets, news, and institutional data sources.
- **Why it's interesting:** MIDGE already has options-adjacent signals (TA, VIX term structure, COT positioning). When MIDGE's insider + macro + government convergence implies an 80% probability of an event, but Kalshi's market prices it at 55%, the edge is structural — not speed-dependent. This is precisely where MIDGE's slow cross-domain moat shines.
- **Evidence:** CoinDesk reported (2026-02-21) a bot converting $313 → $414,000 in one month by exploiting YES+NO price sums below $1 (pure arb). More relevant: cross-market probability comparison bots achieve 85%+ win rates versus ~50% human, with $16.80/trade thin enough to avoid detection.
- **Source:** https://www.coindesk.com/markets/2026/02/21/how-ai-is-helping-retail-traders-exploit-prediction-market-glitches-to-make-easy-money (accessed 2026-03-07)
- **Fits our case because:** MIDGE's proven z=4.74, p<0.0001 edge means her probability estimates ARE mispriced relative to prediction markets. The bridge from convergence alert to Kalshi/Polymarket order is structurally simpler than equity execution (no slippage, no partial fills, known resolution date). This is the minimum viable path to a self-funding loop.
- **Risks:** Market discovery is hard — MIDGE must find the matching contract for each signal. Prediction market catalogs are sparse. Markets may not exist for every MIDGE signal.

#### 2. Shadow Mode as a Formal Deployment Stage (Not Just Paper)

- **What:** A shadow execution layer that submits REAL orders to a broker API but at minimum position sizes (e.g., $1 per trade), then compares fills, slippage, and partial fill rates against paper simulation.
- **Why it's interesting:** Alpaca paper trading explicitly does NOT simulate market impact or slippage. A shadow mode with $1 real orders creates a calibration dataset for the transition from paper to live — specifically measuring slippage and fill rates at real market prices.
- **Evidence:** The 3commas risk management guide (2025) recommends deploying "1-2% of total intended capital" for 48-72 hours before scaling, then increasing to 10-20% while monitoring. This is effectively shadow mode with small capital.
- **Source:** https://3commas.io/blog/ai-trading-bot-risk-management-guide-2025 (accessed 2026-03-07)
- **Fits our case because:** MIDGE's Kelly sizer already caps at 5% of bankroll. Shadow mode with $1 orders on a $100 account would consume <$5 per trade while generating real slippage data. The paper-to-live confidence gap (winners=0.560 vs. losers=0.565 in replay) suggests the execution layer, not the signal layer, needs calibration.
- **Risks:** Requires a separate broker account for shadow. $1 orders may get different fills than larger orders (different liquidity tier).

#### 3. Autonomous Wallet Pattern (Crypto + Prediction Markets)

- **What:** MIDGE maintains a USDC wallet on Polygon. Profits from Polymarket trades stay on-chain and directly fund the next round of trades, creating a self-reinforcing capital pool without human intervention.
- **Why it's interesting:** AI agents have already demonstrated this pattern at scale. The Polymarket agents framework explicitly requires `POLYGON_WALLET_PRIVATE_KEY` and USDC balance. The entire loop from signal → order → settlement → next order can be automated without a human approving any step.
- **Evidence:** github.com/Polymarket/agents: "Users must load wallets with USDC for trading" — then the system runs autonomously. IOSG (2026 outlook) forecasts prediction market agents as "a new product form," with Piper Sandler estimating $222.5 billion in notional volume for 2026.
- **Source:** https://github.com/Polymarket/agents (accessed 2026-03-07), https://www.kucoin.com/news/flash/iosg-prediction-market-agents-to-emerge-as-new-product-form-in-2026 (accessed 2026-03-07)
- **Fits our case because:** This is the literal self-funding loop described in the research brief. Prediction markets settle automatically (no human needed to close positions). USDC is stable — MIDGE doesn't need to manage currency risk between trades.
- **Risks:** Smart contract risk (Polygon), USDC de-peg risk, private key management is operationally critical — if lost, all funds are permanently lost.

---

### Emerging Approaches

#### 1. Alpaca MCP Server — Direct LLM-to-Broker Integration

- **What:** Alpaca released an official MCP (Model Context Protocol) server (github.com/alpacahq/alpaca-mcp-server) that allows LLM tools to trade directly using natural language, with built-in stocks, ETFs, crypto, and options access.
- **Momentum:** GitHub repo newly created in early 2026. MCP is the emerging standard for LLM-tool integration. This is specifically designed for AI agent trading.
- **Source:** https://github.com/alpacahq/alpaca-mcp-server (accessed 2026-03-07)
- **Fits our case because:** MIDGE does not currently use an LLM decision layer, so MCP is less relevant than the direct SDK. However, if MIDGE adds an LLM synthesis layer (e.g., for plain-language alert routing), the MCP server eliminates the broker integration layer entirely.
- **Maturity risk:** MCP is a new protocol standard. Tool support and stability are still evolving. Not recommended as a primary execution path for MIDGE's current architecture.

#### 2. pmxt / Unified Prediction Market Abstraction

- **What:** Open-source Python library (`pip install pmxt`) that normalizes Polymarket and Kalshi APIs into a single interface, similar to CCXT for crypto exchanges.
- **Momentum:** Mentioned in agentbets.ai's 2026 API reference as an active tool. Dome (a similar commercial product) was acquired by Polymarket in early 2026, suggesting the market is validating this abstraction layer.
- **Source:** https://agentbets.ai/guides/prediction-market-api-reference/ (accessed 2026-03-07)
- **Fits our case because:** If MIDGE targets both Polymarket and Kalshi (which serve different signal types — Kalshi for macro, Polymarket for event-specific), pmxt provides a unified order interface. Reduces code complexity from two broker integrations to one.
- **Maturity risk:** pmxt is community-maintained with limited production evidence. The Dome acquisition by Polymarket may signal consolidation that fragments this space further.

#### 3. Tradier API — Low-Friction US Equities Entry Point

- **What:** Developer-friendly REST API for US equities and options. `lumiwealth-tradier` Python wrapper (January 2026 release). Low fees, paper and live accounts.
- **Momentum:** BrokerChooser (2026) lists Tradier as notable for algorithmic trading despite low customer service scores. `lumiwealth-tradier` just released in January 2026.
- **Source:** https://docs.tradier.com/ (accessed 2026-03-07), https://brokerchooser.com/best-brokers/best-brokers-for-algo-trading-in-the-united-states (accessed 2026-03-07)
- **Fits our case because:** Simpler account approval process than Schwab/IBKR. Paper and live trading use the same code structure.
- **Maturity risk:** Tradier's customer support is rated 2.5/5 — the lowest among top algo brokers. Poor support is a material operational risk for an autonomous system with unexpected edge cases.

#### 4. Schwab Developer API (Post-TD Ameritrade)

- **What:** Schwab launched Trader API in 2024 after absorbing TD Ameritrade accounts. REST API for equities, options, account management.
- **Momentum:** Access requires manual approval (days to weeks in "Approved - Pending" state). Tokens expire every 7 days and must be regenerated — a significant operational challenge for a daemon process.
- **Source:** https://developer.schwab.com/ (accessed 2026-03-07), https://blog.traderspost.io/article/does-td-ameritrade-have-api (accessed 2026-03-07)
- **Fits our case because:** Schwab has the largest US retail brokerage footprint, which could matter for stealth (MIDGE's orders are less identifiable in a massive retail flow).
- **Maturity risk:** 7-day token expiration is a critical operational burden. An autonomous daemon that runs 24/7 needs token refresh automation. This is a solved problem but adds infrastructure. Not recommended as primary broker.

---

### Gaps and Unknowns

1. **Minimum bet sizes on Polymarket and Kalshi are undocumented in public API references.** The agentbets.ai 2026 reference explicitly notes this gap. Minimum viable position sizes need to be tested empirically before designing the self-funding loop.

2. **MIDGE's signal-to-market mapping is unresolved.** MIDGE fires on tickers (e.g., "LMT" or "NVDA"). Prediction market contracts are event-based (e.g., "Will LMT announce a contract >$1B in Q2 2026?"). There is no current mechanism in MIDGE to map a convergence alert to a specific Polymarket/Kalshi market. This is the single biggest unknown for the prediction market path.

3. **Slippage calibration for equity execution is unmeasured.** Alpaca paper trades do not simulate slippage. MIDGE's proven 19.9% win rate comes from paper + replay, not live execution. The actual live win rate under slippage is unknown. Shadow mode is the only way to measure this before committing capital.

4. **MIDGE's stealth architecture is undefined.** The research brief requires MIDGE to be "undetectable as algorithmic." This is achievable (see Stealth section below) but zero stealth measures are currently implemented. Order timing, sizing, and frequency patterns all leave fingerprints.

5. **Autonomous wallet private key management is an unsolved operational security problem.** If MIDGE runs unattended with a funded Polygon wallet, the private key must be accessible to the daemon process. Standard `.env` file storage is a security risk. Hardware security module (HSM) or secrets manager integration has not been researched.

6. **Self-calibrating withdrawal/reinvestment loop is undesigned.** The brief calls for MIDGE to "fund her own compute." This requires: (a) measuring total P&L, (b) withdrawing profits above a threshold, (c) routing funds to a compute payment account. None of these steps have a defined architecture.

7. **Regulatory considerations for autonomous US equity trading are unresearched.** FINRA algorithmic trading rules require surveillance and controls. An autonomous system trading US equities may have disclosure requirements depending on scale.

---

### Synthesis

#### The Minimum Viable Execution Layer (MVEL)

The gap between MIDGE's current state and live execution is precisely **one function**: a broker client that replaces the JSONL append in `_write_paper_trade()`. The structural components — signal generation, confidence gating, Kelly sizing, position tracking — already exist. What is missing:

**Component 1: BrokerClient abstraction** — a thin wrapper with `submit_order(asset, direction, quantity, order_type)` and `get_position(asset)`. Behind this interface: Alpaca for equities/ETFs, CCXT for crypto, Kalshi SDK for macro prediction markets.

**Component 2: RiskGateway** — sits between the convergence alert and BrokerClient. Enforces:
- Portfolio-level drawdown circuit breaker (halt if daily loss > 7% of account)
- Per-position cap (never more than 5% of account in one position — already handled by Kelly cap)
- Correlation limit (no two positions in the same domain cluster)
- Trade frequency cap (max 10/day — matches 3commas industry recommendation)
- Kill switch flag (environment variable `MIDGE_TRADING_ENABLED=false` halts all execution)

**Component 3: FillTracker** — subscribes to broker fill webhooks/WebSocket to update `ActiveTracker` and `PortfolioTracker` with real fills instead of estimated fills. Alpaca provides WebSocket streaming for order updates.

**Component 4: MarketSelector** (for prediction markets only) — maps a convergence alert's ticker + domain combination to a specific Kalshi/Polymarket market. This is the novel piece with no existing analog in MIDGE.

#### Recommended Deployment Pipeline: Paper → Shadow → Live

**Stage 1 (Current): Paper** — MIDGE writes to `paper_trades.jsonl`. No broker API calls. `ActiveTracker` monitors hypothetical positions. This stage is already complete.

**Stage 2: Shadow** — MIDGE calls the Alpaca paper API (not live) using its own API keys. Fills are real simulations with partial fill randomization. Shadow runs for 30-90 days to accumulate real fill data. The key calibration metric: do paper win rates hold when the execution layer adds latency and partial fills?

**Stage 3: Live Micro** — Switch Alpaca endpoint from paper to live. Start with 10% of intended capital ($5,000 on a $50,000 account). Run for 30 days. The Kelly cap at 5% means maximum exposure is $250 per trade. Measure actual vs. paper win rates.

**Stage 4: Live Full** — Gradually increase to full capital after Stage 3 win rates are within 5% of Stage 2 results.

**Stage 5: Prediction Markets** — After equity execution is stable, add Kalshi for macro signals. This is the self-funding path: prediction market profits stay in USDC and fund compute costs before equity profits are withdrawn.

#### Where MIDGE's Edge Fits Best (Stealth + Payoff Analysis)

MIDGE's edge is slow-forming (14-90 day outcome windows), multi-domain, and cross-domain. This does NOT fit:
- High-frequency trading (MIDGE's edge is not speed-based)
- Intraday equity scalping (MIDGE's signals mature over days/weeks)

MIDGE's edge DOES fit:
- **Kalshi macro contracts** — FOMC, CPI, NFP outcome predictions match MIDGE's macro domain perfectly. MIDGE's Granger causality analysis on FRED data provides genuine lead-time over market consensus. Contracts resolve on known dates (no holding-period ambiguity).
- **Equity positions held 5-45 days** — matching MIDGE's existing `timeframe_days=5` and outcome windows of 14-90 days. Alpaca with limit orders (not market orders) minimizes slippage for slow-moving thesis trades.
- **Prediction market events matching government + insider signals** — When Congressional trades + hiring surge + government contract signals converge on a company, a Polymarket event market often exists (earnings surprise, regulatory approval, contract award). MIDGE's cross-domain edge would be genuinely invisible to other prediction market participants who don't have access to these 30 simultaneous data sources.

#### Stealth Architecture

The research brief requires MIDGE to be undetectable as algorithmic. Based on the BJF Trading Group white paper (2026), the key measures are:

- **Order timing jitter** — randomize submission time ±5-120 seconds after signal fires (not immediate)
- **Size variation** — add ±8-15% noise to Kelly-computed share count (round to nearest whole share anyway)
- **Order type rotation** — mix market orders and limit orders (don't always use market orders at signal time)
- **Frequency cap** — max 3-5 positions open simultaneously (not mechanical/systematic appearance)
- **Cross-domain appearance** — MIDGE's signals naturally produce varied ticker lists (not the same tickers repeatedly), which looks like informed discretionary trading, not systematic scanning

For prediction markets specifically: position sizes of $50-$500 per trade are "thin enough to be invisible on any single execution" (CoinDesk, 2026). MIDGE should target this range initially.

#### Recommended First Domain: Kalshi Macro Contracts

**Why Kalshi macro is the right starting domain:**

1. MIDGE already has 6 macro-domain sources (FRED, finnhub economic, COT positioning, VIX term structure, economic calendar, Granger causality on macro-technical relationships)
2. Kalshi contracts have known resolution dates — MIDGE's Thompson Sampler can be updated deterministically the moment the contract resolves (no price-check uncertainty)
3. CFTC regulation means no legal ambiguity for US operation
4. Contract payoffs are linear (1-99 cents → $1 at resolution) — matches MIDGE's "prefer instruments where payoff math is linear" design principle
5. The FIX 4.4 protocol support means institutional-grade connectivity when/if MIDGE scales
6. Minimum capital to be meaningful: a $1,000 USDC allocation can fund 20-100 trades at $10-$50/position — genuinely self-funding compute costs at $20-50/month

**The self-funding loop with Kalshi:**
- MIDGE fires macro convergence alert → `MarketSelector` finds matching Kalshi contract → `RiskGateway` approves → `BrokerClient` submits limit order via kalshi-python → contract resolves → P&L deposited to Kalshi account → Thompson Sampler updated with outcome → withdrawn profits pay compute costs

This loop requires zero human decisions. Guiding Light turns MIDGE on.

---

### Broker/Exchange API Quick Reference

| Broker | Asset Classes | Paper Available | Account Min | Python SDK | Auth Type | Key Constraint |
|--------|--------------|----------------|-------------|------------|-----------|----------------|
| Alpaca | Stocks, ETFs, Crypto | Yes (free) | $0 funded | `alpaca-py` | API key pair | US only; paper ≠ live slippage |
| IBKR | All (global) | Yes (requires funded live) | $500 | `ib_async` | App + session | Requires TWS/Gateway running |
| CCXT | Crypto (100+ exchanges) | Exchange-dependent | Exchange-dependent | `ccxt` / `ccxt.pro` | Exchange API keys | Rate limits vary |
| Kalshi | Event contracts (macro, political) | Yes (demo API) | $0 | `kalshi-python` | RSA key pair | CFTC-regulated; US only |
| Polymarket | Event contracts (anything) | No | USDC on Polygon | `py-clob-client` | HMAC + wallet private key | Requires Polygon wallet |
| Tradier | US stocks, options | Yes | $0 | `lumiwealth-tradier` | OAuth | 7-day token refresh |
| Schwab | US stocks, options | No (paper deprecated) | $0 | `schwab-py` | OAuth | 7-day token expiration |

---

### Appendix: What the Existing Code Already Has (No Rebuilding Required)

The following MIDGE components map directly to execution layer requirements with zero modification:

| Execution Need | Existing MIDGE Component | Status |
|---------------|--------------------------|--------|
| Trade signal generation | `TradeSignal` dataclass, `_write_paper_trade()` | Complete |
| Position sizing | `KellyPositionSizer` (half-Kelly, 5% cap) | Complete |
| Position tracking | `PortfolioTracker` (reads paper_trades.jsonl) | Complete |
| Active monitoring | `ActiveTracker` (20 assets, MFE/MAE tracking) | Complete |
| Exit signal generation | `PortfolioTracker.check_exits()` | Complete |
| Thompson feedback on outcomes | `OutcomeCollector` + `OutcomeTracker` | Complete |
| Confidence gating | `paper_trade_min_confidence=0.45` in `LEARNING_CONFIG` | Complete |
| Combo filter (block known losers) | `paper_trade_min_combo_mean=0.25` in market_hooks.py | Complete |
| Plain-language alert formatting | `mae_core/market/plain_language.py` | Complete |
| Daemon mode | `main.py --daemon --pace 1.0` | Complete |

**The execution gap is surgical, not structural.** MIDGE is missing only: BrokerClient, RiskGateway, FillTracker, and MarketSelector. Everything else is production-ready.
